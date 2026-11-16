// 2-D InfLLM-v2 backward megakernel. Derived from
// csrc/infllm_bwd/infllm_bwd.cu with FOREIGN tasks: column tasks of moved units, run by the SAME
// official column routine with the unit's own parameter block; a completion count per unit fires the trigger that
// returns the unit's FP32 dQ accumulator to its query owner. Own tasks, reductions and completion: as csrc/infllm_bwd.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <vector>
#include <cutlass/numeric_types.h>
#define INFLLM_WANT_BWD
#include "../infllm_bwd/infllm_official.h"
#include "infllm2d_bwd_params.h"
#include "../runtime/rt_engine.cuh"
#include "../runtime/rt_sched.cuh"

#ifndef DIST_SPIN_REPORT
#define DIST_SPIN_REPORT (1L << 26)
#endif

namespace infllm2d_bwd {

using FParams = infllm_bwd::FParams;
using KT = Infllm_bwd_kernel_traits<128, 16, 64, 4, 1, 4, 1, /*Is_V_in_regs=*/true, /*No_double_buffer=*/true, InfllmElem>;
static_assert(KT::kNThreads == 128, "block 0 = 1 scheduler warp + 3 engine warps");

// (as infllm_bwd.cu fill)
static void fill(FParams &p, const FArgs &a) {
    p = {};
    p.is_bf16 = INFLLM_IS_BF16;
    p.q_ptr = a.q; p.k_ptr = a.k; p.v_ptr = a.v; p.o_ptr = a.o;
    p.q_row_stride = a.q_row_stride; p.k_row_stride = a.k_row_stride; p.v_row_stride = a.k_row_stride;
    p.o_row_stride = a.o_row_stride;
    p.q_head_stride = p.k_head_stride = p.v_head_stride = p.o_head_stride = 128;
    p.cu_seqlens_q = const_cast<int *>(a.cu_q); p.cu_seqlens_k = const_cast<int *>(a.cu_k);
    p.seqused_k = const_cast<int *>(a.seqused_k);
    p.softmax_lse_ptr = a.lse;
    p.b = a.b; p.h = 2; p.h_k = 2; p.h_h_k_ratio = 1;
    p.seqlen_q = a.max_seqlen_q; p.seqlen_k = a.max_seqlen_k;
    p.seqlen_q_rounded = (a.max_seqlen_q + 127) / 128 * 128; p.seqlen_k_rounded = (a.max_seqlen_k + 127) / 128 * 128;
    p.d = 128; p.d_rounded = 128;
    p.scale_softmax = a.scale; p.scale_softmax_log2 = a.scale * M_LOG2E;
    p.p_dropout = 1.f; p.p_dropout_in_uint8_t = 255; p.rp_dropout = 1.f; p.scale_softmax_rp_dropout = a.scale;
    p.is_causal = true; p.window_size_left = a.max_seqlen_k; p.window_size_right = 0;
    p.is_seqlens_k_cumulative = true; p.unpadded_lse = true; p.seqlenq_ngroups_swapped = false;
    p.total_q = a.total_q;
    p.blockmask = const_cast<uint64_t *>(a.blockmask);
    p.m_block_dim = 16; p.n_block_dim = 64; p.num_k_heads = 2;
    p.num_blocks_m = a.num_blocks_m; p.num_blocks_n = a.num_blocks_n;
    p.page_block_size = 1; p.num_splits = 1;
    p.do_ptr = a.dout; p.do_row_stride = a.do_row_stride; p.do_head_stride = 128;
    p.dq_ptr = a.dq; p.dq_row_stride = a.dq_row_stride; p.dq_head_stride = 128;
    p.dq_accum_ptr = a.dq_accum; p.dk_accum_ptr = a.dk_accum; p.dv_accum_ptr = a.dv_accum;
    p.dsoftmax_sum = a.dpsum;
    p.deterministic = false; p.dq_accum_split_stride = 0;
    p.acc_row_stride = a.acc_row_stride;
}

// One moved unit = the official varlen causal backward of ONE chunk (batch 1) for ONE KV-head group (h = h_k = 1; K / V
// and the dK / dV accumulators at the group's column). O is not read (Is_first = false: Δ comes precomputed); dQ goes
// only to dq_accum (Is_last = false).
static void fill_unit(FParams &p, const UnitDesc &u) {
    FArgs a{};
    a.q = u.q; a.k = u.k; a.v = u.v; a.o = u.dout; a.dout = u.dout; a.dq = u.q;
    a.q_row_stride = a.o_row_stride = a.do_row_stride = a.dq_row_stride = 128; a.k_row_stride = 512;
    a.lse = reinterpret_cast<float *>(u.lse); a.dpsum = reinterpret_cast<float *>(u.dpsum);
    a.dq_accum = reinterpret_cast<float *>(u.dq_accum);
    a.dk_accum = reinterpret_cast<float *>(u.dk_acc); a.dv_accum = reinterpret_cast<float *>(u.dv_acc); a.acc_row_stride = 512;
    a.cu_q = u.cu_q; a.cu_k = u.cu_k; a.seqused_k = u.seqused; a.b = 1;
    a.blockmask = reinterpret_cast<const uint64_t *>(u.bm);
    a.total_q = u.len * 16; a.num_blocks_m = u.len; a.num_blocks_n = u.nbn;
    a.max_seqlen_q = u.len * 16; a.max_seqlen_k = u.max_seqlen_k; a.scale = u.scale;
    fill(p, a);
    p.h = 1; p.h_k = 1; p.h_h_k_ratio = 1; p.num_k_heads = 1;
}

__device__ __forceinline__ unsigned long long bwd_now() {
    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
#define ITR2(p, i, v) do { if ((p).trace && threadIdx.x == 0) atomicAdd(&(p).trace[i], (unsigned long long)(v)); } while (0)

struct Ctl { int entry; };
constexpr int kCtlBytes = 128;

__device__ __noinline__ void slot_done(const Params &p, int n) {
    if (p.ks_agg_need[n] > 0) rt::red_count(p.sch, p.n_km + p.ka_unit[n]);
    else if (p.ks_push_pe[n] >= 0) rt::fire(p.eng, n);
    else rt::red_count(p.sch, p.ks_push_dst[n]);
}

__device__ __noinline__ void complete_task(const Params &p, int r) {
    if (threadIdx.x == 0) {
        __threadfence();
        const int n = p.task_n[r];
        if (atomicAdd(&p.ks_done[n], 1) + 1 == p.ks_need[n]) { __threadfence(); slot_done(p, n); }
        if (p.task_h[r] == kForeignH) {                      // [2-D] the unit's dQ is complete after its last task
            const int u = p.task_b[r];
            if (atomicAdd(&p.unit_done[u], 1) + 1 == p.unit_need[u]) { __threadfence(); rt::fire(p.eng, p.trig_unit0 + u); }
        }
        __threadfence();
        atomicAdd(p.sch.rows_done, 1);
    }
    __syncthreads();
}

__device__ __forceinline__ float4 add4(float4 a, const float4 b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; return a; }

__device__ __noinline__ void ksum(const Params &p, int m) {
    const int n = p.km_blk[m], rows = p.km_rows[m], l0 = p.km_ptr[m], nl = p.km_ptr[m + 1] - l0;
    const float4 *acc = reinterpret_cast<const float4 *>(p.base + p.acc_off + (int64_t)n * kSlotBytes);
    InfllmElem *dk = reinterpret_cast<InfllmElem *>(p.dk_out) + (int64_t)p.km_out[m] * 256;
    InfllmElem *dv = reinterpret_cast<InfllmElem *>(p.dv_out) + (int64_t)p.km_out[m] * 256;
    for (int e = threadIdx.x; e < rows * (kRowFloats / 4); e += blockDim.x) {
        float4 a = acc[e];
        for (int k = 0; k < nl; ++k)
            a = add4(a, reinterpret_cast<const float4 *>(p.base + p.land_off + (int64_t)p.km_land[l0 + k] * kSlotBytes)[e]);
        const int r = e / (kRowFloats / 4), c4 = e % (kRowFloats / 4);
        InfllmElem *o = (c4 < 64 ? dk : dv) + (int64_t)r * 256 + (c4 & 63) * 4;
        o[0] = InfllmElem(a.x); o[1] = InfllmElem(a.y); o[2] = InfllmElem(a.z); o[3] = InfllmElem(a.w);
    }
    __syncthreads();
}

__device__ __noinline__ void kagg(const Params &p, int u) {
    const int n = p.ka_blk[u], l0 = p.ka_ptr[u], nl = p.ka_ptr[u + 1] - l0;
    float4 *acc = reinterpret_cast<float4 *>(p.base + p.acc_off + (int64_t)n * kSlotBytes);
    for (int e = threadIdx.x; e < kBlockRows * (kRowFloats / 4); e += blockDim.x) {
        float4 a = acc[e];
        for (int k = 0; k < nl; ++k)
            a = add4(a, reinterpret_cast<const float4 *>(p.base + p.land_off + (int64_t)p.ka_land[l0 + k] * kSlotBytes)[e]);
        acc[e] = a;
    }
    __syncthreads();
    if (threadIdx.x == 0) rt::fire(p.eng, n);
}

__global__ void __launch_bounds__(KT::kNThreads) infllm2d_bwd_kernel(const __grid_constant__ FParams fp, const __grid_constant__ Params p) {
    extern __shared__ char smem_[];
    const int tidx = threadIdx.x;
    if (blockIdx.x == 0) {
        if (tidx < 32) rt::sched_block0<0>(p.sch, smem_, *p.sch_b0, &p.eng);
        else rt::engine(p.eng, 32, blockDim.x - 32, 1, reinterpret_cast<volatile int *>(smem_ + KT::kSmemSize1colblock));
        return;
    }
    Ctl &c = *reinterpret_cast<Ctl *>(smem_ + KT::kSmemSize1colblock);
    const FParams *fpu = reinterpret_cast<const FParams *>(p.fpu);
    if (p.trace && tidx == 0) atomicMin(&p.trace[5], bwd_now());
    for (;;) {
        const unsigned long long tw0 = p.trace ? bwd_now() : 0;
        if (tidx == 0) {
            const int pos = atomicAdd(p.tile_counter, 1);
            const volatile int *q = p.sq + pos;
            long n = 0;
            while (*q == 0) {
                __nanosleep(64);
                if (++n % DIST_SPIN_REPORT == 0) printf("[pe %d blk %d] infllm2d bwd queue slot %d empty\n", nvshmem_my_pe(), blockIdx.x, pos);
            }
            c.entry = *q - 1;
            __threadfence();
        }
        __syncthreads();
        const int entry = c.entry;
        __syncthreads();
        const int kind = entry >> 28, pay = entry & ((1 << 28) - 1);
        const unsigned long long tw1 = p.trace ? bwd_now() : 0;
        ITR2(p, 0, tw1 - tw0);
        if (kind == kStop) break;
        if (kind == kTask) {
            const int r = p.sch.task_desc[4 * pay];
            if (p.task_h[r] == kForeignH)                    // [2-D] foreign task: unit task_b, block task_n, its own params
                infllm_flash::compute_dq_dk_dv_1colblock_acc<KT, false, true, false, false, false, true, false, false, false, true>(
                    fpu[p.task_b[r]], 0, 0, p.task_n[r]);
            else
                infllm_flash::compute_dq_dk_dv_1colblock_acc<KT, false, true, false, false, false, true, false, false, false, true>(
                    fp, p.task_b[r], p.task_h[r], p.task_n[r]);
            const unsigned long long tw2 = p.trace ? bwd_now() : 0;
            ITR2(p, 1, tw2 - tw1); ITR2(p, 4, 1);
            complete_task(p, r);
            ITR2(p, 2, (p.trace ? bwd_now() : 0) - tw2);
        } else if (kind == kKSum) {
            ksum(p, pay);
            ITR2(p, 3, (p.trace ? bwd_now() : 0) - tw1);
        } else if (kind == kKAgg) {
            kagg(p, pay);
            ITR2(p, 3, (p.trace ? bwd_now() : 0) - tw1);
        }                                                    // kNop: nothing (incl. the dQ-landing units)
    }
    if (p.trace && tidx == 0) atomicMax(&p.trace[6], bwd_now());
}

}  // namespace infllm2d_bwd

using KT2B = infllm2d_bwd::KT;

int infllm2d_bwd_smem_bytes() { return KT2B::kSmemSize1colblock + infllm2d_bwd::kCtlBytes; }

int infllm2d_bwd_grid() {
    static int g = 0;
    if (!g) {
        int per = 0, nsm = 0, dev = 0;
        cudaGetDevice(&dev);
        cudaFuncSetAttribute(infllm2d_bwd::infllm2d_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, infllm2d_bwd_smem_bytes());
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, infllm2d_bwd::infllm2d_bwd_kernel, KT2B::kNThreads, infllm2d_bwd_smem_bytes());
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        g = per * nsm;
    }
    return g;
}

int64_t infllm2d_bwd_fparams_bytes() { return (int64_t)sizeof(infllm2d_bwd::FParams); }

void infllm2d_bwd_build_units(const infllm2d_bwd::UnitDesc *d, int n, void *dev, cudaStream_t stream) {
    if (!n) return;
    std::vector<infllm2d_bwd::FParams> h(n);
    for (int i = 0; i < n; ++i) infllm2d_bwd::fill_unit(h[i], d[i]);
    cudaMemcpyAsync(dev, h.data(), (size_t)n * sizeof(infllm2d_bwd::FParams), cudaMemcpyHostToDevice, stream);
}

void infllm2d_bwd_launch(const infllm2d_bwd::FArgs &a, const infllm2d_bwd::Params &p, int grid, cudaStream_t stream) {
    infllm2d_bwd::FParams fp;
    infllm2d_bwd::fill(fp, a);
    infllm2d_bwd_grid();
    infllm2d_bwd::infllm2d_bwd_kernel<<<grid, KT2B::kNThreads, infllm2d_bwd_smem_bytes(), stream>>>(fp, p);
}
