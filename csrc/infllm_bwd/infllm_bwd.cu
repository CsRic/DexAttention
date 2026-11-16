// InfLLM-v2 backward megakernel: the OFFICIAL column routine as the compute,
// the shared block-0 scheduler + communication engine (csrc/runtime) as the runtime, one launch per GPU.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cutlass/numeric_types.h>
#define INFLLM_WANT_BWD
#include "infllm_official.h"
#include "infllm_bwd_params.h"
#include "../runtime/rt_engine.cuh"
#include "../runtime/rt_sched.cuh"

#ifndef DIST_SPIN_REPORT
#define DIST_SPIN_REPORT (1L << 26)
#endif

namespace infllm_bwd {

// the official sm86 traits (run_mha_bwd_hdim128, max dynamic smem < 144 KB branch): 16-row m blocks (= one token's 16
// query heads), 64-key columns, 4 warps, V in registers, no double buffer
using KT = Infllm_bwd_kernel_traits<128, 16, 64, 4, 1, 4, 1, /*Is_V_in_regs=*/true, /*No_double_buffer=*/true, InfllmElem>;
static_assert(KT::kNThreads == 128, "block 0 = 1 scheduler warp + 3 engine warps");

// Plain launch arguments (filled by the host from tensors): the official varlen backward's inputs.
struct FArgs {
    void *q, *k, *v, *o, *dout, *dq;            // bf16: q/o/dout/dq [total_q, 2, 128] (group rows); k/v: global layout rows
    int64_t q_row_stride, k_row_stride, o_row_stride, do_row_stride, dq_row_stride;   // elements (head stride = 128)
    float *lse, *dpsum, *dq_accum;              // [2, total_q]; [2, total_q + 128 b]; [total_q + 128 b, 2, 128]
    float *dk_accum, *dv_accum; int64_t acc_row_stride;
    const int *cu_q, *cu_k, *seqused_k; int b;
    const uint64_t *blockmask; int num_blocks_m, num_blocks_n;
    int total_q, max_seqlen_q, max_seqlen_k; float scale;
};

// Mirrors mha_varlen_bwd + set_params_dgrad (flash_api.cpp) for: bf16, d = 128, causal, no dropout / alibi / softcap /
// paging, column blockmask, unpadded LSE, dq_accum / dsoftmax_sum with 128 padding rows per segment.
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
    p.is_causal = true; p.window_size_left = a.max_seqlen_k; p.window_size_right = 0;   // causal: (-1, 0) -> (seqlen_k, 0)
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

__device__ __forceinline__ unsigned long long bwd_now() {
    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
#define ITR(p, i, v) do { if ((p).trace && threadIdx.x == 0) atomicAdd(&(p).trace[i], (unsigned long long)(v)); } while (0)

// Worker control word in DYNAMIC shared memory past the compute routine's area (no static __shared__: it would shift the
// routine's swizzled tiles, see dist_bwd_kernel.h).
struct Ctl { int entry; };
constexpr int kCtlBytes = 128;

// Block n complete here (its last task done): send / book its partial sum (out of line, as dist_bwd::complete_task).
__device__ __noinline__ void slot_done(const Params &p, int n) {
    if (p.ks_agg_need[n] > 0) rt::red_count(p.sch, p.n_km + p.ka_unit[n]);      // this PE is the node gateway
    else if (p.ks_push_pe[n] >= 0) rt::fire(p.eng, n);                           // push to the owner / gateway
    else rt::red_count(p.sch, p.ks_push_dst[n]);                                 // own block: the owner sum
}

__device__ __noinline__ void complete_task(const Params &p, int r) {
    if (threadIdx.x == 0) {
        __threadfence();                                     // release: this block's atomics before the count
        const int n = p.task_n[r];
        if (atomicAdd(&p.ks_done[n], 1) + 1 == p.ks_need[n]) { __threadfence(); slot_done(p, n); }
        __threadfence();
        atomicAdd(p.sch.rows_done, 1);                       // one row = one task
    }
    __syncthreads();
}

__device__ __forceinline__ float4 add4(float4 a, const float4 b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; return a; }

// Owner sum m: block km_blk[m] = own accumulator + landings -> bf16 dK (row floats [0, 256)) / dV ([256, 512)) at the
// output rows km_out[m] .. (km_rows[m] valid rows), in the compute dtype.
__device__ __noinline__ void ksum(const Params &p, int m) {
    const int n = p.km_blk[m], rows = p.km_rows[m], l0 = p.km_ptr[m], nl = p.km_ptr[m + 1] - l0;
    const float4 *acc = reinterpret_cast<const float4 *>(p.base + p.acc_off + (int64_t)n * kSlotBytes);
    InfllmElem *dk = reinterpret_cast<InfllmElem *>(p.dk_out) + (int64_t)p.km_out[m] * 256;
    InfllmElem *dv = reinterpret_cast<InfllmElem *>(p.dv_out) + (int64_t)p.km_out[m] * 256;
    for (int e = threadIdx.x; e < rows * (kRowFloats / 4); e += blockDim.x) {
        float4 a = acc[e];
        for (int k = 0; k < nl; ++k)
            a = add4(a, reinterpret_cast<const float4 *>(p.base + p.land_off + (int64_t)p.km_land[l0 + k] * kSlotBytes)[e]);
        const int r = e / (kRowFloats / 4), c4 = e % (kRowFloats / 4);        // 128 float4 per row: dK [0, 64), dV [64, 128)
        InfllmElem *o = (c4 < 64 ? dk : dv) + (int64_t)r * 256 + (c4 & 63) * 4;
        o[0] = InfllmElem(a.x); o[1] = InfllmElem(a.y); o[2] = InfllmElem(a.z); o[3] = InfllmElem(a.w);
    }
    __syncthreads();
}

// Gateway aggregation u: the node peers' landings of block ka_blk[u] into the own accumulator, then push it to the owner.
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

__global__ void __launch_bounds__(KT::kNThreads) infllm_bwd_kernel(const __grid_constant__ FParams fp, const __grid_constant__ Params p) {
    extern __shared__ char smem_[];
    const int tidx = threadIdx.x;
    if (blockIdx.x == 0) {
        if (tidx < 32) rt::sched_block0<0>(p.sch, smem_, *p.sch_b0, &p.eng);
        else rt::engine(p.eng, 32, blockDim.x - 32, 1, reinterpret_cast<volatile int *>(smem_ + KT::kSmemSize1colblock));
        return;
    }
    Ctl &c = *reinterpret_cast<Ctl *>(smem_ + KT::kSmemSize1colblock);
    if (p.trace && tidx == 0) atomicMin(&p.trace[5], bwd_now());
    for (;;) {
        const unsigned long long tw0 = p.trace ? bwd_now() : 0;
        if (tidx == 0) {
            const int pos = atomicAdd(p.tile_counter, 1);
            const volatile int *q = p.sq + pos;
            long n = 0;
            while (*q == 0) {
                __nanosleep(64);
                if (++n % DIST_SPIN_REPORT == 0) printf("[pe %d blk %d] infllm bwd queue slot %d empty\n", nvshmem_my_pe(), blockIdx.x, pos);
            }
            c.entry = *q - 1;
            __threadfence();
        }
        __syncthreads();
        const int entry = c.entry;
        __syncthreads();
        const int kind = entry >> 28, pay = entry & ((1 << 28) - 1);
        const unsigned long long tw1 = p.trace ? bwd_now() : 0;
        ITR(p, 0, tw1 - tw0);
        if (kind == kStop) break;
        if (kind == kTask) {
            const int r = p.sch.task_desc[4 * pay];      // the scheduler's task = one row = (segment, kv head, block)
            // template args of the official varlen causal launch (run_flash_bwd_seqk_parallel): no dropout, causal,
            // not local, no alibi, uneven MN (varlen), even K, no softcap, Is_first = Is_last = false, Seq_parallel
            infllm_flash::compute_dq_dk_dv_1colblock_acc<KT, false, true, false, false, false, true, false, false, false, true>(
                fp, p.task_b[r], p.task_h[r], p.task_n[r]);
            const unsigned long long tw2 = p.trace ? bwd_now() : 0;
            ITR(p, 1, tw2 - tw1); ITR(p, 4, 1);
            complete_task(p, r);
            ITR(p, 2, (p.trace ? bwd_now() : 0) - tw2);
        } else if (kind == kKSum) {
            ksum(p, pay);
            ITR(p, 3, (p.trace ? bwd_now() : 0) - tw1);
        } else if (kind == kKAgg) {
            kagg(p, pay);
            ITR(p, 3, (p.trace ? bwd_now() : 0) - tw1);
        }                                                    // kNop: nothing
    }
    if (p.trace && tidx == 0) atomicMax(&p.trace[6], bwd_now());
}

// ---- the official preprocess / postprocess kernels (flash_bwd_launch_template.h instantiations) ----------------------
__global__ void __launch_bounds__(KT::kNThreads) dot_do_o_kernel(const __grid_constant__ FParams fp) {
    infllm_flash::compute_dot_do_o</*Clear_dQaccum=*/true, KT>(fp);
}
__global__ void __launch_bounds__(KT::kNThreads) convert_dq_kernel(const __grid_constant__ FParams fp) {
    infllm_flash::convert_dQ<KT>(fp, 1);
}

// Column bitmap of the official backward from the top-k selection: sel [2, L, K] int32 (block ids, -1 = none), pos [L]
// the token's global position; bit t of row (head, block) set when token t selected the block and the block is not past
// the token's own block (the forward's filter idx[idx > qb] = -1). bm [2, nbn, upr] int64, zeroed by the caller.
__global__ void col_bitmap_kernel(const int *__restrict__ sel, const int *__restrict__ pos, int L, int K, int nbn, int upr,
                                  unsigned long long *__restrict__ bm) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= 2LL * L * K) return;
    const int h = (int)(i / ((int64_t)L * K)), rem = (int)(i % ((int64_t)L * K)), t = rem / K;
    const int blk = sel[i];
    if (blk < 0 || blk > pos[t] / 64) return;
    atomicOr(&bm[((int64_t)h * nbn + blk) * upr + (t >> 6)], 1ull << (t & 63));
}

}  // namespace infllm_bwd

using infllm_bwd::KT;

int infllm_bwd_smem_bytes() { return KT::kSmemSize1colblock + infllm_bwd::kCtlBytes; }

// one scheduler block + every resident CTA (the kernel spins: a non-resident CTA would never run)
int infllm_bwd_grid() {
    static int g = 0;
    if (!g) {
        int per = 0, nsm = 0, dev = 0;
        cudaGetDevice(&dev);
        cudaFuncSetAttribute(infllm_bwd::infllm_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, infllm_bwd_smem_bytes());
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, infllm_bwd::infllm_bwd_kernel, KT::kNThreads, infllm_bwd_smem_bytes());
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        g = per * nsm;
    }
    return g;
}

void infllm_bwd_launch(const infllm_bwd::FArgs &a, const infllm_bwd::Params &p, int grid, cudaStream_t stream) {
    infllm_bwd::FParams fp;
    infllm_bwd::fill(fp, a);
    infllm_bwd_grid();                                       // (sets the smem attribute once)
    infllm_bwd::infllm_bwd_kernel<<<grid, KT::kNThreads, infllm_bwd_smem_bytes(), stream>>>(fp, p);
}

void infllm_bwd_prep_launch(const infllm_bwd::FArgs &a, cudaStream_t stream) {   // dpsum = rowsum(dO * O), dq_accum = 0
    infllm_bwd::FParams fp;
    infllm_bwd::fill(fp, a);
    dim3 grid((a.max_seqlen_q + KT::kBlockM - 1) / KT::kBlockM, a.b, 2);
    infllm_bwd::dot_do_o_kernel<<<grid, KT::kNThreads, 0, stream>>>(fp);
}

void infllm_bwd_convert_launch(const infllm_bwd::FArgs &a, cudaStream_t stream) {   // dq = scale * dq_accum (un-permuted) -> bf16
    infllm_bwd::FParams fp;
    infllm_bwd::fill(fp, a);
    static bool set = false;
    if (!set) { cudaFuncSetAttribute(infllm_bwd::convert_dq_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, KT::kSmemdQSize); set = true; }
    dim3 grid((a.max_seqlen_q + KT::kBlockM - 1) / KT::kBlockM, a.b, 2);
    infllm_bwd::convert_dq_kernel<<<grid, KT::kNThreads, KT::kSmemdQSize, stream>>>(fp);
}

void infllm_col_bitmap_launch(const int *sel, const int *pos, int L, int K, int nbn, int upr, unsigned long long *bm, cudaStream_t stream) {
    const int64_t n = 2LL * L * K;
    if (n) infllm_bwd::col_bitmap_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(sel, pos, L, K, nbn, upr, bm);
}
