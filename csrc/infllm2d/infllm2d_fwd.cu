// 2-D InfLLM-v2 forward megakernel. Derived from
// csrc/infllm_fwd/infllm_fwd.cu (the query-stationary executor) with FOREIGN rows: rows of moved units
// computed here for another rank's queries, by the SAME official row routine called with the unit's own parameter block,
// and a completion count per unit that fires the trigger returning the unit's FP32 partial O | LSE to its query owner.
// Own rows: decode, routine and parameters exactly as csrc/infllm_fwd.
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cutlass/numeric_types.h>
#define INFLLM_WANT_FWD
#include "../infllm_bwd/infllm_official.h"
#include "infllm2d_fwd_params.h"
#include "../runtime/rt_engine.cuh"
#include "../runtime/rt_sched.cuh"

#ifndef DIST_SPIN_REPORT
#define DIST_SPIN_REPORT (1L << 26)
#endif

namespace infllm2d_fwd {

// the official hdim128 bf16 forward traits: one warp, 16 query rows (= one token's 16 heads of one KV head), 64 keys
using KT = Infllm_fwd_kernel_traits<128, 16, 64, 1, false, false, InfllmElem>;
struct FParams : Infllm_fwd_params { void *o32_ptr; };   // + fp32 partial O (as infllm_fwd::FParams)
static_assert(KT::kNThreads == 32, "the official routine is single-warp");
constexpr int kComputeWarps = 2, kThreads = 128;                 // 2 x 36 KB slices (as the official's 2 CTAs per SM)
constexpr int kSmemCompute = kComputeWarps * KT::kSmemSize;
constexpr int kCtlBytes = 128;

// (as infllm_fwd.cu fill: bf16, d = 128, causal, no dropout / alibi / softcap / paging, InfLLM-v2 row blockmask)
static void fill(FParams &p, const FArgs &a) {
    p = {};
    p.is_bf16 = INFLLM_IS_BF16;
    p.q_ptr = a.q; p.k_ptr = a.k; p.v_ptr = a.v; p.o_ptr = a.o;
    p.q_row_stride = a.q_row_stride; p.k_row_stride = a.k_row_stride; p.v_row_stride = a.k_row_stride; p.o_row_stride = a.o_row_stride;
    p.q_head_stride = p.k_head_stride = p.v_head_stride = p.o_head_stride = 128;
    p.cu_seqlens_q = const_cast<int *>(a.cu_q); p.cu_seqlens_k = const_cast<int *>(a.cu_k);
    p.seqused_k = const_cast<int *>(a.seqused_k);
    p.p_ptr = nullptr; p.softmax_lse_ptr = a.lse;
    p.b = a.b; p.h = 2; p.h_k = 2; p.h_h_k_ratio = 1;
    p.seqlen_q = a.max_seqlen_q; p.seqlen_k = a.max_seqlen_k;
    p.seqlen_q_rounded = (a.max_seqlen_q + 127) / 128 * 128; p.seqlen_k_rounded = (a.max_seqlen_k + 127) / 128 * 128;
    p.d = 128; p.d_rounded = 128;
    p.softcap = 0.f; p.scale_softmax = a.scale; p.scale_softmax_log2 = a.scale * M_LOG2E;
    p.p_dropout = 1.f; p.p_dropout_in_uint8_t = 255; p.rp_dropout = 1.f; p.scale_softmax_rp_dropout = a.scale;
    p.is_causal = true; p.window_size_left = a.max_seqlen_k; p.window_size_right = 0;
    p.is_seqlens_k_cumulative = true; p.unpadded_lse = true; p.seqlenq_ngroups_swapped = false;
    p.total_q = a.total_q;
    p.blockmask = const_cast<uint64_t *>(a.blockmask);
    p.m_block_dim = 16; p.n_block_dim = 64; p.num_k_heads = 2;
    p.num_blocks_m = a.num_blocks_m; p.num_blocks_n = a.num_blocks_n;
    p.page_block_size = 1; p.alibi_slopes_ptr = nullptr; p.num_splits = 1;
    p.o32_ptr = a.o32;
}

// One moved unit = the official varlen causal call of ONE chunk (batch 1) for ONE KV-head group: h = h_k = 1 (the
// group's 16 query heads are the unit's 16 rows per token; K / V pointers at the group's column of the global K | V rows),
// q [len * 16, 128], FP32 partial O [len * 16, 128] + LSE [len * 16], row blockmask [len, ceil(nbn / 64)].
static void fill_unit(FParams &p, const UnitDesc &u) {
    FArgs a{};
    a.q = u.q; a.k = u.k; a.v = u.v; a.o = u.o32; a.o32 = u.o32;
    a.q_row_stride = a.o_row_stride = 128; a.k_row_stride = 512;
    a.lse = reinterpret_cast<float *>(u.lse);
    a.cu_q = u.cu_q; a.cu_k = u.cu_k; a.seqused_k = u.seqused; a.b = 1;
    a.blockmask = reinterpret_cast<const uint64_t *>(u.bm);
    a.total_q = u.len * 16; a.num_blocks_m = u.len; a.num_blocks_n = u.nbn;
    a.max_seqlen_q = u.len * 16; a.max_seqlen_k = u.max_seqlen_k; a.scale = u.scale;
    fill(p, a);
    p.h = 1; p.h_k = 1; p.h_h_k_ratio = 1; p.num_k_heads = 1;
}

__device__ __forceinline__ unsigned long long fwd_now() {
    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}

__global__ void __launch_bounds__(kThreads, 1) infllm2d_fwd_kernel(const __grid_constant__ FParams fp, const __grid_constant__ Params p) {
    extern __shared__ char smem_[];
    const int tidx = threadIdx.x, w = tidx >> 5, lane = tidx & 31;
    if (blockIdx.x == 0) {
        if (w == 0) rt::sched_block0<0>(p.sch, smem_, *p.sch_b0, &p.eng);
        else if (w == 3) rt::engine(p.eng, 96, 32, 1, reinterpret_cast<volatile int *>(smem_ + kSmemCompute));
        return;
    }
    if (w >= kComputeWarps) return;                              // (2 compute warps per CTA: 2 x 36 KB)
    char *smem = smem_ + w * KT::kSmemSize;
    if (p.trace && tidx == 0) atomicMin(&p.trace[3], fwd_now());
    const FParams *fpu = reinterpret_cast<const FParams *>(p.fpu);
    for (;;) {
        const unsigned long long tw0 = p.trace ? fwd_now() : 0;
        int entry = 0;
        if (lane == 0) {
            const int pos = atomicAdd(p.tile_counter, 1);
            const volatile int *q = p.sq + pos;
            long n = 0;
            while (*q == 0) {
                __nanosleep(64);
                if (++n % DIST_SPIN_REPORT == 0) printf("[pe %d blk %d] infllm2d fwd queue slot %d empty\n", nvshmem_my_pe(), blockIdx.x, pos);
            }
            entry = *q - 1;
            __threadfence();
        }
        entry = __shfl_sync(0xffffffffu, entry, 0);
        const int kind = entry >> 28, pay = entry & ((1 << 28) - 1);
        const unsigned long long tw1 = p.trace ? fwd_now() : 0;
        if (p.trace && lane == 0) atomicAdd(&p.trace[0], tw1 - tw0);
        if (kind == kStop) break;
        if (kind != kTask) continue;                              // kNop (incl. the partial-landing units)
        const int r = p.sch.task_desc[4 * pay];
        if (r >= p.n_rows_own) {                                  // [2-D] a foreign row: unit u, tokens [m0, m1) of it
            const int32_t *fr = p.frow + 3 * (r - p.n_rows_own);
            const int u = fr[0], m0 = fr[1], m1 = fr[2];
            for (int m = m0; m < m1; ++m)
                infllm_flash::compute_attn_1rowblock_w<KT, false, true, false, false, false, true, false, false>(smem, fpu[u], 0, 0, m);
            __syncwarp();
            if (p.trace && lane == 0) { atomicAdd(&p.trace[1], fwd_now() - tw1); atomicAdd(&p.trace[2], m1 - m0); }
            if (lane == 0) {
                __threadfence();
                if (atomicAdd(&p.unit_done[u], 1) + 1 == p.unit_need[u]) { __threadfence(); rt::fire(p.eng, p.trig_unit0 + u); }
                atomicAdd(p.sch.rows_done, 1);
            }
            __syncwarp();
            continue;
        }
        // one task = one row = (part, head, up to 8 tokens): as csrc/infllm_fwd
        const int hp = r / p.nrh, i = r - hp * p.nrh, h = hp & 1, part = hp >> 1;   // rows: part-major, then head
        const int t0_ = part * p.L + i * kTokensPerRow, t1_ = min(t0_ + kTokensPerRow, (part + 1) * p.L);
        int b = 0;
        for (int t = t0_; t < t1_; ++t) {
            while (b + 1 < p.n_seg && p.cu_q[b + 1] <= t * 16) ++b;
            infllm_flash::compute_attn_1rowblock_w<KT, false, true, false, false, false, true, false, false>(smem, fp, b, h, t - p.cu_q[b] / 16);
        }
        __syncwarp();
        if (p.trace && lane == 0) { atomicAdd(&p.trace[1], fwd_now() - tw1); atomicAdd(&p.trace[2], t1_ - t0_); }
        if (lane == 0) { __threadfence(); atomicAdd(p.sch.rows_done, 1); }
        __syncwarp();
    }
    if (p.trace && lane == 0) atomicMax(&p.trace[4], fwd_now());
}

// (as infllm_fwd.cu) the local / remote split bitmap of the own rows
__global__ void split_bitmap_kernel(const int *__restrict__ sel, const uint8_t *__restrict__ is_local, int L, int K, int upr,
                                    unsigned long long *__restrict__ bm) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= 2LL * L * K) return;
    const int blk = sel[i];
    if (blk < 0) return;
    const int h = (int)(i / ((int64_t)L * K)), t = (int)((i / K) % L), part = is_local[blk] ? 0 : 1;
    atomicOr(&bm[((int64_t)h * 2 * L + (int64_t)part * L + t) * upr + (blk >> 6)], 1ull << (blk & 63));
}

// [2-D] the official ROW bitmap of a token range of one group, restricted to the entries on GPU `want`:
// sel / egpu [L, K] (one group, the range's rows; egpu = the entry's GPU), out [L, upr] u64, zeroed by the caller.
__global__ void row_bitmap_kernel(const int *__restrict__ sel, const int16_t *__restrict__ egpu, int want, int L, int K,
                                  int upr, unsigned long long *__restrict__ out) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= (int64_t)L * K) return;
    const int blk = sel[i];
    if (blk < 0 || egpu[i] != want) return;
    atomicOr(&out[(i / K) * upr + (blk >> 6)], 1ull << (blk & 63));
}

// [2-D] the official COLUMN bitmap (backward) of a token range of one group, restricted likewise: out [nbn, ceil(L / 64)]
// u64 zeroed by the caller; token t of the range = bit t (the unit's own token numbering). sel is causal-filtered.
__global__ void col_bitmap_kernel(const int *__restrict__ sel, const int16_t *__restrict__ egpu, int want, int L, int K,
                                  int nbn, unsigned long long *__restrict__ out) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= (int64_t)L * K) return;
    const int blk = sel[i];
    if (blk < 0 || blk >= nbn || egpu[i] != want) return;
    const int t = (int)(i / K), upc = (L + 63) / 64;
    atomicOr(&out[(int64_t)blk * upc + (t >> 6)], 1ull << (t & 63));
}

}  // namespace infllm2d_fwd

void infllm2d_split_bitmap_launch(const int *sel, const uint8_t *is_local, int L, int K, int upr, unsigned long long *bm, cudaStream_t stream) {
    const int64_t n = 2LL * L * K;
    if (n) infllm2d_fwd::split_bitmap_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(sel, is_local, L, K, upr, bm);
}

void infllm2d_unit_bitmap_launch(bool col, const int *sel, const int16_t *egpu, int want, int L, int K, int nbn,
                                 unsigned long long *out, cudaStream_t stream) {
    const int64_t n = (int64_t)L * K;
    if (!n) return;
    if (col) infllm2d_fwd::col_bitmap_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(sel, egpu, want, L, K, nbn, out);
    else infllm2d_fwd::row_bitmap_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(sel, egpu, want, L, K, (nbn + 63) / 64, out);
}

int infllm2d_fwd_smem_bytes() { return infllm2d_fwd::kSmemCompute + infllm2d_fwd::kCtlBytes; }

int infllm2d_fwd_grid() {
    static int g = 0;
    if (!g) {
        int per = 0, nsm = 0, dev = 0;
        cudaGetDevice(&dev);
        cudaFuncSetAttribute(infllm2d_fwd::infllm2d_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, infllm2d_fwd_smem_bytes());
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, infllm2d_fwd::infllm2d_fwd_kernel, infllm2d_fwd::kThreads, infllm2d_fwd_smem_bytes());
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        g = per * nsm;
    }
    return g;
}

int64_t infllm2d_fwd_fparams_bytes() { return (int64_t)sizeof(infllm2d_fwd::FParams); }

// per incoming unit: its official parameter block, written to dev (device memory, n * sizeof(FParams) bytes)
void infllm2d_fwd_build_units(const infllm2d_fwd::UnitDesc *d, int n, void *dev, cudaStream_t stream) {
    if (!n) return;
    std::vector<infllm2d_fwd::FParams> h(n);
    for (int i = 0; i < n; ++i) infllm2d_fwd::fill_unit(h[i], d[i]);
    cudaMemcpyAsync(dev, h.data(), (size_t)n * sizeof(infllm2d_fwd::FParams), cudaMemcpyHostToDevice, stream);   // (pageable: staged now)
}

void infllm2d_fwd_launch(const infllm2d_fwd::FArgs &a, const infllm2d_fwd::Params &p, int grid, cudaStream_t stream) {
    infllm2d_fwd::FParams fp;
    infllm2d_fwd::fill(fp, a);
    infllm2d_fwd_grid();
    infllm2d_fwd::infllm2d_fwd_kernel<<<grid, infllm2d_fwd::kThreads, infllm2d_fwd_smem_bytes(), stream>>>(fp, p);
}
