// InfLLM-v2 forward megakernel: the OFFICIAL row routine as the compute, the
// shared block-0 scheduler / engine (csrc/runtime) as the runtime, one launch per GPU. Mirror of the backward
// (csrc/infllm_bwd): Q stays home, K/V runs come in, O / LSE are written directly (nothing to reduce across GPUs;
// with the local / remote split the two fp32 partials of a token are merged on this GPU).
#include <cuda_runtime.h>
#include <cstdio>
#include <cutlass/numeric_types.h>
#define INFLLM_WANT_FWD
#include "../infllm_bwd/infllm_official.h"
#include "infllm_fwd_params.h"
#include "../runtime/rt_engine.cuh"
#include "../runtime/rt_sched.cuh"

#ifndef DIST_SPIN_REPORT
#define DIST_SPIN_REPORT (1L << 26)
#endif

namespace infllm_fwd {

// the official hdim128 bf16 forward traits: one warp, 16 query rows (= one token's 16 heads of one KV head), 64 keys
using KT = Infllm_fwd_kernel_traits<128, 16, 64, 1, false, false, InfllmElem>;
struct FParams : Infllm_fwd_params { void *o32_ptr; };   // + fp32 partial O (local / remote split)
static_assert(KT::kNThreads == 32, "the official routine is single-warp");
constexpr int kComputeWarps = 2, kThreads = 128;                 // 2 x 36 KB slices (as the official's 2 CTAs per SM)
constexpr int kSmemCompute = kComputeWarps * KT::kSmemSize;
constexpr int kCtlBytes = 128;

struct FArgs {                                   // plain launch arguments (filled by the host from tensors)
    void *q, *k, *v, *o, *o32;                   // bf16: q / o [total_q, 2, 128] (group rows); k / v: global layout rows;
                                                 // o32: fp32 partial O [total_q, 2, 128] (nullptr: the bf16 o)
    int64_t q_row_stride, k_row_stride, o_row_stride;
    float *lse;                                  // [2, total_q]
    const int *cu_q, *cu_k, *seqused_k; int b;
    const uint64_t *blockmask; int num_blocks_m, num_blocks_n;
    int total_q, max_seqlen_q, max_seqlen_k; float scale;
};

// Mirrors mha_varlen_fwd + set_params_fprop for: bf16, d = 128, causal, no dropout / alibi / softcap / paging,
// InfLLM-v2 row blockmask.
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

__device__ __forceinline__ unsigned long long fwd_now() {
    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}

__global__ void __launch_bounds__(kThreads, 1) infllm_fwd_kernel(const __grid_constant__ FParams fp, const __grid_constant__ Params p) {
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
    for (;;) {
        const unsigned long long tw0 = p.trace ? fwd_now() : 0;
        int entry = 0;
        if (lane == 0) {
            const int pos = atomicAdd(p.tile_counter, 1);
            const volatile int *q = p.sq + pos;
            long n = 0;
            while (*q == 0) {
                __nanosleep(64);
                if (++n % DIST_SPIN_REPORT == 0) printf("[pe %d blk %d] infllm fwd queue slot %d empty\n", nvshmem_my_pe(), blockIdx.x, pos);
            }
            entry = *q - 1;
            __threadfence();
        }
        entry = __shfl_sync(0xffffffffu, entry, 0);
        const int kind = entry >> 28, pay = entry & ((1 << 28) - 1);
        const unsigned long long tw1 = p.trace ? fwd_now() : 0;
        if (p.trace && lane == 0) atomicAdd(&p.trace[0], tw1 - tw0);
        if (kind == kStop) break;
        if (kind != kTask) continue;                              // kNop
        const int r = p.sch.task_desc[4 * pay];                   // one task = one row = (part, head, up to 8 tokens)
        const int hp = r / p.nrh, i = r - hp * p.nrh, h = hp & 1, part = hp >> 1;   // rows: part-major, then head
        const int t0_ = part * p.L + i * kTokensPerRow, t1_ = min(t0_ + kTokensPerRow, (part + 1) * p.L);
        int b = 0;
        for (int t = t0_; t < t1_; ++t) {
            while (b + 1 < p.n_seg && p.cu_q[b + 1] <= t * 16) ++b;
            // template args of the official varlen causal launch (run_mha_fwd_hdim128): no dropout, causal, not local,
            // no alibi, uneven MN (varlen), even K, no softcap, no returned softmax
            infllm_flash::compute_attn_1rowblock_w<KT, false, true, false, false, false, true, false, false>(smem, fp, b, h, t - p.cu_q[b] / 16);
        }
        __syncwarp();
        if (p.trace && lane == 0) { atomicAdd(&p.trace[1], fwd_now() - tw1); atomicAdd(&p.trace[2], t1_ - t0_); }
        if (lane == 0) { __threadfence(); atomicAdd(p.sch.rows_done, 1); }
        __syncwarp();
    }
    if (p.trace && lane == 0) atomicMax(&p.trace[4], fwd_now());
}

// The official row bitmap of the local / remote split, straight from the (causal-filtered) selection: sel [2, L, K]
// block ids (-1 = none); bm int64 [2, 2 L, upr] zeroed by the caller: row (head, part * L + token), part 0 = blocks owned
// here (is_local[blk]), part 1 = the others.
__global__ void split_bitmap_kernel(const int *__restrict__ sel, const uint8_t *__restrict__ is_local, int L, int K, int upr,
                                    unsigned long long *__restrict__ bm) {
    const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= 2LL * L * K) return;
    const int blk = sel[i];
    if (blk < 0) return;
    const int h = (int)(i / ((int64_t)L * K)), t = (int)((i / K) % L), part = is_local[blk] ? 0 : 1;
    atomicOr(&bm[((int64_t)h * 2 * L + (int64_t)part * L + t) * upr + (blk >> 6)], 1ull << (blk & 63));
}

}  // namespace infllm_fwd

void infllm_split_bitmap_launch(const int *sel, const uint8_t *is_local, int L, int K, int upr, unsigned long long *bm, cudaStream_t stream) {
    const int64_t n = 2LL * L * K;
    if (n) infllm_fwd::split_bitmap_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(sel, is_local, L, K, upr, bm);
}

int infllm_fwd_smem_bytes() { return infllm_fwd::kSmemCompute + infllm_fwd::kCtlBytes; }

int infllm_fwd_grid() {
    static int g = 0;
    if (!g) {
        int per = 0, nsm = 0, dev = 0;
        cudaGetDevice(&dev);
        cudaFuncSetAttribute(infllm_fwd::infllm_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, infllm_fwd_smem_bytes());
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, infllm_fwd::infllm_fwd_kernel, infllm_fwd::kThreads, infllm_fwd_smem_bytes());
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        g = per * nsm;
    }
    return g;
}

void infllm_fwd_launch(const infllm_fwd::FArgs &a, const infllm_fwd::Params &p, int grid, cudaStream_t stream) {
    infllm_fwd::FParams fp;
    infllm_fwd::fill(fp, a);
    infllm_fwd_grid();
    infllm_fwd::infllm_fwd_kernel<<<grid, infllm_fwd::kThreads, infllm_fwd_smem_bytes(), stream>>>(fp, p);
}
