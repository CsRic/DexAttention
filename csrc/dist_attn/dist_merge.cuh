#pragma once

// Log-sum-exp merge of the partial outputs a Q row collected from its
// contributing devices. Same math as FA3's flash_fwd_combine_kernel, written
// compactly so it can run as a work item INSIDE the persistent kernel rather
// than as a second launch:
//
//   lse_max = max_s lse_s,  scale_s = exp(lse_s - lse_max)
//   lse_out = log(sum_s scale_s) + lse_max
//   O       = sum_s (scale_s / sum scale) * O_s
//
// Commutative and associative, so arrival order does not matter -- which is
// what lets a merge start as soon as its partials are in, without ordering
// the senders.
//
// Layouts (what EpiloguePart / Epilogue write):
//   part_o   TIn [slot * 128 + row][h][d]   (already normalised by 1/row_sum; fp32 in the distributed kernels,
//            fp16 in the standalone merge test)
//   part_lse fp32 [h][slot * 128 + row]

#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_fp16.h>
#include "dist_merge_params.h"
#include "dist_scatter_params.h"

namespace ring_attn {

// One (Q tile, head), any number of partial slots. 256 threads: one thread per
// row for the two per-row statistics (lse_max, 1/sum), then the whole block on
// the 128 x d accumulation. Only the two statistics live in shared memory
// (2 * 128 floats); each slot's weight is recomputed from part_lse inside the
// accumulation. That read is a warp-wide broadcast (all 32 lanes of a warp sit
// on the same row), so recomputing costs one exp per element per slot and no
// per-slot shared-memory budget -- there is no cap on the slot count.
// Layout of a set of 128-row partial (or output) tiles: element of (slot, row r, head h, column c) of O is at
// slot * o_slot + r * nheads * kHeadDim + h * kHeadDim + c, LSE of (slot, h, r) at slot * lse_slot + h * lse_head + r.
// The final output / head-major partials: o_slot = 128 * nheads * kHeadDim, lse_slot = 128, lse_head = tokens.
// Slot-major partials (O then LSE in one contiguous slot, dist_sched_kernel): o_slot / lse_slot = the slot size.
struct MergeLayout { int64_t o_slot, lse_slot, lse_head; };

__device__ __forceinline__ float merge_to_float(float x) { return x; }
__device__ __forceinline__ float merge_to_float(__half x) { return __half2float(x); }
template <typename T> __device__ __forceinline__ T merge_from_float(float x);
template <> __device__ __forceinline__ float merge_from_float<float>(float x) { return x; }
template <> __device__ __forceinline__ __half merge_from_float<__half>(float x) { return __float2half(x); }

// TIn = partial O element (fp32 partials of the dist kernels, fp16 in the standalone test), TOut = output element
// (fp16 final output, or fp32 when a local merge produces the partial that is sent to the Q owner).
template<int kBlockM = 128, int kHeadDim = 128, int kNThreads = 256, typename TIn = __half, typename TOut = __half>
__device__ __forceinline__ void dist_merge_tile(
    void   *__restrict__ o_raw,          // output O (TOut)
    float  *__restrict__ lse_ptr,        // output LSE
    const TIn *__restrict__ part_o,
    const float *__restrict__ part_lse,
    const int32_t *__restrict__ slots, int n_slots,
    int out_row,                         // output slot (which Q tile of the owner's output)
    int bidh, int nheads, MergeLayout in, MergeLayout out,
    float *__restrict__ smem_stats       // [2 * kBlockM] scratch: lse_max | 1/sum
) {
    const int tid = threadIdx.x;
    TOut *__restrict__ o_ptr = reinterpret_cast<TOut *>(o_raw);
    float *__restrict__ s_max = smem_stats;
    float *__restrict__ s_inv = smem_stats + kBlockM;
    const float *__restrict__ lse_h = part_lse + (size_t)bidh * in.lse_head;

    // ---- per-row statistics ---------------------------------------------- //
    for (int r = tid; r < kBlockM; r += kNThreads) {
        float lse_max = -INFINITY;
        #pragma unroll 1
        for (int s = 0; s < n_slots; ++s) {
            lse_max = fmaxf(lse_max, lse_h[(size_t)slots[s] * in.lse_slot + r]);
        }
        const float lm = (lse_max == -INFINITY) ? 0.f : lse_max;
        float ssum = 0.f;
        #pragma unroll 1
        for (int s = 0; s < n_slots; ++s) {
            ssum += __expf(lse_h[(size_t)slots[s] * in.lse_slot + r] - lm);
        }
        const bool empty = (ssum == 0.f || ssum != ssum);
        s_max[r] = lm;
        s_inv[r] = empty ? 0.f : 1.f / ssum;
        lse_ptr[(size_t)out_row * out.lse_slot + (size_t)bidh * out.lse_head + r] =
            empty ? -INFINITY : (__logf(ssum) + lm);
    }
    __syncthreads();

    // ---- accumulate O ---------------------------------------------------- //
    const int total = kBlockM * kHeadDim;
    for (int i = tid; i < total; i += kNThreads) {
        const int r = i / kHeadDim, c = i % kHeadDim;
        const float lm = s_max[r], inv = s_inv[r];
        float acc = 0.f;
        #pragma unroll 1
        for (int s = 0; s < n_slots; ++s) {
            const float w = __expf(lse_h[(size_t)slots[s] * in.lse_slot + r] - lm) * inv;
            const size_t off = (size_t)slots[s] * in.o_slot + (size_t)r * nheads * kHeadDim
                             + (size_t)bidh * kHeadDim + c;
            acc += w * merge_to_float(part_o[off]);
        }
        const size_t oo = (size_t)out_row * out.o_slot + (size_t)r * nheads * kHeadDim
                        + (size_t)bidh * kHeadDim + c;
        o_ptr[oo] = merge_from_float<TOut>(acc);
    }
}

// Shared memory dist_merge_tile needs, in bytes.
constexpr int kMergeSmemBytes = 2 * 128 * (int)sizeof(float);

// Standalone kernel, used by the single-GPU numerics test. In the megakernel
// this body runs as a work item instead of a launch.
__global__ __launch_bounds__(256, 1)
void dist_merge_kernel(__grid_constant__ const Merge_params p) {
    extern __shared__ float smem_stats[];
    const int item = blockIdx.x;
    const int bidh = blockIdx.y;
    if (item >= p.n_merge) return;
    const int beg = p.merge_ptr[item], end = p.merge_ptr[item + 1];
    const int64_t tile = 128LL * p.nheads * 128;
    dist_merge_tile<>(p.o_ptr, p.lse_ptr, p.part_o, p.part_lse,
                      p.merge_slot + beg, end - beg, p.merge_out[item],
                      bidh, p.nheads, MergeLayout{tile, 128, p.lse_stride}, MergeLayout{tile, 128, p.lse_out_stride},
                      smem_stats);
}

// Stage owned tiles into their slots in ONE launch (instead of one copy per tile).
__global__ __launch_bounds__(256, 1)
void dist_scatter_kernel(__grid_constant__ const Scatter_params p) {
    const int i = blockIdx.x;
    if (i >= p.n) return;
    int4       *d = (int4 *)(p.dst + (int64_t)p.dst_idx[i] * p.slot_bytes);
    const int4 *s = (const int4 *)(p.src + (int64_t)p.src_idx[i] * p.slot_bytes);
    const size_t n4 = p.slot_bytes / sizeof(int4);
    for (size_t j = threadIdx.x; j < n4; j += blockDim.x) { d[j] = s[j]; }
}

// Q, K and V staging in ONE launch (blockIdx.y = 0 / 1 / 2).
__global__ __launch_bounds__(256, 1)
void dist_scatter3_kernel(__grid_constant__ const Scatter_params pq, __grid_constant__ const Scatter_params pk,
                          __grid_constant__ const Scatter_params pv) {
    // blockIdx.x = (slot, 64 KiB chunk) spreads each slot over many blocks
    constexpr int64_t kChunk = 64 << 10;
    const Scatter_params &p = blockIdx.y == 0 ? pq : (blockIdx.y == 1 ? pk : pv);
    const int64_t cps = (p.slot_bytes + kChunk - 1) / kChunk;
    const int64_t i = blockIdx.x / cps, c = blockIdx.x % cps;
    if (i >= p.n) return;
    const int64_t off = c * kChunk, nb = min(kChunk, p.slot_bytes - off);
    int4       *d = (int4 *)(p.dst + (int64_t)p.dst_idx[i] * p.slot_bytes + off);
    const int4 *s = (const int4 *)(p.src + (int64_t)p.src_idx[i] * p.slot_bytes + off);
    for (int64_t j = threadIdx.x; j < nb / (int64_t)sizeof(int4); j += blockDim.x) { d[j] = s[j]; }
}

}  // namespace ring_attn
