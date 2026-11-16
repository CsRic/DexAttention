#pragma once

// The ring kernel (ring_fwd_kernel below) is not built in this release; its traits and helpers are used through
// sparse_fwd_kernel.h.

// FA3 hopper headers (include path points at hopper/; namespace flash is
// FA3's; FA2's csrc headers would clash on the same namespace).
#include "utils.h"                      // FA3; must precede block.h (round_up/round_down)
#include "softmax.h"                    // FA3, verbatim
#include "seqlen.h"                     // FA3, verbatim
#include "ring_mainloop_fwd_sm80.hpp"   // our copy of mainloop_fwd_sm80.hpp (see header)
#include "ring_tile_scheduler.hpp"      // our copy of DynamicPersistentTileScheduler (see header)
#include "epilogue_fwd.hpp"             // FA3, verbatim

#include "nvshmem_transport.cuh"
#include "ring_params.h"

#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/arch.h>

namespace ring_attn {


using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Persistent-kernel ring attention forward, all-put transport.
//
//   grid  = [n] blocks (one static full-occupancy wave; blocks loop over the
//           variable work themselves — mandatory under the cooperative
//           collective launch, where an undispatched block would stall the
//           cross-block spins)
//   block = 256 threads (8 warps) on one 128x128 tile — FA3's sm86 geometry
//
// The compute core is FA3's sm80 mainloop (RingCollectiveMainloopFwdSm80) + FA3's persistent-claim structure and
// CollectiveEpilogueFwd (see ring_mainloop_fwd_sm80.hpp for the modifications);
// this file owns only what FA3 has no equivalent for:
//
//   - the five per-ring-step transport phases (grant/arrival/push/compute/
//     retire — see ring_params.h for the protocol)
//   - the tile scheduler (FA3-ported dynamic claim queue with L2 swizzle +
//     LPT, or a static split)
//   - the cross-step online-softmax state: the ring loop is step-outer /
//     tile-inner, so each tile's (acc_o, row_max, row_sum) is spilled to
//     gmem between steps and mma() is re-entered with Is_first_step=false.
//
// Ring step s consumes chunk(s); while computing it, this PE PUTS chunk(s)
// into next_pe's slot (s+1)%2. Per-step phases:
//
//   1. grant     delegate → prev: slot (s+1)%2 is writable (its previous
//                occupant is computed and forwarded)
//   2. arrival   all blocks gate on chunk(s) having landed
//   3. push      after the arrival gate (the push READS chunk(s)):
//                intra-node → st-loop, retired immediately;
//                inter-node → putmem_nbi issued, retired in phase 5
//   4. compute   FA3 mainloop over all (batch, head, Q-tile) work items
//   5. retire    inter-node only: quiet() after a full step of compute has
//                overlapped the RDMA, then signal arrived
//
// Cross-block coordination is local-memory flags fanned out by the delegate
// (block 0), the only NVSHMEM-signal poller.
//
////////////////////////////////////////////////////////////////////////////////////////////////////

// FA3 component assembly for one causal mode. Geometry is FA3's sm86 pick
// (tile_size_fwd_sm8x, hdim 128, fp16): 128x128, 8 warps, 1 smem stage,
// Q in registers (which also means Q and V share smem — 64KB total).
// The causal kernel needs TWO mainloop instantiations: the diagonal ring
// step (s==0) masks against the diagonal (Is_causal=true, BlockMN clamps
// n_block_max), every other contributing step is full attention over the
// chunk (Is_causal=false). Non-causal only ever uses the full one.
// DynamicSched picks the tile scheduler. Defaults to Is_causal (causal needs
// LPT balancing), but the non-causal RING path opts in too — see kDynamicSched.
template<bool Is_causal, bool DynamicSched = Is_causal>
struct Ring_fa3_kernel_traits {
    using Element      = cutlass::half_t;
    using ElementAccum = float;
    using ArchTag      = cutlass::arch::Sm80;
    using index_t      = int64_t;

    static constexpr int kBlockM   = 128;
    static constexpr int kBlockN   = 128;
    static constexpr int kHeadDim  = 128;
    static constexpr int kHeadDimV = 128;
    static constexpr int kNWarps   = 8;
    static constexpr int kStages   = 1;
    static constexpr bool Q_in_regs = true;

    using TileShape_MNK    = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using TileShape_MNK_PV = cute::Shape<Int<kBlockM>, Int<kHeadDimV>, Int<kBlockN>>;

    using MainloopDiag = flash::RingCollectiveMainloopFwdSm80<
        kNWarps, kStages, Q_in_regs, TileShape_MNK, kHeadDimV, Element, ElementAccum,
        ArchTag, /*Is_causal=*/true, /*Is_local=*/false, /*Has_softcap=*/false,
        /*Varlen=*/false, /*PagedKV=*/false, /*AppendKV=*/false, /*PackGQA=*/false, /*Split=*/false>;
    using MainloopFull = flash::RingCollectiveMainloopFwdSm80<
        kNWarps, kStages, Q_in_regs, TileShape_MNK, kHeadDimV, Element, ElementAccum,
        ArchTag, /*Is_causal=*/false, /*Is_local=*/false, /*Has_softcap=*/false,
        /*Varlen=*/false, /*PagedKV=*/false, /*AppendKV=*/false, /*PackGQA=*/false, /*Split=*/false>;

    static constexpr int kNThreads = MainloopFull::NumMmaThreads;   // 256

    using Epilogue = flash::CollectiveEpilogueFwd<
        TileShape_MNK_PV, cute::Shape<_1, _1, _1>, Element, ArchTag, kNThreads,
        /*Varlen=*/false, /*PackGQA=*/false, /*Split=*/false, /*FP8PermuteCol=*/false>;

    using TiledMma      = typename MainloopFull::TiledMma;
    using SeqlenInfo_t  = typename MainloopFull::SeqlenInfo_t;
    static constexpr int kSoftmaxRows = 2 * (2 * kBlockM / kNThreads);   // 2
    using Softmax_t     = flash::Softmax<kSoftmaxRows, /*Max_offset=*/0>;

    // Scheduler: non-causal uses FA3's StaticPersistentTileScheduler
    // (round-robin) — LPT + L2 swizzle is pure
    // overhead when every tile does equal work. Causal keeps the ported
    // DynamicPersistentTileScheduler (LPT load-balancing is essential).
    //
    // The non-causal RING path (P>1) takes the dynamic scheduler too. Its motive
    // is not tile balance (non-causal tiles are uniform) but push exposure: only a dynamic claim lets a block that
    // is busy pushing claim fewer tiles, which is what makes the small push budget pay off. P=1 (no push at all)
    // keeps the static scheduler.
    //
    // Single source of truth: the dynamic scheduler brings a producer/consumer
    // named-barrier protocol that MUST be drained between ring steps. Anything
    // conditioned on "which scheduler" must test THIS, never Is_causal.
    static constexpr bool kDynamicSched = DynamicSched;

    // Push-block budget, set by the scheduler. Static: the straggle is fully exposed, so spread it over 8.
    // Dynamic: the straggle self-heals and only interference remains, so use as
    // few as possible: 2 for causal, 1 for non-causal.
    static constexpr int kNumPushBlocks = !kDynamicSched ? 8 : (Is_causal ? 2 : 1);

    using Scheduler = std::conditional_t<
        kDynamicSched,
        RingDynamicPersistentTileScheduler<kNThreads, kNThreads, /*Split=*/false, /*PackGQA=*/false, /*WarpSpecialized=*/false>,
        flash::StaticPersistentTileScheduler</*Split=*/false>>;
    using SchedulerParams = typename Scheduler::Params;

    // Kernel-level shared storage — FlashAttnFwdSm80::SharedStorage's
    // mainloop/epilogue overlap, as a plain named union (the upstream
    // anonymous-aggregate + padding trick trips older host compilers; the
    // padding is only needed when the epilogue outgrows smem_v + smem_k,
    // which the static_assert rules out here). smem_o aliases smem_v/q —
    // exactly the layout FA3's kernel produces for this geometry. No
    // tile-scheduler smem (our claim broadcast is a separate __shared__ int).
    static_assert(int(sizeof(typename Epilogue::TensorStorage))
                  <= int(sizeof(typename MainloopFull::TensorStorage)),
                  "epilogue smem must fit inside the mainloop smem overlap");
    struct SharedStorage {
        union Tensors {
            typename MainloopFull::TensorStorage mainloop;
            typename Epilogue::TensorStorage epilogue;
        } tensors;
        // Scheduler's tile-count broadcast slot (FA3 FlashAttnFwdSm80 layout).
        alignas(16) typename Scheduler::SharedStorage smem_scheduler;
    };
    static constexpr int kSmemSize = int(sizeof(SharedStorage));

    // Host-built FA3 component Params, passed as a second kernel argument.
    struct LaunchParams {
        typename MainloopDiag::Params mainloop_diag;
        typename MainloopFull::Params mainloop_full;
        typename Epilogue::Params     epilogue;
    };
};

// Spill/restore one Q-tile's online-softmax state (acc_o, row_max, row_sum):
// the loop is step-outer / tile-inner, so every owned tile's state must
// survive across ring steps — too much for registers, round-trips via gmem.
template<typename AccTensor, typename RowTensor>
__device__ __forceinline__ void save_tile_acc_state(
    float *dst, const AccTensor &acc_o, const RowTensor &row_max, const RowTensor &row_sum
) {
    int idx = 0;
    #pragma unroll
    for (int i = 0; i < size(acc_o); ++i)   { dst[idx++] = acc_o(i); }
    #pragma unroll
    for (int i = 0; i < size(row_max); ++i) { dst[idx++] = row_max(i); }
    #pragma unroll
    for (int i = 0; i < size(row_sum); ++i) { dst[idx++] = row_sum(i); }
}

template<typename AccTensor, typename RowTensor>
__device__ __forceinline__ void load_tile_acc_state(
    const float *src, AccTensor &acc_o, RowTensor &row_max, RowTensor &row_sum
) {
    int idx = 0;
    #pragma unroll
    for (int i = 0; i < size(acc_o); ++i)   { acc_o(i)   = src[idx++]; }
    #pragma unroll
    for (int i = 0; i < size(row_max); ++i) { row_max(i) = src[idx++]; }
    #pragma unroll
    for (int i = 0; i < size(row_sum); ++i) { row_sum(i) = src[idx++]; }
}

// Per-tile acc-state float count. Device-side constexpr helper.
template<typename Kernel_traits>
__device__ constexpr int ring_acc_state_floats() {
    typename Kernel_traits::TiledMma tiled_mma{};
    auto acc_o = partition_fragment_C(
        tiled_mma, select<0, 2>(typename Kernel_traits::TileShape_MNK{}));
    return int(size(acc_o)) + 2 * Kernel_traits::kSoftmaxRows;
}

// ── Per-tile processing for one ring step ──────────────────────────────────────
// __noinline__ isolates the mainloop's 255-register allocation from the ring
// scaffold (ring pointers, scheduler state, step counter). Inlined, the two
// compete and ptxas spills; as an ABI call the tile body gets a clean
// allocation and the scaffold's live values are spilled exactly once per tile.
// Takes block_coord and pre-computed scratch pointer directly (no w→coord decode
// inside — the scheduler already produces block_coord).
// __noinline__ for ring: keeps the mainloop's 255-register allocation
// isolated from the ring scaffold. The per-tile ABI call overhead is small
// and avoids register-spill cascades when the ring scaffold + inlined mainloop
// exceed the 255-register budget.
template<typename Kernel_traits, bool Is_diag_step, bool Is_first_contrib>
__device__ __noinline__ void ring_process_tile(
    const typename Kernel_traits::LaunchParams &fa3,
    typename Kernel_traits::Element *chunk_k,
    typename Kernel_traits::Element *chunk_v,
    typename Kernel_traits::SharedStorage &shared_storage,
    cute::tuple<int32_t, int32_t, int32_t, int32_t> block_coord,
    bool is_last, float *scratch, bool is_strict = false
) {
    typename Kernel_traits::MainloopDiag mainloop_diag;
    typename Kernel_traits::MainloopFull mainloop_full;
    typename Kernel_traits::Epilogue     epilogue;
    typename Kernel_traits::TiledMma     tiled_mma;

    Tensor acc_o = partition_fragment_C(
        tiled_mma, select<0, 2>(typename Kernel_traits::TileShape_MNK{}));
    typename Kernel_traits::Softmax_t softmax(
        fa3.mainloop_full.softmax_scale_log2);

    const int bidb = get<2>(block_coord);

    if constexpr (!Is_first_contrib) {
        load_tile_acc_state(scratch, acc_o, softmax.row_max, softmax.row_sum);
    }

    // seqlen_q/k from fa3 (ld.const), tidx from special register.
    const int seqlen_q = get<0>(fa3.mainloop_full.shape_Q);
    const int seqlen_k = get<0>(fa3.mainloop_full.shape_K);
    const int tidx = int(threadIdx.x);

    typename Kernel_traits::SeqlenInfo_t seqlen_info{
        bidb, seqlen_q, seqlen_k, 0,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

    if constexpr (Is_diag_step) {
        mainloop_diag.template mma<Is_first_contrib>(
            fa3.mainloop_diag, chunk_k, chunk_v, acc_o, softmax,
            tidx, seqlen_info, block_coord, shared_storage, is_strict);
    } else {
        mainloop_full.template mma<Is_first_contrib>(
            fa3.mainloop_full, chunk_k, chunk_v, acc_o, softmax,
            tidx, seqlen_info, block_coord, shared_storage, is_strict);
    }

    if (!is_last) {
        save_tile_acc_state(scratch, acc_o, softmax.row_max, softmax.row_sum);
        return;
    }

    // Last contributing step: finalize + epilogue (same as single kernel).
    Tensor final_scale = softmax.finalize();
    softmax.rescale_o(acc_o, final_scale);
    epilogue.store(fa3.epilogue, acc_o, softmax.row_sum, shared_storage,
                   tiled_mma, tidx, block_coord);
}

// ── Ring persistent kernel — single-kernel-style scheduler inside ring loop ────
template<typename Kernel_traits, bool Is_causal, int NumPEs = 0, bool NextIntra = true>
__device__ void ring_fwd_persistent(
    const Ring_fwd_params &params,
    const typename Kernel_traits::LaunchParams &fa3,
    const typename Kernel_traits::SchedulerParams &sched_params_nc
) {
    using Element = typename Kernel_traits::Element;
    using index_t = typename Kernel_traits::index_t;

    extern __shared__ char smem_[];
    auto &shared_storage = *reinterpret_cast<typename Kernel_traits::SharedStorage *>(smem_);
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kNThreads = Kernel_traits::kNThreads;
    constexpr int kSoftmaxRows = Kernel_traits::kSoftmaxRows;

    // P: compile-time when NumPEs>0, runtime fallback when NumPEs==0 (default)
    const int P = (NumPEs > 0) ? NumPEs : params.num_pes;
    const int r = params.my_pe;

    const int num_m_block = params.seqlen_q / kBlockM;
    const int total_work  = params.b * params.h * num_m_block;

    const bool is_delegate = (blockIdx.x == 0);
    // NumPEs==1 → is_push is compile-time false (no ring, no push)
    const bool is_push     = (NumPEs != 1) && ((int)blockIdx.x < params.num_push_blocks);
    // Dedicated comm blocks: push K/V only, never compute.
    // Compute blocks: full attention, report to compute_done gate.
    const int  num_compute_blocks = (NumPEs != 1) ? gridDim.x : gridDim.x;
    // next_intra: compile-time when NumPEs>0, runtime fallback otherwise
    const bool next_intra  = (NumPEs > 0) ? NextIntra : params.next_intra_node;

    // STRIPED causal: every PE computes ALL P steps (each a triangular block —
    // inclusive-diagonal for s<=r, strict/diagonal-excluded for s>r), which
    // balances the causal load. So last_compute_step = P-1 for both causal and
    // non-causal.
    // NumPEs==1 → last_compute_step = 0 (single step, compile-time known)
    const int last_compute_step = (NumPEs == 1) ? 0 : (P - 1);

    const index_t chunk_row_stride   = (index_t)params.h_k * params.d;
    const index_t chunk_batch_stride = (index_t)params.seqlen_k * chunk_row_stride;
    const size_t  chunk_bytes        = (size_t)params.b * chunk_batch_stride * sizeof(Element);
    const size_t  kv_bytes           = 2 * chunk_bytes;

    auto slice_of = [&](size_t total, size_t *off, size_t *len) {
        const size_t units = total / 256;
        const size_t per   = units / params.num_push_blocks;
        const size_t rem   = units % params.num_push_blocks;
        const size_t c     = blockIdx.x;
        *off = (c * per + (c < rem ? c : rem)) * 256;
        *len = (per + (c < rem ? 1 : 0)) * 256;
    };

    // Acc-state scratch stride per thread (all tiles, all steps); the host allocation must match.
    static constexpr int kAccStateFloats = ring_acc_state_floats<Kernel_traits>();

    // ── Sub-allocated pointer offsets (merged to reduce param buffer) ────────
    uint64_t *const sig_dst_free = params.sig_base_ptr;
    uint64_t *const sig_arrived  = params.sig_base_ptr + 1;
    const uint64_t sig_base = params.sig_epoch;
    int *const coord   = params.coord_base;
    int *const compute_done = coord;
    int *const arrived_flag = coord + P;
    int *const push_go      = coord + 2*P;
    int *const push_done    = coord + 3*P;
    int *const tile_count   = coord + 4*P;

    // ── Entry grant ─────────────────────────────────────────────────────────
    if constexpr (NumPEs != 1) {
        if (is_delegate && tidx == 0) {
            ring_signal_set(sig_dst_free, sig_base + 2, params.prev_pe);
        }
    }

    // ── Ring loop ───────────────────────────────────────────────────────────
    for (int s = 0; s < P; ++s) {
        // ── 1. grant ────────────────────────────────────────────────────────
        if constexpr (NumPEs != 1) {
            if (is_delegate && tidx == 0) {
                if (s >= 1) counter_wait_ge(&compute_done[s - 1], num_compute_blocks);
                if (s >= 2) {
                    ring_signal_set(sig_dst_free, sig_base + s + 1, params.prev_pe);
                }
            }
        }

        // ── 2. arrival gate ─────────────────────────────────────────────────
        if (s >= 1) {
            if (tidx == 0) {
                if (is_delegate) {
                    ring_signal_wait_ge(sig_arrived, sig_base + s);
                    __threadfence();
                    flag_set(&arrived_flag[s], 1);
                } else {
                    counter_wait_ge(&arrived_flag[s], 1);
                }
                __threadfence();
            }
            __syncthreads();
        }

        // ── 3. push ─────────────────────────────────────────────────────────
        if constexpr (NumPEs != 1) {
            if (s < P - 1) {
                if (is_delegate && tidx == 0) {
                ring_signal_wait_ge(sig_dst_free, sig_base + s + 1);
                __threadfence();
                flag_set(&push_go[s], 1);
            }
            if (is_push) {
                if (tidx == 0) {
                    counter_wait_ge(&push_go[s], 1);
                    __threadfence();
                }
                __syncthreads();
                char *src[2], *dst[2];
                size_t off, len;
                int nregions;
                const int dslot = (s + 1) & 1;
                if (s == 0) {
                    slice_of(chunk_bytes, &off, &len);
                    src[0] = (char *)params.sym_local_k;
                    src[1] = (char *)params.sym_local_v;
                    dst[0] = (char *)params.sym_buf_k[dslot];
                    dst[1] = (char *)params.sym_buf_v[dslot];
                    nregions = 2;
                } else {
                    slice_of(kv_bytes, &off, &len);
                    src[0] = (char *)params.sym_buf_k[s & 1];
                    dst[0] = (char *)params.sym_buf_k[dslot];
                    nregions = 1;
                }
                if (next_intra) {
                    // skip_push_data: signalling only (off in the shipped paths)
                    if (!params.skip_push_data) {
                        for (int i = 0; i < nregions; ++i) {
                            intra_push_slice_block(dst[i] + params.next_intra_delta + off,
                                                   src[i] + off, len);
                        }
                    }
                    __syncthreads();
                    if (tidx == 0) {
                        __threadfence_system();
                        counter_inc(&push_done[s]);
                    }
                } else if (tidx == 0 && len > 0 && !params.skip_push_data) {
                    for (int i = 0; i < nregions; ++i) {
                        nvshmem_putmem_nbi(dst[i] + off, src[i] + off, len, params.next_pe);
                    }
                }
            }
            if (next_intra && is_delegate && tidx == 0) {
                counter_wait_ge(&push_done[s], params.num_push_blocks);
                ring_signal_set(sig_arrived, sig_base + s + 1, params.next_pe);
            }
        }  // if (s < P - 1)
        }  // if constexpr (NumPEs != 1)

        // ── 4. compute — single-kernel-style scheduler + mainloop ───────────
        if (s <= last_compute_step) {
            Element *chunk_k = (s == 0) ? (Element *)params.sym_local_k
                                        : (Element *)params.sym_buf_k[s & 1];
            Element *chunk_v = (s == 0) ? (Element *)params.sym_local_v
                                        : (Element *)params.sym_buf_v[s & 1];
            // STRIPED: every causal step uses the causal mainloop; s>r wraps to
            // a "future" source chunk whose block excludes the diagonal (strict,
            // attend col<row). Non-causal keeps is_strict=false (no mask).
            const bool is_strict = Is_causal && (s > r);
            const bool is_last = (s == last_compute_step);
            const bool is_first_contrib = (s == 0);

            // Build per-step scheduler params. For non-causal the StaticPersistent
            // params are step-invariant (re-built each step — cost is negligible).
            auto sched_params = [&]() {
                if constexpr (Kernel_traits::kDynamicSched) {
                    flash::TileSchedulerArguments sched_args{
                        num_m_block, params.h, params.b, 1,
                        params.h / params.h_k,
                        params.seqlen_q,
                        params.seqlen_k, params.d, params.d, (int)sizeof(Element),
                        &tile_count[s], nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
                    return Kernel_traits::Scheduler::to_underlying_arguments(sched_args);
                } else {
                    return sched_params_nc;
                }
            }();

            typename Kernel_traits::Scheduler scheduler(&shared_storage.smem_scheduler);
            scheduler.init_consumer();
            const int warp_idx = cutlass::canonical_warp_idx_sync();

            // Hoist loop-invariant params fields to registers
            const int h_l        = params.h;
            float *const acc_scratch = params.acc_scratch_buf;

            CUTLASS_PRAGMA_NO_UNROLL
            for (auto work_tile_info = warp_idx == 0
                     ? scheduler.template get_initial_work</*IsProducerWarp=*/true>(sched_params)
                     : scheduler.template get_initial_work</*IsProducerWarp=*/false>(sched_params);
                 work_tile_info.is_valid(sched_params);
                 work_tile_info = warp_idx == 0
                     ? scheduler.template get_next_work</*IsProducerWarp=*/true>(sched_params, work_tile_info)
                     : scheduler.template get_next_work</*IsProducerWarp=*/false>(sched_params, work_tile_info)) {

                auto block_coord = work_tile_info.get_block_coord(sched_params);
                const int m_block = get<0>(block_coord);
                const int bidh    = get<1>(block_coord);
                const int bidb    = get<2>(block_coord);
                const int w = bidb * h_l * num_m_block + bidh * num_m_block + m_block;

                float *scratch = acc_scratch
                    + ((size_t)w * kNThreads + tidx) * kAccStateFloats;

                // Dispatch to __noinline__ tile processor — register isolation.
                // Is_diag_step = Is_causal: causal steps use the causal mainloop
                // (striped triangular mask, inclusive/strict via is_strict);
                // non-causal uses MainloopFull. Only 2 call sites (first / not).
                if (is_first_contrib) {
                    ring_process_tile<Kernel_traits, /*Is_diag_step=*/Is_causal, /*first=*/true>(
                        fa3, chunk_k, chunk_v, shared_storage,
                        block_coord, is_last, scratch, is_strict);
                } else {
                    ring_process_tile<Kernel_traits, /*Is_diag_step=*/Is_causal, /*first=*/false>(
                        fa3, chunk_k, chunk_v, shared_storage,
                        block_coord, is_last, scratch, is_strict);
                }

                // Prefetch: atomic claim for next tile overlaps save/finalize
                // inside ring_process_tile (same as single kernel).
                scheduler.prefetch_next_work(sched_params, work_tile_info);
            }

            // Drain named-barrier phase for the DYNAMIC scheduler (RingDynamic-
            // PersistentTileScheduler uses producer/consumer barriers that
            // must be reset between ring steps — the barrier hardware state
            // persists across the step boundary). Keyed on the scheduler, not on Is_causal.
            if constexpr (Kernel_traits::kDynamicSched) {
                if (warp_idx == 0) {
                    flash::named_barrier_sync(kNThreads,
                        cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
                }
            }
        }

        // ── compute_done fence ──────────────────────────────────────────────
        // Only compute blocks report; push blocks skip (they just wait at
        // the next step's arrival gate for the remote signal).
        if constexpr (NumPEs != 1) {
            if (s < P - 1) {
                __syncthreads();
                if (tidx == 0) {
                    __threadfence();
                    counter_inc(&compute_done[s]);
                }
            }
        }

        // ── 5. retire inter-node push ───────────────────────────────────────
        if constexpr (NumPEs != 1) {
            if (s < P - 1 && !next_intra) {
                if (is_push && tidx == 0) {
                    nvshmem_quiet();
                    __threadfence();
                    counter_inc(&push_done[s]);
                }
                if (is_delegate && tidx == 0) {
                    counter_wait_ge(&push_done[s], params.num_push_blocks);
                    ring_signal_set(sig_arrived, sig_base + s + 1, params.next_pe);
                }
            }
        }  // if constexpr (NumPEs != 1) retire
    }  // for (int s = 0; s < P; ++s)
}  // ring_fwd_persistent

////////////////////////////////////////////////////////////////////////////////////////////////////

// sm_86 register budget: 256 threads × 255 regs = 65280 / 65536 regs/SM.
// Occupancy is 1 block/SM (limited by shared memory). __launch_bounds__
// tells ptxas to allocate registers for exactly 1 block, avoiding spills
// from conservative occupancy assumptions.
template<typename Kernel_traits, bool Is_causal, int NumPEs = 0, bool NextIntra = true>
__global__ __launch_bounds__(Kernel_traits::kNThreads, 1)
void ring_fwd_kernel(
    Ring_fwd_params params,
    typename Kernel_traits::LaunchParams fa3,
    typename Kernel_traits::SchedulerParams sched_params_nc
) {
    ring_fwd_persistent<Kernel_traits, Is_causal, NumPEs, NextIntra>(params, fa3, sched_params_nc);
}

}  // namespace ring_attn
