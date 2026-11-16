#pragma once

// ── Ported FA3 DynamicPersistentTileScheduler ────────────────────────────────
// Verbatim copy of flash::DynamicPersistentTileScheduler from FA3's
// hopper/tile_scheduler.hpp (renamed to namespace ring_attn), EXCEPT for the
// ring modifications flagged with `// RING`. It is ported rather than called
// directly because ring attention has to modify its coupling: the ring
// replays the claim queue once per communication step, so (1) the semaphore
// must be per-step resettable and (2) the producer/consumer named-barrier
// handshake must re-initialise each step. The scheduler math (L2-swizzle
// sections + longest-processing-time-first decode) and the epilogue-overlapped
// prefetch (prefetch_next_work: thread 0 atomically claims the next tile while
// the current tile's epilogue runs) are unchanged.
//
// RING edits: to_underlying_arguments is HOST_DEVICE (the ring kernel rebuilds
// the params inside the kernel every step) and the semaphore pointer is
// non-const. The per-step reset is done by giving each ring step its own
// counter (ring_fwd_kernel.h: &tile_count[s]), not by a reset hook here.

#include "cutlass/fast_math.h"
#include "cutlass/arch/barrier.h"
#include "named_barrier.hpp"
#include "tile_scheduler.hpp"   // flash::TileSchedulerArguments (shared arg struct)
#include "utils.h"

namespace ring_attn {

using namespace cute;

template<int NumMmaThreads = 256, int NumProducerThreads = 256,
         bool Split = false, bool PackGQA = false, bool WarpSpecialized = false>
class RingDynamicPersistentTileScheduler {

    static_assert(WarpSpecialized || NumProducerThreads == NumMmaThreads);
    static constexpr int NumThreads = WarpSpecialized ? NumMmaThreads + NumProducerThreads : NumMmaThreads;

public:
    using SharedStorage = int;

protected:
    SharedStorage* const tile_count_smem;

public:

    struct Params {
        int const total_blocks;
        cutlass::FastDivmod const m_block_divmod, head_divmod;
        cutlass::FastDivmod const l2_minor_divmod, l2_major_divmod;
        cutlass::FastDivmod const l2_minor_residual_divmod;
        int const num_hb_quotient;
        int* tile_count_semaphore;  // mutable per-step (ring kernel reassigns each step)
    };

    CUTLASS_HOST_DEVICE static Params
    to_underlying_arguments(flash::TileSchedulerArguments const& args) {
        long long const size_one_kv_head = long(args.seqlen_k) * long(args.headdim + args.headdim_v) * long(args.element_size);
        int const size_l2 = 32 * 1024 * 1024;  // 32 MB for K & V
        auto find_log2_floor = [&](int n) { return 31 - cutlass::clz(n); };
        int const swizzle = (size_l2 < size_one_kv_head ? 1 : (1 << find_log2_floor(size_l2 / size_one_kv_head))) * (PackGQA ? 1 : args.qhead_per_khead);
        int const num_hb_remainder = (args.num_head * args.num_batch) % swizzle;
        int const num_split_blocks = args.num_blocks * (!Split ? 1 : args.num_splits);
        assert(args.tile_count_semaphore != nullptr);
        return {num_split_blocks * args.num_head * args.num_batch,
                cutlass::FastDivmod(args.num_blocks), cutlass::FastDivmod(args.num_head),
                cutlass::FastDivmod(swizzle), cutlass::FastDivmod(swizzle * num_split_blocks),
                cutlass::FastDivmod(num_hb_remainder > 0 ? num_hb_remainder : 1),
                (args.num_head * args.num_batch) / swizzle,
                args.tile_count_semaphore};
    }

    static dim3
    get_grid_shape(Params const& params, int num_sm) {
        return {uint32_t(num_sm)};
    }

    struct WorkTileInfo {
        int tile_idx;

        CUTLASS_DEVICE
        bool is_valid(Params const& params) const {
            return tile_idx < params.total_blocks;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            int block, bidh, bidb;
            int l2_mod, bidhb, bidhb_residual;
            bidhb = params.l2_major_divmod.divmod(l2_mod, tile_idx);
            if (bidhb < params.num_hb_quotient) {
                block = params.l2_minor_divmod.divmod(bidhb_residual, l2_mod);
            } else {
                block = params.l2_minor_residual_divmod.divmod(bidhb_residual, l2_mod);
            }
            bidb = params.head_divmod.divmod(bidh, bidhb * params.l2_minor_divmod.divisor + bidhb_residual);
            int split_idx = 0;
            if constexpr (Split) {
                split_idx = params.m_block_divmod.divmod(block, block);
            }
            // Longest-processing-time-first
            block = params.m_block_divmod.divisor - 1 - block;
            return {block, bidh, bidb, split_idx};
        }
    };

    CUTLASS_DEVICE
    RingDynamicPersistentTileScheduler(SharedStorage* const smem_scheduler) : tile_count_smem(smem_scheduler) {};

    template<bool IsProducerWarp = false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        return {int(blockIdx.x)};
    }

    CUTLASS_DEVICE
    void init_consumer() const {
        if (WarpSpecialized || cutlass::canonical_warp_idx_sync() > 0) {
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 /*id*/);  // TileCountSmemEmpty
        }
    }

    CUTLASS_DEVICE
    void prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {
        if (threadIdx.x % NumProducerThreads == 0) {
            current_work.tile_idx = atomicAdd(params.tile_count_semaphore, 1) + int(gridDim.x);
        }
    }

    template<bool IsProducerWarp = false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        if constexpr (IsProducerWarp) {
            int new_tile_idx = __shfl_sync(0xffffffff, current_work.tile_idx, 0 /*lane*/);
            flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 /*id*/);  // TileCountSmemEmpty
            if (threadIdx.x % NumProducerThreads == 0) {
                *tile_count_smem = current_work.tile_idx;
            }
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 /*id*/);  // TileCountSmemFull
            return {new_tile_idx};
        } else {
            flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 /*id*/);  // TileCountSmemFull
            int tile_idx = *tile_count_smem;
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 /*id*/);  // TileCountSmemEmpty
            return {tile_idx};
        }
    }
};

}  // namespace ring_attn
