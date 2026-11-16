#pragma once

// ── Block-sparse persistent kernel (single GPU) ──────────────────────────────
// FA3's sm80 mainloop (RingCollectiveMainloopFwdSm80) + FA3's persistent-claim structure, driven by an arbitrary
// BLOCK-SPARSE mask instead of the dense causal/non-causal geometry.
//
// The dense kernel derives the KV range of a Q tile from causal arithmetic
// (BlockMN::get_n_block_min_max). Here the range comes from a CSR table the
// host built from the mask, so anything the planner world can express --
// sliding window, BigBird, vertical-slash, a prefix tree, or a one-off mask
// handed in as a boolean block matrix -- runs through the same mainloop. Only the diagonal tiles take the elementwise mask; every other
// present tile is FULL and skips masking, which is exactly where the win over
// "dense kernel + additive -inf mask" comes from: the masked-out tiles are
// never loaded, never multiplied.
//
// Load balance is the other half. Rows of a sparse mask have wildly different
// nnz, so a static assignment strands SMs; this keeps FA3's atomic claim queue
// and lets the HOST choose the claim ORDER (longest-row-first) by permuting
// `work_coord`.

#include "ring_fwd_kernel.h"   // FA3 headers + Ring_fa3_kernel_traits + helpers
#include "sparse_desc.h"
#include "sparse_params.h"

namespace ring_attn {

using namespace cute;

// ── Scheduler ────────────────────────────────────────────────────────────────
// FA3's DynamicPersistentTileScheduler protocol (atomic claim + producer-warp
// broadcast through a named barrier, so the claim latency hides under the
// previous tile's epilogue) with ONE change: the tile index is decoded by a
// table lookup rather than by divmod arithmetic. The table is what carries the
// host's ordering into the kernel.
class SparseTileScheduler {
public:
    using SharedStorage = int;

    struct Params {
        int n_work;
        const int32_t *work_coord;   // packed (m_block, bidh, bidb)
        int *tile_count_semaphore;
        int num_m_block;
        int nheads;
        int plain = 0;               // granularity microbenchmark: 1 = plain grid (CTA b runs work item b only, no claim)
    };

    struct WorkTileInfo {
        int tile_idx;

        CUTLASS_DEVICE bool is_valid(Params const &p) const { return tile_idx < p.n_work; }

        CUTLASS_DEVICE cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const &p) const {
            int const packed = p.work_coord[tile_idx];
            int const m_block = packed % p.num_m_block;
            int const rest    = packed / p.num_m_block;
            int const bidh    = rest % p.nheads;
            int const bidb    = rest / p.nheads;
            return {m_block, bidh, bidb, 0};
        }
    };

protected:
    SharedStorage *const tile_count_smem;

public:
    CUTLASS_DEVICE SparseTileScheduler(SharedStorage *const smem) : tile_count_smem(smem) {}

    CUTLASS_DEVICE WorkTileInfo get_initial_work(Params const &) const { return {int(blockIdx.x)}; }

    CUTLASS_DEVICE void init_consumer() const {
        if (cutlass::canonical_warp_idx_sync() > 0) {
            flash::named_barrier_arrive(256, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
        }
    }

    CUTLASS_DEVICE void prefetch_next_work(Params const &p, WorkTileInfo &cur) const {
        if (threadIdx.x % 256 == 0) {
            cur.tile_idx = p.plain ? p.n_work : atomicAdd(p.tile_count_semaphore, 1) + int(gridDim.x);
        }
    }

    template <bool IsProducerWarp = false>
    CUTLASS_DEVICE WorkTileInfo get_next_work(Params const &, WorkTileInfo const &cur) const {
        if constexpr (IsProducerWarp) {
            int new_tile_idx = __shfl_sync(0xffffffff, cur.tile_idx, 0);
            flash::named_barrier_sync(256, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
            if (threadIdx.x % 256 == 0) { *tile_count_smem = cur.tile_idx; }
            flash::named_barrier_arrive(256, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
            return {new_tile_idx};
        } else {
            flash::named_barrier_sync(256, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
            int tile_idx = *tile_count_smem;
            flash::named_barrier_arrive(256, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
            return {tile_idx};
        }
    }
};

// ── Traits ───────────────────────────────────────────────────────────────────
// Same FA3 sm86 geometry as the dense path (128x128, 8 warps, 1 stage, Q in
// regs). Is_causal=true is what gives the mainloop a Mask object able to mask a
// diagonal tile; WHICH tiles get masked is a run-time property of the row
// descriptor, not of this flag.
// kBlockN_ = 64 (SpargeAttention2's 128 x 64 blocks exactly; FA2's own sm86 tile for hdim 128): the
// same mainloop / epilogue, K / V tiles of 64 rows (the row descriptor's KV indices are 64-row tile ids).
template <int kBlockN_ = 128>
struct Sparse_fa3_kernel_traits_t {
    using Element      = cutlass::half_t;
    using ElementAccum = float;
    using ArchTag      = cutlass::arch::Sm80;

    static constexpr int kBlockM   = 128;
    static constexpr int kBlockN   = kBlockN_;
    static constexpr int kHeadDim  = 128;
    static constexpr int kHeadDimV = 128;
    static constexpr int kNWarps   = 8;
    static constexpr int kStages   = 1;
    static constexpr bool Q_in_regs = true;

    using TileShape_MNK    = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using TileShape_MNK_PV = cute::Shape<Int<kBlockM>, Int<kHeadDimV>, Int<kBlockN>>;

    using Mainloop = flash::RingCollectiveMainloopFwdSm80<
        kNWarps, kStages, Q_in_regs, TileShape_MNK, kHeadDimV, Element, ElementAccum,
        ArchTag, /*Is_causal=*/true, /*Is_local=*/false, /*Has_softcap=*/false,
        /*Varlen=*/false, /*PagedKV=*/false, /*AppendKV=*/false, /*PackGQA=*/false,
        /*Split=*/false, /*BlockSparse=*/true>;

    static constexpr int kNThreads = Mainloop::NumMmaThreads;   // 256

    using Epilogue = flash::CollectiveEpilogueFwd<
        TileShape_MNK_PV, cute::Shape<_1, _1, _1>, Element, ArchTag, kNThreads,
        /*Varlen=*/false, /*PackGQA=*/false, /*Split=*/false, /*FP8PermuteCol=*/false>;

    using TiledMma     = typename Mainloop::TiledMma;
    using SeqlenInfo_t = typename Mainloop::SeqlenInfo_t;
    static constexpr int kSoftmaxRows = 2 * (2 * kBlockM / kNThreads);
    using Softmax_t    = flash::Softmax<kSoftmaxRows, /*Max_offset=*/0>;

    using Scheduler       = SparseTileScheduler;
    using SchedulerParams = typename Scheduler::Params;

    static_assert(int(sizeof(typename Epilogue::TensorStorage))
                  <= int(sizeof(typename Mainloop::TensorStorage)),
                  "epilogue smem must fit inside the mainloop smem overlap");
    struct SharedStorage {
        union Tensors {
            typename Mainloop::TensorStorage mainloop;
            typename Epilogue::TensorStorage epilogue;
        } tensors;
        alignas(16) typename Scheduler::SharedStorage smem_scheduler;
    };
    static constexpr int kSmemSize = int(sizeof(SharedStorage));

    struct LaunchParams {
        typename Mainloop::Params mainloop;
        typename Epilogue::Params epilogue;
    };
};

using Sparse_fa3_kernel_traits = Sparse_fa3_kernel_traits_t<128>;

// ── Kernel ───────────────────────────────────────────────────────────────────
template<typename Kernel_traits>
__global__ __launch_bounds__(Kernel_traits::kNThreads, 1)
void sparse_fwd_kernel(
    typename Kernel_traits::LaunchParams fa3,
    Sparse_fwd_params sp,
    typename Kernel_traits::SchedulerParams sched_params
) {
    extern __shared__ char smem_[];
    auto &shared_storage = *reinterpret_cast<typename Kernel_traits::SharedStorage *>(smem_);
    const int tidx = threadIdx.x;

    typename Kernel_traits::Mainloop mainloop;
    typename Kernel_traits::Epilogue epilogue;
    typename Kernel_traits::TiledMma tiled_mma;

    typename Kernel_traits::Scheduler scheduler(&shared_storage.smem_scheduler);
    scheduler.init_consumer();
    const int warp_idx = cutlass::canonical_warp_idx_sync();

    CUTLASS_PRAGMA_NO_UNROLL
    for (auto work = scheduler.get_initial_work(sched_params);
         work.is_valid(sched_params);
         work = warp_idx == 0
             ? scheduler.template get_next_work</*IsProducerWarp=*/true>(sched_params, work)
             : scheduler.template get_next_work</*IsProducerWarp=*/false>(sched_params, work)) {

        auto block_coord = work.get_block_coord(sched_params);
        const int m_block = get<0>(block_coord);

        // The row's KV list. Identical for every (head, batch) — the mask is a
        // property of the sequence, not of the head.
        const int32_t beg = sp.kv_indptr[m_block];
        flash::SparseRowDesc row{ sp.kv_indices + beg,
                                  sp.kv_mask_nb + beg,
                                  sp.kv_indptr[m_block + 1] - beg,
                                  sp.row_n_masked[m_block],
                                  sp.row_mask_mb[m_block],
                                  /*ord_lo=*/0, /*ord_hi=*/0 };

        Tensor acc_o = partition_fragment_C(
            tiled_mma, select<0, 2>(typename Kernel_traits::TileShape_MNK{}));
        typename Kernel_traits::Softmax_t softmax(fa3.mainloop.softmax_scale_log2);
        typename Kernel_traits::SeqlenInfo_t seqlen_info{
            get<2>(block_coord), sp.seqlen_q, sp.seqlen_k, 0,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

        bool const tile_valid = mainloop.template mma</*Is_first_step=*/true>(
            fa3.mainloop, (const typename Kernel_traits::Element *)sp.k_ptr,
            (const typename Kernel_traits::Element *)sp.v_ptr,
            acc_o, softmax, tidx, seqlen_info, block_coord, shared_storage,
            /*is_strict=*/false, &row);

        scheduler.prefetch_next_work(sched_params, work);
        if (tile_valid) {
            Tensor final_scale = softmax.finalize();
            softmax.rescale_o(acc_o, final_scale);
            epilogue.store(fa3.epilogue, acc_o, softmax.row_sum, shared_storage,
                           tiled_mma, tidx, block_coord);
        } else {
            epilogue.store_zero(fa3.epilogue, tidx, block_coord);
        }
    }
}

}  // namespace ring_attn
