#pragma once

// Traits of the distributed forward: same FA3 sm86 geometry and the same
// block-sparse mainloop as `sparse_fwd_kernel.h`. The partial output (fp32 O
// normalised by 1/row_sum, fp32 LSE) is written by EpiloguePart, FA3's non-split
// store path with a 4-byte element (epilogue_fwd_f32.hpp). The final fp16 output
// uses FA3's Epilogue. The output slot is carried in the block coordinate.

#include "../ring_attn/sparse_fwd_kernel.h"
#include "epilogue_fwd_f32.hpp"

namespace ring_attn {

using namespace cute;

// kBlockN_ = 64 (SpargeAttention2's 128 x 64 blocks): K / V tiles of 64 rows; the KV list holds half
// tiles (2 x 128-tile slot + hn), so a row's list can be twice as long (kMaxRowTiles_).
template <int kBlockN_ = 128, int kMaxRowTiles_ = 2048>
struct Dist_fa3_kernel_traits_t {
    using Element        = cutlass::half_t;
    // (partial O element: ElementPartial below)
    using ElementAccum   = float;
    using ArchTag        = cutlass::arch::Sm80;

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

    // Partial O (local pool slots and the partials sent to the Q owner) is stored, sent and merged in fp32
    // (as FA's split-KV keeps fp32 partial accumulators). LSE is fp32. The final
    // output stays fp16. EpiloguePart = FA3's non-split store path with a 4-byte element (epilogue_fwd_f32.hpp).
    using ElementPartial = float;
    using EpiloguePart = flash::CollectiveEpilogueFwdF32<
        TileShape_MNK_PV, cute::Shape<_1, _1, _1>, ElementPartial, ArchTag, kNThreads,
        /*Varlen=*/false, /*PackGQA=*/false, /*Split=*/false, /*FP8PermuteCol=*/false>;
    // Epilogue: FA3's non-split fp16 store, used for the final O of rows this PE alone contributes to (same as
    // sparse_fwd).
    // The split one stores O straight from the accumulator registers, which makes the whole kernel spill.
    using Epilogue = flash::CollectiveEpilogueFwd<
        TileShape_MNK_PV, cute::Shape<_1, _1, _1>, Element, ArchTag, kNThreads,
        /*Varlen=*/false, /*PackGQA=*/false, /*Split=*/false, /*FP8PermuteCol=*/false>;

    using TiledMma     = typename Mainloop::TiledMma;
    using SeqlenInfo_t = typename Mainloop::SeqlenInfo_t;
    static constexpr int kSoftmaxRows = 2 * (2 * kBlockM / kNThreads);
    using Softmax_t    = flash::Softmax<kSoftmaxRows, /*Max_offset=*/0>;

    static constexpr int kMaxRowTiles = kMaxRowTiles_;     // KV tiles per row (256K tokens); planner keeps rows within it
    using Scheduler       = SparseTileScheduler;
    using SchedulerParams = typename Scheduler::Params;

    static_assert(int(sizeof(typename Epilogue::TensorStorage))
                  <= int(sizeof(typename Mainloop::TensorStorage)),
                  "epilogue smem must fit inside the mainloop smem overlap");
    // (kBlockN 64: the mainloop's K / V tiles are half as large, the fp32 partial epilogue (128 x 128 x 4 B) is not; the
    // union below is sized by the larger member, so the overlap is only a layout property of the 128 instantiation)
    static_assert(kBlockN_ != 128 || int(sizeof(typename EpiloguePart::TensorStorage))
                  <= int(sizeof(typename Mainloop::TensorStorage)),
                  "fp32 partial epilogue smem must fit inside the mainloop smem overlap");
    struct SharedStorage {
        union Tensors {
            typename Mainloop::TensorStorage mainloop;
            typename Epilogue::TensorStorage epilogue;
            typename EpiloguePart::TensorStorage epilogue_part;
        } tensors;
        alignas(16) typename Scheduler::SharedStorage smem_scheduler;
        // this row's KV slot list, staged by the mainloop (has_row_slot): a per-tile LDS instead of a global load
        alignas(16) int32_t row_slot[kMaxRowTiles];
    };
    static constexpr int kSmemSize = int(sizeof(SharedStorage));

    // The compute path IS sparse_fwd's: same mainloop instantiation, same epilogue, same shared-memory layout.
    // Everything distributed (push, delegate, gate, partial return, merge) lives outside the mainloop.
    static_assert(std::is_same_v<Mainloop, typename Sparse_fa3_kernel_traits_t<kBlockN_>::Mainloop>, "dist mainloop != sparse_fwd's");
    static_assert(std::is_same_v<Epilogue, typename Sparse_fa3_kernel_traits_t<kBlockN_>::Epilogue>, "dist epilogue != sparse_fwd's");

    struct LaunchParams {
        typename Mainloop::Params mainloop;
        typename EpiloguePart::Params epilogue;    // partial O / LSE slots (fp32 O)
        typename Epilogue::Params epilogue_final;  // this PE's final O / LSE (rows it alone contributes to, fp16 O)
        typename EpiloguePart::Params epilogue_pool;   // dist_sched_kernel: the local pool of per-task partials (fp32 O)
    };
};

using Dist_fa3_kernel_traits = Dist_fa3_kernel_traits_t<128, 2048>;

}  // namespace ring_attn
