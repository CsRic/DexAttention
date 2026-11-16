#pragma once

// The block-sparse persistent kernel, writing a PARTIAL O (fp16) + LSE (fp32)
// instead of a final O. Standalone building block (Python `sparse_fwd_partial`);
// the distributed megakernel runs the same body inline and only borrows this file's
// traits include. Structurally identical to `sparse_fwd_kernel` — same mainloop,
// same host-ordered claim queue — only the epilogue differs.
//
// `m_block` addresses three things at once and that is deliberate: the Q slot it
// reads, the row list it walks, and the partial slot it writes. A device
// produces exactly one partial per Q row it touches, so those indices coincide
// and there is no third table.

#include "dist_traits.h"

namespace ring_attn {

using namespace cute;

template<typename Kernel_traits>
__global__ __launch_bounds__(Kernel_traits::kNThreads, 1)
void dist_partial_kernel(
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
            epilogue.store(fa3.epilogue_final, acc_o, softmax.row_sum, shared_storage,
                           tiled_mma, tidx, block_coord);
        } else {
            epilogue.store_zero(fa3.epilogue_final, tidx, block_coord);
        }
    }
}

}  // namespace ring_attn
