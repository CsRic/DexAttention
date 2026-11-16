// Forward kernel entry points of the extension: the single-GPU block-sparse forward (sparse_fwd), the partial-output
// variant and the log-sum-exp merge used by the numerics tests, the distributed forward megakernel (dist_fwd) and the
// staging / copy-engine signal helpers. Host-side argument packing lives in ring_flash_attn_host.cpp.

#include <nvshmem.h>
#include <nvshmemx.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <cstdlib>

#include "sparse_fwd_kernel.h"
#include "../dist_attn/dist_partial_kernel.h"
#include "../dist_attn/dist_merge.cuh"
#include "../dist_attn/dist_sched_kernel.h"   // includes dist_fwd_kernel.h

#include <cutlass/kernel_launch.h>
#include <cutlass/device_kernel.h>
#include <cutlass/kernel_hardware_info.h>

using namespace ring_attn;

// ── Block-sparse single-GPU launch (sparse_fwd_kernel.h) ─────────────────────
// No NVSHMEM: one persistent wave over a mask-driven claim queue; the claim order is chosen on the host.

template<typename Mainloop, typename Element>
static typename Mainloop::Arguments make_sparse_mainloop_args(const Sparse_fwd_params &p) {
    const int64_t krow   = (int64_t)p.h_k * p.d;
    const int64_t kbatch = (int64_t)p.seqlen_k * krow;
    return typename Mainloop::Arguments {
        (Element const *)p.q_ptr,
        cute::make_shape(p.seqlen_q, p.d, p.h, p.b),
        cute::make_stride(p.q_row_stride, cute::_1{}, p.q_head_stride, p.q_batch_stride),
        (Element *)p.k_ptr,
        cute::make_shape(p.seqlen_k, p.d, p.h_k, p.b),
        cute::make_stride(krow, cute::_1{}, (int64_t)p.d, kbatch),
        (Element *)p.v_ptr,
        p.d,
        cute::make_stride(krow, cute::_1{}, (int64_t)p.d, kbatch),
        nullptr, cute::make_shape(0, 0, 0, 0),
        cute::make_stride((int64_t)0, cute::_1{}, (int64_t)0, (int64_t)0),
        nullptr, cute::make_stride((int64_t)0, cute::_1{}, (int64_t)0, (int64_t)0),
        nullptr, cute::make_stride((int64_t)0, cute::_1{}, (int64_t)0, (int64_t)0),
        nullptr, cute::make_shape(0, 0), cute::make_stride((int64_t)0, cute::_1{}),
        nullptr, cute::make_stride((int64_t)0, cute::_1{}),
        false,
        nullptr, cute::make_shape(0, 0), cute::make_stride((int64_t)0, cute::_1{}),
        p.scale_softmax,
        nullptr, nullptr, nullptr,
        cute::make_stride((int64_t)0, (int64_t)0),
        cute::make_stride((int64_t)0, (int64_t)0),
        cute::make_stride((int64_t)0, (int64_t)0),
        -1, -1, 0,
        0.f,
        1,
        nullptr,
        nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr
    };
}

template <typename Traits>
static void sparse_launch_impl(const Sparse_fwd_params &p, cudaStream_t stream) {
    using Element = typename Traits::Element;

    typename Traits::LaunchParams fa3 {
        Traits::Mainloop::to_underlying_arguments(
            make_sparse_mainloop_args<typename Traits::Mainloop, Element>(p)),
        Traits::Epilogue::to_underlying_arguments(
            typename Traits::Epilogue::Arguments {
                (Element *)p.o_ptr,
                cute::make_shape(p.seqlen_q, p.d, p.h, p.b, 1),
                cute::make_stride(p.o_row_stride, cute::_1{}, p.o_head_stride, p.o_batch_stride, (int64_t)0),
                nullptr,
                cute::make_stride((int64_t)0, cute::_1{}, (int64_t)0, (int64_t)0, (int64_t)0),
                (float *)p.softmax_lse_ptr,
                cute::make_stride(cute::_1{}, (int64_t)p.seqlen_q, (int64_t)p.h * p.seqlen_q, (int64_t)0),
                nullptr,
                cute::make_stride(cute::_1{}, (int64_t)0, (int64_t)0, (int64_t)0),
                p.h_k, nullptr, nullptr
            }),
    };

    const int kNThreads = Traits::kNThreads;
    const int smem_size = Traits::kSmemSize;
    auto kernel = (const void *)sparse_fwd_kernel<Traits>;
    cudaFuncSetAttribute((void *)kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    int max_blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, kernel, kNThreads, smem_size);
    int device = 0, num_sms = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
    const int wave   = std::max(1, max_blocks_per_sm) * std::max(1, num_sms);
    // SPARSE_FWD_PLAIN=1 (granularity microbenchmark): the same routine as a plain grid, one CTA per work item, no
    // claim loop
    static const bool plain = getenv("SPARSE_FWD_PLAIN") && atoi(getenv("SPARSE_FWD_PLAIN"));
    const int grid_x = plain ? std::max(p.n_work, 1) : std::min(wave, std::max(p.n_work, 1));

    typename Traits::SchedulerParams sched_params {
        p.n_work, p.work_coord, p.tile_counter, p.num_m_block, p.h, plain ? 1 : 0
    };

    Sparse_fwd_params p_copy = p;
    void *args[] = { (void *)&fa3, (void *)&p_copy, (void *)&sched_params };
    dim3 grid(grid_x), block(kNThreads);
    cudaLaunchKernel(kernel, grid, block, args, smem_size, stream);
}

extern "C" void ring_flash_attn_sparse_launch(const Sparse_fwd_params &p, cudaStream_t stream) {
    sparse_launch_impl<Sparse_fa3_kernel_traits>(p, stream);
}

// 128 x 64 tiles (SpargeAttention2's blocks): kv_indices are 64-row K / V tile ids
extern "C" void ring_flash_attn_sparse64_launch(const Sparse_fwd_params &p, cudaStream_t stream) {
    sparse_launch_impl<Sparse_fa3_kernel_traits_t<64>>(p, stream);
}

// ── Partial-output launch + standalone merge ─────────────────────────────────
// `sparse_fwd_partial` is the sparse kernel writing a partial output: O_partial (fp16, normalised) + LSE_partial
// (fp32). `merge_partials` is the log-sum-exp combine as its own kernel, used by the single-GPU numerics test; inside
// the distributed megakernel the same body runs as a work item.

extern "C" void dist_partial_launch(const Sparse_fwd_params &p, cudaStream_t stream) {
    using Traits  = Dist_fa3_kernel_traits;
    using Element = typename Traits::Element;

    const int     seqlen_o = p.seqlen_o > 0 ? p.seqlen_o : p.seqlen_q;
    const int64_t lse_hs   = p.lse_head_stride > 0 ? p.lse_head_stride : (int64_t)seqlen_o;
    auto o_stride  = cute::make_stride(p.o_row_stride, cute::_1{}, p.o_head_stride,
                                       p.o_batch_stride, (int64_t)0);
    auto lse_stride = cute::make_stride(cute::_1{}, lse_hs, (int64_t)p.h * lse_hs, (int64_t)0);
    auto const ep_partial =
        Traits::Epilogue::to_underlying_arguments(
            typename Traits::Epilogue::Arguments {
                (Element *)p.o_ptr,                             // ptr_O: the partial (fp16, normalised)
                cute::make_shape(seqlen_o, p.d, p.h, p.b, 1),   // +1 degenerate split dim
                o_stride,
                nullptr,                                        // ptr_O_partial (unused: non-split epilogue)
                o_stride,
                (float *)p.softmax_lse_ptr,                     // ptr_LSE
                lse_stride,
                nullptr,                                        // ptr_LSE_partial (unused)
                lse_stride,
                p.h_k, nullptr, nullptr
            });
    // the standalone partial kernel keeps fp16 output (single-GPU numerics test): it writes through epilogue_final;
    // the fp32 partial-epilogue fields are unused here
    auto const ep_unused =
        Traits::EpiloguePart::to_underlying_arguments(
            typename Traits::EpiloguePart::Arguments {
                (typename Traits::ElementPartial *)p.o_ptr, cute::make_shape(seqlen_o, p.d, p.h, p.b, 1), o_stride,
                nullptr, o_stride, (float *)p.softmax_lse_ptr, lse_stride, nullptr, lse_stride, p.h_k, nullptr, nullptr
            });
    typename Traits::LaunchParams fa3 {
        Traits::Mainloop::to_underlying_arguments(
            make_sparse_mainloop_args<typename Traits::Mainloop, Element>(p)),
        ep_unused, ep_partial, ep_unused };

    const int kNThreads = Traits::kNThreads;
    const int smem_size = Traits::kSmemSize;
    auto kernel = (const void *)dist_partial_kernel<Traits>;
    cudaFuncSetAttribute((void *)kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    int max_blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, kernel, kNThreads, smem_size);
    int device = 0, num_sms = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
    const int wave   = std::max(1, max_blocks_per_sm) * std::max(1, num_sms);
    const int grid_x = std::min(wave, std::max(p.n_work, 1));

    typename Traits::SchedulerParams sched_params {
        p.n_work, p.work_coord, p.tile_counter, p.num_m_block, p.h
    };
    Sparse_fwd_params p_copy = p;
    void *args[] = { (void *)&fa3, (void *)&p_copy, (void *)&sched_params };
    dim3 grid(grid_x), block(kNThreads);
    cudaLaunchKernel(kernel, grid, block, args, smem_size, stream);
}

// `max_slots` is ignored: the merge keeps only two per-row statistics in shared
// memory and handles any slot count. Kept in the signature for the Python ABI.
extern "C" void dist_merge_launch(const Merge_params &p, int /*max_slots*/, cudaStream_t stream) {
    if (p.n_merge <= 0) return;
    const size_t smem = (size_t)kMergeSmemBytes;
    auto kernel = (const void *)dist_merge_kernel;
    cudaFuncSetAttribute((void *)kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    Merge_params p_copy = p;
    void *args[] = { (void *)&p_copy };
    dim3 grid(p.n_merge, p.nheads), block(256);
    cudaLaunchKernel(kernel, grid, block, args, smem, stream);
}

// ── Distributed forward megakernel launch ───────────────────────────────────
template <typename Traits>
static void dist_fwd_launch_impl(const Dist_fwd_params &p, cudaStream_t stream) {
    using Element = typename Traits::Element;

    Sparse_fwd_params shim{};
    shim.q_ptr = p.q_slots; shim.k_ptr = p.k_slots; shim.v_ptr = p.v_slots;
    shim.q_batch_stride = (int64_t)p.seqlen_q * p.h * p.d;
    shim.q_row_stride = p.q_row_stride; shim.q_head_stride = p.q_head_stride;
    shim.b = p.b; shim.h = p.h; shim.h_k = p.h_k;
    shim.seqlen_q = p.seqlen_q; shim.seqlen_k = p.seqlen_k; shim.d = p.d;
    shim.scale_softmax = p.scale_softmax;

    // partial slots: head-major -> rows = NPART*128, one batch; combined (O then LSE per slot) -> each slot is a
    // batch of 128 rows, so the kernels address a partial slot as (row block 0, head, batch = slot)
    auto o_stride  = cute::make_stride(p.o_row_stride, cute::_1{}, p.o_head_stride,
                                       p.part_combined ? p.part_o_slot : (int64_t)p.seqlen_o * p.h * p.d, (int64_t)0);
    auto lse_str   = cute::make_stride(cute::_1{}, p.lse_head_stride,
                                       p.part_combined ? p.part_lse_slot : (int64_t)p.h * p.lse_head_stride,
                                       (int64_t)0);
    const int part_rows = p.part_combined ? 128 : p.seqlen_o;
    const int part_batches = p.part_combined ? p.NPART : p.b;
    const int n_pool = (int)std::max<int64_t>(p.pool_lse_stride, 128);
    auto po_stride  = cute::make_stride(p.o_row_stride, cute::_1{}, p.o_head_stride,
                                        (int64_t)n_pool * p.h * p.d, (int64_t)0);
    auto plse_str   = cute::make_stride(cute::_1{}, (int64_t)n_pool, (int64_t)p.h * n_pool, (int64_t)0);
    // final O [n_owned*128][h][d] fp16, LSE [h][n_owned*128] fp32: the same format EpiloguePart / Epilogue write
    const int n_out = (int)(p.flse_head_stride);
    auto fo_stride  = cute::make_stride(p.fo_row_stride, cute::_1{}, p.fo_head_stride,
                                        (int64_t)n_out * p.h * p.d, (int64_t)0);
    auto flse_str   = cute::make_stride(cute::_1{}, p.flse_head_stride,
                                        (int64_t)p.h * p.flse_head_stride, (int64_t)0);
    typename Traits::LaunchParams fa3 {
        Traits::Mainloop::to_underlying_arguments(
            make_sparse_mainloop_args<typename Traits::Mainloop, Element>(shim)),
        Traits::EpiloguePart::to_underlying_arguments(      // partial slots: fp32 O
            typename Traits::EpiloguePart::Arguments {
                (typename Traits::ElementPartial *)p.part_o, cute::make_shape(part_rows, p.d, p.h, part_batches, 1), o_stride,
                nullptr, o_stride,
                (float *)p.part_lse, lse_str, nullptr, lse_str,
                p.h_k, nullptr, nullptr
            }),
        Traits::Epilogue::to_underlying_arguments(
            typename Traits::Epilogue::Arguments {
                (Element *)p.o_ptr, cute::make_shape(n_out, p.d, p.h, p.b, 1), fo_stride,
                nullptr, fo_stride,
                (float *)p.lse_ptr, flse_str, nullptr, flse_str,
                p.h_k, nullptr, nullptr
            }),
        Traits::EpiloguePart::to_underlying_arguments(      // local pool of per-task partials (sched kernel), fp32 O
            typename Traits::EpiloguePart::Arguments {
                (typename Traits::ElementPartial *)p.pool_o, cute::make_shape(n_pool, p.d, p.h, p.b, 1), po_stride,
                nullptr, po_stride,
                (float *)p.pool_lse, plse_str, nullptr, plse_str,
                p.h_k, nullptr, nullptr
            }),
    };

    const int kNThreads = Traits::kNThreads;
    // sched kernel: + its control words past the compute area (no static shared memory, see SchedCtl)
    const int smem_size = p.row_grp_ptr ? sched_smem_bytes<Traits>() : fwd_smem_bytes<Traits>();
    const void *kernel;
    if constexpr (Traits::kBlockN == 64)                   // 128 x 64 lists: only the block-0 scheduler path (dx)
        kernel = (const void *)dist_sched_kernel<Traits, false>;
    else
        kernel = p.row_grp_ptr ? (p.rect ? (const void *)dist_sched_kernel<Traits, true>      // block-0 task scheduler
                                         : (const void *)dist_sched_kernel<Traits, false>)
               : p.dep_ptr ? (const void *)dist_fwd_kernel<Traits, true>      // ready queue: no tile gate
                           : (const void *)dist_fwd_kernel<Traits, false>;
    cudaFuncSetAttribute((void *)kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    int max_blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, kernel, kNThreads, smem_size);
    int device = 0, num_sms = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
    const int wave = std::max(1, max_blocks_per_sm) * std::max(1, num_sms);
    // +1 for the delegate block, which never takes work.
    // sched kernel: the amount of work is decided at run time -> one full wave (block 0 + workers)
    const int grid_x = p.row_grp_ptr ? std::max(2, wave)
                                     : std::max(2, std::min(wave, std::max(p.n_work, 1) + 1));

    // Block 0 is the delegate only when there is something to poll for or push; otherwise it computes too.
    const int n_reserved = (p.row_grp_ptr || p.n_remote > 0 || p.n_pushq + p.n_pushkv > 0 ||
                            (p.use_engine && p.eng.n_start + p.eng.n_pq > 0)) ? 1 : 0;
    DistTileScheduler::Params sched_params {
        p.n_work, p.work_coord, p.tile_counter, std::max(p.n_compute, 1), p.h, n_reserved
    };

    Dist_fwd_params p_copy = p;
    if (p_copy.b0_bytes <= Traits::kSmemSize) p_copy.b0_state = nullptr;   // block-0 state fits below the control words
    if (p_copy.num_push_blocks > grid_x - p_copy.push_block_base)
        p_copy.num_push_blocks = grid_x - p_copy.push_block_base;
    // sched kernel: block 0 and the push blocks get no static first queue slot (dist_sched_kernel.h)
    if (p.row_grp_ptr)
        sched_params.n_reserved = std::max(1, std::min(grid_x - 1, p_copy.push_block_base + p_copy.num_push_blocks));
    if (p.sp_on) {                                           // the shared scheduler (rt_sched.cuh)
        p_copy.sp.b0_state = p.sp.b0_bytes <= Traits::kSmemSize ? nullptr : p.sp.b0_state;
        p_copy.sp.n_workers = grid_x - 1; p_copy.sp.n_static = grid_x - sched_params.n_reserved;
    }
    void *args[] = { (void *)&fa3, (void *)&p_copy, (void *)&sched_params };
    dim3 grid(grid_x), block(kNThreads);
    cudaLaunchKernel(kernel, grid, block, args, smem_size, stream);
}

extern "C" void dist_fwd_launch(const Dist_fwd_params &p, cudaStream_t stream) {
    if (p.kv_half) dist_fwd_launch_impl<Dist_fa3_kernel_traits_t<64, 4096>>(p, stream);   // 128 x 64 compute lists
    else dist_fwd_launch_impl<Dist_fa3_kernel_traits>(p, stream);
}


extern "C" void dist_scatter3_launch(const Scatter_params &q, const Scatter_params &k, const Scatter_params &v,
                                     cudaStream_t stream) {
    constexpr int64_t kChunk = 64 << 10;                     // must match dist_scatter3_kernel
    auto blocks = [&](const Scatter_params &p) { return (int64_t)p.n * ((p.slot_bytes + kChunk - 1) / kChunk); };
    const int64_t n = std::max(blocks(q), std::max(blocks(k), blocks(v)));
    if (n <= 0) return;
    Scatter_params a = q, b = k, c = v;
    void *args[] = { (void *)&a, (void *)&b, (void *)&c };
    dim3 grid((unsigned)n, 3), block(256);
    cudaLaunchKernel((const void *)dist_scatter3_kernel, grid, block, args, 0, stream);
}

extern "C" void dist_scatter_launch(const Scatter_params &p, cudaStream_t stream) {
    if (p.n <= 0) return;
    Scatter_params c = p;
    void *args[] = { (void *)&c };
    dim3 grid(p.n), block(256);
    cudaLaunchKernel((const void *)dist_scatter_kernel, grid, block, args, 0, stream);
}



// ---- copy-engine push: after a chunk of cudaMemcpyAsync peer copies on a side stream, publish the
// chunk's signal ids on the destination (peer_sig is the P2P-mapped address of its signal words).
// Stream order makes the copies complete before this runs; the system fence orders our writes.
__global__ void dist_ce_signal_kernel(uint64_t *peer_sig, const int32_t *ids, int n, uint64_t epoch) {
    __threadfence_system();
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        *(volatile uint64_t *)&peer_sig[ids[i]] = epoch;
}
extern "C" void dist_ce_signal_launch(uint64_t *peer_sig, const int32_t *ids, int n, uint64_t epoch,
                                      cudaStream_t stream) {
    if (n > 0) dist_ce_signal_kernel<<<1, 128, 0, stream>>>(peer_sig, ids, n, epoch);
}

