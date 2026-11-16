#pragma once
// ── Sparse (list-driven) attention backward, one KV column block per task ─────────────────────────────────────────
// FlashAttention-2's sm80 backward mainloop (compute_dq_dk_dv_1colblock, sequence-parallel flavour: dQ by fp32
// atomicAdd into dq_accum) with ONE change of control flow: instead of walking m_block = m_max-1 .. m_min over a dense
// sequence, a task walks a planner-given ITEM LIST. A task = (KV 64-row block n64, KV head g); an item = (Q 64-row block
// q64, q head hq, causal?) -- so a task may mix Q tiles from anywhere (sparse masks, remote slots) and all q heads of
// its KV head's GQA group (dK/dV are accumulated across the group in registers: no per-q-head dK/dV + sum as in FA2).
// Everything else (tiling, smem layouts, double-buffered sQ, V in registers, the five GEMMs) is FA2's, unchanged.
//
// Storage is SLOT-MAJOR (a slot = one 128-token tile; slot order is arbitrary, positions only matter for the causal
// mask, which is tile-local). Inside a slot rows are [128][heads][d]:
//   q, dout      fp16, slot stride q_slot_stride (elements)    lse, dpsum  fp32 [h][128] per slot: element
//                                                              slot * lse_slot_stride + head * lse_head_stride + row
//   k, v         fp16, slot stride k_slot_stride               dq_accum    fp32, slot stride dq_slot_stride (unscaled,
//                                                              FA2's permuted order inside each 64x128 block)
//   dk, dv       fp16 (direct store; whole column in one task)  or  dk_accum, dv_accum fp32 (atomicAdd), slot stride
//                dkv_slot_stride -- split columns / partial sums
// (A plain [rows, h, d] tensor is the special case q_slot_stride = 128 * h * d, lse_slot_stride = 128.)
// Item encoding (int32): (q_slot << 9) | (mb << 8) | (hq << 1) | causal: 64-row half mb of Q slot q_slot, q head hq;
// causal = a causal diagonal 64x64 block (col > row masked); fully masked blocks are simply not listed.
// Task KV encoding: (kv_slot << 9) | (hn << 8) | g: 64-row half hn of KV slot kv_slot, KV head g.

// FA2's headers in their own namespace (the forward's FA3 headers use `flash`), included by explicit path: the main
// extension's include path also has FA3 headers with the same file names (utils.h, softmax.h, mask.h).
#ifndef FLASH_NAMESPACE
#define FLASH_NAMESPACE fa2bwd
#endif
#include "flash_attn/src/namespace_config.h"
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include "flash_attn/src/kernel_traits.h"
#include "flash_attn/src/utils.h"
#include "flash_attn/src/softmax.h"
#include "flash_attn/src/mask.h"
#include "sparse_bwd_params.h"

namespace sparse_bwd {

using namespace cute;
namespace F = FLASH_NAMESPACE;



__device__ __forceinline__ int item_slot(int it) { return it >> 9; }
__device__ __forceinline__ int item_mb(int it) { return (it >> 8) & 1; }
__device__ __forceinline__ int item_hq(int it) { return (it >> 1) & 127; }
__device__ __forceinline__ bool item_causal(int it) { return it & 1; }

template <int MMA_N, class... Args, class TiledMMA>
CUTE_HOST_DEVICE auto make_tiled_copy_B_warpcontiguousN(Copy_Atom<Args...> const &copy_atom, TiledMMA const &tiled_mma) {
    constexpr int TileShape_N = decltype(tiled_mma.template tile_size_mnk<1>())::value;
    constexpr int TileShape_K = decltype(tiled_mma.template tile_size_mnk<2>())::value;
    using AtomShape_MNK = typename TiledMMA::AtomShape_MNK;
    constexpr int AtomShape_N = decltype(size<1>(AtomShape_MNK{}))::value;
    constexpr int kNWarpsN = TileShape_N / AtomShape_N / 2;
    constexpr int MMAStride_N = MMA_N * AtomShape_N * 2;
    auto t = make_tile(Layout<Shape<Int<AtomShape_N>, Int<kNWarpsN>, _2>, Stride<_1, Int<MMAStride_N>, _8>>{},
                       make_layout(Int<TileShape_K>{}));
    return make_tiled_copy_impl(copy_atom, tiled_mma.get_layoutB_TV(), t);
}
template <int MMA_N, class... Args, class TiledMMA>
CUTE_HOST_DEVICE auto make_tiled_copy_C_warpcontiguousN(Copy_Atom<Args...> const &copy_atom, TiledMMA const &tiled_mma) {
    constexpr int TileShape_M = decltype(tiled_mma.template tile_size_mnk<0>())::value;
    constexpr int TileShape_N = decltype(tiled_mma.template tile_size_mnk<1>())::value;
    using AtomShape_MNK = typename TiledMMA::AtomShape_MNK;
    constexpr int AtomShape_N = decltype(size<1>(AtomShape_MNK{}))::value;
    constexpr int kNWarpsN = TileShape_N / AtomShape_N / 2;
    constexpr int MMAStride_N = MMA_N * AtomShape_N * 2;
    auto t = make_tile(make_layout(Int<TileShape_M>{}),
                       Layout<Shape<Int<AtomShape_N>, Int<kNWarpsN>, _2>, Stride<_1, Int<MMAStride_N>, _8>>{});
    return make_tiled_copy_impl(copy_atom, tiled_mma.get_layoutC_TV(), t);
}

// One task: KV block (kv slot, half hn, head g) x items. Must be entered by all threads; returns with smem free.
template <typename Kernel_traits, bool Atomic_dKV>
inline __device__ void bwd_task(const Params &params, const int kvs, const int hn, const int g, const int *items,
                                const int n_items, char *smem_) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = int64_t;
    const int tidx = threadIdx.x;
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    static_assert(kBlockM == 64 && kBlockN == 64, "items / tasks are 64-row blocks");
    if (n_items <= 0) return;                    // (a 64-key half with no live row: 128 x 64 masks; uniform, before any work)
    constexpr int MMA_N_SdP = kBlockN / decltype(typename Kernel_traits::TiledMmaSdP{}.template tile_size_mnk<1>())::value;
    constexpr int AtomLayoutMS = Kernel_traits::AtomLayoutMSdP;
    constexpr bool Double_buffer = !Kernel_traits::No_double_buffer;

    // per-item element offsets (q / dO / dq_accum share the [rows, h, d] layout; lse / dpsum [h, stride])
    auto q_in = [&](int it) -> index_t {          // inside the slot: half mb, head hq (Q, dO and dQ share it)
        return (index_t)item_mb(it) * kBlockM * params.q_row_stride + (index_t)item_hq(it) * params.d; };
    auto q_off = [&](int it) -> index_t { return (index_t)item_slot(it) * params.q_slot_stride + q_in(it); };
    auto dq_off = [&](int it) -> index_t { return (index_t)item_slot(it) * params.dq_slot_stride + q_in(it); };
    auto l_off = [&](int it) -> index_t {
        return (index_t)item_slot(it) * params.lse_slot_stride + (index_t)item_hq(it) * params.lse_head_stride
             + (index_t)item_mb(it) * kBlockM; };
    const int it0 = items[0];
    const index_t kv_in = (index_t)hn * kBlockN * params.k_row_stride + (index_t)g * params.d;
    const index_t row_offset_k = (index_t)kvs * params.k_slot_stride + kv_in;
    const index_t row_offset_dkv = (index_t)kvs * params.dkv_slot_stride + kv_in;

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q) + q_off(it0)),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.k_row_stride, _1{}));
    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dout) + q_off(it0)),
                             Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.q_row_stride, _1{}));
    // dQ accumulator block of the current item, in FA2's PERMUTED order: register i of thread t is atomically added
    // at the address the 8x32 atomic-add copy layout gives it (coalesced), not at its MMA coordinate; every writer
    // (all tasks, all GPUs' partial sums) uses the same permutation and dq_convert undoes it
    Tensor gdQaccum = make_tensor(make_gmem_ptr(params.dq_accum + dq_off(it0)),
                                  Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.q_row_stride, _1{}));
    typename Kernel_traits::GmemTiledCopydQaccumAtomicAdd gmem_tiled_copy_dQaccum;
    auto gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_thread_slice(tidx);
    Tensor tdQgdQaccum = gmem_thr_copy_dQaccum.partition_D(gdQaccum);
    Tensor gLSE = make_tensor(make_gmem_ptr(params.lse + l_off(it0)), Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gdPsum = make_tensor(make_gmem_ptr(params.dpsum + l_off(it0)), Shape<Int<kBlockM>>{}, Stride<_1>{});

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQdO{});
    Tensor sQt = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOtransposed{});
    Tensor sQtNoSwizzle = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOtransposedNoSwizzle{});
    Tensor sdO = make_tensor(sQ.data() + (Double_buffer ? 2 : 1) * size(sQ), typename Kernel_traits::SmemLayoutQdO{});
    Tensor sdOt = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutQdOtransposed{});
    Tensor sdOtransposedNoSwizzle = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutQdOtransposedNoSwizzle{});
    Tensor sK = make_tensor(sdO.data() + size(sdO), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sKt = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKtransposed{});
    Tensor sKtNoSwizzle = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKtransposedNoSwizzle{});
    Tensor sdS = make_tensor(!Kernel_traits::Is_V_in_regs ? sV.data() + size(sV) : sK.data() + size(sK),
                             typename Kernel_traits::SmemLayoutPdS{});
    Tensor sdSt = make_tensor(sdS.data(), typename Kernel_traits::SmemLayoutPdStransposed{});
    Tensor sdStNoSwizzle = make_tensor(sdS.data(), typename Kernel_traits::SmemLayoutPdStransposedNoSwizzle{});
    Tensor sP = make_tensor(sdS.data() + size(sdS), typename Kernel_traits::SmemLayoutPdS{});
    Tensor sPt = make_tensor(sP.data(), typename Kernel_traits::SmemLayoutPdStransposed{});
    Tensor sPtNoSwizzle = make_tensor(sP.data(), typename Kernel_traits::SmemLayoutPdStransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_dO;     // !Is_first: dO straight to smem
    auto gmem_thr_copy_dO = gmem_tiled_copy_dO.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tdOgdO = gmem_thr_copy_dO.partition_S(gdO);
    Tensor tdOsdO = gmem_thr_copy_dO.partition_D(sdO);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma_sdp.partition_fragment_B(sK);
    Tensor tdPrdO = thr_mma_sdp.partition_fragment_A(sdO);
    Tensor tdPrV = thr_mma_sdp.partition_fragment_B(sV);

    typename Kernel_traits::TiledMmadKV tiled_mma_dkv;
    auto thr_mma_dkv = tiled_mma_dkv.get_thread_slice(tidx);
    Tensor tdKrdSt = thr_mma_dkv.partition_fragment_A(sdStNoSwizzle);
    Tensor tdKrQt = thr_mma_dkv.partition_fragment_B(sQtNoSwizzle);
    Tensor tdVrPt = thr_mma_dkv.partition_fragment_A(sPtNoSwizzle);
    Tensor tdVrdO = thr_mma_dkv.partition_fragment_B(sdOtransposedNoSwizzle);

    typename Kernel_traits::TiledMmadQ tiled_mma_dq;
    auto thr_mma_dq = tiled_mma_dq.get_thread_slice(tidx);
    Tensor tdQrdS = thr_mma_dq.partition_fragment_A(sdS);
    Tensor tdQrKt = thr_mma_dq.partition_fragment_B(sKtNoSwizzle);

    Tensor acc_dk = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});
    Tensor acc_dv = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});

    auto smem_tiled_copy_QdO = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_QdO = smem_tiled_copy_QdO.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_QdO.partition_S(sQ);
    Tensor tdPsdO = smem_thr_copy_QdO.partition_S(sdO);
    auto smem_tiled_copy_KV = make_tiled_copy_B_warpcontiguousN<MMA_N_SdP>(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_KV = smem_tiled_copy_KV.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_KV.partition_S(sK);
    Tensor tdPsV = smem_thr_copy_KV.partition_S(sV);
    auto smem_tiled_copy_PdS = make_tiled_copy_C_warpcontiguousN<MMA_N_SdP>(typename Kernel_traits::SmemCopyAtomPdS{}, tiled_mma_sdp);
    auto smem_thr_copy_PdS = smem_tiled_copy_PdS.get_thread_slice(tidx);
    Tensor tPsP = smem_thr_copy_PdS.partition_D(sP);
    Tensor tdSsdS = smem_thr_copy_PdS.partition_D(sdS);
    auto smem_tiled_copy_PdSt = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dkv);
    auto smem_thr_copy_PdSt = smem_tiled_copy_PdSt.get_thread_slice(tidx);
    Tensor tdVsPt = smem_thr_copy_PdSt.partition_S(sPt);
    Tensor tdKsdSt = smem_thr_copy_PdSt.partition_S(sdSt);
    auto smem_tiled_copy_QdOt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dkv);
    auto smem_thr_copy_QdOt = smem_tiled_copy_QdOt.get_thread_slice(tidx);
    Tensor tdVsdOt = smem_thr_copy_QdOt.partition_S(sdOt);
    Tensor tdKsQt = smem_thr_copy_QdOt.partition_S(sQt);
    auto smem_tiled_copy_dS = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_dq);
    auto smem_thr_copy_dS = smem_tiled_copy_dS.get_thread_slice(tidx);
    Tensor tdQsdS = smem_thr_copy_dS.partition_S(sdS);
    auto smem_tiled_copy_Kt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dq);
    auto smem_thr_copy_Kt = smem_tiled_copy_Kt.get_thread_slice(tidx);
    Tensor tdQsKt = smem_thr_copy_Kt.partition_S(sKt);

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
    Tensor tQcQ = gmem_thr_copy_QKV.partition_D(cQ);
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_D(cKV);
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    __syncthreads();                       // the previous task (persistent loop) may still read this smem
    if (Kernel_traits::Is_V_in_regs) {
        F::copy</*Is_even_MN=*/true, /*Is_even_K=*/true, /*Clear_OOB_MN=*/true>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, kBlockN);
        F::cp_async_fence();
    }
    F::copy<true, true, true>(gmem_tiled_copy_dO, tdOgdO, tdOsdO, tQcQ, tQpQ, kBlockM);
    F::copy<true, true, true>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ, kBlockM);

    Tensor caccS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);
    static_assert(decltype(size<0>(taccScS))::value == 4);
    Tensor taccScS_row = logical_divide(taccScS, Shape<_2>{})(make_coord(0, _), _, 0);
    Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) { lse(mi) = gLSE(get<0>(taccScS_row(mi))); }

    F::copy<true, true, true>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV, kBlockN);
    if (!Kernel_traits::Is_V_in_regs) {
        F::copy<true, true, true>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, kBlockN);
    }
    F::cp_async_fence();

    if (Kernel_traits::Is_V_in_regs) {
        cute::cp_async_wait<1>();
        __syncthreads();
        Tensor tdPrV_copy_view = smem_thr_copy_KV.retile_D(tdPrV);
        CUTE_STATIC_ASSERT_V(size<1>(tdPsV) == size<1>(tdPrV_copy_view));
        cute::copy(smem_tiled_copy_KV, tdPsV, tdPrV_copy_view);
    }
    clear(acc_dv);
    clear(acc_dk);

    int cur = it0;
    for (int ii = 0; ii < n_items; ++ii) {
        const bool has_next = ii + 1 < n_items;
        const int nxt = has_next ? items[ii + 1] : cur;
        Tensor acc_s = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);
        cute::cp_async_wait<0>();
        __syncthreads();

        Tensor dP_sum = make_fragment_like(lse);
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) { dP_sum(mi) = gdPsum(get<0>(taccScS_row(mi))); }

        F::gemm(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma_sdp, smem_tiled_copy_QdO, smem_tiled_copy_KV,
                smem_thr_copy_QdO, smem_thr_copy_KV);
        Tensor scores = make_tensor(acc_s.data(), F::convert_layout_acc_rowcol(acc_s.layout()));
        if (item_causal(cur)) {           // tile-local 64x64 diagonal block: col > row masked
            F::apply_mask_causal(scores, (tidx / 32 / AtomLayoutMS) * MMA_N_SdP * 16, kBlockN,
                                 get<0>(taccScS_row(0)), kBlockM, AtomLayoutMS * 16);
        }
        F::scale_apply_exp2</*scale_max=*/false>(scores, lse, params.scale_softmax_log2);
        Tensor rP = F::convert_type<Element>(acc_s);
        Tensor tPrP = make_tensor(rP.data(), F::convert_layout_acc_Aregs<typename Kernel_traits::TiledMmaSdP>(rP.layout()));
        Tensor tPaP = smem_thr_copy_PdS.retile_S(tPrP);
        cute::copy(smem_tiled_copy_PdS, tPaP, tPsP);

        Tensor acc_dp = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_dp);
        F::gemm</*A_in_regs=*/false, /*B_in_regs=*/Kernel_traits::Is_V_in_regs>(
            acc_dp, tdPrdO, tdPrV, tdPsdO, tdPsV, tiled_mma_sdp, smem_tiled_copy_QdO, smem_tiled_copy_KV,
            smem_thr_copy_QdO, smem_thr_copy_KV);
        Tensor dS = make_tensor(acc_dp.data(), scores.layout());
        #pragma unroll
        for (int mi = 0; mi < size<0>(dS); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(dS); ++ni) { dS(mi, ni) = scores(mi, ni) * (dS(mi, ni) - dP_sum(mi)); }
        }

        Tensor acc_dq = partition_fragment_C(tiled_mma_dq, Shape<Int<kBlockM>, Int<kHeadDim>>{});
        clear(acc_dq);
        const index_t dq_delta = q_off(nxt) - q_off(cur);       // Q / dO: same layout
        const index_t dqa_delta = dq_off(nxt) - dq_off(cur);    // dQ accumulator
        if (Double_buffer && has_next) {
            const int sQ_offset = ii % 2 == 0 ? size(sQ) : -size(sQ);
            tQsQ.data() = tQsQ.data() + sQ_offset;
            tSsQ.data() = tSsQ.data() + sQ_offset;
            tQgQ.data() = tQgQ.data() + dq_delta;
            F::copy</*Is_even_MN=*/true, true>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ);
            F::cp_async_fence();
        }

        Tensor dS_reshaped = make_tensor(dS.data(), acc_dp.layout());
        Tensor tdSrdS = F::convert_type<Element>(dS_reshaped);
        Tensor tdSadS = smem_thr_copy_PdS.retile_S(tdSrdS);
        cute::copy(smem_tiled_copy_PdS, tdSadS, tdSsdS);
        __syncthreads();

        F::gemm(acc_dv, tdVrPt, tdVrdO, tdVsPt, tdVsdOt, tiled_mma_dkv, smem_tiled_copy_PdSt, smem_tiled_copy_QdOt,
                smem_thr_copy_PdSt, smem_thr_copy_QdOt);
        __syncthreads();
        if (has_next) {
            tdOgdO.data() = tdOgdO.data() + dq_delta;
            F::copy<true, true>(gmem_tiled_copy_dO, tdOgdO, tdOsdO, tQcQ, tQpQ);
            F::cp_async_fence();
        }

        F::gemm(acc_dq, tdQrdS, tdQrKt, tdQsdS, tdQsKt, tiled_mma_dq, smem_tiled_copy_dS, smem_tiled_copy_Kt,
                smem_thr_copy_dS, smem_thr_copy_Kt);
        if (has_next) {
            const index_t l_delta = l_off(nxt) - l_off(cur);
            gLSE.data() = gLSE.data() + l_delta;
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) { lse(mi) = gLSE(get<0>(taccScS_row(mi))); }
            gdPsum.data() = gdPsum.data() + l_delta;
        }
        CUTE_STATIC_ASSERT_V(size(acc_dq) == size(tdQgdQaccum));
        #pragma unroll
        for (int i = 0; i < size(acc_dq); ++i) { atomicAdd(&tdQgdQaccum(i), acc_dq(i)); }
        tdQgdQaccum.data() = tdQgdQaccum.data() + dqa_delta;

        F::gemm(acc_dk, tdKrdSt, tdKrQt, tdKsdSt, tdKsQt, tiled_mma_dkv, smem_tiled_copy_PdSt, smem_tiled_copy_QdOt,
                smem_thr_copy_PdSt, smem_thr_copy_QdOt);
        if (Double_buffer) { tdKsQt.data() = tdKsQt.data() + (ii % 2 == 0 ? size(sQ) : -size(sQ)); }
        if (!Double_buffer && has_next) {
            __syncthreads();
            tQgQ.data() = tQgQ.data() + dq_delta;
            F::copy<true, true>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ);
            F::cp_async_fence();
        }
        cur = nxt;
    }

    #pragma unroll
    for (int i = 0; i < size(acc_dk); ++i) { acc_dk(i) *= params.scale_softmax; }

    if constexpr (Atomic_dKV) {
        // fp32 atomicAdd straight from the accumulators (no smem round trip): the task may be one of several on
        // this column (split column / partial sum for a remote owner)
        Tensor cdKV = make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});
        Tensor tcdKV = thr_mma_dkv.partition_C(cdKV);
        float *dk = params.dk_accum + row_offset_dkv, *dv = params.dv_accum + row_offset_dkv;
        #pragma unroll
        for (int i = 0; i < size(acc_dk); ++i) {
            const index_t o = (index_t)get<0>(tcdKV(i)) * params.k_row_stride + get<1>(tcdKV(i));
            atomicAdd(dk + o, acc_dk(i));
            atomicAdd(dv + o, acc_dv(i));
        }
    } else {
        Tensor rdK = F::convert_type<Element>(acc_dk);
        Tensor rdV = F::convert_type<Element>(acc_dv);
        Tensor sdK = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutdKV{});
        Tensor sdV = make_tensor(sdK.data() + size(sdK), typename Kernel_traits::SmemLayoutdKV{});
        auto smem_tiled_copy_dKV = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdKV{}, tiled_mma_dkv);
        auto smem_thr_copy_dKV = smem_tiled_copy_dKV.get_thread_slice(tidx);
        Tensor taccdKrdK = smem_thr_copy_dKV.retile_S(rdK);
        Tensor taccdKsdK = smem_thr_copy_dKV.partition_D(sdK);
        Tensor taccdVrdV = smem_thr_copy_dKV.retile_S(rdV);
        Tensor taccdVsdV = smem_thr_copy_dKV.partition_D(sdV);
        __syncthreads();
        cute::copy(smem_tiled_copy_dKV, taccdKrdK, taccdKsdK);
        cute::copy(smem_tiled_copy_dKV, taccdVrdV, taccdVsdV);
        Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk) + row_offset_k),
                                 Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.k_row_stride, _1{}));
        Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv) + row_offset_k),
                                 Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(params.k_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
        auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
        Tensor tdKsdK = gmem_thr_copy_dKV.partition_S(sdK);
        Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
        Tensor tdVsdV = gmem_thr_copy_dKV.partition_S(sdV);
        Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);
        __syncthreads();
        Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
        cute::copy(gmem_tiled_copy_dKV, tdKsdK, tdKrdK);
        Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
        cute::copy(gmem_tiled_copy_dKV, tdVsdV, tdVrdV);
        Tensor cdKV = make_identity_tensor(make_shape(size<0>(sdK), size<1>(sdK)));
        Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
        Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKgdK)));
        F::copy<true, true, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKVcdKV, tdKVpdKV, kBlockN);
        F::copy<true, true, false, false>(gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdKVcdKV, tdKVpdKV, kBlockN);
    }
}

// Non-persistent launch (step 1): one CTA per task. Tasks with no item write zeros (direct mode) / nothing (atomic).
template <typename Kernel_traits, bool Atomic_dKV>
__global__ void __launch_bounds__(Kernel_traits::kNThreads) sparse_bwd_kernel(const Params params) {
    extern __shared__ char smem_[];
    const int t = blockIdx.x;
    const int ib = params.task_ptr[t], ie = params.task_ptr[t + 1];
    const int tk = params.task_kv[t];
    if (ie > ib) bwd_task<Kernel_traits, Atomic_dKV>(params, tk >> 9, (tk >> 8) & 1, tk & 255, params.items + ib, ie - ib, smem_);
}

// dpsum[hq, row] = sum_d dO * O (fp32); one warp per (row, head); also zeroes dq_accum when asked
static __global__ void dot_do_o_kernel(const __half *dout, const __half *o, float *dpsum, int rows, int h, int d,
                                int64_t lse_stride, float *dq_accum_to_zero) {
    const int w = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
    if (w >= rows * h) return;
    const int r = w / h, hh = w % h;
    const __half *a = dout + ((int64_t)r * h + hh) * d, *b = o + ((int64_t)r * h + hh) * d;
    float s = 0.f;
    for (int i = lane * 2; i < d; i += 64) {
        const __half2 x = *reinterpret_cast<const __half2 *>(a + i), y = *reinterpret_cast<const __half2 *>(b + i);
        const float2 fx = __half22float2(x), fy = __half22float2(y);
        s += fx.x * fy.x + fx.y * fy.y;
    }
    #pragma unroll
    for (int o2 = 16; o2; o2 >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o2);
    if (lane == 0) dpsum[(int64_t)hh * lse_stride + r] = s;
    if (dq_accum_to_zero)
        for (int i = lane; i < d; i += 32) dq_accum_to_zero[((int64_t)r * h + hh) * d + i] = 0.f;
}

// dQ = scale * dq_accum, undoing the in-block permutation of the atomics (same two partitions, same thread)
template <typename Kernel_traits>
__global__ void __launch_bounds__(Kernel_traits::kNThreads) dq_convert_kernel(const float *acc, __half *out,
                                                                            int64_t row_stride, int n64, int h, float scale) {
    constexpr int kBlockM = Kernel_traits::kBlockM, kHeadDim = Kernel_traits::kHeadDim;
    const int b = blockIdx.x / h, hh = blockIdx.x % h, tidx = threadIdx.x;
    const int64_t off = (int64_t)b * kBlockM * row_stride + (int64_t)hh * kHeadDim;
    Tensor g = make_tensor(make_gmem_ptr(acc + off), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(row_stride, _1{}));
    typename Kernel_traits::GmemTiledCopydQaccumAtomicAdd cp;
    Tensor tg = cp.get_thread_slice(tidx).partition_D(g);
    typename Kernel_traits::TiledMmadQ mma;
    Tensor tc = mma.get_thread_slice(tidx).partition_C(make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    CUTE_STATIC_ASSERT_V(size(tg) == size(tc));
    #pragma unroll
    for (int i = 0; i < size(tg); ++i)
        out[off + (int64_t)get<0>(tc(i)) * row_stride + get<1>(tc(i))] = __float2half_rn(tg(i) * scale);
}

// out = scale * acc (fp32 -> fp16)
static __global__ void convert_kernel(const float *acc, __half *out, int64_t n, float scale) {
    const int64_t i = (int64_t)(blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= n) return;
    const float4 a = *reinterpret_cast<const float4 *>(acc + i);
    __half2 *o = reinterpret_cast<__half2 *>(out + i);
    o[0] = __floats2half2_rn(a.x * scale, a.y * scale);
    o[1] = __floats2half2_rn(a.z * scale, a.w * scale);
}

}  // namespace sparse_bwd
