#pragma once
// Plain-data parameters for the block-sparse forward kernel. Host (.cpp) and
// device (.cu) both include this, so no CUDA-only syntax here.
#include <cstdint>

namespace ring_attn {

struct Sparse_fwd_params {
    using index_t = int64_t;

    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;
    void *__restrict__ o_ptr;
    void *__restrict__ softmax_lse_ptr;   // (b, h, seqlen_q) fp32

    index_t q_batch_stride, q_row_stride, q_head_stride;
    index_t o_batch_stride, o_row_stride, o_head_stride;

    int b, h, h_k;
    int seqlen_q, seqlen_k, d;
    float scale_softmax;

    // The OUTPUT may live in a different, larger space than Q: a device writes
    // its partial into one slot of a shared partial region, so the epilogue's
    // seqlen and LSE head stride come from the output tensor, not from Q.
    // Zero means "same as Q" (the plain single-GPU path).
    int     seqlen_o;
    index_t lse_head_stride;

    // ── the mask, as a CSR over Q tiles (see sparse_desc.h) ──
    const int32_t *kv_indptr;      // [num_m_block + 1]
    const int32_t *kv_indices;     // [nnz], ascending within a row, masked last
    const int32_t *kv_mask_nb;     // [nnz], the mask's n_block for each entry
    const int32_t *row_mask_mb;    // [num_m_block], the mask's m_block for each row
    const int32_t *row_n_masked;   // [num_m_block]

    // Claim order. Permuting this is how the host load-balances: the kernel
    // hands tiles out in exactly this order, so putting the heaviest rows first
    // (longest-processing-time) is a one-line host change, not a search.
    const int32_t *work_coord;     // [n_work] packed (bidb * h + bidh) * num_m_block + m_block
    int n_work;
    int num_m_block;

    int *tile_counter;             // device scratch, one int, zeroed per launch
};

}  // namespace ring_attn
