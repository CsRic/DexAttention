#pragma once
#include <cuda_fp16.h>
// Plain-data parameters for the merge, shared between host (.cpp) and device.
// `o_ptr` is void* on purpose: the host TU must not need a CUDA fp16 definition.
#include <cstdint>

namespace ring_attn {

struct Merge_params {
    void    *o_ptr;              // final O,   (tokens, h, d) fp16
    float   *lse_ptr;            // final LSE, [h][tokens]
    const __half *part_o;        // [slot*128 + row][h][d] fp16
    const float *part_lse;       // [h][slot*128 + row]    fp32
    const int32_t *merge_ptr;    // [n_merge + 1]
    const int32_t *merge_slot;   // [nnz] partial slots to combine
    const int32_t *merge_out;    // [n_merge] which output tile each lands in
    int n_merge, nheads;
    int lse_stride;              // tokens in the PARTIAL lse
    int lse_out_stride;          // tokens in the FINAL lse
};

}  // namespace ring_attn
