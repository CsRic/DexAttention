#pragma once
// Plain-data parameters of the list-driven sparse backward (shared by the kernel TU and the host bindings).
#include <cstdint>
namespace sparse_bwd {

struct Params {
    void *q, *k, *v, *dout;              // fp16
    float *lse, *dpsum, *dq_accum;
    void *dk, *dv;                       // fp16 (direct mode)
    float *dk_accum, *dv_accum;          // fp32 (atomic mode)
    int64_t q_row_stride, k_row_stride;  // elements: h*d, hk*d
    int64_t q_slot_stride, k_slot_stride, dq_slot_stride, dkv_slot_stride;   // elements per slot
    int64_t lse_slot_stride, lse_head_stride;                                // floats
    int h, hk, d;
    float scale_softmax, scale_softmax_log2;
    const int *task_kv;                  // [n_tasks]: (kv_slot << 9) | (hn << 8) | g
    const int *task_ptr;                 // [n_tasks + 1] into items
    const int *items;
    int n_tasks;
};

}  // namespace sparse_bwd
