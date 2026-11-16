#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>

#include "sparse_bwd_params.h"
void sparse_bwd_launch(const sparse_bwd::Params &p, bool atomic_dkv, cudaStream_t stream);
void dot_do_o_launch(const at::Tensor &dout, const at::Tensor &o, at::Tensor &dpsum, int64_t lse_stride,
                     c10::optional<at::Tensor> dq_accum, cudaStream_t stream);
void convert_launch(const at::Tensor &acc, at::Tensor &out, float scale, cudaStream_t stream);
void dq_convert_launch(const at::Tensor &acc, at::Tensor &out, float scale, cudaStream_t stream);
int sparse_bwd_smem_bytes();

// q, dout [rows_q, h, d] fp16; k, v [rows_k, hk, d] fp16; lse, dpsum [h, lse_stride] fp32; dq_accum [rows_q, h, d] fp32
// task_kv [T] int32 ((kv_slot << 9) | (hn << 8) | g), task_ptr [T+1], items int32 ((q_slot << 9) | (mb << 8) | (hq << 1) | causal)
// direct mode: dk, dv fp16 [rows_k, hk, d]; atomic mode: dk_accum, dv_accum fp32 [rows_k, hk, d]
void sparse_bwd_run(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor dout, at::Tensor lse, at::Tensor dpsum,
                at::Tensor dq_accum, c10::optional<at::Tensor> dk, c10::optional<at::Tensor> dv,
                c10::optional<at::Tensor> dk_accum, c10::optional<at::Tensor> dv_accum, at::Tensor task_kv,
                at::Tensor task_ptr, at::Tensor items, double scale) {
    TORCH_CHECK(q.dtype() == at::kHalf && q.size(2) == 128 && q.is_contiguous() && k.is_contiguous());
    const bool atomic = dk_accum.has_value();
    sparse_bwd::Params p{};
    p.q = q.data_ptr(); p.k = k.data_ptr(); p.v = v.data_ptr(); p.dout = dout.data_ptr();
    p.lse = lse.data_ptr<float>(); p.dpsum = dpsum.data_ptr<float>(); p.dq_accum = dq_accum.data_ptr<float>();
    if (atomic) { p.dk_accum = dk_accum->data_ptr<float>(); p.dv_accum = dv_accum->data_ptr<float>(); }
    else { p.dk = dk->data_ptr(); p.dv = dv->data_ptr(); }
    p.h = q.size(1); p.hk = k.size(1); p.d = 128;
    p.q_row_stride = (int64_t)p.h * p.d; p.k_row_stride = (int64_t)p.hk * p.d;
    p.q_slot_stride = 128 * p.q_row_stride; p.dq_slot_stride = p.q_slot_stride;      // plain [rows, heads, d] tensors
    p.k_slot_stride = 128 * p.k_row_stride; p.dkv_slot_stride = p.k_slot_stride;
    p.lse_slot_stride = 128; p.lse_head_stride = lse.stride(0);
    p.scale_softmax = scale; p.scale_softmax_log2 = scale * M_LOG2E;
    p.task_kv = task_kv.data_ptr<int>(); p.task_ptr = task_ptr.data_ptr<int>(); p.items = items.data_ptr<int>();
    p.n_tasks = task_kv.numel();
    sparse_bwd_launch(p, atomic, at::cuda::getCurrentCUDAStream());
}
void dot_do_o(at::Tensor dout, at::Tensor o, at::Tensor dpsum, c10::optional<at::Tensor> dq_accum) {
    dot_do_o_launch(dout, o, dpsum, dpsum.stride(0), dq_accum, at::cuda::getCurrentCUDAStream());
}
void dq_convert(at::Tensor acc, at::Tensor out, double scale) { dq_convert_launch(acc, out, scale, at::cuda::getCurrentCUDAStream()); }
void convert(at::Tensor acc, at::Tensor out, double scale) { convert_launch(acc, out, scale, at::cuda::getCurrentCUDAStream()); }

void register_dist_bwd(pybind11::module &m);
void register_sparse_bwd(pybind11::module &m) {
    m.def("sparse_bwd", &sparse_bwd_run, "list-driven sparse attention backward (one CTA per KV-block task)");
    m.def("dot_do_o", &dot_do_o, "dpsum = rowsum(dO * O); optionally zero dq_accum");
    m.def("convert", &convert, "fp32 accumulator * scale -> fp16");
    m.def("dq_convert", &dq_convert, "dq_accum (permuted 64x128 blocks, [rows, h, 128]) * scale -> dQ fp16");
    m.def("smem_bytes", &sparse_bwd_smem_bytes);
}
