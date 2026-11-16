// Launchers for the list-driven sparse backward (sparse_bwd_kernel.h). FA2 headers live in namespace fa2bwd
// (-DFLASH_NAMESPACE=fa2bwd) so this TU links next to the FA3-derived forward.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "sparse_bwd_kernel.h"

using Traits = Flash_bwd_kernel_traits<128, 64, 64, 8, 4, 2, 2, /*Is_V_in_regs=*/true, /*No_double_buffer=*/false,
                                               cutlass::half_t>;

int sparse_bwd_smem_bytes() { return Traits::kSmemSize1colblock; }

void sparse_bwd_launch(const sparse_bwd::Params &p, bool atomic_dkv, cudaStream_t stream) {
    constexpr int smem = Traits::kSmemSize1colblock;
    if (p.n_tasks == 0) return;
    if (atomic_dkv) {
        auto k = &sparse_bwd::sparse_bwd_kernel<Traits, true>;
        static bool set = false;
        if (!set) { cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); set = true; }
        k<<<p.n_tasks, Traits::kNThreads, smem, stream>>>(p);
    } else {
        auto k = &sparse_bwd::sparse_bwd_kernel<Traits, false>;
        static bool set = false;
        if (!set) { cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); set = true; }
        k<<<p.n_tasks, Traits::kNThreads, smem, stream>>>(p);
    }
}

void dot_do_o_launch(const at::Tensor &dout, const at::Tensor &o, at::Tensor &dpsum, int64_t lse_stride,
                     c10::optional<at::Tensor> dq_accum, cudaStream_t stream) {
    const int rows = dout.size(0), h = dout.size(1), d = dout.size(2);
    const int64_t warps = (int64_t)rows * h;
    sparse_bwd::dot_do_o_kernel<<<(warps * 32 + 255) / 256, 256, 0, stream>>>(
        (const __half *)dout.data_ptr(), (const __half *)o.data_ptr(), dpsum.data_ptr<float>(), rows, h, d, lse_stride,
        dq_accum ? dq_accum->data_ptr<float>() : nullptr);
}

void dq_convert_launch(const at::Tensor &acc, at::Tensor &out, float scale, cudaStream_t stream) {
    const int rows = acc.size(0), h = acc.size(1);
    sparse_bwd::dq_convert_kernel<Traits><<<(rows / 64) * h, Traits::kNThreads, 0, stream>>>(
        acc.data_ptr<float>(), (__half *)out.data_ptr(), (int64_t)h * 128, rows / 64, h, scale);
}

void convert_launch(const at::Tensor &acc, at::Tensor &out, float scale, cudaStream_t stream) {
    const int64_t n = acc.numel();
    sparse_bwd::convert_kernel<<<(n / 4 + 255) / 256, 256, 0, stream>>>(acc.data_ptr<float>(), (__half *)out.data_ptr(),
                                                                       n, scale);
}
