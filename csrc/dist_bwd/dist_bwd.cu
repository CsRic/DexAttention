// Launcher of the distributed sparse backward megakernel (dist_bwd_kernel.h). Part of the main extension (NVSHMEM).
#include <cuda_runtime.h>
#include "dist_bwd_kernel.h"

using BwdTraits = Flash_bwd_kernel_traits<128, 64, 64, 8, 4, 2, 2, /*Is_V_in_regs=*/true, /*No_double_buffer=*/false,
                                          cutlass::half_t>;

int dist_bwd_smem_bytes() { return BwdTraits::kSmemSize1colblock + dist_bwd::kCtlBytes; }

// grid: one CTA per SM (80 KB smem -> 1 resident CTA per SM on sm86), block 0 = scheduler, so every CTA is resident
// (the kernel spins on queue entries / signals; a CTA that is not resident would never run)
void dist_bwd_launch(const dist_bwd::Params &p, int grid, cudaStream_t stream) {
    constexpr int smem = BwdTraits::kSmemSize1colblock + dist_bwd::kCtlBytes;   // + worker control words (Ctl)
    auto k = &dist_bwd::dist_bwd_kernel<BwdTraits>;
    static bool set = false;
    if (!set) { cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); set = true; }
    k<<<grid, BwdTraits::kNThreads, smem, stream>>>(p);
}
