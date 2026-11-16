#pragma once
// The official InfLLM-v2 (infllmv2_cuda_impl) headers, with their `flash` namespace and global struct names renamed so
// they cannot collide with the FA2 / FA3 headers of this extension's other translation units (same names, different
// definitions). INFLLM root = the fork's checkout (setup.py adds it to the include path; csrc/flash_attn/src inside it).
#define flash infllm_flash
#define Flash_kernel_traits Infllm_kernel_traits
#define Flash_fwd_kernel_traits Infllm_fwd_kernel_traits
#define Flash_bwd_kernel_traits Infllm_bwd_kernel_traits
#define Qkv_params Infllm_qkv_params
#define Flash_fwd_params Infllm_fwd_params
#define Flash_bwd_params Infllm_bwd_params
#include "csrc/flash_attn/src/flash.h"
#include <cutlass/numeric_types.h>
#ifdef __CUDACC__
#include "csrc/flash_attn/src/kernel_traits.h"
#ifdef INFLLM_WANT_BWD
#include "csrc/flash_attn/src/flash_bwd_preprocess_kernel.h"   // compute_dot_do_o, convert_dQ (dot_do_o used by the kernel)
#include "csrc/flash_attn/src/flash_bwd_kernel.h"              // bwdIterator, the copy helpers; its own routine is unused
#include "infllm_1colblock.h"                                  // the official column routine, accumulate epilogue
#endif
#ifdef INFLLM_WANT_FWD
#include "csrc/flash_attn/src/flash_fwd_kernel.h"              // fwdIterator, the helpers; its own routine is unused
#include "../infllm_fwd/infllm_1rowblock.h"                    // the official row routine, warp-sliced shared memory
#endif
#endif
#undef flash

#ifdef INFLLM_BF16
using InfllmElem = cutlass::bfloat16_t;
#define INFLLM_TORCH_DTYPE torch::kBFloat16
#define INFLLM_IS_BF16 true
#else
using InfllmElem = cutlass::half_t;                  // fp16 build: InfLLM-v2 routines in FP16, not BF16 (as every system)
#define INFLLM_TORCH_DTYPE torch::kHalf
#define INFLLM_IS_BF16 false
#endif

namespace infllm_bwd {
// the official varlen backward parameter set + what the accumulate epilogue needs
struct FParams : Infllm_bwd_params { int64_t acc_row_stride; };
}  // namespace infllm_bwd
