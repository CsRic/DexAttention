"""Builds the Dex CUDA extension `dex.dex_cuda` (torch CUDAExtension, NVSHMEM device-link).

Inputs : DEX_NVSHMEM_HOME (config/env.sh), third_party/{flash-attention,infllmv2_cuda_impl}; optional DEX_INFLLM_DTYPE
         (fp16 | bf16, element type of the InfLLM-v2 routines; bf16 also builds the 2-D InfLLM-v2 executors of csrc/infllm2d)
         and DEX_CUDA_ARCH (default sm_86).
Outputs: build/build_ext.sh defaults to dex/ for FP16 and build/ext_bf16/ for BF16; --out overrides the directory.
The script also generates the headers derived from third_party.
Example: bash build/build_ext.sh"""
import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).parent.resolve()
CSRC = ROOT / "csrc"
FA = ROOT / "third_party" / "flash-attention"
INFLLM_SRC = ROOT / "third_party" / "infllmv2_cuda_impl"
NVSHMEM_HOME = os.environ.get("DEX_NVSHMEM_HOME")
if not NVSHMEM_HOME:
    raise RuntimeError("DEX_NVSHMEM_HOME is not set: copy config/env.example.sh to config/env.sh and source it")

BF16 = os.environ.get("DEX_INFLLM_DTYPE", "fp16") == "bf16"
INFLLM_DT = ["-DINFLLM_BF16", "-DDEX_WITH_INFLLM2D"] if BF16 else []
cxx_flags = ["-O3", "-std=c++17", *INFLLM_DT]
nvcc_flags = [
    "-O3",
    "-std=c++17",
    f"-arch={os.environ.get('DEX_CUDA_ARCH', 'sm_86')}",
    "--use_fast_math",  # as FA2 / FA3: without it every softmax exp2f expands to a range check on the critical path
    "-rdc=true",  # NVSHMEM device symbols
    "-DNVSHMEM_COMPILING_WITH_NVCC=1",
    *INFLLM_DT,
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-Xptxas",
    "-v",
    "-lineinfo",
]

SOURCES = [
    "ring_attn/ring_flash_attn_host.cpp",
    "ring_attn/ring_flash_attn.cu",
    "dist_bwd/bwd_bind.cpp",
    "dist_bwd/sparse_bwd.cu",
    "dist_bwd/dist_bwd.cu",
    "dist_bwd/dist_bwd_host.cpp",
    "infllm_bwd/infllm_bwd.cu",
    "infllm_bwd/infllm_bwd_host.cpp",
    "infllm_fwd/infllm_fwd.cu",
    "infllm_fwd/infllm_fwd_host.cpp",
]
if BF16:
    SOURCES += [
        "infllm2d/infllm2d_fwd.cu",
        "infllm2d/infllm2d_fwd_host.cpp",
        "infllm2d/infllm2d_bwd.cu",
        "infllm2d/infllm2d_bwd_host.cpp",
    ]

setup(
    name="dexattention",
    version="1.0",
    packages=["dex", "dex.planner", "dex.scheduler", "dex.attn", "dex.common"],
    ext_modules=[
        CUDAExtension(
            name="dex.dex_cuda",
            sources=[str(CSRC / s) for s in SOURCES],
            include_dirs=[
                str(FA / "hopper"),  # FA3 headers (mainloop, epilogue, softmax)
                str(CSRC / "ring_attn"),
                str(FA / "csrc" / "cutlass" / "include"),
                str(FA / "csrc"),  # FA2 headers as flash_attn/src/... (backward)
                str(CSRC / "dist_bwd"),
                f"{NVSHMEM_HOME}/include",
                str(INFLLM_SRC),  # official InfLLM-v2 headers (csrc/flash_attn/src/...)
            ],
            library_dirs=[f"{NVSHMEM_HOME}/lib"],
            libraries=["nvshmem_host", "nvshmem_device", "cuda", "ibverbs", "numa"],
            dlink=True,
            dlink_libraries=["nvshmem_device"],
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
