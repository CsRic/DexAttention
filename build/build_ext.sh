#!/bin/bash
# Builds everything compiled in Dex: applies build/patches/{flash-attention,infllmv2_cutlass}.patch, generates the headers
# derived from the InfLLM-v2 kernels, builds the extension (module dex.dex_cuda) and the scheduler libraries. The bf16 build
# also builds InfLLM-v2's own extension (infllm_v2.C in third_party/infllmv2_cuda_impl), which W4's selection and baselines call.
#   build                       Python environment   used by
#   --infllm-dtype fp16         torch of DEX_PYTHON  W1-W3, W5, W6: forward / backward tile kernels
#   --infllm-dtype bf16         torch of DEX_PYTHON  W4: same tile kernels + BF16 InfLLM-v2 routines + the 2-D executors
# The two differ only in the InfLLM-v2 translation units. An extension links against one torch, so a workload with its own
# Python environment gets its own build: set DEX_PYTHON and DEX_CUDA_HOME for that environment and select the result at
# run time with DEX_EXT_DIR=<DIR>.
# Inputs : config/env.sh (DEX_PYTHON, DEX_CUDA_HOME, DEX_NVSHMEM_HOME), third_party/{flash-attention,infllmv2_cuda_impl}.
# Options: --infllm-dtype fp16|bf16 (default fp16); --out DIR (default: dex/ for fp16, build/ext_bf16 for bf16);
#          --infllm-cuda-home DIR (toolkit for infllm_v2.C, default DEX_CUDA_HOME; the evaluation used nvcc 12.6, requirements/w6.txt).
# Outputs: dex_cuda*.so in the output directory, dex/scheduler/lib/*.so, the generated headers under csrc/.
# Example: bash build/build_ext.sh; DEX_PYTHON=/path/to/env2/bin/python bash build/build_ext.sh --infllm-dtype bf16
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PRESET_PYTHON=${DEX_PYTHON:-}
PRESET_CUDA=${DEX_CUDA_HOME:-}
source "$ROOT/config/env.sh"
DEX_PYTHON=${PRESET_PYTHON:-$DEX_PYTHON}
DEX_CUDA_HOME=${PRESET_CUDA:-$DEX_CUDA_HOME}
DTYPE=fp16
OUT=
INFLLM_CUDA=
while [ $# -gt 0 ]; do
  case "$1" in
    --infllm-dtype) DTYPE=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --infllm-cuda-home) INFLLM_CUDA=$2; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
case "$DTYPE" in
  fp16) ;;
  bf16) OUT=${OUT:-$ROOT/build/ext_bf16} ;;
  *) echo "--infllm-dtype must be fp16 or bf16" >&2; exit 2 ;;
esac
TP=$ROOT/third_party
for d in flash-attention infllmv2_cuda_impl; do
  [ -e "$TP/$d/csrc/cutlass/include" ] || { echo "missing $TP/$d: run bash third_party/setup_submodules.sh $d" >&2; exit 1; }
done

# 1. third-party changes (a no-op if already applied)
apply_change() {   # repository, change file
  if git -C "$1" apply --check "$ROOT/build/patches/$2" 2>/dev/null; then git -C "$1" apply "$ROOT/build/patches/$2"; fi
}
apply_change "$TP/flash-attention" flash-attention.patch
apply_change "$TP/infllmv2_cuda_impl/csrc/cutlass" infllmv2_cutlass.patch

# 2. headers derived from the official kernels
"$DEX_PYTHON" "$ROOT/csrc/infllm_fwd/gen_1rowblock.py" "$TP/infllmv2_cuda_impl/csrc/flash_attn/src/flash_fwd_kernel.h"
"$DEX_PYTHON" "$ROOT/csrc/infllm_bwd/gen_1colblock.py" "$TP/infllmv2_cuda_impl/csrc/flash_attn/src/flash_bwd_kernel.h"

# 3. the extension. torch builds without a CUDA toolkit of their own ship the math-library headers as pip packages.
export CUDA_HOME=$DEX_CUDA_HOME
export PATH=$DEX_CUDA_HOME/bin:$(dirname "$DEX_PYTHON"):$PATH
export DEX_NVSHMEM_HOME DEX_INFLLM_DTYPE=$DTYPE
SITE=$("$DEX_PYTHON" -c "import site; print(site.getsitepackages()[0])")
for x in cusparse cusolver curand cufft nccl cudnn cublas; do
  if [ -d "$SITE/nvidia/$x/include" ]; then export CPATH="$SITE/nvidia/$x/include:${CPATH:-}"; fi
done
TAG=$("$DEX_PYTHON" -c "import sys, torch; print(f'py{sys.version_info[0]}{sys.version_info[1]}_torch{torch.__version__}_$DTYPE')")
cd "$ROOT"
BT=${DEX_SCRATCH:-$ROOT/build}/dex_build/$TAG
if [ -n "$OUT" ]; then
  mkdir -p "$OUT"
  "$DEX_PYTHON" setup.py build_ext --build-temp "$BT/temp" --build-lib "$BT/lib"
  cp "$BT"/lib/dex/dex_cuda*.so "$OUT"/
  echo "extension in $OUT: run with DEX_EXT_DIR=$OUT"
else
  "$DEX_PYTHON" setup.py build_ext --inplace --build-temp "$BT/temp" --build-lib "$BT/lib"
fi

# 4. InfLLM-v2's own extension: W4 only, compiled against its own CUTLASS
if [ "$DTYPE" = bf16 ]; then
  IC=${INFLLM_CUDA:-$DEX_CUDA_HOME}
  (cd "$TP/infllmv2_cuda_impl" && CUDA_HOME=$IC PATH=$IC/bin:$PATH "$DEX_PYTHON" setup.py build_ext --inplace)
fi

# 5. the scheduler libraries
bash "$ROOT/dex/scheduler/build.sh"
