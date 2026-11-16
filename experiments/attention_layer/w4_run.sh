#!/usr/bin/env bash
# W4 attention-layer operator tests (MiniCPM4.1-8B, InfLLM-v2, BF16 forward + backward), 16 GPUs.
# Inputs : the 144 cases of workloads/w4_infllm/cases144.txt (36 sequences x layers 0 / 8 / 16 / 24), recorded selections and
#          attention inputs under $DEX_DATA_ROOT/w4_infllm/masks on node 0, $DEX_MODEL_ROOT/MiniCPM4.1-8B on node 0.
# Outputs: $DEX_RESULT_ROOT/attention_layer/w4/<system>/rep<r>.jsonl (+ .log).
# Protocol: per case 2 warm-ups + 10 timed calls per part (exec, bwd, online), median; two independent launches per system and
#          length. Launch 1 runs layers 0 / 16 with the correctness checks in a launch of their own, then layers 8 / 24.
#          The single-GPU reference of the online forward does not fit one GPU's memory at 128K and runs at 32K / 64K only.
# Python environment: DEX_PYTHON from requirements/w6.txt, which serves W4 and W6.
# Environment: profile "operator" (config/runtime_defaults.sh); CUDA_DEVICE_MAX_CONNECTIONS=32 for every system; 2 GiB heap;
#          DEX_EXT_DIR selects the BF16 build (default build/ext_bf16, from build/build_ext.sh --infllm-dtype bf16).
# The executor ablation runs the same driver with the system "decomposed".
# usage: experiments/attention_layer/w4_run.sh [systems, default dex,dex_no_assigner,ring,allgather,sparse_allgather] [reps, default 1,2] [lengths, default 32768,65536,131072]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/config/env.sh"
SYSS=${1:-dex,dex_no_assigner,ring,allgather,sparse_allgather}
REPS=${2:-1,2}
BINS=${3:-32768,65536,131072}
CASES=$ROOT/workloads/w4_infllm/cases144.txt
OUT=$DEX_RESULT_ROOT/attention_layer/w4
export PYTHONPATH=$ROOT/third_party/infllmv2_cuda_impl${PYTHONPATH:+:$PYTHONPATH}
export DEX_ENV_PROFILE=operator
export CUDA_DEVICE_MAX_CONNECTIONS=32
export DEX_EXT_DIR=${DEX_EXT_DIR:-$ROOT/build/ext_bf16}
# Dex's shared forward / backward region takes 85% of this heap
export NVSHMEM_SYMMETRIC_SIZE=2147483648
sel() { grep "_$1:" "$CASES" | grep -E ":($2)\$" | paste -sd,; }
for r in ${REPS//,/ }; do
  for s in ${SYSS//,/ }; do
    mkdir -p "$OUT/$s"
    O=$OUT/$s/rep$r.jsonl
    for S in ${BINS//,/ }; do
      if [ "$r" = 1 ]; then
        F=(--check --fp64)
        [ "$S" != 131072 ] && F+=(--online-ref)
        "$ROOT/tools/launch.sh" experiments/attention_layer/w4_op.py --system "$s" --cases "$(sel "$S" "0|16")" --out "$O" "${F[@]}" >> "$O.log" 2>&1
        "$ROOT/tools/launch.sh" experiments/attention_layer/w4_op.py --system "$s" --cases "$(sel "$S" "8|24")" --out "$O" >> "$O.log" 2>&1
      else
        "$ROOT/tools/launch.sh" experiments/attention_layer/w4_op.py --system "$s" --cases "$(sel "$S" "0|8|16|24")" --out "$O" >> "$O.log" 2>&1
      fi
    done
  done
done
