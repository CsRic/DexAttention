#!/usr/bin/env bash
# W6 end-to-end prefill (Llama-3.1-8B-Instruct + FlexPrefill), 16 GPUs: 9 inputs (workloads/w6_flexprefill/inputs9.txt) per system,
# two launches. Per input: correctness pass + 2 warm-ups + 4 timed prefills, geometric mean. Launch 1 also saves hidden samples.
# Inputs : $DEX_DATA_ROOT/w6_flexprefill/inputs (node 0), $DEX_MODEL_ROOT/Llama-3.1-8B-Instruct
# Outputs: $DEX_RESULT_ROOT/end2end/llama/<system>/rep<r>.{jsonl,log}, rep<r>/ (logits, hidden samples)
# Environment: profile "training", 2 GiB heap; an extension built for this workload's Python environment (requirements/w6.txt)
#          is selected with DEX_EXT_DIR=<DIR>, from DEX_PYTHON=<python> bash build/build_ext.sh --out <DIR>.
# usage: experiments/end2end/llama_run.sh [systems, default dex,dex_no_assigner,allgather,ring,sparse_allgather] [reps, default 1,2]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/config/env.sh"
SYSS=${1:-dex,dex_no_assigner,allgather,ring,sparse_allgather}
REPS=${2:-1,2}
OUT=$DEX_RESULT_ROOT/end2end/llama
export DEX_ENV_PROFILE=training
export DEX_PRELOAD_NVSHMEM=1        # every system loads the extension (its table builders), so its host library resolves first
export PYTHONPATH=$ROOT/third_party/flex_prefill${PYTHONPATH:+:$PYTHONPATH}
# forward-only tables fit a smaller heap, leaving the memory to the model
export NVSHMEM_SYMMETRIC_SIZE=2147483648
for r in ${REPS//,/ }; do
  for s in ${SYSS//,/ }; do
    mkdir -p "$OUT/$s"
    F=()
    [ "$r" = 1 ] && F=(--save_hidden)
    "$ROOT/tools/launch.sh" experiments/end2end/llama_prefill.py --system "$s" --out "$OUT/$s/rep$r.jsonl" \
      --save_dir "$OUT/$s/rep$r" "${F[@]}" >> "$OUT/$s/rep$r.log" 2>&1
  done
done
