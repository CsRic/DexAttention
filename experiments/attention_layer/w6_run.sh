#!/usr/bin/env bash
# W6 attention-layer operator tests (Llama-3.1-8B + FlexPrefill, forward only), 16 GPUs.
# Inputs : the 36 cases of workloads/w6_flexprefill/cases36.txt (dumps under $DEX_DATA_ROOT/w6_flexprefill/qkv on node 0).
# Outputs: $DEX_RESULT_ROOT/attention_layer/w6/<system>/rep<r>.jsonl (+ .log); launch 1 carries the FP64 check.
# Protocol: per case 3 warm-ups + 10 timed calls, median; two independent launches per system.
# Environment: profile "training" (config/runtime_defaults.sh), which also sets CUDA_DEVICE_MAX_CONNECTIONS=32, every system;
#          4 GiB heap. An extension built for this workload's Python environment (requirements/w6.txt) is selected with
#          DEX_EXT_DIR=<DIR>, from DEX_PYTHON=<python> bash build/build_ext.sh --out <DIR>.
# The executor ablation runs the same driver with the system "decomposed".
# usage: experiments/attention_layer/w6_run.sh [systems, default dex,dex_no_assigner,allgather,ring,sparse_allgather] [reps, default 1,2]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/config/env.sh"
SYSS=${1:-dex,dex_no_assigner,allgather,ring,sparse_allgather}
REPS=${2:-1,2}
OUT=$DEX_RESULT_ROOT/attention_layer/w6
export DEX_ENV_PROFILE=training
export DEX_PRELOAD_NVSHMEM=1        # every system loads the extension (its table builders), so its host library resolves first
export PYTHONPATH=$ROOT/third_party/flex_prefill${PYTHONPATH:+:$PYTHONPATH}
# Dex's region takes 85% of this heap; 4 GiB holds the 128K tables
export NVSHMEM_SYMMETRIC_SIZE=4294967296
for r in ${REPS//,/ }; do
  for s in ${SYSS//,/ }; do
    mkdir -p "$OUT/$s"
    F=()
    [ "$r" = 1 ] && F=(--fp64)
    "$ROOT/tools/launch.sh" experiments/attention_layer/w6_op.py --system "$s" --out "$OUT/$s/rep$r.jsonl" "${F[@]}" \
      >> "$OUT/$s/rep$r.log" 2>&1
  done
done
