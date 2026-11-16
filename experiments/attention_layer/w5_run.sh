#!/usr/bin/env bash
# W5 attention-layer operator tests (Wan2.1-1.3B, SpargeAttention2, exact FP16 forward + backward), 16 GPUs.
# Inputs : the 42 cases of workloads/w5_sparge/cases42.txt (dumps under $DEX_DATA_ROOT/w5_sparge/dumps on node 0).
# Outputs: $DEX_RESULT_ROOT/attention_layer/w5/<system>/rep<r>.jsonl (+ .log); launch 1 carries the FP64 check.
# Protocol: per case 2 warm-ups + 10 timed calls per scope, median; two independent launches per system.
# Environment: profile "training" (config/runtime_defaults.sh), every system; 4 GiB heap. An extension built for this
#          workload's Python environment (requirements/w5.txt) is selected with DEX_EXT_DIR=<DIR>, from
#          DEX_PYTHON=<python> bash build/build_ext.sh --out <DIR>.
# The executor ablation runs the same driver with the system "decomposed".
# usage: experiments/attention_layer/w5_run.sh [systems, default dex,dex_no_assigner,ring,allgather,sparse_allgather] [reps, default 1,2]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/config/env.sh"
SYSS=${1:-dex,dex_no_assigner,ring,allgather,sparse_allgather}
REPS=${2:-1,2}
OUT=$DEX_RESULT_ROOT/attention_layer/w5
export DEX_ENV_PROFILE=training
export DEX_PRELOAD_NVSHMEM=1        # every system loads the extension (its table builders), so its host library resolves first
# Dex's shared forward / backward region takes 85% of this heap; 4 GiB holds the 720p tables
export NVSHMEM_SYMMETRIC_SIZE=4294967296
for r in ${REPS//,/ }; do
  for s in ${SYSS//,/ }; do
    mkdir -p "$OUT/$s"
    F=()
    [ "$r" = 1 ] && F=(--fp64)
    "$ROOT/tools/launch.sh" experiments/attention_layer/w5_op.py --system "$s" --out "$OUT/$s/rep$r.jsonl" "${F[@]}" \
      >> "$OUT/$s/rep$r.log" 2>&1
  done
done
