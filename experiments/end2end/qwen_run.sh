#!/usr/bin/env bash
# W2 / W3 training steps (16 GPUs): per length bin the 8 selected inputs in selection order, 4 distinct inputs per step
# (2 steps), 1 warm-up step on the first group, every step timed once; one launch per (workload, bin, system).
# Inputs : masks and Dex plans under $DEX_DATA_ROOT, weights under $DEX_MODEL_ROOT
# Outputs: $DEX_RESULT_ROOT/end2end/<w>/<tag>.jsonl (one line per timed step; a launch that fails leaves one line with
#          status "failed" and the reason found in its log), <tag>_<bin>_<system>.log
# Example: bash experiments/end2end/qwen_run.sh w3 "dex ring allgather" "32k 64k 128k" launch1
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/config/env.sh"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
WORKLOAD=${1:?usage: qwen_run.sh <w2|w3> [systems] [bins] [tag]}
SYSTEMS=${2:-dex ring allgather dcp ua magi}
BINS=${3:-32k 64k 128k}
TAG=${4:-launch1}
OUT="$DEX_RESULT_ROOT/end2end/$WORKLOAD/$TAG.jsonl"
export DEX_ENV_PROFILE=training   # the driver sizes Dex's heap from its regions
export DEX_PRELOAD_NVSHMEM=0      # as the W1-W3 operator tests, in the same Python environment
cd "$ROOT"
for bin in $BINS; do
  ids=$("$DEX_PYTHON" -c "
import sys; sys.path.insert(0, 'experiments/attention_layer')
import static_inputs as SI
print(','.join(c['id'] for c in SI.inputs('$WORKLOAD') if c['length_bin'] == '$bin'))")
  for system in $SYSTEMS; do
    echo "[qwen] $WORKLOAD $bin $system: $ids"
    py=$DEX_PYTHON                                   # DCP and Magi train in their own Python environments
    case $system in
      dcp) py=${DEX_DCP_PYTHON:?set DEX_DCP_PYTHON (config/env.example.sh)} ;;
      magi) py=${DEX_MAGI_PYTHON:?set DEX_MAGI_PYTHON (config/env.example.sh)} ;;
    esac
    log="$DEX_RESULT_ROOT/end2end/$WORKLOAD/${TAG}_${bin}_$system.log"
    mkdir -p "$(dirname "$log")"
    rc=0
    DEX_PYTHON=$py bash tools/launch.sh experiments/end2end/qwen_train.py --workload "$WORKLOAD" --system "$system" --inputs "$ids" \
      --accum 4 --warmup 1 --iters 1 --out "$OUT" 2>&1 | tee "$log" || rc=$?
    if [ "$rc" -ne 0 ]; then
      echo "[qwen] $WORKLOAD $bin $system rc=$rc"
      "$DEX_PYTHON" experiments/end2end/failure_record.py --workload "$WORKLOAD" --system "$system" --inputs "$ids" --rc "$rc" \
        --log "$log" --out "$OUT"
    fi
  done
done
