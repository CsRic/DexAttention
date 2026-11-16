#!/usr/bin/env bash
# W1-W3 attention-call tests (16 GPUs): for each launch (1 = timing + FP64 check, 2 = timing only), workload and chunk of
# inputs, one forward launch and one backward launch of the system. W1: one input per launch; W2 / W3: 8 per launch.
# DCP and Magi run in their own Python environments with their own drivers (baselines/dcp/ladder.py --measure,
# baselines/magi/bench.py); they write the same records. Dex runs with the environment profile "static".
# Inputs : masks and plans under $DEX_DATA_ROOT (workloads/*/gen_masks.py, static_plans.py, experiments/planning_latency)
# Outputs: $DEX_RESULT_ROOT/attention_layer/<w>/<system>_<pass>_launch<k>.jsonl
# Heap: static_op.py sizes NVSHMEM_SYMMETRIC_SIZE from the launch's largest region; the profile does not set it.
# Example: bash experiments/attention_layer/static_run.sh dex "w1 w2 w3" "1 2"
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/config/env.sh"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
SYSTEM=${1:-dex}
WORKLOADS=${2:-w1 w2 w3}
LAUNCHES=${3:-1 2}
DRV=experiments/attention_layer/static_op.py
cd "$ROOT"
case $SYSTEM in
  dcp)
    DEX_PYTHON=${DEX_DCP_PYTHON:?set DEX_DCP_PYTHON (config/env.example.sh)}
    export DEX_PYTHON
    "$DEX_PYTHON" baselines/dcp/ladder.py --workloads $WORKLOADS --measure --launches $LAUNCHES
    exit 0
    ;;
  magi)
    DEX_PYTHON=${DEX_MAGI_PYTHON:?set DEX_MAGI_PYTHON (config/env.example.sh)}
    export DEX_PYTHON
    for launch in $LAUNCHES; do
      fp=(); [ "$launch" = 1 ] && fp=(--fp64)
      for w in $WORKLOADS; do
        bash tools/launch.sh baselines/magi/bench.py --workload "$w" --launch "$launch" "${fp[@]}"
      done
    done
    exit 0
    ;;
  dex) export DEX_ENV_PROFILE=static ;;
esac
ids_of() {   # workload -> input ids, one per line, canonical order
  "$DEX_PYTHON" -c "
import sys; sys.path.insert(0, '$ROOT/experiments/attention_layer')
import static_inputs as SI
print('\n'.join(c['id'] for c in SI.inputs('$1')))"
}
for launch in $LAUNCHES; do
  for w in $WORKLOADS; do
    chunk=8; [ "$w" = w1 ] && chunk=1
    mapfile -t ids < <(ids_of "$w")
    for ((i = 0; i < ${#ids[@]}; i += chunk)); do
      sel=$(IFS=,; echo "${ids[*]:i:chunk}")
      for pass in fwd bwd; do
        echo "[static] launch $launch $w $SYSTEM $pass: $sel"
        bash tools/launch.sh "$DRV" --system "$SYSTEM" --workload "$w" --pass "$pass" --launch "$launch" --ids "$sel" \
          || echo "[static] launch $launch $w $SYSTEM $pass rc=$? ($sel)"
      done
    done
  done
done
