#!/usr/bin/env bash
# Magi planning cost (Table 1, Table 4 planning column), 16 GPUs: time to build magi_attn_flex_key (dispatch + solver + comm metadata), slowest rank, per input.
# Outputs: $DEX_RESULT_ROOT/planning_latency/magi_<w>.jsonl; summary magi.json. Example: bash experiments/planning_latency/baselines_magi.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/config/env.sh"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
for w in w1 w2 w3; do
  DEX_PYTHON=$DEX_MAGI_PYTHON bash tools/launch.sh baselines/magi/bench.py --workload "$w" --plan_only
done
"$DEX_PYTHON" "$ROOT/experiments/planning_latency/baselines_summary.py" magi
