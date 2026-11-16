#!/usr/bin/env bash
# UA planning cost (Table 1, Table 4 planning column), CPU + Gurobi, one worker per node: block-size ladder, 120 s per attempt; planning time per input and pass
# = deployment ILPs + fusion / schedule, recorded in $DEX_DATA_ROOT/plans/ua/{ladder_<w>_<i>.json, manifest.json}; a run of this script is also what produces UA's plans.
# Inputs : masks under $DEX_DATA_ROOT, Gurobi licence (DEX_GUROBI_LICENSE). Output summary: $DEX_RESULT_ROOT/planning_latency/ua.json
# Example: bash experiments/planning_latency/baselines_ua.sh [worker] [nworkers]   (run the worker of each node, then with "merge")
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/config/env.sh"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [ "${1:-}" = merge ]; then
  "$DEX_PYTHON" "$ROOT/baselines/ua/planner.py" --merge
else
  "$DEX_PYTHON" "$ROOT/baselines/ua/planner.py" --worker "${1:-0}" --nworkers "${2:-1}" --workloads w1 w2 w3
fi
"$DEX_PYTHON" "$ROOT/experiments/planning_latency/baselines_summary.py" ua
