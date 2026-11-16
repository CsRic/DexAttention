#!/usr/bin/env bash
# DCP planning cost (Table 1, Table 4 planning column), 16 GPUs for the per-rank planner: block-size ladder, 120 s per attempt, plan time = maximum over GPUs of the official
# planner's wall time. Also produces DCP's plan caches. Inputs: masks + ranges under $DEX_DATA_ROOT, KaHyPar and redis in DCP's environment.
# Outputs: $DEX_DATA_ROOT/plans/dcp/<w>/ladder.json; summary $DEX_RESULT_ROOT/planning_latency/dcp.json. Example: bash experiments/planning_latency/baselines_dcp.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/config/env.sh"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$DEX_PYTHON" "$ROOT/baselines/dcp/ladder.py" --workloads w1 w2 w3
"$DEX_PYTHON" "$ROOT/experiments/planning_latency/baselines_summary.py" dcp
