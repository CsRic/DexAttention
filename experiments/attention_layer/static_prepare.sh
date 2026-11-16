#!/usr/bin/env bash
# W1-W3 inputs and Dex static plans (one GPU): W1 masks, W2 / W3 group selection and masks (from the token counts
# in workloads/w2_w3_rollout/lengths, which sample_rollouts.sh regenerates), head-split plans, then the static planning time.
# Outputs: $DEX_DATA_ROOT/{masks,rollouts,plans/static}, $DEX_RESULT_ROOT/planning_latency/static_plan_time_hs4.json
# Example: bash experiments/attention_layer/static_prepare.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/config/env.sh"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
"$DEX_PYTHON" -m workloads.w1_lambda.gen_masks
"$DEX_PYTHON" -m workloads.w2_w3_rollout.select_groups
"$DEX_PYTHON" -m workloads.w2_w3_rollout.gen_masks
"$DEX_PYTHON" experiments/attention_layer/static_plans.py --workloads w1 w2 w3 --head-split 4
"$DEX_PYTHON" experiments/attention_layer/static_plan_time.py --workloads w1 w2 w3 --head-split 4
