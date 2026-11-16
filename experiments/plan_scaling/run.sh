#!/usr/bin/env bash
# Figure 7: planning cost at 1-32 nodes (4-128 GPUs), weak scaling, measured on one GPU and its host, one process per point.
# Inputs : $DEX_DATA_ROOT/plan_scaling/masks (gen_family_masks.py, built once by this script from rollout_groups.jsonl,
#          w4_masks.txt and the W4 / W5 / W6 selections and dumps)
# Outputs: $DEX_RESULT_ROOT/plan_scaling/worker<w>.jsonl ; then python analysis/fig_plan_scaling.py
# Points are dealt round-robin to the workers; run one worker per node so the CPU-timed scheduler stages do not contend.
# Example: bash experiments/plan_scaling/run.sh 0 4     (worker 0 of 4; run the others on the other nodes)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/config/env.sh"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
WORKER=${1:-0}
NWORKERS=${2:-1}
O=$DEX_RESULT_ROOT/plan_scaling
mkdir -p "$O"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
[ -f "$DEX_DATA_ROOT/plan_scaling/masks/masks_w6.npz" ] || "$DEX_PYTHON" experiments/plan_scaling/gen_family_masks.py
i=0
for P in 4 8 16 32 64 128; do
  [ $((i % NWORKERS)) = "$WORKER" ] && "$DEX_PYTHON" experiments/plan_scaling/plan_scaling.py --wl w1 --P "$P" --hv 4 --out "$O/worker$WORKER.jsonl"
  i=$((i + 1))
done
for seed in 0 1 2; do
  for P in 4 8 16 32 64 128; do
    for spec in "w4 --perhead" "w5" "w6" "w2 --hv 4" "w3 --hv 4"; do
      set -- $spec
      wl=$1; shift
      [ $((i % NWORKERS)) = "$WORKER" ] && "$DEX_PYTHON" experiments/plan_scaling/plan_scaling.py --wl "$wl" --P "$P" --seed "$seed" "$@" --out "$O/worker$WORKER.jsonl"
      i=$((i + 1))
    done
  done
done
