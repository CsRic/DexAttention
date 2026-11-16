#!/usr/bin/env bash
# Smoke test on one node with 4 GPUs: a 16K W1 Lambda input, Dex, Ring and All-gather, forward + backward, launch 1 protocol
# (warm-ups, 10 timed calls, FP64 sampled-row check). Uses its own data and result directories under $DEX_SCRATCH.
# Inputs : config/env.sh, the built extension (build/build_ext.sh). Outputs: the table it prints; exit status 0 if every check passes.
# Takes about 5 minutes (first-run kernel compilation of the planner included).
# Example: bash tests/smoke_4gpu.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/config/env.sh"
BASE=$DEX_SCRATCH/dex_smoke
rm -rf "$BASE"
export DEX_DATA_ROOT=$BASE/data DEX_RESULT_ROOT=$BASE/results DEX_NODES=1 DEX_ENV_PROFILE=static
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$DEX_DATA_ROOT" "$DEX_RESULT_ROOT"
cd "$ROOT"
CUDA_VISIBLE_DEVICES=0 "$DEX_PYTHON" tests/smoke_driver.py plan
ID=w1_16k_sink128_win4k
for s in dex ring allgather; do
  for ps in fwd bwd; do
    bash tools/launch.sh tests/smoke_driver.py op --system "$s" --workload w1 --pass "$ps" --launch 1 --ids "$ID"
  done
done
"$DEX_PYTHON" tests/smoke_driver.py check
