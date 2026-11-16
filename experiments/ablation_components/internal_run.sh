#!/usr/bin/env bash
# Data-reuse, relay and transfer-order study (16 GPUs, one independent run, head splitting off, zigzag placement): plans for the nine
# static inputs, then per arm and pass one launch per input. Independent of Table 5's run.sh.
# Inputs : masks under $DEX_DATA_ROOT. Outputs: $DEX_DATA_ROOT/plans/ablation/internal/, $DEX_RESULT_ROOT/ablation_components/internal/*.jsonl
# Example: bash experiments/ablation_components/internal_run.sh "full load_only no_relay fifo"; python analysis/tables.py component_internal
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/config/env.sh"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
ARMS=${1:-full load_only no_relay fifo}
D=experiments/ablation_components
cd "$ROOT"
"$DEX_PYTHON" $D/internal_plans.py
mapfile -t ids < <("$DEX_PYTHON" -c "
import sys; sys.path.insert(0, '$D')
import internal_plans as IP
print('\n'.join(c['id'] for c in IP.cases()))")
export DEX_ENV_PROFILE=static
for arm in $ARMS; do
  for pass in fwd bwd; do
    for id in "${ids[@]}"; do
      bash tools/launch.sh $D/internal_run.py --arm "$arm" --pass "$pass" --ids "$id" || echo "[internal] $arm $pass $id rc=$?"
    done
  done
done
