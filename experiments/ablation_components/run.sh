#!/usr/bin/env bash
# Table 5 static row set (16 GPUs): for each launch, Magi native on the padded tile layout (fp64 on launch 1), then rows 2 and 3 of Ring, UA, DCP, Magi.
# Native rows of Ring, UA, DCP and the Dex reference come from experiments/attention_layer; the dynamic Ring row reuses the W4-W6 attention-layer results.
# Inputs : experiments/ablation_components/convert_plans.sh. Outputs: $DEX_RESULT_ROOT/ablation_components/*.jsonl
# Example: bash experiments/ablation_components/run.sh "ring ua dcp magi" "1 2"
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/config/env.sh"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
XS=${1:-ring ua dcp magi}
LAUNCHES=${2:-1 2}
cd "$ROOT"
for k in $LAUNCHES; do
  fp=""; [ "$k" = 1 ] && fp=--fp64
  if [[ " $XS " == *" magi "* ]]; then
    for w in w1 w2 w3; do
      ids=$("$DEX_PYTHON" -c "
import sys; sys.path.insert(0, 'experiments/ablation_components')
import sample as S
print(','.join(S.sample()['$w']))")
      DEX_PYTHON=$DEX_MAGI_PYTHON bash tools/launch.sh baselines/magi/bench.py --workload "$w" --ids "$ids" --padded --launch "$k" $fp \
        || echo "[ablation] magi native $w launch $k rc=$?"
    done
  fi
  for x in $XS; do
    for row in 2 3; do
      DEX_ENV_PROFILE=static bash tools/launch.sh experiments/ablation_components/run_fwd.py --x "$x" --row "$row" --launch "$k" || echo "[ablation] $x row $row launch $k rc=$?"
    done
  done
done
