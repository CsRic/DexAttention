#!/usr/bin/env bash
# Converts every system's plans for the 15 inputs (CPU only). Inputs: UA manifest, DCP ladder + plan caches under $DEX_DATA_ROOT/plans.
# Outputs: $DEX_DATA_ROOT/plans/ablation/{ring,ua,dcp,magi}/, manifest.json. Example: bash experiments/ablation_components/convert_plans.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/config/env.sh"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1
D=experiments/ablation_components
cd "$ROOT"
"$DEX_PYTHON" $D/sample.py
"$DEX_PYTHON" $D/ring_order.py
"$DEX_PYTHON" $D/ua_order.py
"$DEX_MAGI_PYTHON" $D/magi_plan.py
while read -r w n; do
  ex=""; [ "$w" = w1 ] && ex=--perhead
  PYTHONPATH="$ROOT/third_party/dcp:$PYTHONPATH" "$DEX_DCP_PYTHON" $D/dcp_plan.py --workload "$w" --name "$n" $ex
  [ "$w" = w1 ] || "$DEX_PYTHON" $D/dcp_perhead.py --workload "$w" --name "$n"
done < <("$DEX_PYTHON" -c "
import sys; sys.path.insert(0, '$D')
import sample as S
for w, c in S.cases(): print(w, c['id'])")
"$DEX_PYTHON" $D/build_manifest.py
