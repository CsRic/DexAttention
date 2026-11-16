#!/usr/bin/env bash
# Figure 6(a): operator tests of W1-W6, every system, two launches (launch 1 with the correctness checks), 16 GPUs.
# Inputs : prepared data under $DEX_DATA_ROOT (docs/reproduction.md), baseline plans (UA, DCP) from the commands below
# Outputs: $DEX_RESULT_ROOT/attention_layer/<w>/ ; then python analysis/fig_attention.py
# One command runs in one Python environment (DEX_PYTHON, requirements/): start W4, W5, W6 separately as docs/reproduction.md shows.
# Example: bash experiments/attention_layer/run.sh            all
#          bash experiments/attention_layer/run.sh w5 w6      selected workloads
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/config/env.sh"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
WS=${*:-static w4 w5 w6}
DCP_PY=${DEX_DCP_PYTHON:-$DEX_PYTHON}
for w in $WS; do
  case $w in
    static)
      bash experiments/attention_layer/static_prepare.sh
      # all baseline plans offline with the block-size ladder (120 s per attempt), then execution only
      bash experiments/planning_latency/baselines_ua.sh 0 1
      bash experiments/planning_latency/baselines_ua.sh merge
      DEX_PYTHON=$DCP_PY bash experiments/planning_latency/baselines_dcp.sh
      bash experiments/planning_latency/baselines_magi.sh
      for s in dex ring ua dcp magi; do bash experiments/attention_layer/static_run.sh "$s" "w1 w2 w3" "1 2"; done
      ;;
    w4) bash experiments/attention_layer/w4_run.sh ;;
    w5) bash experiments/attention_layer/w5_run.sh ;;
    w6) bash experiments/attention_layer/w6_run.sh ;;
    *) echo "unknown workload $w (static w4 w5 w6)" >&2; exit 2 ;;
  esac
done
