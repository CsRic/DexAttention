#!/usr/bin/env bash
# Figure 6(b) and Table 7: complete training steps of W2-W5 and whole-model prefill of W6, every system, 16 GPUs.
# Inputs : prepared data and checkpoints (docs/reproduction.md); W2 / W3 baseline plans from experiments/attention_layer
# Outputs: $DEX_RESULT_ROOT/end2end/{w2,w3,minicpm,wan,llama}/ ; then python analysis/fig_end2end.py
# One command runs in one Python environment (DEX_PYTHON, requirements/): start W4, W5, W6 separately as docs/reproduction.md shows.
# Example: bash experiments/end2end/run.sh            all       bash experiments/end2end/run.sh w5    one workload
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/config/env.sh"
cd "$ROOT"
WS=${*:-w2 w3 w4 w5 w6}
for w in $WS; do
  case $w in
    w2|w3) bash experiments/end2end/qwen_run.sh "$w" "dex ring allgather dcp ua magi" "32k 64k 128k" launch1 ;;
    w4) bash experiments/end2end/minicpm_run.sh ;;
    w5) bash experiments/end2end/wan_run.sh ;;
    w6) bash experiments/end2end/llama_run.sh ;;
    *) echo "unknown workload $w (w2 w3 w4 w5 w6)" >&2; exit 2 ;;
  esac
done
