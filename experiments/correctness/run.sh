#!/usr/bin/env bash
# Numerical agreement: launch 1 of every operator test carries the FP64 sampled-row check; training losses and the W6 greedy
# token come from the end-to-end records. This script only summarizes them (no GPU); W6 prefill's comparison with the saved
# logits and the single-device reference is experiments/end2end/llama_summarize.py.
# Outputs: $DEX_RESULT_ROOT/analysis/correctness.{tex,json}
# Example: bash experiments/correctness/run.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/config/env.sh"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT/analysis"
"$DEX_PYTHON" tables.py correctness
cd "$ROOT"
"$DEX_PYTHON" experiments/end2end/llama_summarize.py
