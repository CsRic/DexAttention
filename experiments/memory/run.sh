#!/usr/bin/env bash
# Table 7: peak memory of the training steps (W2-W5) and W6 prefill. No extra launch: every end-to-end driver records
# peak_actual_GiB (W6: mem_max_over_ranks) in its step or input records; run experiments/end2end/run.sh first.
# Outputs: $DEX_RESULT_ROOT/analysis/table7.{tex,json}
# Magi's allocator measurement is separate: experiments/memory/magi_cache.sh
# Example: bash experiments/memory/run.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/config/env.sh"
cd "$ROOT/analysis"
DEX_RESULT_ROOT=$DEX_RESULT_ROOT "$DEX_PYTHON" tables.py table7
