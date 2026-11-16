#!/usr/bin/env bash
# MagiAttention's allocator measurement: reserved versus allocated memory of calc_attn alone, then W2 / W3 training at 32K and 64K
# without clearing PyTorch's allocator cache. A length whose uncleared run runs out of memory is reported and left unpaired.
# Inputs : masks and weights under $DEX_DATA_ROOT and $DEX_MODEL_ROOT; the cleared runs of experiments/end2end/run.sh (tag launch1)
# Outputs: $DEX_RESULT_ROOT/memory/magi_alloc.jsonl, $DEX_RESULT_ROOT/end2end/<w>/noclear.jsonl; prints the summary
# Example: bash experiments/memory/magi_cache.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/config/env.sh"
cd "$ROOT"
: "${DEX_MAGI_PYTHON:?set DEX_MAGI_PYTHON (config/env.example.sh)}"
mkdir -p "$DEX_RESULT_ROOT/memory"
DEX_PYTHON=$DEX_MAGI_PYTHON bash tools/launch.sh experiments/memory/magi_alloc.py --out "$DEX_RESULT_ROOT/memory/magi_alloc.jsonl"
for w in w2 w3; do
  DEX_MAGI_CLEAR_CACHE=0 bash experiments/end2end/qwen_run.sh "$w" magi "32k 64k" noclear
done
DEX_RESULT_ROOT=$DEX_RESULT_ROOT "$DEX_PYTHON" experiments/memory/magi_cache_summary.py
