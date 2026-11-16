#!/usr/bin/env bash
# Sample the W2 / W3 rollout pool: Qwen3-4B-Thinking-2507 on LongReason 32k / 64k / 128k, 48 prompts per split,
# 16 replies per prompt, at most 8192 tokens per reply (one node; the 128k split uses tensor parallelism 4).
# Inputs : $DEX_DATA_ROOT/longreason, $DEX_MODEL_ROOT/Qwen3-4B-Thinking-2507 (needs vLLM in $DEX_PYTHON)
# Outputs: $DEX_DATA_ROOT/rollouts/R_<split>_max8192_qwen3-4b-thinking.json
# Example: bash workloads/w2_w3_rollout/sample_rollouts.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/config/env.sh"
cd "$ROOT"
for split in 32k 64k 128k; do
  tp=2; [ "$split" = 128k ] && tp=4
  "$DEX_PYTHON" -m workloads.w2_w3_rollout.sample_rollouts --split "$split" --tp "$tp"
done
