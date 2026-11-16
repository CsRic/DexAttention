#!/usr/bin/env bash
# W4 end-to-end training steps (MiniCPM4.1-8B, InfLLM-v2, BF16), 16 GPUs: per length one launch per system with its 12
# sequences (sorted) = one warm-up step on the first group + 3 timed steps of 4 different sequences.
# Inputs : $DEX_DATA_ROOT/w4_infllm/ids/<seq>_<length>.npy (workloads/w4_infllm/prepare_data.py), $DEX_MODEL_ROOT/MiniCPM4.1-8B
# Outputs: $DEX_RESULT_ROOT/end2end/minicpm/steps.jsonl (one record per timed step) + one log per (length, system)
# Environment: DEX_PYTHON from requirements/w6.txt, profile "training", 2 GiB heap, DEX_EXT_DIR default build/ext_bf16.
# usage: experiments/end2end/minicpm_run.sh [systems, default dex,dex_no_assigner,ring,allgather,sparse_allgather] [lengths, default 32768,65536,131072]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/config/env.sh"
SYSS=${1:-dex,dex_no_assigner,ring,allgather,sparse_allgather}
BINS=${2:-32768,65536,131072}
OUT=$DEX_RESULT_ROOT/end2end/minicpm
IDS=$DEX_DATA_ROOT/w4_infllm/ids
mkdir -p "$OUT"
export PYTHONPATH=$ROOT/third_party/infllmv2_cuda_impl${PYTHONPATH:+:$PYTHONPATH}
export DEX_ENV_PROFILE=training
# the BF16 extension (build/build_ext.sh --infllm-dtype bf16)
export DEX_EXT_DIR=${DEX_EXT_DIR:-$ROOT/build/ext_bf16}
# Dex's shared forward / backward region takes 85% of this heap
export NVSHMEM_SYMMETRIC_SIZE=2147483648
for b in ${BINS//,/ }; do
  SEQS=$(ls "$IDS" | grep "_$b.npy\$" | sed 's/\.npy$//' | sort | head -12 | paste -sd,)
  for s in ${SYSS//,/ }; do
    "$ROOT/tools/launch.sh" experiments/end2end/minicpm_train.py --system "$s" --seqs "$SEQS" --accum 4 --warmup 1 --iters 1 \
      --out "$OUT/steps.jsonl" > "$OUT/${b}_$s.log" 2>&1
  done
done
