#!/usr/bin/env bash
# W4 data preparation on node 0: LongBench v2 prompts (12 per length, 6 domains x 2, seed 0) -> token ids; a layer-major
# prefill of MiniCPM4.1-8B with the official InfLLM-v2 kernels -> each layer's selection and the attention inputs of layers
# 0 / 8 / 16 / 24 for all 36 sequences (local to node 0; rank 0 of the operator tests reads them).
# Inputs : $DEX_MODEL_ROOT/MiniCPM4.1-8B, $DEX_DATA_ROOT/LongBench-v2/data.json (zai-org/LongBench-v2)
# Outputs: $DEX_DATA_ROOT/w4_infllm/{seqs.jsonl,ids/,masks/<seq>/L??_topk.npy,masks/<seq>/L??_xattn.npy}
# usage: workloads/w4_infllm/prepare.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/config/env.sh"
cd "$ROOT"
export PYTHONPATH=$ROOT:$ROOT/third_party/infllmv2_cuda_impl${PYTHONPATH:+:$PYTHONPATH}
D=$DEX_DATA_ROOT/w4_infllm
"$DEX_PYTHON" -m workloads.w4_infllm.prepare_data --out "$D"
pids=()
for ((w = 0; w < DEX_GPUS_PER_NODE; w++)); do
  "$DEX_PYTHON" -m workloads.w4_infllm.sample_masks --data "$D" --out "$D/masks" --worker "$w" --nworkers "$DEX_GPUS_PER_NODE" \
    --device "cuda:$w" --x-layers 0,8,16,24 --x-seqs all &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
