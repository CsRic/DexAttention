#!/usr/bin/env bash
# Table 2: standalone FlashAttention, NCCL per-message time, Dex per-tile time. One GPU, then one node, then two nodes.
# Outputs: $DEX_RESULT_ROOT/microbench/{compute_fa2,comm,dex_fwd}.jsonl ; then python analysis/tables.py table2
# The Dex communication overheads of the table are the scheduler's per-message constants (config/calibration.json).
# Example: bash experiments/microbench/run.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/config/env.sh"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
O=$DEX_RESULT_ROOT/microbench
mkdir -p "$O"
cd "$ROOT"
# the standalone FA2 is FlashAttention 2.8.3, not the headers Dex's kernels compile against
PYTHONPATH="${DEX_FA283_PATH:?set DEX_FA283_PATH (config/env.example.sh)}:$PYTHONPATH" CUDA_VISIBLE_DEVICES=0 "$DEX_PYTHON" experiments/microbench/compute.py --out "$O/compute_fa2.jsonl"
CUDA_VISIBLE_DEVICES=0 DEX_NODES=1 DEX_GPUS_PER_NODE=1 DEX_ENV_PROFILE=static bash tools/launch.sh experiments/microbench/dex_tile.py --out "$O/dex_fwd.jsonl"
# intra-node pair: default NCCL; inter-node: default and GPUDirect RDMA forced
DEX_NODES=1 bash tools/launch.sh experiments/microbench/comm.py --pattern pair --tag nccl_default --out "$O/comm.jsonl"
DEX_NODES=2 bash tools/launch.sh experiments/microbench/comm.py --pattern inter --tag nccl_default --out "$O/comm.jsonl"
DEX_NODES=2 NCCL_NET_GDR_LEVEL=SYS NCCL_NET_GDR_READ=1 bash tools/launch.sh experiments/microbench/comm.py --pattern inter --tag nccl_gdr --out "$O/comm.jsonl"
