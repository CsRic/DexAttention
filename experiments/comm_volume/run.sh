#!/usr/bin/env bash
# Table 4 imbalance / NIC / direct-fetch inputs: offline traffic and balance statistics of every system's plans.
# Inputs : masks, plans and selections under $DEX_DATA_ROOT (see common.py), Ring / W6 operator-test records under
#          $DEX_RESULT_ROOT/attention_layer (run experiments/attention_layer first).
# Outputs: $DEX_RESULT_ROOT/comm_volume/{recv,balance,direct_fetch}/ ; then python analysis/tables.py table4
# CPU only except the two W5 steps, which need one idle GPU on the node that holds the W5 dumps.
# DEX_PYTHON from requirements/w6.txt: the W4 scripts import InfLLM-v2's extension (build/build_ext.sh --infllm-dtype bf16).
# Example: bash experiments/comm_volume/run.sh            (all steps)   bash experiments/comm_volume/run.sh cpu   (no GPU steps)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/config/env.sh"
HERE="$REPO/experiments/comm_volume"
PY="$DEX_PYTHON"
DCP_PY="${DEX_DCP_PYTHON:-$DEX_PYTHON}"     # Python environment with the DCP package (third_party/dcp)
MAGI_PY="${DEX_MAGI_PYTHON:-$DEX_PYTHON}"   # Python environment with magi_attention
MODE="${1:-all}"
cd "$REPO"
export PYTHONPATH="$REPO:$REPO/third_party/infllmv2_cuda_impl${PYTHONPATH:+:$PYTHONPATH}"

cpu() { CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" "$@"; }

# received bytes per GPU (forward / backward, intra / inter node)
for W in W1 W2 W3; do cpu "$PY" "$HERE/dex_static_comm.py" --workload "$W"; done
cpu "$PY" "$HERE/ring_static_comm.py"
for W in W1 W2 W3; do cpu env PYTHONPATH="$REPO/third_party/dcp:$PYTHONPATH" "$DCP_PY" "$HERE/dcp_comm.py" --workload "$W"; done
cpu "$PY" "$HERE/ua_comm.py"
cpu "$MAGI_PY" "$HERE/magi_comm.py"
cpu "$PY" "$HERE/w4_comm.py"
cpu "$PY" "$HERE/w5_comm.py" --analytic

# compute balance
cpu env PYTHONPATH="$REPO/third_party/dcp:$PYTHONPATH" "$DCP_PY" "$HERE/balance_dcp_units.py"
cpu "$MAGI_PY" "$HERE/balance_magi_hosts.py"
cpu "$PY" "$HERE/balance_w123.py" W1 W2 W3
cpu "$PY" "$HERE/balance_dynamic.py" w4 w5 w6

# direct-fetch model
cpu "$PY" "$HERE/direct_fetch.py" w123 w4 w6

if [ "$MODE" = all ]; then
  CUDA_VISIBLE_DEVICES="${DEX_COMM_GPU:-0}" "$PY" "$HERE/w5_comm.py" --gpu
  CUDA_VISIBLE_DEVICES="${DEX_COMM_GPU:-0}" "$PY" "$HERE/direct_fetch.py" w5
fi
echo "[comm_volume] done -> $DEX_RESULT_ROOT/comm_volume"
