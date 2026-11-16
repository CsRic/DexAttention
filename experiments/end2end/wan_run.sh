#!/usr/bin/env bash
# W5 end-to-end training steps (Wan2.1-T2V-1.3B, SpargeAttention2), 16 GPUs: per resolution one launch per system with
# its 16 inputs = 4 timed steps of 4 different inputs (every group warmed once, untimed).
# Inputs : $DEX_DATA_ROOT/w5_sparge/e2e_inputs/<res>_p<i>_s<seed>.pt (workloads/w5_sparge/prep_e2e_inputs.py),
#          $DEX_MODEL_ROOT/Wan2.1-T2V-1.3B-Diffusers
# Outputs: $DEX_RESULT_ROOT/end2end/wan/steps.jsonl (one record per timed step) + one log per (resolution, system)
# Environment: profile "training", 4 GiB heap; an extension built for this workload's Python environment (requirements/w5.txt)
#          is selected with DEX_EXT_DIR=<DIR>, from DEX_PYTHON=<python> bash build/build_ext.sh --out <DIR>.
# usage: experiments/end2end/wan_run.sh [systems, default dex,dex_no_assigner,ring,allgather,sparse_allgather] [resolutions, default 480p,720p]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/config/env.sh"
SYSS=${1:-dex,dex_no_assigner,ring,allgather,sparse_allgather}
RESS=${2:-480p,720p}
OUT=$DEX_RESULT_ROOT/end2end/wan
INPUTS=$DEX_DATA_ROOT/w5_sparge/e2e_inputs
mkdir -p "$OUT"
export DEX_ENV_PROFILE=training
export DEX_PRELOAD_NVSHMEM=1        # every system loads the extension (its table builders), so its host library resolves first
# Dex's shared forward / backward region takes 85% of this heap
export NVSHMEM_SYMMETRIC_SIZE=4294967296
# per-layer recomputation allocates many differently sized activations; expandable segments avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for res in ${RESS//,/ }; do
  INS=$(ls "$INPUTS" | grep "^${res}_" | sed 's/\.pt$//' | sort | head -16 | paste -sd,)
  for s in ${SYSS//,/ }; do
    "$ROOT/tools/launch.sh" experiments/end2end/wan_train.py --system "$s" --inputs "$INS" --accum 4 --iters 1 \
      --out "$OUT/steps.jsonl" > "$OUT/${res}_$s.log" 2>&1
  done
done
