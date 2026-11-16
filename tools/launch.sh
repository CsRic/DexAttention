#!/bin/bash
# Multi-node torchrun launcher: starts one torchrun per host of the hostfile (the first locally, the others over ssh) and
# waits for them. The checkout must be at the same path on every node. The caller's exported environment reaches every node.
# Inputs : config/env.sh, config/runtime_defaults.sh; <script> relative to the checkout or absolute, then its arguments.
#          DEX_NODES       the first N hosts only (default: all)
#          DEX_ENV_PROFILE a profile of config/runtime_defaults.sh to apply (operator, training, static; default: none)
#          DEX_PRELOAD_NVSHMEM 1 / 0: preload the NVSHMEM host library; default: 1 when the arguments contain
#                            `--system dex*`, else 0 (baselines)
# Outputs: the script's own; other nodes' lines are prefixed by their node rank. Returns the first non-zero rc, else 0.
# Example: DEX_ENV_PROFILE=training tools/launch.sh experiments/attention_layer/w5_op.py --system dex --out w5.jsonl
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PRESET_PYTHON=${DEX_PYTHON:-}        # a caller's interpreter (a baseline's own environment) wins over config/env.sh
source "$ROOT/config/env.sh"
DEX_PYTHON=${PRESET_PYTHON:-$DEX_PYTHON}
source "$ROOT/config/runtime_defaults.sh"
[ $# -ge 1 ] || { echo "usage: tools/launch.sh <script> [args...]" >&2; exit 2; }
dex_env_common
if [ -n "${DEX_ENV_PROFILE:-}" ]; then "dex_env_$DEX_ENV_PROFILE"; fi
if [ -z "${DEX_PRELOAD_NVSHMEM:-}" ]; then
  DEX_PRELOAD_NVSHMEM=0
  PREV=
  for a in "$@"; do
    case "$PREV:$a" in --system:dex*) DEX_PRELOAD_NVSHMEM=1 ;; esac
    case "$a" in --system=dex*) DEX_PRELOAD_NVSHMEM=1 ;; esac
    PREV=$a
  done
fi
if [ "$DEX_PRELOAD_NVSHMEM" = 1 ]; then dex_env_preload; fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

case "$DEX_HOSTFILE" in /*) HF=$DEX_HOSTFILE ;; *) HF=$ROOT/$DEX_HOSTFILE ;; esac
mapfile -t HOSTS < <(grep -v '^[[:space:]]*\(#\|$\)' "$HF")
NN=${DEX_NODES:-${#HOSTS[@]}}
[ "$NN" -le "${#HOSTS[@]}" ] || { echo "DEX_NODES=$NN exceeds the hosts of $HF" >&2; exit 2; }

TR=(-m torch.distributed.run --nnodes "$NN" --nproc-per-node "$DEX_GPUS_PER_NODE"
    --master-addr "$DEX_MASTER_ADDR" --master-port "$DEX_MASTER_PORT")
# the whole exported environment, minus what describes this login session
EXPORTS=
for v in $(compgen -e); do
  case "$v" in
    _|HOME|USER|LOGNAME|SHELL|SHLVL|PWD|OLDPWD|HOSTNAME|TERM|DISPLAY|MAIL|LS_COLORS|SSH_*|XDG_*|DBUS_*|BASH_*) continue ;;
  esac
  EXPORTS+="export $v=$(printf '%q' "${!v}"); "
done
ARGS=$(printf '%q ' "$@")

PIDS=()
for ((n = 1; n < NN; n++)); do
  (
    ssh -n -o BatchMode=yes "${HOSTS[$n]}" \
      "$EXPORTS cd $(printf '%q' "$ROOT") && exec $(printf '%q' "$DEX_PYTHON") ${TR[*]} --node-rank $n $ARGS" \
      2>&1 | sed -u "s/^/[node $n] /"
  ) &
  PIDS+=($!)
done
RC=0
(cd "$ROOT" && "$DEX_PYTHON" "${TR[@]}" --node-rank 0 "$@") || RC=$?
for p in ${PIDS[@]+"${PIDS[@]}"}; do
  wait "$p" || { r=$?; [ "$RC" -ne 0 ] || RC=$r; }
done
exit "$RC"
