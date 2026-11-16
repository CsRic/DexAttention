#!/usr/bin/env bash
# Table 6: compare execution strategies under Dex assignments. Decomposed execution uses its own gateways,
# step schedule and direct result returns, with batched NCCL and one attention call per step. It changes transport
# and overlap as well as launches. Planning deductions and W4's timing qualification: docs/reproduction.md.
# Inputs : as experiments/attention_layer
# Outputs: $DEX_RESULT_ROOT/ablation_executor/<w1|w2|w3>/decomposed_launch<k>.jsonl, attention_layer/<w4|w5|w6>/decomposed/rep<k>.jsonl
# Example: bash experiments/ablation_executor/run.sh ; python analysis/tables.py table6
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
source "$ROOT/config/env.sh"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
# W1-W3 run in the environment of Dex's static tests; each W4-W6 run script sets its own
export DEX_ENV_PROFILE=static
for launch in 1 2; do
  for w in w1 w2 w3; do
    mapfile -t ids < <("$DEX_PYTHON" -c "
import sys; sys.path.insert(0, 'experiments/attention_layer')
import static_inputs as SI
print('\n'.join(c['id'] for c in SI.inputs('$w')))")
    for ((i = 0; i < ${#ids[@]}; i += 8)); do        # 8 inputs per 16-GPU launch
      sel=$(IFS=,; echo "${ids[*]:i:8}")
      bash tools/launch.sh baselines/decomposed/w123.py --workload "$w" --ids "$sel" --launch "$launch"
    done
  done
done
bash experiments/attention_layer/w4_run.sh decomposed 1,2
bash experiments/attention_layer/w5_run.sh decomposed 1,2
bash experiments/attention_layer/w6_run.sh decomposed 1,2
