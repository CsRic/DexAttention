# Running the evaluation

Start with the build and smoke test in the [README](../README.md). Run the commands below from the repository root. The reported results use four nodes with four RTX A5000 GPUs each, PCIe within nodes and one 200 Gb/s InfiniBand NIC per node. Recalibrate [config/calibration.json](../config/calibration.json) for another cluster.

## Environments

Use [config/env.example.sh](../config/env.example.sh) to configure paths, hosts, interfaces and interpreters. Runtime defaults are in [config/runtime_defaults.sh](../config/runtime_defaults.sh). Use `third_party/setup_submodules.sh` to fetch the pinned dependencies; UltraAttn has a private nested dependency that prevents a plain recursive submodule update.

| Workload or system | Requirements | Extension |
| --- | --- | --- |
| W1–W3; Ring, All-gather, UA | [core.txt](../requirements/core.txt) | Default build |
| W4; W6; optional W2/W3 rollout generation | [w6.txt](../requirements/w6.txt) | BF16 build for W4; separate W6 build |
| W5 | [w5.txt](../requirements/w5.txt) | Separate W5 build |
| W6 input dump and official reference | [w6_official.txt](../requirements/w6_official.txt) | No Dex extension |
| DCP / Magi | [dcp.txt](../requirements/dcp.txt) / [magi.txt](../requirements/magi.txt) | Their own environments |

The requirements files specify versions and non-PyPI installation steps; [third_party/PINS.md](../third_party/PINS.md) records dependency commits and patches. UA requires `DEX_GUROBI_LICENSE` and `DEX_FA283_PATH`; DCP and Magi use `DEX_DCP_PYTHON` and `DEX_MAGI_PYTHON`.

An extension must match its PyTorch environment. Set `DEX_PYTHON` and `DEX_CUDA_HOME` for each build (CUDA 12.9 for W4–W6):

```bash
# W4: replace the path below with your CUDA 12.6 toolkit for InfLLM-v2.
INFLLM_CUDA_HOME=/path/to/cuda-12.6
bash build/build_ext.sh --infllm-dtype bf16 --infllm-cuda-home "$INFLLM_CUDA_HOME"
# Run these separately, from their corresponding environments:
bash build/build_ext.sh --out build/ext_w5
bash build/build_ext.sh --out build/ext_w6
```

Select the matching interpreter and `DEX_EXT_DIR` for each launch. W4 defaults to `build/ext_bf16`; W5 and W6 use their separate builds. Do not run all workloads in one environment.

## Prepare inputs

Weights, generated inputs, plans and raw evaluation records are not bundled. Place model checkpoints under `$DEX_MODEL_ROOT`; preparation scripts write to `$DEX_DATA_ROOT`. The script arguments identify the required checkpoints and datasets.

| Inputs | Preparation entry points, in order |
| --- | --- |
| W1–W3 masks and plans | `bash experiments/attention_layer/static_prepare.sh` (recorded rollout lengths are included) |
| W4 selections and QKV | `bash workloads/w4_infllm/prepare.sh` |
| W5 generation dumps and training inputs | `workloads/w5_sparge/generate.py`, then `prep_e2e_inputs.py` in the same directory |
| W6 inputs and reference | `workloads/w6_flexprefill/prep_inputs.py`, then `dump_qkv.py` and `official_reference.py` in the same directory |

W4 dumps require roughly 754 GB and W5 dumps roughly 234 GB of local storage on node 0. Use the workload-specific environments above for preparation as well.

## Run experiments

The static attention driver prepares baseline plans before timing. Run attention tests before ablations and end-to-end tests, which reuse their plans or records. Each driver lists its arguments and outputs in its header.

```bash
bash experiments/attention_layer/run.sh static   # W1–W3, core environment
bash experiments/end2end/run.sh w2 w3
# In each workload's environment, select w4, w5 or w6 separately:
bash experiments/attention_layer/run.sh w4
bash experiments/end2end/run.sh w4
```

| Paper result | Entry point |
| --- | --- |
| Attention / end-to-end (Figure 6) | Commands above |
| Planning latency (Tables 1, 4) | `experiments/planning_latency/`, also invoked by the static attention driver |
| Kernel and communication measurements (Table 2) | `bash experiments/microbench/run.sh` |
| Traffic and load balance (Table 4) | `bash experiments/comm_volume/run.sh` |
| Component replacement (Table 5) | `bash experiments/ablation_components/convert_plans.sh`, then `run.sh` in the same directory |
| Persistent execution (Table 6) | `bash experiments/ablation_executor/run.sh` |
| Memory (Table 7) | `bash experiments/memory/run.sh` after end-to-end runs |
| Planning scaling (Figure 7) | `bash experiments/plan_scaling/run.sh 0 1` (one worker; requires prepared W4–W6 data) |
| Numerical checks | `bash experiments/correctness/run.sh` after attention and end-to-end runs |

Additional component and cache studies have entry points `experiments/ablation_components/internal_run.sh` and `experiments/memory/magi_cache.sh`.

## Generate results

Records go to `$DEX_RESULT_ROOT`; summaries and figures go to its `analysis/` subdirectory.

```bash
cd analysis
python fig_attention.py
python fig_end2end.py
python fig_plan_scaling.py
python tables.py
CUDA_VISIBLE_DEVICES= python mask_stats.py
python check_expected.py
```

`check_expected.py` reports missing results and mismatches against `expected_ranges.json`; use `--partial` for a subset. Static attention excludes cached planning; dynamic attention includes planning but excludes mask selection; end-to-end includes selection. Other measurement settings are in the drivers and analysis code.

## Result notes

- **Table 4:** static planning measures warmed assigner search over both owner placements. Scheduler/table construction is separate; compilation and graph capture are excluded.
- **Table 2:** Dex communication entries are calibrated fixed overheads, excluding payload time.
- **Table 6:** the execution strategies differ in transport, routing and overlap, not just kernel launches. W4 retains decomposed step-list construction while deducting Dex input assembly, so its ratios are upper bounds. W4/W6 decomposed drivers require home assignments.
- **Accuracy correction:** the paper's training-loss bound of 2e-4 should be 2.7e-4 on W2–W4 and 2.7e-3 on W5. Run `python tables.py correctness` from `analysis/` to check new records.
