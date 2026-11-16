# Implementation guide

```text
mask → assignment (GPU) → schedule (CPU) → tables → persistent GPU execution
```

Data ownership stays fixed. The assignment decides where to compute; the schedule determines the required transfers and task order.

| Stage | Main code | What it does |
| --- | --- | --- |
| Assignment | [plan.py](../dex/planner/plan.py), [cluster.py](../dex/planner/cluster.py), [kv.py](../dex/planner/kv.py) | Cluster Q rows by shared KV, place them under load caps, then try KV column re-splits. Plan across nodes before refining within each node. |
| Cost model | [models.py](../dex/planner/models.py) | Compare candidate assignments using calibrated compute and communication costs. |
| Scheduling | [fastsched_blocks.cpp](../dex/scheduler/csrc/fastsched_blocks.cpp) | Choose gateways, generate messages, within each head, order sends by gain / cost and computation by output destination and expected input readiness. |
| Distributed planning | [local.py](../dex/scheduler/local.py) | Exchange message metadata and arrival estimates so ranks can build schedules locally. |
| Execution tables | [fwd.py](../dex/scheduler/fwd.py), [bwd.py](../dex/scheduler/bwd.py) | Convert schedules into buffers, transfer pieces and task dependencies. |
| Runtime | [rt_sched.cuh](../csrc/runtime/rt_sched.cuh), [rt_engine.cuh](../csrc/runtime/rt_engine.cuh) | Observe arrival signals, dispatch ready work and return partial results. |

Assignment uses batched tensor operations and CUDA graphs. The scheduler uses CPU sorting and linear passes. A persistent kernel on each GPU runs attention workers; NVSHMEM handles inter-node transfers and copy engines handle the default intra-node path.

Algorithm details and implementation constraints are documented beside the code. Machine constants live in [config/calibration.json](../config/calibration.json).
