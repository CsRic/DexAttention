"""Table 2 Dex row: the persistent forward kernel on one GPU, dense non-causal MHA 16 / 16, S = 128 and 8192, every block
pair on GPU 0 (no communication). The amortized tile time is the 8K pass over its (S / 128)^2 tiles; the 128-token call is
the kernel's launch and setup cost. Same forward path and protocol as experiments/attention_layer/static_op.py.
Output : one record per S appended to --out (fields S, fwd_ms)
Example: DEX_NODES=1 DEX_GPUS_PER_NODE=1 tools/launch.sh experiments/microbench/dex_tile.py --out results/microbench/dex_fwd.jsonl
"""
import argparse
import json
import os
import sys
import types

import numpy as np
import torch
import torch.distributed as dist

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [REPO, os.path.join(REPO, "experiments", "attention_layer")]
import static_op as OP  # noqa: E402
from dex.common.env import cfg  # noqa: E402


def make_input(S, scratch):
    n = S // 128
    bt = min(S, 2048); nb = S // bt
    mask = os.path.join(scratch, f"dense_nc_{S}.npz"); plan = os.path.join(scratch, f"plan_nc_{S}.json")
    np.savez(mask, mask=np.ones((n, n), np.int8), tokens=np.int64(S), heads=np.array([16, 16]), block_tokens=np.int64(bt))
    json.dump(dict(cp=1, n_blocks=nb, nodes=1, gpus_per_node=1, block_tokens=bt, owner=[0] * nb, schedule_table=np.zeros((nb, nb), int).tolist()),
              open(plan, "w"))
    return mask, plan


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    dist.init_process_group("nccl")
    from dex import dex_cuda, scheduler
    OP.C, OP.S = dex_cuda, scheduler
    dev = torch.device("cuda", 0)
    scratch = cfg("DEX_SCRATCH")
    built = []
    for S in (8192, 128):                          # the largest region first: later regions fit the first allocation
        mask, plan = make_input(S, scratch)
        built.append((S, OP.fwd_build(dict(id=f"dense_{S}", workload="micro", S=S), mask, plan, 0, 1, dev)))
    os.environ["NVSHMEM_SYMMETRIC_SIZE"] = str(int(OP.fwd_region_bytes(built[0][1]["tables"], 16, 16) * 1.15) + (512 << 20))
    from dex.runtime import finalize_nvshmem, init_nvshmem
    init_nvshmem(dist.group.WORLD)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    for S, b in built:
        r = OP.dex_fwd_case(b, types.SimpleNamespace(launch=2), 0, dev)
        with open(a.out, "a") as f:
            f.write(json.dumps(dict(S=S, fwd_ms=r["lat_ms"], fwd_ms_all=r["lat_ms_all"])) + "\n")
        print(S, r["lat_ms"], flush=True)
    dist.barrier()
    finalize_nvshmem()


if __name__ == "__main__":
    main()
