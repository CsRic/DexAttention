"""MagiAttention's allocator behaviour on its own (no model, no checkpoint): 24 calc_attn calls on one W2 input, forward only and
forward + backward; rank 0 records PyTorch's reserved and allocated memory and its peak allocation after every 4th call.
The gap between the peak allocation and the reserved cache is what cache clearing in training releases.
Inputs : $DEX_DATA_ROOT/masks/w2/<input>.npz
Outputs: --out, JSON lines {mode, call, reserved_GiB, allocated_GiB, peak_GiB}; read by magi_cache_summary.py
Example: DEX_PYTHON=$DEX_MAGI_PYTHON tools/launch.sh experiments/memory/magi_alloc.py --out results/memory/magi_alloc.jsonl
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.distributed as dist

from baselines.magi.e2e import MagiCP
from dex.common.env import cfg

GIB = 2 ** 30
CALLS = 24


def run(mode, op, rank, f):
    """mode: plain_fwd = calc_attn under no_grad, plain_fb = calc_attn + backward"""
    from magi_attention.api import calc_attn

    def mark(call):
        torch.cuda.synchronize()
        if rank == 0:
            rec = dict(mode=mode, call=call, reserved_GiB=round(torch.cuda.memory_reserved() / GIB, 2),
                       allocated_GiB=round(torch.cuda.memory_allocated() / GIB, 2),
                       peak_GiB=round(torch.cuda.max_memory_allocated() / GIB, 2))
            print(json.dumps(rec), flush=True)
            f.write(json.dumps(rec) + "\n")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mark(0)
    Tl = op.tokens.shape[0]
    for i in range(CALLS):
        q, k, v = (torch.randn(Tl, 16, 128, device="cuda", dtype=torch.float16, requires_grad=mode != "plain_fwd") for _ in range(3))
        if mode == "plain_fwd":
            with torch.no_grad():
                o = calc_attn(q=q, k=k, v=v, key=op.key)[0]
        else:
            o = calc_attn(q=q, k=k, v=v, key=op.key)[0]
            o.float().sum().backward()
        del o, q, k, v
        if i % 4 == 3:
            mark(i + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="w2_64k_p648", help="W2 input id (the nb32 mask holds its segment table)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    op = MagiCP(np.load(os.path.join(cfg("DEX_DATA_ROOT"), "masks", "w2", f"{a.input}_nb32.npz")), (16, 16))
    if rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out if rank == 0 else os.devnull, "w") as f:
        for mode in ("plain_fwd", "plain_fb"):
            run(mode, op, rank, f)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
