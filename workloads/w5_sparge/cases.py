"""W5 operator-test cases and the shared timing helper.

A case is one self-attention call of a generation: `<generation> <call>` per line of cases42.txt (2 resolutions x
denoising steps {0, 5, 15, 25, 35, 45, 49} x layers 0 / 15 / 29, the generation rotating over the 16 of a resolution).
The dump of a generation is $DEX_DATA_ROOT/w5_sparge/dumps/<generation>/rank<r>/call<i>.npz (r = 0..3: the Ulysses
ranks of the generating run, each holding 3 of the 12 heads; q / k fp16 [1, 3, S, 128], step, call_in_step, seqlen)."""

import os
import time

import numpy as np
import torch
import torch.distributed as dist

from dex.common.env import cfg

HERE = os.path.dirname(os.path.abspath(__file__))
CASES42 = os.path.join(HERE, "cases42.txt")
DUMP_RANKS = 4


def dump_dir(generation):
    return os.path.join(cfg("DEX_DATA_ROOT"), "w5_sparge", "dumps", generation)


def read_cases(path=CASES42):
    """-> [(generation, call)] in file order (the case index seeds V / dO)"""
    out = []
    for ln in open(path):
        if ln.strip():
            g, c = ln.split()
            out.append((g, int(c)))
    return out


def load_case(generation, call):
    """-> q, k [12, S, 128] fp16 (numpy), meta dict(step, call_in_step, S)"""
    d = dump_dir(generation)
    q, k = [], []
    meta = None
    for r in range(DUMP_RANKS):
        z = np.load(f"{d}/rank{r}/call{call:05d}.npz")
        q.append(z["q"][0])
        k.append(z["k"][0])
        meta = dict(step=int(z["step"]), call_in_step=int(z["call_in_step"]), S=int(z["seqlen"]))
    return np.concatenate(q), np.concatenate(k), meta


def latency(fn, dev, warmup, iters):
    """barrier -> slowest rank done, `warmup` untimed calls, median of `iters` timed calls -> (median ms, all ms)"""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t = torch.tensor([time.perf_counter() - t0], device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        ts.append(t.item() * 1e3)
    return float(np.median(ts)), [round(x, 3) for x in ts]
