"""Table 2 compute column: standalone FlashAttention 2.8.3 calls on one GPU, dense non-causal MHA 16 / 16, d = 128, fp16,
S = 128 ... 8192 (single latency: sync -> call -> sync, median of 10 after 3 warm-ups). Effective TFLOP/s with forward
FLOPs 4 S^2 H d and backward 2.5x forward.
Inputs : FlashAttention 2.8.3 importable (DEX_FA283_PATH)   Output: one record per S appended to --out
Example: python experiments/microbench/compute.py --out results/microbench/compute_fa2.jsonl
"""
import argparse
import json
import os
import time

import numpy as np
import torch

H, D, SIZES = 16, 128, (128, 256, 512, 1024, 2048, 4096, 8192)


def timed(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(); v = []
    for _ in range(iters):
        torch.cuda.synchronize(); a = time.perf_counter(); fn(); torch.cuda.synchronize(); v.append(time.perf_counter() - a)
    return float(np.median(v)) * 1e3


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_backward, _flash_attn_forward
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    dev = "cuda"; torch.manual_seed(0); sc = D ** -0.5
    for S in SIZES:
        q, k, v, do = ((torch.randn(1, S, H, D, device=dev) * 0.5).half() for _ in range(4))
        rec = dict(S=S, H=H, d=D, fa_version=flash_attn.__version__)
        fl = 4.0 * S * S * H * D
        o, lse, _, _ = _flash_attn_forward(q, k, v, 0.0, sc, False, -1, -1, 0.0, None, False)
        rec["fwd_ms"] = timed(lambda: _flash_attn_forward(q, k, v, 0.0, sc, False, -1, -1, 0.0, None, False))
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        rec["bwd_ms"] = timed(lambda: _flash_attn_backward(do, q, k, v, o, lse, dq, dk, dv, 0.0, sc, False, -1, -1, 0.0, None, False))
        rec["fwd_tflops"] = round(fl / rec["fwd_ms"] / 1e9, 2); rec["bwd_tflops"] = round(2.5 * fl / rec["bwd_ms"] / 1e9, 2)
        print(json.dumps(rec), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        del q, k, v, do
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
