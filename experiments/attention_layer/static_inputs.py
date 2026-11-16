"""W1-W3 static inputs: the case list, file locations and the planner's problem (block work matrix) of each input.

Files under $DEX_DATA_ROOT:
  masks/w1/<id>.npz, masks/w1/<id>_ranges.npy                    workloads/w1_lambda/gen_masks.py
  masks/<w2|w3>/<id>_nb<nb>.npz, masks/<w2|w3>/<id>_ranges.npy   workloads/w2_w3_rollout/gen_masks.py
  rollouts/names_<w2|w3>.txt, rollouts/selection.json            workloads/w2_w3_rollout/select_groups.py
  plans/static/hs<hv>/<id>.npz, <id>_fwd.json, <id>_bwd.json     static_plans.py (virtual-head mask + Dex plans)
Plan block counts (forward, backward): W1 (32, 128), W2 / W3 (32, 32); 16 GPUs as 4 nodes x 4 GPUs.
"""
import json
import os

import numpy as np
import torch

from dex.common.env import cfg

TILE, D = 128, 128
P, GPN, NN = 16, 4, 4
W1_TOKENS = (32768, 65536, 131072, 262144)
W1_SINK, W1_WINDOW = 128, 4096
BLOCKS = {"w1": (32, 128), "w2": (32, 32), "w3": (32, 32)}
ROLLOUT_MASK_NB = 32
PLACEMENT = {"w1": "contiguous", "w2": "zigzag", "w3": "zigzag"}   # deployed owner placement per workload


def data_root():
    return cfg("DEX_DATA_ROOT")


def plan_dir(head_split):
    return os.path.join(data_root(), "plans", "static", f"hs{head_split}")


def inputs(workload):
    """-> [dict(id, workload, mask, ranges, S)] in the workload's canonical order (S = real tokens)"""
    root = data_root()
    out = []
    if workload == "w1":
        from workloads.w1_lambda.gen_masks import mask_name
        for t in W1_TOKENS:
            n = mask_name(t, W1_SINK, W1_WINDOW)
            out.append(dict(id=n, workload="w1", mask=os.path.join(root, "masks", "w1", n + ".npz"),
                            ranges=os.path.join(root, "masks", "w1", n + "_ranges.npy"), S=t))
        return out
    sel = json.load(open(os.path.join(root, "rollouts", "selection.json")))
    for c in sel:
        n = f"{workload}_{c['split']}_p{c['idx']}"
        d = os.path.join(root, "masks", workload)
        out.append(dict(id=n, workload=workload, mask=os.path.join(d, f"{n}_nb{ROLLOUT_MASK_NB}.npz"),
                        ranges=os.path.join(d, f"{n}_ranges.npy"), S=c["S"], length_bin=c["split"], rho=c["rho"]))
    return out


def plan_files(case, head_split):
    d = plan_dir(head_split)
    return (os.path.join(d, case["id"] + ".npz"), os.path.join(d, case["id"] + "_fwd.json"),
            os.path.join(d, case["id"] + "_bwd.json"))


def block_problem(m, block_tokens, dev):
    """tile mask [n, n] -> (W [nb, nb] work per block pair in full-tile units (diagonal tile = 0.5), kcols [nb] live
    tile columns per KV block, nb)"""
    k = block_tokens // TILE
    n = m.shape[0]
    nb = n // k
    W = torch.tensor(np.where(m == 2, 0.5, (m > 0) * 1.0).reshape(nb, k, nb, k).sum((1, 3)), dtype=torch.float32, device=dev)
    kc = torch.tensor((m > 0).reshape(n, nb, k).any(0).sum(1).astype(np.float32), device=dev)
    return W, kc, nb


def bdiag(X, hv, fill):
    """hv copies of X on the block diagonal of a [hv n, hv n] matrix (fill elsewhere)"""
    n = X.shape[0]
    Y = torch.full((hv * n, hv * n), fill, dtype=X.dtype, device=X.device)
    for h in range(hv):
        Y[h * n:(h + 1) * n, h * n:(h + 1) * n] = X
    return Y


def virtual_heads(heads, head_split):
    """number of head copies hv (head_split capped at the KV heads) and the per-copy heads (hq / hv, hk / hv)"""
    hq, hk = heads
    hv = head_split if head_split <= hk else hk
    return hv, (hq // hv, hk // hv)
