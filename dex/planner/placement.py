"""Owner placement of the plan blocks. The planner plans for given owners; the placement is chosen before it.

Rule: equal chunks per rank (+-1 block) always; zigzag by default. `choose`: plan the forward and the backward for
every equal-chunk candidate (zigzag, contiguous) and keep the candidate with the lower modelled forward + backward time
(forward model `models.node_time_b` -> `gpu_time_b`, backward model `bwd_gpu_time_b`; the same owners serve both passes,
as in training)."""
import numpy as np
import torch


def zigzag_equal(nb, P):
    """Standard zigzag CP layout with equal block counts (+-1): the sequence is cut into 2P chunks; GPU g owns chunk g
    and chunk 2P-1-g (chunk sizes from the per-GPU counts, split as evenly as possible). -> owner GPU per block [nb]"""
    base, extra = divmod(nb, P)
    cap = np.array([base + (g < extra) for g in range(P)], np.int64)
    first, second = (cap + 1) // 2, cap // 2
    order = [g for g in range(P) for _ in range(first[g])] + [g for g in range(P - 1, -1, -1) for _ in range(second[g])]
    return np.array(order, np.int64)


def contiguous_equal(nb, P):
    """GPU g owns one contiguous run of blocks; equal counts (+-1). -> owner GPU per block [nb]"""
    base, extra = divmod(nb, P)
    return np.repeat(np.arange(P), [base + (g < extra) for g in range(P)]).astype(np.int64)


CANDIDATES = {"zigzag": zigzag_equal, "contiguous": contiguous_equal}


def fwd_time(W, owner, dev_pair, gpn, NN, Qb, Ob, Kb, TT):
    """modelled forward time (s) of a plan: W [n, n] work, owner [n], dev_pair [n, n] GPU per pair (-1 none)"""
    from .models import node_time_b, gpu_time_b
    W = W.float(); dp = torch.where(W > 0, dev_pair, torch.full_like(dev_pair, -1))
    NA = torch.where(dp >= 0, dp // gpn, dp)[None, None]
    _, _, arr = node_time_b(W[None], NA, (owner // gpn)[None], NN, gpn, Qb, Ob, Kb, TT)
    F, _ = gpu_time_b(W[None], dp[None, None], owner[None], gpn, NN, arr, Qb, Ob, Kb, TT)
    return F.max().item()


def bwd_time(W, owner, dev_pair, gpn, NN, heads, B, kcols):
    """modelled backward time (s) of a plan; kcols [n] live tile columns per KV block"""
    from .models import bwd_models_b
    W = W.float(); dp = torch.where(W > 0, dev_pair, torch.full_like(dev_pair, -1))
    _, gt, _, _ = bwd_models_b(heads, B, kcols, gpn, NN)
    F, _ = gt(W, dp[None], owner, torch.zeros(1, device=W.device))
    return F.max().item()


def choose(W, gpn, NN, Qb, Ob, Kb, TT, heads, B, kcols, plan_fn):
    """-> (name, owner [n] numpy, fwd plan [n, n], bwd plan [n, n], {name: (t_fwd, t_bwd)}) for the candidate with the
    lowest modelled t_fwd + t_bwd. plan_fn = planner.plan."""
    n, P = W.shape[0], gpn * NN
    res = {}
    for name, f in CANDIDATES.items():
        ow = f(n, P); ot = torch.as_tensor(ow, device=W.device)
        sf = plan_fn(W, ot, gpn, NN, Qb, Ob, Kb, TT)
        sb = plan_fn(W, ot, gpn, NN, Qb, Ob, Kb, TT, bwd=(heads, B, kcols))
        res[name] = (ow, sf, sb, fwd_time(W, ot, sf, gpn, NN, Qb, Ob, Kb, TT), bwd_time(W, ot, sb, gpn, NN, heads, B, kcols))
    best = min(res, key=lambda k: res[k][3] + res[k][4])
    return best, res[best][0], res[best][1], res[best][2], {k: (v[3], v[4]) for k, v in res.items()}
