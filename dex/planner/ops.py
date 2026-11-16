"""Batched per-unit reductions of the planner (leading candidate dimension C).

Shapes: W [H, m, m] work per block pair (rows = a head's Q blocks, columns = its KV blocks, padding = 0), NA / dev_pair
[C, H, m, m] unit (node or GPU) of every pair (-1 = no pair), owner / home [H, m]. Every reduction is one scatter pass
(no Python loop over the units)."""
import torch

oh = torch.nn.functional.one_hot


def _counts(X, U, src=None):
    """X [C, r, k] unit ids (-1 none) -> [C, r, U] counts (or sums of src [.., r, k])"""
    live = X >= 0
    v = live.float() if src is None else torch.where(live, src.float(), torch.zeros((), device=X.device))
    return torch.zeros((X.shape[0], X.shape[1], U), device=X.device).scatter_add_(2, X.clamp(min=0), v)


def rows_any_b(NA, U):
    C = NA.shape[0]
    return _counts(NA.reshape(C, -1, NA.shape[-1]), U) > 0                   # [C, H * m, U]


def cols_any_b(NA, U):
    C = NA.shape[0]
    return _counts(NA.transpose(-1, -2).reshape(C, -1, NA.shape[-2]), U) > 0


def col_counts_b(NA, U):
    C = NA.shape[0]
    return _counts(NA.transpose(-1, -2).reshape(C, -1, NA.shape[-2]), U)


def unit_sums_b(W, NA, U, mask=None):
    """[C, U]: sum of W over the pairs on every unit (mask [C or 1, H, m, m] restricts the pairs)"""
    C = NA.shape[0]
    live = NA >= 0
    if mask is not None:
        live = live & mask
    v = torch.where(live, W.float().expand_as(NA), torch.zeros((), device=W.device))
    # per row first, then over the rows: a single scatter of every pair into <= U counters would serialise on the
    # atomics. Sums of tile counts: exact either way.
    k = NA.shape[-1]
    return _counts(torch.where(live, NA, torch.full_like(NA, -1)).reshape(C, -1, k), U, v.reshape(C, -1, k)).sum(1)


def _by_home(home, X, NN):
    """H_.T @ X for the one-hot H_ of home [n] (X [C, n, U]) without a GEMM (sums of 0/1 rows, exact): [C, NN, U]"""
    return torch.zeros((X.shape[0], NN, X.shape[2]), device=X.device, dtype=X.dtype).index_add_(1, home, X)


def intra_io_b(use, owner, gpn, NN, b_in, b_out):
    """intra-node bytes per GPU, both directions: use [C, n, P] = GPUs that need block i (one side), owner
    [n]. Same node: the owner sends the input to each reader, each reader sends its partial back (b_out). Another node:
    the first reader there (the relay) receives it from the NIC and forwards it to the node's other readers; their
    partials are collected at the relay first. -> (in [C, P], out [C, P])"""
    C, n, P = use.shape; dev = use.device
    hm = oh(owner, P).bool()
    on = owner // gpn
    homein = (on[:, None] == torch.arange(NN, device=dev)[None, :])        # [n, NN]
    un = (use & ~hm).reshape(C, n, NN, gpn)
    cnt = un.float().sum(3)                                                # [C, n, NN] readers per node
    first = oh(un.float().argmax(3), gpn).bool() & un.any(3)[..., None]    # relay of a remote node
    rest = torch.where(homein[None, :, :, None], un, un & ~first).float()  # readers fed / collected inside the node
    n_rest = torch.where(homein[None], cnt, (cnt - 1).clamp(min=0))        # [C, n, NN]
    inn = b_in * rest + b_out * torch.where(homein[None, :, :, None], 0.0, first.float() * n_rest[..., None])
    out = b_out * rest + b_in * torch.where(homein[None, :, :, None], 0.0, first.float() * n_rest[..., None])
    own_same = (cnt * homein[None].float()).sum(2)                         # [C, n] same-node readers of the owner
    inn = inn.reshape(C, n, P).sum(1) + torch.zeros((C, P), device=dev).index_add_(1, owner, b_out * own_same)
    out = out.reshape(C, n, P).sum(1) + torch.zeros((C, P), device=dev).index_add_(1, owner, b_in * own_same)
    return inn, out


def components_b(A, unit, iters=12):
    """connected components (rows sharing a KV column) inside each unit, per candidate: unit [C, H, m]; A [H, m, m]
    shared or [C, H, m, m] per candidate"""
    m = A.shape[-2]; dev = A.device
    Ah = A.half()
    E = (Ah @ Ah.transpose(-1, -2)) > 0
    E = (E if A.dim() == 4 else E[None]) & (unit[..., :, None] == unit[..., None, :])
    lab = torch.arange(m, device=dev).expand_as(unit).clone()
    for _ in range(iters):
        nb_min = torch.where(E, lab[..., None, :], torch.full_like(lab[..., None, :], m)).min(-1).values
        lab = torch.minimum(lab, nb_min)
        lab = lab.gather(2, lab)
    return lab



def row_counts(NA, U, src=None):
    """[n, U]: per row, the pairs (or the sum of src over the pairs) on every unit."""
    live = NA >= 0
    v = live.float() if src is None else torch.where(live, src.float(), torch.zeros_like(src, dtype=torch.float))
    return torch.zeros((NA.shape[0], U), device=NA.device).scatter_add_(1, NA.clamp(min=0), v)



def fr(X):
    return X.reshape(-1, X.shape[-1])                                     # [H * m, m]: row (h, i) -> h * m + i

