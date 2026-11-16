"""Prepare and score the two levels of assignment.

Node level: cluster and fill Q rows for load-cap candidates, add a home
assignment, then re-split KV and retain each candidate's node assignment.
GPU level: pack each node's work into a compact problem, cluster load-cap
candidates alongside a preset home assignment, then choose by modeled time
(then bytes). Already placed home rows do not participate in clustering.
The packing keeps head-local KV columns separate.
"""
import math

import torch

from .ops import fr, row_counts, rows_any_b, cols_any_b
from .models import node_time_b, gpu_node_c, node_bytes_b, lexpick
from .cluster import ap_init, ap_finish
from .kv import kv_init, kv_round, kv_plan

CHECK_UNIQUE = False   # (tests: assert the GPU-level part targets are distinct)
HB_RH = None        # (compiled Planner: a static Rh bucket; eager: the exact largest (node, head) row count)


def _parts(W, NA, owner, gid, gpn, NN, vmax, R, Ch, xs=None):
    """the GPU level's parts (row, node) with work, in the dense order (row id, node) -- rid = the dense virtual-row
    index -- regrouped per node, ALL heads in one problem (per-(node, head) groups would each be padded to the largest
    group and carry the head's full column axis): node x's parts in (head, row) order (<= R) x their KV
    column blocks indexed INSIDE the row's head (<= Ch per (node, head); heads share no columns, so similarities are
    taken within a head: exact). Returns Wg [NN, 1, R, Ch], rid, row_home [NN, 1, R], kv_home [NN, 1, H, Ch], per
    part slot, order, node work, the columns' block ids idx_c [NN, H, Ch], the rows' heads [NN, 1, R].
    xs (distributed planning): only these nodes' problems (Nx = len(xs) in place of NN; None = all)."""
    H_, m, _ = W.shape; dev = W.device; nr = H_ * m
    Wf, NAf = fr(W), fr(NA)
    S = row_counts(NAf, NN, Wf)                                        # [rows, NN] work of every part
    work = S > 0
    gr = gid.reshape(-1).clamp(min=0)
    key = torch.where(work, gr[:, None] * NN + torch.arange(NN, device=dev)[None, :], torch.full((nr, NN), nr * NN * 4, dtype=torch.long, device=dev))
    order = torch.argsort(key.reshape(-1))[:vmax]                         # part = row * NN + node
    prow, pu = order // NN, order % NN
    pwork = work.reshape(-1)[order]
    Wv = Wf.index_select(0, prow) * (NAf.index_select(0, prow) == pu[:, None])
    ph = prow // m
    pos = (work.long().cumsum(0) - 1)[prow, pu]                          # rank inside the node, (head, row) order
    xsv = node_ids(xs, NN, dev); Nx = xsv.shape[0]
    lid = torch.full((NN,), -1, dtype=torch.long, device=dev).index_copy_(0, xsv, torch.arange(Nx, device=dev))[pu]
    trash = Nx * R
    slot = torch.where(pwork & (pos < R) & (lid >= 0), lid * R + pos, torch.full_like(pos, trash))
    inC = cols_any_b(NA[None], NN)[0].t()[xsv].reshape(Nx, H_, m)        # (head, column) blocks of node x
    cp = inC.long().cumsum(2) - 1                                         # position inside (node, head)
    colpos = torch.where(inC & (cp < Ch), cp, torch.full_like(cp, Ch)).reshape(Nx, nr)
    pc = colpos[lid.clamp(min=0)[:, None], ph[:, None] * m + torch.arange(m, device=dev)[None, :]]   # [vmax, m]
    tgt = torch.where((Wv > 0) & (slot < trash)[:, None] & (pc < Ch), slot[:, None] * Ch + pc, torch.full_like(pc, trash * Ch))
    # plain stores, not accumulate=True: every valid (part slot, head-local column) target is distinct; the invalid
    # entries (most of the [vmax, m] block when heads multiply the rows) all go to the one trash slot, where atomics
    # would serialise
    if CHECK_UNIQUE:
        tv = tgt.reshape(-1)[tgt.reshape(-1) < trash * Ch]
        assert tv.numel() == torch.unique(tv).numel(), "duplicate GPU-level part targets"
    Wg = torch.zeros(trash * Ch + 1, device=dev).index_put_((tgt.reshape(-1),), Wv.reshape(-1), accumulate=False)[:-1].reshape(Nx, 1, R, Ch)
    rhd = torch.full((trash + 1,), -1, dtype=torch.long, device=dev).index_copy_(0, slot, ph)[:trash].reshape(Nx, 1, R)
    rid = torch.full((trash + 1,), -1, dtype=torch.long, device=dev).index_copy_(0, slot, torch.arange(vmax, device=dev))[:trash].reshape(Nx, 1, R)
    po = owner.reshape(-1)[prow]
    rh_ = torch.where(po // gpn == pu, po - pu * gpn, torch.full_like(po, -1))
    rh = torch.full((trash + 1,), -1, dtype=torch.long, device=dev).index_copy_(0, slot, rh_)[:trash].reshape(Nx, 1, R)
    from .kv import compact_idx
    idx_c, val_c = compact_idx(inC.reshape(Nx * H_, m), Ch)               # [Nx H, Ch] column inside its head
    idx_c = (idx_c + (torch.arange(Nx * H_, device=dev) % H_)[:, None] * m).reshape(Nx, H_, Ch)   # -> block id
    oc = owner.reshape(-1)[idx_c]; nodes = xsv[:, None, None]
    kh = torch.where(val_c.reshape(Nx, H_, Ch) & (oc // gpn == nodes), oc - nodes * gpn, torch.full_like(oc, -1))[:, None]
    nodew = torch.zeros(NN, device=dev).index_add_(0, pu, Wv.sum(1))[xsv]
    return Wg, rid, rh, kh, slot, order, nodew, torch.where(val_c.reshape(Nx, H_, Ch), idx_c, torch.full_like(idx_c, -1)), rhd


def _hb_split(Wg, rid, rh, kh, rhd, H_, Rh):
    """(heads never share KV columns, so clustering runs per head) the per-node problems in head-batched form:
    merged rows [Nx, 1, R] in (head, row) order -> [Nx, H, Rh] (row position inside its head; Rh >= the largest
    (node, head) row count). The clustering's per-group unit loads / caps / acceptance order still span all heads
    (cluster.py handles a head axis exactly as the node level does), and cross-head similarity is 0 in both forms, so
    the result equals the merged form's. Returns (Wg, rid, rh, rhd, kh) head-batched + inv [Nx, R] (merged slot ->
    flat head-batched index, -1 = no row)."""
    Nx, _, R, Ch = Wg.shape; dev = Wg.device
    h = rhd[:, 0]                                                         # [Nx, R], -1 = empty slot
    ok = h >= 0
    ohh = torch.nn.functional.one_hot(h.clamp(min=0), H_) * ok[..., None]
    pos = (ohh.cumsum(1) - 1).gather(2, h.clamp(min=0)[..., None])[..., 0]   # position inside the row's head
    inv = torch.where(ok & (pos < Rh), h * Rh + pos, torch.full_like(h, -1))
    tgt = torch.where(inv >= 0, inv, torch.full_like(inv, H_ * Rh))       # (overflow / empty -> trash slot)
    def sc(X, fill):                                                      # [Nx, 1, R, ...] -> [Nx, H, Rh, ...]
        tail = X.shape[3:]
        out = torch.full((Nx, H_ * Rh + 1, *tail), fill, dtype=X.dtype, device=dev)
        out.scatter_(1, tgt.reshape(Nx, R, *([1] * len(tail))).expand(Nx, R, *tail), X[:, 0])
        return out[:, :-1].reshape(Nx, H_, Rh, *tail)
    return (sc(Wg, 0), sc(rid, -1), sc(rh, -1), sc(rhd, -1), kh[:, 0], inv)


def hb_rows(Wg, rhd, H_):
    """the largest (node, head) row count of the merged per-node problems (eager: one sync)"""
    h = rhd[:, 0]
    return int(torch.nn.functional.one_hot(h.clamp(min=0), H_).mul((h >= 0)[..., None]).sum(1).max())


def node_ids(xs, NN, dev):
    """the nodes a (distributed) planner instance solves at the GPU level: xs (a long tensor -- a runtime input, so one
    compiled instance serves any node --, or a tuple) or all"""
    if xs is None:
        return torch.arange(NN, device=dev)
    return xs if torch.is_tensor(xs) else torch.tensor(list(xs), dtype=torch.long, device=dev)


def mbuckets(m, NN):
    """static per-(node, head) capacities: multiples of the mean parts per (node, head)"""
    base = max(1, math.ceil(m / NN))
    fr_ = [1, 1.25, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    return sorted({min(m * NN, math.ceil(f * base / 8) * 8) for f in fr_} | {m * NN})


def static_shapes(sync, H_, m, NN):
    """from the one host sync (virtual rows, rows of the largest node, column blocks of the largest node, of the
    largest (node, head)): the GPU level's static shapes -- virtual-row bucket, the per-node problem size R x Cc (the
    GPU-level KV re-split) and the head-local column count Ch (clustering) (few distinct values: few graph captures)"""
    from .kv import mx_bucket
    vmax = next(b for b in buckets(H_ * m, NN) if b >= int(sync[0]))
    return (vmax, mx_bucket(int(sync[1]), H_ * m), mx_bucket(int(sync[2]), H_ * m), mx_bucket(int(sync[3]), m),
            mx_bucket(int(sync[4]), m))   # (+ rows of the largest (node, head): the head-batched GPU level)



def buckets(n, NN):
    fr = [1, 1.25, 1.5, 2] + list(range(3, NN + 1))
    return sorted({min(NN * n, math.ceil(f * n / 64) * 64) for f in fr if f <= NN})



def node_pre(W, owner, gid, gpn, NN, Qb, Ob, Kb, TT, eps_node, models=None):
    H_, m, _ = W.shape; dev = W.device
    W = W.float(); own_node = owner // gpn
    WQ, WK = (Qb + Ob, Kb) if models is None else (models[2], models[3])
    mean = W.sum() / NN
    own_load = torch.zeros(NN, device=dev).index_add_(0, own_node.reshape(-1), W.sum(2).reshape(-1))
    caps = torch.stack([torch.maximum(own_load.max(), (1 + max(eps_node)) * mean)] + [(1 + e) * mean for e in eps_node])
    Ck = caps.shape[0] - 1; cp = caps[1:]
    ex = lambda X: X[None].expand(Ck, *X.shape)                          # noqa: E731
    state, consts = ap_init(ex(W), ex(own_node), ex(own_node), NN, cp[:, None].expand(Ck, NN), WQ, WK, ex(gid),
                               cp[:, None, None].expand(Ck, H_, m))
    return state, consts, caps


def node_mid(W, owner, gid, gpn, NN, Qb, Ob, Kb, TT, state, consts, caps, models=None):
    W = W.float(); own_node = owner // gpn
    if models is not None:
        ntime, _, WQ, WK = models
    else:
        WQ, WK = Qb + Ob, Kb
        ntime = lambda W_, NA_, hn_: node_time_b(W_, NA_, hn_, NN, gpn, Qb, Ob, Kb, TT)   # noqa: E731
    units = ap_finish(state, consts, NN, WQ, WK)
    unit = torch.cat([own_node[None], units])
    return kv_init(W, own_node, unit, gid, NN, caps, lambda NA_: ntime(W, NA_, own_node)[:2])


def node_kv_round(W, owner, gid, gpn, NN, Qb, Ob, Kb, TT, kst, caps, ncap, models=None, min_gain=0.0):
    W = W.float(); own_node = owner // gpn
    if models is not None:
        ntime, _, WQ, WK = models
    else:
        WQ, WK = Qb + Ob, Kb
        ntime = lambda W_, NA_, hn_: node_time_b(W_, NA_, hn_, NN, gpn, Qb, Ob, Kb, TT)   # noqa: E731
    return kv_round(W, own_node, gid, NN, caps, WQ, WK, lambda NA_: ntime(W, NA_, own_node)[:2], kst, ncap=ncap, min_gain=min_gain)


def node_end(W, owner, gpn, NN, Qb, Ob, Kb, TT, kst, models=None, keep=None):
    H_, m, _ = W.shape
    W = W.float(); own_node = owner // gpn
    if models is not None:
        ntime, _, WQ, WK = models
    else:
        WQ, WK = Qb + Ob, Kb
        ntime = lambda W_, NA_, hn_: node_time_b(W_, NA_, hn_, NN, gpn, Qb, Ob, Kb, TT)   # noqa: E731
    NA = kv_plan(W, kst); C = NA.shape[0]
    if keep is not None:
        Tn = kst[4].double()                                              # (the model time after the re-split)
        Bn = node_bytes_b(W, NA, own_node, NN, WQ, WK).double()
        i1 = torch.argsort(Bn, stable=True)
        sel = i1.index_select(0, torch.argsort(Tn.index_select(0, i1), stable=True))[:keep]
        NA = NA.index_select(0, sel); C = keep
    ra = rows_any_b(NA, NN)
    rn = ra.sum(1).max(1).values                                          # rows (all heads) of the largest node
    ca = cols_any_b(NA, NN)
    cn = ca.sum(1).max(1).values                                          # its column blocks
    ch = ca.reshape(C, H_, m, NN).sum(2).reshape(C, -1).max(1).values     # column blocks of the largest (node, head)
    rh = ra.reshape(C, H_, m, NN).sum(2).reshape(C, -1).max(1).values     # row blocks of the largest (node, head)
    L = torch.stack([ra.reshape(C, -1).sum(1), rn, cn, ch, rh], 1)        # (virtual rows, node rows, node columns, ..)
    return NA, ntime(W, NA, own_node)[2], L


def gpu_pre(W, owner, gid, NA_all, gpn, NN, Qb, Ob, Kb, vmax, R, Ch, eps_gpu, models=None, xs=None, Rh=None):
    H_, m, _ = W.shape; dev = W.device
    W = W.float()
    WQ, WK = (Qb + Ob, Kb) if models is None else (models[2], models[3])
    gc = list(eps_gpu) + ["home"]; E = len(gc); C = NA_all.shape[0]
    prep = [_parts(W, NA_all[c], owner, gid, gpn, NN, vmax, R, Ch, xs) for c in range(C)]
    st = lambda i: torch.stack([p[i] for p in prep])                    # noqa: E731
    Wg, rid, rh, kh, slot, order, nodew, idxc, rhd = (st(i) for i in range(9))
    NN = Wg.shape[1]                                                      # (the nodes solved here)
    G = C * E * NN
    rep = lambda X: X[:, None].expand(C, E, *X.shape[1:]).reshape(G, *X.shape[2:])   # noqa: E731
    epsv = torch.stack([torch.full((), float(max(eps_gpu) if e == "home" else e), device=dev, dtype=nodew.dtype) for e in gc])
    homeE = torch.stack([torch.full((), e == "home", device=dev, dtype=torch.bool) for e in gc])
    capu = ((1 + epsv)[None, :, None] * nodew[:, None, :] / gpn)
    cbu = capu[..., None].expand(C, E, NN, gpn).reshape(G, gpn)
    # head-batched clustering problems: [G, H, Rh, Ch] in place of [G, 1, R (all heads' rows), Ch]
    Hh = H_
    if Rh is None:                                                        # (eager without a bucket: the exact count)
        Rh = hb_rows(Wg.reshape(C * NN, 1, R, -1), rhd.reshape(C * NN, 1, R), Hh)
    hb = _hb_split(Wg.reshape(C * NN, 1, R, -1), rid.reshape(C * NN, 1, R), rh.reshape(C * NN, 1, R),
                   kh.reshape(C * NN, *kh.shape[2:]), rhd.reshape(C * NN, 1, R), Hh, max(Rh, 1))
    Wh, ridh, rhh, rhdh, khh, inv = (x.reshape(C, NN, *x.shape[1:]) for x in hb)
    Rh = Wh.shape[3]
    preset = torch.where(homeE[None, :, None, None, None], rhh[:, None], torch.full_like(rhh[:, None], -1)).expand(C, E, NN, Hh, Rh).reshape(G, Hh, Rh)
    state, consts = ap_init(rep(Wh), rep(rhh), rep(khh), gpn, cbu, WQ, WK, rep(ridh), cbu[:, :1, None].expand(G, Hh, Rh), preset=preset)
    return state, consts, (slot, order, Wg, rh, kh, idxc, rhd, inv)


def gpu_eval(W, owner, NA_all, arr_all, gpn, NN, Qb, Ob, Kb, TT, eps_gpu, state, consts, ctx, models=None, xs=None):
    """fill, then every (node candidate, GPU candidate, node) judged on that node's compact problem (a node's terms depend only on its
    own pairs); per node the best GPU candidate (lexpick). -> (ug, pick [C, Nx],
    T_loc [C] = the slowest of these nodes' best times, B_all [C] node-level bytes). A node candidate's time is the
    max of T_loc over all nodes: with xs (distributed planning) an all-reduce(max) across the node groups."""
    W = W.float(); own_node = owner // gpn
    WQ, WK = (Qb + Ob, Kb) if models is None else (models[2], models[3])
    slot, order, Wg, rh, kh, idxc, rhd, inv = ctx
    xsv = node_ids(xs, NN, W.device); Nx = xsv.shape[0]
    E = len(eps_gpu) + 1; C = NA_all.shape[0]; G = C * E * Nx
    rep = lambda X: X[:, None].expand(C, E, *X.shape[1:]).reshape(G, *X.shape[2:])   # noqa: E731
    ugh = ap_finish(state, consts, gpn, WQ, WK)                           # [G, H, Rh] (head-batched)
    invr = rep(inv)                                                       # [G, R] merged slot -> head-batched index
    ug = torch.where(invr >= 0, ugh.reshape(G, -1).gather(1, invr.clamp(min=0)), torch.zeros_like(invr))[:, None]   # [G, 1, R]
    Wr = rep(Wg)
    ugv = torch.where(Wr[:, 0].sum(2) > 0, ug[:, 0], torch.full_like(ug[:, 0], -1))   # [G, R] (-1: no part)
    rhl = rep(torch.where(rh >= 0, rh, torch.full_like(rh, gpn)))
    khl = rep(torch.where(kh >= 0, kh, torch.full_like(kh, gpn)))           # [G, 1, H, Ch]
    rhr = rep(rhd)
    arr = arr_all[:, xsv][:, None, :].expand(C, E, Nx).reshape(G)
    if models is None:
        F, Fc = gpu_node_c(Wr, ugv, rhr, rhl, khl, arr, gpn, Qb, Ob, Kb, TT)
    else:
        gt = models[1]
        vc = idxc >= 0
        kcc = rep((gt.kcols.reshape(-1).float()[idxc.clamp(min=0)] * vc).reshape(C, Nx, -1))   # [G, H Ch]
        nic = rep(gt.nic(NA_all, owner)[:, xsv])                          # [G]
        F, Fc = gt.node_c(Wr, ugv, rhr, rhl, khl, kcc, nic)
    Fall = F.reshape(C, E, Nx)
    pick = lexpick(Fall, Fc.reshape(C, E, Nx), 1)                         # [C, Nx]
    T_loc = Fall.min(1).values.max(1).values
    B_all = node_bytes_b(W, NA_all, own_node, NN, WQ, WK).double()
    return ug, pick, T_loc, B_all


def pick_best(T_all, B_all):
    """the node candidate with the least time, ties by bytes"""
    return torch.where(T_all == T_all.min(), B_all, torch.full_like(B_all, float("inf"))).argmin()


def gpu_final(W, NA_all, gpn, NN, ctx, ug, pick, best, xs=None):
    """the GPU of every pair of these nodes (others -1) for the picked node candidate best"""
    H_, m, _ = W.shape; dev = W.device; nr = H_ * m; live = W > 0
    slot, order, Wg, rh, kh, idxc, rhd = ctx[:7]
    xsv = node_ids(xs, NN, dev); Nx = xsv.shape[0]; C = NA_all.shape[0]; R = Wg.shape[-2]
    E = ug.shape[0] // (C * Nx)
    b1 = best.reshape(1)
    ugb = ug.reshape(C, E, Nx, R).index_select(0, b1)[0]                  # [E, Nx, R]
    pk = pick.index_select(0, b1)[0]                                       # [Nx]
    gsel = ugb.gather(0, pk[None, :, None].expand(1, Nx, R))[0] + xsv[:, None] * gpn
    sl = slot.index_select(0, b1)[0]; od = order.index_select(0, b1)[0]
    gpu_of_part = torch.cat([gsel.reshape(-1), gsel.new_zeros(1)]).gather(0, sl)
    pg = torch.zeros(nr * NN, dtype=torch.long, device=dev).scatter_(0, od, gpu_of_part).reshape(H_, m, NN)
    NAb = NA_all.index_select(0, b1)[0]
    inx = torch.zeros(NN, dtype=torch.bool, device=dev).index_fill_(0, xsv, True)
    mine = live & inx[NAb.clamp(min=0)]
    return torch.where(mine, pg.gather(2, NAb.clamp(min=0)), torch.full_like(NAb, -1))


def gpu_post(W, owner, NA_all, arr_all, gpn, NN, Qb, Ob, Kb, TT, eps_gpu, state, consts, ctx, models=None):
    """centralized GPU level end: gpu_eval for all nodes, the best node candidate, gpu_final -> (plan, (T, B, best))"""
    ug, pick, T_all, B_all = gpu_eval(W, owner, NA_all, arr_all, gpn, NN, Qb, Ob, Kb, TT, eps_gpu, state, consts, ctx, models)
    best = pick_best(T_all, B_all)
    return gpu_final(W, NA_all, gpn, NN, ctx, ug, pick, best), (T_all, B_all, best)
