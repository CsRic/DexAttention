"""Reassign whole KV columns within each candidate group pair after Q-row placement.
The same KV column can still be assigned to different destinations in other groups.

kv_init groups rows connected by shared KV on the same destination; a
destination can hold multiple groups. kv_round shortlists pairs by shared KV,
constructs one quota-based split per pair, and scores communication saved.
Pairs must use disjoint destination sets and fit the destination-count and work caps.
Each destination votes for its best positive-gain candidate; accept a pair only
when all its destinations agree. Commit the selected splits together only if
the cost model improves by min_gain. The caller increases the limit on destinations per merged group.

The gkv_* functions apply the same steps to compact problems inside each node,
scored by its local model.
"""
import torch

from .ops import oh, components_b, col_counts_b, rows_any_b, cols_any_b, unit_sums_b
from .models import gpu_local_c

GPU_KV_ROUNDS = 4                          # KV rounds per level at the GPU level (the node level runs 1)


def kv_init(W, home_unit, unit_row, gid, U, cap, model):
    """the KV re-split's set-up as a state: (sid, UM, colunit, alive, T_cur, Tc_cur, active) + constants. W [H, m, m]
    shared by the candidates or [C, H, m, m] per candidate (the GPU level's compact per-node problems)"""
    C = unit_row.shape[0]; H_, m, mc = W.shape[-3], W.shape[-2], W.shape[-1]; dev = W.device   # (rows m, columns mc)
    A = (W > 0).float(); live = W > 0
    sid = components_b(A, unit_row)
    alive = torch.zeros((C, H_, m), device=dev).scatter_add_(2, sid, torch.ones((C, H_, m), device=dev)) > 0
    UM = oh(unit_row, U).bool()
    colunit = torch.where(live, unit_row[..., None].expand(C, H_, m, mc), torch.full((C, H_, m, mc), -1, dtype=torch.long, device=dev))
    colunit = torch.full((C, H_, m, mc), -1, dtype=torch.long, device=dev).scatter_reduce(
        2, sid[..., None].expand(C, H_, m, mc), colunit, "amax", include_self=True)
    NA = torch.where(live, colunit.gather(2, sid[..., None].expand(C, H_, m, mc)), torch.full_like(colunit, -1))
    T_cur, Tc_cur = model(NA)
    active = T_cur > Tc_cur * (1 + 1e-6)
    return (sid, UM, colunit, alive, T_cur, Tc_cur, active)


def kv_plan(W, state):
    sid, colunit = state[0], state[2]
    C = sid.shape[0]; H_, m, mc = W.shape[-3], W.shape[-2], W.shape[-1]
    return torch.where(W > 0, colunit.gather(2, sid[..., None].expand(C, H_, m, mc)), torch.full_like(colunit, -1))


def kv_round(W, home_unit, gid, U, cap, WQ, WK, model, state, K=64, sub_rounds=3, ncap=None, min_gain=0.0):
    """one KV re-split round: candidate set pairs (top K by shared KV), each fusion evaluated on its own (bytes
    gained: communication before minus after, unit caps), the best non-conflicting ones accepted (sub-rounds), then the model check of the
    round. ncap: at most ncap units per fused group (level n). Returns (state, any fusion kept).
    Square W [.., m, m] (node level: home_unit / gid per block) or, per candidate, rectangular [C, H, m rows, mc
    columns] (the GPU level's compact problems): home_unit = (row homes [C, H, m], column homes [C, H, mc]), gid =
    the columns' global ids (tie parity)."""
    sid, UM, colunit, alive, T_cur, Tc_cur, active = state
    C = sid.shape[0]; H_, m, mc = W.shape[-3], W.shape[-2], W.shape[-1]; dev = W.device
    idx = torch.arange(m, device=dev); idc = torch.arange(mc, device=dev)
    A = (W > 0).float(); Ah = A.half(); Wf = W.float(); live = W > 0
    home_r, home_c = home_unit if isinstance(home_unit, tuple) else (home_unit, home_unit)
    hm = oh(home_r, U).bool(); hmc = hm if home_c is home_r else oh(home_c, U).bool()
    even = (gid % 2 == 0)
    ci = torch.arange(C, device=dev)[:, None]
    hx = (lambda X: X[ci, hK]) if W.dim() == 4 else (lambda X: X[hK])   # noqa: E731  (per-candidate W, home, gid)
    cap = cap.reshape(C, -1)               # one cap per candidate, or one per unit
    plan = lambda: torch.where(live, colunit.gather(2, sid[..., None].expand(C, H_, m, mc)), torch.full_like(colunit, -1))   # noqa: E731
    if model is not None:
        active = active & (T_cur > Tc_cur * (1 + 1e-6))
        saved = (sid.clone(), UM.clone(), colunit.clone(), alive.clone())
    sx = sid[..., None].expand(C, H_, m, mc)
    CNS = torch.zeros((C, H_, m, mc), device=dev).scatter_add_(2, sx, A.expand(C, H_, m, mc))
    CWS = torch.zeros((C, H_, m, mc), device=dev).scatter_add_(2, sx, Wf.expand(C, H_, m, mc))
    CMS = (CNS > 0).half()
    NA = torch.where(live, colunit.gather(2, sx), torch.full_like(colunit, -1))
    UC = col_counts_b(NA, U).reshape(C, H_, mc, U)
    RN = rows_any_b(NA, U).reshape(C, H_, m, U)
    load = unit_sums_b(Wf, NA, U)                                     # [C, U]
    qb = ((RN | hm).sum(-1) - 1).clamp(min=0).float()                  # [C, H, m]
    sh = CMS @ CMS.transpose(-1, -2)
    UMh = UM.half()
    disj = (UMh @ UMh.transpose(-1, -2)) == 0
    ok = alive[..., :, None] & alive[..., None, :] & disj & (idx[:, None] < idx[None, :]) & (sh > 0)
    if ncap is not None:                                                  # level n: at most ncap units per group
        nu = UM.sum(-1)
        ok = ok & ((nu[..., :, None] + nu[..., None, :]) <= ncap)
    shv, flat = torch.where(ok, sh, torch.full_like(sh, -1.0)).reshape(C, -1).topk(K, dim=1)
    hK = flat // (m * m); s = (flat // m) % m; t = flat % m; valid = shv > 0     # [C, K]
    V = UM[ci, hK, s] | UM[ci, hK, t]                                 # [C, K, U]
    cs, ct, rs, rt = CNS[ci, hK, s], CNS[ci, hK, t], CWS[ci, hK, s], CWS[ci, hK, t]   # [C, K, m]
    xs, xt = colunit[ci, hK, s], colunit[ci, hK, t]
    Cm = (cs > 0) | (ct > 0)
    hk = hx(home_c)                                                   # [C, K, mc]
    hin = V.gather(2, hk)
    both = (cs > 0) & (ct > 0)
    keep = torch.where(both, torch.where((rs > rt) | ((rs == rt) & hx(even)), xs, xt), torch.where(cs > 0, xs, xt))
    pref = torch.where(hin, hk, keep)
    oxs = oh(xs.clamp(min=0), U).float() * (xs >= 0)[..., None]; oxt = oh(xt.clamp(min=0), U).float() * (xt >= 0)[..., None]
    wcol = (rs + rt) * Cm
    wold = ((rs[..., None] * oxs) + (rt[..., None] * oxt)).sum(2)     # [C, K, U]
    base = load[:, None] - wold
    Vf = V.float(); tot = wcol.sum(2)
    lvl = ((base * Vf).sum(2) + tot) / Vf.sum(2).clamp(min=1)
    quota = (lvl[..., None] - base).clamp(min=0) * Vf
    quota = quota * (tot / quota.sum(2).clamp(min=1e-9))[..., None]
    cq = quota.cumsum(2)
    key = torch.where(Cm, pref * (2 * mc) + (~hin).long() * mc + idc, torch.full_like(pref, 4 * mc * U))
    order = key.argsort(2)
    ws = wcol.gather(2, order)
    pos = torch.minimum(ws.cumsum(2) - ws / 2, (tot * (1 - 1e-6))[..., None])
    us = torch.searchsorted(cq.contiguous(), pos.contiguous(), right=True).clamp(max=U - 1)
    fill = torch.empty_like(us).scatter_(2, order, us)
    newcol = torch.where(Cm, fill, torch.full_like(keep, -1))
    onew = oh(newcol.clamp(min=0), U).bool() & (newcol >= 0)[..., None]   # [C, K, m, U]
    UCk = UC[ci, hK]                                                  # [C, K, m, U]
    rest = UCk - cs[..., None] * oxs - ct[..., None] * oxt
    hmK = hx(hm); hmcK = hx(hmc)                                      # (row / column homes)
    kvb = ((UCk > 0) | hmcK).sum(-1) - 1
    kva = ((rest > 0) | onew | hmcK).sum(-1) - 1
    dKV = WK * (kva.clamp(min=0) - kvb.clamp(min=0)).float().sum(2)
    sidK = sid[ci, hK]
    mem = (sidK == s[..., None]) | (sidK == t[..., None])
    if W.dim() == 4 and H_ == 1:           # per-node problems: one product over all K pairs (no [C, K, m, mc] copy of A)
        need = (Ah[:, 0] @ onew.half().permute(0, 2, 1, 3).reshape(C, mc, K * U)).reshape(C, m, K, U).permute(0, 2, 1, 3) > 0
    else:
        need = (hx(Ah) @ onew.half()) > 0
    qa = ((need | hmK).sum(-1) - 1).clamp(min=0).float()
    dQ = WQ * ((qa - qb[ci, hK]) * mem.float()).sum(2)
    wnew = ((rs + rt)[..., None] * onew.float()).sum(2)
    la = load[:, None] - wold + wnew
    feas = valid & ((la <= cap[:, None, :]) | ~V).all(2)
    # dQ and dKV are after-minus-before costs; negate their sum for savings.
    gain = torch.where(feas, -(dQ + dKV), torch.full_like(dQ, -float("inf"))).double()
    gain = gain - torch.arange(K, device=dev).double() * 1e-3
    used = torch.zeros((C, U), dtype=torch.bool, device=dev); acc = torch.zeros((C, K), dtype=torch.bool, device=dev)
    for _ in range(sub_rounds):
        g_ = torch.where((V & used[:, None]).any(2), torch.full_like(gain, -float("inf")), gain)
        best = torch.where(V, g_[..., None], torch.full_like(g_[..., None].expand(C, K, U), -float("inf"))).max(1).values
        take = (g_ > 0) & torch.where(V, best[:, None] == g_[..., None], torch.ones_like(V)).all(2)
        acc = acc | take
        used = used | (V & take[..., None]).any(1)
    if model is not None:
        acc = acc & active[:, None]
    kk = torch.arange(K, device=dev).expand(C, K); nf = H_ * m
    fs, ft = hK * m + s, hK * m + t
    ks = torch.full((C, nf), -1, dtype=torch.long, device=dev).scatter_reduce(
        1, torch.where(acc, fs, torch.full_like(fs, nf - 1)), torch.where(acc, kk, torch.full_like(kk, -1)), "amax").reshape(C, H_, m)
    kt = torch.full((C, nf), -1, dtype=torch.long, device=dev).scatter_reduce(
        1, torch.where(acc, ft, torch.full_like(ft, nf - 1)), torch.where(acc, kk, torch.full_like(kk, -1)), "amax").reshape(C, H_, m)
    kts = kt.gather(2, sid)
    sid = torch.where(kts >= 0, s.gather(1, kts.clamp(min=0).reshape(C, -1)).reshape(C, H_, m), sid)
    ksf = ks.clamp(min=0).reshape(C, -1)
    UM = torch.where(ks[..., None] >= 0, V.gather(1, ksf[..., None].expand(C, nf, U)).reshape(C, H_, m, U), UM)
    colunit = torch.where(ks[..., None] >= 0, newcol.gather(1, ksf[..., None].expand(C, nf, mc)).reshape(C, H_, m, mc), colunit)
    alive = alive & ~(kt >= 0)
    if model is not None:
        T_new, Tc_new = model(plan())
        ok_ = T_new <= T_cur * (1 - min_gain)   # (min_gain: the round must gain that share of the model time)
        o3, o4 = ok_[:, None, None], ok_[:, None, None, None]
        sid = torch.where(o3, sid, saved[0]); UM = torch.where(o4, UM, saved[1])
        colunit = torch.where(o4, colunit, saved[2]); alive = torch.where(o3, alive, saved[3])
        active = active & ok_
        T_cur = torch.where(ok_, T_new, T_cur); Tc_cur = torch.where(ok_, Tc_new, Tc_cur)
    kept = (acc.any(1) & ok_) if model is not None else acc.any(1)
    return (sid, UM, colunit, alive, T_cur, Tc_cur, active), kept


def gkv_prep(W, dp, owner, gid, gpn, NN, mr, mc, eps=0.1, kcols=None, xs=None):
    """GPU-level set-up on the GPU plan dp [H, m, m], as compact per-node problems (a dense form would hold NN
    candidates x NN heads x m x m elements). Candidate x = node x
    keeps only its own pairs: its row blocks (a pair on node x; <= mr) x the column blocks they touch (<= mc), all
    heads in ONE problem (heads share no columns: block-diagonal, exact; per-head groups would pad every head to the
    largest one), both compacted in (head, block) order (idx_r / idx_c), units = the node's gpn GPUs + one slot
    "owned outside the node" (gpn). A node's local model depends only on its own pairs, so the compact problems are
    exact. Cap per GPU = max(its node's max GPU work, (1 + eps) x its node's mean).
    -> (Wc, homes (row, column), gid of the columns, capc, kcc, (idx_r, idx_c), urow)"""
    H_, m, _ = W.shape; dev = W.device; P = NN * gpn; n = H_ * m
    W = W.float(); live = W > 0
    nd = torch.where(live, dp // gpn, torch.full_like(dp, -1))
    from .levels import node_ids
    xsv = node_ids(xs, NN, dev); Nx = xsv.shape[0]                        # (xs: only these nodes; None = all)
    idx_r, val_r = compact_idx(rows_any_b(nd[None], NN).reshape(n, NN).t()[xsv], mr)   # (h, row) blocks of node x
    idx_c, val_c = compact_idx(cols_any_b(nd[None], NN).reshape(n, NN).t()[xsv], mc)   # (h, column) blocks
    hr_, hc_ = idx_r // m, idx_c // m
    same = (hr_[:, :, None] == hc_[:, None, :]) & val_r[:, :, None] & val_c[:, None, :]   # [NN, mr, mc] (heads disjoint)
    fl = (idx_r[:, :, None] * m + (idx_c % m)[:, None, :]).clamp(max=n * m - 1)            # into [H, m, m]
    cg = lambda X: X.reshape(-1)[fl][:, None]                             # noqa: E731  [H, m, m] -> [NN, 1, mr, mc]
    xs = xsv[:, None, None]
    onx = same[:, None] & (cg(nd) == xs[..., None])
    Wc = torch.where(onx, cg(W), torch.zeros((), device=dev))
    dpl = torch.where(onx, cg(dp) - xs[..., None] * gpn, torch.full((), -1, dtype=torch.long, device=dev))
    loc = lambda o: torch.where(o // gpn == xs[:, 0], o - xs[:, 0] * gpn, torch.full_like(o, gpn))   # noqa: E731
    of = owner.reshape(-1)
    hr, hc = loc(of[idx_r])[:, None], loc(of[idx_c])[:, None]             # [NN, 1, mr] / [NN, 1, mc]
    gidc = torch.where(val_c, gid.reshape(-1)[idx_c], torch.full_like(idx_c, -1))[:, None]
    urow = dpl.amax(-1)                                                   # [NN, 1, mr] local GPU of each row part
    urow = torch.where(urow >= 0, urow, hr)
    ld = unit_sums_b(W, dp[None], P)[0]                                   # [P] work per GPU
    nl = ld.reshape(NN, gpn)[xsv]
    cap = torch.maximum(nl.max(1, keepdim=True).values, (1 + eps) * nl.mean(1, keepdim=True)).expand(Nx, gpn)
    capc = torch.cat([cap, torch.full((Nx, 1), float("inf"), device=dev)], 1)
    if kcols is None:
        kcc = torch.zeros((Nx, mc), device=dev)
    else:
        kcc = torch.where(val_c, kcols.reshape(-1)[idx_c].float(), torch.zeros((), device=dev))
    return Wc, (hr, hc), gidc, capc, kcc, (idx_r, idx_c), urow


def gkv_prep_hb(W, dp, owner, gid, gpn, NN, mr, mc, eps=0.1, kcols=None, xs=None):
    """gkv_prep in head-batched form: node x's problem as [H, mr, mc] -- per head its own row blocks
    (<= mr) x the column blocks they touch (<= mc) -- in place of one [all heads' rows x all heads' columns] block-
    diagonal matrix. The candidates' K best pairs, the unit loads / caps and the local model still span all heads of
    the node (kv_round handles a head axis); a pair never spans two heads in either form."""
    H_, m, _ = W.shape; dev = W.device; P = NN * gpn; n = H_ * m
    W = W.float(); live = W > 0
    nd = torch.where(live, dp // gpn, torch.full_like(dp, -1))
    from .levels import node_ids
    xsv = node_ids(xs, NN, dev); Nx = xsv.shape[0]
    rin = rows_any_b(nd[None], NN).reshape(H_, m, NN).permute(2, 0, 1)[xsv].reshape(Nx * H_, m)
    cin = cols_any_b(nd[None], NN).reshape(H_, m, NN).permute(2, 0, 1)[xsv].reshape(Nx * H_, m)
    ir, vr = compact_idx(rin, mr); ic, vc = compact_idx(cin, mc)           # positions inside the head
    hh = (torch.arange(Nx * H_, device=dev) % H_)[:, None]
    idx_r = (ir + hh * m).reshape(Nx, H_, mr); idx_c = (ic + hh * m).reshape(Nx, H_, mc)   # -> block ids
    val_r, val_c = vr.reshape(Nx, H_, mr), vc.reshape(Nx, H_, mc)
    fl = (idx_r[..., :, None] * m + (idx_c % m)[..., None, :]).clamp(max=n * m - 1)        # into [H, m, m]
    cg = lambda X: X.reshape(-1)[fl]                                      # noqa: E731  [H, m, m] -> [Nx, H, mr, mc]
    xs_ = xsv[:, None, None, None]
    onx = val_r[..., :, None] & val_c[..., None, :] & (cg(nd) == xs_)
    Wc = torch.where(onx, cg(W), torch.zeros((), device=dev))
    dpl = torch.where(onx, cg(dp) - xs_ * gpn, torch.full((), -1, dtype=torch.long, device=dev))
    x3 = xsv[:, None, None]
    loc = lambda o: torch.where(o // gpn == x3, o - x3 * gpn, torch.full_like(o, gpn))   # noqa: E731
    of = owner.reshape(-1)
    hr, hc = loc(of[idx_r]), loc(of[idx_c])                               # [Nx, H, mr] / [Nx, H, mc]
    gidc = torch.where(val_c, gid.reshape(-1)[idx_c], torch.full_like(idx_c, -1))
    urow = dpl.amax(-1)
    urow = torch.where(urow >= 0, urow, hr)
    ld = unit_sums_b(W, dp[None], P)[0]
    nl = ld.reshape(NN, gpn)[xsv]
    cap = torch.maximum(nl.max(1, keepdim=True).values, (1 + eps) * nl.mean(1, keepdim=True)).expand(Nx, gpn)
    capc = torch.cat([cap, torch.full((Nx, 1), float("inf"), device=dev)], 1)
    if kcols is None:
        kcc = torch.zeros((Nx, H_ * mc), device=dev)
    else:
        kcc = torch.where(val_c, kcols.reshape(-1)[idx_c].float(), torch.zeros((), device=dev)).reshape(Nx, H_ * mc)
    return Wc, (hr, hc), gidc, capc, kcc, (idx_r, idx_c), urow


def gkv_finish_hb(W, Wc, st, idx, gpn, NN, xs=None):
    """gkv_finish for gkv_prep_hb's head-batched problems"""
    H_, m, _ = W.shape; live = W > 0; dev = W.device; idx_r, idx_c = idx
    pl_ = kv_plan(Wc, st)                                                 # [Nx, H, mr, mc]
    from .levels import node_ids
    xs = node_ids(xs, NN, dev)[:, None, None, None]
    g = torch.where(pl_ >= 0, pl_ + xs * gpn, torch.full_like(pl_, -1))
    flat = idx_r[..., :, None] * m + (idx_c % m)[..., None, :]
    flat = torch.where(g >= 0, flat, torch.full_like(flat, H_ * m * m))
    out = torch.full((H_ * m * m + 1,), -1, dtype=torch.long, device=dev).scatter_(0, flat.reshape(-1), g.reshape(-1))
    return torch.where(live, out[:-1].reshape(H_, m, m), torch.full_like(out[:-1].reshape(H_, m, m), -1))


def node_slots_hb(dp, gpn, NN):
    """the largest numbers of row blocks / column blocks of one (node, head) with a pair on that node"""
    H_, m, _ = dp.shape
    nd = torch.where(dp >= 0, dp // gpn, torch.full_like(dp, -1))
    ra = rows_any_b(nd[None], NN)[0].reshape(H_, m, NN).sum(1); ca = cols_any_b(nd[None], NN)[0].reshape(H_, m, NN).sum(1)
    return int(ra.max()), int(ca.max())


def compact_idx(inS, mx):
    """inS [NN, n] -> the first mx set positions of every row in index order (idx [NN, mx], val = a real one)"""
    NN, n = inS.shape; dev = inS.device
    pos = inS.long().cumsum(1) - 1
    slot = torch.where(inS & (pos < mx), pos, torch.full_like(pos, mx))
    idx = torch.zeros((NN, mx + 1), dtype=torch.long, device=dev).scatter_(1, slot, torch.arange(n, device=dev).expand(NN, n))[:, :mx]
    val = torch.zeros((NN, mx + 1), dtype=torch.bool, device=dev).scatter_(1, slot, inS)[:, :mx]
    return idx, val


def gkv_init(Wc, homes, gidc, capc, kcc, urow, gpn, gtc):
    """the KV re-split state of the compact problems (gtc = the compact local model)"""
    return kv_init(Wc, homes[0], urow, gidc, gpn + 1, capc, lambda NA: gtc(Wc, NA, homes, kcc))


def gkv_round(Wc, homes, gidc, capc, kcc, st, n, gpn, gtc, WQ, WK, K=64, sub_rounds=3, min_gain=0.0):
    """one GPU-level KV round at level n -> (state, any fusion kept, any candidate still active). A round that keeps
    nothing leaves the state unchanged (or switches the candidate off), so the level's later rounds would repeat it:
    the caller stops the level there and stops everything once no candidate is active (exact early exit)."""
    st, kept = kv_round(Wc, homes, gidc, gpn + 1, capc, WQ, WK, lambda NA: gtc(Wc, NA, homes, kcc), st, K, sub_rounds,
                        ncap=n, min_gain=min_gain)
    return st, kept.any(), st[6].any()


def gkv_finish(W, Wc, st, idx, gpn, NN, xs=None):
    """node x's part of the plan from candidate x (local GPU ids, compact indices) -> GPU plan [H, m, m] (with xs: only
    these nodes' pairs, the rest -1)"""
    H_, m, _ = W.shape; live = W > 0; dev = W.device; idx_r, idx_c = idx
    pl_ = kv_plan(Wc, st)[:, 0]                                           # [Nx, mr, mc]
    from .levels import node_ids
    xs = node_ids(xs, NN, dev)[:, None, None]
    g = torch.where(pl_ >= 0, pl_ + xs * gpn, torch.full_like(pl_, -1))
    flat = idx_r[:, :, None] * m + (idx_c % m)[:, None, :]                # (h, row) * m + column (same head)
    flat = torch.where(g >= 0, flat, torch.full_like(flat, H_ * m * m))   # (pairs of different nodes are disjoint)
    out = torch.full((H_ * m * m + 1,), -1, dtype=torch.long, device=dev).scatter_(0, flat.reshape(-1), g.reshape(-1))
    return torch.where(live, out[:-1].reshape(H_, m, m), torch.full_like(out[:-1].reshape(H_, m, m), -1))


def gkv_models(gpn, NN, Qb, Ob, Kb, TT, models):
    """(compact local model gtc(Wc, dp, homel, kcc), WQ, WK, kcols or None): forward = gpu_local_c, backward =
    bwd_models_b(...)[1].local_c"""
    if models is None:
        gt = lambda Wc, dp_, hl_, kc_: gpu_local_c(Wc, dp_, hl_, gpn, Qb, Ob, Kb, TT)   # noqa: E731
        return gt, Qb + Ob, Kb, None
    return models[1].local_c, models[2], models[3], models[1].kcols


def node_slots(dp, gpn, NN):
    """the largest numbers of row blocks / column blocks (all heads) with a pair on one node: the compact problems'
    size (mr, mc)"""
    nd = torch.where(dp >= 0, dp // gpn, torch.full_like(dp, -1))
    return int(rows_any_b(nd[None], NN)[0].sum(0).max()), int(cols_any_b(nd[None], NN)[0].sum(0).max())


def mx_bucket(s, m):
    """static per-node problem size (few graph captures): >= 8 (topk K = 64 candidate pairs need mx^2 >= 64), steps
    x1.25 in multiples of 8, at most max(m, 8)"""
    b = 8
    while b < s:
        b = max(b + 8, -(-(b * 5 // 4) // 8) * 8)
    return min(b, max(m, 8))


def gpu_kv_split(W, dp, owner, gid, gpn, NN, arr_all, best, Qb, Ob, Kb, TT, models=None, min_gain=0.01, kv_cost=0.0):
    """eager GPU-level KV re-split of the picked plan dp (best / arr_all: unused, the local model has no arrivals):
    levels n = 2..gpn, GPU_KV_ROUNDS rounds each; a level that does not lower a candidate's time by more than kv_cost
    switches it off. Sync-free apart from the compact size (the eager form of Planner's per-round graphs)."""
    gtc, WQ, WK, kc = gkv_models(gpn, NN, Qb, Ob, Kb, TT, models)
    sr, sc = node_slots_hb(dp, gpn, NN)                                   # (head-batched)
    mr, mc = mx_bucket(sr, W.shape[1]), mx_bucket(sc, W.shape[1])
    Wc, hl, gc, capc, kcc, idx, urow = gkv_prep_hb(W, dp, owner, gid, gpn, NN, mr, mc, kcols=kc)
    st = gkv_init(Wc, hl, gc, capc, kcc, urow, gpn, gtc)
    for n in range(2, gpn + 1):
        T0 = st[4]
        for _ in range(GPU_KV_ROUNDS):
            st, _, _ = gkv_round(Wc, hl, gc, capc, kcc, st, n, gpn, gtc, WQ, WK, min_gain=min_gain)
        st = st[:6] + (st[6] & (T0 - st[4] > kv_cost),)
    return gkv_finish_hb(W, Wc, st, idx, gpn, NN)


def gkv_step_prep(W, out, owner, gid, gpn, NN, mr, mc, Qb, Ob, Kb, TT, models=None, xs=None):
    """Planner graph: set-up -> (Wc, homes, gidc, capc, kcc, idx, state)"""
    gtc, _, _, kc = gkv_models(gpn, NN, Qb, Ob, Kb, TT, models)
    Wc, hl, gc, capc, kcc, idx, urow = gkv_prep_hb(W, out, owner, gid, gpn, NN, mr, mc, kcols=kc, xs=xs)   # (head-batched)
    return Wc, hl, gc, capc, kcc, idx, gkv_init(Wc, hl, gc, capc, kcc, urow, gpn, gtc)


def gkv_step_round(Wc, hl, gc, capc, kcc, st, n, gpn, NN, Qb, Ob, Kb, TT, models=None, min_gain=0.01):
    """Planner graph: one KV round at level n -> (state, kept flag, active flag)"""
    gtc, WQ, WK, _ = gkv_models(gpn, NN, Qb, Ob, Kb, TT, models)
    return gkv_round(Wc, hl, gc, capc, kcc, st, n, gpn, gtc, WQ, WK, min_gain=min_gain)
