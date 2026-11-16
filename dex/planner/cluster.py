"""Cluster Q rows and place them on destinations (nodes or GPUs).

Each round computes shared-KV merge similarity and destination affinity once.
Mutually preferred pairs merge; their clusters remain unplaced until the next
round. Other clusters request destinations. Per-destination sorting and work
prefix sums accept only requests that fit. Bounded matching subrounds exclude
already paired clusters without recomputing scores.

Finish retries placement, splits unplaced clusters into original rows, retries
capacity-checked placement twice, then fills load gaps with whole rows. Only
the last step can exceed the placement cap.
Tensor state: CM = cluster KV membership, cw = work, QA = Q-home benefit,
UCM = destination reusable KV, lab = row-to-cluster map, alive = active cluster,
unit = assigned destination, load = destination work.
"""
import torch

oh = torch.nn.functional.one_hot


def _aff(CM, UCM, rhead, U):
    """cluster-unit KV overlap [G, Hn, n, U]. Head-local columns (rhead [G, Hn, n] = each row's head; the
    GPU level's per-node problems index a row's KV columns inside its own head): UCM [G, Hn, Hh U, Ch], the row's
    head picked"""
    X = CM @ UCM.transpose(2, 3)
    if rhead is None:
        return X
    return X.gather(3, rhead.clamp(min=0)[..., None] * U + torch.arange(U, device=X.device))


def _sim(CM, rhead):
    """cluster-cluster shared KV columns (head-local columns: only rows of one head share columns)"""
    S = CM @ CM.transpose(2, 3)
    return S if rhead is None else S * (rhead[..., :, None] == rhead[..., None, :]).half()


def _ucm_add(UCM, sel, CM, rhead, U):
    """the units take the KV columns of the clusters sel [G, Hn, n, U] accepted"""
    if rhead is not None:
        Hh = UCM.shape[2] // U
        sel = (oh(rhead.clamp(min=0), Hh).half()[..., :, None] * sel[..., None, :]).reshape(*sel.shape[:3], Hh * U)
    return torch.maximum(UCM, (sel.transpose(2, 3) @ CM > 0).half())


def _accept(prop, uA, bA, unit, load, UCM, cw, CM, capu, U, rhead=None):
    """proposals accepted per group in (unit, -score) order while the unit's load stays under its cap. Returns unit,
    load, UCM."""
    G, Hn, n = prop.shape; dev = prop.device
    flat = lambda X: X.reshape(G, Hn * n)                                # noqa: E731
    fp, fu, fb, fc = flat(prop), flat(uA), flat(bA), flat(cw)
    key = torch.where(fp, fu.double() * 1e13 - fb.double(), torch.full((G, Hn * n), float("inf"), device=dev, dtype=torch.float64))
    order = torch.argsort(key, dim=1, stable=True)
    ps = fp.gather(1, order); us = fu.gather(1, order).clamp(min=0)
    ws = torch.where(ps, fc.gather(1, order), torch.zeros_like(fc))
    tot = torch.zeros((G, U), device=dev).scatter_add(1, us, ws)
    off = torch.cumsum(tot, 1) - tot
    segcs = torch.cumsum(ws, 1) - off.gather(1, us)
    acc_s = ps & (load.gather(1, us) + segcs <= capu.gather(1, us))
    acc = torch.zeros((G, Hn * n), dtype=torch.bool, device=dev).scatter(1, order, acc_s).reshape(G, Hn, n)
    unit = torch.where(acc, uA, unit)
    load = load + torch.zeros((G, U), device=dev).scatter_add(1, fu.clamp(min=0), flat(torch.where(acc, cw, torch.zeros_like(cw))))
    sel = oh(uA.clamp(min=0), U).half() * acc[..., None].half()
    UCM = _ucm_add(UCM, sel, CM, rhead, U)
    return unit, load, UCM, acc.any()


def ap_init(W, row_home, kv_home, U, capu, WQ, WK, rid, row_cap, preset=None, rhead=None):
    """-> (state, consts). rhead [G, Hn, n] (optional): head-local KV columns -- W [G, Hn, n, Ch] indexes a row's
    columns inside its head, kv_home [G, Hn, Hh, Ch]"""
    G, Hn, n, nb = W.shape; dev = W.device
    valid = rid >= 0
    idx = torch.arange(n, device=dev).expand(G, Hn, n)
    CM = (W > 0).half()
    cw = W.float().sum(3)
    CM0, cw0 = CM.clone(), cw.clone()
    QA = oh(row_home.clamp(min=0), U).float() * (row_home >= 0).float()[..., None] * WQ
    if rhead is None:
        UCM = (oh(kv_home.clamp(min=0), U).half() * (kv_home >= 0).half()[..., None]).transpose(2, 3).contiguous()
    else:
        Hh, Ch = kv_home.shape[-2], kv_home.shape[-1]
        UCM = (oh(kv_home.clamp(min=0), U).half() * (kv_home >= 0).half()[..., None]).transpose(-1, -2).reshape(G, Hn, Hh * U, Ch).contiguous()
    QA0 = QA.clone()
    lab = idx.clone(); alive = valid.clone()
    unit = torch.full((G, Hn, n), -1, dtype=torch.long, device=dev)
    load = torch.zeros((G, U), device=dev)
    if preset is not None:
        fx = (preset >= 0) & valid
        unit = torch.where(fx, preset, unit)
        load = load.scatter_add(1, preset.clamp(min=0).reshape(G, -1), torch.where(fx, cw, torch.zeros_like(cw)).reshape(G, -1))
        sel0 = oh(preset.clamp(min=0), U).half() * fx[..., None].half()
        UCM = _ucm_add(UCM, sel0, CM, rhead, U)
    ri = rid.clamp(min=0)
    hsh = ((ri[..., :, None] * 7919 + ri[..., None, :] * 7919 + (ri[..., :, None] * ri[..., None, :]) * 104729) % 1000003).float() / 1000003.0
    tie = WK * 1e-3 * hsh
    state = (CM, cw, QA, UCM, lab, alive, unit, load)
    consts = (valid, idx, tie, row_cap, capu, CM0, cw0, QA0, row_home, kv_home, rhead)
    return state, consts


def ap_round(state, consts, U, WQ, WK, sub_rounds=4):
    """one clustering round -> (next state, changed flag)"""
    CM, cw, QA, UCM, lab, alive, unit, load = state
    valid, idx, tie, capr, capu, CM0, cw0, QA0 = consts[:8]
    rhead = consts[10] if len(consts) > 10 else None
    G, Hn, n = cw.shape; dev = cw.device
    eye = torch.eye(n, dtype=torch.bool, device=dev)[None, None]
    neg = torch.full((), -1.0, device=dev)
    gath = lambda X, j: X.gather(2, j[..., None].expand(-1, -1, -1, X.shape[3]))   # noqa: E731
    capg = capu[:, None, None, :]
    free = alive & (unit < 0)
    # Placement affinity: Q-home savings plus reusable KV, subject to load caps.
    SA = QA + WK * _aff(CM, UCM, rhead, U).float()
    SA = torch.where(free[..., None] & (load[:, None, None, :] + cw[..., None] <= capg), SA, neg)
    bA, uA = SA.max(3)
    SC0 = WK * _sim(CM, rhead).float()
    joint = cw[..., :, None] + cw[..., None, :]
    SC0 = torch.where(SC0 > 0, SC0 + tie + WK * 1e-3 * (1.0 - joint / capr[..., :, None]), SC0)
    okC0 = free[..., :, None] & free[..., None, :] & (joint <= capr[..., :, None]) & ~eye & (SC0 > 0)
    best_c = torch.where(okC0, SC0, neg).max(3).values
    # Select disjoint mutual pairs against this round's fixed score matrices.
    mate = torch.full((G, Hn, n), -1, dtype=torch.long, device=dev)
    for _ in range(sub_rounds):
        fr = free & (mate < 0)
        okC = fr[..., :, None] & fr[..., None, :] & (joint <= capr[..., :, None]) & ~eye & (SC0 > 0)
        bC, jC = torch.where(okC, SC0, neg).max(3)
        want_c = fr & (bC > bA) & (bC > 0)
        mutual = want_c & want_c.gather(2, jC) & (jC.gather(2, jC) == idx)
        mate = torch.where(mutual, jC, mate)
    # Paired clusters merge now but cannot request a destination until next round.
    matched = mate >= 0
    want_a = free & ~matched & (bA > 0) & (bA >= best_c)
    partner = torch.where((mate >= 0) & (mate > idx), mate, torch.full_like(mate, -1))
    away = (mate >= 0) & (mate < idx)
    hp = partner >= 0; pc = partner.clamp(min=0)
    CM = torch.where(hp[..., None], torch.maximum(CM, gath(CM, pc)), CM)
    cw = torch.where(hp, cw + cw.gather(2, pc), cw)
    QA = torch.where(hp[..., None], QA + gath(QA, pc), QA)
    tgt = torch.where(away, mate, idx)
    lab = tgt.gather(2, lab)
    alive = alive & ~away
    CM = torch.where(away[..., None], torch.zeros_like(CM), CM)
    cw = torch.where(away, torch.zeros_like(cw), cw)
    QA = torch.where(away[..., None], torch.zeros_like(QA), QA)
    unit, load, UCM, acc_any = _accept(want_a, uA, bA, unit, load, UCM, cw, CM, capu, U, rhead)
    return (CM, cw, QA, UCM, lab, alive, unit, load), hp.any() | acc_any


def ap_finish(state, consts, U, WQ, WK, forced_rounds=1):
    """the fill: forced rounds, then the rows of unplaced clusters one by one, then least-loaded. -> unit [G, H, m]"""
    CM, cw, QA, UCM, lab, alive, unit, load = state
    valid, idx, tie, capr, capu, CM0, cw0, QA0 = consts[:8]
    rhead = consts[10] if len(consts) > 10 else None
    G, Hn, n = cw.shape; dev = cw.device; nb = CM.shape[3]
    neg = torch.full((), -1.0, device=dev)
    capg = capu[:, None, None, :]
    for _ in range(forced_rounds):
        free = alive & (unit < 0)
        SA = QA + WK * _aff(CM, UCM, rhead, U).float()
        SA = torch.where(free[..., None] & (load[:, None, None, :] + cw[..., None] <= capg), SA + 1.0, neg)
        bA, uA = SA.max(3)
        unit, load, UCM, _ = _accept(free & (bA > 0), uA, bA, unit, load, UCM, cw, CM, capu, U, rhead)
    Af = CM0.float()
    rowunit = unit.gather(2, lab)
    left = (rowunit < 0) & valid
    lab = torch.where(left, idx, lab)
    sel = left.float()
    cw = torch.zeros((G, Hn, n), device=dev).scatter_add(2, lab, cw0 * sel)
    CM = (torch.zeros((G, Hn, n, nb), device=dev).scatter_add(2, lab[..., None].expand(-1, -1, -1, nb), Af * sel[..., None]) > 0).half()
    QA = torch.zeros((G, Hn, n, U), device=dev).scatter_add(2, lab[..., None].expand(-1, -1, -1, U), QA0 * sel[..., None])
    piece = torch.zeros((G, Hn, n), device=dev).scatter_add(2, lab, sel) > 0
    unit = torch.full((G, Hn, n), -1, dtype=torch.long, device=dev)
    for _ in range(2):
        free = piece & (unit < 0)
        SA = QA + WK * _aff(CM, UCM, rhead, U).float()
        SA = torch.where(free[..., None] & (load[:, None, None, :] + cw[..., None] <= capg), SA + 1.0, neg)
        bA, uA = SA.max(3)
        unit, load, UCM, _ = _accept(free & (bA > 0), uA, bA, unit, load, UCM, cw, CM, capu, U, rhead)
    rowunit = torch.where(left, unit.gather(2, lab), rowunit)
    free = rowunit < 0
    return torch.where(free, water_fill(free & valid, cw0, load), rowunit)


def water_fill(free, w, load):
    """the last rows nothing could take, placed while updating the loads: rather than all to the one least-loaded
    unit (a whole leftover could double a GPU's work), they fill the units up to one common level in one pass: the
    level lam solves sum_u max(0, lam - load_u) = the leftover work, each unit's room is max(0, lam - load_u), the
    rooms are laid end to end in unit-id order, and each row (in row order, so a head's rows stay together) goes to the
    room that holds the midpoint of its cumulative work. Rows are not sorted by size; a row can overrun its room by up
    to half its own work. free / w [G, H, n], load [G, U] -> unit [G, H, n]"""
    G, Hn, n = w.shape; U = load.shape[1]; dev = w.device
    wf = torch.where(free, w, torch.zeros_like(w)).reshape(G, -1)
    S = wf.sum(1)                                                         # [G] leftover work
    Ls = load.sort(1).values                                              # ascending
    k = torch.arange(1, U + 1, device=dev, dtype=load.dtype)
    lam_k = (S[:, None] + Ls.cumsum(1)) / k                               # level if the k lowest units take it all
    nxt = torch.cat([Ls[:, 1:], torch.full_like(Ls[:, :1], float("inf"))], 1)
    ok = lam_k <= nxt                                                     # the first k for which the level fits
    kk = torch.where(ok, torch.arange(U, device=dev)[None].expand(G, U), torch.full((G, U), U, device=dev, dtype=torch.long)).min(1).values
    lam = lam_k.gather(1, kk.clamp(max=U - 1)[:, None])                   # [G, 1]
    room = (lam - load).clamp(min=0)                                      # [G, U] (units in id order)
    croom = room.cumsum(1)
    pos = (wf.cumsum(1) - wf / 2).clamp(max=(croom[:, -1:] * (1 - 1e-6)))
    u = torch.searchsorted(croom.contiguous(), pos.contiguous(), right=True).clamp(max=U - 1)
    return u.reshape(G, Hn, n)

