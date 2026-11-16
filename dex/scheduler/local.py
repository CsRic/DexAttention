"""Build each rank's schedule using node-local work and metadata exchanges.

1. Reconstruct the node's incoming messages and their compute gain / cost.
2. Gateways send message records to owners; each owner sorts its full NIC queue.
3. Owners return expected arrivals to the receiving node's ranks.
4. Reconstruct forward arrivals and compute order; emit this rank's tables.
An all-reduce MAX agrees on symmetric buffer sizes. Backward also exchanges
partial-gradient landing slots from receivers to contributors.

Gateway loads here are scoped to (sender, receiving node), so ranks can rebuild
routes independently. Tables match global planning with that same rule (mode 3);
the global default also balances forwarding across senders (mode 5).
Exchanges carry plan metadata, not attention tensors.
"""
import ctypes
import os

import numpy as np

from . import fwd as fs
from . import bwd as bp


P_ = lambda a: a.ctypes.data_as(ctypes.c_void_p)   # noqa: E731


def enabled(P):
    """DIST_LOCAL_PLAN: 1 = always, 0 = never, auto (default) = from 64 GPUs, where the global schedule's cost starts
    to matter"""
    v = os.environ.get("DIST_LOCAL_PLAN", "auto")
    return v == "1" or (v == "auto" and P >= 64)


_GROUP = None


def _group():
    """a gloo (CPU / TCP) group, created once per process: the records exchanged are small host-side arrays."""
    global _GROUP
    import torch.distributed as dist
    if _GROUP is None:
        if dist.get_backend() == "gloo":
            _GROUP = dist.group.WORLD
        else:
            _GROUP = dist.new_group(backend="gloo")
    return _GROUP


def _dev():
    import torch
    import torch.distributed as dist
    return torch.device("cuda", torch.cuda.current_device()) if dist.get_backend(_group()) == "nccl" else torch.device("cpu")


def _a2a(rows, width):
    """all_to_all of int64 rows: rows[q] = array [k, width] for rank q -> list of the arrays received from each rank"""
    import torch
    import torch.distributed as dist
    P = len(rows)
    dev = _dev()
    cnt = torch.tensor([len(r) for r in rows], dtype=torch.int64, device=dev)
    rcnt = torch.empty_like(cnt)
    dist.all_to_all_single(rcnt, cnt, group=_group())
    rc = rcnt.tolist()
    send = torch.from_numpy(np.ascontiguousarray(np.concatenate([np.asarray(r, np.int64).reshape(-1, width) for r in rows]))).to(dev).reshape(-1)
    recv = torch.empty(sum(rc) * width, dtype=torch.int64, device=dev)
    dist.all_to_all_single(recv, send, output_split_sizes=[c * width for c in rc], input_split_sizes=[len(r) * width for r in rows],
                           group=_group())
    out = recv.cpu().numpy().reshape(-1, width)
    off = np.concatenate([[0], np.cumsum(rc)])
    return [out[off[q]:off[q + 1]] for q in range(P)]


def _allmax(x):
    import torch
    import torch.distributed as dist
    t = torch.as_tensor(np.asarray(x, np.int64), device=_dev())
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=_group())
    return t.cpu().numpy()


f2i = lambda x: np.asarray(x, np.float64).view(np.int64)                # noqa: E731  (bit-exact doubles in int64)
i2f = lambda x: np.asarray(x, np.int64).view(np.float64)                # noqa: E731


def _canon(M, i, gpn):
    """a message's key in the global construction order (kind; intra pass, then relay by (sender, node))"""
    s, d, k, b0, c, t, par = (int(x) for x in M[i])
    if t == 1: return (k, 0, s, d, b0, 0)
    if t == 0: return (k, 1, s, d // gpn, b0, 0)
    return (k, 1, int(M[par][0]), d // gpn, b0, 1 + d)


def _fetch(L, P):
    """the schedule just computed by fsb_schedule (this library): messages, positions, landings, rows, static data"""
    L.fsb_nmsgs.restype = ctypes.c_int64; L.fsb_nrows.restype = ctypes.c_int64; L.fsb_ngroups.restype = ctypes.c_int64
    nm, nr = int(L.fsb_nmsgs()), int(L.fsb_nrows())
    msgs = np.empty((nm, 7), np.int64); pos = np.empty(nm, np.int64); land = np.empty(nm)
    rp = np.empty(P + 1, np.int64); rb = np.empty(nr, np.int64); rg = np.empty(nr, np.int64)
    L.fsb_fetch(P_(msgs), P_(pos), P_(land), P_(rp), P_(rb), P_(rg))
    u = np.empty(nm); st = np.empty(nm); mt = np.empty(nm)
    L.fsb_fetch_static(P_(u), P_(st)); L.fsb_fetch_mtime(P_(mt))
    return dict(msgs=msgs, pos=pos, land=land, row_ptr=rp, row_blk=rb, row_group=rg, u=u, start=st, mt=mt, ng=int(L.fsb_ngroups()))


def _split_by(X, key, P):
    """rows X [n, w] grouped by key in [0, P) (stable: row order inside a group) -> list of P arrays"""
    o = np.argsort(key, kind="stable")
    cnt = np.bincount(key, minlength=P)
    return np.split(X[o], np.cumsum(cnt)[:-1])


def _exchange_landings(L, pe, P, gpn, RA, heads_of):
    """x1 / B / x2: returns this rank's own NIC sends (msg7 rows, start, land), set into the library the landings of
    the NIC messages into its node (fsb_set_ext_land). heads_of(b0 array) -> head of each first tile's block."""
    M = RA["msgs"]
    idx = np.nonzero((M[:, 5] == 0) & (M[:, 1] == pe))[0]               # NIC messages this rank receives (gateway)
    Mi = M[idx]                                                         # (k, s, d, b0, cnt, head, u bits, mt bits)
    X = np.stack([Mi[:, 2], Mi[:, 0], Mi[:, 1], Mi[:, 3], Mi[:, 4], np.asarray(heads_of(Mi[:, 3]), np.int64),
                  np.ascontiguousarray(RA["u"][idx], np.float64).view(np.int64),
                  np.ascontiguousarray(RA["mt"][idx], np.float64).view(np.int64)], 1).astype(np.int64).reshape(-1, 8)
    got = _a2a(_split_by(X, Mi[:, 0], P), 8)
    mine = np.concatenate(got).reshape(-1, 8)                          # B: this rank's NIC link, all nodes
    u, mt = i2f(mine[:, 6]), i2f(mine[:, 7])
    # order (head, -(u / mt), kind, receiver node, first tile) = the global static order's tie-break (message
    # construction order)
    o = np.lexsort((mine[:, 3], mine[:, 2] // gpn, mine[:, 0], -(u / mt), mine[:, 5]))
    ms = mine[o]; mto = mt[o]
    oend = np.cumsum(mto)                                               # sequential sums: tl += mt per message
    ost = np.concatenate([[0.0], oend[:-1]]) if len(oend) else np.zeros(0)
    k, s, d, b0, c = ms[:, 0], ms[:, 1], ms[:, 2], ms[:, 3], ms[:, 4]
    own7 = np.stack([s, d, k, b0, c, np.zeros_like(s), np.full_like(s, -1)], 1).reshape(-1, 7)
    Y = np.stack([k, s, d, b0, ost.view(np.int64), oend.view(np.int64)], 1).reshape(-1, 6)
    dn = d // gpn
    byn = _split_by(Y, dn, (P + gpn - 1) // gpn)                        # every rank of the receiver's node
    back = [byn[q // gpn] for q in range(P)]
    lands = np.concatenate(_a2a(back, 6)).reshape(-1, 6)
    k4 = np.ascontiguousarray(lands[:, :4]); est, eld = np.ascontiguousarray(i2f(lands[:, 4])), np.ascontiguousarray(i2f(lands[:, 5]))
    L.fsb_set_ext_land(ctypes.c_int64(len(k4)), P_(k4), P_(est), P_(eld))
    return (own7, ost, oend), None


def _canon_keys(M, gpn):
    """_canon of every message, as 6 key columns (k, pass, sender, receiver or its node, first tile, 0 or 1 + receiver)"""
    s, d, k, b0, t, par = M[:, 0], M[:, 1], M[:, 2], M[:, 3], M[:, 5], M[:, 6]
    t1, t2 = t == 1, t == 2
    psend = np.where(t2, M[np.where(t2, par, 0), 0] if len(M) else s, s)
    return [k, np.where(t1, 0, 1), psend, np.where(t1, d, d // gpn), b0, np.where(t2, 1 + d, 0)]


def _load_local(L, pe, P, gpn, RC, own):
    """this rank's messages (the node's touching it, parents of its forwards, its own NIC sends) in the global message
    order, positions = its sends ranked by (start, global order) -- loaded as the library's schedule (numpy; the
    per-message version _load_local_py is used if two messages have the same order key)"""
    M = RC["msgs"]
    hit = (M[:, 0] == pe) | (M[:, 1] == pe)
    selm = hit.copy()
    selm[M[hit & (M[:, 5] == 2), 6]] = True                            # parents of this rank's forwards
    sel = np.nonzero(selm)[0]
    own7, ost, oend = own
    xs = np.nonzero(own7[:, 1] // gpn != pe // gpn)[0]                  # (always all: NIC sends leave the node)
    Kc = [x[sel] for x in _canon_keys(M, gpn)]
    Ox = own7[xs]
    Kx = [Ox[:, 2], np.ones(len(xs), np.int64), Ox[:, 0], Ox[:, 1] // gpn, Ox[:, 3], np.zeros(len(xs), np.int64)]
    K = [np.concatenate([a, b]).astype(np.int64) for a, b in zip(Kc, Kx)]
    o = np.lexsort(K[::-1])
    if len(o) > 1:
        Ks = np.stack([x[o] for x in K], 1)
        if (Ks[1:] == Ks[:-1]).all(1).any():                            # equal keys: the set order mattered
            return _load_local_py(L, pe, P, gpn, RC, own)
    nC = len(sel)
    rowsC = M[sel].copy()
    pos_of = np.full(len(M), -1, np.int64)                              # RC index -> position in the sorted list
    inv = np.empty(len(o), np.int64); inv[o] = np.arange(len(o))
    pos_of[sel] = inv[:nC]
    rowsC[:, 6] = np.where(rowsC[:, 6] >= 0, pos_of[np.maximum(rowsC[:, 6], 0)], rowsC[:, 6])
    msg7 = np.ascontiguousarray(np.concatenate([rowsC, Ox]).astype(np.int64)[o]).reshape(-1, 7)
    land = np.ascontiguousarray(np.concatenate([RC["land"][sel], np.asarray(oend, np.float64)[xs]])[o], np.float64)
    start = np.concatenate([RC["start"][sel], np.asarray(ost, np.float64)[xs]])[o]
    pos = np.full(len(o), -1, np.int64)
    mine = np.nonzero(msg7[:, 0] == pe)[0]                               # (key order already; stable by start)
    mine = mine[np.argsort(start[mine], kind="stable")]
    pos[mine] = np.arange(len(mine))
    rp, rb, rg = (np.ascontiguousarray(RC[k]) for k in ("row_ptr", "row_blk", "row_group"))
    L.fsb_load_out(ctypes.c_int64(len(msg7)), P_(msg7), P_(pos), P_(land), ctypes.c_int64(P), P_(rp),
                   ctypes.c_int64(len(rb)), P_(rb), P_(rg), ctypes.c_int64(RC["ng"]))


def _load_local_py(L, pe, P, gpn, RC, own):
    """the per-message version of _load_local (used when message keys tie)"""
    M = RC["msgs"]
    sel = set()
    for i in range(len(M)):
        if M[i][0] == pe or M[i][1] == pe:
            sel.add(i)
            if M[i][5] == 2: sel.add(int(M[i][6]))
    items = [(_canon(M, i, gpn), M[i].copy(), RC["land"][i], RC["start"][i], ("C", i)) for i in sel]
    own7, ost, oend = own
    for j in range(len(own7)):
        r = own7[j]
        if r[1] // gpn == pe // gpn: continue                          # (never: NIC sends leave the node)
        items.append(((int(r[2]), 1, int(r[0]), int(r[1]) // gpn, int(r[3]), 0), r.copy(), oend[j], ost[j], ("X", j)))
    items.sort(key=lambda x: x[0])
    idx = {it[4]: j for j, it in enumerate(items)}
    msg7 = np.stack([it[1] for it in items]).astype(np.int64) if items else np.zeros((0, 7), np.int64)
    for j, it in enumerate(items):
        if msg7[j][6] >= 0: msg7[j][6] = idx[("C", int(it[1][6]))]
    land = np.ascontiguousarray([it[2] for it in items], np.float64); start = np.asarray([it[3] for it in items])
    pos = np.full(len(items), -1, np.int64)
    mine = sorted([j for j in range(len(items)) if msg7[j][0] == pe], key=lambda j: (start[j], items[j][0]))
    for r_, j in enumerate(mine): pos[j] = r_
    rp, rb, rg = (np.ascontiguousarray(RC[k]) for k in ("row_ptr", "row_blk", "row_group"))
    L.fsb_load_out(ctypes.c_int64(len(msg7)), P_(msg7), P_(pos), P_(land), ctypes.c_int64(P), P_(rp),
                   ctypes.c_int64(len(rb)), P_(rb), P_(rg), ctypes.c_int64(RC["ng"]))


def fwd_tables(pe, P, B, pw, owner, per, device, block_head=None, heads=(32, 8), d=128, gpn=4):
    """fwd.schedule_blocks + tables_blocks for rank pe, per-rank (see the module doc). Returns (tables, tm)."""
    L = fs._LIB
    owner = np.asarray(owner, np.int64)
    sel = (np.asarray(B["pdev"]) // gpn) == pe // gpn
    bh = np.asarray(block_head, np.int64) if block_head is not None else None
    heads_of = (lambda b0: bh[b0 // per]) if bh is not None else (lambda b0: np.zeros(len(b0), np.int64))   # noqa: E731
    Bs = [np.asarray(B[k_])[sel] for k_ in ("bq", "bk", "pdev")] + [np.asarray(pw)[sel]] + [np.asarray(B[k_])[sel] for k_ in ("prow", "pcol")]

    def run():
        fs.schedule_blocks(*Bs, owner, P, per, gpn=gpn, block_head=block_head, heads=heads, d=d, static=True, gw=3)
        return _fetch(L, P)
    L.fsb_set_ext_land(ctypes.c_int64(0), None, None, None)
    try:
        RA = run()
        own, _ = _exchange_landings(L, pe, P, gpn, RA, heads_of)
        RC = run()
    finally:
        L.fsb_set_ext_land(ctypes.c_int64(0), None, None, None)
    _load_local(L, pe, P, gpn, RC, own)
    me = fs.tables_blocks(pe, B, owner, P, per, device, gpn=gpn, sizes_only=True)
    NQ, NKV, NPART = (int(x) for x in _allmax(me))
    L.fsb_set_sym(ctypes.c_int64(NQ), ctypes.c_int64(NKV), ctypes.c_int64(NPART), ctypes.c_int(0))
    try:
        tables, tm = fs.tables_blocks(pe, B, owner, P, per, device, gpn=gpn)
    finally:
        L.fsb_set_sym(ctypes.c_int64(-1), ctypes.c_int64(-1), ctypes.c_int64(-1), ctypes.c_int(0))
    return tables, tm


def bwd_tables(pe, P, B, owner, per, h, hk, gpn=4, timing=None):
    """bwd.build_fast for rank pe, per-rank (see the module doc). Returns build_fast's (glob, {tabs, stats})."""
    L = bp._blib()
    owner = np.asarray(owner, np.int64)
    sel = (np.asarray(B["pdev"]) // gpn) == pe // gpn
    store = {}

    def fetch(Lx):
        store["R"] = _fetch(Lx, P)
        return True
    bf = lambda **kw: bp.build_fast(B, owner, P, per, pe, h, hk, gpn=gpn, sched_sel=sel, gw=3, **kw)   # noqa: E731
    L.fsb_set_ext_land(ctypes.c_int64(0), None, None, None)
    try:
        bf(hook=fetch); RA = store["R"]
        own, _ = _exchange_landings(L, pe, P, gpn, RA, lambda b0: np.zeros(len(b0), np.int64))
        bf(hook=fetch); RC = store["R"]
    finally:
        L.fsb_set_ext_land(ctypes.c_int64(0), None, None, None)

    def load(Lx):
        _load_local(Lx, pe, P, gpn, RC, own)
        return False
    load.replaces_schedule = True          # (build_fast skips its own fsb_schedule: the hook loads the schedule)
    big = ctypes.c_int64(1 << 40)
    L.bwd_set_sym(big, big, big, big, ctypes.c_int(1))
    try:
        me, lists = bf(hook=load, sizes_only=True)                      # pass 1: own sizes and landing list
    finally:
        L.bwd_set_sym(*(ctypes.c_int64(-1) for _ in range(4)), ctypes.c_int(0))
    X4 = np.concatenate([np.stack([np.full(len(lists[k]), k, np.int64), lists[k][:, 0].astype(np.int64),
                                   np.full(len(lists[k]), pe, np.int64), np.arange(len(lists[k]), dtype=np.int64)], 1)
                         for k in range(2)]).reshape(-1, 4)            # x4: slot of every contribution, to its maker
    C4 = np.concatenate([lists[k][:, 1].astype(np.int64) for k in range(2)])
    outs = np.concatenate(_a2a(_split_by(X4, C4, P), 4)).reshape(-1, 4)
    sizes = _allmax(me)
    L.bwd_set_sym(*(ctypes.c_int64(int(x)) for x in sizes), ctypes.c_int(0))
    o4 = np.ascontiguousarray(outs, np.int64)
    L.bwd_set_out(ctypes.c_int64(len(o4)), P_(o4))
    try:
        res = bf(hook=load, timing=timing)                              # pass 2: the tables
    finally:
        L.bwd_set_sym(*(ctypes.c_int64(-1) for _ in range(4)), ctypes.c_int(0))
        L.bwd_set_out(ctypes.c_int64(0), None)
    return res
