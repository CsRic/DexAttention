"""Turn a forward assignment into messages, send order and compute order.

schedule_blocks delegates to fastsched_blocks.cpp:
1. Deduplicate Q/KV requirements; choose gateways and add local forwards.
2. Order each channel by profit = attributed compute gain / transfer cost.
3. Order Q rows by output destination, expected input readiness and group work.
Both orders are head-major when masks differ by head.

tables_blocks converts the schedule into this rank's buffers and kernel tables.
Messages are contiguous data runs; tables split them into independently
signaled pieces. Compact KV slots contain owned inputs followed by remote inputs
needed on this rank. block_pairs and block_pairs_heads supply live block pairs,
tile counts and masks; schedule_blocks converts the counts to work estimates.
"""
import ctypes
import os

import numpy as np

from ..common import calibration as _cal

_LIB = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", "libfsched.so"))
_P = lambda a: a.ctypes.data_as(ctypes.c_void_p)                       # noqa: E731
for _f in ("fsb_schedule", "fsb_tables", "fsb_tab_size", "fsb_tab_total"):
    getattr(_LIB, _f).restype = ctypes.c_int64

GiB = _cal.GiB
RELAY = int(os.environ.get("DIST_FS_RELAY", "1"))   # relay ablation (0 = no intra-node relay)
# link model (config/calibration.json "links"): T_TILE = one full tile, GQA 32/8, all heads (scaled by query heads);
# BW_INTER = one GPU's share of its node's NIC when all four send; BW_INTRA = one copy-engine stream (sender -> one
# peer), all streams busy; BW_FW = a gateway's single relay-forward stream; LAT_* = per-message latency
LINK = dict(T_TILE=_cal.fwd_model()["T_TILE"], **_cal.links(),
            MAX_BYTES=16 << 20)     # transfer chunk (smaller chunks split rows into too many groups)
GW_MODE = 5                         # relay gateway rule (fastsched_blocks.cpp g_gw_mode); per-rank planning uses 3


def schedule_blocks(bq, bk, pdev, pwork, prow, pcol, owner, P, per, gpn=4, block_head=None, heads=(32, 8), d=128,
                    static=True, gw=GW_MODE):
    """the schedule from plan-block data: pwork = compute time of each block pair (s), prow / pcol = its live tile rows
    / columns (bit masks), owner / block_head per plan block. Returns msgs [nm, 7] = (sender, receiver, kind, first
    block, count, type, parent), pos, land, and per GPU (CSR row_ptr) the block rows in compute order with their row
    group. The library keeps this schedule for tables_blocks."""
    L = LINK
    hq, hk = heads
    qb, kvb = 128 * hq * d * 2, 2 * 128 * hk * d * 2
    cap = L["MAX_BYTES"] / L["BW_INTER"]
    _LIB.fsb_set_chunk_tiles(ctypes.c_int64(max(1, int(L["MAX_BYTES"] // qb))), ctypes.c_int64(max(1, int(L["MAX_BYTES"] // kvb))))
    _LIB.fsb_set_chunk_inter(ctypes.c_int64(0), ctypes.c_int64(0))
    c64 = lambda x: np.ascontiguousarray(x, np.int64)                   # noqa: E731
    bq, bk, pdev, owner = c64(bq), c64(bk), c64(pdev), c64(owner)
    pwork = np.ascontiguousarray(pwork, np.float64)
    prow, pcol = np.ascontiguousarray(prow, np.uint64), np.ascontiguousarray(pcol, np.uint64)
    bh = c64(block_head) if block_head is not None else None
    bw = np.array([L["BW_INTER"], L["BW_INTRA"], L["BW_FW"]], np.float64)
    lat = np.array([L["LAT_INTER"], L["LAT_INTRA"]], np.float64)
    nb = len(owner)
    _LIB.fsb_set_gateway_mode(ctypes.c_int(gw))
    rw = L["BW_INTER"] / L["BW_FW"]          # relay work in time units (set every call: the library keeps it)
    _LIB.fsb_set_relay_cost(ctypes.c_double(rw * qb if gw in (4, 5) else 0.0), ctypes.c_double(rw * kvb if gw in (4, 5) else 0.0))
    nm = _LIB.fsb_schedule(ctypes.c_int64(nb), ctypes.c_int64(per), ctypes.c_int64(P), ctypes.c_int64(gpn),
                           ctypes.c_int64(len(bq)), _P(bq), _P(bk), _P(pdev), _P(pwork), _P(prow), _P(pcol),
                           _P(owner), _P(bh) if bh is not None else ctypes.c_void_p(0), ctypes.c_double(qb),
                           ctypes.c_double(kvb), ctypes.c_double(1e15), ctypes.c_double(0), ctypes.c_int(RELAY), _P(bw), _P(lat),
                           ctypes.c_double(cap), ctypes.c_int(int(static)))
    nr = _LIB.fsb_nrows()
    msgs = np.empty((nm, 7), np.int64); pos = np.empty(nm, np.int64); land = np.empty(nm, np.float64)
    row_ptr = np.empty(P + 1, np.int64); row_blk = np.empty(nr, np.int64); row_group = np.empty(nr, np.int64)
    _LIB.fsb_fetch(_P(msgs), _P(pos), _P(land), _P(row_ptr), _P(row_blk), _P(row_group))
    return dict(msgs=msgs, pos=pos, land=land, row_ptr=row_ptr, row_blk=row_blk, row_group=row_group,
                stats=dict(n_msgs=nm, makespan_comm=float(land.max(initial=0.0)), n_rows=nr,
                           n_row_groups=int(_LIB.fsb_ngroups())))


def block_pairs(m, dev_blk, per, owner=None):
    """plan-block inputs from a dense kernel-tile mask m [nv, nv] (0 / FULL 1 / DIAG 2; virtual blocks for per-head
    problems) and the plan's GPU per block pair: live pairs (bq, bk, GPU), their FULL / DIAG tile counts, tile-row /
    tile-column bit masks, and the live / DIAG tile bitmaps (per x per bits, row-major, packed little-endian into
    ceil(per*per/64) uint64 words). Used by the W1-W3 front-end and drivers."""
    nv = m.shape[0]; nb = nv // per; assert per <= 64
    R = m.reshape(nb, per, nb, per)
    live = (R != 0).any((1, 3))
    bq, bk = np.nonzero(live)
    blk = R[bq, :, bk, :]                                                  # [npair, per, per]
    lv, dg = blk != 0, blk == 2
    bits = (np.uint64(1) << np.arange(per, dtype=np.uint64))
    prow = (lv.any(2).astype(np.uint64) * bits).sum(-1, dtype=np.uint64)
    pcol = (lv.any(1).astype(np.uint64) * bits).sum(-1, dtype=np.uint64)
    nw = (per * per + 63) // 64
    def pack(x):
        b = np.packbits(x.reshape(len(x), -1), axis=1, bitorder="little")
        b = np.concatenate([b, np.zeros((len(x), nw * 8 - b.shape[1]), np.uint8)], 1)
        return np.ascontiguousarray(b).view(np.uint64).reshape(len(x), nw)
    return _ring_full(dict(bq=bq.astype(np.int64), bk=bk.astype(np.int64), pdev=np.asarray(dev_blk)[bq, bk].astype(np.int64),
                nfull=(lv & ~dg).sum((1, 2)), ndiag=dg.sum((1, 2)), prow=prow, pcol=pcol, bits=pack(lv), dbits=pack(dg)),
                      nb, per, owner)


def _ring_full(d, nb, per, owner, nbh=None):
    """DIST_RING_FULL=1 (ablation: Ring's data flow = no plan, FULL communication): every GPU receives every K / V
    block of every head it computes -- as upstream Ring passes every chunk to every rank -- including blocks nothing is
    computed with. Added as transfer-only block pairs on that GPU: no live tile (bits 0, no work), every tile column
    (pcol all ones), no rows (prow 0); blocks the GPU owns are skipped (no transfer). Head of block b = b // nbh (the
    head-virtual layout; nbh: blocks per head, DIST_RING_FULL_NBH or the builder's own, default one head). Needs every
    pair of a row block on one GPU (Q-owner assignments: Ring, USP) so an added (row, column) pair is not already present."""
    import os
    if os.environ.get("DIST_DENSE_KV") == "1":           # whole K / V blocks: every tile column of a REMOTE block (a GPU's
        if owner is None:                                # own block needs no transfer; widening it would make the owner a
            d["pcol"] = _dense_kv(d["pcol"], per)        # contributor of tiles it never computes, whose local reduction
        else:                                            # input would never arrive)
            own_ = np.asarray(owner)[:nb]
            rem = own_[np.asarray(d["bk"])] != np.asarray(d["pdev"])
            d["pcol"] = np.where(rem, _dense_kv(d["pcol"], per), d["pcol"]).astype(np.uint64)
    if os.environ.get("DIST_RING_FULL") != "1":
        return d
    assert owner is not None, "DIST_RING_FULL needs the plan's owner per block"
    owner = np.asarray(owner)[:nb]
    nbh = int(os.environ.get("DIST_RING_FULL_NBH", nbh or nb))
    full = np.uint64((1 << per) - 1) if per < 64 else np.uint64(0xFFFFFFFFFFFFFFFF)
    nw = d["bits"].shape[1]
    have = set(zip(d["pdev"].tolist(), d["bk"].tolist()))
    rows_of = {}                                                        # (GPU, head) -> a row block it computes
    for q, g in zip(d["bq"].tolist(), d["pdev"].tolist()):
        rows_of.setdefault((g, q // nbh), q)
    pairs_q = {(q, k_) for q, k_ in zip(d["bq"].tolist(), d["bk"].tolist())}
    add = []
    for (g, h), q in rows_of.items():
        for b in range(h * nbh, min(nb, (h + 1) * nbh)):
            if owner[b] != g and (g, b) not in have:
                assert (q, b) not in pairs_q, "DIST_RING_FULL: row block computed on several GPUs"
                add.append((q, b, g))
    if not add:
        return d
    a_ = np.asarray(add, np.int64)
    n = len(a_)
    ext = dict(bq=a_[:, 0], bk=a_[:, 1], pdev=a_[:, 2], nfull=np.zeros(n, d["nfull"].dtype), ndiag=np.zeros(n, d["ndiag"].dtype),
               prow=np.zeros(n, np.uint64), pcol=np.full(n, full, np.uint64), bits=np.zeros((n, nw), np.uint64),
               dbits=np.zeros((n, nw), np.uint64))
    out = {k_: np.concatenate([np.asarray(d[k_]), ext[k_]]) for k_ in ext}      # (pair-indexed keys only; block_head
                                                                                 # is per block, unchanged)
    o = np.lexsort((out["bk"], out["bq"]))                               # the usual row-major pair order
    return {k_: v[o] for k_, v in out.items()} | {k_: v for k_, v in d.items() if k_ not in out}


def _dense_kv(pcol, per):
    """DIST_DENSE_KV=1 (ablation): the Ring data flow on this executor -- every tile column of a live block
    pair is transferred (all of the block's K / V; the backward also returns all its dK / dV), compute unchanged (bits)."""
    import os
    if os.environ.get("DIST_DENSE_KV") != "1":
        return pcol
    return np.full_like(pcol, np.uint64((1 << per) - 1) if per < 64 else np.uint64(0xFFFFFFFFFFFFFFFF))


def block_pairs_heads_torch(mask_heads, dev_blk, per, owner=None):
    """block_pairs_heads on a torch tensor (GPU: the online per-layer planning of dynamic masks) --
    the same pooling (live block pairs, work per pair) and the same per-pair bit fields, identical outputs (numpy, as
    the scheduler takes them). mask_heads [H, n, n] (any int / bool dtype, nonzero = live); dev_blk [H nbh, H nbh]
    numpy or torch. Sums of distinct powers of two in int64 = the uint64 OR bit for bit (bit 63 included)."""
    import torch
    H, n, w = mask_heads.shape; nbh = n // per; assert per <= 64
    dev = mask_heads.device
    half = w == 2 * n                       # [H, n, 2 n]: 64-key halves (column 2 t + hn) -> also hbits (128 x 64 compute)
    if half:
        R2 = (mask_heads != 0).reshape(H, nbh, per, nbh, 2 * per)
        R = R2.reshape(H, nbh, per, nbh, per, 2).any(-1)                 # the union per 128 tile
    else:
        R = (mask_heads != 0).reshape(H, nbh, per, nbh, per)             # [h, i, a, j, b]
    live = R.any(4).any(2)
    h, i, j = torch.nonzero(live, as_tuple=True)                         # (h, i, j) row-major, as np.nonzero
    blk = R[h, i, :, j, :]                                               # [npair, per, per]
    bits = torch.ones(per, dtype=torch.int64, device=dev) << torch.arange(per, device=dev)
    prow = (blk.any(2).to(torch.int64) * bits).sum(-1)
    pcol = (blk.any(1).to(torch.int64) * bits).sum(-1)
    nw = (per * per + 63) // 64
    flat = torch.nn.functional.pad(blk.reshape(len(h), -1), (0, nw * 64 - per * per)).reshape(len(h), nw, 64)
    w64 = torch.ones(64, dtype=torch.int64, device=dev) << torch.arange(64, device=dev)
    words = (flat.to(torch.int64) * w64).sum(-1)                         # [npair, nw]: bit f % 64 of word f // 64
    bq, bk = h * nbh + i, h * nbh + j
    nfull = blk.sum((1, 2))
    u64 = lambda x: x.cpu().numpy().astype(np.int64).view(np.uint64)    # noqa: E731
    bq_n, bk_n = bq.cpu().numpy().astype(np.int64), bk.cpu().numpy().astype(np.int64)
    pdev = dev_blk[bq, bk].cpu().numpy() if isinstance(dev_blk, torch.Tensor) else np.asarray(dev_blk)[bq_n, bk_n]
    d = dict(bq=bq_n, bk=bk_n, pdev=np.asarray(pdev).astype(np.int64), nfull=nfull.cpu().numpy().astype(np.int64),
             ndiag=np.zeros(len(bq_n), np.int64), prow=u64(prow), pcol=u64(pcol), bits=np.ascontiguousarray(u64(words)),
             dbits=np.zeros((len(bq_n), nw), np.uint64), block_head=np.repeat(np.arange(H), nbh))
    if half:                                # per pair: bit a * 2 per + 2 b + hn of its live halves; work = halves / 2
        assert per <= 32, "128 x 64 mode: a tile row's 2 per halves must fit one 64-bit word"
        blk2 = R2[h, i, :, j, :]                                         # [npair, per, 2 per]
        nw2 = (2 * per * per + 63) // 64
        flat2 = torch.nn.functional.pad(blk2.reshape(len(h), -1), (0, nw2 * 64 - 2 * per * per)).reshape(len(h), nw2, 64)
        d["hbits"] = np.ascontiguousarray(u64((flat2.to(torch.int64) * w64).sum(-1)))
        d["nhalf"] = blk2.sum((1, 2)).cpu().numpy().astype(np.int64)
    return _ring_full(d, H * nbh, per, owner, nbh)


def block_pairs_heads(mask_heads, dev_blk, per, owner=None):
    """block_pairs for a per-head mask [H, n, n] (virtual blocks h * n / per + i) without the dense [H n, H n] matrix."""
    H, n, _ = mask_heads.shape; nbh = n // per; assert per <= 64
    R = mask_heads.reshape(H, nbh, per, nbh, per) != 0                    # [h, i, a, j, b]
    live = R.any((2, 4))
    h, i, j = np.nonzero(live)
    blk = R[h, i, :, j, :]                                                 # [npair, per, per]
    bits = (np.uint64(1) << np.arange(per, dtype=np.uint64))
    prow = (blk.any(2).astype(np.uint64) * bits).sum(-1, dtype=np.uint64)
    pcol = (blk.any(1).astype(np.uint64) * bits).sum(-1, dtype=np.uint64)
    nw = (per * per + 63) // 64
    def pack(x):
        b = np.packbits(x.reshape(len(x), -1), axis=1, bitorder="little")
        b = np.concatenate([b, np.zeros((len(x), nw * 8 - b.shape[1]), np.uint8)], 1)
        return np.ascontiguousarray(b).view(np.uint64).reshape(len(x), nw)
    bq, bk = (h * nbh + i).astype(np.int64), (h * nbh + j).astype(np.int64)
    return _ring_full(dict(bq=bq, bk=bk, pdev=np.asarray(dev_blk)[bq, bk].astype(np.int64), nfull=blk.sum((1, 2)),
                ndiag=np.zeros(len(bq), np.int64), prow=prow, pcol=pcol, bits=pack(blk),
                dbits=np.zeros((len(bq), nw), np.uint64), block_head=np.repeat(np.arange(H), nbh)), H * nbh, per, owner, nbh)


# Legacy table keys follow the positional C++ ABI. In the packed C++ output,
# kv_indices holds seg_cb, kv_mask_nb holds item_direct, and kv_ord_sig holds
# seg_loc; tables_blocks replaces kv_indices with the GPU-expanded index buffer.
_TAB_NAMES = ["kv_indptr", "kv_indices", "kv_mask_nb", "row_n_masked", "row_mask_mb", "work_coord", "wait_ptr",
              "wait_sig", "opush_pe", "opush_slot", "merge_out", "merge_ptr", "merge_slot", "mwait_ptr", "mwait_sig",
              "remote_sig", "stageq_src", "stageq_dst", "stagekv_src", "stagekv_dst", "pushq_src", "pushq_pe",
              "pushq_dst", "pushkv_src", "pushkv_pe", "pushkv_dst", "merge_self", "row_nremote", "row_qwait",
              "kv_ord_sig", "push_order", "item_q", "mself_ptr", "mself_idx", "ce_runs", "ce_sig", "pushq_cnt",
              "pushkv_cnt"] + [f"_sched{i}" for i in range(15)] + ["_sched_ce_pout", "_sched_fw"]
_CPU_TABS = {"ce_runs": 7, "_sched_ce_pout": 3, "_sched_fw": 7}          # host-read tables: CPU, [n, cols]
_PIN = {}


def _pinned(n):
    """a reused pinned staging buffer (allocating pinned memory per call is slow); the previous call's
    host-to-device copy is waited for before it is overwritten"""
    import torch
    b = _PIN.get("buf")
    if b is None or b.numel() < n:
        b = _PIN["buf"] = torch.empty(max(n, int(1.25 * (b.numel() if b is not None else 0))), dtype=torch.int32, pin_memory=True)
        _PIN["ev"] = None
    if _PIN.get("ev") is not None:
        _PIN["ev"].synchronize()
    return b[:n]


def _pinned_done():
    import torch
    _PIN["ev"] = torch.cuda.Event(); _PIN["ev"].record()


def tables_blocks(pe, B, owner, P, per, device, gpn=4, sizes_only=False):
    """the sched kernel's tables for PE `pe` from the block pairs B (block_pairs) and the schedule computed last by
    schedule_blocks for the same plan. Returns (tables, {owned tiles, items, tiles}). sizes_only (per-rank planning):
    this PE's symmetric sizes only."""
    import torch
    _LIB.fsb_set_kv_compact(ctypes.c_int(1))
    _LIB.fsb_set_gpu_expand(ctypes.c_int(0))
    c64 = lambda x: np.ascontiguousarray(x, np.int64)                   # noqa: E731
    u64 = lambda x: np.ascontiguousarray(x, np.uint64)                  # noqa: E731
    owner = c64(owner); nb = len(owner)
    bits, dbits = u64(B["bits"]), u64(B["dbits"])
    if sizes_only:
        _LIB.fsb_set_sym(ctypes.c_int64(1), ctypes.c_int64(1), ctypes.c_int64(1), ctypes.c_int(1))
    hb = u64(B["hbits"]) if "hbits" in B else None                       # (128 x 64 compute lists)
    if hb is not None:
        _LIB.fsb_set_half(_P(hb), ctypes.c_int64(hb.shape[1]))
    n = _LIB.fsb_tables(ctypes.c_int64(pe), ctypes.c_int64(nb), ctypes.c_int64(per), ctypes.c_int64(P),
                        ctypes.c_int64(gpn), ctypes.c_int64(len(B["bq"])), _P(c64(B["bq"])), _P(c64(B["bk"])),
                        _P(c64(B["pdev"])), _P(u64(B["prow"])), _P(u64(B["pcol"])), _P(bits), _P(dbits),
                        ctypes.c_int64(bits.shape[1]), _P(owner), ctypes.c_int(1), ctypes.c_int(0))
    if hb is not None:
        _LIB.fsb_set_half(ctypes.c_void_p(0), ctypes.c_int64(0))
    if sizes_only:
        me = np.zeros(3, np.int64); _LIB.fsb_get_sym_me(_P(me))
        _LIB.fsb_set_sym(ctypes.c_int64(-1), ctypes.c_int64(-1), ctypes.c_int64(-1), ctypes.c_int(0))
        return me
    total = int(_LIB.fsb_tab_total())
    host = _pinned(total) if torch.device(device).type == "cuda" else torch.empty(total, dtype=torch.int32)   # (cpu: timing only)
    sizes_ = np.empty(n, np.int64)
    _LIB.fsb_tab_pack(ctypes.c_void_p(host.data_ptr()), _P(sizes_))
    sizes = sizes_.tolist()
    dev = host.to(device, non_blocking=True)
    if torch.device(device).type == "cuda":
        _pinned_done()
    sc = np.empty(12, np.int64); _LIB.fsb_tab_scalars(_P(sc))
    n_q, n_kv, n_part, n_compute, n_merge, n_owned, NQ, NKV, NPART, n_pool, ntile = (int(x) for x in sc[:11])
    # all device views in one torch.split and the host copies from one numpy view (cheaper than slicing per table)
    dparts = torch.split(dev, sizes)
    hnp = host.numpy()
    offs = np.concatenate([[0], np.cumsum(sizes_)]).tolist()
    view = lambda k: dparts[k]                                        # noqa: E731
    tables = {}
    for k, name in enumerate(_TAB_NAMES):
        tables[name] = host[offs[k]:offs[k + 1]].clone().reshape(-1, _CPU_TABS[name]) if name in _CPU_TABS else dparts[k]
    nbase = len(_TAB_NAMES)
    # block 0 of the kernel writes the per-row KV lists (dims[23]): buffers only; compact KV slots + geometry (dims[24])
    tables["kv_indices"] = torch.empty(ntile + 1, dtype=torch.int32, device=dev.device)
    tables["kv_ord_sig"] = torch.empty(ntile + 1, dtype=torch.int32, device=dev.device)
    tables["_kv_geom"] = torch.empty(ntile + 1, dtype=torch.int32, device=dev.device)
    dxt = [view(1), view(nbase), view(nbase + 1), view(29), view(nbase + 2), view(nbase + 3)]
    order = _TAB_NAMES[:38]
    empty = dev[:0]
    for k in range(38, 43):                                            # padding up to the scheduler's tables
        tables[f"_pad{k}"] = empty; order.append(f"_pad{k}")
    order += _TAB_NAMES[38:]
    dxn = ["_dx_seg_cb", "_dx_seg_lo", "_dx_seg_hi", "_dx_seg_loc", "_dx_row_seg_ptr", "_dx_kslot"]
    for nm, t in zip(dxn, dxt):
        tables[nm] = t
    tables["dx_off"] = len(order); tables["dx_kslot"] = 1
    order += dxn
    tables["geom_off"] = len(order)
    order += ["_kv_geom"]
    tables["_order"] = order
    # host copies of what the shared engine's message table is built from (launch.engine_tables; no device sync)
    hv = lambda k: hnp[offs[k]:offs[k + 1]].copy()                    # noqa: E731
    tables["_eng_host"] = dict(pe=int(pe), gpn=int(gpn), opush_pe=hv(8), opush_slot=hv(9), pushq_src=hv(20),
                               pushq_pe=hv(21), pushq_dst=hv(22), pushkv_src=hv(23), pushkv_pe=hv(24), pushkv_dst=hv(25),
                               push_order=hv(30), pushq_cnt=hv(36), pushkv_cnt=hv(37), remote_sig=hv(15),
                               stageq_dst=hv(17), stagekv_dst=hv(19), ce=tables["ce_runs"].numpy().copy(),
                               fw=tables["_sched_fw"].numpy().copy(), ce_pout=tables["_sched_ce_pout"].numpy().copy())
    tables.update(n_q=n_q, n_kv=n_kv, n_part=n_part, n_compute=n_compute, n_merge=n_merge, n_owned=n_owned, NQ=NQ,
                  NKV=NKV, NPART=NPART, ranges_mode=1, sched_off=43, n_pool=n_pool, kv_half=int(hb is not None))
    owned = (np.flatnonzero(owner == pe)[:, None] * per + np.arange(per)).reshape(-1)   # (tiles ascending)
    return tables, dict(owned=owned.tolist(), items=n_compute, tiles=int(sizes[1]))
