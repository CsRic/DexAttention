"""Schedule backward work on transposed block pairs; build tables in the original orientation.

KV columns are the task's resident side: kind 0 is KV, kind 1 is the Q bundle.
The shared scheduler chooses gateways, orders input sends by gain / cost,
and orders compute tasks. Gateways forward inputs and aggregate FP32 partial
gradients before returning them. Both dQ and dKV have return dependencies.

bwd_tables emits task/item lists, messages, symmetric buffers and scratch state.
Transfer pieces carry readiness signals; the default intra-node path uses
copy engines.
"""
import ctypes
import os
import time

import numpy as np

from .fwd import LINK, GW_MODE
from ..common import calibration as _cal

T = 128
TABLE_ORDER = ["task_kv", "task_ptr", "items", "task_qs_ptr", "task_qs", "task_need", "init_tasks", "qs_need",
               "ks_need", "qs_push_pe", "qs_push_dst", "ks_push_pe", "ks_push_dst", "qm_need", "qm_out", "qm_local",
               "qm_ptr", "qm_land", "km_need", "km_out", "km_local", "km_ptr", "km_land", "remote_sig",
               "sig_task_ptr", "sig_task", "sig_qm", "sig_km", "qs_agg_need", "qs_agg_ptr", "qs_agg_land",
               "ks_agg_need", "ks_agg_ptr", "ks_agg_land", "sig_qa", "sig_ka", "sig_fwd", "msg", "trig_ptr", "trig_lane",
               "row_grp_ptr", "grp_beg", "grp_end", "grp_dep", "grp_need", "row_qdep", "row_order", "sig_grp_ptr",
               "sig_grp", "sig_row_ptr", "sig_row", "red_need", "red_kind", "red_pay", "red_n", "sig_red_ptr", "sig_red",
               "qa_unit", "ka_unit"]

RECT_ORDER = ["rect_g_rect", "rect_g_col", "rect_g_k0", "rect_g_cnt", "rect_g_ks", "rect_pp", "rect_pair", "rect_bk",
              "rect_pair_bq", "rect_bits", "rect_dbits", "rect_prow", "rect_pcol", "rect_qslot", "rect_kslot",
              "rect_exp_order"]


def rect_half_bits(B, tabs, nb):
    """B["hbits"] (per global pair) reordered as the rect tables' local pairs (rect_pair_bq = row block of local pair q,
    rect_bk[R] = column block of rectangle R holding pairs rect_pair[rect_pp[R]:rect_pp[R + 1]]) -> (int32 view, nw2)"""
    hb = np.ascontiguousarray(B["hbits"], np.uint64); nw2 = hb.shape[1]
    pbq = np.asarray(tabs["rect_pair_bq"], np.int64); pp = np.asarray(tabs["rect_pp"], np.int64)
    pair = np.asarray(tabs["rect_pair"], np.int64); rbk = np.asarray(tabs["rect_bk"], np.int64)
    loc_bk = np.empty(len(pbq), np.int64)
    loc_bk[pair[:pp[-1]]] = np.repeat(rbk, np.diff(pp))
    gk = np.asarray(B["bq"], np.int64) * nb + np.asarray(B["bk"], np.int64)
    o = np.argsort(gk, kind="stable"); lk = pbq * nb + loc_bk
    pos = np.searchsorted(gk[o], lk)
    assert np.array_equal(gk[o][pos], lk), "rect pair not among the block pairs"
    out = hb[o[pos]]
    return (np.ascontiguousarray(out).view(np.int32).reshape(-1) if len(out) else np.zeros(2, np.int32)), nw2


def expand_plan(plan, n_tiles):
    """fwd-format plan (owner per block, schedule_table per block pair) -> per-tile owner and per-tile-pair PE."""
    B = plan["block_tokens"]; k = B // T
    own = np.repeat(np.asarray(plan["owner"], np.int64), k)[:n_tiles]
    st = np.asarray(plan["schedule_table"], np.int64)
    dev = np.repeat(np.repeat(st, k, 0), k, 1)[:n_tiles, :n_tiles]
    return own, dev


def region_layout(glob, h, hk, d=128):
    """byte offsets of [sig, qb, kv, dqa, dkva, dql, dkl] and per-slot bytes [qb, kv, dqa, dkva]; total bytes."""
    al = lambda x: (x + 255) // 256 * 256                 # noqa: E731
    qb = 2 * (T * h * d * 2) + 2 * (h * T * 4); kv = 2 * (T * hk * d * 2)
    dqa = T * h * d * 4; dkva = 2 * T * hk * d * 4
    NQS, NKS, NDQL, NDKL = glob["NQS"], glob["NKS"], glob["NDQL"], glob["NDKL"]
    offs, o = [], 0
    for sz in (8 * (NQS + NKS + NDQL + NDKL), NQS * qb, NKS * kv, NQS * dqa, NKS * dkva, NDQL * dqa, NDKL * dkva):
        offs.append(o); o = al(o + sz)
    return offs, [qb, kv, dqa, dkva], max(o, int(glob.get("REGION", 0)))   # (per-PE layout: the busiest PE's total)


def scratch_layout(stats, glob, h, hk, n_workers):
    """int32 offsets of [sq, sq_tail, tile_counter, task_cnt, qs_done, ks_done, qm_cnt, km_cnt, n_enq, quiet_gate, mail,
    qa_cnt, ka_cnt, pq, pq_tail, grp_cnt, red_cnt, task_desc, row_ntask, row_cnt, rows_done, sched_b0]
    (the last seven: the shared scheduler's per-call state; sched_b0 = rt::SchedB0, 3 x 512 + 2 ints)
    (the queue gets one padding entry per push: every push is an even number of entries, see dist_bwd_kernel.h);
    rectangle mode (stats rect = 1) adds [avail_bits, exp_next, done_q, done_tail, rowbuf (3 warps x 64), arena_off]."""
    ng = stats.get("n_groups", stats["n_rows"])
    sizes = [stats["tasks"] + stats["n_qm"] * (2 * h + 1) + stats["n_km"] * (hk + 1) + 2 * n_workers + 64
             + 2 * (stats["n_fwd"] + stats["n_qagg"] + stats["n_kagg"]), 1, 1,
             stats["tasks"] + 1, glob["NQS"], glob["NKS"], stats["n_qm"] + 1, stats["n_km"] + 1, 1, 1,
             32 * (n_workers // 2 + 2), glob["NQS"], glob["NKS"], stats["n_pq"] + 32, 1,
             ng + 1, stats["n_red"] + 1, 4 * ng + 4, stats["n_rows"] + 1, stats["n_rows"] + 1, 1,
             3 * 512 + 2]
    if stats.get("rect"):
        sizes += [ng // 32 + 1, 1, stats["n_rows"] + 1, 1, 256, 1]
    offs, o = [], 0
    for s in sizes:
        offs.append(o); o += (s + 31) // 32 * 32
    return offs, o


_BLIB = None
BWD_ITEM_S = _cal.load()["links"]["bwd_item_s"] / _cal.load()["backward"]["profile_count"]   # seconds per item per GPU
BWD_MAX_BYTES = 4 << 20            # transfer chunk of the backward


def _blib():
    global _BLIB
    if _BLIB is None:
        _BLIB = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", "libbwdplan.so"))
        for f in ("fsb_schedule", "bwd_tables", "bwd_tab_size", "bwd_msg_rows", "bwd_ce_rows", "bwd_tab_total", "bwd_in_size"):
            getattr(_BLIB, f).restype = ctypes.c_int64
    return _BLIB


def build_fast(B, owner_blk, P, per, pe, h, hk, gpn=4, d=128, timing=None, sched_sel=None, hook=None, sizes_only=False,
               gw=GW_MODE):
    """this PE's backward tables from the block pairs B (fwd.block_pairs). -> (glob {NQS, NKS, NDQL, NDKL},
    {tabs, stats}). timing: dict to receive schedule / tables seconds. Per-rank planning (local.py): sched_sel = the
    pairs scheduled (this node's), hook(lib) after the schedule (read / replace it; True = stop), sizes_only = pass 1."""
    L = _blib()
    c64 = lambda x: np.ascontiguousarray(x, np.int64)        # noqa: E731
    u64 = lambda x: np.ascontiguousarray(x, np.uint64)       # noqa: E731
    P_ = lambda a: a.ctypes.data_as(ctypes.c_void_p)         # noqa: E731
    owner = c64(owner_blk); nb = len(owner)
    bq, bk, pdev = c64(B["bq"]), c64(B["bk"]), c64(B["pdev"])
    prow, pcol = u64(B["prow"]), u64(B["pcol"])
    bits, dbits = u64(B["bits"]), u64(B["dbits"])
    qb = 2 * (T * h * d * 2) + 2 * (h * T * 4); kvb = 2 * (T * hk * d * 2)
    skip = hook is not None and getattr(hook, "replaces_schedule", False)   # (per-rank passes: the hook loads it)
    nf, ndg = (np.asarray(B["nfull"]), np.asarray(B["ndiag"])) if sched_sel is None else \
        (np.asarray(B["nfull"])[sched_sel], np.asarray(B["ndiag"])[sched_sel])   # (elementwise: = the full array's selection)
    pwork = None if skip else np.ascontiguousarray((nf + 0.5 * ndg) * 4 * h * BWD_ITEM_S, np.float64)
    Lk = LINK
    bw = np.array([Lk["BW_INTER"], Lk["BW_INTRA"], Lk["BW_FW"]], np.float64)
    lat = np.array([Lk["LAT_INTER"], Lk["LAT_INTRA"]], np.float64)
    cap = Lk["MAX_BYTES"] / Lk["BW_INTER"]
    dqa, dkva = T * h * d * 4, 2 * T * hk * d * 4             # fp32 partial sums per tile
    # gateway relay work per tile and other reader: the input forwarded + the partial summed (transposed kind 0 = KV)
    rw = Lk["BW_INTER"] / Lk["BW_FW"] if gw in (4, 5) else 1.0
    L.fsb_set_relay_cost(ctypes.c_double(rw * (kvb + dkva)), ctypes.c_double(rw * (qb + dqa)))
    L.fsb_set_gateway_mode(ctypes.c_int(gw))
    L.fsb_set_chunk_tiles(ctypes.c_int64(max(1, int(BWD_MAX_BYTES // qb))), ctypes.c_int64(max(1, int(BWD_MAX_BYTES // kvb))))
    t0 = time.perf_counter()
    if not skip:
        sq_, sk_, sd_, sr_, sc_ = (bq, bk, pdev, prow, pcol) if sched_sel is None else \
            (c64(bq[sched_sel]), c64(bk[sched_sel]), c64(pdev[sched_sel]), u64(prow[sched_sel]), u64(pcol[sched_sel]))
        L.fsb_schedule(ctypes.c_int64(nb), ctypes.c_int64(per), ctypes.c_int64(P), ctypes.c_int64(gpn),
                       ctypes.c_int64(len(sq_)), P_(sk_), P_(sq_), P_(sd_), P_(pwork), P_(sc_), P_(sr_), P_(owner),
                       ctypes.c_void_p(0), ctypes.c_double(kvb), ctypes.c_double(qb), ctypes.c_double(1e15),
                       ctypes.c_double(0), ctypes.c_int(int(os.environ.get("DIST_FS_RELAY", "1"))), P_(bw), P_(lat),
                       ctypes.c_double(cap), ctypes.c_int(1))
    t1 = time.perf_counter()
    if hook is not None and hook(L):
        return None
    nt = L.bwd_tables(ctypes.c_int64(pe), ctypes.c_int64(nb), ctypes.c_int64(per), ctypes.c_int64(P),
                      ctypes.c_int64(gpn), ctypes.c_int64(len(bq)), P_(bq), P_(bk), P_(pdev), P_(prow), P_(pcol),
                      P_(bits), P_(dbits), ctypes.c_int64(bits.shape[1]), P_(owner), ctypes.c_int64(h), ctypes.c_int64(hk),
                      ctypes.c_int64(d), ctypes.c_int(1), ctypes.c_int(0), ctypes.c_int(1))
    if sizes_only:                                           # per-rank planning pass 1: own sizes + landing lists
        me = np.zeros(4, np.int64); L.bwd_get_sym_me(P_(me))
        lists = []
        for k in range(2):
            tc = np.empty(2 * int(L.bwd_in_size(ctypes.c_int(k))), np.int32); L.bwd_get_in(ctypes.c_int(k), P_(tc))
            lists.append(tc.reshape(-1, 2))
        return me, lists
    order = TABLE_ORDER + RECT_ORDER
    assert nt == len(order), (nt, len(order))
    buf = np.empty(int(L.bwd_tab_total()), np.int32); sz = np.empty(len(order), np.int64)
    L.bwd_tab_pack(P_(buf), P_(sz))
    off = np.concatenate([[0], np.cumsum(sz)])
    tabs = {name: buf[off[k]:off[k + 1]] for k, name in enumerate(order)}
    nmsg, nce = L.bwd_msg_rows(), L.bwd_ce_rows()
    msg = np.empty((nmsg, 4), np.int64); ce = np.empty((nce, 6), np.int64); sc = np.empty(27, np.int64)
    L.bwd_fetch(P_(msg), P_(ce), P_(sc))
    tabs["msg"] = msg; tabs["ce"] = ce
    t2 = time.perf_counter()
    if timing is not None:
        timing.update(schedule_s=t1 - t0, tables_s=t2 - t1)
    NQS, NKS, NDQL, NDKL = (int(x) for x in sc[:4])
    keys = ["tasks", "items", "n_init", "n_qm", "n_km", "n_remote", "pushq", "pushkv", "n_fwd", "n_qagg", "n_kagg",
            "n_pq", "n_start", "n_ce", "n_rows", "n_red", "n_fwd_msgs", "n_groups", "n_rect", "G1", "arena", "nw", "per"]
    stats = {k: int(v) for k, v in zip(keys, sc[4:])}
    stats["rect"] = 1
    if "hbits" in B:                         # 128 x 64 tiles: the rect tables' local pairs' half bitmaps
        tabs["rect_hbits"], stats["nw2"] = rect_half_bits(B, tabs, nb)
    L.bwd_region_max.restype = ctypes.c_int64
    return dict(NQS=NQS, NKS=NKS, NDQL=NDQL, NDKL=NDKL, REGION=int(L.bwd_region_max())), dict(tabs=tabs, stats=stats)
