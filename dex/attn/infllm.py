"""Dex on W4 (MiniCPM4.1-8B, InfLLM-v2 top-k attention, 32 query / 2 KV heads, BF16): the persistent executors around
the OFFICIAL InfLLM-v2 routines (dex_cuda.infllm_* / infllm2d_*: block 0 = scheduler + communication engine, the other
CTAs run the official forward row routine / backward column routine), their plan tables, the planner front end, and the
training-step attention ops.

  Dex (system `dex`)                       Plan2D + InfllmFwd2D / InfllmBwd2D: the two KV-head groups are planned
                                               as ONE problem by Dex's planner, then the home gate; queries may move
                                               (Q-side transfer + FP32 partial O / dQ returned to the owner)
  Dex without assigner (`dex_no_assigner`) InfllmFwd / InfllmBwd: every query stays on its zigzag owner; the same
                                               transport (copy engines inside a node, NVSHMEM across nodes, one gateway
                                               per remote node that relays to the node's other readers)

Layouts (official varlen "group rows"): q / o / do / dq [L * 16, 2, 128] bf16, the rank's chunks concatenated in `my`
order; lse [2, L * 16] fp32; k_own / v_own [L, 2, 128] bf16; sel [2, L, K] int32 causal-filtered global 64-token block
ids (-1 = none). K | V live in a symmetric region in the global layout (row = token, 1 KB: [K h0 | K h1 | V h0 | V h1]);
runs (<= RUN_MAX blocks inside one chunk) are the transfer unit.

    fw = InfllmFwd2D(DA, Plan2D(chunks, my, S, P, rank, dev)); o, lse = fw.forward(q, k_own, v_own, sel, scale)
"""
import os
import time

import numpy as np
import torch
import torch.distributed as dist

from dex.common.calibration import infllm_entry_times
from workloads.w4_infllm import reuse
from workloads.w4_infllm.common import (GH, chunk_rank, filter_causal, from_rows, gpus_per_node, pad_concat, select_cp,
                                        to_rows)
from workloads.w4_infllm.stage1_fast import merge_n

RUN_MAX = 8                    # 64-token blocks per transfer run (runs never cross a chunk boundary)
D = 128
G = 2                          # KV-head groups
K_NOP = 4                      # queue kind of the landing units (both 2-D kernels)
# planner calibration (config/calibration.json): seconds per ENTRY (one token x one selected 64-key block, 16 query heads)
# of the official forward row routine / backward column routine on one GPU, measured on real W4 selections
_ENTRY_TIMES = infllm_entry_times()
T_ENTRY_FWD = _ENTRY_TIMES["T_ENTRY_FWD"]
T_ENTRY_BWD = _ENTRY_TIMES["T_ENTRY_BWD"]
# home gate: a plan that moves queries is kept only if the model gives home / plan > 1 + eps; estimates within the margin
# eps count as ties and home, which moves no queries, is kept
GATE_EPS = {"fwd": 0.040, "bwd": 0.046}


def ext():
    """the compiled extension (dex/dex_cuda*.so)"""
    from dex import dex_cuda
    return dex_cuda


def AL(x):
    """256-byte alignment (scalars and arrays)"""
    r = (np.asarray(x, np.int64) + 255) // 256 * 256
    return int(r) if r.ndim == 0 else r


# ================================================================ plan tables (every rank computes the same global
# plan from the global needs matrix and takes its own rows)


BLK = 64                       # tokens per block (official n_block_dim)
ROW_BYTES = 1024               # one K | V row: 512 bf16
SLOT_BYTES = BLK * 512 * 4     # accumulator / landing slot: 64 rows x (dK 256 | dV 256) fp32
K_SUM, K_AGG = 1, 2            # queue kinds (infllm_bwd_params.h)
BWD_TABLE_ORDER = ["task_b", "task_h", "task_n", "ks_need", "ks_agg_need", "ka_unit", "ks_push_pe", "ks_push_dst",
               "km_blk", "km_out", "km_rows", "km_ptr", "km_land", "ka_blk", "ka_ptr", "ka_land", "msg", "trig_ptr",
               "trig_lane", "remote_sig", "row_grp_ptr", "grp_beg", "grp_end", "grp_dep", "row_qdep", "row_order", "sig_grp_ptr",
               "sig_grp", "sig_row_ptr", "sig_row", "red_need", "red_kind", "red_pay", "red_n", "sig_red_ptr", "sig_red"]


def static_plan(own, chunks, S, gpn, run_max):
    """once per (S, P): runs (<= run_max blocks inside one chunk) and their owners"""
    nb = len(own)
    cstart = np.zeros(nb, bool)
    for a_, _ in chunks:
        assert a_ % BLK == 0, "chunks must be block-aligned"
        cstart[a_ // BLK] = True
    rs, b = [], 0
    while b < nb:
        rs.append(b)
        e = b + 1
        while e < nb and not cstart[e] and e - b < run_max:
            e += 1
        b = e
    rs = np.array(rs, np.int64)
    rl = np.diff(np.concatenate([rs, [nb]]))
    P = int(own.max()) + 1
    nn = (P + gpn - 1) // gpn
    return dict(own=own, nb=nb, S=S, rs=rs, rl=rl, run_of=np.repeat(np.arange(len(rs)), rl), run_owner=own[rs],
                r0=rs * BLK, nr=np.minimum((rs + rl) * BLK, S) - rs * BLK, P=P, gpn=gpn, nn=nn, NG=gpn + nn,
                n_run_sig=len(rs) * (gpn + nn), node_of=own // gpn, chunks=chunks)


def gateways(st, rneed, b_):
    """gateway (rank) of every cross-node run into node b_ that some GPU there needs."""
    gpn, P, ro = st["gpn"], st["P"], st["run_owner"]
    gp = np.arange(b_ * gpn, min(P, b_ * gpn + gpn))
    k = np.nonzero(((ro // gpn) != b_) & rneed[gp].any(0))[0]
    want = rneed[gp][:, k]
    j0 = ro[k] % gpn
    # prefer the owner's rail-aligned GPU; else the first selecting GPU at or after a per-run rotating start (spreads
    # the relay load over the node's GPUs)
    n = len(gp)
    stt = (j0 + 1 + k) % n
    rot = (stt[None, :] + np.arange(n)[:, None]) % n
    fb = rot[want[rot, np.arange(len(k))[None, :]].argmax(0), np.arange(len(k))]
    return gp, k, want, np.where(want[j0, np.arange(len(k))], j0, fb)


def global_plan(st, needs, relay=True):
    """the part every rank computes identically: gateways, partial routes, landing slots."""
    P, gpn, nn, own, node_of, run_of = st["P"], st["gpn"], st["nn"], st["own"], st["node_of"], st["run_of"]
    nb = st["nb"]
    rneed = np.logical_or.reduceat(needs, st["rs"], axis=1)                       # [P, nruns]
    gw = np.full((nn, len(st["rs"])), -1, np.int64)
    for b_ in range(nn):
        gp, k, want, gwj = gateways(st, rneed, b_)
        gw[b_, k] = gp[gwj]
    rk = np.arange(P)
    E = needs & (own[None, :] != rk[:, None])                                     # (r, n): r contributes to block n
    R, N = np.nonzero(E)
    same = node_of[N] == R // gpn
    if relay and nn > 1:
        g = gw[R // gpn, run_of[N]]
        e1 = same
        e2 = ~same & (g != R)
        # kind 2: (gateway, n) for every (node, n) with the owner elsewhere and any needer in the node (the gateway may not
        # need n itself): it aggregates the node's partials and pushes one to the owner
        node_need = needs.reshape(nn, gpn, nb).any(1) if P % gpn == 0 else \
            np.stack([needs[b * gpn:(b + 1) * gpn].any(0) for b in range(nn)])     # [nn, nb]
        B3, N3 = np.nonzero(node_need & (node_of[None, :] != np.arange(nn)[:, None]))
        G3 = gw[B3, run_of[N3]]
        assert (G3 >= 0).all()
        sR = np.concatenate([R[e1], R[e2], G3]); sN = np.concatenate([N[e1], N[e2], N3])
        sD = np.concatenate([own[N[e1]], g[e2], own[N3]])
        kind = np.concatenate([np.zeros(e1.sum(), np.int8), np.ones(e2.sum(), np.int8), np.full(len(G3), 2, np.int8)])
    else:
        sR, sN, sD = R, N, own[N]
        kind = np.zeros(len(R), np.int8)
    # landing slots: per receiver, numbered by (block, sender)
    order = np.lexsort((sR, sN, sD))
    sR, sN, sD, kind = sR[order], sN[order], sD[order], kind[order]
    first = np.searchsorted(sD, rk, "left"); count = np.searchsorted(sD, rk, "right") - first
    land = np.arange(len(sD)) - first[sD]
    return dict(gw=gw, rneed=rneed, sR=sR, sN=sN, sD=sD, kind=kind, land=land, first=first, count=count,
                NL=int(count.max()) if len(count) else 0)


def bwd_region_layout(st, NL):
    """byte offsets [sig, kv, acc, land] of the symmetric region and its total size; NSIG = n_run_sig + NL."""
    al = lambda x: (x + 255) // 256 * 256                 # noqa: E731
    NSIG = st["n_run_sig"] + NL
    offs, o = [], 0
    for sz in (8 * NSIG, st["nb"] * BLK * ROW_BYTES, st["nb"] * SLOT_BYTES, max(NL, 1) * SLOT_BYTES):
        offs.append(o); o = al(o + sz)
    return offs, o, NSIG


def bwd_scratch_layout(n_tasks, nb, n_km, n_ka, n_pq, n_workers):
    """int32 offsets of [sq, sq_tail, tile_counter, ks_done, n_enq, pq, pq_tail, red_cnt, task_desc, row_ntask, grp_cnt,
    rows_done, sched_b0] (infllm_bwd_host.cpp)."""
    sizes = [n_tasks + n_km + n_ka + n_workers + 64, 1, 1, nb, 1, n_pq + 32, 1, n_km + n_ka + 1, 4 * n_tasks + 4,
             n_tasks + 1, n_tasks + 1, 1, 3 * 512 + 2]
    offs, o = [], 0
    for s in sizes:
        offs.append(o); o += (s + 31) // 32 * 32
    return offs, o


def rank_tables(st, G, needs, rank, tasks, my, n_workers, offs):
    """this rank's tables (vectorized; the global plan restricted to this rank). tasks: int arrays (b, h, n, cnt) of its
    column tasks (cnt = tokens, the work); my: its chunk indices in output order; offs: region byte offsets."""
    P, gpn, own, nb, run_of, NG, n_run_sig = st["P"], st["gpn"], st["own"], st["nb"], st["run_of"], st["NG"], st["n_run_sig"]
    rs, r0, nr, ro, chunks, S = st["rs"], st["r0"], st["nr"], st["run_owner"], st["chunks"], st["S"]
    gw, rneed, node_of = G["gw"], G["rneed"], st["node_of"]
    me_n = rank // gpn
    kv_off, acc_off, land_off = offs[1], offs[2], offs[3]
    nm = np.arange(me_n * gpn, min(P, me_n * gpn + gpn))                     # my node's ranks
    tb, th, tn, tc = (np.asarray(x, np.int64) for x in tasks)
    order = np.lexsort((tn, -tc))                                             # work descending
    tb, th, tn, tc = tb[order], th[order], tn[order], tc[order]
    n_tasks = len(tb)
    ks_need = np.bincount(tn, minlength=nb).astype(np.int32)
    sig_runs = lambda ks, d: ks * NG + np.where(node_of[rs[ks]] == d // gpn, d % gpn, gpn + d // gpn)   # noqa: E731
    # ---- polled signals: input runs (needed, not owned), then my landings
    in_runs = np.nonzero(rneed[rank] & (ro != rank))[0]
    n_in = len(in_runs)
    in_sig = sig_runs(in_runs, rank) if n_in else np.zeros(0, np.int64)
    nL = int(G["count"][rank]); l0 = int(G["first"][rank])
    LN, LK = G["sN"][l0:l0 + nL], G["kind"][l0:l0 + nL]                      # my landings: (block, sender) sorted
    remote_sig = np.concatenate([in_sig, n_run_sig + np.arange(nL)]).astype(np.int32)
    nsig = len(remote_sig)
    jr = np.full(len(rs), -1, np.int64); jr[in_runs] = np.arange(n_in)
    grp_dep = np.where(own[tn] == rank, -1, jr[run_of[tn]]).astype(np.int32)
    rows = np.nonzero(grp_dep >= 0)[0]
    sig_grp = rows[np.argsort(grp_dep[rows], kind="stable")]
    sig_grp_ptr = np.zeros(nsig + 1, np.int64)
    sig_grp_ptr[1:n_in + 1] = np.cumsum(np.bincount(grp_dep[rows], minlength=n_in)[:n_in]) if n_in else []
    sig_grp_ptr[n_in + 1:] = sig_grp_ptr[n_in]
    # ---- my output rows: chunks in `my` order
    out0, o_ = {}, 0
    for ci in my:
        out0[ci] = o_; o_ += chunks[ci][1] - chunks[ci][0]
    blk_chunk = np.empty(nb, np.int64)
    for ci, (a_, b_) in enumerate(chunks):
        blk_chunk[a_ // BLK:(b_ + BLK - 1) // BLK] = ci
    # ---- owner sums: my owned blocks with any contribution (landings of a block are contiguous in LN)
    ownb = np.nonzero(own == rank)[0]
    lo = np.searchsorted(LN, ownb, "left"); hi = np.searchsorted(LN, ownb, "right")
    need = (ks_need[ownb] > 0).astype(np.int64) + (hi - lo)
    keep = need > 0
    km_blk, km_need, klo, khi = ownb[keep], need[keep], lo[keep], hi[keep]
    n_km = len(km_blk)
    km_ptr = np.concatenate([[0], np.cumsum(khi - klo)])
    km_land_t = np.concatenate([np.arange(x, y) for x, y in zip(klo, khi)]) if n_km else np.zeros(0, np.int64)
    km_of = np.full(nb, -1, np.int64); km_of[km_blk] = np.arange(n_km)
    # ---- my routes (sender == me), in the global (dst, block, sender) order
    me = np.nonzero(G["sR"] == rank)[0]
    mN, mD, mK, mL = G["sN"][me], G["sD"][me], G["kind"][me], G["land"][me]
    ks_push_pe = np.full(nb, -1, np.int32); ks_push_dst = np.full(nb, -1, np.int32)
    ks_agg_need = np.zeros(nb, np.int32); ka_unit = np.full(nb, -1, np.int32)
    ks_push_pe[mN] = mD
    push_dst = np.full(nb, -1, np.int64); push_land = np.full(nb, -1, np.int64)
    push_dst[mN] = mD; push_land[mN] = mL
    gwm = mK == 2
    ka_blk = mN[gwm]
    n_ka = len(ka_blk)
    ka_of = np.full(nb, -1, np.int64); ka_of[ka_blk] = np.arange(n_ka); ka_unit[ka_blk] = np.arange(n_ka)
    alo = np.searchsorted(LN, ka_blk, "left"); ahi = np.searchsorted(LN, ka_blk, "right")
    ka_land_lists = [np.arange(x, y)[LK[x:y] == 1] for x, y in zip(alo, ahi)]
    ka_need = np.array([int(ks_need[n] > 0) + len(l_) for n, l_ in zip(ka_blk, ka_land_lists)], np.int64)
    ks_agg_need[ka_blk] = ka_need
    ka_ptr = np.concatenate([[0], np.cumsum([len(l_) for l_ in ka_land_lists])]) if n_ka else np.zeros(1, np.int64)
    ka_land_t = np.concatenate(ka_land_lists) if n_ka and ka_ptr[-1] else np.zeros(0, np.int64)
    ob = ownb[ks_need[ownb] > 0]
    ks_push_dst[ob] = km_of[ob]
    km_rows = np.minimum(BLK, S - km_blk * BLK)
    kc = blk_chunk[km_blk]
    km_out = np.array([out0[int(c_)] for c_ in kc], np.int64) + km_blk * BLK - np.array([chunks[int(c_)][0] for c_ in kc], np.int64) if n_km else np.zeros(0, np.int64)
    red_need = np.concatenate([km_need, ka_need]).astype(np.int32)
    red_kind = np.array([K_SUM] * n_km + [K_AGG] * n_ka, np.int32)
    red_pay = np.concatenate([np.arange(n_km), np.arange(n_ka)]).astype(np.int32)
    red_n = np.ones(n_km + n_ka, np.int32)
    sig_red = np.where(own[LN] == rank, km_of[LN], n_km + ka_of[LN]) if nL else np.zeros(0, np.int64)
    sig_red_ptr = np.zeros(nsig + 1, np.int64)
    sig_red_ptr[n_in + 1:] = np.arange(1, nL + 1)
    # ---- messages. engine (inter-node): start = my input runs to other nodes' gateways; trigger n = block n's partial
    # to another node's owner. copy engine (intra-node): start runs to same-node needers (kind 0), relay forwards after
    # the run's arrival here (kind 1), partial pushes to a same-node owner / gateway after trigger flag n (kind 2, in
    # the blocks' expected completion order: local blocks first, by the work queued before them + their own).
    ko = np.nonzero(ro == rank)[0]                                            # my runs
    src_k = kv_off + r0 * ROW_BYTES; byt_k = nr * ROW_BYTES
    kk, dd = np.nonzero(rneed[nm][:, ko].T & (nm != rank)[None, :])           # k-major, d-minor
    ks_, ds_ = ko[kk], nm[dd]
    ce_start = np.stack([src_k[ks_], src_k[ks_], byt_k[ks_], ds_, sig_runs(ks_, ds_), np.zeros(len(ks_), np.int64),
                         np.zeros(len(ks_), np.int64)], 1) if len(ks_) else np.zeros((0, 7), np.int64)
    kk, bb = np.nonzero((gw[:, ko].T >= 0) & (np.arange(st["nn"]) != me_n)[None, :])
    ks_, gs_ = ko[kk], gw[bb, ko[kk]]
    start = np.stack([src_k[ks_], src_k[ks_], byt_k[ks_], gs_, sig_runs(ks_, gs_)], 1) if len(ks_) else np.zeros((0, 5), np.int64)
    n_trig = nb + nsig
    trig_lane = np.ones(n_trig, np.int32)
    cum = (np.cumsum(tc) - tc) / max(1, n_workers) + tc
    est = np.full(nb, np.inf)
    np.minimum.at(est, tn, np.where(own[tn] == rank, 0.0, 1e9) + cum)
    pb = np.nonzero(push_dst >= 0)[0]                                         # blocks with a partial push
    same = (push_dst[pb] // gpn) == me_n
    pm = np.stack([acc_off + pb * SLOT_BYTES, land_off + push_land[pb] * SLOT_BYTES, np.full(len(pb), SLOT_BYTES),
                   push_dst[pb], n_run_sig + push_land[pb]], 1) if len(pb) else np.zeros((0, 5), np.int64)
    inter_rows = pm[~same]
    trig_lane[pb[same]] = 2
    o_ = np.argsort(est[pb[same]], kind="stable")
    ce_done = np.concatenate([pm[same][o_], np.full((int(same.sum()), 1), 2), pb[same][o_][:, None]], 1) if same.any() else np.zeros((0, 7), np.int64)
    msgs = np.concatenate([start, inter_rows]) if len(inter_rows) else start
    n_pq = len(inter_rows)
    trig_ptr = np.zeros(n_trig + 1, np.int64)
    trig_ptr[0] = len(start)
    trig_ptr[1:nb + 1] = len(start) + np.cumsum(np.bincount(pb[~same], minlength=nb)) if len(pb) else len(start)
    trig_ptr[nb + 1:] = trig_ptr[nb]
    jf = np.nonzero(gw[me_n, in_runs] == rank)[0] if n_in else np.zeros(0, np.int64)   # runs I forward (gateway)
    jj, dd = np.nonzero(rneed[nm][:, in_runs[jf]].T & (nm != rank)[None, :]) if len(jf) else (np.zeros(0, np.int64),) * 2
    js_, ds_ = jf[jj], nm[dd]; kf = in_runs[js_]
    ce_fwd = np.stack([src_k[kf], src_k[kf], byt_k[kf], ds_, sig_runs(kf, ds_), np.ones(len(kf), np.int64),
                       remote_sig[js_].astype(np.int64)], 1) if len(kf) else np.zeros((0, 7), np.int64)
    trig_lane[nb + jf] = 2
    ct = np.concatenate([ce_start, ce_fwd, ce_done]).astype(np.int64)
    ce_tab = np.stack([ct[:, 0], ct[:, 1], ct[:, 2], ct[:, 3] + (ct[:, 4] << 32), ct[:, 5] + (ct[:, 6] << 32),
                       np.ones(len(ct), np.int64)], 1) if len(ct) else np.zeros((0, 6), np.int64)
    mt = msgs.astype(np.int64)
    msg = np.stack([mt[:, 0], mt[:, 1], mt[:, 2], mt[:, 3] + (mt[:, 4] << 32)], 1) if len(mt) else np.zeros((1, 4), np.int64)
    I = lambda x: np.ascontiguousarray(np.asarray(x, np.int32))               # noqa: E731
    tabs = dict(task_b=I(tb), task_h=I(th), task_n=I(tn), ks_need=I(ks_need), ks_agg_need=I(ks_agg_need), ka_unit=I(ka_unit),
                ks_push_pe=I(ks_push_pe), ks_push_dst=I(ks_push_dst), km_blk=I(km_blk), km_out=I(km_out), km_rows=I(km_rows),
                km_ptr=I(km_ptr), km_land=I(km_land_t), ka_blk=I(ka_blk), ka_ptr=I(ka_ptr), ka_land=I(ka_land_t),
                msg=np.ascontiguousarray(msg).view(np.int32).reshape(-1), trig_ptr=I(trig_ptr), trig_lane=I(trig_lane),
                remote_sig=I(remote_sig),
                row_grp_ptr=I(np.arange(n_tasks + 1)), grp_beg=I(np.arange(n_tasks)), grp_end=I(np.arange(1, n_tasks + 1)),
                grp_dep=I(grp_dep), row_qdep=I(np.full(n_tasks, -1)), row_order=I(np.arange(n_tasks)),
                sig_grp_ptr=I(sig_grp_ptr), sig_grp=I(sig_grp), sig_row_ptr=I(np.zeros(nsig + 1)),
                sig_row=I(np.zeros(0)), red_need=I(red_need), red_kind=I(red_kind), red_pay=I(red_pay), red_n=I(red_n),
                sig_red_ptr=I(sig_red_ptr), sig_red=I(sig_red))
    for k_ in BWD_TABLE_ORDER:                                                  # every table at least one element (data_ptr)
        if tabs[k_].size == 0:
            tabs[k_] = np.zeros(1, np.int32)
    dims = dict(nb=nb, n_tasks=n_tasks, n_km=n_km, n_ka=n_ka, n_remote=nsig, n_start=len(start), n_pq=n_pq)
    stats = dict(bytes_in=int(byt_k[in_runs].sum()) if n_in else 0, landings=nL, n_ce=len(ct), n_inter_push=n_pq,
                 partial_out=len(pb) * SLOT_BYTES, n_fwd=len(jf))
    return tabs, dims, stats, np.ascontiguousarray(ce_tab)


# ---------------------------------------------------------------- forward tables (query-stationary executor)
TOKENS_PER_ROW = 8
FWD_TABLE_ORDER = ["trig_ptr", "trig_lane", "remote_sig", "row_grp_ptr", "grp_beg", "grp_end",
               "grp_dep", "grp_need", "row_qdep", "row_order", "sig_grp_ptr", "sig_grp", "sig_row_ptr", "sig_row", "red_need",
               "red_kind", "red_pay", "red_n", "sig_red_ptr", "sig_red"]


def input_plan(st, needs):
    """gateways of every (node, run) from the global needs matrix (every rank computes the same)."""
    rneed = np.logical_or.reduceat(needs, st["rs"], axis=1)
    gw = np.full((st["nn"], len(st["rs"])), -1, np.int64)
    for b_ in range(st["nn"]):
        gp, k, want, gwj = gateways(st, rneed, b_)
        gw[b_, k] = gp[gwj]
    return dict(rneed=rneed, gw=gw)


def fwd_region_layout(st):
    """byte offsets [sig, kv], total bytes, NSIG (= the backward region's prefix)"""
    al = lambda x: (x + 255) // 256 * 256                 # noqa: E731
    NSIG = st["n_run_sig"]
    return [0, al(8 * NSIG)], al(al(8 * NSIG) + st["nb"] * BLK * ROW_BYTES), NSIG


def fwd_scratch_layout(n_rows, n_remote, NSIG, n_workers):
    """int32 offsets of [sq, sq_tail, tile_counter, n_enq, pq, pq_tail, red_cnt, task_desc, row_ntask, grp_cnt, rows_done,
    sched_b0] + total (infllm_fwd_host.cpp)"""
    sizes = [n_rows + 2 * n_workers + 64, 1, 1, 1, 32, 1, 1, 4 * n_rows + 4, n_rows + 1, n_rows + 1, 1, 3 * 512 + 2]
    offs, o = [], 0
    for s in sizes:
        offs.append(o); o += (s + 31) // 32 * 32
    return offs + [o]


def recv_tables(st, rank, rneed_me, L):
    """the receiver / scheduler side, from THIS rank's needs only (rneed_me [nruns] bool): polled signals (the runs it
    receives), rows = (part, head, 8 tokens). grp_dep / grp_need / sig_grp_ptr / sig_grp come from the driver (GPU)."""
    gpn, nb, NG, rs, ro, node_of = st["gpn"], st["nb"], st["NG"], st["rs"], st["run_owner"], st["node_of"]
    sig_runs = lambda ks, d: ks * NG + np.where(node_of[rs[ks]] == d // gpn, d % gpn, gpn + d // gpn)   # noqa: E731
    in_runs = np.nonzero(rneed_me & (ro != rank))[0]
    n_in = len(in_runs)
    remote_sig = (sig_runs(in_runs, rank) if n_in else np.zeros(0, np.int64)).astype(np.int32)
    nrh = (L + TOKENS_PER_ROW - 1) // TOKENS_PER_ROW; n_rows = 4 * nrh
    I = lambda x: np.ascontiguousarray(np.asarray(x, np.int32))               # noqa: E731
    z1 = np.zeros(1, np.int32)
    tabs = dict(remote_sig=I(remote_sig) if n_in else z1,
                row_grp_ptr=I(np.arange(n_rows + 1)), grp_beg=I(np.arange(n_rows)), grp_end=I(np.arange(1, n_rows + 1)),
                grp_dep=z1, grp_need=z1, row_qdep=I(np.full(n_rows, -1)), row_order=I(np.arange(n_rows)),
                sig_grp_ptr=z1, sig_grp=z1, sig_row_ptr=I(np.zeros(n_in + 1)), sig_row=z1,
                red_need=z1, red_kind=z1, red_pay=z1, red_n=z1, sig_red_ptr=I(np.zeros(n_in + 1)), sig_red=z1,
                trig_ptr=I(np.zeros(n_in + 1)), trig_lane=I(np.ones(max(n_in, 1))))
    dims = dict(nb=nb, L=L, nrh=nrh, n_rows=n_rows, n_remote=n_in, n_pq=0)
    return tabs, dims, in_runs


def send_tables(st, G, rank, in_runs, offs):
    """the sender side, from the global plan: engine start messages (runs to other nodes' gateways), copy-engine rows
    (same-node runs, relay forwards), trig_lane (2 = forward by copy engine)."""
    P, gpn, NG, rs, r0, nr, ro, node_of = st["P"], st["gpn"], st["NG"], st["rs"], st["r0"], st["nr"], st["run_owner"], st["node_of"]
    gw, rneed = G["gw"], G["rneed"]
    me_n = rank // gpn
    nm = np.arange(me_n * gpn, min(P, me_n * gpn + gpn))
    kv_off = offs[1]
    sig_runs = lambda ks, d: ks * NG + np.where(node_of[rs[ks]] == d // gpn, d % gpn, gpn + d // gpn)   # noqa: E731
    n_in = len(in_runs)
    ko = np.nonzero(ro == rank)[0]
    src_k = kv_off + r0 * ROW_BYTES; byt_k = nr * ROW_BYTES
    kk, dd = np.nonzero(rneed[nm][:, ko].T & (nm != rank)[None, :])
    ks_, ds_ = ko[kk], nm[dd]
    ce_start = np.stack([src_k[ks_], src_k[ks_], byt_k[ks_], ds_, sig_runs(ks_, ds_), np.zeros(len(ks_), np.int64),
                         np.zeros(len(ks_), np.int64)], 1) if len(ks_) else np.zeros((0, 7), np.int64)
    kk, bb = np.nonzero((gw[:, ko].T >= 0) & (np.arange(st["nn"]) != me_n)[None, :])
    ks_, gs_ = ko[kk], gw[bb, ko[kk]]
    start = np.stack([src_k[ks_], src_k[ks_], byt_k[ks_], gs_, sig_runs(ks_, gs_)], 1) if len(ks_) else np.zeros((0, 5), np.int64)
    jf = np.nonzero(gw[me_n, in_runs] == rank)[0] if n_in else np.zeros(0, np.int64)
    jj, dd = np.nonzero(rneed[nm][:, in_runs[jf]].T & (nm != rank)[None, :]) if len(jf) else (np.zeros(0, np.int64),) * 2
    js_, ds_ = jf[jj], nm[dd]; kf = in_runs[js_]
    remote_sig = sig_runs(in_runs, rank) if n_in else np.zeros(0, np.int64)
    ce_fwd = np.stack([src_k[kf], src_k[kf], byt_k[kf], ds_, sig_runs(kf, ds_), np.ones(len(kf), np.int64),
                       remote_sig[js_]], 1) if len(kf) else np.zeros((0, 7), np.int64)
    trig_lane = np.ones(max(n_in, 1), np.int32); trig_lane[jf] = 2
    ct = np.concatenate([ce_start, ce_fwd]).astype(np.int64)
    ce_tab = np.stack([ct[:, 0], ct[:, 1], ct[:, 2], ct[:, 3] + (ct[:, 4] << 32), ct[:, 5] + (ct[:, 6] << 32),
                       np.ones(len(ct), np.int64)], 1) if len(ct) else np.zeros((0, 6), np.int64)
    mt = start.astype(np.int64)
    msg = np.stack([mt[:, 0], mt[:, 1], mt[:, 2], mt[:, 3] + (mt[:, 4] << 32)], 1) if len(mt) else np.zeros((0, 4), np.int64)
    stats = dict(bytes_in=int(byt_k[in_runs].sum()) if n_in else 0, n_ce=len(ct), n_fwd=len(jf), n_inter=len(start))
    return np.ascontiguousarray(msg), np.ascontiguousarray(ce_tab), trig_lane, stats


def own_blocks(chunks, S, P):
    own = np.empty((S + BLK - 1) // BLK, np.int64)
    for ci, (a_, b_) in enumerate(chunks):
        own[a_ // BLK:(b_ + BLK - 1) // BLK] = chunk_rank(ci, P)
    return own


# ================================================================ query-stationary executors (Dex without assigner)
class InfllmFwd:
    """forward: needs all-gather, receiver tables (GPU), sender tables (host, from the global needs), ONE persistent
    launch whose workers run the official row routine on (KV head, 8 tokens) rows; a row waits only for the remote runs
    its selection reads. Every item is split into a local part and a remote part (duplicated q rows) and the two
    partials are LSE-merged, so local work starts before any transfer lands."""

    def __init__(self, DA, chunks, my, S, gpn=4):
        self.DA, self.chunks, self.my, self.S, self.gpn = DA, chunks, my, S, gpn
        self.rank, self.P = dist.get_rank(), dist.get_world_size()
        self.dev = torch.device("cuda", torch.cuda.current_device())
        self.st = static_plan(own_blocks(chunks, S, self.P), chunks, S, gpn, RUN_MAX)
        self.grid = int(DA.infllm_fwd_grid())
        self.n_workers = self.grid - 1
        self.offs, total, self.NSIG = fwd_region_layout(self.st)
        self.region = DA.bwd_sym_alloc(int(total))                  # collective (shared with the backward)
        self.pe_delta, self.pe_inter = DA.bwd_pe_info()
        self.base = self.region.data_ptr()
        self.Ls = [chunks[ci][1] - chunks[ci][0] for ci in my]
        self.L = sum(self.Ls)
        self.cu_q = torch.tensor(np.concatenate([[0], np.cumsum(self.Ls) * GH]), dtype=torch.int32, device=self.dev)
        self.cu_k = torch.zeros(len(my) + 1, dtype=torch.int32, device=self.dev)
        self.seqused = torch.tensor([chunks[ci][1] for ci in my], dtype=torch.int32, device=self.dev)
        self.kv = self.region[self.offs[1]:self.offs[1] + self.st["nb"] * BLK * ROW_BYTES].view(torch.bfloat16).view(-1, 512)
        self.run_of_dev = torch.from_numpy(self.st["run_of"]).to(self.dev)
        self.is_local = torch.from_numpy((self.st["own"] == self.rank).astype(np.uint8)).to(self.dev).contiguous()
        self.tok8 = torch.arange(self.L, device=self.dev) // 8
        self.n_late = torch.zeros(1, dtype=torch.int32, device=self.dev)   # (kernel argument: no late-published messages)
        self.side = torch.cuda.Stream()
        # the local / remote split: segments [0, nseg) = the local parts, [nseg, 2 nseg) = the remote parts (duplicated q)
        self.cu_q2 = torch.cat([self.cu_q, self.cu_q[1:] + self.L * GH]).contiguous()
        self.cu_k2 = torch.zeros(2 * len(my) + 1, dtype=torch.int32, device=self.dev)
        self.seqused2 = torch.cat([self.seqused, self.seqused]).contiguous()
        self.o32 = torch.empty(2 * self.L * GH, 2, 128, dtype=torch.float32, device=self.dev)
        self.lse2 = torch.empty(2, 2 * self.L * GH, dtype=torch.float32, device=self.dev)
        self.scratch = None

    def forward(self, q, k_own, v_own, sel, scale, timer=None, o=None, cache=None):
        """one layer -> O bf16 [L * 16, 2, 128], LSE fp32 [2, L * 16]. The needs all-gather is issued first (async); the
        GPU prep (K / V staging, split bitmap, duplicated q) and the receiver tables (this rank's own needs, no host
        sync) follow; then the sender tables from the global needs, and the launch. cache (training step): an empty
        dict stores this call's plan, a filled one replays it (no exchange, no planning)."""
        DA, st, dev = self.DA, self.st, self.dev
        c = cache.get("plan") if cache is not None else None
        if c is not None:
            return self._forward_cached(q, k_own, v_own, c, scale, timer, o)
        # 1. needs (blocks each rank has a task on), async; mine on the host (runs)
        m = torch.zeros(st["nb"], dtype=torch.uint8, device=dev)
        x = sel.flatten().long()
        m.scatter_reduce_(0, x.clamp(min=0), (x >= 0).to(torch.uint8), reduce="amax")
        needs_g = torch.empty(self.P, st["nb"], dtype=torch.uint8, device=dev)
        work = dist.all_gather_into_tensor(needs_g, m, async_op=True)
        rneed_me = np.logical_or.reduceat(m.bool().cpu().numpy(), st["rs"])
        # 2. GPU prep that needs no plan
        r = 0
        for ci, L in zip(self.my, self.Ls):
            a_ = self.chunks[ci][0]
            self.kv[a_:a_ + L, :256] = k_own[r:r + L].reshape(L, -1)
            self.kv[a_:a_ + L, 256:] = v_own[r:r + L].reshape(L, -1)
            r += L
        bm = DA.infllm_split_bitmap(sel, self.is_local, st["nb"])
        q2 = torch.cat([q, q])
        # 3. the receiver side: polled signals, rows, dependencies (GPU, no host sync)
        tabs, dims, in_runs = recv_tables(st, self.rank, rneed_me, self.L)
        so = fwd_scratch_layout(dims["n_rows"], dims["n_remote"], self.NSIG, self.n_workers)
        if self.scratch is None or self.scratch.numel() < so[-1]:
            self.scratch = torch.zeros(so[-1], dtype=torch.int32, device=dev)
        dev_tabs = [torch.from_numpy(tabs[k]).to(dev, non_blocking=True) for k in FWD_TABLE_ORDER]
        nruns, nrh, n_rows, n_in = len(st["rs"]), dims["nrh"], dims["n_rows"], dims["n_remote"]
        run_j = torch.full((nruns,), -1, dtype=torch.int64, device=dev)
        if n_in:
            run_j[torch.from_numpy(in_runs).to(dev)] = torch.arange(n_in, device=dev)
        j = run_j[self.run_of_dev[sel.clamp(min=0).long()]]
        j = torch.where(sel >= 0, j, torch.full_like(j, -1))
        row = (2 * nrh + torch.arange(2, device=dev)[:, None] * nrh + (self.tok8)[None, :])[:, :, None].expand_as(j)
        msk = j >= 0
        dep = torch.zeros(n_in + 1, n_rows, dtype=torch.bool, device=dev)
        dep[j[msk], row[msk]] = True
        dep = dep[:n_in]
        sig_grp_ptr = torch.zeros(n_in + 1, dtype=torch.int32, device=dev)
        sig_grp_ptr[1:] = torch.cumsum(dep.sum(1), 0).to(torch.int32)
        pairs = torch.nonzero_static(dep, size=max(1, n_in * n_rows), fill_value=0)      # row-major = by signal, padded
        sig_grp = pairs[:, 1].to(torch.int32).contiguous()
        grp_need = dep.sum(0).to(torch.int32)
        grp_dep = torch.where(grp_need > 0, torch.zeros_like(grp_need), torch.full_like(grp_need, -1))
        for k_, t_ in (("grp_dep", grp_dep), ("grp_need", grp_need), ("sig_grp_ptr", sig_grp_ptr), ("sig_grp", sig_grp)):
            dev_tabs[FWD_TABLE_ORDER.index(k_)] = t_.contiguous()
        # 4. the sender side. The main stream waits for the needs all-gather (an NCCL kernel) so the persistent kernel
        #    cannot take every SM before it ran; the needs are copied to the host from the side stream, which first
        #    waits for the main stream (the copy must read the gathered needs).
        work.wait()
        main = torch.cuda.current_stream()
        with torch.cuda.stream(self.side):
            self.side.wait_stream(main)
            needs_c = needs_g.to("cpu", non_blocking=True)
        self.side.synchronize()
        needs = needs_c.bool().numpy()
        Gp = input_plan(st, needs)
        msg, ce, trig_lane, stats = send_tables(st, Gp, self.rank, in_runs, self.offs)
        dev_tabs[FWD_TABLE_ORDER.index("trig_lane")] = torch.from_numpy(trig_lane).to(dev)
        msg_dev = torch.from_numpy(msg.view(np.int32).reshape(-1) if len(msg) else np.zeros(8, np.int32)).to(dev)
        n_start, ce_t = len(msg), torch.from_numpy(ce)
        d = dims
        dd = [d["nb"], d["L"], int(sel.size(2)), 2 * len(self.my), d["nrh"], d["n_rows"], d["n_remote"], n_start,
              d["n_pq"], self.NSIG, self.grid]
        if timer is not None:
            timer.mark("plan", barrier=False)
        self.epoch = DA.infllm_fwd_run(self.base, list(self.offs), dd, [msg_dev] + dev_tabs, self.scratch, list(so),
                                       self.pe_delta, self.pe_inter, q2, q2, self.o32, self.lse2, self.cu_q2, self.cu_k2,
                                       self.seqused2, bm, max(self.Ls) * GH, self.S, scale, ce_t, self.n_late,
                                       torch.zeros(0, dtype=torch.int64, device=dev))
        if cache is not None:                                      # this call's plan, for the recompute of the same layer
            dt = list(dev_tabs)
            dt[FWD_TABLE_ORDER.index("trig_lane")] = torch.from_numpy(trig_lane).to(dev)
            cache["plan"] = dict(bm=bm, dims=dims, so=so, dev_tabs=dt, n_sel=int(sel.size(2)), stats=stats, needs=needs,
                                 msg_dev=torch.from_numpy(msg.view(np.int32).reshape(-1) if len(msg) else np.zeros(8, np.int32)).to(dev),
                                 n_start=len(msg), ce_t=torch.from_numpy(ce))
        out = merge_n(self.o32, self.lse2, 2, out_dtype=q.dtype)
        if o is not None:
            o.copy_(out); out = o
        l2 = torch.where(self.lse2 == float("inf"), torch.full_like(self.lse2, float("-inf")), self.lse2)   # +inf = empty part
        lse = torch.logaddexp(l2[:, :self.L * GH], l2[:, self.L * GH:])
        self.last = dict(dims=dims, stats=stats)
        return out, lse

    def _forward_cached(self, q, k_own, v_own, c, scale, timer, o):
        """forward() with a stored plan: K / V staging, the launch with every table and start message known, the merge
        -- the same computation as the call that planned it."""
        DA, dev = self.DA, self.dev
        r = 0
        for ci, L in zip(self.my, self.Ls):
            a_ = self.chunks[ci][0]
            self.kv[a_:a_ + L, :256] = k_own[r:r + L].reshape(L, -1)
            self.kv[a_:a_ + L, 256:] = v_own[r:r + L].reshape(L, -1)
            r += L
        q2 = torch.cat([q, q])
        so = c["so"]
        if self.scratch is None or self.scratch.numel() < so[-1]:
            self.scratch = torch.zeros(so[-1], dtype=torch.int32, device=dev)
        d = c["dims"]
        dd = [d["nb"], d["L"], c["n_sel"], 2 * len(self.my), d["nrh"], d["n_rows"], d["n_remote"], c["n_start"],
              d["n_pq"], self.NSIG, self.grid]
        if timer is not None:
            timer.mark("plan", barrier=False)
        # fresh copies of the stored tables: the kernel consumes some of them in place (dependency counters)
        tabs_ = [c["msg_dev"].clone()] + [t_.clone() for t_ in c["dev_tabs"]]
        self.epoch = DA.infllm_fwd_run(self.base, list(self.offs), dd, tabs_, self.scratch, list(so),
                                       self.pe_delta, self.pe_inter, q2, q2, self.o32, self.lse2, self.cu_q2, self.cu_k2,
                                       self.seqused2, c["bm"].clone(), max(self.Ls) * GH, self.S, scale, c["ce_t"],
                                       self.n_late, torch.zeros(0, dtype=torch.int64, device=dev))
        out = merge_n(self.o32, self.lse2, 2, out_dtype=q.dtype)
        if o is not None:
            o.copy_(out); out = o
        l2 = torch.where(self.lse2 == float("inf"), torch.full_like(self.lse2, float("-inf")), self.lse2)
        lse = torch.logaddexp(l2[:, :self.L * GH], l2[:, self.L * GH:])
        self.last = dict(dims=d, stats=c["stats"])
        return out, lse


class InfllmBwd:
    """backward: column tasks (segment, KV head, block) of the official backward routine; remote K / V runs as in the
    forward; partial dK / dV of a remote block are FP32-summed per node by the gateway and pushed once to the owner,
    which sums them into its own accumulator (landing slots in the symmetric region)."""

    def __init__(self, DA, chunks, my, S, gpn=4):
        self.DA, self.chunks, self.my, self.S, self.gpn = DA, chunks, my, S, gpn
        self.rank, self.P = dist.get_rank(), dist.get_world_size()
        self.dev = torch.device("cuda", torch.cuda.current_device())
        self.st = static_plan(own_blocks(chunks, S, self.P), chunks, S, gpn, RUN_MAX)
        self.grid = int(DA.infllm_bwd_grid())
        self.n_workers = self.grid - 1
        # worst-case landing count: every (receiver, block, sender) pair -> region sized once for the dense case
        self.NL_cap = global_plan(self.st, np.ones((self.P, self.st["nb"]), bool))["NL"]
        self.offs, total, self.NSIG = bwd_region_layout(self.st, self.NL_cap)
        self.region = DA.bwd_sym_alloc(int(total))                  # collective
        self.pe_delta, self.pe_inter = DA.bwd_pe_info()
        self.base = self.region.data_ptr()
        self.Ls = [chunks[ci][1] - chunks[ci][0] for ci in my]
        self.L = sum(self.Ls)
        self.cu_q = torch.tensor(np.concatenate([[0], np.cumsum(self.Ls) * GH]), dtype=torch.int32, device=self.dev)
        self.cu_k = torch.zeros(len(my) + 1, dtype=torch.int32, device=self.dev)
        self.seqused = torch.tensor([chunks[ci][1] for ci in my], dtype=torch.int32, device=self.dev)
        self.pos = torch.cat([torch.arange(*chunks[ci], device=self.dev) for ci in my]).to(torch.int32)
        b = len(my)
        self.dpsum = torch.empty(2, self.L * GH + 128 * b, dtype=torch.float32, device=self.dev)
        self.dq_accum = torch.empty(self.L * GH + 128 * b, 2, 128, dtype=torch.float32, device=self.dev)
        self.scratch = None
        self.kv = self.region[self.offs[1]:self.offs[1] + self.st["nb"] * BLK * ROW_BYTES].view(torch.bfloat16).view(-1, 512)

    def needs(self, sel):
        """[P, nb] bool: blocks each rank has a task on (any head, after the causal filter); one small all-gather."""
        m = torch.zeros(self.st["nb"], dtype=torch.uint8, device=self.dev)
        x = sel.flatten().long()
        m.scatter_reduce_(0, x.clamp(min=0), (x >= 0).to(torch.uint8), reduce="amax")
        out = torch.empty(self.P, self.st["nb"], dtype=torch.uint8, device=self.dev)
        dist.all_gather_into_tensor(out, m)
        return out.bool().cpu().numpy()

    def tasks(self, sel):
        """(b, h, n, cnt) of this rank's column tasks: tokens per (segment, head, block)."""
        nb, Ls = self.st["nb"], self.Ls
        out = [[], [], [], []]
        t0 = 0
        for b, L in enumerate(Ls):
            x = sel[:, t0:t0 + L].long()
            for h in range(2):
                v = x[h][x[h] >= 0]
                cnt = torch.bincount(v, minlength=nb).cpu().numpy()
                n = np.nonzero(cnt)[0]
                out[0].append(np.full(len(n), b)); out[1].append(np.full(len(n), h)); out[2].append(n); out[3].append(cnt[n])
            t0 += L
        return tuple(np.concatenate(o) for o in out)

    def plan_host(self, sel_cpu, needs):
        """the host part of plan(): tasks, global plan, this rank's tables, from a CPU copy of the selection and the
        forward's global needs matrix (the same filtered selection, so no exchange). Runs on the training step's worker
        thread ahead of the backward; no GPU work, no collective."""
        nb, Ls = self.st["nb"], self.Ls
        out = [[], [], [], []]
        t0 = 0
        x_all = sel_cpu.astype(np.int64)
        for b, L in enumerate(Ls):                                   # (= tasks(): tokens per (segment, head, block))
            x = x_all[:, t0:t0 + L]
            for h in range(2):
                v = x[h][x[h] >= 0]
                cnt = np.bincount(v, minlength=nb)
                n = np.nonzero(cnt)[0]
                out[0].append(np.full(len(n), b)); out[1].append(np.full(len(n), h)); out[2].append(n); out[3].append(cnt[n])
            t0 += L
        tasks = tuple(np.concatenate(o) for o in out)
        Gp = global_plan(self.st, needs)
        assert Gp["NL"] <= self.NL_cap
        tabs, dims, stats, ce = rank_tables(self.st, Gp, needs, self.rank, tasks, self.my, self.n_workers, self.offs)
        so, sn = bwd_scratch_layout(dims["n_tasks"], dims["nb"], dims["n_km"], dims["n_ka"], dims["n_pq"], self.n_workers)
        return dict(tabs_np=tabs, dims=dims, stats=stats, scr_offs=so, scr_n=sn, G=Gp, ce=torch.from_numpy(ce))

    def plan_finish(self, hp):
        """plan_host's result -> plan() format (tables to the device; the scratch grown if needed)"""
        if self.scratch is None or self.scratch.numel() < hp["scr_n"]:
            self.scratch = torch.zeros(hp["scr_n"], dtype=torch.int32, device=self.dev)
        return dict(tabs=[torch.from_numpy(hp["tabs_np"][k]).to(self.dev, non_blocking=True) for k in BWD_TABLE_ORDER],
                    dims=hp["dims"], stats=hp["stats"], scr_offs=hp["scr_offs"], G=hp["G"], ce=hp["ce"])

    def plan(self, sel, needs=None):
        """this rank's tables (numpy -> device) from the filtered selection; needs: the global matrix (else exchanged)."""
        if needs is None:
            needs = self.needs(sel)
        tasks = self.tasks(sel)
        Gp = global_plan(self.st, needs)
        assert Gp["NL"] <= self.NL_cap
        tabs, dims, stats, ce = rank_tables(self.st, Gp, needs, self.rank, tasks, self.my, self.n_workers, self.offs)
        so, sn = bwd_scratch_layout(dims["n_tasks"], dims["nb"], dims["n_km"], dims["n_ka"], dims["n_pq"], self.n_workers)
        if self.scratch is None or self.scratch.numel() < sn:
            self.scratch = torch.zeros(sn, dtype=torch.int32, device=self.dev)
        dev_tabs = [torch.from_numpy(tabs[k]).to(self.dev, non_blocking=True) for k in BWD_TABLE_ORDER]
        return dict(tabs=dev_tabs, dims=dims, stats=stats, scr_offs=so, G=Gp, ce=torch.from_numpy(ce))

    def run(self, q, o, do, lse, k_own, v_own, sel, plan, scale, dq=None, dk=None, dv=None):
        DA, st = self.DA, self.st
        assert q.shape == (self.L * GH, 2, 128) and lse.shape == (2, self.L * GH)
        dq = torch.empty_like(q) if dq is None else dq
        dk = torch.zeros_like(k_own) if dk is None else dk.zero_()        # blocks nobody selected stay zero
        dv = torch.zeros_like(v_own) if dv is None else dv.zero_()
        r = 0                                     # own K / V rows into the region (global layout)
        for ci, L in zip(self.my, self.Ls):
            a_ = self.chunks[ci][0]
            self.kv[a_:a_ + L, :256] = k_own[r:r + L].reshape(L, -1)
            self.kv[a_:a_ + L, 256:] = v_own[r:r + L].reshape(L, -1)
            r += L
        bm = DA.infllm_col_bitmap(sel.contiguous(), self.pos, st["nb"])                  # the column bitmap
        DA.infllm_bwd_prep(q, o, do, dq, lse, self.dpsum, self.dq_accum, self.cu_q, max(self.Ls) * GH, scale)   # official
        d = plan["dims"]
        dims = [d["nb"], d["n_tasks"], d["n_km"], d["n_ka"], d["n_remote"], d["n_start"], d["n_pq"], self.NSIG, self.NL_cap, self.grid]
        DA.infllm_bwd_run(self.base, list(self.offs), dims, plan["tabs"], self.scratch, list(plan["scr_offs"]),
                          self.pe_delta, self.pe_inter, q, o, do, dq, lse, self.dpsum, self.dq_accum, self.cu_q, self.cu_k,
                          self.seqused, bm, max(self.Ls) * GH, self.S, scale, dk, dv, plan["ce"])
        DA.infllm_bwd_convert_dq(q, o, do, dq, lse, self.dpsum, self.dq_accum, self.cu_q, max(self.Ls) * GH, scale)
        return dq, dk, dv


# ================================================================ executors with query movement (Dex)
class Plan2D:
    """the plan stage shared by the forward and the backward: planner input from the selection (entries per
    (group, query chunk, key chunk)), Dex's compiled planner on the two KV-head groups as one problem (block =
    chunk, zigzag owners), the home gate, rank 0's plan broadcast, entry GPUs, the global unit list and the needs /
    unit-union all-reduce. A unit = (computing GPU c, group g, query chunk qc) with some pair of (g, qc) placed on c
    other than the chunk's owner: its Q (+ bitmap rows) travel to c and its FP32 partial O (dQ) travel back.

    Joint planning: the executor moves K / V of BOTH KV-head groups for every needed key chunk (global K | V layout
    [S, 2, 128]), so the groups are planned as one problem: W = sum over groups of the entries [1, m, m], heads (32, 2)
    for the bytes and the backward model, kcols = per key chunk the max over groups; the single plan is applied to
    both groups (a query chunk's groups move together). Backward W is scaled so that the model's item term
    (b_item x 4 x hq x W) equals the measured official time per entry: W x T_ENTRY_BWD / (b_item x 4 x 32).
    Home gate: the plan's model time vs the home plan's (every pair on its query owner) under the same model (forward:
    node_time_b -> gpu_time_b; backward: bwd_gpu_time_b); home is kept unless home / plan > 1 + GATE_EPS. Evaluated on
    the GPU without a host sync (one CUDA graph per kind); rank 0's result travels in the plan broadcast."""
    _PLANNERS = {}

    def __init__(self, chunks, my, S, P, rank, dev, gpn=4):
        self.chunks, self.my, self.S, self.P, self.rank, self.dev, self.gpn = chunks, my, S, P, rank, dev, gpn
        self.m = len(chunks)                                                # plan blocks per group (= 2P chunks)
        self.nb = (S + BLK - 1) // BLK
        self.nbn = self.nb
        self.upr = (self.nbn + 63) // 64
        own = np.empty(self.nb, np.int64); bch = np.empty(self.nb, np.int64)
        for ci, (a_, b_) in enumerate(chunks):
            own[a_ // BLK:(b_ + BLK - 1) // BLK] = chunk_rank(ci, P); bch[a_ // BLK:(b_ + BLK - 1) // BLK] = ci
        self.own_blk = own
        self.cown = np.array([chunk_rank(ci, P) for ci in range(self.m)], np.int64)    # owner of every chunk
        self.clen = np.array([b_ - a_ for a_, b_ in chunks], np.int64)
        self.Ls = [chunks[ci][1] - chunks[ci][0] for ci in my]
        self.L = sum(self.Ls)
        self.tok0 = {ci: int(sum(self.Ls[:i])) for i, ci in enumerate(my)}         # token offset of my chunk ci
        self.blk_chunk = torch.as_tensor(bch, device=dev)
        self.tchunk = torch.cat([torch.full((L,), ci, dtype=torch.long, device=dev) for ci, L in zip(my, self.Ls)])
        self.tloc = torch.cat([torch.full((L,), i, dtype=torch.long, device=dev) for i, L in enumerate(self.Ls)])
        self.ob = torch.as_tensor(np.tile(self.cown, (G, 1)), device=dev)              # [G, m] owner per plan block
        self.gid = torch.arange(G * self.m, device=dev).view(G, self.m)
        self.B = float(np.mean(self.clen))
        self.NN = max(1, P // gpn)
        self.tplan = {}
        self.TIMING = False

    # ------------------------------------------------------------------ plan-stage timing (synchronised; operator tests only)
    def _t0(self):
        if self.TIMING:
            torch.cuda.synchronize()
        return time.perf_counter()

    def _tick(self, k, t0):
        if not self.TIMING:
            return 0.0
        torch.cuda.synchronize(); t1 = time.perf_counter(); self.tplan[k] = self.tplan.get(k, 0.0) + (t1 - t0) * 1e3
        return t1

    # ------------------------------------------------------------------ planner input + plan
    def stats(self, sel):
        """per rank: entries per (g, own chunk i, key chunk) and the selected blocks per group -> all-gathered ->
        W [G, m, m] (entries, query chunk x key chunk) and kblk [G, nb] (blocks selected by anyone)"""
        P, m, nb, dev = self.P, self.m, self.nb, self.dev
        ok = sel >= 0
        s_ = sel.clamp(min=0).long()
        kc = self.blk_chunk[s_]
        g_ = torch.arange(G, device=dev)[:, None, None].expand_as(s_)
        idx = torch.where(ok, (g_ * 2 + self.tloc[None, :, None]) * m + kc, G * 2 * m)      # (invalid -> dump slot,
        cnt = torch.bincount(idx.reshape(-1), minlength=G * 2 * m + 1)[:G * 2 * m].to(torch.int32)   # no host sync)
        bm = torch.zeros(G * nb + 1, dtype=torch.uint8, device=dev)
        bm[torch.where(ok, g_ * nb + s_, G * nb).reshape(-1)] = 1
        bm = bm[:G * nb]
        nb4 = (G * nb + 3) // 4
        buf = torch.zeros(G * 2 * m + nb4, dtype=torch.int32, device=dev)
        buf[:G * 2 * m] = cnt
        buf.view(torch.uint8)[4 * G * 2 * m:4 * G * 2 * m + G * nb] = bm
        out = torch.empty(P, buf.numel(), dtype=torch.int32, device=dev)
        dist.all_gather_into_tensor(out, buf)
        W = torch.zeros(G, m, m, dtype=torch.float32, device=dev)
        my_all = torch.as_tensor([[r, 2 * P - 1 - r] for r in range(P)], device=dev)     # [P, 2] chunks of rank r
        c_ = out[:, :G * 2 * m].view(P, G, 2, m).float()
        W[:, my_all.reshape(-1), :] = c_.permute(1, 0, 2, 3).reshape(G, P * 2, m)
        kblk = out.view(torch.uint8)[:, 4 * G * 2 * m:4 * G * 2 * m + G * nb].view(P, G, nb).amax(0)
        return W, kblk

    def plan(self, W, kblk, kinds):
        """{kind: pb [G, m, m] long} (GPU of every (group, query chunk, key chunk)), identical on every rank"""
        from dex import planner as pl
        from dex.planner import models as M_
        m, dev, gpn, NN = self.m, self.dev, self.gpn, self.NN
        HJ = (G * GH, G)
        t0 = self._t0()
        Wj = W.sum(0, keepdim=True)                                         # [1, m, m] entries of both groups
        kcols = torch.zeros(G, m, device=dev).index_add_(1, self.blk_chunk, kblk.view(G, self.nb).float()) / 2.0
        kcj = kcols.amax(0, keepdim=True)                                   # [1, m] (tile columns, max over groups)
        Qb, Ob, Kb = pl.fwd_bytes(HJ, self.B, D)
        coef = M_.BWD_COEF[HJ]
        if not hasattr(self, "_gate_g"):                                    # (persistent: the gate's CUDA graphs
            self._gate_g = {}                                               # read these tensors on every replay)
            self._ob1 = self.ob[:1].contiguous()
            self._homef = self._ob1[:, :, None].expand(1, m, m).contiguous()
        ob1, homef = self._ob1, self._homef
        gid1 = torch.arange(m, device=dev).view(1, m)
        out, tms = [], []
        for k_ in kinds:
            bwd = k_ == "bwd"
            key = (k_, "joint", m, NN)
            if key not in Plan2D._PLANNERS:
                Plan2D._PLANNERS[key] = pl.Planner(1, m, gpn, NN, Qb, Ob, Kb, T_ENTRY_FWD,
                                                   bwd=(HJ, self.B, kcj) if bwd else None)
            Gp = Plan2D._PLANNERS[key]
            if bwd:
                Gp.retarget(Qb, Ob, Kb, T_ENTRY_FWD, B=self.B, kcols=kcj)
                Wk = Wj * (T_ENTRY_BWD / (coef[1] * 4 * HJ[0]))             # item term = the official column time
            else:
                Gp.retarget(Qb, Ob, Kb, T_ENTRY_FWD)
                Wk = Wj
            pb = Gp(Wk, ob1, gid1).clone()                                  # [1, m, m]
            t0 = self._tick(f"planner_{k_}_ms", t0)

            def gate(Wk, pb, kcj, Wj, k_=k_, bwd=bwd):
                Wn = Wj > 0
                home = torch.where(Wn, homef, torch.full_like(homef, -1))   # (model view: live pairs only)
                dp = torch.stack([home, torch.where(Wn, pb, torch.full_like(pb, -1))])     # [2 (home, plan), 1, m, m]
                if bwd:
                    F = M_.bwd_gpu_time_b(Wk, dp, ob1, gpn, NN, None, M_.bwd_bytes(HJ, self.B, D), coef, kcj, HJ)[0].amax(1)
                else:
                    NA = torch.where(dp >= 0, dp // gpn, dp)
                    _, _, arr = M_.node_time_b(Wk, NA, ob1 // gpn, NN, gpn, Qb, Ob, Kb, T_ENTRY_FWD)
                    F = M_.gpu_time_b(Wk, dp, ob1, gpn, NN, arr, Qb, Ob, Kb, T_ENTRY_FWD)[0].amax(1)
                move = F[0] > F[1] * (1.0 + GATE_EPS[k_])              # (0-d GPU bool: no sync)
                mv = Wn & (pb != homef)                                     # (the planner's raw plan: live pairs off
                st_ = torch.stack([F[0].double() * 1e9, F[1].double() * 1e9, mv.sum().double(),   # their owner, entries
                                   (Wj * mv).sum().double(), (Wj * mv * ((pb // gpn) != (homef // gpn))).sum().double()])   # moved, cross-node)
                return torch.where(move, pb, homef), st_
            gk = self._gate_g.get(k_)                                       # (one CUDA graph per kind: the model
            if gk is None:                                                  # evaluation is many small kernels)
                from dex.planner.plan import _capture
                bufs = (Wk.clone(), pb.clone(), kcj.clone(), Wj.clone())
                gk = self._gate_g[k_] = (_capture(lambda: gate(*bufs)), bufs)
            (g_, (pbf, st_)), bufs = gk
            for b_, x_ in zip(bufs, (Wk, pb, kcj, Wj)):
                b_.copy_(x_)
            g_.replay()
            pbf, st_ = pbf.clone(), st_.clone()
            out.append(pbf.expand(G, m, m))
            tms.append(st_)
            t0 = self._tick(f"gate_{k_}_ms", t0)
        K_ = len(kinds)
        buf = torch.cat([torch.stack(out).reshape(-1), torch.stack(tms).reshape(-1).round().long()])
        dist.broadcast(buf, 0)                                              # (one broadcast: plans + model times)
        self._tick("plan_bcast_ms", t0)
        pbs = buf[:K_ * G * m * m].view(K_, G, m, m)
        self._gate_raw = (tuple(kinds), buf[K_ * G * m * m:].view(K_, 5), pbs, Wj.sum())
        return {k_: pbs[i] for i, k_ in enumerate(kinds)}

    def gate_info(self):
        """(host read, outside timed windows) the last plan call: per kind model times of the home plan and of the
        planner's plan (ms), their ratio, whether home was kept, pairs moved off their query owner"""
        if getattr(self, "_gate_raw", None) is None:
            return None
        kinds, t, pbs, wsum = self._gate_raw
        t = t.cpu().numpy().astype(np.float64); t[:, :2] /= 1e6; wsum = max(float(wsum), 1.0)
        own = self.ob[:, :, None].expand_as(pbs[0])
        res = {}
        for i, k_ in enumerate(kinds):
            moved = int((pbs[i] != own).sum().item())
            res[k_] = dict(home_ms=round(float(t[i, 0]), 4), plan_ms=round(float(t[i, 1]), 4),
                           ratio=round(float(t[i, 0] / max(t[i, 1], 1e-12)), 5), eps=GATE_EPS[k_],
                           kept_home=bool(moved == 0), pairs_off_owner=moved, raw_pairs_moved=int(t[i, 2]),
                           raw_entries_moved_frac=round(float(t[i, 3]) / wsum, 4), raw_cross_node_frac=round(float(t[i, 4]) / wsum, 4))
        return res

    # ------------------------------------------------------------------ units + exchange
    def units(self, pb):
        """candidate units from the plan: (c, g, qc) with some pair of (g, qc) on c != owner(qc), sorted; unit_id [P, G, m]"""
        P, m = self.P, self.m
        cand = torch.zeros(P, G, m, dtype=torch.bool, device=self.dev)
        g_, q_, k_ = torch.meshgrid(torch.arange(G, device=self.dev), torch.arange(m, device=self.dev),
                                    torch.arange(m, device=self.dev), indexing="ij")
        cand[pb.reshape(-1), g_.reshape(-1), q_.reshape(-1)] = True
        qcs = torch.arange(m, device=self.dev).repeat(G); gs = torch.arange(G, device=self.dev).repeat_interleave(m)
        cand[torch.as_tensor(self.cown, device=self.dev)[qcs], gs, qcs] = False         # (the owner itself: not a unit)
        U = torch.nonzero(cand)                                                # sorted (c, g, qc)
        uid = torch.full((P, G, m), -1, dtype=torch.long, device=self.dev)
        if len(U):
            uid[U[:, 0], U[:, 1], U[:, 2]] = torch.arange(len(U), device=self.dev)
        return U, uid

    def exchange(self, sel, pb):
        """entry GPUs e [2, L, K] int16 and ONE all-reduce(MAX) of X [P + NU, nb] u8: rows < P = blocks GPU c computes
        with, row P + u = blocks of candidate unit u -> (e, host needs [P, nb] bool, the units kept (c, g, qc, union))"""
        return self.exchange_multi(sel, [pb])[0]

    def exchange_multi(self, sel, pbs):
        """exchange() of several plans (the training step: forward + backward) in ONE all-reduce"""
        P, nb, dev = self.P, self.nb, self.dev
        ok = sel >= 0
        s_ = sel.clamp(min=0).long()
        g_ = torch.arange(G, device=dev)[:, None, None].expand_as(s_)
        qc = self.tchunk[None, :, None].expand_as(s_)
        kcs = self.blk_chunk[s_]
        parts, meta = [], []
        for pb in pbs:
            U, uid = self.units(pb)
            NU = len(U)
            e = pb[g_, qc, kcs]                                                # [2, L, K] entry GPU
            X = torch.zeros((P + NU) * nb + 1, dtype=torch.uint8, device=dev)   # (+1: dump slot, no host sync)
            X[torch.where(ok, e * nb + s_, (P + NU) * nb).reshape(-1)] = 1
            if NU:
                u = uid[e, g_, qc]
                fo = ok & (e != self.rank)
                X[torch.where(fo, (P + u) * nb + s_, (P + NU) * nb).reshape(-1)] = 1
            parts.append(X[:-1]); meta.append((U, NU, e))
        Xa = torch.cat(parts) if len(parts) > 1 else parts[0]
        dist.all_reduce(Xa, op=dist.ReduceOp.MAX)
        Xall = Xa.cpu().numpy().astype(bool)
        out, o = [], 0
        for U, NU, e in meta:
            Xh = Xall[o:o + (P + NU) * nb].reshape(P + NU, nb); o += (P + NU) * nb
            needs = Xh[:P]
            Uh = U.cpu().numpy()
            keep = Xh[P:].any(1) if NU else np.zeros(0, bool)
            units = dict(c=Uh[keep, 0] if NU else np.zeros(0, np.int64), g=Uh[keep, 1] if NU else np.zeros(0, np.int64),
                         qc=Uh[keep, 2] if NU else np.zeros(0, np.int64), union=Xh[P:][keep] if NU else np.zeros((0, nb), bool))
            units["o"] = self.cown[units["qc"]]
            units["len"] = self.clen[units["qc"]]
            units["end"] = np.array([self.chunks[c_][1] for c_ in units["qc"]], np.int64)
            out.append((e.to(torch.int16), needs, units))
        return out


# ---------------------------------------------------------------------------------------------------- forward
class InfllmFwd2D:
    """the forward with query movement. Region: [sig | kv | per-rank 2-D area (Q send / Q landing / partial O out /
    partial landing)]; the region is allocated once at the cap (NVSHMEM_SYMMETRIC_SIZE x 0.85, shared with
    InfllmBwd2D) and every call's layout is checked against it. The workers run the official row routine on own rows
    and on the rows of incoming units; the owner LSE-merges the returned FP32 partials."""

    def __init__(self, DA, pl, cap=None):
        self.DA, self.pl = DA, pl
        self.chunks, self.my, self.S, self.gpn = pl.chunks, pl.my, pl.S, pl.gpn
        self.rank, self.P, self.dev = pl.rank, pl.P, pl.dev
        self.st = static_plan(pl.own_blk, pl.chunks, pl.S, pl.gpn, RUN_MAX)
        self.grid = int(DA.infllm2d_fwd_grid())
        self.n_workers = self.grid - 1
        self.NSIG0 = self.st["n_run_sig"]
        self.NUmax = self.P * G * pl.m
        self.NSIG = self.NSIG0 + 2 * self.NUmax
        self.offs = [0, AL(8 * self.NSIG)]
        self.a2d = AL(self.offs[1] + self.st["nb"] * BLK * ROW_BYTES)          # the per-rank 2-D area starts here
        self.cap = cap or int(int(os.environ.get("NVSHMEM_SYMMETRIC_SIZE", str(4 << 30))) * 0.85)
        self.region = DA.bwd_sym_alloc(self.cap)                              # (collective) shared with the backward
        self.pe_delta, self.pe_inter = DA.bwd_pe_info()
        self.base = self.region.data_ptr()
        self.Ls, self.L = pl.Ls, pl.L
        dev = self.dev
        self.cu_q = torch.tensor(np.concatenate([[0], np.cumsum(self.Ls) * GH]), dtype=torch.int32, device=dev)
        self.kv = self.region[self.offs[1]:self.offs[1] + self.st["nb"] * BLK * ROW_BYTES].view(torch.bfloat16).view(-1, 512)
        self.run_of_dev = torch.from_numpy(self.st["run_of"]).to(dev)
        self.is_local = torch.from_numpy((self.st["own"] == self.rank).astype(np.uint8)).to(dev).contiguous()
        self.tok8 = torch.arange(self.L, device=dev) // 8
        self.cu_q2 = torch.cat([self.cu_q, self.cu_q[1:] + self.L * GH]).contiguous()
        self.cu_k2 = torch.zeros(2 * len(self.my) + 1, dtype=torch.int32, device=dev)
        self.seqused = torch.tensor([self.chunks[ci][1] for ci in self.my], dtype=torch.int32, device=dev)
        self.seqused2 = torch.cat([self.seqused, self.seqused]).contiguous()
        self.o32 = torch.empty(2 * self.L * GH, 2, 128, dtype=torch.float32, device=dev)
        self.lse2 = torch.empty(2, 2 * self.L * GH, dtype=torch.float32, device=dev)
        self.fpu = torch.empty(1, dtype=torch.uint8, device=dev)
        self.scratch = None
        self.used = 0
        self.last = {}

    # per-unit byte sizes
    def qslot(self, n):                       # landed bitmap rows + Q (bf16, one group)
        return AL(n * self.pl.upr * 8) + AL(n * GH * D * 2)

    def oslot(self, n):                       # FP32 partial O + LSE (one group)
        return AL(n * GH * D * 4) + AL(n * GH * 4)

    def layout(self, units):
        """per rank: [Q send (units it owns) | Q landing (units it computes) | partial O out (it computes) | partial
        landing (it owns)] after a2d; -> dict of per-unit offsets (identical on every rank) and the region bytes used"""
        P = self.P
        c, o, ln = units["c"], units["o"], units["len"]
        n = len(c)
        qs, ql, oo, ol = (np.zeros(n, np.int64) for _ in range(4))
        top = 0
        for r in range(P):
            x = self.a2d
            for u in np.nonzero(o == r)[0]:
                qs[u] = x; x += self.qslot(ln[u])
            for u in np.nonzero(c == r)[0]:
                ql[u] = x; x += self.qslot(ln[u])
            for u in np.nonzero(c == r)[0]:
                oo[u] = x; x += self.oslot(ln[u])
            for u in np.nonzero(o == r)[0]:
                ol[u] = x; x += self.oslot(ln[u])
            top = max(top, x)
        if top > self.cap:                                                    # (the caller runs the home plan)
            return None
        self.used = max(self.used, top)
        return dict(qs=qs, ql=ql, oo=oo, ol=ol, top=top)

    def forward(self, q, k_own, v_own, sel, scale, timer=None, o=None, cache=None, plans=None, pre=None):
        """one layer -> O bf16 [L * 16, 2, 128], LSE fp32 [2, L * 16]. plans: {'fwd': pb} given (else planned here);
        cache (training-step reuse): an empty dict stores this call's plan (tables, units, pb), a filled one replays it."""
        pl = self.pl
        c_ = cache.get("plan2d") if cache is not None else None
        if c_ is None:
            t0 = pl._t0()
            if plans is None:
                W, kblk = pl.stats(sel)
                t0 = pl._tick("stats_ms", t0)
                plans = pl.plan(W, kblk, ("fwd",))
                t0 = pl._t0()
            pb = plans["fwd"]
            e, needs, units = pre if pre is not None else pl.exchange(sel, pb)      # (pre: the joint exchange of a training step)
            if self.layout(units) is None:                                    # memory cap: the plan's buffers do not fit
                self.cap_home_plans = getattr(self, "cap_home_plans", 0) + 1    # the region; this call runs the home plan
                pb = pl.ob[:, :, None].expand_as(pb).contiguous()             # (every rank decides the same: the layout
                e, needs, units = pl.exchange(sel, pb)                        # is global); counted
            t0 = pl._tick("exchange_ms", t0)
            c_ = self._tables(sel, e, needs, units, pb)
            pl._tick("tables_ms", t0)
            if cache is not None:
                cache["plan2d"] = c_
        return self._launch(q, k_own, v_own, sel, scale, c_, timer, o)

    def _tables(self, sel, e, needs, units, pb):
        """everything of one call that depends only on the selection and the plan (reused by the recompute)"""
        DA, st, dev, rank = self.DA, self.st, self.dev, self.rank
        lay = self.layout(units)
        NU = len(units["c"])
        mine_in = np.nonzero(units["c"] == rank)[0]                            # units I compute (foreign rows)
        mine_out = np.nonzero(units["o"] == rank)[0]                           # units I own (Q out, partial in)
        # own rows: the selection restricted to the entries on this rank
        sel_own = torch.where(e == rank, sel, torch.full_like(sel, -1)).contiguous()
        bm = DA.infllm2d_split_bitmap(sel_own, self.is_local, st["nb"])
        # ---- receiver side (existing recv_tables for the own rows)
        rneed_me = np.logical_or.reduceat(needs[rank], st["rs"])
        tabs, dims, in_runs = recv_tables(st, rank, rneed_me, self.L)
        n_in, nrh, n_own_rows = dims["n_remote"], dims["nrh"], dims["n_rows"]
        n_qi, n_oo = len(mine_in), len(mine_out)
        lens_in = units["len"][mine_in]
        nr_u = (lens_in + 7) // 8
        n_rows = n_own_rows + int(nr_u.sum())
        remote_sig = np.concatenate([tabs["remote_sig"][:n_in] if n_in else np.zeros(0, np.int32),
                                     self.NSIG0 + mine_in, self.NSIG0 + self.NUmax + mine_out]).astype(np.int32)
        nsig = len(remote_sig)
        # foreign row descriptors
        fr_u = np.repeat(np.arange(n_qi), nr_u)
        fr_m0 = np.concatenate([np.arange(0, ln_, 8) for ln_ in lens_in]) if n_qi else np.zeros(0, np.int64)
        fr_m1 = np.minimum(fr_m0 + 8, np.repeat(lens_in, nr_u)) if n_qi else np.zeros(0, np.int64)
        frow = np.stack([fr_u, fr_m0, fr_m1], 1).astype(np.int32).reshape(-1) if n_qi else np.zeros(3, np.int32)
        unit_need = nr_u.astype(np.int32) if n_qi else np.zeros(1, np.int32)
        frow_base = n_own_rows + np.concatenate([[0], np.cumsum(nr_u)[:-1]]) if n_qi else np.zeros(0, np.int64)
        # dependencies: own rows from the restricted selection (GPU, as InfllmFwd); foreign rows: every remote run
        # of the unit's block union
        nruns = len(st["rs"])
        run_j = torch.full((nruns,), -1, dtype=torch.int64, device=dev)
        if n_in:
            run_j[torch.from_numpy(in_runs).to(dev)] = torch.arange(n_in, device=dev)
        j = run_j[self.run_of_dev[sel_own.clamp(min=0).long()]]
        j = torch.where(sel_own >= 0, j, torch.full_like(j, -1))
        row = (2 * nrh + torch.arange(2, device=dev)[:, None] * nrh + (self.tok8)[None, :])[:, :, None].expand_as(j)
        msk = j >= 0
        dep = torch.zeros(n_in + 1, n_rows, dtype=torch.bool, device=dev)
        dep[j[msk], row[msk]] = True
        jr = np.full(nruns, -1, np.int64)
        if n_in:
            jr[in_runs] = np.arange(n_in)
        fj, frr = [], []
        for k, u in enumerate(mine_in):
            ub = np.nonzero(units["union"][u])[0]
            jj = np.unique(jr[st["run_of"][ub]]); jj = jj[jj >= 0]
            rr = frow_base[k] + np.arange(nr_u[k])
            fj.append(np.repeat(jj, len(rr))); frr.append(np.tile(rr, len(jj)))
        if fj and sum(len(x) for x in fj):
            dep[torch.from_numpy(np.concatenate(fj)).to(dev), torch.from_numpy(np.concatenate(frr)).to(dev)] = True
        dep = dep[:n_in]
        sig_grp_ptr = torch.zeros(nsig + 1, dtype=torch.int32, device=dev)
        sig_grp_ptr[1:n_in + 1] = torch.cumsum(dep.sum(1), 0).to(torch.int32)
        sig_grp_ptr[n_in + 1:] = sig_grp_ptr[n_in]
        pairs = torch.nonzero_static(dep, size=max(1, n_in * n_rows), fill_value=0)
        sig_grp = pairs[:, 1].to(torch.int32).contiguous()
        grp_need = dep.sum(0).to(torch.int32)
        grp_dep = torch.where(grp_need > 0, torch.zeros_like(grp_need), torch.full_like(grp_need, -1))
        row_qdep = np.full(n_rows, -1, np.int32)
        sig_row_ptr = np.zeros(nsig + 1, np.int64); sig_row = []
        for k in range(n_qi):
            rr = frow_base[k] + np.arange(nr_u[k])
            row_qdep[rr] = n_in + k
            sig_row.append(rr)
        cnt_row = np.zeros(nsig, np.int64)
        cnt_row[n_in:n_in + n_qi] = nr_u
        sig_row_ptr[1:] = np.cumsum(cnt_row)
        sig_row = np.concatenate(sig_row).astype(np.int32) if n_qi else np.zeros(1, np.int32)
        row_order = np.concatenate([n_own_rows + np.arange(n_rows - n_own_rows), np.arange(n_own_rows)]).astype(np.int32)
        # partial-landing units (no-op): signal n_in + n_qi + l -> unit l
        red_need = np.ones(max(n_oo, 1), np.int32); red_kind = np.full(max(n_oo, 1), K_NOP, np.int32)
        red_pay = np.zeros(max(n_oo, 1), np.int32); red_n = np.ones(max(n_oo, 1), np.int32)
        sig_red_ptr = np.zeros(nsig + 1, np.int64)
        sig_red_ptr[n_in + n_qi + 1:] = np.arange(1, n_oo + 1)
        sig_red = np.arange(n_oo, dtype=np.int32) if n_oo else np.zeros(1, np.int32)
        # ---- sender side: KV runs (existing send_tables; gateways from the global needs), Q sends, partial returns
        Gk = input_plan(st, needs)
        msg_kv, ce_kv, trig_lane_kv, stats = send_tables(st, Gk, rank, in_runs, self.offs)
        me_n = rank // self.gpn
        same = lambda d: d // self.gpn == me_n                                 # noqa: E731
        start_q, ce_q = [], []
        for u in mine_out:                                                     # Q (+ bitmap rows) of my units, start
            c = int(units["c"][u]); m_ = (lay["qs"][u], lay["ql"][u], self.qslot(units["len"][u]), c, self.NSIG0 + u)
            (ce_q.append(m_ + (0, 0)) if same(c) else start_q.append(m_))
        n_trig = nsig + max(n_qi, 1)
        trig_lane = np.ones(n_trig, np.int32)
        trig_lane[:len(trig_lane_kv)] = trig_lane_kv[:min(len(trig_lane_kv), n_trig)]
        trig_msgs, ce_ret = [], []
        trig_ptr = np.zeros(n_trig + 1, np.int64)
        n_pq = 0
        for k, u in enumerate(mine_in):                                        # partial returns: trigger nsig + k
            o_ = int(units["o"][u]); t_ = nsig + k
            m_ = (lay["oo"][u], lay["ol"][u], self.oslot(units["len"][u]), o_, self.NSIG0 + self.NUmax + u)
            if same(o_):
                ce_ret.append(m_ + (2, t_)); trig_lane[t_] = 2
            else:
                trig_msgs.append((t_, m_)); n_pq += 1
        start = [tuple(x) for x in start_q] + [tuple(r_[:3]) + (int(r_[3]) & 0xffffffff, int(r_[3]) >> 32) for r_ in msg_kv]
        n_start = len(start)
        cnt_t = np.zeros(n_trig, np.int64)
        for t_, _ in trig_msgs:
            cnt_t[t_] += 1
        trig_ptr[0] = n_start; trig_ptr[1:] = n_start + np.cumsum(cnt_t)
        allm = start + [m_ for _, m_ in sorted(trig_msgs)]
        mt = np.asarray(allm, np.int64).reshape(-1, 5) if allm else np.zeros((0, 5), np.int64)
        msg = np.stack([mt[:, 0], mt[:, 1], mt[:, 2], mt[:, 3] + (mt[:, 4] << 32)], 1) if len(mt) else np.zeros((1, 4), np.int64)
        def ce6(rows):                                                          # 7-tuples -> rt::CeMsg rows
            r7 = np.asarray(rows, np.int64).reshape(-1, 7)
            return np.stack([r7[:, 0], r7[:, 1], r7[:, 2], r7[:, 3] + (r7[:, 4] << 32), r7[:, 5] + (r7[:, 6] << 32),
                             np.ones(len(r7), np.int64)], 1)
        ce = np.concatenate([ce6(ce_q), np.asarray(ce_kv, np.int64).reshape(-1, 6), ce6(ce_ret)], 0)
        I = lambda x: torch.from_numpy(np.ascontiguousarray(np.asarray(x, np.int32).reshape(-1) if np.asarray(x).size else np.zeros(1, np.int32))).to(dev)   # noqa: E731
        T = dict(trig_ptr=I(trig_ptr), trig_lane=I(trig_lane), remote_sig=I(remote_sig),
                 row_grp_ptr=I(np.arange(n_rows + 1)), grp_beg=I(np.arange(n_rows)), grp_end=I(np.arange(1, n_rows + 1)),
                 grp_dep=grp_dep.contiguous(), grp_need=grp_need.contiguous(), row_qdep=I(row_qdep), row_order=I(row_order),
                 sig_grp_ptr=sig_grp_ptr.contiguous(), sig_grp=sig_grp, sig_row_ptr=I(sig_row_ptr), sig_row=I(sig_row),
                 red_need=I(red_need), red_kind=I(red_kind), red_pay=I(red_pay), red_n=I(red_n), sig_red_ptr=I(sig_red_ptr),
                 sig_red=I(sig_red))
        dev_tabs = [torch.from_numpy(np.ascontiguousarray(msg).view(np.int32).reshape(-1)).to(dev)] + \
                   [T[k_] for k_ in FWD_TABLE_ORDER] + [I(frow), I(unit_need)]
        # incoming units' parameter blocks: (q_off, bm_off, o32_off, lse_off, g, len, nbn, max_seqlen_k)
        ln = units["len"]
        uc = np.stack([lay["ql"][mine_in] + AL(ln[mine_in] * self.pl.upr * 8), lay["ql"][mine_in], lay["oo"][mine_in],
                       lay["oo"][mine_in] + AL(ln[mine_in] * GH * D * 4), units["g"][mine_in], ln[mine_in],
                       np.full(n_qi, self.pl.nbn), np.full(n_qi, self.S)], 1).astype(np.int64) if n_qi else np.zeros((0, 8), np.int64)
        ucu = torch.from_numpy(np.stack([np.zeros(n_qi), ln[mine_in] * GH, np.zeros(n_qi), np.zeros(n_qi),
                                         units["end"][mine_in]], 1).astype(np.int32) if n_qi else np.zeros((1, 5), np.int32)).to(dev)
        n_red = n_oo
        so = fwd2d_scratch_layout(n_rows, nsig, n_red, self.n_workers, n_qi, n_pq)
        dd = [st["nb"], self.L, int(sel.size(2)), 2 * len(self.my), nrh, n_rows, nsig, n_start, n_pq, self.NSIG, self.grid,
              n_own_rows, n_red, nsig, n_qi]
        stats.update(units=NU, units_in=n_qi, units_out=n_oo, q_bytes_out=int(sum(self.qslot(units["len"][u]) for u in mine_out)),
                     o_bytes_in=int(sum(self.oslot(units["len"][u]) for u in mine_out)),
                     kv_bytes_in=stats["bytes_in"], region_top=lay["top"], cap_home_plans=getattr(self, "cap_home_plans", 0))
        return dict(bm=bm, dims=dd, so=so, dev_tabs=dev_tabs, ce=torch.from_numpy(np.ascontiguousarray(ce, np.int64)),
                    units=units, lay=lay, mine_in=mine_in, mine_out=mine_out, uc=torch.from_numpy(uc), ucu=ucu,
                    sel_e=(sel, e), stats=stats, pb=pb)

    def _launch(self, q, k_own, v_own, sel, scale, c_, timer, o):
        DA, dev, pl = self.DA, self.dev, self.pl
        units, lay, e = c_["units"], c_["lay"], c_["sel_e"][1]
        r = 0
        for ci, L in zip(self.my, self.Ls):                                   # own K / V rows into the region
            a_ = self.chunks[ci][0]
            self.kv[a_:a_ + L, :256] = k_own[r:r + L].reshape(L, -1)
            self.kv[a_:a_ + L, 256:] = v_own[r:r + L].reshape(L, -1)
            r += L
        reg = self.region
        for u in c_["mine_out"]:                                              # outgoing units: bitmap rows + Q (group g)
            g, qc, ln_, cdst = int(units["g"][u]), int(units["qc"][u]), int(units["len"][u]), int(units["c"][u])
            t0_ = pl.tok0[qc]; off = int(lay["qs"][u]); nbm = ln_ * pl.upr * 8
            DA.infllm2d_unit_bitmap(False, sel[g, t0_:t0_ + ln_].contiguous(), e[g, t0_:t0_ + ln_].contiguous(), cdst,
                                    pl.nbn, reg[off:off + nbm].view(torch.int64))
            qo = off + AL(nbm)
            reg[qo:qo + ln_ * GH * D * 2].view(torch.bfloat16).view(ln_ * GH, D).copy_(q[t0_ * GH:(t0_ + ln_) * GH, g])
        q2 = torch.cat([q, q])
        so = c_["so"]
        if self.scratch is None or self.scratch.numel() < so[-1]:
            self.scratch = torch.zeros(so[-1], dtype=torch.int32, device=dev)
        nb_ = len(c_["mine_in"]) * int(DA.infllm2d_fwd_param_bytes())
        if self.fpu.numel() < max(nb_, 1):
            self.fpu = torch.empty(max(nb_, 1), dtype=torch.uint8, device=dev)
        if timer is not None:
            timer.mark("plan", barrier=False)
        tabs_ = [t_.clone() for t_ in c_["dev_tabs"]]                          # (the kernel consumes some in place)
        self.epoch = DA.infllm2d_fwd_run(self.base, list(self.offs), c_["dims"], tabs_, self.scratch, list(so), self.pe_delta,
                                         self.pe_inter, q2, q2, self.o32, self.lse2, self.cu_q2, self.cu_k2, self.seqused2,
                                         c_["bm"], max(self.Ls) * GH, self.S, scale, c_["ce"], c_["uc"], c_["ucu"], self.fpu)
        L16 = self.L * GH
        l2 = torch.where(self.lse2 == float("inf"), torch.full_like(self.lse2, float("-inf")), self.lse2)
        lse = torch.logaddexp(l2[:, :L16], l2[:, L16:])
        mo = c_["mine_out"]
        if len(mo) == 0:                                                      # nothing moved here: the existing merge
            out = merge_n(self.o32, self.lse2, 2, out_dtype=q.dtype)
        else:
            oo = merge_n(self.o32, self.lse2, 2, out_dtype=torch.float32).view(-1, D)     # [(r, g), D]
            Os, Ls_, rows, gs = [], [], [], []
            for u in mo:
                g, qc, ln_ = int(units["g"][u]), int(units["qc"][u]), int(units["len"][u])
                off = int(lay["ol"][u])
                Os.append(reg[off:off + ln_ * GH * D * 4].view(torch.float32).view(ln_ * GH, D))
                lo = off + AL(ln_ * GH * D * 4)
                Ls_.append(reg[lo:lo + ln_ * GH * 4].view(torch.float32))
                rows.append(torch.arange(pl.tok0[qc] * GH, (pl.tok0[qc] + ln_) * GH, device=dev)); gs.append(g)
            Oc = torch.cat(Os); Lc = torch.cat(Ls_)
            Lc = torch.where(Lc == float("inf"), torch.full_like(Lc, float("-inf")), Lc)
            ri = torch.cat(rows); gi = torch.cat([torch.full((len(x),), g, device=dev, dtype=torch.long) for x, g in zip(rows, gs)])
            li = gi * L16 + ri; fi = ri * 2 + gi
            M = lse.reshape(-1).clone()
            M.scatter_reduce_(0, li, Lc, reduce="amax")
            Ms = torch.where(M == float("-inf"), torch.zeros_like(M), M)
            w0 = torch.exp(lse.reshape(-1) - Ms)
            den = w0.clone()
            num = oo * w0.view(2, L16).t().reshape(-1)[:, None]
            wu = torch.exp(Lc - Ms[li])
            num.index_add_(0, fi, Oc * wu[:, None]); den.index_add_(0, li, wu)
            den_o = den.view(2, L16).t().reshape(-1)
            out = torch.where(den_o[:, None] > 0, num / den_o.clamp(min=1e-30)[:, None], torch.zeros_like(num))
            out = out.view(L16, 2, D).to(q.dtype)
            lse = torch.where(den > 0, Ms + torch.log(den.clamp(min=1e-30)), torch.full_like(den, float("-inf"))).view(2, L16)
        if o is not None:
            o.copy_(out); out = o
        self.last = dict(stats=c_["stats"], dims=c_["dims"])
        return out, lse


def fwd2d_scratch_layout(n_rows, n_remote, n_red, n_workers, n_units, n_pq):
    """int32 offsets of [sq, sq_tail, tile_counter, n_enq, pq, pq_tail, red_cnt, task_desc, row_ntask, grp_cnt, rows_done,
    sched_b0, (reserved), unit_done] + total (infllm2d_fwd_host.cpp)"""
    sizes = [n_rows + n_red + 2 * n_workers + 64, 1, 1, 1, n_pq + 32, 1, n_red + 1, 4 * n_rows + 4, n_rows + 1, n_rows + 1,
             1, 3 * 512 + 2, 1, n_units + 1]
    offs, o = [], 0
    for s in sizes:
        offs.append(o); o += (s + 31) // 32 * 32
    return offs + [o]


# ---------------------------------------------------------------------------------------------------- backward
class InfllmBwd2D:
    """the backward with query movement. Region: [sig | kv | acc | land (the query-stationary layout) | per-rank 2-D
    area (bundle send / bundle landing / dQ accumulator / dQ landing)]; the shared region (InfllmFwd2D's cap). A moved
    unit sends its bundle (column bitmap | Q | dO | LSE | Delta) to the computing GPU, which returns the unit's FP32 dQ.
    plan(sel) = planner (backward models) + exchange + tables; ahead() + plan_host() = the same split into the
    collective part and the host part (the training step's worker thread)."""

    def __init__(self, DA, pl, cap=None):
        self.DA, self.pl = DA, pl
        self.chunks, self.my, self.S, self.gpn = pl.chunks, pl.my, pl.S, pl.gpn
        self.rank, self.P, self.dev = pl.rank, pl.P, pl.dev
        self.st = static_plan(pl.own_blk, pl.chunks, pl.S, pl.gpn, RUN_MAX)
        self.grid = int(DA.infllm2d_bwd_grid())
        self.n_workers = self.grid - 1
        self.NL_cap = global_plan(self.st, np.ones((self.P, self.st["nb"]), bool))["NL"]
        self.NSIG0 = self.st["n_run_sig"] + self.NL_cap
        self.NUmax = self.P * G * pl.m
        self.NSIG = self.NSIG0 + 2 * self.NUmax
        offs, o = [], 0
        for sz in (8 * self.NSIG, self.st["nb"] * BLK * ROW_BYTES, self.st["nb"] * SLOT_BYTES, max(self.NL_cap, 1) * SLOT_BYTES):
            offs.append(o); o = AL(o + sz)
        self.offs, self.a2d = offs, o
        self.cap = cap or int(int(os.environ.get("NVSHMEM_SYMMETRIC_SIZE", str(4 << 30))) * 0.85)
        self.region = DA.bwd_sym_alloc(self.cap)
        self.pe_delta, self.pe_inter = DA.bwd_pe_info()
        self.base = self.region.data_ptr()
        self.Ls, self.L = pl.Ls, pl.L
        dev = self.dev
        self.cu_q = torch.tensor(np.concatenate([[0], np.cumsum(self.Ls) * GH]), dtype=torch.int32, device=dev)
        self.cu_k = torch.zeros(len(self.my) + 1, dtype=torch.int32, device=dev)
        self.seqused = torch.tensor([self.chunks[ci][1] for ci in self.my], dtype=torch.int32, device=dev)
        self.pos = torch.cat([torch.arange(*self.chunks[ci], device=dev) for ci in self.my]).to(torch.int32)
        b = len(self.my)
        self.dpsum = torch.empty(2, self.L * GH + 128 * b, dtype=torch.float32, device=dev)
        self.dq_accum = torch.empty(self.L * GH + 128 * b, 2, 128, dtype=torch.float32, device=dev)
        self.kv = self.region[offs[1]:offs[1] + self.st["nb"] * BLK * ROW_BYTES].view(torch.bfloat16).view(-1, 512)
        self.seg0 = {ci: int(self.cu_q[i].item()) + 128 * i for i, ci in enumerate(self.my)}   # dpsum / dq_accum row of segment
        self.fpu = torch.empty(1, dtype=torch.uint8, device=dev)
        self.scratch = None
        self.used = 0
        self.last = {}

    def bslot(self, n):                       # landed bundle: column bitmap | Q | dO | LSE | Delta (+ 128 pad)
        nbm = self.pl.nbn * ((n + 63) // 64) * 8
        return AL(nbm) + 2 * AL(n * GH * D * 2) + AL(n * GH * 4) + AL((n * GH + 128) * 4)

    def bparts(self, n):                      # offsets inside a bundle: bm, q, do, lse, dpsum
        nbm = AL(self.pl.nbn * ((n + 63) // 64) * 8); qb = AL(n * GH * D * 2)
        return 0, nbm, nbm + qb, nbm + 2 * qb, nbm + 2 * qb + AL(n * GH * 4)

    def dqslot(self, n):                      # the unit's FP32 dq_accum (+ 128 pad rows), sent back without the pad
        return AL((n * GH + 128) * D * 4)

    def layout(self, units):
        c, o, ln = units["c"], units["o"], units["len"]
        n = len(c)
        bs, bl, dqa, dql = (np.zeros(n, np.int64) for _ in range(4))
        top = 0
        for r in range(self.P):
            x = self.a2d
            for u in np.nonzero(o == r)[0]:
                bs[u] = x; x += self.bslot(ln[u])
            for u in np.nonzero(c == r)[0]:
                bl[u] = x; x += self.bslot(ln[u])
            for u in np.nonzero(c == r)[0]:
                dqa[u] = x; x += self.dqslot(ln[u])
            for u in np.nonzero(o == r)[0]:
                dql[u] = x; x += AL(ln[u] * GH * D * 4)
            top = max(top, x)
        if top > self.cap:
            return None
        self.used = max(self.used, top)
        return dict(bs=bs, bl=bl, dqa=dqa, dql=dql, top=top)

    def plan(self, sel, plans=None):
        """planner (backward models; unless plans={'bwd': pb}) + exchange + this rank's tables"""
        pl = self.pl
        t0 = pl._t0()
        if plans is None:
            W, kblk = pl.stats(sel)
            t0 = pl._tick("bwd_stats_ms", t0)
            plans = pl.plan(W, kblk, ("bwd",))
            t0 = pl._t0()
        pb = plans["bwd"]
        e, needs, units = pl.exchange(sel, pb)
        if self.layout(units) is None:                                        # memory cap: as InfllmFwd2D.forward
            self.cap_home_plans = getattr(self, "cap_home_plans", 0) + 1
            pb = pl.ob[:, :, None].expand_as(pb).contiguous()
            e, needs, units = pl.exchange(sel, pb)
        t0 = pl._tick("bwd_exchange_ms", t0)
        sel_own = torch.where(e == self.rank, sel, torch.full_like(sel, -1)).contiguous()
        hp = self.plan_host(None, needs, units, own_tasks=self.tasks_gpu(sel_own))
        out = self.plan_finish(hp, sel_own, e)
        pl._tick("bwd_tables_ms", t0)
        return out

    def tasks_gpu(self, sel_own):
        """(b, h, n, cnt) of the own column tasks on the GPU (as InfllmBwd.tasks: tokens per (segment, head, block)), one
        host copy"""
        nb = self.st["nb"]
        cnts, t0 = [], 0
        for b, L in enumerate(self.Ls):
            x = sel_own[:, t0:t0 + L].long()
            ok = x >= 0
            hh = torch.arange(2, device=self.dev)[:, None, None].expand_as(x)
            cnts.append(torch.bincount((hh * nb + x.clamp(min=0))[ok], minlength=2 * nb))
            t0 += L
        c = torch.stack(cnts).view(len(self.Ls), 2, nb).cpu().numpy()
        b_, h_, n_ = np.nonzero(c)
        return b_, h_, n_, c[b_, h_, n_]

    def ahead(self, sel, pb, pre=None):
        """(training step) the collective part on the calling thread (or given: pre, the joint forward + backward
        exchange) -> a closure for the worker thread (host tables only)"""
        e, needs, units = pre if pre is not None else self.pl.exchange(sel, pb)
        if self.layout(units) is None:                                        # memory cap: the home plan
            self.cap_home_plans = getattr(self, "cap_home_plans", 0) + 1
            e, needs, units = self.pl.exchange(sel, self.pl.ob[:, :, None].expand_as(pb).contiguous())
        sel_own = torch.where(e == self.rank, sel, torch.full_like(sel, -1)).contiguous()
        h = torch.empty(sel_own.shape, dtype=torch.int32, pin_memory=True); h.copy_(sel_own, non_blocking=True)
        ev = torch.cuda.Event(); ev.record()

        def job():
            ev.synchronize()
            return self.plan_host(h.numpy(), needs, units)
        return job, sel_own, e

    def plan_host(self, sel_own_np, needs, units, own_tasks=None):
        """host tables (no GPU work, no collective): own tasks from the restricted selection (or given, own_tasks),
        foreign tasks (unit, block) of every block of an incoming unit's union, rank_tables on the combined list + the
        2-D additions"""
        st, rank, nb = self.st, self.rank, self.st["nb"]
        out = [[], [], [], []]
        if own_tasks is not None:
            for i_ in range(4):
                out[i_].append(np.asarray(own_tasks[i_], np.int64))
        else:
            t0 = 0
            x_all = sel_own_np.astype(np.int64)
            for b, L in enumerate(self.Ls):                                   # own tasks (as InfllmBwd.plan_host)
                x = x_all[:, t0:t0 + L]
                for h in range(2):
                    v = x[h][x[h] >= 0]
                    cnt = np.bincount(v, minlength=nb)
                    n = np.nonzero(cnt)[0]
                    out[0].append(np.full(len(n), b)); out[1].append(np.full(len(n), h)); out[2].append(n); out[3].append(cnt[n])
                t0 += L
        mine_in = np.nonzero(units["c"] == rank)[0]
        mine_out = np.nonzero(units["o"] == rank)[0]
        fun = []
        for k, u in enumerate(mine_in):                                      # foreign tasks: (unit k, block n), h = 2
            n = np.nonzero(units["union"][u])[0]
            out[0].append(np.full(len(n), k)); out[1].append(np.full(len(n), 2)); out[2].append(n)
            out[3].append(np.full(len(n), int(units["len"][u])))
            fun.append(len(n))
        tasks = tuple(np.concatenate(o_) for o_ in out)
        Gb = global_plan(st, needs)
        assert Gb["NL"] <= self.NL_cap
        tabs, dims, stats, ce = rank_tables(st, Gb, needs, rank, tasks, self.my, self.n_workers, self.offs)
        lay = self.layout(units)
        n_qi, n_oo = len(mine_in), len(mine_out)
        tb, th = tabs["task_b"], tabs["task_h"]
        n_tasks = dims["n_tasks"]
        tb, th = tb[:n_tasks], th[:n_tasks]
        nsig_q = dims["n_remote"]
        remote_sig = np.concatenate([tabs["remote_sig"][:nsig_q], self.NSIG0 + mine_in,
                                     self.NSIG0 + self.NUmax + mine_out]).astype(np.int32)
        nsig = len(remote_sig)
        extend = lambda p_, n_: np.concatenate([p_[:nsig_q + 1], np.full(n_, p_[nsig_q])])   # noqa: E731  (CSR, appended empty rows)
        sig_grp_ptr = extend(tabs["sig_grp_ptr"], nsig - nsig_q)
        # foreign tasks wait for their unit's bundle (row_qdep); bundle signal j -> its tasks (sig_row)
        row_qdep = np.full(n_tasks, -1, np.int32)
        fr = np.nonzero(th == 2)[0]
        row_qdep[fr] = nsig_q + tb[fr]
        order = np.argsort(tb[fr], kind="stable")
        sig_row = fr[order].astype(np.int32)
        cnt = np.zeros(nsig, np.int64); cnt[nsig_q:nsig_q + n_qi] = np.bincount(tb[fr], minlength=n_qi)[:n_qi] if n_qi else []
        sig_row_ptr = np.concatenate([[0], np.cumsum(cnt)])
        unit_need = (np.bincount(tb[fr], minlength=n_qi).astype(np.int32) if n_qi else np.zeros(1, np.int32))
        # dQ-landing units (no-op) after the owner sums / gateway aggregations: signal nsig_q + n_qi + l -> unit
        n_km, n_ka = dims["n_km"], dims["n_ka"]
        nred0 = n_km + n_ka
        red_need = np.concatenate([tabs["red_need"][:nred0], np.ones(n_oo, np.int32)])
        red_kind = np.concatenate([tabs["red_kind"][:nred0], np.full(n_oo, K_NOP, np.int32)])
        red_pay = np.concatenate([tabs["red_pay"][:nred0], np.zeros(n_oo, np.int32)])
        red_n = np.concatenate([tabs["red_n"][:nred0], np.ones(n_oo, np.int32)])
        sig_red_ptr = extend(tabs["sig_red_ptr"], nsig - nsig_q)
        sig_red_ptr[nsig_q + n_qi + 1:] = sig_red_ptr[nsig_q] + np.arange(1, n_oo + 1)
        sig_red = np.concatenate([tabs["sig_red"][:sig_red_ptr[nsig_q]], nred0 + np.arange(n_oo)]).astype(np.int32)
        # row order: foreign tasks first (other GPUs wait for their dQ), then the own tasks in rank_tables' order
        row_order = np.concatenate([fr, np.nonzero(th != 2)[0]]).astype(np.int32)
        # triggers: [0, nb) block pushes (rank_tables), [nb, nb + n_qi) dQ returns, then the relay range (rank_tables'
        # [nb, nb + nsig_q) moved, + empty triggers for the appended signals)
        me_n = rank // self.gpn
        same = lambda d: d // self.gpn == me_n                                 # noqa: E731
        msg_q = tabs["msg"].view(np.int64).reshape(-1, 4)
        tp_q = tabs["trig_ptr"].astype(np.int64)
        n_start_q = dims["n_start"]
        n_msg_q = int(tp_q[nb])                                          # start + block-push messages
        start_b, ce_b = [], []
        for u in mine_out:                                                   # bundles of my units (start messages)
            c = int(units["c"][u]); m_ = (lay["bs"][u], lay["bl"][u], self.bslot(units["len"][u]), c, self.NSIG0 + u)
            (ce_b.append(m_ + (0, 0)) if same(c) else start_b.append(m_))
        ns = len(start_b)
        ret_m, ce_r = [], []
        lanes_q = np.ones(max(n_qi, 0), np.int32)
        for k, u in enumerate(mine_in):                                      # dQ returns: trigger nb + k
            o_ = int(units["o"][u]); t_ = nb + k
            m_ = (lay["dqa"][u], lay["dql"][u], int(units["len"][u]) * GH * D * 4, o_, self.NSIG0 + self.NUmax + u)
            if same(o_):
                ce_r.append(m_ + (2, t_)); lanes_q[k] = 2
            else:
                ret_m.append((k, m_))
        n_trig = nb + n_qi + nsig
        trig_ptr = np.zeros(n_trig + 1, np.int64)
        trig_ptr[:nb + 1] = tp_q[:nb + 1] + ns
        cq = np.zeros(n_qi, np.int64)
        for k, _ in ret_m:
            cq[k] += 1
        trig_ptr[nb + 1:nb + n_qi + 1] = trig_ptr[nb] + np.cumsum(cq)
        trig_ptr[nb + n_qi + 1:] = trig_ptr[nb + n_qi]
        trig_lane = np.ones(n_trig, np.int32)
        trig_lane[:nb] = tabs["trig_lane"][:nb]
        trig_lane[nb:nb + n_qi] = lanes_q
        trig_lane[nb + n_qi:nb + n_qi + nsig_q] = tabs["trig_lane"][nb:nb + nsig_q]
        pk = lambda rows: (np.stack([np.asarray([r_[0] for r_ in rows]), np.asarray([r_[1] for r_ in rows]),   # noqa: E731
                                     np.asarray([r_[2] for r_ in rows]),
                                     np.asarray([r_[3] for r_ in rows]) + (np.asarray([r_[4] for r_ in rows], np.int64) << 32)], 1)
                           if rows else np.zeros((0, 4), np.int64))
        msg = np.concatenate([pk(start_b), msg_q[:n_start_q], msg_q[n_start_q:n_msg_q],
                              pk([m_ for _, m_ in sorted(ret_m)])]).astype(np.int64)
        if len(msg) == 0:
            msg = np.zeros((1, 4), np.int64)
        n_pq = dims["n_pq"] + len(ret_m)

        def ce6(rows):
            r7 = np.asarray(rows, np.int64).reshape(-1, 7)
            return np.stack([r7[:, 0], r7[:, 1], r7[:, 2], r7[:, 3] + (r7[:, 4] << 32), r7[:, 5] + (r7[:, 6] << 32),
                             np.ones(len(r7), np.int64)], 1)
        ce_all = np.concatenate([ce6(ce_b), np.asarray(ce, np.int64).reshape(-1, 6), ce6(ce_r)], 0)
        tabs = dict(tabs)
        tabs.update(msg=np.ascontiguousarray(msg).view(np.int32).reshape(-1), trig_ptr=trig_ptr.astype(np.int32),
                    trig_lane=trig_lane, remote_sig=remote_sig, row_qdep=row_qdep, row_order=row_order,
                    sig_grp_ptr=sig_grp_ptr.astype(np.int32), sig_row_ptr=sig_row_ptr.astype(np.int32),
                    sig_row=sig_row if len(sig_row) else np.zeros(1, np.int32), red_need=red_need, red_kind=red_kind,
                    red_pay=red_pay, red_n=red_n, sig_red_ptr=sig_red_ptr.astype(np.int32),
                    sig_red=sig_red if len(sig_red) else np.zeros(1, np.int32))
        for k_ in BWD_TABLE_ORDER:
            if np.asarray(tabs[k_]).size == 0:
                tabs[k_] = np.zeros(1, np.int32)
        ln = units["len"]
        uc = np.stack([lay["bl"][mine_in] + self.bparts_v(ln[mine_in], 1), lay["bl"][mine_in] + self.bparts_v(ln[mine_in], 2),
                       lay["bl"][mine_in] + self.bparts_v(ln[mine_in], 3), lay["bl"][mine_in] + self.bparts_v(ln[mine_in], 4),
                       lay["dqa"][mine_in], lay["bl"][mine_in], units["g"][mine_in], ln[mine_in],
                       np.full(n_qi, self.pl.nbn), np.full(n_qi, self.S)], 1).astype(np.int64) if n_qi else np.zeros((0, 10), np.int64)
        ucu = np.stack([np.zeros(n_qi), ln[mine_in] * GH, np.zeros(n_qi), np.zeros(n_qi), units["end"][mine_in]], 1).astype(np.int32) \
            if n_qi else np.zeros((1, 5), np.int32)
        so, sn = bwd2d_scratch_layout(n_tasks, nb, nred0 + n_oo, n_pq, self.n_workers, n_qi)
        dd = [nb, n_tasks, n_km, n_ka, nsig, ns + n_start_q, n_pq, self.NSIG, self.NL_cap, self.grid, n_oo, nb, n_qi, nb + n_qi]
        stats = dict(stats)
        stats.update(units=len(units["c"]), units_in=n_qi, units_out=n_oo,
                     bundle_bytes_out=int(sum(self.bslot(ln[u]) for u in mine_out)),
                     dq_bytes_in=int(sum(ln[u] * GH * D * 4 for u in mine_out)), region_top=lay["top"],
                     cap_home_plans=getattr(self, "cap_home_plans", 0))
        return dict(tabs_np=tabs, dims=dd, stats=stats, scr_offs=so, scr_n=sn, ce=torch.from_numpy(np.ascontiguousarray(ce_all)),
                    units=units, lay=lay, mine_in=mine_in, mine_out=mine_out, uc=torch.from_numpy(uc), ucu_np=ucu,
                    unit_need=unit_need)

    def bparts_v(self, ln, i):
        return np.array([self.bparts(int(x))[i] for x in ln], np.int64)

    def plan_finish(self, hp, sel_own, e):
        dev = self.dev
        if self.scratch is None or self.scratch.numel() < hp["scr_n"]:
            self.scratch = torch.zeros(hp["scr_n"], dtype=torch.int32, device=dev)
        I = lambda x: torch.from_numpy(np.ascontiguousarray(np.asarray(x, np.int32).reshape(-1))).to(dev, non_blocking=True)   # noqa: E731
        tabs = [I(hp["tabs_np"][k_]) for k_ in BWD_TABLE_ORDER] + [I(hp["unit_need"])]
        out = dict(hp)
        out.update(tabs=tabs, ucu=I(hp["ucu_np"]).view(-1, 5), sel_own=sel_own, e=e)
        return out

    def run(self, q, o, do, lse, k_own, v_own, sel, plan, scale, dq=None, dk=None, dv=None):
        DA, st, dev, pl = self.DA, self.st, self.dev, self.pl
        dq = torch.empty_like(q) if dq is None else dq
        dk = torch.zeros_like(k_own) if dk is None else dk.zero_()
        dv = torch.zeros_like(v_own) if dv is None else dv.zero_()
        r = 0
        for ci, L in zip(self.my, self.Ls):
            a_ = self.chunks[ci][0]
            self.kv[a_:a_ + L, :256] = k_own[r:r + L].reshape(L, -1)
            self.kv[a_:a_ + L, 256:] = v_own[r:r + L].reshape(L, -1)
            r += L
        sel_own, e = plan["sel_own"], plan["e"]
        bm = DA.infllm_col_bitmap(sel_own, self.pos, st["nb"])
        DA.infllm_bwd_prep(q, o, do, dq, lse, self.dpsum, self.dq_accum, self.cu_q, max(self.Ls) * GH, scale)
        units, lay, reg = plan["units"], plan["lay"], self.region
        for u in plan["mine_out"]:                                           # bundles: column bitmap | Q | dO | LSE | Delta
            g, qc, ln_, cdst = int(units["g"][u]), int(units["qc"][u]), int(units["len"][u]), int(units["c"][u])
            t0_ = pl.tok0[qc]; base = int(lay["bs"][u]); pbm, pq, pdo, pls, pdp = self.bparts(ln_)
            nbm = pl.nbn * ((ln_ + 63) // 64)
            DA.infllm2d_unit_bitmap(True, sel[g, t0_:t0_ + ln_].contiguous(), e[g, t0_:t0_ + ln_].contiguous(), cdst, pl.nbn,
                                    reg[base + pbm:base + pbm + nbm * 8].view(torch.int64))
            R = slice(t0_ * GH, (t0_ + ln_) * GH)
            reg[base + pq:base + pq + ln_ * GH * D * 2].view(torch.bfloat16).view(-1, D).copy_(q[R, g])
            reg[base + pdo:base + pdo + ln_ * GH * D * 2].view(torch.bfloat16).view(-1, D).copy_(do[R, g])
            reg[base + pls:base + pls + ln_ * GH * 4].view(torch.float32).copy_(lse[g, R])
            s0 = self.seg0[qc]
            reg[base + pdp:base + pdp + ln_ * GH * 4].view(torch.float32).copy_(self.dpsum[g, s0:s0 + ln_ * GH])
        for u in plan["mine_in"]:                                            # the incoming units' dq accumulators
            x = int(lay["dqa"][u]); reg[x:x + self.dqslot(int(units["len"][u]))].zero_()
        nb_ = len(plan["mine_in"]) * int(DA.infllm2d_bwd_param_bytes())
        if self.fpu.numel() < max(nb_, 1):
            self.fpu = torch.empty(max(nb_, 1), dtype=torch.uint8, device=dev)
        tabs_ = [t_.clone() for t_ in plan["tabs"]]
        DA.infllm2d_bwd_run(self.base, list(self.offs), plan["dims"], tabs_, self.scratch, list(plan["scr_offs"]),
                            self.pe_delta, self.pe_inter, q, o, do, dq, lse, self.dpsum, self.dq_accum, self.cu_q, self.cu_k,
                            self.seqused, bm, max(self.Ls) * GH, self.S, scale, dk, dv, plan["ce"], plan["uc"], plan["ucu"], self.fpu)
        mo = plan["mine_out"]
        if len(mo):                                                          # landed dQ partials into the own accumulator
            dqa = self.dq_accum.view(-1, D)
            Xs, idx = [], []
            for u in mo:
                g, qc, ln_ = int(units["g"][u]), int(units["qc"][u]), int(units["len"][u])
                x = int(lay["dql"][u])
                Xs.append(reg[x:x + ln_ * GH * D * 4].view(torch.float32).view(-1, D))
                s0 = self.seg0[qc]
                idx.append((torch.arange(s0, s0 + ln_ * GH, device=dev) * 2 + g))
            dqa.index_add_(0, torch.cat(idx), torch.cat(Xs))
        DA.infllm_bwd_convert_dq(q, o, do, dq, lse, self.dpsum, self.dq_accum, self.cu_q, max(self.Ls) * GH, scale)
        self.last = dict(stats=plan["stats"], dims=plan["dims"])
        return dq, dk, dv


def bwd2d_scratch_layout(n_tasks, nb, n_red, n_pq, n_workers, n_units):
    """int32 offsets of [sq, sq_tail, tile_counter, ks_done, n_enq, pq, pq_tail, red_cnt, task_desc, row_ntask, grp_cnt,
    rows_done, sched_b0, unit_done] + total (infllm2d_bwd_host.cpp)"""
    sizes = [n_tasks + n_red + n_workers + 64, 1, 1, nb, 1, n_pq + 32, 1, n_red + 1, 4 * n_tasks + 4, n_tasks + 1,
             n_tasks + 1, 1, 3 * 512 + 2, n_units + 1]
    offs, o = [], 0
    for s in sizes:
        offs.append(o); o += (s + 31) // 32 * 32
    return offs + [o], o


# ================================================================ training-step attention ops (autograd via reuse.CPAttn)
class _EvMark:
    """Record a CUDA event at the forward path's mark('plan'); the 2-D path includes input staging before this mark."""

    def __init__(self, ev):
        self.ev = ev

    def mark(self, k, barrier=False):
        if k == "plan":
            self.ev.record()


def _selection(q, k, chunks, my, Ls, S, pos):
    """the layer's causal-filtered selection [2, L, K] (computed on the first forward, reused by the recompute)"""
    qs, ks, r = {}, {}, 0
    for ci, L in zip(my, Ls):
        qs[ci] = q[r:r + L]; ks[ci] = k[r:r + L]; r += L
    return filter_causal(pad_concat(select_cp(qs, ks, chunks, my, S), my).to(torch.int32), pos).clone()


class DexInfllmOp:
    """Dex (system `dex`) in a training step. The layer's first forward selects, plans the forward AND the backward
    (planner input all-gather, joint planner + home gate, one broadcast), runs both exchanges in one all-reduce and hands
    the backward's host tables to the worker thread; the recompute replays the stored forward plan; the backward
    collects the tables built ahead. Plan-stage spans are recorded as CUDA events on the compute stream (no sync):
    forward = planner input all-gather -> kernel input staging; backward = collecting the ahead-built tables."""

    def __init__(self, chunks, my, S, cfg):
        DA = ext()
        self.chunks, self.my, self.S, self.cfg = chunks, my, S, cfg
        dev = torch.device("cuda", torch.cuda.current_device())
        self.pl = Plan2D(chunks, my, S, dist.get_world_size(), dist.get_rank(), dev, gpn=gpus_per_node(dist.get_world_size()))
        self.fw = InfllmFwd2D(DA, self.pl)
        self.bw = InfllmBwd2D(DA, self.pl)
        assert self.fw.region.data_ptr() == self.bw.region.data_ptr(), "the shared region moved"
        self.pos = torch.cat([torch.arange(*chunks[ci], device=dev) for ci in my]).to(torch.int32)
        self.Ls = [chunks[ci][1] - chunks[ci][0] for ci in my]
        self.scale = cfg.head_dim ** -0.5
        self.pev = {"fwd": [], "bwd": []}

    def fwd(self, q, k, v):
        c = reuse.cache_get()
        if c is None:
            c = dict(sel=_selection(q, k, self.chunks, self.my, self.Ls, self.S, self.pos))
            reuse.cache_put(c)
        sel_f = c["sel"]
        q_rows = to_rows(q); k = k.contiguous(); v = v.contiguous()
        plans = pre = timer = None
        if "plan2d" not in c:                                          # the layer's first forward: plan (fwd + bwd)
            e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
            self.pev["fwd"].append((e0, e1)); timer = _EvMark(e1)
            e0.record()
            W, kblk = self.pl.stats(sel_f)
            plans = self.pl.plan(W, kblk, ("fwd", "bwd"))
            pre, preb = self.pl.exchange_multi(sel_f, [plans["fwd"], plans["bwd"]])     # both exchanges, one all-reduce
            job, sel_own, e = self.bw.ahead(sel_f, plans["bwd"], pre=preb)             # backward host tables: worker
            c["bplan"] = (reuse.pool().submit(job), sel_own, e)
        o_rows, lse = self.fw.forward(q_rows, k, v, sel_f, self.scale, timer=timer, cache=c, plans=plans, pre=pre)
        self.aux = c
        o_rows = o_rows.clone(); lse = lse.clone()
        return from_rows(o_rows, self.cfg.n_heads).contiguous(), (q_rows, k, v, o_rows, lse, sel_f)

    def bwd(self, saved, do, aux=None):
        q_rows, k, v, o_rows, lse, sel_f = saved
        if aux is not None and "bplan" in aux:
            e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
            e0.record(); self.pev["bwd"].append((e0, e1))
            fut, sel_own, e = aux["bplan"]
            plan = self.bw.plan_finish(fut.result(), sel_own, e)
            e1.record()
            reuse.order_ranks(q_rows.device)
        else:
            plan = self.bw.plan(sel_f)
        dq, dk, dv = self.bw.run(q_rows, o_rows, to_rows(do), lse, k, v, sel_f, plan, self.scale)
        return from_rows(dq, self.cfg.n_heads).contiguous(), dk, dv

    def pop_plan_ms(self):
        """(after a synchronize) summed plan-stage spans from the last call on, ms: (forward, backward, plan calls)"""
        f = sum(a_.elapsed_time(b_) for a_, b_ in self.pev["fwd"]); b = sum(a_.elapsed_time(b_) for a_, b_ in self.pev["bwd"])
        n = len(self.pev["fwd"])
        self.pev = {"fwd": [], "bwd": []}
        return f, b, n

    def heap_used(self):
        return max(self.fw.used, self.bw.used)

    def __call__(self, q, k, v):
        return reuse.CPAttn.apply(q, k, v, self)


class DexInfllmNoAssignerOp:
    """Dex without assigner (system `dex_no_assigner`) in a training step: query-stationary executors; the layer's
    first forward stores its plan for the recompute and submits the backward's host tables to the worker thread (from
    the forward's needs matrix: no extra collective)."""

    def __init__(self, chunks, my, S, cfg):
        DA = ext()
        self.chunks, self.my, self.S, self.cfg = chunks, my, S, cfg
        self.fw = InfllmFwd(DA, chunks, my, S, gpn=gpus_per_node(dist.get_world_size()))
        self.bw = InfllmBwd(DA, chunks, my, S, gpn=gpus_per_node(dist.get_world_size()))
        # one shared symmetric region (bwd_sym_alloc grows it in place): the forward's view must stay valid
        assert self.fw.region.data_ptr() == self.bw.region.data_ptr(), "the shared region moved"
        dev = torch.device("cuda", torch.cuda.current_device())
        self.pos = torch.cat([torch.arange(*chunks[ci], device=dev) for ci in my]).to(torch.int32)
        self.Ls = [chunks[ci][1] - chunks[ci][0] for ci in my]
        self.scale = cfg.head_dim ** -0.5

    def fwd(self, q, k, v):
        c = reuse.cache_get()
        if c is None:
            c = dict(sel=_selection(q, k, self.chunks, self.my, self.Ls, self.S, self.pos))
            reuse.cache_put(c)
        sel_f = c["sel"]
        q_rows = to_rows(q); k = k.contiguous(); v = v.contiguous()
        o_rows, lse = self.fw.forward(q_rows, k, v, sel_f, self.scale, cache=c)
        if "bplan" not in c:                                           # the backward plan, searched ahead (worker thread)
            needs = c["plan"]["needs"]
            c["bplan"] = reuse.pool().submit(lambda s_=sel_f, n_=needs: self.bw.plan_host(s_.cpu().numpy(), n_))
        self.aux = c
        o_rows = o_rows.clone(); lse = lse.clone()
        return from_rows(o_rows, self.cfg.n_heads).contiguous(), (q_rows, k, v, o_rows, lse, sel_f)

    def bwd(self, saved, do, aux=None):
        q_rows, k, v, o_rows, lse, sel_f = saved
        if aux is not None and "bplan" in aux:
            plan = self.bw.plan_finish(aux["bplan"].result())
            reuse.order_ranks(q_rows.device)
        else:
            plan = self.bw.plan(sel_f)
        dq, dk, dv = self.bw.run(q_rows, o_rows, to_rows(do), lse, k, v, sel_f, plan, self.scale)
        return from_rows(dq, self.cfg.n_heads).contiguous(), dk, dv

    def heap_used(self):
        return max(self.fw.region.numel(), self.bw.region.numel())

    def __call__(self, q, k, v):
        return reuse.CPAttn.apply(q, k, v, self)


# ================================================================ operator-test front ends (one layer, given selection)
class DexInfllmLayer:
    """the operator test's view of Dex (`dex`) / Dex without assigner (`dex_no_assigner`) on one layer with a
    given selection; kind 'fwd' builds the forward executor, 'bwd' the backward one (built untimed, once per part)."""

    def __init__(self, system, kind, chunks, my, S, P, rank, dev):
        DA = ext()
        self.system, self.kind = system, kind
        gpn = gpus_per_node(P)
        self.pl = Plan2D(chunks, my, S, P, rank, dev, gpn=gpn) if system == "dex" else None
        if kind == "fwd":
            self.ex = InfllmFwd2D(DA, self.pl) if system == "dex" else InfllmFwd(DA, chunks, my, S, gpn=gpn)
        else:
            self.ex = InfllmBwd2D(DA, self.pl) if system == "dex" else InfllmBwd(DA, chunks, my, S, gpn=gpn)
        self.Ls = self.ex.Ls

    def fwd(self, q_rows, k_own, v_own, sel_f, scale, timer=None):
        o, _ = self.ex.forward(q_rows, k_own, v_own, sel_f, scale, timer=timer)
        return o

    def bwd(self, q_rows, o_rows, do_rows, lse, k_own, v_own, sel_f, scale, timer=None):
        plan = self.ex.plan(sel_f)
        if timer is not None:
            timer.mark("plan", barrier=False)
        return self.ex.run(q_rows, o_rows, do_rows, lse, k_own, v_own, sel_f, plan, scale)

    def fwd_stats(self):
        """transfer statistics of the last forward call"""
        if self.system == "dex":
            st = self.ex.last["stats"]
            return dict(nsig=self.ex.last["dims"][6], bytes_in=st["kv_bytes_in"],
                        stats={k_: (int(v_) if isinstance(v_, (int, np.integer)) else v_) for k_, v_ in st.items()})
        return dict(nsig=self.ex.last["dims"]["n_remote"], bytes_in=self.ex.last["stats"]["bytes_in"])

    def bwd_stats(self):
        st = self.ex.last["stats"] if getattr(self.ex, "last", None) else {}
        return {k_: (int(v_) if isinstance(v_, (int, np.integer)) else v_) for k_, v_ in st.items()}

    def heap_used(self):
        return getattr(self.ex, "used", 0)
