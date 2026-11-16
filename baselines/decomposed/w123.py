"""Dex's W1-W3 plans (head split 4: forward plan for the forward, backward plan for the backward) on a decomposed executor, the
W1-W3 row of the executor ablation (Table 6). It keeps the assignment and attention routines, with one remote input
copy per receiving node. It uses its own gateway selection and step schedule. Both executors return forward partials
directly; this executor also returns backward gradients directly, without Dex's gateway pre-reduction.
Batched NCCL transfers replace NVSHMEM/copy-engine transport; attention runs once per step rather than with tile-level
overlap. This compares execution strategies, not kernel-launch overhead alone.

Units    : Q unit = global tile t of the virtual (head-split) sequence with all its HQ heads of the owner (forward: Q; backward: Q,
           dO fp16 + LSE, Delta fp32); KV unit = tile t, HK heads (K + V). A GPU needs a unit when one of the tile pairs the plan
           gives it reads it.
Data flow: a unit a remote node needs crosses the network once, owner -> a landing GPU of that node that needs it (needers sorted,
           landing = needers[key % #needers]); the landing GPU forwards it inside the node; same-node needers get it directly.
Steps    : one batch_isend_irecv per step on a comm stream, one packed message per peer per step (Q units then KV units sorted by
           key), compute on the default stream as data arrives:
             step 0: owner -> {landing, same-node needers}   | compute pairs whose Q and KV are local
             step 1: landing -> other needers                | compute pairs ready after step 0
             step 2: -                                       | compute pairs ready after step 1
             step 3: results to the owners, direct (forward: merged partial O fp32 + LSE per moved Q unit; backward: dQ accumulator
                     fp32 per moved Q unit, dK + dV fp32 per moved KV unit), merged / summed in fp32 there
           compute = C.sparse_fwd once per step (CSR over this GPU's Q buffer x KV slots, full tiles first, the causal diagonal
           last), C.sparse_bwd once per step (KV-head-major 64-row tasks). Step lists and kernel tables are built once per plan
           outside the timed call.
Driver   : same inputs and protocol as experiments/attention_layer/static_op.py (fwd: q, k, v ~ N(0, 0.25) fp16, exact padding
           correction timed; bwd: Q, K, V, dO, O random, LSE = 6, Delta = rowsum(dO O), copy-in + steps + returns + dQ conversion timed).
           --launch 1: FP32 sampled tiles / single-GPU oracle + FP64 sampled rows. --launch > 1 generates owned tiles only for the backward pass.
Output   : $DEX_RESULT_ROOT/ablation_executor/<w>/decomposed_launch<k>.jsonl
Example  : tools/launch.sh baselines/decomposed/w123.py --workload w2 --ids a,b --launch 1"""
import argparse
import datetime
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "experiments", "attention_layer")]
import static_inputs as SI  # noqa: E402
from dex.common import fp64_check as F64  # noqa: E402
from dex.common.env import cfg  # noqa: E402
from dex.common.pad_correct import correct as pad_correct  # noqa: E402

from dex import dex_cuda as C  # noqa: E402

T, D = 128, 128
GPN = SI.GPN
LSE_EMPTY = -1e4
TILE = T




def relay(r, key, src, node_of, me):
    """r receiver, key unit id, src owner (one row per (receiver, unit) need) -> step-0 / step-1
    (src, dst, key) and this rank's received keys (sorted) with their arrival step (1 = step 0, 2 = forwarded)"""
    nr, ns = node_of[r], node_of[src]
    intra = nr == ns
    ie = np.nonzero(~intra)[0]
    nn = int(node_of.max()) + 1
    gk = key[ie] * nn + nr[ie]
    o = np.lexsort((r[ie], gk)); ie = ie[o]; gk = gk[o]
    if len(ie):
        _, gstart, gcnt = np.unique(gk, return_index=True, return_counts=True)
    else:
        gstart = gcnt = np.zeros(0, np.int64)
    gid = np.repeat(np.arange(len(gstart)), gcnt)
    land = r[ie][gstart + (key[ie][gstart] % np.maximum(gcnt, 1))][gid] if len(ie) else np.zeros(0, np.int64)
    a = (np.r_[src[intra], src[ie][gstart]], np.r_[r[intra], land[gstart] if len(ie) else np.zeros(0, np.int64)],
         np.r_[key[intra], key[ie][gstart]])
    fw = r[ie] != land
    b = (land[fw], r[ie][fw], key[ie][fw])
    ka, kb = a[2][a[1] == me], b[2][b[1] == me]
    mk = np.r_[ka, kb]; st = np.r_[np.ones(len(ka), np.int64), np.full(len(kb), 2)]
    o = np.argsort(mk, kind="stable")
    return a, b, mk[o], st[o]


def _plist(s_, d_, k_, me, P):
    return ([np.sort(k_[(s_ == me) & (d_ == p)]) for p in range(P)], [np.sort(k_[(d_ == me) & (s_ == p)]) for p in range(P)])


def _i32(x):
    x = x if x.numel() else torch.zeros(1, dtype=torch.int64, device=x.device)
    return x.to(torch.int32).contiguous()


class DecompW123:
    def __init__(self, C, m, plan_path, hq, hk, rank, P, dev, kind):
        """m: virtual tile mask [n, n] (0 / 1 / 2 causal diagonal); plan: Dex's head-split plan (fwd or bwd); kind fwd|bwd"""
        self.C, self.hq, self.hk, self.rank, self.P, self.dev, self.kind = C, hq, hk, rank, P, dev, kind
        self.cs = torch.cuda.Stream(device=dev) if dev.type == "cuda" else None
        pl = json.load(open(plan_path))
        n = m.shape[0]; per = pl["block_tokens"] // T; nbp = n // per
        devb = np.asarray(pl["schedule_table"], np.int64)[:nbp, :nbp]; ownb = np.asarray(pl["owner"], np.int64)[:nbp]
        own = np.repeat(ownb, per)[:n]
        self.n, self.own = n, own
        self.owned = np.flatnonzero(own == rank)                        # local tiles, ascending (Dex's owned order)
        node_of = np.arange(P) // GPN
        i, j = np.nonzero(m)
        kd = m[i, j] == 2
        g = devb[i // per, j // per]
        assert (g >= 0).all(), "a live tile pair without a GPU in the plan"
        self.moved_pairs = int((g != own[i]).sum())
        # needs: unit not owned by the computing GPU, unique (GPU, tile)
        qn = np.unique(g[own[i] != g] * n + i[own[i] != g]); kn = np.unique(g[own[j] != g] * n + j[own[j] != g])
        L = {}
        for nm, nd in (("q", qn), ("kv", kn)):
            r_, key = nd // n, nd % n
            L[nm] = (relay(r_, key, own[key], node_of, rank), r_, key)
        me = rank
        lpos = np.full(n, -1, np.int64); lpos[self.owned] = np.arange(len(self.owned))
        nl = len(self.owned)
        (aq, bq, mkq, stq), rq_all, kq_all = L["q"]; (ak, bk, mkk, stk), rk_all, kk_all = L["kv"]
        NQ, NK = nl + len(mkq), nl + len(mkk)
        self.nl, self.NQ, self.NK = nl, NQ, NK
        T_ = lambda x: torch.as_tensor(x, device=dev)  # noqa: E731
        steps = []
        for si, (sq_, sk_) in enumerate(((aq, ak), (bq, bk))):
            sq, rq = _plist(*sq_, me, P); sk, rk = _plist(*sk_, me, P)
            srcq = (lambda ks: lpos[ks]) if si == 0 else (lambda ks: nl + np.searchsorted(mkq, ks))
            srck = (lambda ks: lpos[ks]) if si == 0 else (lambda ks: nl + np.searchsorted(mkk, ks))
            steps.append(dict(sq=[T_(srcq(x)) if len(x) else None for x in sq], sk=[T_(srck(x)) if len(x) else None for x in sk],
                              rq=[T_(nl + np.searchsorted(mkq, x)) if len(x) else None for x in rq],
                              rk=[T_(nl + np.searchsorted(mkk, x)) if len(x) else None for x in rk]))
        self.steps = steps
        # returns (step 3): my received units -> their owners; what the others return to me
        ownq = own[mkq] if len(mkq) else mkq; ownk = own[mkk] if len(mkk) else mkk
        ret = dict(q_send=[], k_send=[], q_recv=[], k_recv=[])
        for p in range(P):
            x = np.flatnonzero(ownq == p) + nl; ret["q_send"].append(T_(x) if len(x) else None)
            x = np.flatnonzero(ownk == p) + nl; ret["k_send"].append(T_(x) if len(x) else None)
        for r_, key, out in ((rq_all, kq_all, ret["q_recv"]), (rk_all, kk_all, ret["k_recv"])):
            mine = own[key] == me
            for p in range(P):
                ks = np.sort(key[mine & (r_ == p)])
                out.append(T_(lpos[ks]) if len(ks) else None)
        self.ret = ret
        # this rank's compute entries: slots, ready step
        sel = g == me
        i, j, kd = i[sel], j[sel], kd[sel]
        qslot = np.where(own[i] == me, lpos[i], nl + np.searchsorted(mkq, i)) if len(i) else i
        kslot = np.where(own[j] == me, lpos[j], nl + np.searchsorted(mkk, j)) if len(j) else j
        stepq = np.zeros(n, np.int64); stepq[mkq] = stq
        stepk = np.zeros(n, np.int64); stepk[mkk] = stk
        step = np.maximum(np.where(own[i] == me, 0, stepq[i]), np.where(own[j] == me, 0, stepk[j]))
        if len(mkq):
            assert (mkq[np.searchsorted(mkq, i[own[i] != me])] == i[own[i] != me]).all()
        if len(mkk):
            assert (mkk[np.searchsorted(mkk, j[own[j] != me])] == j[own[j] != me]).all()
        self.rowg = np.r_[self.owned, mkq]                               # global tile of each Q buffer row
        self.tabs = [self._fwd_tab(qslot[step == s], kslot[step == s], kd[step == s], j[step == s]) if kind == "fwd" else
                     self._bwd_tab(qslot[step == s], kslot[step == s], kd[step == s]) for s in range(3)]
        self.n_entries = int(sel.sum())
        # statistics: inter-node KV + Q bytes received per node (relay data flow, inputs only)
        qb, kb = T * hq * D * 2 * (2 if kind == "bwd" else 1) + (2 * hq * T * 4 if kind == "bwd" else 0), 2 * T * hk * D * 2
        rx = np.zeros(P // GPN)
        for (a_s, a_d, _), bytes_ in ((aq, qb), (ak, kb)):
            x = node_of[a_s] != node_of[a_d]
            rx += np.bincount(node_of[a_d[x]], minlength=P // GPN) * bytes_
        self.inter_node_max_MiB = float(rx.max()) / 2**20

    # ------------------------------------------------------------------------------------------- tables
    def _fwd_tab(self, qs, ks, kd, col):
        if len(qs) == 0:
            return None
        dev, NQ, NK, hq = self.dev, self.NQ, self.NK, self.hq
        o = np.lexsort((col, kd.astype(np.int8), qs))                 # per row: full tiles ascending, causal last
        qs, ks, kd, col = qs[o], ks[o], kd[o], col[o]
        cnt = np.bincount(qs, minlength=NQ); nmask = np.bincount(qs[kd], minlength=NQ)
        indptr = np.zeros(NQ + 1, np.int64); indptr[1:] = np.cumsum(cnt)
        shift = NK - NQ
        nb = col + shift
        order = np.argsort(-cnt, kind="stable"); order = order[cnt[order] > 0]
        work = (np.arange(hq)[:, None] * NQ + order[None, :]).reshape(-1)
        t = lambda x: torch.as_tensor(np.asarray(x, np.int64) if len(x) else np.zeros(1, np.int64), device=dev).to(torch.int32)  # noqa: E731
        return (t(indptr), t(ks), t(nb), t(nmask), t(self.rowg), t(work))

    def _bwd_tab(self, qs, ks, kd):
        """KV-head-major tasks (KV slot, half) x items (Q slot, 64-row half, query head, causal) as items_for"""
        if len(qs) == 0:
            return None
        dev, hq, hk = self.dev, self.hq, self.hk
        r_grp = hq // hk
        qs_, ks_, kd_ = (torch.as_tensor(x, device=dev) for x in (qs, ks, kd))
        rows = []
        for hn in (0, 1):
            for mb in (1, 0):
                ok = (~kd_) | torch.tensor(mb >= hn, device=dev)
                causal = kd_ & torch.tensor(mb == hn, device=dev)
                rows.append(torch.stack([ks_[ok], torch.full_like(qs_[ok], hn), qs_[ok], torch.full_like(qs_[ok], mb),
                                         causal[ok].long()], 1))
        E = torch.cat(rows)                                             # [e, (kslot, hn, qslot, mb, causal)]
        G = torch.arange(hk, device=dev)
        Eg = E[None].expand(hk, -1, -1).reshape(-1, 5); gg = G[:, None].expand(hk, len(E)).reshape(-1)
        # task key: g, kslot, hn; inside a task: q slot descending, mb 1 then 0
        NQ = self.NQ
        key = ((gg * self.NK + Eg[:, 0]) * 2 + Eg[:, 1]) * (2 * NQ) + (NQ - 1 - Eg[:, 2]) * 2 + (1 - Eg[:, 3])
        o = torch.argsort(key); Eg, gg, key = Eg[o], gg[o], key[o]
        tkey = key // (2 * NQ)
        ut, cnt = torch.unique_consecutive(tkey, return_counts=True)
        tg = ut // (2 * self.NK); tks = (ut // 2) % self.NK; thn = ut % 2
        task = (tks << 9) | (thn << 8) | tg
        hqs = gg[:, None] * r_grp + torch.arange(r_grp, device=dev)[None, :]     # [e, r_grp]
        it = (Eg[:, 2:3] << 9) | (Eg[:, 3:4] << 8) | (hqs << 1) | Eg[:, 4:5]
        ptr = torch.zeros(len(cnt) + 1, dtype=torch.int64, device=dev); ptr[1:] = torch.cumsum(cnt * r_grp, 0)
        return (_i32(task), _i32(ptr), _i32(it.reshape(-1)))

    # ------------------------------------------------------------------------------------------- exchange
    def _xchg(self, sends, recvs):
        ops, post = [], []
        for p, parts in enumerate(sends):
            parts = [x for x in parts if x is not None]
            if parts:
                buf = torch.cat([x.reshape(-1).view(torch.uint8) for x in parts]).contiguous()
                ops.append(dist.P2POp(dist.isend, buf, p))
        for p, parts in enumerate(recvs):
            parts = [x for x in parts if x is not None]
            if parts:
                nb = sum(X.numel() // X.shape[0] * len(rows) * X.element_size() for dst, rows in parts for X in dst)
                buf = torch.empty(nb, dtype=torch.uint8, device=self.dev)
                ops.append(dist.P2POp(dist.irecv, buf, p)); post.append((buf, parts))
        if ops:
            for rq in dist.batch_isend_irecv(ops):
                rq.wait()
        for buf, parts in post:
            o_ = 0
            for dst, rows in parts:
                for X in dst:
                    w = X.numel() // X.shape[0] * len(rows) * X.element_size()
                    X[rows] = buf[o_:o_ + w].view(X.dtype).view((len(rows),) + tuple(X.shape[1:])); o_ += w

    def _send_parts(self, st, qbufs, kbufs):
        sends, recvs = [], []
        for p in range(self.P):
            s_ = []
            if st["sq"][p] is not None:
                s_ += [X[st["sq"][p]] for X in qbufs]
            if st["sk"][p] is not None:
                s_ += [X[st["sk"][p]] for X in kbufs]
            sends.append(s_)
            r_ = []
            if st["rq"][p] is not None:
                r_.append((qbufs, st["rq"][p]))
            if st["rk"][p] is not None:
                r_.append((kbufs, st["rk"][p]))
            recvs.append(r_)
        return sends, recvs

    def _steps(self, qbufs, kbufs, comp):
        cur = torch.cuda.current_stream(); cs = self.cs
        cs.wait_stream(cur)
        with torch.cuda.stream(cs):
            self._xchg(*self._send_parts(self.steps[0], qbufs, kbufs))
        comp(0)
        cur.wait_stream(cs)
        with torch.cuda.stream(cs):
            self._xchg(*self._send_parts(self.steps[1], qbufs, kbufs))
        comp(1)
        cur.wait_stream(cs)
        comp(2)
        for X in qbufs + kbufs:
            X.record_stream(cs)

    # ------------------------------------------------------------------------------------------- forward
    def fwd(self, ql, kl, vl):
        """ql [nl T, HQ, D], kl / vl [nl T, HK, D] fp16 (owned tiles) -> o [nl T, HQ, D] fp16, lse [HQ, nl T] fp32"""
        C, dev, hq, hk, nl, NQ, NK = self.C, self.dev, self.hq, self.hk, self.nl, self.NQ, self.NK
        Qb = torch.empty(NQ, T, hq, D, dtype=torch.float16, device=dev)
        Kb = torch.empty(NK, T, hk, D, dtype=torch.float16, device=dev); Vb = torch.empty_like(Kb)
        Qb[:nl] = ql.view(nl, T, hq, D); Kb[:nl] = kl.view(nl, T, hk, D); Vb[:nl] = vl.view(nl, T, hk, D)
        scale = D ** -0.5
        parts = []

        def comp(s):
            tb = self.tabs[s]
            if tb is None:
                return
            out = torch.zeros(1, NQ * T, hq, D, dtype=torch.float16, device=dev)
            lse = torch.full((1, hq, NQ * T), LSE_EMPTY, dtype=torch.float32, device=dev)
            C.sparse_fwd(Qb.view(1, -1, hq, D), Kb.view(1, -1, hk, D), Vb.view(1, -1, hk, D), out, lse, *tb, scale)
            parts.append((out[0], lse[0]))
        self._steps([Qb], [Kb, Vb], comp)
        if parts:
            Ls = torch.stack([p_[1] for p_ in parts]); m = Ls.amax(0); w = torch.exp(Ls - m)          # [k, HQ, NQ T]
            ws = w.sum(0)
            O = sum((w[j_] / ws).t()[:, :, None] * parts[j_][0].float() for j_ in range(len(parts)))    # [NQ T, HQ, D]
            Lm = m + torch.log(ws)
        else:
            O = torch.zeros(NQ * T, hq, D, device=dev); Lm = torch.full((hq, NQ * T), LSE_EMPTY, device=dev)
        O = O.view(NQ, T, hq, D); Lt = Lm.view(hq, NQ, T).permute(1, 0, 2)                              # tile-major views
        rt, P = self.ret, self.P
        sends = [[O[rt["q_send"][p]], Lt[rt["q_send"][p]].contiguous()] if rt["q_send"][p] is not None else [] for p in range(P)]
        recvs, post = [], []
        for p in range(P):
            rows = rt["q_recv"][p]
            if rows is None:
                recvs.append([]); continue
            ob = torch.empty(len(rows), T, hq, D, device=dev); lb = torch.empty(len(rows), hq, T, device=dev)
            post.append((rows, ob, lb)); recvs.append([((ob, lb), torch.arange(len(rows), device=dev))])
        self._xchg(sends, recvs)
        Ol, Ll = O[:nl], Lt[:nl].contiguous()                                                         # [nl, T, HQ, D], [nl, HQ, T]
        for rows, ob, lb in post:
            l0 = Ll[rows]; mx = torch.maximum(l0, lb); w0 = torch.exp(l0 - mx); w1 = torch.exp(lb - mx)
            f0 = (w0 / (w0 + w1)).transpose(1, 2)[..., None]; f1 = (w1 / (w0 + w1)).transpose(1, 2)[..., None]
            Ol[rows] = f0 * Ol[rows] + f1 * ob
            Ll[rows] = mx + torch.log(w0 + w1)
        return Ol.reshape(nl * T, hq, D).to(torch.float16), Ll.permute(1, 0, 2).reshape(hq, nl * T)

    # ------------------------------------------------------------------------------------------- backward
    def bwd(self, ql, kl, vl, dol, lsel, deltal):
        """owned tiles: q / do [nl T, HQ, D], k / v [nl T, HK, D] fp16, lse / delta [HQ, nl T] fp32 -> dq, dk, dv fp16"""
        C, dev, hq, hk, nl, NQ, NK = self.C, self.dev, self.hq, self.hk, self.nl, self.NQ, self.NK
        scale = D ** -0.5
        Qb = torch.empty(NQ, T, hq, D, dtype=torch.float16, device=dev); dOb = torch.empty_like(Qb)
        Lt = torch.empty(NQ, hq, T, device=dev); Dt = torch.empty_like(Lt)
        Kb = torch.empty(NK, T, hk, D, dtype=torch.float16, device=dev); Vb = torch.empty_like(Kb)
        Qb[:nl] = ql.view(nl, T, hq, D); dOb[:nl] = dol.view(nl, T, hq, D)
        Lt[:nl] = lsel.view(hq, nl, T).permute(1, 0, 2); Dt[:nl] = deltal.view(hq, nl, T).permute(1, 0, 2)
        Kb[:nl] = kl.view(nl, T, hk, D); Vb[:nl] = vl.view(nl, T, hk, D)
        dq_acc = torch.zeros(NQ * T, hq, D, device=dev)
        dk_acc = torch.zeros(NK * T, hk, D, device=dev); dv_acc = torch.zeros_like(dk_acc)

        def comp(s):
            tb = self.tabs[s]
            if tb is None:
                return
            Lh = Lt.permute(1, 0, 2).reshape(hq, NQ * T); Dh = Dt.permute(1, 0, 2).reshape(hq, NQ * T)
            bk = torch.zeros_like(dk_acc); bv = torch.zeros_like(dv_acc)
            C.sparse_bwd(Qb.view(-1, hq, D), Kb.view(-1, hk, D), Vb.view(-1, hk, D), dOb.view(-1, hq, D), Lh, Dh,
                         dq_acc, None, None, bk, bv, *tb, scale)
            dk_acc.add_(bk); dv_acc.add_(bv)
        self._steps([Qb, dOb, Lt, Dt], [Kb, Vb], comp)
        rt, P = self.ret, self.P
        dq3, dk3, dv3 = dq_acc.view(NQ, T, hq, D), dk_acc.view(NK, T, hk, D), dv_acc.view(NK, T, hk, D)
        sends, recvs, post = [], [], []
        for p in range(P):
            s_ = []
            if rt["q_send"][p] is not None:
                s_.append(dq3[rt["q_send"][p]])
            if rt["k_send"][p] is not None:
                s_ += [dk3[rt["k_send"][p]], dv3[rt["k_send"][p]]]
            sends.append(s_)
            r_ = []
            if rt["q_recv"][p] is not None:
                b = torch.empty(len(rt["q_recv"][p]), T, hq, D, device=dev); r_.append(((b,), torch.arange(len(b), device=dev)))
                post.append((dq3, rt["q_recv"][p], b))
            if rt["k_recv"][p] is not None:
                bk_ = torch.empty(len(rt["k_recv"][p]), T, hk, D, device=dev); bv_ = torch.empty_like(bk_)
                r_.append(((bk_, bv_), torch.arange(len(bk_), device=dev)))
                post += [(dk3, rt["k_recv"][p], bk_), (dv3, rt["k_recv"][p], bv_)]
            recvs.append(r_)
        self._xchg(sends, recvs)
        for X, rows, b in post:
            X[rows] += b
        dq = torch.empty(nl * T, hq, D, dtype=torch.float16, device=dev)
        C.dq_convert(dq_acc[:nl * T].contiguous(), dq, scale)
        return dq, dk_acc[:nl * T].half(), dv_acc[:nl * T].half()


def latency(fn, iters=10, warmup=3):
    for _ in range(warmup):
        dist.barrier(); torch.cuda.synchronize(); fn(); torch.cuda.synchronize()
    dist.barrier()
    ts = []
    for _ in range(iters):
        dist.barrier(); torch.cuda.synchronize()
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        t = torch.tensor([time.perf_counter() - t0], device="cuda"); dist.all_reduce(t, op=dist.ReduceOp.MAX)
        ts.append(t.item() * 1e3)
    return float(np.median(ts)), [round(x, 3) for x in ts]


def tmax(x, dev):
    t = torch.tensor([float(x)], device=dev); dist.all_reduce(t, op=dist.ReduceOp.MAX); return t.item()


def run_fwd(ff, ranges, z, m, a, rank, P, dev, rec):
    T = int(z["tokens"]); HQ, HK = (int(x) for x in z["heads"]); scale = 1.0 / math.sqrt(D)
    t0 = time.perf_counter()
    ex = DecompW123(C, m, ff, HQ, HK, rank, P, dev, "fwd")
    rec["fwd_setup_s"] = round(tmax(time.perf_counter() - t0, dev), 2)
    rec["fwd_moved_pairs"] = ex.moved_pairs
    rec["fwd_inter_node_max_MiB"] = round(tmax(ex.inter_node_max_MiB, dev), 2)
    g = torch.Generator(device=dev).manual_seed(1234)
    q = (torch.randn(1, T, HQ, D, generator=g, device=dev, dtype=torch.float16) * 0.5)
    k = (torch.randn(1, T, HK, D, generator=g, device=dev, dtype=torch.float16) * 0.5)
    v = (torch.randn(1, T, HK, D, generator=g, device=dev, dtype=torch.float16) * 0.5)
    PAD = "tok_valid" in z.files
    if PAD:
        valid = torch.as_tensor(z["tok_valid"].astype(bool), device=dev); npad_g = torch.as_tensor(z["npad"].astype(np.int32), device=dev)
        k[:, ~valid] = 0; v[:, ~valid] = 0
    owned = ex.owned
    idx = torch.as_tensor((owned[:, None] * TILE + np.arange(TILE)).reshape(-1), device=dev)
    ql, kl, vl = q[0, idx].contiguous(), k[0, idx].contiguous(), v[0, idx].contiguous()
    res = {}

    def run():
        o, lse = ex.fwd(ql, kl, vl)
        if PAD:
            pad_correct(o, lse, npad_g[idx])
        res.update(o=o, lse=lse)
    rec["fwd_ms"], rec["fwd_all"] = latency(run)
    if a.check:
        run(); torch.cuda.synchronize()
        out = res["o"]; err = 0.0; errm = 0.0
        rng = np.random.RandomState(rank)
        for li in (rng.choice(len(owned), min(2, len(owned)), replace=False) if len(owned) else []):
            t = int(owned[li]); q0 = t * TILE
            cols = np.flatnonzero(m[t] > 0); causal_ = bool(m[t, t] == 2)
            if PAD and not bool(valid[q0:q0 + TILE].any()):
                continue
            ksel = torch.cat([k[:, int(c) * TILE:int(c) * TILE + TILE] for c in cols], 1)
            vsel = torch.cat([v[:, int(c) * TILE:int(c) * TILE + TILE] for c in cols], 1)
            L_ = ksel.shape[1]; grp_ = HQ // HK
            outs_ = []
            for hq_ in range(HQ):
                s_ = (q[0, q0:q0 + TILE, hq_].float() @ ksel[0, :, hq_ // grp_].float().T) * scale
                if causal_:
                    s_ = s_.masked_fill(torch.arange(L_, device=dev)[None, :] > torch.arange(TILE, device=dev)[:, None] + L_ - TILE, -float("inf"))
                if PAD:
                    kv_ok = torch.cat([valid[int(c) * TILE:int(c) * TILE + TILE] for c in cols])
                    s_ = s_.masked_fill(~kv_ok[None, :], -float("inf"))
                outs_.append(torch.softmax(s_, -1) @ vsel[0, :, hq_ // grp_].float())
            ref = torch.stack(outs_, 1)
            got = out[li * TILE:(li + 1) * TILE].float()
            if PAD:
                rv = valid[q0:q0 + TILE]; ref = ref[rv]; got = got[rv]
            err = max(err, ((got - ref).norm() / ref.norm().clamp_min(1e-12)).item())
            errm = max(errm, ((got - ref).abs().max() / ref.abs().max()).item())
        rec["fwd_err"] = tmax(err, dev); rec["fwd_err_maxabs"] = tmax(errm, dev)
        if a.fp64:
            rec["fwd_err_fp64"] = F64.check_virtual(q[0], k[0], v[0], None, np.load(ranges), scale,
                                                    dict(o=out), z, pos_rows=idx.cpu().numpy(), group=dist.group.WORLD)
    del ex, q, k, v, ql, kl, vl, res
    torch.cuda.empty_cache()


def bwd_oracle(m, h, hk, Q, K, V, dO, LSE, Delta, S, dev, scale):
    """the single-GPU kernel on the whole problem (KV-head-major tasks)"""
    n = m.shape[0]; r_grp = h // hk
    tk, tp, it = [], [0], []
    for gg in range(hk):
        for j in range(n):
            rows = np.flatnonzero(m[:, j])
            for hn in range(2):
                for i in rows[::-1]:
                    for mb in (1, 0):
                        if m[i, j] == 2:
                            if mb < hn:
                                continue
                            causal = mb == hn
                        else:
                            causal = False
                        it += [(int(i) << 9) | (mb << 8) | (hq << 1) | int(causal) for hq in range(gg * r_grp, (gg + 1) * r_grp)]
                tk.append((j << 9) | (hn << 8) | gg); tp.append(len(it))
    i32 = lambda x: torch.tensor(x, dtype=torch.int32, device=dev)  # noqa: E731
    dq_acc = torch.zeros(S, h, D, device=dev); dk_acc = torch.zeros(S, hk, D, device=dev); dv_acc = torch.zeros_like(dk_acc)
    C.sparse_bwd(Q, K, V, dO, LSE, Delta, dq_acc, None, None, dk_acc, dv_acc, i32(tk), i32(tp), i32(it), scale)
    dq_r = torch.empty_like(Q); C.dq_convert(dq_acc, dq_r, scale)
    return dq_r, dk_acc.half(), dv_acc.half()


def run_bwd(bf, ranges, z, m, a, rank, P, dev, rec):
    h, hk = (int(x) for x in z["heads"]); n = m.shape[0]; S = n * TILE; scale = D ** -0.5
    t0 = time.perf_counter()
    ex = DecompW123(C, m, bf, h, hk, rank, P, dev, "bwd")
    rec["bwd_setup_s"] = round(tmax(time.perf_counter() - t0, dev), 2)
    rec["bwd_moved_pairs"] = ex.moved_pairs
    rec["bwd_inter_node_max_MiB"] = round(tmax(ex.inter_node_max_MiB, dev), 2)
    g = torch.Generator(device=dev).manual_seed(1234)
    rnd = lambda *s: (torch.randn(*s, generator=g, device=dev, dtype=torch.float16) * 0.5)  # noqa: E731
    PAD = "tok_valid" in z.files
    valid = torch.as_tensor(z["tok_valid"].astype(bool), device=dev) if PAD else None
    rows = torch.as_tensor((ex.owned[:, None] * TILE + np.arange(TILE)).reshape(-1), device=dev)
    if a.check:
        Q, K, V, dO, O = rnd(S, h, D), rnd(S, hk, D), rnd(S, hk, D), rnd(S, h, D), rnd(S, h, D)
        LSE = torch.full((h, S), 6.0, device=dev)
        if PAD:
            K[~valid] = 0; V[~valid] = 0; dO[~valid] = 0
        Delta = (dO.float() * O.float()).sum(-1).t().contiguous()
        Ql, dOl, Kl, Vl = Q[rows], dO[rows], K[rows], V[rows]; Ll, Dl = LSE[:, rows].contiguous(), Delta[:, rows].contiguous()
    else:
        nr = rows.numel()
        Ql, dOl, Kl, Vl, Ol = rnd(nr, h, D), rnd(nr, h, D), rnd(nr, hk, D), rnd(nr, hk, D), rnd(nr, h, D)
        if PAD:
            Kl[~valid[rows]] = 0; Vl[~valid[rows]] = 0; dOl[~valid[rows]] = 0
        Ll = torch.full((h, nr), 6.0, device=dev); Dl = (dOl.float() * Ol.float()).sum(-1).t().contiguous(); del Ol
    res = {}

    def run():
        dq, dk, dv = ex.bwd(Ql, Kl, Vl, dOl, Ll, Dl)
        res.update(dq=dq, dk=dk, dv=dv)
    run(); torch.cuda.synchronize()
    if a.check:
        if a.fp64:
            pr = rows.cpu().numpy()
            rec["bwd_err_fp64"] = F64.check_virtual(Q, K, V, dO, np.load(ranges), scale,
                                                    dict(dq=res["dq"], dk=res["dk"], dv=res["dv"]), z, pos_rows=pr, pos_kv=pr,
                                                    group=dist.group.WORLD, lse_in=LSE.t(), o_in=O)
        ref = bwd_oracle(m, h, hk, Q, K, V, dO, LSE, Delta, S, dev, scale)
        e = []
        for got, r_ in ((res["dq"], ref[0]), (res["dk"], ref[1]), (res["dv"], ref[2])):
            rr_ = r_[rows].float(); gg = got.float()
            if PAD:
                ok = valid[rows]; rr_ = rr_[ok]; gg = gg[ok]
            e.append(((gg - rr_).norm() / rr_.norm().clamp_min(1e-12)).item())
        et = torch.tensor(e, device=dev); dist.all_reduce(et, op=dist.ReduceOp.MAX)
        rec["bwd_errs"] = [round(x, 6) for x in et.tolist()]
        del ref, Q, K, V, dO, O, LSE, Delta
        torch.cuda.empty_cache()
    rec["bwd_ms"], rec["bwd_all"] = latency(run)
    del ex, res
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workload", required=True, choices=["w1", "w2", "w3"])
    ap.add_argument("--ids", default="", help="comma list of input ids (default: all)")
    ap.add_argument("--head-split", type=int, default=4)
    ap.add_argument("--launch", type=int, default=1, help="1 = timing + checks, > 1 = timing only")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    a.check = a.fp64 = int(a.launch == 1)
    out = a.out or os.path.join(cfg("DEX_RESULT_ROOT"), "ablation_executor", a.workload, f"decomposed_launch{a.launch}.jsonl")
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2))
    rank, P = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank % GPN)))
    dev = torch.device("cuda", torch.cuda.current_device())
    done = {json.loads(ln)["input_id"] for ln in open(out)} if os.path.exists(out) else set()
    for c in SI.inputs(a.workload):
        if c["id"] in done or (a.ids and c["id"] not in a.ids.split(",")):
            continue
        fm, ff, bf = SI.plan_files(c, a.head_split)
        z = np.load(fm)
        m = z["mask"].astype(np.int8)
        rec = dict(input_id=c["id"], workload=a.workload.upper(), S=c["S"], system="decomposed", launch=a.launch, heads=[int(x) for x in z["heads"]], tiles=int(m.shape[0]))
        try:
            run_fwd(ff, c["ranges"], z, m, a, rank, P, dev, rec)
            run_bwd(bf, c["ranges"], z, m, a, rank, P, dev, rec)
        except Exception as ex:  # noqa: BLE001  one input does not lose the launch
            rec["error"] = f"{type(ex).__name__}: {str(ex)[:400]}"
            print(f"[rank {rank}] {c['id']}: {rec['error']}", flush=True)
        if rank == 0:
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print("[decomposed]", json.dumps({k: v for k, v in rec.items() if not k.endswith("_all")}), flush=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
