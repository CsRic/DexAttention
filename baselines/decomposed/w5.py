"""Dex's W5 plan on a decomposed executor (forward + backward), the W5 row of the executor ablation.

Plan     : rebuilt with Dex's assignment procedure (dex.attn.sparge.DexSparge: the layer's per-head 128 x 64 mask all-gathered, the assigner
           per head (forward plan for the forward, backward plan for the backward), rank 0's plan broadcast). W5's plans
           move work off the query owner, so query-side data travels too.
Units    : Q unit = (head h, global tile t) of the query owner (forward: Q; backward: Q, dO fp16 + LSE, delta fp32); KV unit
           = (h, t) of the key owner (K + V). A GPU needs a unit when one of its block pairs (pb[h, i, j] == GPU) reads it.
Data flow: node-level input reuse with this executor's own gateway selection: a unit a remote node needs crosses the network once, owner -> a landing GPU of that node
           that needs it (needers sorted, landing = needers[key % #needers]); the landing GPU forwards it inside the
           node; same-node needers get it directly.
Steps    : host-launched NCCL (one batch_isend_irecv per step on a communication stream, one packed message per peer per
           step: Q units then KV units, sorted by key), compute on the default stream as data arrives:
             step 0: owner -> {landing, same-node needers}      | compute pairs whose Q and KV are both local
             step 1: landing -> other needers of its node       | compute pairs ready after step 0
             step 2: -                                          | compute pairs ready after step 1
             step 3: results back to the owners (forward: merged partial O fp32 + LSE per moved Q unit; backward: dQ fp32
                     per moved Q unit, dK + dV fp32 per moved KV unit), sent directly to the owner
           compute = the kernels every W5 system uses: forward one sparse_fwd64 per step (128 x 64 blocks); backward one
           sparse_bwd per step (task = (KV slot, half), items = the Q slot's two 64-row halves), dQ accumulated, dK / dV per
           step into a zeroed block then added. Forward partials merged in fp32 by LSE at the computing GPU, then at the
           owner; backward partial sums in fp32.
Differences from Dex's persistent executor: no per-message send order (a step's messages go together), no tile-level
overlap inside a step, no copy-engine / NVSHMEM transport, backward gradients returned directly (without Dex's gateway pre-reduction); the
step lists and kernel tables are this executor's planning."""

import numpy as np
import torch
import torch.distributed as dist

from dex.attn.sparge import GPUS_PER_NODE, DexSparge
from workloads.w5_sparge import selection as SA
from workloads.w5_sparge.common import LSE_EMPTY, SCALE, T, D, Reuse, cuda_ext


def _i32(x, dev):
    x = x if x.numel() else torch.zeros(1, dtype=torch.int64, device=dev)
    return x.to(torch.int32).contiguous()


def relay(r, key, src, node_of, me):
    """r receiver, key unit id, src owner (one row per (receiver, unit) need) -> step-0 / step-1 (src, dst, key) and this
    rank's received keys (sorted) with their arrival step (1 = step 0, 2 = forwarded in step 1)"""
    nr, ns = node_of[r], node_of[src]
    intra = nr == ns
    ie = np.nonzero(~intra)[0]
    nn = int(node_of.max()) + 1
    gk = key[ie] * nn + nr[ie]
    o = np.lexsort((r[ie], gk))
    ie = ie[o]
    gk = gk[o]
    if len(ie):
        _, gstart, gcnt = np.unique(gk, return_index=True, return_counts=True)
    else:
        gstart = gcnt = np.zeros(0, np.int64)
    gid = np.repeat(np.arange(len(gstart)), gcnt)
    land = r[ie][gstart + (key[ie][gstart] % np.maximum(gcnt, 1))][gid] if len(ie) else np.zeros(0, np.int64)
    a = (
        np.r_[src[intra], src[ie][gstart]],
        np.r_[r[intra], land[gstart] if len(ie) else np.zeros(0, np.int64)],
        np.r_[key[intra], key[ie][gstart]],
    )
    fw = r[ie] != land
    b = (land[fw], r[ie][fw], key[ie][fw])
    ka, kb = a[2][a[1] == me], b[2][b[1] == me]
    mk = np.r_[ka, kb]
    st = np.r_[np.ones(len(ka), np.int64), np.full(len(kb), 2)]
    o = np.argsort(mk, kind="stable")
    return a, b, mk[o], st[o]


def _plist(s_, d_, k_, me, P):
    """per peer: keys I send (sorted), keys I receive (sorted)"""
    sends = [np.sort(k_[(s_ == me) & (d_ == p)]) for p in range(P)]
    recvs = [np.sort(k_[(d_ == me) & (s_ == p)]) for p in range(P)]
    return sends, recvs


class DecomposedSparge(DexSparge):
    def __init__(self, lay, H):  # NCCL only: no symmetric region
        self.DA, self.lay, self.H, self.assigner = cuda_ext(), lay, H, True
        P = lay.P
        self.own_v = np.tile(lay.block_owners(), H)
        self.planners, self.tplan = {}, {}
        self.used = 0
        self.cs = torch.cuda.Stream(device=lay.tok.device)
        self.node_of = np.arange(P) // GPUS_PER_NODE

    # ---------------------------------------------------------------------------------------------- geometry
    def _geom(self):
        lay = self.lay
        P, per, n_pad = lay.P, lay.per, lay.n_pad
        t = np.arange(n_pad)
        c = t // per
        own = np.where(c < P, c, 2 * P - 1 - c)
        loc = np.where(c < P, 0, per) + t % per
        return own, loc

    def _build(self, mh, pb, kind):
        """step lists + kernel tables of this rank for the plan pb [H, nbh, nbh] on the mask mh [H, n_pad, 2 n_pad]"""
        lay, H, P, me, dev = self.lay, self.H, self.lay.P, self.lay.rank, mh.device
        n_pad, nl, per = lay.n_pad, lay.nl, lay.per
        own_np, loc_np = self._geom()
        own = torch.as_tensor(own_np, device=dev)
        loc = torch.as_tensor(loc_np, device=dev)
        h, qt, c = torch.nonzero(mh, as_tuple=True)
        kt = c // 2
        g = pb[h, qt // per, kt // per]
        NU = H * n_pad
        qneed = torch.unique(g[own[qt] != g] * NU + (h * n_pad + qt)[own[qt] != g])
        kneed = torch.unique(g[own[kt] != g] * NU + (h * n_pad + kt)[own[kt] != g])
        qn, kn = qneed.cpu().numpy(), kneed.cpu().numpy()
        lists = {}
        for nm, nd in (("q", qn), ("kv", kn)):
            r_, key = nd // NU, nd % NU
            src = own_np[key % n_pad]
            lists[nm] = relay(r_, key, src, self.node_of, me)
        nlq = H * nl
        mkq, stq = lists["q"][2], lists["q"][3]
        mkk, stk = lists["kv"][2], lists["kv"][3]
        NQ, NK = nlq + len(mkq), nlq + len(mkk)

        def lslot(keys):
            return (keys // n_pad) * nl + loc_np[keys % n_pad]

        steps = []
        for si in (0, 1):
            sq, rq = _plist(*lists["q"][si], me, P)
            sk, rk = _plist(*lists["kv"][si], me, P)
            src_q = lslot if si == 0 else (lambda ks: nlq + np.searchsorted(mkq, ks))
            src_k = lslot if si == 0 else (lambda ks: nlq + np.searchsorted(mkk, ks))
            steps.append(
                dict(
                    sq=[torch.as_tensor(src_q(x), device=dev) if len(x) else None for x in sq],
                    sk=[torch.as_tensor(src_k(x), device=dev) if len(x) else None for x in sk],
                    rq=[torch.as_tensor(nlq + np.searchsorted(mkq, x), device=dev) if len(x) else None for x in rq],
                    rk=[torch.as_tensor(nlq + np.searchsorted(mkk, x), device=dev) if len(x) else None for x in rk],
                )
            )
        # returns: my received units back to their owners (moved Q: partial O / dQ; moved KV: dK + dV)
        ret = dict(q_send=[], q_recv=[], k_send=[], k_recv=[])
        if len(mkq):
            ownq = own_np[mkq % n_pad]
        for p in range(P):
            ks = mkq[ownq == p] if len(mkq) else mkq
            ret["q_send"].append(torch.as_tensor(nlq + np.searchsorted(mkq, ks), device=dev) if len(ks) else None)
        ownk = own_np[mkk % n_pad] if len(mkk) else mkk
        for p in range(P):
            ks = mkk[ownk == p] if len(mkk) else mkk
            ret["k_send"].append(torch.as_tensor(nlq + np.searchsorted(mkk, ks), device=dev) if len(ks) else None)
        # what the others return to me: units I own that p received (p received unit u iff (p, u) is in the needs)
        for nm, nd, out in (("q", qn, ret["q_recv"]), ("kv", kn, ret["k_recv"])):
            r_, key = nd // NU, nd % NU
            mine = own_np[key % n_pad] == me
            for p in range(P):
                ks = np.sort(key[mine & (r_ == p)])
                out.append(torch.as_tensor(lslot(ks), device=dev) if len(ks) else None)
        # this rank's compute entries (pairs on me), slots, ready step
        sel = g == me
        h, qt, c, kt = h[sel], qt[sel], c[sel], kt[sel]
        uq, uk = h * n_pad + qt, h * n_pad + kt
        lq_slot = torch.full((NU,), -1, dtype=torch.int64, device=dev)
        lq_step = torch.zeros(NU, dtype=torch.int64, device=dev)
        lk_slot = torch.full((NU,), -1, dtype=torch.int64, device=dev)
        lk_step = torch.zeros(NU, dtype=torch.int64, device=dev)
        if len(mkq):
            t_ = torch.as_tensor(mkq, device=dev)
            lq_slot[t_] = nlq + torch.arange(len(mkq), device=dev)
            lq_step[t_] = torch.as_tensor(stq, device=dev)
        if len(mkk):
            t_ = torch.as_tensor(mkk, device=dev)
            lk_slot[t_] = nlq + torch.arange(len(mkk), device=dev)
            lk_step[t_] = torch.as_tensor(stk, device=dev)
        oq, ok = own[qt] == me, own[kt] == me
        qs = torch.where(oq, h * nl + loc[qt], lq_slot[uq])
        ks = torch.where(ok, h * nl + loc[kt], lk_slot[uk])
        assert bool((qs >= 0).all()) and bool((ks >= 0).all()), "a needed unit has no slot"
        step = torch.maximum(torch.where(oq, 0, lq_step[uq]), torch.where(ok, 0, lk_step[uk]))
        hn = c % 2
        tabs = []
        for s in range(3):
            m_ = step == s
            if not bool(m_.any()):
                tabs.append(None)
                continue
            q_, k_, hn_ = qs[m_], ks[m_], hn[m_]
            if kind == "fwd":
                col = 2 * k_ + hn_  # 64-key slot
                o = torch.argsort(q_ * (2 * NK) + col)
                q_, col = q_[o], col[o]
                cnt = torch.bincount(q_, minlength=NQ)
                ptr = torch.zeros(NQ + 1, dtype=torch.int64, device=dev)
                ptr[1:] = torch.cumsum(cnt, 0)
                order = torch.argsort(-cnt, stable=True)
                work = order[cnt[order] > 0]
                tabs.append(
                    (
                        _i32(ptr, dev),
                        _i32(col, dev),
                        _i32(col, dev),
                        _i32(torch.zeros(NQ, dtype=torch.int64, device=dev), dev),
                        _i32(torch.arange(NQ, device=dev), dev),
                        _i32(work, dev),
                    )
                )
            else:
                tk = k_ * 2 + hn_
                o = torch.argsort(tk * NQ + q_)
                tk, q_ = tk[o], q_[o]
                ut, cnt = torch.unique_consecutive(tk, return_counts=True)
                kvs, hh = ut // 2, ut % 2
                task_kv = ((kvs << 9) | (hh << 8)).to(torch.int32)
                items = torch.stack([(q_ << 9) | (1 << 8), q_ << 9], 1).reshape(-1).to(torch.int32)
                ptr = torch.zeros(len(cnt) + 1, dtype=torch.int64, device=dev)
                ptr[1:] = torch.cumsum(2 * cnt, 0)
                tabs.append((task_kv.contiguous(), ptr.to(torch.int32).contiguous(), items.contiguous()))
        tb = 2 * T * D * 2
        a_s, a_d = lists["kv"][0][0], lists["kv"][0][1]
        inter_rx = (
            np.bincount(self.node_of[a_d[self.node_of[a_s] != self.node_of[a_d]]], minlength=P // GPUS_PER_NODE) * tb
        )
        return dict(
            steps=steps, ret=ret, tabs=tabs, NQ=NQ, NK=NK, nlq=nlq, inter_kv_node_max_MiB=float(inter_rx.max()) / 2**20
        )

    # ---------------------------------------------------------------------------------------------- exchange
    def _xchg(self, sends, recvs):
        """sends / recvs: per peer a list of row tensors to send / a list of (destination tensors, rows) -> one packed
        message per peer (raw bytes), unpacked into the destination rows"""
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
                buf = torch.empty(nb, dtype=torch.uint8, device=parts[0][1].device)
                ops.append(dist.P2POp(dist.irecv, buf, p))
                post.append((buf, parts))
        if ops:
            for rq in dist.batch_isend_irecv(ops):
                rq.wait()
        for buf, parts in post:
            o_ = 0
            for dst, rows in parts:
                for X in dst:
                    w = X.numel() // X.shape[0] * len(rows) * X.element_size()
                    X[rows] = buf[o_ : o_ + w].view(X.dtype).view((len(rows),) + tuple(X.shape[1:]))
                    o_ += w

    def _send_parts(self, st, qbufs, kbufs):
        P = self.lay.P
        sends, recvs = [], []
        for p in range(P):
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

    def _steps(self, tab, qbufs, kbufs, comp):
        """step 0-1 transfers on the communication stream, overlapping the compute of the previous step's data"""
        cur = torch.cuda.current_stream()
        cs = self.cs
        cs.wait_stream(cur)
        with torch.cuda.stream(cs):
            self._xchg(*self._send_parts(tab["steps"][0], qbufs, kbufs))
        comp(0)
        cur.wait_stream(cs)
        with torch.cuda.stream(cs):
            self._xchg(*self._send_parts(tab["steps"][1], qbufs, kbufs))
        comp(1)
        cur.wait_stream(cs)
        comp(2)
        for X in qbufs + kbufs:
            X.record_stream(cs)

    # ---------------------------------------------------------------------------------------------- plan
    def _plan_tab(self, fm, kind):
        lay = self.lay
        t0 = self._t0()
        mh = self._mask_global(lay.tile_rows(fm))
        self._tick(f"{kind}_mask_ms", t0)
        pb = self._plan(mh, (kind,))[kind]
        t0 = self._t0()
        tab = self._build(mh, pb, kind)
        self._tick(f"{kind}_decomp_tables_ms", t0)
        tab["moved_pairs"] = int(
            (pb != torch.as_tensor(self.own_v, device=pb.device).view(pb.shape[0], -1)[:, :, None]).sum()
        )
        return tab

    # ---------------------------------------------------------------------------------------------- forward
    def fwd(self, q, k, v):
        lay, H, dev = self.lay, self.H, q.device
        c = Reuse.cache()
        fm = c["fm"] if c is not None and "fm" in c else SA.select_split(q, k, lay)
        tab = self._plan_tab(fm, "fwd")
        self.last_tab = tab
        nlq, NQ, NK = tab["nlq"], tab["NQ"], tab["NK"]
        Qb = torch.empty(NQ, T, D, dtype=torch.float16, device=dev)
        Kb = torch.empty(NK, T, D, dtype=torch.float16, device=dev)
        Vb = torch.empty_like(Kb)
        Qb[:nlq] = lay.pad(q).reshape(nlq, T, D)
        Kb[:nlq] = lay.pad(k).reshape(nlq, T, D)
        Vb[:nlq] = lay.pad(v).reshape(nlq, T, D)
        parts = []

        def comp(s):
            tb = tab["tabs"][s]
            if tb is None:
                return
            out = torch.zeros(1, NQ * T, 1, D, dtype=torch.float16, device=dev)
            lse = torch.full((1, 1, NQ * T), LSE_EMPTY, dtype=torch.float32, device=dev)
            self.DA.sparse_fwd64(Qb.view(1, -1, 1, D), Kb.view(1, -1, 1, D), Vb.view(1, -1, 1, D), out, lse, *tb, SCALE)
            parts.append((out[0, :, 0].float(), lse[0, 0]))

        self._steps(tab, [Qb], [Kb, Vb], comp)
        if parts:
            Ls = torch.stack([p_[1] for p_ in parts])
            m = Ls.amax(0)
            w = torch.exp(Ls - m)
            O = sum(w[j][:, None] * parts[j][0] for j in range(len(parts))) / w.sum(0)[:, None]
            L_ = m + torch.log(w.sum(0))
        else:
            O = torch.zeros(NQ * T, D, device=dev)
            L_ = torch.full((NQ * T,), LSE_EMPTY, device=dev)
        O = O.view(NQ, T, D)
        L_ = L_.view(NQ, T)
        # step 3: moved Q units' partials back to their owners (fp32 O + LSE), merged there
        rt = tab["ret"]
        P = lay.P
        sends = [[O[rt["q_send"][p]], L_[rt["q_send"][p]]] if rt["q_send"][p] is not None else [] for p in range(P)]
        rbufs, recvs = [], []
        for p in range(P):
            rows = rt["q_recv"][p]
            if rows is None:
                recvs.append([])
                continue
            ob = torch.empty(len(rows), T, D, device=dev)
            lb = torch.empty(len(rows), T, device=dev)
            rbufs.append((rows, ob, lb))
            recvs.append([((ob, lb), torch.arange(len(rows), device=dev))])
        self._xchg(sends, recvs)
        for rows, ob, lb in rbufs:
            l0 = L_[rows]
            m = torch.maximum(l0, lb)
            w0 = torch.exp(l0 - m)
            w1 = torch.exp(lb - m)
            O[rows] = (w0[..., None] * O[rows] + w1[..., None] * ob) / (w0 + w1)[..., None]
            L_[rows] = m + torch.log(w0 + w1)
        o = O[:nlq].view(H, lay.L_pad, D).to(torch.float16)
        lse = L_[:nlq].reshape(H, lay.L_pad)
        return lay.unpad(o), (fm, lay.unpad(lse))

    # ---------------------------------------------------------------------------------------------- backward
    def bwd(self, q, k, v, o, do, fm, lse, cache=None):
        lay, H, dev, DA = self.lay, self.H, q.device, self.DA
        tab = self._plan_tab(fm, "bwd")
        nlq, NQ, NK = tab["nlq"], tab["NQ"], tab["NK"]
        q, k, v, o, do = (lay.pad(x) for x in (q, k, v, o, do))
        lse = lay.pad(lse)
        Qb = torch.empty(NQ, T, D, dtype=torch.float16, device=dev)
        dOb = torch.empty_like(Qb)
        Lb = torch.empty(NQ, T, device=dev)
        Db = torch.empty(NQ, T, device=dev)
        Kb = torch.empty(NK, T, D, dtype=torch.float16, device=dev)
        Vb = torch.empty_like(Kb)
        Qb[:nlq] = q.reshape(nlq, T, D)
        dOb[:nlq] = do.reshape(nlq, T, D)
        Lb[:nlq] = lse.reshape(nlq, T)
        Kb[:nlq] = k.reshape(nlq, T, D)
        Vb[:nlq] = v.reshape(nlq, T, D)
        dps = torch.empty(1, nlq * T, device=dev)
        DA.dot_do_o(do.reshape(nlq * T, 1, D), o.reshape(nlq * T, 1, D), dps, None)
        Db[:nlq] = dps.view(nlq, T)
        dq_acc = torch.zeros(NQ * T, 1, D, device=dev)
        dk_acc = torch.zeros(NK * T, 1, D, device=dev)
        dv_acc = torch.zeros_like(dk_acc)

        def comp(s):
            tb = tab["tabs"][s]
            if tb is None:
                return
            bk = torch.zeros_like(dk_acc)
            bv = torch.zeros_like(dv_acc)
            DA.sparse_bwd(
                Qb.view(-1, 1, D),
                Kb.view(-1, 1, D),
                Vb.view(-1, 1, D),
                dOb.view(-1, 1, D),
                Lb.view(1, -1),
                Db.view(1, -1),
                dq_acc,
                None,
                None,
                bk,
                bv,
                *tb,
                SCALE,
            )
            dk_acc.add_(bk)
            dv_acc.add_(bv)

        self._steps(tab, [Qb, dOb, Lb, Db], [Kb, Vb], comp)
        # step 3: dQ of moved Q units, dK + dV of moved KV units back to their owners (fp32), summed there
        rt, P = tab["ret"], lay.P
        dq3, dk3, dv3 = dq_acc.view(NQ, T, D), dk_acc.view(NK, T, D), dv_acc.view(NK, T, D)
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
                b = torch.empty(len(rt["q_recv"][p]), T, D, device=dev)
                r_.append(((b,), torch.arange(len(b), device=dev)))
                post.append((dq3, rt["q_recv"][p], b))
            if rt["k_recv"][p] is not None:
                bk_ = torch.empty(len(rt["k_recv"][p]), T, D, device=dev)
                bv_ = torch.empty_like(bk_)
                r_.append(((bk_, bv_), torch.arange(len(bk_), device=dev)))
                post += [(dk3, rt["k_recv"][p], bk_), (dv3, rt["k_recv"][p], bv_)]
            recvs.append(r_)
        self._xchg(sends, recvs)
        for X, rows, b in post:
            X[rows] += b
        dq = torch.empty(nlq * T, 1, D, dtype=torch.float16, device=dev)
        DA.dq_convert(dq_acc[: nlq * T].contiguous(), dq, SCALE)
        L = lay.L_pad
        return (
            lay.unpad(dq.view(H, L, D)),
            lay.unpad(dk_acc[: nlq * T].view(H, L, D).half()),
            lay.unpad(dv_acc[: nlq * T].view(H, L, D).half()),
        )
