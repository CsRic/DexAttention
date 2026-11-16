"""Dex's W4 plan on a decomposed executor (forward + backward), the W4 row of the executor ablation. Same interface as
baselines.allgather.infllm.SparseAllgatherInfllm.

Plan     : rebuilt with Dex's assignment procedure (dex.attn.infllm.Plan2D: planner input all-gather, joint planner + home gate, one broadcast,
           the needs / unit all-reduce). The decomposed executor runs home plans only (a moved unit raises RuntimeError).
Unit     : one 64-token key block n (K | V of both KV-head groups, [64, 2, 2, 128] bf16), owner = its chunk's owner; GPU
           c needs n when needs[c, n] (Plan2D.exchange) and c does not own it.
Data flow: node-level input reuse with this executor's own gateway selection: a block a remote NODE needs crosses the network once, owner -> a landing GPU of that node
           that needs it (needers sorted, landing = needers[n % #needers]); the landing GPU forwards it inside the node;
           same-node needers get it directly.
Steps    : host-launched NCCL (one batch_isend_irecv per step on a communication stream, one packed message per peer,
           blocks sorted), compute on the default stream as data arrives:
             step 0: owner -> {landing, same-node needers} | compute the entries on own blocks
             step 1: landing -> other needers of its node  | compute the entries on blocks received in step 0
             step 2: -                                     | compute the entries on blocks forwarded in step 1
           compute = the official InfLLM-v2 call of Sparse all-gather (Infllmv2AttnVarlenFunc.forward / .backward on the
           compact keys, selection remapped), with the selection restricted to the step's blocks. Forward partials (O, LSE)
           merged in FP32 by LSE; backward: dQ of every step summed in FP32, dK / dV summed in FP32 per store row, then the
           received blocks' FP32 dK / dV returned directly to their owners and summed there.
Differences from Dex's persistent executor: no per-message send order (a step's messages go together), no row-level
overlap inside a step, no copy-engine / NVSHMEM transport, dK / dV returned directly (no gateway pre-reduction); the step
lists are this executor's planning and stay inside the timed window."""
import numpy as np
import torch
from infllm_v2.infllmv2_sparse_attention import Infllmv2AttnVarlenFunc as IF

from baselines.allgather.infllm import SparseAllgatherInfllm, _Ctx, official_bwd
from dex.attn.infllm import Plan2D
from workloads.w4_infllm.common import BLK, GH, gpus_per_node

LSE_EMPTY = -1e4


def relay(r, key, src, node_of, me):
    """r receiver, key unit id, src owner, one row per (receiver, unit) need -> step-0 / step-1 (src, dst, key) and this
    rank's received keys (sorted) with their arrival step (1 / 2)"""
    nr, ns = node_of[r], node_of[src]
    intra = nr == ns
    ie = np.nonzero(~intra)[0]
    nn = int(node_of.max()) + 1
    gk = key[ie] * nn + nr[ie]
    o = np.lexsort((r[ie], gk)); ie = ie[o]; gk = gk[o]
    z = np.zeros(0, np.int64)
    if len(ie):
        _, gstart, gcnt = np.unique(gk, return_index=True, return_counts=True)
        gid = np.repeat(np.arange(len(gstart)), gcnt)
        land = r[ie][gstart + (key[ie][gstart] % gcnt)][gid]
    else:
        gstart, land = z, z
    a = (np.r_[src[intra], src[ie][gstart]], np.r_[r[intra], land[gstart] if len(ie) else z], np.r_[key[intra], key[ie][gstart]])
    fw = r[ie] != land
    b = (land[fw], r[ie][fw], key[ie][fw])
    ka, kb = a[2][a[1] == me], b[2][b[1] == me]
    mk = np.r_[ka, kb]; st = np.r_[np.ones(len(ka), np.int64), np.full(len(kb), 2)]
    o = np.argsort(mk, kind="stable")
    return a, b, mk[o], st[o]


class DecomposedInfllm(SparseAllgatherInfllm):
    def __init__(self, chunks, my, S, P, rank, dev):
        super().__init__(chunks, my, S, P, rank, dev)
        gpn = gpus_per_node(P)
        self.pl = Plan2D(chunks, my, S, P, rank, dev, gpn=gpn)
        self.node_of = np.arange(P) // gpn
        self.cs = torch.cuda.Stream(device=dev)

    # ------------------------------------------------------------------ plan + step lists
    def lists(self, sel, kind="fwd"):
        pl, P, r, nb = self.pl, self.P, self.rank, self.nb
        t0 = pl._t0()
        W, kblk = pl.stats(sel)
        t0 = pl._tick(f"{kind}_stats_ms", t0)
        pb = pl.plan(W, kblk, (kind,))[kind]
        t0 = pl._t0()
        e, needs, units = pl.exchange(sel, pb)
        t0 = pl._tick(f"{kind}_exchange_ms", t0)
        if len(units["c"]):
            raise RuntimeError("the plan moves query units: the decomposed W4 executor supports home plans only")
        need = needs & (self.own[None, :] != np.arange(P)[:, None])          # [P, nb] blocks c needs and does not own
        rr, key = np.nonzero(need)
        a, b, mk, st = relay(rr, key, self.own[key], self.node_of, r)
        crow = self.orow.copy()
        crow[mk] = self.L + np.arange(len(mk)) * BLK                         # [own rows ; received blocks in key order]
        avail = np.full(nb, -1, np.int64); avail[self.own == r] = 0; avail[mk] = st
        steps = []
        for (s_, d_, k_) in (a, b):
            sends = [np.sort(k_[(s_ == r) & (d_ == p)]) for p in range(P)]
            recvs = [np.sort(k_[(d_ == r) & (s_ == p)]) for p in range(P)]
            steps.append((sends, recvs))
        # dK / dV return (backward): my received blocks back to their owners; blocks I own that p received
        ret_send = [mk[self.own[mk] == p] for p in range(P)]
        ret_recv = [np.sort(key[(rr == p) & (self.own[key] == r)]) for p in range(P)]
        L = dict(steps=steps, crow=torch.as_tensor(crow, device=self.dev), crow_np=crow, nrecv=len(mk),
                 avail=torch.as_tensor(avail, device=self.dev), ret_send=ret_send, ret_recv=ret_recv, bytes_in=len(mk) * BLK * 1024,
                 bytes_out=int(sum(len(x) for x in steps[0][0]) + sum(len(x) for x in steps[1][0])) * BLK * 1024)
        pl._tick(f"{kind}_step_lists_ms", t0)
        return L

    def _brows(self, Lst, blocks):
        """store rows of blocks (own: clamped like SparseAllgatherInfllm._rows; received: 64 rows each)"""
        b = torch.as_tensor(Lst["crow_np"][blocks], device=self.dev)
        rows = b[:, None] + self.ar64[None, :]
        own = torch.as_tensor(self.own[blocks] == self.rank, device=self.dev)[:, None]
        return torch.where(own, rows.clamp(max=self.L - 1), rows).reshape(-1)

    def _xchg(self, store, Lst, s):
        sends, recvs = Lst["steps"][s]
        sb = [store[self._brows(Lst, b)].contiguous() if len(b) else None for b in sends]
        rb = [torch.empty(len(b) * BLK, *store.shape[1:], dtype=store.dtype, device=store.device) if len(b) else None for b in recvs]
        self._exchange(sb, rb)
        for b, t in zip(recvs, rb):
            if t is not None:
                store[self._brows(Lst, b)] = t

    def _stepped(self, Lst, store, comp):
        cur = torch.cuda.current_stream(); cs = self.cs
        cs.wait_stream(cur)
        with torch.cuda.stream(cs):
            self._xchg(store, Lst, 0)
        comp(0)
        cur.wait_stream(cs)
        with torch.cuda.stream(cs):
            self._xchg(store, Lst, 1)
        comp(1)
        cur.wait_stream(cs)
        comp(2)
        store.record_stream(cs)

    def _sel_step(self, Lst, sel, s):
        a = Lst["avail"][sel.clamp(min=0).long()]
        return torch.where((sel >= 0) & (a == s), sel, torch.full_like(sel, -1))

    # ------------------------------------------------------------------ forward / backward
    def fwd(self, q, k, v, sel):
        Lst = self.lists(sel, "fwd")
        store = torch.empty(self.L + Lst["nrecv"] * BLK, 2, 2, 128, dtype=k.dtype, device=k.device)
        store[:self.L] = torch.stack([k, v], 1)
        parts = []
        cnt = torch.stack([((sel >= 0) & (Lst["avail"][sel.clamp(min=0).long()] == s)).sum() for s in range(3)]).cpu().tolist()

        def comp(s):
            if cnt[s] == 0:
                return
            ss = self._sel_step(Lst, sel, s)
            K, V, cu_k, mk, tk, _ = self.compact(Lst, store, ss)
            o, lse, _ = IF.forward(_Ctx(), q.contiguous(), K, V, self._cu_q(), cu_k, max(self.Ls) * GH, mk, 0.0, None, True, (-1, -1),
                                   0.0, None, False, True, None, tk)
            lt = lse.t()                                                     # [L * 16, 2]
            emp = ~torch.isfinite(lt)
            lt = torch.where(emp, torch.full_like(lt, LSE_EMPTY), lt)
            parts.append((torch.where(emp[..., None], torch.zeros((), device=o.device), o.float()), lt))
        self._stepped(Lst, store, comp)
        if not parts:
            return torch.zeros_like(q), torch.full((2, q.shape[0]), LSE_EMPTY, dtype=torch.float32, device=self.dev)
        Ls_ = torch.stack([p_[1] for p_ in parts]); m = Ls_.amax(0); w = torch.exp(Ls_ - m)
        O = sum(w[j][..., None] * parts[j][0] for j in range(len(parts))) / w.sum(0)[..., None]
        lse = (m + torch.log(w.sum(0))).t().contiguous()
        self.last = dict(bytes_in=Lst["bytes_in"], bytes_out=Lst["bytes_out"])
        return O.to(q.dtype), lse

    def bwd(self, q, o, do, lse, k, v, sel):
        Lst = self.lists(sel, "bwd")
        n_st = self.L + Lst["nrecv"] * BLK
        store = torch.empty(n_st, 2, 2, 128, dtype=k.dtype, device=k.device)
        store[:self.L] = torch.stack([k, v], 1)
        acc = torch.zeros(n_st, 2, 2, 128, dtype=torch.float32, device=self.dev)
        dq_acc = torch.zeros(q.shape, dtype=torch.float32, device=self.dev)
        cnt = torch.stack([((sel >= 0) & (Lst["avail"][sel.clamp(min=0).long()] == s)).sum() for s in range(3)]).cpu().tolist()
        qc, oc, lc, doc = q.contiguous(), o.contiguous(), lse.contiguous(), do.contiguous()

        def comp(s):
            if cnt[s] == 0:
                return
            ss = self._sel_step(Lst, sel, s)
            K, V, cu_k, mk, tk, segs = self.compact(Lst, store, ss)
            dq, dK, dV = official_bwd(qc, K, V, oc, lc, self._cu_q(), cu_k, max(self.Ls) * GH, mk, tk, doc, q.shape[-1] ** -0.5)
            dq_acc.add_(dq.float())
            acc.index_add_(0, torch.cat([r_ for _, r_ in segs]), torch.stack([dK, dV], 1).float())
        self._stepped(Lst, store, comp)
        # received blocks' FP32 dK / dV back to their owners (direct), summed there
        sb = [acc[self._brows(Lst, b)].contiguous() if len(b) else None for b in Lst["ret_send"]]
        rb = [torch.empty(len(b) * BLK, 2, 2, 128, dtype=torch.float32, device=self.dev) if len(b) else None for b in Lst["ret_recv"]]
        self._exchange(sb, rb)
        own = acc[:self.L]
        for b, t in zip(Lst["ret_recv"], rb):
            if t is not None:
                own.index_add_(0, self._rows(b), t)
        self.last = dict(bytes_in=Lst["bytes_in"], bytes_out=Lst["bytes_out"])
        return dq_acc.to(q.dtype), own[:, 0].to(k.dtype), own[:, 1].to(k.dtype)
