"""Dex's W6 plan on a decomposed executor (forward only), the W6 row of the executor ablation.

Plan     : rebuilt with Dex's assignment procedure (dex.attn.flexprefill.DexFlex: bit-packed mask all-gather, assigner per KV-head unit, home
           gate, one broadcast). This executor supports home assignments only: every block pair runs on its zigzag query owner. A plan
           that moves work raises RuntimeError (query and partial-O messages are not part of this executor).
Data flow: node-level input reuse with this executor's own gateway selection: the transfer unit is (KV head g, global tile t) = K + V; a unit a remote node needs crosses
           the network once, owner -> a landing GPU of that node that needs it (needers sorted, landing =
           needers[(g n + t) % #needers]); the landing GPU forwards it inside the node; same-node needers get it directly.
Steps    : host-launched NCCL (one batch_isend_irecv per step on a communication stream, one contiguous message per peer,
           units sorted by (g, t)), compute on the default stream as data arrives:
             step 0: owner -> {landing GPUs, same-node needers}   | compute the rank's local key tiles
             step 1: landing -> other needers of the node         | compute the tiles received in step 0
             step 2: -                                            | compute the tiles forwarded in step 1
           each compute = one sparse_fwd (the kernel every W6 system uses) over the rank's rows x that step's key tiles,
           into its own (O fp16, LSE fp32); the three partials are merged in fp32 by LSE at the owner.
Differences from Dex's persistent executor: no per-message send order, no tile-level overlap inside a step, no copy-engine /
NVSHMEM transport; the step lists and kernel tables are this executor's planning."""

import time

import numpy as np
import torch
import torch.distributed as dist

from dex.attn.flexprefill import DexFlex
from workloads.w6_flexprefill.common import (
    GPUS_PER_NODE,
    GQ,
    HK,
    HQ,
    LSE_EMPTY,
    SCALE,
    T,
    D,
    cuda_ext,
    entries,
    from_q,
    fwd_tables,
    mask_global,
    to_kv,
    to_q,
)


class DecomposedFlex:
    plan_timing = False

    def __init__(self, lay):
        self.gate = DexFlex(lay, True)  # Dex's mask exchange + assigner + home gate
        self.DA = cuda_ext()
        self.cs = torch.cuda.Stream(device=lay.dev)
        self.tplan, self.assigner = {}, True
        self.retarget(lay)

    def retarget(self, lay):
        self.lay = lay
        self.gate.retarget(lay)
        self.node_of = np.arange(lay.P) // GPUS_PER_NODE

    @property
    def gate_log(self):
        return self.gate.gate_log

    def _t0(self):
        if DecomposedFlex.plan_timing:
            torch.cuda.synchronize()
        return time.perf_counter()

    def _tick(self, key, t0):
        if not DecomposedFlex.plan_timing:
            return 0.0
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        self.tplan[key] = (t1 - t0) * 1e3
        return t1

    def tables(self, mrows):
        lay, g8 = self.lay, self.gate
        DexFlex.plan_timing = DecomposedFlex.plan_timing
        g8.tplan.clear()
        t0 = self._t0()
        mh = mask_global(mrows, lay)
        self._tick("mask_ms", t0)
        pb = g8.plan(mh)  # (planner_ms, gate_ms, plan_bcast_ms in g8.tplan)
        DexFlex.plan_timing = False
        self.tplan.update(g8.tplan)
        t0 = self._t0()
        home = g8._ob[:, :, None].expand_as(pb)
        live = mh.view(HK, GQ, lay.nbh, lay.per, lay.nbh, lay.per).ne(0).any(5).any(3).any(1)
        if bool(((pb != home) & live).any()):
            raise RuntimeError("plan moves work off the query owner: this executor supports home plans only")
        tab = self._build(mh, mrows)
        self._tick("decomp_tables_ms", t0)
        self.last_mh, self.last_pb = mh, pb
        return tab

    def _build(self, mh, mrows):
        lay, P, n, nl, me = self.lay, self.lay.P, self.lay.n, self.lay.nl, self.lay.rank
        dev = lay.dev
        # need[g, r, t]: rank r's rows (any of KV head g's 4 query heads) read key tile t, t not owned by r
        need = mh[:, lay.rows_of].ne(0).view(HK, GQ, P, nl, n).any(3).any(1)  # [8, P, n]
        need &= lay.t_own[None, None, :] != torch.arange(P, device=dev)[None, :, None]
        g, r, t = [x.cpu().numpy() for x in torch.nonzero(need, as_tuple=True)]  # (the host sync)
        town = lay.t_own.cpu().numpy()
        tloc = lay.t_loc.cpu().numpy()
        src = town[t]
        nr, ns = self.node_of[r], self.node_of[src]
        key = g.astype(np.int64) * n + t  # the unit id
        intra = nr == ns
        # inter-node: group units by (unit, receiving node) -> landing = needers[(g n + t) % #needers]
        ie = np.nonzero(~intra)[0]
        NN = P // GPUS_PER_NODE
        gk = key[ie] * NN + nr[ie]
        o = np.lexsort((r[ie], gk))
        ie = ie[o]
        gk = gk[o]
        _, gstart, gcnt = np.unique(gk, return_index=True, return_counts=True)
        gid = np.repeat(np.arange(len(gstart)), gcnt)
        land = r[ie][gstart + (key[ie][gstart] % gcnt)][gid]
        # step 0: owner -> same-node needer, owner -> landing (once per node); step 1: landing -> other needers
        a_src = np.r_[src[intra], src[ie][gstart]]
        a_dst = np.r_[r[intra], land[gstart]]
        a_key = np.r_[key[intra], key[ie][gstart]]
        fw = r[ie] != land
        b_src, b_dst, b_key = land[fw], r[ie][fw], key[ie][fw]
        # my received units -> slots 8 nl + index (each unit reaches a rank once: direct, as landing, or forwarded)
        mk = np.sort(np.r_[a_key[a_dst == me], b_key[b_dst == me]])
        mstep = np.r_[np.ones((a_dst == me).sum(), np.int64), np.full((b_dst == me).sum(), 2)]
        mstep = mstep[np.argsort(np.r_[a_key[a_dst == me], b_key[b_dst == me]], kind="stable")]
        nloc = HK * nl

        def rslot(keys):
            return nloc + np.searchsorted(mk, keys)

        def lslot(keys):
            return (keys // n) * nl + tloc[keys % n]

        def msgs(s_, d_, k_, src_slot):
            sends, recvs = [], []
            for peer in range(P):
                ks = np.sort(k_[(s_ == me) & (d_ == peer)])
                if len(ks):
                    sends.append((peer, torch.as_tensor(src_slot(ks), device=dev)))
                kr = np.sort(k_[(d_ == me) & (s_ == peer)])
                if len(kr):
                    recvs.append((peer, torch.as_tensor(rslot(kr), device=dev)))
            return sends, recvs

        step0 = msgs(a_src, a_dst, a_key, lslot)
        step1 = msgs(b_src, b_dst, b_key, rslot)
        # compute entries of this rank: step 0 = local key tile, 1 = received in step 0, 2 = forwarded in step 1
        h, i, tt, kind = entries(mrows)
        gg = h // GQ
        lookup_slot = torch.full((HK * n,), -1, dtype=torch.int64, device=dev)
        lookup_step = torch.zeros(HK * n, dtype=torch.int64, device=dev)
        if len(mk):
            mkd = torch.as_tensor(mk, device=dev)
            lookup_slot[mkd] = torch.as_tensor(rslot(mk), device=dev)
            lookup_step[mkd] = torch.as_tensor(mstep, device=dev)
        own = lay.t_own[tt] == me
        u = gg * n + tt
        slot = torch.where(own, gg * nl + lay.t_loc[tt], lookup_slot[u])
        step = torch.where(own, torch.zeros_like(u), lookup_step[u])
        assert bool((slot >= 0).all()), "a needed key tile has no slot"
        R = HQ * nl
        NKV = nloc + len(mk)
        tabs = fwd_tables(h * nl + i, kind, slot, step, 3, R, NKV - R, tt)
        tb = 2 * T * D * 2
        inter_rx = np.bincount(self.node_of[a_dst[self.node_of[a_src] != self.node_of[a_dst]]], minlength=NN) * tb
        return dict(
            steps=(step0, step1),
            tabs=tabs,
            NKV=NKV,
            R=R,
            nrecv=len(mk),
            inter_node_max_MiB=float(inter_rx.max()) / 2**20,
            inter_node_total_MiB=float(inter_rx.sum()) / 2**20,
        )

    def _exchange(self, Kb, Vb, sends, recvs):
        """on the communication stream: pack K + V of the units per peer, one isend / irecv per peer, scatter into the slots"""
        ops, rbufs = [], []
        for peer, sl in sends:
            buf = torch.stack([Kb[sl], Vb[sl]], 1).contiguous()  # [m, 2, T, D]
            ops.append(dist.P2POp(dist.isend, buf, peer))
            rbufs.append(None)
        for peer, sl in recvs:
            buf = torch.empty(len(sl), 2, T, D, dtype=Kb.dtype, device=Kb.device)
            ops.append(dist.P2POp(dist.irecv, buf, peer))
            rbufs.append((sl, buf))
        if ops:
            for rq in dist.batch_isend_irecv(ops):
                rq.wait()
        for x in rbufs:
            if x is not None:
                sl, buf = x
                Kb[sl] = buf[:, 0]
                Vb[sl] = buf[:, 1]

    def _comp(self, qv, Kb, Vb, tb):
        out = torch.zeros(1, qv.shape[1], 1, D, dtype=torch.float16, device=qv.device)
        lse = torch.full((1, 1, qv.shape[1]), LSE_EMPTY, dtype=torch.float32, device=qv.device)
        self.DA.sparse_fwd(qv, Kb.view(1, -1, 1, D), Vb.view(1, -1, 1, D), out, lse, *tb, SCALE)
        return out[0, :, 0].float(), lse[0, 0]

    def run(self, q16, k16, v16, tab):
        lay = self.lay
        nl, dev = lay.nl, lay.dev
        Kb = torch.empty(tab["NKV"], T, D, dtype=torch.float16, device=dev)
        Vb = torch.empty_like(Kb)
        Kb[: HK * nl] = to_kv(k16).view(HK * nl, T, D)
        Vb[: HK * nl] = to_kv(v16).view(HK * nl, T, D)
        qv = to_q(q16)[None].contiguous()
        cur = torch.cuda.current_stream(dev)
        cs = self.cs
        parts = []
        (s0, r0), (s1, r1) = tab["steps"]
        cs.wait_stream(cur)
        with torch.cuda.stream(cs):
            self._exchange(Kb, Vb, s0, r0)  # step 0 transfers ...
        if tab["tabs"][0] is not None:
            parts.append(self._comp(qv, Kb, Vb, tab["tabs"][0]))  # ... overlap the local tiles
        cur.wait_stream(cs)
        with torch.cuda.stream(cs):
            self._exchange(Kb, Vb, s1, r1)  # step 1 forwards ...
        if tab["tabs"][1] is not None:
            parts.append(self._comp(qv, Kb, Vb, tab["tabs"][1]))  # ... overlap the step-0 tiles
        cur.wait_stream(cs)
        if tab["tabs"][2] is not None:
            parts.append(self._comp(qv, Kb, Vb, tab["tabs"][2]))
        Kb.record_stream(cs)
        Vb.record_stream(cs)
        # fp32 LSE merge of the partials at the owner
        L = torch.stack([p[1] for p in parts])  # [k, R T]
        m = L.amax(0)
        w = torch.exp(L - m)
        o = sum(w[j][:, None] * parts[j][0] for j in range(len(parts))) / w.sum(0)[:, None]
        lse = m + torch.log(w.sum(0))
        return from_q(o.to(torch.float16)[:, None], lay.L), lse

    def fwd(self, q16, k16, v16, mrows):
        return self.run(q16, k16, v16, self.tables(mrows))
