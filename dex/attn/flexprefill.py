"""Dex attention op for W6 (Llama-3.1-8B + FlexPrefill, forward only).

Per call: the mask rows all-gathered (bit-packed); the assigner per KV-head unit (8 units, heads (4, 1), W = live head-tiles
summed over the unit's 4 query heads, causal diagonal 0.5) gives the GPU of every unit block pair, applied to the unit's 4
query heads; block pairs over a 32-virtual-head space whose KV blocks live in virtual head 4 g, so each (KV head, tile) is
transferred once per receiver; schedule + tables (heads (1, 1)); one persistent forward launch.
Home gate: the assigner's plan is kept only if model(home) / model(plan) > 1 + eps, with the forward time model on the
assigner's own inputs, home = every pair on its zigzag query owner, eps = 0.040: estimates within this margin of home count
as ties and home, which moves no queries, is kept.
The gate runs on the GPU in one CUDA graph per length; rank 0's plan, gate outcome and model times go out in one broadcast.
assigner=False ("dex_no_assigner"): every pair on its zigzag query owner, no gate, no collective.
System interface of workloads.w6_flexprefill.common."""

import os
import time

import numpy as np
import torch
import torch.distributed as dist

from dex import planner as pl
from dex import scheduler as S
from dex.planner.plan import _capture
from workloads.w6_flexprefill.common import GPUS_PER_NODE, GQ, HK, HQ, SCALE, T, D, cuda_ext, from_q, mask_global, to_q

GATE_EPS = 0.040
REGION_FRACTION = 0.85  # share of the symmetric heap the region takes


def block_pairs_flex(mh, pb, per):
    """the scheduler's block-pair inputs of the 32-virtual-head problem: query head h's mask mh[h] [n, n] (0 / 1 / 2) pairs Q
    block (h, i) = h nbh + i with KV block (4 (h // 4), j) = 4 (h // 4) nbh + j; GPU of a pair = the plan of its KV-head
    unit pb[h // 4, i, j]. Same bit fields as dex.scheduler.block_pairs."""
    Hh, n, _ = mh.shape
    nbh = n // per
    assert per <= 64
    dev = mh.device
    R = mh.view(Hh, nbh, per, nbh, per)
    live = (R != 0).any(4).any(2)
    h, i, j = torch.nonzero(live, as_tuple=True)
    blk = R[h, i, :, j, :]
    lv, dg = blk != 0, blk == 2
    bits = torch.ones(per, dtype=torch.int64, device=dev) << torch.arange(per, device=dev)
    prow = (lv.any(2).to(torch.int64) * bits).sum(-1)
    pcol = (lv.any(1).to(torch.int64) * bits).sum(-1)
    nw = (per * per + 63) // 64
    w64 = torch.ones(64, dtype=torch.int64, device=dev) << torch.arange(64, device=dev)

    def pack(x):
        f = torch.nn.functional.pad(x.reshape(len(x), -1).to(torch.int64), (0, nw * 64 - per * per)).view(
            len(x), nw, 64
        )
        return (f * w64).sum(-1)

    u = h // GQ
    bq, bk = h * nbh + i, (u * GQ) * nbh + j
    pdev = pb[u, i, j]
    u64 = lambda x: np.ascontiguousarray(x.cpu().numpy().astype(np.int64)).view(np.uint64)  # noqa: E731
    return dict(
        bq=bq.cpu().numpy().astype(np.int64),
        bk=bk.cpu().numpy().astype(np.int64),
        pdev=pdev.cpu().numpy().astype(np.int64),
        nfull=(lv & ~dg).sum((1, 2)).cpu().numpy().astype(np.int64),
        ndiag=dg.sum((1, 2)).cpu().numpy().astype(np.int64),
        prow=u64(prow),
        pcol=u64(pcol),
        bits=np.ascontiguousarray(u64(pack(lv))),
        dbits=np.ascontiguousarray(u64(pack(dg))),
    )


def _fwd_region_bytes(t):
    q_slot = kv_slot = 128 * D * 2
    return (
        t["NQ"] * q_slot
        + 2 * t["NKV"] * kv_slot
        + t["NPART"] * (2 * q_slot + 128 * 4)
        + 8 * (t["NQ"] + t["NKV"] + t["NPART"])
    )


class DexFlex:
    plan_timing = False  # synchronised per-stage plan timings into self.tplan (tests only)
    _PLANNERS = {}

    def __init__(self, lay, assigner=True):
        self.lay, self.DA, self.assigner = lay, cuda_ext(), assigner
        cap = int(int(os.environ.get("NVSHMEM_SYMMETRIC_SIZE", str(4 << 30))) * REGION_FRACTION)
        self.region = self.DA.bwd_sym_alloc(cap)  # (collective) the shared region, once
        self.cap, self.used, self.tplan = cap, 0, {}
        self.planners = DexFlex._PLANNERS
        self._gate_g, self.gate_log = {}, []
        self.retarget(lay)

    def retarget(self, lay):
        self.lay = lay
        nbh = lay.nbh
        self.ob = torch.as_tensor(lay.own_b, device=lay.dev)  # owner per block of one head [nbh]
        self._ob = self.ob[None].expand(HK, nbh).contiguous()
        self._gid = torch.arange(HK * nbh, device=lay.dev).view(HK, nbh)
        self.own_v = np.tile(lay.own_b, HQ)  # owner per virtual block [32 nbh]
        self.bhead = np.repeat(np.arange(HQ) // GQ, nbh)  # send order: by KV-head unit
        self.kv32 = torch.zeros(HQ, lay.L, D, dtype=torch.float16, device=lay.dev)  # virtual K / V (heads 4 g used)
        self.vv32 = torch.zeros_like(self.kv32)

    def _t0(self):
        if DexFlex.plan_timing:
            torch.cuda.synchronize()
        return time.perf_counter()

    def _tick(self, key, t0):
        if not DexFlex.plan_timing:
            return 0.0
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        self.tplan[key] = (t1 - t0) * 1e3
        return t1

    def plan(self, mh):
        """-> [8, nbh, nbh] GPU of every unit block pair"""
        M_ = pl.models
        lay = self.lay
        nbh, per = lay.nbh, lay.per
        if not self.assigner:
            return self._ob[:, :, None].expand(HK, nbh, nbh).contiguous()
        t0 = self._t0()
        wt = torch.where(mh == 2, 0.5, (mh != 0).float())
        Wb = wt.view(HK, GQ, nbh, per, nbh, per).sum((1, 3, 5))
        B = per * T
        Qb, Ob, Kb = pl.fwd_bytes((GQ, 1), B, D)
        TT = pl.FWD_MODEL["T_TILE"] / 32  # (W counts head-tiles)
        gpn = GPUS_PER_NODE
        NN = lay.P // gpn
        if "fwd" not in self.planners:
            self.planners["fwd"] = pl.Planner(HK, nbh, gpn, NN, Qb, Ob, Kb, TT)
        Gp = self.planners["fwd"]
        Gp.retarget(Qb, Ob, Kb, TT)
        pb = Gp(Wb, self._ob, self._gid).clone()
        t0 = self._tick("planner_ms", t0)
        ob = self._ob
        homef = ob[:, :, None].expand(HK, nbh, nbh).contiguous()

        def gate(Wk, pb_):
            Wn = Wk > 0
            dp = torch.stack(
                [torch.where(Wn, homef, torch.full_like(homef, -1)), torch.where(Wn, pb_, torch.full_like(pb_, -1))]
            )
            NA = torch.where(dp >= 0, dp // gpn, dp)
            _, _, arr = M_.node_time_b(Wk, NA, ob // gpn, NN, gpn, Qb, Ob, Kb, TT)
            Fm = M_.gpu_time_b(Wk, dp, ob, gpn, NN, arr, Qb, Ob, Kb, TT)[0].amax(1)
            move = Fm[0] > Fm[1] * (1.0 + GATE_EPS)  # (0-d GPU bool: no sync)
            mv = Wn & (pb_ != homef)
            st_ = torch.stack(
                [
                    Fm[0].double() * 1e9,
                    Fm[1].double() * 1e9,
                    mv.sum().double(),
                    (Wk * mv).sum().double(),
                    (Wk * mv * ((pb_ // gpn) != (homef // gpn))).sum().double(),
                ]
            )
            return torch.where(move, pb_, homef), st_

        gk = self._gate_g.get(B)
        if gk is None:  # (the graph reads ob / homef on every replay:
            bufs = (Wb.clone(), pb.clone())  #  keep them alive with it)
            gk = self._gate_g[B] = (_capture(lambda: gate(*bufs)), bufs, (ob, homef))
        (g_, (pbf, st_)), bufs, _keep = gk
        bufs[0].copy_(Wb)
        bufs[1].copy_(pb)
        g_.replay()
        buf = torch.cat([pbf.reshape(-1).long(), st_.round().long()])
        t0 = self._tick("gate_ms", t0)
        dist.broadcast(buf, 0)  # (one broadcast: plan + gate statistics)
        self._tick("plan_bcast_ms", t0)
        self.gate_log.append(buf[-5:])  # home ns, plan ns, moved pairs, moved W, cross-node W
        self.last_W = Wb
        return buf[: HK * nbh * nbh].view(HK, nbh, nbh)

    def _shift_causal_geometry(self, ftab):
        """the tables give every item row_mask_mb = it_row + NQ - NKV with it_row = h n + t (query head h, global tile t)
        while its diagonal KV tile sits at (4 (h // 4)) n + t; the causal mask reads only (KV geometry - row_mask_mb), so
        row_mask_mb -= (h % 4) n makes the diagonal causal"""
        lay = self.lay
        t = ftab["row_mask_mb"]
        if t.numel() and ftab["n_compute"] > 0:
            it_row = t[: ftab["n_compute"]].long() - (ftab["NQ"] - ftab["NKV"])
            t[: ftab["n_compute"]] -= (((it_row // lay.n) % GQ) * lay.n).to(t.dtype)

    def tables(self, mrows):
        lay, dev = self.lay, self.lay.dev
        t0 = self._t0()
        mh = mask_global(mrows, lay)
        self._tick("mask_ms", t0)
        pb = self.plan(mh)
        t0 = self._t0()
        B = block_pairs_flex(mh, pb, lay.per)
        pw = (B["nfull"] + 0.5 * B["ndiag"]) * S.LINK["T_TILE"] / 32
        S.schedule_blocks(
            B["bq"],
            B["bk"],
            B["pdev"],
            pw,
            B["prow"],
            B["pcol"],
            self.own_v,
            lay.P,
            lay.per,
            block_head=self.bhead,
            heads=(1, 1),
            d=D,
        )
        ftab, tm = S.tables_blocks(lay.rank, B, self.own_v, lay.P, lay.per, dev)
        self._shift_causal_geometry(ftab)
        self._tick("pairs_sched_tables_ms", t0)
        assert len(tm["owned"]) == HQ * lay.nl, (len(tm["owned"]), HQ * lay.nl)
        self.used = max(self.used, int(_fwd_region_bytes(ftab)))
        self.last_mh, self.last_pb, self.last_B = mh, pb, B
        return ftab

    def run(self, q16, k16, v16, ftab):
        lay = self.lay
        self.kv32[0::GQ] = k16.transpose(0, 1)
        self.vv32[0::GQ] = v16.transpose(0, 1)
        out = torch.zeros(1, HQ * lay.L, 1, D, dtype=torch.float16, device=lay.dev)
        lse = torch.full((1, 1, HQ * lay.L), -float("inf"), dtype=torch.float32, device=lay.dev)
        S.call(
            self.DA, to_q(q16)[None], self.kv32.view(1, -1, 1, D), self.vv32.view(1, -1, 1, D), out, lse, ftab, SCALE
        )
        return from_q(out[0], lay.L), lse[0, 0]

    def fwd(self, q16, k16, v16, mrows):
        return self.run(q16, k16, v16, self.tables(mrows))


def dex_bytes(ftab, lay):
    """bytes this rank sends per destination from the actual message tables (inter-node engine messages + intra-node
    copy-engine runs incl. relay forwards and partial O), by kind -> [3, P] float64 (0 = Q / KV pushes + relay forwards,
    1 = partial O, 2 = all)"""
    eng = ftab["_eng_dev"]
    P = lay.P
    out = np.zeros((3, P))
    msg = (
        eng["msg"].view(torch.int64).view(-1, 4).cpu().numpy()
        if eng["msg"].numel() >= 8
        else np.zeros((0, 4), np.int64)
    )
    n_start = eng["n_start"]
    nm = len(msg) if eng["msg"].numel() >= 8 and (n_start + eng["n_pq"]) > 0 else 0
    for k_ in range(min(nm, n_start + eng["n_pq"])):
        b, pe = int(msg[k_, 2]), int(msg[k_, 3]) & 0xFFFFFFFF
        out[0 if k_ < n_start else 1, pe] += b
    for row in eng["ce"].numpy():
        b, pe, kind = int(row[2]), int(row[3]) & 0xFFFFFFFF, int(row[4]) & 0xFFFFFFFF
        out[1 if kind == 2 else 0, pe] += b
    out[2] = out[0] + out[1]
    return out
