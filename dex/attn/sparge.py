"""Dex attention op for W5 (Wan2.1 + SpargeAttention2, 12 heads, per-head 128 x 64 masks), forward and backward.

Per call: the selection split by owner (one all-reduce of the pooled keys); the layer's per-head mask rows all-gathered
into [H, n_pad, 2 n_pad] (128-row tiles x 64-key halves); the assigner (dex.planner.Planner) plans every
(head, zigzag chunk) block pair on virtual head blocks (work = live halves / 2, owners = the zigzag owners, heads (1, 1));
rank 0's plan is broadcast (the planner's GPU sums use atomics, so one broadcast keeps every rank on one plan); block
pairs, the forward schedule and this rank's tables; one persistent forward launch. Backward: the persistent backward
(dist_bwd_run) on a plan built from the same mask. assigner=False ("dex_no_assigner") puts every block pair on its
query owner and keeps the rest of the pipeline.
In the training step the recomputation reuses the forward tables, and the backward plan is searched ahead on a worker
thread right after the forward (its GPU search stays on the calling thread). Forward and backward share one symmetric
region, allocated once at 85% of the symmetric heap.
System interface of workloads.w5_sparge.common."""

import os
import time

import numpy as np
import torch
import torch.distributed as dist

from dex import planner as pl
from dex import scheduler as S
from workloads.w5_sparge import selection as SA
from workloads.w5_sparge.common import LSE_EMPTY, SCALE, T, D, Reuse, cuda_ext, view_tiles

GPUS_PER_NODE = int(os.environ.get("DEX_GPUS_PER_NODE", "4"))
REGION_FRACTION = 0.85  # share of the symmetric heap the shared fwd / bwd region takes
_POOL = []


def _pool():
    if not _POOL:
        from concurrent.futures import ThreadPoolExecutor

        dev_ = torch.cuda.current_device()
        _POOL.append(ThreadPoolExecutor(max_workers=1, initializer=lambda: torch.cuda.set_device(dev_)))
    return _POOL[0]


def tables_to_dev(tabs):
    def t32(x):
        x = np.ascontiguousarray(x)
        x = x.view(np.int32).reshape(-1) if x.dtype == np.int64 else np.ascontiguousarray(x, np.int32).reshape(-1)
        return torch.from_numpy(x if len(x) else np.zeros(2, np.int32)).cuda()

    return [t32(tabs[k]) for k in S.TABLE_ORDER + S.RECT_ORDER]


class DexSparge:
    plan_timing = False  # synchronised per-stage plan timings into self.tplan (tests only)

    def __init__(self, lay, H, assigner=True):
        DA = cuda_ext()
        self.DA, self.lay, self.H, self.assigner = DA, lay, H, assigner
        P = lay.P
        self.own_v = np.tile(lay.block_owners(), H)  # virtual blocks h * 2P + c
        self.n_sm = torch.cuda.get_device_properties(lay.tok.device).multi_processor_count
        cap = int(int(os.environ.get("NVSHMEM_SYMMETRIC_SIZE", str(4 << 30))) * REGION_FRACTION)
        self.region = DA.bwd_sym_alloc(cap)  # (collective) the shared region, never regrown
        self.cap = cap
        self.pe_delta, self.pe_inter = DA.bwd_pe_info()
        self.last = None
        self._s1 = torch.zeros(1, device=lay.tok.device)
        self.used = 0  # bytes of the shared region actually used
        self.planners, self.tplan = {}, {}
        assert P % GPUS_PER_NODE == 0
        S.local._group()

    def _t0(self):
        if self.plan_timing:
            torch.cuda.synchronize()
        return time.perf_counter()

    def _tick(self, key, t0):
        if not self.plan_timing:
            return 0.0
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        self.tplan[key] = (t1 - t0) * 1e3
        return t1

    def _plan(self, mh, kinds):
        """the assigner on the layer's per-head mask -> {kind: per-head plan [H, nbh, nbh] (GPU of every block pair)}.
        Virtual head blocks (head h, zigzag chunk c): work W = live 64-key halves / 2, owners = the zigzag owners,
        heads (1, 1), block = per tiles; the backward models take the live key tiles per block column (kcols)."""
        lay, H = self.lay, self.H
        per = lay.per
        nbh = lay.n_pad // per
        dev = mh.device
        if not self.assigner:
            q = torch.as_tensor(self.own_v, device=dev).view(H, nbh)
            return {k_: q[:, :, None].expand(H, nbh, nbh).contiguous() for k_ in kinds}
        t0 = self._t0()
        m_ = mh != 0
        Wb = m_.view(H, nbh, per, nbh, 2 * per).sum((2, 4)).float() * 0.5
        m_ = m_.view(H, lay.n_pad, lay.n_pad, 2).any(-1)
        kb = m_.any(1).view(H, nbh, per).sum(2).float()
        if not hasattr(self, "_ob"):
            self._ob = torch.as_tensor(self.own_v, device=dev).view(H, nbh)
            self._gid = torch.arange(H * nbh, device=dev).view(H, nbh)
        B = per * T
        Qb, Ob, Kb = pl.fwd_bytes((1, 1), B, D)
        TT = pl.FWD_MODEL["T_TILE"] / 32
        gpn = GPUS_PER_NODE
        NN = lay.P // gpn
        out = []
        for k_ in kinds:
            bwd = k_ == "bwd"
            if k_ not in self.planners:
                self.planners[k_] = pl.Planner(H, nbh, gpn, NN, Qb, Ob, Kb, TT, bwd=((1, 1), B, kb) if bwd else None)
            G = self.planners[k_]
            if bwd:
                G.retarget(Qb, Ob, Kb, TT, B=B, kcols=kb)
            info = {}
            out.append(G(Wb, self._ob, self._gid, info=info if self.plan_timing else None).clone())
            t0 = self._tick(f"planner_{k_}_ms", t0)
            if self.plan_timing:
                self.tplan[f"rounds_{k_}"] = info.get("rounds")
        pb = torch.stack(out)
        dist.broadcast(pb, 0)
        self._tick("plan_bcast_ms", t0)
        return {k_: pb[i] for i, k_ in enumerate(kinds)}

    def _devb(self, pb):
        """per-head plan [H, nbh, nbh] -> the virtual [H nbh, H nbh] GPU-of-pair matrix (block diagonal)"""
        H = self.H
        nbh = pb.shape[1]
        full = torch.full((H * nbh, H * nbh), -1, dtype=torch.long, device=pb.device)
        ar = torch.arange(H, device=pb.device)
        full.view(H, nbh, H, nbh)[ar, :, ar, :] = pb
        return full

    def _switch(self, kind):
        """forward and backward stage into the shared region before their entry barriers: at a switch every rank must
        have finished the other kind's call and the host must not run ahead"""
        if self.last is not None and self.last != kind:
            dist.all_reduce(self._s1)
            torch.cuda.current_stream().synchronize()
        self.last = kind

    def _bplan(self, mh, devb):
        lay = self.lay
        B = S.block_pairs_heads_torch(mh, devb, lay.per, owner=self.own_v)
        return S.build_fast(B, self.own_v, lay.P, lay.per, lay.rank, 1, 1)

    def _mask_global(self, mrows):
        """every rank's tile-mask rows (all-gather) assembled on the GPU -> [H, n_pad, 2 n_pad] uint8"""
        lay = self.lay
        mr = mrows.to(torch.uint8)
        buf = torch.empty((lay.P,) + tuple(mr.shape), dtype=torch.uint8, device=mr.device)
        dist.all_gather_into_tensor(buf, mr.contiguous())
        if not hasattr(self, "_rows_of"):
            self._rows_of = torch.from_numpy(np.concatenate([lay.tiles_of(r) for r in range(lay.P)])).to(mr.device)
        w = mr.shape[2]
        mh = torch.empty((self.H, lay.n_pad, w), dtype=torch.uint8, device=mr.device)
        mh[:, self._rows_of] = buf.permute(1, 0, 2, 3).reshape(self.H, -1, w)
        return mh

    def fwd(self, q, k, v):
        lay, H, dev = self.lay, self.H, q.device
        c = Reuse.cache()
        if c is not None and "ftab" in c:
            fm, ftab = c["fm"], c["ftab"]
        else:
            fm = c["fm"] if c is not None and "fm" in c else SA.select_split(q, k, lay)
            t0 = self._t0()
            mh = self._mask_global(lay.tile_rows(fm))
            self._tick("mask_ms", t0)
            plans = self._plan(mh, ("fwd", "bwd") if (c is not None and Reuse.ahead) else ("fwd",))
            t0 = self._t0()
            # block pairs on the GPU; the [H, n_pad, 2 n_pad] mask also yields the half bitmaps (kBlockN 64 lists)
            B = S.block_pairs_heads_torch(mh, self._devb(plans["fwd"]), lay.per, owner=self.own_v)
            work = B["nhalf"] * 0.5 if "nhalf" in B else B["nfull"] + 0.5 * B["ndiag"]
            pw = work * S.LINK["T_TILE"] / 32
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
                block_head=B["block_head"],
                heads=(1, 1),
                d=D,
            )
            ftab, tm = S.tables_blocks(lay.rank, B, self.own_v, lay.P, lay.per, dev)
            self._tick("pairs_sched_tables_ms", t0)
            assert len(tm["owned"]) == H * lay.nl, (len(tm["owned"]), H * lay.nl)
            t_ = ftab
            fb = (
                t_["NQ"] * 128 * D * 2
                + 2 * t_["NKV"] * 128 * D * 2
                + t_["NPART"] * (2 * 128 * D * 2 + 128 * 4)
                + 8 * (t_["NQ"] + t_["NKV"] + t_["NPART"])
            )
            self.used = max(self.used, int(fb))
            if c is not None:
                c.update(fm=fm, ftab=ftab, mh=mh)
                if Reuse.ahead:  # mask + backward plan to pinned host memory
                    hb = torch.empty(mh.shape, dtype=torch.uint8, pin_memory=True)
                    hb.copy_(mh, non_blocking=True)
                    db = self._devb(plans["bwd"])
                    hd = torch.empty(db.shape, dtype=torch.long, pin_memory=True)
                    hd.copy_(db, non_blocking=True)
                    ev = torch.cuda.Event()
                    ev.record()

                    def job(hb=hb, hd=hd, ev=ev):
                        ev.synchronize()
                        return self._bplan(torch.from_numpy(hb.numpy()), hd)

                    c["bplan"] = _pool().submit(job)
        self._switch("fwd")
        out = torch.zeros(1, H * lay.L_pad, 1, D, dtype=torch.float16, device=dev)
        lse = torch.full((1, 1, H * lay.L_pad), LSE_EMPTY, dtype=torch.float32, device=dev)
        S.call(self.DA, view_tiles(lay.pad(q)), view_tiles(lay.pad(k)), view_tiles(lay.pad(v)), out, lse, ftab, SCALE)
        lse = torch.where(torch.isfinite(lse), lse, torch.full_like(lse, LSE_EMPTY))
        return lay.unpad(out.view(H, lay.L_pad, D)), (fm, lay.unpad(lse.view(H, lay.L_pad)))

    def bwd(self, q, k, v, o, do, fm, lse, cache=None):
        DA, lay, H, dev = self.DA, self.lay, self.H, q.device
        c = cache
        if c is not None and "bplan" in c:
            glob, my_ = c["bplan"].result()
        else:
            t0 = self._t0()
            mh = c["mh"] if c is not None and "mh" in c else self._mask_global(lay.tile_rows(fm))
            db = self._devb(self._plan(mh, ("bwd",))["bwd"])
            t0 = self._t0()
            glob, my_ = self._bplan(mh, db)
            self._tick("bwd_pairs_sched_tables_ms", t0)
        self._switch("bwd")
        q, k, v, o, do = (lay.pad(x) for x in (q, k, v, o, do))
        lse = lay.pad(lse)
        L = lay.L_pad
        st = my_["stats"]
        offs, sizes, total = S.region_layout(glob, 1, 1)
        assert total <= self.cap, (total, self.cap)
        self.used = max(self.used, int(total))
        tabs = tables_to_dev(my_["tabs"])
        half = "rect_hbits" in my_["tabs"]  # 128 x 64: the half bitmaps after the rect tables
        if half:
            tabs.append(torch.from_numpy(np.ascontiguousarray(my_["tabs"]["rect_hbits"])).cuda())
        ce_tab = torch.from_numpy(np.ascontiguousarray(my_["tabs"]["ce"], np.int64))
        soffs, ssize = S.scratch_layout(st, glob, 1, 1, self.n_sm - 1)
        scratch = torch.zeros(ssize, dtype=torch.int32, device=dev)
        no = H * lay.nl
        NQS, NKS = glob["NQS"], glob["NKS"]
        region = self.region
        qb = region[offs[1] : offs[1] + NQS * sizes[0]].view(NQS, sizes[0])
        kvr = region[offs[2] : offs[2] + NKS * sizes[1]].view(NKS, sizes[1])
        qbytes = kbytes = T * D * 2
        vq = q.reshape(H * L, 1, D)
        vdo = do.reshape(H * L, 1, D)
        vk = k.reshape(H * L, 1, D)
        vv = v.reshape(H * L, 1, D)
        vo = o.reshape(H * L, 1, D)
        lse_v = lse.reshape(1, H * L).contiguous()
        dps = torch.empty(1, H * L, device=dev)
        DA.dot_do_o(vdo, vo, dps, None)
        qb[:no, 0:qbytes].view(torch.float16).copy_(vq.view(no, T * D))
        qb[:no, qbytes : 2 * qbytes].view(torch.float16).copy_(vdo.view(no, T * D))
        qb[:no, 2 * qbytes : 2 * qbytes + T * 4].view(torch.float32).copy_(lse_v.view(no, T))
        qb[:no, 2 * qbytes + T * 4 : 2 * qbytes + 2 * T * 4].view(torch.float32).copy_(dps.view(no, T))
        kvr[:no, 0:kbytes].view(torch.float16).copy_(vk.view(no, T * D))
        kvr[:no, kbytes : 2 * kbytes].view(torch.float16).copy_(vv.view(no, T * D))
        dq = torch.zeros(H * L, 1, D, device=dev, dtype=torch.float16)
        dk = torch.zeros_like(dq)
        dv = torch.zeros_like(dq)
        dims = [
            NQS,
            NKS,
            glob["NDQL"],
            glob["NDKL"],
            1,
            1,
            D,
            st["tasks"],
            st["n_init"],
            st["n_qm"],
            st["n_km"],
            st["n_remote"],
            st["n_start"],
            0,
            st["n_qagg"] + st["n_kagg"],
            st["n_pq"],
        ]
        dims += [1, st["n_rect"], st["G1"], st["per"], st["nw"], st["arena"], st["n_groups"]]
        if half:
            dims += [1, st["nw2"]]
        DA.dist_bwd_run(
            offs, sizes, dims, tabs, self.pe_delta, self.pe_inter, scratch, soffs, dq, dk, dv, SCALE, self.n_sm, ce_tab
        )
        return lay.unpad(dq.view(H, L, D)), lay.unpad(dk.view(H, L, D)), lay.unpad(dv.view(H, L, D))


def make(lay, H, assigner=True):
    return DexSparge(lay, H, assigner)
