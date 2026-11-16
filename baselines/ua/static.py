"""UA on W1-W3: UltraAttn's plans on the decomposed executor, one attention-layer test per input and pass.

Plans: $DEX_DATA_ROOT/plans/ua/manifest.json {"<input>|fwd" | "<input>|bwd": {"plan": file, "mask": file of the plan's
block layout} or {"na": reason}} (baselines/ua/planner.py). Forward: Q, K, V ~ N(0, 0.5^2) fp16 (seed 1234), pad K / V zero;
pad correction after the call (timed), one first-call accumulator state per call. Backward: Q, K, V, dO, O random, LSE = 6,
D = rowsum(dO * O), pad K / V / dO zero. Each rank loads its owned blocks before timing; received blocks are part of the
timed call. Launch 1 adds the FP64 sampled-row check on real tokens.
`UAStatic` is the op with both plans on one deployment (the forward's), used in training: fwd / bwd / tokens, with pools
held only between begin() and end()."""
import json
import math
import os

import numpy as np
import torch
import torch.distributed as dist

from baselines.ua.plan_adapter import ua_plan_to_fusion_schedule
from baselines.ua.pool_exec import EMPTY_LSE, PoolTiles, fusion_forward_pooled
from baselines.ua.pool_exec_bwd import PoolTilesBwd, fusion_backward_pooled
from dex.common import fp64_check as F64
from dex.common.env import cfg
from dex.common.pad_correct import correct as pad_correct

D = 128


def connect_all(rank, world, dev):
    """exchange one small message with every peer at once: NCCL connects a pair on first use and a first send blocks until
    the peer is inside NCCL, so the plans' first steps must not be the ones that open the connections"""
    tx = torch.zeros(world, 16, device=dev)
    rx = torch.zeros(world, 16, device=dev)
    ops = [dist.P2POp(dist.isend, tx[p], p) for p in range(world) if p != rank] + \
          [dist.P2POp(dist.irecv, rx[p], p) for p in range(world) if p != rank]
    for r in dist.batch_isend_irecv(ops):
        r.wait()
    torch.cuda.synchronize()
    dist.barrier()


def manifest():
    return json.load(open(os.path.join(cfg("DEX_DATA_ROOT"), "plans", "ua", "manifest.json")))


def run_fwd(case, ent, protocol, rank, dev):
    z = np.load(ent["mask"])
    m = z["mask"].astype(np.int8)
    T = int(z["tokens"])
    HQ, HK = (int(x) for x in z["heads"])
    scale = 1.0 / D ** 0.5
    plan = json.load(open(ent["plan"]))
    B = plan["block_tokens"]
    g = torch.Generator(device=dev).manual_seed(1234)
    Qg = torch.randn(1, T, HQ, D, generator=g, device=dev, dtype=torch.float16) * 0.5
    Kg = torch.randn(1, T, HK, D, generator=g, device=dev, dtype=torch.float16) * 0.5
    Vg = torch.randn(1, T, HK, D, generator=g, device=dev, dtype=torch.float16) * 0.5
    PAD = "tok_valid" in z.files
    if PAD:
        valid = torch.as_tensor(z["tok_valid"].astype(bool), device=dev)
        npad = torch.as_tensor(z["npad"].astype(np.int32), device=dev)
        Kg[:, ~valid] = 0
        Vg[:, ~valid] = 0
    sched = ua_plan_to_fusion_schedule(plan, rank)
    blk = lambda X, b: X[:, b * B:(b + 1) * B].contiguous()  # noqa: E731
    pt = PoolTiles(m, sched, HQ, HK, D, torch.float16, dev)
    pt.load({b: blk(Qg, b) for b in sched.owned_q}, {b: (blk(Kg, b), blk(Vg, b)) for b in sched.owned_q})

    def run():
        pt.acc.zero_()
        pt.lse.fill_(EMPTY_LSE)
        o = fusion_forward_pooled(sched, pt, scale, partial_dtype=torch.float32)
        if PAD:
            for b, ob in o.items():
                s, e = pt.qrows(b)
                pad_correct(ob[0], pt.lse[:, s:e].clone(), npad[b * B:(b + 1) * B])
        return o

    ms, ms_all = protocol["latency"](run)
    e64 = None
    if protocol["fp64"]:
        out = run()
        torch.cuda.synchronize()
        rid = torch.as_tensor(F64.real_ids(z, T), device=dev)
        rv = rid >= 0
        bl = sorted(out)
        pos = torch.cat([torch.arange(b * B, (b + 1) * B, device=dev) for b in bl]) if bl else torch.zeros(0, dtype=torch.long, device=dev)
        O_ = torch.cat([out[b][0] for b in bl]) if bl else Qg.new_zeros((0, HQ, D))
        rl = rid[pos]
        ok = rl >= 0
        e64 = F64.check(Qg[0][rv], Kg[0][rv], Vg[0][rv], None, np.load(case["ranges"]), scale, dict(o=O_[ok]),
                        q_rows=rl[ok].cpu().numpy(), group=dist.group.WORLD)
    return protocol["record"](case, "ua", "fwd", protocol["launch"], ms=ms, ms_all=ms_all, e64=e64, block_tokens=B,
                              steps=len(sched.steps))


def run_bwd(case, ent, protocol, rank, dev):
    z = np.load(ent["mask"])
    m = z["mask"].astype(np.int8)
    S = int(z["tokens"])
    h, hk = (int(x) for x in z["heads"])
    scale = D ** -0.5
    g = torch.Generator(device=dev).manual_seed(1234)
    rnd = lambda *s: (torch.randn(*s, generator=g, device=dev, dtype=torch.float16) * 0.5)  # noqa: E731
    Q, K, V, dO, O = rnd(S, h, D), rnd(S, hk, D), rnd(S, hk, D), rnd(S, h, D), rnd(S, h, D)
    LSE = torch.full((h, S), 6.0, device=dev)
    valid = torch.as_tensor(z["tok_valid"].astype(bool), device=dev) if "tok_valid" in z.files else torch.ones(S, dtype=torch.bool, device=dev)
    K[~valid] = 0
    V[~valid] = 0
    dO[~valid] = 0
    Dl = (dO.float() * O.float()).sum(-1).t().contiguous()
    plan = json.load(open(ent["plan"]))
    B = plan["block_tokens"]
    assert plan["cp"] == dist.get_world_size() and plan.get("fob") == 1, ent["plan"]
    sched = ua_plan_to_fusion_schedule(plan, rank)
    pt = PoolTilesBwd(m, sched, h, hk, D, torch.float16, dev)
    own = list(sched.owned_q)
    blk = lambda X, b: X[b * B:(b + 1) * B]  # noqa: E731
    pt.load({b: (blk(Q, b), blk(dO, b), Dl[:, b * B:(b + 1) * B], LSE[:, b * B:(b + 1) * B]) for b in own},
            {b: (blk(K, b), blk(V, b)) for b in own})
    for b in own:
        pt.make_out(b)
    run = lambda: fusion_backward_pooled(sched, pt, scale, partial_dtype=torch.float32)  # noqa: E731
    e64 = None
    if protocol["fp64"]:
        dq, dkv = run()
        torch.cuda.synchronize()
        rid = torch.as_tensor(F64.real_ids(z, S), device=dev)
        rv = rid >= 0
        pos = torch.cat([torch.arange(b * B, (b + 1) * B, device=dev) for b in own]) if own else torch.zeros(0, dtype=torch.long, device=dev)
        rl = rid[pos]
        ok = rl >= 0
        DQ = torch.cat([dq[b] for b in own]) if own else torch.zeros(0, h, D, device=dev)
        DK = torch.cat([dkv[b][0] for b in own]) if own else torch.zeros(0, hk, D, device=dev)
        DV = torch.cat([dkv[b][1] for b in own]) if own else torch.zeros(0, hk, D, device=dev)
        e64 = F64.check(Q[rv], K[rv], V[rv], dO[rv], np.load(case["ranges"]), scale, dict(dq=DQ[ok], dk=DK[ok], dv=DV[ok]),
                        q_rows=rl[ok].cpu().numpy(), kv_rows=rl[ok].cpu().numpy(), group=dist.group.WORLD, lse_in=LSE[:, rv].t(), o_in=O[rv])
    ms, ms_all = protocol["latency"](run)
    return protocol["record"](case, "ua", "bwd", protocol["launch"], ms=ms, ms_all=ms_all, e64=e64, block_tokens=B,
                              steps=len(sched.steps))


def run(cases, pass_, protocol):
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", torch.cuda.current_device())
    connect_all(rank, world, dev)
    man = manifest()
    recs = []
    for c in cases:
        ent = man.get(f"{c['id']}|{pass_}", dict(na="no plan"))
        if "na" in ent:
            recs.append(protocol["record"](c, "ua", pass_, protocol["launch"], status="N/A", reason=ent["na"]))
            continue
        try:
            recs.append((run_fwd if pass_ == "fwd" else run_bwd)(c, ent, protocol, rank, dev))
        except Exception as ex:  # noqa: BLE001  one failing input does not lose the launch
            recs.append(protocol["record"](c, "ua", pass_, protocol["launch"], status="error", reason=f"{type(ex).__name__}: {str(ex)[:300]}"))
        torch.cuda.empty_cache()
        dist.barrier()
    return recs


class UAStatic:
    """training op on one token layout: UA's forward plan and its backward plan built on the forward's deployment (same
    owner and block size). q [Tl, Hq, D], k / v [Tl, Hk, D] fp16 in UA's local order (owned blocks ascending)."""

    def __init__(self, mask, fwd_plan, bwd_plan, heads, pad=None, device=None):
        self.rank, self.P = dist.get_rank(), dist.get_world_size()
        self.dev = device or torch.device("cuda")
        self.hq, self.hk = heads
        self.m = np.asarray(mask, np.int8)
        pf, pb = json.load(open(fwd_plan)), json.load(open(bwd_plan))
        assert pf["owner"] == pb["owner"] and pf["block_tokens"] == pb["block_tokens"], "UA fwd / bwd deployments differ"
        assert pb.get("fob") == 1, bwd_plan
        self.B = B = pf["block_tokens"]
        self.sf = ua_plan_to_fusion_schedule(pf, self.rank)
        self.sb = ua_plan_to_fusion_schedule(pb, self.rank)
        self.own = [b for b in range(pf["n_blocks"]) if pf["owner"][b] == self.rank]
        self.ptf = PoolTiles(self.m, self.sf, self.hq, self.hk, D, torch.float16, self.dev)
        self.ptb = PoolTilesBwd(self.m, self.sb, self.hq, self.hk, D, torch.float16, self.dev)
        self.tokens = torch.as_tensor((np.asarray(self.own)[:, None] * B + np.arange(B)).reshape(-1), device=self.dev)
        self.pad = None
        if pad is not None:
            valid, npad = pad
            self.pad = (torch.as_tensor(np.asarray(valid, bool), device=self.dev)[self.tokens],
                        torch.as_tensor(np.asarray(npad, np.int32), device=self.dev)[self.tokens])
        self.scale = 1.0 / math.sqrt(D)
        # one op per input: the pool tensors exist only between begin() and end(); the plan caches stay
        self._spec = {}
        for nm, pool in (("f", self.ptf), ("b", self.ptb)):
            self._spec[nm] = {a: (v.shape, v.dtype) for a, v in vars(pool).items() if isinstance(v, torch.Tensor) and v.numel() > 1 << 16}
        self.end()

    def begin(self):
        for nm, pool in (("f", self.ptf), ("b", self.ptb)):
            for a, (sh, dt) in self._spec[nm].items():
                setattr(pool, a, torch.empty(sh, dtype=dt, device=self.dev))

    def end(self):
        for nm, pool in (("f", self.ptf), ("b", self.ptb)):
            for a in self._spec[nm]:
                setattr(pool, a, None)

    def _blocks(self, x):
        return {b: x[i * self.B:(i + 1) * self.B] for i, b in enumerate(self.own)}

    def fwd(self, q, k, v):
        if self.pad is not None:
            k = k.masked_fill(~self.pad[0][:, None, None], 0)
            v = v.masked_fill(~self.pad[0][:, None, None], 0)
        qb, kb, vb = self._blocks(q), self._blocks(k), self._blocks(v)
        # every call starts from the pool's initial accumulator: the first-writer flags replay the first call's order
        self.ptf.acc.zero_()
        self.ptf.lse.fill_(EMPTY_LSE)
        self.ptf.load({b: qb[b][None] for b in self.own}, {b: (kb[b][None], vb[b][None]) for b in self.own})
        o_ = fusion_forward_pooled(self.sf, self.ptf, self.scale, partial_dtype=torch.float32)
        Tl = q.shape[0]
        out = torch.empty(Tl, self.hq, D, device=self.dev, dtype=torch.float16)
        lse = torch.empty(self.hq, Tl, device=self.dev, dtype=torch.float32)
        for i, b in enumerate(self.own):
            s, e = self.ptf.qrows(b)
            out[i * self.B:(i + 1) * self.B].copy_(o_[b][0])
            lse[:, i * self.B:(i + 1) * self.B].copy_(self.ptf.lse[:, s:e])
        if self.pad is not None:
            pad_correct(out, lse, self.pad[1])
            out.masked_fill_(~self.pad[0][:, None, None], 0)
        return out, lse

    def bwd(self, q, k, v, o, lse, do):
        if self.pad is not None:
            z = ~self.pad[0][:, None, None]
            k, v, do = k.masked_fill(z, 0), v.masked_fill(z, 0), do.masked_fill(z, 0)
        Dl = (do.float() * o.float()).sum(-1).t().contiguous()
        B = self.B
        rows = {b: (q[i * B:(i + 1) * B], do[i * B:(i + 1) * B], Dl[:, i * B:(i + 1) * B], lse[:, i * B:(i + 1) * B])
                for i, b in enumerate(self.own)}
        cols = {b: (k[i * B:(i + 1) * B], v[i * B:(i + 1) * B]) for i, b in enumerate(self.own)}
        self.ptb.load(rows, cols)
        for b in self.own:
            self.ptb.make_out(b)
        dq_, dkv_ = fusion_backward_pooled(self.sb, self.ptb, self.scale, partial_dtype=torch.float32)
        dq = torch.cat([dq_[b] for b in self.own]).to(torch.float16)
        dk = torch.cat([dkv_[b][0] for b in self.own]).to(torch.float16)
        dv = torch.cat([dkv_[b][1] for b in self.own]).to(torch.float16)
        if self.pad is not None:
            dk.masked_fill_(z, 0)
            dv.masked_fill_(z, 0)
        return dq, dk, dv
