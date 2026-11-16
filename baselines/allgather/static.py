"""All-gather on W1-W3: K / V all-gathered (rank-major, each rank's tiles padded to the largest rank), one sparse_fwd per
rank over its query rows, zigzag ownership of 128-token tiles. Backward: all-gather K / V, sparse_bwd, reduce-scatter of the
fp32 dK / dV to the owners. Also the kernel table builders (csr_rows, items_for) shared by the tile-kernel baselines.
Layout: q [Tl, Hq, D], k / v [Tl, Hk, D] fp16, o fp16, lse [Hq, Tl] fp32, in ascending owned-tile order.
`run` is the attention-layer test (experiments/attention_layer/static_op.py); `AllGatherStatic` the op used by it and by
training (cp_attention interface: fwd, bwd, tokens)."""
import math

import numpy as np
import torch
import torch.distributed as dist

from dex.common import fp64_check as F64
from dex.common.pad_correct import correct as pad_correct

from dex import dex_cuda as C

TILE, D = 128, 128


def csr_rows(m, rows, slot_of_tile, nheads, dev, shift=0):
    """sparse_fwd tables for local Q rows `rows` (global tile ids) against KV stored at slot_of_tile[global tile]. The
    kernel's causal geometry offsets the diagonal by seqlen_k - seqlen_q: shift = that / 128 is added to the columns' mask
    coordinate so a diagonal tile (row == col) is masked exactly causally."""
    indptr, idx, nb, nmask = [0], [], [], []
    for a in rows:
        cols = np.flatnonzero(m[a] > 0)
        kinds = m[a][cols]
        o = np.lexsort((cols, (kinds == 2).astype(np.int8)))
        cols = cols[o]
        idx += [int(slot_of_tile[c]) for c in cols]
        nb += (cols + shift).tolist()
        nmask.append(int((kinds == 2).sum()))
        indptr.append(len(idx))
    nnz = np.diff(indptr)
    order = np.argsort(-nnz, kind="stable")
    order = order[nnz[order] > 0]
    work = [h * len(rows) + int(r) for h in range(nheads) for r in order]
    t = lambda x: torch.tensor(x if len(x) else [0], dtype=torch.int32, device=dev)  # noqa: E731
    return t(indptr), t(idx), t(nb), t(nmask), t([int(a) for a in rows]), t(work)


def items_for(m, q_tiles, q_pos, kv_tiles, kv_pos, h, hk):
    """sparse_bwd tasks over KV tiles kv_tiles (stored at kv_pos) x Q tiles q_tiles (stored at q_pos), KV-head-major"""
    r_grp = h // hk
    tk, tp, it = [], [0], []
    for g in range(hk):
        for j, js in zip(kv_tiles, kv_pos):
            rows = [(i, s) for i, s in zip(q_tiles, q_pos) if m[i, j]]
            if not rows:
                continue
            for hn in range(2):
                n0 = len(it)
                for i, s in rows[::-1]:
                    for mb in (1, 0):
                        if m[i, j] == 2:
                            if mb < hn:
                                continue
                            causal = mb == hn
                        else:
                            causal = False
                        it += [(int(s) << 9) | (mb << 8) | (hq << 1) | int(causal) for hq in range(g * r_grp, (g + 1) * r_grp)]
                if len(it) > n0:
                    tk.append((int(js) << 9) | (hn << 8) | g)
                    tp.append(len(it))
    i32 = lambda x: torch.tensor(x if x else [0], dtype=torch.int32, device="cuda")  # noqa: E731
    return (i32(tk), i32(tp), i32(it)), len(tk)


def zigzag_owner(n, P):
    """[n] rank of every tile: chunks r and 2P-1-r of 2P equal chunks"""
    c = n // (2 * P)
    assert c * 2 * P == n, (n, P)
    own = np.empty(n, np.int64)
    for r in range(P):
        own[r * c:(r + 1) * c] = r
        own[(2 * P - 1 - r) * c:(2 * P - r) * c] = r
    return own


class AllGatherStatic:
    def __init__(self, mask, heads, owner=None, pad=None, device=None):
        """mask [n, n] tile mask; owner [n] rank per tile (default zigzag); pad = (tok_valid [T], npad [T]) or None"""
        self.rank, self.P = dist.get_rank(), dist.get_world_size()
        self.dev = device or torch.device("cuda")
        self.m = np.asarray(mask, np.int8)
        self.hq, self.hk = heads
        n = self.m.shape[0]
        self.own = zigzag_owner(n, self.P) if owner is None else np.asarray(owner, np.int64)
        self.per = [np.flatnonzero(self.own == r) for r in range(self.P)]
        self.owned = self.per[self.rank]
        self.no = len(self.owned)
        self.cap = max(map(len, self.per))
        self.tokens = torch.as_tensor((self.owned[:, None] * TILE + np.arange(TILE)).reshape(-1), device=self.dev)
        self.scale = 1.0 / math.sqrt(D)
        self.pad = None
        if pad is not None:
            valid, npad = pad
            self.pad = (torch.as_tensor(np.asarray(valid, bool), device=self.dev)[self.tokens],
                        torch.as_tensor(np.asarray(npad, np.int32), device=self.dev)[self.tokens])
        slot = np.zeros(n, np.int64)
        for r in range(self.P):
            slot[self.per[r]] = r * self.cap + np.arange(len(self.per[r]))
        self.ftabs = csr_rows(self.m, self.owned, slot, self.hq, self.dev, shift=self.P * self.cap - self.no)
        kv_t = np.concatenate(self.per)
        kv_p = np.concatenate([r * self.cap + np.arange(len(self.per[r])) for r in range(self.P)])
        self.btasks, self.bnt = items_for(self.m, self.owned, np.arange(self.no), kv_t, kv_p, self.hq, self.hk)

    def _zero_pads(self, k, v):
        if self.pad is None:
            return k, v
        z = ~self.pad[0][:, None, None]
        return k.masked_fill(z, 0), v.masked_fill(z, 0)

    def _gather(self, k, v):
        T, hk, no, cap = TILE, self.hk, self.no, self.cap
        Kp = torch.zeros(cap * T, hk, D, device=self.dev, dtype=torch.float16)
        Vp = torch.zeros_like(Kp)
        Kp[:no * T] = k
        Vp[:no * T] = v
        Kg = torch.empty(self.P * cap * T, hk, D, device=self.dev, dtype=torch.float16)
        Vg = torch.empty_like(Kg)
        dist.all_gather_into_tensor(Kg, Kp)
        dist.all_gather_into_tensor(Vg, Vp)
        return Kg, Vg

    def fwd(self, q, k, v):
        k, v = self._zero_pads(k, v)
        Kg, Vg = self._gather(k, v)
        out = torch.zeros(1, self.no * TILE, self.hq, D, device=self.dev, dtype=torch.float16)
        lse = torch.full((1, self.hq, self.no * TILE), -float("inf"), device=self.dev, dtype=torch.float32)
        C.sparse_fwd(q[None].contiguous(), Kg[None], Vg[None], out, lse, *self.ftabs, self.scale)
        o, l_ = out[0], lse[0]
        if self.pad is not None:
            pad_correct(o, l_, self.pad[1])
            o.masked_fill_(~self.pad[0][:, None, None], 0)     # pad rows are not outputs
        return o.contiguous(), l_.contiguous()

    def bwd(self, q, k, v, o, lse, do):
        k, v = self._zero_pads(k, v)
        T, h, hk, no, cap = TILE, self.hq, self.hk, self.no, self.cap
        delta = (do.float() * o.float()).sum(-1).t().contiguous()
        dqa = torch.zeros(no * T, h, D, device=self.dev)
        dq = torch.empty(no * T, h, D, device=self.dev, dtype=torch.float16)
        Kg, Vg = self._gather(k, v)
        dKg = torch.zeros(self.P * cap * T, hk, D, device=self.dev)
        dVg = torch.zeros_like(dKg)
        if self.bnt:
            C.sparse_bwd(q, Kg, Vg, do, lse, delta, dqa, None, None, dKg, dVg, *self.btasks, self.scale)
        dKm = torch.empty(cap * T, hk, D, device=self.dev)
        dVm = torch.empty_like(dKm)
        dist.reduce_scatter_tensor(dKm, dKg)
        dist.reduce_scatter_tensor(dVm, dVg)
        dk, dv = dKm[:no * T].half(), dVm[:no * T].half()
        C.dq_convert(dqa, dq, self.scale)
        if self.pad is not None:
            z = ~self.pad[0][:, None, None]
            dk.masked_fill_(z, 0)
            dv.masked_fill_(z, 0)
        return dq, dk, dv


def static_inputs(z, T, HQ, HK, dev):
    """the operator tests' global tensors: q, k, v, do ~ N(0, 0.5^2) fp16, constant seed; pad K / V / dO zero"""
    g = torch.Generator(device=dev).manual_seed(1234)
    q, k, v, do = (torch.randn(T, h_, D, generator=g, device=dev, dtype=torch.float16) * 0.5 for h_ in (HQ, HK, HK, HQ))
    valid = None
    if "tok_valid" in z.files:
        valid = torch.as_tensor(z["tok_valid"].astype(bool), device=dev)
        k[~valid] = 0
        v[~valid] = 0
        do[~valid] = 0
    return q, k, v, do, valid


def run_op(case, op, tokens, pass_, protocol, system, dev):
    """one input of the attention-layer test for an op with fwd / bwd in local tile order"""
    z = np.load(case["mask"])
    T = int(z["tokens"])
    HQ, HK = (int(x) for x in z["heads"])
    q, k, v, do, _ = static_inputs(z, T, HQ, HK, dev)
    ql, kl, vl, dol = (x[tokens].contiguous() for x in (q, k, v, do))
    res = {}

    def fwd():
        res["o"], res["lse"] = op.fwd(ql, kl, vl)

    def bwd():
        res["dq"], res["dk"], res["dv"] = op.bwd(ql, kl, vl, res["o"], res["lse"], dol)

    if pass_ == "fwd":
        ms, ms_all = protocol["latency"](fwd)
    else:
        fwd()
        ms, ms_all = protocol["latency"](bwd)
    torch.cuda.synchronize()
    e64 = None
    if protocol["fp64"]:
        rid = torch.as_tensor(F64.real_ids(z, T), device=dev)
        rv = rid >= 0
        rl = rid[tokens]
        ok = rl >= 0
        rows = rl[ok].cpu().numpy()
        rg = np.load(case["ranges"])
        scale = 1.0 / math.sqrt(D)
        if pass_ == "fwd":
            e64 = F64.check(q[rv], k[rv], v[rv], None, rg, scale, dict(o=res["o"][ok]), q_rows=rows, group=dist.group.WORLD)
        else:
            e64 = F64.check(q[rv], k[rv], v[rv], do[rv], rg, scale, dict(dq=res["dq"][ok], dk=res["dk"][ok], dv=res["dv"][ok]),
                            q_rows=rows, kv_rows=rows, group=dist.group.WORLD)
    return protocol["record"](case, system, pass_, protocol["launch"], ms=ms, ms_all=ms_all, e64=e64)


def run(cases, pass_, protocol):
    dev = torch.device("cuda", torch.cuda.current_device())
    recs = []
    for c in cases:
        z = np.load(c["mask"])
        HQ, HK = (int(x) for x in z["heads"])
        pad = (z["tok_valid"], z["npad"]) if "tok_valid" in z.files else None
        try:
            op = AllGatherStatic(z["mask"], (HQ, HK), pad=pad, device=dev)
            recs.append(run_op(c, op, op.tokens, pass_, protocol, "allgather", dev))
            del op
        except Exception as ex:  # noqa: BLE001  one failing input does not lose the launch
            recs.append(protocol["record"](c, "allgather", pass_, protocol["launch"], status="error", reason=f"{type(ex).__name__}: {str(ex)[:300]}"))
        torch.cuda.empty_cache()
        dist.barrier()
    return recs
