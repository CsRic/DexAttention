"""Dex's context-parallel attention op for static masks (W1-W3): autograd-free fwd / bwd (DexCP, DexCPHeads).

One op per (mask, forward plan, backward plan); both plans share the owners, so every rank holds the tokens of the
tiles it owns, in ascending tile order ("local order"), for every layer. Forward: the scheduler's tables (built once)
drive one persistent forward kernel; rollout padding is corrected exactly afterwards. Backward: the backward tables
(built once) and one persistent backward kernel over Q, dO, LSE, Delta, K, V staged into the symmetric region.
Tensors: q [Tl, Hq, D], k / v [Tl, Hk, D] fp16 contiguous; o [Tl, Hq, D] fp16; lse [Hq, Tl] fp32.
`DexCPHeads` runs plans made with a head split (the plan's `heads_virtual`): copy g of the block-diagonal virtual
mask holds query heads [g Hq/hv, (g+1) Hq/hv) and their KV heads over the whole sequence.
The autograd entry shared by every system's training op lives in dex.attn.cp_common.
"""
import json
import math
import os

import numpy as np
import torch
import torch.distributed as dist

from dex.common.pad_correct import correct as pad_correct

TILE, D = 128, 128
S = C = None


def _load_core():
    """the scheduler and the compiled extension, loaded with the first op"""
    global S, C
    if C is None:
        from dex import dex_cuda, scheduler
        S, C = scheduler, dex_cuda


def fwd_region_bytes(t, hq, hk):
    """symmetric bytes of a forward region: Q / K / V slots, fp32 partial O + LSE slots, one signal word per slot"""
    q_slot, kv_slot = 128 * hq * D * 2, 128 * hk * D * 2
    return (t["NQ"] * q_slot + 2 * t["NKV"] * kv_slot + t["NPART"] * (2 * q_slot + hq * 128 * 4)
            + 8 * (t["NQ"] + t["NKV"] + t["NPART"]))


def _t32(x):
    """int32 device tensor of a host table (int64 tables as int32 word pairs)"""
    x = np.ascontiguousarray(x)
    x = x.view(np.int32).reshape(-1) if x.dtype == np.int64 else np.ascontiguousarray(x, np.int32).reshape(-1)
    return torch.from_numpy(x if len(x) else np.zeros(2, np.int32)).cuda()


class DexCP:
    def __init__(self, mask, fwd_plan, bwd_plan, heads, pad=None, device=None):
        """mask [nt, nt] int8 tile mask (0 / 1 / 2); fwd_plan / bwd_plan: plan json paths; heads (Hq, Hk);
        pad: None or (tok_valid [T] bool, npad [T] int32) numpy arrays (rollout padding)"""
        _load_core()
        self.rank, self.P = dist.get_rank(), dist.get_world_size()
        self.dev = device or torch.device("cuda")
        self.m = np.asarray(mask, np.int8)
        self.hq, self.hk = heads
        n = self.m.shape[0]
        devb, ownb, meta = S.load_plan(fwd_plan)
        per = meta["block_tokens"] // TILE
        B = S.block_pairs(self.m, devb, per)
        pw = (B["nfull"] + 0.5 * B["ndiag"]) * S.LINK["T_TILE"] * self.hq / 32
        if S.local.enabled(self.P):
            self.ftab, tm = S.local.fwd_tables(self.rank, self.P, B, pw, ownb, per, self.dev, heads=heads, d=D)
        else:
            S.schedule_blocks(B["bq"], B["bk"], B["pdev"], pw, B["prow"], B["pcol"], ownb, self.P, per, heads=heads, d=D)
            self.ftab, tm = S.tables_blocks(self.rank, B, ownb, self.P, per, self.dev)
        self.owned = np.asarray(tm["owned"], np.int64)
        pl = json.load(open(bwd_plan))
        own_t, _ = S.expand_plan(pl, n)
        assert np.array_equal(np.flatnonzero(own_t == self.rank), np.sort(self.owned)), "fwd / bwd plans differ in owners"
        pb = pl["block_tokens"] // TILE
        nbp = n // pb
        Bp = S.block_pairs(self.m, np.asarray(pl["schedule_table"])[:nbp, :nbp], pb)
        ownbb = np.asarray(pl["owner"])[:nbp]
        if S.local.enabled(self.P):
            self.bglob, self.bmine = S.local.bwd_tables(self.rank, self.P, Bp, ownbb, pb, self.hq, self.hk)
        else:
            self.bglob, self.bmine = S.build_fast(Bp, ownbb, self.P, pb, self.rank, self.hq, self.hk)
        self.no = len(self.owned)
        self.tokens = torch.as_tensor((self.owned[:, None] * TILE + np.arange(TILE)).reshape(-1), device=self.dev)
        self.pad = None
        if pad is not None:
            valid, npad = pad
            self.pad = (torch.as_tensor(np.asarray(valid, bool), device=self.dev)[self.tokens],
                        torch.as_tensor(np.asarray(npad, np.int32), device=self.dev)[self.tokens])
        self.scale = 1.0 / math.sqrt(D)
        self.region = None
        self._btabs = None

    # ---- symmetric memory (collective) ------------------------------------------------------------------------
    def heap_bytes(self):
        """(forward region bytes, backward region bytes)"""
        return fwd_region_bytes(self.ftab, self.hq, self.hk), S.region_layout(self.bglob, self.hq, self.hk)[2]

    @staticmethod
    def shared_region():
        """True when the extension lets the forward reuse the backward region's data slots"""
        f = getattr(C, "dist_share_region", None)
        return bool(f and f())

    def heap_used_bytes(self):
        """symmetric memory this op allocates: shared region -> max(fwd data, bwd) + the forward's signal words"""
        f, b = self.heap_bytes()
        fsig = 8 * (self.ftab["NQ"] + self.ftab["NKV"] + self.ftab["NPART"])
        return max(f - fsig, b) + fsig if DexCP.shared_region() else f + b

    @staticmethod
    def init_symmetric(instances):
        """after torch.distributed init: size the NVSHMEM heap for every instance (largest region when shared, else
        the sum of the forward regions + the largest backward region; margin 2% of the larger region when shared, else of
        the forward sum, + 128 MiB), bootstrap NVSHMEM and
        allocate the backward region shared by all instances"""
        from dex.runtime import init_nvshmem
        sh = DexCP.shared_region()
        f = (max if sh else sum)(x.heap_bytes()[0] for x in instances)
        b = max(x.heap_bytes()[1] for x in instances)
        t = torch.tensor([f, b], device="cuda", dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        if sh:
            size = int(max(t[0].item(), t[1].item()) * 1.02 + (128 << 20))
        else:
            size = int(t[0].item() * 1.02 + t[1].item() + (128 << 20))
        os.environ["NVSHMEM_SYMMETRIC_SIZE"] = str(size)
        init_nvshmem(dist.group.WORLD)
        region = C.bwd_sym_alloc(int(max(t[0].item(), t[1].item()) if sh else t[1].item()))
        info = C.bwd_pe_info()
        for x in instances:
            x.region, x.pe_info = region, info
        return region

    # ---- forward / backward -----------------------------------------------------------------------------------
    def fwd(self, q, k, v):
        """-> o [Tl, Hq, D] fp16, lse [Hq, Tl] fp32 (padding-corrected)"""
        Tl = q.shape[0]
        out = torch.zeros(1, Tl, self.hq, D, device=self.dev, dtype=torch.float16)
        lse = torch.full((1, self.hq, Tl), -float("inf"), device=self.dev, dtype=torch.float32)
        if self.pad is not None:
            k = k.masked_fill(~self.pad[0][:, None, None], 0)
            v = v.masked_fill(~self.pad[0][:, None, None], 0)
        S.call(C, q[None].contiguous(), k[None].contiguous(), v[None].contiguous(), out, lse, self.ftab, self.scale)
        if self.pad is not None:
            pad_correct(out[0], lse[0], self.pad[1])
            out[0].masked_fill_(~self.pad[0][:, None, None], 0)
        return out[0], lse[0]

    def _bwd_setup(self):
        st, glob = self.bmine["stats"], self.bglob
        self._btabs = [_t32(self.bmine["tabs"][k_]) for k_ in S.TABLE_ORDER + S.RECT_ORDER]
        self._ce = torch.from_numpy(np.ascontiguousarray(self.bmine["tabs"]["ce"], np.int64))
        self._n_sm = torch.cuda.get_device_properties(0).multi_processor_count
        self._soffs, ssize = S.scratch_layout(st, glob, self.hq, self.hk, self._n_sm - 1)
        self._scratch = torch.zeros(ssize, dtype=torch.int32, device=self.dev)
        self._dims = [glob["NQS"], glob["NKS"], glob["NDQL"], glob["NDKL"], self.hq, self.hk, D, st["tasks"], st["n_init"],
                      st["n_qm"], st["n_km"], st["n_remote"], st["n_start"], 0, st["n_qagg"] + st["n_kagg"], st["n_pq"],
                      1, st["n_rect"], st["G1"], st["per"], st["nw"], st["arena"], st["n_groups"]]

    def bwd(self, q, k, v, o, lse, do):
        """-> dq, dk, dv (fp16) for this rank's tokens"""
        if self._btabs is None:
            self._bwd_setup()
        h, hk, no, T = self.hq, self.hk, self.no, TILE
        offs, sizes, _ = S.region_layout(self.bglob, h, hk)
        NQS, NKS = self.bglob["NQS"], self.bglob["NKS"]
        if self.pad is not None:
            k = k.masked_fill(~self.pad[0][:, None, None], 0)
            v = v.masked_fill(~self.pad[0][:, None, None], 0)
        delta = (do.float() * o.float()).sum(-1).t().contiguous()           # [h, Tl]
        qb = self.region[offs[1]:offs[1] + NQS * sizes[0]].view(NQS, sizes[0])
        kvr = self.region[offs[2]:offs[2] + NKS * sizes[1]].view(NKS, sizes[1])
        qbytes, kbytes = T * h * D * 2, T * hk * D * 2
        v16 = lambda x, a_, b_: x[:no, a_:b_].view(torch.float16)  # noqa: E731
        v32 = lambda x, a_, b_: x[:no, a_:b_].view(torch.float32)  # noqa: E731
        v16(qb, 0, qbytes).copy_(q.reshape(no, T * h * D))
        v16(qb, qbytes, 2 * qbytes).copy_(do.reshape(no, T * h * D))
        v32(qb, 2 * qbytes, 2 * qbytes + h * T * 4).copy_(lse.view(h, no, T).permute(1, 0, 2).reshape(no, h * T))
        v32(qb, 2 * qbytes + h * T * 4, 2 * qbytes + 2 * h * T * 4).copy_(
            delta.view(h, no, T).permute(1, 0, 2).reshape(no, h * T))
        v16(kvr, 0, kbytes).copy_(k.reshape(no, T * hk * D))
        v16(kvr, kbytes, 2 * kbytes).copy_(v.reshape(no, T * hk * D))
        dq = torch.zeros(no * T, h, D, device=self.dev, dtype=torch.float16)
        dk = torch.zeros(no * T, hk, D, device=self.dev, dtype=torch.float16)
        dv = torch.zeros_like(dk)
        self._scratch.zero_()
        C.dist_bwd_run(offs, sizes, self._dims, self._btabs, self.pe_info[0], self.pe_info[1], self._scratch, self._soffs,
                       dq, dk, dv, self.scale, self._n_sm, self._ce)
        if self.pad is not None:          # pad K / V were zero in the forward: their gradient is 0
            dk.masked_fill_(~self.pad[0][:, None, None], 0)
            dv.masked_fill_(~self.pad[0][:, None, None], 0)
        return dq, dk, dv


def _to_v(x, hv):
    """[Tl, H, D] -> [hv * Tl, H / hv, D]: copy g = heads [g H/hv, (g+1) H/hv)"""
    Tl, H, D_ = x.shape
    return x.view(Tl, hv, H // hv, D_).transpose(0, 1).reshape(hv * Tl, H // hv, D_)


def _from_v(x, hv):
    """inverse of _to_v"""
    T2, h, D_ = x.shape
    return x.view(hv, T2 // hv, h, D_).transpose(0, 1).reshape(T2 // hv, hv * h, D_)


class DexCPHeads(DexCP):
    """DexCP on a head-split plan with hv virtual heads. Every copy has the same owners (the plan repeats the
    placement per copy), so each rank keeps the same tokens for all heads; the op permutes the model's [Tl, H, D]
    tensors into the virtual local order (copy-major, then this rank's tiles ascending) and back."""

    def __init__(self, mask, fwd_plan, bwd_plan, heads, pad=None, device=None):
        pl = json.load(open(fwd_plan))
        hqv, hkv = pl["heads_virtual"]
        hq, hk = heads
        hv = hq // hqv
        assert hq == hv * hqv and hk == hv * hkv, (heads, pl["heads_virtual"])
        m = np.asarray(mask, np.int8)
        n = m.shape[0]
        mv = np.kron(np.eye(hv, dtype=np.int8), m).astype(np.int8)
        padv = None if pad is None else (np.tile(np.asarray(pad[0], bool), hv), np.tile(np.asarray(pad[1], np.int32), hv))
        super().__init__(mv, fwd_plan, bwd_plan, (hqv, hkv), pad=padv, device=device)
        self.hv, self.hq_model, self.hk_model = hv, hq, hk
        no1 = len(self.owned) // hv
        base = np.asarray(self.owned[:no1])
        assert len(self.owned) == hv * no1 and np.array_equal(
            np.asarray(self.owned), np.concatenate([g * n + base for g in range(hv)])), "head-split plan: owners differ between copies"
        self.vtokens = self.tokens
        self.tokens = self.vtokens[:no1 * TILE]                # the model's token positions (copy 0)

    def fwd(self, q, k, v):
        hv = self.hv
        o, lse = super().fwd(_to_v(q, hv), _to_v(k, hv), _to_v(v, hv))
        hqv, T2 = lse.shape
        return _from_v(o, hv), lse.view(hqv, hv, T2 // hv).transpose(0, 1).reshape(hv * hqv, T2 // hv)

    def bwd(self, q, k, v, o, lse, do):
        hv = self.hv
        hq, Tl = lse.shape
        lv = lse.view(hv, hq // hv, Tl).transpose(0, 1).reshape(hq // hv, hv * Tl).contiguous()
        dq, dk, dv = super().bwd(_to_v(q, hv), _to_v(k, hv), _to_v(v, hv), _to_v(o, hv), lv, _to_v(do, hv))
        return _from_v(dq, hv), _from_v(dk, hv), _from_v(dv, hv)


def make_op(mask, fwd_plan, bwd_plan, heads, pad=None, device=None):
    """DexCPHeads for a head-split plan (`heads_virtual` in the plan file), else DexCP"""
    cls = DexCPHeads if "heads_virtual" in json.load(open(fwd_plan)) else DexCP
    return cls(mask, fwd_plan, bwd_plan, heads, pad=pad, device=device)
