"""Pieces every W6 attention system shares (forward only).

Layer shape: 32 query heads, 8 KV heads (each KV head serves 4 query heads), d = 128, FP16 inside (q / k / v from the
model's BF16), partial O / LSE FP32. Mask: per query head, 128 x 128 tiles, causal; the diagonal tile is always selected
(value 2 = causal inside the tile). Virtual tiles:
  Q   query head h, local tile i  -> row h * nl + i       (Q as [32 L, 1, D])
  KV  KV head g, local tile       -> slot g * nl + loc    (K / V as [8 L, 1, D]); query head h reads KV head h // 4
Every query head computes its own mask; K / V move per KV head, so the transfer unit is (KV head, tile).
System interface: fwd(q16, k16, v16, mrows) -> (o16 [L, 32, D], lse [32 L] fp32), mrows = this rank's rows [32, nl, n] int8.
"""

import os

import torch
import torch.distributed as dist

T, D, HQ, HK, GQ = 128, 128, 32, 8, 4
LSE_EMPTY = -1e4  # LSE of a row with no live key: finite, so its weight in any merge is exactly 0
SCALE = D**-0.5
GPUS_PER_NODE = int(os.environ.get("DEX_GPUS_PER_NODE", "4"))


def cuda_ext():
    """the compiled extension (block-sparse kernels, persistent executor)"""
    from dex import dex_cuda

    return dex_cuda


def to_q(x):  # [L, 32, D] -> [32 L, 1, D]
    return x.transpose(0, 1).reshape(-1, 1, x.shape[-1])


def from_q(x, L):  # [32 L, 1, D] -> [L, 32, D]
    return x.view(HQ, L, x.shape[-1]).transpose(0, 1)


def to_kv(x):  # [L, 8, D] -> [8 L, 1, D]
    return x.transpose(0, 1).reshape(-1, 1, x.shape[-1])


def entries(mrows):
    """this rank's live (q head, local tile, global key tile) entries; kind = causal diagonal (value 2)"""
    h, i, t = torch.nonzero(mrows, as_tuple=True)
    return h, i, t, mrows[h, i, t] == 2


def fwd_tables(row, kind, slot, grp, ngrp, R, delta, col):
    """sparse_fwd tables for several independent calls at once: per row full tiles first (ascending global key tile `col`),
    the causal one last; work = rows by descending count; causal geometry row_mask_mb = row, kv_mask_nb = row + delta
    (delta = (seqlen_k - seqlen_q) / 128; the mask reads only the difference). -> list (per call) of the 6 tables or None
    """
    dev = row.device
    key = ((grp * R + row) * 2 + kind.long()) * (1 << 24) + col
    o = torch.argsort(key)
    row, kind, slot, grp = row[o], kind[o], slot[o], grp[o]
    cnt = torch.bincount(grp * R + row, minlength=ngrp * R).view(ngrp, R)
    nmask = torch.bincount(grp[kind] * R + row[kind], minlength=ngrp * R).view(ngrp, R)
    nnz = cnt.sum(1)
    worder = torch.argsort(-cnt, dim=1, stable=True)
    nwork = (cnt > 0).sum(1)
    hs = torch.cat([nnz, nwork]).cpu().tolist()  # (the one host sync)
    nnz_h, nwork_h = hs[:ngrp], hs[ngrp:]
    nb = torch.where(kind, row + delta, slot)
    idx_s = torch.split(slot.to(torch.int32), nnz_h)
    nb_s = torch.split(nb.to(torch.int32), nnz_h)
    ptr = torch.zeros(ngrp, R + 1, dtype=torch.int64, device=dev)
    ptr[:, 1:] = torch.cumsum(cnt, 1)
    mb = torch.arange(R, dtype=torch.int32, device=dev)
    out = []
    for s in range(ngrp):
        if nnz_h[s] == 0:
            out.append(None)
            continue
        w = worder[s, : nwork_h[s]].to(torch.int32).contiguous()
        out.append(
            (ptr[s].to(torch.int32), idx_s[s].contiguous(), nb_s[s].contiguous(), nmask[s].to(torch.int32), mb, w)
        )
    return out


def pack_rows(mrows):
    """[32, nl, n] int8 (0 / 1 / 2) -> [32, nl, n / 8] uint8 live bits (bit k of byte c = key tile 8 c + k); the diagonal
    kind is implied by the position"""
    h, nl, n = mrows.shape
    b = (mrows != 0).view(h, nl, n // 8, 8).to(torch.uint8)
    return (b << torch.arange(8, device=mrows.device, dtype=torch.uint8)).sum(-1, dtype=torch.uint8)


def mask_global(mrows, lay):
    """every rank's mask rows (bit-packed NCCL all-gather) -> [32, n, n] int8 (global tile order; 2 on live diagonal)"""
    pk = pack_rows(mrows)
    buf = torch.empty((lay.P,) + tuple(pk.shape), dtype=torch.uint8, device=pk.device)
    dist.all_gather_into_tensor(buf, pk.contiguous())
    bits = (buf[..., None] >> torch.arange(8, device=pk.device, dtype=torch.uint8)) & 1  # [P, 32, nl, n/8, 8]
    mh = torch.empty((HQ, lay.n, lay.n), dtype=torch.int8, device=pk.device)
    mh[:, lay.rows_of] = bits.view(lay.P, HQ, lay.nl, lay.n).permute(1, 0, 2, 3).reshape(HQ, -1, lay.n).to(torch.int8)
    ar = torch.arange(lay.n, device=pk.device)
    mh[:, ar, ar] *= 2
    return mh


def tile_causal_masks(mh, S):
    """the FP64 check's per-head masks: key j visible from row i iff tile (i // 128, j // 128) is live and
    (j // 128 < i // 128 or j <= i)"""
    from dex.common import fp64_check as F64

    class TileCausalMask(F64.BlockMask):
        def to(self, dev):
            return TileCausalMask(self.bm.to(dev), self.qblk, self.kblk, self.S)

        def vis(self, rows, keys):
            rt, kt = rows // T, keys // T
            return self.bm[rt[:, None], kt[None, :]] & ((kt[None, :] < rt[:, None]) | (keys[None, :] <= rows[:, None]))

    return [TileCausalMask(mh[g] != 0, T, T, S) for g in range(mh.shape[0])]
