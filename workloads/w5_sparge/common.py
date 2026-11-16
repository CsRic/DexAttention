"""Pieces every W5 attention system shares: shapes, the per-layer reuse cache of the training step, the K / V
all-gather in padded tile order and the "virtual tile" view of per-head tensors.

System interface (Dex, Dex without assigner, Ring, All-gather, Sparse all-gather, decomposed):
    sys = System(lay, H)
    o, (fm, lse) = sys.fwd(q, k, v)                    q / k / v [H, L, 128] fp16 of this rank's real tokens
    dq, dk, dv = sys.bwd(q, k, v, o, do, fm, lse, cache)
fm = this rank's SpargeAttention2 mask rows [H, nqb_local, nkb] (128 x 64 blocks, real coordinates); compute is exact FP16
block-sparse attention on that mask (forward kBlockN = 64, backward per 64-key half)."""

import torch
import torch.distributed as dist

from dex.common.cp_layout import owned

T, D = 128, 128
SCALE = D**-0.5
LSE_EMPTY = -1e4  # LSE of a row with no live key: finite, so its weight in any merge is exactly 0


def cuda_ext():
    """the compiled extension (block-sparse kernels, persistent executor)"""
    from dex import dex_cuda

    return dex_cuda


class Reuse:
    """Per-layer reuse within one micro-batch of the training step: the backward's recomputation reuses the layer's
    selection (every system) and Dex's forward tables; Dex's backward plan is searched ahead on a worker thread.
    The operator tests use `enabled=True, ahead=False` and preload the mask to time the known-mask scope."""

    enabled = True
    ahead = True
    caches = {}
    layer = [None]

    @classmethod
    def cache(cls):
        """this layer's reuse record (created on its first forward of the micro-batch); None when reuse is off"""
        if not cls.enabled:
            return None
        return cls.caches.setdefault(cls.layer[0], {})

    @classmethod
    def clear(cls):
        cls.caches.clear()


def view_tiles(x):
    """[H, L, D] -> [1, H * L, 1, D]: per-head rows as one sequence of virtual tiles (head h, tile t) -> h * nl + t"""
    return x.reshape(1, x.shape[0] * x.shape[1], 1, x.shape[2])


def gather_padded(x, lay):
    """[H, L_pad, D] per rank (padded chunks) -> [H, n_pad * 128, D] in global tile order (NCCL all-gather)"""
    H, L, D_ = x.shape
    P, per = lay.P, lay.per
    buf = torch.empty(P, H, L, D_, dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(buf, x.contiguous())
    out = torch.empty(H, lay.n_pad * T, D_, dtype=x.dtype, device=x.device)
    for r in range(P):
        o_ = 0
        for c in owned(r, P):
            out[:, c * per * T : (c + 1) * per * T] = buf[r, :, o_ : o_ + per * T]
            o_ += per * T
    return out
