"""W5 (Wan2.1 + SpargeAttention2) Ring attention op: upstream Yunchang ring communication and fp32 merge
(baselines.ring.ring_sparse), with the block-sparse kernels computing, per ring step, only the live part of (this rank's
tiles x the visiting rank's tiles) on the per-head 128 x 64 mask (ring_sparse.HeadTileComp, half=True). The selection is
split by owner (one all-reduce of the pooled keys). System interface of workloads.w5_sparge.common."""

import torch

from baselines.ring import ring_sparse as RS
from workloads.w5_sparge import selection as SA
from workloads.w5_sparge.common import SCALE, D, Reuse, cuda_ext, view_tiles


class RingSparge:
    def __init__(self, lay, H):
        self.DA, self.lay, self.H = cuda_ext(), lay, H

    def fwd(self, q, k, v):
        lay, H = self.lay, self.H
        c = Reuse.cache()
        if c is not None and "fm" in c:
            fm = c["fm"]
        else:
            fm = SA.select_split(q, k, lay)
            if c is not None:
                c["fm"] = fm
        comp = RS.HeadTileComp(self.DA, lay.tile_rows(fm), lay.tiles_of, SCALE, half=True)
        o, lse = RS.ring_forward(None, view_tiles(lay.pad(q)), view_tiles(lay.pad(k)), view_tiles(lay.pad(v)), comp)
        return lay.unpad(o.to(torch.float16).view(H, lay.L_pad, D)), (fm, lay.unpad(lse.float().view(H, lay.L_pad)))

    def bwd(self, q, k, v, o, do, fm, lse, cache=None):
        DA, lay, H, dev = self.DA, self.lay, self.H, q.device
        q, k, v, o, do = (lay.pad(x) for x in (q, k, v, o, do))
        lse = lay.pad(lse)
        L = lay.L_pad
        comp = RS.HeadTileComp(DA, lay.tile_rows(fm), lay.tiles_of, SCALE, half=True)
        vdo = do.reshape(H * L, 1, D)
        vo = o.reshape(H * L, 1, D)
        lse_v = lse.reshape(1, H * L).contiguous()
        dps = torch.empty(1, H * L, device=dev)
        DA.dot_do_o(vdo, vo, dps, None)
        dqa, dk_, dv_ = RS.ring_backward(None, do, q, k, v, o, lse_v, dps, comp)
        dq = torch.empty(H * L, 1, D, dtype=torch.float16, device=dev)
        DA.dq_convert(dqa.view(H * L, 1, D), dq, SCALE)
        return lay.unpad(dq.view(H, L, D)), lay.unpad(dk_.half()), lay.unpad(dv_.half())
