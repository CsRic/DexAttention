"""W5 (Wan2.1 + SpargeAttention2) All-gather and Sparse all-gather attention ops (forward + backward).

All-gather: NCCL all-gather of K / V, SpargeAttention2's selection on the gathered keys, one block-sparse forward
(sparse_fwd64, 128 x 64 blocks) over this rank's rows; backward: all-gather again, one sparse_bwd over per-head task lists,
FP32 reduce-scatter of the partial dK / dV to their owners.
Sparse all-gather: the selection split by owner (one all-reduce of the pooled keys), then each rank fetches exactly the
remote (head, 128-key tile) K / V it needs (need lists all-gathered, one point-to-point exchange), one block-sparse
forward over [own tiles, received tiles]; backward: the same lists and exchange, one sparse_bwd, the FP32 partial dK / dV
of the received tiles sent back to their owners and summed there. The exchange is one batch of point-to-point operations ("batch").
Both use the balanced zigzag layout of dex.common.cp_layout and the system interface of workloads.w5_sparge.common.
"""

import numpy as np
import torch
import torch.distributed as dist

from baselines.allgather.sparse_allgather import Lists, sparse_bwd_tables, sparse_fwd_tables
from baselines.ring import ring_sparse as RS
from dex.common.cp_layout import owned
from workloads.w5_sparge import selection as SA
from workloads.w5_sparge.common import LSE_EMPTY, SCALE, T, D, Reuse, cuda_ext, gather_padded, view_tiles

EXCHANGE_MODE = "batch"


class AllGatherSparge:
    def __init__(self, lay, H):
        self.DA, self.lay, self.H = cuda_ext(), lay, H

    def fwd(self, q, k, v):
        lay, H, dev = self.lay, self.H, q.device
        Kp, Vp = gather_padded(lay.pad(k), lay), gather_padded(lay.pad(v), lay)  # [H, S_pad, D] (padded positions)
        c = Reuse.cache()
        if c is not None and "fm" in c:
            fm = c["fm"]
        else:
            fm = SA.select(SA.pooled_q(q), SA.pooled_k(Kp[:, lay.pos_real]), q.shape[-1] ** -0.5)
            if c is not None:
                c["fm"] = fm
        qp = lay.pad(q)
        out = torch.zeros(1, H * lay.L_pad, 1, D, dtype=torch.float16, device=dev)
        lse = torch.full((1, 1, H * lay.L_pad), LSE_EMPTY, dtype=torch.float32, device=dev)
        # K / V viewed as 64-row tiles: slot h * 2 n_pad + c
        self.DA.sparse_fwd64(
            view_tiles(qp),
            view_tiles(Kp),
            view_tiles(Vp),
            out,
            lse,
            *RS.head_csr(lay.tile_rows(fm), 2 * lay.n_pad),
            SCALE
        )
        return lay.unpad(out.view(H, lay.L_pad, D)), (fm, lay.unpad(lse.view(H, lay.L_pad)))

    def bwd(self, q, k, v, o, do, fm, lse, cache=None):
        DA, lay, H, dev, P = self.DA, self.lay, self.H, q.device, self.lay.P
        q, k, v, o, do = (lay.pad(x) for x in (q, k, v, o, do))
        lse = lay.pad(lse)
        mrows = lay.tile_rows(fm)
        L, n_pad, per = lay.L_pad, lay.n_pad, lay.per
        vq = q.reshape(H * L, 1, D)
        vdo = do.reshape(H * L, 1, D)
        vo = o.reshape(H * L, 1, D)
        lse_v = lse.reshape(1, H * L).contiguous()
        S_pad = n_pad * T
        Kg = gather_padded(k, lay).reshape(H * S_pad, 1, D)
        Vg = gather_padded(v, lay).reshape(H * S_pad, 1, D)
        dps = torch.empty(1, H * L, device=dev)
        dq_acc = torch.zeros(H * L, 1, D, device=dev)
        DA.dot_do_o(vdo, vo, dps, None)
        task_kv, task_ptr, items = RS.head_bwd_tasks(mrows, True)
        dk_acc = torch.zeros(H * S_pad, 1, D, device=dev)
        dv_acc = torch.zeros_like(dk_acc)
        DA.sparse_bwd(vq, Kg, Vg, vdo, lse_v, dps, dq_acc, None, None, dk_acc, dv_acc, task_kv, task_ptr, items, SCALE)
        dq = torch.empty(H * L, 1, D, dtype=torch.float16, device=dev)
        DA.dq_convert(dq_acc, dq, SCALE)
        rs_in = torch.empty(P, 2, H, L, D, device=dev)
        dkv = torch.stack([dk_acc.view(H, S_pad, D), dv_acc.view(H, S_pad, D)])
        for r in range(P):
            o_ = 0
            for c in owned(r, P):
                rs_in[r, :, :, o_ : o_ + per * T] = dkv[:, :, c * per * T : (c + 1) * per * T]
                o_ += per * T
        mine = torch.empty(2, H, L, D, device=dev)
        dist.reduce_scatter_tensor(mine, rs_in)
        return lay.unpad(dq.view(H, L, D)), lay.unpad(mine[0].half()), lay.unpad(mine[1].half())


class SparseAllGatherSparge:
    """need unit = (head, padded 128-key tile) with any live 64-key half in this rank's rows: whole 128-key K / V tiles
    move; the compute lists keep the exact live halves"""

    def __init__(self, lay, H):
        self.DA, self.lay, self.H = cuda_ext(), lay, H

    def _own(self):
        lay = self.lay
        if getattr(self, "_lay_own", None) is not lay:
            P = lay.P
            t_own = np.empty(lay.n_pad, np.int64)
            t_loc = np.empty(lay.n_pad, np.int64)
            for r in range(P):
                tl = lay.tiles_of(r)
                t_own[tl] = r
                t_loc[tl] = np.arange(len(tl))
            dev = lay.tok.device
            self._t_own, self._t_loc = torch.as_tensor(t_own, device=dev), torch.as_tensor(t_loc, device=dev)
            self._lay_own = lay
        return self._t_own, self._t_loc

    def _prep(self, fm):
        """mask rows over 64-key half columns -> need lists and the live entries (row, half column, tile, half)"""
        lay, H = self.lay, self.H
        t_own, t_loc = self._own()
        mr = lay.tile_rows(fm)
        need = mr.view(H, lay.nl, lay.n_pad, 2).any(3).any(1)
        Ls = Lists(need, t_own, t_loc, lay.nl)
        h, i, c = torch.nonzero(mr, as_tuple=True)
        return Ls, h * lay.nl + i, c, c // 2, c % 2

    def fwd(self, q, k, v):
        lay, H, dev = self.lay, self.H, q.device
        c_ = Reuse.cache()
        if c_ is not None and "fm" in c_:
            fm = c_["fm"]
        else:
            fm = SA.select_split(q, k, lay)
            if c_ is not None:
                c_["fm"] = fm
        Ls, row, c, t, hn = self._prep(fm)
        Kc, Vc = Ls.gather_kv(lay.pad(k).view(H * lay.nl, T, D), lay.pad(v).view(H * lay.nl, T, D), mode=EXCHANGE_MODE)
        s128 = Ls.slot[row // lay.nl, t]
        kind = torch.zeros_like(row, dtype=torch.bool)
        tabs = sparse_fwd_tables(row, kind, 2 * s128 + hn, c, H * lay.nl, 0, 1)
        qp = lay.pad(q)
        out = torch.zeros(1, H * lay.L_pad, 1, D, dtype=torch.float16, device=dev)
        lse = torch.full((1, 1, H * lay.L_pad), LSE_EMPTY, dtype=torch.float32, device=dev)
        self.DA.sparse_fwd64(view_tiles(qp), Kc.view(1, -1, 1, D), Vc.view(1, -1, 1, D), out, lse, *tabs, SCALE)
        return lay.unpad(out.view(H, lay.L_pad, D)), (fm, lay.unpad(lse.view(H, lay.L_pad)))

    def bwd(self, q, k, v, o, do, fm, lse, cache=None):
        DA, lay, H, dev = self.DA, self.lay, self.H, q.device
        q, k, v, o, do = (lay.pad(x) for x in (q, k, v, o, do))
        lse = lay.pad(lse)
        L = lay.L_pad
        Ls, row, c, t, hn = self._prep(fm)
        Kc, Vc = Ls.gather_kv(k.view(H * lay.nl, T, D), v.view(H * lay.nl, T, D), mode=EXCHANGE_MODE)
        kind = torch.zeros_like(row, dtype=torch.bool)
        tk = sparse_bwd_tables(row, kind, Ls.slot[row // lay.nl, t], H * lay.nl, 1, hn_only=hn)
        vq = q.reshape(H * L, 1, D)
        vdo = do.reshape(H * L, 1, D)
        vo = o.reshape(H * L, 1, D)
        lse_v = lse.reshape(1, H * L).contiguous()
        dps = torch.empty(1, H * L, device=dev)
        DA.dot_do_o(vdo, vo, dps, None)
        dq_acc = torch.zeros(H * L, 1, D, device=dev)
        nk = Kc.shape[0]
        dK = torch.zeros(nk * T, 1, D, device=dev)
        dV = torch.zeros_like(dK)
        if len(tk[0]):
            DA.sparse_bwd(
                vq, Kc.view(-1, 1, D), Vc.view(-1, 1, D), vdo, lse_v, dps, dq_acc, None, None, dK, dV, *tk, SCALE
            )
        dk, dv = Ls.return_dkv(dK.view(nk, T, D), dV.view(nk, T, D), mode=EXCHANGE_MODE)
        dq = torch.empty(H * L, 1, D, dtype=torch.float16, device=dev)
        DA.dq_convert(dq_acc, dq, SCALE)
        return (lay.unpad(dq.view(H, L, D)), lay.unpad(dk.view(H, L, D).half()), lay.unpad(dv.view(H, L, D).half()))
