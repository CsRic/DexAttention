"""W6 (Llama-3.1-8B + FlexPrefill, forward only) All-gather and Sparse all-gather attention ops.

All-gather: NCCL all-gather of K / V (fp16), one sparse_fwd over this rank's rows x all gathered tiles.
Sparse all-gather: need = union over the 4 query heads of the rank's rows (per KV head), need lists + one exchange of exactly
those K + V tiles (all point-to-point operations in one batch_isend_irecv), one sparse_fwd over [own, received].
System interface of workloads.w6_flexprefill.common; kernel tables are built on the GPU per call."""

import torch
import torch.distributed as dist

from baselines.allgather.sparse_allgather import Lists, sparse_fwd_tables
from workloads.w6_flexprefill.common import (
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
    to_kv,
    to_q,
)

EXCHANGE_MODE = "batch"


class AllGatherFlex:
    def __init__(self, lay):
        self.lay, self.DA = lay, cuda_ext()

    def fwd(self, q16, k16, v16, mrows):
        lay, dev, P = self.lay, q16.device, self.lay.P
        Kg = torch.empty((P,) + tuple(k16.shape), dtype=k16.dtype, device=dev)
        dist.all_gather_into_tensor(Kg, k16)
        Vg = torch.empty_like(Kg)
        dist.all_gather_into_tensor(Vg, v16)
        Kv = Kg.permute(0, 2, 1, 3).reshape(P * HK * lay.L, 1, D)  # slot = r 8 nl + g nl + local tile
        Vv = Vg.permute(0, 2, 1, 3).reshape(P * HK * lay.L, 1, D)
        h, i, t, kind = entries(mrows)
        R = HQ * lay.nl
        slot = lay.t_own[t] * (HK * lay.nl) + (h // GQ) * lay.nl + lay.t_loc[t]
        tabs = fwd_tables(h * lay.nl + i, kind, slot, torch.zeros_like(h), 1, R, P * HK * lay.nl - R, t)[0]
        out = torch.zeros(1, HQ * lay.L, 1, D, dtype=torch.float16, device=dev)
        lse = torch.full((1, 1, HQ * lay.L), LSE_EMPTY, dtype=torch.float32, device=dev)
        self.DA.sparse_fwd(to_q(q16)[None], Kv[None], Vv[None], out, lse, *tabs, SCALE)
        return from_q(out[0], lay.L), lse[0, 0]


class SparseAllGatherFlex:
    def __init__(self, lay):
        self.lay, self.DA = lay, cuda_ext()
        self.last = None

    def fwd(self, q16, k16, v16, mrows):
        lay, dev = self.lay, q16.device
        need = (mrows != 0).any(1).view(HK, GQ, lay.n).any(1)  # [8, n] per KV head: union of its 4 heads
        Ls = Lists(need, lay.t_own, lay.t_loc, lay.nl)
        Kc, Vc = Ls.gather_kv(
            to_kv(k16).view(HK * lay.nl, T, D), to_kv(v16).view(HK * lay.nl, T, D), mode=EXCHANGE_MODE
        )
        h, i, t, kind = entries(mrows)
        R = HQ * lay.nl
        tabs = sparse_fwd_tables(h * lay.nl + i, kind, Ls.slot[h // GQ, t], t, R, HK * lay.nl + Ls.nrecv - R, 1)
        out = torch.zeros(1, R * T, 1, D, dtype=torch.float16, device=dev)
        lse = torch.full((1, 1, R * T), LSE_EMPTY, dtype=torch.float32, device=dev)
        self.DA.sparse_fwd(
            to_q(q16)[None].contiguous(), Kc.view(1, -1, 1, D), Vc.view(1, -1, 1, D), out, lse, *tabs, SCALE
        )
        self.last = Ls
        return from_q(out[0], lay.L), lse[0, 0]
