"""W6 (Llama-3.1-8B + FlexPrefill, forward only) Ring attention op: upstream Yunchang ring communication and fp32 merge
(baselines.ring.ring_sparse), with sparse_fwd computing, per ring step, only the live part of (this rank's rows x the
visiting rank's tiles). System interface of workloads.w6_flexprefill.common."""

import torch

from baselines.ring import ring_sparse as RS
from workloads.w6_flexprefill.common import (
    GQ,
    HK,
    HQ,
    LSE_EMPTY,
    SCALE,
    cuda_ext,
    entries,
    from_q,
    fwd_tables,
    to_kv,
    to_q,
)


class _RingComp:
    def __init__(self, DA, mrows, lay):
        self.DA = DA
        h, i, t, kind = entries(mrows)
        R = HQ * lay.nl
        self.ft = fwd_tables(
            h * lay.nl + i, kind, (h // GQ) * lay.nl + lay.t_loc[t], lay.t_own[t], lay.P, R, HK * lay.nl - R, t
        )

    def fwd(self, src, q, k, v):
        tb = self.ft[src]
        if tb is None:
            return None
        out = torch.zeros(q.shape, device=q.device, dtype=q.dtype)
        lse = torch.full((1, 1, q.shape[1]), LSE_EMPTY, device=q.device, dtype=torch.float32)
        self.DA.sparse_fwd(q.contiguous(), k.contiguous(), v.contiguous(), out, lse, *tb, SCALE)
        return out, lse


class RingFlex:
    def __init__(self, lay):
        self.lay, self.DA = lay, cuda_ext()

    def fwd(self, q16, k16, v16, mrows):
        comp = _RingComp(self.DA, mrows, self.lay)
        o, lse = RS.ring_forward(None, to_q(q16)[None], to_kv(k16)[None], to_kv(v16)[None], comp)
        return from_q(o[0].to(torch.float16), self.lay.L), lse[0, 0].float()
