"""Ring on W1-W3: dense ring communication (ring_sparse) with the block-sparse tile kernels All-gather uses per step,
zigzag tiles. Same op contract as All-gather: fwd / bwd / tokens in the rank's tile order; rollout padding = pad K / V zero,
exact O / LSE correction after the forward, pad-row dK / dV zero.
`run` is the attention-layer test (experiments/attention_layer/static_op.py), an untimed forward warm-up first because
the first ring steps of a launch pay one-off NCCL connection set-up."""
import math

import numpy as np
import torch
import torch.distributed as dist

from baselines.allgather.static import run_op
from baselines.ring import ring_sparse as RS
from dex.common.pad_correct import correct as pad_correct

from dex import dex_cuda as C

TILE, D = 128, 128


class RingStatic:
    def __init__(self, mask, heads, pad=None, device=None):
        self.rank, self.P = dist.get_rank(), dist.get_world_size()
        self.dev = device or torch.device("cuda", torch.cuda.current_device())
        self.hq, self.hk = heads
        self.scale = 1.0 / math.sqrt(D)
        self.comp = RS.TileComp(C, np.asarray(mask, np.int8), self.P, self.rank, self.hq, self.hk, self.scale, self.dev)
        tiles = np.asarray(self.comp.per[self.rank])
        self.tokens = torch.as_tensor((tiles[:, None] * TILE + np.arange(TILE)).reshape(-1), device=self.dev)
        self.pad = None
        if pad is not None:
            valid, npad = pad
            self.pad = (torch.as_tensor(np.asarray(valid, bool), device=self.dev)[self.tokens],
                        torch.as_tensor(np.asarray(npad, np.int32), device=self.dev)[self.tokens])

    def _zero_pads(self, k, v):
        if self.pad is None:
            return k, v
        z = ~self.pad[0][:, None, None]
        return k.masked_fill(z, 0), v.masked_fill(z, 0)

    def fwd(self, q, k, v):
        k, v = self._zero_pads(k, v)
        o, lse = RS.ring_forward(None, q[None].contiguous(), k[None].contiguous(), v[None].contiguous(), self.comp)
        o, lse = o[0], lse[0]
        if self.pad is not None:
            pad_correct(o, lse, self.pad[1])
            o.masked_fill_(~self.pad[0][:, None, None], 0)     # pad rows are not outputs
        return o.contiguous(), lse.contiguous()

    def bwd(self, q, k, v, o, lse, do):
        k, v = self._zero_pads(k, v)
        delta = (do.float() * o.float()).sum(-1).t().contiguous()
        dqa, dk, dv = RS.ring_backward(None, do[None].contiguous(), q[None].contiguous(), k[None].contiguous(),
                                       v[None].contiguous(), o[None], lse[None], delta, self.comp)
        dq = torch.empty(q.shape, device=self.dev, dtype=torch.float16)
        C.dq_convert(dqa[0], dq, self.scale)
        dk, dv = dk[0].half(), dv[0].half()
        if self.pad is not None:
            z = ~self.pad[0][:, None, None]
            dk.masked_fill_(z, 0)
            dv.masked_fill_(z, 0)
        return dq, dk, dv


def run(cases, pass_, protocol):
    dev = torch.device("cuda", torch.cuda.current_device())
    recs, first = [], True
    for c in cases:
        z = np.load(c["mask"])
        HQ, HK = (int(x) for x in z["heads"])
        pad = (z["tok_valid"], z["npad"]) if "tok_valid" in z.files else None
        try:
            op = RingStatic(z["mask"], (HQ, HK), pad=pad, device=dev)
            if first:
                sh = [(op.tokens.numel(), h, D) for h in (HQ, HK, HK)]
                x = [torch.zeros(s, device=dev, dtype=torch.float16) for s in sh]
                for _ in range(3):
                    op.fwd(*x)
                del x
                first = False
            recs.append(run_op(c, op, op.tokens, pass_, protocol, "ring", dev))
            del op
        except Exception as ex:  # noqa: BLE001  one failing input does not lose the launch
            recs.append(protocol["record"](c, "ring", pass_, protocol["launch"], status="error", reason=f"{type(ex).__name__}: {str(ex)[:300]}"))
        torch.cuda.empty_cache()
        dist.barrier()
    return recs
