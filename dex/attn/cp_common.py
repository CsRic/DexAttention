"""Autograd entry of the context-parallel attention ops, shared by every system's training step; it does not load the
compiled extension, so it also serves baselines running in their own Python environments.
`cp_attention(q, k, v, op)` wraps an op's autograd-free fwd / bwd. With selective recomputation the recomputed forward of a
layer reuses the first forward's O / LSE (`RECOMPUTE`).
"""
import torch


class RecomputeCache:
    """Selective recomputation: the backward's recompute of a checkpointed layer keeps the attention's first-forward
    result (O / LSE, or a system's whole saved state) instead of running the attention again; the projections are
    recomputed as usual. The harness sets `key` to the layer id before the layer's attention and clears the cache at
    every micro-batch; every system's op stores and takes its state the same way."""

    def __init__(self):
        self.cache, self.key = {}, None

    def take(self):
        return self.cache.pop(self.key, None) if self.key is not None else None

    def put(self, value):
        if self.key is not None:
            self.cache[self.key] = value

    def clear(self):
        self.cache.clear()


RECOMPUTE = RecomputeCache()


class _CPFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, op):
        hit = RECOMPUTE.take()
        o, lse = hit if hit is not None else op.fwd(q, k, v)
        if hit is None:
            RECOMPUTE.put((o, lse))
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.op = op
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = ctx.op.bwd(q, k, v, o, lse, do.contiguous())
        return dq, dk, dv, None


def cp_attention(q, k, v, op):
    """q [Tl, Hq, D], k / v [Tl, Hk, D] fp16 in op's local order -> o [Tl, Hq, D]; op provides fwd / bwd"""
    return _CPFn.apply(q.contiguous(), k.contiguous(), v.contiguous(), op)
