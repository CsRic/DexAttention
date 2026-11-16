"""Training-step reuse shared by every W4 attention op (Dex and the baselines alike).

With activation checkpointing a layer's forward runs twice: once in the forward pass and once as the recompute inside
the backward. The recompute reuses what the first forward computed instead of computing it again:
  * the top-k selection (same inputs -> same selection) and, for Dex, the forward plan (PLAN_CACHE);
  * the attention's first-forward result O / LSE / selection (OL_CACHE, selective recompute): only the q / k / v
    projections are recomputed.
Both caches are keyed by layer index and cleared at the start of every micro-batch. Dex additionally searches each
layer's backward plan on a worker thread right after the layer's first forward (pool()), so it is ready before the
backward reaches that layer.

    reuse.set_layer(i); o = CPAttn.apply(q, k, v, op)       # op.fwd(q, k, v) -> (o, saved); op.bwd(saved, do, aux)
"""
import torch

from workloads.w4_infllm.common import to_rows

PLAN_CACHE, OL_CACHE = {}, {}
_CUR = [None]
_POOL = []
_SYNC = []


def set_layer(i):
    _CUR[0] = i


def cur_layer():
    return _CUR[0]


def new_microbatch():
    PLAN_CACHE.clear(); OL_CACHE.clear()


def cache_get():
    return PLAN_CACHE.get(_CUR[0])


def cache_put(v):
    PLAN_CACHE[_CUR[0]] = v


def pool():
    """one worker thread (bound to this rank's GPU) for the backward plans searched ahead"""
    if not _POOL:
        from concurrent.futures import ThreadPoolExecutor
        dev_ = torch.cuda.current_device()
        _POOL.append(ThreadPoolExecutor(max_workers=1, initializer=lambda: torch.cuda.set_device(dev_)))
    return _POOL[0]


def order_ranks(dev):
    """a one-element all-reduce on the compute stream: orders every rank's recompute of a layer before that layer's
    backward call (the backward clears signal / accumulator words of the shared symmetric region at its entry), then
    the host waits for the stream as an inline plan's device-to-host copy would"""
    import torch.distributed as dist
    if not _SYNC:
        _SYNC.append(torch.zeros(1, device=dev))
    dist.all_reduce(_SYNC[0])
    torch.cuda.current_stream().synchronize()


class CPAttn(torch.autograd.Function):
    """one context-parallel attention op (any system) as an autograd function; the saved tensors go through
    save_for_backward, so activation checkpointing drops them and the recompute refills them from OL_CACHE"""

    @staticmethod
    def forward(ctx, q, k, v, op):
        hit = OL_CACHE.pop(_CUR[0], None)
        if hit is not None:                                            # the recompute: the first forward's O / LSE / selection
            o, tail, aux = hit                                         # (saved = q rows, k, v | o rows, lse, selection)
            saved = (to_rows(q), k.contiguous(), v.contiguous()) + tail
        else:
            o, saved = op.fwd(q, k, v)
            aux = getattr(op, "aux", None)                             # (this layer's reuse cache: the backward plan ahead)
            OL_CACHE[_CUR[0]] = (o, tuple(saved[3:]), aux)
        ctx.op, ctx.aux = op, aux
        ctx.save_for_backward(*saved)
        return o

    @staticmethod
    def backward(ctx, do):
        dq, dk, dv = ctx.op.bwd(ctx.saved_tensors, do.contiguous(), ctx.aux)
        return dq, dk, dv, None
