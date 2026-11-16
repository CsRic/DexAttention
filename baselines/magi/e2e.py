"""Magi's op for W2 / W3 training, the configuration of its operator tests: exact token-level slices over the real tokens, Magi's
dispatch (the model's token layout per rank = dispatch of the token ids), solver, overlap degree 1, staged all_to_all, fp32 backward
partial reduce. With selective recomputation Magi's own DistAttnFunc forward / backward are called with a proxy ctx: the first
forward stores its O / LSE under the layer; the recompute rebuilds Magi's local q / kv from the recomputed q / k / v with Magi's own
reshape helpers (no comm, no attention) and re-saves the state. PyTorch's allocator cache is cleared after every attention forward and
backward: Magi's dispatch buffers change size from layer to layer, so the caching allocator accumulates freed blocks it cannot reuse;
DEX_MAGI_CLEAR_CACHE=0 keeps them, the setting experiments/memory/magi_cache.sh measures. Tensors: q [Tl, Hq, D], k / v [Tl, Hk, D] fp16."""
import os

import numpy as np
import torch
import torch.distributed as dist

import baselines.magi  # noqa: F401  (kernel backend defaults)
from baselines.magi import staged_a2a
from baselines.magi.slices import slices_from_segments
from dex.attn.cp_common import RECOMPUTE

D = 128
CLEAR_CACHE = os.environ.get("DEX_MAGI_CLEAR_CACHE", "1") != "0"
_INSTALLED = [False]


class _Ctx:                                       # stands in for an autograd ctx inside Magi's Function
    def save_for_backward(self, *t):
        self.saved_tensors = t


class _MagiFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, op):
        from magi_attention.functional.dist_attn import DistAttnFunc
        rt, pc = op.rt, _Ctx()
        hit = RECOMPUTE.take()
        if hit is None:
            out, _, _ = DistAttnFunc.forward(pc, q, k, v, None, rt, None, 0.0, False)
            _, _, o_, lse_, lq, lkv, _ = rt.load_tensors_from_fwd(pc)
            RECOMPUTE.put((o_, lse_, lq, lkv, pc.softmax_scale, pc.softcap))
        else:
            o_, lse_, lq, lkv, sc, cap = hit
            q_, kv_ = rt._maybe_flatten_local_qkv_head_groups(local_q=q, local_kv=(k, v))
            kv_ = rt._maybe_concat(*kv_, need_concat=rt.concat_kv)
            rt.save_tensors_for_bwd(pc, local_q=q_, local_kv=kv_, local_out=o_, local_lse=lse_, last_stage_q=lq, last_stage_kv=lkv, global_sink=None)
            pc.dist_attn_runtime, pc.softmax_scale, pc.softcap = rt, sc, cap
            out = o_
        # Magi's saved tensors go through ctx.save_for_backward, so the checkpoint's hooks can release them between passes
        ctx.save_for_backward(*pc.saved_tensors)
        del pc.saved_tensors
        ctx.pc = pc
        if CLEAR_CACHE:
            torch.cuda.empty_cache()
        return out

    @staticmethod
    def backward(ctx, do):
        from magi_attention.functional.dist_attn import DistAttnFunc
        ctx.pc.saved_tensors = ctx.saved_tensors
        g = DistAttnFunc.backward(ctx.pc, do.contiguous(), None, None)
        del ctx.pc.saved_tensors
        if CLEAR_CACHE:
            torch.cuda.empty_cache()
        return g[0], g[1], g[2], None


class MagiCP:
    def __init__(self, z, heads, device=None):
        if not _INSTALLED[0]:
            staged_a2a.install_event()
            _INSTALLED[0] = True
        from magi_attention.api import dispatch, magi_attn_flex_key
        from magi_attention.api.magi_attn_interface import dist_attn_runtime_dict_mgr
        from magi_attention.common import AttnRanges
        from magi_attention.common.enum import AttnMaskType
        TY = dict(full=AttnMaskType.FULL, causal=AttnMaskType.CAUSAL, bicausal=AttnMaskType.BICAUSAL, invcausal=AttnMaskType.INVCAUSAL)
        self.dev = device or torch.device("cuda")
        self.hq, self.hk = heads
        T, qr, kr, ty = slices_from_segments(z["segs"])
        self.key = magi_attn_flex_key(q_ranges=AttnRanges.from_ranges(qr), k_ranges=AttnRanges.from_ranges(kr), attn_mask_type=[TY[t] for t in ty],
                                      total_seqlen_q=T, total_seqlen_k=T, num_heads_q=self.hq, num_heads_kv=self.hk, head_dim=D, pad_size=0,
                                      cp_group_or_mesh=dist.group.WORLD)
        ids = dispatch(torch.arange(1, T + 1, device=self.dev, dtype=torch.float32)[:, None], key=self.key)[:, 0]
        rid = ids.round().long() - 1                  # real-token id per local row (-1: Magi pad)
        tv = z["tok_valid"].astype(bool) if "tok_valid" in z.files else np.ones(int(z["tokens"]), bool)
        pos = torch.as_tensor(np.flatnonzero(tv), device=self.dev)
        assert len(pos) == T, (len(pos), T)
        self.tokens = torch.where(rid >= 0, pos[rid.clamp(min=0)], torch.full_like(rid, -1))
        self.rt = dist_attn_runtime_dict_mgr.get(self.key).dist_attn_runtime

    def __call__(self, q, k, v):
        return _MagiFn.apply(q.contiguous(), k.contiguous(), v.contiguous(), self)


def make_e2e_op(d):
    """d: id, mask_path, mask, heads, pad, fwd_plan, bwd_plan, workload"""
    return MagiCP(np.load(d["mask_path"]), d["heads"])
