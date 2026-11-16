"""Wan2.1-T2V-1.3B model harness of the W5 training step (Diffusers WanTransformer3DModel, bf16).

Every operation except self-attention is token-local, so each rank runs the transformer on its own tokens (the layout's
`tok`) and the self-attention goes through a context-parallel op (CPAttn) with any W5 attention system. CPAttn casts
q / k / v to fp16 [H, L, 128] (the block-sparse kernels are fp16) and returns bf16. Its backward scales dO by a power of
two 2^k that brings the global max |dO| into [2^-4, 2^-3) (the loss is a mean over ~2M latent elements, so most of dO
would underflow in fp16) and divides the gradients by 2^k in fp32 (gradients are linear in dO).
Selective recomputation: the backward's recomputation of a block keeps the attention's first-forward O / LSE and mask
(q / k / v projections are recomputed, the attention is not run again); every system the same.
Timing marks (CUDA events): attention forward (incl. recomputation), attention backward, and before each attention
backward a deliberate all-rank synchronisation whose wait is reported separately."""

import glob

import torch
import torch.distributed as dist

from .common import Reuse

T_STEP, SIGMA = 500, 0.5  # constant timestep t = 500 of 1000, sigma = t / 1000
DO_TARGET_LOG2 = -4  # loss scaling: global max |dO| -> [2^-4, 2^-3)
EV = {"f0": [], "f1": [], "s0": [], "b0": [], "b1q": [], "b1k": [], "b1v": []}
OL_CACHE = {}
_SYNC_BUF = []


class _Mark(torch.autograd.Function):
    """identity; its backward records a CUDA event into EV[tag]. Before the attention backward ("b0"): the all-rank
    synchronisation of every system (stream-ordered), its wait reported as the phase sync"""

    @staticmethod
    def forward(ctx, x, tag):
        ctx.tag = tag
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        if ctx.tag == "b0":
            s = torch.cuda.Event(enable_timing=True)
            s.record()
            EV["s0"].append(s)
            if not _SYNC_BUF:
                _SYNC_BUF.append(torch.zeros(1, device=g.device))
            dist.all_reduce(_SYNC_BUF[0])
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        EV[ctx.tag].append(e)
        return g, None


def ev_ms(a, b):
    return sum(x.elapsed_time(y) for x, y in zip(EV[a], EV[b]))


def bwd_ms():
    return sum(min(s_.elapsed_time(e) for e in ends) for s_, *ends in zip(EV["b0"], EV["b1q"], EV["b1k"], EV["b1v"]))


class CPAttn(torch.autograd.Function):
    """the context-parallel attention op on bf16 [H, L, D] q / k / v"""

    @staticmethod
    def forward(ctx, q, k, v, sysobj):
        q, k, v = (x.to(torch.float16).contiguous() for x in (q, k, v))
        hit = OL_CACHE.pop(Reuse.layer[0], None)
        if hit is not None:  # the recomputation: the first forward's result
            o, fm, lse = hit
        else:
            o, (fm, lse) = sysobj.fwd(q, k, v)
            OL_CACHE[Reuse.layer[0]] = (o, fm, lse)
        ctx.sysobj, ctx.fm = sysobj, fm
        ctx.cache = Reuse.caches.get(Reuse.layer[0]) if Reuse.enabled else None
        ctx.save_for_backward(q, k, v, o, lse)
        return o.to(torch.bfloat16)

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        m_ = do.float().abs().amax().reshape(1)
        dist.all_reduce(m_, op=dist.ReduceOp.MAX)
        sc_ = torch.exp2(DO_TARGET_LOG2 - torch.floor(torch.log2(m_.clamp(min=1e-30))))
        do = (do.float() * sc_).to(torch.float16).contiguous()
        dq, dk, dv = ctx.sysobj.bwd(q, k, v, o, do, ctx.fm, lse, ctx.cache)
        return (
            (dq.float() / sc_).to(torch.bfloat16),
            (dk.float() / sc_).to(torch.bfloat16),
            (dv.float() / sc_).to(torch.bfloat16),
            None,
        )


class CPAttnProcessor:
    """Diffusers WanAttnProcessor for the self-attention with CPAttn in place of SDPA (q / k / v projections, the
    across-heads RMSNorm and the RoPE of the local tokens exactly as the processor computes them)"""

    def __init__(self, sysobj, layer):
        self.sysobj, self.layer = sysobj, layer

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None):
        assert encoder_hidden_states is None
        query, key, value = attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)
        query, key = attn.norm_q(query), attn.norm_k(key)
        query, key, value = (x.unflatten(2, (attn.heads, -1)) for x in (query, key, value))

        def rope(h, cos, sin):
            x1, x2 = h.unflatten(-1, (-1, 2)).unbind(-1)
            c, s = cos[..., 0::2], sin[..., 1::2]
            out = torch.empty_like(h)
            out[..., 0::2] = x1 * c - x2 * s
            out[..., 1::2] = x1 * s + x2 * c
            return out.type_as(h)

        query, key = rope(query, *rotary_emb), rope(key, *rotary_emb)
        q, k, v = (x[0].transpose(0, 1).contiguous() for x in (query, key, value))  # [H, L, D] bf16
        q, k, v = _Mark.apply(q, "b1q"), _Mark.apply(k, "b1k"), _Mark.apply(v, "b1v")
        Reuse.layer[0] = self.layer
        e0 = torch.cuda.Event(enable_timing=True)
        e0.record()
        EV["f0"].append(e0)
        o = CPAttn.apply(q, k, v, self.sysobj)
        e1 = torch.cuda.Event(enable_timing=True)
        e1.record()
        EV["f1"].append(e1)
        o = _Mark.apply(o, "b0")
        o = o.transpose(0, 1)[None].flatten(2, 3).to(query.dtype)
        o = attn.to_out[0](o)
        return attn.to_out[1](o)


def forward_tokens(model, xt, text, tok, tstep, ckpt=None):
    """WanTransformer3DModel.forward for this rank's tokens `tok` (positions in the patch grid): patch embedding, RoPE,
    condition embedding, the blocks (self-attention through the CP processor), output norm / projection.
    -> [len(tok), 64] in proj_out's per-token order (p_t, p_h, p_w, c)"""
    dt = model.proj_out.weight.dtype
    xt = xt[None].to(dt)
    cos, sin = model.rope(xt)
    rot = (cos[:, tok], sin[:, tok])
    hs = model.patch_embedding(xt).flatten(2).transpose(1, 2)[:, tok]
    temb, tproj, enc, _ = model.condition_embedder(tstep, text[None].to(dt), None)
    tproj = tproj.unflatten(1, (6, -1))
    for blk in model.blocks:
        hs = ckpt(blk, hs, enc, tproj, rot, use_reentrant=False) if ckpt is not None else blk(hs, enc, tproj, rot)
    shift, scale = (model.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)
    hs = (model.norm_out(hs.float()) * (1 + scale) + shift).type_as(hs)
    return model.proj_out(hs)[0]


def patchify_target(tgt, patch):
    """[C, F, H, W] -> [S, pt * ph * pw * C] in proj_out's per-token order"""
    C_, Fl, Hl, Wl = tgt.shape
    pt, ph, pw = patch
    return (
        tgt.reshape(C_, Fl // pt, pt, Hl // ph, ph, Wl // pw, pw)
        .permute(1, 3, 5, 2, 4, 6, 0)
        .reshape((Fl // pt) * (Hl // ph) * (Wl // pw), -1)
    )


def load_model(path, dev):
    """Diffusers' bf16 model without `accelerate`: load fp32, cast every parameter / buffer to bf16 except the modules
    the model keeps in fp32 (_keep_in_fp32_modules)"""
    from diffusers import WanTransformer3DModel
    from safetensors.torch import load_file

    model = WanTransformer3DModel.from_config(WanTransformer3DModel.load_config(path, subfolder="transformer"))
    sd = {}
    for f in sorted(glob.glob(f"{path}/transformer/*.safetensors")):
        sd.update(load_file(f))
    missing, unexpected = model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
    assert not unexpected and all("rope" in k for k in missing), (missing[:5], unexpected[:5])
    del sd
    keep = model._keep_in_fp32_modules
    for name, mod in model.named_modules():
        if any(k in name.split(".") for k in keep):
            continue
        for pn, p in list(mod.named_parameters(recurse=False)) + list(mod.named_buffers(recurse=False)):
            if p.is_floating_point() and not any(k == pn for k in keep):
                p.data = p.data.to(torch.bfloat16)
    model.train()
    return model.to(dev)
