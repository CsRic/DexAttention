"""Llama-3.1-8B-Instruct forward pieces with transformers 4.44's numerics: LlamaRMSNorm (fp32 variance, cast back, weight x),
llama3 RoPE (inv_freq on the CPU in fp32, cos / sin from the fp32 product, cast to bf16), q * cos + rotate_half(q) * sin,
SiLU MLP. Weights: bf16, replicated on every GPU."""

import json
import math
import os

import torch
import torch.nn.functional as F

from dex.common.env import cfg

NLAYER, HID, EPS, HQ, HK, D = 32, 4096, 1e-5, 32, 8, 128
VOCAB = 128256


class Llama:
    def __init__(self, dev):
        path = os.path.join(cfg("DEX_MODEL_ROOT"), "Llama-3.1-8B-Instruct")
        idx = json.load(open(f"{path}/model.safetensors.index.json"))["weight_map"]
        from safetensors import safe_open

        self.w = {}
        for f in sorted(set(idx.values())):
            with safe_open(f"{path}/{f}", framework="pt", device=str(dev)) as fh:
                for k in fh.keys():
                    self.w[k] = fh.get_tensor(k)
        assert len(self.w) == 291 and all(t.dtype == torch.bfloat16 for t in self.w.values())
        c = json.load(open(f"{path}/config.json"))
        rs = c["rope_scaling"]
        assert c["rope_theta"] == 500000.0 and rs["rope_type"] == "llama3" and c["rms_norm_eps"] == EPS
        base, dim = c["rope_theta"], D
        inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        factor, lf, hf, orig = (
            rs["factor"],
            rs["low_freq_factor"],
            rs["high_freq_factor"],
            rs["original_max_position_embeddings"],
        )
        low_w, high_w = orig / lf, orig / hf
        wl = 2 * math.pi / inv
        il = torch.where(wl > low_w, inv / factor, inv)
        sm = (orig / wl - lf) / (hf - lf)
        smoothed = (1 - sm) * il / factor + sm * il
        med = ~(wl < high_w) * ~(wl > low_w)
        self.inv_freq = torch.where(med, smoothed, il).to(dev)

    def rope(self, pos):
        inv = self.inv_freq[None, :, None].float().expand(1, -1, 1)
        freqs = (inv @ pos[None, None, :].float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(torch.bfloat16)[0][:, None], emb.sin().to(torch.bfloat16)[0][:, None]  # [L, 1, D]

    @staticmethod
    def rms(x, w):
        xf = x.to(torch.float32)
        var = xf.pow(2).mean(-1, keepdim=True)
        xf = xf * torch.rsqrt(var + EPS)
        return w * xf.to(x.dtype)

    @staticmethod
    def _rot(x):
        x1, x2 = x[..., : D // 2], x[..., D // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def qkv(self, li, h, cos, sin, padidx):
        """input RMSNorm, projections, RoPE; padded tokens get zero q / k / v"""
        p = f"model.layers.{li}."
        L = h.shape[0]
        x = self.rms(h, self.w[p + "input_layernorm.weight"])
        q = F.linear(x, self.w[p + "self_attn.q_proj.weight"]).view(L, HQ, D)
        k = F.linear(x, self.w[p + "self_attn.k_proj.weight"]).view(L, HK, D)
        v = F.linear(x, self.w[p + "self_attn.v_proj.weight"]).view(L, HK, D)
        q = q * cos + self._rot(q) * sin
        k = k * cos + self._rot(k) * sin
        if padidx is not None:
            q[padidx] = 0
            k[padidx] = 0
            v[padidx] = 0
        return q, k, v

    def post(self, li, h, o):
        """o_proj + residual, post RMSNorm, MLP + residual"""
        p = f"model.layers.{li}."
        h = h + F.linear(o, self.w[p + "self_attn.o_proj.weight"])
        x = self.rms(h, self.w[p + "post_attention_layernorm.weight"])
        return h + F.linear(
            F.silu(F.linear(x, self.w[p + "mlp.gate_proj.weight"])) * F.linear(x, self.w[p + "mlp.up_proj.weight"]),
            self.w[p + "mlp.down_proj.weight"],
        )
