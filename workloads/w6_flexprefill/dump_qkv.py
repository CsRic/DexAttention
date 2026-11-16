"""W6 operator-test data: the official single-device FlexPrefill prefill of Llama-3.1-8B-Instruct (every layer calls
flex_prefill_attention(q, k, v, gamma=0.95, tau=0.1, block_size=128, min_budget=1024, max_budget=None), the official config;
the model runs on two GPUs, layers 0-15 / 16-31, MLP in 8192-token chunks), dumping the post-RoPE q [N, 32, 128], k / v
[N, 8, 128] (bf16 as int16) and the block mask (packed bits [32, nb, ceil(nb / 8)]) of layers 0 / 8 / 16 / 24.
Inputs : $DEX_DATA_ROOT/w6_flexprefill/inputs/<input>.npy, $DEX_MODEL_ROOT/Llama-3.1-8B-Instruct
Outputs: $DEX_DATA_ROOT/w6_flexprefill/qkv/<input>/L<li>_{q,k,v,mask_g095}.npy
Run with two visible GPUs in an environment with transformers 4.44 and the official flex_prefill package.
usage: python dump_qkv.py --inputs ruler_qa_1_i0_32k,..."""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dex.common.env import cfg  # noqa: E402

BS, H, HKV, D = 128, 32, 8, 128
DUMP = (0, 8, 16, 24)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="comma list of input names")
    a = ap.parse_args()
    from transformers import LlamaForCausalLM
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    import flex_prefill.ops.flex_prefill_attention as FPA

    root = os.path.join(cfg("DEX_DATA_ROOT"), "w6_flexprefill")
    cap = {}
    orig = FPA.get_active_blocks

    def wrap(*args, **kw):
        r = orig(*args, **kw)
        cap["block_idx"] = r
        return r

    FPA.get_active_blocks = wrap
    dmap = {"model.embed_tokens": 0, "model.norm": 1, "lm_head": 1, "model.rotary_emb": 0}
    for i in range(32):
        dmap[f"model.layers.{i}"] = 0 if i < 16 else 1
    model = LlamaForCausalLM.from_pretrained(
        os.path.join(cfg("DEX_MODEL_ROOT"), "Llama-3.1-8B-Instruct"), torch_dtype=torch.bfloat16, device_map=dmap
    )
    model.eval()
    rot = model.model.rotary_emb

    def tri_mask(block_idx, nb):
        m = torch.zeros(H, nb * nb, dtype=torch.bool, device=block_idx[0][0].device)
        for h in range(H):
            m[h, block_idx[0][h].long()] = True
        return m.view(H, nb, nb) & torch.ones(nb, nb, dtype=torch.bool, device=m.device).tril()

    @torch.no_grad()
    def run(name):
        ids = torch.from_numpy(np.load(f"{root}/inputs/{name}.npy").astype(np.int64))[None]
        N = ids.shape[1]
        nb = (N + BS - 1) // BS
        h = model.model.embed_tokens(ids.to("cuda:0"))
        pos = torch.arange(N, device="cuda:0")[None]
        c0, s0 = rot(h, pos)
        cs = {0: (c0, s0), 1: (c0.to("cuda:1"), s0.to("cuda:1"))}
        t0 = time.time()
        od = f"{root}/qkv/{name}"
        os.makedirs(od, exist_ok=True)
        for li, layer in enumerate(model.model.layers):
            dev = 0 if li < 16 else 1
            h = h.to(f"cuda:{dev}")
            if li in (0, 16):  # the official code allocates on the current device and caches a causal mask
                torch.cuda.set_device(dev)
                FPA.causal_mask = None
            cos, sin = cs[dev]
            x = layer.input_layernorm(h)
            at = layer.self_attn
            q = at.q_proj(x).view(1, N, H, D).transpose(1, 2)
            k = at.k_proj(x).view(1, N, HKV, D).transpose(1, 2)
            v = at.v_proj(x).view(1, N, HKV, D).transpose(1, 2)
            del x
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            cap.clear()
            o = FPA.flex_prefill_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                gamma=0.95,
                tau=0.1,
                block_size=BS,
                min_budget=1024,
                max_budget=None,
            )
            o = o.reshape(1, N, H * D)
            if li in DUMP:
                for nm_, t_ in (("q", q), ("k", k), ("v", v)):
                    np.save(
                        f"{od}/L{li:02d}_{nm_}.npy", t_[0].transpose(0, 1).contiguous().view(torch.int16).cpu().numpy()
                    )
                np.save(
                    f"{od}/L{li:02d}_mask_g095.npy", np.packbits(tri_mask(cap["block_idx"], nb).cpu().numpy(), axis=-1)
                )
            h = h + at.o_proj(o)
            del o, q, k, v
            for s in range(0, N, 8192):
                hs = h[:, s : s + 8192]
                h[:, s : s + 8192] = hs + layer.mlp(layer.post_attention_layernorm(hs))
        print(f"{name}: N={N} nb={nb} time {time.time() - t0:.1f}s", flush=True)
        del h
        torch.cuda.empty_cache()

    for nm in a.inputs.split(","):
        run(nm)


if __name__ == "__main__":
    main()
