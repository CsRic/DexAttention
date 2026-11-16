"""W6 correctness reference: the official single-device FlexPrefill prefill of Llama-3.1-8B-Instruct (transformers 4.44
LlamaForCausalLM, every layer = flex_prefill_attention with the official config; two GPUs, layers 0-15 / 16-31, MLP in
8192-token chunks). Saves per input the last-token logits (fp32), the final-norm hidden states of the sampled tokens (every
64th real token + the last; bf16 as int16) and their token ids.
Inputs : $DEX_DATA_ROOT/w6_flexprefill/inputs/<input>.npy, $DEX_MODEL_ROOT/Llama-3.1-8B-Instruct
Outputs: $DEX_DATA_ROOT/w6_flexprefill/official/<input>_{logits,hid,tok}.npy
usage: python official_reference.py --inputs a,b,c   (two visible GPUs)"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dex.common.env import cfg  # noqa: E402

H, HKV, D = 32, 8, 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    a = ap.parse_args()
    root = os.path.join(cfg("DEX_DATA_ROOT"), "w6_flexprefill")
    out = f"{root}/official"
    os.makedirs(out, exist_ok=True)
    from transformers import LlamaForCausalLM
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    import flex_prefill.ops.flex_prefill_attention as FPA

    dmap = {"model.embed_tokens": 0, "model.norm": 1, "lm_head": 1, "model.rotary_emb": 0}
    for i in range(32):
        dmap[f"model.layers.{i}"] = 0 if i < 16 else 1
    model = LlamaForCausalLM.from_pretrained(
        os.path.join(cfg("DEX_MODEL_ROOT"), "Llama-3.1-8B-Instruct"), torch_dtype=torch.bfloat16, device_map=dmap
    )
    model.eval()
    rot = model.model.rotary_emb

    @torch.no_grad()
    def run(name):
        ids = torch.from_numpy(np.load(f"{root}/inputs/{name}.npy").astype(np.int64))[None]
        N = ids.shape[1]
        h = model.model.embed_tokens(ids.to("cuda:0"))
        pos = torch.arange(N, device="cuda:0")[None]
        c0, s0 = rot(h, pos)
        cs = {0: (c0, s0), 1: (c0.to("cuda:1"), s0.to("cuda:1"))}
        t0 = time.time()
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
            o = FPA.flex_prefill_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                gamma=0.95,
                tau=0.1,
                block_size=128,
                min_budget=1024,
                max_budget=None,
            )
            o = o.reshape(1, N, H * D)
            h = h + at.o_proj(o)
            del o, q, k, v
            for s in range(0, N, 8192):
                hs = h[:, s : s + 8192]
                h[:, s : s + 8192] = hs + layer.mlp(layer.post_attention_layernorm(hs))
        hn = model.model.norm(h)
        logits = model.lm_head(hn[:, -1]).float()[0]
        tok = np.unique(np.r_[np.arange(0, N, 64), N - 1])
        hs = hn[0, torch.as_tensor(tok, device=hn.device)].contiguous().view(torch.int16).cpu().numpy()
        np.save(f"{out}/{name}_logits.npy", logits.cpu().numpy())
        np.save(f"{out}/{name}_hid.npy", hs)
        np.save(f"{out}/{name}_tok.npy", tok)
        print(f"{name}: N={N} argmax {int(logits.argmax())} time {time.time() - t0:.1f}s", flush=True)
        del h, hn
        torch.cuda.empty_cache()

    for nm in a.inputs.split(","):
        run(nm)


if __name__ == "__main__":
    main()
