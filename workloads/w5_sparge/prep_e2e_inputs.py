"""Inputs of the W5 training step: the 32 generations (8 VBench prompts x seeds {0, 1} x {480p, 720p}) ->
  * latent x0: the generated video (81 frames) -> Wan2.1's VAE encoder (Diffusers AutoencoderKLWan, fp32, tiling) ->
    the posterior mode, normalised as the Wan pipeline expects ((z - latents_mean) / latents_std) -> [16, 21, H/8, W/8] fp32
  * text: the prompt -> umT5-XXL encoder (the pipeline's prompt embedding: max length 512, the embedding cut at the
    prompt's real length and zero padded to 512) -> [512, 4096] bf16
Inputs : $DEX_DATA_ROOT/w5_sparge/videos/<res>_p<i>_s<seed>.mp4 (generate.py), prompts.json,
         $DEX_MODEL_ROOT/Wan2.1-T2V-1.3B-Diffusers
Outputs: $DEX_DATA_ROOT/w5_sparge/e2e_inputs/<res>_p<i>_s<seed>.pt = dict(x0, text, prompt, seed, res, frames, video)
usage: python prep_e2e_inputs.py   (one GPU; needs diffusers, transformers, av)"""

import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from dex.common.env import cfg  # noqa: E402


def read_video(path):
    """every frame as RGB uint8 [F, H, W, 3] (PyAV)"""
    import av

    with av.open(path) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])


def main():
    root = os.path.join(cfg("DEX_DATA_ROOT"), "w5_sparge")
    vids, out_dir = os.path.join(root, "videos"), os.path.join(root, "e2e_inputs")
    model = os.path.join(cfg("DEX_MODEL_ROOT"), "Wan2.1-T2V-1.3B-Diffusers")
    os.makedirs(out_dir, exist_ok=True)
    man = json.load(open(os.path.join(HERE, "prompts.json")))
    dev = torch.device("cuda", 0)
    from diffusers import AutoencoderKLWan
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer, UMT5EncoderModel

    vae = AutoencoderKLWan.from_pretrained(model, subfolder="vae", torch_dtype=torch.float32).to(dev).eval()
    vae.enable_tiling()
    mean = torch.tensor(vae.config.latents_mean, device=dev).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=dev).view(1, -1, 1, 1, 1)
    tok = AutoTokenizer.from_pretrained(model, subfolder="tokenizer")
    te = UMT5EncoderModel.from_pretrained(model, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(dev).eval()
    for res in ("480p", "720p"):
        for i, p in enumerate(man["prompts"]):
            for s in man["seeds"]:
                tag = f"{res}_p{i}_s{s}"
                f = f"{out_dir}/{tag}.pt"
                if os.path.exists(f):
                    continue
                v = glob.glob(f"{vids}/{tag}.mp4")
                assert v, f"missing video {tag}"
                fr = read_video(v[0])
                assert fr.shape[0] == 81, fr.shape
                x = torch.from_numpy(fr).to(dev).permute(3, 0, 1, 2)[None].float() / 127.5 - 1.0  # [1, 3, F, H, W]
                with torch.no_grad():
                    z = vae.encode(x).latent_dist.mode()
                    z = ((z - mean) / std)[0].float().cpu()
                    ti = tok(
                        [prompt_clean(p["prompt"])],
                        padding="max_length",
                        max_length=512,
                        truncation=True,
                        add_special_tokens=True,
                        return_attention_mask=True,
                        return_tensors="pt",
                    )
                    n = int(ti.attention_mask.sum())
                    e = te(ti.input_ids.to(dev), ti.attention_mask.to(dev)).last_hidden_state[0].to(torch.bfloat16)
                    e = torch.cat([e[:n], e.new_zeros(512 - n, e.size(1))]).cpu()
                torch.save(
                    dict(
                        x0=z,
                        text=e,
                        prompt=p["prompt"],
                        seed=s,
                        res=res,
                        frames=int(fr.shape[0]),
                        video=os.path.basename(v[0]),
                    ),
                    f,
                )
                print(tag, tuple(z.shape), f"text tokens {n}", flush=True)


if __name__ == "__main__":
    main()
