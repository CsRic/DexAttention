"""Records the q / k of selected self-attention calls of a Wan generation run by SGLang Diffusion's SpargeAttn backend
(installed in every worker process by dump_site/sitecustomize.py when W5_DUMP_DIR is set).

Wraps spas_sage_attn.core.get_block_map_meansim_fuse_quant, the function that turns (q, k) into the block map the
kernel runs (128-row query blocks x 64-key blocks), and returns its result unchanged. For every call of a selected
denoising step it saves the map the kernel used (bit-packed, "topk"); for the selected layers it also saves this rank's
q / k (fp16). Environment (set by generate.py):
  W5_DUMP_DIR             output directory
  W5_DUMP_CALLS_PER_STEP  SpargeAttn calls per denoising step (layers x CFG passes)
  W5_DUMP_STEPS           comma list of denoising steps to keep
  W5_DUMP_QK              comma list of call_in_step indices whose q / k are saved (call order per step = conditional
                          layers 0..L-1, then unconditional)
File: <dir>/rank<r>/call<i>.npz with step, call_in_step, shape, seqlen, the packed map and its density per head."""

import os

import numpy as np
import torch

_state = {"n": 0}


def install():
    import spas_sage_attn.core as core

    orig = core.get_block_map_meansim_fuse_quant
    out_dir = os.environ["W5_DUMP_DIR"]
    per_step = int(os.environ["W5_DUMP_CALLS_PER_STEP"])
    steps = {int(s) for s in os.environ["W5_DUMP_STEPS"].split(",")}
    qk_calls = {int(x) for x in os.environ.get("W5_DUMP_QK", "").split(",") if x}

    def hooked(q, k, km=None, *args, **kwargs):
        res = orig(q, k, km, *args, **kwargs)
        i = _state["n"]
        _state["n"] += 1
        step, j = i // per_step, i % per_step
        if step not in steps:
            return res
        kw = dict(kwargs)
        kw["return_lut"] = False
        m = orig(q, k, km, *args, **kw)[0].bool()
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        d = os.path.join(out_dir, f"rank{rank}")
        os.makedirs(d, exist_ok=True)
        arrs = dict(
            topk=np.packbits(m.cpu().numpy(), axis=-1), topk_density=m.float().mean(dim=(0, 2, 3)).cpu().numpy()
        )
        if j in qk_calls:
            arrs["q"] = q.detach().to(torch.float16).cpu().numpy()
            arrs["k"] = k.detach().to(torch.float16).cpu().numpy()
        np.savez(
            os.path.join(d, f"call{i:05d}.npz"),
            step=step,
            call_in_step=j,
            shape=np.array(m.shape),
            seqlen=q.shape[-2],
            blkq=kwargs.get("BLKQ", 128),
            blkk=kwargs.get("BLKK", 64),
            **arrs,
        )
        return res

    core.get_block_map_meansim_fuse_quant = hooked
