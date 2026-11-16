"""W2 / W3 masks of the selected rollout groups (selection.json from select_groups).

Per group and plan block count nb: the packed layout (layout.layout: prompt, then the 16 replies, every segment padded
to 128 tokens, inert tail up to a multiple of 128 * nb tokens, so every plan block holds T / nb tokens and every GPU owns
nb / 16 blocks) and its tile mask. Padding rule: pad tokens carry zero K / V, are excluded from every output and error
metric, and the forward output of each real row is corrected exactly for the pad keys it sees (`npad`).

Outputs ($DEX_DATA_ROOT/masks/<w2|w3>/):
  <id>_nb<nb>.npz   mask [nt, nt] int8 (0 / 1 full / 2 causal diagonal), heads [2], tokens, block_tokens, tok_valid [T],
                    npad [T] int32, segs [n, 6], S_valid, rho, name, source
  <id>_ranges.npy   per-token key ranges [S, 2, 2] on the UNPADDED sequence (FP64 reference; baselines that take
                    token ranges)
W3 = Qwen3-4B heads (32 / 8), W2 = Qwen1.5-1.8B heads (16 / 16); the masks are otherwise identical.
Example: python -m workloads.w2_w3_rollout.gen_masks --nbs 16 32 64 128
"""
import argparse
import json
import os

import numpy as np

from dex.common.env import cfg
from workloads.w2_w3_rollout import layout as L

HEADS = {"w3": (32, 8), "w2": (16, 16)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nbs", type=int, nargs="+", default=[16, 32, 64, 128],
                    help="plan block counts to write (Dex plans use 32; other systems choose their own)")
    ap.add_argument("--workloads", nargs="+", default=["w3", "w2"])
    a = ap.parse_args()
    root = cfg("DEX_DATA_ROOT")
    sel = json.load(open(os.path.join(root, "rollouts", "selection.json")))
    for w in a.workloads:
        out = os.path.join(root, "masks", w)
        os.makedirs(out, exist_ok=True)
        hq, hk = HEADS[w]
        for c in sel:
            name = f"{w}_{c['split']}_p{c['idx']}"
            grp = dict(P=c["P"], R=c["R"])
            for nb in a.nbs:
                segs, T = L.layout([grp], L.TILE * nb)
                m = L.tile_mask(segs, T).astype(np.int8)
                valid = L.token_valid(segs, T)
                npad = L.pad_keys_seen(segs, T)
                np.savez(os.path.join(out, f"{name}_nb{nb}.npz"), mask=m, heads=np.array([hq, hk]), tokens=T,
                         block_tokens=T // nb, tok_valid=valid, npad=npad, segs=np.array(segs, np.int64),
                         S_valid=int(valid.sum()), rho=c["rho"], name=f"{name}_nb{nb}",
                         source=f"{c['file']}#{c['idx']}")
            np.save(os.path.join(out, f"{name}_ranges.npy"), L.star_ranges(c["P"], c["R"]))
            print(f"{name}: S_valid {c['S']}, rho {c['rho']:.2f}, heads {hq}/{hk}", flush=True)


if __name__ == "__main__":
    main()
