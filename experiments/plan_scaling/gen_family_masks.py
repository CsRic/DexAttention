"""Real per-input tile masks of the rollout and top-k / top-p families of plan_scaling.py.
  rollout  the groups of rollout_groups.jsonl (stars and trees), laid out as the W2 / W3 masks: prompt, then the replies in
           pre-order, every segment padded to 128 tokens, inert tail up to a multiple of 32 tiles.
  w4       InfLLM-v2: tile (i, j) live iff a token of 128-token tile i selects a 64-key block in tile j under some KV head
           (causal filter applied; diagonal tile kind 2, tail up to a multiple of 32 tiles); one file per line of
           w4_masks.txt, the mask shared across KV heads.
  w5       SpargeAttention2 selection recomputed from the stored q / k of the 42 cases, all 12 heads; a 128 x 128 tile is live
           if either 64-key half is selected; non-causal.
  w6       FlexPrefill masks of the 36 cases, per KV-head unit (8 units of 4 query heads) the union over its heads; causal.
Inputs : rollout_groups.jsonl, w4_masks.txt; recorded selections and dumps under
         $DEX_DATA_ROOT/{w4_infllm/masks,w5_sparge/dumps,w6_flexprefill/qkv}
Outputs: $DEX_DATA_ROOT/plan_scaling/masks/{rollout/<id>.npz, w4_<sequence>_L<layer>.npz, masks_w5.npz, masks_w6.npz}
Example: python experiments/plan_scaling/gen_family_masks.py
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments", "comm_volume"))
import common as C  # noqa: E402
import family_lists as FL  # noqa: E402
from workloads.w2_w3_rollout import layout as L  # noqa: E402
from workloads.w5_sparge.cases import load_case, read_cases  # noqa: E402
from workloads.w5_sparge.selection import pooled_k, pooled_q, select  # noqa: E402

PLAN_BLOCK_TILES = 32   # the tail of every packed mask reaches a whole number of plan blocks


def group_segments(g):
    """rollout group -> segments [start, len, kind, group, parent, real_len] and total tokens; node j's segment is 1 + j"""
    n = L.up(g["P"], L.TILE)
    segs, pos = [[0, n, 0, 0, -1, g["P"]]], n
    for r, parent in g["nodes"]:
        n = L.up(r, L.TILE)
        segs.append([pos, n, 1, 0, 0 if parent < 0 else 1 + parent, r])
        pos += n
    T = L.up(pos, L.TILE * PLAN_BLOCK_TILES)
    if T > pos:
        segs.append([pos, T - pos, 2, -1, -1, 0])
    return segs, T


def rollout(out):
    groups = FL.rollout_groups()
    os.makedirs(os.path.join(out, "rollout"), exist_ok=True)
    for g in groups:
        segs, T = group_segments(g)
        np.savez_compressed(FL.rollout_mask_file(out, g["id"]), mask=L.tile_mask(segs, T).astype(np.int8), tokens=T)
    print("rollout", len(groups), flush=True)


def w4(out):
    for seq, lay in FL.w4_cases():
        tk = np.load(os.path.join(C.w4_selection_dir(), seq, f"L{lay:02d}_topk.npy"), mmap_mode="r")
        S = tk.shape[1]
        n = L.up(L.up(S, L.TILE) // L.TILE, PLAN_BLOCK_TILES)
        m = np.zeros((n, n), np.int8)
        q = np.repeat(np.arange(S), tk.shape[2])
        for h in range(tk.shape[0]):
            b = np.asarray(tk[h]).astype(np.int64).ravel()
            ok = (b >= 0) & (b <= q // 64)
            m[q[ok] // 128, b[ok] // 2] = 1
        m[np.arange(n), np.arange(n)] = 2
        np.savez_compressed(FL.w4_mask_file(out, seq, lay), mask=m, heads=np.array([32, 2]), tokens=n * 128)
        print("w4", seq, lay, n, flush=True)


def w5(out):
    res = {}
    for g, call in read_cases():
        q, k, _meta = load_case(g, call)[:3]
        sel = select(pooled_q(torch.from_numpy(q)), pooled_k(torch.from_numpy(k)), 128 ** -0.5).numpy()
        nq, nk = sel.shape[1], sel.shape[2]
        pad = np.zeros((12, nq, 2 * nq), bool); pad[:, :, :nk] = sel
        res[f"{g}_c{call}"] = pad.reshape(12, nq, nq, 2).any(-1)
        print("w5", g, call, nq, flush=True)
    np.savez_compressed(os.path.join(out, "masks_w5.npz"), **{k: np.packbits(v, axis=-1) for k, v in res.items()},
                        _n=np.array([v.shape[1] for v in res.values()]), _names=np.array(list(res)))


def w6(out):
    res = {}
    for inp, li in C.w6_cases():
        m = np.load(C.w6_dump(inp, li, "mask")); n = m.shape[1]
        b = np.unpackbits(m, axis=-1)[:, :, :n].astype(bool)
        res[f"{inp}_L{li}"] = b.reshape(8, 4, n, n).any(1)
    np.savez_compressed(os.path.join(out, "masks_w6.npz"), **{k: np.packbits(v, axis=-1) for k, v in res.items()},
                        _n=np.array([v.shape[1] for v in res.values()]),
                        _names=np.array([k.replace("_L", ":") for k in res]))
    print("w6", len(res), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", nargs="+", default=["rollout", "w4", "w5", "w6"]); a = ap.parse_args()
    out = os.path.join(C.data_root(), "plan_scaling", "masks")
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(16)
    for w in a.only:
        {"rollout": rollout, "w4": w4, "w5": w5, "w6": w6}[w](out)


if __name__ == "__main__":
    main()
