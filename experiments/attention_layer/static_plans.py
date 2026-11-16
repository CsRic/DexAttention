"""Dex static plans for W1-W3 (one GPU, offline): owner placement, then head-split forward / backward plans.

Per input:
  1. placement: the workload's deployed equal-chunk placement (W1 contiguous, W2 / W3 zigzag), or the one given by
     --placement (auto = the candidate with the lower modelled forward + backward time on the unsplit problem);
  2. head split: the tile mask is replicated block-diagonally for hv = head_split virtual heads (capped at the KV heads;
     MHA 16/16 -> (4, 4) per copy, GQA 32/8 -> (8, 2)), every copy keeping the chosen placement;
  3. planner.plan on the virtual problem at the workload's block counts (forward W1/W2/W3 32 blocks, backward W1 128,
     W2/W3 32); pairs without work get -1.
Outputs (static_inputs.plan_dir): <id>.npz (virtual-head mask: mask, heads per copy, tokens hv * T, tok_valid / npad
tiled, virtual_heads, block_tokens), <id>_fwd.json and <id>_bwd.json (owner per block, GPU per block pair, block_tokens,
heads_virtual, placement). Existing files are kept.
Example: python experiments/attention_layer/static_plans.py --workloads w1 w2 w3 --head-split 4
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(os.path.dirname(HERE))]
import static_inputs as SI  # noqa: E402
from dex import planner as pl  # noqa: E402
from dex.planner.placement import CANDIDATES, bwd_time, fwd_time  # noqa: E402


def pass_problem(m, nb, heads, dev):
    """(W, kcols, B, (Qb, Ob, Kb), TT) of one pass at nb plan blocks for heads (hq, hk)"""
    B = m.shape[0] * SI.TILE // nb
    W, kc, _ = SI.block_problem(m, B, dev)
    return W, kc, B, pl.fwd_bytes(heads, B, SI.D), pl.FWD_MODEL["T_TILE"] * heads[0] / 32


def choose_placement(m, heads, nbs, dev):
    """the candidate with the lowest modelled forward + backward time on the unsplit problem; with equal forward and
    backward block counts this is planner.choose_placement"""
    (Wf, kf, Bf, byf, TTf), (Wb, kb, Bb, byb, TTb) = (pass_problem(m, nb, heads, dev) for nb in nbs)
    if nbs[0] == nbs[1]:
        best, *_ = pl.choose_placement(Wf, SI.GPN, SI.NN, *byf, TTf, heads, Bf, kf, pl.plan)
        return best
    # different block counts per pass (W1): the sum of the two modelled times over each candidate. The evaluated plans use
    # the deployed placement (--placement deployed).
    cost = {}
    for name, f in CANDIDATES.items():
        of = torch.as_tensor(f(Wf.shape[0], SI.P), device=dev)
        ob = torch.as_tensor(f(Wb.shape[0], SI.P), device=dev)
        sf = pl.plan(Wf, of, SI.GPN, SI.NN, *byf, TTf)
        sb = pl.plan(Wb, ob, SI.GPN, SI.NN, *byb, TTb, bwd=(heads, Bb, kb))
        cost[name] = (fwd_time(Wf, of, sf, SI.GPN, SI.NN, *byf, TTf) + bwd_time(Wb, ob, sb, SI.GPN, SI.NN, heads, Bb, kb))
    return min(cost, key=cost.get)


def write_virtual_mask(z, m, hv, case, path):
    out = dict(mask=np.kron(np.eye(hv, dtype=np.int8), m).astype(np.int8), heads=np.array(SI.virtual_heads(
        tuple(int(x) for x in z["heads"]), hv)[1]), tokens=np.array(hv * int(z["tokens"])), name=np.array(case["id"]),
        source=np.array(case["mask"]), virtual_heads=np.array(hv))
    if "block_tokens" in z.files:
        out["block_tokens"] = z["block_tokens"]
    for k in ("tok_valid", "npad"):
        if k in z.files:
            out[k] = np.tile(z[k], hv)
    np.savez(path, **out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workloads", nargs="+", default=["w1", "w2", "w3"])
    ap.add_argument("--head-split", type=int, default=4, help="virtual heads per input (4 in the evaluation)")
    ap.add_argument("--placement", default="deployed", choices=["deployed", "auto", "zigzag", "contiguous"],
                    help="deployed = W1 contiguous, W2 / W3 zigzag; auto = modelled choice on the unsplit problem")
    ap.add_argument("ids", nargs="*", help="only these input ids")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(SI.plan_dir(a.head_split), exist_ok=True)
    for w in a.workloads:
        nbs = SI.BLOCKS[w]
        for case in SI.inputs(w):
            if a.ids and case["id"] not in a.ids:
                continue
            fm, ff, fb = SI.plan_files(case, a.head_split)
            if all(os.path.exists(f) for f in (fm, ff, fb)):
                continue
            z = np.load(case["mask"])
            m = z["mask"].astype(np.int8)
            heads = tuple(int(x) for x in z["heads"])
            hv, hd = SI.virtual_heads(heads, a.head_split)
            place = {"deployed": SI.PLACEMENT[w], "auto": None}.get(a.placement, a.placement)
            place = place or choose_placement(m, heads, nbs, dev)
            if not os.path.exists(fm):
                write_virtual_mask(z, m, hv, case, fm)
            for tag, nb, f in (("fwd", nbs[0], ff), ("bwd", nbs[1], fb)):
                if os.path.exists(f):
                    continue
                W, kc, B, _, _ = pass_problem(m, nb, heads, dev)
                Wm, kcm = SI.bdiag(W, hv, 0.0), kc.repeat(hv)
                ow = torch.as_tensor(CANDIDATES[place](nb, SI.P), device=dev)
                owm = ow.repeat(hv)
                Qb, Ob, Kb = pl.fwd_bytes(hd, B, SI.D)
                TT = pl.FWD_MODEL["T_TILE"] * hd[0] / 32
                p = pl.plan(Wm, owm, SI.GPN, SI.NN, Qb, Ob, Kb, TT, bwd=(hd, B, kcm) if tag == "bwd" else None)
                live = Wm > 0
                json.dump(dict(family=case["id"], strategy=f"head_split{hv}_{tag}", cp=SI.P, nodes=SI.NN,
                               gpus_per_node=SI.GPN, n_blocks=int(Wm.shape[0]), block_tokens=B, owner=owm.tolist(),
                               schedule_table=torch.where(live, p, torch.full_like(p, -1)).tolist(), mask=fm,
                               heads_virtual=list(hd), head_split=hv, placement=place), open(f, "w"))
            print(f"{case['id']}: head split {hv} {hd}, placement {place}", flush=True)


if __name__ == "__main__":
    main()
