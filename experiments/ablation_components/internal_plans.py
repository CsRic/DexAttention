"""Plans of the data-reuse, relay and transfer-order study: head splitting off, zigzag owner placement for every arm, 16 GPUs as
4 nodes x 4 GPUs. Nine static inputs: W1 64K / 128K / 256K and, for W2 and W3, the rollouts p40 (32K), p193 (64K), p722 (128K).
Plan block counts as in static_inputs.BLOCKS (forward, backward): W1 (32, 128), W2 / W3 (32, 32). Arms written here:
  full       Dex's assigner on the unsplit problem (dex.planner.plan)
  load_only  load alone, shared data ignored: whole query block rows, largest work first, each to the least-loaded GPU
The relay and transfer-order arms run the full plans under a scheduler switch (internal_run.py).
Inputs : masks of workloads/w1_lambda and workloads/w2_w3_rollout under $DEX_DATA_ROOT.
Outputs: $DEX_DATA_ROOT/plans/ablation/internal/<id>_<arm>_{fwd,bwd}.json; existing files are kept.
Example: python experiments/ablation_components/internal_plans.py --arms full,load_only
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "experiments", "attention_layer")]
import static_inputs as SI  # noqa: E402
from dex import planner as pl  # noqa: E402
from dex.common.env import cfg  # noqa: E402

ROLLOUTS = ("32k_p40", "64k_p193", "128k_p722")
W1_LENGTHS = (65536, 131072, 262144)


def plan_dir():
    return os.path.join(cfg("DEX_DATA_ROOT"), "plans", "ablation", "internal")


def plan_file(case_id, arm, pass_):
    return os.path.join(plan_dir(), f"{case_id}_{arm}_{pass_}.json")


def cases():
    """the nine inputs in canonical order"""
    ids = {f"{w}_{x}" for w in ("w2", "w3") for x in ROLLOUTS}
    return [c for c in SI.inputs("w1") if c["S"] in W1_LENGTHS] + [c for w in ("w2", "w3") for c in SI.inputs(w) if c["id"] in ids]


def load_only(W):
    """[nb, nb] GPU per live pair: block rows by work descending, each to the least-loaded GPU (ties: the lowest id)"""
    rw = W.sum(1)
    load = np.zeros(SI.P)
    gpu = np.zeros(len(rw), np.int64)
    for i in np.argsort(-rw, kind="stable"):
        g = int(np.argmin(load))
        gpu[i] = g
        load[g] += rw[i]
    return np.where(W > 0, gpu[:, None], -1)


def write_plan(path, case, arm, owner, table, block_tokens, seconds):
    json.dump(dict(family=case["id"], strategy=arm, cp=SI.P, n_blocks=len(owner), nodes=SI.NN, gpus_per_node=SI.GPN, block_tokens=block_tokens,
                   owner=[int(x) for x in owner], schedule_table=np.asarray(table).tolist(), plan_s=round(seconds, 4), mask=case["mask"]), open(path, "w"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="full,load_only")
    ap.add_argument("ids", nargs="*", help="only these input ids")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(plan_dir(), exist_ok=True)
    arms = a.arms.split(",")
    for case in cases():
        if a.ids and case["id"] not in a.ids:
            continue
        z = np.load(case["mask"])
        m = z["mask"].astype(np.int8)
        heads = tuple(int(x) for x in z["heads"])
        for pass_, nb in zip(("fwd", "bwd"), SI.BLOCKS[case["workload"]]):
            B = m.shape[0] * SI.TILE // nb
            W, kc, _ = SI.block_problem(m, B, dev)
            zz = pl.zigzag_equal(nb, SI.P)
            live = (W > 0).cpu().numpy()
            if "full" in arms and not os.path.exists(plan_file(case["id"], "full", pass_)):
                Qb, Ob, Kb = pl.fwd_bytes(heads, B, SI.D)
                TT = pl.FWD_MODEL["T_TILE"] * heads[0] / 32
                t0 = time.perf_counter()
                p = pl.plan(W, torch.tensor(zz, device=dev), SI.GPN, SI.NN, Qb, Ob, Kb, TT, bwd=(heads, B, kc) if pass_ == "bwd" else None)
                write_plan(plan_file(case["id"], "full", pass_), case, "full", zz, np.where(live, p.cpu().numpy(), -1), B, time.perf_counter() - t0)
            if "load_only" in arms and not os.path.exists(plan_file(case["id"], "load_only", pass_)):
                t0 = time.perf_counter()
                t = load_only(W.cpu().numpy())
                write_plan(plan_file(case["id"], "load_only", pass_), case, "load_only", zz, t, B, time.perf_counter() - t0)
            print(f"{case['id']} {pass_}: {nb} blocks", flush=True)


if __name__ == "__main__":
    main()
