"""Static planning time of Dex on W1-W3 (Table 4 static planning column), compiled planner, one GPU.

Same problems as static_plans.py (virtual-head block-diagonal work matrix, block counts per workload). Per input and
pass: one untimed call (compilation / graph capture for a shape seen the first time), then 3 timed calls, median. Fields:
  plan_ms         forward + backward plans at the deployed placement (to_blocks + compiled Planner + scatter back)
  plan_choose_ms  the same for BOTH equal-chunk placements (zigzag, contiguous): the static planning time reported for
                  Dex; the model evaluation that picks between them (choose_pick) is not timed, and the evaluated
                  plans use the deployed placement
  sched_tables_ms schedule + kernel-table build of the deployed plans for one rank (forward: block pairs, schedule,
                  tables; backward: block pairs + build_fast), median of 3 after one untimed call
  identical       the deployed-placement plans equal the plan files of static_plans.py
Output: $DEX_RESULT_ROOT/planning_latency/static_plan_time_hs<hv>.json (one record per input; existing kept)
Example: python experiments/attention_layer/static_plan_time.py --workloads w1 w2 w3
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(os.path.dirname(HERE))]
import static_inputs as SI  # noqa: E402
from dex import planner as pl  # noqa: E402
from dex import scheduler as S  # noqa: E402
from dex.common.env import cfg  # noqa: E402
from dex.planner.placement import CANDIDATES, bwd_time, fwd_time  # noqa: E402

CACHE = {}


def planner(H, m, hd, B, kb, bwd, args):
    """one compiled Planner per (shape, heads, pass); another B / kcols retargets it without recompiling"""
    key = (H, m, tuple(hd), bwd)
    first = key not in CACHE
    if first:
        CACHE[key] = pl.Planner(H, m, SI.GPN, SI.NN, *args, bwd=(hd, B, kb) if bwd else None)
    G = CACHE[key]
    if bwd:
        G.retarget(*args, B=B, kcols=kb)
    else:
        G.retarget(*args)
    return G, first


def one(Wm, owm, kcm, hd, B, bwd, dev):
    Qb, Ob, Kb = pl.fwd_bytes(hd, B, SI.D)
    TT = pl.FWD_MODEL["T_TILE"] * hd[0] / 32
    segs, Wb, ob, gid, kb = pl.to_blocks(Wm, owm, kcm)
    G, first = planner(Wb.shape[0], Wb.shape[1], hd, B, kb, bwd, (Qb, Ob, Kb, TT))
    pb = G(Wb, ob, gid)
    if first:
        pb = G(Wb, ob, gid)
    full = torch.full(Wm.shape, -1, dtype=torch.long, device=dev)
    for h, (s, sz) in enumerate(segs):
        full[s:s + sz, s:s + sz] = pb[h, :sz, :sz]
    return full


def timed(f, cuda=True):
    f()
    ts = []
    r = None
    for _ in range(3):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = f()
        if cuda:
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), r


def sched_tables(mv, plan_f, heads_v, bwd, dev):
    """schedule + tables of one plan file for rank 0 (the per-rank work of the static path)"""
    d = json.load(open(plan_f))
    per = d["block_tokens"] // SI.TILE
    devb, ownb = np.asarray(d["schedule_table"], np.int64), np.asarray(d["owner"], np.int64)

    def run():
        if bwd:
            Bp = S.block_pairs(mv, devb, per, owner=ownb)
            return S.build_fast(Bp, ownb, SI.P, per, 0, heads_v[0], heads_v[1])
        B = S.block_pairs(mv, devb, per, owner=ownb)
        pw = (B["nfull"] + 0.5 * B["ndiag"]) * S.LINK["T_TILE"] * heads_v[0] / 32
        S.schedule_blocks(B["bq"], B["bk"], B["pdev"], pw, B["prow"], B["pcol"], ownb, SI.P, per, heads=heads_v, d=SI.D)
        return S.tables_blocks(0, B, ownb, SI.P, per, dev)
    return timed(run, cuda=not bwd)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workloads", nargs="+", default=["w1", "w2", "w3"])
    ap.add_argument("--head-split", type=int, default=4)
    ap.add_argument("ids", nargs="*")
    a = ap.parse_args()
    dev = torch.device("cuda")
    out = os.path.join(cfg("DEX_RESULT_ROOT"), "planning_latency", f"static_plan_time_hs{a.head_split}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    res = json.load(open(out)) if os.path.exists(out) else {}
    for w in a.workloads:
        for case in SI.inputs(w):
            if (a.ids and case["id"] not in a.ids) or case["id"] in res:
                continue
            z = np.load(case["mask"])
            m = z["mask"].astype(np.int8)
            heads = tuple(int(x) for x in z["heads"])
            hv, hd = SI.virtual_heads(heads, a.head_split)
            fm, ff, fb = SI.plan_files(case, a.head_split)
            mv = np.load(fm)["mask"].astype(np.int8)
            rec = dict(input_id=case["id"], workload=w.upper(), head_split=hv, heads_virtual=list(hd))
            t_dep = t_all = t_tab = 0.0
            ident = True
            pick = {}
            for tag, nb, f in (("fwd", SI.BLOCKS[w][0], ff), ("bwd", SI.BLOCKS[w][1], fb)):
                bwd = tag == "bwd"
                ref = json.load(open(f))
                B = ref["block_tokens"]
                W, kc, _ = SI.block_problem(m, B, dev)
                Wm, kcm = SI.bdiag(W, hv, 0.0), kc.repeat(hv)
                owm = torch.as_tensor(ref["owner"], device=dev)
                t, p = timed(lambda: one(Wm, owm, kcm, hd, B, bwd, dev))
                t_dep += t
                live = Wm > 0
                ident &= bool(np.array_equal(np.asarray(ref["schedule_table"]),
                                             torch.where(live, p, torch.full_like(p, -1)).cpu().numpy()))
                rec[f"{tag}_ms"] = round(t * 1e3, 3)
                rec[f"{tag}_nb"] = nb
                Qb, Ob, Kb = pl.fwd_bytes(hd, B, SI.D)
                TT = pl.FWD_MODEL["T_TILE"] * hd[0] / 32
                for cname, cf in CANDIDATES.items():
                    o = torch.as_tensor(cf(nb, SI.P), device=dev).repeat(hv)
                    t2, p2 = timed(lambda: one(Wm, o, kcm, hd, B, bwd, dev))
                    t_all += t2
                    pick.setdefault(cname, 0.0)
                    pick[cname] += float(bwd_time(Wm, o, p2, SI.GPN, SI.NN, hd, B, kcm) if bwd else
                                         fwd_time(Wm, o, p2, SI.GPN, SI.NN, Qb, Ob, Kb, TT))
                t_tab += sched_tables(mv, f, hd, bwd, dev)
            rec.update(plan_ms=round(t_dep * 1e3, 3), plan_choose_ms=round(t_all * 1e3, 3),
                       sched_tables_ms=round(t_tab * 1e3, 3), identical=ident, choose_pick=min(pick, key=pick.get))
            res[case["id"]] = rec
            json.dump(res, open(out, "w"), indent=1)
            print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
