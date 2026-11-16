"""UA as the replaced system X: UA's forward plan with its own gpu_kernels order. A plan whose blocks exceed the forward scheduler's 64
tiles is refined losslessly: each coarse block -> f fine blocks with the same owner, each fine pair inherits its coarse pair's GPU (-1 where
the mask has no live tile), and a comp item's rows / cols and a comm item's block expand to their f fine blocks at the same step.
Output: $DEX_DATA_ROOT/plans/ablation/ua/<id>_ua.json. Example: python experiments/ablation_components/ua_order.py"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import sample as SM  # noqa: E402
from dex.common.env import cfg  # noqa: E402

TILE, CAP = 128, 64


def refine(plan, tm):
    N, B = plan["n_blocks"], plan["block_tokens"]
    per = B // TILE
    if per <= CAP:
        return plan
    f = next(f for f in range(2, per + 1) if per % f == 0 and per // f <= CAP)
    assert N * f * (per // f) == tm.shape[0], (N, f, per, tm.shape)
    Nf, pf = N * f, per // f
    live = tm.reshape(Nf, pf, Nf, pf).any(axis=(1, 3))
    fine = np.repeat(np.repeat(np.asarray(plan["schedule_table"]), f, 0), f, 1)
    fine[~live] = -1
    ex = lambda b: [int(b) * f + i for i in range(f)]  # noqa: E731
    gk = []
    for g in plan["gpu_kernels"]:
        out = []
        for it in g:
            if it["kind"] == "comp":
                out.append(dict(it, rows=[x for b in it["rows"] for x in ex(b)], cols=[x for b in it.get("cols", []) for x in ex(b)]))
            else:
                out += [dict(it, block=x) for x in ex(it["block"])]
        gk.append(out)
    return dict(plan, n_blocks=Nf, block_tokens=B // f, owner=[int(plan["owner"][i // f]) for i in range(Nf)], schedule_table=fine.tolist(), gpu_kernels=gk,
                refined_from=dict(n_blocks=N, block_tokens=B, factor=f))


def main():
    man = json.load(open(os.path.join(cfg("DEX_DATA_ROOT"), "plans", "ua", "manifest.json")))
    out = SM.ablation_dir("ua")
    for _, c in SM.cases():
        e = man[f"{c['id']}|fwd"]
        d = json.load(open(e["plan"]))
        d["mask"] = e["mask"]
        d = refine(d, np.load(e["mask"])["mask"] != 0)
        json.dump(d, open(f"{out}/{c['id']}_ua.json", "w"))
        print(c["id"], d["n_blocks"], "x", d["block_tokens"], flush=True)


if __name__ == "__main__":
    main()
