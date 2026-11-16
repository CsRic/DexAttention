"""Select the 24 W2 / W3 rollout groups from the sampled pool (deterministic, independent of any measurement).

Pool: every (prompt, 16 replies) group of the token-count files workloads/w2_w3_rollout/lengths/R_*_<tag>.json (or --pool, the
output directory of sample_rollouts.py) whose longest path P + max(R) fits
the model context, in source order (file name, prompt index). S = P + sum(R), rho = P / S (prompt sharing fraction).
  S bins   = S tertiles of the pool;
  rho bins = rho quartiles inside each of the two shorter S bins, the first 2 groups per cell (16 groups);
  longest S bin: the 8 groups whose rho is nearest to the quantiles (i + 0.5) / 8 of the 16 groups already selected
                 (each group used once), so all three lengths share one rho distribution.
W3 (Qwen3-4B, 32/8 heads) and W2 (Qwen1.5-1.8B shape, 16/16) use the same 24 sequences.
Outputs ($DEX_DATA_ROOT/rollouts/): selection.json, inventory.md, names_w3.txt, names_w2.txt
Example: python -m workloads.w2_w3_rollout.select_groups
"""
import argparse
import glob
import json
import os

import numpy as np

from dex.common.env import cfg

PER_CELL = 2
CONTEXT = 262144


def candidates(root, tag):
    out = []
    for f in sorted(glob.glob(os.path.join(root, f"R_*_{tag}*.json"))):
        grp = {}
        for r in json.load(open(f))["responses"]:
            grp.setdefault(r["idx"], []).append(r)
        for idx, g in sorted(grp.items()):
            P = g[0]["P"]
            R = [r["R"] for r in g]
            S = P + sum(R)
            if P + max(R) > CONTEXT:
                continue
            out.append(dict(file=os.path.basename(f), split=os.path.basename(f).split("_")[1], idx=idx, P=P, S=S,
                            rho=P / S, R=R))
    return out


def input_id(c, workload):
    return f"{workload}_{c['split']}_p{c['idx']}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="qwen3-4b-thinking")
    ap.add_argument("--pool", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "lengths"),
                    help="directory of the R_*_<tag>.json files")
    a = ap.parse_args()
    root = os.path.join(cfg("DEX_DATA_ROOT"), "rollouts")
    os.makedirs(root, exist_ok=True)
    pool = candidates(a.pool, a.tag)
    S = np.array([c["S"] for c in pool], float)
    se = np.quantile(S, [0, 1 / 3, 2 / 3, 1])
    sel = []
    md = ["# W2 / W3 group selection", "", f"pool: {len(pool)} groups; S tertile edges (tokens): " +
          ", ".join(f"{x:.0f}" for x in se), "", "| S bin | n | rho quartile edges | selected (id: S, rho) |",
          "|---|---|---|---|"]
    for si in range(3):
        lo, hi = se[si], se[si + 1]
        grp = [c for c in pool if (lo <= c["S"] < hi) or (si == 2 and c["S"] == hi)]
        picked = []
        if si == 2:
            ref = np.sort([c["rho"] for c in sel])
            tgt = np.quantile(ref, (np.arange(4 * PER_CELL) + 0.5) / (4 * PER_CELL))
            left = list(grp)
            for t in tgt:
                c = min(left, key=lambda c: abs(c["rho"] - t))
                left.remove(c)
                sel.append(dict(c, s_bin=[int(lo), int(hi)], rho_target=float(t), s_bin_i=si, rho_bin_i=-1))
                picked.append(f"{input_id(c, 'w3')}: {c['S']}, {c['rho']:.2f} (target {t:.2f})")
            md.append(f"| [{lo / 1024:.0f}K, {hi / 1024:.0f}K] | {len(grp)} | rho matched to bins 1-2 | " +
                      "; ".join(picked) + " |")
            continue
        re_ = np.quantile([c["rho"] for c in grp], [0, .25, .5, .75, 1])
        for ri in range(4):
            cs = [c for c in grp if (re_[ri] <= c["rho"] < re_[ri + 1]) or (ri == 3 and c["rho"] == re_[4])]
            for c in cs[:PER_CELL]:
                sel.append(dict(c, s_bin=[int(lo), int(hi)], rho_bin=[float(re_[ri]), float(re_[ri + 1])], s_bin_i=si,
                                rho_bin_i=ri))
                picked.append(f"{input_id(c, 'w3')}: {c['S']}, {c['rho']:.2f}")
        md.append(f"| [{lo / 1024:.0f}K, {hi / 1024:.0f}K] | {len(grp)} | " + " / ".join(f"{x:.2f}" for x in re_) +
                  " | " + "; ".join(picked) + " |")
    assert len(sel) == 3 * 4 * PER_CELL, len(sel)
    json.dump(sel, open(os.path.join(root, "selection.json"), "w"), indent=1)
    open(os.path.join(root, "inventory.md"), "w").write("\n".join(md) + "\n")
    for w in ("w3", "w2"):
        open(os.path.join(root, f"names_{w}.txt"), "w").write("\n".join(input_id(c, w) for c in sel) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
