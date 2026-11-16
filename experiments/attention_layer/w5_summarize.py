"""W5 operator-test summary: $DEX_RESULT_ROOT/attention_layer/w5/<system>/rep{1,2}.jsonl -> summary.{json,md}.
Per case and system: each launch's median of its timed calls, the launches combined by geometric mean. Speedup =
baseline / Dex per case, geometric mean per resolution and overall, for the forward with the mask known, the forward
with selection and the backward. Also: Dex vs Dex without assigner, the decomposed executor vs Dex (if present),
Dex's online plan stages (median over cases), FP64 worst relative L2 per system (launch 1), launch 2 / launch 1.
Diagnostic only: the article's values come from analysis/ (tables.py, fig_*.py), which uses geometric means across
inputs and does not read this summary.
usage: python experiments/attention_layer/w5_summarize.py"""

import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from dex.common.env import cfg  # noqa: E402

SYS = ("dex", "dex_no_assigner", "ring", "allgather", "sparse_allgather", "decomposed")
KINDS = ("fwd_exec_ms", "fwd_online_ms", "bwd_ms")


def gm(x):
    x = list(x)
    return math.exp(sum(map(math.log, x)) / len(x)) if x else float("nan")


def main():
    W = os.path.join(cfg("DEX_RESULT_ROOT"), "attention_layer", "w5")
    R = {}
    for s in SYS:
        for r in (1, 2):
            f = f"{W}/{s}/rep{r}.jsonl"
            if os.path.exists(f):
                R.setdefault(s, {})[r] = {x["name"]: x for x in map(json.loads, open(f))}
    assert "dex" in R, "no Dex results"
    have = [s for s in SYS if s in R]
    names = sorted(set.intersection(*[set(R[s][r]) for s in have for r in R[s]]))
    res = lambda n: n.split("_")[0]  # noqa: E731
    v = {s: {n: {k: gm(R[s][r][n][k] for r in R[s]) for k in KINDS} for n in names} for s in have}
    out, md = dict(n_cases=len(names), systems=have), [
        f"# W5 operator tests ({len(names)} cases)",
        "",
        "Speedup of Dex = baseline / Dex, geometric mean over cases: fwd (mask known) / fwd (with selection) / bwd.",
        "",
        "| res | vs " + " | vs ".join(s for s in have if s != "dex") + " |",
        "|---|" + "---|" * (len(have) - 1),
    ]
    for grp in ("480p", "720p", "all"):
        sel = [n for n in names if grp == "all" or res(n) == grp]
        sp = {b: [gm(v[b][n][k] / v["dex"][n][k] for n in sel) for k in KINDS] for b in have if b != "dex"}
        out[grp] = dict(
            n=len(sel),
            speedup=sp,
            median_ms={s: {k: float(np.median([v[s][n][k] for n in sel])) for k in KINDS} for s in have},
        )
        md.append(f"| {grp} | " + " | ".join(" / ".join(f"{x:.2f}x" for x in sp[b]) for b in sp) + " |")
    stages = {}
    for n in names:
        for r in R["dex"]:
            for k, x in R["dex"][r][n].get("plan_ms", {}).items():
                if isinstance(x, (int, float)):
                    stages.setdefault((res(n), k), []).append(x)
    out["plan_stages_ms"] = {f"{g}/{k}": round(float(np.median(x)), 2) for (g, k), x in sorted(stages.items())}
    md += ["", "Dex online plan stages, median ms: " + json.dumps(out["plan_stages_ms"])]
    err = {}
    for s in have:
        if 1 not in R[s]:
            continue
        e = [x["rel_l2"] for n in names for x in R[s][1][n].get("err_fp64", {}).values()]
        nf = sum(x.get("nonfinite_rows", 0) for n in names for x in R[s][1][n].get("err_fp64", {}).values())
        err[s] = dict(checks=len(e), worst_rel_l2=max(e) if e else None, nonfinite_rows=nf)
    out["correctness"] = err
    md += ["", "FP64 sampled rows (launch 1, O / dQ / dK / dV): " + json.dumps(err)]
    ratio = {
        f"{s}/{k}": round(gm(R[s][2][n][k] / R[s][1][n][k] for n in names), 3)
        for s in have
        if 1 in R[s] and 2 in R[s]
        for k in KINDS
    }
    out["launch_ratio"] = ratio
    md += ["", "Launch 2 / launch 1 (geometric mean): " + json.dumps(ratio)]
    json.dump(out, open(f"{W}/summary.json", "w"), indent=1)
    open(f"{W}/summary.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
