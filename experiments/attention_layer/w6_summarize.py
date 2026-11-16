"""W6 operator-test summary: $DEX_RESULT_ROOT/attention_layer/w6/<system>/rep{1,2}.jsonl -> summary.{json,md}.
Per case and system: each launch's median of its timed calls, the launches combined by geometric mean. Speedup = baseline /
ours per case, geometric mean per length and overall (for Dex and Dex without assigner); plan stages (median over cases),
moved work, home-gate decisions, received bytes (Dex variants from their message tables; Sparse all-gather from its lists;
All-gather / Ring analytic, K + V fp16 of 8 KV heads: per GPU (P-1) L_kv; per node All-gather 3/4 of all K / V, Ring (P-1) L_kv),
FP64 worst relative L2, launch 2 / launch 1.
Diagnostic only: the article's values come from analysis/ (tables.py, fig_*.py), which uses geometric means across
inputs and does not read this summary.
usage: python experiments/attention_layer/w6_summarize.py"""

import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from dex.common.env import cfg  # noqa: E402

SYS = ["dex", "dex_no_assigner", "allgather", "ring", "sparse_allgather"]
LENS = ["32k", "64k", "128k"]
P = 16


def gm(v):
    v = [x for x in v if x is not None and x > 0]
    return math.exp(np.mean(np.log(v))) if v else float("nan")


def main():
    O = os.path.join(cfg("DEX_RESULT_ROOT"), "attention_layer", "w6")
    R = {}
    for s in SYS + ["decomposed"]:
        for r in (1, 2):
            f = f"{O}/{s}/rep{r}.jsonl"
            if os.path.exists(f):
                for ln in open(f):
                    x = json.loads(ln)
                    R.setdefault(s, {}).setdefault(x["name"], {})[r] = x
    names = sorted(set.intersection(*[set(R[s]) for s in SYS]))
    Lof = lambda n: n.split("_")[-2]  # noqa: E731
    lat = lambda s, n, k="fwd_ms": gm(r.get(k) for r in R[s][n].values())  # noqa: E731
    out = dict(n_cases=len(names), per_len={})
    md = [
        f"# W6 Llama-3.1-8B + FlexPrefill operator tests ({len(names)} cases)",
        "",
        "Speedup = baseline / ours, geometric mean over cases (wins = cases ours is faster).",
        "",
        "| S | n | Dex vs All-gather | vs Ring | vs Sparse all-gather | vs no assigner | no assigner vs All-gather | vs Ring | vs Sparse all-gather | geomean ms |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for Ln in LENS + ["all"]:
        ns = [n for n in names if Ln == "all" or Lof(n) == Ln]
        if not ns:
            continue
        d = dict(n=len(ns))
        for ours in ("dex", "dex_no_assigner"):
            for b in SYS:
                if b != ours:
                    sp = [lat(b, n) / lat(ours, n) for n in ns]
                    d[f"{ours}_vs_{b}"] = gm(sp)
                    d[f"{ours}_vs_{b}_wins"] = int(sum(x > 1 for x in sp))
        d["geomean_ms"] = {s: gm(lat(s, n) for n in ns) for s in SYS}
        if "decomposed" in R:
            nd = [n for n in ns if n in R["decomposed"]]
            d["decomposed_over_dex"] = gm(lat("decomposed", n) / lat("dex", n) for n in nd)
            d["decomposed_over_dex_kernel"] = gm(
                lat("decomposed", n, "kernel_ms") / lat("dex", n, "kernel_ms") for n in nd
            )
        for s in ("dex", "dex_no_assigner"):
            recs = [next(iter(R[s][n].values())) for n in ns]
            ks = sorted({k for x in recs for k in x.get("plan_ms", {})})
            d[f"{s}_plan_ms"] = {
                k: float(np.median([x["plan_ms"][k] for x in recs if k in x.get("plan_ms", {})])) for k in ks
            }
            d[f"{s}_recv_max_MiB"] = float(np.median([x["bytes"]["recv_max_MiB"] for x in recs]))
            d[f"{s}_inter_node_max_MiB"] = float(np.median([x["bytes"]["inter_node_max_MiB"] for x in recs]))
        recs = [next(iter(R["sparse_allgather"][n].values())) for n in ns]
        d["sparse_allgather_recv_max_MiB"] = float(np.median([x["bytes"]["recv_max_MiB"] for x in recs]))
        Lkv = [
            next(iter(R["dex"][n].values()))["S_pad"] / P * 8 * 128 * 2 * 2 / 2**20 for n in ns
        ]  # one rank's K + V
        d["allgather_recv_max_MiB"] = d["ring_recv_max_MiB"] = float(np.median(Lkv)) * (P - 1)
        d["allgather_inter_node_max_MiB"] = float(np.median(Lkv)) * 12
        d["ring_inter_node_max_MiB"] = float(np.median(Lkv)) * (P - 1)
        A = [next(iter(R["dex"][n].values())).get("assign") for n in ns]
        d["moved_work_frac_median"] = float(np.median([x["moved_work_frac"] for x in A]))
        d["cases_all_home"] = int(sum(x["moved_pairs"] == 0 for x in A))
        G = [r["gate_kept_home_calls"] for n in ns for r in R["dex"][n].values()]
        d["gate_calls_kept_home"] = [int(sum(x[0] for x in G)), int(sum(x[1] for x in G))]
        d["max_err_fp64"] = {
            s: max((R[s][n].get(1, {}).get("err_fp64") or {}).get("o", {}).get("rel_l2", 0) for n in ns) for s in SYS
        }
        d["launch2_over_launch1"] = {
            s: gm(R[s][n][2]["fwd_ms"] / R[s][n][1]["fwd_ms"] for n in ns if 1 in R[s][n] and 2 in R[s][n]) for s in SYS
        }
        out["per_len"][Ln] = d
        c = lambda ours, b: f"{d[f'{ours}_vs_{b}']:.2f} ({d[f'{ours}_vs_{b}_wins']})"  # noqa: E731
        md.append(
            f"| {Ln} | {d['n']} | {c('dex', 'allgather')} | {c('dex', 'ring')} | {c('dex', 'sparse_allgather')} | "
            f"{c('dex', 'dex_no_assigner')} | {c('dex_no_assigner', 'allgather')} | {c('dex_no_assigner', 'ring')} | "
            f"{c('dex_no_assigner', 'sparse_allgather')} | "
            + " / ".join(f"{d['geomean_ms'][s]:.1f}" for s in SYS)
            + " |"
        )
    json.dump(out, open(f"{O}/summary.json", "w"), indent=1)
    open(f"{O}/summary.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
