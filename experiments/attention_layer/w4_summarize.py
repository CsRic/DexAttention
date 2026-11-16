"""W4 operator-test summary: $DEX_RESULT_ROOT/attention_layer/w4/<system>/rep{1,2}.jsonl -> summary.{json,md}.
Per case, part and system: each launch's median of its timed calls, the launches combined by geometric mean. Speedup =
baseline / Dex per case, geometric mean per length and overall, for exec, online and bwd. Also: the plan span (slowest
rank's plan phase; the online-cost numbers), the home gate, the correctness of launch 1, and the executor ablation:
decomposed / Dex with and without planning (Dex's latency minus its plan span; the decomposed executor has no
separate plan phase, so Dex's plan span is subtracted from both and its step-list construction stays timed).
Diagnostic only: the article's values come from analysis/ (tables.py, fig_*.py), which uses geometric means across
inputs and does not read this summary.
usage: python experiments/attention_layer/w4_summarize.py"""
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from dex.common.env import cfg  # noqa: E402

SYS = ("dex", "dex_no_assigner", "ring", "allgather", "sparse_allgather", "decomposed")
KINDS = ("exec", "online", "bwd")


def gm(x):
    x = list(x)
    return math.exp(sum(map(math.log, x)) / len(x)) if x else float("nan")


def main():
    W = os.path.join(cfg("DEX_RESULT_ROOT"), "attention_layer", "w4")
    R = {}
    for s in SYS:
        for r in (1, 2):
            f = f"{W}/{s}/rep{r}.jsonl"
            if os.path.exists(f):
                R.setdefault(s, {})[r] = {x["name"]: x for x in map(json.loads, open(f))}
    assert "dex" in R, "no Dex results"
    have = [s for s in SYS if s in R]
    full = {s: set.intersection(*[set(R[s][r]) for r in R[s]]) for s in have}
    bin_of = lambda n: int(n.split(":")[0].rsplit("_", 1)[1])          # noqa: E731
    bins = sorted({bin_of(n) for n in full["dex"]})
    t = lambda s, n, k: gm(R[s][r][n][k]["median_ms"] for r in R[s])   # noqa: E731
    out, md = dict(n_cases={s: len(full[s]) for s in have}), ["# W4 operator tests", "",
                                                             "Speedup = system / Dex per case, geometric mean: exec / online / bwd", "",
                                                             "| length | system | cases | exec | online | bwd |", "|---|---|---|---|---|---|"]
    for S in bins + ["all"]:
        for s in have[1:]:
            ks = [n for n in full["dex"] & full[s] if S == "all" or bin_of(n) == S]
            sp = {k: gm(t(s, n, k) / t("dex", n, k) for n in ks) for k in KINDS}
            sp["fwd_bwd"] = gm((t(s, n, "exec") + t(s, n, "bwd")) / (t("dex", n, "exec") + t("dex", n, "bwd")) for n in ks)
            out[f"{s}/{S}"] = dict(n=len(ks), **sp)
            md.append(f"| {S} | {s} | {len(ks)} | {sp['exec']:.3f} | {sp['online']:.3f} | {sp['bwd']:.3f} |")
    # plan span (online cost) + plan stages + gate
    md += ["", "## Dex plan span, ms (geometric mean over cases): exec / bwd", ""]
    gate, stages = {"fwd": [0, 0], "bwd": [0, 0]}, {}
    for S in bins:
        ks = [n for n in full["dex"] if bin_of(n) == S]
        sp = {k: gm(gm(R["dex"][r][n][k]["plan_span_ms"] for r in R["dex"]) for n in ks) for k in ("exec", "bwd")}
        out[f"plan_span/{S}"] = sp
        md.append(f"- {S}: {sp['exec']:.2f} / {sp['bwd']:.2f}")
    for r in R["dex"]:
        for x in R["dex"][r].values():
            for part, g_ in (("exec", "fwd"), ("bwd", "bwd")):
                info = x[part]["info"][0]
                gate[g_][0] += 1; gate[g_][1] += int(info["gate"][g_]["kept_home"])
                for k_, v_ in info["plan_ms"].items():
                    stages.setdefault(f"{part}/{k_}/{bin_of(x['name'])}", []).append(v_)
    out["gate_calls_home_kept"] = gate
    out["plan_stages_ms"] = {k_: float(np.median(v_)) for k_, v_ in sorted(stages.items())}
    md += ["", f"Home gate (plan calls, home kept): {json.dumps(gate)}", ""]
    # correctness
    md += ["## Correctness (launch 1, layers 0 / 16)", "", "| system | checks | worst FP64 rel L2 | per tensor | max rel vs All-gather |", "|---|---|---|---|---|"]
    for s in have:
        worst, n, ag = {}, 0, 0.0
        for x in R[s].get(1, {}).values():
            for part in ("exec", "bwd", "online"):
                c = x[part].get("check") or {}
                for tn, v in (c.get("fp64") or {}).items():
                    if isinstance(v, dict) and "rel_l2" in v:
                        n += 1; worst[tn] = max(worst.get(tn, 0.0), v["rel_l2"])
                ag = max([ag] + [c[k_] for k_ in ("max_rel", "dq", "dk", "dv") if isinstance(c.get(k_), float)])
        out[f"correctness/{s}"] = dict(checks=n, worst_per_tensor=worst, max_rel_vs_allgather=ag)
        md.append(f"| {s} | {n} | {max(worst.values(), default=0.0):.2e} | " + ", ".join(f"{k_} {v:.1e}" for k_, v in sorted(worst.items())) + f" | {ag:.2e} |")
    # executor ablation
    if "decomposed" in have:
        ks = sorted(full["dex"] & full["decomposed"])
        e = {}
        for n in ks:
            for k in ("exec", "bwd"):
                f_, d_ = [], []
                for r in R["dex"]:
                    fp = R["dex"][r][n][k]
                    f_.append(fp["median_ms"] - fp["plan_span_ms"]); d_.append(R["decomposed"][r][n][k]["median_ms"] - fp["plan_span_ms"])
                e[(n, k)] = (gm(f_), gm(d_))
        ab = dict(fwd=gm(e[(n, "exec")][1] / e[(n, "exec")][0] for n in ks), bwd=gm(e[(n, "bwd")][1] / e[(n, "bwd")][0] for n in ks),
                  total=gm((e[(n, "exec")][1] + e[(n, "bwd")][1]) / (e[(n, "exec")][0] + e[(n, "bwd")][0]) for n in ks))
        out["ablation_executor_excl_planning"] = ab
        md += ["", f"## Executor ablation, decomposed / Dex excluding planning ({len(ks)} cases): forward {ab['fwd']:.3f}, "
                   f"backward {ab['bwd']:.3f}, both {ab['total']:.3f}; forward including planning {out['decomposed/all']['exec']:.3f}"]
    open(f"{W}/summary.md", "w").write("\n".join(md) + "\n")
    json.dump(out, open(f"{W}/summary.json", "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
