"""W6 end-to-end prefill summary: $DEX_RESULT_ROOT/end2end/llama/<system>/rep{1,2}.jsonl + the saved logits / hidden samples ->
summary.{json,md}. Latency per input and launch = geometric mean of its timed prefills; launches combined by geometric mean;
speedup = baseline / ours, geometric mean over the inputs of a length. Also: per-length breakdown (selection / attention / rest,
mean over ranks), home-gate decisions, and correctness of launch 1: logits and hidden samples against All-gather and the
official reference ($DEX_DATA_ROOT/w6_flexprefill/official), greedy-token match, masks identical to All-gather.
usage: python experiments/end2end/llama_summarize.py   (on the node that holds the saved files)"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from dex.common.env import cfg  # noqa: E402

SYS = ["dex", "dex_no_assigner", "allgather", "ring", "sparse_allgather"]
LENS = ["32k", "64k", "128k"]
EPS = 0.040


def gm(x):
    x = np.asarray([v for v in x if np.isfinite(v)])
    return float(np.exp(np.log(x).mean())) if len(x) else float("nan")


def err(a, b):
    """a vs reference b: relative L2, max |a - b| / max |b|"""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b)), float(np.abs(a - b).max() / np.abs(b).max())


def bf16(x):
    return (x.astype(np.int32) << 16).view(np.float32)


def main():
    O = os.path.join(cfg("DEX_RESULT_ROOT"), "end2end", "llama")
    OFF = os.path.join(cfg("DEX_DATA_ROOT"), "w6_flexprefill", "official")

    def load(s, r):
        f = f"{O}/{s}/rep{r}.jsonl"
        return {x["input"]: x for x in map(json.loads, open(f))} if os.path.exists(f) else {}

    R = {s: {r: load(s, r) for r in (1, 2)} for s in SYS}
    inputs = open(os.path.join(ROOT, "workloads", "w6_flexprefill", "inputs9.txt")).read().split()

    def lat(s, i):
        v = [gm(R[s][r][i]["prefill_all"]) for r in (1, 2) if i in R[s][r]]
        return gm(v) if v else float("nan")

    def bd(s, i):
        v = [R[s][r][i]["breakdown_mean_ms"] for r in (1, 2) if i in R[s][r]]
        return {k: gm([x[k] for x in v]) for k in v[0]} if v else None

    out = dict(per_input={i: {s: lat(s, i) for s in SYS} for i in inputs}, per_len={}, correctness={})
    md = [
        "# W6 end-to-end prefill (Llama-3.1-8B-Instruct + FlexPrefill)",
        "",
        "| S | Dex vs All-gather | vs Ring | vs Sparse all-gather | vs no assigner | no assigner vs All-gather | vs Ring | geomean ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for Ln in LENS + ["all"]:
        ins = [i for i in inputs if Ln == "all" or i.endswith("_" + Ln)]
        rec = {}
        for ours in ("dex", "dex_no_assigner"):
            for b in SYS:
                if b != ours:
                    rec[f"{ours}_vs_{b}"] = gm(lat(b, i) / lat(ours, i) for i in ins)
        rec["geomean_ms"] = {s: gm(lat(s, i) for i in ins) for s in SYS}
        bs = {s: [x for x in (bd(s, i) for i in ins) if x] for s in SYS}
        rec["breakdown_ms"] = {s: {k: gm(x[k] for x in v) for k in v[0]} for s, v in bs.items() if v}
        out["per_len"][Ln] = rec
        md.append(
            f"| {Ln} | {rec['dex_vs_allgather']:.3f} | {rec['dex_vs_ring']:.3f} | {rec['dex_vs_sparse_allgather']:.3f} | "
            f"{rec['dex_vs_dex_no_assigner']:.3f} | {rec['dex_no_assigner_vs_allgather']:.3f} | {rec['dex_no_assigner_vs_ring']:.3f} | "
            + " / ".join(f"{rec['geomean_ms'][s]:.0f}" for s in SYS)
            + " |"
        )
    k_, n_ = 0, 0
    for r in (1, 2):
        for x in R["dex"][r].values():
            g = np.array(x["gate_first"], dtype=np.float64)
            k_ += int((g[:, 0] <= g[:, 1] * (1 + EPS)).sum())
            n_ += len(g)
    out["gate_kept_home_first_calls"] = [k_, n_]
    md += ["", f"Home gate, first calls of both launches: home kept in {k_} of {n_} layer plans."]
    md += [
        "",
        "| system | input | logits vs All-gather rel L2 / max rel | greedy = All-gather | hidden vs All-gather rel L2 | logits vs official rel L2 | greedy = official | masks = All-gather |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in SYS:
        for i in inputs:
            x, ag = R[s][1].get(i), R["allgather"][1].get(i)
            if not x or not ag:
                continue
            d, da = f"{O}/{s}/rep1", f"{O}/allgather/rep1"
            lg, la, lo = (
                np.load(f"{d}/{i}_logits.npy"),
                np.load(f"{da}/{i}_logits.npy"),
                np.load(f"{OFF}/{i}_logits.npy"),
            )
            e_la, e_ha = err(lg, la), err(bf16(np.load(f"{d}/{i}_hid.npy")), bf16(np.load(f"{da}/{i}_hid.npy")))
            e_lo = err(lg, lo)
            same = sum(1 for a_, b_ in zip(x["mask_hash"], ag["mask_hash"]) if a_ == b_)
            rec = dict(
                logits_vs_ag=e_la,
                argmax_eq_ag=bool(lg.argmax() == la.argmax()),
                hid_vs_ag=e_ha,
                logits_vs_off=e_lo,
                argmax_eq_off=bool(lg.argmax() == lo.argmax()),
                masks_same_layers=same,
                mask_vs_dump=x["mask_vs_dump"],
            )
            out["correctness"].setdefault(s, {})[i] = rec
            md.append(
                f"| {s} | {i} | {e_la[0]:.2e} / {e_la[1]:.2e} | {rec['argmax_eq_ag']} | {e_ha[0]:.2e} | {e_lo[0]:.2e} | "
                f"{rec['argmax_eq_off']} | {same}/32 |"
            )
    json.dump(out, open(f"{O}/summary.json", "w"), indent=1)
    open(f"{O}/summary.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
