"""W4 training-step summary: $DEX_RESULT_ROOT/end2end/minicpm/steps.jsonl -> summary.{json,md}. Per length and system the
geometric mean of step time and its attention forward / backward, gradient sync and plan spans over the timed steps, the
peak memory, and the step-time ratio system / Dex over paired steps (same input group), geometric mean.
usage: python experiments/end2end/minicpm_summarize.py"""
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from dex.common.env import cfg  # noqa: E402

SYS = ("dex", "dex_no_assigner", "ring", "allgather", "sparse_allgather")
FIELDS = ("step_ms", "attn_fwd_ms", "attn_bwd_ms", "grad_sync_ms", "plan_total_ms_max")


def gm(x):
    x = list(x)
    return math.exp(sum(map(math.log, x)) / len(x)) if x else float("nan")


def main():
    W = os.path.join(cfg("DEX_RESULT_ROOT"), "end2end", "minicpm")
    R = {}
    for r in map(json.loads, open(f"{W}/steps.jsonl")):
        R.setdefault((r["S_padded"], r["system"]), {})[tuple(r["seqs"])] = r
    out, md = {}, ["# W4 training steps", "", "| length | system | steps | step ms | attn fwd | attn bwd | grad sync | plan | system / Dex | peak GiB |",
                   "|---|---|---|---|---|---|---|---|---|---|"]
    for S in sorted({k[0] for k in R}):
        ref = R.get((S, "dex"), {})
        for s in SYS:
            d = R.get((S, s))
            if not d:
                continue
            v = {f: gm(x[f] for x in d.values()) if all(f in x for x in d.values()) else float("nan") for f in FIELDS}
            common = set(d) & set(ref)
            v["ratio"] = gm(d[g]["step_ms"] / ref[g]["step_ms"] for g in common)
            v["peak_actual_GiB"] = max(x["peak_actual_GiB"] for x in d.values())
            out[f"{S}/{s}"] = dict(n=len(d), **v)
            md.append(f"| {S} | {s} | {len(d)} | " + " | ".join(f"{v[f]:.1f}" for f in FIELDS) + f" | {v['ratio']:.3f} | {v['peak_actual_GiB']:.2f} |")
    open(f"{W}/summary.md", "w").write("\n".join(md) + "\n")
    json.dump(out, open(f"{W}/summary.json", "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
