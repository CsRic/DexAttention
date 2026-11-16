"""W5 end-to-end summary: $DEX_RESULT_ROOT/end2end/wan/steps.jsonl -> summary.{json,md}.
Per resolution and system: mean / median step time, attention forward / backward, peak memory, and the per-step speedup
of Dex against every other system on the same input groups (geometric mean over the 4 steps).
Diagnostic only: the article's values come from analysis/ (tables.py, fig_*.py), which uses geometric means across
inputs and does not read this summary.
usage: python experiments/end2end/wan_summarize.py"""

import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from dex.common.env import cfg  # noqa: E402

SYS = ("dex", "dex_no_assigner", "ring", "allgather", "sparse_allgather")
RES = {32760: "480p", 75600: "720p"}


def gm(x):
    x = list(x)
    return math.exp(sum(map(math.log, x)) / len(x)) if x else float("nan")


def main():
    O = os.path.join(cfg("DEX_RESULT_ROOT"), "end2end", "wan")
    recs = [json.loads(x) for x in open(f"{O}/steps.jsonl")]
    out, md = {}, [
        "# W5 end-to-end training steps (Wan2.1-T2V-1.3B)",
        "",
        "| res | system | step s mean / median | attn fwd s | attn bwd s | peak GiB | Dex speedup (geomean over steps) |",
        "|---|---|---|---|---|---|---|",
    ]
    for S, rs in sorted(RES.items()):
        cur = {s: {tuple(r["inputs"]): r for r in recs if r["S"] == S and r["system"] == s} for s in SYS}
        for s in SYS:
            rr = list(cur[s].values())
            if not rr:
                continue
            st = [r["step_ms"] / 1e3 for r in rr]
            keys = sorted(set(cur[s]) & set(cur["dex"]))
            sp = gm(cur[s][k]["step_ms"] / cur["dex"][k]["step_ms"] for k in keys) if keys else float("nan")
            out.setdefault(rs, {})[s] = dict(
                step_s_mean=float(np.mean(st)),
                step_s_median=float(np.median(st)),
                attn_fwd_s=float(np.mean([r["attn_fwd_ms"] for r in rr])) / 1e3,
                attn_bwd_s=float(np.mean([r["attn_bwd_ms"] for r in rr])) / 1e3,
                peak_actual_GiB=rr[0].get("peak_actual_GiB"),
                dex_speedup=sp,
                losses=[round(cur[s][k]["loss"], 5) for k in sorted(cur[s])],
            )
            o = out[rs][s]
            md.append(
                f"| {rs} | {s} | {o['step_s_mean']:.2f} / {o['step_s_median']:.2f} | {o['attn_fwd_s']:.2f} | "
                f"{o['attn_bwd_s']:.2f} | {o['peak_actual_GiB']} | {'-' if s == 'dex' else f'{sp:.2f}x'} |"
            )
    json.dump(out, open(f"{O}/summary.json", "w"), indent=1)
    open(f"{O}/summary.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
