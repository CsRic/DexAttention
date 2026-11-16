"""Figure 7: assigner and scheduler time vs cluster size (weak scaling: 8K tokens per GPU, 4 GPUs per node, 1-32 nodes).

Lines: forward solid, backward dashed. Rollout and top-k / top-p are families of real masks sampled 3 times per point
(seeded packing order): line = geometric mean over the family's workloads of their geometric mean over samples, band =
min-max over all samples. W6 is forward only, so its backward points are not used.
Inputs : $DEX_RESULT_ROOT/plan_scaling/*.jsonl (experiments/plan_scaling/plan_scaling.py)
Outputs: $DEX_RESULT_ROOT/analysis/plan_scaling.{pdf,json}
Example: python analysis/fig_plan_scaling.py
"""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import common as C  # noqa: E402
from geomean import gm  # noqa: E402

LINES = [("lambda", r"$\Lambda$", "#0072B2", "o"), ("rollout", "rollout", "#1A1A1A", "^"), ("topk", "top-k/top-p", "#D55E00", "D")]
FAMS = dict(lambda_=[("w1", True)], rollout=[("w2", True), ("w3", True)], topk=[("w4", True), ("w5", True), ("w6", False)])
KEYS = dict(planner=dict(fwd="planner_fwd_ms", bwd="planner_bwd_ms"), scheduler=dict(fwd="meta_fwd_ms_max", bwd="meta_bwd_ms_max"))


def collect():
    rows = [r for f in sorted(glob.glob(os.path.join(C.ROOT, "plan_scaling", "*.jsonl"))) for r in C.read_jsonl(f)]
    data = {}
    for name, fams in FAMS.items():
        fam_names = [f for f, _ in fams]
        Ps = sorted({r["P"] for r in rows if r["workload"] in fam_names})
        out = dict(nodes=[P // 4 for P in Ps], families={})
        for k, kk in KEYS.items():
            out[k], out[k + "_lo"], out[k + "_hi"] = {}, {}, {}
            for d, c in kk.items():
                line, lo, hi = [], [], []
                for P in Ps:
                    per_fam, allv = [], []
                    for f, has_bwd in fams:
                        if d == "bwd" and not has_bwd:
                            continue
                        v = [r[c] for r in rows if r["workload"] == f and r["P"] == P and r.get(c) is not None]
                        if v:
                            per_fam.append(gm(v)); allv += v
                            out["families"].setdefault(f, {}).setdefault(f"{k}_{d}", {})[P // 4] = round(gm(v), 3)
                    line.append(gm(per_fam)); lo.append(min(allv)); hi.append(max(allv))
                out[k][d], out[k + "_lo"][d], out[k + "_hi"][d] = line, lo, hi
        data[name.rstrip("_")] = out
    return data


def main():
    data = collect()
    out = C.ensure_out()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": 0.7})
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.0), sharey=True, gridspec_kw=dict(wspace=0.06))
    for ax, key, title in zip(axes, ("planner", "scheduler"), ("Assigner", "Scheduler")):
        for w, lab, col, mk in LINES:
            d = data[w]
            if w != "lambda":
                for dd, al in (("fwd", 0.16), ("bwd", 0.08)):
                    ax.fill_between(d["nodes"], d[key + "_lo"][dd], d[key + "_hi"][dd], color=col, alpha=al, lw=0)
            ax.plot(d["nodes"], d[key]["fwd"], color=col, marker=mk, ms=3.2, lw=1.2, label=lab)
            ax.plot(d["nodes"], d[key]["bwd"], color=col, marker=mk, ms=3.2, lw=1.0, ls="--", mfc="white")
        ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 16, 32]); ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
        ax.set_xlabel("Nodes (4 GPUs each)")
        ax.text(0.03, 0.96, title, transform=ax.transAxes, ha="left", va="top", fontweight="bold")
        ax.grid(axis="y", which="major", color="#e6e6e6", lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Time (ms)")
    axes[0].set_yscale("log", base=2)
    axes[0].set_ylim(0.35, 140)
    axes[0].set_yticks([0.5, 1, 2, 4, 8, 16, 32, 64, 128]); axes[0].set_yticklabels(["0.5", "1", "2", "4", "8", "16", "32", "64", "128"])
    axes[0].yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    h1 = axes[1].legend(loc="upper left", bbox_to_anchor=(0.0, 0.86), ncol=3, frameon=False, fontsize=8.5, handlelength=1.6, columnspacing=1.0)
    axes[1].add_artist(h1)
    axes[1].legend(handles=[Line2D([], [], color="#555", lw=1.2, label="forward"), Line2D([], [], color="#555", lw=1.0, ls="--", label="backward")],
                   loc="upper left", bbox_to_anchor=(0.0, 0.72), ncol=2, frameon=False, fontsize=8.5, handlelength=1.6, columnspacing=1.0)
    fig.savefig(os.path.join(out, "plan_scaling.pdf"), bbox_inches="tight", pad_inches=0.01)
    json.dump(data, open(os.path.join(out, "plan_scaling.json"), "w"), indent=1)
    t = data["topk"]
    print(f"top-k / top-p at {t['nodes'][-1]} nodes: assigner fwd {t['planner']['fwd'][-1]:.1f} ms, bwd {t['planner']['bwd'][-1]:.1f} ms; "
          f"range fwd {t['planner_lo']['fwd'][-1]:.1f}-{t['planner_hi']['fwd'][-1]:.1f}, bwd {t['planner_lo']['bwd'][-1]:.1f}-{t['planner_hi']['bwd'][-1]:.1f}")


if __name__ == "__main__":
    main()
