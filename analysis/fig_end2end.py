"""Figure 6(b): complete training steps (W2-W5) and whole-model prefill (W6).

A bar's total is the geometric mean over the timed steps (W6: over the inputs of a length); its segments keep the
proportions of the geometric means of attention forward, attention backward and the rest. Labels: per-step (W6: per-input)
geometric mean of (strongest baseline / ours), steps paired by input group (W4: by position in the shared sorted
sequence list). Also prints the headline over the
training configurations, the non-attention share of W4 steps and the selection share of W6 prefill.
A system with step records in a group is always drawn. One without is labelled N/A with the reason a record states
(unavailable(): the failure record of its training launch, or the operator-test records showing that a baseline has no
plan for an input of the group); without such a record the label is "N/A: no result". No reason is assumed.
Inputs : $DEX_RESULT_ROOT/end2end/...   Outputs: $DEX_RESULT_ROOT/analysis/end2end.{pdf,json}
Example: python analysis/fig_end2end.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import common as C  # noqa: E402
from bars_common import draw  # noqa: E402
from fig_attention import COL, tint  # noqa: E402
from geomean import gm, ratio_gm, strongest  # noqa: E402

TRAIN_ORDER = {"W2": ("dex", "ring", "allgather", "dcp", "ua", "magi"), "W3": ("dex", "ring", "allgather", "dcp", "ua", "magi"),
               "W4": ("dex", "dex_no_assigner", "ring", "allgather", "sparse_allgather"),
               "W5": ("dex", "dex_no_assigner", "ring", "allgather", "sparse_allgather")}
W6_ORDER = ("dex", "dex_no_assigner", "ring", "allgather", "sparse_allgather")
GROUPS = {"W2": ("32K", "64K", "128K"), "W3": ("32K", "64K", "128K"), "W4": ("32K", "64K", "128K"), "W5": ("480p", "720p"), "W6": ("32K", "64K", "128K")}
TITLE = {"W2": "W2  Qwen1.5-1.8B rollout", "W3": "W3  Qwen3-4B rollout", "W4": "W4  MiniCPM4.1-8B top-k", "W5": "W5  Wan2.1-1.3B hybrid",
         "W6": "W6  Llama-3.1-8B FlexPrefill (prefill)"}


def unavailable(W, g, s, failures):
    """(class, source) of a system without a training step in a group, from records only; None when no record says why.
    1. the failure record its training launch left (experiments/end2end/qwen_run.sh);
    2. W2 / W3: the operator-test records of the group's inputs. A training step needs the baseline's forward plan of every
       input of the group, so an input recorded there without one (status other than ok on the forward) rules the step out."""
    if (s, g) in failures:
        return failures[(s, g)], "training failure record"
    if W in ("W2", "W3"):
        why = [w for n, v in C.excluded(W).get(s, {}).items() if C.group_of(W, n).lstrip("~") == g
               for _k, ps, _st, w in v if ps == "fwd"]
        if why:
            return C.na_kind(why), "operator-test records"
    return None


def step_key(r, i):
    """pairs a step across systems: its input group when recorded, otherwise its position"""
    return tuple(r["inputs"]) if r.get("inputs") else i


def collect():
    data, cases, extra = {}, {}, dict(w4_non_attention={}, w6_selection={}, unavailable={})
    for W, order in TRAIN_ORDER.items():
        tr = C.training(W)
        fl = C.training_failures(W)
        data[W] = []
        for g in GROUPS[W]:
            d, cs = {}, {}
            for s in order:
                rows = tr.get((s, g), [])
                if not rows:
                    u = unavailable(W, g, s, fl)
                    if u:
                        extra["unavailable"].setdefault(f"{W} {g}", {})[C.DISPLAY[s]] = dict(kind=u[0], source=u[1])
                    continue
                seg = [gm(r[k] / 1e3 for r in rows) for k in ("attn_fwd_ms", "attn_bwd_ms")]
                st = gm(r["step_ms"] / 1e3 for r in rows)
                seg.append(max(st - sum(seg), 0.0))
                d[C.DISPLAY[s]] = tuple(x * st / sum(seg) for x in seg)
                cs[s] = {step_key(r, i): r["step_ms"] / 1e3 for i, r in enumerate(rows)}
                if W == "W4" and s == "dex":
                    extra["w4_non_attention"][g] = gm((r["step_ms"] - r["attn_fwd_ms"] - r["attn_bwd_ms"]) / r["step_ms"] for r in rows)
            data[W].append((g, d)); cases[(W, g)] = cs
    pf = C.prefill()
    data["W6"] = []
    for g in GROUPS["W6"]:
        tag = "_" + g.lower()
        d, cs = {}, {}
        for s in W6_ORDER:
            ins = {i: rr for i, rr in pf.get(s, {}).items() if i.endswith(tag)}
            if not ins:
                continue
            lat = {i: gm(C.prefill_latency(x) for x in rr.values()) for i, rr in ins.items()}
            bd = lambda k: gm(gm(x["breakdown_mean_ms"][k] for x in rr.values()) for rr in ins.values())   # noqa: E731
            tot = gm(lat.values()) / 1e3; att = bd("attention") / 1e3
            d[C.DISPLAY[s]] = (att, 0.0, tot - att)
            cs[s] = {i: v / 1e3 for i, v in lat.items()}
            if s == "dex":
                extra["w6_selection"][g] = bd("selection") / 1e3 / tot
        data["W6"].append((g, d)); cases[("W6", g)] = cs
    return data, cases, extra


def main():
    data, cases, extra = collect()
    out = C.ensure_out()
    speed = {}
    for (W, g), cs in cases.items():
        bl = [b for b in cs if b not in C.OURS]
        speed[(W, g)] = {C.DISPLAY[o]: (lambda r: (r[0], C.DISPLAY.get(r[1], r[1])))(strongest(cs, o, bl)) for o in C.OURS if o in cs}
    plt.rcParams.update({"hatch.linewidth": 0.5, "font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": 0.7,
                         "xtick.major.width": 0.7, "ytick.major.width": 0.7, "axes.edgecolor": "#333333",
                         "xtick.color": "#222222", "ytick.color": "#222222"})
    order = {W: [C.DISPLAY[s] for s in o] for W, o in TRAIN_ORDER.items()}
    order["W6"] = [C.DISPLAY[s] for s in W6_ORDER]
    LE = dict(edgecolor="black", linewidth=0.5)
    seg_h = [Patch(facecolor="#888888", label="attention forward", **LE), Patch(facecolor=tint("#888888"), label="attention backward", **LE),
             Patch(facecolor="#FFFFFF", label="rest of the step", **LE)]

    def segments(s, v):
        return [(v[0], dict(color=COL[s])), (v[1], dict(color=tint(COL[s]))), (v[2], dict(color="#FFFFFF"))]
    for W in data:
        data[W] = [(g, d) for g, d in data[W] if "Dex" in d]
    fig = plt.figure(figsize=(11.5, 3.5))
    summary = draw(fig, [["W2", "W3"], ["W4", "W5", "W6"]], data, order, TITLE, COL, ("Dex", "Dex w/o assigner"), segments, "s",
                   lambda W, s, lab: "N/A: " + extra["unavailable"].get(f"{W} {lab}", {}).get(s, {}).get("kind", "no result"),
                   [(order["W2"], seg_h), (order["W4"], seg_h)], speedups=speed,
                   label_shift={("W4", "32K"): 0.06}, inside={("W4", g, "Dex w/o assigner") for g in GROUPS["W4"]})
    fig.savefig(os.path.join(out, "end2end.pdf"), bbox_inches="tight", pad_inches=0.01)
    train = [v["Dex"][0] for (W, g), v in speed.items() if W != "W6" and "Dex" in v and v["Dex"][1]]
    headline = gm(train) if train else None                # None: no training configuration has a baseline yet
    pairs = {f"{W} {g}": {f"{C.DISPLAY[a]}/{C.DISPLAY[b]}": round(ratio_gm(cs[a], cs[b]), 3) for a in cs for b in cs if a != b}
             for (W, g), cs in cases.items()}
    w6_cost = {g: ratio_gm(cases[("W6", g)]["dex"], cases[("W6", g)]["dex_no_assigner"]) - 1 for g in GROUPS["W6"]
               if {"dex", "dex_no_assigner"} <= set(cases[("W6", g)])}
    extra["w6_assignment_added_fraction"] = w6_cost
    json.dump(dict(data=data, speedup_vs_strongest=summary, headline_training_geomean=headline, n_training_configs=len(train),
                   pairs=pairs, **extra), open(os.path.join(out, "end2end.json"), "w"), indent=1)
    for k, v in summary.items():
        print(k, v)
    print(f"training headline: {headline:.2f}x over {len(train)} configurations" if train else "training headline: no configuration with a baseline")
    print("W4 non-attention share:", {k: round(v, 3) for k, v in extra["w4_non_attention"].items()})
    print("W6 selection share:", {k: round(v, 3) for k, v in extra["w6_selection"].items()})
    print("W6 assignment adds:", {k: round(v, 3) for k, v in w6_cost.items()})
    for k, v in extra["unavailable"].items():
        for s, u in v.items():
            print(f"[coverage] {k} {s}: no training step, {u['kind']} ({u['source']})")


if __name__ == "__main__":
    main()
