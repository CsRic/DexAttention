"""Figure 6(a): attention-call latency of the six workloads (static planning excluded, dynamic planning included).

Each bar's total is the geometric mean of the group's per-input totals (forward + backward; W6 forward only); its
segments keep the proportions of the per-segment geometric means. Labels: per-input geometric mean of (strongest
baseline / Dex), and of (strongest baseline / Dex w/o assigner) in gray.
A bar missing from a group is labelled with the exclusion class recorded for it (TLE, OOM, unsupported, error, no
result). attn_bars.json lists per group and system the ids of the inputs used and of those shared with Dex, and
every other input of the group with its recorded status and reason; a group's inputs are all those any system has a
record of. Per workload and per group it gives the number of inputs with a record and with a complete Dex result next
to the evaluation's input counts (common.N_INPUTS, N_GROUP_INPUTS), so inputs that no system recorded show up as a
shortfall. The recorded IDs are also compared with the evaluation's input list (common.input_ids: the case lists of
workloads/; W2 / W3 need the selection under $DEX_DATA_ROOT, without it only their counts are compared): IDs outside
the list, listed IDs without a record, and IDs outside the drawn groups are stored. Shortfalls, exclusions and such IDs
are printed; check_expected.py turns them into missing or failed values.
Inputs : $DEX_RESULT_ROOT/attention_layer/<w>/...   Outputs: $DEX_RESULT_ROOT/analysis/attn_bars.{pdf,json} and
         ablation_fwd_w456.json (forward-only ratios of W4-W6 consumed by Table 5).
Example: python analysis/fig_attention.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import common as C  # noqa: E402
from bars_common import draw  # noqa: E402
from geomean import gm, gm_pos, ratio_gm, strongest  # noqa: E402

COL = {"Dex": "#1A1A1A", "Dex w/o assigner": "#8C8C8C", "Ring": "#D55E00", "All-gather": "#56B4E9", "Sparse all-gather": "#009E73",
       "DCP": "#0072B2", "UA": "#F0E442", "Magi": "#CC79A7"}
TITLE = {"W1": "W1  Qwen1.5-1.8B $\\Lambda$", "W2": "W2  Qwen1.5-1.8B rollout", "W3": "W3  Qwen3-4B rollout", "W4": "W4  MiniCPM4.1-8B top-k",
         "W5": "W5  Wan2.1-1.3B hybrid", "W6": "W6  Llama-3.1-8B FlexPrefill (fwd)"}
WORK = ("W1", "W2", "W3", "W4", "W5", "W6")


def tint(c, a=0.55):
    """flat lighter shade of c for the backward segment"""
    c = c.lstrip("#"); r, g, b = (int(c[i:i + 2], 16) for i in (0, 2, 4))
    return "#%02x%02x%02x" % tuple(int(v + (255 - v) * a) for v in (r, g, b))


def total(e):
    return e["fwd"] + (e["bwd"] or 0.0)


def complete(W, table):
    """cases with every pass of the workload"""
    return {n: e for n, e in table.items() if e.get("fwd") and (W == "W6" or e.get("bwd"))}


def collect():
    """-> data {W: [(group, {display: segments})]}, speedup {(W, group): {display: (x, baseline)}}, cases, forward cases,
    coverage {(W, group): {system: dict(used [ids], common_with_dex [ids], excluded {id: reason}, na)}},
    inputs {W: dict(expected, recorded, dex_complete (inside the drawn groups), groups {group: dict(expected, recorded,
    dex_complete)}, outside_groups [ids], id_list (whether the input list was available), unexpected [ids],
    absent [ids])}"""
    data, speed, cases, fwd_cases, cover, inputs = {}, {}, {}, {}, {}, {}
    for W in WORK:
        tab = C.combined(W)
        exc = C.excluded(W)
        data[W] = []
        seen = {n for s in C.SYSTEMS[W] for n in list(tab.get(s, {})) + list(exc.get(s, {}))}
        grouped, per = set(), {}
        for grp in C.GROUPS[W]:
            bars, cs, fw = {}, {}, {}
            done = {s: complete(W, C.in_group(tab.get(s, {}), grp)) for s in C.SYSTEMS[W]}
            exg = {s: {n: v for n, v in exc.get(s, {}).items() if C.group_of(W, n) == grp} for s in C.SYSTEMS[W]}
            ref = {n for s in C.SYSTEMS[W] for n in list(C.in_group(tab.get(s, {}), grp)) + list(exg[s])}   # the group's inputs
            grouped |= ref
            per[grp] = dict(expected=C.N_GROUP_INPUTS[W], recorded=len(ref), dex_complete=len(done["dex"]))
            cv = cover[(W, grp)] = {}
            for s in C.SYSTEMS[W]:
                t = done[s]
                ex = {n: "; ".join(sorted({f"{st}: {why}" if why else st for _k, _p, st, why in v})) for n, v in exg[s].items()}
                for n in sorted(ref - set(t)):
                    ex.setdefault(n, "incomplete record" if n in tab.get(s, {}) else "no record")
                why = [w for v in exg[s].values() for _k, _p, _st, w in v]
                cv[s] = dict(used=sorted(t), common_with_dex=sorted(set(done["dex"]) & set(t)), excluded=ex,
                             na=None if t else C.na_kind(why))
                if not t:
                    continue
                e = list(t.values())
                seg = [gm_pos(x["fwd"] - x["fwd_plan"] for x in e), gm_pos(x["fwd_plan"] for x in e),
                       gm_pos((x["bwd"] or 0.0) - x["bwd_plan"] for x in e), gm_pos(x["bwd_plan"] for x in e)]
                tg = gm(total(x) for x in e)
                bars[C.DISPLAY[s]] = tuple(v * tg / sum(seg) for v in seg)
                cs[s] = {n: total(x) for n, x in t.items()}
                fw[s] = {n: x["fwd"] for n, x in t.items()}
            data[W].append((grp, bars))
            bl = [b for b in cs if b not in C.OURS]
            speed[(W, grp)] = {C.DISPLAY[o]: (lambda r: (r[0], C.DISPLAY.get(r[1], r[1])))(strongest(cs, o, bl)) for o in C.OURS if o in cs}
            cases[(W, grp)] = cs; fwd_cases[(W, grp)] = fw
        ids = C.input_ids(W)                                        # None: W2 / W3 without the selection file
        listed = set().union(*ids.values()) if ids else set()
        inputs[W] = dict(expected=C.N_INPUTS[W], recorded=len(seen), dex_complete=sum(p["dex_complete"] for p in per.values()),
                         groups=per, outside_groups=sorted(seen - grouped), id_list=ids is not None,
                         unexpected=sorted(seen - listed) if ids else [], absent=sorted(listed - seen) if ids else [])
    return data, speed, cases, fwd_cases, cover, inputs


def _na(cover, W, grp, display):
    """the recorded exclusion class of a system without a bar in a group ("TLE", "OOM", "unsupported", ...)"""
    for s, c in cover.get((W, grp), {}).items():
        if C.DISPLAY[s] == display:
            return c["na"]
    return None


def segments(s, v):
    f, p, b, q = v
    return [(f, dict(color=COL[s])), (p, dict(color="white", hatch="//////")), (b, dict(color=tint(COL[s]))),
            (q, dict(color="white", hatch="\\\\\\\\\\\\"))]


def main():
    data, speed, cases, fwd_cases, cover, inputs = collect()
    out = C.ensure_out()
    plt.rcParams.update({"hatch.linewidth": 0.5, "font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": 0.7,
                         "xtick.major.width": 0.7, "ytick.major.width": 0.7, "axes.edgecolor": "#333333",
                         "xtick.color": "#222222", "ytick.color": "#222222"})
    order = {W: [C.DISPLAY[s] for s in C.SYSTEMS[W]] for W in WORK}
    LE = dict(edgecolor="black", linewidth=0.5)
    seg_h = [Patch(facecolor="#888888", label="forward", **LE), Patch(facecolor=tint("#888888"), label="backward", **LE),
             Patch(facecolor="white", hatch="//////", label="forward planning", **LE),
             Patch(facecolor="white", hatch="\\\\\\", label="backward planning", **LE)]
    for W in WORK:                                                  # a group without a Dex result is not drawn
        for g, d in data[W]:
            if "Dex" not in d:
                print(f"[coverage] {W} {g}: no complete Dex result, group not drawn")
        data[W] = [(g, d) for g, d in data[W] if "Dex" in d]
    for W, c in inputs.items():                                     # inputs no system recorded, or without a Dex result
        if c["recorded"] != c["expected"] or c["dex_complete"] != c["expected"] or c["outside_groups"]:
            print(f"[coverage] {W}: {c['expected']} inputs in the evaluation, {c['recorded']} with a record, {c['dex_complete']} with a "
                  f"complete Dex result" + (f", {len(c['outside_groups'])} outside the drawn groups" if c["outside_groups"] else ""))
        for g, p in c["groups"].items():
            if p["recorded"] != p["expected"] or p["dex_complete"] != p["expected"]:
                print(f"[coverage] {W} {g}: {p['expected']} inputs in the evaluation, {p['recorded']} with a record, "
                      f"{p['dex_complete']} with a complete Dex result")
        if not c["id_list"]:
            print(f"[coverage] {W}: input IDs not compared with the evaluation's list (no rollouts/selection.json under DEX_DATA_ROOT)")
        for key, text in (("unexpected", "not in the evaluation's input list"), ("absent", "of the evaluation's input list without a record")):
            if c[key]:
                print(f"[coverage] {W}: {len(c[key])} input(s) {text}: {', '.join(c[key][:6])}" + (" ..." if len(c[key]) > 6 else ""))
    for (W, g), cv in cover.items():                                # every exclusion is listed, none is silent
        if not cv.get("dex", {}).get("used"):
            continue                                                # (reported above: the group is not drawn)
        for s, c in cv.items():
            if c["excluded"] or c["na"]:
                print(f"[coverage] {W} {g} {C.DISPLAY[s]}: {len(c['used'])} inputs used, {len(c['common_with_dex'])} shared with Dex, "
                      f"{len(c['excluded'])} excluded" + (f", bar N/A ({c['na']})" if c["na"] else ""))
    fig = plt.figure(figsize=(11.5, 3.5))
    summary = draw(fig, [["W1", "W2", "W3"], ["W4", "W5", "W6"]], data, order, TITLE, COL, ("Dex", "Dex w/o assigner"), segments, "ms",
                   lambda W, s, lab: "N/A: " + (_na(cover, W, lab, s) or "no result"),
                   [(order["W1"], seg_h[:2]), (order["W4"], seg_h)], speedups=speed)
    fig.savefig(os.path.join(out, "attn_bars.pdf"), bbox_inches="tight", pad_inches=0.01)
    pairs = {f"{W} {g}": {f"{C.DISPLAY[a]}/{C.DISPLAY[b]}": round(ratio_gm(cs[a], cs[b]), 3) for a in cs for b in cs if a != b}
             for (W, g), cs in cases.items()}
    cov = {f"{W} {g}": {C.DISPLAY[s]: c for s, c in cv.items()} for (W, g), cv in cover.items()}
    json.dump(dict(data=data, speedup_vs_strongest=summary, pairs=pairs, coverage=cov, inputs=inputs), open(os.path.join(out, "attn_bars.json"), "w"), indent=1)
    fw = {f"{W} {g}": {f"{C.DISPLAY[a]}/{C.DISPLAY[b]}": ratio_gm(cs[a], cs[b]) for a in ("ring", "dex_no_assigner") for b in ("dex_no_assigner", "dex")
                       if a != b and a in cs and b in cs} for (W, g), cs in fwd_cases.items() if W in ("W4", "W5", "W6")}
    json.dump(fw, open(os.path.join(out, "ablation_fwd_w456.json"), "w"), indent=1)
    for k, v in summary.items():
        print(k, v)


if __name__ == "__main__":
    main()
