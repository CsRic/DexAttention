"""Compares the regenerated numbers (analysis/*.json under $DEX_RESULT_ROOT) with expected_ranges.json. Run after
fig_attention.py, fig_end2end.py, fig_plan_scaling.py, tables.py and mask_stats.py.

The values to compare are enumerated from expected_ranges.json and the workloads' groups, never from what a summary
happens to contain: a summary file that is absent, empty, or lacks a group or an entry yields missing values, not a
skipped check. Every compared value ends in one of four states: pass (within the tolerance), fail (outside it, or not a
finite number), missing (not regenerated) and not applicable (the article reports no value there, null in
expected_ranges.json). All four are counted and every fail and missing value is listed.
Input coverage of the operator tests (attn_bars.json): per length / resolution group the number of inputs with a complete
Dex result must equal the evaluation's (fewer = missing, more = fail); an input outside the drawn groups or outside the
evaluation's input list is a fail. Where fig_attention.py had no input list (W2 / W3 without the selection under
$DEX_DATA_ROOT) only the counts are compared, which is reported. The training headline is compared only when it covers
every training configuration (fewer = missing).
Three entries of expected_ranges.json are reference values and are not compared: the UA planning timeouts (their count
depends on which inputs cross the time limit on the machine), the Figure 7 planning time (read from the figure) and the
training configurations the article reports as unavailable (training_reported_unavailable; printed next to what the
records state, which is the only source of the regenerated table and figure).
Example: python analysis/check_expected.py             full reproduction: exit status 1 on any fail or missing value
         python analysis/check_expected.py --partial   a subset of the experiments: exit status 1 on fails only
"""
import json
import math
import os
import sys

import common as C

EXP = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "expected_ranges.json")))
TOL = EXP["rel_tolerance"]
bad, missing, notes = [], [], []           # missing: (summary, text); notes: printed, not counted
ABSENT = set()                             # summaries without a file
count = dict(passed=0, failed=0, missing=0, not_applicable=0)
_cur = [None]                              # the summary the following checks read
WS = ("W1", "W2", "W3", "W4", "W5", "W6")


def load(name):
    """the summary's object; {} when the file is absent or holds no object, so that each of its expected values is
    counted as missing"""
    _cur[0] = name
    f = os.path.join(C.OUT, f"{name}.json")
    if not os.path.exists(f):
        ABSENT.add(name)
        return {}
    d = json.load(open(f))
    return d if isinstance(d, dict) else {}


def get(d, *keys):
    """nested lookup (dict keys, list positions), None when any level is absent"""
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        elif isinstance(d, (list, tuple)) and isinstance(k, int) and k < len(d):
            d = d[k]
        else:
            return None
    return d


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _check(tag, got, ok, desc):
    if got is None:
        missing.append((_cur[0], f"{tag}: not regenerated")); count["missing"] += 1
    elif not _num(got) or not math.isfinite(got):
        bad.append(f"{tag}: regenerated {got!r} is not a finite number"); count["failed"] += 1
    elif not ok(got):
        bad.append(f"{tag}: regenerated {got:.4g}, {desc}"); count["failed"] += 1
    else:
        count["passed"] += 1


def cmp(tag, got, want, tol=TOL):
    if want is None:
        count["not_applicable"] += 1
        return
    _check(tag, got, lambda g: abs(g - want) <= tol * abs(want), f"reported {want:.4g}")


def within(tag, got, lo, hi, tol=TOL):
    _check(tag, got, lambda g: lo * (1 - tol) <= g <= hi * (1 + tol), f"reported range {lo:.4g}-{hi:.4g}")


def speedups():
    for fig, key in (("attn_bars", "fig6a_speedup_vs_strongest"), ("end2end", "fig6b_speedup_vs_strongest")):
        d = load(fig)
        for grp, want in EXP[key].items():
            cmp(f"{fig} {grp} Dex", get(d, "speedup_vs_strongest", grp, 0), want[0])
            if len(want) > 1:
                cmp(f"{fig} {grp} w/o assigner", get(d, "speedup_vs_strongest", grp, 2), want[1])
        if fig == "attn_bars":
            coverage(d)
    n, n_all = d.get("n_training_configs"), sum(len(C.GROUPS[W]) for W in ("W2", "W3", "W4", "W5"))
    if d.get("headline_training_geomean") is not None and _num(n) and n < n_all:
        missing.append((_cur[0], f"training headline: covers {n} of the {n_all} training configurations")); count["missing"] += 1
    else:
        cmp("training headline", d.get("headline_training_geomean"), EXP["training_headline"])
    for k, sec, W in (("w4_non_attention_share", "w4_non_attention", "W4"), ("w6_selection_share", "w6_selection", "W6"),
                      ("w6_assignment_added_fraction", "w6_assignment_added_fraction", "W6")):
        for g in C.GROUPS[W]:
            within(f"{k} {g}", get(d, sec, g), *EXP[k], tol=0.1)


def coverage(d):
    """input coverage of the operator tests, from the inputs section of attn_bars.json"""
    for W in WS:
        c = get(d, "inputs", W)
        want = C.N_GROUP_INPUTS[W]
        for g in C.GROUPS[W]:              # per group: inputs with a complete Dex result
            n = get(c, "groups", g, "dex_complete")
            if _num(n) and math.isfinite(n) and n < want:
                missing.append((_cur[0], f"{W} {g}: {n} of {want} inputs have a complete Dex result")); count["missing"] += 1
            else:
                cmp(f"{W} {g} inputs with a complete Dex result", n, want, tol=0.0)
        for key, text in (("outside_groups", "outside the drawn groups"), ("unexpected", "not in the evaluation's input list")):
            v = get(c, key)
            if not isinstance(v, list):
                missing.append((_cur[0], f"{W} inputs {text}: not regenerated")); count["missing"] += 1
            elif v:
                bad.append(f"{W}: {len(v)} input(s) {text}: {', '.join(map(str, v[:6]))}" + (" ..." if len(v) > 6 else ""))
                count["failed"] += 1
            else:
                count["passed"] += 1
        if get(c, "id_list") is False:
            notes.append(f"{W}: input IDs not compared with the evaluation's list (no rollouts/selection.json under "
                         f"DEX_DATA_ROOT when fig_attention.py ran); counts only")


def tables():
    t1 = load("table1"); e = EXP["table1_table4_planning"]
    for k in ("ua_s", "dcp_s", "magi_s", "dex_static_ms", "dex_dynamic_ms", "dex_no_assigner_dynamic_ms", "assignment_w5_w6_ms"):
        within(f"planning {k} low", get(t1, k, 0), e[k][0], e[k][1], tol=0.1)
        within(f"planning {k} high", get(t1, k, 1), e[k][0], e[k][1], tol=0.1)
    t2 = load("table2")
    for S, want in EXP["table2"].items():
        for k, v in want.items():
            if not isinstance(v, list):
                cmp(f"table2 {S} {k}", get(t2, S, k), v, tol=0.2)
    t4 = load("table4"); e4 = EXP["table4"]
    for s, vals in e4["imbalance"].items():
        for W, v in zip(WS, vals):
            cmp(f"imbalance {W} {s}", get(t4, "imbalance", f"{W}|{s}"), v, tol=0.03)
    for s, vals in e4["nic_mib"].items():
        for W, v in zip(WS, vals):
            cmp(f"NIC {W} {s}", get(t4, "nic", f"{W}|{s}"), v, tol=0.03)
    for W, v in zip(WS, e4["direct_fetch_mib"]):
        cmp(f"direct fetch {W}", get(t4, "direct", W), v, tol=0.03)
        within(f"measured / direct fetch {W}", get(t4, "measured_over_direct", W), *e4["measured_over_direct"], tol=0.1)
    for W, v in e4["less_than_allgather"].items():
        cmp(f"NIC bytes below All-gather {W}", get(t4, "less_than_allgather", W), v, tol=0.1)
    pct = e4["exposed_planning_pct"]
    for k, e, W in (("w4_exposed_pct", pct["w4"], "W4"), ("w5_exposed_fwd_pct", pct["w5_fwd"], "W5"), ("w5_hidden_bwd_pct", pct["w5_hidden_bwd"], "W5")):
        for g in C.GROUPS[W]:
            within(f"{k} {g}", get(t4, "online_cost", k, g), *e, tol=0.1)
    t5 = load("table5")
    for x in ("ua", "dcp", "magi", "ring"):
        for row, v in zip(("1", "2", "3"), EXP["table5"][x]):
            cmp(f"table5 {x} column {row}", get(t5, x, row), v)
    for k, v in EXP["table5"]["dynamic"].items():
        cmp(f"table5 dynamic {k}", get(t5, "dynamic", k), v)
    ci = load("component_internal"); e = EXP["component_internal"]
    cmp("load-only forward", get(ci, "load_only", "fwd"), e["load_only"][0])
    cmp("load-only backward", get(ci, "load_only", "bwd"), e["load_only"][1])
    cmp("no relay forward", get(ci, "no_relay", "fwd"), e["no_relay"][0])
    cmp("no relay backward", get(ci, "no_relay", "bwd"), e["no_relay"][1])
    within("message-generation order, largest change %", ci.get("fifo_max_change_pct"), 0.0, e["fifo_max_change_pct"])
    ms = load("mask_stats"); e = EXP["mask_stats"]
    for tag, W, key, want, tol, groups in (
            ("W4 largest-GPU coverage", "W4", "largest_gpu_coverage", "w4_largest_gpu_coverage", 0.02, ((0, "lowest"), (1, "highest"))),
            ("W5 density", "W5", "density_per_resolution", "w5_density", 0.02, [(g, g) for g in C.GROUPS["W5"]]),
            ("W6 density", "W6", "density_per_length", "w6_density", 0.15, [(g, g) for g in C.GROUPS["W6"]]),
            ("W6 need / All-gather", "W6", "need_over_allgather_per_length", "w6_need_over_allgather", 0.02, [(g, g) for g in C.GROUPS["W6"]])):
        for g, name in groups:
            within(f"mask statistics {tag} {name}", get(ms, W, key, g), *e[want], tol=tol)
    t6 = load("table6")
    for W in WS:
        for k, v in zip(("fwd", "bwd", "total"), EXP["table6"][W]):
            cmp(f"table6 {W} {k}", get(t6, W, k), v)
    cmp("table6 W4 forward including planning", t6.get("W4_fwd_incl_planning"), EXP["table6"]["W4_fwd_incl_planning"])
    t7 = load("table7")
    for k, vals in EXP["table7_gib"].items():
        if k.startswith("_"):
            continue
        for s, v in zip(EXP["table7_gib"]["_order"], vals):
            cmp(f"table7 {k} {s}", get(t7, k, s), v, tol=0.05)
    e2e = load("end2end")
    for k, ref in EXP["training_reported_unavailable"].items():   # reference values: the records decide, not this list
        if k.startswith("_"):
            continue
        for s, kind in ref.items():
            got = get(t7, "_unavailable", k, s) or get(e2e, "unavailable", k.replace("|", " "), C.DISPLAY[s], "kind")
            val = get(t7, k, s)
            notes.append(f"{k} {C.DISPLAY[s]}: the article reports {kind}; the records "
                         + (f"hold a value ({val})" if val is not None else f"state {got}" if got else "state no reason (no result)"))
    va = load("variability")               # informative: printed next to the reported values, but it must be regenerated
    for k in ("median_pct", "p95_pct"):
        _check(f"launch-to-launch variability {k}", va.get(k), lambda g: g >= 0, "a negative percentage")
    if all(_num(va.get(k)) and math.isfinite(va[k]) for k in ("median_pct", "p95_pct")):
        notes.append(f"launch-to-launch: median {va['median_pct']:.2f}%, p95 {va['p95_pct']:.2f}% "
                     f"(reported {EXP['variability']['median_pct']}%, {EXP['variability']['p95_pct']}%)")
    co = load("correctness"); e = EXP["correctness"]
    for W in ("W1", "W2", "W3", "W4", "W5"):
        lim = e["dex_w1_w3_worst_rel_l2" if W in ("W1", "W2", "W3") else "dex_w4_w5_worst_rel_l2"]
        within(f"correctness {W} Dex", get(co, f"{W}|dex", "worst_rel_l2"), 0.0, lim, tol=0.1)
    within("training loss, largest relative difference", co.get("training_loss_max_rel_diff"), 0.0, e["training_loss_max_rel_diff"], tol=0.1)


def main():
    partial = "--partial" in sys.argv[1:]
    speedups(); tables()
    print(f"{count['passed']} passed, {count['failed']} failed, {count['missing']} missing, "
          f"{count['not_applicable']} not applicable")
    for t in notes:
        print("  NOTE     " + t)
    for b in bad:
        print("  FAIL     " + b)
    by = {}
    for name, text in missing:
        by.setdefault(name, []).append(text)
    for name, items in by.items():
        if name in ABSENT:
            print(f"  MISSING  {name}.json: summary file not found ({len(items)} expected values)")
        else:
            for t in items:
                print("  MISSING  " + t)
    if count["passed"] + count["failed"] == 0:
        print("no regenerated value was compared")
        return 1
    if bad or (missing and not partial):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
