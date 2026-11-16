"""Tables 1, 2, 4, 5, 6, 7, the component study of the ablation section, the numerical-agreement statistics and the launch-to-launch variability, from the result
records under $DEX_RESULT_ROOT. Each table is printed as a LaTeX tabular body and written to analysis/<name>.tex, its
numbers to analysis/<name>.json (compared with expected_ranges.json by check_expected.py).
Example: python analysis/tables.py            (everything)
         python analysis/tables.py table5 table6 component_internal
"""
import collections
import decimal
import glob
import json
import os
import re
import statistics
import sys

import common as C
from geomean import gm, gm_pos, percentile, ratio_gm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPN, P = 4, 16


def emit(name, tex, data):
    out = C.ensure_out()
    open(os.path.join(out, f"{name}.tex"), "w").write(tex + "\n")
    json.dump(data, open(os.path.join(out, f"{name}.json"), "w"), indent=1, default=str)
    print(f"% ---- {name}\n{tex}\n")


def rng(values):
    v = [x for x in values if x is not None]
    return (min(v), max(v)) if v else (float("nan"), float("nan"))


def half_up(x, digits=1):
    """decimal rounding with ties away from zero, on the value as printed in the records (7.85 -> 7.9)"""
    return float(decimal.Decimal(str(round(x, 6))).quantize(decimal.Decimal(1).scaleb(-digits), rounding=decimal.ROUND_HALF_UP))


def plan_range(lo, hi):
    """a planning-time range in seconds, as many digits as its magnitude needs"""
    a = f"{lo:.2f}" if lo < 0.1 else f"{lo:.1f}"
    b = f"{hi:.2f}" if hi < 1 else (f"{hi:.1f}" if hi < 100 else f"{hi:.0f}")
    return f"{a}--{b}"


# ---------------------------------------------------------------- planning (Table 1, Table 4 planning column)
def _planning_file(name):
    """records of <root>/planning_latency/<name>: {"records": {key: record}} (baselines_summary.py) or {input id: record}"""
    f = os.path.join(C.ROOT, "planning_latency", name)
    if not os.path.exists(f):
        return []
    d = json.load(open(f))
    d = d.get("records", d) if isinstance(d, dict) else d
    return [dict(v, name=k) for k, v in d.items()] if isinstance(d, dict) else d


def _dynamic_call_ms(system):
    """planning time of every individual call of W4-W6, per pass, metadata exchange and tables included (ms). W4: the slowest
    rank's plan phase of a case and launch. W5: the sum of the plan stages of a case and launch. W6: per case, the median over
    launches (repeats of the same input) of the forward latency minus that of the kernel-only call"""
    out = []
    for W in ("W4", "W5"):
        for lk in C.operator(W).get(system, {}).values():
            for r in lk.values():
                if W == "W4":
                    out += [r["fwd_span"], r["bwd_span"]]
                else:
                    st = r["stages"]
                    out += [sum(st.get(k, 0.0) for k in ("mask_ms", "planner_fwd_ms", "plan_bcast_ms", "pairs_sched_tables_ms")),
                            sum(st.get(k, 0.0) for k in C.BWD_PLAN)]
    launches = C.operator("W6").get(system, {})
    for n in {c for lk in launches.values() for c in lk}:
        rr = [lk[n]["raw"] for lk in launches.values() if n in lk]
        out.append(statistics.median(r["fwd_ms"] for r in rr) - statistics.median(r["kernel_ms"] for r in rr))
    return out


def planning():
    """planning ranges; Dex's dynamic range is min-max over individual calls of W4-W6"""
    res = {}
    fs = _planning_file("static_plan_time_hs4.json")
    res["dex_static_ms"] = rng(r.get("plan_choose_ms") for r in fs)
    res["dex_dynamic_ms"] = rng(_dynamic_call_ms("dex"))
    res["dex_no_assigner_dynamic_ms"] = rng(_dynamic_call_ms("dex_no_assigner"))
    assign = {}
    for W in ("W5", "W6"):
        for lk in C.operator(W).get("dex", {}).values():
            for r in lk.values():
                s = r["stages"]
                assign.setdefault((W, r["group"]), []).append(sum(s.get(k, 0.0) for k in ("planner_fwd_ms", "planner_ms", "gate_ms")))
    res["assignment_w5_w6_ms"] = rng(gm_pos(v) for v in assign.values())
    ua = _planning_file("ua.json"); dcp = _planning_file("dcp.json")
    res["ua_s"] = rng(r["plan_s"] for r in ua if r.get("plan_s")); res["ua_timeouts"] = [sum(1 for r in ua if not r.get("plan_s")), len(ua)]
    res["dcp_s"] = rng(r["plan_s"] for r in dcp if r.get("plan_s"))
    res["magi_s"] = rng(r["plan_s"] for r in _planning_file("magi.json") if r.get("plan_s") is not None)
    return res


def online_cost(p):
    """exposed planning of the training steps: W4 recorded per step; W5 per-call stages x 120 attention calls per step"""
    res = {}
    t4 = C.training("W4")
    res["w4_exposed_pct"] = {g: 100 * gm(r["plan_total_ms_max"] / r["step_ms"] for r in t4[("dex", g)]) for g in ("32K", "64K", "128K") if ("dex", g) in t4}
    t5 = C.training("W5")
    calls = [r for lk in C.operator("W5").get("dex", {}).values() for r in lk.values()]
    for g in ("480p", "720p"):
        rr = [r for r in calls if r["group"] == g]
        if not rr or ("dex", g) not in t5:
            continue
        step = gm(r["step_ms"] for r in t5[("dex", g)])
        res.setdefault("w5_exposed_fwd_pct", {})[g] = 100 * 120 * gm(r["fwd_plan"] for r in rr) / step
        res.setdefault("w5_hidden_bwd_pct", {})[g] = 100 * 120 * gm(r["bwd_plan"] for r in rr) / step
    return res


def table1():
    p = planning()
    tex = "\n".join([
        f"UA & {plan_range(*p['ua_s'])} s \\\\",
        f"DCP & {plan_range(*p['dcp_s'])} s \\\\",
        f"Magi & {C.fmt_range(p['magi_s'][0] * 1e3, p['magi_s'][1] * 1e3)} ms \\\\",
        f"Dex & {C.fmt_range(*p['dex_dynamic_ms'])} ms \\\\",
        f"% Dex without assigner, per pass: {C.fmt_range(*p['dex_no_assigner_dynamic_ms'])} ms"])
    emit("table1", tex, p)


# ---------------------------------------------------------------- Table 2
def scheduler_latencies():
    """Calibrated fixed overheads for Table 2, not fresh full-transfer measurements.
    Payload time is excluded; these come from config/calibration.json links.
    """
    links = json.load(open(os.path.join(REPO, "config", "calibration.json")))["links"]
    return links["lat_intra_s"], links["lat_inter_s"]


def table2():
    d = os.path.join(C.ROOT, "microbench")
    fa2 = {r["S"]: r for r in C.read_jsonl(os.path.join(d, "compute_fa2.jsonl"))}
    comm = C.read_jsonl(os.path.join(d, "comm.jsonl"))
    us = lambda pat, tag, b: next(r["us"] for r in comm if r["pattern"] == pat and r["tag"] == tag and r["bytes"] == b)   # noqa: E731
    rows, data = [], {}
    for S, mib in ((128, 1), (2048, 16), (8192, 64)):
        b = mib << 20
        f = fa2[S]
        intra, inter = us("pair", "nccl_default", b) / 1e3, us("inter", "nccl_gdr", b) / 1e3
        rows.append(f"$B={S}$, KV {mib} MiB & {f['fwd_ms']:.2f} ({f['fwd_tflops']:.1f}) & {intra:.2f} & {inter:.2f} \\\\" if inter < 10 else f"$B={S}$, KV {mib} MiB & {f['fwd_ms']:.2f} ({f['fwd_tflops']:.1f}) & {intra:.2f} & {inter:.1f} \\\\")
        data[str(S)] = dict(fa2_ms=f["fwd_ms"], tflops=f["fwd_tflops"], intra_ms=intra, inter_ms=inter)
    floor = 64 << 10
    fl_e = [r["us"] / 1e3 for r in comm if r["pattern"] == "inter" and r["impl"] == "nccl" and r["bytes"] <= floor]
    data["floor"] = dict(intra_ms=us("pair", "nccl_default", floor) / 1e3, inter_ms=rng(fl_e))
    rows.append(f"NCCL floor ($\\leq64$ KiB) & --- & {data['floor']['intra_ms']:.2f} & {min(fl_e):.2f}--{max(fl_e):.2f} \\\\")
    ours = {r["S"]: r for r in C.read_jsonl(os.path.join(d, "dex_fwd.jsonl"))}
    S = 8192
    tile_ms = ours[S]["fwd_ms"] / (S // 128) ** 2
    tfl = 4.0 * S * S * 16 * 128 / ours[S]["fwd_ms"] / 1e9
    li, le = scheduler_latencies()
    data["dex"] = dict(tile_ms=tile_ms, tflops=tfl, intra_ms=li * 1e3, inter_ms=le * 1e3, launch_ms=ours[128]["fwd_ms"])
    rows.append(f"Dex ($128^2$ tile) & {tile_ms:.4f} ({tfl:.0f}) & {li * 1e3:.3f} & {le * 1e3:.3f} \\\\")
    emit("table2", "\n".join(rows), data)


# ---------------------------------------------------------------- Table 4
def _recv(pattern):
    out = {}
    for f in glob.glob(os.path.join(C.ROOT, "comm_volume", "recv", pattern)):
        for r in C.read_jsonl(f):
            if "recv" in r and (r.get("system") != "UA" or "fwd" in r.get("passes", [])):
                out[r["case"]] = r
    return out


def nic(rec):
    e = rec["recv"]["fwd"]["inter"]
    return max(sum(e[n * GPN:(n + 1) * GPN]) for n in range(P // GPN)) / float(1 << 20)


def table4():
    SRC = {"Dex": "dex_{}.jsonl", "Ring": "ring_{}.jsonl", "DCP": "dcp_{}.jsonl", "UA": "ua_{}.jsonl", "Magi": "magi_{}.jsonl", "All-gather": None}
    res = dict(nic={}, imbalance={}, direct={})
    for W in ("W1", "W2", "W3"):
        rec = {s: _recv(p.format(W)) for s, p in SRC.items() if p}
        common = set.intersection(*[set(v) for v in rec.values()])
        for s, v in rec.items():
            res["nic"][f"{W}|{s}"] = gm(nic(v[c]) for c in common)
        res.setdefault("n_common", {})[W] = len(common)
        df = {r["case"]: r for r in C.read_jsonl(os.path.join(C.ROOT, "comm_volume", "direct_fetch", f"{W}.jsonl"))}
        res["direct"][W] = gm(df[c]["direct_MiB"] for c in common if c in df)
    for W in ("W4", "W5"):
        rec = {s: _recv(f"{W.lower()}_{k}.jsonl") for s, k in (("Dex", "dex"), ("Ring", "ring"), ("All-gather", "allgather"))}
        common = set.intersection(*[set(v) for v in rec.values()])
        for s, v in rec.items():
            res["nic"][f"{W}|{s}"] = gm(nic(v[c]) for c in common)
        res.setdefault("n_common", {})[W] = len(common)
        df = {r["case"]: r for r in C.read_jsonl(os.path.join(C.ROOT, "comm_volume", "direct_fetch", f"{W}.jsonl"))}
        res["direct"][W] = gm(df[c]["direct_MiB"] for c in common if c in df)
    # W6: Dex's message tables recorded by the operator test; Ring / All-gather analytic (K + V fp16 of 8 KV heads):
    # one rank's bytes L, All-gather 3/4 of all K / V enters a node (12 L), Ring (P - 1) L cross one node boundary
    w6 = [r for r in C.operator("W6").get("dex", {}).get(1, {}).values()]
    if w6:
        res["nic"]["W6|Dex"] = gm(r["raw"]["bytes"]["inter_node_max_MiB"] for r in w6)
        L = [r["raw"]["S_pad"] / P * 8 * 128 * 2 * 2 / float(1 << 20) for r in w6]
        res["nic"]["W6|All-gather"] = gm(12 * x for x in L)
        res["nic"]["W6|Ring"] = gm((P - 1) * x for x in L)
        df = C.read_jsonl(os.path.join(C.ROOT, "comm_volume", "direct_fetch", "W6.jsonl"))
        res["direct"]["W6"] = gm(r["direct_MiB"] for r in df)
    for W in ("W1", "W2", "W3", "W4", "W5", "W6"):
        f = os.path.join(C.ROOT, "comm_volume", "balance", f"{W}.json")
        if not os.path.exists(f):
            continue
        bal = json.load(open(f))
        systems = sorted({s for c in bal.values() for s in c})
        common = [c for c, d in bal.items() if all("fwd" in d.get(s, {}) for s in systems)]
        for s in systems:
            res["imbalance"][f"{W}|{s}"] = gm(bal[c][s]["fwd"]["imb"] for c in common)
    p = planning()
    res["planning"] = p
    res["online_cost"] = online_cost(p)
    res["measured_over_direct"] = {W: res["nic"][f"{W}|Dex"] / res["direct"][W] for W in res["direct"] if f"{W}|Dex" in res["nic"]}
    res["less_than_allgather"] = {W: 1 - res["nic"][f"{W}|Dex"] / res["nic"][f"{W}|All-gather"] for W in ("W4", "W6") if f"{W}|All-gather" in res["nic"]}
    WS = ("W1", "W2", "W3", "W4", "W5", "W6")
    cell = lambda t, W, s, f: f.format(res[t][f"{W}|{s}"]) if f"{W}|{s}" in res[t] else "--"   # noqa: E731
    lines = []
    for s, plan in (("Dex", f"{C.fmt_range(*p['dex_static_ms'])}\\,ms & {C.fmt_range(*p['dex_dynamic_ms'])}\\,ms"), ("direct", None),
                    ("Ring", "-- & --"), ("All-gather", "-- & --"),
                    ("Magi", f"{plan_range(*p['magi_s'])}\\,s & --"), ("DCP", f"{plan_range(*p['dcp_s'])}\\,s & --"),
                    ("UA", f"{plan_range(*p['ua_s'])}\\,s & --")):
        if s == "direct":
            lines.append("\\quad direct fetch & & & & & & & & & " + " & ".join(f"{res['direct'].get(W, float('nan')):.0f}" for W in WS) + " \\\\")
            continue
        lines.append(f"{s} & {plan} & " + " & ".join(cell("imbalance", W, s, "{:.2f}") for W in WS) + " & "
                     + " & ".join(cell("nic", W, s, "{:.0f}") for W in WS) + " \\\\")
    emit("table4", "\n".join(lines), res)


# ---------------------------------------------------------------- Table 5
def _component_rows():
    """{(x, row): {input: {launch: forward ms}}} of experiments/ablation_components (Magi's native row: its padded layout)"""
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    for f in glob.glob(os.path.join(C.ROOT, "ablation_components", "*.jsonl")):
        m = re.fullmatch(r"(\w+?)_(row\d|native)_launch(\d+)\.jsonl", os.path.basename(f))
        if not m:
            continue
        row = 1 if m.group(2) == "native" else int(m.group(2)[3:])
        for r in C.read_jsonl(f):
            if r.get("pass", "fwd") == "fwd" and r.get("lat_ms") is not None:
                out[(m.group(1), row)][r["input_id"]][int(m.group(3))] = r["lat_ms"]
    return out


def table5():
    data = _component_rows()
    tabs = {W: C.combined(W) for W in ("W1", "W2", "W3")}
    ours = {n: e["fwd"] for t in tabs.values() for n, e in t.get("dex", {}).items() if e.get("fwd")}
    native = {s: {n: e["fwd"] for t in tabs.values() for n, e in t.get(s, {}).items() if e.get("fwd")} for s in ("ring", "ua", "dcp", "magi")}
    res, lines = {}, ["\\multicolumn{5}{c}{\\textit{Static masks (W1--W3)}} \\\\"]
    names = {"ua": "UA", "dcp": "DCP", "magi": "Magi", "ring": "Ring"}
    for x in ("ua", "dcp", "magi", "ring"):
        row3 = {n: statistics.median(v.values()) for n, v in data[(x, 3)].items()}
        inputs = sorted(row3)
        rel = {}
        for row in (1, 2, 3):
            vals = {n: statistics.median(v.values()) for n, v in data[(x, row)].items()} if (x, row) in data else {}
            if row == 1:
                vals = vals or {n: native[x][n] for n in inputs if n in native[x]}
            perw = {}
            for W in ("w1", "w2", "w3"):
                v = [vals[n] / ours[n] for n in inputs if n.startswith(W + "_") and n in vals and n in ours]
                if v:
                    perw[W] = gm(v)
            rel[row] = gm(perw.values()) if len(perw) == 3 else float("nan")
        res[x] = rel
        res[x + "_wins"] = sum(1 for n in inputs if row3[n] < ours[n]); res[x + "_n"] = len(inputs)
        if x == "ring":
            lines.append(f"Ring & {rel[1]:.2f} & \\multicolumn{{2}}{{c}}{{{rel[3]:.2f}}} & 1.00 \\\\")
        else:
            lines.append(f"{names[x]} & {rel[1]:.2f} & {rel[2]:.2f} & {rel[3]:.2f} & 1.00 \\\\")
    fw = json.load(open(os.path.join(C.OUT, "ablation_fwd_w456.json")))
    per = collections.defaultdict(dict)
    for key, d in fw.items():
        per[key.split()[0]][key] = d
    pick = lambda k: gm(gm(d[k] for d in g.values()) for g in per.values())   # noqa: E731
    res["dynamic"] = dict(ring_native=pick("Ring/Dex"), executor=pick("Dex w/o assigner/Dex"))
    lines += ["\\midrule", "\\multicolumn{5}{c}{\\textit{Dynamic masks (W4--W6)}} \\\\",
              f"Ring & {res['dynamic']['ring_native']:.2f} & \\multicolumn{{2}}{{c}}{{{res['dynamic']['executor']:.2f}}} & 1.00 \\\\"]
    emit("table5", "\n".join(lines), res)


# ---------------------------------------------------------------- data reuse, relay, transfer order (text of the ablation section)
def component_internal():
    """experiments/ablation_components/internal_run.sh: per arm and pass, geometric mean over inputs of arm / full. Inputs an arm
    lacks are left out of its ratio, and the number of inputs entering each ratio is recorded."""
    d = os.path.join(C.ROOT, "ablation_components", "internal")
    lat = {}
    for f in glob.glob(os.path.join(d, "*.jsonl")):
        m = re.fullmatch(r"(\w+)_(fwd|bwd)\.jsonl", os.path.basename(f))
        acc = collections.defaultdict(list)
        for r in C.read_jsonl(f):
            if r.get("lat_ms") is not None:
                acc[r["input_id"]].append(r["lat_ms"])
        lat[(m.group(1), m.group(2))] = {n: statistics.median(v) for n, v in acc.items()}
    res = {}
    for arm in ("load_only", "no_relay", "fifo"):
        res[arm] = {}
        for ps in ("fwd", "bwd"):
            a, f = lat[(arm, ps)], lat[("full", ps)]
            ns = sorted(set(a) & set(f))
            res[arm][ps] = gm(a[n] / f[n] for n in ns)
            res[arm][f"{ps}_n"] = len(ns)
    res["fifo_max_change_pct"] = max(abs(res["fifo"][ps] - 1) * 100 for ps in ("fwd", "bwd"))
    names = dict(load_only="Assignment by load alone", no_relay="Gateway relay off", fifo="Message-generation send order")
    lines = [f"{names[x]} & {res[x]['fwd']:.2f} & {res[x]['bwd']:.2f} \\\\" for x in names]
    lines.append(f"% inputs: load only {res['load_only']['fwd_n']} / {res['load_only']['bwd_n']}; "
                 f"largest generation-order change {res['fifo_max_change_pct']:.2f}%")
    emit("component_internal", "\n".join(lines), res)


# ---------------------------------------------------------------- Table 6
def _static_decomposed(W):
    """{case: {pass: [ms per launch]}} of the decomposed executor on W1-W3"""
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in sorted(glob.glob(os.path.join(C.ROOT, "ablation_executor", W.lower(), "decomposed_launch*.jsonl"))):
        for r in C.read_jsonl(f):
            if r.get("fwd_ms") is not None:
                n = r.get("input_id") or r["name"]
                out[n]["fwd"].append(r["fwd_ms"]); out[n]["bwd"].append(r["bwd_ms"])
    return out


def _ratios(dec, ours):
    """geometric means over cases: forward, backward, summed per-case latency. dec / ours[case] = dict(fwd, bwd)"""
    ns = [n for n in dec if n in ours]
    return dict(n=len(ns), fwd=gm(dec[n]["fwd"] / ours[n]["fwd"] for n in ns), bwd=gm(dec[n]["bwd"] / ours[n]["bwd"] for n in ns),
                total=gm((dec[n]["fwd"] + dec[n]["bwd"]) / (ours[n]["fwd"] + ours[n]["bwd"]) for n in ns))


def _per_case_gm(acc):
    return {n: {k: gm(v) for k, v in d.items()} for n, d in acc.items() if min(len(v) for v in d.values()) == 2}


def table6():
    res = {}
    for W in ("W1", "W2", "W3"):
        fl = C.operator(W).get("dex", {})
        fa = collections.defaultdict(lambda: collections.defaultdict(list))
        for lk in fl.values():
            for n, e in lk.items():
                if e.get("fwd") and e.get("bwd"):
                    fa[n]["fwd"].append(e["fwd"]); fa[n]["bwd"].append(e["bwd"])
        res[W] = _ratios(_per_case_gm(_static_decomposed(W)), _per_case_gm(fa))
    # W4: Dex's measured planning span removed from both latencies per case and launch; the decomposed executor's own
    # step-list construction stays in its time, so these are upper bounds
    d4 = C.operator("W4")
    if "decomposed" in d4:
        fa, da, incl = collections.defaultdict(lambda: collections.defaultdict(list)), collections.defaultdict(lambda: collections.defaultdict(list)), []
        for k, lk in d4["dex"].items():
            for n, e in lk.items():
                x = d4["decomposed"].get(k, {}).get(n)
                if x is None:
                    continue
                for ps in ("fwd", "bwd"):
                    fa[n][ps].append(e[ps] - e[f"{ps}_span"]); da[n][ps].append(x[ps] - e[f"{ps}_span"])
                incl.append(x["fwd"] / e["fwd"])
        res["W4"] = _ratios(_per_case_gm(da), _per_case_gm(fa))
        res["W4_fwd_incl_planning"] = gm(incl)
    # W5: planning stages removed from both sides; each backward plan consumes a gathered mask, so the backward mask gather
    # is removed on both sides (Dex's equals its forward gather, the decomposed executor records its own)
    d5 = C.operator("W5")
    if "decomposed" in d5:
        fa, da = collections.defaultdict(lambda: collections.defaultdict(list)), collections.defaultdict(lambda: collections.defaultdict(list))
        for k, lk in d5["dex"].items():
            for n, e in lk.items():
                p = e["stages"]
                fa[n]["fwd"].append(e["fwd"] - (p["mask_ms"] + p["planner_fwd_ms"] + p["plan_bcast_ms"] + p["pairs_sched_tables_ms"]))
                fa[n]["bwd"].append(e["bwd"] - (p["mask_ms"] + p["planner_bwd_ms"] + p["bwd_pairs_sched_tables_ms"]))
        for lk in d5["decomposed"].values():
            for n, x in lk.items():
                q = x["stages"]
                da[n]["fwd"].append(x["fwd"] - (q["fwd_mask_ms"] + q["planner_fwd_ms"] + q["plan_bcast_ms"] + q["fwd_decomp_tables_ms"]))
                da[n]["bwd"].append(x["bwd"] - (q["bwd_mask_ms"] + q["planner_bwd_ms"] + q["bwd_decomp_tables_ms"]))
        res["W5"] = _ratios(_per_case_gm(da), _per_case_gm(fa))
    # W6: forward only, kernel-only call against Dex's kernel (cached tables on both sides), per length then overall
    d6 = C.operator("W6")
    if "decomposed" in d6:
        kern = lambda s: {n: gm(lk[n]["raw"]["kernel_ms"] for lk in d6[s].values() if n in lk) for n in d6[s][1]}   # noqa: E731
        kf, kd = kern("dex"), kern("decomposed")
        res["W6"] = dict(fwd=gm(ratio_gm({n: v for n, v in kd.items() if C.group_of("W6", n) == g}, kf) for g in C.GROUPS["W6"]),
                         n=len(set(kd) & set(kf)))
    lines = []
    for W in ("W1", "W2", "W3", "W4", "W5", "W6"):
        if W not in res:
            continue
        r = res[W]
        tag = "$^{*}$" if W == "W4" else ""
        lines.append(f"{W}{tag} & {r['fwd']:.2f} & " + (f"{r['bwd']:.2f} & {r['total']:.2f} \\\\" if "bwd" in r else "-- & -- \\\\"))
    emit("table6", "\n".join(lines), res)


# ---------------------------------------------------------------- Table 7
def table7():
    """Peak GPU memory. A cell without a memory record prints "--". It prints OOM only where the training launch left a
    failure record of kind OOM (C.training_failures); _unavailable lists the kind of every recorded failure. The cells the
    article reports as unavailable are reference values of expected_ranges.json, not filled in here."""
    res, lines, why = {}, [], {}
    cols = ("dex", "ring", "allgather", "dcp", "ua", "magi")
    for W in ("W2", "W3", "W4", "W5"):
        tr = C.training(W)
        fl = C.training_failures(W)
        for g in {"W5": ("480p", "720p")}.get(W, ("32K", "64K", "128K")):
            row = {}
            for s in cols:
                pk = [r["peak_actual_GiB"] for r in tr.get((s, g), []) if r.get("peak_actual_GiB") is not None]
                if pk:
                    row[s] = max(pk)
                elif (s, g) in fl:
                    why.setdefault(f"{W}|{g}", {})[s] = fl[(s, g)]
            res[f"{W}|{g}"] = row
    for s, ins in C.prefill().items():
        for inp, rr in ins.items():
            row = res.setdefault(f"W6|{inp.rsplit('_', 1)[1].upper()}", {})
            for r in rr.values():
                if s in cols and "mem_max_over_ranks" in r:
                    row[s] = max(row.get(s, 0.0), r["mem_max_over_ranks"]["peak_actual_GiB"])
    for k, row in res.items():
        W, g = k.split("|")
        cell = lambda s: f"{half_up(row[s]):.1f}" if s in row else ("OOM" if why.get(k, {}).get(s) == "OOM" else "--")   # noqa: E731
        lines.append(f"{W} & {g} & " + " & ".join(cell(s) for s in cols) + " \\\\")
    overhead_train = [r["dex"] - r["ring"] for k, r in res.items() if not k.startswith("W6") and "ring" in r and "dex" in r]
    overhead_w6 = [r["dex"] - r["ring"] for k, r in res.items() if k.startswith("W6") and "ring" in r and "dex" in r]
    res["_dex_over_ring_training_GiB"] = rng(overhead_train)
    res["_dex_over_ring_w6_GiB"] = rng(overhead_w6)
    res["_unavailable"] = why
    emit("table7", "\n".join(lines), res)


# ---------------------------------------------------------------- numerical agreement and variability
def correctness():
    res = {}
    for W in ("W1", "W2", "W3", "W4", "W5", "W6"):
        for s, lk in C.operator(W).items():
            if s == "decomposed" or 1 not in lk:
                continue
            e = [v["rel_l2"] for r in lk[1].values() for v in r["err"].values() if isinstance(v, dict)]
            if e:
                res[f"{W}|{s}"] = dict(worst_rel_l2=max(e), checks=len(e))
    loss = []
    for W in ("W2", "W3", "W4", "W5"):
        tr = C.training(W)
        for (s, g), rows in tr.items():
            if s == "dex":
                continue
            ref = {tuple(r.get("inputs") or [i]): r for i, r in enumerate(tr.get(("dex", g), []))}
            for i, r in enumerate(rows):
                x = ref.get(tuple(r.get("inputs") or [i]))
                key = "loss_per_token" if "loss_per_token" in r else "loss"
                if x and key in r and key in x:
                    loss.append(abs(r[key] - x[key]) / abs(x[key]))
    res["training_loss_max_rel_diff"] = max(loss) if loss else None
    pf = C.prefill()
    match = [x["vs_official"]["argmax_match"] for s in pf for rr in pf[s].values() for x in rr.values() if x.get("vs_official")]
    res["w6_greedy_token_match"] = [sum(map(bool, match)), len(match)]
    lines = [f"{k} & {v['worst_rel_l2']:.1e} \\\\" for k, v in res.items() if isinstance(v, dict)]
    emit("correctness", "\n".join(lines), res)


def variability():
    """launch-to-launch: |launch 2 / launch 1 - 1| per case, pass and system, over all operator tests and W6 prefill"""
    dev = []
    for W in ("W1", "W2", "W3", "W4", "W5", "W6"):
        for s, lk in C.operator(W).items():
            if s == "decomposed" or 1 not in lk or 2 not in lk:
                continue
            for n in set(lk[1]) & set(lk[2]):
                for k in ("fwd", "bwd"):
                    a, b = lk[1][n].get(k), lk[2][n].get(k)
                    if a and b:
                        dev.append(abs(b / a - 1) * 100)
                a, b = lk[1][n]["raw"].get("fwd_online_ms"), lk[2][n]["raw"].get("fwd_online_ms")   # forward with selection
                if a and b:
                    dev.append(abs(b / a - 1) * 100)
    for s, ins in C.prefill().items():
        for rr in ins.values():
            if 1 in rr and 2 in rr:
                dev.append(abs(C.prefill_latency(rr[2]) / C.prefill_latency(rr[1]) - 1) * 100)
    res = dict(n=len(dev), median_pct=statistics.median(dev) if dev else None, p95_pct=percentile(dev, 95), max_pct=max(dev) if dev else None)
    emit("variability", f"% n = {res['n']}, median {res['median_pct']:.1f}%, p95 {res['p95_pct']:.1f}%, max {res['max_pct']:.1f}%", res)


ALL = dict(table1=table1, table2=table2, table4=table4, table5=table5, component_internal=component_internal, table6=table6, table7=table7, correctness=correctness, variability=variability)


def main():
    failed = []
    for k in sys.argv[1:] or ALL:
        try:
            ALL[k]()
        except (OSError, KeyError, ValueError, statistics.StatisticsError) as e:
            failed.append(k)
            print(f"% ---- {k}: missing results ({type(e).__name__}: {e})", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
