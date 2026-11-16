"""Result loading of the summarizers: every table and figure is regenerated from the records under $DEX_RESULT_ROOT
(JSON lines); no absolute path and no raw dump is read here. The loaders below define the record fields; run instructions are in docs/reproduction.md.

  operator(W)  {system: {launch: {case: record}}} with fwd, bwd, fwd_plan, bwd_plan (ms), group, err, raw filled in
  combined(W)  {system: {case: {fwd, bwd, fwd_plan, bwd_plan, group}}}: launches combined per workload rule
  excluded(W)  {system: {case: [(launch, pass, status, reason)]}}: W1-W3 records left out of operator(W)
  na_kind(reasons)  the exclusion class of a list of reasons: "TLE", "unsupported", "OOM", "error" or "no result"
  training(W)  training-step records of W2-W5 grouped by (system, group), in file order
  training_failures(W)  {(system, group): kind} of the training launches recorded as failed
  input_ids(W) the operator tests' input IDs per group, from the case lists of workloads/ (W2 / W3: from the selection
               under $DEX_DATA_ROOT when it is there)
"""
import glob
import json
import math
import os
import re
import statistics

from geomean import gm, gm_pos

ROOT = os.environ.get("DEX_RESULT_ROOT", "results")
OUT = os.path.join(ROOT, "analysis")
DISPLAY = dict(dex="Dex", dex_no_assigner="Dex w/o assigner", ring="Ring", allgather="All-gather",
               sparse_allgather="Sparse all-gather", dcp="DCP", ua="UA", magi="Magi")
OURS = ("dex", "dex_no_assigner")
SYSTEMS = {"W1": ("dex", "ring", "dcp", "ua", "magi"), "W2": ("dex", "ring", "dcp", "ua", "magi"),
           "W3": ("dex", "ring", "dcp", "ua", "magi"),
           "W4": ("dex", "dex_no_assigner", "ring", "allgather", "sparse_allgather"),
           "W5": ("dex", "dex_no_assigner", "ring", "allgather", "sparse_allgather"),
           "W6": ("dex", "dex_no_assigner", "ring", "allgather", "sparse_allgather")}
GROUPS = {"W1": ("32K", "64K", "128K", "256K"), "W2": ("~32K", "~64K", "~128K"), "W3": ("~32K", "~64K", "~128K"),
          "W4": ("32K", "64K", "128K"), "W5": ("480p", "720p"), "W6": ("32K", "64K", "128K")}
N_INPUTS = {"W1": 4, "W2": 24, "W3": 24, "W4": 144, "W5": 42, "W6": 36}   # inputs of the operator tests per workload
N_GROUP_INPUTS = {"W1": 1, "W2": 8, "W3": 8, "W4": 48, "W5": 21, "W6": 12}  # ... and per length / resolution group
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# planning stages of the dynamic workloads: drawn as the hatched bar segments (W5: the mask gather stays in execution;
# W4 / W6: everything before the kernel is planning, the gather included, see _normalize_w4 / _normalize_dynamic)
FWD_PLAN = ("planner_fwd_ms", "planner_ms", "gate_ms", "plan_bcast_ms", "pairs_sched_tables_ms")
BWD_PLAN = ("planner_bwd_ms", "bwd_pairs_sched_tables_ms")


def ensure_out():
    os.makedirs(OUT, exist_ok=True)
    return OUT


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def group_of(W, name, S=None):
    """length or resolution group of an input"""
    m = re.search(r"(?:^|_)(\d+)k(?:_|$)", name)
    if W == "W5":
        for g in ("480p", "720p"):
            if name.startswith(g):
                return g
        return "480p" if (S or 0) < 50000 else "720p"
    if W == "W4":
        for tag, g in (("131072", "128K"), ("65536", "64K"), ("32768", "32K")):
            if f"_{tag}" in name:
                return g
    if m:
        g = f"{m.group(1)}K"
        return "~" + g if W in ("W2", "W3") else g
    if S:
        return f"{round(S / 1024)}K"
    return None


def _launch_files(W, system):
    """(launch, path) of a dynamic workload's system: <w>/<system>/{rep,launch}<k>.jsonl"""
    d = os.path.join(ROOT, "attention_layer", W.lower(), system)
    out = []
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        m = re.fullmatch(r"(?:rep|launch)(\d+)\.jsonl", os.path.basename(f))
        if m:
            out.append((int(m.group(1)), f))
    return out


def _normalize_w4(r):
    """experiments/attention_layer/w4_op.py record: parts exec / bwd / online, each with its timed calls and phases;
    the hatched planning segment is the median over timed calls of the mean over ranks of the plan phase"""
    def plan(p):
        runs = p["runs"][r["warmup"]:]
        return statistics.median(statistics.mean(x.get("plan", 0.0) for x in c["phases"]) for c in runs) * 1e3
    err = {}
    for part in ("exec", "bwd"):
        err.update({k: v for k, v in ((r[part].get("check") or {}).get("fp64") or {}).items() if isinstance(v, dict)})
    return dict(name=r["name"], fwd=r["exec"]["median_ms"], bwd=r["bwd"]["median_ms"], group=group_of("W4", r["name"], r.get("S")),
                stages={}, err=err, fwd_plan=plan(r["exec"]), bwd_plan=plan(r["bwd"]), fwd_span=r["exec"]["plan_span_ms"], bwd_span=r["bwd"]["plan_span_ms"],
                raw=dict(fwd_online_ms=r["online"]["median_ms"]))


def _normalize_dynamic(W, r):
    """one case of W4 / W5 / W6"""
    if W == "W4":
        return _normalize_w4(r)
    stages = r.get("plan_ms") if isinstance(r.get("plan_ms"), dict) else {}
    fwd = r.get("fwd_exec_ms", r.get("fwd_ms"))
    out = dict(name=r["name"], fwd=fwd, bwd=r.get("bwd_ms"), group=r.get("group") or group_of(W, r["name"], r.get("S")),
               stages=stages, err=r.get("err_fp64") or {}, raw=r)
    fp = sum(stages.get(k, 0.0) for k in FWD_PLAN)
    bp = sum(stages.get(k, 0.0) for k in BWD_PLAN)
    if W == "W6" and r.get("kernel_ms") is not None:      # W6: everything before the kernel is planning
        fp = fwd - r["kernel_ms"]
    out.update(fwd_plan=fp or 0.0, bwd_plan=bp or 0.0)
    return out


_EXCLUDED = {}


def excluded(W):
    """{system: {case: [(launch, pass, status, reason)]}} of the W1-W3 records operator(W) leaves out (status other
    than ok, or no latency); filled by operator(W)"""
    if W not in _EXCLUDED:
        operator(W)
    return _EXCLUDED.get(W, {})


def na_kind(reasons):
    """exclusion class of one system on one group from its recorded reasons (an empty list = no record at all)"""
    if not reasons:
        return "no result"
    low = [str(r).lower() for r in reasons]
    for kind, keys in (("TLE", ("did not finish within", "timeout", "timed out")), ("OOM", ("out of memory", "oom")),
                       ("unsupported", ("fewer than", "not supported", "unsupported"))):
        if all(any(k in x for k in keys) for x in low):
            return kind
    return "error"


def operator(W):
    """{system: {launch: {case: normalized record}}}"""
    res = {}
    if W in ("W1", "W2", "W3"):
        exc = _EXCLUDED[W] = {}
        for f in glob.glob(os.path.join(ROOT, "attention_layer", W.lower(), "*_launch*.jsonl")):
            m = re.fullmatch(r"(\w+?)_(fwd|bwd)_launch(\d+)\.jsonl", os.path.basename(f))
            if not m:
                continue
            s, ps, k = m.group(1), m.group(2), int(m.group(3))
            for r in read_jsonl(f):
                if r.get("status", "ok") != "ok" or r.get("lat_ms") is None:
                    exc.setdefault(s, {}).setdefault(r.get("input_id") or "?", []).append(
                        (k, ps, r.get("status", "ok"), r.get("reason") or ("no latency" if r.get("status", "ok") == "ok" else "")))
                    continue
                n = r["input_id"]
                e = res.setdefault(s, {}).setdefault(k, {}).setdefault(n, dict(
                    name=n, group=group_of(W, n, r.get("S")), fwd_plan=0.0, bwd_plan=0.0, err={}, raw=r))
                e[ps] = r["lat_ms"]
                for t, v in (r.get("fp64_rel_l2") or {}).items():
                    e["err"][t] = dict(rel_l2=v)
        return res
    for s in SYSTEMS[W] + ("decomposed",):
        for k, f in _launch_files(W, s):
            for r in read_jsonl(f):
                res.setdefault(s, {}).setdefault(k, {})[r["name"]] = _normalize_dynamic(W, r)
    return res


def _combine(W, vals):
    """launches of one case (repeats of the same call): median on W1-W3, geometric mean on W4-W6; with the two launches
    of the evaluation the W1-W3 median is their arithmetic mean"""
    return statistics.median(vals) if W in ("W1", "W2", "W3") else gm(vals)


def combined(W, data=None):
    """{system: {case: dict(fwd, bwd, fwd_plan, bwd_plan, group)}} over the launches"""
    data = data or operator(W)
    out = {}
    for s, launches in data.items():
        for n in {c for d in launches.values() for c in d}:
            recs = [d[n] for d in launches.values() if n in d]
            e = dict(group=recs[0]["group"])
            for k in ("fwd", "bwd"):
                v = [r[k] for r in recs if r.get(k) is not None]
                e[k] = _combine(W, v) if v else None
            for k in ("fwd_plan", "bwd_plan"):
                e[k] = gm_pos(r.get(k) for r in recs)
            out.setdefault(s, {})[n] = e
    return out


def in_group(table, group):
    return {n: e for n, e in table.items() if e["group"] == group}


def input_ids(W):
    """{group: set of input IDs} of the operator tests as the records name them, or None when the list is not available:
    W1 from its four lengths, W4 - W6 from the case lists of workloads/, W2 / W3 from the selection that
    workloads/w2_w3_rollout/select_groups.py writes under $DEX_DATA_ROOT (None without it)"""
    if W == "W1":
        ids = [f"w1_{k}k_sink128_win4k" for k in (32, 64, 128, 256)]
    elif W in ("W2", "W3"):
        f = os.path.join(os.environ.get("DEX_DATA_ROOT", ""), "rollouts", "selection.json")
        if not os.path.exists(f):
            return None
        ids = [f"{W.lower()}_{c['split']}_p{c['idx']}" for c in json.load(open(f))]
    else:
        f = {"W4": "w4_infllm/cases144.txt", "W5": "w5_sparge/cases42.txt", "W6": "w6_flexprefill/cases36.txt"}[W]
        ids = []
        for ln in open(os.path.join(REPO, "workloads", f)):
            if not ln.strip():
                continue
            if W == "W4":
                ids.append(ln.strip())                              # <sequence>:<layer>
            elif W == "W5":
                g, c = ln.split()
                ids.append(f"{g}_c{int(c):05d}")
            else:
                i, L = ln.strip().split(":")
                ids.append(f"{i}_L{int(L):02d}")
    out = {}
    for n in ids:
        out.setdefault(group_of(W, n), set()).add(n)
    return out


# ---------------------------------------------------------------- training steps and prefill
def _train_files(W):
    """the step files of a workload. noclear.jsonl is left out: it holds the W2 / W3 steps of Magi without clearing the
    allocator cache (experiments/memory/magi_cache.sh), which only experiments/memory/magi_cache_summary.py reads"""
    d = {"W2": "w2", "W3": "w3", "W4": "minicpm", "W5": "wan"}[W]
    return sorted(f for f in glob.glob(os.path.join(ROOT, "end2end", d, "*.jsonl")) if os.path.basename(f) != "noclear.jsonl")


def _train_group(W, r):
    if r.get("group"):
        return r["group"]
    if W == "W5":
        return "480p" if r["S"] < 50000 else "720p"
    if W == "W4":
        return {32768: "32K", 65536: "64K", 131072: "128K"}[r["S_padded"]]
    return group_of(W, r["inputs"][0]).lstrip("~")


def training(W):
    """{(system, group): [step records in file order]}"""
    out = {}
    for f in _train_files(W):
        for r in read_jsonl(f):
            if "step_ms" in r:
                out.setdefault((r.get("system") or r.get("attn"), _train_group(W, r)), []).append(r)
    return out


def training_failures(W):
    """{(system, group): kind} of the training launches that ended without a step and left a record with status
    "failed" (experiments/end2end/qwen_run.sh). kind is the record's own ("OOM", "TLE", "unsupported" or "error"), set by
    experiments/end2end/failure_record.py from the launch's log; records of one group that disagree give "error"."""
    kinds = {}
    for f in _train_files(W):
        for r in read_jsonl(f):
            if "step_ms" not in r and r.get("status") == "failed":
                kinds.setdefault((r.get("system"), _train_group(W, r)), set()).add(r.get("kind") or "error")
    return {k: v.pop() if len(v) == 1 else "error" for k, v in kinds.items()}


def prefill():
    """W6 prefill: {system: {input: {launch: record}}} from end2end/llama/<system>/rep<k>.jsonl"""
    out = {}
    for f in glob.glob(os.path.join(ROOT, "end2end", "llama", "*", "*.jsonl")):
        m = re.fullmatch(r"(?:rep|launch)(\d+)\.jsonl", os.path.basename(f))
        if m:
            s = os.path.basename(os.path.dirname(f))
            for r in read_jsonl(f):
                out.setdefault(s, {}).setdefault(r["input"], {})[int(m.group(1))] = r
    return out


def prefill_latency(rec):
    """one launch of one input: geometric mean of its timed prefills"""
    return gm(rec["prefill_all"])


def fmt_range(lo, hi, digits=0):
    return f"{lo:.{digits}f}--{hi:.{digits}f}"


def isnan(x):
    return isinstance(x, float) and math.isnan(x)
