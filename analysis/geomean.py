"""Aggregation rules of every reported number.

Within a workload-length or resolution group, each baseline's latency ratio to Dex is the geometric mean of the
per-input ratios over the inputs both systems completed (paired steps for training); unsupported, timed-out and
out-of-memory inputs are excluded. The strongest baseline is the one with the smallest group-level ratio. Every average
across different inputs, steps, layers or workloads is a geometric mean. Repeats of the same call keep their median: the
timed iterations inside a launch, and on W1-W3 the launches of one case (common._combine; the median of two launches is
their mean); on W4-W6 the launches of one case are combined by geometric mean.
"""
import math


def gm(values):
    """geometric mean of positive values (empty -> nan)"""
    v = [x for x in values]
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")


def gm_pos(values):
    """geometric mean of the positive entries; 0.0 when none is positive (a bar segment a system does not have)"""
    v = [x for x in values if x is not None and x > 0]
    return gm(v) if v else 0.0


def ratio_gm(a, b):
    """geometric mean over the common keys of a[k] / b[k] (nan without common keys)"""
    ks = sorted(set(a) & set(b))
    return gm(a[k] / b[k] for k in ks) if ks else float("nan")


def strongest(cases, ours, baselines):
    """(speedup, baseline name) of the strongest baseline against `ours`; cases[system] = {input: latency}"""
    cand = [(ratio_gm(cases[b], cases[ours]), b) for b in baselines if b in cases and cases[b]]
    cand = [c for c in cand if not math.isnan(c[0])]
    return min(cand) if cand else (float("nan"), "")


def percentile(values, q):
    """linear-interpolation percentile, q in [0, 100]"""
    s = sorted(values)
    if not s:
        return float("nan")
    x = (len(s) - 1) * q / 100.0
    lo = int(math.floor(x)); hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (x - lo)
