"""Baseline planning times per input and pass -> ranges per workload (Table 1: UA and DCP over both static passes of successful plans, Magi over key builds).
Inputs: plans/ua/manifest.json, plans/dcp/<w>/ladder.json, planning_latency/magi_<w>.jsonl. Output: $DEX_RESULT_ROOT/planning_latency/<system>.json
Example: python experiments/planning_latency/baselines_summary.py ua"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dex.common.env import cfg  # noqa: E402


def main(system):
    data, res = cfg("DEX_DATA_ROOT"), cfg("DEX_RESULT_ROOT")
    recs = {}
    if system == "ua":
        man = json.load(open(os.path.join(data, "plans", "ua", "manifest.json")))
        for k, e in man.items():
            i, ps = k.split("|")
            if ps in ("fwd", "bwd"):
                recs[k] = dict(input_id=i, workload=i.split("_")[0] if i.startswith(("w2", "w3")) else "w1", plan_s=e.get("total_s"), block_tokens=e.get("block_tokens"), na=e.get("na"))
    elif system == "dcp":
        for w in ("w1", "w2", "w3"):
            f = os.path.join(data, "plans", "dcp", w, "ladder.json")
            if os.path.exists(f):
                for i, e in json.load(open(f)).items():
                    recs[f"{w}|{i}"] = dict(input_id=i, workload=w, plan_s=e.get("plan_s"), block_tokens=e.get("B"), na=e.get("why"))
    else:
        for w in ("w1", "w2", "w3"):
            f = os.path.join(res, "planning_latency", f"magi_{w}.jsonl")
            if os.path.exists(f):
                for ln in open(f):
                    r = json.loads(ln)
                    recs[f"{w}|{r['input_id']}"] = dict(input_id=r["input_id"], workload=w, plan_s=r["plan_s"], na=None)
    out = {}
    for w in ("w1", "w2", "w3"):
        ts = [r["plan_s"] for r in recs.values() if r["workload"] == w and r["plan_s"] is not None]
        na = sum(1 for r in recs.values() if r["workload"] == w and r["plan_s"] is None)
        if ts or na:
            out[w] = dict(min_s=min(ts) if ts else None, max_s=max(ts) if ts else None, n=len(ts), n_without_plan=na)
    os.makedirs(os.path.join(res, "planning_latency"), exist_ok=True)
    json.dump(dict(summary=out, records=recs), open(os.path.join(res, "planning_latency", f"{system}.json"), "w"), indent=1)
    print(system, json.dumps(out), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
