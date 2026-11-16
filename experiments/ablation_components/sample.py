"""The 15 static inputs of the component-replacement study: W1 at every length where UA has a forward and a backward plan, and for W2
and W3 per length bin (32K, 64K, 128K) the first two inputs by name that UA can plan in both passes.
`select()` applies the rule to the current plans; fixed_sample() is its result in the evaluation. Output of this module's main:
$DEX_DATA_ROOT/plans/ablation/sample.json. Example: python experiments/ablation_components/sample.py --select"""
import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "experiments", "attention_layer")]
import static_inputs as SI  # noqa: E402
from dex.common.env import cfg  # noqa: E402

_BINS = ("32k_p103", "32k_p14", "64k_p103", "64k_p193", "128k_p40", "128k_p460")
W1_LENGTHS = (32768, 131072, 262144)    # UA has no plan at 64K within the planning limit


def ablation_dir(*sub):
    d = os.path.join(cfg("DEX_DATA_ROOT"), "plans", "ablation", *sub)
    os.makedirs(d, exist_ok=True)
    return d


def fixed_sample():
    w1 = [c["id"] for c in SI.inputs("w1") if c["S"] in W1_LENGTHS]
    return dict(w1=w1, w2=[f"w2_{x}" for x in _BINS], w3=[f"w3_{x}" for x in _BINS])


def select():
    man = json.load(open(os.path.join(cfg("DEX_DATA_ROOT"), "plans", "ua", "manifest.json")))
    ok = lambda i: "plan" in man.get(f"{i}|fwd", {}) and "plan" in man.get(f"{i}|bwd", {})  # noqa: E731
    out = dict(w1=[c["id"] for c in SI.inputs("w1") if ok(c["id"])])
    for w in ("w2", "w3"):
        ids, cs = [], SI.inputs(w)
        for b in ("32k", "64k", "128k"):
            ids += sorted(c["id"] for c in cs if c.get("length_bin") == b and ok(c["id"]))[:2]
        out[w] = ids
    return out


def sample():
    f = os.path.join(ablation_dir(), "sample.json")
    return json.load(open(f)) if os.path.exists(f) else fixed_sample()


def cases():
    """[(workload, case)] of the sample"""
    s = sample()
    return [(w, c) for w in ("w1", "w2", "w3") for c in SI.inputs(w) if c["id"] in s[w]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--select", action="store_true", help="apply the rule instead of the evaluated sample")
    a = ap.parse_args()
    s = select() if a.select else fixed_sample()
    json.dump(s, open(os.path.join(ablation_dir(), "sample.json"), "w"), indent=1)
    print(json.dumps(s, indent=1))
