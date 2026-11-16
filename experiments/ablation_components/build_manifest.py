"""Per replaced system X and input of the study: the plan file (owner + schedule_table + gpu_kernels) and the mask of its layout. Missing plans are listed,
never replaced. DCP: W1 the exact per-head plan, W2 / W3 the approximate per-head conversion.
Output: $DEX_DATA_ROOT/plans/ablation/manifest.json. Example: python experiments/ablation_components/build_manifest.py"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sample as SM  # noqa: E402


def load():
    return json.load(open(os.path.join(SM.ablation_dir(), "manifest.json")))


def main():
    man = {"ring": {}, "ua": {}, "dcp": {}, "magi": {}, "missing": []}
    for w, c in SM.cases():
        n = c["id"]
        dcp = f"{n}_perhead.json" if w == "w1" else f"{n}_phapprox.json"
        for x, f in (("ring", f"{n}_ring.json"), ("ua", f"{n}_ua.json"), ("dcp", dcp), ("magi", f"{n}_magi.json")):
            pf = os.path.join(SM.ablation_dir(x), f)
            if os.path.exists(pf):
                man[x][n] = dict(plan=pf, mask=json.load(open(pf))["mask"])
            else:
                man["missing"].append([x, n, pf])
    json.dump(man, open(os.path.join(SM.ablation_dir(), "manifest.json"), "w"), indent=1)
    print({x: len(v) for x, v in man.items() if x != "missing"}, "missing", man["missing"], flush=True)


if __name__ == "__main__":
    main()
