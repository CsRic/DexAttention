"""Official DCP with a per-input block-size ladder: each input starts at B = min(4096, the largest power of two with >= 16
blocks); if its planning takes > 120 s, B doubles for that input only; fewer than 16 blocks at the current B -> N/A.
Planning: one track.py --plan_only launch per (input, B); the limit is enforced in the driver. Plan time = wall time of the
official planner (prepare_and_gen_args included), maximum over the GPUs.
--measure: one launch per (workload, B) with cached plans, launch 1 with the FP64 check; DCP runs on its own executor.
Outputs: $DEX_DATA_ROOT/plans/dcp/<w>/{ladder.json, cache/, logs/}; attention-layer records from track.py.
Example: python baselines/dcp/ladder.py --workloads w1 w2 w3; python baselines/dcp/ladder.py --workloads w1 w2 w3 --measure --launches 1 2"""
import argparse
import json
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "experiments", "attention_layer")]
import torch  # noqa: E402,F401  (before numpy: DCP's PyTorch build needs a more recent C++ runtime than the one numpy would load)
import numpy as np  # noqa: E402

import static_inputs as SI  # noqa: E402
from dex.common.env import cfg  # noqa: E402

_N = [0]


def launch(args_, log):
    """one tools/launch.sh launch of track.py in DCP's Python; its own redis port range"""
    _N[0] += 1
    kv = int(cfg("DEX_MASTER_PORT")) + 1000 + 100 * (_N[0] % 20)
    env = dict(os.environ, DEX_PYTHON=cfg("DEX_DCP_PYTHON"), PYTHONPATH=_ROOT)
    with open(log, "w") as f:
        return subprocess.run(["bash", os.path.join(_ROOT, "tools", "launch.sh"), "baselines/dcp/track.py", "--kv_base", str(kv)] + args_, cwd=_ROOT, env=env,
                              stdout=f, stderr=subprocess.STDOUT).returncode


def tasks_for(w):
    out = []
    for c in SI.inputs(w):
        out.append(dict(id=c["id"], workload=w, S=c["S"], ranges=c["ranges"], mask=c["mask"]))
    heads = tuple(int(x) for x in np.load(SI.inputs(w)[0]["mask"])["heads"])
    return heads, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workloads", nargs="+", default=["w1", "w2", "w3"])
    ap.add_argument("--plan_limit", type=float, default=120.0)
    ap.add_argument("--B0", type=int, default=4096)
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--launches", type=int, nargs="+", default=[1, 2])
    a = ap.parse_args()
    for w in a.workloads:
        heads, tasks = tasks_for(w)
        d = os.path.join(cfg("DEX_DATA_ROOT"), "plans", "dcp", w)
        os.makedirs(f"{d}/cache", exist_ok=True)
        os.makedirs(f"{d}/logs", exist_ok=True)
        lf = f"{d}/ladder.json"
        lad = json.load(open(lf)) if os.path.exists(lf) else {}
        hs = [str(heads[0]), str(heads[1])]
        if not a.measure:
            for t in tasks:
                if lad.get(t["id"], {}).get("done"):
                    continue
                B, att = min(a.B0, 1 << max(0, (t["S"] // 16).bit_length() - 1)), []
                while True:
                    if t["S"] < 16 * B:
                        lad[t["id"]] = dict(S=t["S"], B=None, attempts=att, done=True, why=f"S < 16 x {B}")
                        break
                    tf = f"{d}/logs/plan_{t['id']}_B{B}_task.json"
                    out = f"{d}/logs/plan_{t['id']}_B{B}.jsonl"
                    json.dump([t], open(tf, "w"))
                    rc = launch(["--tasks", tf, "--block", str(B), "--heads"] + hs + ["--out", out, "--cache", f"{d}/cache", "--plan_only", "--plan_limit", str(a.plan_limit)],
                                f"{d}/logs/plan_{t['id']}_B{B}.log")
                    rec = [json.loads(x) for x in open(out)] if os.path.exists(out) else []
                    att.append(dict(B=B, rc=rc, plan_s=rec[-1]["plan_s"] if rec else None))
                    print(f"[{w}] {t['id']} B {B}: rc {rc}" + (f", plan {rec[-1]['plan_s']} s" if rec else ""), flush=True)
                    if rc == 0 and rec:
                        lad[t["id"]] = dict(S=t["S"], B=B, plan_s=rec[-1]["plan_s"], attempts=att, done=True)
                        break
                    if not os.path.exists(out + ".over"):
                        lad[t["id"]] = dict(S=t["S"], B=None, attempts=att, done=True, why=f"rc {rc} without time-out at B {B}")
                        break
                    B *= 2
                json.dump(lad, open(lf, "w"), indent=1)
            continue
        groups = {}
        for t in tasks:
            b = lad.get(t["id"], {}).get("B")
            if b:
                groups.setdefault(b, []).append(t)
        for r in a.launches:
            for b, ts in sorted(groups.items()):
                tf = f"{d}/logs/B{b}_tasks.json"
                json.dump(ts, open(tf, "w"))
                out = f"{d}/logs/launch{r}_B{b}.jsonl"
                rc = launch(["--tasks", tf, "--block", str(b), "--heads"] + hs + ["--out", out, "--cache", f"{d}/cache", "--launch", str(r)] + (["--fp64"] if r == 1 else []),
                            f"{d}/logs/launch{r}_B{b}.log")
                print(f"[{w}] launch {r} B {b}: rc {rc}, {len(ts)} inputs", flush=True)


if __name__ == "__main__":
    main()
