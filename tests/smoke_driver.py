"""Helper of tests/smoke_4gpu.sh: the W1 inputs, plans and operator test on one node. The drivers of the evaluation run unchanged
on a small input set: one 16K Lambda input, 4 GPUs as one node, 8 plan blocks per pass.
  python tests/smoke_driver.py plan                       masks and head-split plans under $DEX_DATA_ROOT
  tools/launch.sh tests/smoke_driver.py op <static_op.py arguments>
  python tests/smoke_driver.py check                      prints the records; exit status 1 if a check fails
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments", "attention_layer")]
import static_inputs as SI  # noqa: E402

TOKENS = 16384
SYSTEMS = ("dex", "ring", "allgather")
TOL = 2e-3     # FP64 sampled-row relative L2 of a 16-bit run


def configure():
    SI.W1_TOKENS = (TOKENS,)
    SI.BLOCKS["w1"] = (8, 8)
    SI.P, SI.GPN, SI.NN = 4, 4, 1


def check():
    bad = 0
    root = os.path.join(os.environ["DEX_RESULT_ROOT"], "attention_layer", "w1")
    for s in SYSTEMS:
        for ps in ("fwd", "bwd"):
            fs = glob.glob(os.path.join(root, f"{s}_{ps}_launch1.jsonl"))
            recs = [json.loads(x) for f in fs for x in open(f)]
            if not recs or recs[-1]["status"] != "ok":
                print(f"{s} {ps}: no result"); bad += 1; continue
            r = recs[-1]; err = r.get("fp64_rel_l2") or {}
            ok = bool(err) and max(err.values()) < TOL
            bad += not ok
            print(f"{s:10s} {ps}: {r['lat_ms']:8.2f} ms  FP64 relative L2 " + ", ".join(f"{k} {v:.1e}" for k, v in err.items()) + ("" if ok else "  FAILED"))
    print("smoke test " + ("passed" if not bad else "FAILED"))
    return 1 if bad else 0


def main():
    stage = sys.argv[1]
    configure()
    if stage == "check":
        return check()
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    if stage == "plan":
        from workloads.w1_lambda import gen_masks
        sys.argv = ["gen_masks", "--tokens", str(TOKENS)]
        gen_masks.main()
        import static_plans
        sys.argv = ["static_plans", "--workloads", "w1", "--head-split", "4", "--placement", "zigzag"]
        static_plans.main()
    else:
        import static_op
        static_op.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
