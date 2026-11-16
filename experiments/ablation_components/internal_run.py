"""One pass of one arm of the data-reuse, relay and transfer-order study on Dex's scheduler and executor, over a list of inputs.
Arms: full (the unsplit plans of internal_plans.py), load_only (load-only plans), no_relay (full plans, DIST_FS_RELAY=0: every
remote block goes straight to each reader), fifo (full plans, DIST_FS_FIFO=1: send order per link = message generation order; gateways
unchanged). Input-arrival estimates and the resulting compute order are rebuilt under FIFO;
this changes the send-order rule, not only the final send list. Switches are set before importing and building schedules. Forward and backward drivers and the
timing protocol are experiments/attention_layer/static_op.py (3 warm-ups, median of 10 timed calls), without the FP64 check.
Inputs : internal_plans.py. Output: $DEX_RESULT_ROOT/ablation_components/internal/<arm>_<pass>.jsonl (input_id, lat_ms, status)
Example: DEX_ENV_PROFILE=static tools/launch.sh experiments/ablation_components/internal_run.py --arm no_relay --pass fwd --ids w2_32k_p40"""
import argparse
import datetime
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_ROOT, _HERE, os.path.join(_ROOT, "experiments", "attention_layer")]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--arm", required=True, choices=["full", "load_only", "no_relay", "fifo"])
ap.add_argument("--pass", dest="pass_", required=True, choices=["fwd", "bwd"])
ap.add_argument("--ids", default="", help="comma list of input ids (default: all nine)")
A = ap.parse_args()
if A.arm == "no_relay":
    os.environ["DIST_FS_RELAY"] = "0"
elif A.arm == "fifo":
    os.environ["DIST_FS_FIFO"] = "1"

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

import internal_plans as IP  # noqa: E402
import static_inputs as SI  # noqa: E402
import static_op as SO  # noqa: E402
from dex.common.env import cfg  # noqa: E402


def main():
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2))
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dev = torch.device("cuda")
    from dex import dex_cuda, scheduler
    SO.S, SO.C = scheduler, dex_cuda
    plans = "load_only" if A.arm == "load_only" else "full"
    SI.plan_files = lambda case, _hs: (case["mask"], IP.plan_file(case["id"], plans, "fwd"), IP.plan_file(case["id"], plans, "bwd"))
    cases = [c for c in IP.cases() if not A.ids or c["id"] in A.ids.split(",")]
    args = argparse.Namespace(launch=2, head_split=1)        # launch > 1: timing only
    recs = (SO.dex_fwd if A.pass_ == "fwd" else SO.dex_bwd)(cases, args, rank, world, dev)
    if rank == 0:
        f = os.path.join(cfg("DEX_RESULT_ROOT"), "ablation_components", "internal", f"{A.arm}_{A.pass_}.jsonl")
        os.makedirs(os.path.dirname(f), exist_ok=True)
        with open(f, "a") as fh:
            for r in recs:
                r.update(system=f"dex_{A.arm}", launch=1)
                fh.write(json.dumps(r) + "\n")
                print(r["input_id"], A.arm, A.pass_, r["lat_ms"], r["status"], flush=True)


if __name__ == "__main__":
    main()
