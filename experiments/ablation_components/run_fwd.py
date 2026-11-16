"""Forward of another system's plan on Dex's scheduler and executor, one launch over several inputs (component replacement, Table 5).
--row 2 "X plan + X order": no intra-node relay, X's delivery and compute order imposed on the message tables (dex.scheduler.xorder); Ring also with
its full data flow (every K / V block to every GPU). --row 3 "X assignment + Dex schedule": the same plan file on the default scheduler.
Inputs, timing protocol and FP64 check as experiments/attention_layer/static_op.py. The scheduler reads its switches at import, so they are set first.
Output: $DEX_RESULT_ROOT/ablation_components/<x>_row<r>_launch<k>.jsonl
Example: tools/launch.sh experiments/ablation_components/run_fwd.py --x ua --row 2 --launch 1"""
import argparse
import datetime
import json
import math
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_ROOT, _HERE, os.path.join(_ROOT, "experiments", "attention_layer")]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--x", required=True, choices=["ring", "ua", "dcp", "magi"])
ap.add_argument("--row", type=int, required=True, choices=[2, 3])
ap.add_argument("--launch", type=int, default=1)
ap.add_argument("--ids", default="")
A = ap.parse_args()
if A.row == 2:
    os.environ["DIST_FS_RELAY"] = "0"
    if A.x == "ring":
        os.environ["DIST_RING_FULL"] = "1"
        os.environ["DIST_DENSE_KV"] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

import build_manifest as BM  # noqa: E402
import static_op as SO  # noqa: E402
from dex import dex_cuda as C, scheduler as S  # noqa: E402
from dex.common import fp64_check as F64  # noqa: E402
from dex.common.env import cfg  # noqa: E402
from dex.common.pad_correct import correct as pad_correct  # noqa: E402
from dex.scheduler import xorder  # noqa: E402

TILE, D = 128, 128


def build(case, ent, rank, world, devt):
    z = np.load(ent["mask"])
    m = z["mask"].astype(np.int8)
    HQ, HK = (int(x) for x in z["heads"])
    devb, ownb, meta = S.load_plan(ent["plan"])
    assert meta["cp"] == world, (ent["plan"], meta)
    per = meta["block_tokens"] // TILE
    B = S.block_pairs(m, devb, per, owner=ownb)
    pw = (B["nfull"] + 0.5 * B["ndiag"]) * S.LINK["T_TILE"] * HQ / 32
    R = S.schedule_blocks(B["bq"], B["bk"], B["pdev"], pw, B["prow"], B["pcol"], ownb, world, per, heads=(HQ, HK), d=D)
    x_stats = None
    if A.row == 2:
        x_stats = xorder.apply(R, json.load(open(ent["plan"]))["gpu_kernels"], per, world)
    tables, tm = S.tables_blocks(rank, B, ownb, world, per, devt)
    return dict(case=case, z=z, T=int(z["tokens"]), HQ=HQ, HK=HK, tables=tables, tm=tm, x_stats=x_stats)


def run_case(b, rank, devt):
    z, T, HQ, HK, tables, case = b["z"], b["T"], b["HQ"], b["HK"], b["tables"], b["case"]
    scale = 1.0 / math.sqrt(D)
    g = torch.Generator(device=devt).manual_seed(1234)
    q = torch.randn(1, T, HQ, D, generator=g, device=devt, dtype=torch.float16) * 0.5
    k = torch.randn(1, T, HK, D, generator=g, device=devt, dtype=torch.float16) * 0.5
    v = torch.randn(1, T, HK, D, generator=g, device=devt, dtype=torch.float16) * 0.5
    PAD = "tok_valid" in z.files
    if PAD:
        valid = torch.as_tensor(z["tok_valid"].astype(bool), device=devt)
        npad_g = torch.as_tensor(z["npad"].astype(np.int32), device=devt)
        k[:, ~valid] = 0
        v[:, ~valid] = 0
    owned = b["tm"]["owned"]
    idx = torch.tensor([t * TILE + i for t in owned for i in range(TILE)], device=devt, dtype=torch.long)
    ql, kl, vl = q[:, idx].contiguous(), k[:, idx].contiguous(), v[:, idx].contiguous()
    out = torch.zeros_like(ql)
    lse = torch.full((1, HQ, ql.size(1)), -float("inf"), device=devt, dtype=torch.float32)
    npl = npad_g[idx] if PAD else None

    def run():
        S.call(C, ql, kl, vl, out, lse, tables, scale)
        if PAD:
            pad_correct(out[0], lse[0], npl)

    ms, ms_all = SO.latency(run)
    e64 = None
    if A.launch == 1:
        run()
        torch.cuda.synchronize()
        rg = np.load(case["ranges"])
        if "virtual_heads" in z.files:
            e64 = F64.check_virtual(q[0], k[0], v[0], None, rg, scale, dict(o=out[0]), z, pos_rows=idx.cpu().numpy(), group=dist.group.WORLD)
        else:
            rid = torch.as_tensor(F64.real_ids(z, T), device=devt)
            rv = rid >= 0
            rl = rid[idx]
            ok = rl >= 0
            e64 = F64.check(q[0][rv], k[0][rv], v[0][rv], None, rg, scale, dict(o=out[0][ok]), q_rows=rl[ok].cpu().numpy(), group=dist.group.WORLD)
    r = SO.record(case, f"{A.x}_row{A.row}", "fwd", A.launch, ms, ms_all, None, e64, x_stats=b["x_stats"])
    del ql, kl, vl, out, lse
    torch.cuda.empty_cache()
    return r


def main():
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2))
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    devt = torch.device("cuda")
    man = BM.load()[A.x]
    cases = [c for _, c in BM.SM.cases() if c["id"] in man and (not A.ids or c["id"] in A.ids.split(","))]
    built = [build(c, man[c["id"]], rank, world, devt) for c in cases]
    regs = torch.tensor([SO.fwd_region_bytes(b["tables"], b["HQ"], b["HK"]) for b in built], device=devt, dtype=torch.float64)
    dist.all_reduce(regs, op=dist.ReduceOp.MAX)
    order = np.argsort(-regs.cpu().numpy(), kind="stable")      # the largest region first: later regions fit its allocation
    os.environ["NVSHMEM_SYMMETRIC_SIZE"] = str(int(regs.max().item() * 1.15) + (512 << 20))
    from dex.runtime import finalize_nvshmem, init_nvshmem
    init_nvshmem(dist.group.WORLD)
    recs = [run_case(built[i], rank, devt) for i in order]
    dist.barrier()
    finalize_nvshmem()
    if rank == 0:
        f = os.path.join(cfg("DEX_RESULT_ROOT"), "ablation_components", f"{A.x}_row{A.row}_launch{A.launch}.jsonl")
        os.makedirs(os.path.dirname(f), exist_ok=True)
        with open(f, "a") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
                print(r["input_id"], A.x, A.row, r["lat_ms"], r["fp64_rel_l2"], flush=True)


if __name__ == "__main__":
    main()
