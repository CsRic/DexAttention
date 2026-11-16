"""MagiAttention on 16 GPUs: one launch over a list of inputs of a workload. Magi keeps its defaults: its dispatch (MinHeap chunk
balancing over the real tokens), its solver, overlap degree 1, 1-D CP group over all ranks; the all_to_all is staged.
Masks (token level, exact, unpadded real tokens): Lambda per-token ranges; rollouts their segment table. --padded (component
replacement study): the padded tile layout (pad K / V = 0, exact O / LSE correction after the forward, timed) with chunk = the
largest multiple of 128 <= Magi's auto chunk, so its plan maps exactly onto tiles; forward only.
Timing: barrier before every call, slowest rank, 3 warm-ups, median of 10. fwd = calc_attn; bwd = backward of one saved forward.
plan_s = magi_attn_flex_key (dispatch + solver + comm metadata), first build. --plan_only: only that (Table 1).
Launch 1 (--fp64) adds the FP64 sampled-row check. Outputs: $DEX_RESULT_ROOT/attention_layer/<w>/magi_<pass>_launch<k>.jsonl
(--padded: ablation_components/magi_native_launch<k>.jsonl); --plan_only: planning_latency/magi_<w>.jsonl.
Example: DEX_PYTHON=$DEX_MAGI_PYTHON tools/launch.sh baselines/magi/bench.py --workload w3 --launch 1 --fp64"""
import argparse
import datetime
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "experiments", "attention_layer")]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

import baselines.magi  # noqa: E402,F401  (kernel backend defaults)
from baselines.magi import staged_a2a  # noqa: E402
from baselines.magi.slices import slices_from_padded_segments, slices_from_ranges, slices_from_segments  # noqa: E402
import static_inputs as SI  # noqa: E402
from dex.common import fp64_check as F64  # noqa: E402
from dex.common.env import cfg  # noqa: E402
from dex.common.pad_correct import correct as pad_correct  # noqa: E402

D = 128


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workload", required=True, choices=["w1", "w2", "w3"])
    ap.add_argument("--ids", default="")
    ap.add_argument("--launch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--fp64", action="store_true")
    ap.add_argument("--padded", action="store_true")
    ap.add_argument("--plan_only", action="store_true")
    a = ap.parse_args()
    staged_a2a.install_event()
    from magi_attention.api import calc_attn, dispatch, magi_attn_flex_key, undispatch
    from magi_attention.common import AttnRanges
    from magi_attention.common.enum import AttnMaskType
    TY = dict(full=AttnMaskType.FULL, causal=AttnMaskType.CAUSAL, bicausal=AttnMaskType.BICAUSAL, invcausal=AttnMaskType.INVCAUSAL)
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=30))
    rank, W = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank % int(cfg("DEX_GPUS_PER_NODE")))))
    cases = SI.inputs(a.workload)
    if a.ids:
        cases = [c for c in cases if c["id"] in a.ids.split(",")]
    res = cfg("DEX_RESULT_ROOT")

    def tmax(x):
        t = torch.tensor([x], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return t.item()

    def timed(fn):
        ts = []
        for it in range(a.warmup + a.iters):
            torch.cuda.synchronize()
            dist.barrier()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            dt = tmax(time.perf_counter() - t0)
            if it >= a.warmup:
                ts.append(dt * 1e3)
        return float(np.median(ts)), [round(x, 3) for x in ts]

    def emit(c, ps, ms, ms_all, e64, status, reason, **extra):
        if rank != 0:
            return
        keys = ("o",) if ps == "fwd" else ("dq", "dk", "dv")
        e = {k: e64[k] for k in keys if e64 and k in e64}
        rec = dict(input_id=c["id"], workload=a.workload.upper(), S=c["S"], system="magi", **{"pass": ps}, launch=a.launch, lat_ms=ms, lat_ms_all=ms_all,
                   max_rel_err=None, fp64_rel_l2={k: v["rel_l2"] for k, v in e.items()} or None, fp64_rows={k: v["rows"] for k, v in e.items()} or None,
                   status=status, reason=reason, **extra)
        sub = ("ablation_components", f"magi_native_launch{a.launch}.jsonl") if a.padded else ("attention_layer", a.workload, f"magi_{ps}_launch{a.launch}.jsonl")
        f = os.path.join(res, *sub)
        os.makedirs(os.path.dirname(f), exist_ok=True)
        with open(f, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print("[magi]", json.dumps({k: v for k, v in rec.items() if k != "lat_ms_all"}), flush=True)

    for c in cases:
        extra = {}
        ps_list = ("fwd",) if a.padded else ("fwd", "bwd")
        try:
            zpad = None
            z = np.load(c["mask"])
            hq, hk = (int(x) for x in z["heads"])
            if a.workload == "w1":
                T, qr, kr, ty = slices_from_ranges(np.load(c["ranges"]))
            elif a.padded:
                zpad = z
                T = int(z["tokens"])
                qr, kr, ty = slices_from_padded_segments(z["segs"])
            else:
                T, qr, kr, ty = slices_from_segments(z["segs"])
            g = torch.Generator(device="cuda").manual_seed(1234)
            q = torch.randn(T, hq, D, generator=g, device="cuda", dtype=torch.float16)
            k, v = (torch.randn(T, hk, D, generator=g, device="cuda", dtype=torch.float16) for _ in range(2))
            do = torch.randn(T, hq, D, generator=g, device="cuda", dtype=torch.float16)
            chunk = {"chunk_size": max(128, -(-T // (8 * W)) // 128 * 128)} if a.padded else {}
            torch.cuda.synchronize()
            dist.barrier()
            t0 = time.perf_counter()
            key = magi_attn_flex_key(q_ranges=AttnRanges.from_ranges(qr), k_ranges=AttnRanges.from_ranges(kr), attn_mask_type=[TY[t] for t in ty],
                                     total_seqlen_q=T, total_seqlen_k=T, num_heads_q=hq, num_heads_kv=hk, head_dim=D, pad_size=0,
                                     cp_group_or_mesh=dist.group.WORLD, **chunk)
            torch.cuda.synchronize()
            plan_s = round(tmax(time.perf_counter() - t0), 3)
            extra.update(plan_s=plan_s, n_slices=len(qr))
            if a.plan_only:
                if rank == 0:
                    f = os.path.join(res, "planning_latency", f"magi_{a.workload}.jsonl")
                    os.makedirs(os.path.dirname(f), exist_ok=True)
                    with open(f, "a") as fh:
                        fh.write(json.dumps(dict(input_id=c["id"], plan_s=plan_s, n_slices=len(qr))) + "\n")
                del key
                torch.cuda.empty_cache()
                dist.barrier()
                continue
            lnp = None
            if zpad is not None:
                valid = torch.as_tensor(zpad["tok_valid"].astype(bool), device="cuda")
                k[~valid] = 0
                v[~valid] = 0
                lnp = dispatch(torch.as_tensor(zpad["npad"].astype(np.float32), device="cuda")[:, None], key=key)[:, 0]
            lq, lk, lv, ldo = (dispatch(x, key=key) for x in (q, k, v, do))
            lq, lk, lv = (x.detach().requires_grad_() for x in (lq, lk, lv))

            def fwd():
                o_, m_ = calc_attn(q=lq, k=lk, v=lv, key=key)
                if zpad is not None:
                    with torch.no_grad():
                        pad_correct(o_, m_.lse.t(), lnp)
                return o_

            ms_f = timed(fwd)
            lo = fwd()

            def bwd():
                for x in (lq, lk, lv):
                    x.grad = None
                lo.backward(ldo, retain_graph=True)

            ms_b = timed(bwd) if not a.padded else None
            extra["peak_mem_gb"] = round(tmax(torch.cuda.max_memory_allocated() / 2 ** 30), 2)
            e64 = None
            if a.fp64:
                out = undispatch(lo.detach(), key=key)
                rg = np.load(c["ranges"])
                ids = np.arange(T * rank // W, T * (rank + 1) // W)
                if a.padded:
                    rid = F64.real_ids(zpad, T) if zpad is not None else np.arange(T)
                    ids = ids[rid[ids] >= 0]
                    rvm = torch.as_tensor(rid >= 0, device="cuda")
                    e64 = F64.check(q[rvm], k[rvm], v[rvm], None, rg, D ** -0.5, dict(o=out[torch.as_tensor(ids, device="cuda")]), q_rows=rid[ids], group=dist.group.WORLD)
                else:
                    bwd()
                    gq, gk, gv = (undispatch(x.grad, key=key) for x in (lq, lk, lv))
                    it = torch.as_tensor(ids, device="cuda")
                    e64 = F64.check(q, k, v, do, rg, D ** -0.5, dict(o=out[it], dq=gq[it], dk=gk[it], dv=gv[it]), q_rows=ids, kv_rows=ids, group=dist.group.WORLD)
                    del gq, gk, gv
                del out
            emit(c, "fwd", ms_f[0], ms_f[1], e64, "ok", "", **extra)
            if ms_b is not None:
                emit(c, "bwd", ms_b[0], ms_b[1], e64, "ok", "", **extra)
            del key, lq, lk, lv, ldo, lo
        except Exception as e:  # noqa: BLE001  one failing input does not lose the launch
            for ps in ps_list:
                emit(c, ps, None, None, None, "error", f"{type(e).__name__}: {str(e)[:300]}")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
