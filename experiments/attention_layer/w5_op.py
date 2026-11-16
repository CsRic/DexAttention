"""W5 operator test (Wan2.1-1.3B self-attention, SpargeAttention2 selection, exact FP16 forward + backward on 128 x 64
blocks), one system per launch, every case of the case file in it.

Per case (workloads/w5_sparge/cases42.txt: a generation's real q / k of one layer and denoising step; V and dO drawn from
N(0, 0.5^2) with seed 5000 + case index, identical for every system): rank 0 reads the dump and sends every rank its
balanced-zigzag tokens; then, each as barrier -> slowest rank done, `--warmup` untimed + `--iters` timed calls, median:
  fwd_exec_ms    the mask known (the selection's result is cached): each system's whole remaining per-call work
                 (Dex: mask all-gather, assigner, broadcast, block pairs, schedule, tables, persistent forward)
  fwd_online_ms  the same including the selection (split by owner; All-gather: on the gathered keys)
  bwd_ms         the backward (Dex: its plan inside)
Dex variants and the decomposed executor also record the online plan's stages (rank 0, synchronised, 5 separate runs,
median). --fp64 (launch 1): FP64 sampled-row check of O, dQ, dK, dV against exact attention on the selection's
128 x 64 mask (dex.common.fp64_check); the decomposed executor also compares O / LSE / dQ / dK / dV with All-gather.
Systems: dex, dex_no_assigner, ring, allgather, sparse_allgather, decomposed (Dex's plan on the decomposed
executor, baselines/decomposed/w5.py).
Output: one json line per case appended to --out (finished cases are skipped on a rerun).
usage: tools/launch.sh experiments/attention_layer/w5_op.py --system dex --out w5/dex/rep1.jsonl [--fp64]"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from workloads.w5_sparge import cases as C  # noqa: E402
from workloads.w5_sparge.common import Reuse, cuda_ext  # noqa: E402
from dex.common.cp_layout import Layout  # noqa: E402

D = 128
SYSTEMS = ("dex", "dex_no_assigner", "ring", "allgather", "sparse_allgather", "decomposed")
PLAN_RUNS = 5


def make_system(name, lay, H):
    if name in ("dex", "dex_no_assigner"):
        from dex.attn.sparge import DexSparge

        return DexSparge(lay, H, assigner=name == "dex")
    if name == "ring":
        from baselines.ring.sparge import RingSparge

        return RingSparge(lay, H)
    if name == "allgather":
        from baselines.allgather.sparge import AllGatherSparge

        return AllGatherSparge(lay, H)
    if name == "sparse_allgather":
        from baselines.allgather.sparge import SparseAllGatherSparge

        return SparseAllGatherSparge(lay, H)
    from baselines.decomposed.w5 import DecomposedSparge

    return DecomposedSparge(lay, H)


def distribute(ci_, generation, call, rank, P, dev):
    """rank 0 reads the case and sends every rank its tokens of q, k and the seeded V, dO -> (lay, q, k, v, do, meta)"""
    if rank == 0:
        qn, kn, meta = C.load_case(generation, call)
        hdr = torch.tensor([qn.shape[0], qn.shape[1]], device=dev)
    else:
        meta = None
        hdr = torch.zeros(2, dtype=torch.long, device=dev)
    dist.broadcast(hdr, 0)
    H, S = (int(x) for x in hdr)
    lay = Layout(S, P, rank, dev)
    q = torch.empty(H, lay.L, D, dtype=torch.float16, device=dev)
    k, v, do = torch.empty_like(q), torch.empty_like(q), torch.empty_like(q)
    if rank == 0:
        g = torch.Generator(device=dev).manual_seed(5000 + ci_)
        Vf = (torch.randn(H, S, D, device=dev, generator=g) * 0.5).half()
        dOf = (torch.randn(H, S, D, device=dev, generator=g) * 0.5).half()
        Qf, Kf = torch.from_numpy(qn).to(dev), torch.from_numpy(kn).to(dev)
        for r in range(P):
            tk = Layout(S, P, r, dev).tok
            parts = [X[:, tk].contiguous() for X in (Qf, Kf, Vf, dOf)]
            if r == 0:
                for dst, src in zip((q, k, v, do), parts):
                    dst.copy_(src)
            else:
                for t_ in parts:
                    dist.send(t_, r)
        del Qf, Kf, Vf, dOf
    else:
        for t_ in (q, k, v, do):
            dist.recv(t_, 0)
    return lay, H, S, q, k, v, do, meta


def fp64_errors(lay, H, S, q, k, v, do, o, dq, dk, dv, fm, dev):
    """FP64 sampled-row check of O / dQ / dK / dV on the global 128 x 64 mask (real coordinates)"""
    from dex.common import fp64_check as F64

    P, L = lay.P, lay.L
    qb = torch.from_numpy(
        np.concatenate([np.arange(a_ // 128, -(-b_ // 128)) for a_, b_ in (lay.chunks[c] for c in lay.my)])
    ).to(dev)
    nq, nkb = -(-S // 128), lay.nkb
    mfull = torch.zeros(H, nq, nkb, dtype=torch.uint8, device=dev)
    mfull[:, qb] = fm.to(torch.uint8)
    dist.all_reduce(mfull, op=dist.ReduceOp.MAX)
    gtok = torch.empty(P, lay.L_pad, dtype=torch.int64, device=dev)
    mine = torch.full((lay.L_pad,), -1, dtype=torch.int64, device=dev)
    mine[:L] = lay.tok
    dist.all_gather_into_tensor(gtok, mine)

    def tok(X):  # this rank's [H, L, D] -> all tokens [S, H, D]
        Xp = torch.zeros(H, lay.L_pad, D, dtype=X.dtype, device=dev)
        Xp[:, :L] = X
        buf = torch.empty(P, H, lay.L_pad, D, dtype=X.dtype, device=dev)
        dist.all_gather_into_tensor(buf, Xp)
        out = torch.empty(S, H, D, dtype=X.dtype, device=dev)
        for r in range(P):
            sel = gtok[r] >= 0
            out[gtok[r][sel]] = buf[r][:, sel].permute(1, 0, 2)
        return out

    masks = [F64.BlockMask(mfull[h].bool(), 128, 64, S) for h in range(H)]
    rows = lay.tok.cpu().numpy()
    return F64.check(
        tok(q),
        tok(k),
        tok(v),
        tok(do),
        masks,
        D**-0.5,
        dict(o=o.permute(1, 0, 2), dq=dq.permute(1, 0, 2), dk=dk.permute(1, 0, 2), dv=dv.permute(1, 0, 2)),
        q_rows=rows,
        kv_rows=rows,
        group=dist.group.WORLD,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=SYSTEMS)
    ap.add_argument("--cases", default=C.CASES42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fp64", action="store_true")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=10)
    a = ap.parse_args()
    dist.init_process_group("nccl")
    rank, P = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count())))
    torch.cuda.set_device(dev)
    DA = cuda_ext()
    if a.system in ("dex", "dex_no_assigner"):
        DA.init_nvshmem(dist.group.WORLD)
    Reuse.enabled, Reuse.ahead = True, False  # the cache carries only a known mask
    cases = C.read_cases(a.cases)
    done = {r["name"] for r in map(json.loads, open(a.out))} if os.path.exists(a.out) else set()
    sysobjs = {}
    for ci_, (gen, call) in enumerate(cases):
        name = f"{gen}_c{call:05d}"
        if name in done:
            continue
        t_case = time.time()
        lay, H, S, q, k, v, do, meta = distribute(ci_, gen, call, rank, P, dev)
        if (S, H) not in sysobjs:
            sysobjs[(S, H)] = make_system(a.system, lay, H)
        so = sysobjs[(S, H)]
        so.lay = lay
        rec = dict(name=name, generation=gen, S=S, H=H, system=a.system)
        if rank == 0:
            rec.update(step=meta["step"], layer=meta["call_in_step"])
        Reuse.clear()
        Reuse.layer[0] = 0
        o, (fm, lse) = so.fwd(q, k, v)  # the selection, once: the known-mask scope's mask

        def f_exec():
            Reuse.clear()
            Reuse.caches[0] = {"fm": fm}
            return so.fwd(q, k, v)

        def f_online():
            Reuse.clear()
            return so.fwd(q, k, v)

        def f_bwd():
            return so.bwd(q, k, v, o, do, fm, lse, None)

        rec["fwd_exec_ms"], rec["fwd_exec_all"] = C.latency(f_exec, dev, a.warmup, a.iters)
        rec["fwd_online_ms"], rec["fwd_online_all"] = C.latency(f_online, dev, a.warmup, a.iters)
        rec["bwd_ms"], rec["bwd_all"] = C.latency(f_bwd, dev, a.warmup, a.iters)
        if a.system == "decomposed":
            rec["moved_pairs"] = so.last_tab["moved_pairs"]
            rec["inter_kv_node_max_MiB"] = round(so.last_tab["inter_kv_node_max_MiB"], 2)
        if a.system in ("dex", "dex_no_assigner", "decomposed"):  # the online plan's stages (separate runs)
            type(so).plan_timing = True
            tp = {}
            for _ in range(PLAN_RUNS):
                so.tplan.clear()
                f_online()
                f_bwd()
                for k_, v_ in so.tplan.items():
                    tp.setdefault(k_, []).append(v_)
            type(so).plan_timing = False
            rec["plan_ms"] = {
                k_: (round(float(np.median(v_)), 3) if not isinstance(v_[0], (tuple, list)) else v_[-1])
                for k_, v_ in tp.items()
            }
            rec["assigner"] = so.assigner
        dens = torch.tensor([fm.float().sum().item(), float(fm.numel())], device=dev)
        dist.all_reduce(dens)
        rec["density"] = round((dens[0] / dens[1]).item(), 4)
        if a.fp64:
            o, (fm, lse) = f_online()
            dq, dk, dv = f_bwd()
            if a.system == "decomposed":  # vs All-gather on the same selection
                from baselines.allgather.sparge import AllGatherSparge

                ag = AllGatherSparge(lay, H)
                Reuse.clear()
                Reuse.caches[0] = {"fm": fm}
                oa, (_, la) = ag.fwd(q, k, v)
                dqa, dka, dva = ag.bwd(q, k, v, oa, do, fm, la, None)

                def rel(x, y):
                    t_ = torch.tensor(
                        [(x.float() - y.float()).abs().max().item(), y.float().abs().max().item()], device=dev
                    )
                    dist.all_reduce(t_, op=dist.ReduceOp.MAX)
                    return float(t_[0] / t_[1])

                lt = torch.tensor([(lse - la).abs().max().item()], device=dev)
                dist.all_reduce(lt, op=dist.ReduceOp.MAX)
                rec["vs_allgather"] = dict(
                    o=rel(o, oa), lse_abs=float(lt), dq=rel(dq, dqa), dk=rel(dk, dka), dv=rel(dv, dva)
                )
                del ag, oa, la, dqa, dka, dva
            rec["err_fp64"] = fp64_errors(lay, H, S, q, k, v, do, o, dq, dk, dv, fm, dev)
            torch.cuda.empty_cache()
        rec["wall_s"] = round(time.time() - t_case, 1)
        if rank == 0:
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print("[w5]", json.dumps({k_: v_ for k_, v_ in rec.items() if not k_.endswith("_all")}), flush=True)
        del q, k, v, do, o, lse
        torch.cuda.empty_cache()
        dist.barrier()
    dist.barrier()
    os._exit(0)


if __name__ == "__main__":
    main()
