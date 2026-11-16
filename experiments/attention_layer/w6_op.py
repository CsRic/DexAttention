"""W6 operator test (Llama-3.1-8B + FlexPrefill, forward only), one system per launch, every case in it.

Per case (workloads/w6_flexprefill/cases36.txt, `<input>:<layer>`: the model's real post-RoPE q / k / v and FlexPrefill mask of
that layer, from dump_qkv.py): rank 0 reads the dump, casts BF16 -> FP16, pads to S_pad (32768 / 65536 / 131072: zero
q / k / v) and sends each rank its zigzag rows and mask rows; then, as barrier -> slowest rank, median of --iters timed calls
after --warmup untimed calls (a barrier before each):
  fwd_ms     mask known -> O: each system's whole per-call work (All-gather: gather + tables + kernel; Ring: tables + ring;
             Sparse all-gather: need lists + exchange + tables + kernel; Dex: mask all-gather + assigner + home gate +
             broadcast + block pairs + schedule + tables + persistent forward)
  kernel_ms  (Dex variants, decomposed) the same call on cached tables
Dex variants also: plan stages (5 synchronised runs), home-gate decisions, the plan's moved work, bytes from its message
tables; Sparse all-gather: received tiles per rank. --fp64 (launch 1): FP64 sampled-row check of O (32 groups = query heads,
each with its KV head, the per-head tile-causal mask), 512 rows per rank; the decomposed executor also compares O / LSE with
All-gather.
Systems: dex, dex_no_assigner, ring, allgather, sparse_allgather, decomposed (baselines/decomposed/w6.py).
Output: one json line per case appended to --out (finished cases are skipped on a rerun).
usage: tools/launch.sh experiments/attention_layer/w6_op.py --system dex --out w6/dex/rep1.jsonl [--fp64]"""

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
from dex.common.env import cfg  # noqa: E402
from workloads.w6_flexprefill import common as W  # noqa: E402
from dex.common.tile_layout import TileLayout  # noqa: E402

SYSTEMS = ("dex", "dex_no_assigner", "ring", "allgather", "sparse_allgather", "decomposed")
SPAD = {"32k": 32768, "64k": 65536, "128k": 131072}
CASES36 = os.path.join(ROOT, "workloads", "w6_flexprefill", "cases36.txt")
PLAN_RUNS = 5


def make_system(name, lay):
    if name in ("dex", "dex_no_assigner"):
        from dex.attn.flexprefill import DexFlex

        return DexFlex(lay, assigner=name == "dex")
    if name == "ring":
        from baselines.ring.flexprefill import RingFlex

        return RingFlex(lay)
    if name == "allgather":
        from baselines.allgather.flexprefill import AllGatherFlex

        return AllGatherFlex(lay)
    if name == "sparse_allgather":
        from baselines.allgather.flexprefill import SparseAllGatherFlex

        return SparseAllGatherFlex(lay)
    from baselines.decomposed.w6 import DecomposedFlex

    return DecomposedFlex(lay)


def latency(fn, dev, warmup, iters):
    for _ in range(warmup):
        torch.cuda.synchronize()
        dist.barrier()
        fn()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t = torch.tensor([time.perf_counter() - t0], device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        ts.append(t.item() * 1e3)
    return float(np.median(ts)), [round(x, 3) for x in ts]


def load_case(inp, L, S_pad, lay, rank, P, dev):
    """rank 0 reads; every rank gets its zigzag rows (fp16 q [L, 32, D], k / v [L, 8, D]) and mask rows [32, nl, n] int8"""
    n, nl = lay.n, lay.nl
    if rank == 0:
        d = os.path.join(cfg("DEX_DATA_ROOT"), "w6_flexprefill", "qkv", inp)
        q = torch.from_numpy(np.load(f"{d}/L{L:02d}_q.npy")).view(torch.bfloat16)
        k = torch.from_numpy(np.load(f"{d}/L{L:02d}_k.npy")).view(torch.bfloat16)
        v = torch.from_numpy(np.load(f"{d}/L{L:02d}_v.npy")).view(torch.bfloat16)
        N = q.shape[0]
        mk = np.load(f"{d}/L{L:02d}_mask_g095.npy")
        nb = mk.shape[1]
        m = np.unpackbits(mk, axis=-1, count=nb).astype(bool)  # [32, nb, nb]
        assert not np.triu(m, 1).any(), "mask not causal"
        M = np.zeros((W.HQ, n, n), np.int8)
        M[:, :nb, :nb] = m
        M[:, np.arange(n), np.arange(n)] = 2  # diagonal: causal (padding tiles too)
        tri = nb * (nb + 1) / 2
        info = dict(
            nb=int(nb),
            diag_all=bool(m[:, np.arange(nb), np.arange(nb)].all()),
            density=round(float(m.sum()) / (W.HQ * tri), 4),
            density_kvunion=round(float(m.reshape(W.HK, W.GQ, nb, nb).any(1).sum()) / (W.HK * tri), 4),
        )
        qp = torch.zeros(S_pad, W.HQ, W.D, dtype=torch.float16)
        qp[:N] = q.half()
        kp = torch.zeros(S_pad, W.HK, W.D, dtype=torch.float16)
        kp[:N] = k.half()
        vp = torch.zeros(S_pad, W.HK, W.D, dtype=torch.float16)
        vp[:N] = v.half()
        info["fp16_finite"] = bool(torch.isfinite(qp).all() and torch.isfinite(kp).all() and torch.isfinite(vp).all())
        del q, k, v
        Mt = torch.from_numpy(M)
        for r in range(P - 1, -1, -1):
            tk = torch.cat([torch.arange(*lay.chunks[c]) for c in (r, 2 * P - 1 - r)])
            tl = torch.as_tensor(lay.tiles_of(r))
            parts = [qp[tk].to(dev), kp[tk].to(dev), vp[tk].to(dev), Mt[:, tl].contiguous().to(dev)]
            if r == 0:
                q16, k16, v16, mrows = parts
            else:
                for x in parts:
                    dist.send(x.contiguous(), r)
        St = torch.tensor([N], device=dev)
    else:
        q16 = torch.empty(lay.L, W.HQ, W.D, dtype=torch.float16, device=dev)
        k16 = torch.empty(lay.L, W.HK, W.D, dtype=torch.float16, device=dev)
        v16 = torch.empty_like(k16)
        mrows = torch.empty(W.HQ, nl, n, dtype=torch.int8, device=dev)
        for x in (q16, k16, v16, mrows):
            dist.recv(x, 0)
        St = torch.zeros(1, dtype=torch.long, device=dev)
        info = {}
    dist.broadcast(St, 0)
    return q16, k16, v16, mrows, int(St), info


def fp64_error(lay, q16, k16, v16, o16, mrows, S_pad, S_real, real, P, dev):
    from dex.common import fp64_check as F64

    mh = W.mask_global(mrows, lay)
    gtok = torch.empty(P, lay.L, dtype=torch.int64, device=dev)
    dist.all_gather_into_tensor(gtok, lay.tok.contiguous())

    def glob(X):
        buf = torch.empty((P,) + tuple(X.shape), dtype=X.dtype, device=dev)
        dist.all_gather_into_tensor(buf, X.contiguous())
        out = torch.empty((S_pad,) + tuple(X.shape[1:]), dtype=X.dtype, device=dev)
        out[gtok.reshape(-1)] = buf.reshape((-1,) + tuple(X.shape[1:]))
        return out[:S_real]

    rows = lay.tok[real].cpu().numpy()
    masks = W.tile_causal_masks(mh, S_real)
    kg = glob(k16).repeat_interleave(W.GQ, 1)
    vg = glob(v16).repeat_interleave(W.GQ, 1)
    return F64.check(
        glob(q16), kg, vg, None, masks, W.SCALE, dict(o=o16[real]), q_rows=rows, kv_rows=None, group=dist.group.WORLD
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=SYSTEMS)
    ap.add_argument("--cases", default=CASES36)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fp64", action="store_true")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    a = ap.parse_args()
    dist.init_process_group("nccl")
    rank, P = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count())))
    torch.cuda.set_device(dev)
    gpn = W.GPUS_PER_NODE
    planned = a.system in ("dex", "dex_no_assigner", "decomposed")
    gated = a.system in ("dex", "decomposed")
    if planned:
        W.cuda_ext().init_nvshmem(dist.group.WORLD)
    if planned:
        from dex.attn.flexprefill import GATE_EPS, dex_bytes
    torch.set_grad_enabled(False)
    cases = [c.split(":") for c in open(a.cases).read().split()]
    done = {r["name"] for r in map(json.loads, open(a.out))} if os.path.exists(a.out) else set()
    so = ag = None
    for inp, L in cases:
        L = int(L)
        name = f"{inp}_L{L:02d}"
        if name in done:
            continue
        t_case = time.time()
        S_pad = SPAD[inp.split("_")[-1]]
        lay = TileLayout(S_pad, P, rank, dev)
        q16, k16, v16, mrows, S_real, info = load_case(inp, L, S_pad, lay, rank, P, dev)
        lay.s_real = S_real
        if so is None:
            so = make_system(a.system, lay)
        elif hasattr(so, "retarget"):
            so.retarget(lay)
        else:
            so.lay = lay
        rec = dict(name=name, input=inp, layer=L, S=S_real, S_pad=S_pad, system=a.system, **info)
        fwd = lambda: so.fwd(q16, k16, v16, mrows)  # noqa: E731
        if gated:
            so.gate_log.clear()
        o16, lse = fwd()  # (first call: compile / caches)
        real = lay.tok < S_real
        if gated:
            g0 = [int(x) for x in so.gate_log[0].cpu().tolist()]
            rec["gate_first"] = dict(
                home_ns=g0[0],
                plan_ns=g0[1],
                raw_moved_pairs=g0[2],
                raw_moved_W=g0[3],
                raw_cross_node_W=g0[4],
                kept_home=bool(g0[0] <= g0[1] * (1 + GATE_EPS)),
            )
            so.gate_log.clear()
        if a.system == "decomposed" and a.fp64:
            if ag is None:
                from baselines.allgather.flexprefill import AllGatherFlex

                ag = AllGatherFlex(lay)
            else:
                ag.lay = lay
            oa, la = ag.fwd(q16, k16, v16, mrows)
            e = torch.stack(
                [
                    (
                        (o16[real].float() - oa[real].float()).abs().max()
                        / oa[real].float().abs().max().clamp_min(1e-30)
                    ),
                    (lse.view(W.HQ, lay.L)[:, real] - la.view(W.HQ, lay.L)[:, real]).abs().max(),
                ]
            )
            dist.all_reduce(e, op=dist.ReduceOp.MAX)
            rec["err_vs_allgather"] = dict(o_max_rel=float(e[0]), lse_max_abs=float(e[1]))
            del oa, la
        rec["fwd_ms"], rec["fwd_all"] = latency(fwd, dev, a.warmup, a.iters)
        if gated:
            g = torch.stack(so.gate_log).cpu().numpy()
            rec["gate_kept_home_calls"] = [int((g[:, 0] <= g[:, 1] * (1 + GATE_EPS)).sum()), int(len(g))]
            rec["gate_eps"] = GATE_EPS
            so.gate_log.clear()
        if planned:
            ftab = so.tables(mrows)
            rec["kernel_ms"], rec["kernel_all"] = latency(lambda: so.run(q16, k16, v16, ftab), dev, a.warmup, a.iters)
            cls = type(so)
            cls.plan_timing = True
            tp = {}
            for _ in range(PLAN_RUNS):
                so.tplan.clear()
                so.fwd(q16, k16, v16, mrows)
                for k_, v_ in so.tplan.items():
                    tp.setdefault(k_, []).append(v_)
            cls.plan_timing = False
            rec["plan_ms"] = {k_: round(float(np.median(v_)), 3) for k_, v_ in tp.items()}
            rec["assigner"] = so.assigner
        if a.system == "decomposed":
            tab = so.tables(mrows)
            nm = torch.tensor(
                [
                    float(sum(len(x[1]) for x in tab["steps"][0][1])),
                    float(sum(len(x[1]) for x in tab["steps"][1][1])),
                    float(tab["nrecv"]),
                ],
                device=dev,
            )
            gn = torch.empty(P, 3, device=dev)
            dist.all_gather_into_tensor(gn, nm)
            if rank == 0:
                gg = gn.cpu().numpy()
                rec["units"] = dict(
                    step0_recv_max=int(gg[:, 0].max()), step1_recv_max=int(gg[:, 1].max()), recv_max=int(gg[:, 2].max())
                )
                rec["bytes"] = dict(
                    inter_node_max_MiB=round(tab["inter_node_max_MiB"], 2),
                    inter_node_total_MiB=round(tab["inter_node_total_MiB"], 2),
                )
        elif planned:
            rec["heap_used_GiB"] = round(so.used / 2**30, 3)
            ftab = so.tables(mrows)
            so.run(q16, k16, v16, ftab)  # (the engine tables of this plan)
            bs = torch.as_tensor(dex_bytes(ftab, lay), device=dev)  # [3, P] sent by me -> per receiver
            allb = torch.empty((P,) + tuple(bs.shape), dtype=bs.dtype, device=dev)
            dist.all_gather_into_tensor(allb, bs.contiguous())
            if rank == 0:
                ab = allb.cpu().numpy()  # [sender, kind, receiver]
                node = np.arange(P) // gpn
                recv = ab[:, 2, :].sum(0)
                inter = np.array([ab[node != nd][:, 2][:, node == nd].sum() for nd in range(P // gpn)])
                rec["bytes"] = dict(
                    recv_max_MiB=round(recv.max() / 2**20, 2),
                    recv_mean_MiB=round(recv.mean() / 2**20, 2),
                    recv_partialO_MiB=round(ab[:, 1, :].sum() / 2**20, 2),
                    inter_node_max_MiB=round(inter.max() / 2**20, 2),
                    inter_node_total_MiB=round(inter.sum() / 2**20, 2),
                )
                pb = so.last_pb.cpu().numpy()
                Wb = so.last_W.cpu().numpy() if so.assigner else None
                ob = lay.own_b
                home = pb == ob[None, :, None]
                if Wb is None:
                    Wb = (
                        so.last_mh.float()
                        .view(W.HK, W.GQ, lay.nbh, lay.per, lay.nbh, lay.per)
                        .ne(0)
                        .sum((1, 3, 5))
                        .cpu()
                        .numpy()
                    )
                live = Wb > 0
                rec["assign"] = dict(
                    live_pairs=int(live.sum()),
                    moved_pairs=int((live & ~home).sum()),
                    moved_work_frac=round(float(Wb[~home].sum() / max(Wb.sum(), 1e-9)), 4),
                    moved_inter_node_pairs=int((live & (pb // gpn != (ob // gpn)[None, :, None])).sum()),
                    rows_moved=int((live & ~home).any(2).sum()),
                    live_rows=int(live.any(2).sum()),
                )
        elif a.system == "sparse_allgather":
            Ls = so.last
            nr = torch.tensor([float(Ls.nrecv)], device=dev)
            src = torch.as_tensor(np.repeat(np.arange(P), Ls.recv_splits), device=dev)
            ninter = torch.tensor([float((src // gpn != rank // gpn).sum().item())], device=dev)
            g = torch.empty(P, 2, device=dev)
            dist.all_gather_into_tensor(g, torch.cat([nr, ninter]))
            if rank == 0:
                tb = 2 * W.T * W.D * 2  # K + V of one (KV head, tile)
                gg = g.cpu().numpy()
                rec["bytes"] = dict(
                    recv_max_MiB=round(float(gg[:, 0].max()) * tb / 2**20, 2),
                    recv_mean_MiB=round(float(gg[:, 0].mean()) * tb / 2**20, 2),
                    inter_recv_total_MiB=round(float(gg[:, 1].sum()) * tb / 2**20, 2),
                )
        if a.fp64:
            rec["err_fp64"] = fp64_error(lay, q16, k16, v16, o16, mrows, S_pad, S_real, real, P, dev)
            torch.cuda.empty_cache()
        rec["wall_s"] = round(time.time() - t_case, 1)
        if rank == 0:
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print("[w6]", json.dumps({k_: v_ for k_, v_ in rec.items() if not k_.endswith("_all")}), flush=True)
        del q16, k16, v16, mrows, o16, lse
        torch.cuda.empty_cache()
        dist.barrier()
    dist.barrier()
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # exit at once so the other ranks' collectives abort
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
