"""W1-W3 attention-call test on 16 GPUs: one pass (forward or backward) of one system over a list of inputs.

Protocol (every system): each timed call starts after a global barrier; its latency is the slowest rank's barrier ->
completion; 3 warm-up calls, then the median of 10 timed calls. Launch 1 also runs the FP64 sampled-row check
(dex.common.fp64_check: 512 sampled query rows / key rows per GPU, relative L2 over all sampled rows; padding rows
excluded) and, for Dex, an fp32 check (forward: 2 sampled 128-row tiles per rank; backward: the single-GPU
backward kernel on the whole problem); later launches time only.
Q / K / V (and dO) are N(0, 0.5^2) fp16 with a constant seed; rollout pad tokens have zero K / V (and dO).

Dex (--system dex) runs the head-split plans of static_plans.py: forward = the persistent forward kernel on the
scheduler's tables (+ the exact padding correction, timed); backward = staging of Q, dO, LSE, Delta, K, V into the
symmetric region + the persistent backward kernel (staging timed). Schedules and tables are built before timing
(static planning is cached). Ring, All-gather and UA dispatch to baselines.<system>.static.run; DCP and Magi have their own
drivers in their own environments (baselines/dcp/ladder.py --measure, baselines/magi/bench.py; static_run.sh selects them).

Output: one JSON line per input appended to $DEX_RESULT_ROOT/attention_layer/<workload>/<system>_<pass>_launch<k>.jsonl
fields: input_id, workload, S, system, pass, launch, lat_ms, lat_ms_all, max_rel_err, fp64_rel_l2, fp64_rows, status,
reason (+ system-specific extras).
Example: tools/launch.sh experiments/attention_layer/static_op.py --system dex --workload w3 --pass fwd --launch 1 \
           --ids w3_32k_p103,w3_32k_p193
"""
import argparse
import datetime
import importlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(os.path.dirname(HERE))]
import static_inputs as SI  # noqa: E402
from dex.common import fp64_check as F64  # noqa: E402
from dex.common.env import cfg  # noqa: E402
from dex.common.pad_correct import correct as pad_correct  # noqa: E402

TILE, D = 128, 128
S = C = None             # scheduler and compiled extension: Dex only (baselines may run in other environments)
WARMUP, ITERS = 3, 10
FWD_CHECK_TILES = 2


def latency(fn, iters=ITERS, warmup=WARMUP):
    """-> (median ms, all ms): every call starts right after a global barrier; slowest rank's barrier -> done"""
    for _ in range(warmup):
        dist.barrier()
        torch.cuda.synchronize()
        fn()
        torch.cuda.synchronize()
    dist.barrier()
    ts = []
    for _ in range(iters):
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t = torch.tensor([time.perf_counter() - t0], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        ts.append(t.item() * 1e3)
    return float(np.median(ts)), [round(x, 3) for x in ts]


def record(case, system, pass_, launch, ms=None, ms_all=None, err=None, e64=None, status="ok", reason="", **extra):
    return dict(input_id=case["id"], workload=case["workload"].upper(), S=case["S"], system=system, **{"pass": pass_},
                launch=launch, lat_ms=ms, lat_ms_all=ms_all, max_rel_err=err,
                fp64_rel_l2={k: v["rel_l2"] for k, v in (e64 or {}).items()} or None,
                fp64_rows={k: v["rows"] for k, v in (e64 or {}).items()} or None, status=status, reason=reason, **extra)


# ---- Dex forward -------------------------------------------------------------------------------------------------
def fwd_region_bytes(t, hq, hk):
    q_slot, kv_slot = 128 * hq * D * 2, 128 * hk * D * 2
    return (t["NQ"] * q_slot + 2 * t["NKV"] * kv_slot + t["NPART"] * (2 * q_slot + hq * 128 * 4)
            + 8 * (t["NQ"] + t["NKV"] + t["NPART"]))


def fwd_build(case, mask_path, plan_path, rank, world, devt):
    """schedule + this rank's forward tables of one input"""
    z = np.load(mask_path)
    m = z["mask"].astype(np.int8)
    HQ, HK = (int(x) for x in z["heads"])
    devb, ownb, meta = S.load_plan(plan_path)
    assert meta["cp"] == world, (plan_path, meta)
    per = meta["block_tokens"] // TILE
    B = S.block_pairs(m, devb, per, owner=ownb)
    pw = (B["nfull"] + 0.5 * B["ndiag"]) * S.LINK["T_TILE"] * HQ / 32
    if S.local.enabled(world):
        tables, tm = S.local.fwd_tables(rank, world, B, pw, ownb, per, devt, heads=(HQ, HK), d=D)
    else:
        S.schedule_blocks(B["bq"], B["bk"], B["pdev"], pw, B["prow"], B["pcol"], ownb, world, per, heads=(HQ, HK), d=D)
        tables, tm = S.tables_blocks(rank, B, ownb, world, per, devt)
    return dict(case=case, z=z, m=m, T=int(z["tokens"]), HQ=HQ, HK=HK, tables=tables, tm=tm)


def dex_fwd(cases, args, rank, world, devt):
    built = []
    for c in cases:
        fm, ff, _ = SI.plan_files(c, args.head_split)
        built.append(fwd_build(c, fm, ff, rank, world, devt))
    regs = torch.tensor([fwd_region_bytes(b["tables"], b["HQ"], b["HK"]) for b in built], device=devt, dtype=torch.float64)
    dist.all_reduce(regs, op=dist.ReduceOp.MAX)
    # the largest region first (same order on every rank): every later region fits the first allocation
    order = np.argsort(-regs.cpu().numpy(), kind="stable")
    os.environ["NVSHMEM_SYMMETRIC_SIZE"] = str(int(regs.max().item() * 1.15) + (512 << 20))
    from dex.runtime import finalize_nvshmem, init_nvshmem
    init_nvshmem(dist.group.WORLD)
    recs = [dex_fwd_case(built[i], args, rank, devt) for i in order]
    dist.barrier()
    finalize_nvshmem()
    return recs


def dex_fwd_case(b, args, rank, devt):
    z, m, T, HQ, HK, tables, case = b["z"], b["m"], b["T"], b["HQ"], b["HK"], b["tables"], b["case"]
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
    ms, ms_all = latency(run)
    check = args.launch == 1
    err = torch.zeros(1, device=devt)
    run()
    torch.cuda.synchronize()
    if len(owned):
        rng = np.random.RandomState(rank)
        for li in rng.choice(len(owned), min(FWD_CHECK_TILES, len(owned)), replace=False):
            t = owned[li]
            q0 = t * TILE
            cols = np.flatnonzero(m[t] > 0)
            causal = bool(m[t, t] == 2)
            if PAD and not bool(valid[q0:q0 + TILE].any()):
                continue
            ksel = torch.cat([k[:, int(c) * TILE:int(c) * TILE + TILE] for c in cols], 1)
            vsel = torch.cat([v[:, int(c) * TILE:int(c) * TILE + TILE] for c in cols], 1)
            L_, grp = ksel.shape[1], HQ // HK
            outs = []
            for h in range(HQ):                           # fp32 reference on the row's live tiles
                s_ = (q[0, q0:q0 + TILE, h].float() @ ksel[0, :, h // grp].float().T) * scale
                if causal:
                    s_ = s_.masked_fill(torch.arange(L_, device=devt)[None, :] >
                                        torch.arange(TILE, device=devt)[:, None] + L_ - TILE, -float("inf"))
                if PAD:
                    kv_ok = torch.cat([valid[int(c) * TILE:int(c) * TILE + TILE] for c in cols])
                    s_ = s_.masked_fill(~kv_ok[None, :], -float("inf"))
                outs.append(torch.softmax(s_, -1) @ vsel[0, :, h // grp].float())
            ref = torch.stack(outs, 1)[None]
            got = out[:, li * TILE:(li + 1) * TILE]
            if PAD:
                rv = valid[q0:q0 + TILE]
                ref, got = ref[:, rv], got[:, rv]
            e = ((got.float() - ref).norm() / ref.norm().clamp_min(1e-12)).item()
            err = torch.maximum(err, torch.tensor([e], device=devt))
    e64 = None
    if check:
        ranges = np.load(case["ranges"])
        e64 = F64.check_virtual(q[0], k[0], v[0], None, ranges, scale, dict(o=out[0]), z, pos_rows=idx.cpu().numpy(),
                                group=dist.group.WORLD)
    dist.all_reduce(err, op=dist.ReduceOp.MAX)
    stat = torch.tensor([float(tables["NQ"]), float(tables["NKV"]), float(tables["NPART"])], device=devt)
    dist.all_reduce(stat)
    r = record(case, "dex", "fwd", args.launch, ms, ms_all, err.item(), e64, slots_q=int(stat[0]),
               slots_kv=int(stat[1]), slots_parto=int(stat[2]))
    del ql, kl, vl, out, lse
    torch.cuda.empty_cache()
    return r


# ---- Dex backward ------------------------------------------------------------------------------------------------
def _t32(x):
    x = np.ascontiguousarray(x)
    x = x.view(np.int32).reshape(-1) if x.dtype == np.int64 else np.ascontiguousarray(x, np.int32).reshape(-1)
    return torch.from_numpy(x if len(x) else np.zeros(2, np.int32)).cuda()


def bwd_build(case, mask_path, plan_path, rank, P):
    z = np.load(mask_path)
    m = z["mask"].astype(np.int8)
    h, hk = (int(x) for x in z["heads"])
    n = m.shape[0]
    pl = json.load(open(plan_path))
    own, _ = S.expand_plan(pl, n)
    pb = pl["block_tokens"] // TILE
    nbp = n // pb
    devb, ownb = np.asarray(pl["schedule_table"])[:nbp, :nbp], np.asarray(pl["owner"])[:nbp]
    Bp = S.block_pairs(m, devb, pb, owner=ownb)
    if S.local.enabled(P):
        glob, mine = S.local.bwd_tables(rank, P, Bp, ownb, pb, h, hk)
    else:
        glob, mine = S.build_fast(Bp, ownb, P, pb, rank, h, hk)
    return dict(case=case, z=z, m=m, h=h, hk=hk, n=n, S=n * TILE, own=own, glob=glob, mine=mine)


def dex_bwd(cases, args, rank, P, dev):
    built = []
    for c in cases:
        fm, _, fb = SI.plan_files(c, args.head_split)
        built.append(bwd_build(c, fm, fb, rank, P))
    need = max(S.region_layout(b["glob"], b["h"], b["hk"])[2] for b in built)
    os.environ["NVSHMEM_SYMMETRIC_SIZE"] = str((need + (1 << 30) + (1 << 26) - 1) // (1 << 26) * (1 << 26))
    from dex.runtime import finalize_nvshmem, init_nvshmem
    init_nvshmem(dist.group.WORLD)
    region = C.bwd_sym_alloc(need)
    pe_delta, pe_inter = C.bwd_pe_info()
    n_sm = torch.cuda.get_device_properties(0).multi_processor_count
    recs = []
    for b in built:
        recs.append(dex_bwd_case(b, args, region, pe_delta, pe_inter, n_sm, rank, P, dev))
        torch.cuda.empty_cache()
    dist.barrier()
    finalize_nvshmem()
    return recs


def dex_bwd_case(b, args, region, pe_delta, pe_inter, n_sm, rank, P, dev):
    z, m, h, hk, n, S_, own, glob, my, case = (b[k] for k in ("z", "m", "h", "hk", "n", "S", "own", "glob", "mine", "case"))
    T = TILE
    scale = D ** -0.5
    check = args.launch == 1
    g = torch.Generator(device=dev).manual_seed(1234)
    rnd = lambda *s: torch.randn(*s, generator=g, device=dev, dtype=torch.float16) * 0.5  # noqa: E731
    PAD = "tok_valid" in z.files
    valid = torch.as_tensor(z["tok_valid"].astype(bool), device=dev) if PAD else None
    ref = None
    if check:            # global tensors: the references need the whole problem; LSE is a constant plausible value
        Q, K, V, dO, O = rnd(S_, h, D), rnd(S_, hk, D), rnd(S_, hk, D), rnd(S_, h, D), rnd(S_, h, D)
        LSE = torch.full((h, S_), 6.0, device=dev)
        if PAD:          # pad tokens: K = V = 0 and no loss on them (dO = 0), as in training
            K[~valid] = 0
            V[~valid] = 0
            dO[~valid] = 0
        Delta = (dO.float() * O.float()).sum(-1).t().contiguous()
        tk, tp, it = [], [0], []                     # single-GPU backward kernel on every live 64 x 64 block
        r_grp = h // hk
        for gg in range(hk):
            for j in range(n):
                rows = np.flatnonzero(m[:, j])
                for hn in range(2):
                    for i in rows[::-1]:
                        for mb in (1, 0):
                            if m[i, j] == 2:
                                if mb < hn:
                                    continue
                                causal = mb == hn
                            else:
                                causal = False
                            it += [(int(i) << 9) | (mb << 8) | (hq << 1) | int(causal)
                                   for hq in range(gg * r_grp, (gg + 1) * r_grp)]
                    tk.append((j << 9) | (hn << 8) | gg)
                    tp.append(len(it))
        i32 = lambda x: torch.tensor(x, dtype=torch.int32, device=dev)  # noqa: E731
        dq_acc = torch.zeros(S_, h, D, device=dev)
        dk_acc = torch.zeros(S_, hk, D, device=dev)
        dv_acc = torch.zeros_like(dk_acc)
        C.sparse_bwd(Q, K, V, dO, LSE, Delta, dq_acc, None, None, dk_acc, dv_acc, i32(tk), i32(tp), i32(it), scale)
        dq_r = torch.empty_like(Q)
        C.dq_convert(dq_acc, dq_r, scale)
        ref = (dq_r, dk_acc.half(), dv_acc.half())
        del dq_acc, dk_acc, dv_acc
        torch.cuda.synchronize()
    st = my["stats"]
    offs, sizes, total = S.region_layout(glob, h, hk)
    NQS, NKS = glob["NQS"], glob["NKS"]
    tabs = [_t32(my["tabs"][k_]) for k_ in S.TABLE_ORDER + S.RECT_ORDER]
    ce_tab = torch.from_numpy(np.ascontiguousarray(my["tabs"]["ce"], np.int64))
    soffs, ssize = S.scratch_layout(st, glob, h, hk, n_sm - 1)
    scratch = torch.zeros(ssize, dtype=torch.int32, device=dev)
    oq = np.flatnonzero(own == rank)
    rows = (torch.as_tensor(oq, device=dev)[:, None] * T + torch.arange(T, device=dev)).reshape(-1)
    if check:
        Ql, dOl, Kl, Vl = Q[rows], dO[rows], K[rows], V[rows]
        Ll, Dl = LSE[:, rows], Delta[:, rows]
    else:                # the same distributions, generated locally
        nr = rows.numel()
        Ql, dOl, Kl, Vl, Ol = rnd(nr, h, D), rnd(nr, h, D), rnd(nr, hk, D), rnd(nr, hk, D), rnd(nr, h, D)
        if PAD:
            Kl[~valid[rows]] = 0
            Vl[~valid[rows]] = 0
            dOl[~valid[rows]] = 0
        Ll = torch.full((h, nr), 6.0, device=dev)
        Dl = (dOl.float() * Ol.float()).sum(-1).t().contiguous()
        del Ol
    no = len(oq)
    qb = region[offs[1]:offs[1] + NQS * sizes[0]].view(NQS, sizes[0])
    kvr = region[offs[2]:offs[2] + NKS * sizes[1]].view(NKS, sizes[1])
    qbytes, kbytes = T * h * D * 2, T * hk * D * 2
    v16 = lambda x, a_, b_: x[:no, a_:b_].view(torch.float16)  # noqa: E731
    v32 = lambda x, a_, b_: x[:no, a_:b_].view(torch.float32)  # noqa: E731

    def stage():         # owned tiles = slots [0, no)
        v16(qb, 0, qbytes).copy_(Ql.view(no, T * h * D))
        v16(qb, qbytes, 2 * qbytes).copy_(dOl.view(no, T * h * D))
        v32(qb, 2 * qbytes, 2 * qbytes + h * T * 4).copy_(Ll.view(h, no, T).permute(1, 0, 2).reshape(no, h * T))
        v32(qb, 2 * qbytes + h * T * 4, 2 * qbytes + 2 * h * T * 4).copy_(Dl.view(h, no, T).permute(1, 0, 2).reshape(no, h * T))
        v16(kvr, 0, kbytes).copy_(Kl.view(no, T * hk * D))
        v16(kvr, kbytes, 2 * kbytes).copy_(Vl.view(no, T * hk * D))
    dq = torch.zeros(no * T, h, D, device=dev, dtype=torch.float16)
    dk = torch.zeros(no * T, hk, D, device=dev, dtype=torch.float16)
    dv = torch.zeros_like(dk)
    dims = [NQS, NKS, glob["NDQL"], glob["NDKL"], h, hk, D, st["tasks"], st["n_init"], st["n_qm"], st["n_km"],
            st["n_remote"], st["n_start"], 0, st["n_qagg"] + st["n_kagg"], st["n_pq"],
            1, st["n_rect"], st["G1"], st["per"], st["nw"], st["arena"], st["n_groups"]]

    def run():
        stage()
        C.dist_bwd_run(offs, sizes, dims, tabs, pe_delta, pe_inter, scratch, soffs, dq, dk, dv, scale, n_sm, ce_tab)
    err, errs, e64, status, reason = None, None, None, "ok", ""
    ms = ms_all = ms_stage = None
    try:
        run()
        torch.cuda.synchronize()
        if check:
            pr = rows.cpu().numpy()
            e64 = F64.check_virtual(Q, K, V, dO, np.load(case["ranges"]), scale, dict(dq=dq, dk=dk, dv=dv), z, pos_rows=pr,
                                    pos_kv=pr, group=dist.group.WORLD, lse_in=LSE.t(), o_in=O)
            e = []
            for got, r_ in ((dq, ref[0]), (dk, ref[1]), (dv, ref[2])):
                rr = r_[rows].float()
                if PAD:
                    ok = valid[rows]
                    rr, got = rr[ok], got[ok]
                e.append(((got.float() - rr).norm() / rr.norm().clamp_min(1e-12)).item())
            et = torch.tensor(e, device=dev)
            dist.all_reduce(et, op=dist.ReduceOp.MAX)
            err, errs = et.max().item(), [round(x, 5) for x in et.tolist()]
        for _ in range(WARMUP):
            dist.barrier()
            torch.cuda.synchronize()
            run()
            torch.cuda.synchronize()
        ts, tst = [], []
        for _ in range(ITERS):               # timed call, then the staging alone (reported separately)
            dist.barrier()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            run()
            torch.cuda.synchronize()
            x = torch.tensor([time.perf_counter() - t0], device=dev)
            dist.all_reduce(x, op=dist.ReduceOp.MAX)
            ts.append(x.item() * 1e3)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            stage()
            torch.cuda.synchronize()
            tst.append(time.perf_counter() - t1)
        ms, ms_all, ms_stage = float(np.median(ts)), [round(x, 3) for x in ts], float(np.median(tst)) * 1e3
    except Exception as ex:  # noqa: BLE001  (report the input, continue with the next one)
        status, reason = "error", repr(ex)[:300]
        print(f"[rank {rank}] {case['id']}: {ex}", flush=True)
    return record(case, "dex", "bwd", args.launch, ms, ms_all, err, e64, status, reason, errs_dq_dk_dv=errs,
                  staging_ms=ms_stage, region_MiB=total / 2 ** 20)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", default="dex", choices=["dex", "ring", "allgather", "ua"])
    ap.add_argument("--workload", required=True, choices=["w1", "w2", "w3"])
    ap.add_argument("--pass", dest="pass_", required=True, choices=["fwd", "bwd"])
    ap.add_argument("--launch", type=int, default=1, help="1 = timing + correctness checks, >1 = timing only")
    ap.add_argument("--ids", default="", help="comma list of input ids (default: every input of the workload)")
    ap.add_argument("--head-split", type=int, default=4, help="Dex: head split of the plans")
    ap.add_argument("--out", default="", help="default $DEX_RESULT_ROOT/attention_layer/<w>/<system>_<pass>_launch<k>.jsonl")
    a = ap.parse_args()
    cases = SI.inputs(a.workload)
    if a.ids:
        want = a.ids.split(",")
        cases = [c for c in cases if c["id"] in want]
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2))
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dev = torch.device("cuda")
    if a.system == "dex":
        global S, C
        from dex import dex_cuda as C, scheduler as S
        recs = (dex_fwd if a.pass_ == "fwd" else dex_bwd)(cases, a, rank, world, dev)
    else:
        mod = importlib.import_module(f"baselines.{a.system}.static")
        protocol = dict(warmup=WARMUP, iters=ITERS, launch=a.launch, fp64=a.launch == 1, latency=latency, record=record)
        recs = mod.run(cases, a.pass_, protocol)
    if rank == 0:
        out = a.out or os.path.join(cfg("DEX_RESULT_ROOT"), "attention_layer", a.workload,
                                    f"{a.system}_{a.pass_}_launch{a.launch}.jsonl")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "a") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
                print(f"{r['input_id']:28s} {a.system} {a.pass_} {r['lat_ms']} ms  fp64 {r['fp64_rel_l2']}  {r['status']}",
                      flush=True)


if __name__ == "__main__":
    main()
