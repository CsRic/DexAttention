"""W5 (Wan2.1-1.3B, SpargeAttention2 selection, 12 / 12 heads, FP16, 128 x 64 masks): per-GPU received bytes.
Layout = dex.common.cp_layout.Layout for every system (2P zigzag chunks of `per` tile slots).

--analytic (CPU, no data beyond the case list):
  Ring        K, V [1, H L_pad, 1, 128] fp16 ring: forward P-1 steps from rank - 1; backward P-1 K + V steps + P fp32
              dK + dV sends of the same shape.
  All-gather  NCCL all_gather of K and V [H, L_pad, 128] fp16 (again in backward) + fp32 reduce_scatter (backward);
              ring order: rank r receives the P-1 chunk steps from rank r - 1.
--gpu (one GPU, the node holding the case dumps): Dex, re-planned exactly as its W5 path (selection -> Planner on the
  GPU -> block pairs -> schedule_blocks -> per PE tables / build_fast). The planner's GPU sums use atomics, so a re-plan
  can differ slightly from the timed run's plan; repeated identical calls are compared and reported.
Inputs : common.w5_cases()
Output : $DEX_RESULT_ROOT/comm_volume/recv/w5_{ring,allgather}.jsonl (--analytic), w5_dex.jsonl (--gpu)
Example: python experiments/comm_volume/w5_comm.py --analytic ; CUDA_VISIBLE_DEVICES=0 python experiments/comm_volume/w5_comm.py --gpu
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

H = 12
M32 = 0xFFFFFFFF
LPAD = {"480p": 2048, "720p": 4864}     # padded tokens per rank


def ring_model(recv, ps, nbytes):
    for r in range(C.P):
        C.add(recv, ps, (r - 1) % C.P, r, nbytes)


def analytic():
    ring, ag = [], []
    for d, _call, name in C.w5_cases():
        res = os.path.basename(d).split("_")[0]
        L = LPAD[res]
        kv16, kv32 = 2 * H * L * C.D * 2, 2 * H * L * C.D * 4
        base = dict(workload="W5", case=name, res=res, L_pad=L)
        r = C.empty_recv(); ring_model(r, "fwd", 15 * kv16); ring_model(r, "bwd", 15 * kv16); ring_model(r, "bwd", 16 * kv32)
        ring.append(base | dict(system="Ring", recv=r))
        a = C.empty_recv(); ring_model(a, "fwd", 15 * kv16); ring_model(a, "bwd", 15 * kv16); ring_model(a, "bwd", 15 * kv32)
        ag.append(base | dict(system="All-gather", recv=a))
    C.write_jsonl(os.path.join(C.out_dir("recv"), "w5_ring.jsonl"), ring)
    C.write_jsonl(os.path.join(C.out_dir("recv"), "w5_allgather.jsonl"), ag)
    print(f"[w5] analytic: {len(ring)} cases")


def dex_tables(S, launch, mh, X, plans, dev):
    """per-GPU received bytes of the Dex message tables built from forward / backward plans [H, nbh, nbh]"""
    import w5_common as W5
    recv = C.empty_recv()
    Bf = S.block_pairs_heads_torch(mh, W5.expand(plans["fwd"], X["nbh"], dev), X["per"], owner=X["own_v"])
    work = Bf["nhalf"] * 0.5 if "nhalf" in Bf else Bf["nfull"] + 0.5 * Bf["ndiag"]
    S.schedule_blocks(Bf["bq"], Bf["bk"], Bf["pdev"], work * S.LINK["T_TILE"] / 32, Bf["prow"], Bf["pcol"], X["own_v"], C.P,
                      X["per"], block_head=Bf["block_head"], heads=(1, 1), d=C.D)
    for pe in range(C.P):
        tables, _ = S.tables_blocks(pe, Bf, X["own_v"], C.P, X["per"], "cpu")
        eng = launch.engine_tables(tables, 1, 1, C.D, "cpu")
        for _, _, nbytes, pw_ in eng["msg"].numpy().view(np.int64).reshape(-1, 4).tolist():
            if nbytes:
                C.add(recv, "fwd", pe, pw_ & M32, nbytes)
        for _, _, nbytes, pw_, _kw, _ in eng["ce"].numpy().reshape(-1, 6).tolist():
            C.add(recv, "fwd", pe, pw_ & M32, nbytes)
    if "bwd" in plans:
        Bb = S.block_pairs_heads_torch(mh, W5.expand(plans["bwd"], X["nbh"], dev), X["per"], owner=X["own_v"])
        for pe in range(C.P):
            _, mine = S.build_fast(Bb, X["own_v"], C.P, X["per"], pe, 1, 1)
            for _, _, nbytes, pw_ in mine["tabs"]["msg"].tolist():
                if nbytes:
                    C.add(recv, "bwd", pe, pw_ & M32, nbytes)
            for _, _, nbytes, pw_, _kw, _ in mine["tabs"]["ce"].tolist():
                C.add(recv, "bwd", pe, pw_ & M32, nbytes)
    return recv


def gpu(only):
    import torch
    import w5_common as W5
    from dex import planner as pl
    from dex import scheduler as S
    from dex.scheduler import launch
    dev = torch.device("cuda", 0)
    planners, recs = {}, []
    for d, call, name in C.w5_cases():
        if (only and not any(o in name for o in only)) or not os.path.isdir(d):
            continue
        t0 = time.time()
        mh, lay0, _lays, dens, S_ = W5.mask_of(d, call, dev)
        X = W5.plan_inputs(mh, lay0, dev)
        Qb, Ob, Kb = pl.fwd_bytes((1, 1), X["B"], C.D); TT = pl.FWD_MODEL["T_TILE"] / 32
        plans, same = {}, {}
        for kind in ("fwd", "bwd"):
            key = (kind, S_)
            if key not in planners:
                planners[key] = pl.Planner(H, X["nbh"], C.GPN, C.P // C.GPN, Qb, Ob, Kb, TT,
                                           bwd=((1, 1), X["B"], X["kb"]) if kind == "bwd" else None)
            G = planners[key]
            if kind == "bwd":
                G.retarget(Qb, Ob, Kb, TT, B=X["B"], kcols=X["kb"])
            plans[kind] = G(X["Wb"], X["ob"], X["gid"]).clone()
            same[kind] = bool(torch.equal(G(X["Wb"], X["ob"], X["gid"]), plans[kind]))
        recv = dex_tables(S, launch, mh, X, plans, dev)
        recs.append(dict(system="Dex", workload="W5", case=name, S=S_, density=round(dens, 4), recv=recv, plan_repeat_identical=same))
        print(f"[w5] {name} fwd NIC {C.nic_max(recv['fwd']['inter']) / C.MiB:.1f} MiB repeat-identical {same} {time.time() - t0:.1f}s", flush=True)
        del mh
        torch.cuda.empty_cache()
    C.write_jsonl(os.path.join(C.out_dir("recv"), "w5_dex.jsonl"), recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analytic", action="store_true"); ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    if a.analytic:
        analytic()
    if a.gpu:
        gpu([x for x in a.only.split(",") if x])


if __name__ == "__main__":
    main()
