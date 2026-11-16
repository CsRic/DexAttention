"""Dex on W1-W3: per-GPU received bytes from the per-PE message tables the kernels execute, rebuilt on the CPU from
the saved head_split=4 plans (no planner run, no GPU).

Forward: block_pairs -> schedule_blocks (global schedule, gateway relay) -> per PE tables_blocks -> engine_tables:
  msg rows (src_off, dst_off, bytes, pe | sig << 32) = inter-node input pushes to the receiving node's gateway and
  inter-node partial O (fp32 O + fp32 LSE per row); copy-engine rows (src, dst, bytes, pe | sig << 32, kind | word << 32,
  1) = intra-node direct inputs (kind 0), gateway relay forwards (kind 1), partial O (kind 2). Every row is sent by this
  PE to `pe` (low 32 bits).
Backward: block_pairs -> build_fast per PE: msg rows (the first n_start are input pushes, the rest fp32 dQ / dK+dV
  partials) and copy-engine rows (kind 0 input, 1 relay forward, 2 partial). Inputs per Q tile = Q + dO fp16 + LSE +
  Delta fp32; per KV tile = K + V fp16.
Check: the forward input bytes of the per-PE tables equal the global schedule's message list.
Inputs : common.static_names / mask_path / dex_plan_paths.   Output: $DEX_RESULT_ROOT/comm_volume/recv/dex_<W>.jsonl
Example: CUDA_VISIBLE_DEVICES= python experiments/comm_volume/dex_static_comm.py --workload W2
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

M32 = 0xFFFFFFFF
KIND = {0: "input", 1: "relay forward", 2: "partial"}


def fwd_counts(S, launch, m, plan, hq, hk):
    devb, ownb, meta = launch.load_plan(plan)
    per = meta["block_tokens"] // C.TILE
    B = S.block_pairs(m, devb, per, owner=ownb)
    pw = (B["nfull"] + 0.5 * B["ndiag"]) * S.LINK["T_TILE"] * hq / 32
    R = S.schedule_blocks(B["bq"], B["bk"], B["pdev"], pw, B["prow"], B["pcol"], ownb, C.P, per, heads=(hq, hk), d=C.D)
    msgs = np.asarray(R["msgs"]).reshape(-1, 7)
    qb, kvb = C.TILE * hq * C.D * 2, 2 * C.TILE * hk * C.D * 2
    g_in = np.zeros((C.P, C.P))
    for s_, r_, kind, _first, cnt, _typ, _par in msgs.tolist():
        g_in[s_, r_] += cnt * (qb if kind == 0 else kvb)
    recv = C.empty_recv(); kinds = Counter(); t_in = np.zeros((C.P, C.P))
    for pe in range(C.P):
        tables, _ = S.tables_blocks(pe, B, ownb, C.P, per, "cpu")
        eng = launch.engine_tables(tables, hq, hk, C.D, "cpu")
        mt = eng["msg"].numpy().view(np.int64).reshape(-1, 4)
        for i, (_so, _do, nbytes, pw_) in enumerate(mt.tolist()):
            if nbytes == 0:
                continue
            dst = pw_ & M32; k = "input" if i < eng["n_start"] else "partial"
            C.add(recv, "fwd", pe, dst, nbytes); kinds[(k, "intra" if C.node(pe) == C.node(dst) else "inter")] += nbytes
            if k == "input":
                t_in[pe, dst] += nbytes
        for _so, _do, nbytes, pw_, kw, _ in eng["ce"].numpy().reshape(-1, 6).tolist():
            dst, kind = pw_ & M32, kw & M32
            C.add(recv, "fwd", pe, dst, nbytes); kinds[(KIND[kind], "intra" if C.node(pe) == C.node(dst) else "inter")] += nbytes
            if kind in (0, 1):
                t_in[pe, dst] += nbytes
    np.fill_diagonal(t_in, 0); np.fill_diagonal(g_in, 0)
    return recv, kinds, dict(fwd_inputs_match_global_schedule=bool(np.allclose(t_in, g_in)), fwd_per=per)


def bwd_counts(S, m, plan_json, hq, hk):
    pl = json.load(open(plan_json))
    pb = pl["block_tokens"] // C.TILE; nbp = m.shape[0] // pb
    devb = np.asarray(pl["schedule_table"])[:nbp, :nbp]; ownb = np.asarray(pl["owner"])[:nbp]
    Bp = S.block_pairs(m, devb, pb, owner=ownb)
    recv = C.empty_recv(); kinds = Counter()
    for pe in range(C.P):
        _, mine = S.build_fast(Bp, ownb, C.P, pb, pe, hq, hk)
        tb, ns = mine["tabs"], mine["stats"]["n_start"]
        for i, (_so, _do, nbytes, pw_) in enumerate(tb["msg"].tolist()):
            if nbytes == 0:
                continue
            dst = pw_ & M32; k = "input" if i < ns else "partial"
            C.add(recv, "bwd", pe, dst, nbytes); kinds[(k, "intra" if C.node(pe) == C.node(dst) else "inter")] += nbytes
        for _so, _do, nbytes, pw_, kw, _ in tb["ce"].tolist():
            dst, kind = pw_ & M32, kw & M32
            C.add(recv, "bwd", pe, dst, nbytes); kinds[(KIND[kind], "intra" if C.node(pe) == C.node(dst) else "inter")] += nbytes
    return recv, kinds, dict(bwd_per=pb)


def main():
    C.assert_no_cuda()
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True, choices=["W1", "W2", "W3"])
    ap.add_argument("--only", default="", help="comma-separated name substrings")
    a = ap.parse_args()
    from dex import scheduler as S
    from dex.scheduler import launch
    only = [x for x in a.only.split(",") if x]
    recs = []
    for name in C.static_names(a.workload):
        if only and not any(o in name for o in only):
            continue
        t0 = time.time()
        mask, pf, pb = C.dex_plan_paths(a.workload, name)
        rec = dict(system="Dex", workload=a.workload, case=name)
        miss = [p for p in (mask, pf, pb) if not os.path.exists(p)]
        if miss:
            rec["na"] = f"missing {miss}"; recs.append(rec); continue
        z = np.load(mask); m = z["mask"].astype(np.int8); hq, hk = (int(x) for x in z["heads"])
        rf, kf, cf = fwd_counts(S, launch, m, pf, hq, hk)
        rb, kb, cb = bwd_counts(S, m, pb, hq, hk)
        recv = C.empty_recv(); recv["fwd"] = rf["fwd"]; recv["bwd"] = rb["bwd"]
        rec.update(heads=[hq, hk], recv=recv, checks=cf | cb,
                   by_kind={"fwd": {f"{k}|{l}": v for (k, l), v in kf.items()}, "bwd": {f"{k}|{l}": v for (k, l), v in kb.items()}})
        print(f"[dex] {a.workload} {name} fwd NIC {C.nic_max(recv['fwd']['inter']) / C.MiB:.1f} MiB "
              f"check {cf['fwd_inputs_match_global_schedule']} {time.time() - t0:.1f}s", flush=True)
        recs.append(rec)
    C.write_jsonl(os.path.join(C.out_dir("recv"), f"dex_{a.workload}.jsonl"), recs)


if __name__ == "__main__":
    main()
