"""W4 (MiniCPM4.1 InfLLM-v2, 32 query / 2 KV heads, BF16): per-GPU received bytes of Dex, Ring and All-gather from the
recorded selections (<seq>/L??_topk.npy, int16 [2, S, 96] block ids, -1 = none). CPU only.

Dex: zigzag chunks of 64-token blocks (rank r owns chunks r and 2P-1-r); needs[r, n] = a token of r's chunks selects
  block n under some KV head after the causal filter; static_plan (runs of at most 8 blocks); forward =
  input_plan + send_tables per rank: msg rows (inter-node run to the node's gateway) + copy-engine rows (same-node
  direct, gateway relay forward), nr * ROW_BYTES (K | V, both KV heads, bf16 = 1024 B per token). Backward = the same
  input runs + global_plan partial routes, one SLOT_BYTES per (sender, block) edge (same node -> owner; other node ->
  gateway, which aggregates -> owner).
Ring: K, V [mx, 2, 128] bf16 zero-padded to mx = the largest owned token count; forward P-1 sends from rank - 1;
  backward P-1 K + V sends + P fp32 dK + dV sends.
All-gather: NCCL ring all_gather of [2, mx, 2, 128] bf16 (forward; again in backward) + fp32 reduce_scatter of dK / dV;
  ring order: rank r receives the P-1 chunk steps from rank r - 1.
Inputs : common.w4_selection_dir(), common.w4_cases()
Output : $DEX_RESULT_ROOT/comm_volume/recv/w4_{dex,ring,allgather}.jsonl
Example: CUDA_VISIBLE_DEVICES= python experiments/comm_volume/w4_comm.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

from dex.attn.infllm import BLK, ROW_BYTES, RUN_MAX, SLOT_BYTES, fwd_region_layout, global_plan, input_plan, send_tables, static_plan  # noqa: E402

M32 = 0xFFFFFFFF


def zigzag_chunks(S, P_, align=BLK):
    n = 2 * P_
    nb = (S + align - 1) // align
    q, rem = divmod(nb, n)
    b = [0]
    for i in range(n):
        b.append(min(S, b[-1] + (q + (i < rem)) * align))
    return [(b[i], b[i + 1]) for i in range(n)]


def chunk_owner(ci, P_):
    return ci if ci < P_ else 2 * P_ - 1 - ci


def block_owner(chunks, nb):
    own = np.empty(nb, np.int64)
    for ci, (a_, b_) in enumerate(chunks):
        own[a_ // BLK:(b_ + BLK - 1) // BLK] = chunk_owner(ci, C.P)
    return own


def needs_matrix(tk, chunks, nb):
    """needs[r, n]: rank r has a token selecting block n (any KV head), after the causal filter"""
    needs = np.zeros((C.P, nb), bool)
    for ci, (c0, c1) in enumerate(chunks):
        r = chunk_owner(ci, C.P)
        s = np.asarray(tk[:, c0:c1, :]).astype(np.int64)
        qb = (np.arange(c0, c1) // BLK)[None, :, None]
        s = np.where(s > qb, -1, s)
        needs[r, np.unique(s[s >= 0])] = True
    return needs


def dex(S, needs, chunks):
    nb = needs.shape[1]
    own = block_owner(chunks, nb)
    st = static_plan(own, chunks, S, C.GPN, RUN_MAX)
    offs, _, _ = fwd_region_layout(st)
    Gi = input_plan(st, needs)
    recv = C.empty_recv(); bytes_in = np.zeros(C.P)
    for r in range(C.P):
        in_runs = np.nonzero(Gi["rneed"][r] & (st["run_owner"] != r))[0]
        msg, ce, _, stt = send_tables(st, Gi, r, in_runs, offs)
        bytes_in[r] = stt["bytes_in"]
        for _, _, b, pw in msg.tolist():
            for ps in ("fwd", "bwd"):
                C.add(recv, ps, r, pw & M32, b)
        for _, _, b, pw, _kw, _ in ce.tolist():
            for ps in ("fwd", "bwd"):
                C.add(recv, ps, r, pw & M32, b)
    G = global_plan(st, needs)
    assert (G["gw"] == Gi["gw"]).all()
    for s_, d_ in zip(G["sR"].tolist(), G["sD"].tolist()):
        C.add(recv, "bwd", s_, d_, SLOT_BYTES)
    tot_in = np.array([recv["fwd"]["intra"][d] + recv["fwd"]["inter"][d] for d in range(C.P)])
    return recv, dict(fwd_recv_equals_plan_bytes_in=bool(np.array_equal(tot_in, bytes_in)))


def ring_model(recv, ps, per_step, steps):
    for r in range(C.P):
        C.add(recv, ps, (r - 1) % C.P, r, per_step * steps)


def main():
    C.assert_no_cuda()
    out = {"dex": [], "ring": [], "allgather": []}
    for c in C.w4_cases():
        seq, lay = c.split(":")
        f = os.path.join(C.w4_selection_dir(), seq, f"L{int(lay):02d}_topk.npy")
        if not os.path.exists(f):
            print(f"[w4] missing {f}", flush=True); continue
        t0 = time.time()
        tk = np.load(f, mmap_mode="r"); S = tk.shape[1]
        chunks = zigzag_chunks(S, C.P); nb = (S + BLK - 1) // BLK
        mx = max(sum(chunks[ci][1] - chunks[ci][0] for ci in (r, 2 * C.P - 1 - r)) for r in range(C.P))
        base = dict(workload="W4", case=f"{seq}_L{int(lay):02d}", S=S)
        recv, chk = dex(S, needs_matrix(tk, chunks, nb), chunks)
        out["dex"].append(base | dict(system="Dex", recv=recv, checks=chk))
        kv16, kv32 = 2 * mx * 2 * C.D * 2, 2 * mx * 2 * C.D * 4
        rr = C.empty_recv(); ring_model(rr, "fwd", kv16, C.P - 1); ring_model(rr, "bwd", kv16, C.P - 1); ring_model(rr, "bwd", kv32, C.P)
        out["ring"].append(base | dict(system="Ring", recv=rr))
        ag = C.empty_recv(); ring_model(ag, "fwd", kv16, C.P - 1); ring_model(ag, "bwd", kv16, C.P - 1); ring_model(ag, "bwd", kv32, C.P - 1)
        out["allgather"].append(base | dict(system="All-gather", recv=ag))
        print(f"[w4] {base['case']} S={S} check {chk['fwd_recv_equals_plan_bytes_in']} {time.time() - t0:.1f}s", flush=True)
    for k, v in out.items():
        C.write_jsonl(os.path.join(C.out_dir("recv"), f"w4_{k}.jsonl"), v)


if __name__ == "__main__":
    main()
