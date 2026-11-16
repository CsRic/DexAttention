"""Per-GPU compute balance on the dynamic workloads (Table 4 imbalance column, W4 / W5 / W6). CPU only.

  w4  Dex, Ring and All-gather all keep the zigzag token placement and compute their own query tokens (Dex keeps every
      query at its owner on W4), so the three rows are identical. Unit: per query token and KV head, the visible keys of
      its selected 64-key blocks after the causal filter (64, or t - 64 n + 1 in its own block), x 16 query heads.
  w5  Ring / All-gather: rank r computes the rows of its own tile slots. Dex: the planner's GPU of every (head, Q block,
      KV block) pair; the timed runs do not store plans, so the plan is recomputed with the eager reference planner
      (dex.planner.plan_blocks) on the same mask, with the node-level KV refinement capped at the number of levels the
      timed run recorded (plan stage "rounds_<pass>"[1]). Unit: real query rows x real keys of every selected 128 x 64 half.
  w6  every system computes its own query rows under zigzag (Dex keeps every query home on all W6 cases). Unit: real
      token-pair area (rows / columns clipped to the real length N, diagonal tile causal), summed over the 32 heads.
Output : $DEX_RESULT_ROOT/comm_volume/balance/<W>.json  {case: {system: {fwd: {imb, work}, bwd: {...}}}}
Example: CUDA_VISIBLE_DEVICES= python experiments/comm_volume/balance_dynamic.py w4 w5 w6
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

P = C.P


def stats(w):
    w = np.asarray(w, np.float64)
    return dict(imb=float(w.max() / w.mean()), work=w.tolist())


def same(s):
    return {"fwd": s, "bwd": s}


def w4():
    import w4_comm as W4
    res = {}
    for c in C.w4_cases():
        seq, lay = c.split(":")
        f = os.path.join(C.w4_selection_dir(), seq, f"L{int(lay):02d}_topk.npy")
        if not os.path.exists(f):
            continue
        tk = np.load(f, mmap_mode="r"); S = tk.shape[1]
        area = np.zeros(P)
        for ci, (c0, c1) in enumerate(W4.zigzag_chunks(S, P)):
            s = np.asarray(tk[:, c0:c1, :]).astype(np.int64)
            t = np.arange(c0, c1)[None, :, None]; qb = t // W4.BLK
            s = np.where(s > qb, -1, s)
            vis = np.where(s == qb, t - W4.BLK * s + 1, W4.BLK)
            area[W4.chunk_owner(ci, P)] += float(np.where(s >= 0, vis, 0).sum()) * 16
        st = same(stats(area))
        res[f"{seq}_L{int(lay):02d}"] = {"Dex": st, "Ring": st, "All-gather": st}
    json.dump(res, open(os.path.join(C.out_dir("balance"), "W4.json"), "w"))
    print(f"[balance] W4: {len(res)} cases")


def w5():
    import torch
    import w5_common as W5
    from dex import planner as pl
    from dex.planner.plan import plan_blocks
    dev = torch.device("cpu")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    timed = {}
    for r in C.attention_records("W5", "dex"):
        timed[r["name"]] = r.get("plan_stages_ms") or r.get("plan_ms") or {}
    res = {}
    for d, call, name in C.w5_cases():
        if not os.path.isdir(d):
            continue
        mh, lay0, lays, _dens, _S = W5.mask_of(d, call, dev)
        X = W5.plan_inputs(mh, lay0, dev)
        H, per, nbh = W5.H, X["per"], X["nbh"]
        pos = lay0.pos_real.numpy()
        vq = np.bincount(pos // 128, minlength=lay0.n_pad).astype(np.int64)
        vk = np.bincount(pos // 64, minlength=2 * lay0.n_pad).astype(np.int64)
        mhn = mh.numpy().astype(bool)
        Ah = mhn * vq[None, :, None] * vk[None, None, :]
        rw = np.array([Ah[:, lay0.tiles_of(r)].sum() for r in range(P)], np.float64)
        Ab = Ah.reshape(H, nbh, per, nbh, 2 * per).sum((2, 4))
        Qb, Ob, Kb = pl.fwd_bytes((1, 1), X["B"], C.D); TT = pl.FWD_MODEL["T_TILE"] / 32
        ours = {}
        for kind in ("fwd", "bwd"):
            rr = timed.get(name, {}).get(f"rounds_{kind}")
            models = pl.bwd_models_b((1, 1), X["B"], X["kb"], C.GPN, P // C.GPN) if kind == "bwd" else None
            pb, _extra, _rounds = plan_blocks(X["Wb"], X["ob"], X["gid"], C.GPN, P // C.GPN, Qb, Ob, Kb, TT, models=models,
                                              kv_cost=0.0, max_kv_levels=int(rr[1]) if rr else None)
            pbn = pb.numpy()
            ours[kind] = stats(np.array([Ab[pbn == g].sum() for g in range(P)], np.float64))
        st = same(stats(rw))
        res[name] = {"Dex": ours, "Ring": st, "All-gather": st}
        print(f"[balance] W5 {name} Ring {st['fwd']['imb']:.3f} Dex {ours['fwd']['imb']:.3f}", flush=True)
    json.dump(res, open(os.path.join(C.out_dir("balance"), "W5.json"), "w"))


def w6():
    T = C.TILE
    SPAD = {"32k": 32768, "64k": 65536, "128k": 131072}
    res = {}
    for inp, L in C.w6_cases():
        mk = np.load(C.w6_dump(inp, L, "mask")); nb = mk.shape[1]
        m = np.unpackbits(mk, axis=-1, count=nb).astype(bool)
        N = np.load(C.w6_dump(inp, L, "q"), mmap_mode="r").shape[0]
        per = SPAD[inp.split("_")[-1]] // T // (2 * P)
        ln = np.minimum(T, np.maximum(0, N - np.arange(nb) * T)).astype(np.float64)
        area_row = (m * ln[None, None, :]).sum(2) * ln[None, :]
        dsel = m[:, np.arange(nb), np.arange(nb)]
        area_row = (area_row - dsel * (ln * ln - ln * (ln + 1) / 2)[None, :]).sum(0)
        aw = np.zeros(P)
        for r in range(P):
            for t in list(range(r * per, (r + 1) * per)) + list(range((2 * P - 1 - r) * per, (2 * P - r) * per)):
                if t < nb:
                    aw[r] += area_row[t]
        st = same(stats(aw))
        res[f"{inp}_L{int(L):02d}"] = {"Dex": st, "Ring": st, "All-gather": st}
    json.dump(res, open(os.path.join(C.out_dir("balance"), "W6.json"), "w"))
    print(f"[balance] W6: {len(res)} cases")


if __name__ == "__main__":
    C.assert_no_cuda()
    for arg in sys.argv[1:] or ["w4", "w5", "w6"]:
        {"w4": w4, "w5": w5, "w6": w6}[arg]()
