"""Figure 7: planning cost vs cluster size on ONE GPU and its host, weak scaling (8K tokens per GPU, 4 GPUs per node).
One point per process; no multi-GPU execution is involved.

Stages:
  planner   the compiled planner (torch.compile + CUDA graphs), forward and backward models: first call, then the median of
            3 replays, peak memory
  metadata  what every rank runs before the kernel, per sampled rank -> max over the sampled ranks; per rank the second of
            two calls. P < 64: global schedule (median of 3 calls) + this rank's tables (backward: build_fast).
            P >= 64: per-rank local planning with the collectives replaced by records of a global reference schedule with the
            same rules (static order, gateway mode 3); timing only, the exchanges themselves are not included.
Workloads (block pairs built directly, zigzag owners), each packed one input after another up to the target length, in an
order sampled by --seed; the last input is cut to its top-left corner, which stays a valid mask:
  w1  sink + window, MHA 16/16              w2  rollout masks, MHA 16/16      w3  rollout masks, GQA 32/8 (--hv 4 for both)
  w4  the 5 InfLLM-v2 masks of w4_masks.txt, GQA 32/2 (--perhead)             w5  SpargeAttention2 per-head masks, 12 heads
  w6  FlexPrefill per-KV-head masks, 8 units of 4 query heads, forward used only
The rollout masks are the star and tree groups of rollout_groups.jsonl: 83 for w2, 85 for w3.
Inputs : $DEX_DATA_ROOT/plan_scaling/masks (gen_family_masks.py)
Output : one record per point appended to --out
Example: python experiments/plan_scaling/plan_scaling.py --wl w3 --P 64 --hv 4 --seed 1 --out results/plan_scaling/w0.jsonl
"""
import argparse
import ctypes
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
import family_lists as FL  # noqa: E402
from dex import planner as pl  # noqa: E402
from dex import scheduler as S  # noqa: E402
from dex.common.env import cfg  # noqa: E402
from dex.planner.placement import bwd_time, fwd_time  # noqa: E402
from dex.scheduler import local as LO  # noqa: E402

GPN, D = 4, 128
SEED = [0]


def fam_dir():
    return os.path.join(cfg("DEX_DATA_ROOT"), "plan_scaling", "masks")


def popc(x):
    x = np.ascontiguousarray(x, np.uint64)
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)


def pairs_from_tiles(tq, tk, kind, nt, per):
    """tile list (row, col, kind 1 full / 2 causal diagonal tile) -> block pairs B (scheduler format) + work W [nb, nb]"""
    nb = nt // per; nw = (per * per + 63) // 64
    key = (tq // per) * nb + tk // per
    o = np.argsort(key, kind="stable"); key, tq, tk, kind = key[o], tq[o], tk[o], kind[o]
    uk, cnt = np.unique(key, return_counts=True)
    pid = np.repeat(np.arange(len(uk)), cnt)
    pos = ((tq % per) * per + (tk % per)).astype(np.int64)
    one = lambda p: np.left_shift(np.uint64(1), (p & 63).astype(np.uint64))   # noqa: E731
    bits = np.zeros((len(uk), nw), np.uint64); dbits = np.zeros_like(bits)
    np.bitwise_or.at(bits, (pid, pos >> 6), one(pos))
    dg = kind == 2
    np.bitwise_or.at(dbits, (pid[dg], pos[dg] >> 6), one(pos[dg]))
    prow = np.zeros(len(uk), np.uint64); pcol = np.zeros(len(uk), np.uint64)
    np.bitwise_or.at(prow, pid, one(tq % per)); np.bitwise_or.at(pcol, pid, one(tk % per))
    ndiag = np.bincount(pid, weights=dg, minlength=len(uk)).astype(np.int64)
    B = dict(bq=(uk // nb).astype(np.int64), bk=(uk % nb).astype(np.int64), nfull=cnt - ndiag, ndiag=ndiag,
             prow=prow, pcol=pcol, bits=bits, dbits=dbits)
    W = np.zeros((nb, nb), np.float32); W[B["bq"], B["bk"]] = B["nfull"] + 0.5 * ndiag
    return B, W, nb


def lambda_pairs(nt, per, sink=1, win=32):
    i = np.repeat(np.arange(nt), win); j = i - np.tile(np.arange(win), nt)
    ok = j >= 0; i, j = i[ok], j[ok]
    s_i = np.repeat(np.arange(win, nt), sink); s_j = np.tile(np.arange(sink), nt - win)
    tq = np.concatenate([i, s_i]); tk = np.concatenate([j, s_j])
    return pairs_from_tiles(tq, tk, np.where(tq == tk, 2, 1), nt, per)


def rollout_pairs(nt, per, names):
    """real tile masks (files in list order) packed one after another in a seeded shuffled order (cycled) until nt tiles"""
    names = [names[j] for j in np.random.default_rng(SEED[0]).permutation(len(names))]
    tq, tk, kd, off, i = [], [], [], 0, 0
    while off < nt:
        m = np.load(names[i % len(names)])["mask"]; i += 1
        n = min(m.shape[0], nt - off)
        r, c = np.nonzero(m[:n, :n])
        tq.append(r + off); tk.append(c + off); kd.append(m[r, c].astype(np.int64)); off += n
    return pairs_from_tiles(np.concatenate(tq), np.concatenate(tk), np.concatenate(kd), nt, per)


def perhead_pack(fam, nt, per, seed):
    """per-head tile masks packed per head in a seeded order until nt tiles per head; head h of every packed case goes to
    virtual head h. W5: 12 heads, non-causal; W6: 8 KV-head units, causal (diagonal tile kind 2)."""
    z = np.load(os.path.join(fam_dir(), f"masks_{fam}.npz"))
    names = list(z["_names"]); ns = list(z["_n"])
    order = np.random.default_rng(seed).permutation(len(names))
    causal = fam == "w6"
    H = 8 if causal else 12
    tq, tk, kd = [], [], []
    for h in range(H):
        off, i = 0, 0
        while off < nt:
            j = order[i % len(order)]; i += 1
            key = names[j].replace(":", "_L") if causal else names[j]
            n0 = int(ns[j]); m = np.unpackbits(z[key][h], axis=-1, count=n0).astype(bool)
            n = min(n0, nt - off)
            r, c = np.nonzero(m[:n, :n])
            tq.append(r + off + h * nt); tk.append(c + off + h * nt)
            kd.append(np.where((r == c) & causal, 2, 1).astype(np.int64)); off += n
    B, W, nb = pairs_from_tiles(np.concatenate(tq), np.concatenate(tk), np.concatenate(kd), H * nt, per)
    nbh = nt // per
    return B, W, nb, (np.arange(nb) // nbh).astype(np.int64), nbh, H


def med_ms(ts):
    return round(float(np.median(ts)) * 1e3, 3)


def timed(f, it=3):
    ts = []
    for _ in range(it):
        torch.cuda.synchronize(); t0 = time.perf_counter(); r = f(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return r, ts


class Emu:
    """replaces scheduler.local's collectives for one rank: queued all_to_all results, all-reduce = 2 x own sizes"""

    def __init__(self, results):
        self.q = list(results)

    def a2a(self, rows, width):
        r = self.q.pop(0)
        return r if r is not None else [np.zeros((0, width), np.int64) for _ in rows]

    @staticmethod
    def allmax(x):
        return 2 * np.asarray(x, np.int64)


def exchange_records(R, pe, P, head_of):
    """from a global schedule: what pe receives in the first exchange (its NIC messages, from each gateway) and the second
    (the landings of the NIC messages into its node, from each sender)"""
    M = R["msgs"]; nic = M[:, 5] == 0; f2i = LO.f2i
    x1 = [[] for _ in range(P)]
    for i in np.nonzero(nic & (M[:, 0] == pe))[0]:
        s, d, k, b0, c = (int(x) for x in M[i][:5])
        x1[d].append([k, s, d, b0, c, head_of(b0), f2i(R["u"][i]), f2i(R["mt"][i])])
    x2 = [[] for _ in range(P)]
    for i in np.nonzero(nic & (M[:, 1] // GPN == pe // GPN))[0]:
        s, d, k, b0, c = (int(x) for x in M[i][:5])
        x2[s].append([k, s, d, b0, f2i(R["start"][i]), f2i(R["land"][i])])
    return [np.asarray(r, np.int64).reshape(-1, 8) for r in x1], [np.asarray(r, np.int64).reshape(-1, 6) for r in x2]


def local_fwd(pe, P, B, pw, owner, per, bh, heads, dev):
    L = S.fwd._LIB
    S.schedule_blocks(B["bq"], B["bk"], B["pdev"], pw, B["prow"], B["pcol"], owner, P, per, block_head=bh, heads=heads,
                      d=D, static=True, gw=3)
    R = LO._fetch(L, P)
    head_of = (lambda b0: int(bh[b0 // per])) if bh is not None else (lambda b0: 0)   # noqa: E731
    x1, x2 = exchange_records(R, pe, P, head_of)
    e = Emu([x1, x2]); LO._a2a, LO._allmax = e.a2a, e.allmax
    t0 = time.perf_counter()
    LO.fwd_tables(pe, P, B, pw, owner, per, dev, block_head=bh, heads=heads, d=D)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def local_bwd(pe, P, B, owner, per, h, hk):
    L = S.bwd._blib(); store = {}

    def fetch(Lx):
        store["R"] = LO._fetch(Lx, P)
        return True
    S.build_fast(B, owner, P, per, pe, h, hk, gw=3, hook=fetch)
    x1, x2 = exchange_records(store["R"], pe, P, lambda b0: 0)
    e = Emu([x1, x2, None]); LO._a2a, LO._allmax = e.a2a, e.allmax
    L.bwd_set_out_dummy(ctypes.c_int(1))
    try:
        t0 = time.perf_counter(); LO.bwd_tables(pe, P, B, owner, per, h, hk); t = time.perf_counter() - t0
    finally:
        L.bwd_set_out_dummy(ctypes.c_int(0))
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wl", required=True, choices=["w1", "w2", "w3", "w4", "w5", "w6"]); ap.add_argument("--P", type=int, required=True)
    ap.add_argument("--tokens-per-gpu-k", type=int, default=8); ap.add_argument("--per", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--ranks", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hv", type=int, default=0, help="head split: N virtual heads (MHA 16/16: 4 -> (4,4); GQA 32/8: 4 -> (8,2))")
    ap.add_argument("--perhead", action="store_true", help="one virtual head per KV head (W4: the planning unit of its Dex path)")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    dev = torch.device("cuda"); torch.zeros(1, device=dev)
    P, NN = a.P, a.P // GPN
    SEED[0] = a.seed
    size_k = a.tokens_per_gpu_k * P
    rec = dict(workload=a.wl, P=P, size=size_k, seed=a.seed)
    t0 = time.perf_counter()
    per = a.per; nt = size_k * 1024 // 128; bh = None
    if a.wl == "w1":
        B, W, nb = lambda_pairs(nt, per); heads = (16, 16); owner = pl.zigzag_equal(nb, P)
    elif a.wl == "w4":
        files = [FL.w4_mask_file(fam_dir(), s, lay) for s, lay in FL.w4_cases()]
        B, W, nb = rollout_pairs(nt, per, files); heads = (32, 2); owner = pl.zigzag_equal(nb, P)
    elif a.wl in ("w2", "w3"):
        files = [FL.rollout_mask_file(fam_dir(), g["id"]) for g in FL.rollout_groups(a.wl)]
        B, W, nb = rollout_pairs(nt, per, files); heads = (16, 16) if a.wl == "w2" else (32, 8); owner = pl.zigzag_equal(nb, P)
    else:
        B, W, nb, bh, nbh, Hh = perhead_pack(a.wl, nt, per, a.seed)
        heads = (1, 1) if a.wl == "w5" else (4, 1)
        owner = np.tile(pl.zigzag_equal(nbh, P), Hh); rec.update(perhead=True, virtual_heads=Hh)
    if (a.perhead or a.hv) and a.wl in ("w1", "w2", "w3", "w4"):
        hv = a.hv or heads[1]
        assert heads[0] % hv == 0 and heads[1] % hv == 0, (heads, hv)
        nb0 = nb                                        # the block pairs replicated per virtual head (head h: + h nb0)
        B = {k_: (np.concatenate([np.asarray(v) + h * nb0 for h in range(hv)]) if k_ in ("bq", "bk")
                  else np.concatenate([np.asarray(v)] * hv)) for k_, v in B.items()}
        W = np.kron(np.eye(hv, dtype=np.float32), W); nb = hv * nb0; nbh = nb0
        bh = (np.arange(nb) // nbh).astype(np.int64); owner = np.tile(np.asarray(owner, np.int64), hv)
        heads = (heads[0] // hv, heads[1] // hv); rec.update(perhead=True, virtual_heads=hv)
    owner = np.asarray(owner, np.int64)
    hq, hk = heads; Bt = per * 128
    rec.update(per=per, block_tokens=Bt, blocks=nb, pairs=int(len(B["bq"])), live_tiles=int(popc(B["bits"].reshape(-1)).sum()),
               build_s=round(time.perf_counter() - t0, 2))
    Qb, Ob, Kb = pl.fwd_bytes(heads, Bt, D); TT = pl.FWD_MODEL["T_TILE"] * hq / 32
    colm = np.zeros(nb, np.uint64); np.bitwise_or.at(colm, B["bk"], B["pcol"]); kcols = popc(colm).astype(np.float32)
    Wt = torch.from_numpy(W).to(dev); ot = torch.as_tensor(owner, device=dev)
    pdev = {}
    for tag in ("fwd", "bwd"):
        try:
            torch._dynamo.reset(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            kt = torch.as_tensor(kcols, device=dev)
            segs, Wb, ob, gid, kb = pl.to_blocks(Wt, ot, kt if tag == "bwd" else None)
            H, m = Wb.shape[0], Wb.shape[1]
            t1 = time.perf_counter()
            G = pl.Planner(H, m, GPN, NN, Qb, Ob, Kb, TT, bwd=(heads, Bt, kb) if tag == "bwd" else None)
            pb = G(Wb, ob, gid); torch.cuda.synchronize(); rec[f"planner_{tag}_first_s"] = round(time.perf_counter() - t1, 2)
            pb, ts = timed(lambda: G(Wb, ob, gid))
            rec[f"planner_{tag}_ms"] = med_ms(ts); rec[f"planner_{tag}_peak_GiB"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
            full = torch.full((nb, nb), -1, dtype=torch.long, device=dev)
            for h, (s, sz) in enumerate(segs):
                full[s:s + sz, s:s + sz] = pb[h, :sz, :sz]
            try:
                rec[f"model_{tag}_ms"] = round(1e3 * (fwd_time(Wt, ot, full, GPN, NN, Qb, Ob, Kb, TT) if tag == "fwd"
                                                      else bwd_time(Wt, ot, full, GPN, NN, heads, Bt, kt)), 3)
            except torch.OutOfMemoryError:
                rec[f"model_{tag}_error"] = "OOM"
            pdev[tag] = full[torch.as_tensor(B["bq"], device=dev), torch.as_tensor(B["bk"], device=dev)].cpu().numpy().astype(np.int64)
            del G, full
        except torch.OutOfMemoryError as e:
            rec[f"planner_{tag}_error"] = "OOM: " + str(e).split(".")[0]
        torch.cuda.empty_cache()
    del Wt
    for tag in ("fwd", "bwd"):
        if tag not in pdev:
            pdev[tag] = owner[B["bq"]]; rec[f"pdev_{tag}"] = "owner (planner not run)"
    ranks = sorted({0, P // 2 + 1, P - 1})[:a.ranks]
    local = P >= 64                                                     # the runtime default of scheduler.local.enabled
    rec.update(ranks_sampled=ranks, metadata_path="local (emulated exchanges)" if local else "global schedule")
    try:
        Bf = dict(B, pdev=pdev["fwd"]); pw = (Bf["nfull"] + 0.5 * Bf["ndiag"]) * S.LINK["T_TILE"] * hq / 32
        if local:
            ts = [local_fwd(r, P, Bf, pw, owner, per, bh, heads, dev) for r in ranks for _ in range(2)][1::2]
            rec["meta_fwd_ms_max"] = round(max(ts) * 1e3, 3)
        else:
            R, ts = timed(lambda: S.schedule_blocks(Bf["bq"], Bf["bk"], Bf["pdev"], pw, Bf["prow"], Bf["pcol"], owner, P, per,
                                                    block_head=bh, heads=heads, d=D))
            rec["sched_fwd_ms"] = med_ms(ts); rec["msgs_fwd"] = int(len(R["msgs"]))
            tt = [timed(lambda: S.tables_blocks(r, Bf, owner, P, per, dev), 2)[1][-1] for r in ranks]
            rec["tables_fwd_ms_max"] = round(max(tt) * 1e3, 3)
            rec["meta_fwd_ms_max"] = round(rec["sched_fwd_ms"] + rec["tables_fwd_ms_max"], 3)
    except Exception as e:  # noqa: BLE001  (one failed stage leaves the point's other stages valid)
        rec["fwd_meta_error"] = repr(e)[:300]; traceback.print_exc()
    try:
        Bb = dict(B, pdev=pdev["bwd"])
        if local:
            ts = [local_bwd(r, P, Bb, owner, per, hq, hk) for r in ranks for _ in range(2)][1::2]
            rec["meta_bwd_ms_max"] = round(max(ts) * 1e3, 3)
        else:
            tt = []
            for r in ranks:
                S.build_fast(Bb, owner, P, per, r, hq, hk)
                t1 = time.perf_counter(); S.build_fast(Bb, owner, P, per, r, hq, hk); tt.append(time.perf_counter() - t1)
            rec["meta_bwd_ms_max"] = round(max(tt) * 1e3, 3)
    except Exception as e:  # noqa: BLE001
        rec["bwd_meta_error"] = repr(e)[:300]; traceback.print_exc()
    print(json.dumps(rec), flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
