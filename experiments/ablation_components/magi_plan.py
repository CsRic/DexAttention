"""MagiAttention as the replaced system X: Magi's own dispatch plan and execution order -> plan JSON (owner, schedule_table, gpu_kernels),
so its plan can run on Dex's executor. Runs exactly Magi's functions without CUDA: a gloo group of 16 CPU processes (the solver's
all_gather_object calls only); GroupCollectiveArg / CommMeta are replaced by recorders, so no comm kernel or CUDA tensor is created.
Magi's settings: chunk = ceil(T / (MIN_CHUNKS_PER_RANK cp)) rounded down to a multiple of 128, its padding, MinHeap dispatch with equal chunk
counts, QO comm off, static solver, overlap degree 1: every slice is computed on the rank holding its Q; stage 0 = slices against the rank's
own K / V, stage 1 = its single remote stage, whose K / V are fetched during stage 0. Dispatch runs on the padded tile layout of the mask
file, so every chunk is whole 128-token tiles and the plan maps exactly onto the tile mask (comps are Magi's own slices in its list order).
Plan: block_tokens = 128, owner[tile], schedule_table[q][kv] (= owner[q], -1 dead), gpu_kernels: comp {rows, cols, start 0 | 1, fused}, comm
{block, src, dst, io 'i', rc 'c', role, start 1}.
Validation: Magi slice areas re-derived from the band geometry == AttnSlice.area; every live tile covered exactly once and fully, no dead tile
touched; every comp's Q local, its K / V local (step 0) or received in time; remote K / V tokens per rank == Magi's num_remote_kv_tokens.
Output: $DEX_DATA_ROOT/plans/ablation/magi/<id>_magi.json (+ raw/). Example (CPU only, Magi's Python):
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 $DEX_MAGI_PYTHON experiments/ablation_components/magi_plan.py"""
import argparse
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [_ROOT, os.path.dirname(os.path.abspath(__file__))]
import numpy as np  # noqa: E402

import baselines.magi  # noqa: E402,F401  (kernel backend defaults, typing)
from baselines.magi.slices import slices_from_padded_segments, slices_from_ranges  # noqa: E402
import sample as SM  # noqa: E402

TILE, CP, GPN = 128, 16, 4
TYPES = ("full", "causal", "bi_causal", "inv_causal")


def inputs():
    for w, c in SM.cases():
        z = np.load(c["mask"])
        yield dict(name=c["id"], kind="lambda" if w == "w1" else "rollout", heads=tuple(int(x) for x in z["heads"]), mask=c["mask"], ranges=c["ranges"])


def magi_slices(inp):
    if inp["kind"] == "lambda":
        T, qr, kr, ty = slices_from_ranges(np.load(inp["ranges"]))
        return T, qr, kr, ty
    z = np.load(inp["mask"])
    qr, kr, ty = slices_from_padded_segments(z["segs"])
    return int(z["tokens"]), qr, kr, ty


def rl(rs):
    return [[int(r.start), int(r.end)] for r in rs]


def solve_worker(rank, jobs, port):
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=f"tcp://localhost:{port}", rank=rank, world_size=CP)
    from magi_attention import env
    from magi_attention.api.functools import apply_padding, compute_pad_size
    from magi_attention.common import AttnRanges
    from magi_attention.common.enum import AttnMaskType
    from magi_attention.config import DistAttnConfig
    from magi_attention.meta._make_dispatch_meta import make_dispatch_meta_from_qk_ranges
    from magi_attention.meta.solver import dist_attn_solver as DS

    class Rec:                                                 # records Magi's arguments, builds nothing
        def __init__(self, **kw):
            self.__dict__.update(kw)
    DS.GroupCollectiveArg = Rec; DS.CommMeta = Rec
    assert not env.comm.is_qo_comm_enable() and not env.comm.is_hierarchical_comm_enable()
    assert not env.comm.is_native_grpcoll_enable() and not env.general.is_flatten_head_groups_enable()
    cfg = DistAttnConfig()
    for inp in jobs:
        T, qr, kr, ty = magi_slices(inp)
        t0 = time.perf_counter()
        auto = -(-T // (env.general.min_chunks_per_rank() * CP))
        chunk = min(auto, auto // TILE * TILE)
        q, k = AttnRanges.from_ranges(qr), AttnRanges.from_ranges(kr)
        mt = [AttnMaskType(t if t in TYPES else {"bicausal": "bi_causal", "invcausal": "inv_causal"}[t]) for t in ty]
        pad = compute_pad_size(T, CP, chunk)
        if pad:
            q, k, mt = apply_padding(q_ranges=q, k_ranges=k, attn_mask_type=mt, total_seqlen=T, pad_size=pad)
        Tm = T + pad
        dq, dk = make_dispatch_meta_from_qk_ranges(
            q_ranges=q, k_ranges=k, attn_mask_type=mt, total_seqlen_q=Tm, total_seqlen_k=Tm, chunk_size=chunk,
            cp_size=CP, cp_rank=rank, dispatch_config=cfg.dispatch_config, is_same_source=True,
            is_q_permutable=True, is_k_permutable=True, uneven_shard=cfg.dispatch_config.uneven_shard)
        t1 = time.perf_counter()
        sv = DS.DistAttnSolver(num_heads_q=inp["heads"][0], num_heads_kv=inp["heads"][1], head_dim=128,
                               overlap_config=cfg.overlap_config, cp_group=dist.group.WORLD)
        sv.solve(q_ranges=q, k_ranges=k, attn_mask_type=mt, dispatch_meta_q=dq, dispatch_meta_k=dk)
        cm = sv.make_comm_meta()
        t2 = time.perf_counter()
        he = sv.host_rank_entry_this_rank
        sl = lambda lst: [[int(x.q_range.start), int(x.q_range.end), int(x.k_range.start), int(x.k_range.end),
                           x.mask_type.value, int(x.area)] for x in lst]
        rec = dict(host_q=rl(he.host_q_ranges_global), host_k=rl(he.host_k_ranges_global),
                   host_slices=sl(he.attn_calc_host_slice_local_list),
                   bucket_global=sl(he.attn_calc_slice_global_list),
                   stages=[dict(remote_k=rl(e.remote_k_ranges_global), slices=sl(e.attn_calc_remote_slice_local_list))
                           for e in sv.remote_rank_entry_per_stage_this_rank],
                   gc=[dict(input_split=list(g.input_split_size_list), output_split=list(g.output_split_size_list),
                            dst=[list(x) for x in g.dst_indices_list], src=list(g.src_index_list))
                       for g in cm.kv_group_collective_args_list],
                   num_remote_kv_tokens_per_stage=list(cm.num_remote_kv_tokens_per_stage),
                   overlap_degree=sv.overlap_degree, overlap_chunk_size=sv.overlap_chunk_size,
                   overlap_num_chunks=sv.overlap_num_chunks, t_dispatch_s=t1 - t0, t_solve_s=t2 - t1)
        allr = [None] * CP
        dist.all_gather_object(allr, rec)
        if rank == 0:
            meta = dict(name=inp["name"], kind=inp["kind"], heads=list(inp["heads"]), T=T,
                        auto_chunk=auto, chunk=chunk, magi_pad=pad, partitions=[list(map(int, p)) for p in dq.partitions],
                        n_slices=len(qr), overlap_config=repr(cfg.overlap_config), dispatch_alg=repr(cfg.dispatch_config.alg),
                        mask=inp["mask"],
                        q_ranges=qr, k_ranges=kr, types=ty, ranks=allr)
            json.dump(meta, open(f"{SM.ablation_dir('magi', 'raw')}/{inp['name']}.json", "w"))
            print(f"[magi] {inp['name']}: T {T} chunk {chunk} pad {pad} degree {sv.overlap_degree} "
                  f"dispatch {t1 - t0:.2f}s solve {t2 - t1:.2f}s", flush=True)
        dist.barrier()
    dist.destroy_process_group()


# ----------------------------------------------------------------------------------------------- geometry
BIG = 1 << 40


def band(ty, qa, qb, ka, kb):
    """Magi mask type on a (local) slice -> inclusive bounds on (k - q): CAUSAL = bottom-right aligned,
    INVCAUSAL = top-left aligned, BICAUSAL = both (magi_bench.token_mask / AttnSlice.area semantics)"""
    nq, nk = qb - qa, kb - ka
    lo, hi = -BIG, BIG
    if ty in ("causal", "bi_causal"):
        hi = ka - qa + nk - nq
    if ty in ("inv_causal", "bi_causal"):
        lo = ka - qa
    return lo, hi


def pieces(rng_list, a, b):
    """local [a, b) in the concatenation of merged global ranges -> [(local_start, local_end, global_offset)]"""
    out, pos = [], 0
    for s, e in rng_list:
        n = e - s
        lo, hi = max(a, pos), min(b, pos + n)
        if lo < hi:
            out.append((lo, hi, s - pos))
        pos += n
    return out


def tile_areas(q0, q1, k0, k1, lo, hi):
    """global rectangle [q0,q1) x [k0,k1) with lo <= k - q <= hi -> {(qt, kt): area}"""
    res = {}
    if lo <= -BIG // 2 and hi >= BIG // 2:                    # FULL: closed form
        for qt in range(q0 // TILE, (q1 - 1) // TILE + 1):
            nq = min(q1, (qt + 1) * TILE) - max(q0, qt * TILE)
            for kt in range(k0 // TILE, (k1 - 1) // TILE + 1):
                res[(qt, kt)] = nq * (min(k1, (kt + 1) * TILE) - max(k0, kt * TILE))
        return res
    q = np.arange(q0, q1, dtype=np.int64)
    kl = np.maximum(k0, q + lo); kh = np.minimum(k1 - 1, q + hi)
    ok = kh >= kl
    q, kl, kh = q[ok], kl[ok], kh[ok]
    if not len(q):
        return res
    kts = np.arange(kl.min() // TILE, kh.max() // TILE + 1)
    ov = np.clip(np.minimum(kh[:, None], kts[None, :] * TILE + TILE - 1) - np.maximum(kl[:, None], kts[None, :] * TILE) + 1, 0, None)
    qt = q // TILE
    for t in np.unique(qt):
        s = ov[qt == t].sum(0)
        for j in np.flatnonzero(s):
            res[(int(t), int(kts[j]))] = int(s[j])
    return res


def slice_area(lo, hi, qa, qb, ka, kb):
    q = np.arange(qa, qb, dtype=np.int64)
    return int(np.clip(np.minimum(kb - 1, q + hi) - np.maximum(ka, q + lo) + 1, 0, None).sum())


# ----------------------------------------------------------------------------------------------- conversion
def tile_full_area(m):
    return np.where(m == 1, TILE * TILE, np.where(m == 2, TILE * (TILE + 1) // 2, 0)).astype(np.int64)


def owner_of_tokens(d):
    c, Tm = d["chunk"], d["T"] + d["magi_pad"]
    own = np.full(Tm, -1, np.int64)
    for r, part in enumerate(d["partitions"]):
        for ch in part:
            own[ch * c:(ch + 1) * c] = r
    assert (own >= 0).all()
    return own


def rank_stats(d):
    """Magi-native per-rank numbers (its own token coordinates)"""
    area = np.array([sum(s[5] for s in r["host_slices"]) + sum(s[5] for st in r["stages"] for s in st["slices"])
                     for r in d["ranks"]], np.int64)
    rkv = np.array([sum(r["num_remote_kv_tokens_per_stage"]) for r in d["ranks"]], np.int64)
    gc_out = np.array([sum(sum(g["output_split"]) for g in r["gc"]) for r in d["ranks"]], np.int64)
    inter = np.array([sum(n for g in r["gc"] for n, s in zip(g["output_split"], g["src"]) if s // GPN != i // GPN)
                      for i, r in enumerate(d["ranks"])], np.int64)
    return dict(area=area, remote_kv=rkv, gc_out=gc_out, remote_kv_inter=inter)


def summ(x):
    x = np.asarray(x, np.float64)
    return dict(max=float(x.max()), mean=float(x.mean()), min=float(x.min()), max_over_mean=float(x.max() / max(x.mean(), 1e-12)),
                total=float(x.sum()))


def build_gk_two_stage(owner, m, comps_by_rank=None):
    """comm list (remote K/V tiles, start 1) from the comps; comps either given (Magi order) or derived by stage"""
    gk = [[] for _ in range(CP)]
    live = m > 0
    if comps_by_rank is None:
        comps_by_rank = [[] for _ in range(CP)]
        for g in range(CP):
            for stage in (0, 1):
                colset = {}
                for qt in np.flatnonzero(owner == g):
                    cols = [int(k) for k in np.flatnonzero(live[qt]) if (owner[k] == g) == (stage == 0)]
                    if not cols:
                        continue
                    if qt in cols and m[qt, qt] == 2:
                        comps_by_rank[g].append(dict(kind="comp", rows=[int(qt)], cols=[int(qt)], start=float(stage), fused=False))
                    rest = tuple(c for c in cols if not (c == qt and m[qt, qt] == 2))
                    if rest:
                        colset.setdefault(rest, []).append(int(qt))
                for cs, qs in colset.items():
                    comps_by_rank[g].append(dict(kind="comp", rows=qs, cols=list(cs), start=float(stage),
                                                 fused=len(qs) > 1 or len(cs) > 1))
    sent = set()
    for g in range(CP):
        for c in comps_by_rank[g]:
            for k in c["cols"]:
                if owner[k] != g and (k, g) not in sent:
                    sent.add((k, g))
                    gk[int(owner[k])].append(dict(kind="comm", block=int(k), src=int(owner[k]), dst=g, io="i", rc="c", role="send", start=1.0))
                    gk[g].append(dict(kind="comm", block=int(k), src=int(owner[k]), dst=g, io="i", rc="c", role="recv", start=1.0))
    for g in range(CP):                                       # comps first, Magi list order inside a step
        gk[g] = comps_by_rank[g] + gk[g]
    return gk


def validate(owner, m, gk):
    """exact cover of live tiles by comps on the Q owner; K/V local or received with start <= step"""
    N = len(owner); cnt = np.zeros((N, N), np.int32); have = {}
    for g in range(CP):
        for it in gk[g]:
            if it["kind"] == "comm" and it["dst"] == g:
                assert owner[it["block"]] == it["src"]
                have[(it["block"], g)] = min(have.get((it["block"], g), 9), it["start"])
    for g in range(CP):
        for it in gk[g]:
            if it["kind"] != "comp":
                continue
            for qt in it["rows"]:
                assert owner[qt] == g, "Q not local"
                for kt in it["cols"]:
                    cnt[qt, kt] += 1
                    if owner[kt] != g:
                        assert have.get((kt, g), 9) <= it["start"], ("K/V arrives late", qt, kt, g)
    assert (cnt[m > 0] == 1).all(), f"live tiles covered != 1: {int((cnt[m > 0] != 1).sum())}"
    assert (cnt[m == 0] == 0).all(), "dead tile computed"
    rkv = np.array([sum(1 for (k, d) in have if d == g) for g in range(CP)]) * TILE
    return rkv


def tile_load(owner, m):
    w = np.where(m == 1, 1.0, np.where(m == 2, 0.5, 0.0))
    return np.bincount(owner, weights=w.sum(1), minlength=CP)


def convert_pad(d, m):
    """exact: comps = Magi's own slices mapped back to global tiles"""
    T = d["T"]; N = m.shape[0]
    assert N * TILE == T, (N, T)
    own_tok = owner_of_tokens(d)[:T]
    owner = own_tok[::TILE].copy()
    assert (own_tok.reshape(N, TILE) == owner[:, None]).all(), "a Magi chunk is not whole tiles"
    full = tile_full_area(m)
    cov = {}; comps = [[] for _ in range(CP)]; area_mismatch = 0; multi_piece = 0
    for g, r in enumerate(d["ranks"]):
        for stage, (slices, krng) in enumerate([(r["host_slices"], r["host_k"])] +
                                               [(st["slices"], st["remote_k"]) for st in r["stages"]]):
            for qa, qb, ka, kb, ty, ar in slices:
                lo, hi = band(ty, qa, qb, ka, kb)
                if slice_area(lo, hi, qa, qb, ka, kb) != ar:
                    area_mismatch += 1
                qp, kp = pieces(r["host_q"], qa, qb), pieces(krng, ka, kb)
                multi_piece += len(qp) * len(kp) > 1
                tiles = {}
                for (la, lb, oq) in qp:
                    for (ma, mb, ok) in kp:
                        for key, v in tile_areas(la + oq, lb + oq, ma + ok, mb + ok, lo + ok - oq, hi + ok - oq).items():
                            tiles[key] = tiles.get(key, 0) + v
                for key, v in tiles.items():
                    cov.setdefault(key, []).append((g, stage, v))
                # comps: diagonal causal tiles 1x1, the rest grouped by equal column sets (Magi slice order kept)
                byrow = {}
                for (qt, kt) in sorted(tiles):
                    byrow.setdefault(qt, []).append(kt)
                colset = {}
                for qt, cols in byrow.items():
                    if qt in cols and m[qt, qt] == 2:
                        comps[g].append(dict(kind="comp", rows=[qt], cols=[qt], start=float(stage), fused=False))
                    rest = tuple(c for c in cols if not (c == qt and m[qt, qt] == 2))
                    if rest:
                        colset.setdefault(rest, []).append(qt)
                for cs, qs in colset.items():
                    comps[g].append(dict(kind="comp", rows=qs, cols=list(cs), start=float(stage), fused=len(qs) > 1 or len(cs) > 1))
    bad = [(k, v) for k, v in cov.items() if len(v) != 1 or v[0][2] != full[k]]
    assert area_mismatch == 0, f"{area_mismatch} Magi slices: band area != AttnSlice.area"
    assert not bad, f"{len(bad)} tiles not covered exactly once / fully, e.g. {bad[:3]}"
    live = set(zip(*np.nonzero(m > 0)))
    assert set(cov) == {(int(a), int(b)) for a, b in live}, "coverage != live tiles"
    for (qt, kt), v in cov.items():
        g, stage, _ = v[0]
        assert g == owner[qt] and (stage == 0) == (owner[kt] == g), ("stage / owner mismatch", qt, kt)
    gk = build_gk_two_stage(owner, m, comps)
    rkv = validate(owner, m, gk)
    st = rank_stats(d)
    assert (rkv == st["remote_kv"]).all() and (st["gc_out"] == st["remote_kv"]).all(), \
        ("remote K / V tokens differ from Magi's comm meta", rkv.tolist(), st["remote_kv"].tolist(), st["gc_out"].tolist())
    return owner, gk, dict(multi_piece_slices=int(multi_piece), tiles_checked=len(cov))


def convert():
    for inp in inputs():
        z = np.load(inp["mask"])
        m = z["mask"].astype(np.int8)
        d = json.load(open(f"{SM.ablation_dir('magi', 'raw')}/{inp['name']}.json"))
        owner, gk, extra = convert_pad(d, m)
        st = rank_stats(d)
        info = dict(T_magi=d["T"], auto_chunk=d["auto_chunk"], chunk=d["chunk"], magi_pad=d["magi_pad"], overlap_degree=d["ranks"][0]["overlap_degree"],
                    n_slices=d["n_slices"], magi_area=summ(st["area"]), magi_remote_kv_tokens=summ(st["remote_kv"]),
                    magi_remote_kv_inter_node_tokens=summ(st["remote_kv_inter"]), tile_load=summ(tile_load(owner, m)),
                    cpu_dispatch_s=max(r["t_dispatch_s"] for r in d["ranks"]), cpu_solve_s=max(r["t_solve_s"] for r in d["ranks"]), **extra)
        plan = dict(family=inp["name"], strategy="magi_pad128", cp=CP, nodes=CP // GPN, gpus_per_node=GPN, n_blocks=m.shape[0], block_tokens=TILE, mask=inp["mask"],
                    heads=list(inp["heads"]), exact=True, owner=owner.tolist(), schedule_table=np.where(m > 0, owner[:, None], -1).tolist(), gpu_kernels=gk, magi=info)
        json.dump(plan, open(f"{SM.ablation_dir('magi')}/{inp['name']}_magi.json", "w"))
        print(f"[conv] {inp['name']}: chunk {d['chunk']} pad {d['magi_pad']}, area max/mean {info['magi_area']['max_over_mean']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--convert_only", action="store_true", help="re-run the conversion from the dumps, without Magi")
    a = ap.parse_args()
    if not a.convert_only:
        assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only: run with CUDA_VISIBLE_DEVICES="
        import torch.multiprocessing as mp
        from dex.common.env import cfg
        mp.spawn(solve_worker, args=(list(inputs()), int(cfg("DEX_MASTER_PORT")) + 77), nprocs=CP, join=True)
    convert()


if __name__ == "__main__":
    main()
