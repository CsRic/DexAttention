"""W4 operator test (MiniCPM4.1-8B InfLLM-v2, 32 query / 2 KV heads, BF16), one system per launch, every case given in it.

Per case (sequence : layer): rank 0 loads the layer's attention weights, the recorded attention input and the recorded
selection and sends every rank its zigzag chunks; Q / K / V are projected once, outside every timed window. Then three
parts, each `--warmup` + `--iters` calls, one call = barrier -> slowest rank done, median of the timed calls:
  exec     the selection known: each system's whole remaining forward work (Dex: planner input all-gather, joint
           planner + home gate, broadcast, needs / unit exchange, tables, persistent forward + merge)
  bwd      the backward given O / LSE of an untimed official forward and a seeded dO (its plan inside)
  online   the forward including stage 1 and the selection (compressed keys exchanged; All-gather: on the gathered keys;
           Ring: the compressed keys circulate)
Every call is run twice: once with phase marks (per-rank `phases`; `plan` = everything before the kernel) and once unmarked
(`e2e`, the reported latency). plan_span_ms = median over timed calls of the slowest rank's plan phase (the online-cost
span); Dex also records the plan stages (`plan_ms`, 3 synchronised separate runs) and the home-gate result.
--check (launch 1, layers 0 / 16): exec and bwd compared with All-gather; with --fp64 also the FP64 sampled-row check of O,
dQ, dK, dV; online compared with All-gather, with --online-ref also with the single-GPU official forward on rank 0.
Systems: dex, dex_no_assigner, ring, allgather, sparse_allgather, decomposed (Dex's plan on a decomposed executor).
Output: one json line per case appended to --out (finished cases are skipped on a rerun).
usage: tools/launch.sh experiments/attention_layer/w4_op.py --system dex --cases <seq:layer,...> --out w4/dex/rep1.jsonl"""
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
from workloads.w4_infllm import check as CK  # noqa: E402
from workloads.w4_infllm.common import (GH, Timer, compressed_keys_cp, filter_causal, from_rows, owned, pad_concat,  # noqa: E402
                                        select_chunk, to_rows, zigzag_chunks)
from workloads.w4_infllm.minicpm_model import (LONG_FACTOR_128K, Cfg, Layer, Store, rope_cos_sin, select_topk,  # noqa: E402
                                               sparse_attention)

DT = torch.bfloat16            # the official InfLLM-v2 build is BF16 only
SYSTEMS = ("dex", "dex_no_assigner", "ring", "allgather", "sparse_allgather", "decomposed")
DEX = ("dex", "dex_no_assigner")
PLAN_RUNS = 3
NEEDS_INFERENCE = ("ring", "allgather", "sparse_allgather", "decomposed")   # official autograd entry points: no graph


def _nanmax(xs):
    return max((float("inf") if x != x else x) for x in xs)            # (NaN = failure; plain max() drops it)


def bcast_tensors(d, src=0):
    rank = dist.get_rank()
    meta = [[(k, tuple(v.shape), str(v.dtype)) for k, v in d.items()]] if rank == src else [None]
    dist.broadcast_object_list(meta, src)
    dev = torch.device("cuda", torch.cuda.current_device())
    out = {}
    for k, shp, dt in meta[0]:
        t = d[k].to(dev).contiguous() if rank == src else torch.empty(shp, dtype=getattr(torch, dt[6:]), device=dev)
        dist.broadcast(t, src)
        out[k] = t
    return out


def timed(fn, a, events, P):
    """warm-ups + timed calls -> runs [{e2e seconds, phases per rank}] (the first a.warmup are the warm-ups)"""
    res = []
    for _ in range(a.warmup + a.iters):
        tm = Timer(True, events=events)
        fn(tm)
        torch.cuda.synchronize(); dist.barrier()
        t_ = time.perf_counter()
        fn(Timer(False))
        torch.cuda.synchronize(); dist.barrier()
        e2e = time.perf_counter() - t_
        allt = [None] * P
        dist.all_gather_object(allt, tm.t)
        res.append(dict(e2e=e2e, phases=allt))
    return res


def summary(runs, a):
    t = runs[a.warmup:]
    return dict(median_ms=float(np.median([r["e2e"] for r in t])) * 1e3,
                plan_span_ms=float(np.median([max(p.get("plan", 0.0) for p in r["phases"]) for r in t])) * 1e3, runs=runs)


def plan_stages(layer, fn):
    """Dex's plan stages: 3 synchronised separate runs (aligned ranks: the stage times exclude arrival skew)"""
    pl = layer.pl
    pl.TIMING = True; tp = {}
    for _ in range(PLAN_RUNS):
        torch.cuda.synchronize(); dist.barrier()
        pl.tplan.clear(); fn(Timer(False))
        for k_, v_ in pl.tplan.items():
            tp.setdefault(k_, []).append(v_)
    pl.TIMING = False
    return {k_: round(float(np.median(v_)), 3) for k_, v_ in tp.items()}


class Case:
    """one (sequence, layer): the attention input, projections, selection and layout of this rank"""

    def __init__(self, a, cfg_, dev, seq, lay):
        self.seq, self.lay, self.dev = seq, lay, dev
        self.rank, self.P = dist.get_rank(), dist.get_world_size()
        rank, P = self.rank, self.P
        if rank == 0:
            store = Store(a.model, dev)
            p = f"model.layers.{lay}."
            W = store.load([k for k in store.map if k.startswith(p + "self_attn.")])
            W = {k[len(p):]: v.to(DT) for k, v in W.items()}
            x = torch.from_numpy(np.load(os.path.join(a.masks, seq, f"L{lay:02d}_xattn.npy"))).view(torch.bfloat16)
            St = torch.tensor([x.size(0)], device=dev)
        else:
            W, x, St = {}, None, torch.zeros(1, dtype=torch.long, device=dev)
        dist.broadcast(St, 0)
        S = self.S = int(St)
        W = bcast_tensors(W)
        self.layer = Layer(cfg_, lay, None, W=W)
        self.c = cfg_
        self.scale = cfg_.head_dim ** -0.5
        self.x = x
        self.cos, self.sin = rope_cos_sin(cfg_, S, dev, LONG_FACTOR_128K if S > cfg_.orig_max else None)
        self.chunks = chunks = zigzag_chunks(S, P)
        self.my = my = owned(rank, P)
        self.Ls = [chunks[ci][1] - chunks[ci][0] for ci in my]
        xs = {}
        for r in range(P):
            for ci in owned(r, P):
                c0, c1 = chunks[ci]
                if rank == 0:
                    t = x[c0:c1].to(dev)
                    if r == 0:
                        xs[ci] = t
                    else:
                        dist.send(t.contiguous(), r)
                elif rank == r:
                    t = torch.empty(c1 - c0, cfg_.hidden, dtype=DT, device=dev)
                    dist.recv(t, 0)
                    xs[ci] = t
        dist.barrier()
        self.sel = {}                                                  # recorded selection [2, L, 96] per own chunk
        tk = np.load(os.path.join(a.masks, seq, f"L{lay:02d}_topk.npy"), mmap_mode="r") if rank == 0 else None
        for r in range(P):
            for ci in owned(r, P):
                c0, c1 = chunks[ci]
                if rank == 0:
                    t = torch.from_numpy(np.ascontiguousarray(tk[:, c0:c1, :]).astype(np.int32)).to(dev)
                    if r == 0:
                        self.sel[ci] = t
                    else:
                        dist.send(t, r)
                elif rank == r:
                    t = torch.empty(2, c1 - c0, 96, dtype=torch.int32, device=dev); dist.recv(t, 0); self.sel[ci] = t
        dist.barrier()
        self.loc = {ci: self.layer.qkv(xs[ci], self.cos[chunks[ci][0]:chunks[ci][1]], self.sin[chunks[ci][0]:chunks[ci][1]]) for ci in my}
        self.pos = torch.cat([torch.arange(*chunks[ci], device=dev) for ci in my]).to(torch.int32)
        self.nh = cfg_.n_heads

    def inputs(self):
        """(q rows, k, v) of the own chunks concatenated; input assembly stays inside the timed window"""
        my, loc = self.my, self.loc
        return (torch.cat([to_rows(loc[ci][0]) for ci in my]).contiguous(), torch.cat([loc[ci][1] for ci in my]).contiguous(),
                torch.cat([loc[ci][2] for ci in my]).contiguous())

    def outs(self, o):
        out, r = {}, 0
        for ci, L in zip(self.my, self.Ls):
            out[ci] = from_rows(o[r:r + L * GH], self.nh).reshape(L, -1)
            r += L * GH
        return out


def make_exec(system, cs):
    """-> (fn(timer) -> {own chunk: O [L, 32 * 128]}, the executor object). Executors are built here, untimed."""
    from baselines.allgather.infllm import AllgatherInfllm, SparseAllgatherInfllm
    if system in DEX:
        from dex.attn.infllm import DexInfllmLayer
        ly = DexInfllmLayer(system, "fwd", cs.chunks, cs.my, cs.S, cs.P, cs.rank, cs.dev)

        def fn(tm):
            tm.start()
            sel_f = filter_causal(torch.cat([cs.sel[ci] for ci in cs.my], 1).to(torch.int32), cs.pos)
            o = ly.fwd(*cs.inputs(), sel_f, cs.scale, timer=tm)
            tm.mark("attn", barrier=False)
            tm.mark("wait")
            return cs.outs(o)
        return fn, ly
    if system == "allgather":
        ag = AllgatherInfllm(cs.chunks, cs.my, cs.S, cs.P, cs.c)
        return lambda tm: ag.fwd(cs.loc, cs.sel, tm), ag
    if system == "ring":
        from baselines.ring.infllm import RingInfllm
        rg = RingInfllm(cs.chunks, cs.my, cs.S, cs.P, cs.rank, cs.scale)
        return lambda tm: rg.fwd(cs.loc, cs.sel, tm, cs.nh), rg
    if system == "decomposed":
        from baselines.decomposed.w4 import DecomposedInfllm as Ex
    else:
        Ex = SparseAllgatherInfllm
    pp = Ex(cs.chunks, cs.my, cs.S, cs.P, cs.rank, cs.dev)

    def fn(tm):
        tm.start()
        sel_f = filter_causal(torch.cat([cs.sel[ci] for ci in cs.my], 1).to(torch.int32), cs.pos)
        o, _ = pp.fwd(*cs.inputs(), sel_f)
        tm.mark("attn", barrier=False)
        tm.mark("wait")
        return cs.outs(o)
    return fn, pp


def make_online(system, cs):
    """-> (fn(timer) -> ({own chunk: O}, {own chunk: selection}), Dex's layer object or None): the forward including
    stage 1 and the selection"""
    from baselines.allgather.infllm import AllgatherInfllm, SparseAllgatherInfllm
    if system == "allgather":
        ag = AllgatherInfllm(cs.chunks, cs.my, cs.S, cs.P, cs.c)
        return (lambda tm: ag.fwd_online(lambda: cs.loc, tm)), None
    if system == "ring":
        from baselines.ring.infllm import RingInfllm
        rg = RingInfllm(cs.chunks, cs.my, cs.S, cs.P, cs.rank, cs.scale)
        return (lambda tm: rg.fwd_online(lambda: cs.loc, tm, cs.nh)), None
    ly = None
    if system in DEX:
        from dex.attn.infllm import DexInfllmLayer
        ly = DexInfllmLayer(system, "fwd", cs.chunks, cs.my, cs.S, cs.P, cs.rank, cs.dev)
        run = lambda sel_f, tm: ly.fwd(*cs.inputs(), sel_f, cs.scale, timer=tm)                    # noqa: E731
    else:
        if system == "decomposed":
            from baselines.decomposed.w4 import DecomposedInfllm as Ex
        else:
            Ex = SparseAllgatherInfllm
        pp = Ex(cs.chunks, cs.my, cs.S, cs.P, cs.rank, cs.dev)
        run = lambda sel_f, tm: pp.fwd(*cs.inputs(), sel_f)[0]                                     # noqa: E731

    def fn(tm):
        tm.start()
        tm.mark("local_proj", barrier=False)
        kc, kc2 = compressed_keys_cp({ci: cs.loc[ci][1] for ci in cs.my}, cs.chunks, cs.my, cs.S)
        tm.mark("compress", barrier=False)
        sel = {ci: select_chunk(cs.loc[ci][0], kc, kc2, cs.chunks[ci][0]) for ci in cs.my}
        del kc, kc2
        tm.mark("select", barrier=False)
        sel_f = filter_causal(pad_concat(sel, cs.my).to(torch.int32), cs.pos)
        o = run(sel_f, tm)
        tm.mark("attn", barrier=False)
        tm.mark("wait")
        return cs.outs(o), sel
    return fn, ly


def infer(fn, system):
    """the official entry points run under inference mode on the baselines' forward paths"""
    if system not in NEEDS_INFERENCE:
        return fn

    def wrapped(tm):
        with torch.inference_mode():
            return fn(tm)
    return wrapped


def part_exec(a, cs, system):
    from baselines.allgather.infllm import AllgatherInfllm
    fn, ex = make_exec(system, cs)
    fn = infer(fn, system)
    torch.cuda.synchronize(); dist.barrier()
    check = None
    if a.check:
        outs = fn(Timer(False))
        with torch.inference_mode():
            ref = AllgatherInfllm(cs.chunks, cs.my, cs.S, cs.P, cs.c).fwd(cs.loc, cs.sel, Timer(False))
        rd = torch.tensor([_nanmax(((outs[ci].float() - ref[ci].float()).norm() / ref[ci].float().norm()).item() for ci in cs.my)], device=cs.dev)
        dist.all_reduce(rd, op=dist.ReduceOp.MAX)
        check = dict(max_rel=float(rd.item()))
        if a.fp64:
            check["fp64"] = CK.fwd_check(cs.loc, outs, cs.sel, cs.chunks, cs.my, cs.S, cs.scale)
    res = timed(fn, a, True, cs.P)
    info = {}
    if system in DEX:
        info = ex.fwd_stats()
        info = dict(info, bytes_in_MB=info.pop("bytes_in") / 2 ** 20)
    elif system in ("sparse_allgather", "decomposed"):
        info = dict(bytes_in_MB=ex.last["bytes_in"] / 2 ** 20)
    if system == "dex":
        info["plan_ms"] = plan_stages(ex, fn)
        info["heap_used_GiB"] = round(ex.heap_used() / 2 ** 30, 3)
        info["gate"] = ex.pl.gate_info()
    allinfo = [None] * cs.P
    dist.all_gather_object(allinfo, info)
    return dict(summary(res, a), check=check, info=allinfo)


def part_bwd(a, cs, system):
    with torch.inference_mode(False), torch.enable_grad():
        return _part_bwd(a, cs, system)


def _part_bwd(a, cs, system):
    """O and LSE from an untimed official forward over the gathered K / V (identical for every system); dO seeded"""
    from infllm_v2 import infllmv2_attn_varlen_func

    from baselines.allgather.infllm import AllgatherInfllm, gather_kv
    dev, c, my, chunks = cs.dev, cs.c, cs.my, cs.chunks
    q_rows, k_own, v_own = (t.clone() for t in cs.inputs())
    sel_f = filter_causal(torch.cat([cs.sel[ci].clone() for ci in my], 1).to(torch.int32).contiguous(), cs.pos)
    g = torch.Generator(device=dev); g.manual_seed(1234 + cs.lay)
    do_rows = (torch.randn(q_rows.shape, device=dev, dtype=torch.float32, generator=g) * 0.1).to(DT)
    K, V, _ = gather_kv(k_own, v_own, chunks, my, cs.S, cs.P, c)
    o_rows, lse, t0 = [], [], 0
    for ci, L in zip(my, cs.Ls):
        c1 = chunks[ci][1]
        cu_q = torch.tensor([0, L * GH], dtype=torch.int32, device=dev); cu_k = torch.tensor([0, c1], dtype=torch.int32, device=dev)
        o, l_, _ = infllmv2_attn_varlen_func(q_rows[t0 * GH:(t0 + L) * GH], K[:c1], V[:c1], cu_q, cu_k, L * GH, c1,
                                             causal=True, topk_idx=sel_f[:, t0:t0 + L].contiguous(), return_attn_probs=True)
        o_rows.append(o); lse.append(l_); t0 += L
    o_rows = torch.cat(o_rows).contiguous(); lse = torch.cat(lse, 1).contiguous()
    del K, V
    ag = AllgatherInfllm(chunks, my, cs.S, cs.P, c)
    ly = None
    if system in DEX:
        from dex.attn.infllm import DexInfllmLayer
        ly = DexInfllmLayer(system, "bwd", chunks, my, cs.S, cs.P, cs.rank, dev)

        def fn(tm):
            tm.start()
            dq, dk, dv = ly.bwd(q_rows, o_rows, do_rows, lse, k_own, v_own, sel_f, cs.scale, timer=tm)
            tm.mark("attn", barrier=False)
            tm.mark("wait")
            return dq, dk, dv
    elif system == "ring":
        from baselines.ring.infllm import RingInfllm
        rg = RingInfllm(chunks, my, cs.S, cs.P, cs.rank, cs.scale)
        fn = lambda tm: rg.bwd(q_rows, o_rows, do_rows, lse, k_own, v_own, sel_f, tm)          # noqa: E731
    elif system == "allgather":
        fn = lambda tm: ag.bwd(q_rows, o_rows, do_rows, lse, k_own, v_own, sel_f, cs.scale, tm)   # noqa: E731
    else:
        if system == "decomposed":
            from baselines.decomposed.w4 import DecomposedInfllm as Ex
        else:
            from baselines.allgather.infllm import SparseAllgatherInfllm as Ex
        pp = Ex(chunks, my, cs.S, cs.P, cs.rank, dev)

        def fn(tm):
            tm.start()
            dq, dk, dv = pp.bwd(q_rows, o_rows, do_rows, lse, k_own, v_own, sel_f)
            tm.mark("attn", barrier=False)
            tm.mark("wait")
            return dq, dk, dv
    check = None
    if a.check:
        outs = fn(Timer(False))
        ref = ag.bwd(q_rows, o_rows, do_rows, lse, k_own, v_own, sel_f, cs.scale, Timer(False), recompute=True)
        rel = lambda x, y: ((x.float() - y.float()).norm() / y.float().norm().clamp(min=1e-20)).item()   # noqa: E731
        e = torch.tensor([rel(outs[i], ref[i]) for i in range(3)], device=dev)
        dist.all_reduce(e, op=dist.ReduceOp.MAX)
        check = dict(dq=e[0].item(), dk=e[1].item(), dv=e[2].item())
        if a.fp64:
            n = q_rows.shape[0] // GH; nh = q_rows.shape[1] * GH
            lse_tok = lse.reshape(lse.shape[0], n, GH).permute(1, 0, 2).reshape(n, nh)
            check["fp64"] = CK.bwd_check(from_rows(q_rows, nh), k_own, v_own, from_rows(do_rows, nh), from_rows(o_rows, nh), lse_tok, sel_f,
                                         cs.pos, cs.S, cs.scale, from_rows(outs[0], nh), outs[1], outs[2])
    fn(Timer(False))                                                   # warm
    res = timed(fn, a, system != "allgather", cs.P)
    info = {}
    if system == "dex":
        info = dict(plan_ms=plan_stages(ly, fn), stats=ly.bwd_stats(), heap_used_GiB=round(ly.heap_used() / 2 ** 30, 3), gate=ly.pl.gate_info())
    allinfo = [None] * cs.P
    dist.all_gather_object(allinfo, info)
    return dict(summary(res, a), check=check, info=allinfo)


def part_online(a, cs, system):
    from baselines.allgather.infllm import AllgatherInfllm
    fn, ly = make_online(system, cs)
    fn = infer(fn, system)
    torch.cuda.synchronize(); dist.barrier()
    check = None
    if a.check:
        outs, sel = fn(Timer(False))
        check = {}
        if system != "allgather":
            with torch.inference_mode():
                b_outs, b_sel = AllgatherInfllm(cs.chunks, cs.my, cs.S, cs.P, cs.c).fwd_online(lambda: cs.loc, Timer(False))
            nes = torch.tensor([sum(int(not torch.equal(sel[ci], b_sel[ci])) for ci in cs.my)], device=cs.dev)
            dist.all_reduce(nes)
            rd = torch.tensor([_nanmax(((outs[ci].float() - b_outs[ci].float()).norm() / b_outs[ci].float().norm()).item() for ci in cs.my)], device=cs.dev)
            dist.all_reduce(rd, op=dist.ReduceOp.MAX)
            check.update(max_rel=float(rd.item()), selections_differ=int(nes.item()))
            del b_outs, b_sel
        if a.online_ref:                                               # the single-GPU official forward on rank 0
            got = {}
            for r in range(cs.P):
                for ci in owned(r, cs.P):
                    c0, c1 = cs.chunks[ci]
                    if r == cs.rank and cs.rank != 0:
                        dist.send(outs[ci].contiguous(), 0)
                    elif r == cs.rank:
                        got[ci] = outs[ci]
                    elif cs.rank == 0:
                        t = torch.empty(c1 - c0, cs.c.hidden, dtype=DT, device=cs.dev)
                        dist.recv(t, r)
                        got[ci] = t
            if cs.rank == 0:
                wo = cs.layer.W["self_attn.o_proj.weight"]
                cp = torch.nn.functional.linear(torch.cat([got[ci] for ci in range(2 * cs.P)], 0), wo).float()
                torch.cuda.empty_cache()
                q, k, v = cs.layer.qkv(cs.x.to(cs.dev), cs.cos, cs.sin)
                o = sparse_attention(q, k, v, select_topk(q, k))
                ref = torch.nn.functional.linear(o.reshape(cs.S, -1), wo).float()
                check["rel_1gpu"] = ((cp - ref).norm() / ref.norm()).item()
                del cp, ref, q, k, v, o
            dist.barrier()
    res = timed(fn, a, system != "allgather", cs.P)
    xtra = dict(gate=ly.pl.gate_info(), stats=ly.fwd_stats()["stats"]) if system == "dex" else {}     # (rank 0's view)
    return dict(summary(res, a), check=check, **xtra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=SYSTEMS)
    ap.add_argument("--cases", required=True, help="comma list of seq:layer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None, help="default $DEX_MODEL_ROOT/MiniCPM4.1-8B")
    ap.add_argument("--masks", default=None, help="sample_masks.py output (default $DEX_DATA_ROOT/w4_infllm/masks)")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--check", action="store_true", help="compare with All-gather")
    ap.add_argument("--fp64", action="store_true", help="with --check: FP64 sampled-row check")
    ap.add_argument("--online-ref", action="store_true", help="with --check: single-GPU official forward on rank 0")
    a = ap.parse_args()
    a.model = a.model or os.path.join(cfg("DEX_MODEL_ROOT"), "MiniCPM4.1-8B")
    a.masks = a.masks or os.path.join(cfg("DEX_DATA_ROOT"), "w4_infllm", "masks")
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    rank, P = dist.get_rank(), dist.get_world_size()
    if a.system in DEX:
        os.environ.setdefault("NVSHMEM_SYMMETRIC_SIZE", str(2 << 30))
        from dex.attn.infllm import ext
        ext().init_nvshmem(dist.group.WORLD)
    dev = torch.device("cuda", torch.cuda.current_device())
    torch.set_grad_enabled(False)
    cfg_ = Cfg(a.model)
    done = set()
    if rank == 0 and os.path.exists(a.out):
        done = {json.loads(line)["name"] for line in open(a.out)}
    box = [done]
    dist.broadcast_object_list(box, 0)
    done = box[0]
    for c in a.cases.split(","):
        seq, lay = c.split(":")
        lay = int(lay)
        name = f"{seq}:{lay}"
        if name in done:
            continue
        cs = Case(a, cfg_, dev, seq, lay)
        r_exec = part_exec(a, cs, a.system)
        r_bwd = part_bwd(a, cs, a.system)
        torch.cuda.empty_cache()
        r_online = part_online(a, cs, a.system)
        if rank == 0:
            print(f"{name} S={cs.S} {a.system} exec {r_exec['median_ms']:.2f} bwd {r_bwd['median_ms']:.2f} "
                  f"online {r_online['median_ms']:.2f} ms", flush=True)
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            with open(a.out, "a") as f:
                f.write(json.dumps(dict(name=name, seq=seq, layer=lay, S=cs.S, P=P, system=a.system, warmup=a.warmup, iters=a.iters,
                                        chunks=cs.chunks, exec=r_exec, bwd=r_bwd, online=r_online)) + "\n")
        del cs
        torch.cuda.empty_cache()
        dist.barrier()
    dist.barrier()
    dist.destroy_process_group()
    if a.system in DEX:                                              # no static teardown behind the NVSHMEM runtime
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
