"""W6 end-to-end prefill: Llama-3.1-8B-Instruct + FlexPrefill (gamma 0.95, tau 0.1, min_budget 1024, block 128) with context
parallelism over all GPUs (zigzag), one attention system per launch, every input in it.

Per input: one forward of the whole model (32 layers) over the padded prompt; each rank processes its own tokens. Per layer,
all timed: rest (input RMSNorm, q / k / v, RoPE, padded tokens' q / k / v = 0) -> selection (workloads/w6_flexprefill/selection.py:
distributed FlexPrefill indexing, no full K on any GPU) -> attention (the system's whole per-call work in FP16) -> rest (o_proj,
MLP); then the final RMSNorm and the last real token's logits. Weights: bf16, replicated.
Calls per input: one untimed correctness pass (logits + sampled final hidden states saved; per-layer selected-tile counts and
mask hashes; layers 0 / 8 / 16 / 24 masks vs the dumps; logits vs the official reference if present), --warmup untimed prefills,
--iters timed prefills (barrier -> slowest rank); prefill_ms = their geometric mean. Breakdown: CUDA events per rank on the
compute stream (selection / attention / rest) of the median timed call. Memory: PyTorch peak + memory outside PyTorch, counting
only the symmetric memory Dex's tables use.
Systems: dex, dex_no_assigner, ring, allgather, sparse_allgather.
Output: one json line per input appended to --out; correctness files under --save_dir (finished inputs are skipped on a rerun).
usage: tools/launch.sh experiments/end2end/llama_prefill.py --system dex --out llama/dex/rep1.jsonl --save_dir llama/dex/rep1 [--save_hidden]
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from dex.common.env import cfg  # noqa: E402
from workloads.w6_flexprefill import selection as FD  # noqa: E402
from workloads.w6_flexprefill.common import cuda_ext, mask_global  # noqa: E402
from dex.common.tile_layout import TileLayout  # noqa: E402
from workloads.w6_flexprefill.model import HID, HQ, NLAYER, VOCAB, Llama  # noqa: E402

SYSTEMS = ("dex", "dex_no_assigner", "ring", "allgather", "sparse_allgather")
SPAD = {"32k": 32768, "64k": 65536, "128k": 131072}
INPUTS9 = os.path.join(ROOT, "workloads", "w6_flexprefill", "inputs9.txt")
DUMP_LAYERS = (0, 8, 16, 24)


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
    from baselines.allgather.flexprefill import SparseAllGatherFlex

    return SparseAllGatherFlex(lay)


class Ctx:
    """per-input context: padding, the last real token, the sampled tokens and the selection"""

    def __init__(self, lay, N):
        self.lay, self.N = lay, N
        P = lay.P
        pad = (lay.tok >= N).nonzero().flatten()
        self.padidx = pad if pad.numel() else None
        li = (lay.tok == N - 1).nonzero().flatten()
        self.last_loc = int(li[0]) if li.numel() else None
        self.last_rank = int(
            np.argmax(
                [
                    ((np.r_[np.arange(*lay.chunks[r]), np.arange(*lay.chunks[2 * P - 1 - r])]) == N - 1).any()
                    for r in range(P)
                ]
            )
        )
        self.smp = (((lay.tok % 64 == 0) & (lay.tok < N)) | (lay.tok == N - 1)).nonzero().flatten()
        self.sel = FD.FlexDist(lay, N)
        self.rx, self.nqa = [], []


def prefill(M, so, ctx, ids_loc, ev=None, hook=None):
    lay = ctx.lay
    if ev is not None:
        ev[0].record()
    h = F.embedding(ids_loc, M.w["model.embed_tokens.weight"])
    cos, sin = M.rope(lay.tok)
    ctx.rx.clear()
    ctx.nqa.clear()
    for li in range(NLAYER):
        q, k, v = M.qkv(li, h, cos, sin, ctx.padidx)
        if ev is not None:
            ev[1 + 3 * li].record()
        mrows = FD.drive_nccl(ctx.sel.run(q, k))
        ctx.rx.append(sum(ctx.sel.rx.values()))
        ctx.nqa.append(ctx.sel.n_qa)
        if ev is not None:
            ev[2 + 3 * li].record()
        o16, _ = so.fwd(q.half(), k.half(), v.half(), mrows)
        o = o16.to(torch.bfloat16).reshape(lay.L, HID)
        if ev is not None:
            ev[3 + 3 * li].record()
        if hook is not None:
            hook(li, mrows)
        del q, k, v, o16
        h = M.post(li, h, o)
    hn = M.rms(h, M.w["model.norm.weight"])
    logits = F.linear(hn[ctx.last_loc], M.w["lm_head.weight"]).float() if ctx.last_loc is not None else None
    if ev is not None:
        ev[1 + 3 * NLAYER].record()
    return hn, logits


def breakdown(ev):
    tot = ev[0].elapsed_time(ev[-1])
    sel = sum(ev[1 + 3 * li].elapsed_time(ev[2 + 3 * li]) for li in range(NLAYER))
    att = sum(ev[2 + 3 * li].elapsed_time(ev[3 + 3 * li]) for li in range(NLAYER))
    return [tot, sel, att, tot - sel - att]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=SYSTEMS)
    ap.add_argument("--inputs", default=INPUTS9)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save_dir", required=True, help="logits / hidden samples of the correctness pass (rank 0)")
    ap.add_argument("--warmup", type=int, default=2, help="untimed prefills after the correctness pass")
    ap.add_argument("--iters", type=int, default=4, help="timed prefills")
    ap.add_argument("--save_hidden", action="store_true")
    a = ap.parse_args()
    dist.init_process_group("nccl")
    rank, P = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count())))
    torch.cuda.set_device(dev)
    root = os.path.join(cfg("DEX_DATA_ROOT"), "w6_flexprefill")
    torch.set_grad_enabled(False)
    t_load = time.time()
    M = Llama(dev)
    torch.cuda.synchronize()
    dist.barrier()
    t_load = time.time() - t_load
    dex_sys = a.system in ("dex", "dex_no_assigner")
    gated = a.system == "dex"
    if dex_sys:
        cuda_ext().init_nvshmem(dist.group.WORLD)
        from dex.attn.flexprefill import GATE_EPS
    inputs = open(a.inputs).read().split()
    done = {r["input"] for r in map(json.loads, open(a.out))} if os.path.exists(a.out) else set()
    so = None
    if rank == 0:
        os.makedirs(a.save_dir, exist_ok=True)
    for inp in inputs:
        if inp in done:
            continue
        t_case = time.time()
        torch.cuda.reset_peak_memory_stats()
        S_pad = SPAD[inp.split("_")[-1]]
        lay = TileLayout(S_pad, P, rank, dev)
        if rank == 0:
            ids = np.load(f"{root}/inputs/{inp}.npy").astype(np.int64)
            Nt = torch.tensor([len(ids)], device=dev)
        else:
            Nt = torch.zeros(1, dtype=torch.int64, device=dev)
        dist.broadcast(Nt, 0)
        N = int(Nt)
        idt = torch.zeros(S_pad, dtype=torch.int64, device=dev)
        if rank == 0:
            idt[:N] = torch.from_numpy(ids).to(dev)
        dist.broadcast(idt, 0)
        ids_loc = idt[lay.tok].contiguous()
        lay.s_real = N
        ctx = Ctx(lay, N)
        if so is None:
            so = make_system(a.system, lay)
        elif hasattr(so, "retarget"):
            so.retarget(lay)
        else:
            so.lay = lay
        rec = dict(
            input=inp,
            system=a.system,
            N=N,
            S_pad=S_pad,
            load_s=round(t_load, 1),
            nvshmem_symmetric=os.environ.get("NVSHMEM_SYMMETRIC_SIZE"),
        )
        # ---- correctness pass (untimed; also the first warm-up: compiles / captures)
        cnt = torch.zeros(NLAYER, dtype=torch.int64, device=dev)
        hashes, vs_dump = [], {}

        def hook(li, mrows):
            cnt[li] = (mrows != 0).sum()
            hashes.append(hashlib.md5(mrows.cpu().numpy().tobytes()).digest()[:8])
            if li in DUMP_LAYERS:
                mh = mask_global(mrows, lay)
                f = f"{root}/qkv/{inp}/L{li:02d}_mask_g095.npy"
                if rank == 0 and os.path.exists(f):
                    mk = np.load(f)
                    nb = mk.shape[1]
                    m = np.unpackbits(mk, axis=-1, count=nb).astype(bool)
                    n = lay.n
                    Mr = np.zeros((HQ, n, n), np.int8)
                    Mr[:, :nb, :nb] = m
                    Mr[:, np.arange(n), np.arange(n)] = 2
                    vs_dump[li] = dict(diff=int((mh.cpu().numpy() != Mr).sum()), sel=int((Mr != 0).sum()))
                del mh

        if gated:
            so.gate_log.clear()
        tc = time.time()
        hn, logits = prefill(M, so, ctx, ids_loc, hook=hook)
        torch.cuda.synchronize()
        rec["first_call_s"] = round(time.time() - tc, 2)
        rx = torch.tensor(ctx.rx, dtype=torch.float64, device=dev)
        arx = torch.empty(P, NLAYER, dtype=torch.float64, device=dev)
        dist.all_gather_into_tensor(arx, rx)
        rec["sel_rx_MiB_per_layer_max_rank"] = round(float(arx.max(0).values.mean()) / 2**20, 3)
        rec["sel_n_query_aware"] = list(ctx.nqa)
        dist.all_reduce(cnt)
        hv = torch.tensor(np.frombuffer(b"".join(hashes), dtype=np.int64).copy(), device=dev)
        ah = torch.empty(P, NLAYER, dtype=torch.int64, device=dev)
        dist.all_gather_into_tensor(ah, hv)
        lg = logits if logits is not None else torch.empty(VOCAB, dtype=torch.float32, device=dev)
        dist.broadcast(lg, ctx.last_rank)
        ns = len(ctx.smp)
        mx = torch.tensor([ns], device=dev)
        dist.all_reduce(mx, op=dist.ReduceOp.MAX)
        mx = int(mx)
        hs = torch.zeros(mx, HID, dtype=torch.bfloat16, device=dev)
        hs[:ns] = hn[ctx.smp]
        ts = torch.full((mx,), -1, dtype=torch.int64, device=dev)
        ts[:ns] = lay.tok[ctx.smp]
        ahs = torch.empty(P, mx, HID, dtype=torch.bfloat16, device=dev)
        dist.all_gather_into_tensor(ahs, hs)
        ats = torch.empty(P, mx, dtype=torch.int64, device=dev)
        dist.all_gather_into_tensor(ats, ts)
        if rank == 0:
            tsn = ats.reshape(-1).cpu().numpy()
            ok = tsn >= 0
            o_ = np.argsort(tsn[ok])
            hid = ahs.reshape(-1, HID)[torch.as_tensor(np.nonzero(ok)[0][o_], device=dev)].contiguous()
            np.save(f"{a.save_dir}/{inp}_logits.npy", lg.cpu().numpy())
            if a.save_hidden:
                np.save(f"{a.save_dir}/{inp}_hid.npy", hid.view(torch.int16).cpu().numpy())
                np.save(f"{a.save_dir}/{inp}_tok.npy", tsn[ok][o_])
            hh = ah.cpu().numpy()
            rec["mask_hash"] = [hashlib.md5(hh[:, li].tobytes()).hexdigest()[:12] for li in range(NLAYER)]
            rec["sel_tiles"] = cnt.cpu().tolist()
            rec["mask_vs_dump"] = vs_dump
            rec["argmax"] = int(lg.argmax())
            ref = f"{root}/official/{inp}_logits.npy"
            if os.path.exists(ref):
                r_ = np.load(ref)
                l_ = lg.cpu().numpy()
                rec["vs_official"] = dict(
                    argmax_match=bool(int(r_.argmax()) == int(l_.argmax())),
                    rel_l2=float(np.linalg.norm(l_ - r_) / np.linalg.norm(r_)),
                    max_rel=float(np.abs(l_ - r_).max() / np.abs(r_).max()),
                )
        if gated:
            rec["gate_first"] = [[int(x) for x in t.cpu().tolist()] for t in so.gate_log]
        del hn, logits, lg, ahs, hs
        torch.cuda.empty_cache()
        # ---- warm-up + timed prefills
        fwd = lambda ev=None: prefill(M, so, ctx, ids_loc, ev=ev)  # noqa: E731
        for _ in range(a.warmup):
            torch.cuda.synchronize()
            dist.barrier()
            fwd()
        ts_, bd, kept = [], [], []
        for _ in range(a.iters):
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(2 + 3 * NLAYER)]
            if gated:
                so.gate_log.clear()
            torch.cuda.synchronize()
            dist.barrier()
            t0 = time.perf_counter()
            fwd(ev)
            torch.cuda.synchronize()
            t = torch.tensor([time.perf_counter() - t0], device=dev)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            ts_.append(t.item() * 1e3)
            b = torch.tensor(breakdown(ev), device=dev)
            ab = torch.empty(P, 4, device=dev)
            dist.all_gather_into_tensor(ab, b)
            bd.append(ab.cpu().numpy())
            if gated:
                g = torch.stack(so.gate_log).cpu().numpy()
                kept.append(int((g[:, 0] <= g[:, 1] * (1 + GATE_EPS)).sum()))
        rec["prefill_ms"] = float(np.exp(np.mean(np.log(ts_))))
        rec["prefill_all"] = [round(x, 2) for x in ts_]
        bd = np.stack(bd)  # [iters, P, 4] (total, sel, attn, rest) ms
        im = int(np.argsort(ts_)[len(ts_) // 2])
        names = ["gpu_total", "selection", "attention", "rest"]
        rec["breakdown_mean_ms"] = dict(zip(names, np.round(bd[im].mean(0), 2).tolist()))
        rec["breakdown_slowest_ms"] = dict(zip(names, np.round(bd[im][bd[im][:, 0].argmax()], 2).tolist()))
        rec["breakdown_rank_attention_ms"] = np.round(bd[im][:, 2], 1).tolist()
        if gated:
            rec["gate_kept_home_per_call"] = kept
            rec["gate_eps"] = GATE_EPS
        free_, total_ = torch.cuda.mem_get_info()
        dev_used = (total_ - free_) / 2**30
        nontorch = dev_used - torch.cuda.memory_reserved() / 2**30
        heap_res = heap_used = 0.0
        if dex_sys:
            heap_res = int(os.environ.get("NVSHMEM_SYMMETRIC_SIZE", "0")) / 2**30
            heap_used = so.used / 2**30
        pk = torch.cuda.max_memory_allocated() / 2**30
        mem = torch.tensor(
            [pk, dev_used, nontorch, heap_res, heap_used, pk + nontorch - (heap_res - heap_used)], device=dev
        )
        dist.all_reduce(mem, op=dist.ReduceOp.MAX)
        rec.update(
            mem_max_over_ranks=dict(
                zip(
                    [
                        "peak_torch_GiB",
                        "device_used_GiB",
                        "nontorch_GiB",
                        "heap_reserved_GiB",
                        "heap_used_GiB",
                        "peak_actual_GiB",
                    ],
                    [round(x, 3) for x in mem.tolist()],
                )
            )
        )
        rec["wall_s"] = round(time.time() - t_case, 1)
        if rank == 0:
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(
                "[llama]",
                json.dumps({k_: v_ for k_, v_ in rec.items() if k_ not in ("mask_hash", "sel_tiles", "gate_first")}),
                flush=True,
            )
        del ctx
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
