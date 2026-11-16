"""W5 end-to-end: training steps of Wan2.1-T2V-1.3B's denoising transformer (forward + backward, no optimizer) with
context parallelism over all GPUs and SpargeAttention2 self-attention, one attention system per launch.

  * input: a generation's latent x0 and prompt embedding (workloads/w5_sparge/prep_e2e_inputs.py); constant timestep
    t = 500 of 1000, sigma = 0.5: xt = (1 - sigma) x0 + sigma eps (eps: one seed per input), target eps - x0 (Wan's
    flow-matching objective), loss = MSE over all latent elements.
  * model: workloads/w5_sparge/model.py (bf16), per-layer activation recomputation (checkpoint per block, the attention's
    O / LSE / mask kept), parameters replicated, one gradient all-reduce per block per step.
  * layout: balanced zigzag (dex/common/cp_layout.py); self-attention = exact FP16 block-sparse attention on the
    SpargeAttention2 128 x 64 mask (every system); the recomputation reuses the layer's selection and Dex's forward
    tables; Dex's backward plan is searched ahead on a worker thread.
  * steps: the inputs of one resolution in consecutive groups of --accum different inputs per step; the warm-up runs
    every group once untimed (the assigner captures a CUDA graph per shape bucket), then --iters passes over the
    groups are timed. One record per timed step: step time (barrier -> slowest rank), attention forward / backward, gradient
    synchronisation, the pre-backward sync wait, loss, gradient norm, and the peak memory (PyTorch peak + memory outside
    PyTorch, counting only the symmetric memory Dex's tables use).
Systems: dex, dex_no_assigner, ring, allgather, sparse_allgather.
usage: tools/launch.sh experiments/end2end/wan_train.py --system dex --inputs 480p_p0_s0,... --out wan.jsonl"""

import argparse
import json
import os
import sys
import time
import zlib

import torch
import torch.distributed as dist

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from dex.common.env import cfg  # noqa: E402
from workloads.w5_sparge import model as WM  # noqa: E402
from workloads.w5_sparge.common import Reuse, cuda_ext  # noqa: E402
from dex.common.cp_layout import Layout  # noqa: E402

SYSTEMS = ("dex", "dex_no_assigner", "ring", "allgather", "sparse_allgather")


def make_system(name, lay, H):
    if name in ("dex", "dex_no_assigner"):
        from dex.attn.sparge import DexSparge

        return DexSparge(lay, H, assigner=name == "dex")
    if name == "ring":
        from baselines.ring.sparge import RingSparge

        return RingSparge(lay, H)
    if name == "allgather":
        from baselines.allgather.sparge import AllGatherSparge

        return AllGatherSparge(lay, H)
    from baselines.allgather.sparge import SparseAllGatherSparge

    return SparseAllGatherSparge(lay, H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=SYSTEMS)
    ap.add_argument("--inputs", required=True, help="comma list of generation tags of ONE resolution")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--iters", type=int, default=1, help="timed passes over the input groups")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dist.init_process_group("nccl")
    rank, P = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    DA = cuda_ext()
    dex_sys = a.system in ("dex", "dex_no_assigner")
    if dex_sys:
        DA.init_nvshmem(dist.group.WORLD)
    model = WM.load_model(os.path.join(cfg("DEX_MODEL_ROOT"), "Wan2.1-T2V-1.3B-Diffusers"), dev)
    H = model.config.num_attention_heads
    inp_dir = os.path.join(cfg("DEX_DATA_ROOT"), "w5_sparge", "e2e_inputs")
    INP = []
    for tg in a.inputs.split(","):
        d = torch.load(f"{inp_dir}/{tg}.pt")
        x0 = d["x0"].to(dev)
        g = torch.Generator(device="cpu").manual_seed(zlib.crc32(tg.encode()))
        eps = torch.randn(x0.shape, generator=g).to(dev)
        INP.append(dict(tag=tg, xt=((1 - WM.SIGMA) * x0 + WM.SIGMA * eps), tgt=(eps - x0), text=d["text"].to(dev)))
    C_, Fl, Hl, Wl = INP[0]["xt"].shape
    assert all(d["xt"].shape == INP[0]["xt"].shape for d in INP), "one resolution per launch"
    pt, ph, pw = model.config.patch_size
    S = (Fl // pt) * (Hl // ph) * (Wl // pw)
    lay = Layout(S, P, rank, dev)
    for d in INP:
        d["tgt_l"] = WM.patchify_target(d.pop("tgt"), (pt, ph, pw))[lay.tok].contiguous()
    sysobj = make_system(a.system, lay, H)
    for i, blk in enumerate(model.blocks):
        blk.attn1.set_processor(WM.CPAttnProcessor(sysobj, i))
    params = [p for p in model.parameters() if p.requires_grad]
    tstep = torch.tensor([WM.T_STEP], device=dev)
    from torch.utils.checkpoint import checkpoint

    Reuse.enabled, Reuse.ahead = True, True

    def micro(i):
        Reuse.clear()
        WM.OL_CACHE.clear()
        d = INP[i]
        pred = WM.forward_tokens(model, d["xt"], d["text"], lay.tok, tstep, checkpoint)
        loss = ((pred.float() - d["tgt_l"]) ** 2).sum() / (S * pred.shape[-1])  # this rank's share of the global MSE
        loss.backward()
        return loss.detach()

    xev = []

    def step(group):
        lv = sum(micro(i) for i in group)
        x0_ = torch.cuda.Event(enable_timing=True)
        x0_.record()
        for blk in list(model.blocks) + [None]:  # one flat all-reduce per block (+ the rest)
            ps = [
                p
                for p in (
                    blk.parameters()
                    if blk is not None
                    else (p for n, p in model.named_parameters() if not n.startswith("blocks."))
                )
                if p.grad is not None
            ]
            if ps:
                gs = [p.grad for p in ps]
                flat = torch._utils._flatten_dense_tensors(gs)
                dist.all_reduce(flat)
                for g_, f_ in zip(gs, torch._utils._unflatten_dense_tensors(flat, gs)):
                    g_.copy_(f_)
        x1_ = torch.cuda.Event(enable_timing=True)
        x1_.record()
        xev.append((x0_, x1_))
        return lv

    if len(INP) == 1:
        groups = [[0] * a.accum]
    else:
        assert len(INP) % a.accum == 0, (len(INP), a.accum)
        groups = [list(range(j, j + a.accum)) for j in range(0, len(INP), a.accum)]
    warm = list(groups)
    sched = warm + [g for _ in range(a.iters) for g in groups]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    recs = []
    for i, grp in enumerate(sched):
        for k_ in WM.EV:
            WM.EV[k_].clear()
        xev.clear()
        for p in params:
            p.grad = None
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        lv = step(grp)
        torch.cuda.synchronize()
        x = torch.tensor([time.perf_counter() - t0], device=dev)
        dist.all_reduce(x, op=dist.ReduceOp.MAX)
        dist.all_reduce(lv)
        gn = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in params if p.grad is not None)).item()
        brk = torch.tensor(
            [WM.ev_ms("f0", "f1"), WM.bwd_ms(), sum(u.elapsed_time(w) for u, w in xev), WM.ev_ms("s0", "b0")],
            device=dev,
        )
        dist.all_reduce(brk, op=dist.ReduceOp.MAX)
        if i >= len(warm):
            recs.append(
                dict(
                    inputs=[INP[j]["tag"] for j in grp],
                    step_ms=round(x.item() * 1e3, 1),
                    attn_fwd_ms=round(brk[0].item(), 1),
                    attn_bwd_ms=round(brk[1].item(), 1),
                    grad_sync_ms=round(brk[2].item(), 1),
                    phase_sync_ms=round(brk[3].item(), 1),
                    loss=lv.item() / len(grp),
                    grad_norm=gn,
                )
            )
        if rank == 0:
            print(
                f"[wan] step {i}{' (warm-up)' if i < len(warm) else ''}: {x.item() * 1e3:.1f} ms (attn fwd {brk[0]:.0f}, "
                f"attn bwd {brk[1]:.0f}, grad sync {brk[2]:.0f}, phase sync {brk[3]:.0f} ms), loss {lv.item() / len(grp):.5f}",
                flush=True,
            )
    free_, total_ = torch.cuda.mem_get_info()
    dev_used = (total_ - free_) / 2**30
    nontorch = dev_used - torch.cuda.memory_reserved() / 2**30
    heap_res = heap_used = 0.0
    if dex_sys:
        heap_res = int(os.environ.get("NVSHMEM_SYMMETRIC_SIZE", "0")) / 2**30
        heap_used = sysobj.used / 2**30
    peak_actual = torch.cuda.max_memory_allocated() / 2**30 + nontorch - (heap_res - heap_used)
    mem = torch.tensor(
        [torch.cuda.max_memory_allocated() / 2**30, dev_used, nontorch, heap_res, heap_used, peak_actual], device=dev
    )
    dist.all_reduce(mem, op=dist.ReduceOp.MAX)
    if rank == 0:
        mf = dict(
            peak_mem_GiB=round(mem[0].item(), 2),
            device_used_end_GiB=round(mem[1].item(), 2),
            nontorch_GiB=round(mem[2].item(), 2),
            heap_reserved_GiB=round(mem[3].item(), 2),
            heap_used_GiB=round(mem[4].item(), 2),
            peak_actual_GiB=round(mem[5].item(), 2),
        )
        for r in recs:
            r.update(model="Wan2.1-T2V-1.3B", S=S, system=a.system, accum=a.accum, t=WM.T_STEP, sigma=WM.SIGMA, **mf)
            print(json.dumps(r), flush=True)
            if a.out:
                with open(a.out, "a") as f:
                    f.write(json.dumps(r) + "\n")
    dist.barrier()
    if dex_sys:
        DA.finalize_nvshmem()
    os._exit(0)


if __name__ == "__main__":
    main()
