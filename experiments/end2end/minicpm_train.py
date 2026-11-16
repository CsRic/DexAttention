"""W4 end-to-end training step: MiniCPM4.1-8B prefill forward + next-token loss + backward over the whole model (32 layers),
context parallel on every GPU (zigzag chunks r and 2P-1-r), BF16 (the official InfLLM-v2 build), no optimizer step.
Attention per system, each on its own executor: dex, dex_no_assigner, ring, allgather, sparse_allgather. The top-k
selection is computed in the forward by the official selection code and held constant in the backward.

Memory / step configuration: activation checkpointing per decoder layer + checkpointed loss chunks; HSDP (parameters sharded
over the node's GPUs, replicated across nodes, one cross-node gradient all-reduce per step); `--accum` micro-batches per step,
each a different sequence. Selective recomputation: the backward's recompute reuses the first forward's O / LSE, selection and
forward plan (workloads.w4_infllm.reuse); only the q / k / v projections are recomputed (All-gather also re-gathers K / V).
Dex searches each layer's backward plan on a worker thread after the layer's first forward.

Timing: one step = accum x (forward + loss + backward) + gradient sync, barrier before, CUDA-synchronized, slowest rank; one
warm-up step on the first group, then one record per timed step. Attention forward (incl. recompute) / backward / cross-node
sync from CUDA events on the compute stream; every system has one all-rank stream-ordered sync before each attention backward,
reported apart as phase_sync_ms (the forward's rank imbalance). Dex: plan-stage spans per step (plan_*_ms).
Peak memory: peak_actual_GiB = PyTorch peak allocated + memory outside PyTorch - unused NVSHMEM heap, max over ranks.

Inputs : $DEX_DATA_ROOT/w4_infllm/ids/<seq>.npy, $DEX_MODEL_ROOT/MiniCPM4.1-8B
Output : one json line per timed step appended to --out
usage  : tools/launch.sh experiments/end2end/minicpm_train.py --system dex --seqs <s1,...,s12> --accum 4 --out steps.jsonl"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from dex.common.env import cfg  # noqa: E402
from workloads.w4_infllm import minicpm_model as MM  # noqa: E402
from workloads.w4_infllm import reuse  # noqa: E402
from workloads.w4_infllm.common import gpus_per_node  # noqa: E402

DT = torch.bfloat16
SYSTEMS = ("dex", "dex_no_assigner", "ring", "allgather", "sparse_allgather")
DEX = ("dex", "dex_no_assigner")
LOSS_CHUNK = 2048              # tokens per checkpointed cross-entropy chunk
EV = {"f0": [], "f1": [], "s0": [], "b0": [], "b1q": [], "b1k": [], "b1v": []}
_SYNC_BUF = []


class _Gather(torch.autograd.Function):
    """HSDP: the full parameter from its shard (all-gather over the node on the compute stream); backward = reduce-scatter
    of the full gradient into the shard (synchronous)"""

    @staticmethod
    def forward(ctx, shard, shape, n, group):
        full = shard.new_empty(dist.get_world_size(group) * shard.numel())
        dist.all_gather_into_tensor(full, shard.contiguous(), group=group)
        ctx.n, ctx.group = n, group
        return full[:n].view(shape)

    @staticmethod
    def backward(ctx, g):
        P = dist.get_world_size(ctx.group); g = g.reshape(-1)
        m = -(-ctx.n // P)
        if m * P != ctx.n:
            g = F.pad(g, (0, m * P - ctx.n))
        out = torch.empty(m, dtype=g.dtype, device=g.device)
        dist.reduce_scatter_tensor(out, g.contiguous(), group=ctx.group)
        return out, None, None, None


class ShardParam:
    """a parameter's 1/P shard over the node's ranks"""

    def __init__(self, t, P, rank, group):
        self.shape, self.n, self.group = t.shape, t.numel(), group
        m = -(-self.n // P)
        flat = F.pad(t.reshape(-1), (0, m * P - self.n))
        self.shard = torch.nn.Parameter(flat[rank * m:(rank + 1) * m].clone())

    def full(self):
        return _Gather.apply(self.shard, self.shape, self.n, self.group)


class _Mark(torch.autograd.Function):
    """identity; its backward records a CUDA event into EV[tag]. Tag b0 (the attention backward starts) first runs the
    all-rank sync: stream-ordered, no host block, so the backward's timing starts with every rank aligned."""

    @staticmethod
    def forward(ctx, x, tag):
        ctx.tag = tag
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        if ctx.tag == "b0":
            s = torch.cuda.Event(enable_timing=True); s.record(); EV["s0"].append(s)
            if not _SYNC_BUF:
                _SYNC_BUF.append(torch.zeros(1, device=g.device))
            dist.all_reduce(_SYNC_BUF[0])
        e = torch.cuda.Event(enable_timing=True); e.record(); EV[ctx.tag].append(e)
        return g, None


def ev_ms(a, b):
    return sum(x.elapsed_time(y) for x, y in zip(EV[a], EV[b]))


def bwd_ms():
    """attention backward per layer: from its start to the FIRST of the q / k / v gradient marks"""
    return sum(min(s_.elapsed_time(e) for e in ends) for s_, *ends in zip(EV["b0"], EV["b1q"], EV["b1k"], EV["b1v"]))


def make_op(system, chunks, my, S, mcfg):
    if system == "dex":
        from dex.attn.infllm import DexInfllmOp as Op
    elif system == "dex_no_assigner":
        from dex.attn.infllm import DexInfllmNoAssignerOp as Op
    elif system == "ring":
        from baselines.ring.infllm import RingOp as Op
    elif system == "allgather":
        from baselines.allgather.infllm import AllgatherOp as Op
    else:
        from baselines.allgather.infllm import SparseAllgatherOp as Op
    return Op(chunks, my, S, mcfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=SYSTEMS)
    ap.add_argument("--seqs", required=True, help="comma list of sequences of one length bin; consecutive groups of --accum form a step")
    ap.add_argument("--model", default=None, help="default $DEX_MODEL_ROOT/MiniCPM4.1-8B")
    ap.add_argument("--ids", default=None, help="default $DEX_DATA_ROOT/w4_infllm/ids")
    ap.add_argument("--accum", type=int, default=4, help="micro-batches per step")
    ap.add_argument("--warmup", type=int, default=1, help="warm-up steps on the first group")
    ap.add_argument("--iters", type=int, default=1, help="passes over all groups (12 sequences, accum 4: 3 timed steps per pass)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    a.model = a.model or os.path.join(cfg("DEX_MODEL_ROOT"), "MiniCPM4.1-8B")
    a.ids = a.ids or os.path.join(cfg("DEX_DATA_ROOT"), "w4_infllm", "ids")
    dist.init_process_group("nccl")
    rank, P = dist.get_rank(), dist.get_world_size()
    GPN = gpus_per_node(P)
    lr = int(os.environ.get("LOCAL_RANK", rank % GPN))                 # (explicit index: safetensors' device="cuda" means cuda:0)
    torch.cuda.set_device(lr); dev = torch.device("cuda", lr)
    node_g = cross_g = None
    for n_ in range(P // GPN):                                         # every rank creates every group, same order
        g_ = dist.new_group(list(range(n_ * GPN, (n_ + 1) * GPN)))
        if rank // GPN == n_:
            node_g = g_
    for l_ in range(GPN):
        g_ = dist.new_group(list(range(l_, P, GPN)))
        if rank % GPN == l_:
            cross_g = g_
    DA = None
    if a.system in DEX:
        from dex.attn.infllm import ext
        DA = ext()
        os.environ.setdefault("NVSHMEM_SYMMETRIC_SIZE", str(2 << 30))
        DA.init_nvshmem(dist.group.WORLD)
    mcfg = MM.Cfg(a.model)
    # one launch = one length bin: its sequences all pad to the same S, so the chunk layout and every executor are built once
    INP = []
    for s_ in a.seqs.split(","):
        ids = np.load(os.path.join(a.ids, f"{s_}.npy")).astype(np.int64)
        INP.append(dict(name=s_, ids=ids, S0=len(ids), S=-(-len(ids) // (2 * P * 64)) * (2 * P * 64)))
    S = INP[0]["S"]                                                    # equal 64-aligned zigzag chunks
    assert all(d_["S"] == S for d_ in INP), [(d_["name"], d_["S"]) for d_ in INP]
    cs = S // (2 * P)
    chunks = [(i * cs, (i + 1) * cs) for i in range(2 * P)]
    my = [rank, 2 * P - 1 - rank]
    pos = np.concatenate([np.arange(*chunks[ci]) for ci in my])
    for d_ in INP:
        ids = np.concatenate([d_["ids"], np.zeros(S - d_["S0"], np.int64)])
        tgt = np.where(pos + 1 < d_["S0"], ids[np.minimum(pos + 1, S - 1)], -100)
        d_["ids_l"] = torch.as_tensor(ids[pos], device=dev); d_["tgt_l"] = torch.as_tensor(tgt, device=dev)
        d_["n_tgt"] = d_["S0"] - 1
    cos, sin = MM.rope_cos_sin(mcfg, S, dev, MM.LONG_FACTOR_128K if S > mcfg.orig_max else None)
    cos_l, sin_l = cos[torch.as_tensor(pos, device=dev)], sin[torch.as_tensor(pos, device=dev)]
    # weights: bf16 parameters, sharded over the node
    store = MM.Store(a.model, dev)
    names = ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
             "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight", "input_layernorm.weight",
             "post_attention_layernorm.weight"]

    def param(t):
        return ShardParam(t.to(DT), GPN, rank % GPN, node_g)
    layers = []
    for i in range(mcfg.n_layers):
        W = store.load([f"model.layers.{i}.{n}" for n in names])
        layers.append({n: param(W[f"model.layers.{i}.{n}"]) for n in names}); del W
    W = store.load(["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"])
    top = {k: param(v) for k, v in W.items()}; del W
    torch.cuda.empty_cache()
    leaf = [p.shard for d in layers + [top] for p in d.values()]
    attn = make_op(a.system, chunks, my, S, mcfg)
    rs = mcfg.scale_depth / math.sqrt(mcfg.n_layers)
    nq, nkv, hd = mcfg.n_heads, mcfg.n_kv, mcfg.head_dim
    LKEY = {id(Wl_): i_ for i_, Wl_ in enumerate(layers)}

    def layer_fwd(h, Wl):
        reuse.set_layer(LKEY[id(Wl)])
        g = {n: p.full() for n, p in Wl.items()}
        x = MM.rmsnorm(h, g["input_layernorm.weight"], mcfg.eps)
        q = MM.apply_rope(F.linear(x, g["self_attn.q_proj.weight"]).view(-1, nq, hd), cos_l, sin_l)
        k = MM.apply_rope(F.linear(x, g["self_attn.k_proj.weight"]).view(-1, nkv, hd), cos_l, sin_l)
        v = F.linear(x, g["self_attn.v_proj.weight"]).view(-1, nkv, hd)
        q, k, v = _Mark.apply(q, "b1q"), _Mark.apply(k, "b1k"), _Mark.apply(v, "b1v")   # (backward: attention bwd done)
        e0 = torch.cuda.Event(enable_timing=True); e0.record(); EV["f0"].append(e0)
        o = attn(q, k, v)
        e1 = torch.cuda.Event(enable_timing=True); e1.record(); EV["f1"].append(e1)
        o = _Mark.apply(o, "b0")                                       # (its backward: the attention backward starts)
        h = h + F.linear(o.reshape(o.size(0), -1), g["self_attn.o_proj.weight"]) * rs
        x = MM.rmsnorm(h, g["post_attention_layernorm.weight"], mcfg.eps)
        y = F.linear(F.silu(F.linear(x, g["mlp.gate_proj.weight"])) * F.linear(x, g["mlp.up_proj.weight"]), g["mlp.down_proj.weight"])
        return h + y * rs

    def ce_chunk(xc, Wh, tc):
        return F.cross_entropy(F.linear(xc, Wh).float(), tc, ignore_index=-100, reduction="sum")

    def micro(i_):
        reuse.new_microbatch()                                         # (this micro-batch's own selections / plans / O)
        tgt_l = INP[i_]["tgt_l"]
        h = F.embedding(INP[i_]["ids_l"], top["model.embed_tokens.weight"].full()) * mcfg.scale_emb
        for Wl in layers:
            h = checkpoint(layer_fwd, h, Wl, use_reentrant=False)
        x = MM.rmsnorm(h, top["model.norm.weight"].full(), mcfg.eps) / (mcfg.hidden / mcfg.dim_model_base)
        Wh = top["lm_head.weight"].full()
        loss = h.new_zeros((), dtype=torch.float32)
        for s in range(0, x.size(0), LOSS_CHUNK):                      # chunked fp32 cross entropy, each chunk checkpointed
            loss = loss + checkpoint(ce_chunk, x[s:s + LOSS_CHUNK], Wh, tgt_l[s:s + LOSS_CHUNK], use_reentrant=False)
        loss.backward()
        return loss.detach()

    xev = []

    def step(group):
        """one micro-batch per input of `group` (gradient accumulation into the shards' .grad), then the cross-node sync"""
        lv = sum(micro(i_) for i_ in group)
        x0 = torch.cuda.Event(enable_timing=True); x0.record()
        for d_ in layers + [top]:                                      # one flat all-reduce per layer
            gs = [z_.shard.grad for z_ in d_.values() if z_.shard.grad is not None]
            if gs:
                flat = torch._utils._flatten_dense_tensors(gs)
                dist.all_reduce(flat, group=cross_g)
                for g_, f_ in zip(gs, torch._utils._unflatten_dense_tensors(flat, gs)):
                    g_.copy_(f_)
        x1 = torch.cuda.Event(enable_timing=True); x1.record(); xev.append((x0, x1))
        return lv

    recs = []
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()     # (peak = the training steps, not the weight load)
    assert len(INP) % a.accum == 0, (len(INP), a.accum)
    groups = [list(range(j, j + a.accum)) for j in range(0, len(INP), a.accum)]
    sched = [groups[0]] * a.warmup + [g for _ in range(a.iters) for g in groups]
    for i, grp in enumerate(sched):
        for k_ in EV:
            EV[k_].clear()
        xev.clear()
        for p in leaf:
            p.grad = None
        dist.barrier(); torch.cuda.synchronize(); t0 = time.perf_counter()
        lv = step(grp)
        torch.cuda.synchronize()
        x = torch.tensor([time.perf_counter() - t0], device=dev); dist.all_reduce(x, op=dist.ReduceOp.MAX)
        dist.all_reduce(lv)
        brk = torch.tensor([ev_ms("f0", "f1"), bwd_ms(), sum(u.elapsed_time(w) for u, w in xev), ev_ms("s0", "b0")], device=dev)
        dist.all_reduce(brk, op=dist.ReduceOp.MAX)
        pms = None
        if hasattr(attn, "pop_plan_ms"):                               # plan-stage spans of this step (sum over layers x
            f_, b_, n_ = attn.pop_plan_ms()                            # micro-batches; max and mean over ranks)
            pm = torch.tensor([f_, b_, f_ + b_, f_ + b_], device=dev, dtype=torch.float64)
            pmx = pm.clone(); dist.all_reduce(pmx, op=dist.ReduceOp.MAX); dist.all_reduce(pm)
            pms = dict(plan_fwd_ms_max=round(pmx[0].item(), 1), plan_bwd_ms_max=round(pmx[1].item(), 1),
                       plan_total_ms_max=round(pmx[2].item(), 1), plan_total_ms_mean=round(pm[3].item() / P, 1), plan_calls=n_)
        n_tgt = sum(INP[j]["n_tgt"] for j in grp)
        if i >= a.warmup:
            recs.append(dict(seqs=[INP[j]["name"] for j in grp], step_ms=round(x.item() * 1e3, 1), attn_fwd_ms=round(brk[0].item(), 1),
                             attn_bwd_ms=round(brk[1].item(), 1), grad_sync_ms=round(brk[2].item(), 1),
                             phase_sync_ms=round(brk[3].item(), 1), loss_per_token=lv.item() / max(n_tgt, 1), **(pms or {})))
        if rank == 0:
            print(f"[train] iter {i}: {x.item() * 1e3:.1f} ms (attn fwd incl. recompute {brk[0]:.0f}, attn bwd {brk[1]:.0f}, "
                  f"cross-node grad sync {brk[2]:.0f}, phase sync wait {brk[3]:.0f} ms), loss/token {lv.item() / max(n_tgt, 1):.4f}, "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    # PyTorch peak allocated + memory outside PyTorch at the end (device used - PyTorch reserved: NCCL, CUDA context, the
    # NVSHMEM heap) - (heap reserved - heap Dex's shared forward / backward region used); other systems: no heap term
    free_, total_ = torch.cuda.mem_get_info()
    dev_used = (total_ - free_) / 2**30
    nontorch = dev_used - torch.cuda.memory_reserved() / 2**30
    heap_res = heap_used = 0.0
    if a.system in DEX:
        heap_res = int(os.environ["NVSHMEM_SYMMETRIC_SIZE"]) / 2**30
        heap_used = attn.heap_used() / 2**30
    peak_actual = torch.cuda.max_memory_allocated() / 2**30 + nontorch - (heap_res - heap_used)
    mem = torch.tensor([torch.cuda.max_memory_allocated() / 2**30, dev_used, nontorch, heap_res, heap_used, peak_actual], device=dev)
    dist.all_reduce(mem, op=dist.ReduceOp.MAX)
    if rank == 0:
        mfields = dict(peak_mem_GiB=round(mem[0].item(), 2), device_used_end_GiB=round(mem[1].item(), 2), nontorch_GiB=round(mem[2].item(), 2),
                       heap_reserved_GiB=round(mem[3].item(), 2), heap_used_GiB=round(mem[4].item(), 2), peak_actual_GiB=round(mem[5].item(), 2))
        for r in recs:
            r.update(model="MiniCPM4.1-8B", S_padded=S, system=a.system, accum=a.accum, **mfields)
            print(json.dumps(r), flush=True)
            if a.out:
                open(a.out, "a").write(json.dumps(r) + "\n")
    dist.barrier()
    if DA is not None:                                                 # (NVSHMEM down first, then a hard exit)
        DA.finalize_nvshmem()
    os._exit(0)


if __name__ == "__main__":
    main()
