"""W2 / W3 training steps: actor forward + backward of Qwen1.5-1.8B (W2) or Qwen3-4B (W3) on packed rollout groups.

Context parallelism over all 16 ranks (micro-batch 1, no optimizer step). Each rank holds the tokens of the tiles its
attention op owns (local order); embedding, norms, MLP and loss are token-local; attention is the system's CP op
(Dex: dex.attn.static; others: baselines.<system>.e2e). Position ids follow each token's
path (a reply continues the prompt's positions); pad tokens are masked exactly. Token ids are deterministic random.
Loss: next-token cross entropy over real reply tokens whose previous token is in the same segment (fp32, chunked over
1024 tokens, each chunk recomputed in the backward), summed over ranks.
Memory: HSDP (parameters sharded over the node's 4 GPUs and replicated across nodes; gradients reduce-scattered in
the node per micro-batch and all-reduced across nodes once per step); activation checkpointing per decoder layer with
selective recomputation (the recomputed layer reuses the attention's first-forward result); FP16 weights.
Step: --accum micro-batches = --accum DIFFERENT inputs (consecutive groups of the input list). Every attention backward
is preceded by an all-rank synchronization whose wait is reported separately (phase_sync_ms). Timing: barrier ->
step -> slowest rank; --warmup steps on the first group, then --iters passes over all groups, one record per step.
Peak memory (peak_actual_GiB, max over ranks): PyTorch max allocated + device memory outside PyTorch at the end
(NCCL, CUDA context, NVSHMEM heap) - the unused part of the NVSHMEM heap (reserved - used by Dex's regions).
Output: one JSON line per timed step appended to --out.
Example: tools/launch.sh experiments/end2end/qwen_train.py --workload w3 --system dex --inputs w3_32k_p103,... \
           --out $DEX_RESULT_ROOT/end2end/w3/dex.jsonl
"""
import argparse
import importlib
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # every system: no fragmentation
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "attention_layer"), os.path.dirname(os.path.dirname(HERE))]
import static_inputs as SI  # noqa: E402
from dex.attn.cp_common import RECOMPUTE, cp_attention  # noqa: E402
from dex.common.env import cfg  # noqa: E402

TILE = 128
GPN = 4
MODELS = {"w2": "Qwen1.5-1.8B-Chat", "w3": "Qwen3-4B-Thinking-2507"}
CE_CHUNK = 1024


def path_positions(segs, T):
    """position id of every token: a segment starts where its parent's path ends (prompt at 0)"""
    pos = np.zeros(T, np.int64)
    end = {}
    for i, (s, L, kind, g, par, real) in enumerate(segs):
        base = 0 if par < 0 else end[par]
        pos[s:s + L] = base + np.arange(L)
        end[i] = base + real
    return pos


def loss_targets(segs, T, ids):
    """[T] target id (-100 = no loss): the next token inside the same segment, real reply tokens only"""
    tgt = np.full(T, -100, np.int64)
    for s, L, kind, g, par, real in segs:
        if kind == 1 and real > 1:
            tgt[s:s + real - 1] = ids[s + 1:s + real]
    return tgt


class _Gather(torch.autograd.Function):
    """the full parameter from its shard (all-gather over the node group on the compute stream); backward =
    reduce-scatter of the full gradient into the shard (synchronous: nothing runs beside a persistent attention kernel)"""

    @staticmethod
    def forward(ctx, shard, shape, n, group):
        full = shard.new_empty(dist.get_world_size(group) * shard.numel())
        dist.all_gather_into_tensor(full, shard.contiguous(), group=group)
        ctx.n, ctx.group = n, group
        return full[:n].view(shape)

    @staticmethod
    def backward(ctx, g):
        P = dist.get_world_size(ctx.group)
        g = g.reshape(-1)
        m = -(-ctx.n // P)
        if m * P != ctx.n:
            g = torch.nn.functional.pad(g, (0, m * P - ctx.n))
        out = torch.empty(m, dtype=g.dtype, device=g.device)
        dist.reduce_scatter_tensor(out, g.contiguous(), group=ctx.group)
        return out, None, None, None


class ShardedParam:
    """a parameter's 1/P shard over `group`"""

    def __init__(self, t, P, rank, group):
        self.shape, self.n, self.group = t.shape, t.numel(), group
        m = -(-self.n // P)
        flat = torch.nn.functional.pad(t.reshape(-1), (0, m * P - self.n))
        self.shard = torch.nn.Parameter(flat[rank * m:(rank + 1) * m].clone())

    def full(self):
        return _Gather.apply(self.shard, self.shape, self.n, self.group)


EV = {"f0": [], "f1": [], "s0": [], "b0": [], "b1q": [], "b1k": [], "b1v": []}
_SYNC_BUF = []


class _Mark(torch.autograd.Function):
    """identity; its backward records a CUDA event into EV[tag]. Tag b0 (the attention backward starts) first runs an
    all-rank all-reduce on the stream, so the attention backward starts with every rank aligned"""

    @staticmethod
    def forward(ctx, x, tag):
        ctx.tag = tag
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        if ctx.tag == "b0":
            s = torch.cuda.Event(enable_timing=True)
            s.record()
            EV["s0"].append(s)
            if not _SYNC_BUF:
                _SYNC_BUF.append(torch.zeros(1, device=g.device))
            dist.all_reduce(_SYNC_BUF[0])
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        EV[ctx.tag].append(e)
        return g, None


def ev_ms(a, b):
    return sum(x.elapsed_time(y) for x, y in zip(EV[a], EV[b]))


def bwd_ms():
    """attention backward per layer: from its start to the first of the q / k / v gradient marks"""
    return sum(min(s_.elapsed_time(e) for e in ends) for s_, *ends in zip(EV["b0"], EV["b1q"], EV["b1k"], EV["b1v"]))


LAYER_W = ["input_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
           "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight", "post_attention_layernorm.weight",
           "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]


def rms(x, w, eps):
    xf = x.float()
    return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def build_model(model_dir, dev):
    """Qwen3 (W3) or Qwen2 (W2: q / k / v biases, no q / k RMSNorm, untied lm_head), FP16"""
    from transformers import AutoConfig
    cfg_ = AutoConfig.from_pretrained(model_dir)
    if cfg_.model_type == "qwen2":
        from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM as CLS
    else:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM as CLS
    cfg_.torch_dtype = torch.float16
    with torch.device(dev):
        model = CLS.from_pretrained(model_dir, torch_dtype=torch.float16, device_map={"": dev})
    model.train()
    return model, cfg_


def make_ops(a, INP, hq, hk):
    """per input: the system's CP op (attribute `tokens`: this rank's global token positions, -1 = filler) and its
    attention callable"""
    if a.system == "dex":
        from dex.attn.static import DexCP, make_op
        ops = [make_op(d["m"], d["fwd_plan"], d["bwd_plan"], (hq, hk), pad=d["pad"]) for d in INP]
        DexCP.init_symmetric(ops)
        for d, op in zip(INP, ops):
            d["op"], d["attn"] = op, (lambda o_: (lambda q, k, v: cp_attention(q, k, v, o_)))(op)
        return
    mod = importlib.import_module(f"baselines.{a.system}.e2e")
    for d in INP:
        op = mod.make_e2e_op(dict(id=d["id"], mask_path=d["path"], mask=d["m"], heads=(hq, hk), pad=d["pad"],
                                  fwd_plan=d.get("fwd_plan"), bwd_plan=d.get("bwd_plan"), workload=a.workload))
        d["op"] = op
        d["attn"] = op if callable(op) else (lambda o_: (lambda q, k, v: cp_attention(q, k, v, o_)))(op)
        if hasattr(op, "begin"):
            d["begin"], d["end"] = op.begin, op.end


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workload", required=True, choices=["w2", "w3"])
    ap.add_argument("--system", default="dex", choices=["dex", "ring", "allgather", "dcp", "ua", "magi"])
    ap.add_argument("--inputs", required=True, help="comma list of input ids (a multiple of --accum)")
    ap.add_argument("--masks", default="", help="comma list of mask files (default: the inputs' nb32 masks; a system "
                    "planning at another block count passes its own layouts)")
    ap.add_argument("--fwd-plans", default="", help="comma list (default Dex: the head-split plans)")
    ap.add_argument("--bwd-plans", default="")
    ap.add_argument("--head-split", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4, help="micro-batches (distinct inputs) per step")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dist.init_process_group("nccl")
    rank, P = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank % GPN)))
    dev = torch.device("cuda")
    cases = {c["id"]: c for c in SI.inputs(a.workload)}
    ids = a.inputs.split(",")
    if a.system != "dex" and not a.masks:        # a system with its own layouts / plans names them per input
        mod = importlib.import_module(f"baselines.{a.system}.e2e")
        res = mod.resolve_inputs(ids, a.workload) if hasattr(mod, "resolve_inputs") else None
        if res is not None:
            a.masks, a.fwd_plans, a.bwd_plans = (",".join(res[k]) for k in ("masks", "fwd_plans", "bwd_plans"))
    masks = a.masks.split(",") if a.masks else [cases[i]["mask"] for i in ids]
    if a.system == "dex" and not a.fwd_plans:
        fps = [SI.plan_files(cases[i], a.head_split)[1] for i in ids]
        bps = [SI.plan_files(cases[i], a.head_split)[2] for i in ids]
    else:
        fps = a.fwd_plans.split(",") if a.fwd_plans else [None] * len(ids)
        bps = a.bwd_plans.split(",") if a.bwd_plans else [None] * len(ids)
    assert len(ids) == len(masks) == len(fps) == len(bps)
    INP = []
    for i, mk, fp, bp in zip(ids, masks, fps, bps):
        z = np.load(mk)
        INP.append(dict(id=i, name=os.path.basename(mk), path=mk, m=z["mask"].astype(np.int8), T=int(z["tokens"]),
                        segs=z["segs"].tolist(), pad=(z["tok_valid"], z["npad"]) if "tok_valid" in z.files else None,
                        heads=tuple(int(x) for x in z["heads"]), fwd_plan=fp, bwd_plan=bp))
    hq, hk = INP[0]["heads"]
    assert all(d["heads"] == (hq, hk) for d in INP)
    CUR = [0]
    make_ops(a, INP, hq, hk)
    attn = lambda q, k, v: INP[CUR[0]]["attn"](q, k, v)               # noqa: E731
    model, mcfg = build_model(os.path.join(cfg("DEX_MODEL_ROOT"), MODELS[a.workload]), dev)
    for d in INP:
        T = d["T"]
        ids_ = np.random.RandomState(0).randint(0, mcfg.vocab_size, T)
        ref = d["path"].rsplit("_nb", 1)[0] + "_nb32.npz"
        if d["pad"] is not None and not d["path"].endswith("_nb32.npz"):
            # a layout at another block count pads differently: the real tokens take the ids they have in the nb32
            # layout, so every system trains on the same tokens
            z32 = np.load(ref)
            ids32 = np.random.RandomState(0).randint(0, mcfg.vocab_size, int(z32["tokens"]))
            ids_ = np.zeros(T, ids32.dtype)
            ids_[np.flatnonzero(np.asarray(d["pad"][0], bool))] = ids32[np.flatnonzero(z32["tok_valid"].astype(bool))]
        pos = path_positions(d["segs"], T)
        tgt = loss_targets(d["segs"], T, ids_)
        tl = d["op"].tokens.cpu().numpy()
        fill = tl < 0
        tl = np.where(fill, 0, tl)
        d["ids_l"] = torch.as_tensor(ids_[tl], device=dev)
        d["pos_l"] = torch.as_tensor(np.where(fill, 0, pos[tl]), device=dev)[None]
        d["tgt_l"] = torch.as_tensor(np.where(fill, -100, tgt[tl]), device=dev)
        d["n_tgt"] = int((tgt >= 0).sum())
    inner = model.model
    rot = inner.rotary_emb
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    QWEN2 = mcfg.model_type == "qwen2"
    layer_w = LAYER_W
    if QWEN2:
        layer_w = [w for w in LAYER_W if "_norm.weight" not in w or "layernorm" in w] + \
            ["self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias"]
    eps = mcfg.rms_norm_eps
    hd = getattr(mcfg, "head_dim", None) or mcfg.hidden_size // mcfg.num_attention_heads
    layers = list(inner.layers)
    emb, nrm = inner.embed_tokens.weight, inner.norm.weight
    TIED = model.lm_head.weight.data_ptr() == emb.data_ptr()
    head_w = model.lm_head.weight
    node_g = cross_g = None                                            # every rank creates every group, same order
    for n_ in range(P // GPN):
        g_ = dist.new_group(list(range(n_ * GPN, (n_ + 1) * GPN)))
        if rank // GPN == n_:
            node_g = g_
    for l_ in range(GPN):
        g_ = dist.new_group(list(range(l_, P, GPN)))
        if rank % GPN == l_:
            cross_g = g_
    zr = rank % GPN
    LW = [{n: ShardedParam(dict(l_.named_parameters())[n].data, GPN, zr, node_g) for n in layer_w} for l_ in layers]
    TOP = {"emb": ShardedParam(emb.data, GPN, zr, node_g), "norm": ShardedParam(nrm.data, GPN, zr, node_g)}
    if not TIED:
        TOP["head"] = ShardedParam(head_w.data, GPN, zr, node_g)
    for p_ in model.parameters():
        p_.data = torch.empty(0, dtype=p_.dtype, device=dev)
        p_.requires_grad_(False)
    torch.cuda.empty_cache()
    leaf = [z_.shard for d_ in LW + [TOP] for z_ in d_.values()]
    F_ = torch.nn.functional
    LKEY = {id(Wl_): i_ for i_, Wl_ in enumerate(LW)}

    def attn_layer(Wl, h, cos, sin):
        RECOMPUTE.key = LKEY[id(Wl)]
        g = {n: w.full() for n, w in Wl.items()}
        x = rms(h, g["input_layernorm.weight"], eps)
        hs = (*x.shape[:-1], -1, hd)
        if QWEN2:
            q = F_.linear(x, g["self_attn.q_proj.weight"], g["self_attn.q_proj.bias"]).view(hs).transpose(1, 2)
            k = F_.linear(x, g["self_attn.k_proj.weight"], g["self_attn.k_proj.bias"]).view(hs).transpose(1, 2)
            v = F_.linear(x, g["self_attn.v_proj.weight"], g["self_attn.v_proj.bias"]).view(hs).transpose(1, 2)
        else:
            q = rms(F_.linear(x, g["self_attn.q_proj.weight"]).view(hs), g["self_attn.q_norm.weight"], eps).transpose(1, 2)
            k = rms(F_.linear(x, g["self_attn.k_proj.weight"]).view(hs), g["self_attn.k_norm.weight"], eps).transpose(1, 2)
            v = F_.linear(x, g["self_attn.v_proj.weight"]).view(hs).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        q, k, v = _Mark.apply(q, "b1q"), _Mark.apply(k, "b1k"), _Mark.apply(v, "b1v")
        e0 = torch.cuda.Event(enable_timing=True)
        e0.record()
        EV["f0"].append(e0)
        o = attn(q[0].transpose(0, 1), k[0].transpose(0, 1), v[0].transpose(0, 1))      # [Tl, H, D]
        e1 = torch.cuda.Event(enable_timing=True)
        e1.record()
        EV["f1"].append(e1)
        o = _Mark.apply(o, "b0")
        h = h + F_.linear(o.reshape(1, o.shape[0], -1), g["self_attn.o_proj.weight"])
        x = rms(h, g["post_attention_layernorm.weight"], eps)
        return h + F_.linear(F_.silu(F_.linear(x, g["mlp.gate_proj.weight"])) * F_.linear(x, g["mlp.up_proj.weight"]),
                             g["mlp.down_proj.weight"])

    from torch.utils.checkpoint import checkpoint

    def ce_chunk(hc, W, tc):
        return F_.cross_entropy((hc @ W.t()).float(), tc, ignore_index=-100, reduction="sum")

    def micro(i):
        CUR[0] = i
        RECOMPUTE.clear()
        if "begin" in INP[i]:
            INP[i]["begin"]()
        ids_l, pos_l, tgt_l = INP[i]["ids_l"], INP[i]["pos_l"], INP[i]["tgt_l"]
        E = TOP["emb"].full()
        h = F_.embedding(ids_l[None], E)
        del E                       # the lookup's backward needs only the indices
        cos, sin = rot(h, pos_l)
        for Wl in LW:
            h = checkpoint(attn_layer, Wl, h, cos, sin, use_reentrant=False)
        h = rms(h, TOP["norm"].full(), eps)[0]
        loss = h.new_zeros((), dtype=torch.float32)
        W = TOP["emb" if TIED else "head"].full()
        for s in range(0, h.shape[0], CE_CHUNK):
            loss = loss + checkpoint(ce_chunk, h[s:s + CE_CHUNK], W, tgt_l[s:s + CE_CHUNK], use_reentrant=False)
        loss.backward()
        if "end" in INP[i]:
            INP[i]["end"]()
        return loss.detach()

    xev = []

    def step(group):
        """one micro-batch per input of `group` (gradients accumulate into the shards), then the cross-node all-reduce"""
        lv = sum(micro(i) for i in group)
        x0 = torch.cuda.Event(enable_timing=True)
        x0.record()
        for d_ in LW + [TOP]:                                          # one flat all-reduce per layer
            gs = [z_.shard.grad for z_ in d_.values() if z_.shard.grad is not None]
            if gs:
                flat = torch._utils._flatten_dense_tensors(gs)
                dist.all_reduce(flat, group=cross_g)
                for g_, f_ in zip(gs, torch._utils._unflatten_dense_tensors(flat, gs)):
                    g_.copy_(f_)
        x1 = torch.cuda.Event(enable_timing=True)
        x1.record()
        xev.append((x0, x1))
        return lv

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()          # the peak counts the training steps, not the model load
    assert len(INP) % a.accum == 0, (len(INP), a.accum)
    groups = [list(range(j, j + a.accum)) for j in range(0, len(INP), a.accum)]
    sched = [groups[0]] * a.warmup + [g for _ in range(a.iters) for g in groups]
    recs = []
    for i, grp in enumerate(sched):
        for k_ in EV:
            EV[k_].clear()
        xev.clear()
        for p_ in leaf:
            p_.grad = None
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        lv = step(grp)
        torch.cuda.synchronize()
        x = torch.tensor([time.perf_counter() - t0], device=dev)
        dist.all_reduce(x, op=dist.ReduceOp.MAX)
        dist.all_reduce(lv)
        brk = torch.tensor([ev_ms("f0", "f1"), bwd_ms(), sum(u.elapsed_time(w) for u, w in xev), ev_ms("s0", "b0")],
                           device=dev)
        dist.all_reduce(brk, op=dist.ReduceOp.MAX)                    # per component, the slowest rank
        if i >= a.warmup:
            recs.append(dict(inputs=[INP[j]["id"] for j in grp], step_ms=round(x.item() * 1e3, 1),
                             attn_fwd_ms=round(brk[0].item(), 1), attn_bwd_ms=round(brk[1].item(), 1),
                             grad_sync_ms=round(brk[2].item(), 1), phase_sync_ms=round(brk[3].item(), 1),
                             loss_per_token=lv.item() / max(sum(INP[j]["n_tgt"] for j in grp), 1)))
        if rank == 0:
            print(f"[e2e] step {i}: {x.item() * 1e3:.1f} ms (attn fwd incl. recompute {brk[0]:.0f}, attn bwd {brk[1]:.0f}, "
                  f"cross-node grad sync {brk[2]:.0f}, phase sync wait {brk[3]:.0f} ms), "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    free_, total_ = torch.cuda.mem_get_info()
    dev_used = (total_ - free_) / 2**30
    nontorch = dev_used - torch.cuda.memory_reserved() / 2**30
    heap_res = heap_used = 0.0
    if a.system == "dex":
        heap_res = int(os.environ.get("NVSHMEM_SYMMETRIC_SIZE", "0")) / 2**30
        heap_used = max(d["op"].heap_used_bytes() for d in INP) / 2**30   # one shared region: the largest input's
    peak_actual = torch.cuda.max_memory_allocated() / 2**30 + nontorch - (heap_res - heap_used)
    mem = torch.tensor([torch.cuda.max_memory_allocated() / 2**30, dev_used, nontorch, heap_res, heap_used, peak_actual],
                       device=dev)
    dist.all_reduce(mem, op=dist.ReduceOp.MAX)
    if rank == 0:
        mf = dict(peak_mem_GiB=round(mem[0].item(), 2), device_used_end_GiB=round(mem[1].item(), 2),
                  nontorch_GiB=round(mem[2].item(), 2), heap_reserved_GiB=round(mem[3].item(), 2),
                  heap_used_GiB=round(mem[4].item(), 2), peak_actual_GiB=round(mem[5].item(), 2))
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "a") as f:
            for r in recs:
                r.update(workload=a.workload.upper(), system=a.system, accum=a.accum, **mf)
                f.write(json.dumps(r) + "\n")
                print(json.dumps(r), flush=True)
    dist.barrier()
    if a.system == "dex":
        from dex.runtime import finalize_nvshmem   # NVSHMEM down first, then a hard exit without NCCL teardown
        finalize_nvshmem()
    os._exit(0)


if __name__ == "__main__":
    main()
