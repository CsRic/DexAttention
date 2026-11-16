"""Layer-major prefill of MiniCPM4.1-8B with InfLLM-v2 sparse attention; records the selection of every layer.

One process per GPU, no communication. Each worker owns a token-balanced subset of sequences, keeps their hidden
states ([S, 4096] bf16) in pinned host memory, and for every layer loads that layer's weights once and pushes each
sequence through it. Writes

  <out>/<seq>/L{l:02d}_topk.npy   int16 [2 kv-heads, S, 96]: selected 64-token block ids per (KV head, query token),
                                   sorted, -1 = none
  <out>/<seq>/L{l:02d}_xattn.npy  for --x-layers x --x-seqs: the attention input (after input_layernorm), bf16 viewed
                                   as int16 [S, 4096]; the operator tests replay these layers
  <out>/<seq>/meta.json           n_tokens, final NLL / greedy accuracy (sanity check)

Inputs : <data>/seqs.jsonl + <data>/ids/*.npy (prepare_data.py), $DEX_MODEL_ROOT/MiniCPM4.1-8B
Example (4 GPUs of one node, attention inputs of layers 0/8/16/24 for every sequence):
  for w in 0 1 2 3; do python -m workloads.w4_infllm.sample_masks --data $DEX_DATA_ROOT/w4_infllm \\
      --out $DEX_DATA_ROOT/w4_infllm/masks --worker $w --nworkers 4 --device cuda:$w \\
      --x-layers 0,8,16,24 --x-seqs all & done; wait
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from dex.common.env import cfg as env_cfg
from workloads.w4_infllm.minicpm_model import LONG_FACTOR_128K, Cfg, EmbedHead, Layer, Store, rope_cos_sin


def log(w, *a):
    print(f"[w{w} {time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="default $DEX_MODEL_ROOT/MiniCPM4.1-8B")
    ap.add_argument("--data", default=None, help="prepare_data.py output (default $DEX_DATA_ROOT/w4_infllm)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--only", default="", help="comma list of sequence names")
    ap.add_argument("--truncate", type=int, default=0)
    ap.add_argument("--layers", type=int, default=0, help="run only the first N layers (0 = all)")
    ap.add_argument("--x-layers", default="", help="layers whose attention input is stored")
    ap.add_argument("--x-seqs", default="", help="sequences whose attention input is stored ('all' = every one)")
    a = ap.parse_args()
    model = a.model or os.path.join(env_cfg("DEX_MODEL_ROOT"), "MiniCPM4.1-8B")
    data = a.data or os.path.join(env_cfg("DEX_DATA_ROOT"), "w4_infllm")

    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    torch.set_grad_enabled(False)
    cfg = Cfg(model)
    store = Store(model, dev)
    rows = [json.loads(line) for line in open(os.path.join(data, "seqs.jsonl"))]
    if a.only:
        keep = set(a.only.split(","))
        rows = [r for r in rows if r["name"] in keep]
    rows.sort(key=lambda r: -r["n_tokens"])
    load = [0] * a.nworkers
    mine = []
    for r in rows:
        w = min(range(a.nworkers), key=lambda i: load[i])
        load[w] += r["n_tokens"]
        if w == a.worker:
            mine.append(r)
    if not mine:
        return
    x_layers = {int(x) for x in a.x_layers.split(",") if x}
    x_all = a.x_seqs == "all"
    x_seqs = set(a.x_seqs.split(",")) if a.x_seqs and not x_all else set()
    n_layers = a.layers or cfg.n_layers

    eh = EmbedHead(cfg, store)
    seqs = []
    for r in mine:
        ids = torch.from_numpy(np.load(os.path.join(data, "ids", r["name"] + ".npy")).astype(np.int64))
        if a.truncate:
            ids = ids[:a.truncate]
        h = eh.embed(ids.to(dev)).cpu().pin_memory()
        os.makedirs(os.path.join(a.out, r["name"]), exist_ok=True)
        S = ids.numel()
        cos, sin = rope_cos_sin(cfg, S, "cpu", LONG_FACTOR_128K if S > cfg.orig_max else None)
        seqs.append(dict(row=r, ids=ids, h=h, cos=cos, sin=sin))
    log(a.worker, f"{len(seqs)} seqs, {sum(s['ids'].numel() for s in seqs)} tokens")

    for l in range(n_layers):
        t0 = time.time()
        layer = Layer(cfg, l, store)
        for s in seqs:
            t1 = time.time()
            name = s["row"]["name"]
            h = s["h"].to(dev, non_blocking=True)
            cap = {"want_x": l in x_layers and (x_all or name in x_seqs)}
            layer.forward(h, s["cos"].to(dev), s["sin"].to(dev), cap)
            s["h"].copy_(h)
            if "topk" in cap:
                np.save(os.path.join(a.out, name, f"L{l:02d}_topk.npy"),
                        cap["topk"].to(torch.int16).cpu().numpy())
            if "x_attn" in cap:
                np.save(os.path.join(a.out, name, f"L{l:02d}_xattn.npy"),
                        cap["x_attn"].view(torch.int16).cpu().numpy())
            del h, cap
            torch.cuda.synchronize()
            log(a.worker, f"L{l:02d} {name} S={s['ids'].numel()} {time.time() - t1:.1f}s")
        del layer
        torch.cuda.empty_cache()
        log(a.worker, f"layer {l} done {time.time() - t0:.1f}s peak "
                      f"{torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GiB")

    if n_layers == cfg.n_layers:
        for s in seqs:
            nll, acc = eh.nll(s["h"].to(dev), s["ids"].to(dev))
            meta = dict(s["row"], nll=nll, ppl=float(np.exp(nll)), greedy_acc=acc,
                        n_tokens_run=int(s["ids"].numel()))
            json.dump(meta, open(os.path.join(a.out, s["row"]["name"], "meta.json"), "w"), indent=1)
            log(a.worker, f"{s['row']['name']} nll={nll:.3f} ppl={np.exp(nll):.2f} acc={acc:.3f}")


if __name__ == "__main__":
    main()
