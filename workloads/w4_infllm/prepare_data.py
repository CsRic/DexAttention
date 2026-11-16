"""Build the long-context prefill inputs of W4: LongBench v2 items wrapped in MiniCPM4.1's chat template (thinking
mode) and tokenized with its tokenizer.

Items are bucketed by token length: for bucket B we take items whose full prompt is >= B tokens and truncate the
CONTEXT from the middle (LongBench's own truncation rule) so the prompt is ~B tokens. Per (domain, bucket) `--per`
items are drawn with a seeded shuffle (6 domains x 2 = 12 sequences per length).

Inputs : $DEX_MODEL_ROOT/MiniCPM4.1-8B (tokenizer), $DEX_DATA_ROOT/LongBench-v2/data.json
Outputs: <out>/seqs.jsonl (one line per sequence: name, id, domain, bucket, n_tokens, answer) and
         <out>/ids/<id>_<bucket>.npy (int32 token ids)
Example: python -m workloads.w4_infllm.prepare_data --out $DEX_DATA_ROOT/w4_infllm
"""
import argparse
import json
import os
import random

import numpy as np
from transformers import PreTrainedTokenizerFast

from dex.common.env import cfg

TEMPLATE = (
    "Please read the following text and answer the question below.\n\n"
    "<text>\n{context}\n</text>\n\n"
    "What is the correct answer to this question: {question}\n"
    "Choices:\n(A) {A}\n(B) {B}\n(C) {C}\n(D) {D}\n\n"
    "Format your response as follows: \"The correct answer is (insert answer here)\"."
)


def build(tok, item, ctx_ids, keep):
    """Prompt token ids with the context truncated to `keep` tokens (middle cut)."""
    if keep < len(ctx_ids):
        half = keep // 2
        ctx_ids = ctx_ids[:half] + ctx_ids[len(ctx_ids) - (keep - half):]
    context = tok.decode(ctx_ids)
    text = TEMPLATE.format(context=context, question=item["question"], A=item["choice_A"],
                           B=item["choice_B"], C=item["choice_C"], D=item["choice_D"])
    prompt = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=True)
    return [tok.bos_token_id] + tok.encode(prompt, add_special_tokens=False)   # LlamaTokenizer add_bos_token=True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="MiniCPM4.1-8B checkpoint (default $DEX_MODEL_ROOT/MiniCPM4.1-8B)")
    ap.add_argument("--data", default=None, help="LongBench v2 data.json (default $DEX_DATA_ROOT/LongBench-v2/data.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--buckets", default="32768,65536,131072")
    ap.add_argument("--per", type=int, default=2, help="items per (domain, bucket)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    model = a.model or os.path.join(cfg("DEX_MODEL_ROOT"), "MiniCPM4.1-8B")
    data_path = a.data or os.path.join(cfg("DEX_DATA_ROOT"), "LongBench-v2", "data.json")

    tok = PreTrainedTokenizerFast.from_pretrained(model)   # AutoTokenizer needs the remote-code config
    data = json.load(open(data_path))
    rng = random.Random(a.seed)
    buckets = [int(b) for b in a.buckets.split(",")]
    os.makedirs(os.path.join(a.out, "ids"), exist_ok=True)

    ctx = {}                                           # context token counts (tokenize once)
    for it in data:
        ctx[it["_id"]] = tok.encode(it["context"], add_special_tokens=False)
    overhead = len(build(tok, data[0], [], 0))         # prompt length without the context
    by_dom = {}
    for it in data:
        by_dom.setdefault(it["domain"], []).append(it)

    used = set()
    rows = []
    for B in buckets:
        for dom in sorted(by_dom):
            cands = [it for it in by_dom[dom]
                     if len(ctx[it["_id"]]) + overhead + 512 >= B and it["_id"] not in used]
            rng.shuffle(cands)
            for it in cands[:a.per]:
                keep = B - overhead - len(tok.encode(it["question"] + it["choice_A"] + it["choice_B"]
                                                     + it["choice_C"] + it["choice_D"],
                                                     add_special_tokens=False)) - 64
                ids = build(tok, it, ctx[it["_id"]], keep)
                ids = ids[:B]
                used.add(it["_id"])
                name = f"{it['_id']}_{B}"
                np.save(os.path.join(a.out, "ids", name + ".npy"), np.asarray(ids, dtype=np.int32))
                rows.append(dict(name=name, id=it["_id"], domain=dom, sub_domain=it["sub_domain"],
                                 bucket=B, n_tokens=len(ids), answer=it["answer"],
                                 ctx_tokens=len(ctx[it["_id"]])))
                print(f"{name:40s} {dom:38s} B={B:7d} n={len(ids):7d} ctx={len(ctx[it['_id']])}", flush=True)
    with open(os.path.join(a.out, "seqs.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} sequences, {sum(r['n_tokens'] for r in rows) / 1e6:.2f}M tokens")


if __name__ == "__main__":
    main()
