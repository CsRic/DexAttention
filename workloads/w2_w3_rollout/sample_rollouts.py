"""Sample GRPO-style rollout groups on LongReason with vLLM and record their token lengths.

The LongReason prompt is the single chat message; chat template with thinking enabled and the generation prompt; n replies per
prompt (temperature 1.0, top-p 1.0, at most --max-tokens each); prefix caching shares the prompt across the replies.
Per reply: P (prompt tokens), R (generated tokens), finish reason, whether the thinking block closed, whether the final
answer letter matches the ground truth. Only the lengths are needed downstream (the masks depend on P and R only).

Inputs : $DEX_DATA_ROOT/longreason/data/<split>-*.parquet (LongReason, columns prompt / answer / example_idx),
         $DEX_MODEL_ROOT/<model>
Output : $DEX_DATA_ROOT/rollouts/R_<split>_max<max_tokens>_<tag>.json  {meta, responses: [{idx, P, R, ...}]}
Example: python -m workloads.w2_w3_rollout.sample_rollouts --split 64k --tp 2
"""
import argparse
import glob
import json
import os
import re
import time

import numpy as np

from dex.common.env import cfg

DEFAULT_MODEL = "Qwen3-4B-Thinking-2507"
DEFAULT_TAG = "qwen3-4b-thinking"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, help="LongReason split: 32k, 64k or 128k")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="directory name under $DEX_MODEL_ROOT")
    ap.add_argument("--tag", default=DEFAULT_TAG, help="output file suffix (the selection reads *_<tag>.json)")
    ap.add_argument("--prompts", type=int, default=48)
    ap.add_argument("--n", type=int, default=16, help="replies per prompt")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--tp", type=int, default=4, help="tensor parallel GPUs (128k: 4, 32k / 64k: 2)")
    ap.add_argument("--gpu-util", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-text", action="store_true", help="also write the reply texts to $DEX_SCRATCH")
    a = ap.parse_args()
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    data = os.path.join(cfg("DEX_DATA_ROOT"), "longreason")
    model = os.path.join(cfg("DEX_MODEL_ROOT"), a.model)
    files = sorted(glob.glob(os.path.join(data, f"data/{a.split}-*.parquet")))
    rows = [r for f in files for r in pq.read_table(f, columns=["prompt", "answer", "example_idx"]).to_pylist()]
    rng = np.random.RandomState(a.seed)
    pick = sorted(rng.choice(len(rows), min(a.prompts, len(rows)), replace=False).tolist())
    tok = AutoTokenizer.from_pretrained(model)
    texts = [tok.apply_chat_template([{"role": "user", "content": rows[i]["prompt"]}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=True) for i in pick]
    plen = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
    llm = LLM(model, tensor_parallel_size=a.tp, max_model_len=max(plen) + a.max_tokens + 64, enable_prefix_caching=True,
              gpu_memory_utilization=a.gpu_util, seed=a.seed, max_num_batched_tokens=8192,
              compilation_config=dict(max_cudagraph_capture_size=256))
    sp = SamplingParams(n=a.n, temperature=a.temperature, top_p=1.0, max_tokens=a.max_tokens, seed=a.seed)
    t0 = time.time()
    outs = llm.generate(texts, sp)
    dt = time.time() - t0
    recs, raw = [], []
    for k, (i, o) in enumerate(zip(pick, outs)):
        gt = rows[i]["answer"].strip()
        for c in o.outputs:
            closed = "</think>" in c.text
            tail = c.text.split("</think>")[-1] if closed else ""
            m = re.findall(r"\b([A-J])\b", tail)
            recs.append(dict(idx=i, example_idx=rows[i]["example_idx"], P=plen[k], R=len(c.token_ids),
                             finish=c.finish_reason, think_closed=closed, correct=bool(m and m[-1] == gt[:1])))
            raw.append(dict(idx=i, text=c.text))
    gen_tok = sum(r["R"] for r in recs)
    meta = dict(split=a.split, prompts=len(pick), n=a.n, max_tokens=a.max_tokens, temperature=a.temperature,
                model=a.model, seconds=dt, gen_tokens=gen_tok, tok_per_s=gen_tok / dt)
    out = os.path.join(cfg("DEX_DATA_ROOT"), "rollouts")
    os.makedirs(out, exist_ok=True)
    json.dump(dict(meta=meta, responses=recs), open(os.path.join(out, f"R_{a.split}_max{a.max_tokens}_{a.tag}.json"), "w"))
    if a.save_text:
        json.dump(raw, open(os.path.join(cfg("DEX_SCRATCH"), f"text_{a.split}_max{a.max_tokens}_{a.tag}.json"), "w"))
    R = np.array([r["R"] for r in recs])
    print(json.dumps(meta))
    print(f"P median {int(np.median(plen))}  R p10/p50/p90/max {np.percentile(R, 10):.0f}/{np.median(R):.0f}/"
          f"{np.percentile(R, 90):.0f}/{R.max()}  truncated {np.mean([r['finish'] == 'length' for r in recs]):.1%}")


if __name__ == "__main__":
    main()
