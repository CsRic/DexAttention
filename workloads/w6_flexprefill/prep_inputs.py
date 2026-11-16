"""W6 token inputs: per length L in {32K, 64K, 128K}: RULER niah_multikey_2 #0, RULER qa_1 #0 (RULER generator, meta-llama3
template, num_samples=3, seed 42), InfiniteBench longbook_qa_eng (first record with a book of at least 128K tokens;
FlexPrefill's prompt, middle truncation to L-40 tokens, Llama-3 chat header).
Inputs : $DEX_DATA_ROOT/w6_flexprefill/ruler/<L>/<task>/validation.jsonl, infbench/longbook_qa_eng.jsonl,
         $DEX_MODEL_ROOT/Llama-3.1-8B-Instruct (tokenizer)
Outputs: $DEX_DATA_ROOT/w6_flexprefill/inputs/<input>.npy + inputs.json (names as in inputs9.txt)
usage: python prep_inputs.py"""

import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dex.common.env import cfg  # noqa: E402

TPL = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
BODY = "Read the book and answer the question. Be very concise in your answer.\n\n{}\n\nQuestion: {}\nAnswer:"


def main():
    from transformers import AutoTokenizer

    root = os.path.join(cfg("DEX_DATA_ROOT"), "w6_flexprefill")
    out, ruler = f"{root}/inputs", f"{root}/ruler"
    os.makedirs(out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(os.path.join(cfg("DEX_MODEL_ROOT"), "Llama-3.1-8B-Instruct"))
    ib = None
    for line in open(f"{root}/infbench/longbook_qa_eng.jsonl"):
        r = json.loads(line)
        if len(tok(r["context"], add_special_tokens=False).input_ids) >= 131072:
            ib = r
            break
    meta = []

    def save(name, L, ids, src):
        a = np.array(ids, np.int32)
        np.save(f"{out}/{name}.npy", a)
        meta.append(dict(name=name, L=L, n=len(ids), src=src, md5=hashlib.md5(a.tobytes()).hexdigest()))

    for L in (32768, 65536, 131072):
        for task in ("niah_multikey_2", "qa_1"):
            d = json.loads(open(f"{ruler}/{L}/{task}/validation.jsonl").readline())
            ids = tok(d["input"], add_special_tokens=False).input_ids
            save(f"ruler_{task}_i{d['index']}_{L // 1024}k", L, ids, f"RULER {task} index {d['index']}")
        pre, post = TPL.split("{}")
        cids = tok(BODY.format(ib["context"], ib["input"]), add_special_tokens=False).input_ids
        a = tok(pre, add_special_tokens=False).input_ids
        b = tok(post, add_special_tokens=False).input_ids
        keep = L - 40 - len(a) - len(b)
        if len(cids) > keep:
            h = keep // 2
            cids = cids[:h] + cids[len(cids) - (keep - h) :]
        save(
            f"infbench_longbook_qa_eng_i{ib['id']}_{L // 1024}k",
            L,
            a + cids + b,
            f"InfiniteBench longbook_qa_eng id {ib['id']}, middle-truncated",
        )
    json.dump(meta, open(f"{out}/inputs.json", "w"), indent=1)
    for m in meta:
        print(m)


if __name__ == "__main__":
    main()
