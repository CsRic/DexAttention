"""Prints MagiAttention's allocated versus reserved memory and the cost of clearing PyTorch's allocator cache during training.

Overhead of a training step = step_ms with clearing / step_ms without - 1, paired on the step's inputs. Only steps of the bins
given by --bins that have both runs enter, so a bin whose uncleared run did not fit reports no pair.
Inputs : $DEX_RESULT_ROOT/memory/magi_alloc.jsonl (magi_alloc.py); $DEX_RESULT_ROOT/end2end/<w>/{launch1,noclear}.jsonl
Outputs: stdout only
Example: python experiments/memory/magi_cache_summary.py
"""
import argparse
import decimal
import json
import math
import os
import re


def half_up(x):
    """one decimal, ties away from zero on the value as recorded (4.55 -> 4.6), as the tables of analysis/"""
    return decimal.Decimal(str(x)).quantize(decimal.Decimal("0.1"), rounding=decimal.ROUND_HALF_UP)


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def steps(path):
    """{inputs tuple: step_ms} of Magi's training steps; a later record of the same inputs replaces an earlier one"""
    return {tuple(r["inputs"]): r["step_ms"] for r in read_jsonl(path) if r.get("system") == "magi" and "step_ms" in r}


def bin_of(inputs):
    m = re.search(r"_(\d+k)_p", inputs[0])
    return m.group(1) if m else "?"


def summarize_alloc(path):
    rows = [r for r in read_jsonl(path) if r["mode"] == "plain_fb"]
    if not rows:
        print(f"allocator: no records in {path}")
        return
    last = max(rows, key=lambda r: r["call"])
    print(f"allocator (calc_attn + backward, {last['call']} calls, rank 0): {half_up(last['peak_GiB'])} GiB allocated at peak, "
          f"{half_up(last['reserved_GiB'])} GiB reserved")


def summarize_overhead(root, cleared_tag, uncleared_tag, bins):
    every = []
    for w in ("w2", "w3"):
        d = os.path.join(root, "end2end", w)
        on, off = steps(os.path.join(d, f"{cleared_tag}.jsonl")), steps(os.path.join(d, f"{uncleared_tag}.jsonl"))
        for b in bins:
            pairs = [(on[k], off[k]) for k in on if k in off and bin_of(k) == b]
            if not pairs:
                print(f"{w.upper()} {b}: no paired steps (no uncleared run)")
                continue
            ratios = [x / y for x, y in pairs]
            gm = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
            every += ratios
            print(f"{w.upper()} {b}: {len(pairs)} steps, cleared / uncleared {gm:.3f} (overhead {100 * (gm - 1):.1f}%; "
                  f"per step {100 * (min(ratios) - 1):.1f}-{100 * (max(ratios) - 1):.1f}%)")
    if every:
        print(f"overhead range over {len(every)} steps: {100 * (min(every) - 1):.0f}-{100 * (max(every) - 1):.0f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("DEX_RESULT_ROOT", "results"))
    ap.add_argument("--cleared-tag", default="launch1", help="tag of the end-to-end run with clearing")
    ap.add_argument("--uncleared-tag", default="noclear", help="tag of the run with DEX_MAGI_CLEAR_CACHE=0")
    ap.add_argument("--bins", nargs="+", default=["32k", "64k"])
    a = ap.parse_args()
    summarize_alloc(os.path.join(a.root, "memory", "magi_alloc.jsonl"))
    summarize_overhead(a.root, a.cleared_tag, a.uncleared_tag, a.bins)


if __name__ == "__main__":
    main()
