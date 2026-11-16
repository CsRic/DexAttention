"""Mask statistics of the dynamic workloads, computed on the CPU from their selections alone (no attention call, no result record):
  W4  per GPU, the 64-token key blocks selected by the GPU's queries under some KV head (causal filter applied) as a share of the blocks
      up to its last query block; range over cases of the largest GPU's share, and per length the geometric mean over cases
      of the mean over the case's GPUs
  W5  SpargeAttention2 selection on the dumped Q / K: selected 128 x 64 blocks / all blocks; geometric mean per resolution
  W6  FlexPrefill tile mask: selected tiles / causal tiles per query head; K + V bytes a GPU must receive under need-based fetching
      (remote tiles of its per-KV-head unions, the slowest GPU) / All-gather's bytes; geometric means over cases per length
Inputs : $DEX_DATA_ROOT/w4_infllm/masks/<seq>/L??_topk.npy, w5_sparge/dumps/<generation>/rank<r>/call<i>.npz,
         w6_flexprefill/qkv/<input>/L??_mask_g095.npy, for the cases of workloads/*/cases*.txt; absent cases are skipped and counted.
Outputs: $DEX_RESULT_ROOT/analysis/mask_stats.json (compared by check_expected.py), a summary on stdout.
Example: CUDA_VISIBLE_DEVICES= python analysis/mask_stats.py w4 w5 w6
"""
import collections
import json
import os
import sys

import numpy as np

import common as C
from geomean import gm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from dex.common.env import cfg  # noqa: E402

P = 16
W4_BLOCK, TILE, W6_HQ, W6_HK, W6_GQ, D = 64, 128, 32, 8, 4, 128


def data_path(*sub):
    return os.path.join(cfg("DEX_DATA_ROOT"), *sub)


def case_lines(*sub):
    return [x.strip() for x in open(os.path.join(REPO, "workloads", *sub)) if x.strip()]


# ---------------------------------------------------------------- W4
def w4_chunks(S):
    """2P zigzag chunks of whole 64-token blocks, the remainder spread over the first chunks"""
    nb = (S + W4_BLOCK - 1) // W4_BLOCK
    q, rem = divmod(nb, 2 * P)
    b = [0]
    for i in range(2 * P):
        b.append(min(S, b[-1] + (q + (i < rem)) * W4_BLOCK))
    return [(b[i], b[i + 1]) for i in range(2 * P)]


def w4_case(path):
    """-> coverage of one case per GPU [P]"""
    tk = np.load(path, mmap_mode="r")
    S = tk.shape[1]
    chunks = w4_chunks(S)
    nb = (S + W4_BLOCK - 1) // W4_BLOCK
    cov = np.zeros(P)
    for r in range(P):
        last = 0
        need = np.zeros(nb, bool)
        for c0, c1 in (chunks[r], chunks[2 * P - 1 - r]):
            if c1 <= c0:
                continue
            s = np.asarray(tk[:, c0:c1, :]).astype(np.int64)
            s = np.where(s > (np.arange(c0, c1)[None, :, None] // W4_BLOCK), -1, s)
            need[s[s >= 0]] = True
            last = max(last, (c1 - 1) // W4_BLOCK)
        cov[r] = need[:last + 1].sum() / (last + 1)
    return cov


def w4():
    mean, largest, missing = collections.defaultdict(list), [], 0
    for line in case_lines("w4_infllm", "cases144.txt"):
        seq, lay = line.split(":")
        f = data_path("w4_infllm", "masks", seq, f"L{int(lay):02d}_topk.npy")
        if not os.path.exists(f):
            missing += 1
            continue
        cov = w4_case(f)
        mean[C.group_of("W4", seq)].append(float(cov.mean()))
        largest.append(float(cov.max()))
    return dict(n=len(largest), missing=missing, largest_gpu_coverage=[min(largest), max(largest)],
                mean_gpu_coverage_per_length={g: gm(v) for g, v in mean.items()})


# ---------------------------------------------------------------- W5
def w5():
    sys.path.insert(0, os.path.join(REPO, "experiments", "comm_volume"))
    import torch
    import w5_common as W5
    from workloads.w5_sparge.cases import dump_dir, read_cases
    dev = torch.device("cpu")
    per, cases, missing = collections.defaultdict(list), {}, 0
    for gen, call in read_cases():
        d = dump_dir(gen)
        if not all(os.path.exists(f"{d}/rank{r}/call{call:05d}.npz") for r in range(W5.H // 3)):
            missing += 1
            continue
        dens = W5.mask_of(d, call, dev)[3]
        per[gen.split("_")[0]].append(dens)
        cases[f"{gen}_c{call:05d}"] = dens
    return dict(n=len(cases), missing=missing, density_per_resolution={g: gm(v) for g, v in per.items()}, cases=cases)


# ---------------------------------------------------------------- W6
def w6_owner(S_pad):
    per = S_pad // (2 * P * TILE)
    return np.repeat([c if c < P else 2 * P - 1 - c for c in range(2 * P)], per)


def w6_case(path, S_pad):
    """-> (density, need-based K + V bytes of the slowest GPU / All-gather's, MiB of the slowest GPU)"""
    mk = np.load(path)
    nb = mk.shape[1]
    m = np.unpackbits(mk, axis=-1, count=nb).astype(bool)                # [32, nb, nb]
    tri = nb * (nb + 1) / 2
    own = w6_owner(S_pad)
    union = m.reshape(W6_HK, W6_GQ, nb, nb).any(1)                       # per KV head: the union over its query heads
    need = [int((union[:, own[:nb] == r].any(1) & (own[:nb] != r)).sum()) for r in range(P)]
    tile_bytes = TILE * D * 2 * 2                                        # K + V fp16 of one tile and KV head
    allgather = (P - 1) * (S_pad // P) * W6_HK * D * 2 * 2
    return float(m.sum()) / (W6_HQ * tri), max(need) * tile_bytes / allgather, max(need) * tile_bytes / float(1 << 20)


def w6():
    dens, ratio, cases, missing = collections.defaultdict(list), collections.defaultdict(list), {}, 0
    for line in case_lines("w6_flexprefill", "cases36.txt"):
        inp, lay = line.split(":")
        f = data_path("w6_flexprefill", "qkv", inp, f"L{int(lay):02d}_mask_g095.npy")
        if not os.path.exists(f):
            missing += 1
            continue
        g = C.group_of("W6", inp)
        d, r, mib = w6_case(f, int(g[:-1]) * 1024)
        dens[g].append(d); ratio[g].append(r)
        cases[f"{inp}_L{int(lay):02d}"] = dict(density=d, need_over_allgather=r, need_max_MiB=mib)
    return dict(n=len(cases), missing=missing, density_per_length={g: gm(v) for g, v in dens.items()},
                need_over_allgather_per_length={g: gm(v) for g, v in ratio.items()}, cases=cases)


def main():
    res = {}
    for w in sys.argv[1:] or ("w4", "w5", "w6"):
        res[w.upper()] = dict(w4=w4, w5=w5, w6=w6)[w]()
        print(w.upper(), json.dumps({k: v for k, v in res[w.upper()].items() if k != "cases"}), flush=True)
    f = os.path.join(C.ensure_out(), "mask_stats.json")
    if os.path.exists(f):
        res = {**json.load(open(f)), **res}
    json.dump(res, open(f, "w"), indent=1)


if __name__ == "__main__":
    main()
