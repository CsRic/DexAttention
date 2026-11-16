"""DCP per-work-unit compute assignment for the balance statistics, from the same cached plans as dcp_comm.py.
Each rank's pickle holds the global WorkloadSpec. Per work unit (one query head x one Q block x one KV block):
device = work_to_device_map, area = sum over the unit's local_attn_mask row ranges of (end - start) (DCP's own exact
token-level mask of the block pair). Check: the block pairs in rank 0's and rank 15's execution plans equal the number of
work units mapped to them.
Output : $DEX_RESULT_ROOT/comm_volume/balance/dcp/<W>_<case>.npz + meta.json
Example: CUDA_VISIBLE_DEVICES= PYTHONPATH=third_party/dcp python experiments/comm_volume/balance_dcp_units.py --jobs 4
"""
import torch  # noqa: F401  (import before numpy keeps the C++ runtime load order of the DCP package)
import argparse
import json
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def n_pairs(ex):
    n = 0
    for ins in ex.instructions:
        if type(ins).__name__ in ("AttnInstr", "AttnBackwardInstr"):
            t = ins.kv_block_table
            n += sum(len(x) for x in (t.block_table if hasattr(t, "block_table") else t))
    return n


def units(wl):
    bm = wl.block_mapping
    ids = sorted(bm.work_id_to_meta)
    dev = np.empty(len(ids), np.int64); area = np.empty(len(ids), np.int64)
    for i, w in enumerate(ids):
        m = bm.work_id_to_meta[w]
        d = wl.work_to_device_map[w]; dev[i] = d[0] * C.GPN + d[1]
        lm = m.local_attn_mask.to(torch.int64)
        area[i] = int((lm[:, 1] - lm[:, 0]).clamp(min=0).sum())
    return dict(dev=dev, area=area)


def job(args):
    W, name, B, cdir, out = args
    res = {}
    for r in (0, C.P - 1):
        fw, bw, wlf, wlb = pickle.load(open(os.path.join(cdir, f"{name}_B{B}.cp{r}.pkl"), "rb"))[0]
        if r == 0:
            U = {ps: units(wl) for ps, wl in (("fwd", wlf), ("bwd", wlb))}
            np.savez(out, **{f"{ps}_{k}": v for ps in U for k, v in U[ps].items()})
        for ps, ex in (("fwd", fw), ("bwd", bw)):
            res[f"{ps}_rank{r}_pairs_eq_units"] = n_pairs(ex) == int((U[ps]["dev"] == r).sum())
    return W, name, res


def main():
    C.assert_no_cuda()
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=4); a = ap.parse_args()
    od = C.out_dir("balance", "dcp")
    tasks = []
    for W in ("W1", "W2", "W3"):
        f = os.path.join(C.out_dir("recv"), f"dcp_{W}.jsonl")
        for r in map(json.loads, open(f)):
            if "recv" in r:
                tasks.append((W, r["case"], r["B"], os.path.join(C.dcp_dir(W), "cache"), os.path.join(od, f"{W}_{r['case']}.npz")))
    meta = {}
    with Pool(a.jobs) as pool:
        for W, name, res in pool.imap_unordered(job, tasks):
            meta[f"{W}|{name}"] = res; print(f"[dcp units] {W} {name} {res}", flush=True)
    json.dump(meta, open(os.path.join(od, "meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
