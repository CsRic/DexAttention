"""Ring as the replaced system X: plan = the zigzag query-stationary plan at 32 blocks (every pair on its Q owner), order = upstream
Ring's full data flow: at step h GPU g holds K / V of GPU (g - h) mod P (every block, needed or not) and computes its rows against
them. A transfer reaching g at step h counts as delivered then (the executor sends it from the owner directly).
Output: $DEX_DATA_ROOT/plans/ablation/ring/<id>_ring.json. Example: python experiments/ablation_components/ring_order.py"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import sample as SM  # noqa: E402
from dex.planner.placement import zigzag_equal  # noqa: E402

P, GPN, NB, TILE = 16, 4, 32, 128


def ring_full_gk(st, owner):
    rows = [np.flatnonzero(owner == g) for g in range(P)]
    gk = [[] for _ in range(P)]
    for h in range(P):
        for g in range(P):
            o = (g - h) % P
            for q in rows[g]:
                cols = [int(c) for c in rows[o] if st[q, c] == g]
                if cols:
                    gk[g].append(dict(kind="comp", rows=[int(q)], cols=cols, start=float(h), fused=len(cols) > 1))
            if h:
                for b in rows[o]:
                    for gg in ((g - 1) % P, g):
                        gk[gg].append(dict(kind="comm", block=int(b), src=(g - 1) % P, dst=g, io="i", rc="c", start=float(h)))
    return gk


def main():
    out = SM.ablation_dir("ring")
    for _, c in SM.cases():
        z = np.load(c["mask"])
        m = z["mask"]
        T = int(z["tokens"])
        B = T // NB
        k = B // TILE
        live = (m.reshape(NB, k, NB, k) != 0).any((1, 3))
        own = zigzag_equal(NB, P)
        st = np.where(live, own[:, None], -1)
        d = dict(family=c["id"], strategy="zigzag_query_stationary+ring_full_order", cp=P, n_blocks=NB, nodes=P // GPN, gpus_per_node=GPN, block_tokens=B,
                 owner=own.tolist(), schedule_table=st.tolist(), gpu_kernels=ring_full_gk(st, own), mask=c["mask"])
        json.dump(d, open(f"{out}/{c['id']}_ring.json", "w"))
        print(c["id"], "block_tokens", B, "comm items", sum(it["kind"] == "comm" for g in d["gpu_kernels"] for it in g), flush=True)


if __name__ == "__main__":
    main()
