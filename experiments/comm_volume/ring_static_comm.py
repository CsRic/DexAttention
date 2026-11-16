"""Ring (dense ring communication, sparse compute) on W1-W3: analytic per-GPU received bytes.

Forward: P - 1 steps, each rank sends its current K and V shard to rank + 1 whatever the mask. Backward: the same P - 1
K / V steps plus P sends of the fp32 dK / dV accumulators (the last one returns them home). Shard = S_run / P tokens
(S_run = the padded length of the input's tile mask), K and V fp16 [S_run / P, Hkv, 128]. Only the first GPU of each node
receives across the node boundary.
Inputs : the tile masks (common.mask_path)
Output : $DEX_RESULT_ROOT/comm_volume/recv/ring_<W>.jsonl
Example: python experiments/comm_volume/ring_static_comm.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import common as C  # noqa: E402


def main():
    for W in ("W1", "W2", "W3"):
        recs = []
        for name in C.static_names(W):
            z = np.load(C.mask_path(W, name))
            S, (hq, hk) = z["mask"].shape[0] * C.TILE, (int(x) for x in z["heads"])
            assert S % (2 * C.P * C.TILE) == 0, (name, S)
            sh = S // C.P
            kv16, kv32 = 2 * sh * hk * C.D * 2, 2 * sh * hk * C.D * 4
            recv = C.empty_recv()
            for d in range(C.P):
                s = (d - 1) % C.P
                C.add(recv, "fwd", s, d, (C.P - 1) * kv16)
                C.add(recv, "bwd", s, d, (C.P - 1) * kv16 + C.P * kv32)
            recs.append(dict(system="Ring", workload=W, case=name, S_run=S, heads=[hq, hk], recv=recv))
        C.write_jsonl(os.path.join(C.out_dir("recv"), f"ring_{W}.jsonl"), recs)
        print(f"[ring] {W}: {len(recs)} cases")


if __name__ == "__main__":
    main()
