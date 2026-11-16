"""Magi per-rank compute assignment for the balance statistics (CPU only, Python environment with magi_attention).
Same dispatch as magi_comm.py; Magi computes, on rank r, the attention of the query tokens it hosts
(dispatch_meta_q.host_ranges_per_rank[r]) against all keys they attend. Also records Magi's own bucket area per rank.
Output : $DEX_RESULT_ROOT/comm_volume/balance/magi_hosts.json
Example: CUDA_VISIBLE_DEVICES= python experiments/comm_volume/balance_magi_hosts.py
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from magi_comm import slices_from_ranges, slices_from_segments  # noqa: E402


def hosts(T, qr, kr, ty):
    from magi_attention import env
    from magi_attention.api.functools import apply_padding, compute_pad_size
    from magi_attention.common import AttnRanges
    from magi_attention.common.enum import AttnMaskType
    from magi_attention.meta._make_dispatch_meta import make_bucket_per_rank_from_qk_ranges, make_dispatch_meta_from_qk_ranges
    from magi_attention.meta.solver.dispatch_solver import DispatchConfig
    TY = dict(full=AttnMaskType.FULL, causal=AttnMaskType.CAUSAL, bicausal=AttnMaskType.BICAUSAL, invcausal=AttnMaskType.INVCAUSAL)
    mt = [TY[t] for t in ty]
    q, k = AttnRanges.from_ranges(qr), AttnRanges.from_ranges(kr)
    chunk = math.ceil(T / (env.general.min_chunks_per_rank() * C.P))
    pad = compute_pad_size(T, C.P, chunk)
    Tp = T
    if pad > 0:
        q, k, mt = apply_padding(q_ranges=q, k_ranges=k, attn_mask_type=mt, total_seqlen=T, pad_size=pad)
        Tp = T + pad
    dm_q, _ = make_dispatch_meta_from_qk_ranges(q_ranges=q, k_ranges=k, attn_mask_type=mt, total_seqlen_q=Tp, total_seqlen_k=Tp,
                                                chunk_size=chunk, cp_size=C.P, cp_rank=0, dispatch_config=DispatchConfig(),
                                                is_same_source=True, is_q_permutable=True, is_k_permutable=True)
    buckets = make_bucket_per_rank_from_qk_ranges(q_ranges=q, k_ranges=k, attn_mask_type=mt, dispatch_meta=dm_q)
    hq = [[[int(a.start), int(a.end)] for a in dm_q.host_ranges_per_rank[r]._ranges] for r in range(C.P)]
    return dict(T=T, chunk=chunk, pad=pad, host_q=hq, bucket_area=[int(b.area) for b in buckets])


def main():
    C.assert_no_cuda()
    out = {}
    for W in ("W1", "W2", "W3"):
        for name in C.static_names(W):
            src = C.magi_input(W, name)
            if "ranges" in src:
                T, qr, kr, ty = slices_from_ranges(np.load(src["ranges"]))
            else:
                T, qr, kr, ty = slices_from_segments(np.load(src["segs"])["segs"])
            out[f"{W}|{name}"] = hosts(T, qr, kr, ty)
            print(W, name, flush=True)
    json.dump(out, open(os.path.join(C.out_dir("balance"), "magi_hosts.json"), "w"))


if __name__ == "__main__":
    main()
