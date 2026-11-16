"""Magi (MagiAttention native, 1-D communication) on W1-W3: per-GPU received bytes, recomputed on the CPU from Magi's
own dispatch metadata (its meta functions take cp_size / cp_rank explicitly, so no process group is needed).

  chunk, pad  chunk = ceil(T / (min_chunks_per_rank * cp)), compute_pad_size, apply_padding (an empty FULL slice)
  dispatch    make_dispatch_meta_from_qk_ranges(..., cp_size=16, cp_rank=0, DispatchConfig(), permutable q / k)
  buckets     make_bucket_per_rank_from_qk_ranges(q, k, types, dispatch_meta_q)
  host_k[r]   dispatch_meta_k.host_ranges_per_rank[r].merge()
  remote_k[d] buckets[d].k_ranges.find_hole_ranges(host_k[d], is_other_merged=True)
  tok[s][d]   remote_k[d].find_overlap_ranges(host_k[s]).total_seqlen
Bytes: forward group-cast of K and V fp16 (tok * 2 * Hkv * 128 * 2 at d from s); backward the same group-cast again plus
the group-reduce of fp32 partial dK / dV back to the owner (tok * 2 * Hkv * 128 * 4 at s from d). No Q / O traffic.
Inputs : common.magi_input (W1 key ranges, W2 / W3 rollout segment tables)
Output : $DEX_RESULT_ROOT/comm_volume/recv/magi_<W>.jsonl
Example: CUDA_VISIBLE_DEVICES= python experiments/comm_volume/magi_comm.py   (Python environment with magi_attention)
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

from baselines.magi.slices import slices_from_ranges, slices_from_segments  # noqa: E402


def magi_recv_tokens(T, qr, kr, ty):
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
    dm_q, dm_k = make_dispatch_meta_from_qk_ranges(q_ranges=q, k_ranges=k, attn_mask_type=mt, total_seqlen_q=Tp,
                                                   total_seqlen_k=Tp, chunk_size=chunk, cp_size=C.P, cp_rank=0,
                                                   dispatch_config=DispatchConfig(), is_same_source=True,
                                                   is_q_permutable=True, is_k_permutable=True)
    buckets = make_bucket_per_rank_from_qk_ranges(q_ranges=q, k_ranges=k, attn_mask_type=mt, dispatch_meta=dm_q)
    host = [r.merge() for r in dm_k.host_ranges_per_rank]
    remote = [buckets[r].k_ranges.find_hole_ranges(host[r], is_other_merged=True) for r in range(C.P)]
    tok = np.zeros((C.P, C.P), dtype=np.int64)
    for d in range(C.P):
        for s in range(C.P):
            if s != d:
                tok[s, d] = remote[d].find_overlap_ranges(host[s], is_self_merged=True, is_other_merged=True).total_seqlen
        assert tok[:, d].sum() == remote[d].total_seqlen, d
    return tok, dict(chunk=chunk, pad=pad)


def main():
    C.assert_no_cuda()
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=""); a = ap.parse_args()
    only = [x for x in a.only.split(",") if x]
    for W, (hq, hk) in (("W1", (16, 16)), ("W2", (16, 16)), ("W3", (32, 8))):
        recs = []
        for name in C.static_names(W):
            if only and not any(o in name for o in only):
                continue
            src = C.magi_input(W, name)
            if "ranges" in src:
                T, qr, kr, ty = slices_from_ranges(np.load(src["ranges"]))
            else:
                T, qr, kr, ty = slices_from_segments(np.load(src["segs"])["segs"])
            tok, info = magi_recv_tokens(T, qr, kr, ty)
            kv16, dkv32 = 2 * hk * C.D * 2, 2 * hk * C.D * 4
            recv = C.empty_recv()
            for s in range(C.P):
                for d in range(C.P):
                    if tok[s, d]:
                        C.add(recv, "fwd", s, d, tok[s, d] * kv16)
                        C.add(recv, "bwd", s, d, tok[s, d] * kv16)
                        C.add(recv, "bwd", d, s, tok[s, d] * dkv32)
            recs.append(dict(system="Magi", workload=W, case=name, S=T, heads=[hq, hk], recv=recv, **info))
        if recs:
            C.write_jsonl(os.path.join(C.out_dir("recv"), f"magi_{W}.jsonl"), recs)
            print(f"[magi] {W}: {len(recs)} cases")


if __name__ == "__main__":
    main()
