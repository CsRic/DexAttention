"""Exact token-level Magi slices of a mask: q ranges, k ranges and mask types. No torch.
rollout segment table [start, len, kind, group, parent, real_len]: each real segment attends every ancestor FULL + itself CAUSAL.
per-token key ranges [T, R, 2] (half-open): run-length grouped (start const / end +1 = CAUSAL, both +1 = BICAUSAL, both
const = FULL, start +1 / end const = INVCAUSAL)."""
import numpy as np


def slices_from_segments(segs):
    """rollout segment table -> (T, q_ranges, k_ranges, types) over the compacted real tokens"""
    real = [(i, s) for i, s in enumerate(segs) if s[2] != 2 and s[5] > 0]
    start, pos = {}, 0
    for i, s in real:
        start[i] = pos
        pos += int(s[5])
    qr, kr, ty = [], [], []
    for i, s in real:
        q0, q1 = start[i], start[i] + int(s[5])
        a = int(s[4])
        anc = []
        while a >= 0:
            anc.append(a)
            a = int(segs[a][4])
        for a in reversed(anc):
            if a in start:
                qr.append([q0, q1])
                kr.append([start[a], start[a] + int(segs[a][5])])
                ty.append("full")
        qr.append([q0, q1])
        kr.append([q0, q1])
        ty.append("causal")
    return pos, qr, kr, ty


def slices_from_padded_segments(segs):
    """the same segments on the padded tile layout (segments padded to 128 tokens, inert tail included): the tile mask exactly"""
    qr, kr, ty = [], [], []
    for s_ in segs:
        anc, a_ = [], int(s_[4])
        while a_ >= 0:
            anc.append(a_)
            a_ = int(segs[a_][4])
        for a_ in reversed(anc):
            qr.append([int(s_[0]), int(s_[0] + s_[1])])
            kr.append([int(segs[a_][0]), int(segs[a_][0] + segs[a_][1])])
            ty.append("full")
        qr.append([int(s_[0]), int(s_[0] + s_[1])])
        kr.append([int(s_[0]), int(s_[0] + s_[1])])
        ty.append("causal")
    return qr, kr, ty


def slices_from_ranges(R):
    """per-token key ranges [T, nr, 2] (half-open, empty = start == end) -> (T, q_ranges, k_ranges, types), exact"""
    T, nr, _ = R.shape
    qr, kr, ty = [], [], []
    for j in range(nr):
        s, e = R[:, j, 0].astype(np.int64), R[:, j, 1].astype(np.int64)
        q = 0
        while q < T:
            if s[q] >= e[q]:
                q += 1
                continue
            ds = 0 if q + 1 >= T or s[q + 1] >= e[q + 1] else s[q + 1] - s[q]
            de = 0 if q + 1 >= T or s[q + 1] >= e[q + 1] else e[q + 1] - e[q]
            if (ds, de) not in ((0, 0), (0, 1), (1, 1), (1, 0)):
                ds, de = 0, 0                           # irregular step: a one-row FULL slice
            r = q + 1
            while r < T and s[r] < e[r] and s[r] - s[r - 1] == ds and e[r] - e[r - 1] == de:
                r += 1
            n = r - q
            qr.append([q, r])
            kr.append([int(s[q]), int(e[r - 1]) if de else int(e[q])])
            ty.append({(0, 0): "full", (0, 1): "causal", (1, 1): "bicausal", (1, 0): "invcausal"}[(ds, de)])
            if n == 1:
                ty[-1] = "full"
            q = r
    return T, qr, kr, ty
