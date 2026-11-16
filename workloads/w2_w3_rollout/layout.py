"""Token layout and tile mask of a packed rollout group.

A group = one prompt (P tokens) followed by its replies, each padded to a multiple of 128 tokens so that every
128 x 128 tile is full, causal-diagonal or empty. A reply token sees the whole prompt and its own reply causally; a
prompt token sees the prompt causally. The sequence ends with an inert tail segment (attends only itself) that pads
the total to a multiple of the plan block, so every plan block has the same token count.

Segments are rows [start, len, kind, group, parent, real_len]; kind 0 = prompt, 1 = reply, 2 = inert tail.
Pad tokens never take part in attention: their K / V are zero and the forward output is corrected exactly
(dex.common.pad_correct, which also provides `token_valid` / `pad_keys_seen`, the per-token data of the correction).
"""
import numpy as np

from dex.common.pad_correct import ancestors, pad_keys_seen, token_valid  # noqa: F401

TILE = 128


def up(x, m):
    return -(-x // m) * m


def layout(groups, block):
    """groups: dicts with P and R (reply lengths); block: plan block in tokens. -> (segments, total tokens)"""
    segs, pos = [], 0
    for g, grp in enumerate(groups):
        pi = len(segs)
        L = up(grp["P"], TILE)
        segs.append([pos, L, 0, g, -1, grp["P"]])
        pos += L
        for r in grp["R"]:
            L = up(r, TILE)
            segs.append([pos, L, 1, g, pi, r])
            pos += L
    T = up(pos, block)
    if T > pos:
        segs.append([pos, T - pos, 2, -1, -1, 0])
    return segs, T



def tile_mask(segs, T):
    """[nt, nt] uint8 at tile granularity: 1 = full, 2 = causal diagonal, 0 = empty"""
    nt = T // TILE
    m = np.zeros((nt, nt), np.uint8)
    for i, (s, L, kind, g, par, _) in enumerate(segs):
        q0, q1 = s // TILE, (s + L) // TILE
        for a in ancestors(segs, i):
            ps, pl = segs[a][0], segs[a][1]
            m[q0:q1, ps // TILE:(ps + pl) // TILE] = 1
        own = np.tril(np.ones((q1 - q0, q1 - q0), np.uint8))
        np.fill_diagonal(own, 2)
        m[q0:q1, q0:q1] = own
    return m



def star_ranges(P, R):
    """[S, 2, 2] int32 key ranges on the UNPADDED token sequence (prompt, then the replies): prompt tokens see the
    prompt causally; a reply token sees the whole prompt (range 0) and its reply causally (range 1)"""
    S = P + sum(R)
    r = np.zeros((S, 2, 2), np.int32)
    r[:P, 0, 1] = np.arange(P) + 1
    pos = P
    for R_ in R:
        j = np.arange(R_)
        r[pos:pos + R_, 0, 1] = P
        r[pos:pos + R_, 1, 0] = pos
        r[pos:pos + R_, 1, 1] = pos + j + 1
        pos += R_
    return r
