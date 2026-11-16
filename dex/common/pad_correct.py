"""Rollout padding: segments are stored padded to 128-token tiles; the pad tokens take no part in attention, for every
system, and no system's compute kernel changes.

Exact construction, the same for all systems:
  1. the pad tokens' K and V are zero  -> a pad key scores exactly 0 for every query (q . 0 = 0) and adds nothing to
     the numerator (V = 0); it only adds exp(0) = 1 per visible pad key to the softmax denominator;
  2. the number of pad keys a real query row sees, n_pad(row), is known from the layout: every ancestor segment is
     seen in full (its pad tail included), the row's own segment only causally (a real row never reaches its own pad
     tail, which sits after every real token of the segment);
  3. after the forward (after every merge, so the partial-O path of each system is untouched):
        LSE_true = LSE + log1p(-n_pad * exp(-LSE)),   O_true = O * exp(LSE - LSE_true)
     (merging partials adds their denominators, so the pad terms of all partials add up to n_pad(row) as well).
  Backward with the true O / LSE is exact without a correction: P of a pad key multiplies K_pad = 0 in dQ and its dK /
  dV rows are discarded; real dK / dV only see real P = exp(s - LSE_true).
Pad query rows (and the inert tail segment) are not outputs: their rows are excluded from every error metric.
Limit: the correction subtracts n_pad from the denominator; it loses precision only if the real denominator is below
~1e-4 * n_pad, i.e. every real score of a row is far below 0 (not the case for softmax-scaled attention scores of
real models; checked per run by the error against the token-exact reference).
segs: rollout layout rows [start, len, kind, group, parent, real_len] (start / len 128-aligned)."""
import numpy as np

TILE = 128


def ancestors(segs, i):
    out, p = [], segs[i][4]
    while p >= 0:
        out.append(p); p = segs[p][4]
    return out


def token_valid(segs, T):
    """[T] bool: real tokens (kind 0 / 1, position < real_len inside the segment)"""
    v = np.zeros(T, bool)
    for s, L, kind, g, par, real in segs:
        if kind != 2:
            v[s:s + real] = True
    return v


def pad_keys_seen(segs, T):
    """[T] int32: pad keys visible to each real query token (0 for pad / inert tokens: they are not outputs)"""
    pad = np.zeros(T, np.int32)
    for i, (s, L, kind, g, par, real) in enumerate(segs):
        if kind == 2:
            continue
        pad[s:s + real] = sum(segs[a][1] - segs[a][5] for a in ancestors(segs, i))
    return pad


def correct(O, LSE, npad):
    """O [T, H, D] (any float dtype), LSE [H, T] fp32 natural log, npad [T] int (torch tensors on one device), in
    place; returns (O, LSE). Rows with npad = 0 are unchanged."""
    import torch
    n = npad.to(LSE.dtype)[None, :]
    lt = LSE + torch.log1p(-n * torch.exp(-LSE))
    f = torch.exp(LSE - lt).t()[:, :, None]                 # [T, H, 1]
    O.copy_((O.float() * f).to(O.dtype)); LSE.copy_(lt)
    return O, LSE


def zero_pads(x, valid):
    """K or V [T, H, D]: pad tokens -> 0 (in place); valid [T] bool torch tensor"""
    x[~valid] = 0
    return x
