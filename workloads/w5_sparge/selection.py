"""SpargeAttention2 block selection (Algorithm 1 of the SpargeAttention2 report), the training setting used by W5.

Q is mean-pooled per 128-row block and K per 64-key block; block scores P = softmax(pool(Q) pool(K)^T / sqrt(d)) per query
block row; the mask is Top-k(P, k) U Top-p(P, p):
  * Top-k: the int(k * nk) largest key blocks of the row (at least 1)
  * Top-p: the smallest prefix of the row's blocks sorted by score whose cumulative probability reaches p
No self-similarity filter and no key smoothing (neither is part of Algorithm 1). Wan2.1-1.3B: k = 0.03, p = 0.2.

  pooled_q(q)            -> [H, nq, D] fp32 (mean over the block's real rows)
  pooled_k(k)            -> [H, nk, D] fp32
  select(pq, pk, scale)  -> mask [H, nq, nk] bool (128 x 64 blocks)
  select_split(q, k, lay) -> this rank's mask rows, with the pooled keys assembled by one all-reduce
The pooled keys are per-block means, so a context-parallel rank pools its own key blocks and assembles the others' with
one all-reduce, exactly as the single-GPU computation."""

import numpy as np
import torch
import torch.distributed as dist

TOPK, TOPP = 0.03, 0.2
BLKQ, BLKK = 128, 64


def _pool(x, blk):
    H, L, D = x.shape
    nb = (L + blk - 1) // blk
    xp = torch.nn.functional.pad(x.float(), (0, 0, 0, nb * blk - L))
    cnt = torch.full((nb,), float(blk), device=x.device)
    cnt[-1] = L - (nb - 1) * blk
    return xp.reshape(H, nb, blk, D).sum(2) / cnt[None, :, None]


def pooled_q(q):
    return _pool(q, BLKQ)


def pooled_k(k):
    return _pool(k, BLKK)


def select(pq, pk, scale, topk=TOPK, topp=TOPP):
    """pq [H, nq, D], pk [H, nk, D] (fp32) -> [H, nq, nk] bool"""
    nk = pk.shape[1]
    P = torch.softmax(pq @ pk.transpose(1, 2) * scale, -1)
    srt, idx = torch.sort(P, -1, descending=True)
    nkeep = max(1, int(topk * nk))
    cdf = torch.cumsum(srt, -1)
    ptop = (cdf < topp).sum(-1, keepdim=True) + 1
    keep_sorted = torch.arange(nk, device=P.device)[None, None, :] < torch.clamp(
        torch.maximum(ptop, torch.full_like(ptop, nkeep)), max=nk
    )
    m = torch.zeros_like(P, dtype=torch.bool)
    m.scatter_(-1, idx, keep_sorted)
    return m


def key_blocks_of(lay, r):
    """global 64-key block ids held by rank r (its chunk order)"""
    return np.concatenate(
        [np.arange(a // BLKK, (b + BLKK - 1) // BLKK) for a, b in (lay.chunks[c] for c in lay.owned(r))]
    )


def select_split(q, k, lay):
    """the selection split by owner (Ring, Sparse all-gather, Dex): every rank pools its own 64-key blocks, one
    all-reduce assembles the pooled keys (each block written by its owner only, so the sum is exact), then the scores and
    the selection of this rank's 128-row query blocks. q / k [H, L, D] -> [H, nqb_local, nkb] bool"""
    H, dev = q.shape[0], q.device
    kb_me = torch.from_numpy(key_blocks_of(lay, lay.rank)).to(dev)
    pk = torch.zeros(H, lay.nkb, q.shape[2], dtype=torch.float32, device=dev)
    pk[:, kb_me] = pooled_k(k)
    dist.all_reduce(pk)
    return select(pooled_q(q), pk, q.shape[-1] ** -0.5)
