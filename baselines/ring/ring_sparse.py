"""Ring: every rank's K / V circulate through all ranks (P - 1 steps, whatever the mask), so the communication stays
dense; the merge of per-step partial outputs is upstream's fp32 LSE merge. Communication and merge follow upstream
Yunchang's ring_flash_attn_forward / ring_flash_attn_backward (third_party/yunchang, yunchang/ring/ring_flash_attn.py);
the per-step kernel call is replaced by a compute object, so each step computes the mask's live part of the (this rank's
queries, visiting chunk) pair and skips a pair with nothing live, and receive buffers are reused across steps.
Token placement: zigzag (chunks r and 2P-1-r of 2P equal chunks).

A compute object provides
  fwd(src, q, k, v) -> (block_out [1, s, h, d], block_lse [1, h, s] fp32) or None (nothing live); rows with no live key
      in this pair carry a finite, very negative LSE (LSE_EMPTY: weight 0 in the merge, no fp32 cancellation)
  bwd(src, do, q, k, v, o, lse, delta, dq_acc, dk_blk, dv_blk) -> bool (computed); adds this pair's dQ into dq_acc and
      writes dK / dV into the zeroed fp32 dk_blk / dv_blk."""
import os
import sys

import numpy as np
import torch

from dex.common.env import repo_root

_YC = os.path.join(repo_root(), "third_party", "yunchang")
if _YC not in sys.path:
    sys.path.insert(0, _YC)
from yunchang.ring.utils import RingComm, update_out_and_lse  # noqa: E402

LSE_EMPTY = -1e4


def zigzag_tiles(n_tiles, P, r):
    """global tile ids held by rank r: zigzag chunks r and 2P-1-r of 2P equal chunks"""
    c = n_tiles // (2 * P)
    assert c * 2 * P == n_tiles, (n_tiles, P)
    return list(range(r * c, (r + 1) * c)) + list(range((2 * P - 1 - r) * c, (2 * P - r) * c))


def _send_recv(comm, x, into=None):
    """comm.send_recv on a tensor or on each tensor of a tuple. `into`: upstream's recv_tensor argument (a reused
    receive buffer of the same shape)."""
    if isinstance(x, tuple):
        return tuple(comm.send_recv(t, None if into is None else r) for t, r in zip(x, into or [None] * len(x)))
    return comm.send_recv(x, into)


def _like(x):
    return tuple(torch.empty_like(t) for t in x) if isinstance(x, tuple) else torch.empty_like(x)


# Receive buffers: two ping-pong buffers per tensor per call (upstream's recv_tensor argument), so a long training step
# does not grow the allocator's reserved pool with one buffer per step; the step k + 1 receive reuses the buffer step
# k - 1 received into, whose send and compute finished before step k's wait.
def ring_forward(group, q, k, v, comp):
    """upstream ring_flash_attn_forward with comp.fwd as the per-step kernel -> out (q.dtype), lse [1, h, s] fp32"""
    comm = RingComm(group)
    out = None
    lse = None
    next_k, next_v = None, None
    rk = rv = None
    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            if rk is None:
                rk, rv = [_like(k), _like(k)], [_like(v), _like(v)]
            next_k = _send_recv(comm, k, rk[step % 2])
            next_v = _send_recv(comm, v, rv[step % 2])
            comm.commit()
        r = comp.fwd((comm.rank - step) % comm.world_size, q, k, v)
        if r is not None:
            out, lse = update_out_and_lse(out, lse, r[0], r[1])
        if step + 1 != comm.world_size:
            comm.wait()
            k = next_k
            v = next_v
    if out is None:                     # a rank whose queries see nothing at all
        out = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
        lse = torch.full((q.shape[0], q.shape[1], q.shape[2], 1), -1e4, dtype=torch.float32, device=q.device)
    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse.contiguous()


def ring_backward(group, do, q, k, v, o, lse, delta, comp):
    """upstream ring_flash_attn_backward with comp.bwd as the per-step kernel -> dq_acc (fp32, in whatever scale the
    compute object's kernel produces), dk, dv (fp32, this rank's own chunk)"""
    kv_comm = RingComm(group)
    d_kv_comm = RingComm(group)
    dq = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    dk, dv = None, None
    next_dk, next_dv = None, None
    next_k, next_v = None, None
    W = kv_comm.world_size
    rk, rv = ([_like(k), _like(k)], [_like(v), _like(v)]) if W > 1 else (None, None)
    blk_k = [torch.empty(k.shape, dtype=torch.float32, device=k.device) for _ in range(2)]   # block dK / dV (ping-pong:
    blk_v = [torch.empty(v.shape, dtype=torch.float32, device=v.device) for _ in range(2)]   # one may still be in flight)
    rdk = [torch.empty_like(blk_k[0]) for _ in range(2)]; rdv = [torch.empty_like(blk_v[0]) for _ in range(2)]
    for step in range(W):
        if step + 1 != W:
            next_k = kv_comm.send_recv(k, rk[step % 2])
            next_v = kv_comm.send_recv(v, rv[step % 2])
            kv_comm.commit()
        block_dk = blk_k[step % 2].zero_()
        block_dv = blk_v[step % 2].zero_()
        done = comp.bwd((kv_comm.rank - step) % W, do, q, k, v, o, lse, delta, dq, block_dk, block_dv)
        if step == 0:
            dk, dv = block_dk, block_dv               # computed or not: zeros start the accumulators
        elif done:
            d_kv_comm.wait()
            dk = block_dk.add_(next_dk)
            dv = block_dv.add_(next_dv)
        else:
            d_kv_comm.wait()
            dk = next_dk
            dv = next_dv
        if step + 1 != W:
            kv_comm.wait()
            k = next_k
            v = next_v
        next_dk = d_kv_comm.send_recv(dk, rdk[step % 2])
        next_dv = d_kv_comm.send_recv(dv, rdv[step % 2])
        d_kv_comm.commit()
    d_kv_comm.wait()
    return dq, next_dk, next_dv


# ---- block-sparse tile masks: the kernels All-gather and Dex's workers use ----------------------------------------
class TileComp:
    """per-step compute = C.sparse_fwd / C.sparse_bwd on the live tiles of (this rank's zigzag rows) x (the source
    rank's zigzag columns). m: tile mask [n, n] (0 dead, 1 full, 2 causal diagonal)."""

    def __init__(self, C, m, P, rank, hq, hk, scale, dev):
        from baselines.allgather.static import csr_rows, items_for
        self.C, self.hq, self.hk, self.scale = C, hq, hk, scale
        n = m.shape[0]
        self.per = [np.array(zigzag_tiles(n, P, r)) for r in range(P)]
        rows = self.per[rank]
        self.ft, self.bt = {}, {}
        for src in range(P):
            cols = self.per[src]
            if not m[np.ix_(rows, cols)].any():
                continue
            slot = np.zeros(n, np.int64)
            slot[cols] = np.arange(len(cols))
            mm = np.zeros_like(m)
            mm[np.ix_(rows, cols)] = m[np.ix_(rows, cols)]
            self.ft[src] = csr_rows(mm, rows, slot, hq, dev, shift=0)
            self.bt[src] = items_for(mm, rows, np.arange(len(rows)), cols, np.arange(len(cols)), hq, hk)

    def fwd(self, src, q, k, v):
        if src not in self.ft:
            return None
        out = torch.zeros(q.shape, device=q.device, dtype=q.dtype)
        lse = torch.full((1, self.hq, q.shape[1]), LSE_EMPTY, device=q.device, dtype=torch.float32)
        self.C.sparse_fwd(q.contiguous(), k.contiguous(), v.contiguous(), out, lse, *self.ft[src], self.scale)
        return out, lse

    def bwd(self, src, do, q, k, v, o, lse, delta, dq_acc, dk_blk, dv_blk):
        if src not in self.bt or not self.bt[src][1]:
            return False
        tasks, _ = self.bt[src]
        self.C.sparse_bwd(q[0], k[0], v[0], do[0], lse[0], delta, dq_acc[0], None, None, dk_blk[0], dv_blk[0], *tasks, self.scale)
        return True


def head_csr(sub, ns):
    """sparse_fwd tables for per-head tile masks over 'virtual' tiles: query tile (h, i) -> h * nl + i (q / out / lse
    viewed [1, H * nl * 128, 1, D]), key tile (h, j) -> slot h * ns + j (K / V viewed [1, H * ns * 128, 1, D]).
    sub [H, nl, ns] bool. Non-causal: every live tile full (nmask 0). Built on the GPU (csr_rows' format)."""
    H, nl, _ = sub.shape
    hh, ii, jj = torch.nonzero(sub, as_tuple=True)                     # sorted by (h, i), then j
    row = hh * nl + ii
    cnt = torch.bincount(row, minlength=H * nl)
    indptr = torch.zeros(H * nl + 1, dtype=torch.int64, device=sub.device); indptr[1:] = torch.cumsum(cnt, 0)
    i32 = lambda x: (x if x.numel() else torch.zeros(1, dtype=torch.int64, device=sub.device)).to(torch.int32).contiguous()   # noqa: E731
    order = torch.argsort(-cnt, stable=True)
    work = order[cnt[order] > 0]
    return (i32(indptr), i32(hh * ns + jj), i32(jj), i32(torch.zeros(H * nl, dtype=torch.int64, device=sub.device)),
            i32(torch.arange(H * nl, device=sub.device)), i32(work))


def head_bwd_tasks(sub, half):
    """sparse_bwd task lists (64 x 64 units) for per-head masks over virtual tiles. sub [H, nl, ns] (128 x 128
    tiles; both 64-key halves of a live tile -- the union) or, half=True, [H, nl, 2 ns] (column 2 j + hn = key tile j,
    64-key half hn: SpargeAttention2's 128 x 64 blocks exactly). Task = (key tile h * ns + j, half hn), items = the query
    tiles h * nl + i with both 64-row halves (mb 1, 0), non-causal. -> (task_kv, task_ptr, items) int32"""
    H, nl, w = sub.shape
    ns = w // 2 if half else w
    tk, its, cnts = [], [], []
    for hn in (0, 1):
        s_ = sub[:, :, hn::2] if half else sub
        hh, jj, ii = torch.nonzero(s_.permute(0, 2, 1), as_tuple=True)    # sorted by (h, j), then i
        upair, cnt = torch.unique_consecutive(hh * ns + jj, return_counts=True)
        qslot = (hh * nl + ii).to(torch.int32)
        its.append(torch.stack([(qslot << 9) | (1 << 8), qslot << 9], 1).reshape(-1))
        tk.append((upair.to(torch.int32) << 9) | (hn << 8)); cnts.append(2 * cnt)
    cnt = torch.cat(cnts)
    ptr = torch.zeros(len(cnt) + 1, dtype=torch.int64, device=sub.device); ptr[1:] = torch.cumsum(cnt, 0)
    return torch.cat(tk).contiguous(), ptr.to(torch.int32).contiguous(), torch.cat(its).contiguous()


class HeadTileComp:
    """per-step backward = the FA2 block-sparse backward (sparse_bwd) on per-head 128 x 128 tile masks, over 'virtual'
    tiles (head h, tile t) -> h * n + t, exactly as the All-gather backward builds its task lists (per call).
    Layout: q / do / o / dq [H, L, D], lse / delta [1, H*L]; K / V of the visiting rank [H, L, D] (same tile count per
    rank). mrows [H, nl, n_pad] bool (this rank's tiles x all tiles); tiles_of(r): rank r's global tile ids (its order)."""

    def __init__(self, DA, mrows, tiles_of, scale, T=128, half=False):
        """half=True: mrows [H, nl, 2 n_pad] over 64-key half tiles (128 x 64: sparse_fwd64, per-half bwd tasks)"""
        self.DA, self.mrows, self.tiles_of, self.scale, self.T, self.half = DA, mrows, tiles_of, scale, T, half

    def _cols(self, src, dev):
        c = torch.as_tensor(self.tiles_of(src), device=dev)
        return (2 * c[:, None] + torch.arange(2, device=dev)).reshape(-1) if self.half else c

    def fwd(self, src, q, k, v):
        """sparse_fwd on (this rank's tiles x src's tiles), per-head masks as virtual
        tiles: q / k / v [1, H * L, 1, D] (head-major, the rank's tile order). -> (out [1, H L, 1, D], lse [1, 1, H L])"""
        sub = self.mrows[:, :, self._cols(src, q.device)]
        if not bool(sub.any()):
            return None
        out = torch.zeros(q.shape, device=q.device, dtype=q.dtype)
        lse = torch.full((1, 1, q.shape[1]), LSE_EMPTY, device=q.device, dtype=torch.float32)
        f = self.DA.sparse_fwd64 if self.half else self.DA.sparse_fwd
        f(q.contiguous(), k.contiguous(), v.contiguous(), out, lse, *head_csr(sub, sub.shape[2]), self.scale)
        return out, lse

    def bwd(self, src, do, q, k, v, o, lse, delta, dq_acc, dk_blk, dv_blk):
        H, L, D = q.shape
        sub = self.mrows[:, :, self._cols(src, q.device)]                 # [H, nl, nl_src (x 2)]
        if not bool(sub.any()):
            return False
        task_kv, task_ptr, items = head_bwd_tasks(sub, self.half)
        v1 = lambda x: x.reshape(H * x.shape[1], 1, D)                                 # noqa: E731
        self.DA.sparse_bwd(v1(q), v1(k), v1(v), v1(do), lse, delta, v1(dq_acc), None, None, v1(dk_blk), v1(dv_blk),
                           task_kv, task_ptr, items, self.scale)
        return True
