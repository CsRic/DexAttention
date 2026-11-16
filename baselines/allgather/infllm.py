"""W4 (MiniCPM4.1-8B, InfLLM-v2) All-gather and Sparse all-gather, on the official InfLLM-v2 kernels (BF16).

All-gather: NCCL all-gather of K / V into the full sequence, the official selection on the gathered compressed keys,
the official InfLLM-v2 forward per own chunk; backward: all-gather K / V again, the official backward per chunk, the
chunks' partial dK / dV summed in FP32 and reduce-scattered (FP32) to their owners. The operator test times the
backward without a forward recompute (O / LSE given), as the training step's selective recompute does.
Sparse all-gather: query-stationary on the zigzag owners. Each rank all-gathers a need bitmap of the remote 64-token
blocks its tokens selected; ONE batch_isend_irecv exchange moves exactly those K | V blocks; per own chunk the keys are
compacted ([selected blocks below the chunk, ascending | the chunk's own tokens], selection remapped) and ONE official
call computes both chunks. Backward: the same exchange, the official backward on the compact arrays, FP32 dK / dV summed
per key row, the received blocks' FP32 dK / dV returned point to point and summed at their owners.
Layouts as dex.attn.infllm: q / o / do [L * 16, 2, 128] group rows; lse [2, L * 16]; k / v [L, 2, 128];
sel [2, L, K] int32 causal-filtered global block ids."""
import numpy as np
import torch
import torch.distributed as dist
from infllm_v2.infllmv2_sparse_attention import Infllmv2AttnVarlenFunc as IF
from infllm_v2.infllmv2_sparse_attention import cuda_topk_to_uint64

from workloads.w4_infllm import reuse
from workloads.w4_infllm.common import (BLK, GH, chunk_rank, filter_causal, from_rows, owned, pad_concat, select_chunk,
                                        select_cp, to_rows)
from workloads.w4_infllm.minicpm_model import SPARSE, compress_k


# ================================================================ All-gather
def gather_kv(k_own, v_own, chunks, my, S, P, c):
    """full K, V [S, 2, 128] on every rank (one all-gather of each rank's chunks, padded to the longest rank)"""
    mine = torch.stack([k_own, v_own])
    sizes = [chunks[a][1] - chunks[a][0] + chunks[b][1] - chunks[b][0] for a, b in (owned(r, P) for r in range(P))]
    pad = mine.new_zeros(2, max(sizes), c.n_kv, c.head_dim)
    pad[:, :mine.size(1)] = mine
    buf = mine.new_empty(P, *pad.shape)
    dist.all_gather_into_tensor(buf, pad)
    K = torch.empty(S, c.n_kv, c.head_dim, device=mine.device, dtype=mine.dtype); V = torch.empty_like(K)
    for r in range(P):
        o = 0
        for ci in owned(r, P):
            a_, b_ = chunks[ci]
            K[a_:b_] = buf[r, 0, o:o + b_ - a_]; V[a_:b_] = buf[r, 1, o:o + b_ - a_]; o += b_ - a_
    return K, V, sizes


def reduce_scatter(dk, dv, chunks, my, S, P, sizes, dtype=torch.float32):
    """full-length partial dK / dV [S, 2, 128] -> this rank's owned rows (sum over ranks), rank-major padded layout"""
    dev = dk.device
    buf = torch.zeros(P, 2, max(sizes), dk.size(1), dk.size(2), dtype=dtype, device=dev)
    for r in range(P):
        o = 0
        for ci in owned(r, P):
            a_, b_ = chunks[ci]
            buf[r, 0, o:o + b_ - a_] = dk[a_:b_]; buf[r, 1, o:o + b_ - a_] = dv[a_:b_]; o += b_ - a_
    out = torch.empty_like(buf[0])
    dist.reduce_scatter_tensor(out, buf)
    n = sizes[dist.get_rank()]
    return out[0, :n], out[1, :n]


def attn_chunk(q, K, V, c0, idx):
    """the official InfLLM-v2 forward of one query chunk (positions [c0, c0 + L)) over K / V [0, c0 + L)"""
    from infllm_v2 import infllmv2_attn_varlen_func
    L, end = q.size(0), c0 + q.size(0)
    dev = q.device
    cu_q = torch.tensor([0, L], dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0, end], dtype=torch.int32, device=dev)
    return infllmv2_attn_varlen_func(q.contiguous(), K[:end], V[:end], cu_q, cu_k, L, end, causal=True, topk_idx=idx)


class _Ctx:
    """stands in for the official autograd ctx: its forward / backward run unchanged on it"""

    saved_tensors = None

    def save_for_backward(self, *t):
        self.saved_tensors = t


def official_bwd(q, K, V, o, lse, cu_q, cu_k, max_q, max_k, tk, do, scale):
    """Infllmv2AttnVarlenFunc.backward given the forward's O / LSE (the forward blockmask built from the selection by the
    official topk_to_uint64)"""
    bm, _ = cuda_topk_to_uint64(tk, max_k, 64)
    fc = _Ctx()
    fc.saved_tensors = (q, K, V, o, lse, cu_q, cu_k, bm, torch.zeros(2, dtype=torch.int64, device=q.device))
    fc.dropout_p, fc.max_seqlen_q, fc.max_seqlen_k, fc.softmax_scale = 0.0, max_q, max_k, scale
    fc.causal, fc.window_size, fc.softcap, fc.alibi_slopes, fc.deterministic, fc.topk_idx = True, (-1, -1), 0.0, None, False, tk
    return IF.backward(fc, do)[:3]


class AllgatherInfllm:
    """operator-test front end of All-gather (forward given a selection, online forward, backward without recompute)"""

    def __init__(self, chunks, my, S, P, c):
        self.chunks, self.my, self.S, self.P, self.c = chunks, my, S, P, c
        self.Ls = [chunks[ci][1] - chunks[ci][0] for ci in my]

    def fwd(self, loc, sel, timer):
        """loc[ci] = (q [L, 32, D], k, v); sel[ci] [2, L, 96] -> {ci: O [L, 32 * D]}"""
        timer.start()
        k_own = torch.cat([loc[ci][1] for ci in self.my]); v_own = torch.cat([loc[ci][2] for ci in self.my])
        K, V, _ = gather_kv(k_own, v_own, self.chunks, self.my, self.S, self.P, self.c)
        timer.mark("comm", barrier=False)
        outs = {ci: attn_chunk(loc[ci][0], K, V, self.chunks[ci][0], sel[ci]) for ci in self.my}
        timer.mark("attn", barrier=False)
        timer.mark("wait")
        return {ci: o.reshape(o.size(0), -1) for ci, o in outs.items()}

    def fwd_online(self, loc_fn, timer):
        """from the attention input: projections, all-gather K / V, compressed keys and the official selection on the
        gathered keys, the official forward per chunk -> ({ci: O}, {ci: selection})"""
        timer.start()
        loc = loc_fn()
        timer.mark("local_proj")
        k_own = torch.cat([loc[ci][1] for ci in self.my]); v_own = torch.cat([loc[ci][2] for ci in self.my])
        K, V, _ = gather_kv(k_own, v_own, self.chunks, self.my, self.S, self.P, self.c)
        timer.mark("comm_allgather")
        kc = compress_k(K, SPARSE["kernel_size"], SPARSE["kernel_stride"])
        kc2 = compress_k(K, SPARSE["kernel_size"] * 4, SPARSE["kernel_stride"] * 4)
        timer.mark("compress", barrier=False)
        sel = {ci: select_chunk(loc[ci][0], kc, kc2, self.chunks[ci][0]) for ci in self.my}
        timer.mark("select", barrier=False)
        outs = {ci: attn_chunk(loc[ci][0], K, V, self.chunks[ci][0], sel[ci]) for ci in self.my}
        timer.mark("attn", barrier=False)
        timer.mark("wait")
        return {ci: o.reshape(o.size(0), -1) for ci, o in outs.items()}, sel

    def bwd(self, q_rows, o_rows, do_rows, lse, k_own, v_own, sel_f, scale, timer, recompute=False, dtype=torch.float32):
        """all-gather K / V -> per-chunk official backward -> FP32 sum + reduce-scatter of dK / dV. recompute=True: the
        official autograd (forward recomputed), the reference of the correctness checks."""
        from infllm_v2 import infllmv2_attn_varlen_func
        timer.start()
        dev, S = q_rows.device, self.S
        K, V, sizes = gather_kv(k_own, v_own, self.chunks, self.my, S, self.P, self.c)
        timer.mark("comm")
        if recompute:
            K, V = K.requires_grad_(), V.requires_grad_()
        dq = torch.empty_like(q_rows)
        dk32 = torch.zeros(S, self.c.n_kv, self.c.head_dim, dtype=torch.float32, device=dev); dv32 = torch.zeros_like(dk32)
        t0 = 0
        for ci, L in zip(self.my, self.Ls):
            c1 = self.chunks[ci][1]
            R = slice(t0 * GH, (t0 + L) * GH)
            cu_q = torch.tensor([0, L * GH], dtype=torch.int32, device=dev); cu_k = torch.tensor([0, c1], dtype=torch.int32, device=dev)
            tk = sel_f[:, t0:t0 + L].contiguous()
            if recompute:
                qr = q_rows[R].detach().requires_grad_()
                o = infllmv2_attn_varlen_func(qr, K[:c1], V[:c1], cu_q, cu_k, L * GH, c1, causal=True, topk_idx=tk)
                gq, gk, gv = torch.autograd.grad(o, (qr, K, V), do_rows[R])
                gk, gv = gk[:c1], gv[:c1]    # gradients with respect to the whole K, V; a chunk reads only its preceding keys
            else:
                gq, gk, gv = official_bwd(q_rows[R].contiguous(), K[:c1], V[:c1], o_rows[R].contiguous(), lse[:, R].contiguous(),
                                          cu_q, cu_k, L * GH, c1, tk, do_rows[R].contiguous(), scale)
            dq[R] = gq; dk32[:c1] += gk.float(); dv32[:c1] += gv.float(); t0 += L
        timer.mark("attn")
        dk, dv = reduce_scatter(dk32.to(dtype), dv32.to(dtype), self.chunks, self.my, S, self.P, sizes, dtype)
        timer.mark("reduce")
        return dq, dk, dv


class AllgatherOp:
    """All-gather in a training step: K / V all-gathered, the official selection on the gathered keys, the official
    autograd forward per own chunk (on a stand-in ctx); its saved state minus q / K / V is kept for the recompute, which
    re-gathers K / V and does not run the attention. Backward: the official backward per chunk, dK / dV summed in FP32
    and reduce-scattered to the owners."""

    def __init__(self, chunks, my, S, cfg):
        self.chunks, self.my, self.S, self.cfg = chunks, my, S, cfg
        self.Ls = [chunks[ci][1] - chunks[ci][0] for ci in my]
        self.P = dist.get_world_size()

    def __call__(self, q, k, v):
        return _AllgatherKeep.apply(q, k, v, self)


class _AllgatherKeep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, s):
        K, V, sizes = gather_kv(k.contiguous(), v.contiguous(), s.chunks, s.my, s.S, s.P, s.cfg)
        c = reuse.cache_get()
        if c is None:
            kc = compress_k(K, SPARSE["kernel_size"], SPARSE["kernel_stride"])
            kc2 = compress_k(K, SPARSE["kernel_size"] * 4, SPARSE["kernel_stride"] * 4)
            c = dict(sel={}); r = 0
            for ci, L in zip(s.my, s.Ls):
                c["sel"][ci] = select_chunk(q[r:r + L], kc, kc2, s.chunks[ci][0]).clone()
                r += L
            reuse.cache_put(c)
        hit = reuse.OL_CACHE.pop(reuse.cur_layer(), None)
        outs, fcs, r = [], [], 0
        for i, (ci, L) in enumerate(zip(s.my, s.Ls)):
            c1 = s.chunks[ci][1]
            qc = q[r:r + L].contiguous(); r += L
            if hit is not None:
                fc = hit[1][i]
                fc.saved_tensors = (qc, K[:c1], V[:c1]) + fc.kept
                outs.append(hit[0][i])
            else:
                fc = _Ctx()
                cu_q = torch.tensor([0, L], dtype=torch.int32, device=q.device)
                cu_k = torch.tensor([0, c1], dtype=torch.int32, device=q.device)
                outs.append(IF.forward(fc, qc, K[:c1], V[:c1], cu_q, cu_k, L, c1, 0.0, None, True, (-1, -1), 0.0, None, False,
                                       False, None, c["sel"][ci]))
                fc.kept = fc.saved_tensors[3:]; fc.saved_tensors = None
            fcs.append(fc)
        if hit is None:
            # non-reentrant checkpointing runs the backward of THIS forward's node; the recompute only refills the
            # stand-in ctxs' q / K / V in place
            reuse.OL_CACHE[reuse.cur_layer()] = (outs, fcs)
            ctx.fcs, ctx.Kshape, ctx.sizes = fcs, K.shape, sizes
        ctx.save_for_backward(q, k, v)                                 # (so the recompute reaches this op)
        ctx.s = s
        return torch.cat(outs, 0)

    @staticmethod
    def backward(ctx, do):
        s = ctx.s
        ctx.saved_tensors                                              # (unpacking runs the recompute: q / K / V refilled)
        assert all(fc.saved_tensors is not None for fc in ctx.fcs), "the layer must be recomputed (activation checkpointing)"
        dK = torch.zeros(ctx.Kshape, dtype=torch.float32, device=do.device); dV = torch.zeros_like(dK)
        dqs, r = [], 0
        for (ci, L), fc in zip(zip(s.my, s.Ls), ctx.fcs):
            c1 = s.chunks[ci][1]
            dq, dk, dv = IF.backward(fc, do[r:r + L].contiguous())[:3]; r += L
            dqs.append(dq); dK[:c1] += dk.float(); dV[:c1] += dv.float()
        for fc in ctx.fcs:                                             # release now: the graph (and this ctx) lives until
            fc.saved_tensors = fc.kept = None                          # the whole backward ends; kept, every layer's full
        ctx.fcs = None                                                 # K / V would stay resident
        dk, dv = reduce_scatter(dK, dV, s.chunks, s.my, s.S, s.P, ctx.sizes, torch.float32)
        return torch.cat(dqs, 0), dk.to(do.dtype), dv.to(do.dtype), None


# ================================================================ Sparse all-gather
class SparseAllgatherInfllm:
    def __init__(self, chunks, my, S, P, rank, dev):
        self.chunks, self.my, self.S, self.P, self.rank, self.dev = chunks, my, S, P, rank, dev
        self.nb = (S + BLK - 1) // BLK
        own = np.empty(self.nb, np.int64)
        for ci, (a_, b_) in enumerate(chunks):
            own[a_ // BLK:(b_ + BLK - 1) // BLK] = chunk_rank(ci, P)
        self.own = own
        self.Ls = [chunks[ci][1] - chunks[ci][0] for ci in my]
        self.L = sum(self.Ls)
        self.tok0 = [int(sum(self.Ls[:i])) for i in range(len(my))]
        orow = np.full(self.nb, -1, np.int64)                                # own block -> first local token row
        for i, ci in enumerate(my):
            a_, b_ = chunks[ci]
            n0, n1 = a_ // BLK, (b_ + BLK - 1) // BLK
            orow[n0:n1] = self.tok0[i] + np.arange(n1 - n0) * BLK
        self.orow = orow
        self.is_own = torch.as_tensor(own == rank, device=dev)
        self.ar64 = torch.arange(BLK, device=dev)

    # ------------------------------------------------------------------ need bitmaps -> send / receive lists
    def lists(self, sel):
        m = torch.zeros(self.nb, dtype=torch.uint8, device=self.dev)
        x = sel.flatten().long()
        m.scatter_reduce_(0, x.clamp(min=0), (x >= 0).to(torch.uint8), reduce="amax")
        m[self.is_own] = 0
        need = torch.empty(self.P, self.nb, dtype=torch.uint8, device=self.dev)
        dist.all_gather_into_tensor(need, m)
        nh = need.cpu().numpy().astype(bool)                                 # (the one host sync)
        r = self.rank
        send = [np.nonzero(nh[d] & (self.own == r))[0] for d in range(self.P)]
        recv = [np.nonzero(nh[r] & (self.own == s))[0] for s in range(self.P)]
        rb = np.concatenate(recv) if sum(len(x) for x in recv) else np.zeros(0, np.int64)
        crow = self.orow.copy()                                              # block -> row of the combined store
        crow[rb] = self.L + np.arange(len(rb)) * BLK                          # [own K|V rows ; received blocks]
        return dict(send=send, recv=recv, nrecv=len(rb), crow=torch.as_tensor(crow, device=self.dev),
                    bytes_in=len(rb) * BLK * 1024, bytes_out=sum(len(x) for x in send) * BLK * 1024)

    def _rows(self, blocks):
        """local token rows of own blocks (64 each; clamped inside the own rows: only the sequence's last block can be
        short and it is never sent -- only its own chunk selects it)"""
        b = torch.as_tensor(self.orow[blocks], device=self.dev)
        return (b[:, None] + self.ar64[None, :]).reshape(-1).clamp(max=self.L - 1)

    def _exchange(self, send_bufs, recv_bufs):
        ops = []
        for d, t in enumerate(send_bufs):
            if t is not None and t.numel() and d != self.rank:
                ops.append(dist.P2POp(dist.isend, t, d))
        for s, t in enumerate(recv_bufs):
            if t is not None and t.numel() and s != self.rank:
                ops.append(dist.P2POp(dist.irecv, t, s))
        if ops:
            for w in dist.batch_isend_irecv(ops):
                w.wait()

    def gather(self, Lst, kv):
        """kv [L, 2, 2, 128] (K | V of the own tokens) -> the combined store [L + nrecv * 64, 2, 2, 128]"""
        sb = [kv[self._rows(b)].contiguous() if len(b) else None for b in Lst["send"]]
        rb = [torch.empty(len(b) * BLK, *kv.shape[1:], dtype=kv.dtype, device=kv.device) if len(b) else None for b in Lst["recv"]]
        self._exchange(sb, rb)
        parts = [kv] + [t for t in rb if t is not None]
        return torch.cat(parts, 0) if len(parts) > 1 else kv

    def compact(self, Lst, store, sel):
        """-> (K, V [sum len_b, 2, 128], cu_k, max_k, remapped selection [2, L, K], per-segment (block list, rows))"""
        Ks, segs, rs = [], [], []
        cu = [0]
        for i, ci in enumerate(self.my):
            c0 = self.chunks[ci][0]; L = self.Ls[i]; t0 = self.tok0[i]; q0 = c0 // BLK
            s_ = sel[:, t0:t0 + L]
            below = (s_ >= 0) & (s_ < q0)
            Bb = torch.unique(s_[below])                                      # ascending
            nB = int(Bb.numel())
            rows = (Lst["crow"][Bb.long()][:, None] + self.ar64[None, :]).reshape(-1) if nB else torch.zeros(0, dtype=torch.long, device=self.dev)
            rows = torch.cat([rows, torch.arange(t0, t0 + L, device=self.dev)])
            Ks.append(store[rows])
            r_ = torch.where(below, torch.searchsorted(Bb, s_.contiguous()).to(torch.int32), s_ - q0 + nB)
            rs.append(torch.where(s_ >= 0, r_, torch.full_like(s_, -1)).to(torch.int32))
            segs.append((Bb, rows))
            cu.append(cu[-1] + nB * BLK + L)
        KV = torch.cat(Ks, 0)
        cu_k = torch.tensor(cu, dtype=torch.int32, device=self.dev)
        return KV[:, 0].contiguous(), KV[:, 1].contiguous(), cu_k, max(np.diff(cu)), torch.cat(rs, 1).contiguous(), segs

    def _cu_q(self):
        return torch.tensor(np.concatenate([[0], np.cumsum(self.Ls) * GH]), dtype=torch.int32, device=self.dev)

    def fwd(self, q, k, v, sel):
        """-> O [L * 16, 2, 128], LSE [2, L * 16]"""
        Lst = self.lists(sel)
        store = self.gather(Lst, torch.stack([k, v], 1))
        K, V, cu_k, mk, tk, segs = self.compact(Lst, store, sel)
        o, lse, _ = IF.forward(_Ctx(), q.contiguous(), K, V, self._cu_q(), cu_k, max(self.Ls) * GH, mk, 0.0, None, True,
                               (-1, -1), 0.0, None, False, True, None, tk)
        self.last = dict(bytes_in=Lst["bytes_in"], bytes_out=Lst["bytes_out"])
        return o, lse

    def bwd(self, q, o, do, lse, k, v, sel):
        """official backward given the forward's O / LSE -> dq (bf16), dk, dv (bf16 of the FP32 sums)"""
        Lst = self.lists(sel)
        store = self.gather(Lst, torch.stack([k, v], 1))
        K, V, cu_k, mk, tk, segs = self.compact(Lst, store, sel)
        dq, dK, dV = official_bwd(q.contiguous(), K, V, o.contiguous(), lse.contiguous(), self._cu_q(), cu_k,
                                  max(self.Ls) * GH, mk, tk, do.contiguous(), q.shape[-1] ** -0.5)
        self.last = dict(bytes_in=Lst["bytes_in"], bytes_out=Lst["bytes_out"])
        n_st = self.L + Lst["nrecv"] * BLK
        acc = torch.zeros(n_st, 2, 2, 128, dtype=torch.float32, device=self.dev)   # FP32 per store row (own | received)
        acc.index_add_(0, torch.cat([r_ for _, r_ in segs]), torch.stack([dK, dV], 1).float())
        sb, rb, off = [None] * self.P, [None] * self.P, self.L       # received blocks' FP32 dK / dV back to their owners
        for s, b in enumerate(Lst["recv"]):
            if len(b):
                sb[s] = acc[off:off + len(b) * BLK].contiguous(); off += len(b) * BLK
        for d, b in enumerate(Lst["send"]):
            if len(b):
                rb[d] = torch.empty(len(b) * BLK, 2, 2, 128, dtype=torch.float32, device=self.dev)
        self._exchange(sb, rb)
        own = acc[:self.L]
        for d, b in enumerate(Lst["send"]):
            if len(b):
                own.index_add_(0, self._rows(b), rb[d])
        return dq, own[:, 0].to(k.dtype), own[:, 1].to(k.dtype)


class SparseAllgatherOp:
    """Sparse all-gather in a training step: the selection split by owner (compressed keys exchanged); forward = need
    all-gather + one point-to-point exchange + ONE official call on the compact keys; backward = the same exchange (it is
    communication, not attention, so it also runs after the recompute) + the official backward + FP32 dK / dV returned
    to the owners."""

    def __init__(self, chunks, my, S, cfg):
        self.chunks, self.my, self.S, self.cfg = chunks, my, S, cfg
        dev = torch.device("cuda", torch.cuda.current_device())
        self.pp = SparseAllgatherInfllm(chunks, my, S, dist.get_world_size(), dist.get_rank(), dev)
        self.pos = torch.cat([torch.arange(*chunks[ci], device=dev) for ci in my]).to(torch.int32)
        self.Ls = [chunks[ci][1] - chunks[ci][0] for ci in my]

    def fwd(self, q, k, v):
        c = reuse.cache_get()
        if c is None:
            qs, ks, r = {}, {}, 0
            for ci, L in zip(self.my, self.Ls):
                qs[ci] = q[r:r + L]; ks[ci] = k[r:r + L]; r += L
            sel_f = filter_causal(pad_concat(select_cp(qs, ks, self.chunks, self.my, self.S), self.my).to(torch.int32), self.pos).clone()
            c = dict(sel=sel_f)
            reuse.cache_put(c)
        sel_f = c["sel"]
        q_rows = to_rows(q); k = k.contiguous(); v = v.contiguous()
        o_rows, lse = self.pp.fwd(q_rows, k, v, sel_f)
        return from_rows(o_rows, self.cfg.n_heads).contiguous(), (q_rows, k, v, o_rows.contiguous(), lse.contiguous(), sel_f)

    def bwd(self, saved, do, aux=None):
        q_rows, k, v, o_rows, lse, sel_f = saved
        dq, dk, dv = self.pp.bwd(q_rows, o_rows, to_rows(do), lse, k, v, sel_f)
        return from_rows(dq, self.cfg.n_heads).contiguous(), dk, dv

    def __call__(self, q, k, v):
        return reuse.CPAttn.apply(q, k, v, self)
