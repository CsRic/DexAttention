"""W4 (MiniCPM4.1-8B, InfLLM-v2) Ring: upstream Yunchang's ring communication and FP32 merge (baselines.ring.ring_sparse:
every K / V chunk visits every rank, so the communication is dense), with the official InfLLM-v2 kernels computing,
per ring step, each (own query chunk, visiting key chunk) pair on the selected 64-token blocks inside that key chunk
(re-indexed to it; causal only on the query's own chunk; a later chunk is never selected). Zigzag placement.
Selection: in the operator test's online forward Ring circulates the compressed keys (ring_stage1); in the training step
it uses the same exchange of compressed keys as every other system (workloads.w4_infllm.common.select_cp).
Layouts: q / o / dq [1, N, 2, 128] group rows (N = sum of own chunk lengths x 16); lse [1, 2, N]; k / v [1, Lmax, 2, 128]
= the owner's chunks in owned() order, zero padded to the longest rank."""
import torch
import torch.distributed as dist

from baselines.ring.ring_sparse import ring_backward, ring_forward
from workloads.w4_infllm import reuse
from workloads.w4_infllm.common import (GH, filter_causal, from_rows, halo_allgather, key_counts, kinds, own_windows,
                                        owned, pad_concat, pick_blocks, select_cp, to_rows, windows)
from workloads.w4_infllm.stage1_fast import Stage1Seg

LSE_EMPTY = -1e4               # rows with no live key in a pair: finite, so their merge weight is 0 without NaN


class InfllmComp:
    """per-step compute = the official InfLLM-v2 kernels (infllm_v2.C.varlen_fwd / varlen_bwd, the routines
    infllmv2_attn_varlen_func wraps) per (own query chunk, visiting key chunk) pair.
      chunks: [(c0, c1)] of the 2P zigzag chunks; owned(r) -> the rank's chunk ids; sel: {own chunk: [Hk, L, K] int32
      global block ids (-1 none), causally filtered}."""

    def __init__(self, chunks, owned_fn, rank, P, sel, scale, block=64):
        import infllm_v2.C as IC
        from infllm_v2.blockmask_to_uint64 import blockmask_to_uint64
        from infllm_v2.topk_to_uint64 import topk_to_uint64
        from infllm_v2.uint64_to_bool import uint64_to_bool
        self.IC, self.t2u, self.u2b, self.b2u = IC, topk_to_uint64, uint64_to_bool, blockmask_to_uint64
        self.chunks, self.owned, self.sel, self.scale, self.blk = chunks, owned_fn, sel, scale, block
        self.my = owned_fn(rank)
        self.qoff, o = {}, 0
        for ci in self.my:
            self.qoff[ci] = o; o += (chunks[ci][1] - chunks[ci][0]) * GH
        self.N = o
        self._fw, self._bw, self._nb = {}, {}, {}      # per own chunk: {key chunk: fwd row mask / bwd column mask}

    def _pairs(self, src):
        """(own chunk ci, its row range, key chunk kj, its row range in the src buffer, causal)"""
        out, koff = [], 0
        for kj in self.owned(src):
            k0, k1 = self.chunks[kj]
            for ci in self.my:
                if k0 <= self.chunks[ci][0]:
                    L = self.chunks[ci][1] - self.chunks[ci][0]
                    out.append((ci, self.qoff[ci], L, kj, koff, k1 - k0, kj == ci))
            koff += k1 - k0
        return out

    def _prep(self, ci):
        """every key chunk's local selection of own chunk ci at once: one batched remap and one topk_to_uint64 ->
        {kj: [Hk, L, words]} (the official conversion sets one bit per entry and skips -1 wherever it is)"""
        if ci in self._fw:
            return
        c0 = self.chunks[ci][0]
        kjs = [kj for kj in range(len(self.chunks)) if self.chunks[kj][0] <= c0]
        Lks = [self.chunks[kj][1] - self.chunks[kj][0] for kj in kjs]
        nbs = [(Lk + self.blk - 1) // self.blk for Lk in Lks]
        idx = self.sel[ci]
        dev = idx.device
        b0 = torch.tensor([self.chunks[kj][0] // self.blk for kj in kjs], dtype=torch.int32, device=dev)[:, None, None, None]
        nb = torch.tensor(nbs, dtype=torch.int32, device=dev)[:, None, None, None]
        rel = idx[None].to(torch.int32) - b0
        li = torch.where((rel >= 0) & (rel < nb), rel, torch.full_like(rel, -1)).contiguous()
        if len({(n + 63) // 64 for n in nbs}) == 1:                    # one word count: one batched conversion
            U = self.t2u(li, max(Lks), self.blk)[0]
            self._fw[ci] = {kj: U[i] for i, kj in enumerate(kjs)}
        else:
            self._fw[ci] = {kj: self.t2u(li[i].contiguous(), Lks[i], self.blk)[0] for i, kj in enumerate(kjs)}
        self._nb.update({(ci, kj): n for kj, n in zip(kjs, nbs)})

    def _idx(self, ci, kj):
        self._prep(ci)
        return self._fw[ci][kj]

    def _bwd_mask(self, ci, kj):
        """the backward's column mask, as the official autograd builds it: forward row mask -> bool -> transpose ->
        uint64 (per key block: the query rows that selected it); all key chunks of ci in one batch"""
        if ci not in self._bw:
            self._prep(ci)
            kjs = list(self._fw[ci]); nbmax = max(self._nb[(ci, kj)] for kj in kjs)
            U = torch.stack([self._fw[ci][kj] for kj in kjs]) if len({self._fw[ci][kj].shape for kj in kjs}) == 1 else None
            if U is not None:
                Bm = self.u2b(U, nbmax).transpose(-1, -2).contiguous()                 # [n, Hk, nbmax, L]
                M = self.b2u(Bm)[0]                                                   # [n, Hk, nbmax, words(L)]
                self._bw[ci] = {kj: M[i, :, :self._nb[(ci, kj)]].contiguous() for i, kj in enumerate(kjs)}
            else:
                self._bw[ci] = {kj: self.b2u(self.u2b(self._fw[ci][kj], self._nb[(ci, kj)]).transpose(1, 2).contiguous())[0]
                                for kj in kjs}
        return self._bw[ci][kj]

    def fwd(self, src, q, k, v):
        from yunchang.ring.utils import update_out_and_lse
        pairs = self._pairs(src)
        if not pairs:
            return None
        dev = q.device
        out = torch.zeros(q.shape, device=dev, dtype=torch.float32)
        lse = torch.full((1, q.shape[2], q.shape[1]), LSE_EMPTY, device=dev, dtype=torch.float32)
        for ci, r0, L, kj, k0, Lk, causal in pairs:
            n = L * GH
            cu_q = torch.tensor([0, n], dtype=torch.int32, device=dev); cu_k = torch.tensor([0, Lk], dtype=torch.int32, device=dev)
            ob, lb, _, _ = self.IC.varlen_fwd(q[0, r0:r0 + n], k[0, k0:k0 + Lk], v[0, k0:k0 + Lk], None, cu_q, cu_k, None, None,
                                              None, None, n, Lk, 0.0, self.scale, False, causal, -1, -1, 0.0, False, None,
                                              self._idx(ci, kj))
            lb = torch.where(torch.isinf(lb), torch.full_like(lb, LSE_EMPTY), lb)      # rows that selected nothing here
            o_, l_ = update_out_and_lse(out[:, r0:r0 + n], lse[:, :, r0:r0 + n].transpose(1, 2).unsqueeze(-1), ob[None], lb[None])
            out[:, r0:r0 + n] = o_; lse[:, :, r0:r0 + n] = l_.squeeze(-1).transpose(1, 2)
        return out.to(q.dtype), lse

    def bwd(self, src, do, q, k, v, o, lse, delta, dq_acc, dk_blk, dv_blk):
        pairs = self._pairs(src)
        if not pairs:
            return False
        dev = q.device
        for ci, r0, L, kj, k0, Lk, causal in pairs:
            n = L * GH
            cu_q = torch.tensor([0, n], dtype=torch.int32, device=dev); cu_k = torch.tensor([0, Lk], dtype=torch.int32, device=dev)
            dq_t = torch.empty_like(q[0, r0:r0 + n])
            dk_t = torch.zeros_like(k[0, k0:k0 + Lk]); dv_t = torch.zeros_like(v[0, k0:k0 + Lk])
            self.IC.varlen_bwd(do[0, r0:r0 + n].contiguous(), q[0, r0:r0 + n].contiguous(), k[0, k0:k0 + Lk].contiguous(),
                               v[0, k0:k0 + Lk].contiguous(), o[0, r0:r0 + n].contiguous(), lse[0, :, r0:r0 + n].contiguous(),
                               dq_t, dk_t, dv_t, cu_q, cu_k, None, n, Lk, 0.0, self.scale, False, causal, -1, -1, 0.0, False,
                               self._bwd_mask(ci, kj), None, None)
            dq_acc[0, r0:r0 + n] += dq_t.float(); dk_blk[0, k0:k0 + Lk] += dk_t.float(); dv_blk[0, k0:k0 + Lk] += dv_t.float()
        return True


def ring_stage1(qs, k_own, chunks, my, S):
    """Ring's own selection: every rank's compressed-key windows rotate P-1 times, twice -- first the coarse keys (online
    per-row max / sum = the LSE), then the fine keys (score columns with the final LSE); each step overlaps the NCCL
    send / recv of the next shard with the segment kernels."""
    P, rank = dist.get_world_size(), dist.get_rank()
    ow = own_windows(k_own, chunks, my, S, halo_allgather(k_own, my))
    cnt = [[windows(*chunks[ci], S, size, stp) for size, stp in kinds()] for ci in range(2 * P)]
    st = {ci: Stage1Seg(qs[ci], *key_counts(chunks[ci][1])) for ci in my}
    ref = next(iter(k_own.values()))
    for kind, method in ((1, "lse"), (0, "score")):
        sizes = [sum(cnt[cj][kind][1] - cnt[cj][kind][0] for cj in owned(r, P)) for r in range(P)]
        cur = ref.new_zeros(max(sizes), *ref.shape[1:])
        o = 0
        for (_, j0, j1, t) in ow[kind]:
            cur[o:o + j1 - j0] = t; o += j1 - j0
        nxt = torch.empty_like(cur)
        for step in range(P):
            reqs = []
            if step < P - 1:
                reqs = dist.batch_isend_irecv([dist.P2POp(dist.isend, cur, (rank + 1) % P),
                                               dist.P2POp(dist.irecv, nxt, (rank - 1) % P)])
            src = (rank - step) % P
            off = 0
            for cj in owned(src, P):
                j0, j1 = cnt[cj][kind]
                seg = cur[off:off + j1 - j0]
                for ci in my:
                    if j0 < (st[ci].n2 if kind == 1 else st[ci].n1):
                        getattr(st[ci], method)(seg, j0)
                off += j1 - j0
            for r_ in reqs:
                r_.wait()
            cur, nxt = nxt, cur
    return {ci: pick_blocks(st[ci].out, chunks[ci][0], st[ci].n1) for ci in my}


def _kv_padded(k_own, v_own, chunks, P):
    """this rank's K / V [1, Lmax, 2, 128], zero padded to the longest rank's token count"""
    mx = max(sum(chunks[cj][1] - chunks[cj][0] for cj in owned(r, P)) for r in range(P))
    k = k_own.new_zeros(1, mx, *k_own.shape[1:]); v = torch.zeros_like(k)
    k[0, :k_own.shape[0]] = k_own; v[0, :v_own.shape[0]] = v_own
    return k, v


class RingInfllm:
    """operator-test front end of Ring"""

    def __init__(self, chunks, my, S, P, rank, scale):
        self.chunks, self.my, self.S, self.P, self.rank, self.scale = chunks, my, S, P, rank, scale
        self.Ls = [chunks[ci][1] - chunks[ci][0] for ci in my]

    def _comp(self, sel):
        return InfllmComp(self.chunks, lambda r: owned(r, self.P), self.rank, self.P, sel, self.scale)

    def _outs(self, o, nh):
        outs, r0 = {}, 0
        for ci, L in zip(self.my, self.Ls):
            outs[ci] = from_rows(o[0, r0:r0 + L * GH], nh).reshape(L, -1); r0 += L * GH
        return outs

    def fwd(self, loc, sel, timer, nh):
        """loc[ci] = (q, k, v) token layout; sel {ci: [2, L, K]} -> {ci: O [L, nh * 128]}"""
        timer.start()
        k, v = _kv_padded(torch.cat([loc[ci][1] for ci in self.my]), torch.cat([loc[ci][2] for ci in self.my]), self.chunks, self.P)
        q = torch.cat([to_rows(loc[ci][0]) for ci in self.my])[None]
        o, _ = ring_forward(None, q, k, v, self._comp(sel))
        outs = self._outs(o, nh)
        timer.mark("attn", barrier=False)
        timer.mark("wait")
        return outs

    def fwd_online(self, loc_fn, timer, nh):
        """projections, Ring's selection (ring_stage1), the ring forward -> ({ci: O}, {ci: selection})"""
        timer.start()
        loc = loc_fn()
        timer.mark("local_proj", barrier=False)
        sel = ring_stage1({ci: loc[ci][0] for ci in self.my}, {ci: loc[ci][1] for ci in self.my}, self.chunks, self.my, self.S)
        timer.mark("select", barrier=False)
        k, v = _kv_padded(torch.cat([loc[ci][1] for ci in self.my]), torch.cat([loc[ci][2] for ci in self.my]), self.chunks, self.P)
        q = torch.cat([to_rows(loc[ci][0]) for ci in self.my])[None]
        o, _ = ring_forward(None, q, k, v, self._comp(sel))
        outs = self._outs(o, nh)
        timer.mark("attn", barrier=False)
        timer.mark("wait")
        return outs, sel

    def bwd(self, q_rows, o_rows, do_rows, lse, k_own, v_own, sel_f, timer):
        timer.start()
        sel_c, t0 = {}, 0
        for ci, L in zip(self.my, self.Ls):
            sel_c[ci] = sel_f[:, t0:t0 + L]; t0 += L
        k, v = _kv_padded(k_own, v_own, self.chunks, self.P)
        dq, dk, dv = ring_backward(None, do_rows[None], q_rows[None], k, v, o_rows[None], lse[None], None, self._comp(sel_c))
        timer.mark("attn", barrier=False)
        timer.mark("wait")
        Lo = k_own.shape[0]
        return dq[0].to(q_rows.dtype), dk[0, :Lo], dv[0, :Lo]


class RingOp:
    """Ring in a training step (autograd via reuse.CPAttn); the selection as every other system's"""

    def __init__(self, chunks, my, S, cfg):
        self.chunks, self.my, self.S, self.cfg = chunks, my, S, cfg
        self.P, self.rank = dist.get_world_size(), dist.get_rank()
        dev = torch.device("cuda", torch.cuda.current_device())
        self.pos = {ci: torch.arange(*chunks[ci], device=dev).to(torch.int32) for ci in my}
        self.Ls = [chunks[ci][1] - chunks[ci][0] for ci in my]
        self.scale = cfg.head_dim ** -0.5

    def _comp(self, sel):
        return InfllmComp(self.chunks, lambda r: owned(r, self.P), self.rank, self.P, sel, self.scale)

    def fwd(self, q, k, v):
        qs, ks, r = {}, {}, 0
        for ci, L in zip(self.my, self.Ls):
            qs[ci] = q[r:r + L]; ks[ci] = k[r:r + L]; r += L
        c = reuse.cache_get()
        if c is None:
            sel = select_cp(qs, ks, self.chunks, self.my, self.S)
            padded = pad_concat(sel, self.my).to(torch.int32)          # (one width for every chunk, -1 padded)
            sel, t0 = {}, 0
            for ci, L in zip(self.my, self.Ls):
                sel[ci] = filter_causal(padded[:, t0:t0 + L], self.pos[ci]).clone(); t0 += L
            c = dict(sel=sel)
            reuse.cache_put(c)
        sel = c["sel"]
        q_rows = torch.cat([to_rows(qs[ci]) for ci in self.my]).contiguous()
        kp, vp = _kv_padded(k, v, self.chunks, self.P)
        o, lse = ring_forward(None, q_rows[None], kp, vp, self._comp(sel))
        o_rows = o[0].contiguous()
        sel_t = torch.cat([sel[ci] for ci in self.my], 1)
        return from_rows(o_rows, self.cfg.n_heads).contiguous(), (q_rows, k.contiguous(), v.contiguous(), o_rows,
                                                                  lse[0].contiguous(), sel_t)

    def bwd(self, saved, do, aux=None):
        q_rows, k, v, o_rows, lse, sel_t = saved
        sel, t0 = {}, 0
        for ci, L in zip(self.my, self.Ls):
            sel[ci] = sel_t[:, t0:t0 + L]; t0 += L
        kp, vp = _kv_padded(k, v, self.chunks, self.P)
        dq, dk, dv = ring_backward(None, to_rows(do)[None], q_rows[None], kp, vp, o_rows[None], lse[None], None, self._comp(sel))
        Lo = k.shape[0]
        return from_rows(dq[0].to(q_rows.dtype), self.cfg.n_heads).contiguous(), dk[0, :Lo].to(k.dtype), dv[0, :Lo].to(v.dtype)

    def __call__(self, q, k, v):
        return reuse.CPAttn.apply(q, k, v, self)
