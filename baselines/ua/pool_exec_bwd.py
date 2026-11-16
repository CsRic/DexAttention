"""Backward of the decomposed executor for tile masks (UltraAttn's backward execution plan, dump_plan_structure.py --fob 1:
UA's dependent graph + fused plan for fob=1, converted by plan_adapter as the forward). Communication = UA's backward:
  rows  'i','r'  Q, dO, D = rowsum(dO * O), LSE          (UA sends D, not O)
  cols  'i','c'  K, V
  rows  'o','r'  partial dQ    -> the row block's owner, summed
  cols  'o','c'  partial dK, dV -> the column block's owner, summed
Partials travel in fp32.

Compute: the official flash-attention 2 backward kernel (flash_attn_interface._flash_attn_backward), one call per
maximal rectangle of a step's tile mask, reading the pools in place:
  full      contiguous q tiles x one contiguous run of live key tiles shared by all of them (causal=False)
  causal    q tiles r..r+n x key tiles a..r+n, the diagonal tiles being the mask's causal (2) tiles (causal=True; bottom-right
            alignment gives each q tile exactly keys a .. its own diagonal)
The kernel computes D itself as rowsum(dO * out). UA ships D, not O, so each row block gets out' = dO * D / |dO|^2 per
row and head (computed once when the block is loaded / received): rowsum(dO * out') = D up to fp16 rounding of out'
(rows with dO = 0, e.g. padding, get out' = 0 and D = 0).

Accumulators: dQ over every row slot, dK / dV over every column slot, fp32, zeroed at the start of each call; the kernel's
fp16 gradients of each rectangle are added in, received partials are added into the owner's accumulator.
"""
import numpy as np
import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import _flash_attn_backward

from baselines.ua.pool_exec import T, _launch, _wait


class PoolTilesBwd:
    def __init__(self, mask, sched, H, Hk, D, dtype, device):
        self.m = np.asarray(mask, np.int8)
        self.B = sched.block_tokens; self.k = self.B // T
        self.H, self.Hk, self.D = H, Hk, D
        qs, ks = set(sched.init_q) | set(sched.owned_q), set(sched.init_kv) | set(sched.owned_q)
        for st in sched.steps:
            for c in st.comp:
                qs.update(c.q_blocks); ks.update(c.kv_blocks)
            for c in st.comm:
                if c.is_recv(sched.rank) or c.is_send(sched.rank):
                    (qs if c.rc == "r" else ks).update(c.blocks)
        self.qb, self.kb = sorted(qs), sorted(ks)
        self.qslot = {b: i for i, b in enumerate(self.qb)}; self.kslot = {b: i for i, b in enumerate(self.kb)}
        nq, nk = len(self.qb) * self.B, len(self.kb) * self.B
        self.q = torch.empty(nq, H, D, dtype=dtype, device=device)
        self.do = torch.empty(nq, H, D, dtype=dtype, device=device)
        self.out = torch.empty(nq, H, D, dtype=dtype, device=device)       # out' (see the module docstring)
        self.dd = torch.empty(H, nq, dtype=torch.float32, device=device)   # D
        self.lse = torch.empty(H, nq, dtype=torch.float32, device=device)
        self.kp = torch.empty(nk, Hk, D, dtype=dtype, device=device)
        self.vp = torch.empty(nk, Hk, D, dtype=dtype, device=device)
        self.dq = torch.zeros(nq, H, D, dtype=torch.float32, device=device)
        self.dk = torch.zeros(nk, Hk, D, dtype=torch.float32, device=device)
        self.dv = torch.zeros(nk, Hk, D, dtype=torch.float32, device=device)
        self.dev = device
        self.cache = {}

    # ---- pool views ---------------------------------------------------------------------------------------------
    def qrows(self, b):
        s = self.qslot[b] * self.B; return s, s + self.B

    def krows(self, b):
        s = self.kslot[b] * self.B; return s, s + self.B

    def load(self, rows, cols):
        """owned blocks into their slots (outside the timed call). rows[b] = (Q, dO, D [H, B], LSE [H, B]) of block b,
        cols[b] = (K, V)"""
        for b, (q, do, d, l) in rows.items():
            s, e = self.qrows(b)
            self.q[s:e].copy_(q); self.do[s:e].copy_(do); self.dd[:, s:e].copy_(d); self.lse[:, s:e].copy_(l)
        for b, (k, v) in cols.items():
            s, e = self.krows(b); self.kp[s:e].copy_(k); self.vp[s:e].copy_(v)

    def make_out(self, b):
        """out' of row block b from its dO and D: rowsum(dO * out') = D"""
        s, e = self.qrows(b)
        do = self.do[s:e].float()
        n2 = (do * do).sum(-1)                                          # [B, H]
        c = torch.where(n2 > 0, self.dd[:, s:e].t() / n2.clamp_min(1e-30), torch.zeros_like(n2))
        self.out[s:e].copy_((do * c.unsqueeze(-1)).to(self.out.dtype))

    def begin(self):
        self.dq.zero_(); self.dk.zero_(); self.dv.zero_()

    # ---- one rectangle -> maximal FA rectangles ----------------------------------------------------------------
    def _plan(self, qb, kvb):
        k = self.k
        gq = np.concatenate([np.arange(b * k, (b + 1) * k) for b in qb])
        pq = np.concatenate([np.arange(self.qslot[b] * k, (self.qslot[b] + 1) * k) for b in qb])
        gk = np.concatenate([np.arange(b * k, (b + 1) * k) for b in sorted(kvb)])
        pk = np.concatenate([np.arange(self.kslot[b] * k, (self.kslot[b] + 1) * k) for b in sorted(kvb)])
        sub = self.m[np.ix_(gq, gk)]
        full, caus = [], []                                   # (q pool tile, k pool tile start, end)
        for r in range(len(gq)):
            cols = np.flatnonzero(sub[r])
            if not len(cols):
                continue
            pos = pk[cols]
            cuts = np.flatnonzero((np.diff(pos) != 1) | (sub[r, cols[:-1]] == 2)) + 1
            for run, rp in zip(np.split(cols, cuts), np.split(pos, cuts)):
                assert not (sub[r, run[:-1]] == 2).any()
                if sub[r, run[-1]] == 2:
                    caus.append((int(pq[r]), int(rp[0]), int(rp[-1]) + 1))
                else:
                    full.append((int(pq[r]), int(rp[0]), int(rp[-1]) + 1))
        ops = []
        # full: same key run, consecutive q pool tiles -> one call
        full.sort(key=lambda x: (x[1], x[2], x[0]))
        i = 0
        while i < len(full):
            q0, a, b = full[i]; j = i + 1
            while j < len(full) and full[j][1:] == (a, b) and full[j][0] == full[j - 1][0] + 1:
                j += 1
            ops.append((q0, q0 + (j - i), a, b, False)); i = j
        # causal: q tiles r, r+1, ... with key runs a..r, a..r+1, ... (same start, end = own diagonal + 1) -> one call
        caus.sort(key=lambda x: (x[1], x[0]))
        i = 0
        while i < len(caus):
            q0, a, b = caus[i]; j = i + 1
            while (j < len(caus) and caus[j][1] == a and caus[j][0] == caus[j - 1][0] + 1
                   and caus[j][2] == caus[j - 1][2] + 1):
                j += 1
            ops.append((q0, q0 + (j - i), a, caus[j - 1][2], True)); i = j
        ops.sort(key=lambda o: -(o[1] - o[0]) * (o[3] - o[2]))      # big first
        return ops

    def compute(self, qb, kvb, scale):
        key = (tuple(qb), tuple(kvb))
        ops = self.cache.get(key)
        if ops is None:
            ops = self.cache[key] = self._plan(qb, kvb)
        for q0, q1, a, b, causal in ops:
            qs, qe, ks, ke = q0 * T, q1 * T, a * T, b * T
            q = self.q[qs:qe].unsqueeze(0); do = self.do[qs:qe].unsqueeze(0); o = self.out[qs:qe].unsqueeze(0)
            kk = self.kp[ks:ke].unsqueeze(0); vv = self.vp[ks:ke].unsqueeze(0)
            lse = self.lse[:, qs:qe].unsqueeze(0).contiguous()
            dq = torch.empty_like(q); dk = torch.empty_like(kk); dv = torch.empty_like(vv)
            _flash_attn_backward(do, q, kk, vv, o, lse, dq, dk, dv, 0.0, scale, causal, -1, -1, 0.0, None, False, None)
            self.dq[qs:qe].add_(dq[0]); self.dk[ks:ke].add_(dk[0]); self.dv[ks:ke].add_(dv[0])


def fusion_backward_pooled(sched, pt, softmax_scale, partial_dtype=torch.float32):
    """One backward call on pooled buffers (pt = PoolTilesBwd, owned blocks loaded, out' of owned rows made).
    Returns ({owned row block: dQ [B, H, D] fp32}, {owned column block: (dK, dV) [B, Hk, D] fp32})."""
    rank, B, H, Hk, D = sched.rank, pt.B, pt.H, pt.Hk, pt.D
    pt.begin()

    def build_comm(step, io):
        ops, merges, made = [], [], []
        for c in step.comm:
            if c.io != io:
                continue
            for blk in c.blocks:
                if io == "i" and c.rc == "r":
                    s, e = pt.qrows(blk)
                    ts = (pt.q[s:e], pt.do[s:e], pt.dd[:, s:e], pt.lse[:, s:e])
                    if c.is_send(rank):
                        ops += [(dist.isend, t.contiguous() if not t.is_contiguous() else t, c.dst) for t in ts]
                    elif c.is_recv(rank):
                        bufs = (pt.q[s:e], pt.do[s:e], torch.empty(H, B, device=pt.dev), torch.empty(H, B, device=pt.dev))
                        ops += [(dist.irecv, t, c.src) for t in bufs]
                        made.append((blk, bufs[2], bufs[3]))
                elif io == "i":
                    s, e = pt.krows(blk)
                    if c.is_send(rank):
                        ops += [(dist.isend, pt.kp[s:e], c.dst), (dist.isend, pt.vp[s:e], c.dst)]
                    elif c.is_recv(rank):
                        ops += [(dist.irecv, pt.kp[s:e], c.src), (dist.irecv, pt.vp[s:e], c.src)]
                elif c.rc == "r":                                   # partial dQ -> the row owner
                    s, e = pt.qrows(blk)
                    if c.is_send(rank):
                        ops.append((dist.isend, pt.dq[s:e].to(partial_dtype), c.dst))
                    elif c.is_recv(rank):
                        g = torch.empty(B, H, D, device=pt.dev, dtype=partial_dtype)
                        ops.append((dist.irecv, g, c.src)); merges.append(("r", blk, g, None))
                else:                                               # partial dK, dV -> the column owner
                    s, e = pt.krows(blk)
                    if c.is_send(rank):
                        ops += [(dist.isend, pt.dk[s:e].to(partial_dtype), c.dst), (dist.isend, pt.dv[s:e].to(partial_dtype), c.dst)]
                    elif c.is_recv(rank):
                        gk = torch.empty(B, Hk, D, device=pt.dev, dtype=partial_dtype); gv = torch.empty_like(gk)
                        ops += [(dist.irecv, gk, c.src), (dist.irecv, gv, c.src)]; merges.append(("c", blk, gk, gv))
        return ops, merges, made

    def finish_inputs(made):                  # received row blocks: D / LSE into the pools, then out'
        for blk, d, l in made:
            s, e = pt.qrows(blk); pt.dd[:, s:e].copy_(d); pt.lse[:, s:e].copy_(l); pt.make_out(blk)

    def drain(merges):
        for kind, blk, a, b in merges:
            if kind == "r":
                s, e = pt.qrows(blk); pt.dq[s:e].add_(a.float())
            else:
                s, e = pt.krows(blk); pt.dk[s:e].add_(a.float()); pt.dv[s:e].add_(b.float())

    steps = sched.steps
    if steps:
        o_, _, made = build_comm(steps[0], "i"); in_reqs = _launch(o_)
    else:
        in_reqs, made = [], []
    out_reqs, out_merge = [], []
    for i, step in enumerate(steps):
        _wait(in_reqs); finish_inputs(made)
        if i + 1 < len(steps):
            o_, _, made = build_comm(steps[i + 1], "i"); in_reqs = _launch(o_)
        else:
            in_reqs, made = [], []
        for c in step.comp:
            pt.compute(c.q_blocks, c.kv_blocks, softmax_scale)
        o_ops, o_m, _ = build_comm(step, "o")
        out_reqs += _launch(o_ops); out_merge += o_m
        if len(out_reqs) >= 48:
            _wait(out_reqs); drain(out_merge); out_reqs, out_merge = [], []
    _wait(out_reqs); drain(out_merge)
    dq = {b: pt.dq[slice(*pt.qrows(b))] for b in sched.owned_q}
    dkv = {b: (pt.dk[slice(*pt.krows(b))], pt.dv[slice(*pt.krows(b))]) for b in sched.owned_q}
    return dq, dkv
