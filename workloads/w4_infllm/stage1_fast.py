"""InfLLM-v2 stage 1 (block-selection scoring) with a large tile: the same math as the official
`infllmv2_attn_stage1` (infllmv2_cuda_impl, compute_attn_1rowblock_splitkv_stage1), a different tiling; plus the
LSE merges of partial outputs in the official row layout (used by every context-parallel executor of W4).

Math, per KV head g, query token t of a chunk of L tokens, its 16 query heads h:
  pass 1 (LSE over the coarse keys k_c2, 128-token windows every 64 tokens, n2 of them):
      s = q_h . k_c2_j;  m_h = max_j s, l_h = sum_j exp2((s - m_h) * scale * log2 e)  over visible j
  pass 2 (fine keys k_c, 32-token windows every 16 tokens, n1 of them):
      p_h(t, j) = exp2(s * scale * log2 e - m_h * scale * log2 e) / l_h ;  score[g, t, j] = bf16(sum_h p_h(t, j))
  visible keys (causal, official integer formula, C division): for stride st and key count nK,
      cmq = (L - st + 1) / st ;  lim(t) = min(nK, max(0, (t + 1) / st - 1 + nK - cmq)) ;  j visible iff j < lim(t)
  masked entries are 0; NaN -> 0 (as the official Python wrapper).
  early exit (official: `if (n_block_min >= n_block_max_c) return;`, i.e. the token's row stays 0) when
      maxc(t) = (t - 63) / 64 + (((16 * n1 + 15) - 63) / 64 - (L - 63) / 64) <= 0     (C division)
The official kernel computes one token's 16 heads (16 rows) per one-warp CTA and re-reads every key tile per token.
Pass 2 is dense causal (no per-token sparsity), so a tile of TOK tokens x 16 heads = 16*TOK rows shares every key
tile: the classic FlashAttention shape. Here TOK = 8 (128 rows) in Triton.

merge_n: LSE-merge of NP partial outputs (bf16 or fp32 rows, LSE +inf = an empty partial, the official convention).
"""
import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634
G = 16


def _cdiv_c(a, b):          # C integer division (truncation toward zero)
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def early_exit(L, n1, device):
    t = torch.arange(L, device=device)
    tm = t - 63
    ctr = torch.where(tm >= 0, tm // 64, -((-tm) // 64))
    return ctr + (_cdiv_c(16 * n1 + 15 - 63, 64) - _cdiv_c(L - 63, 64)) <= 0


def limits(L, nK, st, device):
    cmq = _cdiv_c(L - st + 1, st)
    t = torch.arange(L, device=device)
    return torch.clamp(torch.clamp((t + 1) // st - 1 + nK - cmq, min=0), max=nK)


@torch.no_grad()
def stage1_ref(q, k1, k2, scale=None):
    """fp32 torch reference of the official math. q [L, 32, 128] bf16, k1 [n1, 2, 128], k2 [n2, 2, 128]."""
    L, H, D = q.shape
    Hk = k1.shape[1]
    scale = D ** -0.5 if scale is None else scale
    sl2 = scale * LOG2E
    n1, n2 = k1.shape[0], k2.shape[0]
    lim1, lim2 = limits(L, n1, 16, q.device), limits(L, n2, 64, q.device)
    out = torch.zeros(Hk, L, n1, dtype=q.dtype, device=q.device)
    for g in range(Hk):
        qg = q[:, g * G:(g + 1) * G].float()                                  # [L, 16, D]
        s2 = torch.einsum("thd,jd->thj", qg, k2[:, g].float())
        s2 = s2.masked_fill(torch.arange(n2, device=q.device)[None, None, :] >= lim2[:, None, None], float("-inf"))
        m = s2.amax(-1, keepdim=True)
        l = torch.exp2((s2 - m) * sl2).nan_to_num(0.0).sum(-1, keepdim=True)
        ms = torch.where(m == float("-inf"), torch.zeros_like(m), m * sl2)
        s1 = torch.einsum("thd,jd->thj", qg, k1[:, g].float())
        p = torch.exp2(s1 * sl2 - ms) * (1.0 / l)
        p = p.masked_fill(torch.arange(n1, device=q.device)[None, None, :] >= lim1[:, None, None], 0.0)
        out[g] = p.sum(1).to(out.dtype)
    out[:, early_exit(L, n1, q.device)] = 0
    return torch.where(torch.isnan(out), torch.zeros_like(out), out)


@triton.jit
def _stage1_kernel(Q, K1, K2, OUT, L, n1, n2, cmq1, cmq2, offc, sl2,
                   stride_qt, stride_qh, stride_k1, stride_k1h, stride_k2, stride_k2h, stride_og, stride_ot,
                   TOK: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    g = tl.program_id(1)
    R: tl.constexpr = TOK * 16
    r = tl.arange(0, R)
    tok = pid * TOK + r // 16
    head = g * 16 + r % 16
    d = tl.arange(0, D)
    qmask = tok < L
    q = tl.load(Q + tok[:, None] * stride_qt + head[:, None] * stride_qh + d[None, :], mask=qmask[:, None], other=0.0)
    # per-row visible limits (official integer formula; (t+1)//st >= 0 so floor == C division)
    lim2 = tl.minimum(tl.maximum((tok + 1) // 64 - 1 + n2 - cmq2, 0), n2)
    lim1 = tl.minimum(tl.maximum((tok + 1) // 16 - 1 + n1 - cmq1, 0), n1)
    last = tl.minimum(pid * TOK + TOK, L) - 1
    end2 = tl.minimum(tl.maximum((last + 1) // 64 - 1 + n2 - cmq2, 0), n2)
    end1 = tl.minimum(tl.maximum((last + 1) // 16 - 1 + n1 - cmq1, 0), n1)
    # pass 1: online max / sum over the coarse keys
    m = tl.full([R], float("-inf"), tl.float32)
    lsum = tl.zeros([R], tl.float32)
    cols = tl.arange(0, BN)
    for j0 in range(0, end2, BN):
        j = j0 + cols
        k = tl.load(K2 + j[:, None] * stride_k2 + g * stride_k2h + d[None, :], mask=(j < n2)[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k))
        s = tl.where(j[None, :] < lim2[:, None], s, float("-inf"))
        mn = tl.maximum(m, tl.max(s, 1))
        mns = tl.where(mn == float("-inf"), 0.0, mn)
        lsum = lsum * tl.exp2((m - mns) * sl2) + tl.sum(tl.exp2((s - mns[:, None]) * sl2), 1)
        m = mn
    ms = tl.where(m == float("-inf"), 0.0, m * sl2)
    inv = 1.0 / lsum
    # pass 2: probabilities over the fine keys, summed over each token's 16 heads
    otok = pid * TOK + tl.arange(0, TOK)
    tm = otok - 63
    early = (tl.where(tm >= 0, tm // 64, -((-tm) // 64)) + offc) <= 0
    for j0 in range(0, end1, BN):
        j = j0 + cols
        k = tl.load(K1 + j[:, None] * stride_k1 + g * stride_k1h + d[None, :], mask=(j < n1)[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k))
        p = tl.exp2(s * sl2 - ms[:, None]) * inv[:, None]
        p = tl.where(j[None, :] < lim1[:, None], p, 0.0)
        ps = tl.sum(tl.reshape(p, [TOK, 16, BN]), 1)                         # [TOK, BN]
        ps = tl.where(ps != ps, 0.0, ps)                                     # NaN -> 0
        ps = tl.where(early[:, None], 0.0, ps)
        tl.store(OUT + g * stride_og + otok[:, None] * stride_ot + j[None, :], ps.to(OUT.dtype.element_ty),
                 mask=(otok < L)[:, None] & (j < n1)[None, :])


@torch.no_grad()
def stage1_fast(q, k1, k2, scale=None, TOK=8, BN=64, num_warps=8, num_stages=2):
    """same math as infllmv2_attn_stage1(...)[:, :L] for one segment with a different tiling (scores agree up to
    accumulation order): [Hk, L, n1] bf16."""
    L, H, D = q.shape
    Hk = k1.shape[1]
    scale = D ** -0.5 if scale is None else scale
    n1, n2 = k1.shape[0], k2.shape[0]
    q = q.contiguous()
    k1 = k1 if k1.stride(-1) == 1 else k1.contiguous()       # strided rows are fine (e.g. a 1 KB-row region)
    k2 = k2 if k2.stride(-1) == 1 else k2.contiguous()
    out = torch.zeros(Hk, L, n1, dtype=q.dtype, device=q.device)
    if L == 0 or n1 == 0:
        return out
    grid = (triton.cdiv(L, TOK), Hk)
    _stage1_kernel[grid](q, k1, k2, out, L, n1, max(n2, 0), _cdiv_c(L - 16 + 1, 16), _cdiv_c(L - 64 + 1, 64),
                         _cdiv_c(16 * n1 + 15 - 63, 64) - _cdiv_c(L - 63, 64), scale * LOG2E, q.stride(0), q.stride(1), k1.stride(0), k1.stride(1),
                         k2.stride(0) if n2 else 0, k2.stride(1) if n2 else 0, out.stride(0), out.stride(1),
                         TOK=TOK, BN=BN, D=D, num_warps=num_warps, num_stages=num_stages)
    return out


@triton.jit
def _merge_kernel(ACC, LSE, OB, LB, R, HK: tl.constexpr, D: tl.constexpr, BR: tl.constexpr):
    """acc [R, HK, D] fp32 (official row layout), lse [HK, R]; merge one partial (ob bf16 [R, HK, D], lb [HK, R])."""
    pid = tl.program_id(0)
    h = tl.program_id(1)
    r = pid * BR + tl.arange(0, BR)
    d = tl.arange(0, D)
    m = r < R
    lo = tl.load(LSE + h * R + r, mask=m, other=float("-inf"))
    ln = tl.load(LB + h * R + r, mask=m, other=float("-inf"))
    mx = tl.maximum(lo, ln)
    mxs = tl.where(mx == float("-inf"), 0.0, mx)
    eo = tl.exp(lo - mxs)
    en = tl.exp(ln - mxs)
    tot = eo + en
    merged = tl.where(tot > 0, mxs + tl.log(tot), float("-inf"))
    wo = tl.where(tot > 0, eo / tot, 0.0)
    wn = tl.where(tot > 0, en / tot, 0.0)
    off = r[:, None] * (HK * D) + h * D + d[None, :]
    a = tl.load(ACC + off, mask=m[:, None], other=0.0)
    o = tl.load(OB + off, mask=m[:, None], other=0.0).to(tl.float32)
    tl.store(ACC + off, a * wo[:, None] + o * wn[:, None], mask=m[:, None])
    tl.store(LSE + h * R + r, merged, mask=m)


def lse_merge_(acc, lse, ob, lb, BR=64):
    """in place: (acc, lse) <- LSE-merge of (acc, lse) and (ob, lb). One pass over the data."""
    R, HK, D = acc.shape
    _merge_kernel[(triton.cdiv(R, BR), HK)](acc, lse, ob.contiguous(), lb.contiguous(), R, HK=HK, D=D, BR=BR,
                                            num_warps=4)


# ---------------------------------------------------------------- key-segment variants (ring / pushed shards)
# The keys of one pass may arrive in segments (a shard of the compressed keys at a time). Pass 1 folds a segment
# of coarse keys [kbase, kbase + nseg) into running per-row (max, sum); pass 2 writes the score columns of a
# segment of fine keys using the final (max, sum). Visibility limits use the GLOBAL key counts n1 / n2.
@triton.jit
def _lse_seg_kernel(Q, K2, M, LS, L, n2, cmq2, kbase, nseg, sl2, stride_qt, stride_qh, stride_k2, stride_k2h,
                    TOK: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    g = tl.program_id(1)
    R: tl.constexpr = TOK * 16
    r = tl.arange(0, R)
    tok = pid * TOK + r // 16
    head = g * 16 + r % 16
    d = tl.arange(0, D)
    qmask = tok < L
    q = tl.load(Q + tok[:, None] * stride_qt + head[:, None] * stride_qh + d[None, :], mask=qmask[:, None], other=0.0)
    lim2 = tl.minimum(tl.maximum((tok + 1) // 64 - 1 + n2 - cmq2, 0), n2)
    last = tl.minimum(pid * TOK + TOK, L) - 1
    end2 = tl.minimum(tl.minimum(tl.maximum((last + 1) // 64 - 1 + n2 - cmq2, 0), n2) - kbase, nseg)
    ridx = g * (L * 16) + pid * R + r                          # rows of this kv head: (token, head) = t*16 + h
    rm = pid * R + r < L * 16
    m = tl.load(M + ridx, mask=rm, other=float("-inf"))
    lsum = tl.load(LS + ridx, mask=rm, other=0.0)
    cols = tl.arange(0, BN)
    for j0 in range(0, end2, BN):
        jl = j0 + cols
        k = tl.load(K2 + jl[:, None] * stride_k2 + g * stride_k2h + d[None, :], mask=(jl < nseg)[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k))
        s = tl.where(((kbase + jl)[None, :] < lim2[:, None]) & (jl < nseg)[None, :], s, float("-inf"))
        mn = tl.maximum(m, tl.max(s, 1))
        mns = tl.where(mn == float("-inf"), 0.0, mn)
        mos = tl.where(m == float("-inf"), 0.0, m)
        lsum = lsum * tl.exp2((mos - mns) * sl2) + tl.sum(tl.exp2((s - mns[:, None]) * sl2), 1)   # lsum = 0 while m = -inf
        m = mn
    tl.store(M + ridx, m, mask=rm)
    tl.store(LS + ridx, lsum, mask=rm)


@triton.jit
def _score_seg_kernel(Q, K1, M, LS, OUT, L, n1, cmq1, offc, kbase, nseg, sl2, stride_qt, stride_qh, stride_k1,
                      stride_k1h, stride_og, stride_ot, TOK: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    g = tl.program_id(1)
    R: tl.constexpr = TOK * 16
    r = tl.arange(0, R)
    tok = pid * TOK + r // 16
    head = g * 16 + r % 16
    d = tl.arange(0, D)
    qmask = tok < L
    q = tl.load(Q + tok[:, None] * stride_qt + head[:, None] * stride_qh + d[None, :], mask=qmask[:, None], other=0.0)
    lim1 = tl.minimum(tl.maximum((tok + 1) // 16 - 1 + n1 - cmq1, 0), n1)
    last = tl.minimum(pid * TOK + TOK, L) - 1
    end1 = tl.minimum(tl.minimum(tl.maximum((last + 1) // 16 - 1 + n1 - cmq1, 0), n1) - kbase, nseg)
    ridx = g * (L * 16) + pid * R + r
    rm = pid * R + r < L * 16
    m = tl.load(M + ridx, mask=rm, other=float("-inf"))
    lsum = tl.load(LS + ridx, mask=rm, other=0.0)
    ms = tl.where(m == float("-inf"), 0.0, m * sl2)
    inv = 1.0 / lsum
    otok = pid * TOK + tl.arange(0, TOK)
    tm = otok - 63
    early = (tl.where(tm >= 0, tm // 64, -((-tm) // 64)) + offc) <= 0
    cols = tl.arange(0, BN)
    for j0 in range(0, end1, BN):
        jl = j0 + cols
        k = tl.load(K1 + jl[:, None] * stride_k1 + g * stride_k1h + d[None, :], mask=(jl < nseg)[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k))
        p = tl.exp2(s * sl2 - ms[:, None]) * inv[:, None]
        p = tl.where((kbase + jl)[None, :] < lim1[:, None], p, 0.0)
        ps = tl.sum(tl.reshape(p, [TOK, 16, BN]), 1)
        ps = tl.where(ps != ps, 0.0, ps)
        ps = tl.where(early[:, None], 0.0, ps)
        tl.store(OUT + g * stride_og + otok[:, None] * stride_ot + (kbase + jl)[None, :], ps.to(OUT.dtype.element_ty),
                 mask=(otok < L)[:, None] & (jl < nseg)[None, :])


class Stage1Seg:
    """stage 1 for one query chunk (L tokens, global key counts n1 / n2), keys fed in segments."""

    def __init__(self, q, n1, n2, scale=None, TOK=8, BN=64):
        self.q = q if q.stride(-1) == 1 else q.contiguous()
        self.L, H, self.D = q.shape
        self.Hk = H // G
        self.n1, self.n2, self.TOK, self.BN = n1, n2, TOK, BN
        self.sl2 = (self.D ** -0.5 if scale is None else scale) * LOG2E
        self.m = torch.full((self.Hk, self.L * G), float("-inf"), device=q.device)
        self.l = torch.zeros(self.Hk, self.L * G, device=q.device)
        self.out = torch.zeros(self.Hk, self.L, n1, dtype=q.dtype, device=q.device)
        self.grid = (triton.cdiv(self.L, TOK), self.Hk)

    def lse(self, k2seg, kbase):                 # k2seg [nseg, Hk, D] = coarse keys kbase .. kbase + nseg - 1
        nseg = k2seg.shape[0]
        if nseg == 0 or self.L == 0:
            return
        _lse_seg_kernel[self.grid](self.q, k2seg, self.m, self.l, self.L, self.n2, _cdiv_c(self.L - 63, 64), kbase,
                                   nseg, self.sl2, self.q.stride(0), self.q.stride(1), k2seg.stride(0),
                                   k2seg.stride(1), TOK=self.TOK, BN=self.BN, D=self.D, num_warps=8, num_stages=2)

    def score(self, k1seg, kbase):
        nseg = k1seg.shape[0]
        if nseg == 0 or self.L == 0:
            return
        _score_seg_kernel[self.grid](self.q, k1seg, self.m, self.l, self.out, self.L, self.n1,
                                     _cdiv_c(self.L - 15, 16), _cdiv_c(16 * self.n1 + 15 - 63, 64) - _cdiv_c(self.L - 63, 64),
                                     kbase, nseg, self.sl2, self.q.stride(0), self.q.stride(1), k1seg.stride(0),
                                     k1seg.stride(1), self.out.stride(0), self.out.stride(1), TOK=self.TOK,
                                     BN=self.BN, D=self.D, num_warps=8, num_stages=2)


@triton.jit
def _merge2_kernel(O2, L2, OUT, R, HK: tl.constexpr, D: tl.constexpr, BR: tl.constexpr):
    """out[r] = LSE-merge of two partials o2[r], o2[R + r] (bf16, official row layout [rows, HK, D]);
    lse2 [HK, 2R] natural log, +inf (= nothing selected, official convention) treated as -inf."""
    pid = tl.program_id(0)
    h = tl.program_id(1)
    r = pid * BR + tl.arange(0, BR)
    d = tl.arange(0, D)
    m = r < R
    la = tl.load(L2 + h * 2 * R + r, mask=m, other=float("-inf"))
    lb = tl.load(L2 + h * 2 * R + R + r, mask=m, other=float("-inf"))
    la = tl.where(la == float("inf"), float("-inf"), la)
    lb = tl.where(lb == float("inf"), float("-inf"), lb)
    mx = tl.maximum(la, lb)
    mxs = tl.where(mx == float("-inf"), 0.0, mx)
    ea = tl.exp(la - mxs)
    eb = tl.exp(lb - mxs)
    tot = ea + eb
    wa = tl.where(tot > 0, ea / tot, 0.0)
    wb = tl.where(tot > 0, eb / tot, 0.0)
    off = r[:, None] * (HK * D) + h * D + d[None, :]
    a = tl.load(O2 + off, mask=m[:, None], other=0.0).to(tl.float32)
    b = tl.load(O2 + R * HK * D + off, mask=m[:, None], other=0.0).to(tl.float32)
    tl.store(OUT + off, (a * wa[:, None] + b * wb[:, None]).to(OUT.dtype.element_ty), mask=m[:, None])


def merge2(o2, lse2, BR=64):
    R2, HK, D = o2.shape
    R = R2 // 2
    out = torch.empty(R, HK, D, dtype=o2.dtype, device=o2.device)
    _merge2_kernel[(triton.cdiv(R, BR), HK)](o2, lse2.contiguous(), out, R, HK=HK, D=D, BR=BR, num_warps=4)
    return out


@triton.jit
def _mergen_kernel(ON, LN, OUT, R, NP: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, BR: tl.constexpr):
    """out[r] = LSE-merge of NP partials on[p*R + r] (bf16, official row layout); lse [HK, NP*R], +inf = empty."""
    pid = tl.program_id(0)
    h = tl.program_id(1)
    r = pid * BR + tl.arange(0, BR)
    d = tl.arange(0, D)
    m = r < R
    mx = tl.full([BR], float("-inf"), tl.float32)
    for p in tl.static_range(NP):
        l = tl.load(LN + h * NP * R + p * R + r, mask=m, other=float("-inf"))
        l = tl.where(l == float("inf"), float("-inf"), l)
        mx = tl.maximum(mx, l)
    mxs = tl.where(mx == float("-inf"), 0.0, mx)
    tot = tl.zeros([BR], tl.float32)
    acc = tl.zeros([BR, D], tl.float32)
    off = r[:, None] * (HK * D) + h * D + d[None, :]
    for p in tl.static_range(NP):
        l = tl.load(LN + h * NP * R + p * R + r, mask=m, other=float("-inf"))
        l = tl.where(l == float("inf"), float("-inf"), l)
        e = tl.exp(l - mxs)
        tot += e
        acc += tl.load(ON + p * R * HK * D + off, mask=m[:, None], other=0.0).to(tl.float32) * e[:, None]
    acc = acc / tl.where(tot > 0, tot, 1.0)[:, None]
    tl.store(OUT + off, acc.to(OUT.dtype.element_ty), mask=m[:, None])


def merge_n(on, lsen, np_, BR=64, out_dtype=None):
    RN, HK, D = on.shape
    R = RN // np_
    out = torch.empty(R, HK, D, dtype=out_dtype or on.dtype, device=on.device)
    _mergen_kernel[(triton.cdiv(R, BR), HK)](on, lsen.contiguous(), out, R, NP=np_, HK=HK, D=D, BR=BR, num_warps=4)
    return out
