"""Shared correctness check for every system of the static tests: an FP64 reference on SAMPLED rows.

Each system is checked against an independent FP64 reference of the input's mask, on the REAL tokens (padding
excluded), in the real-token order (prompt, then the replies; the order of the range files). The mask is given as
per-token key ranges [S, R, 2] (start, end), as the range files of the mask generators (`<input>_ranges.npy`);
`ranges_causal(S)` is a helper that builds the ranges of a dense causal mask.

Per rank (all ranks call `check` collectively when a process group is given):
  1. LSE and D = rowsum(dO * O) of ALL query rows, FP64: each rank computes a 1/P slice of the rows, then all-gather.
  2. Samples (seed + rank): up to n_sample query rows among the rows this rank holds outputs for (O, dQ), and up to
     n_sample key rows among the key rows it holds (dK, dV).
  3. FP64 reference on the samples: O and dQ of the sampled query rows (all keys), dK and dV of the sampled key rows
     (all queries, the LSE / D of step 1; GQA: summed over the q heads of the group).
  4. Error per tensor = relative L2 over all sampled rows of all ranks: ||sys - ref||_2 / ||ref||_2 (sums of squares
     all-reduced); also the max abs error / max |ref|.
Inputs on every rank: the GLOBAL q [S, hq, d], k, v [S, hk, d], do [S, hq, d] (any float dtype, any device) in real-token
order, and this rank's system outputs with the global real-token ids of their rows.
A backward run on given saved statistics (the attention-layer test feeds random O and a constant LSE): pass lse_in [S, hq]
and o_in [S, hq, d]; the reference is then the backward formula on those (D = rowsum(dO * o_in)), step 1 is skipped.
usage:  err = check(q, k, v, do, ranges, scale, out=dict(o=..., dq=..., dk=..., dv=...), q_rows=..., kv_rows=...)
        ranges_for_mask(npz path) -> the range file of a static mask; real_ids(z, T) -> real-token id per padded position"""
import math
import os
import re

import numpy as np
import torch


def ranges_for_mask(npz_path):
    """[S, R, 2] key ranges of a static mask file (<dir>/<input>[_nb<k>].npz -> <dir>/<input>_ranges.npy)"""
    d, f = os.path.split(npz_path)
    base = re.sub(r"_nb\d+$", "", f[:-4])
    return np.load(os.path.join(d, f"{base}_ranges.npy"))


def real_ids(z, T):
    """real-token id of every position of the padded layout (-1 = padding); z = the loaded mask npz"""
    if "tok_valid" not in z.files:
        return np.arange(T)
    v = z["tok_valid"].astype(bool); r = np.full(len(v), -1, np.int64); r[v] = np.arange(int(v.sum()))
    return r


def ranges_causal(S):
    r = np.zeros((S, 1, 2), np.int64); r[:, 0, 1] = np.arange(1, S + 1)
    return r


class BlockMask:
    """One head's block bitmap in place of key ranges (block top-k selection): bm [nqb, nkb] bool, query row i
    sees key j iff bm[i // qblk, j // kblk] (non-causal). Accepted wherever a ranges tensor is (one per head in grouped mode)."""
    def __init__(self, bm, qblk, kblk, S):
        self.bm = torch.as_tensor(bm, dtype=torch.bool); self.qblk, self.kblk, self.S = qblk, kblk, S
        self.device = self.bm.device

    def to(self, dev):
        return BlockMask(self.bm.to(dev), self.qblk, self.kblk, self.S)

    def vis(self, rows, keys):
        return self.bm[(rows // self.qblk)[:, None], (keys // self.kblk)[None, :]]


def _vis(rg, rows, k0, k1, dev):
    """bool [len(rows), k1 - k0]: key k visible from query row (any range)"""
    if isinstance(rg, BlockMask):
        return rg.vis(rows, torch.arange(k0, k1, device=dev))
    ks = torch.arange(k0, k1, device=dev)[None, :]
    m = torch.zeros((len(rows), k1 - k0), dtype=torch.bool, device=dev)
    for j in range(rg.shape[1]):
        s = rg[rows, j, 0][:, None]; e = rg[rows, j, 1][:, None]
        m |= (ks >= s) & (ks < e)
    return m


def _kspan(rg, rows):
    """[lo, hi): the key span the rows can see at all (to skip invisible key chunks)"""
    if isinstance(rg, BlockMask):
        return 0, rg.S
    st, en = rg[rows, :, 0], rg[rows, :, 1]
    live = en > st
    lo = int(torch.where(live, st, torch.full_like(st, 1 << 62)).min()); hi = int(torch.where(live, en, torch.zeros_like(en)).max())
    return lo, max(lo, hi)


def _fwd_rows(q, k, v, rg, rows, scale, ck):
    """FP64 O [n, hq, d], LSE [n, hq] of the query rows `rows` (LongTensor)"""
    dev = q.device; hq, hk = q.shape[1], k.shape[1]; g = hq // hk
    Q = q[rows].double()                                                   # [n, hq, d]
    n = len(rows); m = torch.full((n, hq), -math.inf, dtype=torch.float64, device=dev)
    l = torch.zeros((n, hq), dtype=torch.float64, device=dev); acc = torch.zeros_like(Q)
    lo, hi = _kspan(rg, rows)
    for k0 in range(lo, hi, ck):
        k1 = min(hi, k0 + ck); vis = _vis(rg, rows, k0, k1, dev)
        if not vis.any():
            continue
        K = k[k0:k1].double().repeat_interleave(g, 1); V = v[k0:k1].double().repeat_interleave(g, 1)
        s = torch.einsum("nhd,khd->nhk", Q, K) * scale
        s = s.masked_fill(~vis[:, None, :], -math.inf)
        mn = torch.maximum(m, s.amax(-1)); a = torch.exp(m - mn); a = torch.where(torch.isfinite(mn), a, torch.zeros_like(a))
        p = torch.exp(s - mn[..., None]); p = torch.where(torch.isfinite(mn)[..., None], p, torch.zeros_like(p))
        l = l * a + p.sum(-1); acc = acc * a[..., None] + torch.einsum("nhk,khd->nhd", p, V); m = mn
    O = acc / l.clamp(min=1e-300)[..., None]
    return O, m + torch.log(l.clamp(min=1e-300))


def _dq_rows(q, k, v, do, rg, rows, scale, lse, D, ck):
    dev = q.device; hq, hk = q.shape[1], k.shape[1]; g = hq // hk
    Q = q[rows].double(); dO = do[rows].double(); L = lse[rows]; Dr = D[rows]
    dq = torch.zeros_like(Q); lo, hi = _kspan(rg, rows)
    for k0 in range(lo, hi, ck):
        k1 = min(hi, k0 + ck); vis = _vis(rg, rows, k0, k1, dev)
        if not vis.any():
            continue
        K = k[k0:k1].double().repeat_interleave(g, 1); V = v[k0:k1].double().repeat_interleave(g, 1)
        p = torch.exp(torch.einsum("nhd,khd->nhk", Q, K) * scale - L[..., None]) * vis[:, None, :]
        ds = p * (torch.einsum("nhd,khd->nhk", dO, V) - Dr[..., None])
        dq += torch.einsum("nhk,khd->nhd", ds, K) * scale
    return dq


def _dkv_rows(q, k, v, do, rg, keys, scale, lse, D, cq):
    """FP64 dK, dV [n, hk, d] of the key rows `keys`: every query row that sees them (all S rows, chunked)"""
    dev = q.device; S, hq, hk = q.shape[0], q.shape[1], k.shape[1]; g = hq // hk
    K = k[keys].double().repeat_interleave(g, 1); V = v[keys].double().repeat_interleave(g, 1)
    dk = torch.zeros_like(K); dv = torch.zeros_like(V)
    # first query row that can see any sampled key (with non-causal ranges, rows before min(keys) can see the keys too;
    # rows before kmin have every range end <= min(keys))
    if isinstance(rg, BlockMask):
        kmin = 0
    else:
        ok = torch.nonzero(rg[:, :, 1].max(1).values > int(keys.min())).flatten()
        kmin = int(ok[0]) if len(ok) else S
    for r0 in range(kmin, S, cq):                                          # (causal-type masks: rows < key never see it)
        rows = torch.arange(r0, min(S, r0 + cq), device=dev)
        vis = rg.vis(rows, keys) if isinstance(rg, BlockMask) else torch.zeros((len(rows), len(keys)), dtype=torch.bool, device=dev)
        for j in range(0 if isinstance(rg, BlockMask) else rg.shape[1]):
            s = rg[rows, j, 0][:, None]; e = rg[rows, j, 1][:, None]
            vis |= (keys[None, :] >= s) & (keys[None, :] < e)
        if not vis.any():
            continue
        Q = q[rows].double(); dO = do[rows].double()
        p = torch.exp(torch.einsum("nhd,khd->nhk", Q, K) * scale - lse[rows][..., None]) * vis[:, None, :]
        dv += torch.einsum("nhk,nhd->khd", p, dO)
        ds = p * (torch.einsum("nhd,khd->nhk", dO, V) - D[rows][..., None])
        dk += torch.einsum("nhk,nhd->khd", ds, Q) * scale
    n = len(keys)
    return dk.reshape(n, hk, g, -1).sum(2), dv.reshape(n, hk, g, -1).sum(2)


def check(q, k, v, do, ranges, scale, out, q_rows=None, kv_rows=None, group=None, n_sample=512, seed=0, ck=4096, cq=1024,
          lse_in=None, o_in=None):
    """-> dict tensor -> (rel L2, max abs err / max |ref|, sampled rows over all ranks). See the module docstring.
    ranges [S, R, 2]: one key mask per token for every head. GROUPED (InfLLM-v2: the selection differs per
    KV head): ranges [G, S, R, 2] with G = KV heads; group g = query heads [g*hq/G, (g+1)*hq/G) and KV head g; every group
    is checked on the same sampled rows / keys and the error sums of all groups are combined before the reduction."""
    import torch.distributed as dist
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if isinstance(ranges, (list, tuple)) and isinstance(ranges[0], BlockMask):    # per-head block masks
        rg = [m.to(dev) for m in ranges]
    else:
        rg = torch.as_tensor(np.asarray(ranges) if not torch.is_tensor(ranges) else ranges, dtype=torch.int64, device=dev)
    if not isinstance(rg, list) and rg.dim() == 3:
        sums = _sums(q, k, v, do, rg, scale, out, q_rows, kv_rows, group, n_sample, seed, ck, cq, lse_in, o_in)
    else:
        G, hq = len(rg), q.shape[1]; hs = hq // G
        assert k.shape[1] == G, (k.shape, G)
        sums = {}
        for g in range(G):
            hsl = slice(g * hs, (g + 1) * hs)
            og = {nm: (t[:, hsl] if nm in ("o", "dq") else t[:, g:g + 1]) for nm, t in out.items()}
            sg = _sums(q[:, hsl], k[:, g:g + 1], v[:, g:g + 1], do[:, hsl] if do is not None else None, rg[g], scale, og,
                       q_rows, kv_rows, group, n_sample, seed, ck, cq,
                       lse_in[:, hsl] if lse_in is not None else None, o_in[:, hsl] if o_in is not None else None)
            for nm, x in sg.items():
                if nm not in sums:
                    sums[nm] = list(x)
                else:                                            # e^2, ref^2 add; maxima max; rows once; non-finite add
                    t_ = sums[nm]; sums[nm] = [t_[0] + x[0], t_[1] + x[1], max(t_[2], x[2]), max(t_[3], x[3]), t_[4], t_[5] + x[5]]
    res = {}
    for nm in ("o", "dq", "dk", "dv"):
        t = torch.tensor(sums.get(nm, [0.0, 0.0, 0.0, 0.0, 0, 0]), dtype=torch.float64, device=dev)
        if group is not None:
            a = t[[0, 1, 4, 5]].clone(); b = t[[2, 3]].clone()
            dist.all_reduce(a, group=group); dist.all_reduce(b, op=dist.ReduceOp.MAX, group=group)
            t = torch.stack([a[0], a[1], b[0], b[1], a[2], a[3]])
        if t[4] > 0:
            res[nm] = dict(rel_l2=float(torch.sqrt(t[0] / t[1].clamp(min=1e-300))), max_rel=float(t[2] / t[3].clamp(min=1e-300)), rows=int(t[4]),
                           nonfinite_rows=int(t[5]))
    return res


def _sums(q, k, v, do, rg, scale, out, q_rows, kv_rows, group, n_sample, seed, ck, cq, lse_in, o_in):
    """per-rank error sums {name: [sum e^2, sum ref^2, max |e|, max |ref|, rows, non-finite rows]} for one mask"""
    import torch.distributed as dist
    dev = rg.device
    rank = dist.get_rank(group) if group is not None else 0
    world = dist.get_world_size(group) if group is not None else 1
    q, k, v = q.to(dev), k.to(dev), v.to(dev); do = do.to(dev) if do is not None else None
    S, hq = q.shape[0], q.shape[1]
    need_bwd = any(x in out for x in ("dq", "dk", "dv"))
    lse = D = None
    if need_bwd and lse_in is not None:                                    # given saved statistics (backward-only test)
        lse = lse_in.to(dev).double(); D = torch.empty_like(lse)
        for r0 in range(0, S, 4096):
            D[r0:r0 + 4096] = (do[r0:r0 + 4096].double() * o_in[r0:r0 + 4096].to(dev).double()).sum(-1)
    elif need_bwd:                                                         # 1. LSE, D of all rows (slices, all-gather)
        bnd = [S * i // world for i in range(world + 1)]
        mine = torch.arange(bnd[rank], bnd[rank + 1], device=dev)
        lse_s = torch.empty((len(mine), hq), dtype=torch.float64, device=dev); D_s = torch.empty_like(lse_s)
        for r0 in range(0, len(mine), 512):
            rows = mine[r0:r0 + 512]
            O, L = _fwd_rows(q, k, v, rg, rows, scale, ck)
            lse_s[r0:r0 + len(rows)] = L; D_s[r0:r0 + len(rows)] = (do[rows].double() * O).sum(-1)
        if group is not None:
            mx = max(bnd[i + 1] - bnd[i] for i in range(world))
            pad = lambda t: torch.cat([t, t.new_zeros((mx - len(t), hq))])          # noqa: E731
            gl = [torch.empty((mx, hq), dtype=torch.float64, device=dev) for _ in range(world)]
            gd = [torch.empty_like(gl[0]) for _ in range(world)]
            dist.all_gather(gl, pad(lse_s), group=group); dist.all_gather(gd, pad(D_s), group=group)
            lse = torch.cat([gl[i][:bnd[i + 1] - bnd[i]] for i in range(world)])
            D = torch.cat([gd[i][:bnd[i + 1] - bnd[i]] for i in range(world)])
        else:
            lse, D = lse_s, D_s
    rng = np.random.default_rng(seed + rank)
    sums = {}

    def acc(name, sysv, ref):
        e = (sysv.double().to(dev) - ref)
        bad = int((~torch.isfinite(e.reshape(e.shape[0], -1))).any(-1).sum())   # non-finite system rows (NaN-safe max below)
        sums[name] = [float((e * e).sum()), float((ref * ref).sum()), float(torch.nan_to_num(e.abs(), nan=float("inf")).max()),
                      float(ref.abs().max()), int(ref.shape[0]), bad]

    # (a rank owning only padding rows samples nothing but still joins the all-reduce)
    if q_rows is not None and len(q_rows) and ("o" in out or "dq" in out):  # 2-3. sampled query rows
        qr = np.asarray(q_rows); pick = np.sort(rng.choice(len(qr), min(n_sample, len(qr)), replace=False))
        rows = torch.as_tensor(qr[pick], device=dev)
        O, L = _fwd_rows(q, k, v, rg, rows, scale, ck)
        if "o" in out:
            acc("o", out["o"][torch.as_tensor(pick, device=out["o"].device)], O)
        if "dq" in out:
            acc("dq", out["dq"][torch.as_tensor(pick, device=out["dq"].device)], _dq_rows(q, k, v, do, rg, rows, scale, lse, D, ck))
    if kv_rows is not None and len(kv_rows) and ("dk" in out or "dv" in out):   # sampled key rows
        kr = np.asarray(kv_rows); pick = np.sort(rng.choice(len(kr), min(n_sample, len(kr)), replace=False))
        keys = torch.as_tensor(kr[pick], device=dev)
        dk, dv = _dkv_rows(q, k, v, do, rg, keys, scale, lse, D, cq)
        for nm, ref in (("dk", dk), ("dv", dv)):
            if nm in out:
                acc(nm, out[nm][torch.as_tensor(pick, device=out[nm].device)], ref)
    return sums


def check_virtual(q, k, v, do, ranges, scale, out, z, pos_rows=None, pos_kv=None, group=None, n_sample=512, seed=0,
                  ck=4096, cq=1024, lse_in=None, o_in=None):
    """FP64 check of a run in the VIRTUAL-HEAD layout (head split): the mask file z holds hv copies of the
    input's padded sequence (block-diagonal), copy g = heads [g*hq/hv, (g+1)*hq/hv) of the original input; position p of the
    virtual layout = (copy p // T, padded token p % T). q, k, v, do: the GLOBAL virtual tensors [hv*T, h/hv, d]; out: this
    rank's outputs (o / dq rows at virtual positions pos_rows, dk / dv rows at pos_kv); ranges: the ORIGINAL input's key
    ranges. Every copy is checked like an input of its own (same sampled-row rule, n_sample rows per copy), and the error
    sums of all copies are combined before the reduction (as the grouped mode of check)."""
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    hv = int(z["virtual_heads"]); Tv = q.shape[0]; T = Tv // hv
    if "tok_valid" in z.files:                       # real-token id per padded position of ONE copy (tok_valid is tiled)
        tv = z["tok_valid"][:T].astype(bool); r_ = np.full(T, -1, np.int64); r_[tv] = np.arange(int(tv.sum()))
    else:
        r_ = np.arange(T)
    rid = torch.as_tensor(r_, device=dev)
    rv = rid >= 0
    rg = torch.as_tensor(np.asarray(ranges), dtype=torch.int64, device=dev)
    pr = torch.as_tensor(pos_rows, device=dev) if pos_rows is not None else None
    pk = torch.as_tensor(pos_kv, device=dev) if pos_kv is not None else None
    sums = {}
    for g in range(hv):
        sl = slice(g * T, (g + 1) * T)
        og, qr_, kr_ = {}, None, None
        if pr is not None:
            sel = (pr >= g * T) & (pr < (g + 1) * T)
            rl = rid[(pr[sel] - g * T)]; ok = rl >= 0; qr_ = rl[ok].cpu().numpy()
            for nm in ("o", "dq"):
                if nm in out:
                    og[nm] = out[nm][sel.to(out[nm].device)][ok.to(out[nm].device)]
        if pk is not None:
            sel = (pk >= g * T) & (pk < (g + 1) * T)
            rl = rid[(pk[sel] - g * T)]; ok = rl >= 0; kr_ = rl[ok].cpu().numpy()
            for nm in ("dk", "dv"):
                if nm in out:
                    og[nm] = out[nm][sel.to(out[nm].device)][ok.to(out[nm].device)]
        pick = lambda t: t[sl][rv.to(t.device)] if t is not None else None          # noqa: E731
        sg = _sums(pick(q), pick(k), pick(v), pick(do), rg, scale, og, qr_, kr_, group, n_sample, seed + 7919 * g, ck, cq,
                   pick(lse_in), pick(o_in))
        for nm, x in sg.items():
            if nm not in sums:
                sums[nm] = list(x)
            else:                                    # e^2, ref^2 add; maxima max; rows add (different rows per copy)
                t_ = sums[nm]; sums[nm] = [t_[0] + x[0], t_[1] + x[1], max(t_[2], x[2]), max(t_[3], x[3]), t_[4] + x[4], t_[5] + x[5]]
    return _reduce(sums, group, dev)


def _reduce(sums, group, dev):
    import torch.distributed as dist
    res = {}
    for nm in ("o", "dq", "dk", "dv"):
        t = torch.tensor(sums.get(nm, [0.0, 0.0, 0.0, 0.0, 0, 0]), dtype=torch.float64, device=dev)
        if group is not None:
            a = t[[0, 1, 4, 5]].clone(); b = t[[2, 3]].clone()
            dist.all_reduce(a, group=group); dist.all_reduce(b, op=dist.ReduceOp.MAX, group=group)
            t = torch.stack([a[0], a[1], b[0], b[1], a[2], a[3]])
        if t[4] > 0:
            res[nm] = dict(rel_l2=float(torch.sqrt(t[0] / t[1].clamp(min=1e-300))), max_rel=float(t[2] / t[3].clamp(min=1e-300)), rows=int(t[4]),
                           nonfinite_rows=int(t[5]))
    return res


def ranges_from_topk(topk, block=64):
    """InfLLM-v2: selected block ids [G, S, n] (sorted, -1 = none) -> grouped ranges [G, S, n, 2]: token i of group g sees
    keys [block * b, min(block * b + block, i + 1)) for every selected b (causal inside the diagonal block)"""
    t = torch.as_tensor(topk).long()
    S = t.shape[1]; i = torch.arange(S, device=t.device)[None, :, None]
    st = (t * block).clamp(min=0); en = torch.minimum(st + block, i + 1)
    live = t >= 0
    return torch.stack([torch.where(live, st, torch.zeros_like(st)), torch.where(live, en, torch.zeros_like(en))], -1)


def selftest():
    """CPU / 1 GPU: a small star mask, GQA 4:1, against a dense FP64 autograd reference"""
    torch.manual_seed(0); S, hq, hk, d = 700, 8, 2, 32; P_, R = 300, [100, 150, 150]
    rg = np.zeros((S, 2, 2), np.int64); rg[:P_, 0, 1] = np.arange(1, P_ + 1); pos = P_
    for r in R:
        j = np.arange(r); rg[pos:pos + r, 0, 1] = P_; rg[pos:pos + r, 1, 0] = pos; rg[pos:pos + r, 1, 1] = pos + j + 1; pos += r
    q, k, v, do = (torch.randn(S, h, d, dtype=torch.float64) * .5 for h in (hq, hk, hk, hq))
    vis = torch.zeros(S, S, dtype=torch.bool)
    for j in range(2):
        vis |= (torch.arange(S)[None] >= torch.as_tensor(rg[:, j, 0])[:, None]) & (torch.arange(S)[None] < torch.as_tensor(rg[:, j, 1])[:, None])
    qq, kk, vv = (x.clone().requires_grad_() for x in (q, k, v)); g = hq // hk; sc = d ** -0.5
    s = torch.einsum("qhd,khd->hqk", qq, kk.repeat_interleave(g, 1)) * sc
    o = torch.einsum("hqk,khd->qhd", torch.softmax(s.masked_fill(~vis[None], -math.inf), -1), vv.repeat_interleave(g, 1))
    o.backward(do)
    out = dict(o=o.detach().float(), dq=qq.grad.float(), dk=kk.grad.float(), dv=vv.grad.float())
    e = check(q, k, v, do, rg, sc, out, q_rows=np.arange(S), kv_rows=np.arange(S), n_sample=256, ck=128, cq=64)
    print("selftest (fp32-rounded exact outputs, expect ~1e-7):", {kk_: round(vv_["rel_l2"], 9) for kk_, vv_ in e.items()})
    assert all(x["rel_l2"] < 1e-6 for x in e.values()), e


if __name__ == "__main__":
    selftest()
