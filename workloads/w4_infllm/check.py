"""W4 correctness: the shared FP64 sampled-row check (dex.common.fp64_check, grouped mode: the selection differs per KV
head) for every W4 system.

Mask = the selection the system used ([2 KV heads, tokens, 96] block ids, -1 none; 64-token blocks, causal inside the
diagonal block) -> per-group key ranges. Each rank checks its own rows:
  forward : O on 512 sampled query rows per rank
  backward: dQ on 512 sampled query rows, dK / dV on 512 sampled keys per rank, given the forward statistics the backward
            was run with (its O and LSE)
Global tensors are assembled from every rank's owned tokens (all-gather), on every rank."""
import torch
import torch.distributed as dist

from dex.common import fp64_check as F64

BLOCK = 64


def gather_tok(x_own, pos, S):
    """x_own [n_own, ...] at global token ids pos [n_own] (this rank) -> the global [S, ...] tensor on every rank"""
    P = dist.get_world_size(); dev = x_own.device
    n = torch.tensor([x_own.shape[0]], device=dev); ns = [torch.zeros_like(n) for _ in range(P)]
    dist.all_gather(ns, n); mx = int(max(t.item() for t in ns))
    pad = x_own.new_zeros((mx,) + tuple(x_own.shape[1:])); pad[:x_own.shape[0]] = x_own
    pp = torch.full((mx,), -1, dtype=torch.long, device=dev); pp[:len(pos)] = pos.long()
    xs = [torch.empty_like(pad) for _ in range(P)]; ps = [torch.empty_like(pp) for _ in range(P)]
    dist.all_gather(xs, pad); dist.all_gather(ps, pp)
    out = x_own.new_zeros((S,) + tuple(x_own.shape[1:]))
    for x, p, k in zip(xs, ps, ns):
        k = int(k.item()); out[p[:k]] = x[:k]
    return out


def gather_sel(sel_own, pos, S):
    """selection [2, n_own, K] -> global [2, S, K]"""
    return gather_tok(sel_own.permute(1, 0, 2).contiguous(), pos, S).permute(1, 0, 2).contiguous()


def fwd_check(loc, outs, sel, chunks, my, S, scale):
    """loc[ci] = (q [L, 32, D], k [L, 2, D], v [L, 2, D]); outs[ci] = O [L, 32 * D] (token layout); sel[ci] [2, L, 96]"""
    dev = loc[my[0]][0].device
    pos = torch.cat([torch.arange(*chunks[ci], device=dev) for ci in my])
    q = torch.cat([loc[ci][0] for ci in my]); k = torch.cat([loc[ci][1] for ci in my]); v = torch.cat([loc[ci][2] for ci in my])
    Q, K, V = gather_tok(q, pos, S), gather_tok(k, pos, S), gather_tok(v, pos, S)
    TK = gather_sel(torch.cat([sel[ci] for ci in my], 1).to(torch.int32), pos, S)
    rg = F64.ranges_from_topk(TK.to(dev), BLOCK)
    O = torch.cat([outs[ci].reshape(outs[ci].shape[0], Q.shape[1], Q.shape[2]) for ci in my])
    e = F64.check(Q, K, V, None, rg, scale, dict(o=O), q_rows=pos.cpu().numpy(), group=dist.group.WORLD)
    del Q, K, V, TK, rg
    torch.cuda.empty_cache()
    return e


def bwd_check(q_tok, k_own, v_own, do_tok, o_tok, lse_tok, sel_own, pos, S, scale, dq_tok, dk_own, dv_own):
    """this rank's owned tokens (token layout): q / dO / O / dQ [n, 32, D], LSE [n, 32], K / V / dK / dV [n, 2, D],
    selection [2, n, 96]; pos [n] global ids"""
    dev = q_tok.device
    Q, K, V = gather_tok(q_tok, pos, S), gather_tok(k_own, pos, S), gather_tok(v_own, pos, S)
    dO, O, L = gather_tok(do_tok, pos, S), gather_tok(o_tok, pos, S), gather_tok(lse_tok.float(), pos, S)
    rg = F64.ranges_from_topk(gather_sel(sel_own.to(torch.int32), pos, S).to(dev), BLOCK)
    p = pos.cpu().numpy()
    e = F64.check(Q, K, V, dO, rg, scale, dict(dq=dq_tok, dk=dk_own, dv=dv_own), q_rows=p, kv_rows=p,
                  group=dist.group.WORLD, lse_in=L, o_in=O)
    del Q, K, V, dO, O, L, rg
    torch.cuda.empty_cache()
    return e
