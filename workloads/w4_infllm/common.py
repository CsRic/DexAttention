"""Context-parallel layout and selection of W4, shared by every system (Dex and the baselines).

Layout: the sequence is split into 2P zigzag chunks (multiples of 64 tokens, remainder spread evenly); rank r owns
chunks r and 2P-1-r. Selection under CP: each rank compresses its own key windows (plus the first HALO K rows of the
next chunk), the pooled keys are all-gathered (1/16 + 1/64 of K), and the selection runs per own chunk on the same
compressed keys as a selection on the gathered K (each window is compress_k over the same K rows).
Row layouts: q / o [L, 32, 128] token layout <-> [L * 16, 2, 128] "group rows" (the official InfLLM-v2 varlen layout);
selections [2 KV heads, L, 96] int32 global 64-token block ids, -1 = none.
"""
import time

import numpy as np
import torch
import torch.distributed as dist
from infllm_v2 import max_pooling_1d_varlen

from dex.common.cp_layout import owned
from dex.common.env import cfg
from workloads.w4_infllm.minicpm_model import SPARSE, compress_k
from workloads.w4_infllm.stage1_fast import stage1_fast

ALIGN = 64                    # chunk alignment = InfLLM-v2 block size
BLK = 64
GH, HKV_D = 16, 128           # query heads per KV head, head dim
HALO = 64                     # K rows of the next chunk a chunk's last compressed windows reach into


def gpus_per_node(P):
    return min(int(cfg("DEX_GPUS_PER_NODE")), P)


def zigzag_chunks(S, P, align=ALIGN):
    n = 2 * P
    nb = (S + align - 1) // align
    q, rem = divmod(nb, n)
    b = [0]
    for i in range(n):
        b.append(min(S, b[-1] + (q + (i < rem)) * align))
    return [(b[i], b[i + 1]) for i in range(n)]


def chunk_owner(ci, P):
    """(rank, slot in that rank's owned list) of chunk ci"""
    return (ci, 0) if ci < P else (2 * P - 1 - ci, 1)


def chunk_rank(ci, P):
    return ci if ci < P else 2 * P - 1 - ci


def key_counts(c1, sp=SPARSE):
    """compressed keys visible to a query chunk ending at c1: (n1 fine, n2 coarse)"""
    ks, st = sp["kernel_size"], sp["kernel_stride"]
    return max(0, (c1 - ks) // st + 1), max(0, (c1 - 4 * ks) // (4 * st) + 1)


def pick_blocks(score, c0, n1, sp=SPARSE):
    """stage-1 scores [Hk, L, n1] -> top-96 64-token blocks per (KV head, token) (official pooling + top-k)."""
    L = score.size(1)
    dev = score.device
    cu_q = torch.tensor([0, L], dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0, n1], dtype=torch.int32, device=dev)
    lb = sp["window_size"] // sp["block_size"]
    bs = max_pooling_1d_varlen(score.contiguous(), cu_q, cu_k, torch.tensor([c0], dtype=torch.int32, device=dev),
                               L, n1, local_blocks=lb, init_blocks=sp["init_blocks"],
                               block_size=sp["block_size"], stride=sp["kernel_stride"])
    k = min(sp["topk"] + lb, bs.size(-1))
    idx = bs.topk(k, dim=-1).indices.sort(-1).values
    qb = (torch.arange(L, device=dev) + c0) // sp["block_size"]
    idx[idx > qb[None, :, None]] = -1
    return idx.to(torch.int32)


def select_chunk(q, kc, kc2, c0, sp=SPARSE):
    """selection for queries q = positions [c0, c0+L) given the full-sequence compressed keys (only entries fully
    inside [0, c0+L) are visible); stage 1 with the large-tile kernel (same math as the official stage 1 with a different
    tiling: scores agree up to accumulation order), then the official block pick"""
    L = q.size(0)
    end = c0 + L
    ks, st = sp["kernel_size"], sp["kernel_stride"]
    n1 = max(0, (end - ks) // st + 1)
    n2 = max(0, (end - 4 * ks) // (4 * st) + 1)
    score = stage1_fast(q, kc[:n1].contiguous(), kc2[:n2].contiguous())
    return pick_blocks(score, c0, n1, sp)


def windows(c0, c1, S, size, st):
    """compressed-key windows (start = st*j) that START inside chunk [c0, c1): [j0, j1)"""
    j0 = (c0 + st - 1) // st
    j1 = min((c1 - 1) // st + 1, (S - size) // st + 1)
    return j0, max(j0, j1)


def kinds():
    ks, st = SPARSE["kernel_size"], SPARSE["kernel_stride"]
    return ((ks, st), (4 * ks, 4 * st))                 # kind 0: k_c (32/16, fine), kind 1: k_c2 (128/64, coarse)


def halo_allgather(k_own, my):
    """first HALO K rows of every chunk (small all-gather): heads[ci] -> [HALO, Hkv, D]"""
    P = dist.get_world_size()
    heads = torch.stack([k_own[ci][:HALO] for ci in my])
    hb = heads.new_empty(P, *heads.shape)
    dist.all_gather_into_tensor(hb, heads)
    return lambda ci: hb[chunk_owner(ci, P)[0], chunk_owner(ci, P)[1]]


def own_windows(k_own, chunks, my, S, head_of):
    """this rank's compressed-key windows: {kind: [(ci, j0, j1, tensor [j1-j0, Hkv, D])]} (windows START in chunk ci;
    the last ones reach HALO rows into chunk ci+1, given by head_of(ci+1)). Each window is compress_k over the same K rows
    as in compress_k(K_full), so the values match it up to the reduction order of the mean."""
    P = dist.get_world_size()
    out = {}
    for kind, (size, stp) in enumerate(kinds()):
        out[kind] = []
        for ci in my:
            c0, c1 = chunks[ci]
            j0, j1 = windows(c0, c1, S, size, stp)
            if j1 == j0:
                continue
            ext = k_own[ci]
            if ci + 1 < 2 * P:
                ext = torch.cat([ext, head_of(ci + 1)], 0)
            out[kind].append((ci, j0, j1, compress_k(ext[j0 * stp - c0:(j1 - 1) * stp + size - c0], size, stp)))
    return out


def compressed_keys_cp(k_own, chunks, my, S):
    """k_c / k_c2 of the whole sequence without gathering K: own windows (halo all-gather), then the pooled keys are
    all-gathered. The same values as compress_k(K_full) up to the reduction order of the mean."""
    P = dist.get_world_size()
    kinds_ = kinds()
    ow = own_windows(k_own, chunks, my, S, halo_allgather(k_own, my))
    parts = [t for kind in range(2) for (_, _, _, t) in ow[kind]]
    ref = next(iter(k_own.values()))
    cnt = [[windows(*chunks[ci], S, size, stp) for size, stp in kinds_] for ci in range(2 * P)]
    per_rank = [sum(j1 - j0 for ci in owned(r, P) for j0, j1 in cnt[ci]) for r in range(P)]
    mine = torch.cat(parts, 0) if parts else ref.new_zeros(0, *ref.shape[1:])
    pad = mine.new_zeros(max(per_rank), *mine.shape[1:])
    pad[:mine.size(0)] = mine
    buf = pad.new_empty(P, *pad.shape)
    dist.all_gather_into_tensor(buf, pad)
    out = []
    for kind in range(2):
        segs = []
        for ci in range(2 * P):                         # global window order = chunk order
            r, _ = chunk_owner(ci, P)
            o = 0
            for kk in range(2):                          # offset inside rank r's contribution: kinds x own chunks
                for cj in owned(r, P):
                    j0, j1 = cnt[cj][kk]
                    if kk == kind and cj == ci:
                        segs.append(buf[r, o:o + j1 - j0])
                    o += j1 - j0
        out.append(torch.cat(segs, 0))
    return out[0], out[1]


def select_cp(qs, ks, chunks, my, S):
    """{own chunk: selection [2, L, K]} from the own q / k chunks (compressed keys exchanged, official selection)"""
    kc, kc2 = compressed_keys_cp(ks, chunks, my, S)
    return {ci: select_chunk(qs[ci], kc, kc2, chunks[ci][0]) for ci in my}


def to_rows(q):
    """[L, 32, 128] -> official group-row layout [L * 16, 2, 128]"""
    return q.reshape(q.size(0), -1, GH, HKV_D).transpose(1, 2).reshape(-1, q.size(1) // GH, HKV_D).contiguous()


def from_rows(o, nh):
    return o.reshape(-1, GH, nh // GH, HKV_D).transpose(1, 2).reshape(-1, nh, HKV_D)


def filter_causal(sel, pos):
    """drop blocks after the token's own block (-1). sel [2, L, K] int32, pos [L] global token ids."""
    qb = (pos // BLK)[None, :, None]
    return torch.where(sel > qb, torch.full_like(sel, -1), sel).to(torch.int32).contiguous()


def pad_concat(sel, my):
    """{own chunk: [2, L, K_ci]} -> [2, L_total, max K] (-1 padded), chunks in `my` order"""
    kmax = max(x.size(-1) for x in sel.values())
    return torch.cat([torch.nn.functional.pad(sel[ci], (0, kmax - sel[ci].size(-1)), value=-1) for ci in my], 1)


class Timer:
    """phase timer of the operator tests. A barrier mark synchronizes and closes a phase on every rank; with
    events=True, a non-barrier mark only records a CUDA event (no sync: the overlap of side streams and transports
    stays intact) and the next barrier mark resolves the pending events."""

    def __init__(self, on, events=False):
        self.on, self.t, self.ev, self.pend = on, {}, events, []

    def start(self):
        if self.on:
            torch.cuda.synchronize(); dist.barrier(); self._l = time.perf_counter()
            if self.ev:
                self._e = torch.cuda.Event(enable_timing=True); self._e.record()

    def mark(self, k, barrier=True):
        if not self.on:
            return
        if self.ev and not barrier:
            e = torch.cuda.Event(enable_timing=True); e.record(); self.pend.append((k, e))
            return
        torch.cuda.synchronize()
        if self.ev and self.pend:
            prev = self._e
            for kk, e in self.pend:
                self.t[kk] = self.t.get(kk, 0.0) + prev.elapsed_time(e) / 1e3
                prev = e
            end = torch.cuda.Event(enable_timing=True); end.record(); end.synchronize()
            self._l = time.perf_counter() - prev.elapsed_time(end) / 1e3     # k gets the time from the last event on
            self.pend = []
        if barrier:
            dist.barrier()
        n = time.perf_counter()
        self.t[k] = self.t.get(k, 0.0) + n - self._l
        self._l = n
        if self.ev:
            self._e = torch.cuda.Event(enable_timing=True); self._e.record()


def block_owner(chunks, S, P):
    nb = (S + ALIGN - 1) // ALIGN
    own = np.empty(nb, np.int64)
    for ci, (a, b) in enumerate(chunks):
        own[a // ALIGN:(b + ALIGN - 1) // ALIGN] = chunk_rank(ci, P)
    return own
