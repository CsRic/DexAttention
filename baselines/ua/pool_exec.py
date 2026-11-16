"""Forward of the decomposed executor for tile masks: global steps (a step's input point-to-point is launched during the
previous step's compute and waited before its own; partial O returns are launched after the step's compute and merged
when drained), on one launch of host-level NCCL per step.

Pools: every Q / K / V block a rank ever holds has a slot in one pool per tensor (Q [nQ B, H, D], K / V [nK B, Hk, D]),
slots in global block order; received blocks land directly in their slot, sends read slot views, and a rectangle's key
runs map to pool rows, so one flash-attn call reads them in place (strided-batch forward for rows with equal runs at a
constant step, flash_attn_with_kvcache with per-row cache_leftpad / cache_seqlens otherwise). Pools are allocated and owned
blocks loaded outside the timed call (load()).
Partial results go into an fp32 accumulator pool over the rank's Q rows (out [nQ B, H, D], lse [H, nQ B]): the first writer
of a row assigns, later ones merge by LSE in place. "First writer" flags are recorded on the first call and reused."""
import numpy as np
import torch
import torch.distributed as dist
import flash_attn_2_cuda as _fa
from flash_attn import flash_attn_with_kvcache, flash_attn_varlen_func

T = 128
EMPTY_LSE = -1.0e4
GATHER_RUNS = 3
GATHER_MAX_TILES = 8192   # key tiles per packed call


def _launch(ops):
    if not ops:
        return []
    return dist.batch_isend_irecv([dist.P2POp(fn, t, p) for (fn, t, p) in ops])


def _wait(reqs):
    for r in reqs:
        r.wait()


class PoolTiles:
    def __init__(self, mask, sched, H, Hk, D, dtype, device):
        self.m = np.asarray(mask, np.int8)
        self.B = sched.block_tokens; self.k = self.B // T
        self.H, self.Hk, self.D = H, Hk, D
        qs, ks = set(sched.init_q) | set(sched.owned_q), set(sched.init_kv)
        for st in sched.steps:
            for c in st.comp:
                qs.update(c.q_blocks); ks.update(c.kv_blocks)
            for c in st.comm:
                if c.io == "i" and c.is_recv(sched.rank):
                    (qs if c.rc == "r" else ks).update(c.blocks)
        self.qb, self.kb = sorted(qs), sorted(ks)
        self.qslot = {b: i for i, b in enumerate(self.qb)}; self.kslot = {b: i for i, b in enumerate(self.kb)}
        nq, nk = len(self.qb) * self.B, len(self.kb) * self.B
        self.q = torch.empty(nq, H, D, dtype=dtype, device=device)
        self.kp = torch.empty(nk, Hk, D, dtype=dtype, device=device)
        self.vp = torch.empty(nk, Hk, D, dtype=dtype, device=device)
        self.acc = torch.zeros(nq, H, D, dtype=torch.float32, device=device)
        self.lse = torch.full((H, nq), EMPTY_LSE, dtype=torch.float32, device=device)
        self.dev = device
        self.cache, self.flags, self.frozen, self.cursor = {}, [], False, 0
        self.touched = np.zeros(len(self.qb) * self.k, bool)
        self.untouched = None

    # ---- pool views -------------------------------------------------------------------------------------------
    def qrows(self, b):
        s = self.qslot[b] * self.B
        return s, s + self.B

    def qview(self, b):
        s, e = self.qrows(b); return self.q[s:e]

    def kvview(self, b):
        s = self.kslot[b] * self.B; return self.kp[s:s + self.B], self.vp[s:s + self.B]

    def load(self, Q_local, KV_local):
        """owned blocks into their slots (outside the timed call). Q_local[b]: [1, B, H, D]; KV_local[b]: (K, V)."""
        for b, x in Q_local.items():
            self.qview(b).copy_(x[0])
        for b, (kx, vx) in KV_local.items():
            kk, vv = self.kvview(b); kk.copy_(kx[0]); vv.copy_(vx[0])

    # ---- first-writer bookkeeping ------------------------------------------------------------------------------
    def begin(self):
        self.cursor = 0
        if self.frozen and len(self.untouched):            # rows nobody writes (none in a causal mask) stay empty
            for t in self.untouched:
                self.acc[t * T:(t + 1) * T].zero_(); self.lse[:, t * T:(t + 1) * T].fill_(EMPTY_LSE)
        elif not self.frozen:
            self.touched[:] = False

    def end(self):
        if not self.frozen:
            self.frozen = True
            self.untouched = np.flatnonzero(~self.touched)

    def _first(self, tiles):
        if self.frozen:
            f = self.flags[self.cursor]
        else:
            f = not self.touched[tiles].any(); self.touched[tiles] = True; self.flags.append(f)
        self.cursor += 1
        return f

    # ---- merge into the accumulator ------------------------------------------------------------------------------
    def merge(self, rows, o, l, first):
        """rows: (start, end) row slice or a LongTensor of rows; o [r, H, D] (any float dtype), l [H, r] fp32."""
        if isinstance(rows, tuple):
            o0, l0 = self.acc[rows[0]:rows[1]], self.lse[:, rows[0]:rows[1]]
            if first:
                o0.copy_(o); l0.copy_(l)
            else:
                o0.lerp_(o.float(), torch.sigmoid(l - l0).t().unsqueeze(-1)); l0.copy_(torch.logaddexp(l0, l))
            return
        if first:
            self.acc.index_copy_(0, rows, o.float()); self.lse.index_copy_(1, rows, l)
            return
        o0, l0 = self.acc.index_select(0, rows), self.lse.index_select(1, rows)
        o0.lerp_(o.float(), torch.sigmoid(l - l0).t().unsqueeze(-1))
        self.acc.index_copy_(0, rows, o0); self.lse.index_copy_(1, rows, torch.logaddexp(l0, l))

    def merge_partial(self, b, o, l):
        """a received partial O of q block b: o [B, H, D], l [H, B]."""
        s, e = self.qrows(b)
        self.merge((s, e), o, l, self._first(np.arange(s // T, e // T)))

    # ---- one rectangle -------------------------------------------------------------------------------------------
    def _plan(self, qb, kvb):
        k = self.k
        gq = np.concatenate([np.arange(b * k, (b + 1) * k) for b in qb])
        pq = np.concatenate([np.arange(self.qslot[b] * k, (self.qslot[b] + 1) * k) for b in qb])
        gk = np.concatenate([np.arange(b * k, (b + 1) * k) for b in sorted(kvb)])
        pk = np.concatenate([np.arange(self.kslot[b] * k, (self.kslot[b] + 1) * k) for b in sorted(kvb)])
        sub = self.m[np.ix_(gq, gk)]
        rowruns = []
        for r in range(len(gq)):
            cols = np.flatnonzero(sub[r])
            if not len(cols):
                continue
            pos = pk[cols]                                  # pool tile of every live key tile (increasing)
            cuts = np.flatnonzero((np.diff(pos) != 1) | (sub[r, cols[:-1]] == 2)) + 1
            runs_ = []
            for run, rp in zip(np.split(cols, cuts), np.split(pos, cuts)):
                causal = bool(sub[r, run[-1]] == 2)
                assert not (sub[r, run[:-1]] == 2).any()
                runs_.append((int(pq[r]), int(rp[0]), int(rp[-1]) + 1, causal))
            rowruns.append(runs_)
        # fragmented rows: one call per run would be one kernel per run index; when some row of the
        # rectangle has more than GATHER_RUNS full-tile runs, every multi-run row instead puts all its full tiles in
        # ONE gathered sequence (K/V copied tile by tile), all such rows in one packed varlen call
        items, gathered = [], []
        frag = max((sum(1 for x in rr if not x[3]) for rr in rowruns), default=0) > GATHER_RUNS
        for rr in rowruns:
            full = [x for x in rr if not x[3]]
            if frag and len(full) > 1:
                gathered.append((rr[0][0], [t for x in full for t in range(x[1], x[2])]))
                items += [x for x in rr if x[3]]
            else:
                items += rr
        items.sort(key=lambda x: (x[3], x[2] - x[1], x[0], x[1]))
        ops, singles, i = [], [], 0
        while i < len(items):
            r0, a0, b0, c0 = items[i]; L = b0 - a0
            j = i + 1; d = None
            while j < len(items):
                r, a, b, c = items[j]
                if c != c0 or b - a != L or r != items[j - 1][0] + 1:
                    break
                step = a - items[j - 1][1]
                if step < 0 or (d is not None and step != d):
                    break
                d = step; j += 1
            if j - i >= 2:
                ops.append(dict(kind="g", r0=r0, nb=j - i, a0=a0, L=L, d=d, causal=c0))
            else:
                singles.append(items[i])
            i = j
        ops.sort(key=lambda o: -o["nb"] * o["L"])
        cnt, layers = {}, {}
        for r, a, b, c in sorted(singles):
            li = cnt.get(r, 0); cnt[r] = li + 1
            layers.setdefault((li, c), []).append((r, a, b))
        for (li, c), its in sorted(layers.items()):
            rr = [r for r, _, _ in its]
            contig = rr == list(range(rr[0], rr[0] + len(rr)))
            ops.append(dict(kind="p", tiles=rr,
                            rows=None if contig else torch.cat([torch.arange(r * T, (r + 1) * T) for r in rr]).to(self.dev),
                            qtiles=None if contig else torch.as_tensor(rr, device=self.dev),
                            lp=torch.tensor([a * T for _, a, _ in its], dtype=torch.int32, device=self.dev),
                            sl=torch.tensor([b * T for _, _, b in its], dtype=torch.int32, device=self.dev), causal=c))
        # one packed varlen call per chunk of gathered rows; a chunk copies at most GATHER_MAX_TILES key tiles (the copy
        # is K and V of every gathered tile: 2 * T * Hk * D * 2 bytes each -- bounded so a large tree does not OOM)
        chunks, cur, nt = [], [], 0
        for g in gathered:
            if cur and nt + len(g[1]) > GATHER_MAX_TILES:
                chunks.append(cur); cur, nt = [], 0
            cur.append(g); nt += len(g[1])
        if cur:
            chunks.append(cur)
        for gch in reversed(chunks):
            rr = [r for r, _ in gch]
            keys = np.concatenate([np.asarray(ts) for _, ts in gch])          # pool TILE ids (tile-contiguous copy)
            ck = np.concatenate([[0], np.cumsum([len(ts) * T for _, ts in gch])])
            ops.insert(0, dict(kind="x", tiles=rr,
                               rows=torch.cat([torch.arange(r * T, (r + 1) * T) for r in rr]).to(self.dev),
                               keys=torch.as_tensor(keys, device=self.dev), qt=torch.as_tensor(rr, device=self.dev),
                               cq=torch.arange(0, (len(rr) + 1) * T, T, dtype=torch.int32, device=self.dev),
                               ck=torch.as_tensor(ck, dtype=torch.int32, device=self.dev),
                               mk=int(max(len(ts) for _, ts in gch) * T)))
        for op in ops:
            op["tl"] = np.arange(op["r0"], op["r0"] + op["nb"]) if op["kind"] == "g" else np.asarray(op["tiles"])
        return ops

    def compute(self, qb, kvb, scale):
        key = (tuple(qb), tuple(kvb))
        ops = self.cache.get(key)
        if ops is None:
            ops = self.cache[key] = self._plan(qb, kvb)
        H, Hk, D = self.H, self.Hk, self.D
        rs = Hk * D
        for op in ops:
            first = self._first(op["tl"])
            if op["kind"] == "g":
                r0, nb, a0, L, d = op["r0"], op["nb"], op["a0"], op["L"], op["d"]
                qq = self.q[r0 * T:(r0 + nb) * T].view(nb, T, H, D)
                kk = self.kp.as_strided((nb, L * T, Hk, D), (d * T * rs, rs, D, 1), self.kp.storage_offset() + a0 * T * rs)
                vv = self.vp.as_strided((nb, L * T, Hk, D), (d * T * rs, rs, D, 1), self.vp.storage_offset() + a0 * T * rs)
                o, l = _fa.fwd(qq, kk, vv, None, None, 0.0, scale, op["causal"], -1, -1, 0.0, False, None)[:2]
                self.merge((r0 * T, (r0 + nb) * T), o.view(nb * T, H, D), l.permute(1, 0, 2).reshape(H, nb * T), first)
            elif op["kind"] == "x":                             # gathered fragmented rows, one varlen call
                # tile-granular gathers through a float64 view (same bytes; index_select is faster on 8-byte elements)
                g64 = lambda x, idx, h: x.view(-1, T * h * D).view(torch.float64).index_select(0, idx).view(x.dtype).view(-1, h, D)  # noqa: E731
                qq = g64(self.q, op["qt"], H)
                kk = g64(self.kp, op["keys"], Hk)
                vv = g64(self.vp, op["keys"], Hk)
                o, l, _ = flash_attn_varlen_func(qq, kk, vv, op["cq"], op["ck"], T, op["mk"], softmax_scale=scale,
                                                 causal=False, return_attn_probs=True)
                self.merge(op["rows"], o, l[:, :o.shape[0]].float(), first)
            else:
                nb = len(op["tiles"]); r0 = op["tiles"][0]
                qt = self.q.view(-1, T, H, D)
                qq = qt[r0:r0 + nb] if op["qtiles"] is None else qt.index_select(0, op["qtiles"])
                kc = self.kp.unsqueeze(0).expand(nb, -1, -1, -1); vc = self.vp.unsqueeze(0).expand(nb, -1, -1, -1)
                o, l = flash_attn_with_kvcache(qq, kc, vc, cache_seqlens=op["sl"], cache_leftpad=op["lp"],
                                               softmax_scale=scale, causal=op["causal"], num_splits=1,
                                               return_softmax_lse=True)
                rows = (r0 * T, (r0 + nb) * T) if op["qtiles"] is None else op["rows"]
                self.merge(rows, o.reshape(nb * T, H, D), l.permute(1, 0, 2).reshape(H, nb * T), first)


def fusion_forward_pooled(sched, pt, softmax_scale, partial_dtype=torch.float32, out_dtype=torch.float16):
    """One forward call on pooled buffers (pt = PoolTiles, owned blocks loaded). Returns {owned q block: [1, B, H, D]}."""
    rank, B, H, D = sched.rank, pt.B, pt.H, pt.D
    pt.begin()

    def build_comm(step, io):
        ops, merges = [], []
        for c in step.comm:
            if c.io != io:
                continue
            for blk in c.blocks:
                if io == "i" and c.rc == "r":
                    if c.is_send(rank):
                        ops.append((dist.isend, pt.qview(blk), c.dst))
                    elif c.is_recv(rank):
                        ops.append((dist.irecv, pt.qview(blk), c.src))
                elif io == "i":
                    kk, vv = pt.kvview(blk)
                    if c.is_send(rank):
                        ops.append((dist.isend, kk, c.dst)); ops.append((dist.isend, vv, c.dst))
                    elif c.is_recv(rank):
                        ops.append((dist.irecv, kk, c.src)); ops.append((dist.irecv, vv, c.src))
                else:                                           # partial O + LSE back to the owner
                    if c.is_send(rank):
                        s, e = pt.qrows(blk)
                        ops.append((dist.isend, pt.acc[s:e].to(partial_dtype), c.dst))
                        ops.append((dist.isend, pt.lse[:, s:e].to(partial_dtype).contiguous(), c.dst))
                    elif c.is_recv(rank):
                        ob = torch.empty(B, H, D, device=pt.dev, dtype=partial_dtype)
                        lb = torch.empty(H, B, device=pt.dev, dtype=partial_dtype)
                        ops.append((dist.irecv, ob, c.src)); ops.append((dist.irecv, lb, c.src))
                        merges.append((blk, ob, lb))
        return ops, merges

    def drain(merges):
        for blk, ob, lb in merges:
            pt.merge_partial(blk, ob, lb.float())

    steps = sched.steps
    in_reqs = _launch(build_comm(steps[0], "i")[0]) if steps else []
    out_reqs, out_merge = [], []
    for i, step in enumerate(steps):
        _wait(in_reqs)
        in_reqs = _launch(build_comm(steps[i + 1], "i")[0]) if i + 1 < len(steps) else []
        for c in step.comp:
            pt.compute(c.q_blocks, c.kv_blocks, softmax_scale)
        o_ops, o_m = build_comm(step, "o")
        out_reqs += _launch(o_ops); out_merge += o_m
        if len(out_reqs) >= 48:
            _wait(out_reqs)
            drain(out_merge)
            out_reqs, out_merge = [], []
    _wait(out_reqs)
    drain(out_merge)
    pt.end()
    return {b: pt.acc[slice(*pt.qrows(b))].to(out_dtype).unsqueeze(0) for b in sched.owned_q}
