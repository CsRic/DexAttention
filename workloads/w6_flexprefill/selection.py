"""Distributed FlexPrefill block selection (official flex_prefill.ops.flex_prefill_attention.get_active_blocks,
gqa_interleave=False) without the full K on any GPU. Llama-3.1-8B setting: gamma 0.95, tau 0.1, min_budget 1024 tokens
(= 8 blocks), max_budget None, block 128.

The official selection, per layer (q [1, N, 32, 128], k [1, N, 8, 128], N = real tokens):
  last_q = the last 128 real queries / sqrt(d); qk = last_q . k^T (bf16, all N keys); causal mask on the last 128 keys;
  p = softmax(qk) fp32; per query head: vertical[c] = mean_i p[i, c], slash[c] = sum_i p[i, c - 127 + i] / 128 (token-level
  diagonal sums; c = the diagonal's last key); budgets from the sorted cumsum of vertical / slash (gamma coverage,
  >= min_budget); block-pooled vertical / slash; pooled K -> avg_qk = softmax(mean(last_q) . pooled K); sqrt-JS(avg_qk,
  pooled vertical) < tau -> "query-aware" head (budgets = min_budget + a block top-p over softmax(pooled Q . pooled K^T),
  gamma * nb coverage of the flattened [nb, nb] matrix); vertical / slash top-k -> block index sets.

Distributed (P ranks, zigzag), per layer:
  1 broadcast   the last 128 real query rows (bf16) from their 1-2 owner ranks
  2 sendrecv    each chunk receives the 127 keys before it from the previous chunk's owner, so every diagonal sum is
                computed whole on the rank holding its last key
  3 all-gather  per-rank row max of qk over its own keys -> global max (exact)
  4 all-gather  per-rank row sum of exp(qk - max) -> global sum in rank order (sum over ranks)
  5 all-gather  per rank: token-level vertical + slash of its own keys, pooled Q and pooled K of its own blocks
  then every rank runs the official code from the budget step on, on tensors of the official shapes; the query-aware loop
  of the j-th query-aware head runs on rank j % P only and
  6 all-gather  its block bitmap (nb^2 bits per head)
  so all ranks hold one full mask; each keeps its own rows.
Every per-element operation is the official one on the same inputs. The softmax row sum is the fp32 sum of per-rank partial
sums (rank order), so p can differ from the official p by 1 ulp; the official query-aware top-p scan is itself
nondeterministic by +-1 element at the coverage boundary, so one computation per head keeps all ranks on one mask.

Generator protocol: FlexDist.run(q_loc, k_loc) yields communication requests and receives their results:
  ("bcast", tensor, src) -> tensor; ("allgather", tensor) -> [P, *shape]; ("sendrecv", sends [(dst, t)], recvs [(src, shape,
  dtype)]) -> [received tensors]. drive_nccl() executes them with torch.distributed."""

import math

import flex_prefill.ops.flex_prefill_attention as FPA
import torch

BS, HQ, HK, D, GQ = 128, 32, 8, 128, 4
GAMMA, TAU, MIN_BUDGET = 0.95, 0.1, math.ceil(1024 / 128)
HALO = BS - 1


class FlexDist:
    def __init__(self, lay, N):
        self.lay, self.N = lay, N
        P, cs = lay.P, lay.cs
        self.nb = math.ceil(N / BS)
        self.owner = lambda c: c if c < P else 2 * P - 1 - c  # noqa: E731
        self.locoff = lambda c: 0 if c < P else cs  # noqa: E731  chunk c's offset in its owner
        self.my = []  # (chunk, local offset, start, n_real)
        for i, c in enumerate(lay.my):
            a = lay.chunks[c][0]
            self.my.append((c, i * cs, a, max(0, min(cs, N - a))))
        # last-q parts: global [N-128, N) split by chunk -> (src rank, row offset in last_q, src local offset, length)
        self.lq_parts = []
        s = N - BS
        while s < N:
            c = s // cs
            e = min(N, (c + 1) * cs)
            self.lq_parts.append((self.owner(c), s - (N - BS), self.locoff(c) + (s - c * cs), e - s))
            s = e
        # halo: chunk c (start a, 0 < a < N) needs keys [a-127, a) = the last 127 rows of chunk c-1 (owner(c-1))
        self.recvs, self.sends = [], []
        for c, lo, a, nr in self.my:
            if 0 < a < N:
                self.recvs.append((c, self.owner(c - 1)))
        for c, lo, a, nr in self.my:
            if c + 1 < 2 * P and (c + 1) * cs < N:
                self.sends.append((self.owner(c + 1), lo + cs - HALO))

    def run(self, q_loc, k_loc):
        """q_loc [L, 32, D] bf16, k_loc [L, 8, D] bf16 (this rank's zigzag tokens; padded tokens are ignored).
        Generator; returns this rank's mask rows [32, nl, n] int8 (1 selected, 2 = causal diagonal tile)."""
        lay, N, dev = self.lay, self.N, q_loc.device
        rank, P, cs = lay.rank, lay.P, lay.cs
        self.rx = {}  # bytes received by this rank, per step
        # 1 the last 128 real queries
        lq = torch.empty(BS, HQ, D, dtype=q_loc.dtype, device=dev)
        for src, ro, so, ln in self.lq_parts:
            buf = (
                q_loc[so : so + ln].contiguous()
                if rank == src
                else torch.empty(ln, HQ, D, dtype=q_loc.dtype, device=dev)
            )
            buf = yield ("bcast", buf, src)
            lq[ro : ro + ln] = buf
            if rank != src:
                self.rx["last_q"] = self.rx.get("last_q", 0) + buf.numel() * buf.element_size()
        last_q = lq[None] / math.sqrt(D)
        # 2 halo keys
        sends = [(dst, k_loc[o : o + HALO].contiguous()) for dst, o in self.sends if dst != rank]
        recvs = [(src, (HALO, HK, D), k_loc.dtype) for c, src in self.recvs if src != rank]
        got = yield ("sendrecv", sends, recvs)
        self.rx["halo"] = sum(int(math.prod(s)) * 2 for _, s, _ in recvs)
        halo, gi = {}, 0
        for c, src in self.recvs:
            if src == rank:  # (chunk P-1 -> chunk P, same rank)
                o = self.locoff(c - 1) + cs - HALO
                halo[c] = k_loc[o : o + HALO]
            else:
                halo[c] = got[gi]
                gi += 1
        # 3 logits per chunk over [halo (127; zeros before token 0), own real keys]
        qks = []
        for c, lo, a, nr in self.my:
            if nr == 0:
                qks.append(None)
                continue
            h = halo.get(c, torch.zeros(HALO, HK, D, dtype=k_loc.dtype, device=dev))
            ke = torch.cat([h, k_loc[lo : lo + nr]], 0)
            qk = torch.einsum("bihgd, bjhgd -> bhgij", last_q.view(1, BS, HK, GQ, D), ke.view(1, HALO + nr, HK, 1, D))
            pos = torch.arange(a - HALO, a + nr, device=dev)  # global key position of each column
            i = torch.arange(BS, device=dev)
            dead = (pos[None, :] - (N - BS)) > i[:, None]  # official causal mask on the last 128
            dead |= pos[None, :] < 0
            qk.masked_fill_(dead[None, None, None], float("-inf"))
            qks.append(qk)
        mx = torch.full((HK, GQ, BS), float("-inf"), device=dev)
        for qk in qks:
            if qk is not None:
                mx = torch.maximum(mx, qk[0, ..., HALO:].float().amax(-1))
        allmx = yield ("allgather", mx.contiguous())
        self.rx["row_max"] = (P - 1) * mx.numel() * 4
        M = allmx.amax(0)
        sm = torch.zeros(HK, GQ, BS, device=dev)
        es = []
        for qk in qks:
            if qk is None:
                es.append(None)
                continue
            e = torch.exp(qk[0].float() - M[..., None])  # (official epilogue: exp(x - max) / sum)
            sm = sm + e[..., HALO:].sum(-1)
            es.append(e)
        allsm = yield ("allgather", sm.contiguous())
        self.rx["row_sum"] = (P - 1) * sm.numel() * 4
        S = allsm.sum(0)
        # 4 token-level vertical / slash of own keys, pooled Q / K of own blocks
        L, nl = lay.L, lay.nl
        vs = torch.zeros(2, HQ, L, dtype=torch.float32, device=dev)
        pq = torch.zeros(nl, HQ, D, dtype=q_loc.dtype, device=dev)
        pk = torch.zeros(nl, HK, D, dtype=k_loc.dtype, device=dev)
        for (c, lo, a, nr), e in zip(self.my, es):
            if e is None:
                continue
            p = (e / S[..., None]).view(1, HQ, BS, HALO + nr)  # [1, (h g), i, j] (official rearrange)
            vs[0, :, lo : lo + nr] = p[..., HALO:].contiguous().mean(-2)[0]
            vs[1, :, lo : lo + nr] = (FPA.sum_all_diagonal_matrix(p) / BS)[0, :, HALO:]
            nbc = math.ceil(nr / BS)
            b0 = lo // BS
            pk[b0 : b0 + nbc] = FPA.triton_bnhd_pool(k_loc[lo : lo + nr][None], BS)[0]
            padq = nbc * BS - nr
            for hh in range(HQ):  # (official layout: per head [nb, 128, d])
                pq[b0 : b0 + nbc, hh] = (
                    torch.nn.functional.pad(q_loc[lo : lo + nr, hh, :], (0, 0, 0, padq), value=0)
                    .view(nbc, BS, D)
                    .mean(1)
                )
        # 5 one all-gather of everything the global stage needs
        pack = torch.cat(
            [vs.view(torch.uint8).reshape(-1), pq.view(torch.uint8).reshape(-1), pk.view(torch.uint8).reshape(-1)]
        )
        allp = yield ("allgather", pack)
        self.rx["vectors_pooled"] = (P - 1) * pack.numel()
        rows = yield from self.global_stage(allp, lq)
        return rows

    def unpack(self, allp):
        lay, N = self.lay, self.N
        P, L, nl, dev = lay.P, lay.L, lay.nl, allp.device
        nv, nq = 2 * HQ * L * 4, nl * HQ * D * 2
        vs = allp[:, :nv].contiguous().view(torch.float32).view(P, 2, HQ, L)
        pq = allp[:, nv : nv + nq].contiguous().view(torch.bfloat16).view(P, nl, HQ, D)
        pk = allp[:, nv + nq :].contiguous().view(torch.bfloat16).view(P, nl, HK, D)
        gtok = torch.cat(
            [torch.cat([torch.arange(*lay.chunks[c], device=dev) for c in (r, 2 * P - 1 - r)]) for r in range(P)]
        )
        gblk = lay.rows_of  # rank-major local tiles -> global tile
        V = torch.empty(2, HQ, lay.S, dtype=torch.float32, device=dev)
        V[:, :, gtok] = vs.permute(1, 2, 0, 3).reshape(2, HQ, P * L)
        PQ = torch.empty(lay.n, HQ, D, dtype=torch.bfloat16, device=dev)
        PQ[gblk] = pq.reshape(P * nl, HQ, D)
        PK = torch.empty(lay.n, HK, D, dtype=torch.bfloat16, device=dev)
        PK[gblk] = pk.reshape(P * nl, HK, D)
        return (
            V[0, :, :N][None].contiguous(),
            V[1, :, :N][None].contiguous(),
            PQ[: self.nb],
            PK[: self.nb][None].contiguous(),
        )

    def global_stage(self, allp, lq):
        """the official get_active_blocks body from the budget step on: vertical / slash / pooled K / pooled Q come from the
        gather, and the query-aware loop body of head qa[j] runs on rank j % P only. Generator (one all-gather)."""
        vertical, slash, PQ, pooled_k = self.unpack(allp)
        N, dev = self.N, allp.device
        block_size, gamma, min_budget, tau = BS, GAMMA, MIN_BUDGET, TAU
        batch_size, seq_len, num_heads, head_dim = 1, N, HQ, D
        num_share_q_heads = GQ
        num_blocks = math.ceil(seq_len / block_size)
        max_budget = min(math.ceil(2147483647 / block_size), num_blocks)  # (official: max_budget None)
        last_q = lq[None] / math.sqrt(head_dim)
        num_vertical_blocks = FPA.score_cover_topk(vertical, gamma) // 128 + 1
        num_slash_blocks = FPA.score_cover_topk(slash, gamma) // 128 + 1
        num_vertical_blocks[num_vertical_blocks < min_budget] = min_budget
        num_vertical_blocks[num_vertical_blocks > max_budget] = max_budget
        num_slash_blocks[num_slash_blocks < min_budget] = min_budget
        num_slash_blocks[num_slash_blocks > max_budget] = max_budget
        vertical = FPA.torch_bhn_sumpool(vertical, block_size)
        slash = FPA.torch_bhn_sumpool(slash, block_size)
        avg_k = pooled_k.repeat_interleave(num_share_q_heads, 2)
        avg_qk = torch.einsum("bihd, bjhd -> bhij", last_q.mean(1, keepdim=True), avg_k).squeeze(2)
        avg_qk = torch.softmax(avg_qk, dim=-1, dtype=torch.float32)
        kl_div = FPA.square_root_js_divergence(avg_qk, vertical)
        block_sparse_mask = kl_div < tau
        num_vertical_blocks[block_sparse_mask] = min_budget
        num_slash_blocks[block_sparse_mask] = min_budget
        vertical[..., :1] = torch.inf
        slash[..., -1:] = torch.inf
        num_slash_blocks = num_slash_blocks.view(batch_size * num_heads)
        slash = slash.view(batch_size * num_heads, -1)
        slash_topk = (num_blocks - 1) - slash.topk(min(num_slash_blocks.max().item(), num_blocks), -1).indices
        slash_topk[
            torch.arange(slash_topk.shape[-1], device=num_slash_blocks.device)[None, :] >= num_slash_blocks[:, None]
        ] = 0
        slash_topk = slash_topk.view(batch_size, num_heads, -1)
        num_vertical_blocks = num_vertical_blocks.view(batch_size * num_heads)
        vertical = vertical.view(batch_size * num_heads, -1)
        vertical_topk = vertical.topk(min(num_vertical_blocks.max().item(), num_blocks), -1).indices
        vertical_topk[
            torch.arange(vertical_topk.shape[-1], device=num_vertical_blocks.device)[None, :]
            >= num_vertical_blocks[:, None]
        ] = 0
        vertical_topk = vertical_topk.view(batch_size, num_heads, -1)
        block_idx = FPA.transform_veritcal_slash_idx(vertical_topk, slash_topk, num_blocks)
        block_causal_mask = None
        # query-aware heads: the list is identical on every rank; head qa[j] is computed by rank j % P only, bitmaps all-gathered
        qa = block_sparse_mask.nonzero().tolist()
        P, rank = self.lay.P, self.lay.rank
        kq = (len(qa) + P - 1) // P
        nbits = num_blocks * num_blocks
        nby = (nbits + 7) // 8
        mine = torch.zeros(max(kq, 1), nby, dtype=torch.uint8, device=dev)
        w8 = (1 << torch.arange(8, device=dev, dtype=torch.int32)).to(torch.uint8)
        for j, (b, h) in enumerate(qa):
            if j % P != rank:
                continue
            if block_causal_mask is None:
                block_causal_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=dev, dtype=torch.bool))
            pad_q = math.ceil(seq_len / block_size) * block_size - seq_len
            avg_q = PQ[:, h, :].clone()  # (= pad(q[b, :, h, :]).view(nb, bs, d).mean(1))
            avg_q[-1, :] = avg_q[-1, :] * block_size / (block_size - pad_q)
            attn = torch.einsum("id, jd -> ij", avg_q / math.sqrt(head_dim), avg_k[b, :, h, :]).masked_fill_(
                ~block_causal_mask, float("-inf")
            )
            attn = torch.softmax(attn, dim=-1, dtype=torch.float32).view(-1)
            block_topk = FPA.score_cover_idx(attn, gamma * num_blocks)
            bm = torch.zeros(nby * 8, dtype=torch.bool, device=dev)
            bm[block_topk.long()] = True
            mine[j // P] = (bm.view(nby, 8).to(torch.uint8) * w8).sum(1, dtype=torch.uint8)
        extra = {}
        self.n_qa = len(qa)
        if qa:
            allq = yield ("allgather", mine)  # [P, kq, nby]
            self.rx["qa_bitmaps"] = (P - 1) * mine.numel()
            for j, (b, h) in enumerate(qa):
                bits = (allq[j % P, j // P][:, None] >> torch.arange(8, device=dev, dtype=torch.uint8)) & 1
                extra[h] = bits.reshape(-1)[:nbits].bool()
        return self.to_rows(block_idx[0], extra)

    def to_rows(self, idx, extra):
        """official block index lists -> this rank's mask rows [32, nl, n] int8 (lower triangle; diagonal = 2; padded tile
        rows: only their diagonal)"""
        lay, nb, dev = self.lay, self.nb, self.lay.dev
        n, nl = lay.n, lay.nl
        m = torch.zeros(HQ, nb * nb, dtype=torch.bool, device=dev)
        for h in range(HQ):
            m[h, idx[h].long()] = True
            if h in extra:
                m[h] |= extra[h]  # (= unique(cat(vs_idx, block_topk)))
        m = m.view(HQ, nb, nb) & torch.ones(nb, nb, dtype=torch.bool, device=dev).tril()
        gt = lay.gtile
        real = gt < nb
        out = torch.zeros(HQ, nl, n, dtype=torch.int8, device=dev)
        out[:, real.nonzero().flatten(), :nb] = m[:, gt[real]].to(torch.int8)
        out[:, torch.arange(nl, device=dev), gt] = 2
        return out


def drive_nccl(gen):
    """execute a FlexDist.run generator with torch.distributed (NCCL); returns its result"""
    import torch.distributed as dist

    P = dist.get_world_size()
    res = None
    while True:
        try:
            op = gen.send(res)
        except StopIteration as e:
            return e.value
        if op[0] == "bcast":
            dist.broadcast(op[1], op[2])
            res = op[1]
        elif op[0] == "allgather":
            t = op[1]
            res = torch.empty((P,) + tuple(t.shape), dtype=t.dtype, device=t.device)
            dist.all_gather_into_tensor(res, t)
        elif op[0] == "sendrecv":
            sends, recvs = op[1], op[2]
            res = [torch.empty(s, dtype=dt, device=torch.cuda.current_device()) for _, s, dt in recvs]
            ops = [dist.P2POp(dist.isend, t, d) for d, t in sends] + [
                dist.P2POp(dist.irecv, b, s) for (s, _, _), b in zip(recvs, res)
            ]
            if ops:
                for r in dist.batch_isend_irecv(ops):
                    r.wait()
        else:
            raise ValueError(op[0])
