"""Build a tile assignment by node-level search followed by GPU-level refinement.

plan_blocks is the eager reference; Planner runs compiled CUDA-graph rounds
with host convergence checks. Both cluster load-cap candidates alongside a
home assignment, then search node-level KV re-splits when communication
bound. GPU-level KV re-splitting is optional. Each node candidate is refined
within nodes before the final cost-model choice.

Online node-level search stops when the best improvement is no larger than
kv_cost (default: measured cost of one re-split round). The eager default is
zero; pass kv_cost=0.0 to use that stopping rule online. GPU-level search uses
zero in both forms.

Inputs: W = work per block pair, owner = data owner, gid = block ID (-1 padding),
Qb/Ob/Kb = bytes per block, TT = seconds per tile. Output: executing GPU for each
live pair (-1 otherwise). plan converts full masks to/from per-head blocks.
"""

import torch
import torch._functorch.config as _fc

from .cluster import ap_round
from .kv import GPU_KV_ROUNDS, gpu_kv_split, gkv_step_prep, gkv_step_round, gkv_finish_hb
from .levels import node_pre, node_mid, node_kv_round, node_end, gpu_pre, gpu_post, static_shapes, gpu_eval, gpu_final, pick_best
from .models import bwd_models_b

_fc.enable_autograd_cache = True           # AOTAutograd disk cache: shortens a fresh process' first call (same graphs)

EPS_NODE = (0.05, 0.1, 0.25, 0.5)          # node-level candidates: work cap (1 + eps) x mean (+ "home")
EPS_GPU = (0.1, 0.25, 0.5)                 # GPU-level candidates (+ "home")
MAX_KV_LEVELS = None                       # node-level KV re-split: optional cap on the levels (None = all, n = 2..NN)


def plan_blocks(W, owner, gid, gpn, NN, Qb, Ob, Kb, TT, models=None, eps_node=EPS_NODE, eps_gpu=EPS_GPU, rounds=8,
                sub_rounds=4, rounds_gpu=8, keep=None, kv_max_rounds=1, kv_cost=0.0, gpu_kv=True, min_gain=0.01,
                max_kv_levels=MAX_KV_LEVELS):
    """eager reference of Planner on per-head blocks -> (plan [H, m, m], extra (T_all, B_all, best), rounds)"""
    WQ, WK = (Qb + Ob, Kb) if models is None else (models[2], models[3])

    def run(state, consts, U, R):
        r = 0
        for _ in range(R):
            state, ch = ap_round(state, consts, U, WQ, WK, sub_rounds); r += 1
            if not bool(ch):
                break
        return state, r
    state, consts, caps = node_pre(W, owner, gid, gpn, NN, Qb, Ob, Kb, TT, eps_node, models)
    state, r1 = run(state, consts, NN, rounds)
    kst = node_mid(W, owner, gid, gpn, NN, Qb, Ob, Kb, TT, state, consts, caps, models)
    rk = 0
    for n in range(2, (NN if max_kv_levels is None else min(NN, max_kv_levels + 1)) + 1):
        if not bool(kst[6].any()):
            break
        T0 = kst[4]
        for _ in range(kv_max_rounds):
            kst, kept = node_kv_round(W, owner, gid, gpn, NN, Qb, Ob, Kb, TT, kst, caps, n, models, min_gain); rk += 1
            if not bool(kept.any()):
                break
        if not float(T0.min() - kst[4].min()) > kv_cost:   # the best candidate gained <= kv_cost
            break
    NA, arr, L = node_end(W, owner, gpn, NN, Qb, Ob, Kb, TT, kst, models, keep)
    vmax, R, Cc, Ch, Rh = static_shapes(L.max(0).values.tolist(), W.shape[0], W.shape[1], NN)
    state, consts, ctx = gpu_pre(W, owner, gid, NA, gpn, NN, Qb, Ob, Kb, vmax, R, Ch, eps_gpu, models, Rh=Rh)
    state, r2 = run(state, consts, gpn, rounds_gpu)
    out, extra = gpu_post(W, owner, NA, arr, gpn, NN, Qb, Ob, Kb, TT, eps_gpu, state, consts, ctx, models)
    if gpu_kv:
        out = gpu_kv_split(W, out, owner, gid, gpn, NN, arr, extra[2], Qb, Ob, Kb, TT, models, min_gain)
    return out, extra, (r1, rk, r2)


CAPTURE_STATS = dict(n=0, s=0.0)        # (statistics) graph captures made and host seconds spent in them
                                        # (incl. any torch.compile triggered by their warm-up runs)


def _capture(f):
    """CUDA graph of f (two warm-up runs on a side stream) -> (graph, outputs)"""
    import time as _tc
    _t0 = _tc.perf_counter()
    try:
        return _capture_impl(f)
    finally:
        CAPTURE_STATS["n"] += 1; CAPTURE_STATS["s"] += _tc.perf_counter() - _t0


def _capture_impl(f):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2): f()
    torch.cuda.current_stream().wait_stream(s)
    G = torch.cuda.CUDAGraph()
    # thread_local: another thread's sync (e.g. a backward plan-ahead worker waiting on an event) during a capture would
    # invalidate a global-mode capture; only this thread's own unsafe calls are prohibited
    with torch.cuda.graph(G, capture_error_mode="thread_local"): out = f()
    return G, out


class Planner:
    """plan_blocks with torch.compile + CUDA graphs for one (H, m) shape. Graphs: node pre, clustering chunk (`chunk`
    rounds, state in place, changed flag), node mid (fill + KV init), one KV round per level n, node end; per GPU-level
    static-shape bucket: GPU pre, clustering chunk, GPU post, GPU-level KV set-up, one KV round per level, finish.
    Host loop with one tiny sync per chunk / KV round (flags) and one for the GPU level's shapes; every early exit is
    exact (a round that changes nothing would repeat). kv_cost None = one node KV round's replay time, measured at the
    first call (the node-level stop rule: the search goes on to the next level only if this level gained more than that;
    0.0 = any gain, the eager default). info: T_ms, rounds (node clustering, node KV, GPU clustering, GPU KV)."""

    def __init__(self, H, m, gpn, NN, Qb, Ob, Kb, TT, models=None, eps_node=EPS_NODE, eps_gpu=EPS_GPU, rounds=8,
                 sub_rounds=4, rounds_gpu=8, chunk=2, keep=None, kv_max_rounds=1, kv_cost=None, gpu_kv=True,
                 min_gain=0.01, dev="cuda", bwd=None, max_kv_levels=MAX_KV_LEVELS, xs=None, reduce=None):
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        self.H, self.m, self.gpn, self.NN = H, m, gpn, NN
        # runtime constants: 0-d GPU buffers rather than Python floats baked into the compiled code, so another block
        # size B (bytes per block) or other kcols reuse the same compiled code AND the same captured graphs (retarget()
        # copies into the buffers). bwd = (heads, B, kcols) builds the backward models on these buffers (models= with
        # Python floats also works, with B baked into the compiled code).
        f64 = lambda x: torch.tensor(float(x), dtype=torch.float64, device=dev)   # noqa: E731
        self.cQb, self.cOb, self.cKb, self.cTT = f64(Qb), f64(Ob), f64(Kb), f64(TT)
        self.args = (self.cQb, self.cOb, self.cKb, self.cTT)
        self.bwd_heads = None
        if bwd is not None:
            from .models import bwd_bytes, bwd_models_t
            heads, B_, kc = bwd
            self.bwd_heads = tuple(heads)
            self.cby = [f64(x) for x in bwd_bytes(heads, B_)]
            self.ckc = torch.as_tensor(kc, device=dev).float().clone()
            models = bwd_models_t(heads, self.cby, self.ckc, gpn, NN)
        self.eps_node, self.eps_gpu = tuple(eps_node), tuple(eps_gpu)
        self.W = torch.zeros((H, m, m), device=dev); self.owner = torch.zeros((H, m), dtype=torch.long, device=dev)
        self.gid = torch.zeros((H, m), dtype=torch.long, device=dev)
        self.models = models
        self.rounds, self.sub, self.rounds_gpu, self.chunk, self.keep, self.kvr = rounds, sub_rounds, rounds_gpu, chunk, keep, kv_max_rounds
        self.kv_cost, self.gpu_kv, self.min_gain = kv_cost, gpu_kv, min_gain
        self.kv_levels = list(range(2, (NN if max_kv_levels is None else min(NN, max_kv_levels + 1)) + 1))   # node caps per level
        # distributed planning: xs = the nodes this instance solves at the GPU level (None = all), reduce =
        # the all-reduce(max) of the per-candidate node times across instances (host callable on a [C] tensor). The
        # result is then the plan of xs's pairs only (others -1); the instances' plans are disjoint.
        self.xs = None if xs is None else torch.tensor([int(x) for x in xs], dtype=torch.long, device=dev)
        self.reduce = reduce                                              # (xs: a buffer; set_nodes switches it)
        self.WQ, self.WK = (self.cQb + self.cOb, self.cKb) if models is None else (models[2], models[3])
        c = lambda f: torch.compile(f, dynamic=False)                    # noqa: E731
        self.c_round, self.c_node_pre, self.c_gpu_pre = c(ap_round), c(node_pre), c(gpu_pre)
        self.c_mid, self.c_kv, self.c_end, self.c_gpu_post = c(node_mid), c(node_kv_round), c(node_end), c(gpu_post)
        self.c_gkv_prep, self.c_gkv_round, self.c_gkv_fin = c(gkv_step_prep), c(gkv_step_round), c(gkv_finish_hb)
        self.c_gpu_eval, self.c_gpu_final = c(gpu_eval), c(gpu_final)
        self.g = {}
        self.reuse_buckets = True                                         # (see __call__)
        self.force_key, self.last_key = None, None

    def set_nodes(self, xs):
        """distributed planning: solve these nodes (same count) next call -- no recompile, no recapture"""
        self.xs.copy_(torch.as_tensor(list(xs), dtype=torch.long))

    def retarget(self, Qb, Ob, Kb, TT, B=None, kcols=None):
        """updated byte / time constants (e.g. another block size B) and, for the backward, B and kcols [H, m]: in place"""
        for t, v in zip(self.args, (Qb, Ob, Kb, TT)):
            t.fill_(float(v))
        if self.bwd_heads is not None:
            from .models import bwd_bytes
            for t, v in zip(self.cby, bwd_bytes(self.bwd_heads, B)):
                t.fill_(float(v))
            self.ckc.copy_(torch.as_tensor(kcols, device=self.ckc.device).float())
            self.WQ.copy_(self.cby[0] + self.cby[1]); self.WK.copy_(self.cby[2] + self.cby[3])
        elif self.models is None:
            self.WQ.copy_(self.cQb + self.cOb); self.WK.copy_(self.cKb)

    def _chunk(self, state, consts, U):
        def f():
            st = state
            for _ in range(self.chunk):
                st, ch = self.c_round(st, consts, U, self.WQ, self.WK, self.sub)
            for b, x in zip(state, st):
                b.copy_(x)
            return ch
        return _capture(f)

    def _rounds(self, g_chunk, flag, R):
        n = 0
        for _ in range(-(-R // self.chunk)):
            g_chunk.replay(); n += self.chunk
            if not bool(flag):
                break
        return n

    def _node_graphs(self):
        Qb, Ob, Kb, TT = self.args; gpn, NN, md = self.gpn, self.NN, self.models
        pre = _capture(lambda: self.c_node_pre(self.W, self.owner, self.gid, gpn, NN, Qb, Ob, Kb, TT, self.eps_node, md))
        state, consts, caps = pre[1]
        chk = self._chunk(state, consts, NN)
        mid = _capture(lambda: self.c_mid(self.W, self.owner, self.gid, gpn, NN, Qb, Ob, Kb, TT, state, consts, caps, md))
        kst = mid[1]
        kvg = {}
        for n in self.kv_levels:
            def f(n=n):
                st, kept = self.c_kv(self.W, self.owner, self.gid, gpn, NN, Qb, Ob, Kb, TT, kst, caps, n, md, self.min_gain)
                for b, x in zip(kst, st):
                    b.copy_(x)
                return kept.any(), (st[4] < 0).any()                      # (kept flag; dummy keeps outputs static)
            kvg[n] = _capture(f)
        end = _capture(lambda: self.c_end(self.W, self.owner, gpn, NN, Qb, Ob, Kb, TT, kst, md, self.keep))
        if self.kv_cost is None and NN < 2:                                # one node: no node-level KV levels
            self.kv_cost = 0.0
        if self.kv_cost is None:                                           # one KV round's replay time (the stop rule)
            import time as _t
            ts = []
            for _ in range(7):
                torch.cuda.synchronize(); a = _t.perf_counter(); kvg[2][0].replay(); torch.cuda.synchronize()
                ts.append(_t.perf_counter() - a)
            self.kv_cost = sorted(ts)[3]
        return pre, chk, mid, kvg, torch.zeros_like(kst[4]), end

    def _gpu_graphs(self, NA_all, arr_all, vmax, mr, mc, ch, rh):
        Qb, Ob, Kb, TT = self.args; gpn, NN, md = self.gpn, self.NN, self.models
        xs = self.xs
        pre2 = _capture(lambda: self.c_gpu_pre(self.W, self.owner, self.gid, NA_all, gpn, NN, Qb, Ob, Kb, vmax, mr, ch, self.eps_gpu, md, xs, rh))
        st2, c2, ctx = pre2[1]
        chk2 = self._chunk(st2, c2, gpn)
        if xs is None:
            post = _capture(lambda: self.c_gpu_post(self.W, self.owner, NA_all, arr_all, gpn, NN, Qb, Ob, Kb, TT, self.eps_gpu,
                                                    st2, c2, ctx, md))
            o_ = post[1][0]
        else:                              # distributed: evaluate these nodes, (host all-reduce + pick), finalize
            ev = _capture(lambda: self.c_gpu_eval(self.W, self.owner, NA_all, arr_all, gpn, NN, Qb, Ob, Kb, TT, self.eps_gpu,
                                                  st2, c2, ctx, md, xs))
            ug_, pick_, _, _ = ev[1]
            bestbuf = torch.zeros((), dtype=torch.long, device=self.W.device)
            fn = _capture(lambda: self.c_gpu_final(self.W, NA_all, gpn, NN, ctx, ug_, pick_, bestbuf, xs))
            post = (ev, fn, bestbuf); o_ = fn[1]
        gk = None
        if self.gpu_kv:                    # set-up graph, one round graph per level (state in place; flags kept /
                                           # active), finish graph; compact per-node problems (rows mr, columns mc)
            gp_ = _capture(lambda: self.c_gkv_prep(self.W, o_, self.owner, self.gid, gpn, NN, rh, ch, Qb, Ob, Kb, TT, md, xs))   # (head-batched: per (node, head))
            Wc, hl, gc, capc, kcc, idx, gst = gp_[1]
            grd = {}
            for n in range(2, gpn + 1):
                def f(n=n):
                    st, ka, aa = self.c_gkv_round(Wc, hl, gc, capc, kcc, gst, n, gpn, NN, Qb, Ob, Kb, TT, md, self.min_gain)
                    for b, x in zip(gst, st):
                        b.copy_(x)
                    return ka, aa
                grd[n] = _capture(f)
            fin = _capture(lambda: self.c_gkv_fin(self.W, Wc, gst, idx, gpn, NN, xs))
            gk = (gp_, grd, fin, torch.zeros_like(gst[4]))
        return pre2, chk2, post, gk

    def __call__(self, W, owner, gid, info=None):
        gpn, NN = self.gpn, self.NN
        import time as _t
        t_start = _t.perf_counter()
        self.W.copy_(W); self.owner.copy_(owner); self.gid.copy_(gid)
        if "node" not in self.g:
            self.g["node"] = self._node_graphs()
        pre, chk, mid, kvg, T0, end = self.g["node"]
        pre[0].replay()
        r1 = self._rounds(chk[0], chk[1], self.rounds)
        mid[0].replay()
        kst = mid[1]; rk = 0
        for n in self.kv_levels:
            if not bool(kst[6].any()):
                break
            T0.copy_(kst[4])
            for _ in range(self.kvr):
                kvg[n][0].replay(); rk += 1
                if not bool(kvg[n][1][0]):
                    break
            if not float(T0.min() - kst[4].min()) > self.kv_cost:          # the best candidate gained <= one round
                break
        end[0].replay()
        NA_all, arr_all, L_all = end[1]
        sync = L_all.max(0).values.tolist()
        t_node = _t.perf_counter() - t_start                              # (the node level ends at this host sync)
        key = static_shapes(sync, self.H, self.m, NN)                     # (vmax, per-node rows mr x columns mc,
        if self.force_key is not None:                                    # head-local columns ch)
            assert all(f >= k for f, k in zip(self.force_key, key)), (self.force_key, key)
            key = tuple(self.force_key)                                   # (verification: a larger bucket on purpose)
        elif key not in self.g and self.reuse_buckets:
            # the bucket values are capacities: an already-captured bucket at least as large in every dimension holds
            # this problem too -> reuse the smallest such one rather than capture another (the same inputs re-planned
            # can land one bucket off: the node level's float sums use atomics)
            dom = [k for k in self.g if isinstance(k, tuple) and len(k) == 5 and all(a >= b for a, b in zip(k, key))]
            if dom:
                key = min(dom, key=lambda k: (k[0] * k[1] * k[2] * k[3] * k[4], k))
        if key not in self.g:
            self.g[key] = self._gpu_graphs(NA_all, arr_all, *key)
        self.last_key = key
        vmax, mr, mc, ch, rh = key
        pre2, chk2, post, gk = self.g[key]
        pre2[0].replay()
        r2 = self._rounds(chk2[0], chk2[1], self.rounds_gpu)
        if self.xs is None:
            post[0].replay()
            out, extra = post[1]
        else:
            ev, fn, bestbuf = post
            ev[0].replay()
            _, _, T_loc, B_all = ev[1]
            T_all = self.reduce(T_loc.clone())                            # max over all node groups (one collective)
            best = pick_best(T_all, B_all); bestbuf.copy_(best)
            fn[0].replay(); out = fn[1]; extra = (T_all, B_all, best)
        rg = 0
        if gk is not None:
            gp_, grd, fin, T0g = gk
            gp_[0].replay(); gst = gp_[1][6]
            if bool(gst[6].any()):
                for n in range(2, gpn + 1):
                    T0g.copy_(gst[4]); aa = True
                    for _ in range(GPU_KV_ROUNDS):
                        grd[n][0].replay(); rg += 1
                        ka_, aa_ = grd[n][1]
                        if not bool(ka_):  # (one sync: kept and active of this round)
                            aa = bool(aa_); break
                    gst[6].logical_and_(T0g - gst[4] > 0)
                    if not aa or not bool(gst[6].any()):
                        break
            fin[0].replay(); out = fin[1]
        if info is not None:
            info.update(T_ms=[float(x) * 1e3 for x in extra[0]], rounds=(r1, rk, r2, rg), vmax=vmax, mr=mr, mc=mc, ch=ch, rh=rh,
                        kv_cost_ms=self.kv_cost * 1e3, t_node_ms=t_node * 1e3)
        return out


def head_blocks(W):
    """the contiguous diagonal blocks of W that no pair crosses (per-head masks: the virtual heads; one shared mask: one block):
    a boundary follows block i iff no pair (row <= i < col) or (col <= i < row). Returns [(start, size)]."""
    Wb = W > 0; n = W.shape[0]; dev = W.device
    idx = torch.arange(n, device=dev)
    rmax = torch.where(Wb, idx[None, :], torch.full_like(Wb, -1, dtype=torch.long)).max(1).values
    cmax = torch.where(Wb, idx[:, None], torch.full_like(Wb, -1, dtype=torch.long)).max(0).values
    reach = torch.cummax(torch.maximum(rmax, cmax), 0).values
    ends = (torch.nonzero(reach <= idx).flatten() + 1).tolist()
    if not ends or ends[-1] != n:
        ends.append(n)
    out, s = [], 0
    for e in ends:
        out.append((s, e - s)); s = e
    # tiny segments (a head's padding blocks attend only to themselves) join the previous one: still block-diagonal,
    # so the plan is the same, but the heads stay equal-sized (no near-empty padded heads)
    big = max(sz for _, sz in out)
    merged = []
    for st, sz in out:
        if merged and sz * 4 < big:
            merged[-1] = (merged[-1][0], merged[-1][1] + sz)
        else:
            merged.append((st, sz))
    return merged


def to_blocks(W, owner, kcols=None):
    """full [n, n] W, owner [n] (and kcols [n]) -> per-head blocks (segs, Wb [H, m, m], ob [H, m], gid [H, m], kb [H, m])"""
    dev = W.device
    segs = head_blocks(W)
    H, m = len(segs), max(sz for _, sz in segs)
    Wb = torch.zeros((H, m, m), device=dev); ob = torch.zeros((H, m), dtype=torch.long, device=dev)
    gid = torch.full((H, m), -1, dtype=torch.long, device=dev); kb = torch.zeros((H, m), dtype=torch.float32, device=dev)
    for h, (s, sz) in enumerate(segs):
        Wb[h, :sz, :sz] = W[s:s + sz, s:s + sz].float(); ob[h, :sz] = owner[s:s + sz]
        gid[h, :sz] = torch.arange(s, s + sz, device=dev)
        if kcols is not None:
            kb[h, :sz] = torch.as_tensor(kcols, device=dev)[s:s + sz].float()
    return segs, Wb, ob, gid, kb


def plan(W, owner, gpn, NN, Qb, Ob, Kb, TT, bwd=None, keep=None, gpu_kv=True):
    """full W [n, n], owner [n] -> GPU of every pair [n, n] (eager). bwd = (heads, B, kcols [n]) for the backward."""
    n = W.shape[0]
    segs, Wb, ob, gid, kb = to_blocks(W, owner, None if bwd is None else bwd[2])
    models = None if bwd is None else bwd_models_b(bwd[0], bwd[1], kb, gpn, NN)
    pb = plan_blocks(Wb, ob, gid, gpn, NN, Qb, Ob, Kb, TT, models=models, keep=keep, gpu_kv=gpu_kv)[0]
    out = torch.full((n, n), -1, dtype=torch.long, device=W.device)
    for h, (s, sz) in enumerate(segs):
        out[s:s + sz, s:s + sz] = pb[h, :sz, :sz]
    return out
