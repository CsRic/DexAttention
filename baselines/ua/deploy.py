"""UltraAttn deployment search (context tiling), the first stage of the UA baseline planner.

A block problem is an N x N matrix of block weights (fraction of live token pairs per block pair; 0 = dead) over plan blocks of B tokens.
UltraAttn coarsens it to a P x P grid of context chunks (P from the dynamic-load-imbalance rule or given), sets chunk
ownership by the sequential context remap, and assigns every live chunk pair to a device with an ILP that minimises
MCV, the maximum of per-device inbound and outbound traffic, under a compute cap; nodes first, then devices inside each
node. The MCV coefficients are the executor's per-block bytes of Q, K+V and partial O (forward) or of the backward's
Q-side bundle, K+V, dQ and dK+dV partials. Solver: Gurobi (licence file from DEX_GUROBI_LICENSE) or HiGHS.
Output: Assignment(dev [N, N] device per live block pair, -1 dead; owner [N] device of each block)."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np

FULL, DIAG = 1.0, 0.5


@dataclass
class BlockProblem:
    """block-granular attention problem: w [N, N] tile weights (fraction of live token pairs per block pair)"""

    name: str
    n_blocks: int
    w: np.ndarray
    seqs: list = field(default_factory=list)
    segs: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def weights(self) -> np.ndarray:
        return self.w

    def mask(self) -> np.ndarray:
        return self.w > 0

    def to_json(self) -> str:
        """rows stored as [k_start, k_end, weight] runs"""
        rows = []
        for q in range(self.n_blocks):
            runs, r, k = [], self.w[q], 0
            while k < self.n_blocks:
                if r[k] == 0:
                    k += 1
                    continue
                k0, v = k, float(r[k])
                while k < self.n_blocks and r[k] == v:
                    k += 1
                runs.append([k0, k, v])
            rows.append(runs)
        return json.dumps(dict(name=self.name, n_blocks=self.n_blocks, seqs=[list(s) for s in self.seqs],
                               segs=[list(s) for s in self.segs], meta=self.meta, rows=rows), indent=1)

    @staticmethod
    def from_json(s: str) -> "BlockProblem":
        d = json.loads(s)
        N = d["n_blocks"]
        w = np.zeros((N, N), dtype=np.float32)
        for q, runs in enumerate(d["rows"]):
            for k0, k1, v in runs:
                w[q, k0:k1] = v
        return BlockProblem(d["name"], N, w, [tuple(x) for x in d["seqs"]], [tuple(x) for x in d["segs"]],
                            d.get("meta", {}))


@dataclass
class Topo:
    """GPU topology and the bytes of one block of each tensor (MCV coefficients)."""

    n_nodes: int = 4
    gpus_per_node: int = 4
    q_bytes: float = 1.0
    kv_bytes: float = 2.0
    o_bytes: float = 1.0
    inter_penalty: float = 12.0  # relative cost of an inter-node byte in the greedy completion
    bwd_bytes: tuple = None  # backward: (Q-side bundle, K+V, dQ partial, dK+dV partial) per block

    @property
    def R(self) -> int:
        return self.n_nodes * self.gpus_per_node

    def node_of(self, d):
        return np.asarray(d) // self.gpus_per_node


@dataclass
class Assignment:
    name: str
    dev: np.ndarray
    owner: np.ndarray
    search_time: float = 0.0
    info: dict = field(default_factory=dict)


def tiles_of(prob):
    w = prob.weights()
    qs, ks = np.nonzero(w)
    return qs, ks, w[qs, ks].astype(np.float64)


def _owner_zigzag(N, R):
    own = np.empty(N, dtype=np.int64)
    chunk = np.arange(N) * (2 * R) // N
    for c in range(2 * R):
        own[chunk == c] = c if c < R else (2 * R - 1 - c)
    return own


def greedy(prob, topo, slack: float = 0.05, owner=None, **kw):
    """communication-aware list scheduling (heaviest tile first, cheapest marginal fetch, hard load cap); used only
    when the context-tiling ILP has no solution at any relaxed cap"""
    t0 = time.perf_counter()
    N, R = prob.n_blocks, topo.R
    owner = _owner_zigzag(N, R) if owner is None else owner
    qs, ks, ws = tiles_of(prob)
    cap = ws.sum() / R * (1 + slack)
    node = topo.node_of(np.arange(R))
    link = np.where(node[None, :] == node[:, None], 1.0, topo.inter_penalty)
    has_q = np.zeros((N, R), dtype=bool)
    has_kv = np.zeros((N, R), dtype=bool)
    has_q[np.arange(N), owner] = True
    has_kv[np.arange(N), owner] = True
    load = np.zeros(R)
    dev = np.full((N, N), -1, dtype=np.int64)
    for t in np.argsort(-ws):
        q, k, wt = qs[t], ks[t], ws[t]
        cost = (~has_q[q]) * (topo.q_bytes + topo.o_bytes) * link[owner[q]] + (~has_kv[k]) * topo.kv_bytes * link[owner[k]]
        cost = cost + np.where(load + wt > cap, 1e6, 0.0) + 1e-3 * load
        d = int(np.argmin(cost))
        dev[q, k] = d
        has_q[q, d] = has_kv[k, d] = True
        load[d] += wt
    return Assignment("greedy", dev, owner, time.perf_counter() - t0)


def _coarsen(qc, kc, ws):
    """aggregate block tiles onto the P x P grid of context chunks"""
    if len(qc) == 0:
        return qc, kc, ws
    key = qc.astype(np.int64) * (int(kc.max()) + 1) + kc
    uniq, inv = np.unique(key, return_inverse=True)
    agg = np.zeros(len(uniq))
    np.add.at(agg, inv, ws)
    first = np.zeros(len(uniq), dtype=np.int64)
    first[inv] = np.arange(len(inv))
    return qc[first], kc[first], agg


def context_remap(P: int, CP: int) -> np.ndarray:
    """chunk -> rank, the sequential remap Cmap(p) = floor(p / (P / CP))"""
    return (np.arange(P) * CP) // P


def partition_degree(prob, CP: int, theta_dli: float = 0.05, cap: int | None = None) -> int:
    """smallest P (multiple of CP) whose chunk grid reaches dynamic load imbalance <= theta_dli"""
    N = prob.n_blocks
    cap = cap or N
    qs, ks, ws = tiles_of(prob)
    P = CP
    while True:
        chunk = (np.arange(N) * P) // N
        _, _, cw = _coarsen(chunk[qs], chunk[ks], ws)
        comp = float(cw.sum()) / (N / P) ** 2
        dli = np.ceil(comp / CP) / (comp / CP) - 1
        if dli <= theta_dli or P * 2 > min(cap, N):
            return P
        P *= 2


# MCV coefficients (forward: Q, K+V, O bytes; backward: _MCV_BWD), set by ultraattn() from the topology
_MCV_COEF = (1.0, 2.0, 1.0)
_MCV_BWD = None


def _mcv_coefs():
    """(inbound coefficients of A, B, C, D), (outbound ...): A = remote rows read, B = remote columns read, C = own rows
    read elsewhere, D = own columns read elsewhere"""
    if _MCV_BWD is not None:
        qin, kv, dq, dkv = _MCV_BWD
        return (qin, kv, dq, dkv), (dq, dkv, qin, kv)
    cq, ckv, co = _MCV_COEF
    return (cq, ckv, co, 0.0), (co, 0.0, cq, ckv)


def _gurobi_env():
    lic = os.environ.get("DEX_GUROBI_LICENSE")
    if lic:
        os.environ.setdefault("GRB_LICENSE_FILE", lic)


def _mcv_gurobi(rs, cs_, w, P, CP, cmap, tau, time_limit, gap, threads=0):
    _gurobi_env()
    import gurobipy as gp
    from gurobipy import GRB

    T = len(rs)
    m = gp.Model("context_tiling")
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = gap
    if threads:
        m.Params.Threads = threads
    x = m.addVars(T, CP, vtype=GRB.BINARY, name="x")
    H = m.addVars(CP, P, lb=0.0, ub=1.0, name="H")
    V = m.addVars(CP, P, lb=0.0, ub=1.0, name="V")
    MCV = m.addVar(lb=0.0, name="MCV")
    m.addConstrs((x.sum(t, "*") == 1 for t in range(T)), "alloc")
    for t in range(T):
        for g in range(CP):
            m.addConstr(H[g, rs[t]] >= x[t, g])
            m.addConstr(V[g, cs_[t]] >= x[t, g])
    m.addConstrs((gp.quicksum(w[t] * x[t, g] for t in range(T)) <= tau for g in range(CP)), "balance")
    for g in range(CP):
        mine = np.flatnonzero(cmap == g)
        other = np.flatnonzero(cmap != g)
        A = gp.quicksum(H[g, r] for r in other)
        B = gp.quicksum(V[g, c] for c in other)
        C = gp.quicksum(H[k, r] for r in mine for k in range(CP) if k != g)
        D = gp.quicksum(V[k, c] for c in mine for k in range(CP) if k != g)
        ci, co_ = _mcv_coefs()
        m.addConstr(ci[0] * A + ci[1] * B + ci[2] * C + ci[3] * D <= MCV)
        m.addConstr(co_[0] * A + co_[1] * B + co_[2] * C + co_[3] * D <= MCV)
    m.setObjective(MCV, GRB.MINIMIZE)
    m.optimize()
    if m.SolCount == 0:
        return None, dict(status=int(m.Status), solver="gurobi")
    xs = np.array([[x[t, g].X for g in range(CP)] for t in range(T)])
    return xs.argmax(axis=1), dict(status=int(m.Status), mcv=float(m.ObjVal), gap=float(m.MIPGap), solver="gurobi",
                                   timeout=int(m.Status == 9))


def _mcv_highs(rs, cs_, w, P, CP, cmap, tau, time_limit, gap):
    """the same model on HiGHS (scipy.optimize.milp)"""
    from scipy import sparse
    from scipy.optimize import Bounds, LinearConstraint, milp

    T = len(rs)
    nx = T * CP
    oH, oV, oM = nx, nx + CP * P, nx + 2 * CP * P
    nvar = oM + 1

    def X(t, g):
        return t * CP + g

    def Hh(g, r):
        return oH + g * P + r

    def Vv(g, c):
        return oV + g * P + c

    rows, cols, vals, lb, ub = [], [], [], [], []

    def add(idx, coef, lo, hi):
        r = len(lb)
        rows.extend([r] * len(idx))
        cols.extend(idx)
        vals.extend(coef)
        lb.append(lo)
        ub.append(hi)

    for t in range(T):
        add([X(t, g) for g in range(CP)], [1.0] * CP, 1, 1)
    for t in range(T):
        for g in range(CP):
            add([X(t, g), Hh(g, rs[t])], [1.0, -1.0], -np.inf, 0)
            add([X(t, g), Vv(g, cs_[t])], [1.0, -1.0], -np.inf, 0)
    for g in range(CP):
        add([X(t, g) for t in range(T)], list(w), -np.inf, tau)
    for g in range(CP):
        mine = np.flatnonzero(cmap == g)
        other = np.flatnonzero(cmap != g)
        for cf in _mcv_coefs():
            expr = [(Hh(g, r), cf[0]) for r in other] + [(Vv(g, c), cf[1]) for c in other] + \
                   [(Hh(k, r), cf[2]) for r in mine for k in range(CP) if k != g] + \
                   [(Vv(k, c), cf[3]) for c in mine for k in range(CP) if k != g]
            expr = [(i, v) for i, v in expr if v != 0.0]
            add([i for i, _ in expr] + [oM], [v for _, v in expr] + [-1.0], -np.inf, 0)
    Amat = sparse.csr_matrix((vals, (rows, cols)), shape=(len(lb), nvar))
    cost = np.zeros(nvar)
    cost[oM] = 1.0
    integrality = np.zeros(nvar)
    integrality[:nx] = 1
    bounds = Bounds(np.zeros(nvar), np.r_[np.ones(nx + 2 * CP * P), np.inf])
    res = milp(c=cost, constraints=LinearConstraint(Amat, lb, ub), integrality=integrality, bounds=bounds,
               options=dict(time_limit=time_limit, mip_rel_gap=gap, disp=False))
    if res.x is None:
        return None, dict(status=int(res.status))
    return (res.x[:nx].reshape(T, CP).argmax(axis=1),
            dict(status=int(res.status), mcv=float(res.fun), gap=float(getattr(res, "mip_gap", np.nan)),
                 timeout=int(res.status == 1)))


def _solve_mcv(rs, cs_, w, P, CP, cmap, tau, time_limit, gap, solver):
    """solver 'gurobi' (UltraAttn's solver; raises when no licence is usable) or 'highs'"""
    if solver == "gurobi":
        return _mcv_gurobi(rs, cs_, w, P, CP, cmap, tau, time_limit, gap)
    out, meta = _mcv_highs(rs, cs_, w, P, CP, cmap, tau, time_limit, gap)
    meta["solver"] = "highs"
    return out, meta


# absolute time.perf_counter() deadline for every ILP solve of ultraattn() (None: per-solve time_limit only); each solve
# gets min(time_limit, time left), at least 0.5 s
_DEADLINE = None


def _ilp_with_tau(rs, cs_, w, P, CP, cmap, tau0, time_limit, gap, solver):
    """tau = ceil(COMP / CP) is the strictest cap; relaxed by 10 % per attempt if infeasible"""
    tau = tau0
    for _ in range(6):
        tl = time_limit if _DEADLINE is None else min(time_limit, max(0.5, _DEADLINE - time.perf_counter()))
        out, meta = _solve_mcv(rs, cs_, w, P, CP, cmap, tau, tl, gap, solver)
        if out is not None:
            meta["tau"] = tau
            return out, meta
        tau *= 1.1
    return None, dict(status=-1, tau=tau)


def ultraattn(prob, topo, theta_dli: float = 0.05, time_limit: float = 30.0, gap: float = 0.02, P=None,
              solver: str = "gurobi", **kw):
    """hierarchical context tiling: node level, then device level inside each node"""
    global _MCV_COEF, _MCV_BWD
    _MCV_COEF = (float(topo.q_bytes), float(topo.kv_bytes), float(topo.o_bytes))
    _MCV_BWD = tuple(float(x) for x in topo.bwd_bytes) if getattr(topo, "bwd_bytes", None) else None
    t0 = time.perf_counter()
    N, R = prob.n_blocks, topo.R
    Pg = int(P) if P else partition_degree(prob, R, theta_dli)
    Pg = max(int(np.ceil(Pg / R)) * R, R)
    Pg = min(Pg, N)
    chunk_of = (np.arange(N) * Pg) // N
    qs, ks, ws = tiles_of(prob)
    cr, cc, cw = _coarsen(chunk_of[qs], chunk_of[ks], ws)
    cwn = cw / ((N / Pg) ** 2)
    info = dict(P=Pg, chunk=N / Pg, cells=len(cr), remap="seq")
    cmap_rank = context_remap(Pg, R)
    cmap_node = cmap_rank // topo.gpus_per_node
    if topo.n_nodes > 1:
        tau = np.ceil(cwn.sum() / topo.n_nodes)
        out, meta = _ilp_with_tau(cr, cc, cwn, Pg, topo.n_nodes, cmap_node, tau, time_limit, gap, solver)
        info["ilp_node"] = meta
        if out is None:
            sub = greedy(prob, topo)
            return Assignment("ultraattn(greedy)", sub.dev, sub.owner, time.perf_counter() - t0, info)
        cell_node = out
    else:
        cell_node = np.zeros(len(cr), dtype=np.int64)
    cell_dev = np.empty(len(cr), dtype=np.int64)
    gpn = topo.gpus_per_node
    for n in range(topo.n_nodes):
        sel = np.flatnonzero(cell_node == n)
        base = n * gpn
        if sel.size == 0:
            continue
        local = np.where(cmap_node == n, cmap_rank - base, -1)
        tau = np.ceil(cwn[sel].sum() / gpn)
        out, meta = _ilp_with_tau(cr[sel], cc[sel], cwn[sel], Pg, gpn, local, tau, time_limit, gap, solver)
        info[f"ilp_dev{n}"] = meta
        if out is None:
            sub = greedy(prob, topo)
            return Assignment("ultraattn(greedy)", sub.dev, sub.owner, time.perf_counter() - t0, info)
        cell_dev[sel] = base + out
    dev = np.full((N, N), -1, dtype=np.int64)
    cell_of = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(cr, cc))}
    idx = np.array([cell_of[(chunk_of[q], chunk_of[k])] for q, k in zip(qs, ks)])
    dev[qs, ks] = cell_dev[idx]
    owner = cmap_rank[chunk_of]
    metas = [v for v in info.values() if isinstance(v, dict)]
    info["ilp_timeout"] = int(any(m.get("timeout") for m in metas))
    info["solver"] = metas[0].get("solver", "?") if metas else "-"
    return Assignment("ultraattn", dev, owner, time.perf_counter() - t0, info)


STRATEGIES = {"ultraattn": ultraattn}
