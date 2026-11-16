"""Estimate assignment completion time; all times are seconds.

Forward: finish = max(compute, remote arrival + dependent compute tail)
                 + fixed completion cost.
Compute scales caller-supplied tile time TT by A_COMP. Arrival uses incoming
and outgoing bytes, bandwidth and latency; KAPPA weights interference from the
opposite NIC direction. Shared inputs enter a node once. GPU-level estimates
also include intra-node copies and gateway forwarding.

Backward GPU-level: finish = max(task/item compute, copy-engine time, NIC time) + C0.
The node-level backward model omits the copy-engine term.
GPU-local KV re-splitting omits inter-node arrivals and adds a local transfer
tail to max(compute, intra-node time).

Constants come from config/calibration.json, with separate entries for routines,
passes and link scopes; those entries are reused across workloads. Planner NIC
bandwidth models node aggregate traffic; scheduler bandwidth models a sender's
queue. They are effective model rates, not interchangeable hardware link specs.
KAPPA is an interference coefficient, not a fraction constrained to [0, 1].
The model ranks assignments, not individual task start times. The forward
expression equals max(compute, arrival) + residual_tail, where residual_tail
is the dependent tail not already covered by remaining compute, plus the fixed
completion cost.
"""
import torch

from .ops import oh, unit_sums_b, rows_any_b, cols_any_b, _by_home, intra_io_b

from ..common import calibration as _cal

GiB = _cal.GiB
# machine constants: config/calibration.json (units, and how to measure them on another cluster, are documented there)
FWD_MODEL = _cal.fwd_model()
BW_FW = _cal.load()["forward"]["bw_fw_gib_s"] * GiB   # one gateway's relay-forward stream
BWD_MODEL = _cal.bwd_model()
# backward compute profile: seconds per task / per item, by (query heads, KV heads)
BWD_COEF = _cal.bwd_coef()
# tail weight of intra-node traffic in the GPU-level KV re-split's local view (the share of the transfer time that does
# not overlap the receiver's compute), forward / backward
GAMMA_FWD = _cal.load()["forward"]["gamma_local"]
GAMMA_BWD = _cal.load()["backward"]["gamma_local"]


def fwd_bytes(heads, B, d=128):
    """per plan block, bytes: (Q, partial O + LSE, K + V) of the forward. Q / K / V fp16; partial O fp32 (stored, sent
    and merged in fp32, as split-KV attention keeps its partial accumulators) + fp32 LSE."""
    hq, hk = heads
    return B * hq * d * 2, B * hq * d * 4 + B * hq * 4, 2 * B * hk * d * 2


def bwd_bytes(heads, B, d=128):
    """per plan block: (Q-side bundle in, dQ partial back, K+V in, dK+dV partial back), bytes."""
    hq, hk = heads
    return (2 * B * hq * d * 2 + 2 * B * hq * 4, B * hq * d * 4, 2 * B * hk * d * 2, 2 * B * hk * d * 4)


def relay_tail_b(urow, ucol, owner, gpn, NN, Qb, Kb):
    """[C, P]: per GPU the latest relay hop it waits for after its node's arrival (seconds). urow / ucol [C, n, P]."""
    C, n, P = urow.shape; dev = urow.device
    remote = (owner // gpn)[:, None] != torch.arange(NN, device=dev)[None, :]          # [n, NN]
    tail = torch.zeros((C, P), device=dev)
    for use, b in ((urow, Qb), (ucol, Kb)):
        un = use.reshape(C, n, NN, gpn) & remote[None, :, :, None]
        pos = (un.float().cumsum(3) - 1) * un.float()                                    # 0 = gateway, j = j-th hop
        tail = torch.maximum(tail, (pos * (b / BW_FW)).amax(1).reshape(C, P))
    return tail


def node_time_b(W, NA, home_node, NN, gpn, Qb, Ob, Kb, TT, m=None):
    """forward, node level: NA [C, H, m, m] node per pair -> F [C], compute part [C], arrivals [C, NN]"""
    m = m or FWD_MODEL
    W = W.float(); dev = W.device
    live = NA >= 0
    Cc = unit_sums_b(W, NA, NN) * TT
    gated = live & ((NA != home_node[:, :, None]) | (NA != home_node[:, None, :]))
    R = unit_sums_b(W, NA, NN, gated) * TT
    urow, ucol = rows_any_b(NA, NN), cols_any_b(NA, NN)
    hm = oh(home_node.reshape(-1), NN).bool()
    qx = (urow & ~hm).float(); kx = (ucol & ~hm).float()
    hr = home_node.reshape(-1)
    QX, KX = _by_home(hr, qx, NN), _by_home(hr, kx, NN)                           # (= H_.T @ qx, exact)
    X = Qb * QX + Ob * QX.transpose(1, 2) + Kb * KX                       # [C, NN, NN]
    X = X * (1 - torch.eye(NN, device=dev))
    xin, xout = X.sum(1), X.sum(2)
    tin = torch.where(xin > 0, (xin + m["KAPPA"] * xout) / m["B_NIC"], torch.zeros_like(xin))
    tsend = (xout + m["KAPPA"] * xin) / m["B_NIC"]
    arr = torch.maximum(tin, torch.where(X > 0, tsend[:, :, None], torch.zeros_like(X)).max(1).values) + m["C_ARR"]
    Cg, Rg = Cc / gpn, R / gpn
    F = torch.maximum(m["A_COMP"] * Cg, arr + m["GAMMA"] * m["A_COMP"] * Rg) + m["C_END"]
    return F.max(1).values, m["A_COMP"] * Cg.max(1).values + m["C_END"], arr


def gpu_time_b(W, dev_pair, owner, gpn, NN, arr, Qb, Ob, Kb, TT, m=None, local=False):
    """forward, GPU level: dev_pair [C, H, m, m], arr [C, NN] -> per node F [C, NN], compute part [C, NN].
    local: the GPU level's own view (per GPU max(compute, intra-node bytes both directions / B_INTRA) + GAMMA_FWD x the
    intra term, no inter-node arrival; arr is ignored)."""
    m = m or FWD_MODEL
    W = W.float(); dev = W.device; P = NN * gpn; C = dev_pair.shape[0]
    live = dev_pair >= 0
    node_pair = torch.where(live, dev_pair // gpn, torch.full_like(dev_pair, -1))
    own_node = owner // gpn
    Cc = unit_sums_b(W, dev_pair, P) * TT
    gated = live & ((own_node[:, :, None] != node_pair) | (own_node[:, None, :] != node_pair))
    R = unit_sums_b(W, dev_pair, P, gated) * TT
    urow, ucol = rows_any_b(dev_pair, P), cols_any_b(dev_pair, P)
    ow = owner.reshape(-1)
    if local:
        i1, o1 = intra_io_b(urow, ow, gpn, NN, Qb, Ob)
        i2, o2 = intra_io_b(ucol, ow, gpn, NN, Kb, 0.0)
        It = torch.maximum(i1 + i2, o1 + o2) / m["B_INTRA"]
        F = torch.maximum(m["A_COMP"] * Cc, It) + GAMMA_FWD * It + m["C_END"]
        return F.reshape(C, NN, gpn).max(2).values, (m["A_COMP"] * Cc).reshape(C, NN, gpn).max(2).values + m["C_END"]
    # intra-node bytes on the receiving side (owner -> same-node readers, gateway -> the node's other readers, partials back)
    on = ow // gpn; n = ow.shape[0]
    hm = oh(ow, P).bool()
    homein = on[:, None] == torch.arange(NN, device=dev)[None, :]
    intra = torch.zeros((C, P), device=dev)
    for use, fwd, back, has_back in ((urow, Qb, Ob, True), (ucol, Kb, 0.0, False)):   # (structural flag: Ob may be a tensor)
        un = use.reshape(C, n, NN, gpn)
        rec = un & ~hm.reshape(n, NN, gpn)
        first = oh(un.float().argmax(3), gpn).bool() & un.any(3)[..., None]
        rec = torch.where(homein[..., None], rec, rec & ~first)
        intra = intra + fwd * rec.float().sum(1).reshape(C, P)
        if has_back:
            nrec = (rec.float().sum(3) * homein.float()).sum(2)              # [C, n]
            intra = intra + back * (hm.float()[None] * nrec[..., None]).sum(1)
    gnode = torch.arange(P, device=dev) // gpn
    a = torch.maximum(arr[:, gnode] - m["C_ARR"], intra.reshape(C, NN, gpn).max(2).values[:, gnode] / m["B_INTRA"]) + m["C_ARR"]
    a = torch.maximum(a, arr[:, gnode] + relay_tail_b(urow, ucol, ow, gpn, NN, Qb, Kb))   # serial gateway forwards
    F = torch.maximum(m["A_COMP"] * Cc, a + m["GAMMA"] * m["A_COMP"] * R) + m["C_END"]
    return F.reshape(C, NN, gpn).max(2).values, (m["A_COMP"] * Cc).reshape(C, NN, gpn).max(2).values + m["C_END"]


def bwd_node_time_b(W, NA, home_node, NN, gpn, by, coef, kcols, heads):
    """backward, node level -> F [C], compute part [C], arrivals (zeros: no gated-arrival term)"""
    m = BWD_MODEL
    W = W.float(); dev = W.device; hq, hk = heads; a_t, b_i = coef; C = NA.shape[0]
    qin, dq, kv, dkv = by
    ucol = cols_any_b(NA, NN)
    items = unit_sums_b(W, NA, NN) * 4 * hq
    tasks = (ucol.float() * kcols.reshape(-1)[None, :, None]).sum(1) * 2 * hk
    Cc = (b_i * items + a_t * tasks) / gpn
    urow = rows_any_b(NA, NN)
    H_ = oh(home_node.reshape(-1), NN).float(); hm = H_.bool()
    qx, kx = (urow & ~hm).float(), (ucol & ~hm).float()
    hr = home_node.reshape(-1)
    QX, KX = _by_home(hr, qx, NN), _by_home(hr, kx, NN)
    X = qin * QX + dq * QX.transpose(1, 2) + kv * KX + dkv * KX.transpose(1, 2)
    X = X * (1 - torch.eye(NN, device=dev))
    nic = ((X.sum(2) + m["kappa"] * X.sum(1)) / m["B_nic"]).max(1).values
    Tc = m["k_comp"] * Cc.max(1).values
    return torch.maximum(Tc, nic) + m["c0"], Tc + m["c0"], torch.zeros((C, NN), device=dev)


def bwd_gpu_time_b(W, dev_pair, owner, gpn, NN, arr, by, coef, kcols, heads, local=False):
    """backward, GPU level -> per node F [C, NN], compute part [C, NN]. Copy-engine transfers overlap compute:
    max(k_comp C, E / B_eng) per GPU, then the node's NIC time. local: + GAMMA_BWD x E / B_eng, no NIC."""
    m = BWD_MODEL
    W = W.float(); dev = W.device; P = NN * gpn; hq, hk = heads; a_t, b_i = coef; C = dev_pair.shape[0]
    qin, dq, kv, dkv = by
    urow, ucol = rows_any_b(dev_pair, P), cols_any_b(dev_pair, P)
    items = unit_sums_b(W, dev_pair, P) * 4 * hq
    tasks = (ucol.float() * kcols.reshape(-1)[None, :, None]).sum(1) * 2 * hk
    Cc = b_i * items + a_t * tasks
    ow = owner.reshape(-1); n = ow.shape[0]
    hm = oh(ow, P).bool()
    gnode = torch.arange(P, device=dev) // gpn
    own_node = ow // gpn
    E = torch.zeros((C, P), device=dev)
    Xn = torch.zeros((C, NN, NN), device=dev)
    homein = own_node[:, None] == torch.arange(NN, device=dev)[None, :]
    for use, fwd, back in ((urow, qin, dq), (ucol, kv, dkv)):
        un = (use & ~hm).reshape(C, n, NN, gpn)
        cnt = un.float().sum(3)
        anyu = un.any(3)
        gw = oh(un.float().argmax(3), gpn).bool() & anyu[..., None]
        hm3 = hm.reshape(n, NN, gpn).float()
        E = E + fwd * torch.where(homein[..., None], hm3 * cnt[..., None], gw.float() * (cnt - 1).clamp(min=0)[..., None]).sum(1).reshape(C, P)
        E = E + back * torch.where(homein[..., None], un.float(), (un & ~gw).float()).sum(1).reshape(C, P)
        Mx = _by_home(own_node, (anyu & ~homein).float(), NN)                 # (= Hn.T @ ..., exact) [C, NN, NN]
        Xn = Xn + fwd * Mx + back * Mx.transpose(1, 2)
    i1, o1 = intra_io_b(urow, ow, gpn, NN, qin, dq)                      # (the receiving side too)
    i2, o2 = intra_io_b(ucol, ow, gpn, NN, kv, dkv)
    E = torch.maximum(E, i1 + i2)
    if local:
        F = torch.maximum(m["k_comp"] * Cc, E / m["B_eng"]) + GAMMA_BWD * E / m["B_eng"] + m["c0"]
        return F.reshape(C, NN, gpn).max(2).values, (m["k_comp"] * Cc).reshape(C, NN, gpn).max(2).values + m["c0"]
    nic = (Xn.sum(2) + m["kappa"] * Xn.sum(1)) / m["B_nic"]
    a_ = torch.maximum(m["k_comp"] * Cc, E / m["B_eng"])
    F = torch.maximum(a_, nic[:, gnode]) + m["c0"]
    return F.reshape(C, NN, gpn).max(2).values, (m["k_comp"] * Cc).reshape(C, NN, gpn).max(2).values + m["c0"]


def _intra_io_c(use, homel, gpn, b_in, b_out):
    """intra_io_b inside ONE node per candidate (the GPU-level KV re-split's compact problems): use [C, n,
    gpn] = the node's GPUs that need block i (one side), homel [C, n] = the block's owner as a local GPU id (gpn = owned
    outside the node). Same terms as intra_io_b restricted to the node's GPUs (a node's values depend only on its own
    pairs). -> (in [C, gpn], out [C, gpn])"""
    hm = oh(homel, gpn + 1)[..., :gpn].bool()
    homein = homel < gpn
    un = use & ~hm
    cnt = un.float().sum(2)                                                # [C, n] readers in the node
    first = oh(un.float().argmax(2), gpn).bool() & un.any(2)[..., None]    # relay (a block owned elsewhere)
    rest = torch.where(homein[..., None], un, un & ~first).float()
    n_rest = torch.where(homein, cnt, (cnt - 1).clamp(min=0))
    fr_ = torch.where(homein[..., None], 0.0, first.float() * n_rest[..., None])
    own_same = cnt * homein.float()                                        # same-node readers of an owner here
    inn = (b_in * rest + b_out * fr_).sum(1) + (hm.float() * (b_out * own_same)[..., None]).sum(1)
    out = (b_out * rest + b_in * fr_).sum(1) + (hm.float() * (b_in * own_same)[..., None]).sum(1)
    return inn, out


def gpu_local_c(Wc, dp, homel, gpn, Qb, Ob, Kb, TT, m=None):
    """gpu_time_b(local=True) of node x on its compact problem: Wc [C, H, mr, mc] (candidate = node; its row blocks x
    the columns they touch), dp the pairs' local GPU ids (-1 none), homel = (row homes [C, H, mr], column homes [C, H,
    mc]) as local GPU ids (gpn = outside the node) -> (F [C], compute part [C])"""
    m = m or FWD_MODEL
    C = dp.shape[0]; hr, hc = homel
    Cc = unit_sums_b(Wc, dp, gpn) * TT
    urow, ucol = rows_any_b(dp, gpn), cols_any_b(dp, gpn)
    i1, o1 = _intra_io_c(urow, hr.reshape(C, -1), gpn, Qb, Ob)
    i2, o2 = _intra_io_c(ucol, hc.reshape(C, -1), gpn, Kb, 0.0)
    It = torch.maximum(i1 + i2, o1 + o2) / m["B_INTRA"]
    F = torch.maximum(m["A_COMP"] * Cc, It) + GAMMA_FWD * It + m["C_END"]
    return F.max(1).values, (m["A_COMP"] * Cc).max(1).values + m["C_END"]


def bwd_local_c(Wc, dp, homel, kcc, gpn, by, coef, heads):
    """bwd_gpu_time_b(local=True) of node x on its compact problem (see gpu_local_c); kcc [C, H mc] live tile columns
    of the column blocks"""
    m = BWD_MODEL
    hq, hk = heads; a_t, b_i = coef; qin, dq, kv, dkv = by; C = dp.shape[0]
    urow, ucol = rows_any_b(dp, gpn), cols_any_b(dp, gpn)
    items = unit_sums_b(Wc, dp, gpn) * 4 * hq
    tasks = (ucol.float() * kcc[..., None]).sum(1) * 2 * hk
    Cc = b_i * items + a_t * tasks
    hr, hc = (x.reshape(C, -1) for x in homel)
    E = torch.zeros((C, gpn), device=dp.device)
    for use, hl, fwd, back in ((urow, hr, qin, dq), (ucol, hc, kv, dkv)):
        hm = oh(hl, gpn + 1)[..., :gpn].bool(); homein = hl < gpn
        un = use & ~hm
        cnt = un.float().sum(2); anyu = un.any(2)
        gw = oh(un.float().argmax(2), gpn).bool() & anyu[..., None]
        E = E + fwd * torch.where(homein[..., None], hm.float() * cnt[..., None], gw.float() * (cnt - 1).clamp(min=0)[..., None]).sum(1)
        E = E + back * torch.where(homein[..., None], un.float(), (un & ~gw).float()).sum(1)
    i1, o1 = _intra_io_c(urow, hr, gpn, qin, dq)
    i2, o2 = _intra_io_c(ucol, hc, gpn, kv, dkv)
    E = torch.maximum(E, i1 + i2)
    F = torch.maximum(m["k_comp"] * Cc, E / m["B_eng"]) + GAMMA_BWD * E / m["B_eng"] + m["c0"]
    return F.max(1).values, (m["k_comp"] * Cc).max(1).values + m["c0"]


def _per_node_use(use, home, gpn):
    """use [G, n, gpn] (the node's GPUs that need block i), home [G, n] local owner GPU (gpn = outside) -> the
    non-local model's receiving-side terms inside the node: (rec [G, n, gpn] receivers fed inside the node, homein,
    hm one-hot owner)"""
    hm = oh(home, gpn + 1)[..., :gpn].bool(); homein = home < gpn
    un = use & ~hm
    first = oh(un.float().argmax(2), gpn).bool() & un.any(2)[..., None]    # the relay of a block owned elsewhere
    rec = torch.where(homein[..., None], un, un & ~first)
    return rec, homein, hm, un


def _rows_view(Wg, ug, rhead, Hh, gpn):
    """a node's problem where every row (part) sits on one GPU ug [G, R] (-1 = none), its columns head-local (rhead
    [G, 1, R] the rows' heads, Hh heads): -> (urow [G, R, gpn], ucol [G, Hh Ch, gpn] = the GPUs using each (head,
    column) block -- one product, no per-pair tensor --, row work [G, R])"""
    A = (Wg[:, 0] > 0).half()                                            # [G, R, Ch]
    G, R, Ch = A.shape
    urow = oh(ug.clamp(min=0), gpn).bool() & (ug >= 0)[..., None]
    uh = (oh(rhead[:, 0].clamp(min=0), Hh).half()[..., :, None] * urow.half()[..., None, :]).reshape(G, R, Hh * gpn)
    ucol = ((A.transpose(1, 2) @ uh) > 0).reshape(G, Ch, Hh, gpn).transpose(1, 2).reshape(G, Hh * Ch, gpn)
    return urow, ucol, Wg[:, 0].float().sum(2)


def gpu_node_c(Wg, ug, rhead, rh, kh, arr, gpn, Qb, Ob, Kb, TT, m=None):
    """gpu_time_b (non-local: arrival, relay hops, gated compute) of ONE node on its compact problem, per group
    (the GPU level evaluates candidates per node on the node's own problem). Wg [G, 1, R, Cc]
    (rows = the node's parts, columns = its column blocks), ug [G, R] the local GPU of every part (-1 none), rh [G, 1,
    R] / kh [G, 1, H, Ch] owners as local GPU ids (gpn = another node; columns head-local, rhead [G, 1, R] the rows'
    heads), arr [G] the node's arrival -> (F [G], compute part [G]). A node's terms depend only on its own pairs, so F
    equals the node's entry of the global model."""
    m = m or FWD_MODEL
    G = ug.shape[0]; dev = ug.device; Hh = kh.shape[-2]
    urow, ucol, cw = _rows_view(Wg, ug, rhead, Hh, gpn)
    hr, hc = rh.reshape(G, -1), kh.reshape(G, -1)
    khr = kh[:, 0].gather(1, rhead[:, 0].clamp(min=0)[..., None].expand(-1, -1, kh.shape[-1]))   # [G, R, Ch] row's head
    gw = torch.where((hr >= gpn)[..., None], Wg[:, 0].float(), Wg[:, 0].float() * (khr >= gpn).float()).sum(2)
    Cc = (cw[..., None] * urow.float()).sum(1) * TT                       # (tile counts: exact sums)
    R = (gw[..., None] * urow.float()).sum(1) * TT                        # gated work (row or column owned elsewhere)
    intra = torch.zeros((G, gpn), device=dev)
    for use, home, fwd, back, has_back in ((urow, hr, Qb, Ob, True), (ucol, hc, Kb, 0.0, False)):
        rec, homein, hm, _ = _per_node_use(use, home, gpn)
        intra = intra + fwd * rec.float().sum(1)
        if has_back:
            nrec = rec.float().sum(2) * homein.float()                     # [G, n] receivers of a block owned here
            intra = intra + back * (hm.float() * nrec[..., None]).sum(1)
    a = torch.maximum(arr[:, None] - m["C_ARR"], intra.max(1, keepdim=True).values / m["B_INTRA"]) + m["C_ARR"]
    tail = torch.zeros((G, gpn), device=dev)                             # relay hops (blocks owned elsewhere)
    for use, home, b in ((urow, hr, Qb), (ucol, hc, Kb)):
        un = use & (home >= gpn)[..., None]
        pos = (un.float().cumsum(2) - 1) * un.float()
        tail = torch.maximum(tail, (pos * (b / BW_FW)).amax(1))
    a = torch.maximum(a, arr[:, None] + tail)
    F = torch.maximum(m["A_COMP"] * Cc, a + m["GAMMA"] * m["A_COMP"] * R) + m["C_END"]
    return F.max(1).values, (m["A_COMP"] * Cc).max(1).values + m["C_END"]


def bwd_nic_nodes(NA, owner, gpn, NN, by):
    """the backward model's NIC time per node [C, NN]: determined by the node assignment NA [C, H, m, m] (which node uses
    which block), the same for every GPU-level candidate of a node candidate"""
    m = BWD_MODEL
    qin, dq, kv, dkv = by
    on = owner.reshape(-1) // gpn
    homein = on[:, None] == torch.arange(NN, device=NA.device)[None, :]
    Xn = 0.0
    for use, fwd, back in ((rows_any_b(NA, NN), qin, dq), (cols_any_b(NA, NN), kv, dkv)):
        Mx = _by_home(on, (use & ~homein).float(), NN)                    # [C, NN home, NN reader]
        Xn = Xn + fwd * Mx + back * Mx.transpose(1, 2)
    return (Xn.sum(2) + m["kappa"] * Xn.sum(1)) / m["B_nic"]


def bwd_node_c(Wg, ug, rhead, rh, kh, kcc, nic, gpn, by, coef, heads):
    """bwd_gpu_time_b (non-local) of ONE node on its compact problem per group (see gpu_node_c); kcc [G, Cc] live tile
    columns of the column blocks, nic [G] the node's NIC time (bwd_nic_nodes)"""
    m = BWD_MODEL
    hq, hk = heads; a_t, b_i = coef; qin, dq, kv, dkv = by; G = ug.shape[0]
    urow, ucol, cw = _rows_view(Wg, ug, rhead, kh.shape[-2], gpn)
    items = (cw[..., None] * urow.float()).sum(1) * 4 * hq
    tasks = (ucol.float() * kcc[..., None]).sum(1) * 2 * hk
    Cc = b_i * items + a_t * tasks
    hr, hc = rh.reshape(G, -1), kh.reshape(G, -1)
    E = torch.zeros((G, gpn), device=ug.device)
    for use, home, fwd, back in ((urow, hr, qin, dq), (ucol, hc, kv, dkv)):
        _, homein, hm, un = _per_node_use(use, home, gpn)
        cnt = un.float().sum(2); anyu = un.any(2)
        gw = oh(un.float().argmax(2), gpn).bool() & anyu[..., None]
        E = E + fwd * torch.where(homein[..., None], hm.float() * cnt[..., None], gw.float() * (cnt - 1).clamp(min=0)[..., None]).sum(1)
        E = E + back * torch.where(homein[..., None], un.float(), (un & ~gw).float()).sum(1)
    i1, o1 = _intra_io_c(urow, hr, gpn, qin, dq)
    i2, o2 = _intra_io_c(ucol, hc, gpn, kv, dkv)
    E = torch.maximum(E, i1 + i2)
    a_ = torch.maximum(m["k_comp"] * Cc, E / m["B_eng"])
    F = torch.maximum(a_, nic[:, None]) + m["c0"]
    return F.max(1).values, (m["k_comp"] * Cc).max(1).values + m["c0"]


def bwd_models_b(heads, B, kcols, gpn=4, NN=4):
    """the backward models for the planner: kcols [H, m] live tile columns per KV block -> (node model, GPU model with
    a .local view, WQ, WK)"""
    by = bwd_bytes(heads, B); coef = BWD_COEF[tuple(heads)]
    nt = lambda W, NA, hn: bwd_node_time_b(W, NA, hn, NN, gpn, by, coef, kcols, heads)            # noqa: E731
    gt = lambda W, dp, ow, arr: bwd_gpu_time_b(W, dp, ow, gpn, NN, arr, by, coef, kcols, heads)   # noqa: E731
    gt.local = lambda W, dp, ow, arr: bwd_gpu_time_b(W, dp, ow, gpn, NN, arr, by, coef, kcols, heads, local=True)   # noqa: E731
    gt.local_c = lambda Wc, dp, hl, kcc: bwd_local_c(Wc, dp, hl, kcc, gpn, by, coef, heads)   # noqa: E731
    gt.node_c = lambda Wg, ug, rhd, rh, kh, kcc, nic: bwd_node_c(Wg, ug, rhd, rh, kh, kcc, nic, gpn, by, coef, heads)   # noqa: E731
    gt.nic = lambda NA, ow: bwd_nic_nodes(NA, ow, gpn, NN, by)   # noqa: E731
    gt.kcols = kcols
    return nt, gt, by[0] + by[1], by[2] + by[3]


def bwd_models_t(heads, by, kcols, gpn=4, NN=4):
    """bwd_models_b on runtime buffers: by = 4 0-d tensors (bwd_bytes), kcols [H, m]; WQ / WK are tensors that
    Planner.retarget refreshes in place"""
    coef = BWD_COEF[tuple(heads)]
    nt = lambda W, NA, hn: bwd_node_time_b(W, NA, hn, NN, gpn, by, coef, kcols, heads)            # noqa: E731
    gt = lambda W, dp, ow, arr: bwd_gpu_time_b(W, dp, ow, gpn, NN, arr, by, coef, kcols, heads)   # noqa: E731
    gt.local = lambda W, dp, ow, arr: bwd_gpu_time_b(W, dp, ow, gpn, NN, arr, by, coef, kcols, heads, local=True)   # noqa: E731
    gt.local_c = lambda Wc, dp, hl, kcc: bwd_local_c(Wc, dp, hl, kcc, gpn, by, coef, heads)   # noqa: E731
    gt.node_c = lambda Wg, ug, rhd, rh, kh, kcc, nic: bwd_node_c(Wg, ug, rhd, rh, kh, kcc, nic, gpn, by, coef, heads)   # noqa: E731
    gt.nic = lambda NA, ow: bwd_nic_nodes(NA, ow, gpn, NN, by)   # noqa: E731
    gt.kcols = kcols
    return nt, gt, (by[0] + by[1]).clone(), (by[2] + by[3]).clone()


def node_bytes_b(W, NA, home_unit, U, WQ, WK):
    """[C]: bytes crossing units (one copy per (block, remote unit)): rows x WQ + KV columns x WK"""
    hm = oh(home_unit.reshape(-1), U).bool()
    q = (rows_any_b(NA, U) | hm).sum(-1) - 1
    k = (cols_any_b(NA, U) | hm).sum(-1) - 1
    return WQ * q.clamp(min=0).sum(1) + WK * k.clamp(min=0).sum(1)


def lexpick(F, Fc, dim=0, rtol=1e-6):
    """per node the GPU-level candidate with the least model time; among (near-)ties the least compute part, then the
    first (a communication-bound node gives every candidate the same time)"""
    m = F.min(dim, keepdim=True).values
    tie = F <= m * (1 + rtol)
    return torch.where(tie, Fc, torch.full_like(Fc, float("inf"))).argmin(dim)
