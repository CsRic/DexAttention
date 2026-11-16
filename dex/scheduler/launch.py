"""Forward kernel launch (dex_cuda.dist_fwd) from fwd.tables_blocks' tables, and plan files.

engine_tables: the shared runtime's message table (csrc/runtime: rt_params.h Msg, rt_host.h CeMsg; offsets from the
region's Q slot area; engine mode 3): every message goes by the lane of its destination -- another
node: block 0's engine (NVSHMEM put-with-signal, IBRC / IBGDA); the same node: the copy-engine executor. Messages:
the Q / KV pushes (start; K without a signal, V carrying it; a run of slots = one message signalled on its last
slot), the partial O of each row (trigger = the row), the relay forwards (copy engine, on the gateway's arrival)."""
import json

import numpy as np


def load_plan(path):
    """plan file -> (GPU per block pair [nb, nb] (-1 none), owner GPU per block [nb], meta)"""
    d = json.load(open(path))
    return (np.asarray(d["schedule_table"], dtype=np.int64), np.asarray(d["owner"], dtype=np.int64),
            dict(family=d.get("family"), strategy=d.get("strategy"), cp=d.get("cp"), n_blocks=d.get("n_blocks"),
                 block_tokens=d.get("block_tokens")))


def engine_tables(tables, h, hk, d, device):
    """the message tables (mode 3) of fwd.tables_blocks' tables; cached in `tables`"""
    import torch
    if "_eng_dev" in tables:
        return tables["_eng_dev"]
    H = tables["_eng_host"]; NQ, NKV = tables["NQ"], tables["NKV"]; nit = tables["n_compute"]
    n_rs = len(H["remote_sig"])
    qs, kvs = 128 * h * d * 2, 128 * hk * d * 2
    comb = 128 * h * d * 4 + h * 128 * 4                   # [O fp32 | LSE fp32] per partial slot
    K0, V0, PO = NQ * qs, NQ * qs + NKV * kvs, NQ * qs + 2 * NKV * kvs
    me_node = H["pe"] // H["gpn"]
    inter = lambda pe_: pe_ // H["gpn"] != me_node          # noqa: E731

    def run_msgs(kind, s0, d0, c, pe_, sig):                # a run of c consecutive slots (Q: one message; KV: K, V)
        if kind == 0:
            return [(s0 * qs, d0 * qs, c * qs, pe_, sig)]
        return [(K0 + s0 * kvs, K0 + d0 * kvs, c * kvs, pe_, -1), (V0 + s0 * kvs, V0 + d0 * kvs, c * kvs, pe_, sig)]

    eng_start, ce = [], []                                  # ce: (src, dst, bytes, pe, sig, kind, word)
    nq = len(H["pushq_pe"]); nk = len(H["pushkv_pe"])
    qc = H["pushq_cnt"] if len(H["pushq_cnt"]) else np.ones(nq, np.int32)
    kc = H["pushkv_cnt"] if len(H["pushkv_cnt"]) else np.ones(nk, np.int32)
    order = H["push_order"] if len(H["push_order"]) else np.arange(nq + nk)
    for e in order.tolist():                                # Q / KV pushes, in push_order
        if e < nq:
            c, s0, d0, pe_ = int(qc[e]), int(H["pushq_src"][e]), int(H["pushq_dst"][e]), int(H["pushq_pe"][e])
            ms = run_msgs(0, s0, d0, c, pe_, d0 + c - 1)
        else:
            i = e - nq; c, s0, d0, pe_ = int(kc[i]), int(H["pushkv_src"][i]), int(H["pushkv_dst"][i]), int(H["pushkv_pe"][i])
            ms = run_msgs(1, s0, d0, c, pe_, NQ + d0 + c - 1)
        if inter(pe_):
            eng_start += ms
        else:
            ce += [m_ + (0, 0) for m_ in ms]
    per_dst = {}                                            # intra-node runs: src = owned index -> its staged slot
    for e in H["ce"].reshape(-1, 7).tolist():
        dst, kind, src, dsl, c, sig = e[0], e[1] & 1, e[2], e[3], e[4], e[5]
        stg = H["stageq_dst"] if kind == 0 else H["stagekv_dst"]
        if e[1] >= 2:                                       # already a slot of the send region
            per_dst.setdefault(dst, []).extend(run_msgs(kind, src, dsl, c, dst, sig)); continue
        a = 0
        while a < c:                                        # split where the staged slots stop being consecutive
            b = a + 1
            while b < c and stg[src + b] == stg[src + a] + (b - a):
                b += 1
            per_dst.setdefault(dst, []).extend(run_msgs(kind, int(stg[src + a]), dsl + a, b - a, dst, sig if b == c else -1))
            a = b
    for k in sorted(per_dst):
        ce += [m_ + (0, 0) for m_ in per_dst[k]]
    msg = list(eng_start); n_start = len(msg); trig = [n_start]; n_pq = 0
    lane = np.zeros(nit + n_rs, np.int32)
    ce_rows = []
    for it in range(nit):                                   # partial O of a row
        o = int(H["opush_pe"][it])
        if o >= 0:
            sl = int(H["opush_slot"][it]); ms_ = [(PO + it * comb, PO + sl * comb, comb, o, NQ + NKV + sl)]
            if inter(o):
                msg += ms_; n_pq += 1; lane[it] = 1
            else:
                ce_rows += [(it, m_) for m_ in ms_]; lane[it] = 2
        trig.append(len(msg))
    for j in range(n_rs):                                   # (remote-signal triggers: nothing by the engine)
        trig.append(len(msg))
    mt = np.asarray(msg, np.int64).reshape(-1, 5)
    tab = np.stack([mt[:, 0], mt[:, 1], mt[:, 2], mt[:, 3] + (mt[:, 4] << 32)], 1) if len(mt) else np.zeros((1, 4), np.int64)
    out = dict(msg=torch.from_numpy(np.ascontiguousarray(tab).view(np.int32).reshape(-1)).to(device),
               trig_ptr=torch.tensor(trig, dtype=torch.int32, device=device), n_start=n_start, n_pq=n_pq,
               grp_trig=nit + n_rs)
    # completion messages in the planned compute order (the copy-engine stream sends them in this order)
    rank = {int(r): i for i, r in enumerate(H["ce_pout"].reshape(-1, 3)[:, 0].tolist())} if len(H["ce_pout"]) else {}
    ce_rows.sort(key=lambda z: (rank.get(z[0], len(rank) + z[0]), z[0]))   # (stable: a trigger's messages in order)
    ce += [m_ + (2, t_) for t_, m_ in ce_rows]
    for e in H["fw"].reshape(-1, 7).tolist():               # relay forwards: wait for the arrival here, plan order
        ce += [m_ + (1, e[6]) for m_ in run_msgs(e[1], e[2], e[3], e[4], e[0], e[5])]
    ct = np.asarray(ce, np.int64).reshape(-1, 7)
    cet = np.stack([ct[:, 0], ct[:, 1], ct[:, 2], ct[:, 3] + (ct[:, 4] << 32), ct[:, 5] + (ct[:, 6] << 32),
                    np.ones(len(ct), np.int64)], 1)         # one signal word: a run's receiver waits on its LAST slot
    out.update(trig_lane=torch.from_numpy(lane).to(device), ce=torch.from_numpy(np.ascontiguousarray(cet)))
    tables["_eng_dev"] = out
    return out


def call(mod, q_local, k_local, v_local, out, lse, tables, scale):
    """one forward: block 0 polls from the first instruction, inter-node messages carry their own
    signal (put-with-signal), the shared engine + copy-engine executor (mode 3), block 0 expands the KV lists, compact
    KV slots."""
    tab = [tables[k] for k in tables["_order"]]
    eng = engine_tables(tables, q_local.size(2), k_local.size(2), q_local.size(3), q_local.device)
    dims = [tables["NQ"], tables["NKV"], tables["NPART"], tables["n_compute"], tables["n_merge"],
            int(tables["remote_sig"].numel()), 1, 1, 0, int(tables["ranges_mode"]), 0, int(tables["sched_off"]),
            int(tables["n_pool"]), 0]
    dims += [0] * (21 - len(dims))                 # dims[17..20]: shared engine (message table, trigger CSR, mode 3)
    dims[17:21] = [len(tab), eng["n_start"], eng["n_pq"], 3]   # appended LAST: the host splits them off
    dims += [0] * (23 - len(dims)); dims[22] = eng["grp_trig"]
    tab = tab + [eng["msg"], eng["trig_ptr"], eng["trig_lane"], eng["ce"]]
    dims += [0] * (24 - len(dims)); dims[23] = int(tables["dx_off"])            # block 0 expands the KV lists
    dims += [0] * (26 - len(dims)); dims[24] = int(tables["geom_off"]); dims[25] = int(tables["dx_kslot"])
    dims += [0] * (27 - len(dims)); dims[26] = int(tables.get("kv_half", 0))   # 128 x 64 compute lists (kBlockN 64)
    return mod.dist_fwd(q_local, k_local, v_local, out, lse, tab, dims, scale, 1)
