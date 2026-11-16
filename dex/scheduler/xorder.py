"""Component replacement (Table 5, executor column; forward only): run ANOTHER system's execution order on this executor.

The other system X (UA / DCP / Magi / Ring) gives its plan and schedule in the Yunchang-executor format: owner[block],
schedule_table[q][kv] and gpu_kernels[gpu] = steps (`start`) of compute items (rows x cols) and transfers (block, src,
dst, input 'i' / partial output 'o', Q 'r' / KV 'c'). This executor is dataflow (signals), so only X's ORDER can be
imposed, not X's step barriers:
  - the message set is ours for X's assignment (schedule_blocks; run it with DIST_FS_RELAY=0: X sends every input from
    its owner straight to the computing GPU -- no intra-node relay; DIST_RING_FULL / DIST_DENSE_KV for Ring's full data
    flow);
  - send order: every link's messages in the order X delivers their blocks to the receiver (earliest step at which X
    brings any block of the message to that GPU; ties: X's listing order, then our order). A transfer X routes through
    other GPUs (Ring's hop-by-hop forwarding) counts as delivered at the step it reaches the receiver -- here it is sent
    from the owner directly;
  - landing estimate (used only to order a row's KV segments) = that step;
  - compute order per GPU: our row groups (rows with identical dependencies) ordered by the earliest step at which X
    computes any of their rows (stable: our order within ties).
apply(R, plan) rewrites the schedule kept by the library (fsb_load_out); tables_blocks then builds the tables from it."""
import ctypes

import numpy as np

from . import fwd as fs

BIG = 1e9


def x_steps(gk, P):
    """X's delivery step per (kind, receiver, block) and compute step per (GPU, row block) from gpu_kernels"""
    deliver, comp = {}, {}
    for g in range(P):
        for i, it in enumerate(gk[g]):
            t = float(it["start"]) + i * 1e-6                                # listing order breaks ties
            if it["kind"] == "comp":
                for q in it["rows"]:
                    comp[(g, int(q))] = min(comp.get((g, int(q)), BIG), t)
            elif it.get("io", "i") == "i":
                k = (0 if it["rc"] == "r" else 1, int(it["dst"]), int(it["block"]))
                deliver[k] = min(deliver.get(k, BIG), t)
    return deliver, comp


def apply(R, gk, per, P):
    """R = schedule_blocks' result for X's assignment; gk = X's gpu_kernels. Loads the reordered schedule into the
    library and returns stats (messages X never sends, rows X never computes -- both should be 0)."""
    deliver, comp = x_steps(gk, P)
    msgs = R["msgs"]; nm = len(msgs)
    key = np.full(nm, BIG)
    miss = 0
    for m in range(nm):
        s, d, k, t0, n, typ, par = (int(x) for x in msgs[m])
        if typ == 2:                                  # relay forward: should not occur with DIST_FS_RELAY=0
            raise ValueError("xorder: relay forwards in the schedule; run with DIST_FS_RELAY=0")
        blocks = range(t0 // per, (t0 + n - 1) // per + 1)
        v = min((deliver.get((k, d, b), BIG) for b in blocks), default=BIG)
        if v >= BIG:
            miss += 1
        key[m] = v
    order = np.lexsort((R["pos"], key))                 # X's step, then our position
    pos = np.empty(nm, np.int64); pos[order] = np.arange(nm)
    land = np.where(key < BIG, key, key.max(initial=0.0) + 1.0).astype(np.float64)
    rp, rb, rg = R["row_ptr"], R["row_blk"], R["row_group"]
    nrb, nrows_miss = [], 0
    nrg = []
    new_ptr = [0]
    for g in range(P):
        rows = list(range(rp[g], rp[g + 1]))
        gstep = {}
        for x in rows:
            v = comp.get((g, int(rb[x])), BIG)
            nrows_miss += v >= BIG
            gstep[int(rg[x])] = min(gstep.get(int(rg[x]), BIG), v)
        first = {}
        for i, x in enumerate(rows):
            first.setdefault(int(rg[x]), i)
        rows.sort(key=lambda x: (gstep[int(rg[x])], first[int(rg[x])], x))   # groups by X's step, members kept together
        nrb += [int(rb[x]) for x in rows]; nrg += [int(rg[x]) for x in rows]
        new_ptr.append(len(nrb))
    c64 = lambda a: np.ascontiguousarray(a, np.int64)                       # noqa: E731
    msg7, pos_, rp_, rb_, rg_ = c64(msgs), c64(pos), c64(new_ptr), c64(nrb), c64(nrg)
    land = np.ascontiguousarray(land, np.float64)
    P_ = lambda a: a.ctypes.data_as(ctypes.c_void_p)                         # noqa: E731
    fs._LIB.fsb_load_out(ctypes.c_int64(nm), P_(msg7), P_(pos_), P_(land), ctypes.c_int64(P), P_(rp_),
                         ctypes.c_int64(len(rb_)), P_(rb_), P_(rg_), ctypes.c_int64(R["stats"]["n_row_groups"]))
    return dict(x_msgs_missing=int(miss), x_rows_missing=int(nrows_miss), n_msgs=nm,
                n_distinct_steps=int(len({round(float(v), 9) for v in key if v < BIG})))
