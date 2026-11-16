"""Direct-fetch NIC traffic of Dex's plans (Table 4 "direct fetch" row): the forward inter-node bytes a node would
receive if every GPU fetched each remote input it needs itself and returned each partial result directly to its owner,
without gateway forwarding or pre-merging, next to the same model with forwarding. NIC = bytes entering a node, max over
the nodes, MiB. Forward only.

  w123  tile level on the head_split=4 plans: live tile (i, j) computed on GPU g; Q tile i (Q fp16) and KV tile j
        (K + V fp16) are inputs unless homed on g's node; partial O + LSE (fp32) returns to the row owner's node.
        relay: a node receives each remote tile once and returns one pre-merged partial per (node, row tile).
  w4    needs-based (Dex keeps every query at its owner): node receives each remote 64-token block one of its GPUs
        selects; direct = once per GPU, relay = once per node.
  w5    the GPU-replanned W5 plan (as w5_comm.py --gpu; needs one GPU): Q tiles, 64-key K + V halves, partial O + LSE.
  w6    needs-based per KV-head unit (union of its 4 query heads), 128-token K + V tiles, queries at their owner.
Inputs : as w5_comm.py / dex_static_comm.py / w4_comm.py; W6 FlexPrefill masks (common.w6_dump)
Output : $DEX_RESULT_ROOT/comm_volume/direct_fetch/<W>.jsonl, one record per case: case, relay_MiB, direct_MiB
Example: CUDA_VISIBLE_DEVICES= python experiments/comm_volume/direct_fetch.py w123 w4 w6 ; CUDA_VISIBLE_DEVICES=0 python ... w5
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

GPN, P = C.GPN, C.P


def _tile_nic(ti, tj, gt, own_t, qin, kin, qpart, relay):
    """forward node NIC (max over nodes) for live tiles (ti, tj) computed on GPUs gt; own_t[x] = owner GPU of tile x"""
    nd = gt // GPN; who = nd if relay else gt
    recv = np.zeros(P // GPN)
    for x, size in ((ti, qin), (tj, kin)):
        m = (own_t[x] // GPN) != nd
        u = np.unique(np.stack([who[m], x[m]]), axis=1)
        np.add.at(recv, u[0] if relay else u[0] // GPN, size)
    m = (own_t[ti] // GPN) != nd
    u = np.unique(np.stack([who[m], ti[m]]), axis=1)
    np.add.at(recv, own_t[u[1]] // GPN, qpart)
    return recv.max() / C.MiB


def w123():
    import json
    for W in ("W1", "W2", "W3"):
        recs = []
        for name in C.static_names(W):
            mask, pf, _pb = C.dex_plan_paths(W, name)
            if not (os.path.exists(mask) and os.path.exists(pf)):
                continue
            d = json.load(open(pf))
            t = np.asarray(d["schedule_table"]); own = np.asarray(d["owner"]); hq, hk = d["heads_virtual"]
            k = d["block_tokens"] // C.TILE
            m = np.load(mask)["mask"]
            ti, tj = np.nonzero(m > 0); gt = t[ti // k, tj // k]
            assert (gt >= 0).all(), name
            own_t = np.repeat(own, k)
            T_, D_ = C.TILE, C.D
            args = (T_ * hq * D_ * 2, 2 * T_ * hk * D_ * 2, T_ * hq * (D_ * 4 + 4))
            recs.append(dict(workload=W, case=name, relay_MiB=_tile_nic(ti, tj, gt, own_t, *args, True),
                             direct_MiB=_tile_nic(ti, tj, gt, own_t, *args, False)))
        C.write_jsonl(os.path.join(C.out_dir("direct_fetch"), f"{W}.jsonl"), recs)
        print(f"[direct] {W}: {len(recs)} cases")


def w4():
    import w4_comm as W4
    recs = []
    for c in C.w4_cases():
        seq, lay = c.split(":")
        f = os.path.join(C.w4_selection_dir(), seq, f"L{int(lay):02d}_topk.npy")
        if not os.path.exists(f):
            continue
        tk = np.load(f, mmap_mode="r"); S = tk.shape[1]
        ch = W4.zigzag_chunks(S, P); nb = (S + W4.BLK - 1) // W4.BLK
        needs = W4.needs_matrix(tk, ch, nb); own = W4.block_owner(ch, nb)
        bb = (np.minimum((np.arange(nb) + 1) * W4.BLK, S) - np.arange(nb) * W4.BLK) * W4.ROW_BYTES
        rem = needs & ((own // GPN)[None, :] != (np.arange(P) // GPN)[:, None])
        direct = (rem * bb).sum(1).reshape(P // GPN, GPN).sum(1)
        relay = (rem.reshape(P // GPN, GPN, nb).any(1) * bb).sum(1)
        recs.append(dict(workload="W4", case=f"{seq}_L{int(lay):02d}", relay_MiB=relay.max() / C.MiB, direct_MiB=direct.max() / C.MiB))
    C.write_jsonl(os.path.join(C.out_dir("direct_fetch"), "W4.jsonl"), recs)
    print(f"[direct] W4: {len(recs)} cases")


def w5():
    import torch
    import w5_common as W5
    from dex import planner as pl
    dev = torch.device("cuda", 0)
    QB, KB, PB = C.TILE * C.D * 2, 64 * C.D * 2 * 2, C.TILE * C.D * 4 + C.TILE * 4
    planners, recs = {}, []
    for d, call, name in C.w5_cases():
        if not os.path.isdir(d):
            continue
        mh, lay0, _l, _dens, S_ = W5.mask_of(d, call, dev)
        X = W5.plan_inputs(mh, lay0, dev)
        Qb, Ob, Kb = pl.fwd_bytes((1, 1), X["B"], C.D); TT = pl.FWD_MODEL["T_TILE"] / 32
        if S_ not in planners:
            planners[S_] = pl.Planner(W5.H, X["nbh"], GPN, P // GPN, Qb, Ob, Kb, TT)
        pb = planners[S_](X["Wb"], X["ob"], X["gid"]).cpu().numpy()
        h, i, j = (x.cpu().numpy() for x in torch.nonzero(mh != 0, as_tuple=True))
        per, ownb = X["per"], X["ownb"]
        g = pb[h, i // per, (j // 2) // per]
        oq, ok = ownb[i // per], ownb[(j // 2) // per]

        def model(relay):
            nd = g // GPN; who = nd if relay else g
            recv = np.zeros(P // GPN)
            for key, o, size in (((h, i), oq, QB), ((h, j), ok, KB)):
                m = (o // GPN) != nd
                u = np.unique(np.stack([who[m], key[0][m], key[1][m]]), axis=1)
                np.add.at(recv, u[0] if relay else u[0] // GPN, size)
            m = (oq // GPN) != nd
            u = np.unique(np.stack([who[m], h[m], i[m]]), axis=1)
            np.add.at(recv, ownb[u[2] // per] // GPN, PB)
            return recv.max() / C.MiB
        recs.append(dict(workload="W5", case=name, relay_MiB=model(True), direct_MiB=model(False)))
        del mh
        torch.cuda.empty_cache()
    C.write_jsonl(os.path.join(C.out_dir("direct_fetch"), "W5.jsonl"), recs)
    print(f"[direct] W5: {len(recs)} cases")


def w6():
    HQ, HK = 32, 8
    SPAD = {"32k": 32768, "64k": 65536, "128k": 131072}
    unit = 2 * C.TILE * C.D * 2
    recs = []
    for inp, L in C.w6_cases():
        mk = np.load(C.w6_dump(inp, L, "mask")); nb = mk.shape[1]
        m = np.unpackbits(mk, axis=-1, count=nb).astype(bool)
        n = SPAD[inp.split("_")[-1]] // C.TILE; per = n // (2 * P)
        own = np.repeat(C.zigzag_owner(2 * P), per)[:nb]
        mg = m.reshape(HK, HQ // HK, nb, nb).any(1)
        need = np.zeros((P, HK, nb), bool)
        for r in range(P):
            need[r] = mg[:, own == r, :].any(1)
        rem = need & ((own // GPN)[None, None, :] != (np.arange(P) // GPN)[:, None, None])
        direct = rem.sum((1, 2)).reshape(P // GPN, GPN).sum(1) * unit
        relay = rem.reshape(P // GPN, GPN, HK, nb).any(1).sum((1, 2)) * unit
        recs.append(dict(workload="W6", case=f"{inp}_L{int(L):02d}", relay_MiB=relay.max() / C.MiB, direct_MiB=direct.max() / C.MiB))
    C.write_jsonl(os.path.join(C.out_dir("direct_fetch"), "W6.jsonl"), recs)
    print(f"[direct] W6: {len(recs)} cases")


if __name__ == "__main__":
    for arg in sys.argv[1:] or ["w123", "w4", "w6"]:
        {"w123": w123, "w4": w4, "w5": w5, "w6": w6}[arg]()
