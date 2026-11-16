"""DCP's W2 / W3 plans as approximate per-head virtual-block plans for Dex's executor. DCP places (token block x head) units; on the rollouts
its block boundaries are not 128-token aligned (it plans the unpadded concatenation, the layout pads each sequence) and on GQA the query heads of
one KV head may sit on different GPUs. The executor runs per-head plans as block-diagonal virtual heads (MHA 16/16 -> 16 copies with heads (1, 1), GQA
32/8 -> 8 copies, one per KV-head group, heads (4, 1)). Mapping:
  plan blocks: k tiles of 128 padded tokens (k = the smallest power of two with virtual blocks <= 2048); owner of a plan block = the DCP owner of the
    DCP block holding most of its real tokens; pad-only blocks -> the GPU with the least DCP work;
  compute: virtual pair (copy g, q block j, kv block k) -> the GPU carrying most of DCP's work for it (work units of copy g of every overlapping DCP block
    pair, weighted by the share of that pair's live token pairs inside (j, k)); stage = that GPU's heaviest stage; pairs without a DCP unit -> the q owner,
    last stage;
  order: comp at 2 * stage; an input arrives at DCP's own earliest transfer of any of its (head, DCP block) inputs to that GPU (phase 1: 2 s - 1, phase 2: 2 s),
    else one step before its first use; partial O after the last stage.
stats.approx reports work moved off DCP's (GPU, stage) choice, real tokens whose owner changed, pairs without DCP work.
Run dcp_plan.py first (per-rank extracts). Output: $DEX_DATA_ROOT/plans/ablation/dcp/<id>_phapprox.json + virtual mask.
Example: python experiments/ablation_components/dcp_perhead.py --workload w3 --name <id>"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import dcp_plan as D  # noqa: E402

P, TILE = D.P, D.TILE


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workload", required=True, choices=["w2", "w3"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--max_vblocks", type=int, default=2048)
    a = ap.parse_args()
    L = D.plan_dir(a.workload)
    B = D.ladder_B(a.workload, a.name)
    OUT = D.out_dir()
    ext = f"{OUT}/_ext"
    rp = D.Replay(D.load_extracts(ext, a.name, B)).run(); chk = D.check_replay(rp)
    W, S = rp.W, rp.S
    mask_path = D.default_mask(a.workload, a.name); z = np.load(mask_path); m = z["mask"].astype(np.int8)
    Hq, Hk = (int(x) for x in z["heads"]); Hv = Hk; gq = Hq // Hk
    Tpad = int(z["tokens"]); ntile = m.shape[0]
    tv = z["tok_valid"].astype(bool) if "tok_valid" in z.files else np.ones(Tpad, bool)
    pos = np.flatnonzero(tv); Sreal = len(pos)
    rg = np.load(D.default_ranges(a.workload, a.name), mmap_mode="r")
    k = 1
    while Hv * ntile // k > a.max_vblocks and ntile % (2 * k) == 0 and 2 * k <= 64:
        k *= 2
    bt = k * TILE; nb = ntile // k; n = Hv * nb
    # DCP blocks, owners
    ni = len(W["in_type"])
    nbd = max(W["in_block"]) + 1
    qdev = {}
    for i in range(ni):
        if W["in_type"][i] == "Q":
            qdev.setdefault(W["in_block"][i], set()).add(W["in_dev"][i])
    assert all(len(v) == 1 for v in qdev.values()), "DCP Q block on several GPUs"
    downer = np.array([min(qdev[b]) for b in range(nbd)])
    db_of_real = np.arange(Sreal) // B
    pb_of_real = pos // bt
    cnt = collections.Counter(zip(pb_of_real.tolist(), db_of_real.tolist()))
    plan_db = collections.defaultdict(list)
    for (j, b), c in cnt.items():
        plan_db[j].append((b, c))
    owner = np.full(nb, -1, np.int64); moved_tok = 0
    for j, lst in plan_db.items():
        lst.sort(key=lambda x: (-x[1], x[0])); owner[j] = downer[lst[0][0]]
        moved_tok += sum(c for b, c in lst[1:] if downer[b] != owner[j])
    load = np.zeros(P)
    for w in range(len(W["w_q"])):
        load[W["w_dev"][w]] += W["w_cost"][w]
    pad_only = np.flatnonzero(owner < 0); owner[pad_only] = int(np.argmin(load))
    first_real = np.searchsorted(pos, np.arange(nb) * bt); last_real = np.searchsorted(pos, (np.arange(nb) + 1) * bt)
    # DCP work per (DCP pair, copy g): (dev, stage) -> cost
    unit = collections.defaultdict(lambda: collections.defaultdict(float))
    pair_live = {}
    for w in range(len(W["w_q"])):
        qi, ki = W["w_q"][w], W["w_kv"][w]
        g = W["in_head"][ki]                                   # KV head = copy (MHA: = q head)
        assert W["in_head"][qi] // gq == g, "work unit q head outside its KV head group"
        unit[(W["in_block"][qi], W["in_block"][ki], g)][(W["w_dev"][w], W["w_stage"][w])] += W["w_cost"][w]
    live = (m.reshape(nb, k, nb, k) != 0).any((1, 3))
    lq, lk = np.nonzero(live)
    st = np.full((n, n), -1, np.int64); stg = {}
    tot = moved = 0.0; no_work = 0; pad_pairs = 0
    for j, kk in zip(lq.tolist(), lk.tolist()):
        if not plan_db.get(j) or not plan_db.get(kk):
            for g in range(Hv):
                st[g * nb + j, g * nb + kk] = owner[j]; stg[(g, j, kk)] = 0
            pad_pairs += 1; continue
        cands = [(qb, kb) for qb, _ in plan_db[j] for kb, _ in plan_db[kk]]
        wts = {}
        for qb, kb in cands:                                   # share of the DCP pair's live token pairs inside (j, kk)
            key = (qb, kb)
            if key not in pair_live:
                pair_live[key] = D.live_pairs_ranges(rg, qb * B, min((qb + 1) * B, Sreal), kb * B, min((kb + 1) * B, Sreal))
            q0, q1 = max(first_real[j], qb * B), min(last_real[j], (qb + 1) * B)
            k0, k1 = max(first_real[kk], kb * B), min(last_real[kk], (kb + 1) * B)
            wts[key] = D.live_pairs_ranges(rg, q0, q1, k0, k1) / max(1, pair_live[key])
        for g in range(Hv):
            sc = collections.defaultdict(float)
            for key in cands:
                for (dv, s), c in unit.get((key[0], key[1], g), {}).items():
                    sc[(dv, s)] += c * wts[key]
            vq, vk = g * nb + j, g * nb + kk
            if not sc:
                st[vq, vk] = owner[j]; stg[(g, j, kk)] = S - 1; no_work += 1; continue
            by_d = collections.defaultdict(float)
            for (dv, s), c in sc.items():
                by_d[dv] += c
            dv = min(by_d, key=lambda x: (-by_d[x], x))
            s = min((s for (d_, s) in sc if d_ == dv), key=lambda s_: (-sc[(dv, s_)], s_))
            st[vq, vk] = dv; stg[(g, j, kk)] = s
            t_ = sum(sc.values()); tot += t_; moved += t_ - sc[(dv, s)]
    own_v = [int(owner[i]) for g in range(Hv) for i in range(nb)]
    # DCP's input arrivals: (rc, dst, copy g, DCP block) -> earliest global step
    arr = {}
    for (i, src, dst, s, ph) in rp.transfers:
        rc = "r" if W["in_type"][i] == "Q" else "c"
        g = W["in_head"][i] // (gq if rc == "r" else 1)
        t = float(2 * s - 1 if ph == 1 else 2 * s)
        key = (rc, dst, g, W["in_block"][i]); arr[key] = min(arr.get(key, 1e9), t)
    gk = [[] for _ in range(P)]
    need = collections.defaultdict(lambda: 1e9)                # (rc, gpu, vblock) -> first comp step
    rows = collections.defaultdict(list)                       # (gpu, stage, vq) -> [vk]
    for (g, j, kk), s in stg.items():
        dv = int(st[g * nb + j, g * nb + kk]); rows[(dv, s, g * nb + j)].append(g * nb + kk)
        for rc, b in (("r", j), ("c", kk)):
            if owner[b] != dv:
                need[(rc, dv, g * nb + b)] = min(need[(rc, dv, g * nb + b)], 2.0 * s)
    for (dv, s, vq), cols in rows.items():
        gk[dv].append(dict(kind="comp", rows=[vq], cols=sorted(cols), start=float(2 * s), fused=len(cols) > 1))
    n_dcp = n_syn = 0
    for (rc, dv, vb), t_need in need.items():
        g, j = divmod(vb, nb)
        ts = [arr[(rc, dv, g, b)] for b, _ in plan_db.get(j, []) if (rc, dv, g, b) in arr and arr[(rc, dv, g, b)] < t_need]
        if ts:
            t = min(ts); n_dcp += 1
        else:
            t = t_need - 1; n_syn += 1
        src = own_v[vb]
        for gg in {src, dv}:
            gk[gg].append(dict(kind="comm", block=vb, src=src, dst=dv, io="i", rc=rc, start=t))
    for (dv, s, vq) in rows:
        if own_v[vq] != dv:
            for gg in {dv, own_v[vq]}:
                gk[gg].append(dict(kind="comm", block=vq, src=dv, dst=own_v[vq], io="o", rc="r", start=float(2 * (S - 1) + 1)))
    for g in range(P):
        gk[g].sort(key=lambda x: x["start"])
    vm = D.perhead_mask(mask_path, a.name, Hv)
    approx = dict(work_off_dcp_gpu_and_stage=moved / tot if tot else 0.0, real_tokens_owner_changed=int(moved_tok),
                  real_tokens=Sreal, block_tokens=bt, tiles_per_block=k, virtual_heads=Hv, n_virtual_blocks=n,
                  virtual_pairs_without_dcp_work=no_work, pad_pairs=pad_pairs, inputs_at_dcp_step=n_dcp,
                  inputs_synthesized=n_syn, replay_checks=chk)
    plan = dict(family=a.workload, strategy="dcp_official_perhead_approx", name=a.name, cp=P, n_blocks=n, block_tokens=bt,
                owner=own_v, schedule_table=st.tolist(), gpu_kernels=gk, virtual_heads=Hv, blocks_per_head=nb, mask=vm,
                heads=[gq, 1], source=dict(cache_dir=f"{L}/cache", B=B, n_stages=S), stats=dict(approx=approx))
    out = f"{OUT}/{a.name}_phapprox.json"; json.dump(plan, open(out, "w"))
    print(a.name, json.dumps(approx))


if __name__ == "__main__":
    main()
