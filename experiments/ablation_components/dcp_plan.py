"""DCP as the replaced system X: the official planner's per-rank plan cache -> plan JSON (owner, schedule_table, gpu_kernels).
Source of truth: <name>_B<B>.cp<rank>.pkl = [(fw ExecutionPlan, bw ExecutionPlan, fw WorkloadSpec, bw WorkloadSpec)]:
  placement = WorkloadSpec.input_to_device_map / output_to_device_map; compute = work_to_device_map, work_to_stage_map;
  schedule  = the forward ExecutionPlan.instructions of every rank, replayed: MemcpyInstr, CommLaunchInstr S<k> (inputs of stage k, launched
  before stage k-1's attention), S<k>P2 (intra-node forward of DCP's two-phase routing), AttnInstr, AttnReductionInstr, CommLaunchInstr O
  (partial outputs, after the last stage). Every comm op is identified by tracking buffer-slot contents and matching the k-th send a->b with
  the k-th receive; the replay is cross-checked against the WorkloadSpec.
Mapping (one owner per plan block, head-agnostic): block_tokens = B when DCP block boundaries are 128-multiples of an unpadded layout (Lambda),
else 128; a tile holding tokens of two DCP blocks takes the DCP block with most of its real tokens (owner) and the pair with most live token
pairs (compute); pad-only tiles go to the GPU with the least DCP work, computed locally. A pair whose heads DCP splits over GPUs / stages goes to
the GPU with the largest work share, at its heaviest stage (stats.head_split). Steps: stage s computes at 2 s, phase-1 inputs at 2 s - 1, phase-2
relays at 2 s, partial outputs after the last stage. --perhead (MHA, B-aligned): the exact per-head virtual-block plan with DCP's schedule 1:1.
Two stages: extract (DCP's Python, per rank: plain records, no DCP classes), convert (any Python with numpy).
Output: $DEX_DATA_ROOT/plans/ablation/dcp/<id>.json (+ _perhead.json, .stats.json, _ext/).
Example: CUDA_VISIBLE_DEVICES= PYTHONPATH=third_party/dcp $DEX_DCP_PYTHON experiments/ablation_components/dcp_plan.py --workload w1 --name <id> --perhead"""
import argparse
import collections
import gzip
import json
import os
import pickle
import sys
import time

try:                       # torch before numpy: its libstdc++ requirement in DCP's environment
    import torch  # noqa: F401
except ImportError:
    pass
import numpy as np  # noqa: E402

P, GPN, TILE = 16, 4, 128
_EXT_VERSION = 1


def _root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def inputs_of(workload):
    sys.path[:0] = [_root(), os.path.join(_root(), "experiments", "attention_layer")]
    import static_inputs as SI
    return {c["id"]: c for c in SI.inputs(workload)}


def default_mask(workload, name):
    """Lambda: its mask. Rollouts: of <name>_nb{16,32,64,128}.npz (same tile mask, padding differs by plan block count) the one with the fewest
    padded tokens (ties -> smaller nb)"""
    c = inputs_of(workload)[name]
    if workload == "w1":
        return c["mask"]
    best = None
    base = c["mask"].rsplit("_nb", 1)[0]
    for nb in (16, 32, 64, 128):
        f = f"{base}_nb{nb}.npz"
        if os.path.exists(f):
            t = int(np.load(f)["tokens"])
            if best is None or t < best[0]:
                best = (t, f)
    return best[1]


def default_ranges(workload, name):
    return inputs_of(workload)[name]["ranges"]


def plan_dir(workload):
    from dex.common.env import cfg
    return os.path.join(cfg("DEX_DATA_ROOT"), "plans", "dcp", workload)


def out_dir():
    from dex.common.env import cfg
    d = os.path.join(cfg("DEX_DATA_ROOT"), "plans", "ablation", "dcp")
    os.makedirs(d, exist_ok=True)
    return d


def ladder_B(workload, name):
    r = json.load(open(os.path.join(plan_dir(workload), "ladder.json"))).get(name)
    if not r or not r.get("done") or not r.get("B"):
        raise SystemExit(f"{name}: no finished ladder entry for {workload}")
    return int(r["B"])


def _s(t):
    return str(t).split(".")[-1]


# ------------------------------------------------------------------------------------------------ stage 1: extract
def extract_rank(path, rank):
    """one rank's pickle -> plain record: its forward instruction stream (buffer refs only) + the fwd WorkloadSpec"""
    fw, _bw, wl, _wlb = pickle.load(open(path, "rb"))[0]
    me = fw.node_id * fw.n_devices_per_node + fw.local_device_id
    assert me == rank and fw.n_devices_per_node == GPN and fw.n_nodes * GPN == P, (path, me, rank)
    R = lambda d: int(d[0]) * GPN + int(d[1])                                   # noqa: E731
    bb = lambda b: (b.buffer_type, int(b.index), int(b.n_tokens))                # noqa: E731
    ins = []
    for x in fw.instructions:
        n = type(x).__name__
        if n == "MemcpyInstr":
            ins.append(("memcpy", [(bb(a), bb(b)) for a, b in x.src_dst_pairs]))
        elif n == "CommLaunchInstr":
            ins.append(("launch", x.key, x.stream, [(int(op.comm_type), R(op.peer), bb(op.buffer_block)) for op in x.comm_ops]))
        elif n == "CommWaitInstr":
            ins.append(("wait", x.key, x.stream))
        elif n == "AttnInstr":
            ins.append(("attn", int(x.stage_id), [bb(r[0]) for r in x.q_block_table], [[bb(b) for b in r] for r in x.kv_block_table],
                        [bb(r[0]) for r in x.out_block_table], [bb(r[0]) for r in x.lse_block_table]))
        elif n == "AttnReductionInstr":
            ins.append(("red", int(x.output_is_unpadded), [([bb(a) for a, _ in op.src_buffers], bb(op.dst_buffer[0]), bb(op.dst_buffer[1]))
                                                          for op in x.ops]))
        else:
            ins.append(("other", n))
    bm = wl.block_mapping
    ni, no, nw = len(bm.input_id_to_meta), len(bm.output_id_to_meta), len(bm.work_id_to_meta)
    im = [bm.input_id_to_meta[i] for i in range(ni)]
    om = [bm.output_id_to_meta[i] for i in range(no)]
    wm = [bm.work_id_to_meta[i] for i in range(nw)]
    for m in im + om:
        assert m.seq_id == 0 and m.head_block_size == 1, (m.seq_id, m.head_block_size)
    W = dict(
        in_type=[_s(m.type) for m in im], in_head=[int(m.head_id) for m in im], in_block=[int(m.block_id) for m in im],
        in_ntok=[int(m.n_tokens) for m in im], in_bsize=[int(m.block_size) for m in im], in_dtype=[int(m.dtype) for m in im],
        in_pts=[tuple(m.per_token_shape) for m in im], in_dev=[R(wl.input_to_device_map[i]) for i in range(ni)],
        in_bytes=[float(x) for x in wl.input_sizes],
        out_type=[_s(m.type) for m in om], out_head=[int(m.head_id) for m in om], out_block=[int(m.block_id) for m in om],
        out_dev=[R(wl.output_to_device_map[i]) for i in range(no)], out_bytes=[float(x) for x in wl.output_sizes],
        out_dtype=[int(m.dtype) for m in om],
        w_q=[int(m.q_id) for m in wm], w_kv=[int(m.kv_id) for m in wm], w_out=[int(m.out_id) for m in wm],
        w_lse=[int(m.lse_id) for m in wm], w_head=[int(m.head_id) for m in wm],
        w_dev=[R(wl.work_to_device_map[i]) for i in range(nw)], w_stage=[int(wl.work_to_stage_map[i]) for i in range(nw)],
        w_cost=[float(x) for x in wl.workloads],
        in_buf={R(d): {int(k): int(v) for k, v in m.items()} for d, m in bm.input_id_to_buffer_index.items()},
        out_buf={R(d): {int(k): int(v) for k, v in m.items()} for d, m in bm.output_id_to_buffer_index.items()},
        buffer_dtype={k: int(v) for k, v in bm.buffer_type_to_dtype.items()})
    binfo = {k: dict(n_blocks=int(v.n_blocks), block_size=int(v.block_size), dtype=int(v.dtype), per_token_shape=tuple(v.per_token_shape))
             for k, v in fw.buffer_info.items()}
    return dict(version=_EXT_VERSION, rank=rank, src=os.path.abspath(path), src_mtime=os.path.getmtime(path),
                n_stages=int(fw.n_stages), instrs=ins, buffer_info=binfo, wl=W)


def ext_path(ext_dir, name, B, rank):
    return f"{ext_dir}/{name}_B{B}.ext{rank}.pkl.gz"


def _extract_job(args):
    path, rank, out = args
    t0 = time.time()
    rec = extract_rank(path, rank)
    tmp = out + ".tmp"
    with gzip.open(tmp, "wb", compresslevel=3) as f:
        pickle.dump(rec, f, protocol=4)
    os.replace(tmp, out)
    return rank, time.time() - t0


def extract(cache_dir, name, B, ranks, ext_dir, jobs, force=False):
    os.makedirs(ext_dir, exist_ok=True)
    jobs_left = []
    for r in ranks:
        src = f"{cache_dir}/{name}_B{B}.cp{r}.pkl"
        out = ext_path(ext_dir, name, B, r)
        if not os.path.exists(src):
            continue
        if not force and os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(src):
            continue
        jobs_left.append((src, r, out))
    if not jobs_left:
        return
    if jobs > 1 and len(jobs_left) > 1:
        from multiprocessing import Pool
        with Pool(min(jobs, len(jobs_left))) as pool:
            for r, dt in pool.imap_unordered(_extract_job, jobs_left):
                print(f"[extract] {name} rank {r} {dt:.1f}s", flush=True)
    else:
        for t in jobs_left:
            r, dt = _extract_job(t)
            print(f"[extract] {name} rank {r} {dt:.1f}s", flush=True)


def load_extracts(ext_dir, name, B):
    recs = {}
    for r in range(P):
        f = ext_path(ext_dir, name, B, r)
        if not os.path.exists(f):
            raise SystemExit(f"missing extract of rank {r}: {f} (run --extract_only on the node holding .cp{r}.pkl)")
        with gzip.open(f, "rb") as fh:
            recs[r] = pickle.load(fh)
        assert recs[r]["version"] == _EXT_VERSION and recs[r]["rank"] == r
    return recs


# ------------------------------------------------------------------------------------------------ stage 2a: replay
class Replay:
    """replays the 16 forward instruction streams; identifies every comm op and every attention row"""

    def __init__(self, recs):
        self.recs = recs
        W = recs[0]["wl"]
        for r in range(1, P):                                       # every rank must hold the same spec
            for k in ("in_dev", "w_dev", "w_stage", "out_dev", "w_q", "w_kv"):
                assert recs[r]["wl"][k] == W[k], f"WorkloadSpec field {k} differs between rank 0 and rank {r}"
        self.W = W
        self.S = recs[0]["n_stages"]
        assert all(recs[r]["n_stages"] == self.S for r in range(P))
        self.slot = {}                                              # (rank, buffer type, idx) -> input id / output id
        self.transfers = []                                         # (input id, src, dst, stage k, phase 1|2)
        self.out_transfers = []                                     # (output id, src, dst)
        self.attn = {}                                              # (rank, stage) -> [(q id, [kv ids], out id)]
        self.n_red = 0
        self.errors = []

    def run(self):
        W, recs = self.W, self.recs
        loc = {"Q": "LocalQ", "KV": "LocalKV"}
        for d, m in W["in_buf"].items():
            for i, k in m.items():
                self.slot[(d, loc[W["in_type"][i]], k)] = ("in", i)
        locout = {"OUT": "LocalOut", "LSE": "LocalLSE"}
        for d, m in W["out_buf"].items():
            for o, k in m.items():
                self.slot[(d, locout[W["out_type"][o]], k)] = ("out", o)
        # launches / attns per rank, in stream order
        self.launch = {r: {} for r in range(P)}
        attn_ins = {r: {} for r in range(P)}
        for r in range(P):
            for x in recs[r]["instrs"]:
                if x[0] == "memcpy":
                    for (a, b) in x[1]:
                        self.slot[(r, b[0], b[1])] = self.slot[(r, a[0], a[1])]
                elif x[0] == "launch":
                    assert x[1] not in self.launch[r], (r, x[1])
                    self.launch[r][x[1]] = x[3]
                elif x[0] == "attn":
                    assert x[1] not in attn_ins[r], (r, x[1])
                    attn_ins[r][x[1]] = x
        self.attn_ins = attn_ins
        keys_seen = set(k for r in range(P) for k in self.launch[r])
        self.do_attn(0)
        for s in range(1, self.S):
            for key, ph in ((f"S{s}", 1), (f"S{s}P2", 2)):
                if key in keys_seen:
                    self.do_comm(key, s, ph)
            self.do_attn(s)
        other = keys_seen - {f"S{s}" for s in range(1, self.S)} - {f"S{s}P2" for s in range(1, self.S)} - {"O"}
        assert not other, f"unexpected comm keys {other}"
        self.do_outputs()
        return self

    def _ident(self, r, b, kind):
        v = self.slot.get((r, b[0], b[1]))
        assert v is not None and v[0] == kind, f"rank {r}: slot {b} holds {v}, expected a {kind} block"
        return v[1]

    def _pairs(self, key):
        sends, recvs = collections.defaultdict(list), collections.defaultdict(list)
        for r in range(P):
            for (ct, peer, b) in self.launch[r].get(key, []):
                (sends if ct == 0 else recvs)[(r, peer) if ct == 0 else (peer, r)].append(b)
        assert set(sends) == set(recvs), f"{key}: send / recv peer sets differ"
        return sends, recvs

    def do_comm(self, key, s, ph):
        sends, recvs = self._pairs(key)
        ids = {}
        for (a, b), lst in sends.items():
            rl = recvs[(a, b)]
            assert len(lst) == len(rl), f"{key} {a}->{b}: {len(lst)} sends vs {len(rl)} recvs"
            ids[(a, b)] = [self._ident(a, x, "in") for x in lst]
            for x, y in zip(lst, rl):
                assert x[2] == y[2], f"{key} {a}->{b}: token count mismatch {x} {y}"
        for (a, b), lst in ids.items():                       # apply receives after resolving every send of the key
            for i, y in zip(lst, recvs[(a, b)]):
                self.slot[(b, y[0], y[1])] = ("in", i)
                self.transfers.append((i, a, b, s, ph))

    def do_attn(self, s):
        for r in range(P):
            x = self.attn_ins[r].get(s)
            if x is None:
                continue
            _, _, qt, kvt, ot, lt = x
            rows = []
            for qb, kvl, ob, lb in zip(qt, kvt, ot, lt):
                q = self._ident(r, qb, "in")
                kvs = [self._ident(r, b, "in") for b in kvl]
                rows.append((q, kvs, ob, lb))
            self.attn[(r, s)] = rows

    def do_outputs(self):
        """out slots: AttnInstr rows (the q row's OUT / LSE ids), reductions (dst <- srcs' identity); then O comm"""
        W = self.W
        q2out = {}
        for q, o, l_ in zip(W["w_q"], W["w_out"], W["w_lse"]):
            assert q2out.setdefault(q, (o, l_)) == (o, l_)
        out2lse = {o: l_ for o, l_ in q2out.values()}
        for r in range(P):
            for x in self.recs[r]["instrs"]:
                if x[0] == "attn":
                    for (q, _kvs, ob, lb) in self.attn[(r, x[1])]:
                        o, l_ = q2out[q]
                        self.slot[(r, ob[0], ob[1])] = ("out", o)
                        self.slot[(r, lb[0], lb[1])] = ("out", l_)
                elif x[0] == "red":
                    self.n_red += 1
                    if x[1]:                                   # final reduction into the Local output buffers
                        continue
                    for srcs, dO, dL in x[2]:
                        idents = {self.slot.get((r, b[0], b[1])) for b in srcs}
                        assert len(idents) == 1, f"rank {r}: reduction mixes outputs {idents}"
                        o = idents.pop()
                        self.slot[(r, dO[0], dO[1])] = o
                        self.slot[(r, dL[0], dL[1])] = ("out", out2lse[o[1]])
        if any("O" in self.launch[r] for r in range(P)):
            sends, recvs = self._pairs("O")
            for (a, b), lst in sends.items():
                rl = recvs[(a, b)]
                assert len(lst) == len(rl)
                for x, y in zip(lst, rl):
                    o = self._ident(a, x, "out")
                    self.out_transfers.append((o, a, b))


def check_replay(rp):
    """the replay against the WorkloadSpec; returns a dict of check results (raises on hard inconsistencies)"""
    W = rp.W
    want = collections.defaultdict(set)
    for w, (q, kv, d, s) in enumerate(zip(W["w_q"], W["w_kv"], W["w_dev"], W["w_stage"])):
        want[(d, s)].add((q, kv))
    got = collections.defaultdict(set)
    n_rows_dup = 0
    for (r, s), rows in rp.attn.items():
        for q, kvs, _, _ in rows:
            for kv in kvs:
                if (q, kv) in got[(r, s)]:
                    n_rows_dup += 1
                got[(r, s)].add((q, kv))
    attn_match = all(want[k] == got[k] for k in set(want) | set(got))
    # inputs present when read
    have = collections.defaultdict(dict)                       # rank -> input id -> stage available
    for i, d in enumerate(W["in_dev"]):
        have[d][i] = -1
    for (i, a, b, s, ph) in rp.transfers:
        have[b].setdefault(i, s)
    missing = 0
    relay_src_ok = 0; relay_src_bad = 0
    for (i, a, b, s, ph) in rp.transfers:
        st = have[a].get(i)
        if st is None or st > s or (st == s and ph == 1):
            relay_src_bad += 1
        elif a != W["in_dev"][i]:
            relay_src_ok += 1
    for (r, s), rows in rp.attn.items():
        for q, kvs, _, _ in rows:
            for x in [q] + kvs:
                if have[r].get(x, 99) > s:
                    missing += 1
    # outputs: expected remote partial outputs
    exp = set()
    for w, (o, l_, d) in enumerate(zip(W["w_out"], W["w_lse"], W["w_dev"])):
        for x in (o, l_):
            if W["out_dev"][x] != d:
                exp.add((x, d, W["out_dev"][x]))
    got_o = set(rp.out_transfers)
    return dict(attn_rows_match_workload=bool(attn_match), attn_duplicate_pairs=n_rows_dup, inputs_missing_at_read=missing,
                nonowner_sources_valid=relay_src_ok, nonowner_sources_invalid=relay_src_bad,
                out_transfers_match_workload=(got_o == exp) and len(got_o) == len(rp.out_transfers),
                n_input_transfers=len(rp.transfers), n_output_transfers=len(rp.out_transfers),
                transfers_by_phase=dict(collections.Counter(f"phase{t[4]}" for t in rp.transfers)))


# ------------------------------------------------------------------------------------------------ stage 2b: convert
def live_pairs_ranges(rg, q0, q1, k0, k1):
    """live (query, key) token pairs of real queries [q0, q1) x real keys [k0, k1) (dcp ranges [S, nr, 2])"""
    if q1 <= q0 or k1 <= k0:
        return 0
    x = rg[q0:q1].astype(np.int64)
    s = np.maximum(x[..., 0], k0); e = np.minimum(x[..., 1], k1)
    return int(np.maximum(e - s, 0).sum())


def convert(rp, mask_path, ranges_path, name, track, B, cache_dir, perhead=False):
    W, S = rp.W, rp.S
    z = np.load(mask_path)
    mask = z["mask"]; Tpad = int(z["tokens"]) if "tokens" in z.files else mask.shape[0] * TILE
    heads = tuple(int(x) for x in z["heads"])
    assert mask.shape[0] * TILE == Tpad
    tv = z["tok_valid"].astype(bool) if "tok_valid" in z.files else np.ones(Tpad, bool)
    pos = np.flatnonzero(tv)                                    # real token id -> padded position
    Sreal = len(pos)
    rg = np.load(ranges_path, mmap_mode="r")
    assert rg.shape[0] == Sreal, (rg.shape, Sreal)

    # ---- DCP blocks / heads
    ni = len(W["in_type"])
    qin = [i for i in range(ni) if W["in_type"][i] == "Q"]; kin = [i for i in range(ni) if W["in_type"][i] == "KV"]
    Hq = len({W["in_head"][i] for i in qin}); Hk = len({W["in_head"][i] for i in kin})
    assert (Hq, Hk) == heads, f"DCP heads {(Hq, Hk)} != mask heads {heads}"
    bsz = {W["in_bsize"][i] for i in range(ni)}; assert bsz == {B}, (bsz, B)
    nbd = max(W["in_block"]) + 1
    ntok = {}
    for i in range(ni):
        ntok.setdefault(W["in_block"][i], W["in_ntok"][i])
    assert sum(ntok[b] for b in range(nbd)) == Sreal, (sum(ntok.values()), Sreal)
    d_head = W["in_pts"][qin[0]][-1]

    # (1) placement per DCP block
    qdev = collections.defaultdict(set); kdev = collections.defaultdict(set)
    for i in range(ni):
        (qdev if W["in_type"][i] == "Q" else kdev)[W["in_block"][i]].add(W["in_dev"][i])
    odev = collections.defaultdict(set)
    for o, t in enumerate(W["out_type"]):
        odev[W["out_block"][o]].add(W["out_dev"][o])
    placement = dict(
        dcp_blocks=nbd, q_block_on_several_gpus=sum(len(qdev[b]) > 1 for b in range(nbd)),
        kv_block_on_several_gpus=sum(len(kdev[b]) > 1 for b in range(nbd)),
        q_and_kv_same_gpu=sum(qdev[b] == kdev[b] and len(qdev[b]) == 1 for b in range(nbd)),
        out_lse_on_q_gpu=sum(odev[b] == qdev[b] for b in range(nbd)))
    exact_owner = placement["q_block_on_several_gpus"] == 0 and placement["kv_block_on_several_gpus"] == 0 \
        and placement["q_and_kv_same_gpu"] == nbd and placement["out_lse_on_q_gpu"] == nbd
    if not exact_owner:
        print(f"[warn] {name}: DCP placement is not one GPU per block for Q, KV and O: {placement}")
    downer = np.array([min(qdev[b]) for b in range(nbd)])       # (exact when exact_owner)

    # (2) compute per DCP pair (head-agnostic: GPU with the largest work share; its main stage)
    pw = collections.defaultdict(lambda: collections.defaultdict(float))     # (qb, kb) -> (gpu, stage) -> cost
    gq_split = collections.defaultdict(set)                                   # (qb, kb, kv head) -> gpus (GQA groups)
    for w in range(len(W["w_q"])):
        qb, kb = W["in_block"][W["w_q"][w]], W["in_block"][W["w_kv"][w]]
        pw[(qb, kb)][(W["w_dev"][w], W["w_stage"][w])] += W["w_cost"][w]
        gq_split[(qb, kb, W["in_head"][W["w_kv"][w]])].add(W["w_dev"][w])
    pair_gpu, pair_stage = {}, {}
    tot_cost = moved_cost = stage_moved_cost = 0.0
    n_split_dev = n_split_stage = 0
    for p, d in pw.items():
        by_g = collections.defaultdict(float)
        for (g, s), c in d.items():
            by_g[g] += c
        g = min(by_g, key=lambda x: (-by_g[x], x))
        by_s = {s: c for (gg, s), c in d.items() if gg == g}
        s = min(by_s, key=lambda x: (-by_s[x], x))
        pair_gpu[p], pair_stage[p] = g, s
        tc = sum(d.values()); tot_cost += tc
        moved_cost += tc - by_g[g]
        stage_moved_cost += tc - by_s[s]
        n_split_dev += len(by_g) > 1
        n_split_stage += len({s for (_, s) in d}) > 1
    head_split = dict(dcp_pairs=len(pw), pairs_heads_on_several_gpus=n_split_dev, pairs_heads_in_several_stages=n_split_stage,
                      work_moved_to_majority_gpu=moved_cost / tot_cost,
                      work_off_chosen_gpu_and_stage=stage_moved_cost / tot_cost,
                      kv_head_groups_on_several_gpus=sum(len(v) > 1 for v in gq_split.values()) if Hq != Hk else None,
                      kv_head_groups=len(gq_split) if Hq != Hk else None)

    # (3) granularity
    bstart = np.array([pos[b * B] for b in range(nbd)])
    aligned = bool((bstart % TILE == 0).all() and Tpad == Sreal and all((B % TILE == 0) for _ in [0]))
    bt = B if (aligned and Tpad % B == 0) else TILE
    nb = Tpad // bt
    pb_of_real = pos // bt                                      # plan block of every real token
    db_of_real = np.arange(Sreal) // B
    cnt = collections.Counter(zip(pb_of_real.tolist(), db_of_real.tolist())) if bt == TILE else None
    owner = np.full(nb, -1, np.int64)
    plan_dblocks = collections.defaultdict(list)               # plan block -> [(dcp block, real tokens)]
    if bt == B:
        for b in range(nb):
            plan_dblocks[b] = [(b, ntok[b])]
    else:
        for (j, b), c in cnt.items():
            plan_dblocks[j].append((b, c))
    straddle = 0; owner_changed_tokens = 0
    for j, lst in plan_dblocks.items():
        lst.sort(key=lambda x: (-x[1], x[0]))
        owner[j] = downer[lst[0][0]]
        if len(lst) > 1:
            straddle += 1
            owner_changed_tokens += sum(c for b, c in lst[1:] if downer[b] != owner[j])
    pad_only = np.flatnonzero(owner < 0)          # (assigned below, all to one GPU: they form one causal block)
    # real-token range of each plan block per DCP block (for live-pair weights of straddling tiles)
    first_real = np.searchsorted(pos, np.arange(nb) * bt); last_real = np.searchsorted(pos, (np.arange(nb) + 1) * bt)

    # (live plan pairs from the mask)
    k = bt // TILE
    live = (mask.reshape(nb, k, nb, k) != 0).any((1, 3))
    lq, lk = np.nonzero(live)
    if len(pad_only):
        assert not live[np.ix_(np.setdiff1d(np.arange(nb), pad_only), pad_only)].any() and \
            not live[np.ix_(pad_only, np.setdiff1d(np.arange(nb), pad_only))].any(), "pad tiles interact with real tiles"
        load = np.zeros(P)                         # pad region -> the GPU with the least DCP work (cost units)
        for p_, d_ in pw.items():
            for (g_, _s), c_ in d_.items():
                load[g_] += c_
        owner[pad_only] = int(np.argmin(load))
    st = np.full((nb, nb), -1, np.int64); stage = {}
    n_choice = n_choice_changed = 0; no_dcp_work = 0; pad_pairs = 0; changed_pairs_w = 0.0
    for j, kk in zip(lq.tolist(), lk.tolist()):
        if not plan_dblocks.get(j) or not plan_dblocks.get(kk):
            st[j, kk] = owner[j]; stage[(j, kk)] = 0; pad_pairs += 1
            if owner[kk] != owner[j]:
                raise SystemExit(f"{name}: live pad tile pair ({j},{kk}) with different owners")
            continue
        cands = [(qb, kb) for qb, _ in plan_dblocks[j] for kb, _ in plan_dblocks[kk] if (qb, kb) in pw]
        if not cands:
            no_dcp_work += 1
            # the mask says live but DCP has no work unit: the owner computes it (counted, reported)
            st[j, kk] = owner[j]; stage[(j, kk)] = S - 1
            continue
        if len(cands) > 1:
            n_choice += 1
            wts = []
            for qb, kb in cands:
                q0, q1 = max(first_real[j], qb * B), min(last_real[j], (qb + 1) * B)
                k0, k1 = max(first_real[kk], kb * B), min(last_real[kk], (kb + 1) * B)
                wts.append(live_pairs_ranges(rg, q0, q1, k0, k1))
            o = sorted(range(len(cands)), key=lambda i: (-wts[i], cands[i]))
            c = cands[o[0]]
            if len({pair_gpu[x] for x in cands}) > 1:
                n_choice_changed += 1
                changed_pairs_w += sum(wts[i] for i in o[1:] if pair_gpu[cands[i]] != pair_gpu[c]) / max(1, sum(wts))
        else:
            c = cands[0]
        st[j, kk] = pair_gpu[c]; stage[(j, kk)] = pair_stage[c]
    # DCP pairs with work but no live plan pair (should be none)
    covered = set()
    for j, kk in zip(lq.tolist(), lk.tolist()):
        for qb, _ in plan_dblocks.get(j, []):
            for kb, _ in plan_dblocks.get(kk, []):
                covered.add((qb, kb))
    dcp_pairs_not_in_mask = len(set(pw) - covered)
    gran = dict(block_tokens=bt, n_blocks=nb, padded_tokens=Tpad, real_tokens=Sreal, B=B,
                dcp_block_starts_off_128=int((bstart % TILE != 0).sum()), dcp_blocks=nbd,
                plan_blocks_with_tokens_of_several_dcp_blocks=straddle,
                real_tokens_whose_owner_changed=int(owner_changed_tokens), pad_only_plan_blocks=len(pad_only),
                live_pairs=int(len(lq)), live_pairs_on_pad_blocks=pad_pairs, live_pairs_with_several_dcp_pairs=n_choice,
                of_which_dcp_gpus_differ=n_choice_changed, live_pairs_without_dcp_work=no_dcp_work,
                dcp_pairs_not_covered_by_mask=dcp_pairs_not_in_mask)

    # (4) schedule: comps per (gpu, stage) in DCP's row order; derived block-level comm with DCP's stage / phase / source
    dblock_plan = collections.defaultdict(list)                # dcp block -> plan blocks
    for j, lst in plan_dblocks.items():
        for b, _ in lst:
            dblock_plan[b].append(j)
    rowpos = {}                                                 # (gpu, stage, dcp q block) -> first row position
    for (r, s), rows in rp.attn.items():
        for i, (q, _kvs, _, _) in enumerate(rows):
            rowpos.setdefault((r, s, W["in_block"][q]), i)
    comps = collections.defaultdict(lambda: collections.defaultdict(list))   # gpu -> stage -> q -> [kv]
    for (j, kk), s in stage.items():
        comps[int(st[j, kk])][s].append((j, kk))
    gk = [[] for _ in range(P)]
    cstart = lambda s: 2 * s                                    # noqa: E731
    istart = lambda s, ph: 2 * s - 1 if ph == 1 else 2 * s      # noqa: E731
    ostart = 2 * (S - 1) + 1
    need = {}                                                   # (block, rc, gpu) -> min stage
    for g in range(P):
        for s in sorted(comps[g]):
            rows = collections.defaultdict(list)
            for j, kk in comps[g][s]:
                rows[j].append(kk)
                for b, rc in ((j, "r"), (kk, "c")):
                    if owner[b] != g:
                        need[(b, rc, g)] = min(need.get((b, rc, g), 99), s)
            def rkey(j):
                ps = [rowpos.get((g, s, b)) for b, _ in plan_dblocks.get(j, [])]
                ps = [p_ for p_ in ps if p_ is not None]
                return (min(ps) if ps else 1 << 30, j)
            colset = {}
            order = []
            for j in sorted(rows, key=rkey):
                cs = tuple(sorted(rows[j]))
                if cs not in colset:
                    colset[cs] = []; order.append(cs)
                colset[cs].append(j)
            for cs in order:
                qs = colset[cs]
                gk[g].append(dict(kind="comp", rows=qs, cols=list(cs), start=float(cstart(s)), fused=len(qs) > 1 or len(cs) > 1))
    # DCP's transfers at plan-block level: (plan block, rc, dst) -> [(stage, phase, src, bytes)]
    rec = collections.defaultdict(list)
    rcof = {"Q": "r", "KV": "c"}
    for (i, a, b, s, ph) in rp.transfers:
        for j in dblock_plan[W["in_block"][i]]:
            rec[(j, rcof[W["in_type"][i]], b)].append((s, ph, a, W["in_bytes"][i]))
    chosen = {}                                                 # (block, rc, dst) -> (start, src, origin)
    cstats = collections.Counter()

    def pick(key, s_need):
        rs = [x for x in rec.get(key, []) if x[0] <= s_need]
        if not rs:
            return None
        s0, p0 = min((x[0], x[1]) for x in rs)
        by_src = collections.defaultdict(float)
        for x in rs:
            if (x[0], x[1]) == (s0, p0):
                by_src[x[2]] += x[3]
        src = min(by_src, key=lambda a: (-by_src[a], a))
        return istart(s0, p0), src

    for key, s_need in sorted(need.items(), key=lambda kv: kv[1]):
        pk = pick(key, s_need)
        if pk is None:
            chosen[key] = (istart(s_need, 1), int(owner[key[0]]), "synth"); cstats["synthesized_from_owner"] += 1
        else:
            chosen[key] = (pk[0], pk[1], "dcp"); cstats["dcp_stage_phase_src"] += 1

    def holds(b, rc, g, before):
        if owner[b] == g:
            return True
        c = chosen.get((b, rc, g))
        return c is not None and c[0] < before

    changed = True
    while changed:                                              # relay sources must hold the block one step earlier
        changed = False
        for key in sorted(chosen, key=lambda x: chosen[x][0]):
            b, rc, g = key
            t, src, orig = chosen[key]
            if src == owner[b] or holds(b, rc, src, t):
                continue
            rs = [x for x in rec.get((b, rc, src), []) if istart(x[0], x[1]) < t]
            if rs and (b, rc, src) not in chosen:
                s0, p0 = min((x[0], x[1]) for x in rs)
                by_src = collections.defaultdict(float)
                for x in rs:
                    if (x[0], x[1]) == (s0, p0):
                        by_src[x[2]] += x[3]
                chosen[(b, rc, src)] = (istart(s0, p0), min(by_src, key=lambda a: (-by_src[a], a)), "relay_only")
                cstats["relay_only_receives_added"] += 1
            else:
                chosen[key] = (t, int(owner[b]), orig + "+owner"); cstats["relay_from_owner"] += 1
            changed = True
    comm_list = sorted(((t, src, b, rc, g) for (b, rc, g), (t, src, _o) in chosen.items()), key=lambda x: (x[0], x[1], x[4], x[3], x[2]))
    # within a step: DCP's own op order is per (sender, receiver) list order; we keep (start, src, dst, rc, block)
    for t, src, b, rc, g in comm_list:
        for gg in (src, g):
            gk[gg].append(dict(kind="comm", block=int(b), src=int(src), dst=int(g), io="i", rc=rc, start=float(t)))
    nout = 0
    for g in range(P):
        rowsg = sorted({j for (j, kk) in stage if st[j, kk] == g and owner[j] != g})
        for j in rowsg:
            for gg in (g, int(owner[j])):
                gk[gg].append(dict(kind="comm", block=int(j), src=int(g), dst=int(owner[j]), io="o", rc="r", start=float(ostart)))
            nout += 1
    for g in range(P):                                         # list order = start, then the order built above
        gk[g].sort(key=lambda x: x["start"])
    cstats["input_transfers"] = len(comm_list); cstats["output_transfers"] = nout
    cstats["relay_hops"] = sum(1 for t, src, b, rc, g in comm_list if src != owner[b])

    plan = dict(family=track, strategy="dcp_official", name=name, cp=P, n_blocks=int(nb), block_tokens=int(bt),
                owner=owner.tolist(), schedule_table=st.tolist(), gpu_kernels=gk, mask=os.path.abspath(mask_path),
                heads=list(heads),
                source=dict(cache_dir=os.path.abspath(cache_dir), B=B, n_stages=S, ranks=[rp.recs[r]["src"] for r in range(P)],
                            mtimes=[rp.recs[r]["src_mtime"] for r in range(P)],
                            partial_out_dtype=rp.recs[0]["buffer_info"].get("BufferOut", {}).get("dtype")),
                steps=dict(comp="2*stage", input_phase1="2*stage-1", input_phase2_relay="2*stage", output=ostart),
                stats=dict(placement=placement, exact_owner=exact_owner, head_split=head_split, granularity=gran,
                           comm=dict(cstats)))
    aux = dict(stage=stage, need=need, chosen=chosen, plan_dblocks=plan_dblocks, real_tokens_of_block=last_real - first_real,
               d=d_head, heads=heads, pair_gpu=pair_gpu, pair_stage=pair_stage, pw=pw, downer=downer, ntok=ntok)
    ph = perhead_plan(rp, plan, aux, B, bt, nb, mask, heads) if perhead else None
    return plan, aux, ph


def perhead_plan(rp, plan, aux, B, bt, nb, mask, heads):
    """the EXACT DCP plan as per-head virtual blocks (virtual block v = h * nb + i = DCP's (head h, block i) input; owner
    repeated per head; schedule_table block-diagonal per head) with DCP's own schedule 1:1: every replayed AttnInstr row
    -> one comp (rows=[v_q], cols = its kv list, DCP order) at 2 s; every replayed input transfer -> one comm at 2 s - 1
    (phase 1) / 2 s (phase 2, src = the relaying GPU); every OUT transfer -> one io='o' comm after the last stage.
    Only when Q heads == KV heads (MHA) and plan block = DCP block (no straddling): a GQA KV head serves 4 Q heads and
    has no own virtual block in this layout."""
    W, S = rp.W, rp.S
    Hq, Hk = heads
    if Hq != Hk or bt != B:
        return None
    n = Hq * nb
    owner = [plan["owner"][i] for h in range(Hq) for i in range(nb)]
    v = lambda i: W["in_head"][i] * nb + W["in_block"][i]                  # noqa: E731
    stv = np.full((n, n), -1, np.int64)
    for w in range(len(W["w_q"])):
        qi, ki = W["w_q"][w], W["w_kv"][w]
        assert W["in_head"][ki] == W["in_head"][qi]
        stv[v(qi), v(ki)] = W["w_dev"][w]
    gk = [[] for _ in range(P)]
    for r in range(P):
        for s in range(S):
            for q, kvs, _, _ in rp.attn.get((r, s), []):
                gk[r].append(dict(kind="comp", rows=[v(q)], cols=[v(x) for x in kvs], start=float(2 * s), fused=len(kvs) > 1))
    rcof = {"Q": "r", "KV": "c"}
    for (i, a, b, s, ph) in sorted(rp.transfers, key=lambda t: (2 * t[3] - (1 if t[4] == 1 else 0))):
        t = float(2 * s - 1 if ph == 1 else 2 * s)
        for gg in (a, b):
            gk[gg].append(dict(kind="comm", block=v(i), src=a, dst=b, io="i", rc=rcof[W["in_type"][i]], start=t))
    for (o, a, b) in rp.out_transfers:
        if W["out_type"][o] != "OUT":
            continue
        vb = W["out_head"][o] * nb + W["out_block"][o]
        for gg in (a, b):
            gk[gg].append(dict(kind="comm", block=vb, src=a, dst=b, io="o", rc="r", start=float(2 * (S - 1) + 1)))
    for g in range(P):
        gk[g].sort(key=lambda x: x["start"])
    return dict(family=plan["family"], strategy="dcp_official_perhead", name=plan["name"], cp=P, n_blocks=n,
                block_tokens=bt, owner=owner, schedule_table=stv.tolist(), gpu_kernels=gk, virtual_heads=Hq,
                blocks_per_head=nb, mask=plan["mask"], heads=[1, 1], source=plan["source"], steps=plan["steps"],
                note="per-head virtual blocks (head-major: h * blocks_per_head + i); heads=(1, 1) per virtual block")


# ------------------------------------------------------------------------------------------------ validation
def live_blocks(mask, nb, bt):
    k = bt // TILE
    return (mask.reshape(nb, k, nb, k) != 0).any((1, 3))


def validate(plan, live, rp=None, aux=None):
    """live: [n_blocks, n_blocks] bool, the plan-block pairs with a live tile"""
    owner, st, gk = plan["owner"], np.asarray(plan["schedule_table"]), plan["gpu_kernels"]
    res = {}
    res["live_pairs_assigned"] = bool(((st >= 0) == live).all())
    res["n_live_pairs"] = int(live.sum())
    cover = collections.Counter()
    comp_start = {}
    for g in range(P):
        for x in gk[g]:
            if x["kind"] == "comp":
                for q in x["rows"]:
                    for kv in x["cols"]:
                        cover[(q, kv, g)] += 1
                        comp_start[(q, kv)] = x["start"]
    want = {(int(q), int(kv), int(st[q, kv])) for q, kv in zip(*np.nonzero(st >= 0))}
    res["comp_covers_schedule_exactly_once"] = set(cover) == want and all(v == 1 for v in cover.values())
    # comm consistency: each comm listed on src and dst identically
    occ = collections.Counter()
    for g in range(P):
        for x in gk[g]:
            if x["kind"] == "comm":
                assert g in (x["src"], x["dst"])
                occ[(x["block"], x["src"], x["dst"], x["io"], x["rc"], x["start"])] += 1
    res["comm_on_both_ends"] = all(v == 2 for v in occ.values())
    recv_at = {}
    for (b, s, d, io, rc, t), _ in occ.items():
        if io == "i":
            assert (b, rc, d) not in recv_at, ("duplicate input transfer", b, rc, d)
            recv_at[(b, rc, d)] = (t, s)
    bad_in = 0; bad_relay = 0
    for (q, kv), t in comp_start.items():
        g = int(st[q, kv])
        for b, rc in ((q, "r"), (kv, "c")):
            if owner[b] != g and not ((b, rc, g) in recv_at and recv_at[(b, rc, g)][0] <= t):
                bad_in += 1
    for (b, rc, d), (t, s) in recv_at.items():
        if s != owner[b] and not ((b, rc, s) in recv_at and recv_at[(b, rc, s)][0] < t):
            bad_relay += 1
    res["remote_inputs_missing_or_late"] = bad_in
    res["relay_sources_not_holding"] = bad_relay
    outs = collections.Counter((b, s, d) for (b, s, d, io, rc, t) in occ if io == "o")
    last = collections.defaultdict(float)
    for (q, kv), t in comp_start.items():
        last[(q, int(st[q, kv]))] = max(last[(q, int(st[q, kv]))], t)
    exp = {(q, g, owner[q]) for (q, g) in last if g != owner[q]}
    res["partial_outputs_to_owner_exactly_once"] = set(outs) == exp and all(v == 1 for v in outs.values())
    res["partial_outputs_after_last_comp"] = all(t >= last[(b, s)] for (b, s, d, io, rc, t) in occ if io == "o")
    res["n_input_transfers"] = len(recv_at); res["n_output_transfers"] = len(outs)
    if rp is not None and aux is not None:
        res["volume_vs_dcp"] = volume_vs_dcp(plan, rp, aux, recv_at, outs)
    return res


def volume_vs_dcp(plan, rp, aux, recv_at, outs):
    """moved data in head-token units (real tokens x heads; KV counts K and V once each per kv head): this plan vs DCP's
    actual comm ops (replayed transfers) and DCP's WorkloadSpec sizes (bytes)"""
    W = rp.W
    Hq, Hk = aux["heads"]
    rt = aux["real_tokens_of_block"]
    ours = collections.Counter()
    for (b, rc, d) in recv_at:
        ours["Q" if rc == "r" else "KV"] += int(rt[b]) * (Hq if rc == "r" else Hk)
    for (b, s, d) in outs:
        ours["OUT"] += int(rt[b]) * Hq
    dcp = collections.Counter(); dcp_b = collections.Counter()
    for (i, a, b, s, ph) in rp.transfers:
        dcp[W["in_type"][i]] += W["in_ntok"][i]; dcp_b[W["in_type"][i]] += W["in_bytes"][i]
    for (o, a, b) in rp.out_transfers:
        dcp_b[W["out_type"][o]] += W["out_bytes"][o]
        if W["out_type"][o] == "OUT":
            dcp["OUT"] += aux["ntok"][W["out_block"][o]]
    # DCP's model (dedup per device, no relays): what the assignment needs
    need_in = set(); need_out = set()
    for w in range(len(W["w_q"])):
        d = W["w_dev"][w]
        for i in (W["w_q"][w], W["w_kv"][w]):
            if W["in_dev"][i] != d:
                need_in.add((i, d))
        o = W["w_out"][w]
        if W["out_dev"][o] != d:
            need_out.add((o, d))
    model = collections.Counter()
    for i, d in need_in:
        model[W["in_type"][i]] += W["in_ntok"][i]
    for o, d in need_out:
        model["OUT"] += aux["ntok"][W["out_block"][o]]
    return dict(unit="real tokens x heads", ours=dict(ours), dcp_comm_ops=dict(dcp), dcp_assignment_model=dict(model),
                ours_over_dcp_ops={k: (ours[k] / dcp[k] if dcp[k] else None) for k in ("Q", "KV", "OUT")},
                dcp_bytes_comm_ops=dict(dcp_b))


def perhead_mask(mask_path, name, H):
    """the virtual mask: H block-diagonal copies of this layout's mask"""
    z = np.load(mask_path)
    vm = os.path.join(out_dir(), f"{name}_ph{H}.npz")
    if not os.path.exists(vm):
        gq = int(z["heads"][0]) // int(z["heads"][1])
        o = dict(mask=np.kron(np.eye(H, dtype=np.int8), z["mask"].astype(np.int8)).astype(np.int8), heads=np.array([gq, 1]), tokens=np.array(H * int(z["tokens"])),
                 name=np.array(f"{name}_ph"), source=np.array(mask_path), virtual_heads=np.array(H))
        for key in ("tok_valid", "npad"):
            if key in z.files:
                o[key] = np.tile(z[key], H)
        np.savez(vm, **o)
    return vm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workload", required=True, choices=["w1", "w2", "w3"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--perhead", action="store_true", help="also write <id>_perhead.json (MHA, B-aligned only)")
    a = ap.parse_args()
    t0 = time.time()
    cache_dir = os.path.join(plan_dir(a.workload), "cache")
    B = ladder_B(a.workload, a.name)
    ext_dir = os.path.join(out_dir(), "_ext")
    extract(cache_dir, a.name, B, range(P), ext_dir, a.jobs)
    rp = Replay(load_extracts(ext_dir, a.name, B)).run()
    chk = check_replay(rp)
    mask_path = default_mask(a.workload, a.name)
    plan, aux, ph = convert(rp, mask_path, default_ranges(a.workload, a.name), a.name, a.workload, B, cache_dir, perhead=a.perhead)
    mask = np.load(mask_path)["mask"]
    live = live_blocks(mask, plan["n_blocks"], plan["block_tokens"])
    plan["stats"]["replay_checks"] = chk
    plan["stats"]["validation"] = validate(plan, live, rp, aux)
    out = os.path.join(out_dir(), f"{a.name}.json")
    json.dump(plan, open(out, "w"))
    json.dump({k: plan[k] for k in ("name", "family", "cp", "n_blocks", "block_tokens", "mask", "source", "stats")}, open(out[:-5] + ".stats.json", "w"), indent=1)
    if ph is not None:
        H = ph["virtual_heads"]
        ph["mask"] = perhead_mask(mask_path, a.name, H)
        ph["stats"] = dict(validation=validate(ph, np.kron(np.eye(H, dtype=bool), live)))
        print("[perhead] validation", ph["stats"]["validation"])
        json.dump(ph, open(out[:-5] + "_perhead.json", "w"))
    print(f"[dcp_plan] {a.name} B{B} -> {out} (n_blocks {plan['n_blocks']} x {plan['block_tokens']}) {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
