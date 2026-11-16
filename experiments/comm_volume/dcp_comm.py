"""DCP on W1-W3: per-GPU received bytes from DCP's cached execution plans.

Each rank's plan cache <name>_B<B>.cp<rank>.pkl holds [(fwd ExecutionPlan, bwd ExecutionPlan, fwd WorkloadSpec, bwd
WorkloadSpec)]. The executor issues exactly the comm ops of its CommLaunchInstr instructions (CommOp(comm_type,
peer=(node, dev), buffer_block=(buffer_type, index, n_tokens))); every RECV op of rank r counts
    n_tokens * prod(buffer_info[buffer_type].per_token_shape) * element_size(buffer_info[buffer_type].dtype)
bytes, so DCP's own two-phase inter-node routing is counted on every hop. Peer (node, dev) -> rank node * 4 + dev.
Check: the SEND ops of all ranks addressed to r sum to r's RECV. A case counts only if all 16 ranks' plans are readable.
Inputs : common.dcp_dir(W) (ladder.json + cache/), common.static_names(W); the DCP package (third_party/dcp) importable.
Output : $DEX_RESULT_ROOT/comm_volume/recv/dcp_<W>.jsonl
Example: CUDA_VISIBLE_DEVICES= PYTHONPATH=third_party/dcp python experiments/comm_volume/dcp_comm.py --workload W2 --jobs 3
"""
import torch  # noqa: F401  (import before numpy keeps the C++ runtime load order of the DCP package)
import argparse
import collections
import json
import math
import os
import pickle
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

ESIZE = {0: 2, 1: 2, 2: 4, 3: 4}     # DCP DType: bf16, fp16, fp32, int


def plan_counts(path, rank):
    """-> {pass: {"recv": {"peer|kind": bytes}, "send": {...}}}"""
    fw, bw, _wlf, _wlb = pickle.load(open(path, "rb"))[0]
    out = {}
    for ps, ex in (("fwd", fw), ("bwd", bw)):
        me = ex.node_id * ex.n_devices_per_node + ex.local_device_id
        assert me == rank and ex.n_devices_per_node == C.GPN and ex.n_nodes * C.GPN == C.P, (path, me, rank)
        rc, sd = collections.Counter(), collections.Counter()
        for ins in ex.instructions:
            if type(ins).__name__ != "CommLaunchInstr":
                continue
            for op in ins.comm_ops:
                bb = op.buffer_block
                bi = ex.buffer_info[bb.buffer_type]
                nb = bb.n_tokens * math.prod(bi.per_token_shape) * ESIZE[bi.dtype]
                peer = op.peer[0] * C.GPN + op.peer[1]
                kind = bb.buffer_type.replace("Local", "").replace("Buffer", "")
                (rc if op.comm_type == 1 else sd)[(peer, kind)] += nb
        out[ps] = {"recv": {f"{p}|{k}": v for (p, k), v in rc.items()}, "send": {f"{p}|{k}": v for (p, k), v in sd.items()},
                   "dtypes": {k: ESIZE[v.dtype] for k, v in ex.buffer_info.items()}}
    return out


def job(args):
    name, B, cdir, rank = args
    path = os.path.join(cdir, f"{name}_B{B}.cp{rank}.pkl")
    if not os.path.exists(path):
        return name, rank, None
    try:
        return name, rank, plan_counts(path, rank)
    except Exception as e:  # noqa: BLE001  (an unreadable pickle excludes the case; the reason is recorded)
        return name, rank, {"error": repr(e)[:300]}


def main():
    C.assert_no_cuda()
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True, choices=["W1", "W2", "W3"])
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    tdir = C.dcp_dir(a.workload)
    lad = json.load(open(os.path.join(tdir, "ladder.json")))
    recs, tasks = {}, []
    for n in C.static_names(a.workload):
        v = lad.get(n)
        if not v or not v.get("done") or not v.get("B"):
            recs[n] = dict(system="DCP", workload=a.workload, case=n, na="no final block size in ladder.json")
            continue
        recs[n] = dict(system="DCP", workload=a.workload, case=n, B=v["B"])
        tasks += [(n, v["B"], os.path.join(tdir, "cache"), r) for r in range(C.P)]
    per = collections.defaultdict(dict)
    with Pool(a.jobs) as pool:
        for name, rank, cnt in pool.imap_unordered(job, tasks):
            per[name][rank] = cnt
    out = []
    for n, rec in recs.items():
        if "na" in rec:
            out.append(rec); continue
        recv = C.empty_recv(); sendto = {ps: [0] * C.P for ps in ("fwd", "bwd")}
        have = [r for r in range(C.P) if per[n].get(r) and "error" not in per[n][r]]
        for r in have:
            for ps in ("fwd", "bwd"):
                for key, b in per[n][r][ps]["recv"].items():
                    C.add(recv, ps, int(key.split("|")[0]), r, b)
                for key, b in per[n][r][ps]["send"].items():
                    d = int(key.split("|")[0])
                    if d != r:
                        sendto[ps][d] += b
        rec["ranks_counted"] = have
        if len(have) == C.P:
            rec["recv"] = recv
            rec["send_recv_consistent"] = all(sendto[ps][r] == recv[ps]["intra"][r] + recv[ps]["inter"][r]
                                              for ps in ("fwd", "bwd") for r in range(C.P))
        else:
            rec["na"] = f"plans readable for {len(have)}/{C.P} ranks"
        out.append(rec)
    C.write_jsonl(os.path.join(C.out_dir("recv"), f"dcp_{a.workload}.jsonl"), out)
    print(f"[dcp] {a.workload}: {sum('recv' in r for r in out)}/{len(out)} cases")


if __name__ == "__main__":
    main()
