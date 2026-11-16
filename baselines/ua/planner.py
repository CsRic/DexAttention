"""UA planning with the block-size ladder (planning cost: Table 1 / 4; plans: every UA row).

Per input and pass, block sizes from fine to coarse: B starts at min(4096, the largest power of two with >= 16 blocks) and
doubles when UA's planning does not finish within 120 s; fewer than 16 blocks -> N/A. Rollouts use their <id>_nb<k>.npz
layouts (the padded length depends on k), from the finest k with at least 2048 real tokens per block down to k = 16; Lambda
its one mask at each B. Stages:
  deployment  the context-tiling ILPs in this process (deploy.py, Gurobi); a rung is kept when every ILP solves to
              optimality (status 2) within the budget. Backward MCV uses the backward's bytes.
  fusion / schedule  one persistent process per head layout (third_party/ultraattn search_algo/dump_plan_structure.py
              --batch; start-up recorded once, not charged to inputs), UA's greedy fusion capped at 2 s, forward (fob 0)
              and backward (fob 1) plans; the backward's plan is also built on the forward's deployment for training.
Forward planning time = deployment + fusion / schedule; backward likewise. UA partition degree P: 64 for Lambda and every
backward, UA's load-imbalance rule for rollout forwards.
Outputs under $DEX_DATA_ROOT/plans/ua: problems/<rung>.json, <key>_assignment.json, <key>_plan.json, ladder_<w>_<worker>.json (all
attempts, times), manifest.json ("<input>|fwd|bwd|bwd_on_fwd": {plan, mask, block_tokens, total_s, deploy_s, fusion_s} or {na}).
Example: python baselines/ua/planner.py --workloads w1 w2 w3 [--worker 0 --nworkers 4]; --merge merges workers' records."""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from baselines.ua import deploy as DP  # noqa: E402
from dex.common.env import cfg, repo_root  # noqa: E402

TILE, D = 128, 128
NODES, GPN = 4, 4
BUDGET_S = 120.0         # planning limit per attempt
FUSION_SEARCH_S = 2.0    # UA's fusion search cap
RESERVE_FWD_S = 6.0      # deployment deadline of a forward attempt = budget - this (fusion search + its overhead)
RESERVE_BWD_S = 4.0      # backward: ILP overshoot + write-out
P_LAMBDA, P_BWD = 64, 64
ROLLOUT_MIN_REAL_TOKENS = 2048   # real tokens per plan block of a rollout's finest layout


def root():
    return os.path.join(cfg("DEX_DATA_ROOT"), "plans", "ua")


def ua_repo():
    return os.path.join(repo_root(), "third_party", "ultraattn")


def block_bytes(heads, B):
    """per plan block in MiB: Q, K + V, partial O + LSE (fp32)"""
    hq, hk = heads
    q = B * hq * D * 2
    return q / 2 ** 20, 2 * B * hk * D * 2 / 2 ** 20, (2 * q + B * hq * 4) / 2 ** 20


def block_bytes_bwd(heads, B):
    """backward, per plan block in MiB: Q-side bundle (Q, dO fp16 + LSE, Delta fp32), K + V, dQ partial, dK + dV partial (fp32)"""
    hq, hk = heads
    return ((2 * B * hq * D * 2 + 2 * B * hq * 4) / 2 ** 20, 2 * B * hk * D * 2 / 2 ** 20, B * hq * D * 4 / 2 ** 20, 2 * B * hk * D * 4 / 2 ** 20)


def block_weights(m, B):
    """live token-pair fraction per block pair"""
    k = B // TILE
    nb = m.shape[0] // k
    pairs = np.where(m == 1, TILE * TILE, np.where(m == 2, TILE * (TILE + 1) // 2, 0)).astype(np.float64)
    return pairs.reshape(nb, k, nb, k).sum(axis=(1, 3)) / (B * B)


def write_problem(stem, mask_path, B):
    """-> (problem file, heads)"""
    z = np.load(mask_path)
    m, T = z["mask"], int(z["tokens"])
    nb = T // B
    prob = DP.BlockProblem(stem, nb, block_weights(m, B).astype(np.float32), seqs=[(0, nb)], segs=[(0, nb, "g0")],
                           meta=dict(block_tokens=B, tile=TILE, tokens=T))
    os.makedirs(f"{root()}/problems", exist_ok=True)
    f = f"{root()}/problems/{stem}.json"
    with open(f, "w") as fh:
        fh.write(prob.to_json())
    return f, tuple(int(x) for x in z["heads"])


def rungs(workload, case):
    """[(rung stem, mask file, B)] fine -> coarse, the first with B >= the start rule"""
    Sv = case["S"]
    B0 = min(4096, 1 << max(0, (Sv // 16).bit_length() - 1))
    out = []
    if workload == "w1":
        B = B0
        while Sv >= 16 * B:
            out.append((f"{case['id']}_b{B}", case["mask"], B))
            B *= 2
        return out
    base = re.sub(r"_nb\d+\.npz$", "", case["mask"])
    for nb in (128, 64, 32, 16):
        f = f"{base}_nb{nb}.npz"
        if not os.path.exists(f):
            continue
        if Sv // nb >= ROLLOUT_MIN_REAL_TOKENS:
            out.append((f"{case['id']}_nb{nb}", f, int(np.load(f)["tokens"]) // nb))
    return out


def deploy_attempt(stem, mask_path, B, bwd, P, solver):
    """deployment ILPs of one rung -> attempt record (+ assignment file when finished)"""
    pf, heads = write_problem(stem, mask_path, B)
    prob = DP.BlockProblem.from_json(open(pf).read())
    qb, kvb, ob = block_bytes(heads, B)
    topo = DP.Topo(n_nodes=NODES, gpus_per_node=GPN)
    topo.q_bytes, topo.kv_bytes, topo.o_bytes = qb, kvb, ob
    if bwd:
        topo.bwd_bytes = block_bytes_bwd(heads, B)
    key = f"{stem}_{'uab' if bwd else 'ua'}"
    t0 = time.perf_counter()
    DP._DEADLINE = t0 + BUDGET_S - (RESERVE_BWD_S if bwd else RESERVE_FWD_S)
    try:
        asg = DP.ultraattn(prob, topo, solver=solver, P=P or None, time_limit=BUDGET_S)
    except Exception as e:  # noqa: BLE001  e.g. too few blocks for the topology: this rung cannot run
        DP._DEADLINE = None
        return dict(key=key, finished=False, deploy_s=round(time.perf_counter() - t0, 2), ilp_status={}, error=f"{type(e).__name__}: {e}"[:300])
    t_dep = time.perf_counter() - t0
    DP._DEADLINE = None
    sts = {k: int(v["status"]) for k, v in asg.info.items() if k.startswith("ilp_") and isinstance(v, dict)}
    fin = bool(sts) and all(x == 2 for x in sts.values()) and t_dep <= BUDGET_S
    rec = dict(key=key, finished=fin, deploy_s=round(t_dep, 2), ilp_status=sts, block_tokens=B, heads=list(heads), problem=pf, mask=mask_path)
    if fin:
        rec["assignment"] = f"{root()}/{key}_assignment.json"
        json.dump(dict(dev=asg.dev.tolist(), owner=asg.owner.tolist(), info={k: str(v) for k, v in asg.info.items()}), open(rec["assignment"], "w"))
    print(f"{key}: deployment {t_dep:.1f} s, ILP status {sorted(set(sts.values()))} -> {'finished' if fin else 'not finished'}", flush=True)
    return rec


def install_profile(platform):
    """the executor's measured kernel / communication profile, where UA's Prof_DB reads it"""
    dst = os.path.join(ua_repo(), "database", "local", platform)
    if not os.path.isdir(dst):
        shutil.copytree(os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles", platform), dst)


def fusion_batch(tasks, fob, tag):
    """one persistent dump_plan_structure --batch process per head layout -> {out file: seconds or None}, per layout start-up"""
    groups = {}
    for t in tasks:
        groups.setdefault(f"{t['heads']}x{t['kv_heads']}", []).append(t)
    res, init = {}, {}
    os.makedirs(f"{root()}/logs", exist_ok=True)
    for hs, items in groups.items():
        platform = f"A5000_gqa{hs}"
        install_profile(platform)
        bf = f"{root()}/logs/{tag}_{hs}_{os.getpid()}.json"
        json.dump(items, open(bf, "w"))
        env = dict(os.environ, CLUSTER_NAME="local", PLATFORM=platform, UA_GPU_PER_NODE=str(GPN), UA_FAITHFUL="1", GUROBI_NUM_THREADS="16",
                   PULP_TMPDIR=cfg("DEX_SCRATCH"), NCCL_SOCKET_IFNAME=cfg("DEX_NIC_IFNAME"), GLOO_SOCKET_IFNAME=cfg("DEX_NIC_IFNAME"),
                   PYTHONPATH=repo_root() + os.pathsep + os.environ.get("PYTHONPATH", ""))
        if cfg("DEX_GUROBI_LICENSE", ""):
            env["GRB_LICENSE_FILE"] = cfg("DEX_GUROBI_LICENSE")
        cmd = [cfg("DEX_PYTHON"), "-m", "torch.distributed.run", "--nproc_per_node", "1", "--master_port", str(int(cfg("DEX_MASTER_PORT")) + 100 + os.getpid() % 800),
               "search_algo/dump_plan_structure.py", "--batch", bf, "--problem", "-", "--strategy", "ultraattn", "--cp", str(NODES * GPN), "--nodes", str(NODES),
               "--transform-mode", "ua_greedy", "--rung", "w_kernel_tile", "--fob", str(fob)]
        with open(bf + ".log", "w") as lf:
            subprocess.run(cmd, cwd=ua_repo(), env=env, stdout=lf, stderr=subprocess.STDOUT)
        times = [json.loads(x) for x in open(bf + ".times.jsonl")] if os.path.exists(bf + ".times.jsonl") else [{}]
        init[hs] = times[0].get("init_s")
        for x in times[1:]:
            res[x["out"]] = x["seconds"] if x.get("ok") else None
    return res, init


def plan_input(workload, case, solver):
    """ladder of one input -> {"fwd": attempts + chosen, "bwd": ...}"""
    out = {}
    rs = rungs(workload, case)
    for ps in ("fwd", "bwd"):
        P = P_BWD if ps == "bwd" else (P_LAMBDA if workload == "w1" else 0)
        att = []
        for stem, mask, B in rs:
            r = deploy_attempt(stem, mask, B, ps == "bwd", P, solver)
            att.append(r)
            if r["finished"]:
                break
        ok = [x for x in att if x["finished"]]
        out[ps] = dict(attempts=att, chosen=ok[0] if ok else None, na=None if rs else "fewer than 16 blocks at the start block size")
    return out


def task_of(rec, out, fb=None):
    fb = fb if fb is not None else max(0.5, min(FUSION_SEARCH_S, BUDGET_S - rec["deploy_s"] - 1.0))
    return dict(problem=rec["problem"], assignment=rec["assignment"], out=out, block_tokens=rec["block_tokens"], heads=rec["heads"][0], kv_heads=rec["heads"][1],
                fusion_budget_s=round(fb, 2))


def finish(records):
    """fusion / schedule of every chosen rung, then the manifest entries"""
    fw, bw, bf = [], [], []
    for r in records.values():
        for ps, lst in (("fwd", fw), ("bwd", bw)):
            c = r[ps]["chosen"]
            if c:
                lst.append(task_of(c, f"{root()}/{c['key']}_plan.json"))
        c = r["fwd"]["chosen"]
        if c:
            bf.append(task_of(c, f"{root()}/{c['key']}_on_fwd_bwd_plan.json", fb=FUSION_SEARCH_S))
    tf, _ = fusion_batch(fw, 0, "fusion_fwd")
    tb, _ = fusion_batch(bw, 1, "fusion_bwd")
    tbf, _ = fusion_batch(bf, 1, "fusion_bwd_on_fwd")
    man = {}
    for cid, r in records.items():
        for ps, tm in (("fwd", tf), ("bwd", tb)):
            c = r[ps]["chosen"]
            if not c:
                man[f"{cid}|{ps}"] = dict(na=r[ps]["na"] or "UA's deployment ILPs did not finish within 120 s at every block size")
                continue
            of = f"{root()}/{c['key']}_plan.json"
            ft = tm.get(of)
            if ft is None or not json.load(open(of)).get("gpu_kernels"):
                man[f"{cid}|{ps}"] = dict(na="UA's fusion / schedule stage produced no execution plan")
                continue
            man[f"{cid}|{ps}"] = dict(plan=of, mask=c["mask"], block_tokens=c["block_tokens"], heads=c["heads"], deploy_s=c["deploy_s"], fusion_s=ft,
                                      total_s=round(c["deploy_s"] + ft, 2), within_budget=c["deploy_s"] + ft <= BUDGET_S)
        c = r["fwd"]["chosen"]
        if c and tbf.get(f"{root()}/{c['key']}_on_fwd_bwd_plan.json") is not None:
            man[f"{cid}|bwd_on_fwd"] = dict(plan=f"{root()}/{c['key']}_on_fwd_bwd_plan.json", mask=c["mask"], block_tokens=c["block_tokens"])
    return man


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workloads", nargs="+", default=["w1", "w2", "w3"])
    ap.add_argument("--ids", default="", help="comma list of input ids (default: all)")
    ap.add_argument("--solver", default="gurobi", choices=["gurobi", "highs"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1, help="inputs dealt round-robin; one worker per node so Gurobi has the node's cores")
    ap.add_argument("--merge", action="store_true", help="after all deployment workers: run the fusion stage, write manifest.json")
    ap.add_argument("--deploy-only", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(repo_root(), "experiments", "attention_layer"))
    import static_inputs as SI
    os.makedirs(root(), exist_ok=True)
    recf = lambda w: f"{root()}/ladder_{w}_{a.worker}.json"  # noqa: E731
    if not a.merge:
        for w in a.workloads:
            rec = json.load(open(recf(w))) if os.path.exists(recf(w)) else {}
            for i, c in enumerate(SI.inputs(w)):
                if i % a.nworkers != a.worker or c["id"] in rec or (a.ids and c["id"] not in a.ids.split(",")):
                    continue
                rec[c["id"]] = plan_input(w, c, a.solver)
                json.dump(rec, open(recf(w), "w"), indent=1)
    if a.merge or (a.nworkers == 1 and not a.deploy_only):
        import glob
        records = {}
        for f in sorted(glob.glob(f"{root()}/ladder_w?_*.json")):
            records.update(json.load(open(f)))
        man_f = f"{root()}/manifest.json"
        man = json.load(open(man_f)) if os.path.exists(man_f) else {}
        man.update(finish(records))
        json.dump(man, open(man_f, "w"), indent=1)
        print(f"{len(man)} manifest entries -> {man_f}", flush=True)


if __name__ == "__main__":
    main()
