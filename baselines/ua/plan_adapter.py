"""UltraAttn's dumped plan (third_party/ultraattn search_algo/dump_plan_structure.py) -> per-rank FusionSchedule. No torch.

Plan fields: cp, n_blocks, block_tokens, owner[block], gpu_kernels[gpu] = list of {kind: 'comp' | 'comm', ...}:
  comp: rows[q_blocks], cols[kv_blocks], fused, start, time;  comm: block, src, dst, io ('i' | 'o'), rc ('r' | 'c'), start, time
Steps are global phases: the ILP start times of ALL ranks are bucketed, so a comm edge lands in the same step on its
sender and receiver and each step's batch_isend_irecv has matching sends and receives on every rank. Buckets merge by
interval overlap: two kernels whose [start, start + time] intervals share an instant have no dependency (the ILP forces
succ.start >= pred.end), so they can share a step. Each block moves as its own point-to-point transfer, as in UA's executor."""
import json

from .fusion_schedule import CommOp, CompOp, FusionSchedule, Step


def ua_plan_to_fusion_schedule(plan, rank, time_eps=1e-9):
    if isinstance(plan, str):
        plan = json.load(open(plan))
    cp, N, bt = plan["cp"], plan["n_blocks"], plan["block_tokens"]
    owner = list(plan["owner"])
    gpu_kernels = plan.get("gpu_kernels") or []
    if not gpu_kernels:
        raise ValueError(f"plan (family={plan.get('family')}) has no gpu_kernels: build it with baselines/ua/planner.py")
    kernels = gpu_kernels[rank]
    all_times = sorted({float(k["start"]) for g in gpu_kernels for k in g})
    min_end = {}                       # min end among kernels at each distinct start, all ranks
    for g in gpu_kernels:
        for k in g:
            s = float(k["start"])
            e = s + float(k.get("time", 0.0))
            min_end[s] = min(min_end.get(s, e), e)
    t2b, bi, grp_end = {}, -1, float("-inf")
    for s in all_times:
        if bi < 0 or s >= grp_end - time_eps:     # past every group member's end: a next step
            bi += 1
            grp_end = min_end[s]
        else:                                     # inside all members' intervals: same step
            grp_end = min(grp_end, min_end[s])
        t2b[s] = bi
    nbuckets = bi + 1
    comp_by = [[] for _ in range(nbuckets)]
    comm_by = [[] for _ in range(nbuckets)]
    for k in kernels:
        b = t2b[float(k["start"])]
        if k["kind"] == "comp":
            comp_by[b].append(CompOp(list(k["rows"]), list(k["cols"]), bool(k.get("fused"))))
        else:
            comm_by[b].append(CommOp([int(k["block"])], int(k["src"]), int(k["dst"]), k["io"], k["rc"]))
    steps = [Step(number=b, comm=comm_by[b], comp=comp_by[b]) for b in range(nbuckets)]   # empty steps kept: index = global step
    init_q = [b for b in range(N) if owner[b] == rank]
    return FusionSchedule(world=cp, rank=rank, n_blocks=N, block_tokens=bt, owner=owner, steps=steps, init_q=init_q,
                          init_kv=list(init_q), owned_q=list(init_q))
