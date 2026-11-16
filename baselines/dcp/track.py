"""One launch of the official DCP benchmark over a list of inputs (planning is per input; launches are expensive).

For every task {id, workload, ranges (.npy, two key ranges per token), S, mask}: the official planner at block size --block,
a per-rank plan cache, a planning limit (a rank past it writes <out>.over and exits 17; the ladder then doubles B for that
input); then, without --plan_only, the official executor's forward and backward latency (barrier before every call, slowest
rank, 3 warm-ups, median of 10). An untimed first input pays the launch's one-off NCCL and kernel set-up.
--fp64: FP64 sampled-row check on the real tokens after the timing.
Outputs: rank 0 appends plan records to --out (jsonl: id, S, B, plan_s, cached) and, when timing, attention-layer records to
$DEX_RESULT_ROOT/attention_layer/<w>/dcp_<fwd|bwd>_launch<k>.jsonl.
Example: DEX_PYTHON=$DEX_DCP_PYTHON tools/launch.sh baselines/dcp/track.py --tasks t.json --block 4096 --heads 32 8 \
         --cache $DEX_DATA_ROOT/plans/dcp/w3/cache --out r.jsonl --launch 1 --fp64"""
import argparse
import datetime
import json
import os
import sys
import threading
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "third_party", "dcp", "benchmark", "microbenchmark")]
from dex.common.env import cfg  # noqa: E402

import baselines.dcp  # noqa: E402,F401  (DCP's configuration, before DCP is imported)

os.environ.setdefault("DCP_BENCH_LATENCY", "1")
os.environ.setdefault("DCP_KV_HOST", cfg("DEX_MASTER_ADDR"))
os.environ.setdefault("DCP_REDIS_CMD", cfg("DEX_REDIS_SERVER"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

FP64_SMALL_S = 1 << 17   # from this length the check uses smaller chunks, next to DCP's fp32 buffers


def _assemble(workload, exec_plan, local_t, kind, glob, n_dev_node, rank):
    """place this rank's output blocks into the zeroed global tensor `glob` (the mapping of DCP's reconstruct_*_for_test)"""
    from dcp.core.block_table import BlockType
    cu = list(exec_plan.local_cu_seqlens)
    bm = workload.block_mapping
    for oid in range(len(workload.output_sizes)):
        meta = bm.output_id_to_meta[oid]
        dev = workload.output_to_device_map[oid]
        if dev[0] * n_dev_node + dev[1] != rank or meta.type != kind:
            continue
        h0 = meta.head_id * meta.head_block_size
        hb = meta.head_block_size
        n = meta.n_tokens
        st = meta.block_id * meta.block_size
        off = cu[bm.output_id_to_buffer_index[dev][oid]]
        if kind == BlockType.dKV:
            glob[st:st + n, :, h0:h0 + hb] = local_t[off:off + n]
        else:
            glob[st:st + n, h0:h0 + hb] = local_t[off:off + n]


def fp64_check(args, ad, inp, ranges_path, rank):
    """one more forward + backward on the official executor, outputs assembled to global tensors (each rank places its
    blocks, sum all-reduce); each rank checks a 1 / P slice of the rows"""
    from dcp.core.block_table import BlockType
    from dcp.core.instructions import BufferType as BT
    from dex.common import fp64_check as F64
    executor, fw, local_q, local_kv = ad["dcp"]
    executor.mode = "Forward"
    executor.cuda_graph = None
    executor.deallocate_buffers()
    executor.fw_exec_plan = fw
    executor.prepare(forward=True)
    executor.init_forward_input(local_q, local_kv)
    out, lse = executor.execute(fw, is_forward=True)
    out, lse = out.detach().clone(), lse.detach().clone()
    bw = ad["dcp_bw_exec_plan"]
    torch.cuda.synchronize()
    executor.bw_exec_plan = bw
    executor.switch_to_backward()
    executor.set_local_out_lse(out, lse)
    executor.init_forward_input(local_q, local_kv)
    executor.init_backward_input(ad["dcp_local_dout"])
    for bt_ in (BT.LOCAL_dQ, BT.LOCAL_dKV):              # persistent across calls: one fresh backward = zeroed accumulators
        if executor.buffers.get(bt_) is not None:
            executor.buffers[bt_].zero_()
    dq, dkv = executor.execute(bw, is_forward=False)
    dq, dkv = dq.detach().clone(), dkv.detach().clone()
    q, kv, do = (x.cuda() for x in inp[:3])
    S = q.shape[0]
    nd = args.n_devices_per_node
    O = torch.zeros_like(q)
    dQ = torch.zeros_like(q)
    dKV = torch.zeros_like(kv)
    _assemble(ad["dcp_fw_workload"], fw, out, BlockType.Out, O, nd, rank)
    _assemble(ad["dcp_bw_workload"], bw, dq, BlockType.dQ, dQ, nd, rank)
    _assemble(ad["dcp_bw_workload"], bw, dkv, BlockType.dKV, dKV, nd, rank)
    for t in (O, dQ, dKV):
        dist.all_reduce(t)                               # disjoint blocks: the sum is the assembly
    W = dist.get_world_size()
    ids = np.arange(S * rank // W, S * (rank + 1) // W)
    it = torch.as_tensor(ids, device="cuda")
    small = S >= FP64_SMALL_S
    if small:
        torch.cuda.empty_cache()
    return F64.check(q, kv[:, 0], kv[:, 1], do, np.load(ranges_path), q.shape[-1] ** -0.5, dict(o=O[it], dq=dQ[it], dk=dKV[it, 0], dv=dKV[it, 1]),
                     q_rows=ids, kv_rows=ids, group=dist.group.WORLD, **(dict(ck=1024, cq=256) if small else {}))


def _write_record(t, ps, launch, ms, e64, B):
    keys = ("o",) if ps == "fwd" else ("dq", "dk", "dv")
    e = {k: e64[k] for k in keys if e64 and k in e64}
    rec = dict(input_id=t["id"], workload=t["workload"].upper(), S=t["S"], system="dcp", **{"pass": ps}, launch=launch, lat_ms=ms, lat_ms_all=None,
               max_rel_err=None, fp64_rel_l2={k: v["rel_l2"] for k, v in e.items()} or None, fp64_rows={k: v["rows"] for k, v in e.items()} or None,
               status="ok", reason="", block_tokens=B)
    d = os.path.join(cfg("DEX_RESULT_ROOT"), "attention_layer", t["workload"])
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"dcp_{ps}_launch{launch}.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--heads", type=int, nargs=2, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", required=True, help="plan cache directory")
    ap.add_argument("--plan_only", action="store_true")
    ap.add_argument("--plan_limit", type=float, default=120.0)
    ap.add_argument("--kv_base", type=int, default=0, help="redis port base of this launch (one fresh port per input)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--launch", type=int, default=1)
    ap.add_argument("--fp64", action="store_true")
    me = ap.parse_args()
    hq, hk = me.heads
    sys.argv = [sys.argv[0], "-b", "dcp", "-m", "ranges_file", "--seqlens", "128", "--n-heads", str(hq), "--n-query-groups", str(hk), "--head-dim", "128",
                "--dtype", "float16", "--block-size", str(me.block), "-nd", cfg("DEX_GPUS_PER_NODE"), "--warmup", str(me.warmup), "--iters", str(me.iters)]
    import benchmark_attention as BA
    import utils as U
    args = BA.parse_args()
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=60), pg_options=dist.ProcessGroupNCCL.Options(is_high_priority_stream=True))
    BA.init_env()
    n = dist.get_world_size()
    args.n_devices_per_node = min(args.n_devices_per_node, n)
    cp_groups = [dist.new_group(list(range(n)), backend="nccl")]
    dp_groups = [dist.new_group([j], backend="nccl") for j in range(n)]
    rank = dist.get_rank()
    args.cp_rank, args.dp_rank = rank, 0
    args.all_cp_groups, args.all_dp_groups = cp_groups, dp_groups
    args.cp_group, args.dp_group = cp_groups[0], dp_groups[rank]
    args.cp_world_size, args.curr_cpg_ranks, args.curr_dpg_ranks = n, list(range(n)), [rank]
    os.makedirs(me.cache, exist_ok=True)
    tasks = json.load(open(me.tasks))
    if not me.plan_only and tasks:
        tasks = [dict(tasks[0], _discard=True)] + tasks
    kv_base = me.kv_base or int(cfg("DEX_MASTER_PORT")) + 1000
    for ti, t in enumerate(tasks):
        S, B = int(t["S"]), me.block
        os.environ["DCP_KV_PORT"] = str(kv_base + ti)    # DCP's dataloader starts one redis server per input, kept until exit
        k = max(1, S // B // n)
        args.block_size, args.mem_epsilon, args.seqlens = B, min(0.2, 1.0 / k), [S]
        os.environ["DCP_RANGES_FILE"] = t["ranges"]
        cache = f"{me.cache}/{t['id']}_B{B}"
        os.environ["DCP_PLAN_CACHE"] = cache
        cached = os.path.exists(f"{cache}.cp{rank}.pkl")
        guard = None
        if not cached:
            def _over():
                print(f"[dcp_track] rank {rank}: {t['id']} planning over {me.plan_limit:.0f} s at B {B}", flush=True)
                open(me.out + ".over", "w").close()
                os._exit(17)
            guard = threading.Timer(me.plan_limit, _over)
            guard.daemon = True
            guard.start()
        t0 = time.time()
        inputs = list(BA.generate_input(args, [[S]]))
        arg_dicts = list(BA.prepare_and_gen_args(args, inputs))
        plan_s = time.time() - t0
        if guard is not None:
            guard.cancel()
        ps = torch.tensor([plan_s], device="cuda")
        dist.all_reduce(ps, op=dist.ReduceOp.MAX)
        rec = dict(id=t["id"], S=S, B=B, plan_s=0.0 if cached else round(ps.item(), 2), cached=cached)
        lat = {}
        if not me.plan_only:
            for bt, tag in (("forward", "fwd"), ("backward", "bwd")):
                args.benchmark_type = bt
                torch.cuda.synchronize()
                dist.barrier()
                BA._bench_dcp(args, arg_dicts[0])
                lat[tag] = round(U.LAST["latency_ms"], 3)
        e64 = None
        if me.fp64 and not me.plan_only and not t.get("_discard"):
            e64 = fp64_check(args, arg_dicts[0], inputs[0], t["ranges"], rank)
        if rank == 0 and not t.get("_discard"):
            with open(me.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            for tag, ms in lat.items():
                _write_record(t, tag, me.launch, ms, e64, B)
            print("[dcp_track]", json.dumps(rec), lat, flush=True)
        del arg_dicts, inputs
        torch.cuda.empty_cache()
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
