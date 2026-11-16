"""Table 2 communication columns: per-message exchange time (send and receive posted together), slowest rank, with torch.distributed batch_isend_irecv on
one process group (the decomposed executors' transport), message size 256 B ... 64 MiB, both directions at once:
  pair   one node, one GPU pair (local 0 <-> 1), the other GPUs idle (launch with DEX_NODES=1)
  inter  two nodes x 4 GPUs, every GPU <-> the same local index on the other node (launch with DEX_NODES=2)
--tag nccl_default leaves the transport to NCCL; otherwise the launcher's environment holds (forced GDR). One message: barrier, post send
+ recv, wait; median of 9 after 3 warm-ups, slowest rank.
Output : one record per size appended to --out (rank 0)
Example: DEX_NODES=2 tools/launch.sh experiments/microbench/comm.py --pattern inter --tag nccl_gdr --out results/microbench/comm.jsonl
"""
import argparse
import json
import os
import time

import torch
import torch.distributed as dist

SIZES = [2 ** e for e in range(8, 27)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True, choices=["pair", "inter"])
    ap.add_argument("--tag", default=""); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.tag == "nccl_default":     # NCCL's own transport choice, not the GPUDirect RDMA setting of the launcher's common profile
        for v in ("NCCL_NET_GDR_LEVEL", "NCCL_NET_GDR_READ"):
            os.environ.pop(v, None)
    dist.init_process_group("nccl")
    r, W = dist.get_rank(), dist.get_world_size()
    gpn = int(os.environ["DEX_GPUS_PER_NODE"])
    torch.cuda.set_device(r % gpn)
    node, lg = r // gpn, r % gpn
    if a.pattern == "pair":
        assert W == gpn, "pair: one node"
        peer = {0: 1, 1: 0}.get(r, -1)
    else:
        assert W == 2 * gpn, "inter: two nodes"
        peer = (1 - node) * gpn + lg
    t = torch.zeros(1, device="cuda", dtype=torch.float64)
    for sz in SIZES:
        sb = torch.empty(sz, dtype=torch.uint8, device="cuda"); rb = torch.empty_like(sb)

        def one():
            if peer >= 0:
                for q in dist.batch_isend_irecv([dist.P2POp(dist.isend, sb, peer), dist.P2POp(dist.irecv, rb, peer)]):
                    q.wait()
            torch.cuda.synchronize()

        def timed():
            dist.barrier(); torch.cuda.synchronize(); t0 = time.perf_counter(); one(); return time.perf_counter() - t0
        for _ in range(3):
            timed()
        t[0] = sorted(timed() for _ in range(9))[4]
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        s = t.item()
        if r == 0:
            rec = dict(impl="nccl", pattern=a.pattern, tag=a.tag, bytes=sz, us=round(s * 1e6, 2), GBps=round(sz / s / 1e9, 3))
            print(json.dumps(rec), flush=True)
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
