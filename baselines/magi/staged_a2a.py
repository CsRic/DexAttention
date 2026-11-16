"""Staged all-to-all: W - 1 shift rounds (round s: send to r + s, receive from r - s), installed over
torch.distributed.all_to_all_single. Staged rounds replace NCCL's all-to-all; the exchanged bytes are unchanged.
install_event(): Magi issues all_to_all_single(async_op=True), overlaps it with compute and accepts a Work / Stream / Event
handle: the rounds run on a side stream after the current stream's pending work, the handle is a CUDA event at their end."""
import torch
import torch.distributed as dist


def _splits(sizes, n, W):
    if sizes is None or len(sizes) == 0:
        assert n % W == 0, (n, W)
        return [n // W] * W
    return [int(s) for s in sizes]


class _Done:
    def wait(self, *a, **k):
        return True

    def is_completed(self):
        return True


def staged_all_to_all_single(output, input, output_split_sizes=None, input_split_sizes=None, group=None, async_op=False):
    W = dist.get_world_size(group)
    if W == 1:
        if output.data_ptr() != input.data_ptr():
            output.copy_(input)
        return _Done() if async_op else None
    r = dist.get_rank(group)
    to_global = (lambda i: dist.get_global_rank(group, i)) if group is not None else (lambda i: i)
    if output.data_ptr() == input.data_ptr() or output.untyped_storage().data_ptr() == input.untyped_storage().data_ptr():
        input = input.clone()
    ins = torch.split(input, _splits(input_split_sizes, input.size(0), W), 0)
    outs = torch.split(output, _splits(output_split_sizes, output.size(0), W), 0)
    outs[r].copy_(ins[r])
    for s in range(1, W):
        dst, src = (r + s) % W, (r - s) % W
        ops = []
        if ins[dst].numel():
            ops.append(dist.P2POp(dist.isend, ins[dst].contiguous(), to_global(dst), group=group))
        if outs[src].numel():
            ops.append(dist.P2POp(dist.irecv, outs[src], to_global(src), group=group))
        if ops:
            for w in dist.batch_isend_irecv(ops):
                w.wait()
    return _Done() if async_op else None


_side = {}


def staged_all_to_all_single_event(output, input, output_split_sizes=None, input_split_sizes=None, group=None, async_op=False):
    dev = output.device
    s = _side.get(dev)
    if s is None:
        s = _side[dev] = torch.cuda.Stream(device=dev)
    cur = torch.cuda.current_stream(dev)
    s.wait_stream(cur)
    with torch.cuda.stream(s):
        staged_all_to_all_single(output, input, output_split_sizes, input_split_sizes, group=group, async_op=False)
        ev = torch.cuda.Event()
        ev.record(s)
    output.record_stream(s)
    input.record_stream(s)
    if not async_op:
        cur.wait_event(ev)
        return None
    return ev


def install():
    dist.all_to_all_single = staged_all_to_all_single
    torch.distributed.all_to_all_single = staged_all_to_all_single


def install_event():
    dist.all_to_all_single = staged_all_to_all_single_event
    torch.distributed.all_to_all_single = staged_all_to_all_single_event
