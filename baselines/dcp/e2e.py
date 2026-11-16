"""DCP's op for W2 / W3 training: the official planner's cached plan for this input and the official executor, as its own
training integration uses them.
Token placement = DCP's: every rank holds whole token blocks of B tokens (all heads), real tokens only (DCP plans the unpadded
sequence), in its local buffer order. Attention = the executor's forward / backward on its local block layout, converted
from / to the model's token-major q, k, v, o by index. DCP's executor switches one way (forward plan -> backward plan), so
each micro-batch builds two executors of its input's plan (one for the forward, one for the backward): built in begin(),
dropped in end(), inside the step (timed, as DCP's own set-up is in its training step). With selective recomputation the
recomputed forward reuses the first forward's O / LSE (dex.attn.cp_common.RECOMPUTE) and only re-selects the local layout.
Runs in DCP's environment (PyTorch 2.5.1 with the zero-byte all_to_all DCP requires, third_party/PINS.md)."""
import json
import os
import pickle

import numpy as np
import torch
import torch.distributed as dist

from dex.attn.cp_common import RECOMPUTE
from dex.common import fp64_check as F64
from dex.common.env import cfg


def _rows(workload, exec_plan, kind, dev_key, start_of_block, inputs=True):
    """(local row, local token, head) of every row of the `kind` blocks this device holds (input or output side)"""
    bm = workload.block_mapping
    metas = bm.input_id_to_meta if inputs else bm.output_id_to_meta
    bidx = (bm.input_id_to_buffer_index if inputs else bm.output_id_to_buffer_index)[dev_key]
    cu = exec_plan.local_cu_seqlens
    r_, t_, h_ = [], [], []
    for i, k in bidx.items():
        m = metas[i]
        if str(m.type).split(".")[-1] != kind:
            continue
        assert m.head_block_size == 1 and m.seq_id == 0, (m.head_block_size, m.seq_id)
        off = cu[k]; n = m.n_tokens
        r_.append(np.arange(off, off + n)); t_.append(start_of_block[m.block_id] + np.arange(n)); h_.append(np.full(n, m.head_id))
    r, t, h = (np.concatenate(x) for x in (r_, t_, h_))
    o = np.argsort(r)
    assert np.array_equal(r[o], np.arange(len(r))), f"{kind}: local rows not a contiguous cover"
    return t[o], h[o]


class DcpCP:
    def __init__(self, name, workload, z, heads, device):
        from dcp.core.common import ExecutionContext
        from dcp.core.cost_model import AttnRooflineCostModel, CommunicationCostModel
        self.rank, self.P = dist.get_rank(), dist.get_world_size()
        self.dev = device
        self.hq, self.hk = heads
        d = os.path.join(cfg("DEX_DATA_ROOT"), "plans", "dcp", workload)
        self.B = int(json.load(open(f"{d}/ladder.json"))[name]["B"])
        self.fw_ex, self.bw_ex, self.fw_wl, self.bw_wl = pickle.load(open(f"{d}/cache/{name}_B{self.B}.cp{self.rank}.pkl", "rb"))[0]
        nd = int(cfg("DEX_GPUS_PER_NODE"))
        self.key = (self.rank // nd, self.rank % nd)
        self.ctx = ExecutionContext(nd, self.P // nd, comm_cost_model=CommunicationCostModel(), comp_cost_model=AttnRooflineCostModel())
        # this rank's token blocks in its local (buffer) order: Q blocks of head 0
        bm = self.fw_wl.block_mapping; bidx = bm.input_id_to_buffer_index[self.key]
        blks = sorted((k, bm.input_id_to_meta[i].block_id, bm.input_id_to_meta[i].n_tokens) for i, k in bidx.items()
                      if str(bm.input_id_to_meta[i].type).endswith("Q") and bm.input_id_to_meta[i].head_id == 0)
        start, real, s = {}, [], 0
        for _, b, n in blks:
            start[b] = s; real.append(np.arange(b * self.B, b * self.B + n)); s += n
        real = np.concatenate(real)
        rid = F64.real_ids(z, int(z["tokens"]))                     # padded position -> real-token id (-1 pad)
        inv = np.full(rid.max() + 1, -1, np.int64); inv[rid[rid >= 0]] = np.flatnonzero(rid >= 0)
        self.tokens = torch.as_tensor(inv[real], device=device)      # this rank's tokens, as padded positions
        assert (inv[real] >= 0).all()
        t = lambda x: torch.as_tensor(x, device=device)              # noqa: E731
        self.q_idx = tuple(map(t, _rows(self.fw_wl, self.fw_ex, "Q", self.key, start)))
        self.kv_idx = tuple(map(t, _rows(self.fw_wl, self.fw_ex, "KV", self.key, start)))
        self.o_idx = tuple(map(t, _rows(self.fw_wl, self.fw_ex, "OUT", self.key, start, inputs=False)))
        self.do_idx = tuple(map(t, _rows(self.bw_wl, self.fw_ex, "dOUT", self.key, start)))   # forward local offsets, as get_local_q_kv
        self.dq_idx = tuple(map(t, _rows(self.bw_wl, self.bw_ex, "dQ", self.key, start, inputs=False)))
        self.dkv_idx = tuple(map(t, _rows(self.bw_wl, self.bw_ex, "dKV", self.key, start, inputs=False)))
        self.fx = self.bx = None

    # ---- the per-micro-batch executors (official get_executor; forward: prepare(forward); backward: switch)
    def begin(self):
        from dcp.runtime.flash_attention.executor import get_executor
        mk = lambda: get_executor(1, 128, self.ctx, fw_exec_plan=self.fw_ex, fw_workload=self.fw_wl,     # noqa: E731
                                  bw_exec_plan=self.bw_ex, bw_workload=self.bw_wl)
        self.fx = mk(); self.fx.prepare(forward=True)
        self.bx = mk(); self.bx.switch_to_backward()

    def end(self):
        for e in (self.fx, self.bx):
            if e is not None:
                e.deallocate_all_buffers()
        self.fx = self.bx = None

    def layout(self, q, k, v):
        """this rank's local Q / KV buffers in DCP's order (a local index select, no communication)"""
        (qt, qh), (kt, kh) = self.q_idx, self.kv_idx
        return q[qt, qh][:, None].contiguous(), torch.stack([k[kt, kh], v[kt, kh]], 1)[:, :, None].contiguous()

    def out(self, q, out_l):
        o = torch.zeros(q.shape, dtype=q.dtype, device=q.device)
        ot, oh = self.o_idx
        o[ot, oh] = out_l[:, 0].to(q.dtype)
        return o

    def fwd(self, q, k, v):
        lq, lkv = self.layout(q, k, v)
        out_l, lse_l = self.fx.forward(lq, lkv)
        out_l = out_l.to(q.dtype)                 # (the merged output back to the qkv dtype -- the backward
        return self.out(q, out_l), (lq, lkv, out_l, lse_l)   # reads O in it, and the saved / cached O stays half size)

    def bwd(self, saved, do, kshape):
        lq, lkv, out_l, lse_l = saved
        dt, dh = self.do_idx
        ldo = do[dt, dh][:, None].contiguous()
        dq_l, dkv_l = self.bx.backward(lq, lkv, out_l, ldo, lse_l)
        dq = torch.zeros(do.shape, dtype=do.dtype, device=do.device)
        dk = torch.zeros(kshape, dtype=do.dtype, device=do.device); dv = torch.zeros_like(dk)
        qt, qh = self.dq_idx; kt, kh = self.dkv_idx
        dq[qt, qh] = dq_l[:, 0].to(do.dtype)
        dk[kt, kh] = dkv_l[:, 0, 0].to(do.dtype); dv[kt, kh] = dkv_l[:, 1, 0].to(do.dtype)
        return dq, dk, dv



class _DcpFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, op):
        hit = RECOMPUTE.take()                                     # selective recompute: the first forward's O / LSE; the
        if hit is not None:                                        # local Q / KV layout is re-selected from the recomputed
            out_l, lse_l = hit                                     # q / k / v (an index select)
            lq, lkv = op.layout(q, k, v)
            o, saved = op.out(q, out_l), (lq, lkv, out_l, lse_l)
        else:
            o, saved = op.fwd(q, k, v)
            RECOMPUTE.put((saved[2], saved[3]))
        ctx.op, ctx.kshape = op, k.shape
        ctx.save_for_backward(*saved)
        return o

    @staticmethod
    def backward(ctx, do):
        dq, dk, dv = ctx.op.bwd(ctx.saved_tensors, do.contiguous(), ctx.kshape)
        return dq, dk, dv, None


class DcpAttention(DcpCP):
    def __call__(self, q, k, v):
        return _DcpFn.apply(q.contiguous(), k.contiguous(), v.contiguous(), self)


def make_e2e_op(d):
    """d: id, mask_path, mask, heads, pad, fwd_plan, bwd_plan, workload"""
    return DcpAttention(d["id"], d["workload"], np.load(d["mask_path"]), d["heads"], torch.device("cuda", torch.cuda.current_device()))
