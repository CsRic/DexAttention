"""Sparse all-gather pieces shared by the W5 and W6 ops: per-rank need lists, the point-to-point K / V exchange and the
block-sparse kernel tables over [own tiles, received tiles]."""
import torch
import torch.distributed as dist


def exchange(out, inp, out_splits, in_splits, mode):
    """all-to-all-v along dim 0: inp rows split by destination (in_splits), out rows by source (out_splits); the rank's own
    part must be empty. mode "batch": all point-to-point operations in one batch_isend_irecv; "staged": P - 1 shift rounds
    (round s: send to r + s, receive from r - s), one batch_isend_irecv per round."""
    W, r = dist.get_world_size(), dist.get_rank()
    ins = torch.split(inp, in_splits, 0)
    outs = torch.split(out, out_splits, 0)
    if mode == "batch":
        ops = []
        for s in range(1, W):
            d, src = (r + s) % W, (r - s) % W
            if ins[d].numel():
                ops.append(dist.P2POp(dist.isend, ins[d].contiguous(), d))
            if outs[src].numel():
                ops.append(dist.P2POp(dist.irecv, outs[src], src))
        if ops:
            for w in dist.batch_isend_irecv(ops):
                w.wait()
        return
    assert mode == "staged", mode
    for s in range(1, W):
        d, src = (r + s) % W, (r - s) % W
        ops = []
        if ins[d].numel():
            ops.append(dist.P2POp(dist.isend, ins[d].contiguous(), d))
        if outs[src].numel():
            ops.append(dist.P2POp(dist.irecv, outs[src], src))
        if ops:
            for w in dist.batch_isend_irecv(ops):
                w.wait()


class Lists:
    """the send / receive lists of one call. need [H, n] bool: this rank's needed key tiles (any live entry of its rows,
    own tiles included); t_own / t_loc [n]: owner and owner-local index of every global tile; nl: tiles per rank per head.
    Units = (head h, global tile t), ordered by (peer, h, t). Local slot of a unit = h * nl + t_loc[t].
      recv_units [nrecv, 2] (h, t) of the remote tiles this rank receives (source order), recv_splits per source
      send_idx [nsend] local slots this rank sends (destination order), send_splits per destination
      slot [H, n] compact K / V slot of every needed unit (local: h nl + loc; received: H nl + position), -1 otherwise
    """

    def __init__(self, need, t_own, t_loc, nl):
        P, rank = dist.get_world_size(), dist.get_rank()
        H, n = need.shape
        dev = need.device
        remote = need & (t_own != rank)[None, :]
        allneed = torch.empty((P, H, n), dtype=torch.uint8, device=dev)
        dist.all_gather_into_tensor(allneed, remote.to(torch.uint8).contiguous())
        allneed = allneed.bool()
        # receive: my remote needs by source rank, then (h, t)
        h_r, t_r = torch.nonzero(remote, as_tuple=True)
        src = t_own[t_r]
        o = torch.argsort((src * H + h_r) * n + t_r)
        h_r, t_r, src = h_r[o], t_r[o], src[o]
        # send: units (h, t) I own that rank d needs, by d then (h, t)
        mine = (t_own == rank)[None, None, :] & allneed  # [P, H, n]
        d_s, h_s, t_s = torch.nonzero(mine, as_tuple=True)
        cnt = torch.cat([torch.bincount(src, minlength=P), torch.bincount(d_s, minlength=P)]).cpu().tolist()
        self.recv_splits, self.send_splits = cnt[:P], cnt[P:]
        self.nrecv = len(h_r)
        self.recv_units = torch.stack([h_r, t_r], 1)
        self.send_idx = h_s * nl + t_loc[t_s]
        slot = torch.full((H, n), -1, dtype=torch.long, device=dev)
        loc = need & (t_own == rank)[None, :]
        hl, tl = torch.nonzero(loc, as_tuple=True)
        slot[hl, tl] = hl * nl + t_loc[tl]
        slot[h_r, t_r] = H * nl + torch.arange(self.nrecv, device=dev)
        self.slot = slot
        self.H, self.nl = H, nl

    def gather_kv(self, kt, vt, mode):
        """kt / vt [H nl, 128, D] this rank's virtual K / V tiles -> compact [H nl + nrecv, 128, D] each"""
        sb = torch.stack([kt[self.send_idx], vt[self.send_idx]], 1)  # [nsend, 2, 128, D]
        rb = torch.empty((self.nrecv,) + tuple(sb.shape[1:]), dtype=sb.dtype, device=sb.device)
        exchange(rb, sb, self.recv_splits, self.send_splits, mode)
        return torch.cat([kt, rb[:, 0]]), torch.cat([vt, rb[:, 1]])

    def return_dkv(self, dK, dV, mode):
        """dK / dV [H nl + nrecv, 128, D] fp32 compact partials -> this rank's own [H nl, 128, D]: the received tiles'
        partials go back to their owners (reverse exchange), summed into the owners' tiles"""
        HN = self.H * self.nl
        sb = torch.stack([dK[HN:], dV[HN:]], 1).contiguous()  # [nrecv, 2, 128, D] (source order)
        rb = torch.empty((len(self.send_idx),) + tuple(sb.shape[1:]), dtype=sb.dtype, device=sb.device)
        exchange(rb, sb, self.send_splits, self.recv_splits, mode)
        dk, dv = dK[:HN].clone(), dV[:HN].clone()
        dk.index_add_(0, self.send_idx, rb[:, 0])
        dv.index_add_(0, self.send_idx, rb[:, 1])
        return dk, dv


def sparse_fwd_tables(row, kind, slot, col, R, delta, hq):
    """sparse_fwd tables of one call: entries (query tile row, kind = causal diagonal, K / V tile slot (sparse_fwd64: 64-row
    slot), sort column); per row full tiles first (ascending col), causal last; work = heads x rows by descending count;
    causal geometry row_mask_mb = row, kv_mask_nb = row + delta."""
    dev = row.device
    key = (row * 2 + kind.long()) * (1 << 26) + col
    o = torch.argsort(key)
    row, kind, slot = row[o], kind[o], slot[o]
    cnt = torch.bincount(row, minlength=R)
    nmask = torch.bincount(row[kind], minlength=R)
    worder = torch.argsort(-cnt, stable=True)
    nwork = int((cnt > 0).sum().item())
    ptr = torch.zeros(R + 1, dtype=torch.int64, device=dev)
    ptr[1:] = torch.cumsum(cnt, 0)
    nb = torch.where(kind, row + delta, slot)
    w = (torch.arange(hq, device=dev)[:, None] * R + worder[:nwork][None, :]).reshape(-1)
    return (
        ptr.to(torch.int32),
        slot.to(torch.int32).contiguous(),
        nb.to(torch.int32).contiguous(),
        nmask.to(torch.int32),
        torch.arange(R, dtype=torch.int32, device=dev),
        w.to(torch.int32),
    )


def sparse_bwd_tables(row, kind, kslot, R, hq, hn_only=None):
    """sparse_bwd tasks of one call: tasks (K / V 128-row tile slot, 64-key half hn), hn-major then slot; items = (query tile
    slot, 64-row half mb, head, causal), rows descending, mb 1 then 0, heads ascending; causal diagonal: mb < hn skipped,
    causal iff mb == hn. hn_only [E] (128 x 64 masks): the entry is that half only."""
    dev = row.device
    hn = torch.tensor([0, 0, 1, 1], device=dev)
    mbv = torch.tensor([1, 0, 1, 0], device=dev)
    diag = kind[:, None]
    valid = ~diag | (mbv[None, :] >= hn[None, :])
    if hn_only is not None:
        valid &= hn[None, :] == hn_only[:, None]
    causal = diag & (mbv[None, :] == hn[None, :])
    e_i, c_i = torch.nonzero(valid, as_tuple=True)
    hn_i, mb_i = hn[c_i], mbv[c_i]
    tkey = hn_i * (1 << 24) + kslot[e_i]
    skey = (tkey * (1 << 24) + (R - 1 - row[e_i])) * 2 + (1 - mb_i)
    o = torch.argsort(skey)
    e_i, c_i, tkey, mb_i = e_i[o], c_i[o], tkey[o], mb_i[o]
    it = (row[e_i] << 9) | (mb_i << 8) | causal[e_i, c_i].long()
    items = (it[:, None] | (torch.arange(hq, device=dev)[None, :] << 1)).reshape(-1)
    utask, tcnt = torch.unique_consecutive(tkey, return_counts=True)
    tkv = (((utask % (1 << 24)) << 9) | ((utask // (1 << 24)) << 8)).to(torch.int32)
    ptr = torch.zeros(len(utask) + 1, dtype=torch.int64, device=dev)
    ptr[1:] = torch.cumsum(tcnt * hq, 0)
    return tkv.contiguous(), ptr.to(torch.int32), items.to(torch.int32).contiguous()
