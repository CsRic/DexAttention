"""W5 helpers shared by w5_comm.py, balance_dynamic.py and direct_fetch.py: the per-head selection mask of one operator-test
case exactly as the Dex W5 path builds it (each rank pools its own keys, the pooled keys are assembled globally, each
rank selects its own query rows) and the Dex planner inputs derived from it.

  mask_of(case_dir, call, dev) -> mh [H, n_pad, 2 n_pad] uint8 over padded 64-key half tiles, layout lay0, density
  plan_inputs(mh, lay0, dev)   -> Wb (work per head block pair), ob (owners), gid, kb (live key columns), per, nbh
  expand(pb, H, nbh, dev)      -> [H nbh, H nbh] block-pair table (-1 off the per-head diagonal blocks)
"""
import numpy as np
import torch

from dex.common.cp_layout import Layout
from workloads.w5_sparge.selection import key_blocks_of, pooled_k, pooled_q, select

H, D, T = 12, 128, 128


def load_qk(case_dir, call, dev):
    """the case's stored q / k [H, S, D] (written by the generation hook on 4 GPUs, each holding 3 of the 12 heads)"""
    q, k = [], []
    for r in range(4):
        z = np.load(f"{case_dir}/rank{r}/call{call:05d}.npz")
        q.append(z["q"][0]); k.append(z["k"][0])
    return torch.from_numpy(np.concatenate(q)).to(dev), torch.from_numpy(np.concatenate(k)).to(dev)


def mask_of(case_dir, call, dev, P=16):
    Qf, Kf = load_qk(case_dir, call, dev)
    S = Qf.shape[1]
    lays = [Layout(S, P, r, dev) for r in range(P)]
    lay0 = lays[0]
    pk = torch.zeros(H, lay0.nkb, D, dtype=torch.float32, device=dev)
    for r, lay in enumerate(lays):
        pk[:, torch.from_numpy(key_blocks_of(lay, r)).to(dev)] = pooled_k(Kf[:, lay.tok])
    mh = torch.zeros(H, lay0.n_pad, 2 * lay0.n_pad, dtype=torch.uint8, device=dev)
    sel, tot = 0.0, 0
    for r, lay in enumerate(lays):
        fm = select(pooled_q(Qf[:, lay.tok]), pk, D ** -0.5)
        sel += fm.float().sum().item(); tot += fm.numel()
        mh[:, torch.from_numpy(lay.tiles_of(r)).to(dev)] = lay.tile_rows(fm).to(torch.uint8)
    return mh, lay0, lays, sel / tot, S


def plan_inputs(mh, lay0, dev, P=16):
    per = lay0.per; nbh = lay0.n_pad // per
    m_ = mh != 0
    Wb = m_.view(H, nbh, per, nbh, 2 * per).sum((2, 4)).float() * 0.5
    m128 = m_.view(H, lay0.n_pad, lay0.n_pad, 2).any(-1)
    kb = m128.any(1).view(H, nbh, per).sum(2).float()
    ownb = np.array([c if c < P else 2 * P - 1 - c for c in range(2 * P)], np.int64)
    own_v = np.tile(ownb, H)
    ob = torch.as_tensor(own_v, device=dev).view(H, nbh)
    gid = torch.arange(H * nbh, device=dev).view(H, nbh)
    return dict(Wb=Wb, ob=ob, gid=gid, kb=kb, per=per, nbh=nbh, ownb=ownb, own_v=own_v, B=per * T)


def expand(pb, nbh, dev):
    full = torch.full((H * nbh, H * nbh), -1, dtype=torch.long, device=dev)
    ar = torch.arange(H, device=dev)
    full.view(H, nbh, H, nbh)[ar, :, ar, :] = pb
    return full
