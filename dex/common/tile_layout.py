"""Zigzag token layout of whole tiles (W6, every system): 2P equal chunks of `per` whole 128-token tiles over the padded sequence S_pad,
rank r holding chunks r and 2P-1-r (its tokens in that order = its global tiles ascending)."""

import numpy as np
import torch

T = 128


class TileLayout:
    def __init__(self, S_pad, P, rank, dev):
        assert S_pad % (2 * P * T) == 0, (S_pad, P)
        self.S, self.P, self.rank, self.dev = S_pad, P, rank, dev
        self.s_real = None
        self.per = S_pad // (2 * P * T)
        self.cs = self.per * T
        self.chunks = [(i * self.cs, (i + 1) * self.cs) for i in range(2 * P)]
        self.my = [rank, 2 * P - 1 - rank]
        self.L, self.nl, self.n, self.nbh = 2 * self.cs, 2 * self.per, 2 * P * self.per, 2 * P
        self.tok = torch.cat([torch.arange(*self.chunks[c], device=dev) for c in self.my])
        self.gtile = torch.as_tensor(self.tiles_of(rank), device=dev)  # [nl] global tile of each local tile
        self.own_b = np.array([c if c < P else 2 * P - 1 - c for c in range(2 * P)], np.int64)  # owner per block
        t_own = np.repeat(self.own_b, self.per)  # owner of every global tile
        t_loc = np.zeros(self.n, np.int64)  # its index in the owner's tile list
        for r in range(P):
            t_loc[self.tiles_of(r)] = np.arange(self.nl)
        self.t_own = torch.as_tensor(t_own, device=dev)
        self.t_loc = torch.as_tensor(t_loc, device=dev)
        self.rows_of = torch.as_tensor(np.concatenate([self.tiles_of(r) for r in range(P)]), device=dev)

    def tiles_of(self, r):
        p, P = self.per, self.P
        return np.r_[r * p : (r + 1) * p, (2 * P - 1 - r) * p : (2 * P - r) * p]
