"""Token layout of per-head block masks (W5 operator tests and the Wan training step, every system): 2P zigzag chunks of `per` whole
128-token tiles, rank r holding chunks r and 2P-1-r. per = ceil(n / 2P) for n = ceil(S / 128) real tiles, so
2P * per - n = k pad tiles (k < 2P). The padding is spread: one pad tile at the end of each of the last k chunks, so every
rank's real tile count differs from every other's by at most 1. Every chunk keeps `per` tile slots (the planner, the
scheduler and the buffers see equal blocks); its real tokens are contiguous and come first.

  lay = Layout(S, P, rank, dev)
  lay.chunks          real token ranges [a, b) of the 2P chunks
  lay.tok             this rank's real token ids (its chunks in order)         -> model-side gather
  lay.pad(x) / unpad  [H, L_real, ...] <-> [H, L_pad = 2 per T, ...] (zero pad rows)
  lay.pos_real        padded global position of every real token [S]
  lay.tile_rows(fm)   this rank's rows of a 128 x 64 mask over REAL key blocks -> [H, nl, 2 n_pad] bool over PADDED
                      64-key half tiles (column 2 t + hn = padded tile t, half hn)
  lay.tiles_of(r)     padded global tile ids of rank r (its slot order)"""

import numpy as np
import torch

T = 128


def owned(r, P):
    return [r, 2 * P - 1 - r]


class Layout:
    def __init__(self, S, P, rank, dev):
        self.S, self.P, self.rank = S, P, rank
        n = -(-S // T)
        per = -(-n // (2 * P))
        n_pad = 2 * P * per
        self.n, self.per, self.n_pad = n, per, n_pad
        k = n_pad - n
        pad_c = np.zeros(2 * P, np.int64)
        pad_c[2 * P - k :] = 1
        real_tiles = per - pad_c
        a = np.concatenate([[0], np.cumsum(real_tiles)[:-1]]) * T
        b = np.minimum(a + real_tiles * T, S)
        assert b[-1] == S and np.all(b > a), (S, P, per, k)
        self.chunks = [(int(x), int(y)) for x, y in zip(a, b)]
        self.pad_c = pad_c
        self.my = owned(rank, P)
        self.lens = [self.chunks[c][1] - self.chunks[c][0] for c in self.my]
        self.L = sum(self.lens)
        self.L_pad = 2 * per * T
        self.nl = 2 * per
        self.tok = torch.from_numpy(np.concatenate([np.arange(*self.chunks[c]) for c in self.my])).to(dev)
        self.pidx = torch.from_numpy(
            np.concatenate([j * per * T + np.arange(ln) for j, ln in enumerate(self.lens)])
        ).to(dev)
        self.qslot = torch.from_numpy(
            np.concatenate([j * per + np.arange(-(-ln // T)) for j, ln in enumerate(self.lens)])
        ).to(dev)
        pos = np.concatenate([c * per * T + np.arange(bb - aa) for c, (aa, bb) in enumerate(self.chunks)])
        self.pos_real = torch.from_numpy(pos).to(dev)
        # real 64-key block -> padded half-tile column (chunk starts are whole tiles, so a block never straddles)
        nkb = -(-S // 64)
        self.nkb = nkb
        self.kb_col = torch.from_numpy(pos[np.arange(nkb) * 64] // 64).to(dev)

    def owned(self, r):
        return owned(r, self.P)

    def tiles_of(self, r):
        return np.concatenate([np.arange(c * self.per, (c + 1) * self.per) for c in owned(r, self.P)])

    def pad(self, x):
        out = x.new_zeros((x.shape[0], self.L_pad) + tuple(x.shape[2:]))
        out[:, self.pidx] = x
        return out

    def unpad(self, x):
        return x[:, self.pidx]

    def tile_rows(self, fm):
        """[H, nqb_local, nkb] (this rank's 128-row blocks x REAL 64-key blocks) -> [H, nl, 2 n_pad] over padded half tiles"""
        H = fm.shape[0]
        mr = torch.zeros(H, self.nl, 2 * self.n_pad, dtype=torch.bool, device=fm.device)
        mr[:, self.qslot[:, None], self.kb_col[None, :]] = fm
        return mr

    def block_owners(self):
        """owner rank of every zigzag chunk, [2P]"""
        P = self.P
        return np.array([c if c < P else 2 * P - 1 - c for c in range(2 * P)], np.int64)
