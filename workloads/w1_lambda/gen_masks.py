"""W1 mask generation: attention sink + sliding window at 128-token tile granularity.

Tile (i, j) is visible iff j <= i and (j < sink / 128 or j >= i - window / 128 + 1); the diagonal tile is causal.
The mask is shared by every head (MHA 16/16, head dimension 128).

Outputs, per sequence length, under $DEX_DATA_ROOT/masks/w1/:
  <name>.npz          mask [nt, nt] int8 (0 empty / 1 full / 2 causal diagonal), heads [2], tokens, block_tokens, name
  <name>_ranges.npy   per-token key ranges [S, 2, 2] (start, end) of the same mask: the FP64 reference and the
                      baselines that take token ranges read this file
Example: python -m workloads.w1_lambda.gen_masks --tokens 32768 65536 131072 262144
"""
import argparse
import os

import numpy as np

from dex.common.env import cfg

TILE = 128
HEADS = (16, 16)


def mask_name(tokens, sink, window):
    return f"w1_{tokens // 1024}k_sink{sink}_win{window // 1024}k"


def lambda_mask(nt, sink_tiles, window_tiles):
    """[nt, nt] int8 tile mask: sink tiles and the window (window_tiles including the diagonal) below the diagonal"""
    i = np.arange(nt)[:, None]
    j = np.arange(nt)[None, :]
    m = (((j < sink_tiles) | (j > i - window_tiles)) & (j < i)).astype(np.int8)
    m[np.arange(nt), np.arange(nt)] = 2
    return m


def token_ranges(tokens, sink, window):
    """[S, 2, 2] int32 key ranges of the tile mask at token level. A row whose window reaches the sink tiles sees one
    contiguous range [0, pos + 1) (range 0 empty); otherwise range 0 = the sink tiles and range 1 = the window tiles
    up to the row itself (the diagonal tile is causal)."""
    sink_t, win_t = sink // TILE, window // TILE
    pos = np.arange(tokens)
    tile = pos // TILE
    lo = np.maximum(tile - win_t + 1, 0)
    split = lo > sink_t
    r = np.zeros((tokens, 2, 2), np.int32)
    r[:, 0, 1] = np.where(split, sink_t * TILE, 0)
    r[:, 1, 0] = np.where(split, lo * TILE, 0)
    r[:, 1, 1] = pos + 1
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", type=int, nargs="+", default=[32768, 65536, 131072, 262144])
    ap.add_argument("--sink", type=int, default=128)
    ap.add_argument("--window", type=int, default=4096)
    a = ap.parse_args()
    out = os.path.join(cfg("DEX_DATA_ROOT"), "masks", "w1")
    os.makedirs(out, exist_ok=True)
    for tokens in a.tokens:
        nt = tokens // TILE
        m = lambda_mask(nt, a.sink // TILE, a.window // TILE)
        name = mask_name(tokens, a.sink, a.window)
        np.savez(os.path.join(out, name + ".npz"), mask=m, heads=np.array(HEADS), block_tokens=max(8192, tokens // 16),
                 tokens=tokens, name=name)
        r = token_ranges(tokens, a.sink, a.window)
        np.save(os.path.join(out, name + "_ranges.npy"), r)
        live = int((m > 0).sum())
        print(f"{name}: {nt} tiles, live {live} ({live / (nt * (nt + 1) / 2):.1%} of causal), "
              f"tiles per row {live / nt:.1f}", flush=True)


if __name__ == "__main__":
    main()
