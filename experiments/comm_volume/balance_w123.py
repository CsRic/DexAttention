"""Per-GPU compute balance of every system's plan on W1 / W2 / W3 (Table 4 imbalance column). CPU only.

Unit: exact masked token-pair area x query heads, counted on the real (unpadded) tokens. Oracle: the per-token key ranges
[T, R, 2] of the input; for a contiguous real-row range and key boundaries x, F(x) = sum_rows sum_ranges clip(x - s, 0,
e - s), so any rectangle's area is F(kb) - F(ka) (sorted prefix sums). A padded tile mask maps to real tokens through
its tok_valid field (monotone), so a 128-token padded tile is a contiguous real-token range.
Who computes what:
  Dex  head_split=4 plans: 4 block-diagonal copies of the tile mask; schedule_table[i][j] = GPU of block pair (i, j);
         work x hq / 4 per copy.
  Ring   zigzag: rank r computes its own query rows (2P equal chunks) against all keys.
  UA     plan JSON: schedule_table over the plan's tile mask, block_tokens B.
  DCP    balance/dcp/<W>_<case>.npz (balance_dcp_units.py): device and exact area per work unit.
  Magi   balance/magi_hosts.json (balance_magi_hosts.py): rank computes its hosted query tokens against all keys.
Imbalance = max over GPUs of the GPU's work / mean.
Output : $DEX_RESULT_ROOT/comm_volume/balance/<W>.json  {case: {system: {fwd: {imb, work}, bwd: {...}}}}
Example: CUDA_VISIBLE_DEVICES= python experiments/comm_volume/balance_w123.py W1 W2 W3
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

P, TILE = C.P, C.TILE


def heads(W):
    return (32, 8) if W == "W3" else (16, 16)


class Oracle:
    def __init__(self, R):
        self.s = R[:, :, 0].astype(np.int64); self.e = R[:, :, 1].astype(np.int64)
        self.T = R.shape[0]
        self.rowcum = np.concatenate([[0], np.cumsum(np.clip(self.e - self.s, 0, None).sum(1))])
        self.total = int(self.rowcum[-1])

    def F(self, qa, qb, xs):
        s = self.s[qa:qb].ravel(); e = self.e[qa:qb].ravel()
        ok = s < e; s = np.sort(s[ok]); e = np.sort(e[ok])
        cs = np.concatenate([[0], np.cumsum(s)]); ce = np.concatenate([[0], np.cumsum(e)])
        xs = np.asarray(xs, np.int64)
        i = np.searchsorted(s, xs, "left"); j = np.searchsorted(e, xs, "left")
        return (i * xs - cs[i]) - (j * xs - ce[j])

    def rows(self, qa, qb):
        return int(self.rowcum[min(qb, self.T)] - self.rowcum[min(qa, self.T)])


def tile_area(npz, orc):
    """-> A [n, n] real area per padded tile pair, tile mask m"""
    z = np.load(npz)
    m = z["mask"].astype(np.int8); n = m.shape[0]
    valid = z["tok_valid"].astype(bool) if "tok_valid" in z.files else np.ones(n * TILE, bool)
    cv = np.concatenate([[0], np.cumsum(valid)])
    assert cv[-1] == orc.T, (npz, cv[-1], orc.T)
    xs = cv[np.arange(n + 1) * TILE]
    A = np.zeros((n, n), np.int64)
    for t in range(n):
        A[t] = np.diff(orc.F(xs[t], xs[t + 1], xs))
    return A, m


def stats(work):
    w = np.asarray(work, np.float64)
    return dict(imb=float(w.max() / w.mean()) if w.mean() > 0 else None, work=[float(x) for x in w])


def blk(A, per):
    nb = A.shape[0] // per
    return A[:nb * per, :nb * per].reshape(nb, per, nb, per).sum((1, 3))


def dex(W, n, A, m):
    hq, _ = heads(W)
    out = {}
    split_mask, *plans = C.dex_plan_paths(W, n)
    for ps, pf in zip(("fwd", "bwd"), plans):
        if not os.path.exists(pf):
            continue
        pl = json.load(open(pf))
        hd = pl["heads_virtual"]; hv = hq // hd[0]
        per = pl["block_tokens"] // TILE; ns = m.shape[0]; nbc = ns // per
        st = np.asarray(pl["schedule_table"]); assert st.shape[0] == hv * nbc, (n, st.shape, hv, nbc)
        Ab = blk(A, per)
        work = np.zeros(P)
        for c in range(hv):
            sub = st[c * nbc:(c + 1) * nbc, c * nbc:(c + 1) * nbc]
            for g in range(P):
                work[g] += Ab[sub == g].sum() * hd[0]
        out[ps] = stats(work)
    return out


def ring(W, A):
    hq, _ = heads(W)
    n = A.shape[0]; c = n // (2 * P); assert c * 2 * P == n
    work = np.zeros(P)
    for r in range(P):
        t = list(range(r * c, (r + 1) * c)) + list(range((2 * P - 1 - r) * c, (2 * P - r) * c))
        work[r] = A[t].sum() * hq
    s = stats(work)
    return {"fwd": s, "bwd": s}


def ua(W, n, orc, man, cache):
    hq, _ = heads(W)
    out = {}
    for ps in ("fwd", "bwd"):
        e = man.get(f"{n}|{ps}") or {}
        if "plan" not in e:
            continue
        pl = json.load(open(os.path.join(C.ua_dir(), e["plan"])))
        mp = os.path.join(C.ua_dir(), e["mask"]) if "mask" in e else C.mask_path(W, n)   # UA's tile mask of the plan's block size
        if mp not in cache:
            cache[mp] = tile_area(mp, orc)
        A, m = cache[mp]
        per = pl["block_tokens"] // TILE
        assert pl["n_blocks"] * per == m.shape[0], (e["plan"], m.shape)
        st = np.asarray(pl["schedule_table"]); Ab = blk(A, per)
        out[ps] = stats(np.array([Ab[st == g].sum() * hq for g in range(P)], np.float64))
    return out


def dcp(W, n):
    f = os.path.join(C.out_dir("balance", "dcp"), f"{W}_{n}.npz")
    if not os.path.exists(f):
        return {}
    z = np.load(f)
    return {ps: stats(np.bincount(z[f"{ps}_dev"], weights=z[f"{ps}_area"], minlength=P).astype(np.float64)) for ps in ("fwd", "bwd")}


def magi(W, n, orc, mh):
    hq, _ = heads(W)
    d = mh.get(f"{W}|{n}")
    if d is None:
        return {}
    s = stats(np.array([sum(orc.rows(a, b) for a, b in d["host_q"][r]) * hq for r in range(P)], np.float64))
    return {"fwd": s, "bwd": s}


def main():
    C.assert_no_cuda()
    mfile = os.path.join(C.out_dir("balance"), "magi_hosts.json")
    mh = json.load(open(mfile)) if os.path.exists(mfile) else {}
    uf = os.path.join(C.ua_dir(), "manifest.json")
    man = json.load(open(uf)) if os.path.exists(uf) else {}
    for W in sys.argv[1:] or ["W1", "W2", "W3"]:
        res = {}
        for n in C.static_names(W):
            orc = Oracle(np.load(C.ranges_path(W, n)))
            A, m = tile_area(C.mask_path(W, n), orc)
            res[n] = dict(Dex=dex(W, n, A, m), Ring=ring(W, A), UA=ua(W, n, orc, man, {}), DCP=dcp(W, n), Magi=magi(W, n, orc, mh))
            print(W, n, " ".join(f"{s}:{res[n][s]['fwd']['imb']:.3f}" for s in res[n] if "fwd" in res[n][s]), flush=True)
        json.dump(res, open(os.path.join(C.out_dir("balance"), f"{W}.json"), "w"))


if __name__ == "__main__":
    main()
