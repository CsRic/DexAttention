// Assignment -> messages -> send order -> compute order -> per-rank tables.
//
// 1. Union each reader's required Q/KV blocks. Remote data enters a node once
//    through a reader acting as gateway, then forwards to the other readers.
// 2. Attribute each task's work equally to its final input-delivery messages.
//    This gain divided by transfer cost is profit. Sort each channel by profit
//    within each head; estimate arrivals, including waits for incoming data.
// 3. Group rows with the same input dependencies up to cap_group work. Within
//    each head, prefer outputs for remote nodes, then remote GPUs, then local;
//    break ties by earlier group readiness and larger group work.
//
// Input block pairs carry assigned GPU, work in seconds and live row/column
// tile masks. Messages are contiguous runs; table building cuts transfer pieces.
// fsb_fetch exports the schedule. Runtime checks actual arrival signals.
#include <algorithm>
#include <array>
#include <tuple>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <queue>
#include <vector>
#include <unordered_map>
#ifdef FSB_TIMING
#include <chrono>
#include <cstdio>
#define TP(name) do { auto _n = std::chrono::steady_clock::now(); fprintf(stderr, "  %-12s %7.3f ms\n", name, std::chrono::duration<double, std::milli>(_n - _t0).count()); _t0 = _n; } while (0)
#else
#define TP(name) do {} while (0)
#endif

namespace {
struct Out {
    std::vector<int64_t> ms, md, mk, mb0, mcnt, mtype, mpar, pos;
    std::vector<double> land;
    std::vector<int64_t> row_ptr, row_blk, row_group;     // per GPU (CSR): block rows in priority order, their group
    int64_t n_groups = 0;
    std::vector<double> u, start, mt;                     // static order: compute gain, expected send start, transfer cost
} g_out;
// Per-rank (local) planning support: externally given start / landing of inter-node (NIC) messages, keyed
// by (kind, sender, receiver, first tile) -- the sender orders its NIC messages from ALL nodes (phase B); a node's run
// on its own pairs then uses these instead of ordering the NIC link from the node's messages only. Static order only.
std::unordered_map<uint64_t, std::pair<double, double>> g_ext_land;
// symmetric buffer sizes (NQ, NKV, NPART) for fsb_tables: -1 = compute from every PE's lists (global planning); per-rank
// planning sets the all-reduced maxima. g_sym_me = this rank's own sizes (the all-reduce input); g_sym_only: stop there
int64_t g_sym[3] = {-1, -1, -1}, g_sym_me[3] = {0, 0, 0}, g_sym_used[3] = {0, 0, 0};
int g_sym_only = 0;
inline uint64_t ext_key(int64_t k, int64_t s, int64_t d, int64_t b0) {
    return ((uint64_t)k << 63) | ((uint64_t)s << 44) | ((uint64_t)d << 24) | (uint64_t)b0;
}

inline uint64_t mix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
// Weighted bytes per tile and other reader: input forwarding, plus partial
// gradient aggregation in backward. This converts relay time to the receive
// load's units; it is communication load, not additional attention work.
double g_relay_cost[2] = {0.0, 0.0};
// Gateway policies (fsb_set_gateway_mode):
// 0: receive + relay load across senders; 1: count gateway roles per tile;
// 2: compute estimate + relay time; 3: receive + relay within (sender, node);
// 4: same load criterion as 0, with sender-rotated tie order;
// 5: receive within (sender, node), relay across senders (Python fwd.GW_MODE default).
// Policy 3 lets node-local planners reconstruct routes independently.
int g_gw_mode = 0;  // C++ initial policy; Python sets the policy before scheduling.
// Transfer chunks (fsb_set_chunk_tiles; 0 = off): the table stages (fsb_tables, bwd_tables) cut every run at ABSOLUTE
// tile multiples of the chunk (per kind in the caller's convention: 0 = Q, 1 = KV), each piece with its own signal. The
// schedule search then runs uncapped (whole messages); the parent and the forwards of a relay share the cut points, so
// a gateway forwards chunk by chunk.
int64_t g_chunk_tiles[2] = {0, 0};
// Inter-node transfer chunks (fsb_set_chunk_inter; 0 = as g_chunk_tiles): a separate cut for the inter-node
// messages and their relay forwards (which share the parent's cut points) -- finer pieces unlock the receivers' compute
// and the gateway's forwards earlier; intra-node direct messages keep g_chunk_tiles. Table stage only (no search cost).
int64_t g_chunk_inter[2] = {0, 0};
// Compact KV slots in the forward tables (fsb_set_kv_compact; non-rect): a PE's KV slots = its owned tiles, then the
// remote tiles it reads, ascending, so the KV region does not grow to the whole sequence's KV. The kernel then reads the mask geometry (global tile) from a
// separate list.
int g_kv_compact = 0;
// CSR of the indexes 0..n-1 by key (a stable counting sort: indexes ascending inside a key) -- counted and scattered per
// RUN of equal keys: pair lists hold long runs of one GPU / block, where per-element counter increments form a
// dependency chain.
// x / d for 0 <= x < lim by a multiply-shift (per-tile loops divide by the runtime plan-block size `per`, and a 64-bit
// idiv is slow); exact: m = ceil(2^64 / d) errs below 1 / d for x < 2^64 / d^2, and
// lim is checked against that (else plain division). Powers of two shift.
struct FDiv {
    uint64_t d = 1, m = 0; int sh = 0; bool pow2 = true, plain = false;
    FDiv() = default;
    FDiv(int64_t d_, int64_t lim) : d((uint64_t)d_) {
        pow2 = (d & (d - 1)) == 0;
        if (pow2) { sh = __builtin_ctzll(d); return; }
        m = ~0ull / d + 1;
        plain = (unsigned __int128)(uint64_t)lim * d * d >= ((unsigned __int128)1 << 64);
    }
    inline int64_t q(int64_t x) const {
        if (pow2) return (int64_t)((uint64_t)x >> sh);
        if (plain) return (int64_t)((uint64_t)x / d);
        return (int64_t)(((unsigned __int128)(uint64_t)x * m) >> 64);
    }
    inline int64_t r(int64_t x) const { return x - q(x) * (int64_t)d; }
};
// one stable counting-sort pass of v by key(x) in [0, range) (LSD radix sorts below)
template <class T, class F> void cpass(std::vector<T> &v, std::vector<T> &tmp, int64_t range, F key) {
    std::vector<int32_t> c(range + 1, 0);
    for (const T &x : v) ++c[key(x) + 1];
    for (int64_t i = 0; i < range; ++i) c[i + 1] += c[i];
    tmp.resize(v.size());
    for (const T &x : v) tmp[c[key(x)]++] = x;
    v.swap(tmp);
}
template <class I> void csr_by_key(const int64_t *key, int64_t n, int64_t nkey, std::vector<I> &ptr, std::vector<I> &ix) {
    ptr.assign(nkey + 1, 0); ix.resize(n);
    for (int64_t a = 0; a < n;) {
        int64_t e = a + 1; const int64_t k = key[a];
        while (e < n && key[e] == k) ++e;
        ptr[k + 1] += (I)(e - a); a = e;
    }
    for (int64_t x = 0; x < nkey; ++x) ptr[x + 1] += ptr[x];
    std::vector<I> f(ptr.begin(), ptr.end() - 1);
    for (int64_t a = 0; a < n;) {
        int64_t e = a + 1; const int64_t k = key[a];
        while (e < n && key[e] == k) ++e;
        I w = f[k];
        for (int64_t q = a; q < e; ++q) ix[w++] = (I)q;
        f[k] = w; a = e;
    }
}
}  // namespace

extern "C" {

void fsb_set_relay_cost(double c0, double c1) { g_relay_cost[0] = c0; g_relay_cost[1] = c1; }
void fsb_set_gateway_mode(int m) { g_gw_mode = m; }
void fsb_set_chunk_tiles(int64_t q, int64_t kv) { g_chunk_tiles[0] = q; g_chunk_tiles[1] = kv; }
void fsb_set_chunk_inter(int64_t q, int64_t kv) { g_chunk_inter[0] = q; g_chunk_inter[1] = kv; }
void fsb_set_kv_compact(int v) { g_kv_compact = v; }
void fsb_set_ext_land(int64_t n, const int64_t *key4, const double *start, const double *land) {
    g_ext_land.clear(); g_ext_land.reserve((size_t)n * 2);
    for (int64_t i = 0; i < n; ++i)
        g_ext_land[ext_key(key4[4 * i], key4[4 * i + 1], key4[4 * i + 2], key4[4 * i + 3])] = {start[i], land[i]};
}
// load a schedule computed elsewhere (per-rank planning: the messages touching one rank, its rows) for fsb_tables /
// bwd_tables; msg7 rows = (sender, receiver, kind, first tile, tiles, type, parent index or -1)
void fsb_load_out(int64_t nm, const int64_t *msg7, const int64_t *pos, const double *land, int64_t P,
                  const int64_t *row_ptr, int64_t nr, const int64_t *row_blk, const int64_t *row_group, int64_t ngroups) {
    Out &O = g_out;
    O = Out();
    for (int64_t m = 0; m < nm; ++m) {
        const int64_t *r = msg7 + 7 * m;
        O.ms.push_back(r[0]); O.md.push_back(r[1]); O.mk.push_back(r[2]); O.mb0.push_back(r[3]); O.mcnt.push_back(r[4]);
        O.mtype.push_back(r[5]); O.mpar.push_back(r[6]);
    }
    O.pos.assign(pos, pos + nm); O.land.assign(land, land + nm);
    O.row_ptr.assign(row_ptr, row_ptr + P + 1); O.row_blk.assign(row_blk, row_blk + nr); O.row_group.assign(row_group, row_group + nr);
    O.n_groups = ngroups;
}
void fsb_set_sym(int64_t nq, int64_t nkv, int64_t npart, int only) { g_sym[0] = nq; g_sym[1] = nkv; g_sym[2] = npart; g_sym_only = only; }
void fsb_get_sym_me(int64_t *out) { out[0] = g_sym_me[0]; out[1] = g_sym_me[1]; out[2] = g_sym_me[2]; }
void fsb_get_sym_used(int64_t *out) { out[0] = g_sym_used[0]; out[1] = g_sym_used[1]; out[2] = g_sym_used[2]; }
void fsb_fetch_static(double *u, double *start) {
    std::memcpy(u, g_out.u.data(), g_out.u.size() * 8); std::memcpy(start, g_out.start.data(), g_out.start.size() * 8);
}
void fsb_fetch_mtime(double *mt) { std::memcpy(mt, g_out.mt.data(), g_out.mt.size() * 8); }

// returns the number of messages. Link parameters: bw[3] = inter, intra, forward (bytes/s), lat[2] = inter, intra.
int64_t fsb_schedule(int64_t nb, int64_t per, int64_t P, int64_t gpn, int64_t npair, const int64_t *bq,
                     const int64_t *bk, const int64_t *pdev, const double *pwork, const uint64_t *prow,
                     const uint64_t *pcol, const int64_t *owner, const int64_t *bhead, double qb, double kvb,
                     double max_bytes, double max_bytes_kv, int relay, const double *bw, const double *lat,
                     double cap_group, int static_order) {
    Out &O = g_out;
    O = Out();
#ifdef FSB_TIMING
    auto _t0 = std::chrono::steady_clock::now();
#endif
    const int64_t nv = nb * per;
    // ---- 0. pairs in (block row, GPU, column) order, renumbered: every later pass is sequential. Input sorted by row
    // (the usual row-major list): stable sort by GPU inside each row (usually already sorted); else a counting sort.
    std::vector<int64_t> po(npair);
    bool by_row = true;
    for (int64_t p = 1; p < npair && by_row; ++p) by_row = bq[p] >= bq[p - 1];
    if (by_row) {
        for (int64_t p = 0; p < npair; ++p) po[p] = p;
        for (int64_t a = 0; a < npair;) {
            int64_t e = a + 1; bool same = true;
            while (e < npair && bq[e] == bq[a]) { same &= pdev[e] == pdev[a]; ++e; }
            if (!same) std::stable_sort(po.begin() + a, po.begin() + e, [&](int64_t x, int64_t y) { return pdev[x] < pdev[y]; });
            a = e;
        }
    } else {
        std::vector<int64_t> bc(nb + 1, 0);                             // stable by (row block, GPU) without an nb x P
        for (int64_t p = 0; p < npair; ++p) ++bc[bq[p] + 1];             // array: counting sort by block, then the
        for (int64_t b = 0; b < nb; ++b) bc[b + 1] += bc[b];             // (few) pairs of each block by GPU
        { std::vector<int64_t> f(bc.begin(), bc.end() - 1); for (int64_t p = 0; p < npair; ++p) po[f[bq[p]]++] = p; }
        for (int64_t b = 0; b < nb; ++b)
            if (bc[b + 1] - bc[b] > 1)
                std::stable_sort(po.begin() + bc[b], po.begin() + bc[b + 1], [&](int64_t x, int64_t y) { return pdev[x] < pdev[y]; });
    }
    TP("n:po");
    std::vector<int64_t> sd(npair), sq(npair), sk(npair);
    std::vector<double> sw(npair);
    for (int64_t x = 0; x < npair; ++x) {
        const int64_t p = po[x];
        sd[x] = pdev[p]; sq[x] = bq[p]; sk[x] = bk[p]; sw[x] = pwork[p];
    }
    // ---- 1. needs: per (kind, receiver, plan block) the tile blocks it needs (row masks OR-ed per row first) -- as
    // entries (sender = the block's owner, receiver, block, mask), not a kind x P x blocks array (quadratic in the world
    // size under weak scaling); sorted by (sender, receiver, block) for the direct messages and by (sender, receiver
    // node, block, receiver) for the relayed ones -- the same visiting order as the dense scan (a block nobody needs
    // yields no segment there)
    struct NE { int32_t s, d, blk; uint64_t m; };
    std::vector<NE> ne[2];
    // small P x blocks: a dense OR array, read back in (sender, receiver, block) order (no sort); large: sorted entries
    const bool dense_need = P * nb <= (int64_t)(1 << 21) && P * nb <= 16 * npair;   // (a small run: sparse)
    std::vector<std::vector<int64_t>> owned(P);                     // plan blocks per owner, ascending
    for (int64_t b = 0; b < nb; ++b) owned[owner[b]].push_back(b);
    if (dense_need) {
        std::vector<uint64_t> need(2 * P * nb, 0);                     // [kind][d][blk]
        for (int64_t a = 0; a < npair;) {
            int64_t e = a; uint64_t rm = 0;
            const int64_t d = sd[a];
            for (; e < npair && sd[e] == d && sq[e] == sq[a]; ++e) {
                rm |= prow[po[e]];
                if (owner[sk[e]] != d) need[(1 * P + d) * nb + sk[e]] |= pcol[po[e]];
            }
            if (owner[sq[a]] != d) need[(0 * P + d) * nb + sq[a]] |= rm;
            a = e;
        }
        for (int k = 0; k < 2; ++k)
            for (int64_t s_ = 0; s_ < P; ++s_)
                for (int64_t d = 0; d < P; ++d)
                    for (int64_t b : owned[s_]) {
                        const uint64_t m = need[(k * P + d) * nb + b];
                        if (m) ne[k].push_back({(int32_t)s_, (int32_t)d, (int32_t)b, m});
                    }
    } else for (int64_t a = 0; a < npair;) {
        int64_t e = a; uint64_t rm = 0;
        const int64_t d = sd[a];
        for (; e < npair && sd[e] == d && sq[e] == sq[a]; ++e) {
            rm |= prow[po[e]];
            if (owner[sk[e]] != d) ne[1].push_back({(int32_t)owner[sk[e]], (int32_t)d, (int32_t)sk[e], pcol[po[e]]});
        }
        if (owner[sq[a]] != d) ne[0].push_back({(int32_t)owner[sq[a]], (int32_t)d, (int32_t)sq[a], rm});
        a = e;
    }
    TP("n:entries");
    for (int k = 0; k < 2 && !dense_need; ++k) {                       // sort, merge equal (receiver, block)
        auto &v = ne[k];
        if ((int64_t)(nb + 2 * P) < 8 * (int64_t)v.size()) {           // LSD counting sorts: block, receiver,
            std::vector<NE> tmp;                                        // sender -- stable; equal keys are OR-merged below,
            cpass(v, tmp, nb, [](const NE &x) { return x.blk; });       // so the result equals the comparison sort's
            cpass(v, tmp, P, [](const NE &x) { return x.d; });
            cpass(v, tmp, P, [](const NE &x) { return x.s; });
        } else {                                                        // (sort on a packed key: sender, receiver, block)
            std::vector<std::pair<uint64_t, uint64_t>> kv(v.size());
            for (size_t i = 0; i < v.size(); ++i)
                kv[i] = {((uint64_t)v[i].s << 44) | ((uint64_t)v[i].d << 24) | (uint64_t)v[i].blk, v[i].m};
            std::sort(kv.begin(), kv.end(), [](auto &x, auto &y) { return x.first < y.first; });
            for (size_t i = 0; i < v.size(); ++i)
                v[i] = {(int32_t)(kv[i].first >> 44), (int32_t)((kv[i].first >> 24) & 0xfffff), (int32_t)(kv[i].first & 0xffffff), kv[i].second};
        }
        size_t w = 0;
        for (size_t i = 0; i < v.size(); ++i) {
            if (w && v[w - 1].s == v[i].s && v[w - 1].d == v[i].d && v[w - 1].blk == v[i].blk) v[w - 1].m |= v[i].m;
            else v[w++] = v[i];
        }
        v.resize(w);
    }
    const int64_t NN = (P + gpn - 1) / gpn;
    TP("needs");
    // ---- 2. messages ----------------------------------------------------------------------------------------- //
    std::vector<double> inb(P, 0.0), rload(P, 0.0);                  // inbound bytes; relay work (relay cost on)
    std::vector<int64_t> roles(P, 0);                                   // gateway roles (gateway mode 1)
    std::vector<double> gpu_work(P, 0.0);                               // compute per GPU (gateway mode 2)
    for (int64_t p = 0; p < npair; ++p) gpu_work[pdev[p]] += pwork[p];
    auto add = [&](int64_t s, int64_t d, int64_t k, int64_t b0, int64_t c, int64_t t, int64_t par) {
        O.ms.push_back(s); O.md.push_back(d); O.mk.push_back(k); O.mb0.push_back(b0); O.mcnt.push_back(c);
        O.mtype.push_back(t); O.mpar.push_back(par);
    };
    for (int k = 0; k < 2; ++k) {
        const double bb = k == 0 ? qb : kvb;
        const double mb = (k == 0 || max_bytes_kv <= 0) ? max_bytes : max_bytes_kv;
        const int64_t cap = std::max<int64_t>(1, (int64_t)(mb / bb));
        const std::vector<NE> &nk = ne[k];
        // runs over the tile blocks blks (ascending) of one sender, built from SEGMENTS (bit runs with a constant key
        // inside a plan block, found with bit tricks, not bit by bit): segfn(index, blk, cb) calls cb(first bit,
        // length, key) in order; segments continue across adjacent blocks; emit(start, len, key)
        std::vector<int32_t> blks;
        auto scan = [&](auto segfn, auto emit) {
            int64_t rs = -1, rl = 0; uint64_t rkey = 0;
            for (size_t bi = 0; bi < blks.size(); ++bi)
                segfn(bi, blks[bi], [&](int64_t i, int64_t len, uint64_t key) {
                    const int64_t blk = blks[bi];
                    const int64_t b = blk * per + i;
                    if (rl && b == rs + rl && key == rkey) { rl += len; return; }
                    if (rl) emit(rs, rl, rkey);
                    rs = b; rl = len; rkey = key;
                });
            if (rl) emit(rs, rl, rkey);
        };
        // segments of one mask: runs of set bits (key 1)
        auto seg1 = [](uint64_t m, auto cb) {
            while (m) {
                // ~(m >> i) == 0 only for a full 64-bit word at i == 0, where ctzll(0) is undefined
                const int64_t i = __builtin_ctzll(m); const uint64_t r_ = ~(m >> i);
                const int64_t len = r_ ? __builtin_ctzll(r_) : 64 - i;
                cb(i, len, 1ull);
                m &= ~(((len == 64 ? 0ull : (1ull << len)) - 1) << i);
            }
        };
        size_t dlo = 0;                                                 // the current (s, d) entry range [dlo, dhi)
        auto direct = [&](int64_t s, int64_t d, int64_t typ) {
            scan([&](size_t bi, int64_t, auto cb) { seg1(nk[dlo + bi].m, cb); },
                 [&](int64_t a, int64_t l, uint64_t) {
                     for (int64_t c = 0; c < l; c += cap) {
                         const int64_t cnt = std::min(cap, l - c);
                         add(s, d, k, a + c, cnt, typ, -1);
                         if (typ == 0) inb[d] += cnt * bb;
                     }
                 });
        };
        auto pass_direct = [&](bool intra, int64_t typ) {              // every (s, d) with entries, (s, d, b) order
            for (size_t i = 0; i < nk.size();) {
                size_t j = i;
                while (j < nk.size() && nk[j].s == nk[i].s && nk[j].d == nk[i].d) ++j;
                const int64_t s_ = nk[i].s, d_ = nk[i].d;
                if ((d_ / gpn == s_ / gpn) == intra && d_ != s_) {
                    blks.clear(); for (size_t x = i; x < j; ++x) blks.push_back(nk[x].blk);
                    dlo = i; direct(s_, d_, typ);
                }
                i = j;
            }
        };
        pass_direct(true, 1);                                           // intra-node direct
        if (!relay) { pass_direct(false, 0); continue; }
        // relay: (sender, node, b) order; per (sender, node): its needed blocks and, per block, each GPU's mask
        std::vector<NE> nn_(nk.begin(), nk.end());
        if ((int64_t)(nb + 3 * P) < 8 * (int64_t)nn_.size()) {          // LSD: receiver, block, node, sender
            std::vector<NE> tmp;                                        // (entries are unique per (s, d, block): the same
            cpass(nn_, tmp, P, [](const NE &x) { return x.d; });        //  total order as the comparison sort)
            cpass(nn_, tmp, nb, [](const NE &x) { return x.blk; });
            cpass(nn_, tmp, NN, [&](const NE &x) { return x.d / gpn; });
            cpass(nn_, tmp, P, [](const NE &x) { return x.s; });
        } else
        std::sort(nn_.begin(), nn_.end(), [&](const NE &x, const NE &y) {
            const int64_t xn = x.d / gpn, yn = y.d / gpn;
            return x.s != y.s ? x.s < y.s : xn != yn ? xn < yn : x.blk != y.blk ? x.blk < y.blk : x.d < y.d; });
        std::vector<int32_t> bptr_n;                                    // per block of the range: its entries
        for (size_t i0 = 0; i0 < nn_.size();) {
            size_t j0 = i0;
            const int64_t s = nn_[i0].s, n = nn_[i0].d / gpn;
            while (j0 < nn_.size() && nn_[j0].s == s && nn_[j0].d / gpn == n) ++j0;
            if (n == s / gpn) { i0 = j0; continue; }
            blks.clear(); bptr_n.clear();
            double lin[64] = {0}, lrl[64] = {0};                        // gateway mode 3: loads inside (s, n)
            for (size_t x = i0; x < j0; ++x)
                if (blks.empty() || blks.back() != nn_[x].blk) { blks.push_back(nn_[x].blk); bptr_n.push_back((int32_t)x); }
            bptr_n.push_back((int32_t)j0);
            {
                scan([&](size_t bi, int64_t, auto cb) {                 // key = which GPUs of node n read the bit;
                         uint64_t any = 0, brk = 0, mj[64];              // a segment ends where any GPU's bit changes
                         const int64_t ng_ = std::min(gpn, P - n * gpn);
                         for (int64_t j = 0; j < ng_; ++j) mj[j] = 0;
                         for (int32_t x = bptr_n[bi]; x < bptr_n[bi + 1]; ++x) mj[nn_[x].d - n * gpn] |= nn_[x].m;
                         for (int64_t j = 0; j < ng_; ++j) { any |= mj[j]; brk |= mj[j] ^ (mj[j] << 1); }
                         while (any) {
                             const int64_t i = __builtin_ctzll(any);
                             const uint64_t above = i + 1 < 64 ? brk & ~((2ull << i) - 1) : 0;
                             const int64_t e = above ? __builtin_ctzll(above) : per;
                             uint64_t key = 0;
                             for (int64_t j = 0; j < ng_; ++j) key |= ((mj[j] >> i) & 1ull) << j;
                             cb(i, e - i, key);
                             any &= e >= 64 ? 0ull : ~((1ull << e) - 1);
                         }
                     },
                     [&](int64_t a, int64_t l, uint64_t m) {
                         if (g_gw_mode == 1 && __builtin_popcountll(m) >= 2) {   // per tile block: fewest roles
                             int64_t b = a;
                             while (b < a + l) {
                                 int64_t g = -1;
                                 for (int64_t j = 0; j < gpn; ++j)
                                     if (m >> j & 1) { const int64_t x = n * gpn + j; if (g < 0 || roles[x] < roles[g]) g = x; }
                                 ++roles[g];
                                 const int64_t par = (int64_t)O.ms.size();
                                 add(s, g, k, b, 1, 0, -1);
                                 inb[g] += bb;
                                 for (int64_t j = 0; j < gpn; ++j)
                                     if ((m >> j & 1) && n * gpn + j != g) add(g, n * gpn + j, k, b, 1, 2, par);
                                 ++b;
                             }
                             return;
                         }
                         int64_t g = -1;
                         if (g_gw_mode == 4 || g_gw_mode == 5) {
                             double bk_ = 0.0;
                             for (int64_t jj = 0; jj < gpn; ++jj) {
                                 const int64_t j = (s + jj) % gpn;
                                 if (!(m >> j & 1)) continue;
                                 const int64_t x = n * gpn + j;
                                 const double key = (g_gw_mode == 5 ? lin[j] : inb[x]) + rload[x];
                                 if (g < 0 || key < bk_) { g = x; bk_ = key; }
                             }
                             for (int64_t c = 0; c < l; c += cap) {
                                 const int64_t cnt = std::min(cap, l - c);
                                 const int64_t par = (int64_t)O.ms.size();
                                 add(s, g, k, a + c, cnt, 0, -1);
                                 inb[g] += cnt * bb; lin[g - n * gpn] += cnt * bb;
                                 if (g_relay_cost[k] > 0) rload[g] += cnt * g_relay_cost[k] * (__builtin_popcountll(m) - 1);
                                 for (int64_t j = 0; j < gpn; ++j)
                                     if ((m >> j & 1) && n * gpn + j != g) add(g, n * gpn + j, k, a + c, cnt, 2, par);
                             }
                             return;
                         }
                         if (g_gw_mode == 3) {
                             double bk_ = 0.0;
                             for (int64_t jj = 0; jj < gpn; ++jj) {
                                 const int64_t j = (s + jj) % gpn;
                                 if (!(m >> j & 1)) continue;
                                 const double key = lin[j] + lrl[j];
                                 if (g < 0 || key < bk_) { g = n * gpn + j; bk_ = key; }
                             }
                             for (int64_t c = 0; c < l; c += cap) {
                                 const int64_t cnt = std::min(cap, l - c);
                                 const int64_t par = (int64_t)O.ms.size();
                                 add(s, g, k, a + c, cnt, 0, -1);
                                 lin[g - n * gpn] += cnt * bb;
                                 if (g_relay_cost[k] > 0) lrl[g - n * gpn] += cnt * g_relay_cost[k] * (__builtin_popcountll(m) - 1);
                                 for (int64_t j = 0; j < gpn; ++j)
                                     if ((m >> j & 1) && n * gpn + j != g) add(g, n * gpn + j, k, a + c, cnt, 2, par);
                             }
                             return;
                         }
                         for (int64_t j = 0; j < gpn; ++j)
                             if (m >> j & 1) {
                                 const int64_t x = n * gpn + j;
                                 if (g_gw_mode == 2) {
                                     if (g < 0 || gpu_work[x] + rload[x] / bw[1] < gpu_work[g] + rload[g] / bw[1]) g = x;
                                 } else if (g < 0 || inb[x] + rload[x] < inb[g] + rload[g]) g = x;
                             }
                         for (int64_t c = 0; c < l; c += cap) {
                             const int64_t cnt = std::min(cap, l - c);
                             const int64_t par = (int64_t)O.ms.size();
                             add(s, g, k, a + c, cnt, 0, -1);
                             inb[g] += cnt * bb;
                             if (g_relay_cost[k] > 0) rload[g] += cnt * g_relay_cost[k] * (__builtin_popcountll(m) - 1);
                             for (int64_t j = 0; j < gpn; ++j)
                                 if ((m >> j & 1) && n * gpn + j != g) add(g, n * gpn + j, k, a + c, cnt, 2, par);
                         }
                     });
            }
            i0 = j0;
        }
    }
    TP("messages");
    const int64_t nm = (int64_t)O.ms.size();
    std::vector<double> mtime(nm);
    std::vector<int64_t> mlink(nm), mhead(bhead ? nm : 0);
    for (int64_t m = 0; m < nm; ++m) {
        const double bytes = O.mcnt[m] * (O.mk[m] == 0 ? qb : kvb);
        const int64_t t = O.mtype[m];
        mtime[m] = t == 0 ? bytes / bw[0] + lat[0] : t == 1 ? bytes / bw[1] + lat[1] : bytes / bw[2] + lat[1];
        // links: NIC of the sender, (sender, receiver) inside a node, forward link of the gateway -- intra links
        // indexed inside the node (P x gpn, not P x P: a node's run at 1024 GPUs scanned 1M empty links)
        mlink[m] = t == 0 ? O.ms[m] : t == 1 ? P + O.ms[m] * gpn + O.md[m] % gpn : P + P * gpn + O.ms[m];
        if (bhead) mhead[m] = bhead[O.mb0[m] / per];
    }
    TP("mtime/link");
    // ---- 3. dependencies, row groups, tasks ------------------------------------------------------------------ //
    // per (kind, receiver, plan block): the messages delivering its needed tile blocks (CSR) and a hash of that list
    // keys (kind, receiver, block) = (kind * P + receiver) * nb + block, COMPACTED to the ones some message delivers
    // to (kid: sorted list + binary search; -1 = none) -- no kind x P x blocks arrays
    auto blocks_of = [&](int64_t m, auto f) {
        const int64_t b0 = O.mb0[m] / per, b1 = (O.mb0[m] + O.mcnt[m] - 1) / per;
        for (int64_t b = b0; b <= b1; ++b) f((O.mk[m] * P + O.md[m]) * nb + b);
    };
    // (message, key) entries sorted by key with messages ascending inside a key: an LSD radix sort (block, then kind and
    // receiver) -- linear; it gives the keys, the CSR and its contents in one pass
    std::vector<int64_t> keys, kmsg, kptr(1, 0);
    const bool kdense_ok = 2 * P * nb <= (int64_t)(1 << 22) && 2 * P * nb <= 16 * nm;
    if (kdense_ok) {                                                    // small key space: presence array + counting
        std::vector<int32_t> cnt(2 * P * nb + 1, 0);
        for (int64_t m = 0; m < nm; ++m) blocks_of(m, [&](int64_t key) { ++cnt[key + 1]; });
        for (int64_t key = 0; key < 2 * P * nb; ++key) if (cnt[key + 1]) { keys.push_back(key); kptr.push_back(kptr.back() + cnt[key + 1]); }
    }
    if (!kdense_ok) {
        std::vector<std::pair<int64_t, int64_t>> e0, e1;               // (key, m)
        for (int64_t m = 0; m < nm; ++m) blocks_of(m, [&](int64_t key) { e0.push_back({key, m}); });
        e1.resize(e0.size());
        auto pass = [&](std::vector<std::pair<int64_t, int64_t>> &src, std::vector<std::pair<int64_t, int64_t>> &dst,
                        int64_t nbuck, auto digit) {
            std::vector<int64_t> c(nbuck + 1, 0);
            for (auto &x : src) ++c[digit(x.first) + 1];
            for (int64_t i = 0; i < nbuck; ++i) c[i + 1] += c[i];
            for (auto &x : src) dst[c[digit(x.first)]++] = x;
        };
        pass(e0, e1, nb, [&](int64_t k) { return k % nb; });
        pass(e1, e0, 2 * P, [&](int64_t k) { return k / nb; });
        kmsg.reserve(e0.size());
        for (auto &x : e0) {
            if (keys.empty() || keys.back() != x.first) { keys.push_back(x.first); kptr.push_back(kptr.back()); }
            kmsg.push_back(x.second); ++kptr.back();
        }
    }
    std::vector<int32_t> kdense;                                        // small key space: a dense index
    if (kdense_ok) {
        kdense.assign(2 * P * nb, -1);
        for (size_t i = 0; i < keys.size(); ++i) kdense[keys[i]] = (int32_t)i;
        kmsg.resize(kptr.back());                                       // fill in message order (ascending per key)
        std::vector<int64_t> f(kptr.begin(), kptr.end() - 1);
        for (int64_t m = 0; m < nm; ++m) blocks_of(m, [&](int64_t key) { kmsg[f[kdense[key]]++] = m; });
    }
    auto kid = [&](int64_t key) -> int64_t {
        if (key < 0) return -1;
        if (!kdense.empty()) return kdense[key];
        auto it = std::lower_bound(keys.begin(), keys.end(), key);
        return (it != keys.end() && *it == key) ? it - keys.begin() : -1;
    };
    const int64_t nkey = (int64_t)keys.size();                         // (kptr / kmsg: the CSR built above)
    TP("3:kcsr");
    std::vector<uint64_t> lh(nkey, 0);                                 // hash of every key's message list, one
    for (int64_t key = 0; key < nkey; ++key)                            // sequential pass (0: no message)
        if (kptr[key + 1] > kptr[key]) {
            uint64_t h = 0x12345;
            for (int64_t j = kptr[key]; j < kptr[key + 1]; ++j) h = mix64(h ^ (uint64_t)kmsg[j]);
            lh[key] = h ? h : 1;
        }
    auto lhash_c = [&](int64_t c) -> uint64_t { return c < 0 ? 0 : lh[c]; };   // (compact id)
    TP("3:lhash-def");
    std::vector<int64_t> pkv(npair), pkc(npair);                      // KV dependency key of every task; compact id
    {                                                                   // a pair's key lookup continues from the
        int64_t last = -1, lastc = 0;                                   // previous pair's while keys ascend (pairs of a row
        for (int64_t x = 0; x < npair; ++x) {                           // on one GPU: columns ascending) -- galloping, the
            const int64_t key = owner[sk[x]] != sd[x] ? (1 * P + sd[x]) * nb + sk[x] : -1;   // same index as kid()
            pkv[x] = key;
            if (key < 0 || !kdense.empty()) { pkc[x] = kid(key); continue; }
            int64_t i;
            if (last >= 0 && key > last) {                              // (keys before lastc are < last < key)
                int64_t lo = lastc, hi = lastc, step = 1;
                while (hi < nkey && keys[hi] < key) { lo = hi + 1; hi = lo + step; step <<= 1; }
                if (hi > nkey) hi = nkey;
                i = std::lower_bound(keys.begin() + lo, keys.begin() + hi, key) - keys.begin();
            } else i = std::lower_bound(keys.begin(), keys.end(), key) - keys.begin();
            pkc[x] = (i < nkey && keys[i] == key) ? i : -1;
            last = key; lastc = i;
        }
    }
    TP("3:psort");
    // rows = (GPU, block row): Q list, set of KV lists, work, head, tier
    struct Row { int64_t pe, blk, head, tier, qkey, a, e; uint64_t qs, ks; double work; int64_t group; int64_t qc = -1; };
    std::vector<Row> rows;
    std::vector<double> u_static(static_order ? nm : 0, 0.0);      // static order: unlocked work per message
    auto klen_ = [&](int64_t c) { return c >= 0 ? kptr[c + 1] - kptr[c] : 0; };   // (compact id)
    for (int64_t a = 0; a < npair;) {
        int64_t e = a;
        while (e < npair && sd[e] == sd[a] && sq[e] == sq[a]) ++e;
        Row r{sd[a], sq[a], bhead ? bhead[sq[a]] : 0, 0, -1, a, e, 0, 0, 0.0, -1};
        const int64_t own = owner[r.blk];
        r.tier = own == r.pe ? 2 : (own / gpn != r.pe / gpn ? 0 : 1);
        if (own != r.pe) r.qkey = (0 * P + r.pe) * nb + r.blk;
        r.qc = kid(r.qkey);
        r.qs = lhash_c(r.qc);
        int64_t lastkey = -2;                                           // KV lists in column order (a sequence
        const int64_t nq_ = klen_(r.qc);
        for (int64_t x = a; x < e; ++x) {                               // hash: rows with the same lists in the
            r.work += sw[x];                                            // same order match)
            if (pkv[x] != lastkey) { lastkey = pkv[x]; r.ks = mix64(r.ks ^ lhash_c(pkc[x])); }
            if (static_order) {                                         // a pair's work split over its inputs
                const int64_t nk_ = klen_(pkc[x]);
                if (nq_ + nk_ == 0) continue;
                const double w = sw[x] / (double)(nq_ + nk_);
                if (nq_) for (int64_t j = kptr[r.qc]; j < kptr[r.qc + 1]; ++j) u_static[kmsg[j]] += w;
                if (nk_) for (int64_t j = kptr[pkc[x]]; j < kptr[pkc[x] + 1]; ++j) u_static[kmsg[j]] += w;
            }
        }
        rows.push_back(r);
        a = e;
    }
    TP("3:rows");
    const int64_t nr = (int64_t)rows.size();
    // row groups: same (GPU, head, Q list, KV set) in row order, cut every cap_group of work
    std::vector<int64_t> ro(nr);
    for (int64_t i = 0; i < nr; ++i) ro[i] = i;
    std::sort(ro.begin(), ro.end(), [&](int64_t a, int64_t b) {
        const Row &x = rows[a], &y = rows[b];
        if (x.pe != y.pe) return x.pe < y.pe;
        if (x.head != y.head) return x.head < y.head;
        if (x.qs != y.qs) return x.qs < y.qs;
        if (x.ks != y.ks) return x.ks < y.ks;
        return x.blk < y.blk;
    });
    std::vector<double> gwork;
    std::vector<int64_t> gfirst, gqkey;
    {
        double cum = 0.0; int64_t gbin = -1;
        for (int64_t i = 0; i < nr; ++i) {
            Row &r = rows[ro[i]];
            const bool cls = i == 0 || [&] {
                const Row &q = rows[ro[i - 1]];
                return q.pe != r.pe || q.head != r.head || q.qs != r.qs || q.ks != r.ks;
            }();
            if (cls) cum = 0.0;
            const int64_t bin = (int64_t)(cum / cap_group);
            if (cls || bin != gbin) { gwork.push_back(0.0); gfirst.push_back(r.blk); gqkey.push_back(r.qc); }
            gbin = bin; cum += r.work;
            r.group = (int64_t)gwork.size() - 1;
            gwork.back() += r.work;
        }
    }
    TP("3:groups");
    const int64_t ng = (int64_t)gwork.size();
    O.n_groups = ng;
    // tasks = block pairs (with per-head masks every pair's KV comes from its own message: merging equal dependencies does not
    // shrink the problem). Dependencies: the group's Q list + the pair's KV list.
    const int64_t ntask = npair;
    std::vector<int64_t> tgroup(ntask);
    for (const Row &r : rows) for (int64_t t = r.a; t < r.e; ++t) tgroup[t] = r.group;
    // a task's inputs = its group's Q list (rows of a group have equal Q lists) + its KV list, read in place
    auto for_deps = [&](int64_t t, auto f) {
        const int64_t qk = gqkey[tgroup[t]], kk = pkc[t];               // (compact ids)
        if (qk >= 0) for (int64_t j = kptr[qk]; j < kptr[qk + 1]; ++j) f(kmsg[j]);
        if (kk >= 0) for (int64_t j = kptr[kk]; j < kptr[kk + 1]; ++j) f(kmsg[j]);
    };
    auto ndeps = [&](int64_t t) {
        const int64_t qk = gqkey[tgroup[t]], kk = pkc[t];
        return (qk >= 0 ? kptr[qk + 1] - kptr[qk] : 0) + (kk >= 0 ? kptr[kk + 1] - kptr[kk] : 0);
    };
    TP("3:tdep");
    // message -> dependent tasks, only tasks with >= 2 inputs: a task waiting for one message adds its work to that
    // message's profit once (below) and a landing changes nothing for it
    // (static order: none of this -- the priorities need one pass over the tasks, below)
    std::vector<int64_t> dptr(nm + 1, 0), dpair;
    std::vector<int32_t> nd(ntask);
    for (int64_t t = 0; t < ntask; ++t) {
        nd[t] = (int32_t)ndeps(t);
        if (!static_order && nd[t] >= 2) for_deps(t, [&](int64_t m) { ++dptr[m + 1]; });
    }
    if (!static_order) {
        for (int64_t m = 0; m < nm; ++m) dptr[m + 1] += dptr[m];
        dpair.resize(dptr[nm]);
        std::vector<int64_t> dfill(dptr.begin(), dptr.end() - 1);
        for (int64_t t = 0; t < ntask; ++t)
            if (nd[t] >= 2) for_deps(t, [&](int64_t m) { dpair[dfill[m]++] = t; });
    }
    const int64_t ntk = ntask;
    const double *pw = sw.data();
    TP("deps/groups/tasks");
    // ---- 4. send order ----------------------------------------------------------------------------------------- //
    std::vector<int32_t> missing(ntk);
    std::vector<double> profit(nm, 0.0), half(nm, 0.0);
    std::vector<char> landed(nm, 0), done(nm, 0);
    for (int64_t p = 0; p < (static_order ? 0 : ntk); ++p) {
        missing[p] = nd[p];
        if (missing[p] == 1) for_deps(p, [&](int64_t m) { profit[m] += pw[p]; });
        else if (missing[p] == 2) for_deps(p, [&](int64_t m) { half[m] += pw[p]; });
    }
    const int64_t nlink = P + P * gpn + P;
    // messages by (link, head, m) (counting sort by link, stable sort by head inside) -> buckets = (link, head)
    std::vector<int64_t> lptr(nlink + 1, 0);
    for (int64_t m = 0; m < nm; ++m) ++lptr[mlink[m] + 1];
    for (int64_t l = 0; l < nlink; ++l) lptr[l + 1] += lptr[l];
    std::vector<int64_t> bmsg(nm), lfill(lptr.begin(), lptr.end() - 1);
    for (int64_t m = 0; m < nm; ++m) bmsg[lfill[mlink[m]]++] = m;
    if (bhead)
        for (int64_t l = 0; l < nlink; ++l)
            std::stable_sort(bmsg.begin() + lptr[l], bmsg.begin() + lptr[l + 1],
                             [&](int64_t a, int64_t b) { return mhead[a] < mhead[b]; });
    std::vector<int64_t> bptr(1, 0), link_b0(nlink + 1, 0), mbucket(nm), bleft;
    for (int64_t l = 0; l < nlink; ++l) {
        for (int64_t i = lptr[l]; i < lptr[l + 1]; ++i) {
            if (i == lptr[l] || (bhead && mhead[bmsg[i]] != mhead[bmsg[i - 1]])) bptr.push_back(bptr.back());
            ++bptr.back(); mbucket[bmsg[i]] = (int64_t)bptr.size() - 2;
        }
        link_b0[l + 1] = (int64_t)bptr.size() - 1;
    }
    bleft.resize(bptr.size() - 1);
    for (size_t b = 0; b + 1 < bptr.size(); ++b) bleft[b] = bptr[b + 1] - bptr[b];
    std::vector<int64_t> lfirst(link_b0.begin(), link_b0.end() - 1), left(nlink);
    for (int64_t l = 0; l < nlink; ++l) left[l] = lptr[l + 1] - lptr[l];
    std::vector<int64_t> cptr(nm + 1, 0);                             // relay forwards waiting for a message (CSR)
    for (int64_t m = 0; m < nm; ++m) if (O.mpar[m] >= 0) ++cptr[O.mpar[m] + 1];
    for (int64_t m = 0; m < nm; ++m) cptr[m + 1] += cptr[m];
    std::vector<int64_t> cmsg(cptr[nm]), cfill(cptr.begin(), cptr.end() - 1);
    for (int64_t m = 0; m < nm; ++m) if (O.mpar[m] >= 0) cmsg[cfill[O.mpar[m]]++] = m;
    O.pos.assign(nm, -1); O.land.assign(nm, std::numeric_limits<double>::infinity());
    TP("sim:setup");
    if (static_order) {
        // Static send order: gain is task work divided equally among final remote
        // input-delivery messages; profit = gain / transfer cost. Sort within each
        // head. Gain is attributed work, not work made ready by this message alone.
        // Estimate arrivals in queue order. A forward also waits for its incoming
        // message; these estimates feed the compute ordering below, including FIFO.
        const std::vector<double> &u = u_static;
        std::vector<double> start(nm, 0.0);
        for (int pass = 0; pass < 2; ++pass)                             // links without parents first, then forwards
            for (int64_t l = 0; l < nlink; ++l) {
                const bool fw = l >= P + P * gpn;
                if (fw != (pass == 1) || lptr[l] == lptr[l + 1]) continue;
                static const bool fifo = getenv("DIST_FS_FIFO") && atoi(getenv("DIST_FS_FIFO"));   // transfer-order ablation
                std::sort(bmsg.begin() + lptr[l], bmsg.begin() + lptr[l + 1], [&](int64_t a, int64_t b) {
                    if (bhead && mhead[a] != mhead[b]) return mhead[a] < mhead[b];
                    if (fifo) return a < b;                              // generation order (sender, receiver, block)
                    const double ka = u[a] / mtime[a], kb = u[b] / mtime[b];
                    return ka != kb ? ka > kb : a < b;
                });
                double tl = 0.0;
                const bool ext = l < P && !g_ext_land.empty();         // NIC link ordered by the sender (phase B)
                for (int64_t i = lptr[l]; i < lptr[l + 1]; ++i) {
                    const int64_t m = bmsg[i];
                    if (ext) {
                        auto it = g_ext_land.find(ext_key(O.mk[m], O.ms[m], O.md[m], O.mb0[m]));
                        if (it != g_ext_land.end()) { start[m] = it->second.first; O.land[m] = it->second.second; continue; }
                    }
                    if (O.mpar[m] >= 0) tl = std::max(tl, O.land[O.mpar[m]]);
                    start[m] = tl; tl += mtime[m]; O.land[m] = tl;
                }
            }
        O.u = u; O.start = start; O.mt = mtime;
        std::vector<int64_t> o(nm);
        for (int64_t m = 0; m < nm; ++m) o[m] = m;
        std::sort(o.begin(), o.end(), [&](int64_t a, int64_t b) { return start[a] != start[b] ? start[a] < start[b] : a < b; });
        for (int64_t k = 0; k < nm; ++k) O.pos[o[k]] = k;               // (consistent with every link's order)
    } else {
    // per bucket: count of released unsent messages (the lowest non-empty bucket is the link's current head); the
    // pick scans that bucket (few messages each), first maximum in message order
    const int64_t nbk = (int64_t)bptr.size() - 1;
    std::vector<int64_t> nready(nbk, 0);
    std::vector<char> released(nm, 0);
    // per bucket two lazy max-heaps of (key, -position in the bucket order): released messages with profit > 0 (key
    // profit / transfer time) and all released ones (key half / transfer time) -- a pick pops outdated entries (key no
    // longer current, or sent), so it does not scan the bucket. The pick: the first maximum in bucket order, by profit if any message has profit > 0.
    std::vector<int64_t> mpos(nm);                                      // position of a message in bmsg
    for (int64_t j = 0; j < nm; ++j) mpos[bmsg[j]] = j;
    using HE = std::pair<double, int64_t>;                              // (key, -position)
    std::vector<std::vector<HE>> hpro(nbk), hhalf(nbk);
    auto hpush = [&](std::vector<HE> &h, double key, int64_t m) { h.push_back({key, -mpos[m]}); std::push_heap(h.begin(), h.end()); };
    std::vector<char> bheap(nbk, 0);                                    // heaps only for large buckets (small: scan)
    for (int64_t b = 0; b < nbk; ++b) bheap[b] = bptr[b + 1] - bptr[b] > 64;
    auto note_pro = [&](int64_t m) { if (bheap[mbucket[m]] && released[m] && !done[m] && profit[m] > 0) hpush(hpro[mbucket[m]], profit[m] / mtime[m], m); };
    auto note_half = [&](int64_t m) { if (bheap[mbucket[m]] && released[m] && !done[m]) hpush(hhalf[mbucket[m]], half[m] / mtime[m], m); };
    auto release = [&](int64_t m) { released[m] = 1; ++nready[mbucket[m]]; note_pro(m); note_half(m); };
    for (int64_t m = 0; m < nm; ++m) if (O.mpar[m] < 0) release(m);
    using E = std::pair<double, int64_t>;
    std::priority_queue<E, std::vector<E>, std::greater<E>> inflight, links;
    std::vector<char> parked(nlink, 0);
    for (int64_t l = 0; l < nlink; ++l) if (left[l]) links.push({0.0, l});
    int64_t npos = 0;
    auto land_one = [&](int64_t m, double t) {
        landed[m] = 1;
        for (int64_t j = dptr[m]; j < dptr[m + 1]; ++j) {
            const int64_t p = dpair[j];
            const int32_t left_ = --missing[p];
            if (left_ == 1 || left_ == 2) {
                int64_t o[2], no = 0;
                for_deps(p, [&](int64_t q) { if (no < 2 && !landed[q]) o[no++] = q; });
                if (left_ == 1) { profit[o[0]] += pw[p]; half[o[0]] -= pw[p]; note_pro(o[0]); note_half(o[0]); }
                else { half[o[0]] += pw[p]; half[o[1]] += pw[p]; note_half(o[0]); note_half(o[1]); }
            }
        }
        for (int64_t j = cptr[m]; j < cptr[m + 1]; ++j) {                // a forward is released: wake its link
            const int64_t c = cmsg[j];
            release(c);
            if (parked[mlink[c]]) { parked[mlink[c]] = 0; links.push({t, mlink[c]}); }
        }
    };
#ifdef FSB_TIMING
    { int64_t t1 = 0, t2 = 0, t3 = 0; for (int64_t t = 0; t < ntask; ++t) { int64_t k = nd[t]; (k <= 1 ? t1 : k == 2 ? t2 : t3)++; }
      fprintf(stderr, "  tasks %ld: deps<=1 %ld, 2 %ld, >=3 %ld; tracked deps %ld; msgs %ld\n", (long)ntask, (long)t1, (long)t2, (long)t3, (long)dptr[nm], (long)nm); }
#endif
    TP("sim:setup");
    // events in time order: a landing first when it is not later than the next link event (it may wake a forward)
    while (!links.empty() || !inflight.empty()) {
        if (!inflight.empty() && (links.empty() || inflight.top().first <= links.top().first)) {
            const auto [t, m] = inflight.top(); inflight.pop();
            land_one(m, t);
            continue;
        }
        const auto [t_now, l] = links.top(); links.pop();
        while (lfirst[l] < link_b0[l + 1] && bleft[lfirst[l]] == 0) ++lfirst[l];
        int64_t cb = -1;
        for (int64_t b = lfirst[l]; b < link_b0[l + 1]; ++b) if (nready[b]) { cb = b; break; }
        int64_t best = -1;
        if (cb >= 0) {
            auto top_valid = [&](std::vector<HE> &h, bool pro) -> int64_t {     // drop outdated entries, peek the best
                while (!h.empty()) {
                    const int64_t m = bmsg[-h.front().second];
                    const double cur = (pro ? profit[m] : half[m]) / mtime[m];
                    if (!done[m] && released[m] && (!pro || profit[m] > 0) && cur == h.front().first) return m;
                    std::pop_heap(h.begin(), h.end()); h.pop_back();
                }
                return -1;
            };
            if (bheap[cb]) {
                best = top_valid(hpro[cb], true);
                if (best < 0) best = top_valid(hhalf[cb], false);
            } else {                                                    // small bucket: the scan
                bool any = false;
                for (int64_t j = bptr[cb]; j < bptr[cb + 1]; ++j) {
                    const int64_t m = bmsg[j];
                    if (released[m] && !done[m] && profit[m] > 0) { any = true; break; }
                }
                double bkey = -1.0;
                for (int64_t j = bptr[cb]; j < bptr[cb + 1]; ++j) {
                    const int64_t m = bmsg[j];
                    if (!released[m] || done[m]) continue;
                    const double key = (any ? profit[m] : half[m]) / mtime[m];
                    if (key > bkey) { bkey = key; best = m; }
                }
            }
        }
        if (best < 0) { parked[l] = 1; continue; }
        done[best] = 1; O.pos[best] = npos++; --bleft[mbucket[best]]; --nready[mbucket[best]];
        O.land[best] = t_now + mtime[best];
        inflight.push({O.land[best], best});
        if (--left[l]) links.push({O.land[best], l});
    }
    }
    TP("simulate");
    // ---- 5. compute order: per GPU head-major, then row groups by tier, expected readiness (latest landing among the
    // group's inputs), more work first; rows of a group together, ascending
    std::vector<double> gready(ng, 0.0);
    auto lmax = [&](int64_t key, double v) {
        if (key >= 0) for (int64_t j = kptr[key]; j < kptr[key + 1]; ++j) v = std::max(v, O.land[kmsg[j]]);
        return v;
    };
    for (const Row &r : rows) {                                         // per row: its Q list once, KV lists as they change
        double v = lmax(r.qc, 0.0);
        int64_t last = -2;
        for (int64_t t = r.a; t < r.e; ++t) if (pkv[t] != last) { last = pkv[t]; v = lmax(pkc[t], v); }
        gready[r.group] = std::max(gready[r.group], v);
    }
    std::sort(ro.begin(), ro.end(), [&](int64_t a, int64_t b) {
        const Row &x = rows[a], &y = rows[b];
        if (x.pe != y.pe) return x.pe < y.pe;
        if (x.head != y.head) return x.head < y.head;
        if (x.tier != y.tier) return x.tier < y.tier;
        if (gready[x.group] != gready[y.group]) return gready[x.group] < gready[y.group];
        if (gwork[x.group] != gwork[y.group]) return gwork[x.group] > gwork[y.group];
        if (gfirst[x.group] != gfirst[y.group]) return gfirst[x.group] < gfirst[y.group];
        return x.blk < y.blk;
    });
    O.row_ptr.assign(P + 1, 0);
    for (int64_t i : ro) { ++O.row_ptr[rows[i].pe + 1]; O.row_blk.push_back(rows[i].blk); O.row_group.push_back(rows[i].group); }
    for (int64_t pe = 0; pe < P; ++pe) O.row_ptr[pe + 1] += O.row_ptr[pe];
    TP("rows");
    return nm;
}

int64_t fsb_nrows() { return (int64_t)g_out.row_blk.size(); }
int64_t fsb_nmsgs() { return (int64_t)g_out.ms.size(); }
int64_t fsb_ngroups() { return g_out.n_groups; }

void fsb_fetch(int64_t *msg7, int64_t *pos, double *land, int64_t *row_ptr, int64_t *row_blk, int64_t *row_group) {
    const Out &O = g_out;
    const size_t nm = O.ms.size();
    for (size_t m = 0; m < nm; ++m) {
        int64_t *r = msg7 + 7 * m;
        r[0] = O.ms[m]; r[1] = O.md[m]; r[2] = O.mk[m]; r[3] = O.mb0[m]; r[4] = O.mcnt[m]; r[5] = O.mtype[m]; r[6] = O.mpar[m];
    }
    std::memcpy(pos, O.pos.data(), nm * 8); std::memcpy(land, O.land.data(), nm * 8);
    std::memcpy(row_ptr, O.row_ptr.data(), O.row_ptr.size() * 8);
    std::memcpy(row_blk, O.row_blk.data(), O.row_blk.size() * 8);
    std::memcpy(row_group, O.row_group.data(), O.row_group.size() * 8);
}

}  // extern "C"

// ================================================================================================================ //
// Kernel tables for one PE from the block-level schedule (the sched kernel's tables, in the order of
// dex.scheduler.fwd.tables_blocks). Uses g_out (the schedule just computed for the same plan) plus the plan's block
// pairs with their live / DIAG tile bitmaps.
// Messages ARE the transfer units: every run (inter-node push, copy-engine copy, relay forward) is a message cut where
// the slots stop being consecutive at either end (and, for a forward, where its parent's runs at the gateway are cut).
namespace {
struct Tabs {
    std::vector<std::vector<int32_t>> a;                  // tables in the order of fwd.py's _TAB_NAMES
    std::vector<int64_t> scal;                            // n_q n_kv n_part n_compute n_merge n_owned NQ NKV NPART n_pool ntile
} g_tab;
}  // namespace

extern "C" {

// Block-level table mode (fsb_set_gpu_expand; disabled by default and explicitly set to 0 by fwd.tables_blocks): the tables stage
// hands over the BLOCK-LEVEL segments and leaves the per-row segment lists, the scheduler groups, kv_indptr,
// row_n_masked, row_nremote and row_pool empty, for a caller that expands them itself.
// on: 1 = when the per-row work is large (rows x pairs of their block rows >= g_gx_min: below it the GPU round trip
// costs more than the C++ loop), 2 = always. fsb_gx_used(): whether the last fsb_tables emitted block-level tables.
static int g_gpu_expand = 0, g_gx_used = 0;
static int64_t g_gx_min = 6000;
void fsb_set_gpu_expand(int on) { g_gpu_expand = on; }
void fsb_set_gx_min(int64_t n) { g_gx_min = n; }
// 128 x 64 compute lists (SpargeAttention2's blocks): per pair a per x (2 per) bitmap of live 64-key halves
// (bit a * 2 per + 2 b + hn; nw2 words per pair; bits = their union). Set before fsb_tables, null after. The per-row
// segments then carry half masks (2 per bits, base 2 * cb) and every count (kv_indptr, groups, ntile) is in halves.
// Data movement, slots and signals stay per 128-tile. Needs per <= 32 (a row's halves in one 64-bit word).
static const uint64_t *g_hbits = nullptr;
static int64_t g_nw2 = 0;
void fsb_set_half(const uint64_t *h, int64_t nw2) { g_hbits = h; g_nw2 = nw2; }
int fsb_gx_used() { return g_gx_used; }
int64_t fsb_tables(int64_t pe, int64_t nb, int64_t per, int64_t P, int64_t gpn, int64_t npair, const int64_t *bq,
                   const int64_t *bk, const int64_t *pdev, const uint64_t *prow, const uint64_t *pcol,
                   const uint64_t *bits, const uint64_t *dbits, int64_t nw, const int64_t *owner_blk,
                   int ce_partial, int rect) {
#ifdef FSB_TIMING
    auto _t0 = std::chrono::steady_clock::now();
#endif
    const Out &O = g_out;
    const int64_t nv = nb * per, nm = (int64_t)O.ms.size(), NN = (P + gpn - 1) / gpn;
    const FDiv dper(per, nv + 1);                                       // (t / per, t % per without an idiv)
    auto own = [&](int64_t b) { return owner_blk[dper.q(b)]; };
    std::vector<std::vector<int64_t>> owned(P);                       // plan blocks per owner, ascending
    std::vector<int32_t> ownpos(nb, 0);                                 // a block's position in its owner's list
    for (int64_t b = 0; b < nb; ++b) { ownpos[b] = (int32_t)owned[owner_blk[b]].size(); owned[owner_blk[b]].push_back(b); }
    TP("t:m:owned");
    // ---- who computes / reads what: per PE its (plan block, tile-row / tile-column mask) list, blocks ascending ---- //
    // Built ON DEMAND for the PEs this one exchanges with, from two pair indexes (by computing GPU, by row block;
    // counting sorts), because building every PE's lists from all pairs does not scale to large P. The
    // symmetric sizes (NQ, NKV, NPART = maxima over the PEs) come from fsb_set_sym (the all-reduce of every rank's own
    // sizes in per-rank planning) or, if not given, from every PE's lists (global planning).
    struct BM { int32_t blk; uint64_t m; int32_t pre; };              // pre = rows of the PE's earlier blocks
    // DENSE lists when P x blocks is small next to the pairs: one pass over the pairs gives every PE's
    // (block, OR of row / column masks) and which (PE, block) have a pair -- the per-PE lists (pl) and the
    // contributors of a block (po) are read from it, instead of gathering each PE's / block's pairs from an index
    // (scattered reads over all pairs when a PE exchanges with every other). The same lists either way.
    const bool dense_l = !g_sym_only && P * nb <= ((int64_t)1 << 22) && P * nb <= 16 * npair;   // (sizes-only: pe's lists)
    const bool dense_sz = g_sym[0] < 0 && P * nb <= ((int64_t)1 << 22);
    std::vector<uint64_t> DR, DC, DRp, DCp;                              // [P][nb] masks; presence bits
    if (dense_l || dense_sz) {
        DR.assign((size_t)P * nb, 0); DC.assign((size_t)P * nb, 0);
        if (dense_l) { DRp.assign(((size_t)P * nb + 63) / 64, 0); DCp.assign(((size_t)P * nb + 63) / 64, 0); }
        for (int64_t q = 0; q < npair; ++q) {
            const size_t ir = (size_t)pdev[q] * nb + bq[q], ic = (size_t)pdev[q] * nb + bk[q];
            DR[ir] |= prow[q]; DC[ic] |= pcol[q];
            if (dense_l) { DRp[ir >> 6] |= 1ull << (ir & 63); DCp[ic >> 6] |= 1ull << (ic & 63); }
        }
    }
    // sparse: pair indexes by GPU / by row block, built on first use; this PE's own pairs and the pairs of the
    // blocks it owns come from one filtered pass (all that the sizes-only pass of per-rank planning reads)
    std::vector<int32_t> dptr, dix, rbptr, rbix, pe_pr, pe_rbp, pe_rbi;
    auto pairs_of_pe = [&](int64_t q, int32_t &i0, int32_t &i1) -> const int32_t * {
        if (q == pe) {
            if (pe_rbp.empty()) {                                       // (one pass: own pairs + pairs of owned blocks)
                pe_rbp.assign(nb + 1, 0);
                std::vector<int32_t> hit;
                for (int64_t p = 0; p < npair; ++p) {
                    if (pdev[p] == pe) pe_pr.push_back((int32_t)p);
                    if (owner_blk[bq[p]] == pe) { hit.push_back((int32_t)p); ++pe_rbp[bq[p] + 1]; }
                }
                for (int64_t b = 0; b < nb; ++b) pe_rbp[b + 1] += pe_rbp[b];
                pe_rbi.resize(hit.size());
                std::vector<int32_t> f(pe_rbp.begin(), pe_rbp.end() - 1);
                for (int32_t p : hit) pe_rbi[f[bq[p]]++] = p;
            }
            i0 = 0; i1 = (int32_t)pe_pr.size(); return pe_pr.data();
        }
        if (dptr.empty()) csr_by_key(pdev, npair, P, dptr, dix);
        i0 = dptr[q]; i1 = dptr[q + 1]; return dix.data();
    };
    auto pairs_of_blk = [&](int64_t b, int32_t &i0, int32_t &i1) -> const int32_t * {
        if (owner_blk[b] == pe) { int32_t a_, e_; pairs_of_pe(pe, a_, e_); i0 = pe_rbp[b]; i1 = pe_rbp[b + 1]; return pe_rbi.data(); }
        if (rbptr.empty()) csr_by_key(bq, npair, nb, rbptr, rbix);
        i0 = rbptr[b]; i1 = rbptr[b + 1]; return rbix.data();
    };
    TP("t:m:index");
    const uint64_t full = per == 64 ? ~0ull : ((1ull << per) - 1);
    const bool kvc = g_kv_compact && !rect;
    std::vector<int64_t> nkown(P, 0);
    // owned KV slot of a tile (compact): its owner's earlier blocks x per + the tile's row in its block (a formula: the same
    // numbers as counting the owner's tiles in ascending order)
    if (kvc) for (int64_t q = 0; q < P; ++q) nkown[q] = (int64_t)owned[q].size() * per;
    auto kvown_of = [&](int64_t t) -> int64_t { const int64_t b = dper.q(t); return (int64_t)ownpos[b] * per + (t - b * per); };
    TP("t:m:kvown");
    struct PL { bool done = false; std::vector<BM> lr, lc; std::vector<int32_t> lc_rpre, ownpre; int64_t nqc = 0, nq = 0, nkv = 0; };
    std::vector<PL> pls(P);
    // a PE's (block, OR of masks) list, blocks ascending: a bitmap of the blocks seen + one accumulator per block,
    // O(pairs + block range / 64)
    std::vector<uint64_t> acc_(std::max(nb, P), 0), seen_((std::max(nb, P) + 63) / 64, 0);   // (keys: blocks or GPUs)
    auto build_list = [&](int32_t i0, int32_t i1, const int32_t *ix, const int64_t *blk_, const uint64_t *msk, std::vector<BM> &v) {
        int64_t lo = nb, hi = -1;
        for (int32_t i = i0; i < i1; ++i) {
            const int64_t p = ix[i], b = blk_[p];
            seen_[b >> 6] |= 1ull << (b & 63); acc_[b] |= msk[p];
            lo = std::min(lo, b); hi = std::max(hi, b);
        }
        for (int64_t w = hi < 0 ? 1 : lo >> 6; w <= hi >> 6; ++w) {
            for (uint64_t x = seen_[w]; x; x &= x - 1) {
                const int64_t b = w * 64 + __builtin_ctzll(x);
                v.push_back({(int32_t)b, acc_[b], 0}); acc_[b] = 0;
            }
            seen_[w] = 0;
        }
    };
    auto pl = [&](int64_t q) -> const PL & {
        PL &L_ = pls[q];
        if (L_.done) return L_;
        L_.done = true;
        for (int k = 0; k < 2; ++k) {
            auto &v = k ? L_.lc : L_.lr;
            if (dense_l) {                                              // (row q of the dense masks, blocks ascending)
                const std::vector<uint64_t> &Dm = k ? DC : DR, &Dp = k ? DCp : DRp;
                const size_t a0 = (size_t)q * nb, a1 = a0 + nb;
                for (size_t w = a0 >> 6; w <= (a1 - 1) >> 6 && a1 > a0; ++w) {
                    uint64_t x = Dp[w];
                    if (w == (a0 >> 6)) x &= ~0ull << (a0 & 63);
                    if (w == ((a1 - 1) >> 6) && (a1 & 63)) x &= ~0ull >> (64 - (a1 & 63));
                    for (; x; x &= x - 1) { const size_t i = w * 64 + __builtin_ctzll(x); v.push_back({(int32_t)(i - a0), Dm[i], 0}); }
                }
            } else { int32_t i0, i1; const int32_t *ix = pairs_of_pe(q, i0, i1); build_list(i0, i1, ix, k ? bk : bq, k ? pcol : prow, v); }
            int32_t pre = 0;
            for (auto &e : v) { e.pre = pre; pre += __builtin_popcountll(e.m); }
        }
        L_.nqc = L_.lr.empty() ? 0 : L_.lr.back().pre + __builtin_popcountll(L_.lr.back().m);
        int32_t acc = 0; size_t j = 0;
        L_.ownpre.resize(owned[q].size());
        for (size_t x = 0; x < owned[q].size(); ++x) {                  // owned-not-computed rows of earlier blocks
            const int64_t b = owned[q][x];
            while (j < L_.lr.size() && L_.lr[j].blk < b) ++j;
            const uint64_t rm = (j < L_.lr.size() && L_.lr[j].blk == b) ? L_.lr[j].m : 0ull;
            L_.ownpre[x] = acc; acc += __builtin_popcountll(~rm & full);
        }
        L_.nq = L_.nqc + acc;
        if (kvc) {
            L_.lc_rpre.assign(L_.lc.size(), 0); int32_t a2 = 0;
            for (size_t i = 0; i < L_.lc.size(); ++i) { L_.lc_rpre[i] = a2; if (owner_blk[L_.lc[i].blk] != q) a2 += __builtin_popcountll(L_.lc[i].m); }
            L_.nkv = nkown[q] + a2;
        }
        return L_;
    };
    auto find_bm = [&](const std::vector<BM> &v, int64_t blk) -> const BM * {
        auto it = std::lower_bound(v.begin(), v.end(), (int32_t)blk, [](const BM &x, int32_t y) { return x.blk < y; });
        return (it != v.end() && it->blk == blk) ? &*it : nullptr;
    };
    // the last block looked up per PE and kind: runs of a message walk consecutive tiles of few blocks
    struct Memo { int64_t blk = -1; const BM *f = nullptr; };
    std::vector<Memo> memo_r(P), memo_c(P);
    auto find_r = [&](int64_t q, int64_t blk) -> const BM * {
        Memo &M_ = memo_r[q];
        if (M_.blk != blk) { M_.blk = blk; M_.f = find_bm(pl(q).lr, blk); }
        return M_.f;
    };
    auto find_c = [&](int64_t q, int64_t blk) -> const BM * {
        Memo &M_ = memo_c[q];
        if (M_.blk != blk) { M_.blk = blk; M_.f = find_bm(pl(q).lc, blk); }
        return M_.f;
    };
    std::vector<uint64_t> rowc_me(nb, 0), colr_me(nb, 0);
    for (const BM &e : pl(pe).lr) rowc_me[e.blk] = e.m;
    TP("t:m:pl(pe)");
    for (const BM &e : pl(pe).lc) colr_me[e.blk] = e.m;
    auto colmask = [&](int64_t q, int64_t blk) -> uint64_t { if (q == pe) return colr_me[blk]; const BM *f = find_c(q, blk); return f ? f->m : 0ull; };
    TP("T:masks");
    // ---- Q slots of every PE: rows it computes (ascending), then rows it owns and does not compute ------------ //
    // (a lookup: the PE's (block, mask) list + popcounts; owned-not-computed rows by their owner's block prefix)
    auto qslot_of = [&](int64_t q, int64_t t) -> int64_t {
        const int64_t blk = dper.q(t), i = t - blk * per;
        const uint64_t lo = i ? ((1ull << i) - 1) : 0ull;
        const PL &L_ = pl(q);
        const BM *f = find_r(q, blk);
        if (f && (f->m >> i & 1)) return f->pre + __builtin_popcountll(f->m & lo);
        if (owner_blk[blk] == q) return L_.nqc + L_.ownpre[ownpos[blk]] + __builtin_popcountll(~(f ? f->m : 0ull) & lo);
        return -1;
    };
    std::vector<int32_t> qslot_me(nv, -1);                              // this PE's, dense
    for (int64_t blk = 0; blk < nb; ++blk) {                            // (per block: blocks it neither computes nor owns
        const bool ow = owner_blk[blk] == pe;                           //  have no slot)
        if (!rowc_me[blk] && !ow) continue;
        for (int64_t t = blk * per; t < (blk + 1) * per; ++t) if (rowc_me[blk] >> (t % per) & 1 || ow) qslot_me[t] = (int32_t)qslot_of(pe, t);
    }
    TP("t:q:qslot_me");
    // KV slots: global order (the tile itself) or, compact (g_kv_compact), owned tiles then remote tiles read, ascending
    std::vector<int32_t> kslot_me;
    auto kslot_of = [&](int64_t q, int64_t t) -> int64_t {
        const int64_t blk = dper.q(t), i = t - blk * per;
        if (owner_blk[blk] == q) return kvown_of(t);
        const PL &L_ = pl(q);
        const BM *f = find_c(q, blk);
        if (!f || !(f->m >> i & 1)) return -1;
        return nkown[q] + L_.lc_rpre[f - L_.lc.data()] + __builtin_popcountll(f->m & (i ? ((1ull << i) - 1) : 0ull));
    };
    if (kvc) {
        kslot_me.assign(nv, -1);
        for (int64_t blk = 0; blk < nb; ++blk) {
            const bool ow = owner_blk[blk] == pe;
            if (!colr_me[blk] && !ow) continue;
            for (int64_t t = blk * per; t < (blk + 1) * per; ++t)
                if (ow || (colr_me[blk] >> (t % per) & 1)) kslot_me[t] = (int32_t)kslot_of(pe, t);
        }
    }
    auto slot = [&](int64_t k, int64_t q, int64_t b) -> int64_t {    // KV: global order, or compact slots
        if (k == 0) return q == pe ? qslot_me[b] : qslot_of(q, b);
        if (kvc) return q == pe ? kslot_me[b] : kslot_of(q, b);
        return (own(b) == q || (colmask(q, dper.q(b)) >> dper.r(b) & 1)) ? b : -1;
    };
    TP("T:qslot");
    // ---- partial landing slots: at owner o, after its items, per owned BLOCK (ascending), per contributing PE
    // (ascending), that PE's rows of the block (ascending): slot(row, d) = items + base(block, d) + popcount of d's
    // row mask below the row. Built per (block, contributor) from the row-block index, with no per-row lists. Any bijection works: senders write
    // opush_slot, the owner reads merge_slot, both from here.
    struct CB { int32_t pe; uint64_t m; int32_t base; };              // one contributor of one owned block
    struct PO { bool done = false; std::vector<int32_t> bptr; std::vector<CB> cb; int64_t npart = 0; };
    std::vector<PO> pos_(P);
    std::vector<std::pair<int32_t, uint64_t>> gm;
    auto po = [&](int64_t o) -> const PO & {
        PO &C = pos_[o];
        if (C.done) return C;
        C.done = true;
        C.bptr.assign(owned[o].size() + 1, 0);
        int32_t k = 0;
        for (size_t x = 0; x < owned[o].size(); ++x) {
            const int64_t b = owned[o][x];
            gm.clear();                                                 // (GPU ascending, OR of its pairs' row masks)
            if (dense_l) {
                for (int64_t x2 = 0; x2 < P; ++x2) {
                    const size_t i = (size_t)x2 * nb + b;
                    if (DRp[i >> 6] >> (i & 63) & 1) gm.push_back({(int32_t)x2, DR[i]});
                }
            } else {
                std::vector<BM> tmpv;                                   // (bitmap merge by GPU, as build_list)
                int32_t i0, i1; const int32_t *ix = pairs_of_blk(b, i0, i1);
                build_list(i0, i1, ix, pdev, prow, tmpv);
                for (const BM &e : tmpv) gm.push_back({e.blk, e.m});
            }
            const size_t w = gm.size();
            for (size_t i = 0; i < w; ++i) {
                C.cb.push_back({gm[i].first, gm[i].second, k});
                if (gm[i].first != o) k += __builtin_popcountll(gm[i].second);
            }
            C.bptr[x + 1] = (int32_t)C.cb.size();
        }
        C.npart = pl(o).nqc + k;
        return C;
    };
    auto land_slot = [&](int64_t r, int64_t d) -> int32_t {          // d != owner(r), d contributes to r
        const int64_t o = own(r), x = ownpos[r / per], a = r % per;
        const PO &C = po(o);
        for (int32_t j = C.bptr[x]; j < C.bptr[x + 1]; ++j)
            if (C.cb[j].pe == d) return (int32_t)(pl(o).nqc + C.cb[j].base + __builtin_popcountll(C.cb[j].m & ((1ULL << a) - 1)));
        return -1;
    };
    int64_t NQ, NKV, NPART;
    if (g_sym[0] >= 0) { NQ = g_sym[0]; NKV = kvc ? g_sym[1] : nv; NPART = g_sym[2]; }
    else if (dense_sz) {
        // every PE's sizes in ONE pass over the pairs: per (PE, block) the OR of its row / column masks
        // -> nq = computed rows + owned-not-computed rows, nkv = owned tiles + remote tiles read, npart = computed
        // rows + rows other PEs compute of its blocks. Same numbers as the per-PE lists (pl / po) without building
        // them for all P PEs (the dense masks built above)
        const std::vector<uint64_t> &Rm = DR, &Cm = DC;
        std::vector<int64_t> nqc(P, 0), nq(P, 0), nkv(P, 0), npart(P, 0);
        for (int64_t q = 0; q < P; ++q)
            for (int64_t b = 0; b < nb; ++b) {
                const uint64_t r = Rm[(size_t)q * nb + b];
                nqc[q] += __builtin_popcountll(r);
                if (owner_blk[b] == q) nq[q] += __builtin_popcountll(~r & full);
                else {
                    npart[owner_blk[b]] += __builtin_popcountll(r);
                    if (kvc) nkv[q] += __builtin_popcountll(Cm[(size_t)q * nb + b]);
                }
            }
        NQ = 0; NKV = kvc ? 0 : nv; NPART = 0;
        for (int64_t q = 0; q < P; ++q) {
            NQ = std::max(NQ, nqc[q] + nq[q]); NPART = std::max(NPART, nqc[q] + npart[q]);
            if (kvc) NKV = std::max(NKV, nkown[q] + nkv[q]);
        }
    }
    else {
        NQ = 0; NKV = kvc ? 0 : nv; NPART = 0;
        for (int64_t q = 0; q < P; ++q) {
            NQ = std::max(NQ, pl(q).nq); NPART = std::max(NPART, po(q).npart);
            if (kvc) NKV = std::max(NKV, pl(q).nkv);
        }
    }
    g_sym_used[0] = NQ; g_sym_used[1] = NKV; g_sym_used[2] = NPART;
    g_sym_me[0] = pl(pe).nq; g_sym_me[1] = kvc ? pl(pe).nkv : nv; g_sym_me[2] = po(pe).npart;
    if (g_sym_only) return 0;                                           // per-rank planning: this rank's sizes only
    TP("T:lands");
    // ---- runs of a message: cut where the source or destination slots break (forward: also its parent's) ------- //
    FDiv dck[2][2];                                                     // transfer chunk cut b % ck (kind, inter)
    for (int k = 0; k < 2; ++k) for (int x = 0; x < 2; ++x) {
        const int64_t ck = (x && g_chunk_inter[k] > 0) ? g_chunk_inter[k] : g_chunk_tiles[k];
        if (ck > 0) dck[k][x] = FDiv(ck, nv + 1);
    }
    auto runs = [&](int64_t m, auto emit) {                          // emit(first block, count, sig at md)
        const int64_t k = O.mk[m], s = O.ms[m], d = O.md[m], t = O.mtype[m], b0 = O.mb0[m], c = O.mcnt[m];
        const int64_t ps = t == 2 ? O.ms[O.mpar[m]] : -1;
        int64_t a = b0;
        for (int64_t b = b0 + 1; b <= b0 + c; ++b) {
            bool cut = b == b0 + c;
            if (!cut) {
                cut = slot(k, d, b) != slot(k, d, b - 1) + 1;
                if (!cut && t != 1) cut = slot(k, s, b) != slot(k, s, b - 1) + 1;   // copy engine: owned order
                if (!cut && t == 2) cut = slot(k, ps, b) != slot(k, ps, b - 1) + 1;
                const int64_t ck = (t != 1 && g_chunk_inter[k] > 0) ? g_chunk_inter[k] : g_chunk_tiles[k];
                if (!cut && ck > 0) cut = dck[k][t != 1].r(b) == 0;              // transfer chunk
            }
            if (cut) {
                const int64_t last = slot(k, d, b - 1);
                emit(a, b - a, k == 0 ? last : NQ + last);
                a = b;
            }
        }
    };
    // ---- signals of the runs INTO this PE; landing time of its KV ----------------------------------------------- //
    std::vector<int64_t> q_sig(NQ), kv_sig(NKV), kv_span(NKV, 1), kvmsg(nv, -1);   // (kv_sig / kv_span: per KV slot)
    for (int64_t i = 0; i < NQ; ++i) q_sig[i] = i;
    for (int64_t i = 0; i < NKV; ++i) kv_sig[i] = NQ + i;
    auto kvs_of_tile = [&](int64_t t) -> int64_t { return kvc ? kslot_me[t] : t; };   // this PE's KV slot of a tile
    for (int64_t m = 0; m < nm; ++m) {
        if (O.md[m] != pe) continue;
        const int64_t k = O.mk[m];
        if (k == 1) for (int64_t b = O.mb0[m]; b < O.mb0[m] + O.mcnt[m]; ++b) kvmsg[b] = m;
        runs(m, [&](int64_t a, int64_t n, int64_t sig) {
            const int64_t d0 = slot(k, pe, a);
            for (int64_t j = 0; j < n; ++j) (k == 0 ? q_sig[d0 + j] : kv_sig[d0 + j]) = sig;
            if (k == 1) kv_span[sig - NQ] = n;
        });
    }
    TP("T:sigmaps");
    // ---- items (rows this PE computes, ascending) and their tiles in storage order --------------------------- //
    std::vector<int64_t> it_row, itemidx(nv, -1);
    for (int64_t blk = 0; blk < nb; ++blk)
        for (uint64_t m = rowc_me[blk]; m; m &= m - 1) {
            itemidx[blk * per + __builtin_ctzll(m)] = (int64_t)it_row.size();
            it_row.push_back(blk * per + __builtin_ctzll(m));
        }
    const int64_t nit = (int64_t)it_row.size();
    std::vector<int64_t> mp;                                            // this PE's pairs by (row block, column block)
    for (int64_t p = 0; p < npair; ++p) if (pdev[p] == pe) mp.push_back(p);
    std::stable_sort(mp.begin(), mp.end(), [&](int64_t a, int64_t b) { return bq[a] != bq[b] ? bq[a] < bq[b] : bk[a] < bk[b]; });
    std::vector<int64_t> bptr_(nb + 1, 0);
    for (int64_t p : mp) ++bptr_[bq[p] + 1];
    for (int64_t i = 0; i < nb; ++i) bptr_[i + 1] += bptr_[i];
    auto rowbits = [&](const uint64_t *bm, int64_t i) -> uint64_t {   // row i of a per x per bitmap
        const int64_t pos = i * per, w = pos >> 6, off = pos & 63;
        uint64_t v = bm[w] >> off;
        if (off + per > 64 && w + 1 < nw) v |= bm[w + 1] << (64 - off);
        return per == 64 ? v : v & ((1ull << per) - 1);
    };
    // Storage order is decided per BLOCK ROW: its pairs' columns are cut into column segments with one (signal,
    // landing time) each -- the same for every tile row of the block row -- sorted once by the key
    //   [remote first][latest landing first][higher signal first][column ascending]
    // (segments of different pairs never interleave in column, so this = sorting tiles by (key, column)). A tile
    // row then emits, per segment, its live tiles (row bits AND segment mask), masked-diagonal tiles last.
    struct Seg { int64_t loc; double arr; int64_t sig, first, cb; uint64_t mask; int64_t pair; };
    std::vector<Seg> segs;
    // the tiles themselves are expanded on the GPU from (first column, 64-bit mask, local) per emitted segment
    std::vector<int32_t> indptr(nit + 1, 0), seg_cb, seg_lo, seg_hi, seg_loc, n_masked(nit, 0), row_nremote(nit, 0);
    std::vector<int32_t> row_seg_ptr(nit + 1, 0);                       // row -> its emitted segments (device expansion)
    std::vector<int64_t> g_item, g_beg, g_end, g_sig;
    int64_t ntile = 0;
    {
        size_t est = 0;                                                 // upper bound: rows x pairs of the block row
        for (int64_t blk = 0; blk < nb; ++blk)
            est += (size_t)__builtin_popcountll(rowc_me[blk]) * (bptr_[blk + 1] - bptr_[blk]) * 2;
        for (auto *v : {&seg_cb, &seg_lo, &seg_hi, &seg_loc}) v->reserve(est);
        for (auto *v : {&g_item, &g_beg, &g_end, &g_sig}) v->reserve(est);
    }
    int64_t it = 0;
    int64_t rp_est = 0;                                                  // rows x pairs of their block rows
    for (int64_t blk = 0; blk < nb; ++blk) rp_est += (int64_t)__builtin_popcountll(rowc_me[blk]) * (bptr_[blk + 1] - bptr_[blk]);
    const bool gx = g_gpu_expand && !rect && (g_gpu_expand == 2 || rp_est >= g_gx_min);
    g_gx_used = gx;
    std::vector<int32_t> gx_bs_ptr(1, 0), gx_cb, gx_lo, gx_hi, gx_loc, gx_pair, gx_rbrow, gx_rbi;
    std::vector<int64_t> gx_sig;
    if (gx)                                                             // tiles: every live tile of this PE's pairs
        for (int64_t p = 0; p < npair; ++p) if (pdev[p] == pe) for (int64_t w = 0; w < nw; ++w) ntile += __builtin_popcountll(bits[p * nw + w]);
    // rect: per block row its segments, per row its block row / tile row / state base / non-empty segments
    std::vector<int32_t> brow_seg_ptr(1, 0), r_seg_cb, r_seg_pair, r_row_brow, r_row_bi, r_row_base, r_row_ne;
    std::vector<int64_t> r_seg_mask, r_seg_sig;
    std::vector<int64_t> pair_pos(npair, -1);
    for (size_t x = 0; x < mp.size(); ++x) pair_pos[mp[x]] = (int64_t)x;
    int64_t rowseg = 0;
    // live / masked-diagonal column bits of every (pair of this PE, tile row), extracted once
    std::vector<int64_t> prw((int64_t)mp.size() * per), pdw, prw2;
    bool anyd = false;
    const bool half = g_hbits != nullptr;
    if (half && (per > 32 || gx || rect)) { fprintf(stderr, "fsb_tables: half mode needs per <= 32, no gx / rect\n"); abort(); }
    if (half) {                                                       // a pair row's 2 per half bits
        prw2.resize(prw.size());
        for (size_t x = 0; x < mp.size(); ++x)
            for (int64_t i = 0; i < per; ++i) {
                const uint64_t *hb = g_hbits + mp[x] * g_nw2;
                const int64_t pos = i * 2 * per, w = pos >> 6, off = pos & 63, wd = 2 * per;
                uint64_t v = hb[w] >> off;
                if (off + wd > 64 && w + 1 < g_nw2) v |= hb[w + 1] << (64 - off);
                prw2[x * per + i] = (int64_t)(wd == 64 ? v : v & ((1ull << wd) - 1));
            }
    }
    auto spread2 = [](uint64_t m) {                                     // tile bit b -> half bits 2 b, 2 b + 1
        uint64_t r = 0;
        for (; m; m &= m - 1) { const int b = __builtin_ctzll(m); r |= 3ull << (2 * b); }
        return r;
    };
    for (size_t x = 0; x < mp.size(); ++x)
        for (int64_t i = 0; i < per; ++i) {
            prw[x * per + i] = (int64_t)rowbits(bits + mp[x] * nw, i);
            anyd |= rowbits(dbits + mp[x] * nw, i) != 0;
        }
    if (anyd) {
        pdw.resize(prw.size());
        for (size_t x = 0; x < mp.size(); ++x)
            for (int64_t i = 0; i < per; ++i) pdw[x * per + i] = (int64_t)rowbits(dbits + mp[x] * nw, i);
    }
    for (int64_t blk = 0; blk < nb && it < nit; ++blk) {
        if (!rowc_me[blk]) continue;
        segs.clear();
        for (int64_t x = bptr_[blk]; x < bptr_[blk + 1]; ++x) {        // column segments of the block row
            const int64_t p = mp[x], cb = bk[p] * per, loc = owner_blk[bk[p]] == pe;
            uint64_t cols = pcol[p];
            while (cols) {
                const int64_t j0 = __builtin_ctzll(cols);
                const int64_t sig = loc ? -1 : kv_sig[kvs_of_tile(cb + j0)];
                uint64_t m = 0;
                for (uint64_t c2 = cols; c2; c2 &= c2 - 1) {           // the pair's columns in this signal's run
                    const int64_t j = __builtin_ctzll(c2);
                    if ((loc ? -1 : kv_sig[kvs_of_tile(cb + j)]) != sig) break;
                    m |= 1ull << j;
                }
                const double arr = loc || kvmsg[cb + j0] < 0 ? -1.0 : O.land[kvmsg[cb + j0]];
                segs.push_back({loc, arr, sig, cb + j0, cb, m, p});
                cols &= ~m;
            }
        }
        std::sort(segs.begin(), segs.end(), [](const Seg &a, const Seg &b) {
            if (a.loc != b.loc) return a.loc < b.loc;
            if (a.arr != b.arr) return a.arr > b.arr;
            if (a.sig != b.sig) return a.sig > b.sig;
            return a.first < b.first;
        });
        if (gx) {                                                         // block level only (GPU expands rows)
            const int64_t brow = (int64_t)gx_bs_ptr.size() - 1;
            for (const Seg &q : segs) {
                gx_cb.push_back((int32_t)q.cb); gx_lo.push_back((int32_t)(q.mask & 0xffffffffull));
                gx_hi.push_back((int32_t)(q.mask >> 32)); gx_sig.push_back(q.sig); gx_loc.push_back((int32_t)q.loc);
                gx_pair.push_back((int32_t)pair_pos[q.pair]);
            }
            gx_bs_ptr.push_back((int32_t)gx_cb.size());
            for (uint64_t rm = rowc_me[blk]; rm; rm &= rm - 1, ++it) {
                gx_rbrow.push_back((int32_t)brow); gx_rbi.push_back((int32_t)__builtin_ctzll(rm));
            }
            continue;
        }
        if (rect) {                                                       // rectangle segments: no per-row lists
            const int64_t brow = (int64_t)brow_seg_ptr.size() - 1;
            for (const Seg &q : segs) {
                r_seg_cb.push_back((int32_t)q.cb); r_seg_pair.push_back((int32_t)pair_pos[q.pair]);
                r_seg_mask.push_back((int64_t)q.mask); r_seg_sig.push_back(q.sig);
            }
            brow_seg_ptr.push_back((int32_t)r_seg_cb.size());
            const int64_t ns_b = (int64_t)segs.size();
            for (uint64_t rm = rowc_me[blk]; rm; rm &= rm - 1, ++it) {
                const int64_t i = __builtin_ctzll(rm);
                int32_t nt = 0, nd = 0, ne = 0;
                for (const Seg &q : segs) {
                    const int64_t xr = pair_pos[q.pair] * per + i;
                    const uint64_t m = q.mask & (uint64_t)prw[xr];
                    nt += __builtin_popcountll(m); ne += m != 0;
                    if (anyd) nd += __builtin_popcountll(m & (uint64_t)pdw[xr]);
                }
                r_row_brow.push_back((int32_t)brow); r_row_bi.push_back((int32_t)i); r_row_base.push_back((int32_t)rowseg);
                r_row_ne.push_back(ne);
                rowseg += ns_b;
                n_masked[it] = nd;
                ntile += nt;
                indptr[it + 1] = (int32_t)ntile;
            }
            continue;
        }
        for (uint64_t rm = rowc_me[blk]; rm; rm &= rm - 1, ++it) {   // tile rows of the block row
            const int64_t i = __builtin_ctzll(rm);
            const int64_t base = ntile;
            int64_t gsig = -2;
            for (int dpass = 0; dpass < 2; ++dpass)                       // non-masked tiles, then masked diagonal
                for (const Seg &q : segs) {
                    const int64_t xr = pair_pos[q.pair] * per + i;      // (row masks extracted once, above)
                    const uint64_t lb = (uint64_t)prw[xr], db = anyd ? (uint64_t)pdw[xr] : 0ull;
                    uint64_t m = q.mask & lb & (dpass ? db : (~db & full));
                    if (half && m) m = spread2(m) & (uint64_t)prw2[xr];   // the live 64-key halves of those tiles
                    if (!m) continue;
                    if (q.sig != gsig) {                                 // a group: consecutive tiles, one signal
                        if (gsig != -2) g_end.push_back(ntile - base);
                        g_item.push_back(it); g_beg.push_back(ntile - base); g_sig.push_back(q.sig);
                        gsig = q.sig;
                    }
                    const int32_t n = __builtin_popcountll(m);
                    n_masked[it] += dpass * n; row_nremote[it] += (int32_t)(1 - q.loc) * n;
                    seg_cb.push_back((int32_t)(half ? 2 * q.cb : q.cb)); seg_lo.push_back((int32_t)(m & 0xffffffffull));
                    seg_hi.push_back((int32_t)(m >> 32)); seg_loc.push_back((int32_t)q.loc);
                    ntile += n;
                }
            if (gsig != -2) g_end.push_back(ntile - base);
            indptr[it + 1] = (int32_t)ntile;
            row_seg_ptr[it + 1] = (int32_t)seg_cb.size();
        }
    }
    TP("T:tiles");
    // ---- claim order: the schedule's row-group priority (block rows -> their tile rows) --------------------- //
    std::vector<int64_t> rank(nv, 0);
    for (int64_t x = O.row_ptr[pe]; x < O.row_ptr[pe + 1]; ++x)
        for (int64_t i = 0; i < per; ++i) rank[O.row_blk[x] * per + i] = (x - O.row_ptr[pe]) * per + i;
    std::vector<int32_t> work(nit);
    for (int64_t it = 0; it < nit; ++it) work[it] = (int32_t)it;
    std::stable_sort(work.begin(), work.end(), [&](int32_t a, int32_t b) { return rank[it_row[a]] < rank[it_row[b]]; });
    // ---- partials home ---------------------------------------------------------------------------------------- //
    std::vector<int32_t> opush_pe(nit, -1), opush_slot(nit, -1), item_q(nit), row_qwait(nit, -1);
    for (int64_t it = 0; it < nit; ++it) {
        const int64_t r = it_row[it], o = own(r);
        item_q[it] = qslot_me[r];
        if (o != pe) { opush_pe[it] = (int32_t)o; opush_slot[it] = land_slot(r, pe); row_qwait[it] = (int32_t)q_sig[qslot_me[r]]; }
    }
    TP("T:work/opush");
    // ---- owner merges (rows this PE alone writes go straight to the output: item_direct) ------------------- //
    std::vector<int32_t> item_direct(nit, -1), merge_out, merge_ptr(1, 0), merge_slot, mwait_ptr(1, 0), mwait_sig,
        merge_self, mself_ptr(1, 0), mself_idx;
    std::vector<int64_t> land_here;
    {
        int64_t orow = 0;
        const PO &Cme = po(pe);
        for (size_t xb = 0; xb < owned[pe].size(); ++xb) {
            const int64_t blk = owned[pe][xb];
            for (int64_t i = 0; i < per; ++i, ++orow) {
                const int64_t t = blk * per + i;
                int ncon = 0; bool self = false;
                for (int32_t x = Cme.bptr[xb]; x < Cme.bptr[xb + 1]; ++x) if (Cme.cb[x].m >> i & 1) { ++ncon; self |= Cme.cb[x].pe == pe; }
                if (ncon == 0) continue;
                if (ncon == 1 && self) { item_direct[itemidx[t]] = (int32_t)orow; continue; }
                merge_out.push_back((int32_t)orow);
                int32_t first_self = -1;
                for (int32_t x = Cme.bptr[xb]; x < Cme.bptr[xb + 1]; ++x) {
                    if (!(Cme.cb[x].m >> i & 1)) continue;
                    const int64_t d = Cme.cb[x].pe;
                    if (d == pe) { const int32_t it = (int32_t)itemidx[t]; merge_slot.push_back(it); mself_idx.push_back(it); if (first_self < 0) first_self = it; }
                    else { const int32_t s_ = land_slot(t, d); merge_slot.push_back(s_); mwait_sig.push_back((int32_t)(NQ + NKV + s_)); land_here.push_back(s_); }
                }
                merge_self.push_back(first_self);
                mself_ptr.push_back((int32_t)mself_idx.size());
                merge_ptr.push_back((int32_t)merge_slot.size()); mwait_ptr.push_back((int32_t)mwait_sig.size());
            }
        }
    }
    TP("T:merges");
    // ---- remote signals this PE polls ----------------------------------------------------------------------- //
    std::vector<int64_t> rs;
    for (int64_t it = 0; it < nit; ++it) if (row_qwait[it] >= 0) rs.push_back(row_qwait[it]);
    for (int64_t blk = 0; blk < nb; ++blk)
        if (owner_blk[blk] != pe) for (uint64_t m = colr_me[blk]; m; m &= m - 1) rs.push_back(kv_sig[kvs_of_tile(blk * per + __builtin_ctzll(m))]);
    for (int64_t s_ : land_here) rs.push_back(NQ + NKV + s_);
    std::sort(rs.begin(), rs.end()); rs.erase(std::unique(rs.begin(), rs.end()), rs.end());
    std::vector<int32_t> sig_pos(NQ + NKV + NPART + 1, -1);
    for (size_t j = 0; j < rs.size(); ++j) sig_pos[rs[j]] = (int32_t)j;
    auto sig_index = [&](int64_t id) -> int64_t { return sig_pos[id]; };
    std::vector<int32_t> remote_sig(rs.begin(), rs.end()), rc(rs.size(), 1);
    for (size_t j = 0; j < rs.size(); ++j) if (rs[j] >= NQ && rs[j] < NQ + NKV) rc[j] = (int32_t)kv_span[rs[j] - NQ];
    TP("T:remote_sig");
    // ---- outgoing: inter-node pushes (kernel), copy-engine runs, relay forwards ----------------------------- //
    std::vector<int64_t> idst;
    for (int64_t d = 0; d < P; ++d) if (d / gpn != pe / gpn) idst.push_back(d);
    std::stable_sort(idst.begin(), idst.end(), [&](int64_t a, int64_t b) {
        const int64_t ka = ((a / gpn - pe / gpn) % NN + NN) % NN, kb = ((b / gpn - pe / gpn) % NN + NN) % NN;
        return ka != kb ? ka < kb : a < b;
    });
    std::vector<std::vector<int64_t>> from_pe(P);                      // messages pe -> d, by first block
    for (int64_t m = 0; m < nm; ++m) if (O.ms[m] == pe) from_pe[O.md[m]].push_back(m);
    for (auto &v : from_pe) std::stable_sort(v.begin(), v.end(), [&](int64_t a, int64_t b) { return O.mb0[a] < O.mb0[b]; });
    TP("t:p:from_pe");
    std::vector<int32_t> pq_src, pq_pe, pq_dst, pq_cnt, pk_src, pk_pe, pk_dst, pk_cnt;
    std::vector<std::tuple<int64_t, int, int64_t>> pkey;               // (send pos, kind, index)
    std::vector<int64_t> qpos, kpos;
    // ilv != 0 with transfer chunks: send key = (chunk ordinal in its message, message position). Off.
    constexpr int ilv = 0;
    for (int64_t d : idst)
        for (int k = 0; k < 2; ++k)
            for (int64_t m : from_pe[d]) {
                if (O.mk[m] != k || O.mtype[m] != 0) continue;
                int64_t ord = 0;
                runs(m, [&](int64_t a, int64_t n, int64_t) {
                    auto &src = k ? pk_src : pq_src; auto &pp = k ? pk_pe : pq_pe; auto &dst = k ? pk_dst : pq_dst;
                    auto &cn = k ? pk_cnt : pq_cnt;
                    src.push_back((int32_t)slot(k, pe, a)); pp.push_back((int32_t)d); dst.push_back((int32_t)slot(k, d, a));
                    cn.push_back((int32_t)n); (k ? kpos : qpos).push_back(ilv ? (ord++ << 32) + O.pos[m] : O.pos[m]);
                });
            }
    TP("t:p:runs");
    for (size_t i = 0; i < qpos.size(); ++i) pkey.emplace_back(qpos[i], 0, (int64_t)i);
    for (size_t i = 0; i < kpos.size(); ++i) pkey.emplace_back(kpos[i], 1, (int64_t)(qpos.size() + i));
    std::sort(pkey.begin(), pkey.end());
    std::vector<int32_t> push_order;
    for (auto &e : pkey) push_order.push_back((int32_t)std::get<2>(e));
    TP("T:pushes");
    std::vector<int64_t> ownidx(nv, -1);
    { int64_t k = 0; for (int64_t blk : owned[pe]) for (int64_t i = 0; i < per; ++i) ownidx[blk * per + i] = k++; }
    std::vector<int32_t> ce;                                            // [n, 7]
    for (int64_t d = (pe / gpn) * gpn; d < std::min(P, (pe / gpn + 1) * gpn); ++d) {
        if (d == pe) continue;
        std::vector<std::array<int64_t, 8>> rows;                       // (pos, kind, seq, d, kind, src, dsl, cnt / sig / m)
        std::vector<std::array<int64_t, 7>> rr;
        std::vector<std::pair<int64_t, int64_t>> key;
        for (int k = 0; k < 2; ++k)
            for (int64_t m : from_pe[d]) {
                if (O.mk[m] != k || O.mtype[m] != 1) continue;
                runs(m, [&](int64_t a, int64_t n, int64_t sig) {
                    key.emplace_back(O.pos[m], k);
                    rr.push_back({d, k, ownidx[a], slot(k, d, a), n, sig, m});
                });
            }
        std::vector<int64_t> o(rr.size());
        for (size_t i = 0; i < o.size(); ++i) o[i] = (int64_t)i;
        std::stable_sort(o.begin(), o.end(), [&](int64_t a, int64_t b) { return key[a] < key[b]; });
        for (int64_t i : o) for (int c = 0; c < 7; ++c) ce.push_back((int32_t)rr[i][c]);
    }
    std::vector<int32_t> ce_sig;
    for (size_t i = 0; i < ce.size() / 7; ++i) ce_sig.push_back(ce[7 * i + 5]);
    TP("T:ce");
    std::vector<int32_t> fw;                                            // [n, 7] relay forwards out of pe
    {
        std::vector<std::array<int64_t, 7>> rr;
        std::vector<int64_t> key;
        for (int64_t d = (pe / gpn) * gpn; d < std::min(P, (pe / gpn + 1) * gpn); ++d) {
            if (d == pe) continue;
            for (int k = 0; k < 2; ++k)
                for (int64_t m : from_pe[d]) {
                    if (O.mk[m] != k || O.mtype[m] != 2) continue;
                    // the parent's run signal at this gateway, per block
                    const int64_t par = O.mpar[m];
                    std::vector<int64_t> wt(O.mcnt[par]);
                    {
                        const int64_t pb0 = O.mb0[par];
                        runs(par, [&](int64_t a, int64_t n, int64_t sig) { for (int64_t j = 0; j < n; ++j) wt[a - pb0 + j] = sig; });
                    }
                    runs(m, [&](int64_t a, int64_t n, int64_t sig) {
                        rr.push_back({d, k, slot(k, pe, a), slot(k, d, a), n, sig, wt[a - O.mb0[par]]});
                        key.push_back(O.pos[m]);
                    });
                }
        }
        std::vector<int64_t> o(rr.size());
        for (size_t i = 0; i < o.size(); ++i) o[i] = (int64_t)i;
        std::stable_sort(o.begin(), o.end(), [&](int64_t a, int64_t b) { return key[a] < key[b]; });
        for (int64_t i : o) for (int c = 0; c < 7; ++c) fw.push_back((int32_t)rr[i][c]);
    }
    TP("T:fw");
    // ---- scheduler tables: groups, per-signal lists, pool, merges -------------------------------------------- //
    std::vector<int32_t> r_seg_dep;
    if (rect) {                                                         // segments' signals -> remote_sig indices
        for (int64_t sg : r_seg_sig) r_seg_dep.push_back(sg >= 0 ? (int32_t)sig_index(sg) : -1);
    }
    std::vector<int32_t> gx_dep;
    if (gx) for (int64_t sg_ : gx_sig) gx_dep.push_back(sg_ >= 0 ? (int32_t)sig_index(sg_) : -1);
    const int64_t ngr = (int64_t)g_item.size();
    std::vector<int32_t> row_grp_ptr(gx ? 0 : nit + 1, 0), gb(ngr), ge(ngr), gdep(ngr), row_qdep(nit, -1);
    for (int64_t g = 0; g < ngr; ++g) {
        ++row_grp_ptr[g_item[g] + 1]; gb[g] = (int32_t)g_beg[g]; ge[g] = (int32_t)g_end[g];
        gdep[g] = g_sig[g] >= 0 ? (int32_t)sig_index(g_sig[g]) : -1;
    }
    if (!gx) for (int64_t it = 0; it < nit; ++it) row_grp_ptr[it + 1] += row_grp_ptr[it];
    for (int64_t it = 0; it < nit; ++it) if (row_qwait[it] >= 0) row_qdep[it] = (int32_t)sig_index(row_qwait[it]);
    const int64_t nsig = (int64_t)rs.size();
    auto rev = [&](const std::vector<int32_t> &keys, const std::vector<int32_t> &vals, std::vector<int32_t> &ptr,
                   std::vector<int32_t> &out) {                       // CSR by signal index, values in order
        ptr.assign(nsig + 1, 0);
        for (int32_t k : keys) if (k >= 0) ++ptr[k + 1];
        for (int64_t j = 0; j < nsig; ++j) ptr[j + 1] += ptr[j];
        out.assign(ptr[nsig], 0);
        std::vector<int32_t> f(ptr.begin(), ptr.end() - 1);
        for (size_t i = 0; i < keys.size(); ++i) if (keys[i] >= 0) out[f[keys[i]]++] = vals[i];
    };
    std::vector<int32_t> sgp, sg, srp, sr_, smp, sm, idx_g(ngr), idx_r(nit);
    for (int64_t g = 0; g < ngr; ++g) idx_g[g] = (int32_t)g;
    for (int64_t it = 0; it < nit; ++it) idx_r[it] = (int32_t)it;
    if (rect) {
        std::vector<int32_t> idx_s(r_seg_dep.size());
        for (size_t k = 0; k < idx_s.size(); ++k) idx_s[k] = (int32_t)k;
        rev(r_seg_dep, idx_s, sgp, sg);
    } else if (!gx) {
        rev(gdep, idx_g, sgp, sg);
    }
    rev(row_qdep, idx_r, srp, sr_);
    std::vector<int32_t> mk, mv, merge_need, row_merge(nit, -1);
    for (size_t mi = 0; mi + 1 < merge_ptr.size(); ++mi) {
        for (int32_t j = mwait_ptr[mi]; j < mwait_ptr[mi + 1]; ++j) { mk.push_back((int32_t)sig_index(mwait_sig[j])); mv.push_back((int32_t)mi); }
        for (int32_t j = mself_ptr[mi]; j < mself_ptr[mi + 1]; ++j) row_merge[mself_idx[j]] = (int32_t)mi;
        merge_need.push_back((mwait_ptr[mi + 1] - mwait_ptr[mi]) + (mself_ptr[mi + 1] - mself_ptr[mi]));
    }
    rev(mk, mv, smp, sm);
    std::vector<int32_t> row_pool(gx ? 0 : nit, -1);
    int64_t n_pool = 0;
    for (int64_t it = 0; it < (gx ? 0 : nit); ++it) {
        const int32_t n = rect ? r_row_ne[it] : row_grp_ptr[it + 1] - row_grp_ptr[it];
        if (n > 1) { row_pool[it] = (int32_t)n_pool; n_pool += n; }
    }
    std::vector<int32_t> ce_pout;                                       // intra-node partials by copy engine
    if (ce_partial) {
        std::vector<int64_t> pr(nit);
        for (int64_t k = 0; k < nit; ++k) pr[work[k]] = k;
        std::vector<int64_t> rows_;
        for (int64_t it = 0; it < nit; ++it) if (opush_pe[it] >= 0 && opush_pe[it] / gpn == pe / gpn) rows_.push_back(it);
        std::stable_sort(rows_.begin(), rows_.end(), [&](int64_t a, int64_t b) { return pr[a] < pr[b]; });
        for (int64_t it : rows_) { ce_pout.push_back((int32_t)it); ce_pout.push_back(opush_pe[it]); ce_pout.push_back(opush_slot[it]); }
    }
    TP("T:sched tabs");
    // ---- assemble (order = fwd.py's _TAB_NAMES) ---------------------------------------------------------------- //
    std::vector<int32_t> row_mask_mb, stq_s, stq_d, stk_s, stk_d;
    for (int64_t it = 0; it < nit; ++it) row_mask_mb.push_back((int32_t)(it_row[it] + NQ - NKV));
    if (!nit) row_mask_mb.push_back(0);
    { int32_t k = 0; for (int64_t blk : owned[pe]) for (int64_t i = 0; i < per; ++i, ++k) {
        stq_s.push_back(k); stq_d.push_back(qslot_me[blk * per + i]); stk_s.push_back(k); stk_d.push_back((int32_t)kvs_of_tile(blk * per + i)); } }
    std::vector<int32_t> g_beg32(gb), g_end32(ge);
    TP("T:stage");
    Tabs &T = g_tab;
    T.scal = {pl(pe).nq, nv, po(pe).npart, nit, (int64_t)merge_out.size(), (int64_t)owned[pe].size() * per, NQ, NKV, NPART, n_pool, ntile};
    T.a.clear(); T.a.reserve(55);
    T.a.push_back(std::move(indptr));
    T.a.push_back(std::move(seg_cb));
    T.a.push_back(std::move(item_direct));
    T.a.push_back(std::move(n_masked));
    T.a.push_back(std::move(row_mask_mb));
    T.a.push_back(work);
    T.a.push_back(std::vector<int32_t>(nit + 1, 0));
    T.a.push_back(std::move(rc));
    T.a.push_back(std::move(opush_pe));
    T.a.push_back(std::move(opush_slot));
    T.a.push_back(std::move(merge_out));
    T.a.push_back(std::move(merge_ptr));
    T.a.push_back(std::move(merge_slot));
    T.a.push_back(std::move(mwait_ptr));
    T.a.push_back(std::move(mwait_sig));
    T.a.push_back(std::move(remote_sig));
    T.a.push_back(std::move(stq_s));
    T.a.push_back(std::move(stq_d));
    T.a.push_back(std::move(stk_s));
    T.a.push_back(std::move(stk_d));
    T.a.push_back(std::move(pq_src));
    T.a.push_back(std::move(pq_pe));
    T.a.push_back(std::move(pq_dst));
    T.a.push_back(std::move(pk_src));
    T.a.push_back(std::move(pk_pe));
    T.a.push_back(std::move(pk_dst));
    T.a.push_back(std::move(merge_self));
    T.a.push_back(std::move(row_nremote));
    T.a.push_back(std::move(row_qwait));
    T.a.push_back(std::move(seg_loc));
    T.a.push_back(std::move(push_order));
    T.a.push_back(std::move(item_q));
    T.a.push_back(std::move(mself_ptr));
    T.a.push_back(std::move(mself_idx));
    T.a.push_back(std::move(ce));
    T.a.push_back(std::move(ce_sig));
    T.a.push_back(std::move(pq_cnt));
    T.a.push_back(std::move(pk_cnt));
    T.a.push_back(std::move(row_grp_ptr));
    T.a.push_back(std::move(g_beg32));
    T.a.push_back(std::move(g_end32));
    T.a.push_back(std::move(gdep));
    T.a.push_back(std::move(row_qdep));
    T.a.push_back(std::move(sgp));
    T.a.push_back(std::move(sg));
    T.a.push_back(std::move(srp));
    T.a.push_back(std::move(sr_));
    T.a.push_back(std::move(smp));
    T.a.push_back(std::move(sm));
    T.a.push_back(work);
    T.a.push_back(std::move(row_pool));
    T.a.push_back(std::move(row_merge));
    T.a.push_back(std::move(merge_need));
    T.a.push_back(std::move(ce_pout));
    T.a.push_back(std::move(fw));
    if (rect) {
        auto w64 = [](const std::vector<int64_t> &v) {                // int64 table carried as int32 words
            std::vector<int32_t> o(2 * v.size());
            std::memcpy(o.data(), v.data(), v.size() * 8);
            return o;
        };
        T.a.push_back(std::move(r_row_brow)); T.a.push_back(std::move(r_row_bi)); T.a.push_back(std::move(r_row_base));
        T.a.push_back(std::move(brow_seg_ptr)); T.a.push_back(std::move(r_seg_cb)); T.a.push_back(std::move(r_seg_pair));
        T.a.push_back(std::move(r_seg_dep)); T.a.push_back(w64(r_seg_mask)); T.a.push_back(w64(prw)); T.a.push_back(w64(pdw));
        T.scal.push_back(rowseg);
    }
    T.a.push_back(std::move(seg_lo));                                  // (extra: segment masks, low / high words)
    T.a.push_back(std::move(seg_hi));
    T.a.push_back(std::move(row_seg_ptr));                             // (extra: row -> segments, non-rect)
    if (kvc) T.a.push_back(std::move(kslot_me));                       // (extra, compact KV: tile -> this PE's KV slot)
    if (gx) {                                                           // (extra, GPU expansion: block-level segments)
        auto w64 = [](const std::vector<int64_t> &v) {
            std::vector<int32_t> o(2 * v.size());
            std::memcpy(o.data(), v.data(), v.size() * 8);
            return o;
        };
        T.a.push_back(std::move(gx_bs_ptr)); T.a.push_back(std::move(gx_cb)); T.a.push_back(std::move(gx_lo));
        T.a.push_back(std::move(gx_hi)); T.a.push_back(std::move(gx_dep)); T.a.push_back(std::move(gx_loc));
        T.a.push_back(std::move(gx_pair)); T.a.push_back(std::move(gx_rbrow)); T.a.push_back(std::move(gx_rbi));
        T.a.push_back(w64(prw)); T.a.push_back(w64(pdw));
    }
    TP("T:assemble");
    return (int64_t)T.a.size();
}

int64_t fsb_tab_size(int64_t k) { return (int64_t)g_tab.a[k].size(); }
// all tables back to back into dst (int32), their sizes into sizes[k]
void fsb_tab_pack(int32_t *dst, int64_t *sizes) {
    int64_t off = 0;
    for (size_t k = 0; k < g_tab.a.size(); ++k) {
        sizes[k] = (int64_t)g_tab.a[k].size();
        std::memcpy(dst + off, g_tab.a[k].data(), g_tab.a[k].size() * 4); off += sizes[k];
    }
}
int64_t fsb_tab_total() { int64_t n = 0; for (auto &v : g_tab.a) n += (int64_t)v.size(); return n; }
void fsb_tab_fetch(int64_t k, int32_t *dst) { std::memcpy(dst, g_tab.a[k].data(), g_tab.a[k].size() * 4); }
void fsb_tab_scalars(int64_t *dst) { std::memcpy(dst, g_tab.scal.data(), g_tab.scal.size() * 8); }

}  // extern "C"
