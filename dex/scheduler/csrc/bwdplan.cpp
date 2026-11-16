// BACKWARD tables in C++ from the block-level schedule (dex/scheduler/bwd.py build_fast).
//
// The backward needs the same inputs as the forward -- a GPU computing pairs of Q tile i needs i's Q-side bundle
// (Q, dO, LSE, Delta), one computing pairs of KV tile j needs K_j, V_j -- but its resident side is the KV COLUMN. So
// python runs fsb_schedule (fastsched_blocks.cpp) on the TRANSPOSED pairs (rows = KV column blocks, message kind 0 = KV,
// kind 1 = Q bundle, the backward's byte sizes and per-pair work): messages with relay gateways, the static per-link
// send order, landing times, and the compute order of the column blocks per GPU. bwd_tables then builds one PE's kernel
// tables (dex/scheduler/bwd.py TABLE_ORDER + the message tables):
//   slots        Q bundle / KV slots: owned tiles ascending, then the remote tiles this PE computes, ascending
//   gateways     per (kind, node, tile): the GPU the schedule's relay forwards a cross-node input from; the node's other
//                contributors push their fp32 partial sum to it (aggregated there, one partial crosses to the owner)
//   landings     at an acceptor (owner or gateway): one slot per (tile, contributor) in (tile, contributor) order
//   inputs       every message of the schedule this PE sends, in the send order, cut into runs contiguous at both ends;
//                a run is signalled on its LAST slot and every receiver waits on the run's signal (one polled signal per
//                run); a relay forward waits for its parent run's signal at the gateway
//   tasks        (rect = 1, the mode used): rectangles = a KV plan block x the Q rows of one owner on this PE, in the
//                schedule's compute order; groups = a column x <= 64 rows; block 0 of the kernel materialises the
//                items (the per-tile task mode, rect = 0, is not used)
//   the rest     owner sums, gateway aggregations, the shared block-0 scheduler's tables, the message tables by lane
#include "fastsched_blocks.cpp"
#include <array>
#include <map>
#include <unordered_map>
#include <sys/mman.h>
#include <cstdlib>
#include <thread>
#include <functional>
#include <cstdio>

namespace {
// Allocator for the big item table: 2 MiB-aligned + MADV_HUGEPAGE, so first touch faults 2 MiB pages, not 4 KiB ones
template <class T> struct HugeAlloc {
    using value_type = T;
    HugeAlloc() = default;
    template <class U> HugeAlloc(const HugeAlloc<U> &) {}
    T *allocate(size_t n) {
        const size_t b = ((n * sizeof(T) + (2u << 20) - 1) >> 21) << 21;
        void *p = std::aligned_alloc(2u << 20, b);
        if (!p) throw std::bad_alloc();
        madvise(p, b, MADV_HUGEPAGE);
        return (T *)p;
    }
    void deallocate(T *p, size_t) { std::free(p); }
    template <class U> void construct(U *p) { ::new ((void *)p) U; }  // resize() leaves entries uninitialised
    template <class U, class... A> void construct(U *p, A &&...a) { ::new ((void *)p) U(std::forward<A>(a)...); }
    template <class U> bool operator==(const HugeAlloc<U> &) const { return true; }
    template <class U> bool operator!=(const HugeAlloc<U> &) const { return false; }
};
struct BwdOut {
    std::vector<std::vector<int32_t>> tab;    // python TABLE_ORDER (reserved names empty)
    std::vector<int32_t, HugeAlloc<int32_t>> items;   // tab[2] (kept apart, capacity reused across calls)
    std::vector<int64_t> msg, ce;             // int64 [n, 4] engine messages, int64 [n, 6] copy-engine messages
    std::vector<int64_t> scal;                // NQS NKS NDQL NDKL tasks items n_init n_qm n_km n_remote pushq pushkv
                                              // n_fwd n_qagg n_kagg n_pq n_start n_ce n_rows n_red n_fwd_msgs
                                              // n_groups n_rect G1 R_arena nw per
} g_bwd;
// symmetric sizes NQS NKS NDQL NDKL: -1 = from every PE (global planning); per-rank planning sets the all-reduced maxima.
// g_bsym_me = this rank's own sizes; g_bsym_only: stop once they are known
int64_t g_bsym[4] = {-1, -1, -1, -1}, g_bsym_me[4] = {0, 0, 0, 0};
int g_bsym_only = 0;
// per-rank planning exchange: this rank's landings (pass 1 output, (tile, contributor) in slot order per kind) and the
// slots its partials get at their acceptors (pass 2 input, from the acceptors' lists): key kind * 2^32 + tile
std::vector<int32_t> g_bin[2];
std::unordered_map<int64_t, std::pair<int32_t, int32_t>> g_bout;
int g_bout_dummy = 0;                                                   // timing probes only: missing slots = (0, 0)
}  // namespace

static int64_t g_bregion = 0;     // region bytes every PE allocates: the busiest PE's own layout (per-PE
                                   // layout) or the global-maximum layout
extern "C" {
int64_t bwd_region_max() { return g_bregion; }

void bwd_set_sym(int64_t nqs, int64_t nks, int64_t ndql, int64_t ndkl, int only) {
    g_bsym[0] = nqs; g_bsym[1] = nks; g_bsym[2] = ndql; g_bsym[3] = ndkl; g_bsym_only = only;
}
void bwd_get_sym_me(int64_t *out) { for (int i = 0; i < 4; ++i) out[i] = g_bsym_me[i]; }
void bwd_set_out_dummy(int v) { g_bout_dummy = v; }
int64_t bwd_in_size(int k) { return (int64_t)g_bin[k].size() / 2; }
void bwd_get_in(int k, int32_t *tc) { std::memcpy(tc, g_bin[k].data(), g_bin[k].size() * 4); }
void bwd_set_out(int64_t n, const int64_t *ktal) {                     // rows (kind, tile, acceptor, slot)
    g_bout.clear();
    for (int64_t i = 0; i < n; ++i) g_bout[ktal[4 * i] * (1ll << 32) + ktal[4 * i + 1]] = {(int32_t)ktal[4 * i + 2], (int32_t)ktal[4 * i + 3]};
}

// Call right after fsb_schedule on the transposed pairs of the same plan. bits / dbits: per pair (ORIGINAL orientation:
// bit a * per + b = tile row a, tile column b) the live / causal-diagonal tiles, nw uint64 words each. h, hk: query /
// KV heads; ce_intra: intra-node messages by the copy-engine executor (else block 0's engine). Returns the table count.
int64_t bwd_tables(int64_t pe, int64_t nb, int64_t per, int64_t P, int64_t gpn, int64_t npair, const int64_t *bq,
                   const int64_t *bk, const int64_t *pdev, const uint64_t *prow, const uint64_t *pcol,
                   const uint64_t *bits, const uint64_t *dbits, int64_t nw,
                   const int64_t *owner_blk, int64_t h, int64_t hk, int64_t d, int ce_intra, int ablate, int rect) {
    // rect = 1: RECTANGLE tables instead of per-tile tasks -- the planner only decides the big picture
    // (rectangles = a KV plan block x the Q rows of one owner, their groups = a column x <= 64 rows, coarse inputs);
    // block 0 of the kernel materialises the item lists and counts the completions (dist_bwd_kernel.h rect_block0)
    // ablate (bits): 1 = columns ascending, not the schedule's compute order; 2 = inputs destination-major, not the
    // schedule's send order; 4 = owner groups by owner id, not expected landing
#ifdef FSB_TIMING
    auto _t0 = std::chrono::steady_clock::now();
#endif
    const Out &O = g_out;
    const int64_t r = pe, nv = nb * per, nm = (int64_t)O.ms.size(), NN = (P + gpn - 1) / gpn, T = 128;
    const int64_t rg = h / hk, MAX_QS = 64;
    // q heads per materialised group: rg = one group per KV head holding all its q heads. HG = h / qpg groups per
    // planner group g1:
    // g = qg * G1 + g1, q heads [qg * qpg, (qg + 1) * qpg), KV head (qg * qpg) / rg.
    constexpr int qpg_env = 0;                                  // 0 = all q heads of a KV head in one group
    const int64_t qpg = (qpg_env > 0 && rg % qpg_env == 0) ? qpg_env : rg, HG = h / qpg;
    std::vector<int32_t> owner_t(nv);
    for (int64_t b = 0; b < nb; ++b) for (int64_t t = b * per; t < (b + 1) * per; ++t) owner_t[t] = (int32_t)owner_blk[b];
    const FDiv dper(per, nv + 1);                                       // (t / per without an idiv)
    auto own = [&](int64_t t) -> int64_t { return owner_t[t]; };
    TP("b:t:owner_t");
    auto node = [&](int64_t x) { return x / gpn; };
    auto bit = [&](const uint64_t *w, int64_t p, int64_t k) { return (w[p * nw + (k >> 6)] >> (k & 63)) & 1ull; };
    // ---- who computes what: per tile, the GPUs computing part of its row (kind 0 = Q) / column (1 = KV) ----------- //
    // CSR lists, ascending, built ON DEMAND per (kind, plan block) from pair indexes (by row block, by column block, by
    // computing GPU; counting sorts), because building them for every tile from all pairs does not scale to large P. The symmetric sizes (NQS NKS NDQL NDKL) come from bwd_set_sym (per-rank planning: the all-reduce of
    // every rank's own sizes) or, if not given, from every PE (global planning)
    std::vector<int32_t> bxp[3], bxi[3];                                // pairs by row block / column block / GPU
    csr_by_key(pdev, npair, P, bxp[2], bxi[2]);                         // (counting sorts by runs, csr_by_key)
    // by row / column block: lazily -- per-rank planning reads only the blocks this PE owns: those come from
    // a filtered index (the pairs of owned blocks, one pass); any other block builds the full index on first use. A
    // block's pair list is the same either way (all its pairs) and its consumer (cbuild) does not depend on their order
    std::vector<int32_t> oxp[2], oxi[2];
    auto blk_pairs = [&](int k, int64_t b, int32_t &i0, int32_t &i1) -> const int32_t * {
        if (owner_blk[b] == r) {
            if (oxp[0].empty()) {                                       // (both kinds in one pass over the pairs)
                std::vector<int32_t> hit[2];
                for (int64_t p = 0; p < npair; ++p) {
                    if (owner_blk[bq[p]] == r) hit[0].push_back((int32_t)p);
                    if (owner_blk[bk[p]] == r) hit[1].push_back((int32_t)p);
                }
                for (int kk = 0; kk < 2; ++kk) {
                    const int64_t *key_ = kk == 0 ? bq : bk;
                    oxp[kk].assign(nb + 1, 0);
                    for (int32_t p : hit[kk]) ++oxp[kk][key_[p] + 1];
                    for (int64_t x = 0; x < nb; ++x) oxp[kk][x + 1] += oxp[kk][x];
                    oxi[kk].resize(hit[kk].size());
                    std::vector<int32_t> f(oxp[kk].begin(), oxp[kk].end() - 1);
                    for (int32_t p : hit[kk]) oxi[kk][f[key_[p]]++] = p;
                }
            }
            i0 = oxp[k][b]; i1 = oxp[k][b + 1]; return oxi[k].data();
        }
        if (bxp[k].empty()) csr_by_key(k == 0 ? bq : bk, npair, nb, bxp[k], bxi[k]);
        i0 = bxp[k][b]; i1 = bxp[k][b + 1]; return bxi[k].data();
    };
    TP("b:t:index");
    std::vector<int32_t> cbeg[2], cend[2], cpool[2];                   // per tile: its range in the kind's pool
    std::vector<char> cdone[2];
    for (int k = 0; k < 2; ++k) { cbeg[k].assign(nv, 0); cend[k].assign(nv, 0); cdone[k].assign(nb, 0); cpool[k].reserve(npair * 2); }
    std::vector<std::pair<int32_t, uint64_t>> gm;                      // one block's (GPU, OR of its pairs' masks)
    // (key, OR of masks) of a list of pairs, keys ascending and merged: a bitmap of the keys seen + one accumulator per
    // key, O(pairs + key range / 64)
    std::vector<uint64_t> acc_(std::max(nb, P), 0), seen_((std::max(nb, P) + 63) / 64, 0);
    auto merged = [&](int32_t i0, int32_t i1, const int32_t *ix, const int64_t *key_, const uint64_t *msk, auto keep, auto out) {
        int64_t lo = INT64_MAX, hi = -1;
        for (int32_t i = i0; i < i1; ++i) {
            const int64_t p = ix[i], b = key_[p];
            if (!keep(b)) continue;
            seen_[b >> 6] |= 1ull << (b & 63); acc_[b] |= msk[p];
            lo = std::min(lo, b); hi = std::max(hi, b);
        }
        if (hi < 0) return;
        for (int64_t w = lo >> 6; w <= hi >> 6; ++w) {
            for (uint64_t x = seen_[w]; x; x &= x - 1) {
                const int64_t b = w * 64 + __builtin_ctzll(x);
                out(b, acc_[b]); acc_[b] = 0;
            }
            seen_[w] = 0;
        }
    };
    // DENSE masks when P x blocks is small next to the pairs: one pass gives every (GPU, block)'s OR of
    // row / column masks; a GPU's remote tiles (rl) and a block's contributors (cbuild) are read from it instead of
    // gathering pairs per GPU / block (a zero mask contributes no tile either way: the same lists)
    const bool dense_b = !g_bsym_only && P * nb <= ((int64_t)1 << 22) && P * nb <= 16 * npair;
    std::vector<uint64_t> DM[2];
    if (dense_b) {
        DM[0].assign((size_t)P * nb, 0); DM[1].assign((size_t)P * nb, 0);
        for (int64_t p = 0; p < npair; ++p) { DM[0][(size_t)pdev[p] * nb + bq[p]] |= prow[p]; DM[1][(size_t)pdev[p] * nb + bk[p]] |= pcol[p]; }
    }
    auto cbuild = [&](int k, int64_t b) {
        cdone[k][b] = 1;
        const uint64_t *msk = k ? pcol : prow;
        gm.clear();
        if (dense_b) {
            for (int64_t x = 0; x < P; ++x) { const uint64_t m = DM[k][(size_t)x * nb + b]; if (m) gm.push_back({(int32_t)x, m}); }
        } else {
            int32_t i0, i1;
            const int32_t *ix = blk_pairs(k, b, i0, i1);
            merged(i0, i1, ix, pdev, msk, [](int64_t) { return true; },
                   [&](int64_t g, uint64_t m) { gm.push_back({(int32_t)g, m}); });
        }
        auto &pool = cpool[k];
        for (int64_t a = 0; a < per; ++a) {                             // the block's tiles, GPUs ascending
            cbeg[k][b * per + a] = (int32_t)pool.size();
            for (auto &e : gm) if (e.second >> a & 1) pool.push_back(e.first);
            cend[k][b * per + a] = (int32_t)pool.size();
        }
    };
    auto for_con = [&](int k, int64_t t, auto f) {                     // contributors of tile t, ascending
        const int64_t b_ = dper.q(t);
        if (!cdone[k][b_]) cbuild(k, b_);
        const int32_t *x = cpool[k].data();
        for (int32_t i = cbeg[k][t], e = cend[k][t]; i < e; ++i) f((int64_t)x[i]);
    };
    auto con_n = [&](int k, int64_t t) { const int64_t b_ = dper.q(t); if (!cdone[k][b_]) cbuild(k, b_); return cend[k][t] - cbeg[k][t]; };
    auto has_con = [&](int k, int64_t t, int64_t x) {                 // does GPU x compute part of tile t?
        const int64_t b_ = dper.q(t);
        if (!cdone[k][b_]) cbuild(k, b_);
        const int32_t *p = cpool[k].data();
        for (int32_t i = cbeg[k][t], e = cend[k][t]; i < e; ++i) if (p[i] == x) return true;
        return false;
    };
    // this PE's tiles as CSR by column (cr_*), each column's rows ascending: pairs by row block, bits row-major
    std::vector<int64_t> mine;
    for (int32_t i = bxp[2][r]; i < bxp[2][r + 1]; ++i) mine.push_back(bxi[2][i]);
    std::sort(mine.begin(), mine.end(), [&](int64_t a, int64_t b) { return bq[a] != bq[b] ? bq[a] < bq[b] : a < b; });
    TP("b:t:mine");
    std::vector<int32_t> cr_ptr(rect ? 1 : nv + 1, 0), cr_i;           // (rect: unused)
    std::vector<int8_t> cr_kd;
    for (int pass = 0; pass < (rect ? 0 : 2); ++pass) {                 // (rect: the tiles stay in the pair bits)
        if (pass) {
            for (int64_t j = 0; j < nv; ++j) cr_ptr[j + 1] += cr_ptr[j];
            cr_i.resize(cr_ptr[nv]); cr_kd.resize(cr_ptr[nv]);
        }
        std::vector<int32_t> fill(cr_ptr.begin(), cr_ptr.end() - 1);
        for (int64_t p : mine)
            for (int64_t w = 0; w < nw; ++w)
                for (uint64_t m = bits[p * nw + w]; m; m &= m - 1) {
                    const int64_t k = w * 64 + __builtin_ctzll(m);
                    if (k >= per * per) break;
                    const int64_t j = bk[p] * per + k % per;
                    if (!pass) { ++cr_ptr[j + 1]; continue; }
                    const int32_t x = fill[j]++;
                    cr_i[x] = (int32_t)(bq[p] * per + k / per); cr_kd[x] = (int8_t)(bit(dbits, p, k) ? 2 : 1);
                }
    }
    auto col_n = [&](int64_t j) { return cr_ptr[j + 1] - cr_ptr[j]; };
    TP("B:tiles");
    // ---- slots of every PE: owned tiles ascending, then the remote tiles it computes ascending -------------------- //
    // (no P x tiles map: a tile's position among its owner's tiles, and per GPU the sorted list of the remote tiles it
    // computes -- a lookup is a binary search; this PE's own slots are dense)
    // a tile's position among its owner's tiles = the owner's earlier blocks x per + its row (per block, no per-tile array)
    std::vector<int32_t> ownposb(nb);
    std::vector<int64_t> nown(P, 0);
    for (int64_t b = 0; b < nb; ++b) { ownposb[b] = (int32_t)nown[owner_blk[b]]; nown[owner_blk[b]] += per; }
    auto ownpos_t = [&](int64_t t) -> int64_t { const int64_t b = dper.q(t); return (int64_t)ownposb[b] + (t - b * per); };
    std::vector<std::vector<int64_t>> owned_b(P);                       // plan blocks per owner, ascending
    for (int64_t b = 0; b < nb; ++b) owned_b[owner_blk[b]].push_back(b);
    TP("b:s:own");
    // per GPU on demand: the remote tiles it computes (kind 0: rows, 1: columns), ascending, from its pairs
    struct RL { bool done = false; std::vector<int32_t> t[2]; };
    std::vector<RL> rls(P);
    auto rl = [&](int64_t x) -> const RL & {
        RL &R_ = rls[x];
        if (R_.done) return R_;
        R_.done = true;
        for (int k = 0; k < 2; ++k) {                                   // (block, OR of masks), then the bits
            auto &v = R_.t[k];
            if (dense_b) {
                const uint64_t *row = DM[k].data() + (size_t)x * nb;
                for (int64_t b = 0; b < nb; ++b)
                    if (row[b] && owner_blk[b] != x) for (uint64_t m = row[b]; m; m &= m - 1) v.push_back((int32_t)(b * per + __builtin_ctzll(m)));
                continue;
            }
            merged(bxp[2][x], bxp[2][x + 1], bxi[2].data(), k ? bk : bq, k ? pcol : prow,
                   [&](int64_t b) { return owner_blk[b] != x; },
                   [&](int64_t b, uint64_t m) { for (; m; m &= m - 1) v.push_back((int32_t)(b * per + __builtin_ctzll(m))); });
        }
        return R_;
    };
    // dense: another GPU's remote tiles are never listed -- a slot is its owned tiles + the remote tiles in its
    // earlier blocks (prefix over the dense row) + the bits below in its block: the rank in the list it replaces
    struct RP { bool done = false; std::vector<int32_t> pre[2]; };
    std::vector<RP> rps(dense_b ? P : 0);
    auto rp = [&](int64_t x) -> const RP & {
        RP &R_ = rps[x];
        if (R_.done) return R_;
        R_.done = true;
        for (int k = 0; k < 2; ++k) {
            auto &v = R_.pre[k]; v.assign(nb + 1, 0);
            const uint64_t *row = DM[k].data() + (size_t)x * nb;
            for (int64_t b = 0; b < nb; ++b) v[b + 1] = v[b] + (owner_blk[b] != x ? __builtin_popcountll(row[b]) : 0);
        }
        return R_;
    };
    auto rcount = [&](int k, int64_t x) -> int64_t { return dense_b && x != r ? rp(x).pre[k][nb] : (int64_t)rl(x).t[k].size(); };
    auto rslot = [&](int64_t k, int64_t x, int64_t t) -> int64_t {    // dense, x != r, t not owned by x
        const int64_t b = dper.q(t), i = t - b * per;
        const uint64_t m = DM[k][(size_t)x * nb + b];
        if (!(m >> i & 1)) return -1;
        return nown[x] + rp(x).pre[k][b] + __builtin_popcountll(m & ((1ull << i) - 1));
    };
    int64_t NQS, NKS;
    if (g_bsym[0] >= 0) { NQS = g_bsym[0]; NKS = g_bsym[1]; }
    else {                                                              // global planning: every PE's list in one pass over
        if (!dense_b) {                                                 // (dense: rcount from the dense rows, the same)
            for (int k = 0; k < 2; ++k)                                 // the tiles (ascending: no sort)
                for (int64_t t = 0; t < nv; ++t) for_con(k, t, [&](int64_t c) { if (c != owner_t[t]) rls[c].t[k].push_back((int32_t)t); });
            for (int64_t x = 0; x < P; ++x) rls[x].done = true;
        }
        NQS = 0; NKS = 0;
        for (int64_t x = 0; x < P; ++x) {
            NQS = std::max<int64_t>(NQS, nown[x] + rcount(0, x)); NKS = std::max<int64_t>(NKS, nown[x] + rcount(1, x));
        }
    }
    g_bsym_me[0] = nown[r] + (int64_t)rl(r).t[0].size(); g_bsym_me[1] = nown[r] + (int64_t)rl(r).t[1].size();
    TP("b:s:symref");
    // per-PE layout: every PE lays its region out by ITS OWN slot / landing counts (the global planning
    // mode knows every PE's); senders address a destination by the destination's layout. The region every PE allocates
    // = the busiest PE's total instead of the sum of per-kind maxima. Per-rank planning (injected sizes) keeps the
    // global-maximum layout.
    constexpr bool perpe_env = true;
    const bool perpe = perpe_env && g_bsym[0] < 0 && g_bsym[2] < 0;
    std::vector<int64_t> cQ(P, NQS), cK(P, NKS), cDQ(P, 0), cDK(P, 0);
    if (perpe) {
        for (int64_t x = 0; x < P; ++x) { cQ[x] = nown[x] + rcount(0, x); cK[x] = nown[x] + rcount(1, x); }
        NQS = cQ[r]; NKS = cK[r];
    }
    std::vector<int32_t> myslot[2] = {std::vector<int32_t>(nv, -1), std::vector<int32_t>(nv, -1)};
    std::vector<int32_t> QS_r, KS_r;                                    // this PE: slot -> tile
    for (int k = 0; k < 2; ++k) {
        auto &S = k ? KS_r : QS_r;
        for (int64_t b = 0; b < nb; ++b)
            if (owner_blk[b] == r) for (int64_t t = b * per; t < (b + 1) * per; ++t) { myslot[k][t] = (int32_t)S.size(); S.push_back((int32_t)t); }
        for (int32_t t : rl(r).t[k]) { myslot[k][t] = (int32_t)S.size(); S.push_back(t); }
    }
    TP("B:slots");
    // ---- gateways: the schedule's relay forwards (bwd kind: transposed kind 0 = KV -> 1, 1 = Q -> 0) ------------- //
    // per tile, the (node, gateway) of the relay forwards covering it (no node x tiles array; the last message wins)
    std::vector<int32_t> gw_ptr[2], gw_node[2], gw_pe[2], gw_tile[2];
    for (int k = 0; k < 2; ++k) {
        gw_ptr[k].assign(nv + 1, 0);
        for (int64_t m = 0; m < nm; ++m)
            if (O.mtype[m] == 2 && 1 - O.mk[m] == k) for (int64_t t = O.mb0[m]; t < O.mb0[m] + O.mcnt[m]; ++t) ++gw_ptr[k][t + 1];
        for (int64_t t = 0; t < nv; ++t) gw_ptr[k][t + 1] += gw_ptr[k][t];
        gw_node[k].resize(gw_ptr[k][nv]); gw_pe[k].resize(gw_ptr[k][nv]); gw_tile[k].resize(gw_ptr[k][nv]);
        std::vector<int32_t> f(gw_ptr[k].begin(), gw_ptr[k].end() - 1);
        for (int64_t m = 0; m < nm; ++m)
            if (O.mtype[m] == 2 && 1 - O.mk[m] == k)
                for (int64_t t = O.mb0[m]; t < O.mb0[m] + O.mcnt[m]; ++t) {
                    const int32_t x = f[t]++; gw_node[k][x] = (int32_t)node(O.md[m]); gw_pe[k][x] = (int32_t)O.ms[m]; gw_tile[k][x] = (int32_t)t;
                }
    }
    TP("b:g:gwptr");
    auto gw_of = [&](int64_t k, int64_t nd, int64_t t) -> int64_t {   // gateway of node nd for tile t, or -1
        int64_t g = -1;
        for (int32_t i = gw_ptr[k][t]; i < gw_ptr[k][t + 1]; ++i) if (gw_node[k][i] == nd) g = gw_pe[k][i];
        return g;
    };
    auto dest = [&](int64_t k, int64_t t, int64_t c) -> int64_t {      // where contributor c's partial of tile t lands
        const int64_t g = gw_of(k, node(c), t);
        return (g >= 0 && g != c) ? g : own(t);
    };
    // ---- landings: at acceptor a, (tile ascending, contributor ascending) ------------------------------------------ //
    // per acceptor on demand: its tiles (owned, and those it is a node gateway of) in order, every contributor whose
    // partial lands there (the node's gateway if it has one and is not the contributor, else the owner)
    std::vector<int32_t> gwt_ptr[2], gwt[2];                            // per PE (CSR): tiles it is a node gateway of,
    if (g_bsym[2] >= 0) {                                               // ascending, unique (only the per-rank path)
        for (int k = 0; k < 2; ++k) {                                   // (CSR)
            gwt_ptr[k].assign(P + 1, 0);
            for (int32_t i = 0; i < gw_ptr[k][nv]; ++i) ++gwt_ptr[k][gw_pe[k][i] + 1];
            for (int64_t x = 0; x < P; ++x) gwt_ptr[k][x + 1] += gwt_ptr[k][x];
            gwt[k].resize(gwt_ptr[k][P]);
            std::vector<int32_t> f(gwt_ptr[k].begin(), gwt_ptr[k].end() - 1);
            for (int32_t i = 0; i < gw_ptr[k][nv]; ++i) gwt[k][f[gw_pe[k][i]]++] = gw_tile[k][i];   // (entries: tiles ascending)
            int32_t w = 0;                                              // unique per PE (adjacent repeats), compacted
            for (int64_t x = 0; x < P; ++x) {
                const int32_t a = gwt_ptr[k][x], e = gwt_ptr[k][x + 1];
                gwt_ptr[k][x] = w;
                for (int32_t i = a; i < e; ++i) if (i == a || gwt[k][i] != gwt[k][i - 1]) gwt[k][w++] = gwt[k][i];
            }
            gwt_ptr[k][P] = w; gwt[k].resize(w);
        }
    }
    std::vector<int32_t> gwn(NN, -1);                                   // a tile's gateway per node (last wins)
    // this PE's tiles: gateways of its NIC messages, per tile in message order (CSR over this PE's OWNED
    // tiles: a NIC message's sender owns its tiles; other tiles have none)
    std::vector<int32_t> pg_ptr[2], pg[2];
    bool pg_own = true;                                                 // (else per tile of the sequence: never seen)
    auto pgk = [&](int64_t t) -> int64_t { return pg_own ? ownpos_t(t) : t; };
    if (g_bsym[2] >= 0) {
        std::vector<int64_t> nicm;
        for (int64_t m = 0; m < nm; ++m) if (O.mtype[m] == 0 && O.ms[m] == r) nicm.push_back(m);
        for (int64_t m : nicm) for (int64_t t = O.mb0[m]; t < O.mb0[m] + O.mcnt[m]; ++t) pg_own &= owner_t[t] == r;
        const int64_t no = pg_own ? nown[r] : nv;
        for (int k = 0; k < 2; ++k) pg_ptr[k].assign(no + 1, 0);
        for (int64_t m : nicm) for (int64_t t = O.mb0[m]; t < O.mb0[m] + O.mcnt[m]; ++t) ++pg_ptr[1 - O.mk[m]][pgk(t) + 1];
        for (int k = 0; k < 2; ++k) {
            for (int64_t x = 0; x < no; ++x) pg_ptr[k][x + 1] += pg_ptr[k][x];
            pg[k].resize(pg_ptr[k][no]);
        }
        std::vector<int32_t> f[2] = {std::vector<int32_t>(pg_ptr[0].begin(), pg_ptr[0].end() - 1),
                                     std::vector<int32_t>(pg_ptr[1].begin(), pg_ptr[1].end() - 1)};
        for (int64_t m : nicm) for (int64_t t = O.mb0[m]; t < O.mb0[m] + O.mcnt[m]; ++t) pg[1 - O.mk[m]][f[1 - O.mk[m]][pgk(t)]++] = (int32_t)O.md[m];
    }
    auto pg_range = [&](int k, int64_t t, int32_t &a, int32_t &e) {     // (empty: no NIC message of this PE has t)
        if (pg_ptr[k].empty() || (pg_own && owner_t[t] != r)) { a = e = 0; return; }
        const int64_t x = pgk(t); a = pg_ptr[k][x]; e = pg_ptr[k][x + 1];
    };
    TP("b:g:gwt/pargw");
    // contributors of a tile among one node's GPUs (ascending): from those GPUs' pairs only -- a gateway's landings
    // come from its node, and the full lists would cost a pass over every pair of the plan
    int64_t ncn = -1;
    std::vector<std::vector<std::pair<int32_t, uint64_t>>> ncb[2];       // per block: (GPU, OR of masks), GPUs ascending
    auto for_node_con = [&](int k, int64_t nd, int64_t t, auto f) {
        if (ncn != nd) {
            ncn = nd;
            for (int kk = 0; kk < 2; ++kk) { ncb[kk].assign(nb, {}); }
            for (int64_t x = nd * gpn; x < std::min(P, (nd + 1) * gpn); ++x)
                for (int32_t i = bxp[2][x]; i < bxp[2][x + 1]; ++i) {
                    const int64_t p = bxi[2][i];
                    for (int kk = 0; kk < 2; ++kk) {
                        auto &v = ncb[kk][kk ? bk[p] : bq[p]];
                        const uint64_t m = kk ? pcol[p] : prow[p];
                        if (!v.empty() && v.back().first == (int32_t)x) v.back().second |= m; else v.push_back({(int32_t)x, m});
                    }
                }
        }
        const int64_t b_ = dper.q(t), a_ = t - b_ * per;
        for (auto &e : ncb[k][b_]) if (e.second >> a_ & 1) f((int64_t)e.first);
    };
    struct AL { bool done = false; std::vector<int32_t> t, c; };
    std::vector<AL> als[2] = {std::vector<AL>(P), std::vector<AL>(P)};
    auto al = [&](int k, int64_t a) -> const AL & {
        AL &A = als[k][a];
        if (A.done) return A;
        A.done = true;
        std::vector<int32_t> ts;
        for (int64_t b : owned_b[a]) for (int64_t t = b * per; t < (b + 1) * per; ++t) ts.push_back((int32_t)t);
        if (!gwt_ptr[k].empty()) ts.insert(ts.end(), gwt[k].begin() + gwt_ptr[k][a], gwt[k].begin() + gwt_ptr[k][a + 1]);
        std::sort(ts.begin(), ts.end()); ts.erase(std::unique(ts.begin(), ts.end()), ts.end());
        for (int32_t t : ts) {
            const int64_t o = own(t);
            for (int32_t i = gw_ptr[k][t]; i < gw_ptr[k][t + 1]; ++i) gwn[gw_node[k][i]] = gw_pe[k][i];
            if (o == a) {                                               // own tile: its NIC messages name every node's
                int32_t ja, je; pg_range(k, t, ja, je);                 // gateway (a node with a single reader has no
                    for (int32_t j = ja; j < je; ++j) {                     // forward; that reader lands at the
                        const int32_t g = pg[k][j];                     // owner either way)
                        if (gwn[node(g)] < 0) gwn[node(g)] = g;
                    }
            }
            auto visit = [&](int64_t c) {
                if (c == o) return;
                const int64_t g = gwn[node(c)];
                if (((g >= 0 && g != c) ? g : o) == a) { A.t.push_back(t); A.c.push_back((int32_t)c); }
            };
            if (o == a) for_con(k, t, visit);
            else for_node_con(k, node(a), t, visit);                    // a gateway: only its node's contributors land
            for (int32_t i = gw_ptr[k][t]; i < gw_ptr[k][t + 1]; ++i) gwn[gw_node[k][i]] = -1;
            if (o == a) { int32_t ja, je; pg_range(k, t, ja, je); for (int32_t j = ja; j < je; ++j) gwn[node(pg[k][j])] = -1; }
        }
        return A;
    };
    std::vector<std::pair<int32_t, int32_t>> my_out[2];                 // this PE's partial of tile t -> (acceptor, slot)
    my_out[0].assign(QS_r.size(), {-1, -1}); my_out[1].assign(KS_r.size(), {-1, -1});   // (per slot: tiles it computes)
    std::vector<std::array<int32_t, 3>> my_in[2];                       // landings here: (tile, contributor, slot)
    TP("b:g:alloc");
    int64_t NDQL, NDKL;
    if (g_bsym[2] < 0) {                                                // global planning (all PEs known): one pass over the
        std::vector<int64_t> nland(P);                                  // tiles numbers every acceptor's landings
        for (int k = 0; k < 2; ++k) {
            std::fill(nland.begin(), nland.end(), 0);
            for (int64_t t = 0; t < nv; ++t) {
                const int64_t o = own(t);
                for (int32_t i = gw_ptr[k][t]; i < gw_ptr[k][t + 1]; ++i) gwn[gw_node[k][i]] = gw_pe[k][i];
                for_con(k, t, [&](int64_t c) {
                    if (c == o) return;
                    const int64_t g = gwn[node(c)];
                    const int64_t a_ = (g >= 0 && g != c) ? g : o, l = nland[a_]++;
                    if (c == r) my_out[k][myslot[k][t]] = {(int32_t)a_, (int32_t)l};
                    if (a_ == r) my_in[k].push_back({(int32_t)t, (int32_t)c, (int32_t)l});
                });
                for (int32_t i = gw_ptr[k][t]; i < gw_ptr[k][t + 1]; ++i) gwn[gw_node[k][i]] = -1;
            }
            (k ? NDKL : NDQL) = std::max<int64_t>(1, *std::max_element(nland.begin(), nland.end()));
            for (int64_t x = 0; x < P; ++x) (k ? cDK : cDQ)[x] = std::max<int64_t>(1, nland[x]);
        }
        if (perpe) { NDQL = cDQ[r]; NDKL = cDK[r]; }
        g_bsym_me[2] = std::max<int64_t>(1, (int64_t)my_in[0].size()); g_bsym_me[3] = std::max<int64_t>(1, (int64_t)my_in[1].size());
    } else {                                                            // per-rank planning: the acceptors it talks to
        NDQL = g_bsym[2]; NDKL = g_bsym[3];
        for (int k = 0; k < 2; ++k) {
            g_bin[k].clear();
            for (int32_t t : rl(r).t[k]) {                              // tiles this PE computes and does not own
                if (g_bsym_only) break;
                const auto it = g_bout.find(k * (1ll << 32) + t);       // the acceptor's slot (exchange), or this
                if (it != g_bout.end()) { my_out[k][myslot[k][t]] = it->second; continue; }   // PE's view of its list
                if (g_bout_dummy) { my_out[k][myslot[k][t]] = {0, 0}; continue; }
                const int64_t a_ = dest(k, t, r);
                const AL &A = al(k, a_);
                int64_t l = std::lower_bound(A.t.begin(), A.t.end(), t) - A.t.begin();
                while (A.c[l] != r) ++l;
                my_out[k][myslot[k][t]] = {(int32_t)a_, (int32_t)l};
            }
            const AL &A = al(k, r);
            for (size_t l = 0; l < A.t.size(); ++l) { g_bin[k].push_back(A.t[l]); g_bin[k].push_back(A.c[l]); }
            for (size_t l = 0; l < A.t.size(); ++l) my_in[k].push_back({A.t[l], A.c[l], (int32_t)l});
            if ((int64_t)A.t.size() > (k ? NDKL : NDQL)) {             // injected sizes do not match this rank's view
                fprintf(stderr, "bwd_tables: PE %ld kind %d lands %zu > injected %ld (incomplete gateway info?)\n",
                        (long)r, k, A.t.size(), (long)(k ? NDKL : NDQL));
                return -1;
            }
        }
        g_bsym_me[2] = std::max<int64_t>(1, al(0, r).t.size()); g_bsym_me[3] = std::max<int64_t>(1, al(1, r).t.size());
    }
    if (g_bsym_only) return 0;
    // region layout (dex/scheduler/bwd.py region_layout)
    const int64_t qb = 2 * (T * h * d * 2) + 2 * (h * T * 4), kvb = 2 * (T * hk * d * 2);
    const int64_t dqa = T * h * d * 4, dkva = 2 * T * hk * d * 4;
    if (!perpe) for (int64_t x = 0; x < P; ++x) { cQ[x] = NQS; cK[x] = NKS; cDQ[x] = NDQL; cDK[x] = NDKL; }
    std::vector<std::array<int64_t, 7>> LO(P);
    g_bregion = 0;
    for (int64_t x = 0; x < P; ++x) {
        const int64_t sz[7] = {8 * (cQ[x] + cK[x] + cDQ[x] + cDK[x]), cQ[x] * qb, cK[x] * kvb, cQ[x] * dqa, cK[x] * dkva,
                               cDQ[x] * dqa, cDK[x] * dkva};
        int64_t o = 0;
        for (int i = 0; i < 7; ++i) { LO[x][i] = o; o = (o + sz[i] + 255) / 256 * 256; }
        g_bregion = std::max(g_bregion, o);
    }
    const auto &lo = LO[r];
    auto sig_q = [&](int64_t s) { return s; };
    auto sig_k = [&](int64_t s) { return NQS + s; };
    auto sig_dql = [&](int64_t l) { return NQS + NKS + l; };
    auto sig_dkl = [&](int64_t l) { return NQS + NKS + NDQL + l; };
    TP("B:gw/lands");
    // ---- input runs: a message cut where the sender's or the receiver's slots stop being consecutive (a forward also
    // where its parent's runs are cut); emit(first tile, count, signal at the receiver)
    auto sl_of = [&](int64_t k, int64_t x, int64_t t) -> int64_t {    // slot of tile t on GPU x (-1: none)
        if (x == r) return myslot[k][t];
        if (owner_t[t] == x) return ownpos_t(t);
        if (dense_b) return rslot(k, x, t);
        const auto &v = rl(x).t[k];
        auto it = std::lower_bound(v.begin(), v.end(), (int32_t)t);
        return (it != v.end() && *it == t) ? nown[x] + (it - v.begin()) : -1;
    };
    // slots of GPU x along ascending tiles (a message's tiles): a cursor in x's remote list instead of a search per tile
    struct SlotWalk {                                                   // (x's remote list fetched on first use: a
        int64_t k, x, t0; const int32_t *it, *e, *b;                    // message's sender usually owns its tiles)
    };
    auto walk_begin = [&](int64_t k, int64_t x, int64_t t0) { return SlotWalk{k, x, t0, nullptr, nullptr, nullptr}; };
    auto walk = [&](SlotWalk &w, int64_t t) -> int64_t {              // t ascending across calls
        if (w.x == r) return myslot[w.k][t];
        if (owner_t[t] == w.x) return ownpos_t(t);
        if (dense_b) return rslot(w.k, w.x, t);
        if (!w.b) {
            const auto &v = rl(w.x).t[w.k];
            w.b = v.data(); w.e = v.data() + v.size(); w.it = std::lower_bound(w.b, w.e, (int32_t)w.t0);
        }
        while (w.it != w.e && *w.it < t) ++w.it;
        return (w.it != w.e && *w.it == t) ? nown[w.x] + (w.it - w.b) : -1;
    };
    auto runs = [&](int64_t m, auto emit) {
        const int64_t k = 1 - O.mk[m], s = O.ms[m], rc = O.md[m], ty = O.mtype[m], b0 = O.mb0[m], c = O.mcnt[m];
        const int64_t ps = ty == 2 ? O.ms[O.mpar[m]] : -1;
        SlotWalk wr = walk_begin(k, rc, b0), ws = walk_begin(k, s, b0), wp = walk_begin(k, ps < 0 ? r : ps, b0);
        int64_t a = b0, lr = walk(wr, b0), ls = walk(ws, b0), lp = ty == 2 ? walk(wp, b0) : 0;
        for (int64_t b = b0 + 1; b <= b0 + c; ++b) {
            bool cut = b == b0 + c;
            int64_t xr = lr, xs = ls, xp = lp;
            if (!cut) {
                xr = walk(wr, b); xs = walk(ws, b);
                cut = xr != lr + 1 || xs != ls + 1;
                if (ty == 2) { xp = walk(wp, b); if (!cut) cut = xp != lp + 1; }
            }
            if (!cut && g_chunk_tiles[k] > 0) cut = b % g_chunk_tiles[k] == 0;     // transfer chunk (absolute)
            if (cut) {
                emit(a, b - a, k == 0 ? lr : cQ[rc] + lr);                     // (lr = receiver slot of tile b - 1;
                                                                                // the receiver's layout)
                a = b;
            }
            lr = xr; ls = xs; lp = xp;
        }
    };
    // this PE as a receiver: the signal (run end) each remote input tile waits on, and the run's expected landing
    // stored per SLOT of this PE (every tile it receives is one it computes: a gateway is one of the
    // readers), not per tile of the sequence; a received tile without a slot (never seen) falls back to per tile.
    // Reads of a tile without an entry give the defaults (-1, 0, -1) either way.
    bool dsl = true;
    for (int64_t m = 0; m < nm && dsl; ++m)
        if (O.md[m] == r) for (int64_t t = O.mb0[m]; t < O.mb0[m] + O.mcnt[m]; ++t) if (myslot[1 - O.mk[m]][t] < 0) { dsl = false; break; }
    auto dsz = [&](int k) -> size_t { return dsl ? (k ? KS_r.size() : QS_r.size()) : (size_t)nv; };
    std::vector<int64_t> dep_[2] = {std::vector<int64_t>(dsz(0), -1), std::vector<int64_t>(dsz(1), -1)};
    std::vector<double> dland_[2] = {std::vector<double>(dsz(0), 0.0), std::vector<double>(dsz(1), 0.0)};
    std::vector<int32_t> dlrank_[2] = {std::vector<int32_t>(dsz(0), -1), std::vector<int32_t>(dsz(1), -1)};   // rank of dland among
    auto dx_ = [&](int64_t k, int64_t t) -> int64_t { return dsl ? myslot[k][t] : t; };
    auto dep = [&](int64_t k, int64_t t) -> int64_t { const int64_t i = dx_(k, t); return i < 0 ? -1 : dep_[k][i]; };
    auto dland = [&](int64_t k, int64_t t) -> double { const int64_t i = dx_(k, t); return i < 0 ? 0.0 : dland_[k][i]; };
    auto dlrank = [&](int64_t k, int64_t t) -> int32_t { const int64_t i = dx_(k, t); return i < 0 ? -1 : dlrank_[k][i]; };
    std::vector<int32_t> mrank(nm, -1);                                 // the distinct landings of the messages INTO
    {                                                                   // this PE (only those are ever compared)
        std::vector<double> lv;
        for (int64_t m = 0; m < nm; ++m) if (O.md[m] == r) lv.push_back(O.land[m]);
        std::sort(lv.begin(), lv.end()); lv.erase(std::unique(lv.begin(), lv.end()), lv.end());
        for (int64_t m = 0; m < nm; ++m)
            if (O.md[m] == r) mrank[m] = (int32_t)(std::lower_bound(lv.begin(), lv.end(), O.land[m]) - lv.begin());
    }
    TP("b:d:mrank");
    for (int64_t m = 0; m < nm; ++m)
        if (O.md[m] == r) {
            const int64_t k = 1 - O.mk[m];
            runs(m, [&](int64_t a, int64_t c, int64_t sg) {
                for (int64_t t = a; t < a + c; ++t) { const int64_t i = dx_(k, t); dep_[k][i] = sg; dland_[k][i] = O.land[m]; dlrank_[k][i] = mrank[m]; }
            });
        }
    TP("B:deps");
    // ---- outgoing inputs (in the schedule's send order) and relay forwards ------------------------------------------ //
    std::vector<int64_t> morder;
    for (int64_t m = 0; m < nm; ++m) if (O.ms[m] == r) morder.push_back(m);
    if (ablate & 2) std::sort(morder.begin(), morder.end(), [&](int64_t a, int64_t b) {
        const int64_t ka = 1 - O.mk[a], kb = 1 - O.mk[b];              // Q before KV, destination, tile
        if (ka != kb) return ka < kb;
        if (O.md[a] != O.md[b]) return O.md[a] < O.md[b];
        return O.mb0[a] < O.mb0[b]; });
    else std::sort(morder.begin(), morder.end(), [&](int64_t a, int64_t b) { return O.pos[a] < O.pos[b]; });
    TP("b:i:morder");
    struct M5 { int64_t src, dst, bytes, pe, sig; };
    std::vector<M5> start;                                              // pushes of owned tiles
    std::map<int64_t, std::vector<M5>> fwd;                             // arrival signal here -> forwards
    std::map<int64_t, int64_t> fwd_key;                                 // arrival signal -> its first forward's pos
    int64_t n_pushq = 0, n_pushkv = 0;
    for (int64_t m : morder) {
        const int64_t k = 1 - O.mk[m], rc = O.md[m], ty = O.mtype[m];
        const int64_t cls = k == 0 ? 1 : 2, bb = k == 0 ? qb : kvb;
        runs(m, [&](int64_t a, int64_t c, int64_t sg) {
            const M5 x{lo[cls] + sl_of(k, r, a) * bb, LO[rc][cls] + sl_of(k, rc, a) * bb, c * bb, rc, sg};
            if (ty == 2) {
                const int64_t at = dep(k, a + c - 1);                  // the parent run's signal here
                fwd[at].push_back(x);
                if (!fwd_key.count(at)) fwd_key[at] = O.pos[m];
            } else {
                start.push_back(x); (k == 0 ? n_pushq : n_pushkv) += 1;
            }
        });
    }
    TP("B:inputs");
    // ---- tasks ------------------------------------------------------------------------------------------------------ //
    std::vector<int64_t> cols;                                          // this PE's KV tiles in the compute order
    if (!rect) {                                                        // (rect: no per-tile tasks)
        std::vector<char> seen(nv, 0);
        for (int64_t i = O.row_ptr[r]; i < O.row_ptr[r + 1]; ++i) {
            const int64_t blk = O.row_blk[i];
            for (int64_t j = blk * per; j < (blk + 1) * per; ++j)
                if (col_n(j) && !seen[j]) { seen[j] = 1; cols.push_back(j); }
        }
        for (int64_t j = 0; j < nv; ++j) if (col_n(j) && !seen[j]) cols.push_back(j);   // (defensive)
        if (ablate & 1) std::sort(cols.begin(), cols.end());
    }
    // chunks (Q tiles grouped by the run that delivers them, <= MAX_QS each) per column, computed once: their inputs,
    // their Q slots and their item pattern per half (q slot, m block, causal) -- the KV head only adds q-head bits
    TP("B:t:cols");
    struct Chunk { int64_t col, ks, d0, d1, q0, q1, p0[2], p1[2]; };
    std::vector<Chunk> chunks;
    std::vector<int64_t> cdeps;                                         // CSR over chunks: sorted distinct input signals
    std::vector<int32_t> cqs, cpat;                                     // sorted distinct Q slots; item patterns
    std::vector<int64_t> chunk_ptr{0};
    std::vector<int64_t> dd; std::vector<int32_t> qs;
    std::vector<std::pair<int32_t, int8_t>> rows;                       // one column's (Q tile, kind), grouped
    cpat.reserve(4 * (size_t)(rect ? 0 : cr_ptr[nv]) + 64);
    for (int64_t j : cols) {
        // groups = the Q tiles of one OWNER (a chunk waits for the owner's runs; grouping by run would give more
        // tasks, each with its own cost): this PE's own first, then by the expected landing of the
        // owner's last run here, then owner id; descending tile inside (FA2 walks m_block down)
        std::vector<double> ol(P, -1.0);
        const int32_t c0_ = cr_ptr[j], c1_ = cr_ptr[j + 1];
        for (int32_t x = c0_; x < c1_; ++x) { const int64_t o = own(cr_i[x]); if (o != r) ol[o] = std::max(ol[o], dland(0, cr_i[x])); }
        // owner rank (this PE, then landing, then id); a counting sort by rank keeps the rows ascending, so each
        // group is filled from its end: descending tile inside the group
        std::vector<int64_t> ord(P), rk(P); int64_t no = 0;
        for (int64_t x = 0; x < P; ++x) if (x == r || ol[x] >= 0.0) ord[no++] = x;
        std::sort(ord.begin(), ord.begin() + no, [&](int64_t x, int64_t y) {
            const double kx = x == r ? -2.0 : ((ablate & 4) ? (double)x : ol[x]);
            const double ky = y == r ? -2.0 : ((ablate & 4) ? (double)y : ol[y]);
            return kx != ky ? kx < ky : x < y; });
        for (int64_t x = 0; x < no; ++x) rk[ord[x]] = x;
        std::vector<int32_t> gend(no + 1, 0);
        for (int32_t x = c0_; x < c1_; ++x) ++gend[rk[own(cr_i[x])] + 1];
        for (int64_t g = 0; g < no; ++g) gend[g + 1] += gend[g];
        rows.resize(c1_ - c0_);
        for (int32_t x = c0_; x < c1_; ++x)                             // ascending in, so fill each group backwards
            rows[--gend[rk[own(cr_i[x])] + 1]] = {cr_i[x], cr_kd[x]};
        const int64_t ks = sl_of(1, r, j), kdep = own(j) != r ? dep(1, j) : -1;
        for (size_t a = 0; a < rows.size();) {
            size_t e = a;
            while (e < rows.size() && own(rows[e].first) == own(rows[a].first)) ++e;
            for (size_t c0 = a; c0 < e; c0 += MAX_QS) {
                const size_t c1 = std::min(e, c0 + (size_t)MAX_QS);
                Chunk ch{j, ks, (int64_t)cdeps.size(), 0, (int64_t)cqs.size(), 0, {0, 0}, {0, 0}};
                dd.clear(); qs.clear();
                if (kdep >= 0) dd.push_back(kdep);
                bool asc = true;                                        // one owner: slots follow the tiles, so the
                for (size_t x = c1; x-- > c0;) {                        // descending rows give ascending slots
                    const int64_t i = rows[x].first;
                    if (own(i) != r) dd.push_back(dep(0, i));
                    const int32_t q = (int32_t)sl_of(0, r, i);
                    asc &= qs.empty() || q > qs.back();
                    qs.push_back(q);
                }
                std::sort(dd.begin(), dd.end()); dd.erase(std::unique(dd.begin(), dd.end()), dd.end());
                if (!asc) { std::sort(qs.begin(), qs.end()); qs.erase(std::unique(qs.begin(), qs.end()), qs.end()); }
                cdeps.insert(cdeps.end(), dd.begin(), dd.end()); ch.d1 = (int64_t)cdeps.size();
                cqs.insert(cqs.end(), qs.begin(), qs.end()); ch.q1 = (int64_t)cqs.size();
                for (int hn = 0; hn < 2; ++hn) {
                    ch.p0[hn] = (int64_t)cpat.size();
                    for (size_t x = c0; x < c1; ++x) {
                        const int64_t i = rows[x].first, sq_ = sl_of(0, r, i);
                        for (int mb = 1; mb >= 0; --mb) {
                            int causal = 0;
                            if (rows[x].second == 2) { if (mb < hn) continue; causal = mb == hn; }
                            cpat.push_back((int32_t)((sq_ << 9) | (mb << 8) | causal));
                        }
                    }
                    ch.p1[hn] = (int64_t)cpat.size();
                }
                chunks.push_back(ch);
            }
            a = e;
        }
        chunk_ptr.push_back((int64_t)chunks.size());
    }
    TP("B:t:chunks");
    // tasks, KV-head-major: per head, the columns in the compute order, their chunks, the two halves (twins)
    const int64_t nch = (int64_t)chunks.size();
    int64_t n_items = 0, n_tqs = 0;
    for (const Chunk &ch : chunks) { n_items += (ch.p1[0] - ch.p0[0] + ch.p1[1] - ch.p0[1]) * rg; n_tqs += 2 * (ch.q1 - ch.q0); }
    n_items *= hk; n_tqs *= hk;
    const int64_t nt = 2 * nch * hk;
    std::vector<int32_t> task_kv(nt), task_ptr(nt + 1), tqs_ptr(nt + 1), tqs(n_tqs);
    auto &items = g_bwd.items;
    items.resize(n_items);
    std::vector<int32_t> tchunk(nt);
    {
        int64_t t = 0, ip = 0, qp = 0;
        task_ptr[0] = 0; tqs_ptr[0] = 0;
        for (int64_t g = 0; g < hk; ++g)
            for (int64_t q = 0; q < nch; ++q) {
                const Chunk &ch = chunks[q];
                for (int hn = 0; hn < 2; ++hn, ++t) {
                    ip += (ch.p1[hn] - ch.p0[hn]) * rg;
                    task_ptr[t + 1] = (int32_t)ip;
                    task_kv[t] = (int32_t)((ch.ks << 9) | (hn << 8) | g);
                    std::memcpy(&tqs[qp], &cqs[ch.q0], (ch.q1 - ch.q0) * 4); qp += ch.q1 - ch.q0;
                    tqs_ptr[t + 1] = (int32_t)qp;
                    tchunk[t] = (int32_t)q;
                }
            }
        // the items (q-head bits on the chunk's pattern): the bulk of the output; filled by up to 8 threads above 1M
        // items (ranks of one node share its CPU cores)
        auto fill = [&](int64_t ta, int64_t tb) {
            for (int64_t t_ = ta; t_ < tb; ++t_) {
                const Chunk &ch = chunks[tchunk[t_]];
                const int64_t g = task_kv[t_] & 0xff, hn = (task_kv[t_] >> 8) & 1;
                int64_t ip_ = task_ptr[t_];
                for (int64_t x = ch.p0[hn]; x < ch.p1[hn]; ++x)
                    for (int64_t hq = g * rg; hq < (g + 1) * rg; ++hq) items[ip_++] = cpat[x] | (int32_t)(hq << 1);
            }
        };
        const int64_t nth = std::min<int64_t>(8, std::max<int64_t>(1, n_items >> 20));
        if (nth == 1) fill(0, nt);
        else {
            std::vector<std::thread> th;
            int64_t ta = 0;
            for (int64_t k_ = 1; k_ <= nth; ++k_) {                    // equal item shares, cut at task ends
                const int64_t tb = k_ == nth ? nt
                    : std::upper_bound(task_ptr.begin(), task_ptr.begin() + nt + 1, (int32_t)(n_items * k_ / nth)) - task_ptr.begin() - 1;
                if (tb > ta) th.emplace_back(fill, ta, tb);
                ta = std::max(ta, tb);
            }
            for (auto &x : th) x.join();
        }
    }
    TP("B:t:expand");
    auto ndeps_of = [&](int64_t t) { const Chunk &ch = chunks[tchunk[t]]; return ch.d1 - ch.d0; };
    TP("B:t:lambdas");
    std::vector<int32_t> qs_need(NQS, 0), ks_need(NKS, 0);
    std::vector<int64_t> qlast(NQS, -1), klast(NKS, -1);
    for (int64_t t = 0; t < nt; ++t) {
        for (int32_t q = tqs_ptr[t]; q < tqs_ptr[t + 1]; ++q) { ++qs_need[tqs[q]]; qlast[tqs[q]] = t; }
        ++ks_need[task_kv[t] >> 9]; klast[task_kv[t] >> 9] = t;
    }
    TP("B:tasks");
    // ---- rectangles (rect = 1) ---------------------------------------------------------------------------------- //
    // rectangle = (KV plan block, owner of the Q rows) on this PE, in the schedule's compute order of KV blocks and,
    // inside one, this PE's own rows first, then by the expected landing of the owner's Q here, then owner id; its pairs
    // by row block descending (FA2 walks m_block down). Group = (column, <= 64 consecutive rows of the rectangle's
    // descending row order); inputs = the Q runs of the column's rows in the rectangle + the column's KV run (
    // rect_dep = 1, unused: the whole rectangle's Q runs as inputs). Scheduler rows = groups (one
    // group each, as the per-tile tasks' twin pairs), head-major: g = head * G1 + g1. Completion counts per group:
    // qs_need = KV heads x the (row, column) tiles of the row here, ks_need = KV heads x the column's groups.
    std::vector<int32_t> R_g_rect, R_g_col, R_g_k0, R_g_cnt, R_g_ks, R_rect_pp{0}, R_rect_pair, R_rect_bk, R_pair_bq,
        R_rg_ptr{0}, R_qslot, R_kslot;
    std::vector<uint64_t> R_bits, R_dbits, R_prow, R_pcol;
    std::vector<int64_t> R_rdep_ptr{0}, R_rdep, R_gk;                    // per group: Q inputs (CSR); KV input
    std::vector<int32_t> R_gland;                                       // per group: rank of its inputs' latest landing
    std::vector<int32_t> R_exp_order;                                   // materialisation order (kernel block 0)
    constexpr int rect_dep = 0;
    int64_t n_rect = 0, G1 = 0, R_arena = 0;
    if (rect) {
        R_qslot.resize(nv); R_kslot.resize(nv);
        std::memcpy(R_qslot.data(), myslot[0].data(), nv * 4); std::memcpy(R_kslot.data(), myslot[1].data(), nv * 4);   // (= sl_of(k, r, t))
        std::vector<std::vector<int64_t>> by_bk(nb);
        for (int64_t p : mine) by_bk[bk[p]].push_back(p);
        std::vector<int64_t> bks; std::vector<char> seen(nb, 0);
        for (int64_t i = O.row_ptr[r]; i < O.row_ptr[r + 1]; ++i) {
            const int64_t blk = O.row_blk[i];
            if (!by_bk[blk].empty() && !seen[blk]) { seen[blk] = 1; bks.push_back(blk); }
        }
        for (int64_t b = 0; b < nb; ++b) if (!by_bk[b].empty() && !seen[b]) bks.push_back(b);
        if (ablate & 1) std::sort(bks.begin(), bks.end());
        std::vector<int32_t> col_cnt(per);
        std::vector<int64_t> dd_;
        {                                                               // capacities (groups <= live columns + splits)
            size_t ncol = 0;
            for (int64_t p : mine) ncol += (size_t)__builtin_popcountll(pcol[p]);
            for (auto *v : {&R_g_rect, &R_g_col, &R_g_k0, &R_g_cnt, &R_g_ks, &R_gland}) v->reserve(ncol + 64);
            R_gk.reserve(ncol + 64); R_rdep_ptr.reserve(ncol + 65); R_rdep.reserve(2 * ncol + 64);
            R_bits.reserve(mine.size() * nw); R_dbits.reserve(mine.size() * nw);
            R_prow.reserve(mine.size()); R_pcol.reserve(mine.size()); R_pair_bq.reserve(mine.size()); R_rect_pair.reserve(mine.size());
        }
        std::vector<int64_t> po;
        std::vector<std::vector<int64_t>> cdep(per);                    // per column of the rectangle: Q inputs
        std::vector<int32_t> cland(per);                                //   and their latest landing (rank)
        std::vector<int64_t> touched, ord;
        std::vector<double> ol(P, -1.0);                                // (reset per KV block: only the touched owners)
        for (int64_t B_ : bks) {
            auto &ps = by_bk[B_];
            for (int64_t x : touched) ol[x] = -1.0;
            touched.clear();
            for (int64_t p : ps) {
                const int64_t o = owner_blk[bq[p]];
                if (o == r) continue;
                if (ol[o] < 0.0) touched.push_back(o);
                for (uint64_t m = prow[p]; m; m &= m - 1) ol[o] = std::max(ol[o], dland(0, bq[p] * per + __builtin_ctzll(m)));
            }
            ord.clear(); ord.push_back(r);
            for (int64_t x : touched) if (ol[x] >= 0.0) ord.push_back(x);
            std::sort(ord.begin(), ord.end()); ord.erase(std::unique(ord.begin(), ord.end()), ord.end());
            const int64_t no = (int64_t)ord.size();
            std::sort(ord.begin(), ord.end(), [&](int64_t x, int64_t y) {
                const double kx = x == r ? -2.0 : ((ablate & 4) ? (double)x : ol[x]);
                const double ky = y == r ? -2.0 : ((ablate & 4) ? (double)y : ol[y]);
                return kx != ky ? kx < ky : x < y; });
            for (int64_t oi = 0; oi < no; ++oi) {
                const int64_t o = ord[oi];
                po.clear();
                for (int64_t p : ps) if (owner_blk[bq[p]] == o) po.push_back(p);
                if (po.empty()) continue;
                std::sort(po.begin(), po.end(), [&](int64_t a, int64_t b) { return bq[a] > bq[b]; });
                const int64_t R = n_rect++;
                R_rect_bk.push_back((int32_t)B_);
                std::fill(col_cnt.begin(), col_cnt.end(), 0);
                for (auto &v : cdep) v.clear();                         // (capacity kept)
                std::fill(cland.begin(), cland.end(), -1);
                uint64_t cu = 0, cu_all = 0;
                for (int64_t p : po) cu_all |= pcol[p];                 // (CE order key: the rectangle's last group)
                dd_.clear();
                for (int64_t p : po) {
                    R_rect_pair.push_back((int32_t)R_pair_bq.size());
                    R_pair_bq.push_back((int32_t)bq[p]);
                    for (int64_t w = 0; w < nw; ++w) { R_bits.push_back(bits[p * nw + w]); R_dbits.push_back(dbits[p * nw + w]); }
                    R_prow.push_back(prow[p]); R_pcol.push_back(pcol[p]);
                    cu |= pcol[p];
                    // per row (not per tile): its column mask; consecutive rows of one input run are flushed into
                    // their columns' inputs together
                    const bool remote_rows = owner_blk[bq[p]] != r;     // (one owner per row block)
                    int64_t cur_d = -1; int32_t cur_l = -1; uint64_t cur_m = 0;
                    auto flush = [&]() {
                        for (uint64_t m = cur_m; m; m &= m - 1) {
                            const int64_t b = __builtin_ctzll(m);
                            if (!rect_dep) { auto &v = cdep[b]; if (v.empty() || v.back() != cur_d) v.push_back(cur_d); }
                            cland[b] = std::max(cland[b], cur_l);
                        }
                        cur_m = 0;
                    };
                    for (uint64_t m = prow[p]; m; m &= m - 1) {
                        const int64_t a = __builtin_ctzll(m), i = bq[p] * per + a, k0 = a * per, w = k0 >> 6, o = k0 & 63;
                        uint64_t rm = bits[p * nw + w] >> o;
                        if (o + per > 64 && w + 1 < nw) rm |= bits[p * nw + w + 1] << (64 - o);
                        if (per < 64) rm &= (1ull << per) - 1;
                        for (uint64_t x = rm; x; x &= x - 1) ++col_cnt[__builtin_ctzll(x)];
                        const int64_t sq_ = sl_of(0, r, i);
                        qs_need[sq_] += (int32_t)(HG * __builtin_popcountll(rm)); qlast[sq_] = G1 + __builtin_popcountll(cu_all);
                        if (remote_rows) {
                            dd_.push_back(dep(0, i));
                            if (dep(0, i) != cur_d) { flush(); cur_d = dep(0, i); cur_l = dlrank(0, i); }
                            cur_m |= rm;
                        }
                    }
                    flush();
                }
                R_rect_pp.push_back((int32_t)R_rect_pair.size());
                std::sort(dd_.begin(), dd_.end()); dd_.erase(std::unique(dd_.begin(), dd_.end()), dd_.end());
                for (uint64_t m = cu; m; m &= m - 1) {                   // (only live columns; 0 / 1 entry: sorted)
                    auto &v = cdep[__builtin_ctzll(m)];
                    if (v.size() > 1) { std::sort(v.begin(), v.end()); v.erase(std::unique(v.begin(), v.end()), v.end()); }
                }
                // group order inside the rectangle: GORDER=1 -> row-chunk major (every column's first <= 64 rows, then the
                // next 64, ...): concurrently running groups walk the same Q rows (FA2's lockstep m_block sweep) instead of
                // one column's successive row chunks (each concurrent group a different Q range: L2 misses grow with S)
                constexpr int gorder = 1;
                int64_t kmax = 0;
                for (uint64_t m = cu; m; m &= m - 1) kmax = std::max<int64_t>(kmax, col_cnt[__builtin_ctzll(m)]);
                for (int64_t kk = 0; kk < (gorder ? kmax : 1); kk += MAX_QS)
                for (uint64_t m = cu; m; m &= m - 1) {
                    const int64_t b = __builtin_ctzll(m), j = B_ * per + b, ks_ = sl_of(1, r, j);
                    for (int64_t k0 = gorder ? kk : 0; k0 < (gorder ? std::min<int64_t>(col_cnt[b], kk + 1) : col_cnt[b]); k0 += MAX_QS) {
                        ks_need[ks_] += (int32_t)HG; klast[ks_] = G1;
                        const int64_t c = std::min<int64_t>(MAX_QS, col_cnt[b] - k0);
                        R_g_rect.push_back((int32_t)R); R_g_col.push_back((int32_t)b); R_g_k0.push_back((int32_t)k0);
                        R_g_cnt.push_back((int32_t)c); R_g_ks.push_back((int32_t)ks_);
                        R_gk.push_back(own(j) != r ? dep(1, j) : -1);
                        R_gland.push_back(std::max(cland[b], own(j) != r ? dlrank(1, j) : -1));
                        const auto &qd = rect_dep ? dd_ : cdep[b];
                        R_rdep.insert(R_rdep.end(), qd.begin(), qd.end()); R_rdep_ptr.push_back((int64_t)R_rdep.size());
                        R_arena += 4 * c * rg;                          // both halves, <= 2 items per row and q head
                        ++G1;
                    }
                }
                R_rg_ptr.push_back((int32_t)G1);
            }
        }
        R_arena *= hk;
    }
    TP("B:rect");
    // ---- owner sums ----------------------------------------------------------------------------------------------- //
    std::vector<int32_t> qm_need, qm_out, qm_local, qm_ptr{0}, qm_land, km_need, km_out, km_local, km_ptr{0}, km_land;
    std::vector<int32_t> qm_idx_(nown[r], -1), km_idx_(nown[r], -1);   // per owned tile (by its owned slot)
    struct OIdx { std::vector<int32_t> &v; const std::function<int64_t(int64_t)> &f; int32_t &operator[](int64_t t) { return v[f(t)]; } };
    const std::function<int64_t(int64_t)> opos = [&](int64_t t) { return ownpos_t(t); };
    OIdx qm_idx{qm_idx_, opos}, km_idx{km_idx_, opos};
    // landings here per tile, in my_in order (contributor order): CSR over tiles
    // (by this PE's slot of the tile: it lands only partials of tiles it owns or is a gateway of = computes; a tile
    // without a slot falls back to per tile of the sequence)
    std::vector<int32_t> la_ptr[2], la[2];
    bool la_sl[2] = {true, true};
    for (int k = 0; k < 2; ++k) for (auto &e : my_in[k]) la_sl[k] &= myslot[k][e[0]] >= 0;
    auto lak = [&](int k, int64_t t) -> int64_t { return la_sl[k] ? myslot[k][t] : t; };
    for (int k = 0; k < 2; ++k) {
        const int64_t n_ = la_sl[k] ? (int64_t)(k ? KS_r.size() : QS_r.size()) : nv;
        la_ptr[k].assign(n_ + 1, 0);
        for (auto &e : my_in[k]) ++la_ptr[k][lak(k, e[0]) + 1];
        for (int64_t t = 0; t < n_; ++t) la_ptr[k][t + 1] += la_ptr[k][t];
        la[k].resize(my_in[k].size());
        std::vector<int32_t> f(la_ptr[k].begin(), la_ptr[k].end() - 1);
        for (auto &e : my_in[k]) la[k][f[lak(k, e[0])]++] = e[2];
    }
    auto la_lo = [&](int k, int64_t t) -> int32_t { const int64_t x = lak(k, t); return x < 0 ? 0 : la_ptr[k][x]; };
    auto la_hi = [&](int k, int64_t t) -> int32_t { const int64_t x = lak(k, t); return x < 0 ? 0 : la_ptr[k][x + 1]; };
    auto la_n = [&](int k, int64_t t) { return la_hi(k, t) - la_lo(k, t); };
    TP("b:u:land_at");
    for (int k = 0; k < 2; ++k) {
        auto &need = k ? km_need : qm_need; auto &out = k ? km_out : qm_out; auto &loc = k ? km_local : qm_local;
        auto &ptr = k ? km_ptr : qm_ptr; auto &land = k ? km_land : qm_land; OIdx &idx = k ? km_idx : qm_idx;
        for (int64_t b_ : owned_b[r]) for (int64_t t = b_ * per; t < (b_ + 1) * per; ++t) {   // (owned tiles, ascending)
            if (con_n(k, t) == 0) continue;
            const int64_t l = has_con(k, t, r) ? sl_of(k, r, t) : -1;
            idx[t] = (int32_t)out.size();
            need.push_back((int32_t)(la_n(k, t) + (l >= 0))); out.push_back((int32_t)sl_of(k, r, t));
            loc.push_back((int32_t)l);
            land.insert(land.end(), la[k].begin() + la_lo(k, t), la[k].begin() + la_hi(k, t)); ptr.push_back((int32_t)land.size());
        }
    }
    TP("b:u:owner");
    // where a completed slot's partial goes; a gateway first adds its node's landings (agg need = 1 + landings)
    std::vector<int32_t> qs_pe(NQS, -1), qs_dst(NQS, -1), ks_pe(NKS, -1), ks_dst(NKS, -1);
    std::vector<int32_t> qa_need(NQS, 0), qa_ptr(NQS + 1, 0), qa_land, ka_need(NKS, 0), ka_ptr(NKS + 1, 0), ka_land;
    for (int k = 0; k < 2; ++k) {
        auto &spe = k ? ks_pe : qs_pe; auto &sdst = k ? ks_dst : qs_dst; OIdx &idx = k ? km_idx : qm_idx;
        auto &aneed = k ? ka_need : qa_need; auto &aptr = k ? ka_ptr : qa_ptr; auto &aland = k ? ka_land : qa_land;
        const auto &S = k ? KS_r : QS_r;
        for (int64_t b_ : owned_b[r]) for (int64_t t = b_ * per; t < (b_ + 1) * per; ++t)   // owned tiles it computes
            if (has_con(k, t, r)) sdst[sl_of(k, r, t)] = idx[t];
        for (int32_t t : rl(r).t[k]) { const int64_t s = sl_of(k, r, t); spe[s] = my_out[k][s].first; sdst[s] = my_out[k][s].second; }
        for (int64_t s = 0; s < (k ? NKS : NQS); ++s) {
            if (s < (int64_t)S.size()) {
                const int64_t t = S[s];
                if (own(t) != r && gw_of(k, node(r), t) == r) {
                    aneed[s] = (int32_t)(1 + la_n(k, t));
                    aland.insert(aland.end(), la[k].begin() + la_lo(k, t), la[k].begin() + la_hi(k, t));
                }
            }
            aptr[s + 1] = (int32_t)aland.size();
        }
    }
    TP("B:sums");
    // ---- polled signals: task inputs, landings here, forwarded arrivals --------------------------------------------- //
    std::vector<int64_t> remote;
    remote.insert(remote.end(), cdeps.begin(), cdeps.end());
    remote.insert(remote.end(), R_rdep.begin(), R_rdep.end());
    for (int64_t x : R_gk) if (x >= 0) remote.push_back(x);
    std::unordered_map<int64_t, int32_t> land_q, land_k, land_qa, land_ka;   // landing signal -> owner sum / agg slot
    for (auto &e : my_in[0]) { const int64_t x = sig_dql(e[2]); remote.push_back(x);
        if (own(e[0]) == r) land_q[x] = qm_idx[e[0]]; else land_qa[x] = (int32_t)sl_of(0, r, e[0]); }
    for (auto &e : my_in[1]) { const int64_t x = sig_dkl(e[2]); remote.push_back(x);
        if (own(e[0]) == r) land_k[x] = km_idx[e[0]]; else land_ka[x] = (int32_t)sl_of(1, r, e[0]); }
    for (auto &f : fwd) remote.push_back(f.first);
    {                                                                   // sorted distinct: a bitmap over signal ids
        std::vector<char> seen_(NQS + NKS + NDQL + NDKL + 1, 0);
        for (int64_t x : remote) seen_[x] = 1;
        remote.clear();
        for (int64_t x = 0; x < (int64_t)seen_.size(); ++x) if (seen_[x]) remote.push_back(x);
    }
    const int64_t nrem = (int64_t)remote.size();
    // signal -> its index in remote (sorted): one binary search per chunk input, then per task by the chunk's index
    std::vector<int32_t> sid_of(NQS + NKS + NDQL + NDKL + 1, -1);       // signal id -> its index in remote
    for (int64_t j = 0; j < (int64_t)remote.size(); ++j) sid_of[remote[j]] = (int32_t)j;
    auto sid = [&](int64_t x) { return sid_of[x]; };
    std::vector<int32_t> cdx(cdeps.size());
    for (size_t x = 0; x < cdeps.size(); ++x) cdx[x] = sid(cdeps[x]);
    auto didx_of = [&](int64_t t, auto f) { const Chunk &ch = chunks[tchunk[t]]; for (int64_t x = ch.d0; x < ch.d1; ++x) f(cdeps[x], cdx[x]); };
    std::vector<int32_t> st_ptr{0}, st, sig_qm(nrem, -1), sig_km(nrem, -1), sig_qa(nrem, -1), sig_ka(nrem, -1);
    {
        std::vector<int32_t> cnt(nrem + 1, 0);
        for (int64_t t = 0; t < nt; ++t) didx_of(t, [&](int64_t, int32_t i) { ++cnt[i + 1]; });
        for (int64_t j = 0; j < nrem; ++j) cnt[j + 1] += cnt[j];
        st_ptr.assign(cnt.begin(), cnt.end()); st.resize(cnt[nrem]);
        for (int64_t t = 0; t < nt; ++t) didx_of(t, [&](int64_t, int32_t i) { st[cnt[i]++] = (int32_t)t; });
    }
    for (auto &kv : land_q) sig_qm[sid(kv.first)] = kv.second;
    for (auto &kv : land_k) sig_km[sid(kv.first)] = kv.second;
    for (auto &kv : land_qa) sig_qa[sid(kv.first)] = kv.second;
    for (auto &kv : land_ka) sig_ka[sid(kv.first)] = kv.second;
    std::vector<int32_t> need(nt), init_tasks;
    for (int64_t t = 0; t < nt; ++t) { need[t] = (int32_t)ndeps_of(t); if (!need[t]) init_tasks.push_back((int32_t)t); }
    TP("B:remote");
    // ---- the shared scheduler: rows = twin pairs (row input = the KV run), one group = the chunk's Q runs ---------- //
    const int64_t npair_t = nt / 2;
    std::vector<int32_t> row_qdep(npair_t, -1), grp_need(npair_t, 0), grp_dep(npair_t, -1);
    std::vector<std::vector<int32_t>> sgl(nrem), srl(nrem), sredl(nrem);
    for (int64_t k_ = 0; k_ < npair_t; ++k_) {
        const int64_t t = 2 * k_, ks_ = task_kv[t] >> 9, j = KS_r[ks_];
        const int64_t kdep = own(j) != r ? dep(1, j) : -1;
        didx_of(t, [&](int64_t x, int32_t i) {
            if (x == kdep) { row_qdep[k_] = i; srl[i].push_back((int32_t)k_); return; }
            if (rect) return;              // rect: the groups' lists are rebuilt below (sgl / grp_* replaced)
            if (grp_dep[k_] < 0) grp_dep[k_] = i;
            ++grp_need[k_]; sgl[i].push_back((int32_t)k_);
        });
    }
    const int64_t n_rows_s = rect ? HG * G1 : npair_t, n_grp_s = rect ? HG * G1 : npair_t;
    std::vector<int32_t> R_sg_ptr, R_sg;                                // rect: signal -> groups (CSR, built directly)
    if (rect) {                                                         // rows = groups, head * G1 + g1
        row_qdep.assign(n_rows_s, -1); grp_need.assign(n_grp_s, 0); grp_dep.assign(n_grp_s, -1);
        std::vector<int32_t> gd_ptr(G1 + 1, 0), gd;                     // per g1: its inputs' signal indices
        gd.reserve(R_rdep.size() + G1);
        for (int64_t g1 = 0; g1 < G1; ++g1) {
            for (int64_t x = R_rdep_ptr[g1]; x < R_rdep_ptr[g1 + 1]; ++x) gd.push_back(sid(R_rdep[x]));
            if (R_gk[g1] >= 0) gd.push_back(sid(R_gk[g1]));
            gd_ptr[g1 + 1] = (int32_t)gd.size();
        }
        // per signal: its g1 groups (ascending), then the table written signal by signal (the same entries in the same
        // order -- (g, g1) inside a signal -- as filling it group by group, but sequential, not scattered over a table of
        // HG x sum(deps) entries)
        std::vector<int32_t> s1_ptr(nrem + 1, 0), s1;
        for (int64_t g1 = 0; g1 < G1; ++g1) for (int32_t x = gd_ptr[g1]; x < gd_ptr[g1 + 1]; ++x) ++s1_ptr[gd[x] + 1];
        for (int64_t j = 0; j < nrem; ++j) s1_ptr[j + 1] += s1_ptr[j];
        s1.resize(s1_ptr[nrem]);
        {
            std::vector<int32_t> f1(s1_ptr.begin(), s1_ptr.end() - 1);
            for (int64_t g1 = 0; g1 < G1; ++g1) for (int32_t x = gd_ptr[g1]; x < gd_ptr[g1 + 1]; ++x) s1[f1[gd[x]]++] = (int32_t)g1;
        }
        R_sg_ptr.assign(nrem + 1, 0);
        for (int64_t j = 0; j < nrem; ++j) R_sg_ptr[j + 1] = R_sg_ptr[j] + (int32_t)HG * (s1_ptr[j + 1] - s1_ptr[j]);
        R_sg.resize(R_sg_ptr[nrem]);
        for (int64_t j = 0, w = 0; j < nrem; ++j)
            for (int64_t g = 0; g < HG; ++g)
                for (int32_t x = s1_ptr[j]; x < s1_ptr[j + 1]; ++x) R_sg[w++] = (int32_t)(g * G1 + s1[x]);
        for (int64_t g = 0; g < HG; ++g)
            for (int64_t g1 = 0; g1 < G1; ++g1) {
                const int64_t gg = g * G1 + g1, nd = gd_ptr[g1 + 1] - gd_ptr[g1];
                grp_need[gg] = (int32_t)nd; grp_dep[gg] = nd ? gd[gd_ptr[g1]] : -1;
            }
    }
    TP("B:s:rect");
    if (rect) {                                                         // groups without remote inputs first, then by
        R_exp_order.reserve(n_grp_s);                                   // expected landing of their inputs
        // (landing, group) order without a general sort: the landings are the few message landing times -> rank them,
        // then a counting sort of the groups by rank (stable: group order inside a rank)
        std::vector<int32_t> rk1(G1), cnt(nm + 2, 0);                   // (R_gland: rank among the messages'
        for (int64_t g1 = 0; g1 < G1; ++g1) rk1[g1] = R_gland[g1] + 1;  //  distinct landings, -1 = none)
        for (int64_t g = 0; g < HG; ++g)
            for (int64_t g1 = 0; g1 < G1; ++g1) {
                const int32_t gg = (int32_t)(g * G1 + g1);
                if (grp_need[gg] == 0) R_exp_order.push_back(gg); else ++cnt[rk1[g1] + 1];
            }
        for (size_t j = 0; j + 1 < cnt.size(); ++j) cnt[j + 1] += cnt[j];
        const size_t n0_ = R_exp_order.size();
        R_exp_order.resize(n_grp_s);
        for (int64_t g = 0; g < HG; ++g)
            for (int64_t g1 = 0; g1 < G1; ++g1) {
                const int32_t gg = (int32_t)(g * G1 + g1);
                if (grp_need[gg] > 0) R_exp_order[n0_ + cnt[rk1[g1]]++] = gg;
            }
    }
    TP("B:s:order");
    const int64_t n_qm = (int64_t)qm_out.size(), n_km = (int64_t)km_out.size();
    std::vector<int32_t> qa_unit(NQS, -1), ka_unit(NKS, -1), red_need, red_kind, red_pay, red_n;
    for (int64_t m_ = 0; m_ < n_qm; ++m_) { red_need.push_back(qm_need[m_]); red_kind.push_back(1); red_pay.push_back((int32_t)(m_ * 2 * h)); red_n.push_back((int32_t)(2 * h)); }
    for (int64_t m_ = 0; m_ < n_km; ++m_) { red_need.push_back(km_need[m_]); red_kind.push_back(2); red_pay.push_back((int32_t)(m_ * hk)); red_n.push_back((int32_t)hk); }
    for (int64_t s = 0; s < NQS; ++s) if (qa_need[s] > 0) { qa_unit[s] = (int32_t)red_need.size(); red_need.push_back(qa_need[s]); red_kind.push_back(6); red_pay.push_back((int32_t)s); red_n.push_back(1); }
    for (int64_t s = 0; s < NKS; ++s) if (ka_need[s] > 0) { ka_unit[s] = (int32_t)red_need.size(); red_need.push_back(ka_need[s]); red_kind.push_back(7); red_pay.push_back((int32_t)s); red_n.push_back(1); }
    for (auto &kv : land_q) sredl[sid(kv.first)].push_back(kv.second);
    for (auto &kv : land_k) sredl[sid(kv.first)].push_back((int32_t)(n_qm + kv.second));
    for (auto &kv : land_qa) sredl[sid(kv.first)].push_back(qa_unit[kv.second]);
    for (auto &kv : land_ka) sredl[sid(kv.first)].push_back(ka_unit[kv.second]);
    auto csr = [&](std::vector<std::vector<int32_t>> &L, std::vector<int32_t> &ptr, std::vector<int32_t> &val) {
        ptr.assign(1, 0); val.clear();
        for (auto &v : L) { val.insert(val.end(), v.begin(), v.end()); ptr.push_back((int32_t)val.size()); }
    };
    std::vector<int32_t> sg_ptr, sg, sr_ptr, sr, sred_ptr, sred;
    csr(sgl, sg_ptr, sg); csr(srl, sr_ptr, sr); csr(sredl, sred_ptr, sred);
    if (rect) { sg_ptr.swap(R_sg_ptr); sg.swap(R_sg); }
    TP("B:sched");
    // ---- message tables by lane: engine (inter-node, and intra-node unless ce_intra) / copy engine ------------------ //
    auto is_ce = [&](int64_t p_) { return ce_intra && node(p_) == node(r); };
    std::vector<int64_t> &msg = g_bwd.msg, &ce = g_bwd.ce;
    msg.clear(); ce.clear();
    auto put_msg = [&](const M5 &x) { msg.insert(msg.end(), {x.src, x.dst, x.bytes, x.pe + (x.sig << 32)}); };
    auto put_ce = [&](const M5 &x, int64_t kind, int64_t word) {
        ce.insert(ce.end(), {x.src, x.dst, x.bytes, x.pe + (x.sig << 32), kind + (word << 32), 1});
    };
    for (auto &x : start) if (!is_ce(x.pe)) put_msg(x);
    const int64_t n_start = (int64_t)msg.size() / 4;
    for (auto &x : start) if (is_ce(x.pe)) put_ce(x, 0, 0);
    const int64_t ntrig = NQS + NKS + nrem;
    std::vector<int32_t> trig_ptr{(int32_t)n_start}, lane(ntrig, 0);
    std::vector<std::tuple<int64_t, int64_t, M5>> ce_done, ce_fw;     // (order key, word, message)
    for (int64_t s = 0; s < NQS + NKS; ++s) {
        const bool isq = s < NQS; const int64_t sl = isq ? s : s - NQS;
        const int64_t pe_ = isq ? qs_pe[sl] : ks_pe[sl];
        if (pe_ >= 0) {
            const int64_t dl = isq ? qs_dst[sl] : ks_dst[sl];
            const M5 x = isq ? M5{lo[3] + sl * dqa, LO[pe_][5] + dl * dqa, dqa, pe_, cQ[pe_] + cK[pe_] + dl}
                             : M5{lo[4] + sl * dkva, LO[pe_][6] + dl * dkva, dkva, pe_, cQ[pe_] + cK[pe_] + cDQ[pe_] + dl};
            if (is_ce(pe_)) { ce_done.emplace_back(isq ? qlast[sl] : klast[sl], s, x); lane[s] = 2; }
            else { put_msg(x); lane[s] = 1; }
        }
        trig_ptr.push_back((int32_t)(msg.size() / 4));
    }
    std::vector<int32_t> sig_fwd(nrem, 0);
    for (int64_t j = 0; j < nrem; ++j) {
        auto it = fwd.find(remote[j]);
        if (it != fwd.end())
            for (auto &x : it->second) {
                if (is_ce(x.pe)) { ce_fw.emplace_back(fwd_key[remote[j]], remote[j], x); lane[NQS + NKS + j] |= 2; }
                else { put_msg(x); lane[NQS + NKS + j] |= 1; sig_fwd[j] = 1; }
            }
        trig_ptr.push_back((int32_t)(msg.size() / 4));
    }
    std::stable_sort(ce_done.begin(), ce_done.end(), [](auto &a, auto &b) { return std::get<0>(a) < std::get<0>(b); });
    std::stable_sort(ce_fw.begin(), ce_fw.end(), [](auto &a, auto &b) { return std::get<0>(a) < std::get<0>(b); });
    for (auto &e : ce_done) put_ce(std::get<2>(e), 2, std::get<1>(e));
    for (auto &e : ce_fw) put_ce(std::get<2>(e), 1, std::get<1>(e));
    int64_t n_pq = 0;
    for (int64_t t = 0; t < ntrig; ++t) n_pq += lane[t] & 1;
    TP("B:msgs");
    // ---- pack (python TABLE_ORDER) -------------------------------------------------------------------------------- //
    auto &T_ = g_bwd.tab;
    T_.clear();
    auto v32 = [](const std::vector<int64_t> &v) { return std::vector<int32_t>(v.begin(), v.end()); };
    std::vector<int32_t> row_grp_ptr(n_rows_s + 1), grp_beg(n_grp_s, 0), grp_end(n_grp_s, 1), row_order(n_rows_s);
    for (int64_t k_ = 0; k_ <= n_rows_s; ++k_) row_grp_ptr[k_] = (int32_t)k_;      // one group per row
    if (rect) for (int64_t g = 0; g < n_grp_s; ++g) { grp_beg[g] = (int32_t)g; grp_end[g] = (int32_t)(g + 1); }   // task_desc[1] = group
    for (int64_t k_ = 0; k_ < n_rows_s; ++k_) row_order[k_] = (int32_t)k_;
    const int64_t n_items_ = (int64_t)items.size(), n_init_ = (int64_t)init_tasks.size(), n_red_ = (int64_t)red_need.size();
    int64_t nqa = 0, nka = 0;
    for (auto v : qa_need) nqa += v > 0;
    for (auto v : ka_need) nka += v > 0;
    T_.reserve(64);
    T_.push_back(std::move(task_kv)); T_.push_back(std::move(task_ptr));
    T_.push_back(std::vector<int32_t>());                               // (items: g_bwd.items)
    for (auto *v : {&tqs_ptr, &tqs, &need, &init_tasks, &qs_need, &ks_need, &qs_pe, &qs_dst,
                    &ks_pe, &ks_dst, &qm_need, &qm_out, &qm_local, &qm_ptr, &qm_land, &km_need, &km_out, &km_local,
                    &km_ptr, &km_land})
        T_.push_back(std::move(*v));
    T_.push_back(v32(remote));
    for (auto *v : {&st_ptr, &st, &sig_qm, &sig_km, &qa_need, &qa_ptr, &qa_land, &ka_need, &ka_ptr, &ka_land, &sig_qa,
                    &sig_ka, &sig_fwd})
        T_.push_back(std::move(*v));
    T_.push_back(std::vector<int32_t>());                               // (msg: separate, int64)
    for (auto *v : {&trig_ptr, &lane, &row_grp_ptr, &grp_beg, &grp_end, &grp_dep, &grp_need, &row_qdep, &row_order,
                    &sg_ptr, &sg, &sr_ptr, &sr, &red_need, &red_kind, &red_pay, &red_n, &sred_ptr, &sred, &qa_unit,
                    &ka_unit})
        T_.push_back(std::move(*v));
    if (rect) {                                                         // python RECT_ORDER (u64 tables as int32 pairs)
        auto w32 = [](const std::vector<uint64_t> &v) {
            std::vector<int32_t> o(2 * v.size()); if (!v.empty()) std::memcpy(o.data(), v.data(), v.size() * 8); return o; };
        for (auto *v : {&R_g_rect, &R_g_col, &R_g_k0, &R_g_cnt, &R_g_ks, &R_rect_pp, &R_rect_pair, &R_rect_bk, &R_pair_bq})
            T_.push_back(std::move(*v));
        for (auto *v : {&R_bits, &R_dbits, &R_prow, &R_pcol}) T_.push_back(w32(*v));
        T_.push_back(std::move(R_qslot)); T_.push_back(std::move(R_kslot)); T_.push_back(std::move(R_exp_order));
    }
    int64_t n_fwd = 0, n_fwd_msgs = 0;
    for (auto &f : fwd) { ++n_fwd; n_fwd_msgs += (int64_t)f.second.size(); }
    g_bwd.scal = {NQS, NKS, NDQL, NDKL, rect ? 2 * n_grp_s : nt, rect ? R_arena : n_items_, n_init_, n_qm, n_km, nrem,
                  n_pushq, n_pushkv, n_fwd, nqa, nka, n_pq, n_start, (int64_t)ce.size() / 6, n_rows_s,
                  n_red_, n_fwd_msgs, n_grp_s, n_rect, G1, R_arena, nw, per};
    TP("B:pack");
    return (int64_t)T_.size();
}

int64_t bwd_tab_size(int64_t k) { return k == 2 ? (int64_t)g_bwd.items.size() : (int64_t)g_bwd.tab[k].size(); }
// every table back to back (sizes[k] = its length; one call instead of two per table)
int64_t bwd_tab_total() {
    int64_t n = 0;
    for (size_t k = 0; k < g_bwd.tab.size(); ++k) n += bwd_tab_size((int64_t)k);
    return n;
}
void bwd_tab_pack(int32_t *dst, int64_t *sizes) {
    int64_t off = 0;
    for (size_t k = 0; k < g_bwd.tab.size(); ++k) {
        const int64_t n = bwd_tab_size((int64_t)k);
        sizes[k] = n;
        std::memcpy(dst + off, k == 2 ? g_bwd.items.data() : g_bwd.tab[k].data(), n * 4); off += n;
    }
}
void bwd_tab_copy(int64_t k, int32_t *dst) {
    if (k == 2) std::memcpy(dst, g_bwd.items.data(), g_bwd.items.size() * 4);
    else std::memcpy(dst, g_bwd.tab[k].data(), g_bwd.tab[k].size() * 4);
}
int64_t bwd_msg_rows() { return (int64_t)g_bwd.msg.size() / 4; }
int64_t bwd_ce_rows() { return (int64_t)g_bwd.ce.size() / 6; }
void bwd_fetch(int64_t *msg, int64_t *ce, int64_t *scal) {
    std::memcpy(msg, g_bwd.msg.data(), g_bwd.msg.size() * 8);
    std::memcpy(ce, g_bwd.ce.data(), g_bwd.ce.size() * 8);
    std::memcpy(scal, g_bwd.scal.data(), g_bwd.scal.size() * 8);
}

}  // extern "C"
