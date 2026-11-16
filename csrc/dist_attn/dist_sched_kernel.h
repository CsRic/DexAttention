#pragma once

// ── The dist kernel with a block-0 task scheduler ────────────────────────────
// FA3's persistent kernel, with its dynamic scheduler (an atomic counter) replaced by block 0:
//
//   block 0        never computes. It polls every arrival (Q / KV runs from peers, partials landing here),
//                  tracks which groups of tiles of which rows are now resident, and keeps the task queue fed:
//                  when less than `sched_low` work (tile x head units) is queued, it takes the highest-priority
//                  row that has resident, unissued tiles and issues ALL of them (the consecutive ones) as one task, one queue
//                  entry per head. While the workers are busy, arrivals accumulate and tasks grow; when they are
//                  starving, tasks are cut as soon as data is there.
//   blocks 1..G-1  FA3 workers: claim the next queue entry (atomic counter, as FA3), run sparse_fwd's mainloop +
//                  epilogue on the task's tiles -- no gate, nothing in the loop waits on data -- and account.
//
// A task that covers its whole row writes the row's output directly (final O, or the partial sent to the Q owner).
// Otherwise every task of the row writes a partial into a local pool slot and the block that finishes the row's
// last tile queues a local merge (log-sum-exp) into the row's output. Owner merges (rows owned here, computed in
// part elsewhere) are queued once their landing partials and the local row are in. When every row is written and
// every owner merge queued, block 0 appends one STOP per worker.

#include "dist_fwd_kernel.h"
#include "../runtime/rt_sched.cuh"

namespace ring_attn {

// The scheduler kernel's mainloop issues its K prefetches without the branch past the row's last tile
// (RingCollectiveMainloopFwdSm80::mma<_, BranchlessKNext>); this is a per-kernel codegen choice.
#ifndef DIST_SCHED_BRANCHLESS
#define DIST_SCHED_BRANCHLESS 1
#endif
static constexpr bool kSchedBranchless = DIST_SCHED_BRANCHLESS != 0;

// Control words live in DYNAMIC shared memory past the compute routine's area, at sched_ctl_off<Traits>(): a static
// __shared__ variable shifts the dynamic area, and the mainloop's swizzled tiles then lose their 128-byte alignment
// (dist_bwd_kernel.h Ctl). The kernel has no static shared memory.
static constexpr int kRing = rt::kSchedRing;               // block 0's ring of dispatched batches (work-bounded queue)
struct SchedCtl {
    int last_head, last_grp, entry, owe_quiet, first;      // worker
    int q_left, bcast;                                     // block 0 (bcast: the engine's)
    rt::SchedB0 b0;                                        // block 0's scheduler state (both schedulers)
};
template <typename Traits> constexpr int sched_ctl_off() { return 128 * ((Traits::kSmemSize + 127) / 128); }
template <typename Traits> constexpr int sched_smem_bytes() { return sched_ctl_off<Traits>() + int(sizeof(SchedCtl)); }

// queue entry = ((kind << 28) | payload) + 1, 0 = not written yet
enum : int { kSqCompute = 0, kSqLocalMerge = 1, kSqOwnerMerge = 2, kSqStop = 3 };
__device__ __forceinline__ int sq_entry(int kind, int payload) { return ((kind << 28) | payload) + 1; }

// Append `n` entries kind/payload0+i (any block; the tail is shared with the workers' merge enqueues).
__device__ __forceinline__ void sq_push(const Dist_fwd_params &p, int kind, int payload0, int n) {
    const int base = atomicAdd(p.sq_tail, n);
    for (int i = 0; i < n; ++i) atomicExch(&p.sq[base + i], sq_entry(kind, payload0 + i));
}

// A row's output is complete (the last head of its final writer is done): feed its owner merge, count it written.
// Called by thread 0 of the block that closed the row.
__device__ __noinline__ void sched_row_written(const Dist_fwd_params &p, int r) {
    if (p.timeline) {          // timeline: the row's "compute item" slot = (kernel start, row written)
        unsigned long long *tl = p.timeline + 1 + p.NQ + p.NKV + p.NPART + 2 * (int64_t)r;
        tl[1] = dist_now();                                  // tl[0]: first dispatch of the row (block 0)
    }
    const int m = p.row_merge[r];
    if (m >= 0 && atomicAdd(&p.merge_cnt[m], 1) + 1 == p.merge_need[m]) {
        sq_push(p, kSqOwnerMerge, m * p.h, p.h);
        __threadfence();
        atomicAdd(p.merges_enq, 1);
    }
    __threadfence();
    atomicAdd(p.rows_done, 1);
}

// ---- block 0 ----------------------------------------------------------------------------------------------- //
// Warp 0 only. Shared memory (block 0 never computes, so all of it is free): per-group state (0 waiting, 1 resident,
// 2 issued), per-row "Q here" and task count, per-polled-signal done bits.
// n_workers: blocks that will each read one STOP; n_static: queue slots handed out statically (one per worker that
// starts computing at once -- the push blocks claim their first slot only after pushing), so the claimed frontier is
// tile_counter + n_static.
__device__ __noinline__ void dist_sched_block0(const Dist_fwd_params &p, char *smem, int n_workers, int n_static,
                                               SchedCtl &c) {
    if (threadIdx.x >= 32) return;
    const int lane = threadIdx.x;
    const int nr = p.n_compute, ng = p.n_groups, ns = p.n_remote;
    if (p.b0_state) smem = p.b0_state;                    // too large for shared memory (launcher decides)
    uint8_t *gst = reinterpret_cast<uint8_t *>(smem);
    uint8_t *qok = gst + ng;
    // per row: tasks issued / groups not issued yet -- 16-bit (a row has <= kMaxRowTiles groups), so per-head
    // problems (thousands of rows) still fit in shared memory
    uint16_t *rnt = reinterpret_cast<uint16_t *>(smem + ((ng + nr + 15) & ~15));
    uint16_t *rem = rnt + nr;
    uint32_t *sdone = reinterpret_cast<uint32_t *>(smem + ((((ng + nr + 15) & ~15) + 4 * (int64_t)nr + 15) & ~15));
    int *batch = reinterpret_cast<int *>(sdone + (ns + 31) / 32);        // task ids of one dispatch round
    int &s_nb = c.b0.nb, &s_base = c.b0.base;
    // Work-bounded queue (p.q_work != nullptr): block 0 alone tracks the work it queued and not yet claimed -- a ring
    // of its batches (first slot, entries, work); the claimed front is tile_counter + n_static (workers do nothing
    // extra, so the worker loop stays sparse_fwd's).
    int *rb_base = c.b0.rb_base, *rb_n = c.b0.rb_n, *rb_w = c.b0.rb_w;
    int rb_head = 0, rb_tail = 0, rb_total = 0;            // lane 0 only
    int &s_q_left = c.q_left;                             // q_all_in: inbound Q signals not seen yet
    if (lane == 0 && p.q_all_in) {
        int nq = 0;
        for (int i = 0; i < ns; ++i) nq += p.remote_sig[i] < p.NQ;
        s_q_left = nq;
        if (nq == 0) atomicExch(p.q_all_in, 1);
    }
    for (int r = lane; r < nr; r += 32) rem[r] = p.row_grp_ptr[r + 1] - p.row_grp_ptr[r];
    int cursor = 0;                                       // rows before it (in priority order) are fully issued
    for (int g = lane; g < ng; g += 32) gst[g] = p.grp_dep[g] < 0 ? 1 : 0;
    for (int r = lane; r < nr; r += 32) { qok[r] = p.row_qdep[r] < 0 ? 1 : 0; rnt[r] = 0; }
    for (int w = lane; w < (ns + 31) / 32; w += 32) sdone[w] = 0u;
    __syncwarp();
    if (p.timeline && lane == 0) dist_tl_ev(p)[1] = dist_now();
    int n_tasks = 0, stopped = 0;
    long idle = 0;
    while (!stopped) {
        int progress = 0;
        // 1. arrivals: lanes split the polled ids; each arrival marks its groups / rows and feeds owner merges
        for (int i = lane; i < ns; i += 32) {
            if (sdone[i >> 5] & (1u << (i & 31))) continue;
            const int id = p.remote_sig[i];
            if (*(const volatile uint64_t *)&p.sig_base[id] < p.sig_epoch) continue;
            __threadfence_system();
            if (p.remote_cnt) for (int k = p.remote_cnt[i] - 1; k > 0; --k) atomicExch(&p.arrived[id - k], 1);
            atomicExch(&p.arrived[id], 1);
            if (p.timeline) p.timeline[1 + id] = dist_now();
            atomicOr(&sdone[i >> 5], 1u << (i & 31));
            if (p.use_engine == 2) {               // relay: forward the arrived input inside the node (engine)
                const int t = p.n_compute + i;
                if (p.eng.trig_ptr[t + 1] > p.eng.trig_ptr[t]) rt::pq_push(p.eng, t);
            }
            if (p.q_all_in && id < p.NQ && atomicSub(&s_q_left, 1) == 1) atomicExch(p.q_all_in, 1);
            for (int k = p.sig_grp_ptr[i]; k < p.sig_grp_ptr[i + 1]; ++k) gst[p.sig_grp[k]] = 1;
            for (int k = p.sig_row_ptr[i]; k < p.sig_row_ptr[i + 1]; ++k) qok[p.sig_row[k]] = 1;
            for (int k = p.sig_merge_ptr[i]; k < p.sig_merge_ptr[i + 1]; ++k) {
                const int m = p.sig_merge[k];
                if (atomicAdd(&p.merge_cnt[m], 1) + 1 == p.merge_need[m]) {
                    sq_push(p, kSqOwnerMerge, m * p.h, p.h);
                    __threadfence();
                    atomicAdd(p.merges_enq, 1);
                }
            }
            progress = 1;
        }
        __syncwarp();
        // 2. dispatch: when the queue runs low, lane 0 cuts up to sched_batch tasks (each: the highest-priority row's
        // resident, unissued, consecutive groups) and the warp queues them HEAD-MAJOR across the batch -- head 0 of
        // every task, then head 1, ... -- FA3's order: many rows of one head run together and share KV in L2.
        // (scan: the warp checks 32 rows of the priority order at a time from `cursor`, lane 0 cuts tasks for the
        // candidates in order; rows already fully issued are skipped by the cursor)
        // Two tiers, so a row is split only when that saves idle time: below `sched_low` queued entries, issue rows
        // whose whole remainder is resident (one task, no extra partial); only below `sched_crit` (the workers are
        // about to run dry) also cut rows that are only partly resident.
        int depth = 0;
        if (lane == 0) {
            s_nb = 0;
            const int claimed = *(const volatile int *)p.tile_counter + n_static;
            if (p.q_work) {
                while (rb_head != rb_tail && rb_base[rb_head % kRing] + rb_n[rb_head % kRing] <= claimed) {
                    rb_total -= rb_w[rb_head % kRing]; ++rb_head;
                }
                depth = rb_total;
                if (rb_head != rb_tail) {                         // the oldest batch may be partly claimed
                    const int i = rb_head % kRing, done = claimed - rb_base[i];
                    if (done > 0) depth -= (int)((long long)rb_w[i] * done / rb_n[i]);
                }
                if (rb_tail - rb_head >= kRing) depth = 1 << 30;   // ring full: wait for claims
            } else {
                depth = *(const volatile int *)p.sq_tail - claimed;
            }
            while (cursor < nr && rem[p.row_order[cursor]] == 0) ++cursor;
        }
        depth = __shfl_sync(0xffffffffu, depth, 0);
        cursor = __shfl_sync(0xffffffffu, cursor, 0);
        int nb = 0, nbw = 0;                                     // tasks / their work (tiles x heads) this round
        auto queued = [&]() { return depth + (p.q_work ? nbw : nb * p.h); };
        for (int tier = 0; tier < 2; ++tier) {
            const int thr = tier == 0 ? p.sched_low : p.sched_crit;
            if (queued() >= thr) continue;
            for (int k = cursor; k < nr && nb < p.sched_batch && queued() < thr; k += 32) {
                int cand = 0;
                if (k + lane < nr) {
                    const int r = p.row_order[k + lane];
                    if (rem[r] > 0 && qok[r]) {
                        int ready = 0, waiting = 0;
                        for (int g = p.row_grp_ptr[r]; g < p.row_grp_ptr[r + 1]; ++g) {
                            ready += gst[g] == 1; waiting += gst[g] == 0;
                        }
                        cand = ready > 0 && (tier == 1 || waiting == 0);
                    }
                }
                unsigned m = __ballot_sync(0xffffffffu, cand);
                if (lane == 0) {
                    while (m && nb < p.sched_batch && queued() < thr) {
                        const int r = p.row_order[k + __ffs(m) - 1];
                        m &= m - 1;
                        for (int g0 = p.row_grp_ptr[r]; g0 < p.row_grp_ptr[r + 1] && nb < p.sched_batch; ++g0) {
                            if (gst[g0] != 1) continue;
                            int g1 = g0;
                            while (g1 + 1 < p.row_grp_ptr[r + 1] && gst[g1 + 1] == 1) ++g1;
                            for (int g = g0; g <= g1; ++g) gst[g] = 2;
                            rem[r] -= g1 - g0 + 1;
                            const bool whole = rnt[r] == 0 && g0 == p.row_grp_ptr[r] && g1 + 1 == p.row_grp_ptr[r + 1];
                            if (p.timeline && rnt[r] == 0)       // timeline: the row's first dispatch
                                p.timeline[1 + p.NQ + p.NKV + p.NPART + 2 * (int64_t)r] = dist_now();
                            const int out = whole ? -1 : p.row_pool[r] + rnt[r];
                            ++rnt[r];
                            p.row_ntask[r] = rnt[r];
                            int *d = p.task_desc + 4 * n_tasks;
                            d[0] = r; d[1] = p.grp_beg[g0]; d[2] = p.grp_end[g1]; d[3] = out;
                            nbw += (p.grp_end[g1] - p.grp_beg[g0]) * p.h;
                            batch[nb++] = n_tasks++;
                            g0 = g1;
                        }
                    }
                }
                nb = __shfl_sync(0xffffffffu, nb, 0);
                nbw = __shfl_sync(0xffffffffu, nbw, 0);
                __syncwarp();
            }
        }
        if (lane == 0 && nb) {
            __threadfence();
            s_base = atomicAdd(p.sq_tail, nb * p.h);
            if (p.q_work) {
                const int i = rb_tail % kRing;
                rb_base[i] = s_base; rb_n[i] = nb * p.h; rb_w[i] = nbw; ++rb_tail; rb_total += nbw;
            }
            s_nb = nb;
            if (p.trace) { p.trace[11] = n_tasks; p.trace[13] = dist_now() - p.trace[8]; }
            if (p.timeline) atomicCAS(&dist_tl_ev(p)[2], 0ULL, dist_now());
        }
        __syncwarp();
        if (s_nb) {
            const int nb = s_nb, base = s_base;
            for (int k = lane; k < nb * p.h; k += 32)
                atomicExch(&p.sq[base + k], sq_entry(kSqCompute, batch[k % nb] * p.h + k / nb));
            progress = 1;
        }
        __syncwarp();
        if (lane == 0) {
            // 3. done: every row written, every owner merge queued -> one STOP per worker
            if (*(const volatile int *)p.rows_done == nr && *(const volatile int *)p.merges_enq == p.n_merge) {
                sq_push(p, kSqStop, 0, n_workers);
                stopped = 1;
                if (p.timeline) dist_tl_ev(p)[3] = dist_now();
                if (p.trace) { int sp = 0; for (int r = 0; r < nr; ++r) sp += rnt[r] > 1; p.trace[12] = sp; }
            }
        }
        stopped = __shfl_sync(0xffffffffu, stopped, 0);
        progress = __any_sync(0xffffffffu, progress);
        if (!progress && !stopped) {
            __nanosleep(128);
            if (lane == 0 && ++idle % (DIST_SPIN_REPORT / 4) == 0) {
                int g0n = 0, g1n = 0, g2n = 0, nq = 0, sn = 0, over = 0, under = 0;
                for (int g = 0; g < ng; ++g) { g0n += gst[g] == 0; g1n += gst[g] == 1; g2n += gst[g] == 2; }
                for (int r = 0; r < nr; ++r) {
                    nq += !qok[r];
                    const int t = p.row_tiles_done[r], n = p.kv_indptr[r + 1] - p.kv_indptr[r];
                    over += t > n; under += (rnt[r] > 1 || (rnt[r] == 1 && t > 0)) && t < n && rem[r] == 0;
                }
                int shown = 0;
                for (int i = 0; i < ns; ++i) {
                    if (sdone[i >> 5] & (1u << (i & 31))) continue;
                    ++sn;
                    const int id = p.remote_sig[i];
                    if (shown++ < 4)
                        printf("[pe %d sched] pending signal %d (%s), groups waiting on it %d, merges %d\n",
                               nvshmem_my_pe(), id, id < p.NQ ? "Q" : id < p.NQ + p.NKV ? "KV" : "partial",
                               p.sig_grp_ptr[i + 1] - p.sig_grp_ptr[i], p.sig_merge_ptr[i + 1] - p.sig_merge_ptr[i]);
                }
                printf("[pe %d sched] idle: rows written %d/%d, owner merges queued %d/%d, tasks %d | groups "
                       "wait/ready/issued %d/%d/%d, rows without Q %d, signals pending %d/%d, rows over %d, "
                       "rows all-issued-but-short %d, tail %d claimed %d\n",
                       nvshmem_my_pe(), *(const volatile int *)p.rows_done, nr,
                       *(const volatile int *)p.merges_enq, p.n_merge, n_tasks, g0n, g1n, g2n, nq, sn, ns, over,
                       under, *(const volatile int *)p.sq_tail, *(const volatile int *)p.tile_counter + n_workers);
            }
        }
    }
}

// ---- block 0, rectangle segments (p.rect) ------------------------------------------------------------------- //
// Same loop as dist_sched_block0; the readiness unit is a SEGMENT of a block row (shared by its tile rows), dispatch
// state is per (row, segment): 0 empty for this row (no live tile of it there), 1..64 not issued (= its tiles, counted
// once at start, so dispatch never touches the bitmaps), 255 issued. A task = a row's run of consecutive segments that
// are ready (or empty for it).
__device__ __forceinline__ uint64_t rect_bits(const Dist_fwd_params &p, int s, int bi) {
    return p.seg_mask[s] & p.pair_rows[(int64_t)p.seg_pair[s] * p.per + bi];
}
__device__ __noinline__ void dist_sched_block0_rect(const Dist_fwd_params &p, char *smem, int n_workers, int n_static,
                                                    SchedCtl &c) {
    if (threadIdx.x >= 32) return;
    const int lane = threadIdx.x;
    const int nr = p.n_compute, nsg = p.n_segs, ns = p.n_remote, nrs = p.n_rowseg;
    if (p.b0_state) smem = p.b0_state;
    // layout: sready[nsg] u8 | rst[nrs] u8 | qok[nr] u8 | pad16 | rnt[nr] u16 | rem[nr] u16 | pad16 | sdone | batch
    uint8_t *sready = reinterpret_cast<uint8_t *>(smem);
    uint8_t *rst = sready + nsg;
    uint8_t *qok = rst + nrs;
    int64_t off = ((int64_t)nsg + nrs + nr + 15) & ~15LL;
    uint16_t *rnt = reinterpret_cast<uint16_t *>(smem + off);
    uint16_t *rem = rnt + nr;
    off = (off + 4 * (int64_t)nr + 15) & ~15LL;
    uint32_t *sdone = reinterpret_cast<uint32_t *>(smem + off);
    int *batch = reinterpret_cast<int *>(sdone + (ns + 31) / 32);
    int &s_nb = c.b0.nb, &s_base = c.b0.base;
    int *rb_base = c.b0.rb_base, *rb_n = c.b0.rb_n, *rb_w = c.b0.rb_w;
    int rb_head = 0, rb_tail = 0, rb_total = 0;
    int &s_q_left = c.q_left;
    if (lane == 0 && p.q_all_in) {
        int nq = 0;
        for (int i = 0; i < ns; ++i) nq += p.remote_sig[i] < p.NQ;
        s_q_left = nq;
        if (nq == 0) atomicExch(p.q_all_in, 1);
    }
    for (int s = lane; s < nsg; s += 32) sready[s] = p.seg_dep[s] < 0 ? 1 : 0;
    for (int r = lane; r < nr; r += 32) {
        qok[r] = p.row_qdep[r] < 0 ? 1 : 0; rnt[r] = 0;
        const int b = p.row_brow[r], bi = p.row_bi[r], s0 = p.brow_seg_ptr[b], s1 = p.brow_seg_ptr[b + 1];
        uint8_t *st = rst + p.row_seg_base[r] - s0;
        int n = 0;
        for (int s = s0; s < s1; ++s) { const int c = __popcll(rect_bits(p, s, bi)); st[s] = (uint8_t)c; n += c > 0; }
        rem[r] = (uint16_t)n;
    }
    for (int w = lane; w < (ns + 31) / 32; w += 32) sdone[w] = 0u;
    __syncwarp();
    if (p.timeline && lane == 0) dist_tl_ev(p)[1] = dist_now();
    int cursor = 0, n_tasks = 0, stopped = 0;
    long idle = 0;
    while (!stopped) {
        int progress = 0;
        // 1. arrivals: segments become ready, rows get their Q, owner merges get their partials
        for (int i = lane; i < ns; i += 32) {
            if (sdone[i >> 5] & (1u << (i & 31))) continue;
            const int id = p.remote_sig[i];
            if (*(const volatile uint64_t *)&p.sig_base[id] < p.sig_epoch) continue;
            __threadfence_system();
            if (p.remote_cnt) for (int k = p.remote_cnt[i] - 1; k > 0; --k) atomicExch(&p.arrived[id - k], 1);
            atomicExch(&p.arrived[id], 1);
            if (p.timeline) p.timeline[1 + id] = dist_now();
            atomicOr(&sdone[i >> 5], 1u << (i & 31));
            if (p.use_engine == 2) {               // relay: forward the arrived input inside the node (engine)
                const int t = p.n_compute + i;
                if (p.eng.trig_ptr[t + 1] > p.eng.trig_ptr[t]) rt::pq_push(p.eng, t);
            }
            if (p.q_all_in && id < p.NQ && atomicSub(&s_q_left, 1) == 1) atomicExch(p.q_all_in, 1);
            for (int k = p.sig_grp_ptr[i]; k < p.sig_grp_ptr[i + 1]; ++k) sready[p.sig_grp[k]] = 1;
            for (int k = p.sig_row_ptr[i]; k < p.sig_row_ptr[i + 1]; ++k) qok[p.sig_row[k]] = 1;
            for (int k = p.sig_merge_ptr[i]; k < p.sig_merge_ptr[i + 1]; ++k) {
                const int m = p.sig_merge[k];
                if (atomicAdd(&p.merge_cnt[m], 1) + 1 == p.merge_need[m]) {
                    sq_push(p, kSqOwnerMerge, m * p.h, p.h);
                    __threadfence();
                    atomicAdd(p.merges_enq, 1);
                }
            }
            progress = 1;
        }
        __syncwarp();
        // 2. dispatch (as dist_sched_block0: two tiers, head-major batches)
        int depth = 0;
        if (lane == 0) {
            s_nb = 0;
            const int claimed = *(const volatile int *)p.tile_counter + n_static;
            if (p.q_work) {
                while (rb_head != rb_tail && rb_base[rb_head % kRing] + rb_n[rb_head % kRing] <= claimed) {
                    rb_total -= rb_w[rb_head % kRing]; ++rb_head;
                }
                depth = rb_total;
                if (rb_head != rb_tail) {
                    const int i = rb_head % kRing, done = claimed - rb_base[i];
                    if (done > 0) depth -= (int)((long long)rb_w[i] * done / rb_n[i]);
                }
                if (rb_tail - rb_head >= kRing) depth = 1 << 30;
            } else {
                depth = *(const volatile int *)p.sq_tail - claimed;
            }
            while (cursor < nr && rem[p.row_order[cursor]] == 0) ++cursor;
        }
        depth = __shfl_sync(0xffffffffu, depth, 0);
        cursor = __shfl_sync(0xffffffffu, cursor, 0);
        int nb = 0, nbw = 0;
        auto queued = [&]() { return depth + (p.q_work ? nbw : nb * p.h); };
        for (int tier = 0; tier < 2; ++tier) {
            const int thr = tier == 0 ? p.sched_low : p.sched_crit;
            if (queued() >= thr) continue;
            for (int k = cursor; k < nr && nb < p.sched_batch && queued() < thr; k += 32) {
                int cand = 0;
                if (k + lane < nr) {
                    const int r = p.row_order[k + lane];
                    if (rem[r] > 0 && qok[r]) {
                        const int b = p.row_brow[r], s0 = p.brow_seg_ptr[b], s1 = p.brow_seg_ptr[b + 1];
                        const uint8_t *st = rst + p.row_seg_base[r] - s0;
                        int ready = 0, waiting = 0;
                        for (int s = s0; s < s1; ++s)
                            if (st[s] != 0 && st[s] != 255) { ready += sready[s]; waiting += !sready[s]; }
                        cand = ready > 0 && (tier == 1 || waiting == 0);
                    }
                }
                unsigned m = __ballot_sync(0xffffffffu, cand);
                if (lane == 0) {
                    while (m && nb < p.sched_batch && queued() < thr) {
                        const int r = p.row_order[k + __ffs(m) - 1];
                        m &= m - 1;
                        const int b = p.row_brow[r], s0 = p.brow_seg_ptr[b], s1 = p.brow_seg_ptr[b + 1];
                        uint8_t *st = rst + p.row_seg_base[r] - s0;
                        auto open = [&](int s) { return st[s] != 0 && st[s] != 255; };
                        for (int a = s0; a < s1 && nb < p.sched_batch; ++a) {
                            if (!(open(a) && sready[a])) continue;
                            int last = a, j = a + 1, tiles = 0, k2 = 0;
                            while (j < s1 && (st[j] == 0 || (open(j) && sready[j]))) { if (st[j] != 0) last = j; ++j; }
                            for (int s = a; s <= last; ++s)
                                if (open(s)) { tiles += st[s]; st[s] = 255; ++k2; }
                            rem[r] -= k2;
                            const bool whole = rnt[r] == 0 && rem[r] == 0;
                            if (p.timeline && rnt[r] == 0)
                                p.timeline[1 + p.NQ + p.NKV + p.NPART + 2 * (int64_t)r] = dist_now();
                            const int out = whole ? -1 : p.row_pool[r] + rnt[r];
                            ++rnt[r];
                            p.row_ntask[r] = rnt[r];
                            int *d = p.task_desc + 4 * n_tasks;
                            d[0] = r; d[1] = a; d[2] = last + 1; d[3] = out;
                            nbw += tiles * p.h;
                            batch[nb++] = n_tasks++;
                            a = last;
                        }
                    }
                }
                nb = __shfl_sync(0xffffffffu, nb, 0);
                nbw = __shfl_sync(0xffffffffu, nbw, 0);
                __syncwarp();
            }
        }
        if (lane == 0 && nb) {
            __threadfence();
            s_base = atomicAdd(p.sq_tail, nb * p.h);
            if (p.q_work) {
                const int i = rb_tail % kRing;
                rb_base[i] = s_base; rb_n[i] = nb * p.h; rb_w[i] = nbw; ++rb_tail; rb_total += nbw;
            }
            s_nb = nb;
            if (p.trace) { p.trace[11] = n_tasks; p.trace[13] = dist_now() - p.trace[8]; }
            if (p.timeline) atomicCAS(&dist_tl_ev(p)[2], 0ULL, dist_now());
        }
        __syncwarp();
        if (s_nb) {
            const int nb = s_nb, base = s_base;
            for (int k = lane; k < nb * p.h; k += 32)
                atomicExch(&p.sq[base + k], sq_entry(kSqCompute, batch[k % nb] * p.h + k / nb));
            progress = 1;
        }
        __syncwarp();
        if (lane == 0) {
            if (*(const volatile int *)p.rows_done == nr && *(const volatile int *)p.merges_enq == p.n_merge) {
                sq_push(p, kSqStop, 0, n_workers);
                stopped = 1;
                if (p.timeline) dist_tl_ev(p)[3] = dist_now();
                if (p.trace) { int sp = 0; for (int r = 0; r < nr; ++r) sp += rnt[r] > 1; p.trace[12] = sp; }
            }
        }
        stopped = __shfl_sync(0xffffffffu, stopped, 0);
        progress = __any_sync(0xffffffffu, progress);
        if (!progress && !stopped) {
            __nanosleep(128);
            if (lane == 0 && ++idle % (DIST_SPIN_REPORT / 4) == 0) {
                int sw = 0, sn = 0, rq = 0, rr = 0;
                for (int s = 0; s < nsg; ++s) sw += !sready[s];
                for (int r = 0; r < nr; ++r) { rq += !qok[r]; rr += rem[r] > 0; }
                for (int i = 0; i < ns; ++i) sn += !(sdone[i >> 5] & (1u << (i & 31)));
                printf("[pe %d sched rect] idle: rows written %d/%d, owner merges queued %d/%d, tasks %d | segments "
                       "waiting %d/%d, rows without Q %d, rows with unissued tiles %d, signals pending %d/%d\n",
                       nvshmem_my_pe(), *(const volatile int *)p.rows_done, nr, *(const volatile int *)p.merges_enq,
                       p.n_merge, n_tasks, sw, nsg, rq, rr, sn, ns);
            }
        }
    }
}

// diagnostic counters of this kernel (u64, summed over workers): 0 queue-wait ns, 1 compute-task ns, 2 local-merge
// ns, 3 owner-merge ns, 4 compute tasks, 5 tiles, 6 local merges, 7 owner merges, 8 first start, 9 last end,
// 10 workers done, 11 tasks issued (block 0), 12 rows split (block 0), 13 block-0 last dispatch ns from start
#define STR_ADD(p, i, v) do { if ((p).trace && threadIdx.x == 0) atomicAdd(&(p).trace[i], (unsigned long long)(v)); } while (0)
__device__ __noinline__ void sched_trace_end(const Dist_fwd_params &p, int n_workers) {
    if (!p.trace || threadIdx.x != 0) return;
    atomicMax(&p.trace[9], dist_now());
    if (atomicAdd(&p.trace[10], 1ULL) == (unsigned long long)n_workers) {      // workers + block 0
        __threadfence();
        const unsigned long long *t = p.trace;
        const double nw = n_workers;
        printf("[pe %d sched trace] span %.2f ms | per worker: queue-wait %.2f, compute %.2f, local-merge %.2f, "
               "owner-merge %.2f ms | tasks %llu (%llu tiles), local merges %llu, owner merges %llu | issued %llu, "
               "rows split %llu, last dispatch at %.2f ms\n",
               nvshmem_my_pe(), (t[9] - t[8]) * 1e-6, t[0] * 1e-6 / nw, t[1] * 1e-6 / nw, t[2] * 1e-6 / nw,
               t[3] * 1e-6 / nw, t[4], t[5], t[6], t[7], t[11], t[12], t[13] * 1e-6);
    }
}

// Worker: wait until queue slot `idx` is written (only when nothing is queued yet). Cold, out of line.
__device__ __noinline__ int dist_sq_wait(const Dist_fwd_params &p, int idx) {
    const volatile int *q = p.sq + idx;
    long n = 0;
    while (*q == 0) {
        __nanosleep(64);
        if (++n % DIST_SPIN_REPORT == 0)
            printf("[pe %d blk %3d] task queue slot %d still empty\n", nvshmem_my_pe(), blockIdx.x, idx);
    }
    const int v = *q - 1;
    __threadfence();
    return v;
}

// bounded pool: merge this worker's one-head partial (scratch slot) into row r's accumulator (pool slot r), under the
// (row, head) lock; the first arrival copies. Same log-sum-exp arithmetic as dist_merge_tile (natural log, normalised
// O). Accumulator reads bypass L1 (another block may have written them).
__device__ __noinline__ void dist_acc_merge(const Dist_fwd_params &p, int r, int bidh, float *smem) {
    const int tid = threadIdx.x, nt = blockDim.x, H = p.h, D = p.d;
    const int w = p.pool_scratch0 + (int)blockIdx.x - 1;
    if (blockIdx.x - 1 >= p.pool_nws) __trap();                 // (host sized the scratch for pool_nws workers)
    const int64_t ls = p.pool_lse_stride, key = (int64_t)r * H + bidh;
    float *po = (float *)p.pool_o, *pl = p.pool_lse;
    float *wa = smem, *wb = smem + 128;
    int *s_first = reinterpret_cast<int *>(smem + 256);
    if (tid == 0) {
        while (atomicCAS(p.acc_lock + key, 0, 1) != 0) __nanosleep(32);
        __threadfence();
        const int c = *(volatile int *)(p.acc_cnt + key);
        *s_first = c == 0;
        p.acc_cnt[key] = c + 1;
    }
    __syncthreads();
    const bool first = *s_first;
    float *la = pl + (int64_t)bidh * ls + (int64_t)r * 128;
    const float *lb = pl + (int64_t)bidh * ls + (int64_t)w * 128;
    for (int i = tid; i < 128; i += nt) {
        const float a = first ? -INFINITY : __ldcg(la + i), b = __ldcg(lb + i);
        const float m = fmaxf(a, b);
        if (m == -INFINITY) { wa[i] = 0.f; wb[i] = 0.f; }
        else {
            const float ea = __expf(a - m), eb = __expf(b - m), s = ea + eb;
            wa[i] = ea / s; wb[i] = eb / s;
        }
    }
    __syncthreads();
    for (int idx = tid; idx < 128 * D; idx += nt) {
        const int i = idx / D, c = idx % D;
        const int64_t oa = (((int64_t)r * 128 + i) * H + bidh) * D + c, ob = (((int64_t)w * 128 + i) * H + bidh) * D + c;
        const float x = wa[i] == 0.f ? 0.f : wa[i] * __ldcg(po + oa);
        const float y = wb[i] == 0.f ? 0.f : wb[i] * __ldcg(po + ob);
        po[oa] = x + y;
    }
    for (int i = tid; i < 128; i += nt) {                        // LSE last (the O loop reads only the weights)
        const float a = first ? -INFINITY : __ldcg(la + i), b = __ldcg(lb + i);
        const float m = fmaxf(a, b);
        la[i] = m == -INFINITY ? -INFINITY : m + __logf(__expf(a - m) + __expf(b - m));
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) atomicExch(p.acc_lock + key, 0);
}

// Local merge of row r's task partials (pool) into the row's output, one head.
__device__ __noinline__ void dist_local_merge(const Dist_fwd_params &p, int r, int bidh, float *smem) {
    int *s_slots = reinterpret_cast<int *>(reinterpret_cast<char *>(smem) + 4096);   // past the merge statistics
    const int n = p.pool_acc ? 1 : p.row_ntask[r];                // (bounded pool: the accumulator, pool slot r)
    for (int i = threadIdx.x; i < n; i += blockDim.x) s_slots[i] = p.pool_acc ? r : p.row_pool[r] + i;
    __syncthreads();
    const int dout = p.item_direct ? p.item_direct[r] : -1;
    const MergeLayout pool{128LL * p.h * p.d, 128, p.pool_lse_stride};          // the pool is head-major
    if (dout >= 0) {
        dist_merge_tile<128, 128, 256, float, __half>(p.o_ptr, (float *)p.lse_ptr, (const float *)p.pool_o, p.pool_lse, s_slots, n, dout,
                          bidh, p.h, pool, MergeLayout{128LL * p.h * p.d, 128, p.flse_head_stride}, smem);
    } else {
        dist_merge_tile<128, 128, 256, float, float>(p.part_o, (float *)p.part_lse, (const float *)p.pool_o, p.pool_lse, s_slots, n, r,
                          bidh, p.h, pool, MergeLayout{p.part_o_slot, p.part_lse_slot, p.lse_head_stride}, smem);
    }
    __syncthreads();
}

// Owner merge m (rows owned here with contributions from elsewhere), one head. Its inputs are all in.
__device__ __noinline__ void dist_owner_merge(const Dist_fwd_params &p, int m, int bidh, float *smem) {
    const unsigned long long tm0 = p.timeline ? dist_now() : 0;
    const int beg = p.merge_ptr[m], end = p.merge_ptr[m + 1];
    dist_merge_tile<128, 128, 256, float, __half>(p.o_ptr, (float *)p.lse_ptr, (const float *)p.part_o, (const float *)p.part_lse,
                      p.merge_slot + beg, end - beg, p.merge_out[m], bidh, p.h,
                      MergeLayout{p.part_o_slot, p.part_lse_slot, p.lse_head_stride},
                      MergeLayout{128LL * p.h * p.d, 128, p.flse_head_stride}, smem);
    __syncthreads();
    if (p.timeline && threadIdx.x == 0) {                 // timeline: (start, end) per (merge, head)
        unsigned long long *tl = p.timeline + 1 + p.NQ + p.NKV + p.NPART
                                 + 2 * (int64_t)(p.n_compute > 0 ? p.n_compute : 1) + 2 * ((int64_t)m * p.h + bidh);
        tl[0] = tm0; tl[1] = dist_now();
    }
}

// rect: expand task (row r, segments [s0, s1)) into this block's KV list (warp 0, one lane per segment, a prefix sum
// places each segment's tiles; masked-diagonal tiles last). Returns the count in *n_out and the masked tail in
// *nmask_out (shared). Out of line so the worker's mainloop codegen is untouched.
__device__ __noinline__ void dist_rect_expand(const Dist_fwd_params &p, int r, int s0, int s1, int *lst,
                                              int *n_out, int *nmask_out) {
    const int lane = threadIdx.x;
    if (lane >= 32) return;
    const int bi = p.row_bi[r];
    int base = 0, n_live = 0;
    for (int pass = 0; pass < 2; ++pass) {
        for (int c0 = s0; c0 < s1; c0 += 32) {
            const int s = c0 + lane;
            uint64_t m = 0;
            if (s < s1) {
                const uint64_t dg = p.pair_drows ? p.pair_drows[(int64_t)p.seg_pair[s] * p.per + bi] : 0ull;
                m = rect_bits(p, s, bi) & (pass ? dg : ~dg);
            }
            const int c = __popcll(m);
            int x = c;
            #pragma unroll
            for (int o = 1; o < 32; o <<= 1) { const int y = __shfl_up_sync(0xffffffffu, x, o); if (lane >= o) x += y; }
            int w = base + x - c;
            const int cb = s < s1 ? p.seg_cb[s] : 0;
            for (; m; m &= m - 1) lst[w++] = cb + __ffsll((long long)m) - 1;
            base += __shfl_sync(0xffffffffu, x, 31);
        }
        if (pass == 0) n_live = base;
    }
    if (lane == 0) { *n_out = base; *nmask_out = base - n_live; }
}

// Block 0, warps 1 .. kDxWarps (p.dx_on): the rows' KV lists from their emitted segments, kDxClaim rows per claim in
// dispatch priority (row_order); one fence per claim, then the claim is marked done (dx_bits = per-claim flags) and the
// published prefix *dx_rows (rows row_order[0 .. *dx_rows) are written) is advanced over every done claim (the
// scheduler, rt::sched_block0<2>, dispatches only below it: one read per pass). Lanes take a row's segments 32 at a time, a scan of the popcounts gives
// each one's position. Same lists as the host-side expansion (dex/scheduler/fwd.py tables_blocks): segment order, set
// bits ascending; kv_ord_sig = -1 for a local tile, else NQ + column.
static constexpr int kDxWarps = 3, kDxClaim = 4;
__device__ __noinline__ void dx_block0(const Dist_fwd_params &p) {
    const int lane = threadIdx.x & 31, nr = p.n_compute;
    int32_t *kv = const_cast<int32_t *>(p.kv_indices), *ks = const_cast<int32_t *>(p.kv_ord_sig);
    for (;;) {
        int k0 = lane == 0 ? atomicAdd(p.dx_next, kDxClaim) : 0;
        k0 = __shfl_sync(0xffffffffu, k0, 0);
        if (k0 >= nr) break;
        const int k1 = min(nr, k0 + kDxClaim);
        for (int k = k0; k < k1; ++k) {
            const int r = p.row_order[k], s0 = p.dx_row_seg_ptr[r], s1 = p.dx_row_seg_ptr[r + 1];
            int base = p.kv_indptr[r];
            for (int c = s0; c < s1; c += 32) {
                const int s = c + lane;
                uint64_t m = 0; int cb = 0, loc = 0;
                if (s < s1) {
                    m = (uint64_t)(uint32_t)p.dx_seg_lo[s] | ((uint64_t)(uint32_t)p.dx_seg_hi[s] << 32);
                    cb = p.dx_seg_cb[s]; loc = p.dx_seg_loc[s];
                }
                const int n = __popcll(m);
                int x = n;
                #pragma unroll
                for (int o = 1; o < 32; o <<= 1) { const int y = __shfl_up_sync(0xffffffffu, x, o); if (lane >= o) x += y; }
                int w = base + x - n;
                for (; m; m &= m - 1) {
                    const int j = cb + __ffsll((long long)m) - 1;
                    if (p.kv_half) {                               // j = 2 tile + hn: 64-row K tile 2 slot + hn
                        const int sl = p.dx_kslot[j >> 1];
                        kv[w] = 2 * sl + (j & 1); ks[w] = loc ? -1 : p.NQ + sl; const_cast<int32_t *>(p.kv_geom)[w] = j;
                    } else if (p.dx_kslot) {                       // compact KV: slot, and the tile as geometry
                        const int sl = p.dx_kslot[j];
                        kv[w] = sl; ks[w] = loc ? -1 : p.NQ + sl; const_cast<int32_t *>(p.kv_geom)[w] = j;
                    } else {
                        kv[w] = j; ks[w] = loc ? -1 : p.NQ + j;
                    }
                    ++w;
                }
                base += __shfl_sync(0xffffffffu, x, 31);
            }
        }
        __syncwarp();
        if (lane == 0) {
            __threadfence();
            atomicExch(reinterpret_cast<int *>(p.dx_bits) + k0 / kDxClaim, 1);      // this claim is written
            __threadfence();                       // (two warps finishing neighbouring claims: one sees the other's flag)
            const int nc = (nr + kDxClaim - 1) / kDxClaim;
            int *pc = p.dx_next + 1;                                               // claims published (prefix)
            for (int c = *(volatile int *)pc; c < nc && *(volatile int *)(reinterpret_cast<int *>(p.dx_bits) + c);
                 c = *(volatile int *)pc)
                atomicCAS(pc, c, c + 1);
            atomicMax(p.dx_next + 2, min(nr, *(volatile int *)pc * kDxClaim));    // rows published
        }
        __syncwarp();
    }
}

template<typename Kernel_traits, bool Rect>
__global__ __launch_bounds__(Kernel_traits::kNThreads, 1)
void dist_sched_kernel(
    const __grid_constant__ typename Kernel_traits::LaunchParams fa3,
    const __grid_constant__ Dist_fwd_params p,
    const __grid_constant__ DistTileScheduler::Params sched_params
) {
    extern __shared__ char smem_[];
    const int tidx = threadIdx.x;
    if (p.trace && tidx == 0) atomicCAS(&p.trace[8], 0ULL, dist_now());
    if (p.timeline && tidx == 0) { atomicCAS(&p.timeline[0], 0ULL, dist_now()); atomicMax(&dist_tl_ev(p)[0], dist_now()); }

    SchedCtl &ctl = *reinterpret_cast<SchedCtl *>(smem_ + sched_ctl_off<Kernel_traits>());
    bool owe_quiet = dist_push_phase(p);
    if (blockIdx.x == 0) {
        const int e0 = p.dx_on ? 32 * (1 + kDxWarps) : 32;     // dx: warps 1-3 expand the KV lists
        if (!Rect && p.dx_on && tidx >= 32 && tidx < e0) dx_block0(p);
        else if (p.use_engine && tidx >= e0) rt::engine(p.eng, e0, int(blockDim.x) - e0, 1, &ctl.bcast);
        else if (!Rect && p.sp_on && p.dx_on) rt::sched_block0<2>(p.sp, smem_, ctl.b0, p.use_engine ? &p.eng : nullptr);
        else if (!Rect && p.sp_on) rt::sched_block0(p.sp, smem_, ctl.b0, p.use_engine ? &p.eng : nullptr);
        else if constexpr (Rect) dist_sched_block0_rect(p, smem_, int(gridDim.x) - 1, int(gridDim.x) - sched_params.n_reserved, ctl);
        else dist_sched_block0(p, smem_, int(gridDim.x) - 1, int(gridDim.x) - sched_params.n_reserved, ctl);
        __syncthreads();
        if (owe_quiet) dist_quiet_gated(p);
        sched_trace_end(p, int(gridDim.x) - 1);
        dist_credit_release(p);
        return;
    }

    typename Kernel_traits::Mainloop mainloop;
    typename Kernel_traits::Epilogue epilogue;              // final output (fp16)
    typename Kernel_traits::EpiloguePart epilogue_part;     // partials: pool slots and the row's partial slot (fp32)
    typename Kernel_traits::TiledMma tiled_mma;
    DistTileScheduler scheduler(&reinterpret_cast<typename Kernel_traits::SharedStorage *>(smem_)->smem_scheduler);
    scheduler.init_consumer();
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    int &s_last_head = ctl.last_head, &s_last_grp = ctl.last_grp, &s_entry = ctl.entry, &s_owe_quiet = ctl.owe_quiet;
    if (tidx == 0) s_owe_quiet = owe_quiet;

    // The push blocks (1 .. n_reserved-1) hold NO static first slot: a push block is busy posting its inter-node
    // messages for as long as the NIC queue is full -- up to the whole transfer -- and a slot it held would sit
    // unprocessed all that time.
    // They claim their first slot from the counter once their pushes are posted.
    int &s_first = ctl.first;
    auto work0 = scheduler.get_initial_work(sched_params);
    if (int(blockIdx.x) < sched_params.n_reserved) {
        if (tidx == 0) s_first = atomicAdd(p.tile_counter, 1) + int(gridDim.x) - sched_params.n_reserved;
        __syncthreads();
        work0.tile_idx = s_first;
    }
    CUTLASS_PRAGMA_NO_UNROLL
    for (auto work = work0; ;
         work = warp_idx == 0
             ? scheduler.template get_next_work</*IsProducerWarp=*/true>(sched_params, work)
             : scheduler.template get_next_work</*IsProducerWarp=*/false>(sched_params, work)) {
        const unsigned long long tw0 = p.trace ? dist_now() : 0;
        if (tidx == 0) s_entry = dist_sq_wait(p, work.tile_idx);
        __syncthreads();
        const unsigned long long tw1 = p.trace ? dist_now() : 0;
        STR_ADD(p, 0, tw1 - tw0);
        const int e = s_entry;
        const int kind = e >> 28, payload = e & ((1 << 28) - 1);
        if (kind == kSqStop) break;
        const int item = payload / p.h, bidh = payload % p.h;
        scheduler.prefetch_next_work(sched_params, work);

        if (kind == kSqCompute) {
            // opaque per task: keeps ptxas from hoisting (and spilling) the mainloop's shared-memory addresses
            char *smem_item = smem_;
            asm volatile("" : "+l"(smem_item));
            auto &shared_storage = *reinterpret_cast<typename Kernel_traits::SharedStorage *>(smem_item);
            const int4 d = *reinterpret_cast<const int4 *>(p.task_desc + 4 * item);
            const int r = d.x, tb = d.y, te = d.z, out = d.w;
            int beg = p.kv_indptr[r] + tb, cnt = te - tb;
            // (queued work is tracked by block 0 alone, from the claim counter: workers stay exactly sparse_fwd's)
            bool has_diag = te == p.kv_indptr[r + 1] - p.kv_indptr[r] && p.row_n_masked[r] > 0;
            int n_mask = has_diag ? (p.kv_half ? p.row_n_masked[r] : 1) : 0;   // (halves: a diagonal tile = 2 entries)
            const int32_t *kvl = p.kv_indices;
            if constexpr (Rect) {                 // segments [tb, te) of the row's block row -> this block's KV list
                int *lst = p.rect_list + (int64_t)blockIdx.x * (Kernel_traits::kMaxRowTiles + 2);
                dist_rect_expand(p, r, tb, te, lst, &s_last_grp, &s_last_head);   // (shared scratch, reset below)
                __syncthreads();
                cnt = s_last_grp; n_mask = s_last_head; has_diag = n_mask > 0;
                __syncthreads();
                kvl = lst; beg = 0;
            }
            Tensor acc_o = partition_fragment_C(tiled_mma, select<0, 2>(typename Kernel_traits::TileShape_MNK{}));
            typename Kernel_traits::Softmax_t softmax(fa3.mainloop.softmax_scale_log2);
            typename Kernel_traits::SeqlenInfo_t seqlen_info{
                0, p.seqlen_q, p.seqlen_k, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
            // global-order KV: the slot IS the position, so the slot list doubles as mask geometry; no gate
            const int32_t *geom = Rect ? kvl : p.kv_geom;          // mask geometry (compact KV: the global tiles)
            flash::SparseRowDesc row{ kvl + beg, geom + beg, cnt, n_mask,
                                      p.row_mask_mb[r], 0, 0 };
            const cute::tuple<int32_t, int32_t, int32_t, int32_t> q_coord{
                p.item_q ? p.item_q[r] : r, bidh, 0, 0};
            const bool tile_valid = mainloop.template mma</*Is_first_step=*/true, /*BranchlessKNext=*/kSchedBranchless>(
                fa3.mainloop, (const typename Kernel_traits::Element *)p.k_slots,
                (const typename Kernel_traits::Element *)p.v_slots,
                acc_o, softmax, tidx, seqlen_info, q_coord, shared_storage, /*is_strict=*/false, &row);
            // where: the task's pool slot, else (whole row) the final output or the row's partial slot
            const int dout = (out < 0 && p.item_direct) ? p.item_direct[r] : -1;
            // combined partial slots are addressed through the epilogue's batch dimension (slot = batch, row block 0)
            const bool to_part = out < 0 && dout < 0;
            const cute::tuple<int32_t, int32_t, int32_t, int32_t> o_coord{
                out >= 0 ? (p.pool_acc ? p.pool_scratch0 + (int)blockIdx.x - 1 : out) : (dout >= 0 ? dout : (p.part_combined ? 0 : r)), bidh,
                to_part && p.part_combined ? r : 0, 0};
            const bool fp32_part = dout < 0;                   // pool slot or partial slot: fp32 O
            auto const &ep_part = out >= 0 ? fa3.epilogue_pool : fa3.epilogue;
            if (tile_valid) {
                Tensor final_scale = softmax.finalize();
                softmax.rescale_o(acc_o, final_scale);
                if (fp32_part) epilogue_part.store(ep_part, acc_o, softmax.row_sum, shared_storage, tiled_mma, tidx, o_coord);
                else epilogue.store(fa3.epilogue_final, acc_o, softmax.row_sum, shared_storage, tiled_mma, tidx, o_coord);
            } else {
                if (fp32_part) epilogue_part.store_zero(ep_part, tidx, o_coord);
                else epilogue.store_zero(fa3.epilogue_final, tidx, o_coord);
            }
            STR_ADD(p, 4, 1); STR_ADD(p, 5, cnt);
            if (out < 0) {                                       // the row's output: count heads, push home
                if (dist_return_partial(p, r, s_last_head, s_last_grp) && tidx == 0) s_owe_quiet = 1;
                if (s_last_head && tidx == 0) sched_row_written(p, r);
            } else {                                             // one partial of the row: count its tiles
                __syncthreads();
                if (p.pool_acc) dist_acc_merge(p, r, bidh, reinterpret_cast<float *>(smem_));   // (bounded pool)
                if (tidx == 0) {
                    __threadfence();
                    if (atomicAdd(&p.task_heads[item], 1) == p.h - 1 &&
                        atomicAdd(&p.row_tiles_done[r], cnt) + cnt == p.kv_indptr[r + 1] - p.kv_indptr[r]) {
                        __threadfence();
                        sq_push(p, kSqLocalMerge, r * p.h, p.h);
                    }
                }
            }
            STR_ADD(p, 1, (p.trace ? dist_now() : 0) - tw1);
        } else if (kind == kSqLocalMerge) {
            dist_local_merge(p, item, bidh, reinterpret_cast<float *>(smem_));
            if (dist_return_partial(p, item, s_last_head, s_last_grp) && tidx == 0) s_owe_quiet = 1;
            if (s_last_head && tidx == 0) sched_row_written(p, item);
            STR_ADD(p, 2, (p.trace ? dist_now() : 0) - tw1); STR_ADD(p, 6, 1);
        } else {                                                 // owner merge
            dist_owner_merge(p, item, bidh, reinterpret_cast<float *>(smem_));
            STR_ADD(p, 3, (p.trace ? dist_now() : 0) - tw1); STR_ADD(p, 7, 1);
        }
    }
    __syncthreads();
    if (p.timeline && tidx == 0) atomicMax(&dist_tl_ev(p)[4], dist_now());
    if (s_owe_quiet) dist_quiet_gated(p);
    sched_trace_end(p, int(gridDim.x) - 1);
    dist_credit_release(p);
}

}  // namespace ring_attn
