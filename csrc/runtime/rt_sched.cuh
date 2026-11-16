#pragma once
// Dispatch ready tasks from block 0; attention runs on worker blocks.
// 1. Poll input signals, update dependencies, and release forwards/reductions.
// 2. Refill the worker queue in planned row order, skipping unavailable work.
//    Below sched_low, prefer rows whose remaining inputs are all ready.
//    Below sched_crit, also dispatch ready parts of other rows.
// 3. Once rows finish and reductions are queued, send STOP to each worker.
// SchedParams holds dependency counts, batch limits and worker-gang sizes.
// Expected arrival times guide planning only; signals determine eligibility.
#include <cstdint>
#include "rt_params.h"
#include "rt_engine.cuh"

__device__ __forceinline__ unsigned long long globaltimer() {
    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}

#ifndef RT_SPIN_REPORT
#define RT_SPIN_REPORT (1L << 26)
#endif

namespace rt {

static constexpr int kSchedRing = 512;                     // block 0's ring of dispatched batches (work-bounded queue)
struct SchedB0 {                                           // block 0's control words (dynamic shared memory)
    int nb, base, q_left;
    int rb_base[kSchedRing], rb_n[kSchedRing], rb_w[kSchedRing];
};

__device__ __forceinline__ int sq_entry(int kind, int payload) { return ((kind << 28) | payload) + 1; }

// Append n entries kind / payload0 + i, padded with no-op entries to a multiple of the gang size (any thread).
__device__ __forceinline__ void sq_push(const SchedParams &s, int kind, int payload0, int n) {
    const int n2 = (n + s.gang - 1) / s.gang * s.gang;
    const int base = atomicAdd(s.sq_tail, n2);
    for (int i = 0; i < n2; ++i) atomicExch(&s.sq[base + i], i < n ? sq_entry(kind, payload0 + i) : sq_entry(s.kind_nop, 0));
}

// One input of reduction unit m is in (an arrival or a local completion): queue the unit when it is complete.
__device__ __forceinline__ void red_count(const SchedParams &s, int m) {
    if (atomicAdd(&s.red_cnt[m], 1) + 1 == s.red_need[m]) {
        __threadfence();
        if (s.red_kind) sq_push(s, s.red_kind[m], s.red_pay[m], s.red_n[m]);
        else sq_push(s, s.kind_red, m * s.red_n_all, s.red_n_all);
        __threadfence();
        atomicAdd(s.n_enq, 1);
    }
}

// Warp 0 of block 0. smem: free shared memory for the per-group / per-row state (s.b0_state instead when set).
// eng: the communication engine (relay forwards it sends on arrival), or nullptr.
// kAvail: 1 = grp_max + grp_avail_bits (the backward's rectangle mode), 2 = row_avail prefix (the forward's block-0
// expansion), 0 = neither. Compile-time, because ready() is called per group of every candidate row and the dispatch loop
// is the scheduler's hot path.
template <int kAvail = 0>
static __device__ __noinline__ void sched_block0(const SchedParams &s, char *smem, SchedB0 &c, const EngineParams *eng) {
    if (threadIdx.x >= 32) return;
    const int lane = threadIdx.x;
    const int nr = s.n_rows, ng = s.n_groups, ns = s.n_remote;
    if (s.b0_state) smem = s.b0_state;
    const int nu = s.grp_unit ? s.n_units : 0;
    uint8_t *gst = reinterpret_cast<uint8_t *>(smem);      // per group: 0 waiting, 1 resident, 2 issued
    uint8_t *qok = gst + ng;                                // per row: its own input is here
    uint8_t *ust = qok + nr;                                // per readiness unit: resident (grp_unit only)
    const int64_t o1 = ((int64_t)ng + nr + nu + 15) & ~15LL;
    uint16_t *rnt = reinterpret_cast<uint16_t *>(smem + o1);                      // per row: tasks issued
    uint16_t *rem = rnt + nr;                                                     // per row: groups not issued
    uint32_t *sdone = reinterpret_cast<uint32_t *>(smem + ((o1 + 4 * (int64_t)nr + 15) & ~15LL));
    int *batch = reinterpret_cast<int *>(sdone + (ns + 31) / 32);
    int &s_nb = c.nb, &s_base = c.base;
    int *rb_base = c.rb_base, *rb_n = c.rb_n, *rb_w = c.rb_w;
    int rb_head = 0, rb_tail = 0, rb_total = 0;            // lane 0 only
    for (int r = lane; r < nr; r += 32) rem[r] = s.row_grp_ptr[r + 1] - s.row_grp_ptr[r];
    for (int g = lane; g < ng; g += 32) gst[g] = s.grp_unit ? 0 : (s.grp_dep[g] < 0 ? 1 : 0);
    for (int u = lane; u < nu; u += 32) ust[u] = s.unit_dep[u] < 0 ? 1 : 0;
    // a group is ready: its own state, or (units) not issued and its unit resident; and materialised (grp_avail_bits)
    auto ready = [&](int g) {
        if (kAvail == 1 && !((*(const volatile unsigned *)&s.grp_avail_bits[g >> 5] >> (g & 31)) & 1u)) return false;
        return gst[g] == 1 || (nu && gst[g] == 0 && ust[s.grp_unit[g]]);
    };
    if (s.q_all_in && lane == 0) {                          // q_all_in: inbound Q signals not seen yet
        int n = 0;
        for (int i = 0; i < ns; ++i) n += s.remote_sig[i] < s.q_lim;
        c.q_left = n;
        if (n == 0) atomicExch(s.q_all_in, 1);
    }
    for (int r = lane; r < nr; r += 32) { qok[r] = s.row_qdep[r] < 0 ? 1 : 0; rnt[r] = 0; }
    for (int w = lane; w < (ns + 31) / 32; w += 32) sdone[w] = 0u;
    __syncwarp();
    if (s.tl_ev && lane == 0) s.tl_ev[1] = globaltimer();
    int cursor = 0, n_tasks = 0, stopped = 0;
    long idle = 0;
    unsigned long long st_it = 0, st_dit = 0, st_arr = 0, st_dis = 0, st_q = 0, st_fast = 0, st_slow = 0, st_scan = 0;   // (tl_ev only)
    while (!stopped) {
        int progress = 0;
        const unsigned long long ta_ = s.tl_ev ? globaltimer() : 0;
        // 1. arrivals (lanes over the polled signals; a signal's sig_grp list longer than coop_len is walked by the
        //    whole warp afterwards, so one lane walking a list of thousands does not starve the dispatch)
        auto mark_grp = [&](int g) {
            if (nu) ust[g] = 1;                     // (units: sig_grp lists units)
            else if (!s.grp_need || s.grp_need[g] <= 1 || atomicAdd(&s.grp_cnt[g], 1) + 1 == s.grp_need[g]) gst[g] = 1;
        };
        for (int base = 0; base < ns; base += 32) {
            const int i = base + lane;
            bool coop = false;
            if (i < ns && !(sdone[i >> 5] & (1u << (i & 31)))) {
                const int id = s.remote_sig[i];
                if (*(const volatile uint64_t *)&s.sig[id] >= s.epoch) {
                    __threadfence_system();
                    if (s.arrived) {
                        if (s.remote_cnt) for (int k = s.remote_cnt[i] - 1; k > 0; --k) atomicExch(&s.arrived[id - k], 1);
                        atomicExch(&s.arrived[id], 1);
                    }
                    atomicOr(&sdone[i >> 5], 1u << (i & 31));
                    if (s.tl_sig) s.tl_sig[id] = globaltimer();
                    if (s.q_all_in && id < s.q_lim && atomicSub(&c.q_left, 1) == 1) atomicExch(s.q_all_in, 1);
                    const int k0 = s.sig_grp_ptr[i], k1 = s.sig_grp_ptr[i + 1];
                    coop = s.coop_len > 0 && k1 - k0 > s.coop_len;
                    if (!coop) for (int k = k0; k < k1; ++k) mark_grp(s.sig_grp[k]);
                    for (int k = s.sig_row_ptr[i]; k < s.sig_row_ptr[i + 1]; ++k) qok[s.sig_row[k]] = 1;
                    for (int k = s.sig_red_ptr[i]; k < s.sig_red_ptr[i + 1]; ++k) red_count(s, s.sig_red[k]);
                    if (eng && s.fwd_trig >= 0) {      // relay: the engine forwards the arrived input inside the node
                        const int t = s.fwd_trig + i, l = eng->trig_lane ? eng->trig_lane[t] : 1;
                        if ((l & 1) && eng->trig_ptr[t + 1] > eng->trig_ptr[t]) pq_push(*eng, t);
                    }
                    progress = 1;
                }
            }
            unsigned cm = __ballot_sync(0xffffffffu, coop);
            while (cm) {                            // the long lists of this chunk, one by one, lanes over entries
                const int b = __ffs(cm) - 1; cm &= cm - 1;
                const int j = __shfl_sync(0xffffffffu, i, b);
                const int k0 = s.sig_grp_ptr[j], k1 = s.sig_grp_ptr[j + 1];
                for (int k = k0 + lane; k < k1; k += 32) mark_grp(s.sig_grp[k]);
            }
        }
        __syncwarp();
        const unsigned long long tb_ = s.tl_ev ? globaltimer() : 0;
        // 2. dispatch
        int depth = 0;
        if (lane == 0) {
            s_nb = 0;
            const int claimed = *(const volatile int *)s.tile_counter + s.n_static;
            if (s.q_work) {
                while (rb_head != rb_tail && rb_base[rb_head % kSchedRing] + rb_n[rb_head % kSchedRing] <= claimed) {
                    rb_total -= rb_w[rb_head % kSchedRing]; ++rb_head;
                }
                depth = rb_total;
                if (rb_head != rb_tail) {
                    const int i = rb_head % kSchedRing, done = claimed - rb_base[i];
                    if (done > 0) depth -= (int)((long long)rb_w[i] * done / rb_n[i]);
                }
                if (rb_tail - rb_head >= kSchedRing) depth = 1 << 30;
            } else {
                depth = *(const volatile int *)s.sq_tail - claimed;
            }
            while (cursor < nr && rem[s.row_order[cursor]] == 0) ++cursor;
        }
        depth = __shfl_sync(0xffffffffu, depth, 0);
        cursor = __shfl_sync(0xffffffffu, cursor, 0);
        const int row_lim = kAvail == 2 ? __shfl_sync(0xffffffffu, *(const volatile int *)s.row_avail, 0) : nr;
        int nb = 0, nbw = 0;
        auto queued = [&]() { return depth + (s.q_work ? nbw : nb * s.sub); };
        for (int tier = 0; tier < 2; ++tier) {
            const int thr = tier == 0 ? s.sched_low : s.sched_crit;
            if (queued() >= thr) continue;
            for (int k = cursor; k < nr && nb < s.sched_batch && queued() < thr; k += 32) {
                ++st_scan;
                int cand = 0;
                if (k + lane < row_lim) {
                    const int r = s.row_order[k + lane];
                    if (rem[r] > 0 && qok[r]) {
                        int ready_n = 0, waiting = 0;
                        for (int g = s.row_grp_ptr[r]; g < s.row_grp_ptr[r + 1]; ++g) {
                            const bool rd = ready(g);
                            ready_n += rd; waiting += gst[g] == 0 && !rd;
                        }
                        cand = ready_n > 0 && (tier == 1 || waiting == 0);
                    }
                }
                unsigned m = __ballot_sync(0xffffffffu, cand);
                // every candidate a single-group row (rectangle mode: one group per row): the lanes write their own task, the
                // rank among the candidates = the serial loop's order, the same quota (batch room, queue depth below thr),
                // so dispatching many small tasks is not limited by one lane
                const int r1 = cand ? s.row_order[k + lane] : 0;
                const bool single = !cand || s.row_grp_ptr[r1 + 1] - s.row_grp_ptr[r1] == 1;
                if (!s.q_work && m && __all_sync(0xffffffffu, single)) {   // (work-weighted depth: serial path)
                    int quota = 0, nt0 = 0;
                    if (lane == 0) {
                        const int need = thr - queued();
                        quota = need > 0 ? min(s.sched_batch - nb, (need + s.sub - 1) / s.sub) : 0;
                        nt0 = n_tasks;
                    }
                    quota = __shfl_sync(0xffffffffu, quota, 0); nt0 = __shfl_sync(0xffffffffu, nt0, 0);
                    const int rank = __popc(m & ((1u << lane) - 1));
                    const bool take = cand && rank < quota;
                    int wk = 0;
                    if (take) {
                        const int g0 = s.row_grp_ptr[r1];
                        wk = (s.grp_work ? s.grp_work[g0] : s.grp_end[g0] - s.grp_beg[g0]) * s.sub;
                        gst[g0] = 2; rem[r1] -= 1;
                        const bool whole = rnt[r1] == 0;
                        if (s.tl_row && rnt[r1] == 0) s.tl_row[2 * (int64_t)r1] = globaltimer();
                        const int out = (whole || !s.row_pool) ? -1 : s.row_pool[r1] + rnt[r1];
                        ++rnt[r1];
                        s.row_ntask[r1] = rnt[r1];
                        int *d = s.task_desc + 4 * (nt0 + rank);
                        d[0] = r1; d[1] = s.grp_beg[g0]; d[2] = s.grp_end[g0]; d[3] = out;
                        batch[nb + rank] = nt0 + rank;
                    }
                    for (int o = 16; o; o >>= 1) wk += __shfl_xor_sync(0xffffffffu, wk, o);
                    const int cnt = min(__popc(m), quota);
                    nb += cnt; nbw += wk;
                    if (lane == 0) n_tasks += cnt;
                    st_fast += cnt;
                    __syncwarp();
                    continue;
                }
                if (lane == 0) {
                    while (m && nb < s.sched_batch && queued() < thr) {
                        const int r = s.row_order[k + __ffs(m) - 1];
                        m &= m - 1;
                        for (int g0 = s.row_grp_ptr[r]; g0 < s.row_grp_ptr[r + 1] && nb < s.sched_batch; ++g0) {
                            if (!ready(g0)) continue;
                            int g1 = g0, work = s.grp_work ? s.grp_work[g0] : s.grp_end[g0] - s.grp_beg[g0];
                            while (g1 + 1 < s.row_grp_ptr[r + 1] && ready(g1 + 1) && (kAvail != 1 || g1 - g0 + 1 < s.grp_max)) {
                                ++g1; work += s.grp_work ? s.grp_work[g1] : s.grp_end[g1] - s.grp_beg[g1];
                            }
                            for (int g = g0; g <= g1; ++g) gst[g] = 2;
                            rem[r] -= g1 - g0 + 1;
                            const bool whole = rnt[r] == 0 && g0 == s.row_grp_ptr[r] && g1 + 1 == s.row_grp_ptr[r + 1];
                            if (s.tl_row && rnt[r] == 0) s.tl_row[2 * (int64_t)r] = globaltimer();   // first dispatch
                            const int out = (whole || !s.row_pool) ? -1 : s.row_pool[r] + rnt[r];
                            ++rnt[r];
                            s.row_ntask[r] = rnt[r];
                            int *d = s.task_desc + 4 * n_tasks;
                            d[0] = r; d[1] = s.grp_beg[g0]; d[2] = s.grp_end[g1]; d[3] = out;
                            nbw += work * s.sub;
                            ++st_slow;
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
            s_base = atomicAdd(s.sq_tail, nb * s.sub);
            if (s.q_work) {
                const int i = rb_tail % kSchedRing;
                rb_base[i] = s_base; rb_n[i] = nb * s.sub; rb_w[i] = nbw; ++rb_tail; rb_total += nbw;
            }
            s_nb = nb;
            if (s.trace) { s.trace[11] = n_tasks; s.trace[13] = globaltimer() - s.trace[8]; }
            if (s.tl_ev) atomicCAS(&s.tl_ev[2], 0ULL, globaltimer());
        }
        __syncwarp();
        const unsigned long long tc_ = s.tl_ev ? globaltimer() : 0;
        if (s_nb) {
            // entry k: gang block b = k / (nb * gang), then task, then the gang's member -- gang 1 = sub-major across the
            // batch (the forward's head-major order: many rows of one head share KV in L2); gang 2 with sub 2 = each
            // task's two entries adjacent (the backward's twin halves start together)
            const int nb_ = s_nb, base = s_base, gw = nb_ * s.gang;
            for (int k = lane; k < nb_ * s.sub; k += 32) {
                const int b = k / gw, rr = k % gw, t = rr / s.gang, sb = b * s.gang + rr % s.gang;
                atomicExch(&s.sq[base + k], sq_entry(s.kind_task, batch[t] * s.sub + sb));
            }
            progress = 1;
        }
        __syncwarp();
        if (s.tl_ev) {
            const unsigned long long td_ = globaltimer();
            ++st_it; st_dit += s_nb > 0; st_arr += tb_ - ta_; st_dis += tc_ - tb_; st_q += td_ - tc_;
        }
        if (lane == 0) {
            if (*(const volatile int *)s.rows_done == nr && *(const volatile int *)s.n_enq == s.n_red) {
                sq_push(s, s.kind_stop, 0, s.n_workers * s.gang);
                stopped = 1;
                if (s.tl_ev) {
                    s.tl_ev[3] = globaltimer();
                    s.tl_ev[4] = st_it; s.tl_ev[5] = st_dit; s.tl_ev[6] = st_arr; s.tl_ev[7] = st_dis; s.tl_ev[8] = st_q;
                    s.tl_ev[9] = st_fast; s.tl_ev[10] = st_slow; s.tl_ev[11] = st_scan;
                }
                if (s.trace) { int sp = 0; for (int r = 0; r < nr; ++r) sp += rnt[r] > 1; s.trace[12] = sp; }
            }
        }
        stopped = __shfl_sync(0xffffffffu, stopped, 0);
        progress = __any_sync(0xffffffffu, progress);
        if (!progress && !stopped) {
            __nanosleep(128);
            if (lane == 0 && ++idle % (RT_SPIN_REPORT / 4) == 0) {
                int g0n = 0, g1n = 0, g2n = 0, nq = 0, sn = 0;
                for (int g = 0; g < ng; ++g) { const bool rd = ready(g); g0n += gst[g] == 0 && !rd; g1n += rd; g2n += gst[g] == 2; }
                for (int r = 0; r < nr; ++r) nq += !qok[r];
                for (int i = 0; i < ns; ++i) sn += !(sdone[i >> 5] & (1u << (i & 31)));
                printf("[pe %d rt sched] idle: rows done %d/%d, units queued %d/%d, tasks %d | groups wait/ready/issued "
                       "%d/%d/%d, rows without their input %d, signals pending %d/%d\n", nvshmem_my_pe(),
                       *(const volatile int *)s.rows_done, nr, *(const volatile int *)s.n_enq, s.n_red, n_tasks, g0n, g1n,
                       g2n, nq, sn, ns);
            }
        }
    }
}

}  // namespace rt
