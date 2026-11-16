#pragma once
// Distributed sparse-attention BACKWARD megakernel (one launch per GPU). See dist_bwd_params.h.
// Compute = sparse_bwd::bwd_task (FA2's sm80 backward mainloop, list-driven) unchanged; this file adds the block-0
// scheduler, the input push, slot completion + partial-sum push, and the owner sums.
#include "sparse_bwd_kernel.h"
#include "dist_bwd_params.h"
#include "../ring_attn/nvshmem_transport.cuh"
#include "../runtime/rt_engine.cuh"
#include "../runtime/rt_sched.cuh"

#ifndef DIST_SPIN_REPORT
#define DIST_SPIN_REPORT (1L << 26)
#endif

namespace dist_bwd {

enum : int { kTask = 0, kQSum = 1, kKSum = 2, kStop = 3, kNop = 4, kFwd = 5, kQAgg = 6, kKAgg = 7 };
__device__ __forceinline__ int sq_entry(int kind, int payload) { return ((kind << 28) | payload) + 1; }
// Workers take queue entries in PAIRS (a gang of two blocks, or one block taking both), so every push is an even
// number of entries starting at an even position (an odd push is padded with a no-op entry).
__device__ __forceinline__ void sq_push(const Params &p, int kind, int payload0, int n) {
    const int n2 = (n + 1) & ~1;
    const int base = atomicAdd(p.sq_tail, n2);
    for (int i = 0; i < n2; ++i) atomicExch(&p.sq[base + i], i < n ? sq_entry(kind, payload0 + i) : sq_entry(kNop, 0));
}
// push-queue triggers (the message table's trigger space, dex/scheduler/bwd.py): partial sum of Q slot s, of KV
// slot s, forwards of remote signal j
__device__ __forceinline__ int trig_q(int s) { return s; }
__device__ __forceinline__ int trig_k(const Params &p, int s) { return p.NQS + s; }
__device__ __forceinline__ int trig_fwd(const Params &p, int j) { return p.NQS + p.NKS + j; }
__device__ __forceinline__ int sig_dql(const Params &p, int l) { return p.NQS + p.NKS + l; }
__device__ __forceinline__ int sig_dkl(const Params &p, int l) { return p.NQS + p.NKS + p.NDQL + l; }

// ---- block 0: scheduler (warp 0) -----------------------------------------------------------------------------------
// Tasks are indexed in PRIORITY order (the builder's: KV-head-major, then KV tile, then input group -- the order that
// lets concurrent tasks share a head's Q / dO in L2). An arrival only marks tasks ready (task_cnt == task_need);
// tasks are DISPATCHED in priority order, and only while fewer than sched_low entries wait in the queue -- so the
// queue keeps the priority order instead of the signals' arrival order (as the forward's block-0 scheduler).
// Owner sums (cheap, at the end of a slot's life) are queued as soon as their inputs are in.
__device__ __noinline__ void scheduler(const Params &p, uint32_t *done_bits, uint32_t *disp_bits) {
    if (threadIdx.x >= 32) return;
    const int lane = threadIdx.x;
    for (int w = lane; w < (p.n_remote + 31) / 32; w += 32) done_bits[w] = 0u;
    for (int w = lane; w < (p.n_tasks / 2 + 31) / 32; w += 32) disp_bits[w] = 0u;   // per twin pair
    __syncwarp();
    int cursor = 0, dispatched = 0;                        // pairs before cursor are all dispatched (lane-uniform)
    long idle = 0;
    for (;;) {
        int progress = 0;
        // 1. arrivals
        for (int i = lane; i < p.n_remote; i += 32) {
            if (done_bits[i >> 5] & (1u << (i & 31))) continue;
            if (*(const volatile uint64_t *)&p.sig[p.remote_sig[i]] < p.sig_epoch) continue;
            __threadfence_system();
            atomicOr(&done_bits[i >> 5], 1u << (i & 31));
            for (int k = p.sig_task_ptr[i]; k < p.sig_task_ptr[i + 1]; ++k) atomicAdd(&p.task_cnt[p.sig_task[k]], 1);
            if (p.sig_qm[i] >= 0) {
                const int m = p.sig_qm[i];
                if (atomicAdd(&p.qm_cnt[m], 1) + 1 == p.qm_need[m]) {
                    __threadfence(); sq_push(p, kQSum, m * 2 * p.h, 2 * p.h); atomicAdd(p.n_enq, 1);
                }
            }
            if (p.sig_km[i] >= 0) {
                const int m = p.sig_km[i];
                if (atomicAdd(&p.km_cnt[m], 1) + 1 == p.km_need[m]) {
                    __threadfence(); sq_push(p, kKSum, m * p.hk, p.hk); atomicAdd(p.n_enq, 1);
                }
            }
            if (p.sig_qa[i] >= 0) {                              // a node peer's partial landed at this gateway
                const int s = p.sig_qa[i];
                if (atomicAdd(&p.qa_cnt[s], 1) + 1 == p.qs_agg_need[s]) {
                    __threadfence(); sq_push(p, kQAgg, s, 1); atomicAdd(p.n_enq, 1);
                }
            }
            if (p.sig_ka[i] >= 0) {
                const int s = p.sig_ka[i];
                if (atomicAdd(&p.ka_cnt[s], 1) + 1 == p.ks_agg_need[s]) {
                    __threadfence(); sq_push(p, kKAgg, s, 1); atomicAdd(p.n_enq, 1);
                }
            }
            if (p.sig_fwd[i]) rt::pq_push(p.eng, trig_fwd(p, i));            // an input to forward inside the node
            progress = 1;
        }
        __syncwarp();
        // 2. dispatch: while the queue is short, the next ready TWIN PAIRS (tasks 2k, 2k+1) in priority order; a pair
        //    is pushed as two adjacent entries at an even queue position, so one worker gang starts both together
        int depth = 0;
        if (lane == 0) depth = *(volatile int *)p.sq_tail - *(volatile int *)p.tile_counter;
        depth = __shfl_sync(0xffffffffu, depth, 0);
        const int n_pairs = p.n_tasks >> 1;
        for (int k = cursor; k < n_pairs && depth < p.sched_low; k += 32) {
            const int q = k + lane;
            const bool cand = q < n_pairs && !(disp_bits[q >> 5] & (1u << (q & 31))) &&
                              *(volatile int *)&p.task_cnt[2 * q] >= p.task_need[2 * q] &&
                              *(volatile int *)&p.task_cnt[2 * q + 1] >= p.task_need[2 * q + 1];
            unsigned m = __ballot_sync(0xffffffffu, cand);
            const int room = (p.sched_low - depth + 1) >> 1;
            if (__popc(m) > room) {                             // keep the first `room` in priority order
                unsigned keep = 0; for (int c = 0; c < room; ++c) { keep |= m & (~m + 1); m &= m - 1; }
                m = keep;
            }
            const int nb = __popc(m);
            if (nb) {
                int base = 0;
                if (lane == 0) base = atomicAdd(p.sq_tail, 2 * nb);
                base = __shfl_sync(0xffffffffu, base, 0);
                if (m & (1u << lane)) {
                    const int pos = 2 * __popc(m & ((1u << lane) - 1));
                    atomicOr(&disp_bits[q >> 5], 1u << (q & 31));
                    __threadfence();
                    atomicExch(&p.sq[base + pos], sq_entry(kTask, 2 * q));
                    atomicExch(&p.sq[base + pos + 1], sq_entry(kTask, 2 * q + 1));
                }
                depth += 2 * nb; dispatched += 2 * nb; progress = 1;
            }
            __syncwarp();
        }
        while (cursor < n_pairs && (disp_bits[cursor >> 5] & (1u << (cursor & 31)))) ++cursor;   // lane-uniform
        // 3. done
        int stop = 0;
        if (lane == 0 && dispatched == p.n_tasks && *(volatile int *)p.n_enq == p.n_qm + p.n_km + p.n_xenq) {
            sq_push(p, kStop, 0, 2 * p.n_workers);          // every gang member / solo worker sees a stop
            stop = 1;
        }
        if (__shfl_sync(0xffffffffu, stop, 0)) break;
        progress = __any_sync(0xffffffffu, progress);
        if (!progress) {
            __nanosleep(128);
            if (lane == 0 && ++idle % (DIST_SPIN_REPORT / 4) == 0) {
                int pend = 0, shown = 0;
                for (int i = 0; i < p.n_remote; ++i) {
                    if (done_bits[i >> 5] & (1u << (i & 31))) continue;
                    ++pend;
                    if (shown++ < 4) printf("[pe %d bwd sched] pending signal %d\n", nvshmem_my_pe(), p.remote_sig[i]);
                }
                printf("[pe %d bwd sched] idle: tasks dispatched %d/%d, owner sums queued %d/%d, signals pending %d/%d\n",
                       nvshmem_my_pe(), dispatched, p.n_tasks, *(volatile int *)p.n_enq, p.n_qm + p.n_km, pend, p.n_remote);
            }
        }
    }
}

// ---- owner sums -------------------------------------------------------------------------------------------------------
// dQ: rows [mb*64, +64) of head hq of owned tile out_row = scale * (sum of inputs), un-permuting FA2's dQ order.
template <typename Kernel_traits>
__device__ __noinline__ void qsum(const Params &p, int m, int mb, int hq) {
    using namespace cute;
    constexpr int kBlockM = Kernel_traits::kBlockM, kHeadDim = Kernel_traits::kHeadDim;
    const int64_t row_stride = (int64_t)p.h * p.d;
    const int64_t in_off = (int64_t)mb * kBlockM * row_stride + (int64_t)hq * p.d;     // floats inside a slot
    typename Kernel_traits::GmemTiledCopydQaccumAtomicAdd cp;
    auto thr = cp.get_thread_slice(threadIdx.x);
    float acc[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) acc[i] = 0.f;
    auto add = [&](const float *base) {
        Tensor g = make_tensor(make_gmem_ptr(base + in_off), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(row_stride, _1{}));
        Tensor tg = thr.partition_D(g);
        static_assert(decltype(size(tg))::value == 32);
        #pragma unroll
        for (int i = 0; i < 32; ++i) acc[i] += tg(i);
    };
    if (p.qm_local[m] >= 0) add(reinterpret_cast<const float *>(p.dqa + (int64_t)p.qm_local[m] * p.dqa_bytes));
    for (int k = p.qm_ptr[m]; k < p.qm_ptr[m + 1]; ++k)
        add(reinterpret_cast<const float *>(p.dql + (int64_t)p.qm_land[k] * p.dqa_bytes));
    typename Kernel_traits::TiledMmadQ mma;
    Tensor tc = mma.get_thread_slice(threadIdx.x).partition_C(make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    __half *out = reinterpret_cast<__half *>(p.dq_out) + ((int64_t)p.qm_out[m] * 128 + mb * kBlockM) * row_stride + hq * p.d;
    #pragma unroll
    for (int i = 0; i < 32; ++i) out[(int64_t)get<0>(tc(i)) * row_stride + get<1>(tc(i))] = __float2half_rn(acc[i] * p.scale);
}
// dK, dV: KV head g of owned tile out_row = sum of inputs (dK already scaled in the task epilogue)
__device__ __noinline__ void ksum(const Params &p, int m, int g) {
    const int64_t row_stride = (int64_t)p.hk * p.d, half_floats = 128 * row_stride;
    const int nin = (p.km_local[m] >= 0) + p.km_ptr[m + 1] - p.km_ptr[m];
    for (int e = threadIdx.x; e < 128 * p.d; e += blockDim.x) {
        const int r = e / p.d, c = e % p.d;
        const int64_t o = (int64_t)r * row_stride + (int64_t)g * p.d + c;
        float sk = 0.f, sv = 0.f;
        for (int k = 0; k < nin; ++k) {
            const bool loc = p.km_local[m] >= 0 && k == 0;
            const int slot = loc ? p.km_local[m] : p.km_land[p.km_ptr[m] + k - (p.km_local[m] >= 0)];
            const float *b = reinterpret_cast<const float *>((loc ? p.dkva : p.dkl) + (int64_t)slot * p.dkva_bytes);
            sk += b[o]; sv += b[half_floats + o];
        }
        const int64_t oo = (int64_t)p.km_out[m] * 128 * row_stride + o;
        reinterpret_cast<__half *>(p.dk_out)[oo] = __float2half_rn(sk);
        reinterpret_cast<__half *>(p.dv_out)[oo] = __float2half_rn(sv);
    }
}

__device__ __forceinline__ unsigned long long bwd_now() {
    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
#define BTR(p, i, v) do { if ((p).trace && threadIdx.x == 0) atomicAdd(&(p).trace[i], (unsigned long long)(v)); } while (0)

// A finished compute task: count it into its Q / KV slots; the block completing a slot sends its partial sum (trigger) or
// counts it into the owner sum / gateway aggregation; the second finished half of a twin pair completes the row.
template <typename Ctl_t>
__device__ __noinline__ void complete_task(const Params &p, int t, int ks, Ctl_t &c) {
    const int tidx = threadIdx.x;
    if (p.rect) {                                          // the group's second half queues it for block 0's warp 1
        if (tidx == 0) {
            __threadfence();                               // release: this block's atomics before the count
            if (atomicAdd(&p.row_cnt[t >> 1], 1) == 1) {
                const int pos = atomicAdd(p.done_tail, 1);
                atomicExch(&p.done_q[pos], (t >> 1) + 1);
            }
        }
        __syncthreads();
        return;
    }
    if (tidx == 0) {
        __threadfence();                                   // release: this block's atomics before the count
        int nq = 0;
        for (int k = p.task_qs_ptr[t]; k < p.task_qs_ptr[t + 1]; ++k) {
            const int s = p.task_qs[k];
            if (atomicAdd(&p.qs_done[s], 1) + 1 == p.qs_need[s]) c.qdone[nq++ & 63] = s;
        }
        c.nq = nq;
        c.kdone = atomicAdd(&p.ks_done[ks], 1) + 1 == p.ks_need[ks];
        if (nq || c.kdone) __threadfence();                // acquire: every other block's atomics on them
    }
    __syncthreads();
    const int nq = c.nq, kdone = c.kdone;
    for (int k = 0; k < nq; ++k) {                         // (a task touches <= 64 completed slots at once:
        const int s = c.qdone[k & 63];                     //  the builder caps task_qs at 64)
        if (p.sch_on) {                                    // reduction units: rt::red_count
            if (tidx == 0) {
                if (p.qs_agg_need[s] > 0) rt::red_count(p.sch, p.qa_unit[s]);
                else if (p.qs_push_pe[s] >= 0) rt::fire(p.eng, trig_q(s));
                else rt::red_count(p.sch, p.qs_push_dst[s]);
            }
        } else if (p.qs_agg_need[s] > 0) {                 // node gateway: wait for the node's landings
            if (tidx == 0 && atomicAdd(&p.qa_cnt[s], 1) + 1 == p.qs_agg_need[s]) {
                __threadfence(); sq_push(p, kQAgg, s, 1); atomicAdd(p.n_enq, 1);
            }
        } else if (p.qs_push_pe[s] >= 0) {
            if (tidx == 0) rt::fire(p.eng, trig_q(s));
        } else if (tidx == 0) {
            const int m = p.qs_push_dst[s];
            if (atomicAdd(&p.qm_cnt[m], 1) + 1 == p.qm_need[m]) {
                __threadfence(); sq_push(p, kQSum, m * 2 * p.h, 2 * p.h); atomicAdd(p.n_enq, 1);
            }
        }
    }
    if (kdone && p.sch_on) {
        if (tidx == 0) {
            if (p.ks_agg_need[ks] > 0) rt::red_count(p.sch, p.ka_unit[ks]);
            else if (p.ks_push_pe[ks] >= 0) rt::fire(p.eng, trig_k(p, ks));
            else rt::red_count(p.sch, p.n_qm + p.ks_push_dst[ks]);
        }
    } else if (kdone) {
        if (p.ks_agg_need[ks] > 0) {
            if (tidx == 0 && atomicAdd(&p.ka_cnt[ks], 1) + 1 == p.ks_agg_need[ks]) {
                __threadfence(); sq_push(p, kKAgg, ks, 1); atomicAdd(p.n_enq, 1);
            }
        } else if (p.ks_push_pe[ks] >= 0) {
            if (tidx == 0) rt::fire(p.eng, trig_k(p, ks));
        } else if (tidx == 0) {
            const int m = p.ks_push_dst[ks];
            if (atomicAdd(&p.km_cnt[m], 1) + 1 == p.km_need[m]) {
                __threadfence(); sq_push(p, kKSum, m * p.hk, p.hk); atomicAdd(p.n_enq, 1);
            }
        }
    }
    if (p.sch_on && tidx == 0 && atomicAdd(&p.row_cnt[t >> 1], 1) == 1) {   // the twin pair is done
        __threadfence(); atomicAdd(p.sch.rows_done, 1);
    }
    __syncthreads();
}

// ---- rectangle mode (p.rect): block 0's warp 1 and the workers' two hooks ------------------------------------------
__device__ __forceinline__ void q_slot_done(const Params &p, int s) {          // (as complete_task, shared scheduler)
    if (p.ml) p.ml[s] = bwd_now();
    if (p.qs_agg_need[s] > 0) rt::red_count(p.sch, p.qa_unit[s]);
    else if (p.qs_push_pe[s] >= 0) rt::fire(p.eng, trig_q(s));
    else rt::red_count(p.sch, p.qs_push_dst[s]);
}
__device__ __forceinline__ void k_slot_done(const Params &p, int ks) {
    if (p.ml) p.ml[p.NQS + ks] = bwd_now();
    if (p.ks_agg_need[ks] > 0) rt::red_count(p.sch, p.ka_unit[ks]);
    else if (p.ks_push_pe[ks] >= 0) rt::fire(p.eng, trig_k(p, ks));
    else rt::red_count(p.sch, p.n_qm + p.ks_push_dst[ks]);
}

// Rows of group g1 into rb[0, cnt) (tile << 1 | causal diagonal; warp): the rectangle's pairs (row block
// descending), set bits of column b descending; ranks [k0, k0 + cnt). Returns cnt.
__device__ __forceinline__ int rect_rows(const Params &p, int g1, int lane, int32_t *rb) {
    const int R = p.g_rect[g1], b = p.g_col[g1], k0 = p.g_k0[g1], cnt = p.g_cnt[g1], per = p.per, nw = p.nw;
    int base = 0;
    for (int x = p.rect_pp[R]; x < p.rect_pp[R + 1] && base < k0 + cnt; ++x) {
        const int q = p.rect_pair[x], bq = p.pair_bq[q];
        const uint64_t *bw = p.pair_bits + (int64_t)q * nw, *dw = p.pair_dbits + (int64_t)q * nw;
        for (int c = 0; c < per; c += 32) {
            const int a = per - 1 - c - lane;
            bool set = false, dg = false; int hh = 0;
            if (a >= 0) {
                const int k = a * per + b; set = (bw[k >> 6] >> (k & 63)) & 1; dg = (dw[k >> 6] >> (k & 63)) & 1;
                if (p.pair_hbits && set) {                 // 128 x 64: which 64-key halves of the tile are live
                    const uint64_t *hw = p.pair_hbits + (int64_t)q * p.nw2; const int k2 = a * 2 * per + 2 * b;
                    hh = (int)((hw[k2 >> 6] >> (k2 & 63)) & 1) | ((int)((hw[(k2 + 1) >> 6] >> ((k2 + 1) & 63)) & 1) << 1);
                }
            }
            const unsigned m = __ballot_sync(0xffffffffu, set);
            const int rank = base + __popc(m & ((1u << lane) - 1));
            if (set && rank >= k0 && rank < k0 + cnt)          // tile << 1 | diag, or (half mode) tile << 3 | halves << 1 | diag
                rb[rank - k0] = p.pair_hbits ? (((bq * per + a) << 3) | (hh << 1) | (int)dg) : (((bq * per + a) << 1) | (int)dg);
            base += __popc(m);
        }
    }
    __syncwarp();
    return cnt;
}

// Materialise group g: its tasks 2 g + half in the per-tile format (task_kv; items [task_ptr, task_end) in the arena,
// allocated here) -- the workers' code is the per-tile mode's (warp). Per row, half 0: (m block 1, causal 0),
// (0, diagonal); half 1: (1, diagonal) and, off the diagonal, (0, 0); q heads innermost -- exactly the per-tile
// planner's lists (sparse_bwd item = q_slot << 9 | mb << 8 | hq << 1 | causal).
// 128 x 64 (p.pair_hbits): rect_expand with each row's items only in the halves it has live (rb = tile << 3 | h1 << 2 |
// h0 << 1 | diag): half 0 rows (h0): (mb 1, causal 0), (0, diag); half 1 rows (h1): (1, diag) and, off the diagonal,
// (0, 0). Lanes over rows, q heads inner (warp). A half with no row gets an empty task (bwd_task returns at once).
__device__ __noinline__ void rect_expand_half(const Params &p, int g, int lane, int qpg, int32_t *rb) {
    const int qg = g / p.G1, g1 = g - qg * p.G1, hq0 = qg * qpg, head = hq0 / (p.h / p.hk);
    const int cnt = rect_rows(p, g1, lane, rb);
    const int r0 = lane < cnt ? rb[lane] : 0, r1 = lane + 32 < cnt ? rb[lane + 32] : 0;
    const unsigned a0 = __ballot_sync(0xffffffffu, lane < cnt && (r0 & 2)), a1 = __ballot_sync(0xffffffffu, lane + 32 < cnt && (r1 & 2));
    const unsigned b0 = __ballot_sync(0xffffffffu, lane < cnt && (r0 & 4)), b1 = __ballot_sync(0xffffffffu, lane + 32 < cnt && (r1 & 4));
    const unsigned c0 = __ballot_sync(0xffffffffu, lane < cnt && (r0 & 4) && (r0 & 1));
    const unsigned c1 = __ballot_sync(0xffffffffu, lane + 32 < cnt && (r1 & 4) && (r1 & 1));
    const int n0 = 2 * (__popc(a0) + __popc(a1)) * qpg;
    const int n1 = (2 * (__popc(b0) + __popc(b1)) - __popc(c0) - __popc(c1)) * qpg;
    int off = lane == 0 ? atomicAdd(p.arena_off, n0 + n1) : 0;
    off = __shfl_sync(0xffffffffu, off, 0);
    int32_t *it = p.arena + off;
    for (int hf = 0; hf < 2; ++hf) {
        const int li = lane + 32 * hf;
        if (li >= cnt) continue;
        const int row = hf ? r1 : r0, dg = row & 1, h0 = (row >> 1) & 1, h1 = (row >> 2) & 1, qs = p.qslot_t[row >> 3];
        const unsigned lt = (1u << lane) - 1;
        const int na = hf ? __popc(a0) + __popc(a1 & lt) : __popc(a0 & lt);          // half-0 rows before li
        const int nb_ = hf ? __popc(b0) + __popc(b1 & lt) : __popc(b0 & lt);         // half-1 rows before li
        const int nc = hf ? __popc(c0) + __popc(c1 & lt) : __popc(c0 & lt);          // half-1 diagonal rows before li
        int32_t *o0 = it + 2 * na * qpg, *o1 = it + n0 + (2 * nb_ - nc) * qpg;
        for (int hq = hq0; hq < hq0 + qpg; ++hq) {
            if (h0) { o0[hq - hq0] = (qs << 9) | (1 << 8) | (hq << 1); o0[qpg + hq - hq0] = (qs << 9) | (hq << 1) | dg; }
            if (h1) { o1[hq - hq0] = (qs << 9) | (1 << 8) | (hq << 1) | dg; if (!dg) o1[qpg + hq - hq0] = (qs << 9) | (hq << 1); }
        }
    }
    if (lane == 0) {
        int32_t *tkv = const_cast<int32_t *>(p.task_kv), *tp = const_cast<int32_t *>(p.task_ptr);
        int32_t *te = const_cast<int32_t *>(p.task_end);
        const int ks = p.g_ks[g1];
        tkv[2 * g] = (ks << 9) | head; tkv[2 * g + 1] = (ks << 9) | (1 << 8) | head;
        tp[2 * g] = off; te[2 * g] = off + n0; tp[2 * g + 1] = off + n0; te[2 * g + 1] = off + n0 + n1;
    }
    __syncwarp();
}

__device__ __forceinline__ void rect_expand(const Params &p, int g, int lane, int qpg, int32_t *rb) {
    if (p.pair_hbits) { rect_expand_half(p, g, lane, qpg, rb); return; }
    // g = qg * G1 + g1: q-head group qg = q heads [hq0, hq0 + qpg) of KV head `head` (qpg = h / hg; hg = hk -> the
    // original one-group-per-KV-head decomposition, hg = h -> one q head per group as FA2)
    const int qg = g / p.G1, g1 = g - qg * p.G1, hq0 = qg * qpg, head = hq0 / (p.h / p.hk);
    const int cnt = rect_rows(p, g1, lane, rb);
    const int r0 = lane < cnt ? rb[lane] : 0, r1 = lane + 32 < cnt ? rb[lane + 32] : 0;
    const unsigned d0 = __ballot_sync(0xffffffffu, lane < cnt && (r0 & 1));
    const unsigned d1 = __ballot_sync(0xffffffffu, lane + 32 < cnt && (r1 & 1));
    const int n0 = 2 * cnt * qpg, n1 = (2 * cnt - __popc(d0) - __popc(d1)) * qpg;
    int off = lane == 0 ? atomicAdd(p.arena_off, n0 + n1) : 0;
    off = __shfl_sync(0xffffffffu, off, 0);
    int32_t *it = p.arena + off;
    if (qpg >= 8) {                                        // many q heads: lanes over one row's items (coalesced)
        for (int li = 0; li < cnt; ++li) {
            const int row = li < 32 ? __shfl_sync(0xffffffffu, r0, li) : __shfl_sync(0xffffffffu, r1, li - 32);
            const int dg = row & 1, qs = p.qslot_t[row >> 1];
            const int ndb = li < 32 ? __popc(d0 & ((1u << li) - 1)) : __popc(d0) + __popc(d1 & ((1u << (li - 32)) - 1));
            int32_t *o0 = it + 2 * li * qpg, *o1 = it + n0 + (2 * li - ndb) * qpg;
            for (int e = lane; e < 2 * qpg; e += 32) {
                const int mb = e < qpg ? 1 : 0, hq = hq0 + (e < qpg ? e : e - qpg);
                o0[e] = (qs << 9) | (mb << 8) | (hq << 1) | (mb ? 0 : dg);
                if (e < (dg ? qpg : 2 * qpg)) o1[e] = (qs << 9) | (mb << 8) | (hq << 1) | (mb ? dg : 0);
            }
        }
    } else {                                               // few q heads: lanes over rows
        for (int half = 0; half < 2; ++half) {
            const int li = lane + 32 * half;
            if (li >= cnt) continue;
            const int row = half ? r1 : r0, dg = row & 1, qs = p.qslot_t[row >> 1];
            const unsigned lt = (1u << lane) - 1;
            const int ndb = half ? __popc(d0) + __popc(d1 & lt) : __popc(d0 & lt);   // diagonal rows before li
            int32_t *o0 = it + 2 * li * qpg, *o1 = it + n0 + (2 * li - ndb) * qpg;
            for (int hq = hq0; hq < hq0 + qpg; ++hq) {
                o0[hq - hq0] = (qs << 9) | (1 << 8) | (hq << 1);
                o0[qpg + hq - hq0] = (qs << 9) | (hq << 1) | dg;
                o1[hq - hq0] = (qs << 9) | (1 << 8) | (hq << 1) | dg;
                if (!dg) o1[qpg + hq - hq0] = (qs << 9) | (hq << 1);
            }
        }
    }
    if (lane == 0) {
        int32_t *tkv = const_cast<int32_t *>(p.task_kv), *tp = const_cast<int32_t *>(p.task_ptr);
        int32_t *te = const_cast<int32_t *>(p.task_end);
        const int ks = p.g_ks[g1];
        tkv[2 * g] = (ks << 9) | head; tkv[2 * g + 1] = (ks << 9) | (1 << 8) | head;
        tp[2 * g] = off; te[2 * g] = off + n0; tp[2 * g + 1] = off + n0; te[2 * g + 1] = off + n0 + n1;
    }
    __syncwarp();
}

// Group g finished (both halves): count it into the Q slots of its rows and its column's KV slot; a completed slot
// sends its partial sum or counts into its reduction unit (warp)
__device__ __forceinline__ void rect_group_done(const Params &p, int g, int lane, int32_t *rb) {
    const int g1 = g % p.G1, cnt = rect_rows(p, g1, lane, rb);
    for (int li = lane; li < cnt; li += 32) {
        const int s = p.qslot_t[rb[li] >> (p.pair_hbits ? 3 : 1)];
        if (atomicAdd(&p.qs_done[s], 1) + 1 == p.qs_need[s]) { __threadfence(); q_slot_done(p, s); }
    }
    if (lane == 0) {
        const int ks = p.g_ks[g1];
        if (atomicAdd(&p.ks_done[ks], 1) + 1 == p.ks_need[ks]) { __threadfence(); k_slot_done(p, ks); }
    }
    __syncwarp();                          // (the release fence + rows_done count: once per claimed batch, caller)
}

// Block 0, warps 0 .. kExpWarps - 1 (threads 32 ..): materialise the groups in exp_order, kExpClaim per claim (one fence
// per claim: a fence per group would starve many small groups), and count the finished groups (done_q, in order of
// completion). Counting one group costs slot atomics + fences, too much for one warp on thousands of groups, so every
// warp counts, claiming up to kDrainClaim queue entries at a time from a shared counter (exp_next[1]), one acquire
// fence and one release fence + rows_done add per claim. Warp 0 counts from the start, the others once materialising
// is done (materialising first: a group runs only once materialised). Exits when every group is materialised and
// counted.
static constexpr int kExpWarps = 3, kExpClaim = 8, kDrainClaim = 4;
__device__ __noinline__ void rect_block0(const Params &p) {
    const int lane = threadIdx.x & 31, w = (threadIdx.x >> 5) - 1, G = p.hg * p.G1, rg = p.h / p.hg;   // (rg: q heads per group)
    int32_t *rb = p.rowbuf + 64 * w;
    int *drain_next = p.exp_next + 1;      // (scratch slots are 32-int aligned and zeroed per call)
    bool exp_done = false, drain_done = G == 0;
    long idle = 0;
    int ndone = 0;
    const unsigned long long t0 = bwd_now();
    unsigned long long t_done = 0, t_wait = 0;                       // (trace) time in rect_group_done / waiting
    while (!exp_done || !drain_done) {
        int progress = 0;
        if (!exp_done) {
            int k0 = lane == 0 ? atomicAdd(p.exp_next, kExpClaim) : 0;
            k0 = __shfl_sync(0xffffffffu, k0, 0);
            if (k0 >= G) {
                exp_done = true;
                if (p.trace && lane == 0)
                    printf("[pe %d bwd rect] warp %d: materialised its share of %d groups by %.2f ms\n", nvshmem_my_pe(), w, G, (bwd_now() - t0) * 1e-6);
            } else {
                const int k1 = min(G, k0 + kExpClaim);
                for (int k = k0; k < k1; ++k) rect_expand(p, p.exp_order[k], lane, rg, rb);
                if (lane == 0) {
                    __threadfence();
                    for (int k = k0; k < k1; ++k) { const int g = p.exp_order[k]; atomicOr(&p.avail_bits[g >> 5], 1u << (g & 31)); }
                }
                progress = 1;
            }
        }
        if (!drain_done && (w == 0 || exp_done)) {
            int k0 = -1, k1 = 0;
            if (lane == 0) {
                const int tail = *(volatile int *)p.done_tail, nx = *(volatile int *)drain_next;
                if (nx >= G) k0 = G;
                else if (nx < tail) { const int n = min(kDrainClaim, tail - nx); k0 = atomicAdd(drain_next, n); k1 = min(G, k0 + n); }
            }
            k0 = __shfl_sync(0xffffffffu, k0, 0); k1 = __shfl_sync(0xffffffffu, k1, 0);
            if (k0 >= G) drain_done = true;
            else if (k0 >= 0) {
                const unsigned long long ta = p.trace ? bwd_now() : 0;
                if (lane == 0) for (int k = k0; k < k1; ++k) while (*(volatile int *)&p.done_q[k] == 0) {}
                __syncwarp();
                __threadfence();                                      // acquire: the workers' atomics before their entries
                const unsigned long long tb = p.trace ? bwd_now() : 0;
                for (int k = k0; k < k1; ++k) rect_group_done(p, *(volatile int *)&p.done_q[k] - 1, lane, rb);
                if (lane == 0) { __threadfence(); atomicAdd(p.sch.rows_done, k1 - k0); }
                if (p.trace) { t_wait += tb - ta; t_done += bwd_now() - tb; }
                ndone += k1 - k0;
                progress = 1;
            }
            if (drain_done && p.trace && lane == 0 && G > 0)
                printf("[pe %d bwd rect] warp %d: counted %d of %d groups, all claimed by %.2f ms (in group_done %.2f ms, entry waits %.2f ms)\n",
                       nvshmem_my_pe(), w, ndone, G, (bwd_now() - t0) * 1e-6, t_done * 1e-6, t_wait * 1e-6);
        }
        if (!progress) {
            __nanosleep(256);
            if (lane == 0 && w == 0 && ++idle % (DIST_SPIN_REPORT / 4) == 0)
                printf("[pe %d bwd rect] idle: groups claimed %d/%d, counting claimed %d/%d\n", nvshmem_my_pe(),
                       *(volatile int *)p.exp_next, G, *(volatile int *)drain_next, G);
        }
    }
}

// ---- the kernel -------------------------------------------------------------------------------------------------------
// Worker control words live in DYNAMIC shared memory past the compute routine's area (Ctl): a static __shared__
// variable shifts the dynamic area, and the compute routine's swizzled tiles then lose their alignment.
struct Ctl { int entry, nq, kdone; int qdone[64]; };
constexpr int kCtlBytes = 128 * ((sizeof(Ctl) + 127) / 128);

// Worker gangs: block 1 and, if the count is odd, the last worker are SOLO; the others
// pair up (2,3), (4,5), ... A gang leader takes two queue entries at once and posts their position to its partner, so
// the two twin halves of a KV tile (identical Q-side item streams) start together and read Q / dO in lockstep --
// one DRAM fetch serves both, as with the hardware's pairwise CTA launch.
__device__ __forceinline__ int gang_role(const Params &p, int &gang) {      // 0 leader, 1 partner, 2 solo
    const int b = blockIdx.x;
    if (b == 1) return 2;
    const int w = b - 2, n_pair_workers = (p.n_workers - 1) & ~1;
    if (w >= n_pair_workers) return 2;
    gang = w >> 1;
    return w & 1;
}

// Transfer-only slots (the Ring full-communication ablation): a slot this PE receives and must send a
// partial sum for (push / gateway aggregation) but computes nothing on (need 0) is complete at the start -- its
// accumulator is the zeroed one -- so it takes its completed path right away (as k_slot_done).
// Normal plans have no such slot (a remote slot exists only because a task reads it).
__device__ __noinline__ void zero_slots_done(const Params &p) {
    for (int s = threadIdx.x; s < p.NQS; s += blockDim.x)
        if (p.qs_need[s] == 0 && (p.qs_push_pe[s] >= 0 || p.qs_agg_need[s] > 0)) {
            if (p.sch_on) q_slot_done(p, s);
            else if (p.qs_agg_need[s] > 0) {
                if (atomicAdd(&p.qa_cnt[s], 1) + 1 == p.qs_agg_need[s]) { __threadfence(); sq_push(p, kQAgg, s, 1); atomicAdd(p.n_enq, 1); }
            } else rt::fire(p.eng, trig_q(s));
        }
    for (int ks = threadIdx.x; ks < p.NKS; ks += blockDim.x)
        if (p.ks_need[ks] == 0 && (p.ks_push_pe[ks] >= 0 || p.ks_agg_need[ks] > 0)) {
            if (p.sch_on) k_slot_done(p, ks);
            else if (p.ks_agg_need[ks] > 0) {
                if (atomicAdd(&p.ka_cnt[ks], 1) + 1 == p.ks_agg_need[ks]) { __threadfence(); sq_push(p, kKAgg, ks, 1); atomicAdd(p.n_enq, 1); }
            } else rt::fire(p.eng, trig_k(p, ks));
        }
}

template <typename Kernel_traits>
__global__ void __launch_bounds__(Kernel_traits::kNThreads, 1) dist_bwd_kernel(const __grid_constant__ Params p) {
    extern __shared__ char smem_[];
    const int tidx = threadIdx.x;
    if (blockIdx.x == 0) {
        if (tidx < 32) {
            if (p.sch_on && p.rect) rt::sched_block0<1>(p.sch, smem_, *p.sch_b0, &p.eng);   // shared scheduler (rt_sched.cuh)
            else if (p.sch_on) rt::sched_block0<false>(p.sch, smem_, *p.sch_b0, &p.eng);
            else scheduler(p, reinterpret_cast<uint32_t *>(smem_), reinterpret_cast<uint32_t *>(smem_) + (p.n_remote + 31) / 32);
        } else if (p.rect && tidx < 32 * (1 + kExpWarps)) {
            rect_block0(p);                                // warps 1-3: item lists + completion counting
        } else {
            const int e0 = p.rect ? 32 * (1 + kExpWarps) : 32;
            rt::engine(p.eng, e0, blockDim.x - e0, 1, reinterpret_cast<volatile int *>(smem_ + Kernel_traits::kSmemSize1colblock));
        }
        return;
    }
    Ctl &c = *reinterpret_cast<Ctl *>(smem_ + Kernel_traits::kSmemSize1colblock);
    if (blockIdx.x == 1) zero_slots_done(p);              // transfer-only slots (none in normal plans)
    __syncthreads();
    int gang = 0;
    const int role = gang_role(p, gang);
    volatile int *mail = p.mail + 32 * gang;                 // [0] sequence, [1] queue position of the pair, [2] ack

    if (p.trace && tidx == 0) atomicMin(&p.trace[6], bwd_now());
    int pos = 0, seq = 0;
    for (int it = 0;; ++it) {
        const unsigned long long tw0 = p.trace ? bwd_now() : 0;
        if (tidx == 0) {
            if (role == 0) {                                  // post the pair once the partner has read the last one
                ++seq;
                while (mail[2] < seq - 1) __nanosleep(32);
                pos = atomicAdd(p.tile_counter, 2);
                mail[1] = pos; __threadfence(); mail[0] = seq;
            } else if (role == 1) {
                ++seq;
                while (mail[0] < seq) __nanosleep(32);
                __threadfence();
                pos = mail[1] + 1;
                mail[2] = seq;                                // ack: the leader may post the next pair
            } else {                                          // solo: both entries of a pair, one after the other
                pos = (it & 1) ? pos + 1 : atomicAdd(p.tile_counter, 2);
            }
            const volatile int *q = p.sq + pos;
            long n = 0;
            while (*q == 0) {
                __nanosleep(64);
                if (++n % DIST_SPIN_REPORT == 0) printf("[pe %d blk %d] bwd queue slot %d empty\n", nvshmem_my_pe(), blockIdx.x, pos);
            }
            c.entry = *q - 1;
            __threadfence();
        }
        __syncthreads();
        const int entry = c.entry;
        __syncthreads();
        const int kind = entry >> 28, pay = entry & ((1 << 28) - 1);
        const unsigned long long tw1 = p.trace ? bwd_now() : 0;
        BTR(p, 0, tw1 - tw0);
        if (kind == kStop) break;
        if (kind == kTask) {
            // shared scheduler: payload = dynamic task * 2 + half; the task's row is a twin pair of static tasks
            // (rect: row = group g, tasks 2 g + half -- task_kv / task_ptr / items written by block 0's warps 1-3)
            const int t = p.sch_on ? 2 * p.sch.task_desc[4 * (pay >> 1)] + (pay & 1) : pay;
            const int tk = p.task_kv[t], ks = tk >> 9;
            sparse_bwd::bwd_task<Kernel_traits, /*Atomic_dKV=*/true>(p.sp, ks, (tk >> 8) & 1, tk & 255,
                                                                    p.items + p.task_ptr[t], p.task_end[t] - p.task_ptr[t], smem_);
            // count the task into its slots; the block completing a slot sends / books its accumulator (out of line:
            // cold, and inline its branches shift the compute routine's register allocation)
            __syncthreads();
            const unsigned long long tw2 = p.trace ? bwd_now() : 0;
            BTR(p, 1, tw2 - tw1); BTR(p, 5, 1);
            complete_task(p, t, ks, c);
            BTR(p, 2, (p.trace ? bwd_now() : 0) - tw2);
        } else if (kind == kQSum) {
            const int m = pay / (2 * p.h), k = pay % (2 * p.h);
            qsum<Kernel_traits>(p, m, k / p.h, k % p.h);
            __syncthreads();
            if (p.ml && tidx == 0) atomicMax(&p.ml[p.NQS + p.NKS + m], bwd_now());
            BTR(p, 3, (p.trace ? bwd_now() : 0) - tw1);
        } else if (kind == kKSum) {
            ksum(p, pay / p.hk, pay % p.hk);
            __syncthreads();
            if (p.ml && tidx == 0) atomicMax(&p.ml[p.NQS + p.NKS + p.n_qm + pay / p.hk], bwd_now());
            BTR(p, 4, (p.trace ? bwd_now() : 0) - tw1);
        } else if (kind == kQAgg || kind == kKAgg) {               // gateway: own partial + node landings -> owner
            const bool isq = kind == kQAgg;
            const int s = pay;
            const int64_t bytes = isq ? p.dqa_bytes : p.dkva_bytes;
            float4 *acc = reinterpret_cast<float4 *>((isq ? p.dqa : p.dkva) + (int64_t)s * bytes);
            const char *land = isq ? p.dql : p.dkl;
            const int *lp = isq ? p.qs_agg_ptr : p.ks_agg_ptr, *ll = isq ? p.qs_agg_land : p.ks_agg_land;
            for (int64_t e = tidx; e < bytes / 16; e += blockDim.x) {
                float4 a = acc[e];
                for (int k = lp[s]; k < lp[s + 1]; ++k) {
                    const float4 b = reinterpret_cast<const float4 *>(land + (int64_t)ll[k] * bytes)[e];
                    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
                }
                acc[e] = a;
            }
            __syncthreads();
            if (p.ml && tidx == 0) p.ml[p.NQS + p.NKS + p.n_qm + p.n_km + (isq ? s : p.NQS + s)] = bwd_now();
            if (tidx == 0) rt::fire(p.eng, isq ? trig_q(s) : trig_k(p, s));
        }                                                          // kNop: padding of an odd push
    }
    if (p.trace && tidx == 0) atomicMax(&p.trace[7], bwd_now());
}

}  // namespace dist_bwd
