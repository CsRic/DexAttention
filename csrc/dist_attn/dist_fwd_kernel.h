#pragma once

// ── The distributed forward megakernel ──────────────────────────────────────
// One persistent launch per GPU executes an external assignment end to end:
//
//   block 0            push its share of Q/KV, then DELEGATE — the only NVSHMEM
//                      signal poller (concurrent pollers on one symmetric word
//                      contend heavily), fanning arrivals out through ordinary
//                      device memory. It never computes: partials keep arriving
//                      long after compute starts, so this block has to keep
//                      polling to the end. That costs 1 SM of 64.
//   blocks 1..G-1      push their share, then drain the work queue:
//                        COMPUTE row  gated on its Q and KV arriving; writes a
//                                     partial; the block that finishes the LAST
//                                     head of a row pushes the whole slot home
//                        MERGE  tile  gated on its partials arriving; log-sum-exp
//                                     combine into this PE's final O
//
// The phase order is a DAG — push (depends on nothing) -> compute -> partial
// push -> merge — so waits cannot form a cycle. Compute items are ordered
// before merge items in the queue, so every compute item is claimed before any
// block can park on a merge.

#include "dist_partial_kernel.h"
#include "dist_merge.cuh"
#include "dist_params.h"
#include "../ring_attn/nvshmem_transport.cuh"
#include "../runtime/rt_engine.cuh"

namespace ring_attn {

#ifndef DIST_SPIN_REPORT
#define DIST_SPIN_REPORT 2000000
#endif


using namespace cute;

// Same protocol as SparseTileScheduler, with `n_reserved` blocks held out of the
// queue (block 0 is the delegate). Without the reservation those blocks' initial
// tiles would simply never be executed.
class DistTileScheduler {
public:
    using SharedStorage = int;
    struct Params {
        int n_work;
        const int32_t *work_coord;
        int *tile_count_semaphore;
        int num_m_block;
        int nheads;
        int n_reserved;
    };
    struct WorkTileInfo {
        int tile_idx;
        CUTLASS_DEVICE bool is_valid(Params const &p) const { return tile_idx < p.n_work; }
        CUTLASS_DEVICE cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const &p) const {
            int const packed = p.work_coord[tile_idx];
            int const m_block = packed % p.num_m_block;
            int const rest    = packed / p.num_m_block;
            return {m_block, rest % p.nheads, rest / p.nheads, 0};
        }
    };
protected:
    SharedStorage *const tile_count_smem;
public:
    CUTLASS_DEVICE DistTileScheduler(SharedStorage *const s) : tile_count_smem(s) {}
    CUTLASS_DEVICE WorkTileInfo get_initial_work(Params const &p) const {
        return {int(blockIdx.x) - p.n_reserved};
    }
    CUTLASS_DEVICE void init_consumer() const {
        if (cutlass::canonical_warp_idx_sync() > 0) {
            flash::named_barrier_arrive(256, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
        }
    }
    CUTLASS_DEVICE void prefetch_next_work(Params const &p, WorkTileInfo &cur) const {
        if (threadIdx.x % 256 == 0) {
            cur.tile_idx = atomicAdd(p.tile_count_semaphore, 1)
                         + int(gridDim.x) - p.n_reserved;
        }
    }
    template <bool IsProducerWarp = false>
    CUTLASS_DEVICE WorkTileInfo get_next_work(Params const &, WorkTileInfo const &cur) const {
        if constexpr (IsProducerWarp) {
            int nxt = __shfl_sync(0xffffffff, cur.tile_idx, 0);
            flash::named_barrier_sync(256, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
            if (threadIdx.x % 256 == 0) { *tile_count_smem = cur.tile_idx; }
            flash::named_barrier_arrive(256, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
            return {nxt};
        } else {
            flash::named_barrier_sync(256, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1);
            int t = *tile_count_smem;
            flash::named_barrier_arrive(256, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
            return {t};
        }
    }
};

// ── transport helpers ───────────────────────────────────────────────────────
// `sig_id < 0` means "part of a group, signal after the last one". The fence is
// taken only when a signal follows: only the signal must be ordered after the bytes.
// Cross-call write-after-read guard, pairwise: before this PE writes into dst_pe's slots for call `call_id`,
// dst_pe must have finished call `call_id - 1` (it publishes that from its last exiting block). Only the pairs
// that actually exchange data ever wait on each other -- no global barrier.
__device__ __forceinline__ void dist_wait_credit(const Dist_fwd_params &p, int dst_pe) {
    if (threadIdx.x == 0) {
        unsigned ns = 64;
        while (*(volatile const uint64_t *)&p.credit[dst_pe] + 1 < p.call_id) {
            __nanosleep(ns);
            if (ns < 1024) ns <<= 1;
        }
    }
}

// Timeline event slots after the per-signal / per-item stamps (diagnostic buffer, null in normal runs):
// 0 last block entered the kernel, 1 block 0 initialised, 2 first task queued, 3 STOP queued, 4 last worker left its
// loop, 5 last block entered the credit release, 6 the credit signals to every peer were sent
__device__ __forceinline__ unsigned long long *dist_tl_ev(const Dist_fwd_params &p) {
    return p.timeline + 1 + p.NQ + p.NKV + p.NPART + 2 * (int64_t)(p.n_compute > 0 ? p.n_compute : 1)
         + 2 * (int64_t)p.n_merge * p.h;
}
__device__ __forceinline__ unsigned long long dist_gtimer() {
    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}

__device__ __noinline__ void dist_credit_release(const Dist_fwd_params &p) {
    if (!p.credit) return;
    __syncthreads();
    if (threadIdx.x == 0) {
        if (p.timeline) atomicMax(&dist_tl_ev(p)[5], dist_gtimer());
        __threadfence();
        if (atomicAdd(p.blocks_done, 1) == (int)gridDim.x - 1) {      // the last block: every read is done
            if (p.fw_done)                                           // ... and every relay forward out of here
                while (*(volatile unsigned long long *)p.fw_done < p.sig_epoch) __nanosleep(1000);
            __threadfence_system();
            for (int q = 0; q < p.n_pes; ++q)
                if (q != p.my_pe) nvshmemx_signal_op(&p.credit[p.my_pe], p.call_id, NVSHMEM_SIGNAL_SET, q);
            if (p.timeline) dist_tl_ev(p)[6] = dist_gtimer();
        }
    }
}

__device__ __forceinline__ void dist_put_block(
    const Dist_fwd_params &p, char *dst_local_view, const char *src, size_t nbytes,
    int dst_pe, int sig_id
) {
    if (p.credit) dist_wait_credit(p, dst_pe);
    if (p.pe_inter[dst_pe]) {
        if (threadIdx.x == 0) {
            if (p.put_signal && sig_id >= 0) {
                // The signal rides on the last message of the group, ordered
                // after its own bytes by put-with-signal and after the group's
                // earlier puts to this PE by the fence. No quiet on this path.
                nvshmem_fence();
                nvshmem_putmem_signal_nbi(dst_local_view, src, nbytes, &p.sig_base[sig_id],
                                          p.sig_epoch, NVSHMEM_SIGNAL_SET, dst_pe);
            } else {
                nvshmem_putmem_nbi(dst_local_view, src, nbytes, dst_pe);
            }
        }
        return;   // without put_signal, ordering is closed by a quiet, never here
    }
    if (p.credit) __syncthreads();                     // every thread writes: all wait for thread 0's credit check
    intra_push_slice_block(dst_local_view + p.pe_delta[dst_pe], src, nbytes);
    if (sig_id < 0) { __syncthreads(); return; }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        ring_signal_set(&p.sig_base[sig_id], p.sig_epoch, dst_pe);
    }
}

// At most kMaxQuiet blocks may be inside nvshmem_quiet() at once (NVSHMEM's retire path does not tolerate many
// concurrent callers), and the O-push path can reach it from any compute block.
static constexpr int kMaxQuiet = 4;
__device__ __forceinline__ void dist_quiet_gated(const Dist_fwd_params &p) {
    if (threadIdx.x == 0) {
        while (true) {
            if (atomicAdd(p.quiet_gate, 1) < kMaxQuiet) break;
            atomicSub(p.quiet_gate, 1);
            __nanosleep(256);
        }
        nvshmem_quiet();
        __threadfence_system();
        atomicSub(p.quiet_gate, 1);
    }
    __syncthreads();
}

// ---- diagnostic counters (u64, summed over blocks): 0 push ns, 1 q-wait ns, 2 merge-wait ns,
// 3 compute items, 4 blocks done, 5 first start, 6 last end, 7 compute-item ns (incl. tile gates and
// partial push), 8 merge-item ns, 9 partial-push ns, 10 merge items
__device__ __forceinline__ unsigned long long dist_now() {
    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
#define DTR_ADD(p, i, v) do { if ((p).trace && threadIdx.x == 0) atomicAdd(&(p).trace[i], (unsigned long long)(v)); } while (0)

__device__ __noinline__ void dist_trace_end(const Dist_fwd_params &p) {
    if (!p.trace || threadIdx.x != 0) return;
    atomicMax(&p.trace[6], dist_now());
    if (atomicAdd(&p.trace[4], 1ULL) == gridDim.x - 1) {
        __threadfence();
        const unsigned long long *t = p.trace;
        const double nb = gridDim.x - 1;                     // compute blocks (block 0 delegates)
        printf("[pe %d trace] span %.2f ms | push %.2f ms | per compute block: q-wait %.2f, compute-items %.2f "
               "(of which partial push %.2f), merge-items %.2f (merge-wait %.2f) ms | items %llu compute, %llu merge\n",
               nvshmem_my_pe(), (t[6] - t[5]) * 1e-6, t[0] * 1e-6, t[1] * 1e-6 / nb, t[7] * 1e-6 / nb,
               t[9] * 1e-6 / nb, t[8] * 1e-6 / nb, t[2] * 1e-6 / nb, t[3], t[10]);
    }
}

// Returns whether this block issued any inter-node message (the caller owes a
// quiet before exit when put_signal left them un-retired).
__device__ __noinline__ bool dist_push_phase(const Dist_fwd_params &p) {
    const int pb = int(blockIdx.x) - p.push_block_base;
    if (pb < 0 || pb >= p.num_push_blocks) return false;
    const unsigned long long t_push0 = p.trace ? dist_now() : 0;
    struct PushTimer { const Dist_fwd_params &p; unsigned long long t0;
        __device__ ~PushTimer() { if (p.trace) DTR_ADD(p, 0, dist_now() - t0); } } push_timer{p, t_push0};
    bool inter = false;
    // One queue over both kinds, in push_order when the planner gave one (the
    // scheduler's send order), else all Q then all KV.
    const int n_total = p.n_pushq + p.n_pushkv;
    for (int j = pb; j < n_total; j += p.num_push_blocks) {
        const int e = p.push_order ? p.push_order[j] : j;
        if (e < p.n_pushq) {
            const int i = e;
            const int dpe = p.pushq_pe[i];
            const int cnt = p.pushq_cnt ? p.pushq_cnt[i] : 1;
            char *dst = (char *)p.q_slots + (int64_t)p.pushq_dst[i] * p.q_slot_bytes;
            const char *src = (char *)p.q_slots + (int64_t)p.pushq_src[i] * p.q_slot_bytes;
            dist_put_block(p, dst, src, p.q_slot_bytes * cnt, dpe, p.pushq_dst[i] + cnt - 1);
            inter |= (p.pe_inter[dpe] != 0);
        } else {
            const int i = e - p.n_pushq;
            const int dpe = p.pushkv_pe[i];
            const int cnt = p.pushkv_cnt ? p.pushkv_cnt[i] : 1;
            const int64_t doff = (int64_t)p.pushkv_dst[i] * p.kv_slot_bytes;
            const int64_t soff = (int64_t)p.pushkv_src[i] * p.kv_slot_bytes;
            dist_put_block(p, (char *)p.k_slots + doff, (char *)p.k_slots + soff,
                           p.kv_slot_bytes * cnt, dpe, -1);
            dist_put_block(p, (char *)p.v_slots + doff, (char *)p.v_slots + soff,
                           p.kv_slot_bytes * cnt, dpe, p.NQ + p.pushkv_dst[i] + cnt - 1);
            inter |= (p.pe_inter[dpe] != 0);
        }
    }
    if (inter && p.put_signal) return true;   // signalled per message already
    if (inter) {
        dist_quiet_gated(p);
        if (threadIdx.x == 0) {
            for (int j = pb; j < n_total; j += p.num_push_blocks) {
                const int e = p.push_order ? p.push_order[j] : j;
                if (e < p.n_pushq) {
                    if (p.pe_inter[p.pushq_pe[e]])
                        ring_signal_set(&p.sig_base[p.pushq_dst[e] + (p.pushq_cnt ? p.pushq_cnt[e] - 1 : 0)],
                                        p.sig_epoch, p.pushq_pe[e]);
                } else {
                    const int i = e - p.n_pushq;
                    if (p.pe_inter[p.pushkv_pe[i]])
                        ring_signal_set(&p.sig_base[p.NQ + p.pushkv_dst[i] + (p.pushkv_cnt ? p.pushkv_cnt[i] - 1 : 0)],
                                        p.sig_epoch, p.pushkv_pe[i]);
                }
            }
        }
    }
    return false;
}

// Ready queue: signal j (index into remote_sig) has landed -> count it into the items that need it; every item that is
// now complete gets all its heads appended to the ready queue.
__device__ __forceinline__ void dist_release(const Dist_fwd_params &p, int j) {
    if (!p.dep_ptr) return;
    for (int k = p.dep_ptr[j]; k < p.dep_ptr[j + 1]; ++k) {
        const int it = p.dep_item[k];
        if (atomicAdd(&p.dep_cnt[it], 1) + 1 == p.dep_need[it]) {
            const int base = atomicAdd(p.ready_tail, p.h);
            for (int hh = 0; hh < p.h; ++hh) atomicExch(&p.ready_q[base + hh], hh * p.n_compute + it + 1);
        }
    }
}

// A compute block's wait on a not-yet-released ready-queue slot. Cold (taken only when no work is ready) and kept
// out of the kernel body so it does not touch the mainloop's code generation.
__device__ __noinline__ int dist_ready_wait(const Dist_fwd_params &p, int slot) {
    const volatile int *q = p.ready_q + slot;
    long n = 0;
    while (*q == 0) {
        __nanosleep(64);
        if (++n % DIST_SPIN_REPORT == 0)
            printf("[pe %d blk %3d] ready-queue slot %d still empty\n", nvshmem_my_pe(), blockIdx.x, slot);
    }
    const int v = *q - 1;
    __threadfence();
    return v;
}

// Block 0 only, for the whole run: partials keep landing after compute begins.
//
// Per-head virtual blocks give thousands of remote signal ids, so warp 0 splits the ids by lane (lane l owns ids l,
// l+32, ...) and each lane keeps its own done-bits in shared memory (block 0 never computes, so smem_ is free): no
// atomics on ids that already arrived, 32 polls in flight, and different lanes poll different words. A single thread
// polls when the bitmap does not fit.
__device__ __noinline__ void dist_delegate_serial(const Dist_fwd_params &p) {
    if (threadIdx.x != 0) return;
    int left = p.n_remote;
    long spins = 0;
    while (left > 0) {
        for (int i = 0; i < p.n_remote; ++i) {
            const int id = p.remote_sig[i];
            if (atomicAdd(&p.arrived[id], 0) != 0) continue;
            const volatile uint64_t *w = (const volatile uint64_t *)&p.sig_base[id];
            if (*w >= p.sig_epoch) {
                __threadfence_system();
                if (p.remote_cnt) for (int k = p.remote_cnt[i] - 1; k > 0; --k) atomicExch(&p.arrived[id - k], 1);
                atomicExch(&p.arrived[id], 1);
                dist_release(p, i);
                if (p.timeline) p.timeline[1 + id] = dist_now();
                --left;
            }
        }
        if (left > 0) {
            __nanosleep(256);
            if (++spins % (DIST_SPIN_REPORT / 2) == 0) {
                for (int i = 0; i < p.n_remote; ++i)
                    if (atomicAdd(&p.arrived[p.remote_sig[i]], 0) == 0)
                        printf("[pe %d delegate] no signal for %d\n", nvshmem_my_pe(),
                               p.remote_sig[i]);
            }
        }
    }
}

__device__ __noinline__ void dist_delegate(const Dist_fwd_params &p, uint32_t *done, int cap_words) {
    const int n = p.n_remote;
    const int per_lane = (n + 31) / 32;                    // ids owned by one lane
    const int wpl = (per_lane + 31) / 32;                  // bitmap words per lane
    if (32 * wpl > cap_words) { dist_delegate_serial(p); return; }
    if (threadIdx.x >= 32) return;
    const int lane = threadIdx.x;
    uint32_t *mine = done + lane * wpl;
    for (int w = 0; w < wpl; ++w) mine[w] = 0u;
    int left = lane < n ? (n - lane + 31) / 32 : 0;
    long spins = 0;
    while (__any_sync(0xffffffffu, left > 0)) {
        int got = 0;
        for (int k = 0, i = lane; i < n; ++k, i += 32) {
            if (mine[k >> 5] & (1u << (k & 31))) continue;
            const int id = p.remote_sig[i];
            const volatile uint64_t *w = (const volatile uint64_t *)&p.sig_base[id];
            if (*w >= p.sig_epoch) {
                __threadfence_system();
                if (p.remote_cnt) for (int c = p.remote_cnt[i] - 1; c > 0; --c) atomicExch(&p.arrived[id - c], 1);
                atomicExch(&p.arrived[id], 1);
                dist_release(p, i);
                if (p.timeline) p.timeline[1 + id] = dist_now();
                mine[k >> 5] |= 1u << (k & 31);
                --left; ++got;
            }
        }
        if (!__any_sync(0xffffffffu, got > 0)) {
            __nanosleep(256);
            if (++spins % (DIST_SPIN_REPORT / 2) == 0)
                for (int k = 0, i = lane; i < n; ++k, i += 32)
                    if (!(mine[k >> 5] & (1u << (k & 31))))
                        printf("[pe %d delegate] no signal for %d\n", nvshmem_my_pe(), p.remote_sig[i]);
        }
    }
}

// Every spin is bounded by a report: after ~0.25 s the waiter names the signal it is parked on and keeps going.
__device__ __forceinline__ void dist_spin(const Dist_fwd_params &p, int id,
                                          const char *what) {
    long n = 0;
    while (atomicAdd(&p.arrived[id], 0) == 0) {
        __nanosleep(128);
        if (++n % DIST_SPIN_REPORT == 0)
            printf("[pe %d blk %3d] stuck %s sig %d\n", nvshmem_my_pe(), blockIdx.x, what, id);
    }
}

__device__ __forceinline__ void dist_wait(const Dist_fwd_params &p,
                                          const int32_t *ids, int beg, int end) {
    if (beg == end) return;
    if (threadIdx.x == 0) {
        for (int i = beg; i < end; ++i) dist_spin(p, ids[i], "kv");
        __threadfence();
    }
    __syncthreads();
}

// The partial-O return of one finished compute item, kept OUT of the kernel body (__noinline__): it is cold (once
// per item, a few hundred cycles) but heavy in live state, and inlined it competes with the mainloop for registers.
// Returns whether an un-retired inter-node put was issued (the caller owes a quiet before exit).
__device__ __noinline__ bool dist_return_partial(const Dist_fwd_params &p, int m_block, int &s_last_head,
                                                 int &s_last_grp) {
    const int tidx = threadIdx.x;
    bool owe_quiet = false;
    // A partial slot is complete only once every head has written it, so
    // the block that closes the last head sends the whole slot home in
    // one contiguous transfer instead of h strided ones.
    //
    // The fences are load-bearing: the heads of one slot are written by
    // DIFFERENT blocks, and the block that happens to close the count
    // then reads all of them. Without a release before the atomic and an
    // acquire after it, it can copy a neighbour's not-yet-visible stores.
    __syncthreads();
    if (tidx == 0) {
        __threadfence();
        s_last_head = (atomicAdd(&p.o_head_done[m_block], 1) == p.h - 1);
    }
    __syncthreads();
    if (s_last_head) { __threadfence(); }
    const int grp = (s_last_head && p.item_group) ? p.item_group[m_block] : -1;   // uniform in the block
    if (grp >= 0) {
        // staged return: count this item into its group; the block that completes the group
        // sends the whole contiguous range (release before the count, acquire after)
        if (tidx == 0) {
            __threadfence();
            s_last_grp = (atomicAdd(&p.group_done[grp], 1) == p.group_size[grp] - 1);
        }
        __syncthreads();
        if (s_last_grp && p.use_engine >= 3) {                    // the shared transport sends the group
            if (tidx == 0) rt::fire(p.eng, p.eng_grp_trig + grp);
        } else if (s_last_grp) {
            __threadfence();
            const int dpe = p.group_dst_pe[grp], dslot = p.group_dst_slot[grp];
            const int first = p.group_first[grp], n = p.group_size[grp];
            char *dsto = (char *)p.part_o + (int64_t)dslot * p.o_slot_bytes;
            const char *srco = (char *)p.part_o + (int64_t)first * p.o_slot_bytes;
            const int sig = p.NQ + p.NKV + dslot + n - 1;
            dist_put_block(p, dsto, srco, p.o_slot_bytes * n, dpe, p.part_combined ? sig : -1);
            for (int hh = 0; hh < (p.part_combined ? 0 : p.h); ++hh) {   // combined: the LSE rode along
                char *dstl = (char *)p.part_lse
                           + ((int64_t)hh * p.lse_head_stride + (int64_t)dslot * 128) * 4;
                const char *srcl = (char *)p.part_lse
                           + ((int64_t)hh * p.lse_head_stride + (int64_t)first * 128) * 4;
                dist_put_block(p, dstl, srcl, p.lse_slot_bytes * n, dpe, hh == p.h - 1 ? sig : -1);
            }
            if (p.pe_inter[dpe] && p.put_signal) {
                owe_quiet = true;
            } else if (p.pe_inter[dpe]) {
                dist_quiet_gated(p);
                if (tidx == 0) ring_signal_set(&p.sig_base[sig], p.sig_epoch, dpe);
            }
        }
    } else if (s_last_head && p.opush_pe[m_block] >= 0 && p.hs_pflag && p.pe_inter[p.opush_pe[m_block]]) {
        // unused: inter-node partials always go by NVSHMEM put (hs_pflag is always nullptr, so this branch never runs).
        if (tidx == 0) {
            __threadfence_system();
            *(volatile unsigned long long *)&p.hs_pflag[m_block] = p.sig_epoch;
        }
    } else if (s_last_head && p.opush_pe[m_block] >= 0) {
        if (p.q_all_in) {                     // q_all_in: no partial leaves while Q is still coming in
            if (tidx == 0) { while (*(volatile int *)p.q_all_in == 0) { __nanosleep(256); } }
            __syncthreads();
        }
        const int dpe = p.opush_pe[m_block];
        const int dslot = p.opush_slot[m_block];
        if (p.use_engine >= 2 || (p.use_engine && p.pe_inter[dpe])) {   // the shared transport sends it (trigger = the row:
            if (tidx == 0) rt::fire(p.eng, m_block);                       //  engine queue and / or copy-engine flag)
            return owe_quiet;
        }
        if (p.row_flag && !p.pe_inter[dpe]) {   // the copy engine sends it (see Dist_fwd_params::row_flag)
            if (tidx == 0) {
                __threadfence_system();
                *(volatile unsigned long long *)&p.row_flag[m_block] = p.sig_epoch;
            }
            return owe_quiet;
        }
        char *dsto = (char *)p.part_o + (int64_t)dslot * p.o_slot_bytes;
        const char *srco = (char *)p.part_o + (int64_t)m_block * p.o_slot_bytes;
        // combined slot (O then LSE): one message, signalled; head-major LSE is [h][tokens]: h more messages
        dist_put_block(p, dsto, srco, p.o_slot_bytes, dpe, p.part_combined ? p.NQ + p.NKV + dslot : -1);
        for (int hh = 0; hh < (p.part_combined ? 0 : p.h); ++hh) {
            char *dstl = (char *)p.part_lse
                       + ((int64_t)hh * p.lse_head_stride + (int64_t)dslot * 128) * 4;
            const char *srcl = (char *)p.part_lse
                       + ((int64_t)hh * p.lse_head_stride + (int64_t)m_block * 128) * 4;
            dist_put_block(p, dstl, srcl, p.lse_slot_bytes, dpe,
                           hh == p.h - 1 ? p.NQ + p.NKV + dslot : -1);
        }
        if (p.pe_inter[dpe] && p.put_signal) {
            owe_quiet = true;              // signalled by the last LSE put
        } else if (p.pe_inter[dpe]) {
            dist_quiet_gated(p);
            if (tidx == 0)
                ring_signal_set(&p.sig_base[p.NQ + p.NKV + dslot], p.sig_epoch, dpe);
        }
    }
    return owe_quiet;
}

// One merge item (row `item`, head `bidh`): wait for its partials, log-sum-exp combine. Cold path, __noinline__.
__device__ __noinline__ void dist_merge_item(const Dist_fwd_params &p, int item, int bidh, unsigned long long t_m0,
                                             float *smem) {
    dist_wait(p, p.mwait_sig, p.mwait_ptr[item], p.mwait_ptr[item + 1]);
    // ... and for this PE's own contribution, which arrives by being
    // computed rather than by being signalled.
    if (p.mself_ptr) {                      // every local segment of this row
        if (threadIdx.x == 0) {
            for (int s_ = p.mself_ptr[item]; s_ < p.mself_ptr[item + 1]; ++s_)
                while (atomicAdd(&p.o_head_done[p.mself_idx[s_]], 0) < p.h) { __nanosleep(128); }
            __threadfence();
        }
        __syncthreads();
    } else {
    const int self_row = p.merge_self[item];
    if (self_row >= 0) {
        if (threadIdx.x == 0) {
            while (atomicAdd(&p.o_head_done[self_row], 0) < p.h) { __nanosleep(128); }
            __threadfence();
        }
        __syncthreads();
    }
    }
    if (p.trace) DTR_ADD(p, 2, dist_now() - t_m0);
    const int beg = p.merge_ptr[item], end = p.merge_ptr[item + 1];
    dist_merge_tile<128, 128, 256, float, __half>(p.o_ptr, (float *)p.lse_ptr,
                      (const float *)p.part_o, (const float *)p.part_lse,
                      p.merge_slot + beg, end - beg, p.merge_out[item],
                      bidh, p.h, MergeLayout{p.part_o_slot, p.part_lse_slot, p.lse_head_stride},
                      MergeLayout{128LL * p.h * p.d, 128, p.flse_head_stride}, smem);
    __syncthreads();
}

// ── kernel ──────────────────────────────────────────────────────────────────
// Control words in DYNAMIC shared memory past the compute area (no static __shared__: it would shift the dynamic area
// and misalign the mainloop's swizzled tiles). bcast: block 0's engine.
struct FwdCtl { int last_head, last_grp, coord[3], owe_quiet, packed, bcast; unsigned long long t_item0; };
template <typename Traits> constexpr int fwd_ctl_off() { return 128 * ((Traits::kSmemSize + 127) / 128); }
template <typename Traits> constexpr int fwd_smem_bytes() { return fwd_ctl_off<Traits>() + int(sizeof(FwdCtl)); }

// Ready = true: the ready-queue variant (dims[10]). Compute blocks never wait on data inside an item -- the mainloop
// is instantiated with no tile gate at all, i.e. it is sparse_fwd's -- and only the delegate tracks dependencies.
template<typename Kernel_traits, bool Ready>
__global__ __launch_bounds__(Kernel_traits::kNThreads, 1)
void dist_fwd_kernel(
    // __grid_constant__: the cold-path helpers take `p` by reference; without it the compiler copies the whole
    // struct into every thread's local memory and hot-path reads of p come from there.
    const __grid_constant__ typename Kernel_traits::LaunchParams fa3,
    const __grid_constant__ Dist_fwd_params p,
    const __grid_constant__ DistTileScheduler::Params sched_params
) {
    extern __shared__ char smem_[];
    auto &shared_storage = *reinterpret_cast<typename Kernel_traits::SharedStorage *>(smem_);
    const int tidx = threadIdx.x;
    if (p.trace && tidx == 0) atomicCAS(&p.trace[5], 0ULL, dist_now());
    if (p.timeline && tidx == 0) atomicCAS(&p.timeline[0], 0ULL, dist_now());

    // push_block_base == 1: block 0 is never a push block, so it starts polling
    // at the first instruction instead of after this PE's whole push phase.
    FwdCtl &ctl = *reinterpret_cast<FwdCtl *>(smem_ + fwd_ctl_off<Kernel_traits>());
    bool owe_quiet = dist_push_phase(p);
    if (blockIdx.x == 0 && sched_params.n_reserved > 0) {  // pure delegate (none needed: nothing to poll)
        // warp 0 polls the arrivals; with the shared transport warps 1.. are the PE's communication engine
        if (p.use_engine && tidx >= 32) rt::engine(p.eng, 32, int(blockDim.x) - 32, 1, &ctl.bcast);
        else dist_delegate(p, reinterpret_cast<uint32_t *>(smem_),
                           int(sizeof(typename Kernel_traits::SharedStorage) / sizeof(uint32_t)));
        __syncthreads();
        if (owe_quiet) dist_quiet_gated(p);
        dist_trace_end(p);
        dist_credit_release(p);
        return;
    }

    typename Kernel_traits::Mainloop mainloop;
    typename Kernel_traits::Epilogue epilogue;              // final output (fp16)
    typename Kernel_traits::EpiloguePart epilogue_part;     // partial slots (fp32)
    typename Kernel_traits::TiledMma tiled_mma;

    DistTileScheduler scheduler(&shared_storage.smem_scheduler);
    scheduler.init_consumer();
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    int &s_last_head = ctl.last_head, &s_last_grp = ctl.last_grp;
    // Per-item state that is needed AFTER the mainloop lives in shared memory, not registers: the mainloop uses
    // ~250 of the 255 registers, and every value kept live across it pushes its K/V copy state onto the stack
    // (sparse_fwd, which keeps only the work index live, runs the same mainloop without spilling).
    int *s_coord = ctl.coord;                  // m_block, head, batch of the current item
    int &s_owe_quiet = ctl.owe_quiet;
    unsigned long long &s_t_item0 = ctl.t_item0;
    int &s_packed = ctl.packed;
    if (tidx == 0) s_owe_quiet = owe_quiet;

    CUTLASS_PRAGMA_NO_UNROLL
    for (auto work = scheduler.get_initial_work(sched_params);
         work.is_valid(sched_params);
         work = warp_idx == 0
             ? scheduler.template get_next_work</*IsProducerWarp=*/true>(sched_params, work)
             : scheduler.template get_next_work</*IsProducerWarp=*/false>(sched_params, work)) {

        if (work.tile_idx < 0) continue;   // defensive: only block 0 could get -1, and it returned

        if (work.tile_idx < p.n_compute_work) {
            // Opaque per item: ptxas otherwise hoists every shared-memory address the mainloop derives (16 K/V
            // copy offsets) out of the persistent loop and, short of registers, spills them -- one LDL per use
            // per tile, where recomputing is one IADD.
            char *smem_item = smem_;
#ifndef DIST_NO_OPAQUE
            asm volatile("" : "+l"(smem_item));
#endif
            auto &shared_storage = *reinterpret_cast<typename Kernel_traits::SharedStorage *>(smem_item);
            cute::tuple<int32_t, int32_t, int32_t, int32_t> block_coord;
            if constexpr (Ready) {
                int packed;
                if (work.tile_idx < p.n_init_work) {
                    packed = p.work_coord[work.tile_idx];
                } else {                               // released by the delegate; empty only if nothing is ready
                    if (tidx == 0) s_packed = dist_ready_wait(p, work.tile_idx - p.n_init_work);
                    __syncthreads();
                    packed = s_packed;
                }
                const int nmb = sched_params.num_m_block, rest = packed / nmb;
                block_coord = {packed % nmb, rest % sched_params.nheads, rest / sched_params.nheads, 0};
            } else {
                block_coord = work.get_block_coord(sched_params);
            }
            const int m_block = get<0>(block_coord);

            // Q is needed by every tile of the row, so it is the one thing waited
            // for up front. KV is waited for per tile inside the mainloop (the
            // tile gate below), or for the whole row at once without ord_sig.
            if (tidx == 0) {
                s_coord[0] = m_block; s_coord[1] = get<1>(block_coord); s_coord[2] = get<2>(block_coord);
                if (p.trace || p.timeline) s_t_item0 = dist_now();
            }
            if (!Ready && p.n_remote > 0 && p.row_qwait[m_block] >= 0) {
                if (tidx == 0) {
                    dist_spin(p, p.row_qwait[m_block], "q");
                    DTR_ADD(p, 1, dist_now() - s_t_item0);
                    __threadfence();
                }
            }
            __syncthreads();                       // s_coord visible to all; Q landed

            const int32_t beg = p.kv_indptr[m_block];
            const int32_t cnt = p.kv_indptr[m_block + 1] - beg;

            Tensor acc_o = partition_fragment_C(
                tiled_mma, select<0, 2>(typename Kernel_traits::TileShape_MNK{}));
            typename Kernel_traits::Softmax_t softmax(fa3.mainloop.softmax_scale_log2);
            typename Kernel_traits::SeqlenInfo_t seqlen_info{
                get<2>(block_coord), p.seqlen_q, p.seqlen_k, 0,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

            // ONE mainloop call for the whole row -- sparse_fwd's call, with one addition: `ord_sig`, the per-tile
            // arrival gate. The row is ordered so the tiles that had to travel are consumed last and each is
            // waited for where its cp.async is issued. A row with nothing remote passes no gate at all and runs
            // exactly sparse_fwd's path. Without ord_sig the row is gated as a whole here.
            const bool tilegate = p.kv_ord_sig != nullptr;
            if (!Ready && !tilegate && p.n_remote > 0) {
                dist_wait(p, p.wait_sig, p.wait_ptr[m_block], p.wait_ptr[m_block + 1]);
            }
            const bool gate = tilegate && p.n_remote > 0 && p.row_nremote[m_block] > 0;
            // Ready: everything this item reads has landed before it is queued, so the descriptor carries NO gate
            // (a compile-time null: the gate code folds away and the mainloop is sparse_fwd's).
            flash::SparseRowDesc row = Ready
                ? flash::SparseRowDesc{ p.kv_indices + beg,
                                        (p.item_direct ? p.kv_indices : p.kv_mask_nb) + beg,   // global order: slot = position
                                        cnt, p.row_n_masked[m_block], p.row_mask_mb[m_block], 0, 0 }
                : flash::SparseRowDesc{ p.kv_indices + beg,
                                        (p.item_direct ? p.kv_indices : p.kv_mask_nb) + beg,
                                        cnt, p.row_n_masked[m_block], p.row_mask_mb[m_block], 0, 0,
                                        gate ? p.kv_ord_sig + beg : nullptr, p.arrived };
            // segmented rows: the mainloop reads Q at the item's Q slot; the epilogue (below) still
            // writes the partial at the item index
            const cute::tuple<int32_t, int32_t, int32_t, int32_t> q_coord{
                p.item_q ? p.item_q[m_block] : m_block, get<1>(block_coord), get<2>(block_coord),
                get<3>(block_coord)};
            bool tile_valid = mainloop.template mma</*Is_first_step=*/true>(
                fa3.mainloop, (const typename Kernel_traits::Element *)p.k_slots,
                (const typename Kernel_traits::Element *)p.v_slots,
                acc_o, softmax, tidx, seqlen_info, q_coord, shared_storage,
                /*is_strict=*/false, &row);

            scheduler.prefetch_next_work(sched_params, work);
            // re-read the item's coordinates from shared memory (volatile: nothing kept live across the mainloop)
            const int mb_ = ((volatile int *)s_coord)[0];
            const int hb_ = ((volatile int *)s_coord)[1];
            const int bb_ = ((volatile int *)s_coord)[2];
            // A row only this PE contributes to is final here: write O / LSE straight to the output, the way a
            // single-GPU kernel would -- no partial slot, no merge item.
            const int dout = p.item_direct ? p.item_direct[mb_] : -1;
            const cute::tuple<int32_t, int32_t, int32_t, int32_t> o_coord{dout >= 0 ? dout : mb_, hb_, bb_, 0};
            if (tile_valid) {
                Tensor final_scale = softmax.finalize();
                softmax.rescale_o(acc_o, final_scale);
                if (dout >= 0) epilogue.store(fa3.epilogue_final, acc_o, softmax.row_sum, shared_storage, tiled_mma, tidx, o_coord);
                else epilogue_part.store(fa3.epilogue, acc_o, softmax.row_sum, shared_storage, tiled_mma, tidx, o_coord);
            } else {
                if (dout >= 0) epilogue.store_zero(fa3.epilogue_final, tidx, o_coord);
                else epilogue_part.store_zero(fa3.epilogue, tidx, o_coord);
            }

            const unsigned long long t_po0 = p.trace ? dist_now() : 0;
            if (dist_return_partial(p, mb_, s_last_head, s_last_grp) && tidx == 0) s_owe_quiet = 1;
            if (p.trace) { const unsigned long long t1 = dist_now();
                DTR_ADD(p, 9, t1 - t_po0); DTR_ADD(p, 7, t1 - s_t_item0); DTR_ADD(p, 3, 1); }
            if (p.timeline && tidx == 0) {
                unsigned long long *tl = p.timeline + 1 + p.NQ + p.NKV + p.NPART + 2 * (int64_t)mb_;
                tl[0] = s_t_item0; tl[1] = dist_now();
            }
        } else {
            const int mw   = work.tile_idx - p.n_compute_work;
            const int item = mw / p.h;
            const int bidh = mw % p.h;
            scheduler.prefetch_next_work(sched_params, work);
            const unsigned long long t_m0 = (p.trace || p.timeline) ? dist_now() : 0;
            dist_merge_item(p, item, bidh, t_m0, reinterpret_cast<float *>(smem_));
            if (p.trace) { DTR_ADD(p, 8, dist_now() - t_m0); DTR_ADD(p, 10, 1); }
            if (p.timeline && threadIdx.x == 0) {
                unsigned long long *tl = p.timeline + 1 + p.NQ + p.NKV + p.NPART + 2 * (int64_t)p.n_compute
                                       + 2 * ((int64_t)item * p.h + bidh);
                tl[0] = t_m0; tl[1] = dist_now();
            }
        }
    }
    // put_signal leaves inter-node puts un-retired; their SOURCE slots are
    // restaged by the next call, so they must complete before this block exits.
    __syncthreads();
    if (s_owe_quiet) dist_quiet_gated(p);
    dist_trace_end(p);
    dist_credit_release(p);
}

}  // namespace ring_attn
