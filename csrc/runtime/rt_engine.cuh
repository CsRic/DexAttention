#pragma once
// Move data and publish completion signals for forward and backward kernels.
// A message names source/destination offsets, peer, byte count and signal slot.
// Start messages send existing inputs; triggers release forwards or partial
// outputs after their data is ready. fire() activates the lanes in trig_lane
// (engine, copy engine, or both); the scheduler releases forwards on arrival.
//
// Inter-node: one engine thread posts NVSHMEM put-with-signal operations.
// Callers default to host-enqueued copy-engine operations for intra-node data (rt_host.h).
// Engine-lane alternative: engine warps copy through the peer-memory mapping.
// Only one SM sender per GPU writes peers, limiting L2 contention with workers.
// A message with sig < 0 leaves signaling to a later ordered message.
#include <cstdint>
#include "rt_params.h"
#include <nvshmem.h>
#include <nvshmemx.h>
#include "../ring_attn/nvshmem_transport.cuh"

#ifndef RT_SPIN_REPORT
#define RT_SPIN_REPORT (1L << 26)
#endif

namespace rt {

// Append trigger t to the push queue (any thread; the data it sends must be visible first).
__device__ __forceinline__ void pq_push(const EngineParams &e, int t) {
    __threadfence_system();
    const int idx = atomicAdd(e.pq_tail, 1);
    atomicExch(&e.pq[idx], t + 1);
}

// Trigger t fired (its data is complete): queue it for the engine and / or release its copy-engine messages.
__device__ __forceinline__ void fire(const EngineParams &e, int t) {
    if (e.fire_log) e.fire_log[atomicAdd(e.fire_log - 1, 1)] = t;
    const int l = e.trig_lane ? e.trig_lane[t] : 1;
    if (l & 2) { __threadfence_system(); *(volatile unsigned long long *)&e.trig_flag[t] = e.epoch; }
    if (l & 1) pq_push(e, t);
}

// The engine, run by threads [t0, t0 + nt) of block 0 (whole warps); named barrier `bar_id` among them. bcast: one
// shared int. Runs until every start message and queued trigger is sent; if any inter-node put was posted, thread 0
// calls nvshmem_quiet() before returning.
static __device__ __noinline__ void engine(const EngineParams &e, int t0, int nt, int bar_id, volatile int *bcast) {
    const int t = threadIdx.x - t0;
    bool inter = false;
    int n_start = e.n_start;
    if (e.late_flag) {                                         // the host publishes the start messages after the launch
        if (t == 0) { long n = 0; while (*(volatile const unsigned long long *)e.late_flag < e.epoch) { __nanosleep(256); if (++n % RT_SPIN_REPORT == 0) printf("[pe %d engine] late messages not published\n", nvshmem_my_pe()); } }
        asm volatile("bar.sync %0, %1;" :: "r"(bar_id), "r"(nt) : "memory");
        __threadfence();
        n_start = *(volatile const int *)e.late_n_start;
    }
    uint64_t cleared = 0;                                     // thread 0: peers whose credit was seen (P <= 64)
    auto bar = [&]() { asm volatile("bar.sync %0, %1;" :: "r"(bar_id), "r"(nt) : "memory"); };
    auto send = [&](const Msg m) {                            // by value: the copy loop below must not reload it
        if (e.credit && t == 0 && !(cleared >> m.pe & 1)) {    // the peer finished the previous call (it read its slots)
            unsigned ns = 64;
            while (*(volatile const uint64_t *)&e.credit[m.pe] + 1 < e.call_id) { __nanosleep(ns); if (ns < 1024) ns <<= 1; }
            cleared |= 1ull << m.pe;
        }
        if (e.pe_inter[m.pe]) {
            if (t == 0) {
                if (m.sig >= 0) {
                    nvshmem_fence();
                    nvshmem_putmem_signal_nbi(e.base + m.dst, e.base + m.src, m.bytes, &e.sig[m.sig], e.epoch,
                                              NVSHMEM_SIGNAL_SET, m.pe);
                } else {
                    nvshmem_putmem_nbi(e.base + m.dst, e.base + m.src, m.bytes, m.pe);
                }
            }
            inter = true;
            return;
        }
        if (e.credit) bar();                                   // every thread writes: all wait for thread 0's check
        int4 *__restrict__ d = reinterpret_cast<int4 *>(e.base + m.dst + e.pe_delta[m.pe]);
        const int4 *__restrict__ s = reinterpret_cast<const int4 *>(e.base + m.src);
        const int64_t n16 = m.bytes / 16;
        for (int64_t i = t; i < n16; i += nt) d[i] = s[i];
        bar();
        if (m.sig >= 0 && t == 0) { __threadfence_system(); ring_attn::ring_signal_set(&e.sig[m.sig], e.epoch, m.pe); }
    };
    // Order: trig_first == 0 -> every start message, then the queue. trig_first == 1 -> a queued trigger is served
    // before the next start message (relay forwards and partial sums unblock other GPUs; start messages are the bulk
    // inputs, posted in send order). bcast: next work item, >= 0 trigger, -1 - i start message i.
    int i = 0, k = 0;
    if (e.trig_first == 2) {                                  // inputs first: a fired input
        int lo = 0, lin = 0;                                  // trigger, else the next start message, else the oldest
        constexpr int kUsed = -0x7fffffff;                    // output; consumed entries are overwritten with kUsed
        while (i < n_start || k < e.n_pq) {
            if (t == 0) {
                int w = 0x7fffffff;
                long n = 0;
                while (w == 0x7fffffff) {
                    volatile int *q = e.pq;
                    const int tail = min(*(volatile int *)e.pq_tail, e.n_pq);
                    if (e.in_trig_first >= 0) {               // scan written entries for an input (stop at a hole)
                        for (int x = lin; x < tail; ++x) {
                            const int v = q[x];
                            if (v == 0) break;
                            if (v != kUsed && v - 1 >= e.in_trig_first) { __threadfence(); w = v - 1; q[x] = kUsed; break; }
                            if (x == lin && (v == kUsed || v - 1 < e.in_trig_first)) lin = x + 1;
                        }
                        if (w != 0x7fffffff) break;
                    }
                    if (i < n_start) { w = -1 - i; break; }
                    while (lo < tail && q[lo] == kUsed) ++lo;
                    if (lo < tail && q[lo] != 0) { __threadfence(); w = q[lo] - 1; q[lo] = kUsed; break; }
                    __nanosleep(64);
                    if (++n % RT_SPIN_REPORT == 0) printf("[pe %d engine] push %d/%d not queued\n", nvshmem_my_pe(), k, e.n_pq);
                }
                *bcast = w;
            }
            bar();
            const int w = *bcast;
            bar();
            if (w < 0) { send(e.msg[-1 - w]); ++i; }
            else { for (int j = e.trig_ptr[w]; j < e.trig_ptr[w + 1]; ++j) send(e.msg[j]); ++k; }
        }
        if (inter && t == 0) { nvshmem_quiet(); __threadfence_system(); }
        return;
    }
    while (i < n_start || k < e.n_pq) {
        if (t == 0) {
            const volatile int *q = e.pq + (k < e.n_pq ? k : 0);
            int w;
            if (k < e.n_pq && (i == n_start || (e.trig_first && *q != 0))) {
                long n = 0;
                while (*q == 0) {
                    __nanosleep(64);
                    if (++n % RT_SPIN_REPORT == 0) printf("[pe %d engine] push %d/%d not queued\n", nvshmem_my_pe(), k, e.n_pq);
                }
                __threadfence();
                w = *q - 1;
            } else {
                w = -1 - i;
            }
            *bcast = w;
        }
        bar();
        const int w = *bcast;
        bar();
        if (w < 0) { send(e.msg[-1 - w]); ++i; }
        else { for (int j = e.trig_ptr[w]; j < e.trig_ptr[w + 1]; ++j) send(e.msg[j]); ++k; }
    }
    if (inter && t == 0) { nvshmem_quiet(); __threadfence_system(); }
}

}  // namespace rt
