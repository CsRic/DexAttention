#pragma once

#include <nvshmem.h>
#include <nvshmemx.h>

namespace ring_attn {

// Intra-node push: block-cooperative vectorized st-loop into the P2P-mapped
// remote slot. Same int4 stride pattern NVSHMEM's own
// nvshmemi_memcpy_threadgroup uses on this path, minus its per-call
// barriers/branches. Peer WRITES are posted while peer reads are latency-bound,
// hence all-put. Slices are 256B-aligned so int4 (16B) always is. Completion:
// __syncthreads() after the call, then one thread's __threadfence_system().
__device__ __forceinline__ void intra_push_slice_block(
    void *dst_remote, const void *src_local, size_t nbytes
) {
    int4       *d = reinterpret_cast<int4 *>(dst_remote);
    const int4 *s = reinterpret_cast<const int4 *>(src_local);
    const size_t n = nbytes / sizeof(int4);
    // Not unrolled: a single writer with a lower injection rate interferes less
    // with the computing blocks' L2 traffic.
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        d[i] = s[i];
    }
}


// Point-to-point signal set / wait (uint64 words on the symmetric heap;
// values epoch-monotonic, so all waits are GE).
__device__ __forceinline__ void ring_signal_set(uint64_t *remote_sig, uint64_t value, int pe) {
    nvshmemx_signal_op(remote_sig, value, NVSHMEM_SIGNAL_SET, pe);
}

// Unused in this release (only the ring kernel, which is not built, calls it).
__device__ __forceinline__ void ring_signal_wait_ge(uint64_t *local_sig, uint64_t expected) {
    nvshmem_signal_wait_until(local_sig, NVSHMEM_CMP_GE, expected);
}

// Intra-PE cross-block flags/counters (regular device memory). Only block 0
// polls NVSHMEM signal words (concurrent pollers on one symmetric word contend
// heavily). These helpers are unused in this release (only the ring kernel,
// which is not built, calls them).
__device__ __forceinline__ void flag_set(int *flag, int v)      { atomicExch(flag, v); }
__device__ __forceinline__ void counter_inc(int *counter)       { atomicAdd(counter, 1); }
__device__ __forceinline__ int  counter_load(int *counter)      { return atomicAdd(counter, 0); }
__device__ __forceinline__ void counter_wait_ge(int *counter, int v) {
    while (counter_load(counter) < v) { /* spin on local memory */ }
}

}  // namespace ring_attn
