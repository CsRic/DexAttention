#pragma once

// Plain-data parameter struct shared between host (.cpp) and device (.cu)
// translation units — no CUDA device-only syntax allowed here.
#include <cstdint>

namespace ring_attn {

// All-put ring protocol. Symmetric heap per PE (one nvshmem_malloc):
//
// Block roles (heterogeneous persistent kernel):
//   delegate  — block 0 only: NVSHMEM signal poller, fans out to all.
//   push      — blocks 0..num_push_blocks-1: copy K/V to next PE.
//   compute   — all blocks: attention computation over owned tiles.
//
// Fields annotated with the role(s) that read them. Delegate-only fields
// are cold for all other blocks; the compiler naturally avoids loading them
// through the if (blockIdx.x == 0) guard, but they still occupy the kernel
// parameter buffer.
struct Ring_fwd_params {
    using index_t = int64_t;

    void *__restrict__ q_ptr;
    void *__restrict__ o_ptr;
    void *__restrict__ softmax_lse_ptr;   // (b, h, seqlen_q) fp32

    // Chunk buffers, all contiguous (b, seqlen_k, h_k, d). sym_local_k/v =
    // step-0 K/V source: the raw torch tensors (two separate buffers),
    // EXCEPT when the next edge is inter-node — then the host stages them
    // into the heap's KL/VL (putmem_nbi sources must be symmetric). The
    // ping-pong slots are always on the heap with K|V adjacent.
    void *__restrict__ sym_local_k;
    void *__restrict__ sym_local_v;
    void *__restrict__ sym_buf_k[2];  // KB0, KB1 (each with VB adjacent)
    void *__restrict__ sym_buf_v[2];

    uint64_t *__restrict__ sig_base_ptr;   // 2 u64 words: [0]=dst_free, [1]=arrived
    uint64_t sig_epoch;                    // per-call epoch base

    // Intra-PE cross-block coordination.  Single allocation (5*num_pes ints):
    //   [0*P .. 1*P) compute_done   [1*P .. 2*P) arrived_flag
    //   [2*P .. 3*P) push_go        [3*P .. 4*P) push_done
    //   [4*P .. 5*P) tile_count
    int *__restrict__ coord_base;

    // Per-(work item, thread) online-softmax spill area (step-outer loop).
    float *__restrict__ acc_scratch_buf;

    index_t q_batch_stride, q_row_stride, q_head_stride;
    index_t o_batch_stride, o_row_stride, o_head_stride;

    int b, h, h_k, h_h_k_ratio;
    int seqlen_q, seqlen_k, d;

    float scale_softmax;
    float scale_softmax_log2;

    int my_pe, num_pes;
    int prev_pe, next_pe;

    bool is_causal;

    // Whether the me→next edge stays inside one node (detected on the host
    // via nvshmem_ptr(): P2P-mapped ⇔ same node in practice; note
    // NVSHMEM_DISABLE_P2P=1 makes intra-node neighbours take the inter-node
    // path — useful for exercising the NIC path on one machine).
    bool next_intra_node;

    // Valid iff next_intra_node: byte delta mapping a local symmetric
    // address to next_pe's copy of the same object.
    int64_t next_intra_delta;

    // Blocks that participate in the push (small bound on concurrent nvshmem_quiet() callers).
    int num_push_blocks;

    // Keep the ring signalling but skip the K/V byte movement (off in the shipped paths).
    bool skip_push_data;
};

}  // namespace ring_attn
