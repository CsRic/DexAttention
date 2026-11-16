#pragma once
// Plain-data parameters for the distributed forward megakernel.
#include <cstdint>
#include "../runtime/rt_params.h"

namespace ring_attn {

// One persistent launch per GPU runs an EXTERNAL assignment: push the Q and KV
// tiles peers need, compute the tiles this device was given (wherever their Q
// lives), push each resulting partial back to the Q owner, and merge the
// partials for the rows this device owns.
//
// Slot spaces and the single signal id space they share:
//     q slot s        -> signal id s
//     kv slot s       -> signal id NQ + s
//     partial slot s  -> signal id NQ + NKV + s
// NQ / NKV / NPART are GLOBAL layout sizes, so an id means the same thing on
// every PE -- which is what lets a sender signal a receiver's slot directly.
struct Dist_fwd_params {
    using index_t = int64_t;

    // ---- symmetric buffers ----
    void *__restrict__ q_slots;     // fp16 [NQ*128][h][d]
    void *__restrict__ k_slots;     // fp16 [NKV*128][h_k][d]
    void *__restrict__ v_slots;
    void *__restrict__ part_o;      // fp32 [NPART*128][h][d]
    void *__restrict__ part_lse;    // fp32 [h][NPART*128]

    // ---- this PE's own output (plain device memory) ----
    void *__restrict__ o_ptr;       // fp16 [n_owned*128][h][d]
    void *__restrict__ lse_ptr;     // fp32 [h][n_owned*128]

    index_t q_row_stride, q_head_stride;
    index_t o_row_stride, o_head_stride;     // inside part_o
    index_t fo_row_stride, fo_head_stride;   // final O
    index_t lse_head_stride;                 // NPART*128
    index_t flse_head_stride;                // n_owned*128

    int b, h, h_k, d;
    int seqlen_q, seqlen_k, seqlen_o;        // NQ*128, NKV*128, NPART*128
    float scale_softmax;

    // ---- the mask this PE computes, in slot space ----
    const int32_t *kv_indptr, *kv_indices, *kv_mask_nb;
    const int32_t *row_n_masked, *row_mask_mb;
    const int32_t *work_coord;               // claim order over the compute rows
    int n_compute;                           // rows this PE computes
    int n_compute_work;                      // n_compute * h * b
    int n_work;                              // + n_merge * h
    int *tile_counter;

    // ---- arrivals ----
    uint64_t *sig_base;                      // symmetric, one word per signal id
    uint64_t  sig_epoch;
    // ---- cross-call slot reuse (write-after-read), pairwise ----
    // credit[q] (symmetric, on this PE) = the last call id PE q has FINISHED (it no longer reads any slot of an
    // earlier call). Before writing into q's slots in call `call_id`, wait credit[q] >= call_id - 1. Each PE
    // publishes its finished call id to every peer when its last block exits. nullptr = disabled.
    uint64_t *credit;
    uint64_t  call_id;
    int *blocks_done;
    int my_pe, n_pes;
    int *arrived;                            // device, one int per signal id
    const int32_t *remote_sig;               // ids this PE must wait for
    int n_remote;
    const int32_t *wait_ptr, *wait_sig;      // per compute row: REMOTE KV only
    const int32_t *row_nremote;              // ordinals [0, n) are the remote tail
    const int32_t *row_qwait;                // remote Q signal id, or -1
    const int32_t *kv_ord_sig;               // per ordinal: signal to wait on, or -1

    // ---- sending a finished partial home ----
    const int32_t *opush_pe;                 // per compute row, -1 if owned here
    const int32_t *opush_slot;               // the owner's landing slot
    int *o_head_done;                        // per compute row, counts finished heads
    int *quiet_gate;                         // bounds concurrent nvshmem_quiet()

    // ---- merging the partials for rows this PE owns ----
    const int32_t *merge_out, *merge_ptr, *merge_slot;
    const int32_t *mwait_ptr, *mwait_sig;
    const int32_t *merge_self;   // compute row whose LOCAL partial must be done, or -1
    int n_merge;

    // ---- outgoing Q / KV ----
    const int32_t *pushq_src, *pushq_pe, *pushq_dst;
    const int32_t *pushkv_src, *pushkv_pe, *pushkv_dst;
    int n_pushq, n_pushkv, num_push_blocks;
    // One queue over both kinds: entry e < n_pushq is Q message e, else KV
    // message e - n_pushq. Null -> all Q then all KV.
    const int32_t *push_order;
    // First push block. 0: block 0 pushes and THEN delegates. 1: block 0 only
    // delegates (polls from the start); pushes run on blocks 1..num_push_blocks.
    int push_block_base;
    // Inter-node messages carry their own signal (put-with-signal + fence)
    // instead of one quiet + bulk signal per push block / per partial.
    int put_signal;

    // Per-destination, not per-message: the transport is a property of the edge.
    const int64_t *pe_delta;                 // [P] local->peer byte delta, intra only
    const int32_t *pe_inter;                 // [P] 1 -> NIC path

    // Segmented rows: a compute ITEM is (Q slot item_q[i], its own kv list, its own partial slot i).
    // Several items may share a Q slot (one per KV source segment), so a row can start on the data that
    // has arrived instead of waiting for its last remote tile. nullptr -> item i uses Q slot i.
    const int32_t *item_q;
    // Coalesced pushes: entry i covers cnt[i] consecutive slots at both ends; its signal id is the LAST
    // destination slot's id. nullptr -> one slot per entry.
    const int32_t *pushq_cnt, *pushkv_cnt;
    // Staged partial-O return: item i belongs to group item_group[i] (-1 -> sent alone). A group's items have
    // consecutive item (= partial) slots [group_first, +group_size) here and consecutive landing slots
    // [group_dst_slot, +group_size) on group_dst_pe; the block finishing the group's LAST item sends it as one
    // message, signalled on its last landing slot. group_done: per-call scratch counters.
    const int32_t *item_group, *group_size, *group_first, *group_dst_pe, *group_dst_slot;
    int *group_done;
    // merge: local items whose partial must be finished before merging an owned row (a list, because a
    // row may have several local segments). nullptr -> the single merge_self[] entry.
    const int32_t *mself_ptr, *mself_idx;

    // diagnostic per-call timing counters (u64, see dist_fwd_kernel.h); nullptr in normal runs
    unsigned long long *trace;
    // diagnostic timeline: [t0 | arrival time per signal id (n_sig) | (start, end) per compute item | (start, end)
    // per merge item], globaltimer ns; nullptr in normal runs
    unsigned long long *timeline;
    // unused: inter-node partials always go by NVSHMEM put (the host sets nullptr)
    unsigned long long *hs_pflag;

    // Global-order KV (dims[9]): KV slots are in GLOBAL tile order on every PE (slot = tile = position), so
    // kv_indices doubles as the mask geometry (kv_mask_nb unused) and a remote tile t waits on its own signal
    // NQ + t, which the delegate sets for every tile of an arrived run (remote_cnt[j]: the run ending at
    // remote_sig[j] covers ids remote_sig[j]-cnt+1 .. remote_sig[j]).
    // item_direct[i] >= 0: this PE owns the row and is its only contributor -> the epilogue writes the FINAL
    // O / LSE at that output row (no partial slot, no merge item). Both nullptr when off.
    const int32_t *item_direct, *remote_cnt;
    // Ready queue (dims[10]): block 0 owns every data dependency, compute blocks carry none. Item i needs
    // dep_need[i] distinct remote signals (its Q, its KV runs); dep_item[dep_ptr[j] .. dep_ptr[j+1]) are the items
    // waiting on remote_sig[j]. When the delegate sees signal j it counts it into those items (dep_cnt) and appends
    // every head of each item that became complete to ready_q (entry = packed work coordinate + 1, 0 = not yet).
    // Work index t < n_init_work reads work_coord[t] (the items with no remote input, in claim order); t beyond
    // reads ready_q[t - n_init_work], waiting only while it is still empty -- i.e. only when no work is ready.
    const int32_t *dep_need, *dep_ptr, *dep_item;
    int *dep_cnt, *ready_q, *ready_tail;
    int n_init_work;

    // ---- block-0 task scheduler (dims[11]; dist_sched_kernel.h) ----
    // Block 0 never computes: it watches every arrival and hands the workers their next task, a (row, head, group of
    // tiles whose data is here). Workers run sparse_fwd's mainloop with no gate; a row that ends up in several tasks
    // gets one partial per task (in a local pool) and a local merge, queued by whoever finishes its last tile.
    // Static (host): row r's KV list kv_indices[kv_indptr[r] ..) is in storage order [remote, latest expected
    // arrival first][local][masked diagonal]; its groups (consecutive tiles with one dependency) are
    // [row_grp_ptr[r], row_grp_ptr[r+1]), group g = ordinals [grp_beg[g], grp_end[g]) needing remote_sig[grp_dep[g]]
    // (-1: resident). row_qdep[r]: remote_sig index of the row's Q, or -1. sig_*_ptr / sig_*: per remote_sig index,
    // the groups / rows (Q) / owner merges it feeds. row_order: dispatch priority. row_pool[r]: first pool slot of
    // its partials (-1: one group, never split). row_merge[r]: the owner merge this row's output feeds, or -1;
    // merge_need[m]: inputs of owner merge m (landing partials + the local row).
    const int32_t *row_grp_ptr, *grp_beg, *grp_end, *grp_dep, *row_qdep;
    const int32_t *sig_grp_ptr, *sig_grp, *sig_row_ptr, *sig_row, *sig_merge_ptr, *sig_merge;
    const int32_t *row_order, *row_pool, *row_merge, *merge_need;
    int n_groups, sched_low, sched_crit, sched_batch;
    // per-call scratch (zeroed): the queue and its tail, task descriptors (row, beg, end, out) and per-task finished
    // heads, per-row finished tiles / task count, per-merge inputs seen, rows written, owner merges queued
    int *sq, *sq_tail, *task_desc, *task_heads, *row_tiles_done, *row_ntask, *merge_cnt, *rows_done, *merges_enq;
    void *pool_o; float *pool_lse;            // local partial pool: fp32 [n_pool*128][h][d], fp32 [h][n_pool*128]
    int64_t pool_lse_stride;
    // Bounded pool: a split row's task writes its one-head partial into the worker's
    // scratch slot (pool_scratch0 + worker) and merges it at once into the row's accumulator (pool slot r) under a
    // (row, head) lock; the row's local merge then has one input. Pool = rows + workers slots, independent of
    // the number of tasks. acc_lock / acc_cnt: [n_compute * h], zeroed per call.
    int pool_acc, pool_scratch0, pool_nws;
    int *acc_lock, *acc_cnt;

    // Partial-slot layout (MergeLayout, dist_merge.cuh): elements between consecutive slots of part_o / part_lse
    // (LSE head stride = lse_head_stride). Head-major (default): 128*h*d / 128 with lse_head_stride = NPART*128.
    // part_combined (dist_sched_kernel): each slot is [O 128 x h x d fp32][LSE h x 128 fp32] contiguous, so a
    // partial goes home as ONE message of o_slot_bytes (= the whole slot) instead of 1 + h.
    int64_t part_o_slot, part_lse_slot;
    int part_combined;
    // Non-null -> a partial is pushed home only once every Q this PE receives has landed (block 0 sets *q_all_in).
    // nullptr in the shipped paths.
    int *q_all_in;
    // Work-bounded task queue (sched kernel): queued, not yet started work in
    // tile x head units; block 0 adds a task's tiles x heads before queueing it, a worker subtracts its tiles when it
    // starts the (task, head). sched_low / sched_crit are then work thresholds.
    int *q_work;
    // Block 0's scheduler state (per group / row / signal arrays) lives in shared memory when it fits; for large
    // problems (e.g. per-head virtual blocks: thousands of rows) in this global buffer instead (b0_bytes needed).
    char *b0_state;
    int64_t b0_bytes;
    // Intra-node partials by COPY ENGINE (sched kernel): the block that closes the row stores the call's epoch into
    // row_flag[row], and the host-enqueued copy-engine stream (waiting on that word) copies the slot and signals.
    unsigned long long *row_flag;
    // Intra-node relay (sched kernel): a cross-node Q/KV block several GPUs of a node need crosses the NIC once, to a
    // gateway GPU, whose host-queued copy-engine "forward" stream copies it on to the others (waiting on the
    // gateway's arrival signal). The forward stream finally stores the call's epoch here; the last block waits for
    // it before releasing credits -- remote senders of the next call overwrite the slots the forwards read.
    unsigned long long *fw_done;

    // ---- rectangle segments (sched kernel, rect != 0) ----
    // Readiness is tracked per (block row, column segment): a segment = the columns of one block pair that arrive with
    // one signal, the same for every tile row of the block row (a rectangle of tiles with one data dependency). Block
    // row b's segments [brow_seg_ptr[b], +1) are in storage order; segment s covers columns seg_cb[s] + j for the set
    // bits j of seg_mask[s] in pair seg_pair[s], waits on remote_sig[seg_dep[s]] (-1: resident). Tile row bi of a pair
    // has live columns pair_rows[pair * per + bi] (masked diagonal: pair_drows, nullptr -> none). A compute row r is
    // tile row row_bi[r] of block row row_brow[r]; its per-(row, segment) dispatch state starts at row_seg_base[r].
    // A task (row, segments [s0, s1)) is expanded into a KV list by the worker itself (rect_list: one scratch list of
    // kMaxRowTiles + 2 entries per block); sig_grp_ptr / sig_grp map signals to SEGMENTS.
    int rect, per, n_segs, n_rowseg;
    const int32_t *row_brow, *row_bi, *row_seg_base, *brow_seg_ptr, *seg_cb, *seg_pair, *seg_dep;
    const uint64_t *seg_mask, *pair_rows, *pair_drows;
    int *rect_list;

    // Block-0 expansion (dx_on, dex/scheduler/fwd.py tables_blocks dx; shared scheduler only): the planner hands over
    // each row's emitted SEGMENTS (first column, 64-bit mask as low / high words, local flag; row r's are
    // [dx_row_seg_ptr[r], dx_row_seg_ptr[r + 1])); block 0's warps 1-3 write the KV lists (kv_indices = column, kv_ord_sig
    // = -1 local else NQ + column, at kv_indptr[r] on) row by row in dispatch priority (row_order, dx_next = claim
    // counter) and set the row's groups in dx_bits; the scheduler dispatches only groups whose bit is set
    // (rt::sched_block0<2>: dispatch only below the published row prefix). The workers' code is unchanged.
    int dx_on;
    int kv_half;        // 128 x 64 compute lists (dims[26]): segment bits are 64-key halves (base 2 cb), kBlockN 64
    const int32_t *dx_seg_cb, *dx_seg_lo, *dx_seg_hi, *dx_seg_loc, *dx_row_seg_ptr;
    // compact KV slots: dx_kslot = this PE's tile -> KV slot (block 0 writes kv_indices = slot,
    // kv_geom = tile); kv_geom = the mask geometry the workers hand the mainloop (= kv_indices without compaction)
    const int32_t *dx_kslot, *kv_geom;
    unsigned *dx_bits;
    int *dx_next;

    // Shared communication engine (sched kernel; runtime/rt_engine.cuh): use_engine != 0 ->
    // block 0's engine warps (1..7, or 4..7 with dx_on) send every INTER-node message -- the Q / KV pushes (start
    // messages, in push_order) and the partial O of every row owned on another node (trigger = the row, queued by the block that closes it) -- instead
    // of push block 1 and the workers. use_engine == 2: EVERY message -- also the intra-node Q / KV runs (start
    // messages), the relay forwards (trigger n_compute + j when remote signal j arrives) and the intra-node partials --
    // instead of the copy engines. use_engine == 3: inter-node by the engine, intra-node by the shared copy-engine
    // executor (runtime/rt_host.h) from the same message table -- the transport the backward uses. Offsets are from
    // q_slots (the region's slot area).
    rt::EngineParams eng;
    int use_engine;
    int eng_grp_trig;              // the first staged-group trigger (staged partial O groups: trigger + group)
    // the shared block-0 scheduler (runtime/rt_sched.cuh), filled by the host from the fields above; sp_on = 0: this
    // kernel's own dist_sched_block0
    rt::SchedParams sp;
    int sp_on;

    int NQ, NKV, NPART;
    int64_t q_slot_bytes, kv_slot_bytes, o_slot_bytes, lse_slot_bytes;
};

}  // namespace ring_attn
