#pragma once
// Plain-data parameters of the distributed sparse-attention BACKWARD megakernel.
//
// Per PE, symmetric heap (same layout and sizes on every PE; global maxima NQS, NKS, NDQL, NDKL):
//   sig      u64 [NQS + NKS + NDQL + NDKL]        one word per signal id (set to the call's epoch)
//   qb       Q bundle slots [NQS]: [Q 128 x h x d fp16][dO same][LSE h x 128 fp32][Delta h x 128 fp32]
//   kv       KV slots [NKS]: [K 128 x hk x d fp16][V same]
//   dqa      fp32 dQ accumulators [NQS] (128 x h x d, FA2's permuted 64x128 blocks), one per Q slot computed here
//   dkva     fp32 [dK | dV] accumulators [NKS] (2 x 128 x hk x d)
//   dql      dQ landing slots [NDQL] (fp32, as dqa)      partial sums pushed home by contributors
//   dkl      dKV landing slots [NDKL] (fp32, as dkva)
// Signal ids: Q slot s -> s; KV slot s -> NQS + s; dQ landing l -> NQS + NKS + l; dKV landing l -> NQS+NKS+NDQL+l.
//
// Kernel: block 0 = scheduler (never computes): polls every remote signal; counts each into the tasks / owner sums
// waiting on it; queues a task when all its inputs are resident, an owner sum when all its contributions are in;
// queues one STOP per worker at the end. Block 0's communication engine sends the input pushes (start messages) and
// every partial sum.
// Workers (1..G-1) claim queue entries in order:
//   compute task t: KV slot ks, half hn, KV head g, items (Q slot, half, q head, causal) -> sparse_bwd::bwd_task
//                   (atomic dK/dV into dkva, atomic dQ into dqa); then counts the task into every Q slot / the KV slot
//                   it touched; the block completing a slot either pushes its accumulator to the owner's landing slot
//                   (remote owner) or counts it into the owner sum here.
//   dQ owner sum m, sub (half, q head): sum of the local accumulator and the landing slots -> dQ fp16 (x scale,
//                   un-permuted) at owned row out_row
//   dKV owner sum m, KV head g: sum -> dK, dV fp16
#include <cstdint>
#include "sparse_bwd_params.h"
#include "../runtime/rt_params.h"
namespace rt { struct SchedB0; }

namespace dist_bwd {

struct Params {
    // symmetric regions (this PE's view)
    char *qb, *kv, *dqa, *dkva, *dql, *dkl;
    uint64_t *sig;
    uint64_t sig_epoch;
    int NQS, NKS, NDQL, NDKL;
    int64_t qb_bytes, kv_bytes, dqa_bytes, dkva_bytes;   // per slot

    int h, hk, d;
    float scale, scale_log2;

    // final outputs (plain device memory; rows owned here, [n_owned_q*128, h, d] / [n_owned_kv*128, hk, d])
    void *dq_out, *dk_out, *dv_out;

    // compute tasks
    int n_tasks;
    const int32_t *task_kv;          // (kv_slot << 9) | (hn << 8) | g
    const int32_t *task_ptr, *items; // items as sparse_bwd (q_slot << 9 | mb << 8 | hq << 1 | causal)
    const int32_t *task_end;         // task t's items [task_ptr[t], task_end[t]) (per-tile mode: task_ptr + 1)
    const int32_t *task_qs_ptr, *task_qs;   // distinct Q slots each task touches
    const int32_t *task_need;        // remote inputs per task (0 -> queued at start, in init order)
    const int32_t *init_tasks; int n_init;  // tasks with no remote input, in dispatch order
    // slots: completion counts and what to do when a slot's last task is done
    const int32_t *qs_need, *ks_need;       // tasks per Q / KV slot (0 = not computed here)
    const int32_t *qs_push_pe, *qs_push_dst;   // remote owner: pe, its dQ landing slot; else -1 and the owner sum index
    const int32_t *ks_push_pe, *ks_push_dst;
    // owner sums
    int n_qm, n_km;
    const int32_t *qm_need, *qm_out, *qm_local, *qm_ptr, *qm_land;   // local acc slot or -1, landing slots
    const int32_t *km_need, *km_out, *km_local, *km_ptr, *km_land;
    // remote signals this PE waits for: sig_ids[j]; what each feeds
    int n_remote;
    const int32_t *remote_sig;
    const int32_t *sig_task_ptr, *sig_task;          // tasks needing signal j
    const int32_t *sig_qm, *sig_km;                  // owner sum fed by signal j (landing), or -1
    // node relay (dex/scheduler/bwd.py): this PE as a node GATEWAY. Aggregation: a slot with agg_need > 0 adds the
    // landings agg_land[agg_ptr[s]..] (its node's other contributors) into its own accumulator once its own tasks are
    // done and they all landed (agg_need = 1 + landings), then pushes the sum on (push_pe / push_dst as usual).
    const int32_t *qs_agg_need, *qs_agg_ptr, *qs_agg_land, *ks_agg_need, *ks_agg_ptr, *ks_agg_land;
    const int32_t *sig_qa, *sig_ka;                  // gateway landing: the local slot it adds to, or -1
    // forwards: sig_fwd[j] = 1 when remote signal j is an input this gateway forwards inside its node (on arrival the
    // scheduler queues trigger NQS + NKS + j: copies to the node's other readers)
    const int32_t *sig_fwd;
    int n_xenq;                      // aggregations this PE queues (the scheduler's stop condition)
    // every send: block 0's communication engine (runtime/rt_engine.cuh) -- the input pushes are its start messages;
    // triggers: Q slot s -> s (partial sum home / to the gateway), KV slot s -> NQS + s, forwards of remote signal j ->
    // NQS + NKS + j
    rt::EngineParams eng;
    // the shared block-0 scheduler (runtime/rt_sched.cuh): rows = twin
    // pairs, reduction units = [dQ sums | dKV sums | Q-slot aggregations | KV-slot aggregations]; qa_unit / ka_unit:
    // the unit of a gateway slot; row_cnt: finished halves per pair; sch_b0: block 0's control words (global)
    rt::SchedParams sch;
    int sch_on;
    const int32_t *qa_unit, *ka_unit;
    int *row_cnt;
    rt::SchedB0 *sch_b0;
    // RECTANGLE mode (rect = 1, dex/scheduler/bwd.py RECT_ORDER): the planner only gives rectangles (KV plan block x the
    // Q rows of one owner) and their groups (column x <= 64 rows); scheduler rows = groups (one each, head-major:
    // g = head * G1 + g1). Block 0's warps 1-3 (rect_block0) materialise each group's two tasks 2 g + half (task_kv,
    // task_ptr / task_end, items = arena: device buffers they write) in the planner's expansion order (exp_order:
    // groups without remote inputs first, then by expected input landing), 8 per claim, and set the group's bit in
    // avail_bits (the scheduler dispatches only materialised groups); warps 1-3 also count finished groups (done_q, from
    // the workers) into the slots, firing the partial-sum sends / reduction units. The workers' code is the per-tile
    // mode's (complete_task only queues the group). Freshness: a worker reads a task's tables after its acquire fence
    // on the queue entry, which follows the publication.
    int rect, n_rect, G1, per, nw, hg;   // hg: q-head groups per planner group (n_groups / G1; h / hg q heads each)
    const int32_t *g_rect, *g_col, *g_k0, *g_cnt, *g_ks, *rect_pp, *rect_pair, *rect_bk, *pair_bq, *qslot_t, *kslot_t,
        *exp_order;
    const uint64_t *pair_bits, *pair_dbits, *pair_prow, *pair_pcol;
    // 128 x 64 compute: per local pair a per x (2 per) bitmap of live 64-key halves (bit a * 2 per + 2 b + hn,
    // nw2 words); null = both halves of every live tile. pair_bits stays their union (ranks / counts).
    const uint64_t *pair_hbits; int nw2;
    int32_t *arena, *done_q, *done_tail, *rowbuf, *exp_next, *arena_off;
    unsigned *avail_bits;
    // transport (per destination PE)
    const int64_t *pe_delta;         // local -> peer byte delta of the symmetric heap (intra-node P2P), 0 otherwise
    const int32_t *pe_inter;         // 1 -> NIC path (put-with-signal)
    // per-call scratch (zeroed by the host)
    int *sq, *sq_tail, *tile_counter, *task_cnt, *qs_done, *ks_done, *qm_cnt, *km_cnt, *n_enq, *quiet_gate;
    int *qa_cnt, *ka_cnt;            // aggregation counts per Q / KV slot
    int *mail;                       // per worker gang: [0] sequence, [1] queue position, [2] partner ack (32 ints per gang)
    int sched_low;                   // block 0 dispatches ready tasks (priority order) while fewer are queued
    int n_workers;
    // DEX_TRACE=1: u64 counters summed over workers (ns): 0 queue wait, 1 compute tasks, 2 completion + partial
    // push, 3 dQ owner sums, 4 dKV owner sums, 5 tasks, 6 first worker start, 7 last worker end; else nullptr
    unsigned long long *trace;
    // memory-lifetime stamps (null in normal runs): globaltimer when
    // [Q slot s done (all its tasks) | KV slot done | dQ owner sum m done | dKV owner sum done | Q gateway aggregation
    // done | KV aggregation done]; sizes NQS, NKS, n_qm, n_km, NQS, NKS
    unsigned long long *ml;
    // the compute routine's view of this PE's slots, filled by the host (kernel parameter space, i.e. the constant
    // bank, which the mainloop reads without a shared-memory access)
    sparse_bwd::Params sp;
};

}  // namespace dist_bwd
