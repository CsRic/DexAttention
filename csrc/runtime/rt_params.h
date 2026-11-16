#pragma once
// Plain-data parameters of the shared runtime (host- and device-includable); device code in rt_engine.cuh,
// design in docs/design.md.
#include <cstdint>

namespace rt {

struct Msg { int64_t src, dst, bytes; int32_t pe, sig; };      // 32 bytes; python: int64 [n, 4], col 3 = pe | sig << 32
static_assert(sizeof(Msg) == 32, "Msg is read from an int64 [n, 4] table");

struct EngineParams {
    char *base;                       // this PE's region (offset 0)
    uint64_t *sig;                    // signal words (symmetric; this PE's view)
    uint64_t epoch;
    const Msg *msg;                   // [n_start start messages | trigger messages]
    int n_start;
    const int32_t *trig_ptr;          // trigger t -> msg[trig_ptr[t] .. trig_ptr[t + 1])
    int *pq, *pq_tail;                // push queue: entry = trigger + 1 (0 = not written yet)
    int n_pq;                         // entries this PE queues in this call (the engine stops after them)
    const int64_t *pe_delta;          // [P] byte delta of the peer's P2P view (intra-node), else 0
    const int32_t *pe_inter;          // [P] 1 -> NIC
    const uint64_t *credit;           // pairwise cross-call credits (symmetric, this PE's view), nullptr = off
    uint64_t call_id;
    int trig_first;                   // 1: a queued trigger goes before the next start message (rt_engine.cuh);
                                      // 2: inputs first -- a queued INPUT trigger (id >= in_trig_first, e.g. a relay
                                      // forward) before the next start message, outputs after every start message
    int in_trig_first;                // first input trigger id (trig_first 2), -1 = none
    // per trigger: lane bits (1 = its messages go by the engine -> push queue, 2 = by copy engine -> the host-enqueued
    // stream waits on trig_flag[t] = epoch, rt_host.h); nullptr = all engine
    const int32_t *trig_lane;
    unsigned long long *trig_flag;
    int *fire_log;                    // diagnostic: trigger ids in firing order; fire_log[-1] = count (null in normal runs)
    // late start messages (the InfLLM forward: the kernel launches before the host plan): the engine first waits for
    // *late_flag >= epoch, then takes the start-message count from *late_n_start (msg's contents were written before
    // the flag). nullptr = the n_start above.
    const unsigned long long *late_flag; const int *late_n_start;
};

// The block-0 scheduler (rt_sched.cuh), shared by the forward and the backward: every arrival marks what it unblocks;
// whenever the task queue runs low, ROWS (the resident side of a compute task: fwd a Q tile, bwd a KV tile + head + twin
// pair) are cut into TASKS of consecutive READY GROUPS (items with one set of remote dependencies) in priority order.
struct SchedParams {
    // arrivals: polled signal ids (remote_sig, n_remote), epoch-valued words in `sig`; remote_cnt[j] > 1: signal j
    // closes a run of that many ids (they are all marked in `arrived`, which may be nullptr)
    uint64_t *sig; uint64_t epoch;
    int n_remote; const int32_t *remote_sig, *remote_cnt; int *arrived;
    // rows and groups: row r = groups [row_grp_ptr[r], row_grp_ptr[r + 1]); group g = items [grp_beg[g], grp_end[g]),
    // ready when grp_need[g] signals of sig_grp have landed (grp_need nullptr: one, or none when grp_dep[g] < 0);
    // row_qdep[r]: remote_sig index of the row's own (resident-side) input, or -1
    int n_rows, n_groups;
    const int32_t *row_grp_ptr, *grp_beg, *grp_end, *grp_dep, *grp_need, *row_qdep, *row_order, *row_pool;
    int *grp_cnt;                     // per-call scratch, groups with grp_need > 1
    const int32_t *sig_grp_ptr, *sig_grp, *sig_row_ptr, *sig_row;   // per polled signal: groups / rows it feeds
    // REDUCTION units (owner merges / sums, gateway aggregations): unit m is queued as red_n entries (kind red_kind,
    // payloads red_pay + i) when red_need inputs are in; inputs = arrivals (sig_red CSR) and local completions
    // (rt::red_count from the workers). red_kind nullptr: every unit is (kind_red, m * red_n_all, red_n_all).
    int n_red; const int32_t *red_need, *red_kind, *red_pay, *red_n; int kind_red, red_n_all;
    const int32_t *sig_red_ptr, *sig_red;
    int *red_cnt, *n_enq;             // per-call scratch: inputs seen per unit, units queued
    int *rows_done;                   // rows whose output is complete (stop condition: all rows, all units queued)
    // the task queue: entry = ((kind << 28) | payload) + 1; a task of row r gets `sub` entries (fwd: heads, bwd: the
    // two 64-row halves), queued in GANGS of `gang` consecutive entries (a gang of workers starts them together)
    int *sq, *sq_tail, *tile_counter; int n_static;   // claimed front = *tile_counter + n_static
    int *task_desc, *row_ntask;       // per task (row, first group, end group, pool slot or -1), tasks per row
    int sub, gang, kind_task, kind_stop, kind_nop, n_workers;
    int sched_low, sched_crit, sched_batch;
    int *q_work;                      // non-null: thresholds in work units (items x sub), else in entries
    // relay forwards by the engine: arrival of polled signal j fires trigger fwd_trig + j if it has messages (-1: off)
    int fwd_trig;
    char *b0_state; int64_t b0_bytes; // block 0's arrays when they do not fit in shared memory (else nullptr)
    // readiness UNITS (rectangle segments: one segment is shared by the tile rows of a block row): grp_unit[g] = the unit
    // group g becomes ready with (nullptr: the group itself); sig_grp then lists units; unit_dep[u] < 0: resident
    int n_units; const int32_t *grp_unit, *unit_dep;
    const int32_t *grp_work;          // work of a group (tiles; nullptr: grp_end - grp_beg)
    // diagnostics (nullptr = off): timeline stamps (tl_sig[id] arrival of signal id, tl_row[2 r] first dispatch of
    // row r, tl_ev[1] scheduler ready / [2] first task queued / [3] STOP queued), counters (trace[8] kernel
    // start; written: [11] tasks, [12] rows split, [13] last dispatch from start)
    unsigned long long *tl_sig, *tl_row, *tl_ev, *trace;
    // *q_all_in = 1 once every polled signal with id < q_lim (the inbound Q) has landed (nullptr = off)
    int *q_all_in; int q_lim;
    // only with sched_block0<kAvail = 1>: at most grp_max (>= 1) groups per task, and only groups whose bit is set
    // in grp_avail_bits are dispatched -- helpers (the backward's block-0 warps 1-3) materialise the groups' work lists,
    // in their own order, and set the bits. kAvail = 2: only rows row_order[0 .. *row_avail) are dispatched (helpers
    // that materialise rows in priority order -- the forward's -- publish a prefix: one read per dispatch pass)
    int grp_max; const unsigned *grp_avail_bits; const int *row_avail;
    // arrivals whose sig_grp list is longer than coop_len are processed by the whole warp (lanes over the entries)
    // instead of by one lane (lanes over signals); 0 = off (the forward's many short lists)
    int coop_len;
};

}  // namespace rt
