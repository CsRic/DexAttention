#pragma once
// Plain-data parameters of the InfLLM-v2 backward megakernel.
//
// Per PE, symmetric region (same layout on every PE):
//   sig    u64 [NSIG]                         signal words (this call's epoch when set)
//   kv     bf16 [nb * 64 rows, 512]           K | V, global layout (row = global key token), as the forward's region
//   acc    fp32 [nb * 64 rows, 512]           dK (2 heads x 128) | dV, accumulated by the tasks' epilogues (block n = rows 64 n ..)
//   land   fp32 [NL][64 rows, 512]            landing slots of partial sums (one per (receiver, block, sender), static)
// Signal ids: run arrivals use the forward's static ids (run * NG + group); landing l -> n_run_sig + l.
//
// Kernel: block 0 = shared scheduler (warp 0, runtime/rt_sched.cuh) + communication engine (the other warps,
// runtime/rt_engine.cuh: K/V runs as start messages, then the push queue: partial sums and relay forwards).
// Workers claim queue entries: kTask (row r = (segment, kv head, block): the OFFICIAL InfLLM-v2 column routine, then the
// block's completion count), kKSum (owner sum of an owned block -> bf16 dK/dV rows), kKAgg (gateway: node landings into
// the own accumulator, then push to the owner), kStop.
#include <cstdint>
#include "../runtime/rt_params.h"
namespace rt { struct SchedB0; }

namespace infllm_bwd {

enum : int { kTask = 0, kKSum = 1, kKAgg = 2, kStop = 3, kNop = 4 };
static constexpr int kBlockRows = 64, kRowFloats = 512;                 // accumulator / landing slot: 64 x 512 fp32
static constexpr int64_t kSlotBytes = (int64_t)kBlockRows * kRowFloats * 4;   // 128 KB

struct Params {
    char *base;                       // the symmetric region (this PE's view)
    uint64_t *sig; uint64_t epoch;
    int64_t kv_off, acc_off, land_off;
    int nb;                           // global 64-token blocks (= KV slots)
    // tasks (scheduler rows): (segment, kv head, block)
    int n_tasks;
    const int32_t *task_b, *task_h, *task_n;
    // block completion: tasks per block here; what the completing block does with the block's partial sum
    const int32_t *ks_need;           // 0 = no task here
    const int32_t *ks_agg_need;       // > 0: this PE is the block's node gateway -> reduction unit n_km + ka_unit
    const int32_t *ka_unit;
    const int32_t *ks_push_pe;        // >= 0: push the partial (trigger = block) to this PE (owner or gateway)
    const int32_t *ks_push_dst;       // else: the owner-sum unit here
    // reduction units: [0, n_km) owner sums, [n_km, n_km + n_ka) gateway aggregations
    int n_km, n_ka;
    const int32_t *km_blk, *km_out, *km_rows, *km_ptr, *km_land;   // block, first output row, valid rows, landings
    const int32_t *ka_blk, *ka_ptr, *ka_land;
    // outputs (plain device memory): [own_rows, 2, 128] bf16
    void *dk_out, *dv_out;
    // the shared runtime
    rt::EngineParams eng;
    rt::SchedParams sch;
    rt::SchedB0 *sch_b0;
    // per-call scratch (zeroed by the host)
    int *sq, *sq_tail, *tile_counter, *ks_done, *n_enq;
    int n_workers;
    // DEX_TRACE: u64 counters (ns) summed over workers: 0 queue wait, 1 compute, 2 completion, 3 sums, 4 tasks,
    // 5 first worker start, 6 last worker end; else nullptr
    unsigned long long *trace;
};

}  // namespace infllm_bwd
