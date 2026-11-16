#pragma once
// Plain-data parameters of the 2-D InfLLM-v2 backward megakernel: the
// query-stationary backward of csrc/infllm_bwd (infllm_bwd_params.h, as the first members) plus FOREIGN
// tasks (task_h == 2): column tasks (unit u = task_b, block n) of moved units, run by the same official column routine
// with the unit's own parameter block (fpu[u]); their dK / dV join the block's accumulator slot (existing route to the
// KV owner); when a unit's last task completes, trigger trig_unit0 + u returns the unit's FP32 dQ accumulator.
#include <cstdint>
#include "../runtime/rt_params.h"
namespace rt { struct SchedB0; }

namespace infllm2d_bwd {

enum : int { kTask = 0, kKSum = 1, kKAgg = 2, kStop = 3, kNop = 4 };
static constexpr int kBlockRows = 64, kRowFloats = 512;
static constexpr int64_t kSlotBytes = (int64_t)kBlockRows * kRowFloats * 4;   // 128 KB
static constexpr int kForeignH = 2;                                         // task_h of a foreign task

struct Params {
    char *base;
    uint64_t *sig; uint64_t epoch;
    int64_t kv_off, acc_off, land_off;
    int nb;
    int n_tasks;
    const int32_t *task_b, *task_h, *task_n;
    const int32_t *ks_need;
    const int32_t *ks_agg_need;
    const int32_t *ka_unit;
    const int32_t *ks_push_pe;
    const int32_t *ks_push_dst;
    int n_km, n_ka;
    const int32_t *km_blk, *km_out, *km_rows, *km_ptr, *km_land;
    const int32_t *ka_blk, *ka_ptr, *ka_land;
    void *dk_out, *dv_out;
    rt::EngineParams eng;
    rt::SchedParams sch;
    rt::SchedB0 *sch_b0;
    int *sq, *sq_tail, *tile_counter, *ks_done, *n_enq;
    int n_workers;
    unsigned long long *trace;
    // ---- 2-D extension
    const void *fpu;                  // infllm_bwd::FParams [n_units], one per incoming unit
    int *unit_done;                   // [n_units] per-call scratch: tasks completed
    const int32_t *unit_need;         // [n_units] tasks of the unit
    int trig_unit0;                   // trigger of unit u = trig_unit0 + u
};

// plain launch arguments of the own tasks (as infllm_bwd.cu FArgs)
struct FArgs {
    void *q, *k, *v, *o, *dout, *dq;
    int64_t q_row_stride, k_row_stride, o_row_stride, do_row_stride, dq_row_stride;
    float *lse, *dpsum, *dq_accum;
    float *dk_accum, *dv_accum; int64_t acc_row_stride;
    const int *cu_q, *cu_k, *seqused_k; int b;
    const uint64_t *blockmask; int num_blocks_m, num_blocks_n;
    int total_q, max_seqlen_q, max_seqlen_k; float scale;
};

// one incoming unit: device pointers + sizes (q / dout bf16 [len * 16, 128]; lse f32 [len * 16]; dpsum f32 [len * 16 + 128];
// dq_accum f32 [len * 16 + 128, 128]; dk / dv accumulators at the group's column; bm u64 [nbn, ceil(len / 64)])
struct UnitDesc {
    void *q, *dout, *lse, *dpsum, *dq_accum, *k, *v, *dk_acc, *dv_acc, *bm;
    const int *cu_q, *cu_k, *seqused;
    int len, nbn, max_seqlen_k; float scale;
};

}  // namespace infllm2d_bwd
