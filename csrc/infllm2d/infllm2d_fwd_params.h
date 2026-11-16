#pragma once
// Plain-data parameters of the 2-D InfLLM-v2 forward megakernel: the
// query-stationary forward of csrc/infllm_fwd (infllm_fwd_params.h, as the first members) plus
// FOREIGN rows: rows of moved units (another rank's query chunk, one KV-head group, the entries the plan put on this
// rank). A foreign row runs the same official row routine with the unit's own parameter block (fpu[u], device memory);
// when a unit's last row completes, the worker fires trigger trig_unit0 + u (its FP32 partial O | LSE goes back to the
// query owner). Rows [0, n_rows_own) are the existing own rows, decoded exactly as in csrc/infllm_fwd.
#include <cstdint>
#include "../runtime/rt_params.h"
namespace rt { struct SchedB0; }

namespace infllm2d_fwd {

enum : int { kTask = 0, kStop = 3, kNop = 4 };
static constexpr int kTokensPerRow = 8;   // tokens per scheduler row (= one group)

struct Params {
    char *base; uint64_t *sig; uint64_t epoch;
    int64_t kv_off;
    int nb, L;                        // global blocks; this PE's tokens (all segments)
    const int32_t *cu_q;              // [b + 1] segment starts in rows (= tokens * 16)
    int n_seg, n_rows, nrh;           // rows = own rows (parts x 2 heads x nrh, part-major) + foreign rows
    rt::EngineParams eng;
    rt::SchedParams sch;
    rt::SchedB0 *sch_b0;
    int *sq, *sq_tail, *tile_counter, *n_enq;   // per-call scratch (zeroed by the host)
    int n_workers;
    unsigned long long *trace;        // DEX_TRACE: 0 queue wait, 1 compute, 2 tokens, 3 first start, 4 last end
    // ---- 2-D extension (n_rows_own == n_rows: none)
    int n_rows_own;                   // rows [0, n_rows_own) own (existing decode), [n_rows_own, n_rows) foreign
    const void *fpu;                  // FParams [n_units] (infllm2d_fwd.cu), one per incoming unit
    const int32_t *frow;              // [n_rows - n_rows_own][3]: unit, first token, end token (within the unit)
    int *unit_done;                   // [n_units] per-call scratch: rows completed
    const int32_t *unit_need;         // [n_units] rows of the unit
    int trig_unit0;                   // trigger of unit u = trig_unit0 + u
};

// plain launch arguments of the own rows (as infllm_fwd.cu FArgs)
struct FArgs {
    void *q, *k, *v, *o, *o32;
    int64_t q_row_stride, k_row_stride, o_row_stride;
    float *lse;
    const int *cu_q, *cu_k, *seqused_k; int b;
    const uint64_t *blockmask; int num_blocks_m, num_blocks_n;
    int total_q, max_seqlen_q, max_seqlen_k; float scale;
};

// one incoming unit (host -> infllm2d_fwd_build_units): device pointers + sizes
struct UnitDesc {
    void *q, *bm, *o32, *lse, *k, *v;  // q bf16 [len * 16, 128]; bm u64 [len, ceil(nbn / 64)]; o32 f32 [len * 16, 128];
                                       // lse f32 [len * 16]; k / v: the region's K | V rows at the group's column
    const int *cu_q, *cu_k, *seqused;  // {0, len * 16}, {0, 0}, {chunk end}
    int len, nbn, max_seqlen_k; float scale;
};

}  // namespace infllm2d_fwd
