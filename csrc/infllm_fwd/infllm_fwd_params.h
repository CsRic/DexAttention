#pragma once
// Plain-data parameters of the InfLLM-v2 forward megakernel.
//
// Symmetric region (shared with the backward's layout prefix): sig u64 [NSIG] | kv bf16 [nb * 64 rows, 512] (K | V,
// global layout). Signal ids: run arrivals, the forward's static numbering (run * NG + group).
//
// Kernel (128 threads per block): block 0 = shared scheduler (warp 0; a row is ready when its counted remote runs have
// all landed: grp_need / sig_grp, the tables built on the GPU by the driver) and the communication engine (warp 3:
// inter-node runs); warps 1-2 idle. Workers: warps 0-1 each take queue entries (a task = one ready row = (KV head, 8
// consecutive tokens)) and run the OFFICIAL row routine per token on their own shared-memory slice; warps 2-3 idle.
// Outputs O / LSE are written directly by the routine (Q stays home: nothing to reduce across GPUs; with the local /
// remote split the two fp32 partials of a token are merged on this GPU).
#include <cstdint>
#include "../runtime/rt_params.h"
namespace rt { struct SchedB0; }

namespace infllm_fwd {

enum : int { kTask = 0, kStop = 3, kNop = 4 };
static constexpr int kTokensPerRow = 8;   // tokens per scheduler row (= one group)

struct Params {
    char *base; uint64_t *sig; uint64_t epoch;
    int64_t kv_off;
    int nb, L;                        // global blocks; this PE's tokens (all segments)
    const int32_t *cu_q;              // [b + 1] segment starts in rows (= tokens * 16)
    int n_seg, n_rows, nrh;           // rows = parts x 2 heads x nrh (part-major); token space = parts x L (duplicated q)
    rt::EngineParams eng;
    rt::SchedParams sch;
    rt::SchedB0 *sch_b0;
    int *sq, *sq_tail, *tile_counter, *n_enq;   // per-call scratch (zeroed by the host)
    int n_workers;
    unsigned long long *trace;        // DEX_TRACE: 0 queue wait, 1 compute, 2 tokens, 3 first start, 4 last end
};

}  // namespace infllm_fwd
