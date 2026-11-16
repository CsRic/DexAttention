// Host side of the InfLLM-v2 backward megakernel (tables from
// dex/attn/infllm.py). The symmetric region is the backward region of dist_bwd_host.cpp
// (bwd_sym_alloc / bwd_pe_info), addressed here by its base pointer and byte offsets.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <cstdint>
#include "infllm_bwd_params.h"
#include "../runtime/rt_sched_host.h"
#ifdef INFLLM_BF16
#define INFLLM_DTYPE_HOST torch::kBFloat16
#else
#define INFLLM_DTYPE_HOST torch::kHalf               // fp16 build: InfLLM-v2 routines in FP16, not BF16 (as every system)
#endif
#include "../runtime/rt_host.h"

namespace infllm_bwd {
struct FArgs {                                   // (as infllm_bwd.cu)
    void *q, *k, *v, *o, *dout, *dq;
    int64_t q_row_stride, k_row_stride, o_row_stride, do_row_stride, dq_row_stride;
    float *lse, *dpsum, *dq_accum;
    float *dk_accum, *dv_accum; int64_t acc_row_stride;
    const int *cu_q, *cu_k, *seqused_k; int b;
    const uint64_t *blockmask; int num_blocks_m, num_blocks_n;
    int total_q, max_seqlen_q, max_seqlen_k; float scale;
};
}  // namespace infllm_bwd
int infllm_bwd_smem_bytes();
int infllm_bwd_grid();
void infllm_bwd_launch(const infllm_bwd::FArgs &a, const infllm_bwd::Params &p, int grid, cudaStream_t stream);
void infllm_bwd_prep_launch(const infllm_bwd::FArgs &a, cudaStream_t stream);
void infllm_bwd_convert_launch(const infllm_bwd::FArgs &a, cudaStream_t stream);
void infllm_col_bitmap_launch(const int *sel, const int *pos, int L, int K, int nbn, int upr, unsigned long long *bm, cudaStream_t stream);

namespace {
uint64_t g_epoch = 0;

// q / o / dout / dq: [total_q, 2, 128] bf16 (group rows); lse [2, total_q] f32; dpsum [2, total_q + 128 b] f32;
// dq_accum [total_q + 128 b, 2, 128] f32; cu_q [b + 1] int32 (rows); cu_k [b + 1] int32 (zeros); seqused_k [b] int32
void fill_q(infllm_bwd::FArgs &a, const at::Tensor &q, const at::Tensor &o, const at::Tensor &dout, const at::Tensor &dq,
            const at::Tensor &lse, const at::Tensor &dpsum, const at::Tensor &dq_accum, const at::Tensor &cu_q,
            int64_t max_seqlen_q, double scale) {
    TORCH_CHECK(q.dtype() == INFLLM_DTYPE_HOST && q.dim() == 3 && q.size(1) == 2 && q.size(2) == 128 && q.is_contiguous());
    TORCH_CHECK(o.sizes() == q.sizes() && dout.sizes() == q.sizes() && dq.sizes() == q.sizes());
    TORCH_CHECK(o.is_contiguous() && dout.is_contiguous() && dq.is_contiguous());
    TORCH_CHECK(cu_q.dtype() == torch::kInt32 && lse.dtype() == torch::kFloat && dpsum.dtype() == torch::kFloat &&
                dq_accum.dtype() == torch::kFloat && lse.is_contiguous() && dpsum.is_contiguous() && dq_accum.is_contiguous());
    const int b = (int)cu_q.numel() - 1;
    TORCH_CHECK(lse.size(0) == 2 && lse.size(1) == q.size(0) && dpsum.size(0) == 2 && dpsum.size(1) == q.size(0) + 128 * b &&
                dq_accum.size(0) == q.size(0) + 128 * b && dq_accum.size(1) == 2 && dq_accum.size(2) == 128);
    a.q = q.data_ptr(); a.o = o.data_ptr(); a.dout = dout.data_ptr(); a.dq = dq.data_ptr();
    a.q_row_stride = a.o_row_stride = a.do_row_stride = a.dq_row_stride = 256;
    a.lse = lse.data_ptr<float>(); a.dpsum = dpsum.data_ptr<float>(); a.dq_accum = dq_accum.data_ptr<float>();
    a.cu_q = cu_q.data_ptr<int>(); a.b = b;
    a.total_q = (int)q.size(0); a.max_seqlen_q = (int)max_seqlen_q; a.scale = (float)scale;
}
}  // namespace

// Delta = rowsum(dO * O) into dpsum (official layout), dq_accum zeroed: the official preprocess kernel.
void infllm_bwd_prep(at::Tensor q, at::Tensor o, at::Tensor dout, at::Tensor dq, at::Tensor lse, at::Tensor dpsum,
                     at::Tensor dq_accum, at::Tensor cu_q, int64_t max_seqlen_q, double scale) {
    infllm_bwd::FArgs a{};
    fill_q(a, q, o, dout, dq, lse, dpsum, dq_accum, cu_q, max_seqlen_q, scale);
    infllm_bwd_prep_launch(a, at::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// dq (bf16, group rows) = scale * dq_accum, un-permuting the atomics' block layout: the official convert kernel.
void infllm_bwd_convert_dq(at::Tensor q, at::Tensor o, at::Tensor dout, at::Tensor dq, at::Tensor lse, at::Tensor dpsum,
                           at::Tensor dq_accum, at::Tensor cu_q, int64_t max_seqlen_q, double scale) {
    infllm_bwd::FArgs a{};
    fill_q(a, q, o, dout, dq, lse, dpsum, dq_accum, cu_q, max_seqlen_q, scale);
    infllm_bwd_convert_launch(a, at::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// sel [2, L, K] int32, pos [L] int32 (global token positions) -> the official column bitmap int64 [2, nbn, ceil(L / 64)]
at::Tensor infllm_col_bitmap(at::Tensor sel, at::Tensor pos, int64_t nbn) {
    TORCH_CHECK(sel.dtype() == torch::kInt32 && sel.dim() == 3 && sel.size(0) == 2 && sel.is_contiguous() &&
                pos.dtype() == torch::kInt32 && pos.numel() == sel.size(1) && pos.is_contiguous());
    const int L = (int)sel.size(1), K = (int)sel.size(2), upr = (L + 63) / 64;
    auto bm = torch::zeros({2, nbn, upr}, sel.options().dtype(torch::kInt64));
    infllm_col_bitmap_launch(sel.data_ptr<int>(), pos.data_ptr<int>(), L, K, (int)nbn, upr,
                             reinterpret_cast<unsigned long long *>(bm.data_ptr<int64_t>()), at::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return bm;
}

int64_t infllm_bwd_smem() { return infllm_bwd_smem_bytes(); }
int64_t infllm_bwd_grid_size() { return infllm_bwd_grid(); }

// One backward call. base: the symmetric region (bwd_sym_alloc's view, data_ptr); offs: byte offsets [sig, kv, acc, land];
// dims: [nb, n_tasks, n_km, n_ka, n_remote, n_start, n_pq, NSIG, NL, grid]; tabs: int32 tables in the order of
// dex/attn/infllm.py BWD_TABLE_ORDER; scratch: int32, zeroed here; scr_offs: dex/attn/infllm.py bwd_scratch_layout.
// Inputs (plain device memory, official layouts): q / o / dout / dq [total_q, 2, 128] bf16, lse, dpsum, dq_accum, cu_q,
// cu_k, seqused_k, bm (column bitmap). The K / V rows this PE owns must already be staged into the region (kv_off).
// Outputs: dk_out / dv_out [own_rows, 2, 128] bf16 (owner sums). Returns the epoch.
int64_t infllm_bwd_run(int64_t base, std::vector<int64_t> offs, std::vector<int64_t> dims, std::vector<at::Tensor> tabs,
                       at::Tensor scratch, std::vector<int64_t> scr_offs, at::Tensor pe_delta, at::Tensor pe_inter,
                       at::Tensor q, at::Tensor o, at::Tensor dout, at::Tensor dq, at::Tensor lse, at::Tensor dpsum,
                       at::Tensor dq_accum, at::Tensor cu_q, at::Tensor cu_k, at::Tensor seqused_k, at::Tensor bm,
                       int64_t max_seqlen_q, int64_t max_seqlen_k, double scale, at::Tensor dk_out, at::Tensor dv_out,
                       at::Tensor ce_tab) {
    TORCH_CHECK(base && offs.size() == 4 && dims.size() == 10 && scr_offs.size() == 13);
    TORCH_CHECK(cu_k.dtype() == torch::kInt32 && seqused_k.dtype() == torch::kInt32 && bm.dtype() == torch::kInt64 && bm.is_contiguous());
    TORCH_CHECK(dk_out.dtype() == INFLLM_DTYPE_HOST && dk_out.is_contiguous() && dv_out.is_contiguous() && dk_out.size(1) == 2 && dk_out.size(2) == 128);
    auto stream = at::cuda::getCurrentCUDAStream();
    infllm_bwd::Params p{};
    p.base = reinterpret_cast<char *>(base);
    p.sig = reinterpret_cast<uint64_t *>(p.base + offs[0]); p.kv_off = offs[1]; p.acc_off = offs[2]; p.land_off = offs[3];
    p.nb = (int)dims[0]; p.n_tasks = (int)dims[1]; p.n_km = (int)dims[2]; p.n_ka = (int)dims[3];
    const int n_remote = (int)dims[4], NSIG = (int)dims[7], NL = (int)dims[8];
    p.eng.n_start = (int)dims[5]; p.eng.n_pq = (int)dims[6];
    int grid = (int)dims[9];
    if (grid <= 0) grid = infllm_bwd_grid();
    p.n_workers = grid - 1;
    int k = 0;
    auto T = [&]() -> const int32_t * { TORCH_CHECK(k < (int)tabs.size()); TORCH_CHECK(tabs[k].dtype() == torch::kInt32 && tabs[k].is_cuda()); return tabs[k++].data_ptr<int32_t>(); };
    p.task_b = T(); p.task_h = T(); p.task_n = T();
    p.ks_need = T(); p.ks_agg_need = T(); p.ka_unit = T(); p.ks_push_pe = T(); p.ks_push_dst = T();
    p.km_blk = T(); p.km_out = T(); p.km_rows = T(); p.km_ptr = T(); p.km_land = T();
    p.ka_blk = T(); p.ka_ptr = T(); p.ka_land = T();
    p.eng.msg = reinterpret_cast<const rt::Msg *>(T()); p.eng.trig_ptr = T(); p.eng.trig_lane = T();
    const int64_t n_trig = tabs[k - 1].numel();
    rt::SchedParams &S = p.sch;
    S.remote_sig = T(); S.row_grp_ptr = T(); S.grp_beg = T(); S.grp_end = T(); S.grp_dep = T(); S.row_qdep = T();
    S.row_order = T(); S.sig_grp_ptr = T(); S.sig_grp = T(); S.sig_row_ptr = T(); S.sig_row = T();
    S.red_need = T(); S.red_kind = T(); S.red_pay = T(); S.red_n = T(); S.sig_red_ptr = T(); S.sig_red = T();
    TORCH_CHECK(k == (int)tabs.size(), "table count mismatch: ", k, " vs ", tabs.size());
    p.dk_out = dk_out.data_ptr(); p.dv_out = dv_out.data_ptr();
    // the engine (runtime/rt_engine.cuh): every send of this PE
    p.eng.base = p.base; p.eng.sig = p.sig; p.eng.pe_delta = pe_delta.data_ptr<int64_t>(); p.eng.pe_inter = pe_inter.data_ptr<int32_t>();
    p.eng.credit = nullptr; p.eng.call_id = 0;              // cross-call order: the stream barrier below
    p.eng.trig_first = 2; p.eng.in_trig_first = p.nb;     // input forwards fire before partial outputs
    p.eng.trig_flag = nullptr; p.eng.fire_log = nullptr;
    // intra-node messages by copy engine (rt_host.h): trigger flags (epoch-valued, zeroed once: epochs only grow)
    const int n_ce = ce_tab.numel() ? (int)ce_tab.size(0) : 0;
    TORCH_CHECK(ce_tab.dtype() == torch::kInt64 && !ce_tab.is_cuda() && (n_ce == 0 || ce_tab.size(1) == 6));
    static rt::CeExec ce;
    static unsigned long long *flags = nullptr; static int64_t fcap = 0;
    if (n_trig > fcap) { if (flags) cudaFree(flags); cudaMalloc(&flags, n_trig * 8); cudaMemset(flags, 0, n_trig * 8); fcap = n_trig; }
    p.eng.trig_flag = flags;
    static std::vector<int64_t> delta_h; static int64_t delta_n = 0;
    if ((int64_t)pe_delta.numel() != delta_n) { auto h = pe_delta.cpu(); delta_h.assign(h.data_ptr<int64_t>(), h.data_ptr<int64_t>() + h.numel()); delta_n = h.numel(); }
    // per-call scratch (dex/attn/infllm.py bwd_scratch_layout)
    int *s = scratch.data_ptr<int>();
    p.sq = s + scr_offs[0]; p.sq_tail = s + scr_offs[1]; p.tile_counter = s + scr_offs[2]; p.ks_done = s + scr_offs[3];
    p.n_enq = s + scr_offs[4]; p.eng.pq = s + scr_offs[5]; p.eng.pq_tail = s + scr_offs[6];
    S.red_cnt = s + scr_offs[7]; S.task_desc = s + scr_offs[8]; S.row_ntask = s + scr_offs[9]; S.grp_cnt = s + scr_offs[10];
    S.rows_done = s + scr_offs[11]; p.sch_b0 = reinterpret_cast<rt::SchedB0 *>(s + scr_offs[12]);
    // the shared scheduler (runtime/rt_sched.cuh): rows = tasks, one group each; units = owner sums | gateway aggregations
    S.sig = p.sig; S.n_remote = n_remote; S.remote_cnt = nullptr; S.arrived = nullptr;
    S.n_rows = S.n_groups = p.n_tasks; S.grp_need = nullptr; S.row_pool = nullptr;
    S.n_red = p.n_km + p.n_ka; S.kind_red = 0; S.red_n_all = 0; S.n_enq = p.n_enq;
    S.sq = p.sq; S.sq_tail = p.sq_tail; S.tile_counter = p.tile_counter; S.n_static = 0;
    S.sub = 1; S.gang = 1; S.kind_task = infllm_bwd::kTask; S.kind_stop = infllm_bwd::kStop; S.kind_nop = infllm_bwd::kNop;
    S.n_workers = p.n_workers;
    S.sched_low = 2 * p.n_workers;
    S.sched_crit = S.sched_low; S.sched_batch = 64; S.q_work = nullptr;
    S.fwd_trig = p.nb;                                       // relay forwards: trigger nb + j on the arrival of polled signal j
    S.n_units = 0; S.grp_unit = nullptr; S.unit_dep = nullptr; S.grp_work = nullptr;
    S.tl_sig = S.tl_row = S.tl_ev = S.trace = nullptr; S.q_all_in = nullptr; S.q_lim = 0;
    S.grp_max = 0; S.grp_avail_bits = nullptr; S.row_avail = nullptr;
    S.b0_bytes = rt::sched_b0_bytes(S.n_groups, S.n_rows, S.n_remote, S.sched_batch);
    static void *b0g = nullptr; static int64_t b0cap = 0;
    if (S.b0_bytes <= infllm_bwd_smem_bytes() - 512) S.b0_state = nullptr;   // block 0's shared memory
    else {
        if (S.b0_bytes > b0cap) { if (b0g) cudaFree(b0g); cudaMalloc(&b0g, S.b0_bytes); b0cap = S.b0_bytes; }
        S.b0_state = (char *)b0g;
    }
    // the official compute's parameters
    infllm_bwd::FArgs a{};
    fill_q(a, q, o, dout, dq, lse, dpsum, dq_accum, cu_q, max_seqlen_q, scale);
    TORCH_CHECK(cu_k.numel() == cu_q.numel() && seqused_k.numel() == a.b);
    a.k = p.base + p.kv_off; a.v = p.base + p.kv_off + 256 * 2;      // [rows, 512] bf16: K | V
    a.k_row_stride = 512;
    a.dk_accum = reinterpret_cast<float *>(p.base + p.acc_off); a.dv_accum = a.dk_accum + 256; a.acc_row_stride = 512;
    a.cu_k = cu_k.data_ptr<int>(); a.seqused_k = seqused_k.data_ptr<int>();
    a.blockmask = reinterpret_cast<const uint64_t *>(bm.data_ptr<int64_t>());
    a.num_blocks_m = a.total_q / 16; a.num_blocks_n = (int)bm.size(1); a.max_seqlen_k = (int)max_seqlen_k;
    TORCH_CHECK(bm.size(0) == 2 && bm.size(2) == (a.num_blocks_m + 63) / 64 && a.num_blocks_n >= (a.max_seqlen_k + 63) / 64);
    static at::Tensor trace, tl_sig, tl_row, tl_ev;
    if (getenv("DEX_TRACE") && atoi(getenv("DEX_TRACE"))) {
        if (!trace.defined()) trace = torch::zeros({8}, torch::dtype(torch::kInt64).device(torch::kCUDA));
        trace.zero_(); trace[5] = (int64_t)0x7fffffffffffffffLL;
        p.trace = (unsigned long long *)trace.data_ptr();
        const auto o64 = torch::dtype(torch::kInt64).device(torch::kCUDA);   // the scheduler's timeline (rt_sched.cuh)
        tl_sig = torch::zeros({(int64_t)NSIG + 1}, o64); tl_row = torch::zeros({2 * (int64_t)S.n_rows + 2}, o64); tl_ev = torch::zeros({16}, o64);
        S.tl_sig = (unsigned long long *)tl_sig.data_ptr(); S.tl_row = (unsigned long long *)tl_row.data_ptr();
        S.tl_ev = (unsigned long long *)tl_ev.data_ptr();
    }
    cudaMemsetAsync(scratch.data_ptr(), 0, scratch.numel() * 4, stream);
    cudaMemsetAsync(p.sig, 0, (size_t)NSIG * 8, stream);                 // this call's signal words (epoch-valued)
    cudaMemsetAsync(p.base + p.acc_off, 0, (size_t)p.nb * infllm_bwd::kSlotBytes, stream);   // own accumulators
    p.epoch = ++g_epoch; p.eng.epoch = p.epoch; S.epoch = p.epoch;
    (void)NL;
    // every PE has finished the previous call (no peer still reads a slot this call's pushes overwrite) and has staged
    // this call's inputs before anyone pushes them
    nvshmemx_barrier_all_on_stream(stream);
    const int P = nvshmem_n_pes();
    if (n_ce) { ce.ensure(P); ce.begin(stream, p.epoch); }
    infllm_bwd_launch(a, p, grid, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (n_ce) {
        std::vector<char *> pb(P, nullptr); std::vector<uint64_t *> ps(P, nullptr);
        for (int q = 0; q < P; ++q) { pb[q] = p.base + delta_h[q]; ps[q] = reinterpret_cast<uint64_t *>(pb[q] + offs[0]); }
        ce.enqueue(stream, reinterpret_cast<const rt::CeMsg *>(ce_tab.data_ptr<int64_t>()), n_ce, p.base, pb, ps, p.sig,
                   p.eng.trig_flag, p.epoch, nullptr, 0, nullptr);
    }
    if (p.trace) {
        auto t = trace.cpu(); auto *v = t.data_ptr<int64_t>(); const double nw = p.n_workers;
        printf("[pe %d infllm bwd trace] span %.2f ms | per worker: queue wait %.2f, compute %.2f, completion %.2f, sums %.2f ms | tasks %lld\n",
               nvshmem_my_pe(), (v[6] - v[5]) * 1e-6, v[0] * 1e-6 / nw, v[1] * 1e-6 / nw, v[2] * 1e-6 / nw, v[3] * 1e-6 / nw, (long long)v[4]);
        auto ev = tl_ev.cpu(); const int64_t *e = ev.data_ptr<int64_t>(); const double t0 = (double)e[1];
        auto pct = [&](std::vector<double> &x, double q) { return x.empty() ? 0.0 : x[(size_t)std::min<double>(x.size() - 1, q * x.size())]; };
        std::vector<double> in, ld, dis;
        auto ts = tl_sig.cpu(); const int64_t *a_ = ts.data_ptr<int64_t>();
        std::vector<int> rs(n_remote);
        if (n_remote) cudaMemcpy(rs.data(), S.remote_sig, n_remote * 4, cudaMemcpyDeviceToHost);
        for (int id : rs) if (a_[id]) (id < NSIG - NL ? in : ld).push_back((a_[id] - t0) * 1e-6);
        auto tr = tl_row.cpu(); const int64_t *r = tr.data_ptr<int64_t>();
        for (int i = 0; i < S.n_rows; ++i) if (r[2 * i]) dis.push_back((r[2 * i] - t0) * 1e-6);
        std::sort(in.begin(), in.end()); std::sort(ld.begin(), ld.end()); std::sort(dis.begin(), dis.end());
        printf("[pe %d infllm bwd timeline] from scheduler start (ms): inputs %zu p50 %.1f p90 %.1f last %.1f | landings %zu p50 %.1f p90 %.1f last %.1f | "
               "tasks dispatched p50 %.1f p90 %.1f last %.1f | STOP %.1f\n", nvshmem_my_pe(), in.size(), pct(in, 0.5), pct(in, 0.9),
               in.empty() ? 0.0 : in.back(), ld.size(), pct(ld, 0.5), pct(ld, 0.9), ld.empty() ? 0.0 : ld.back(), pct(dis, 0.5), pct(dis, 0.9),
               dis.empty() ? 0.0 : dis.back(), e[3] ? (e[3] - t0) * 1e-6 : -1.0);
    }
    return (int64_t)p.epoch;
}

void register_infllm_bwd(pybind11::module &m) {
    m.def("infllm_bwd_prep", &infllm_bwd_prep, "official preprocess: dpsum = rowsum(dO * O), dq_accum = 0");
    m.def("infllm_bwd_convert_dq", &infllm_bwd_convert_dq, "official postprocess: dq = scale * dq_accum -> bf16");
    m.def("infllm_col_bitmap", &infllm_col_bitmap, "column bitmap of the official backward from the top-k selection");
    m.def("infllm_bwd_run", &infllm_bwd_run, "one InfLLM-v2 backward megakernel call");
    m.def("infllm_bwd_smem", &infllm_bwd_smem); m.def("infllm_bwd_grid", &infllm_bwd_grid_size);
}
