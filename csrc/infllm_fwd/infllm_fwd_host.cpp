// Host side of the InfLLM-v2 forward megakernel (tables from
// dex/attn/infllm.py). Region: the backward's symmetric region (bwd_sym_alloc), prefix
// [sig | kv].
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include "infllm_fwd_params.h"
#include "../runtime/rt_sched_host.h"
#ifdef INFLLM_BF16
#define INFLLM_DTYPE_HOST torch::kBFloat16
#else
#define INFLLM_DTYPE_HOST torch::kHalf               // fp16 build: InfLLM-v2 routines in FP16, not BF16 (as every system)
#endif
#include "../runtime/rt_host.h"

namespace infllm_fwd {
struct FArgs {                                   // (as infllm_fwd.cu)
    void *q, *k, *v, *o, *o32;
    int64_t q_row_stride, k_row_stride, o_row_stride;
    float *lse;
    const int *cu_q, *cu_k, *seqused_k; int b;
    const uint64_t *blockmask; int num_blocks_m, num_blocks_n;
    int total_q, max_seqlen_q, max_seqlen_k; float scale;
};
}  // namespace infllm_fwd
int infllm_fwd_smem_bytes();
int infllm_fwd_grid();
void infllm_fwd_launch(const infllm_fwd::FArgs &a, const infllm_fwd::Params &p, int grid, cudaStream_t stream);
void infllm_split_bitmap_launch(const int *sel, const uint8_t *is_local, int L, int K, int upr, unsigned long long *bm, cudaStream_t stream);

namespace {
uint64_t g_epoch = 0;
rt::CeExec ce;
struct LateCtx { char *base; uint64_t *sig; unsigned long long *trig_flag; uint64_t epoch; int sig_off; int P; bool late; };
LateCtx g_late{};
std::vector<int64_t> delta_h; int64_t delta_n = 0;
cudaStream_t g_side = nullptr;

void ce_enqueue(cudaStream_t stream, const at::Tensor &ce_tab) {
    const int n_ce = ce_tab.numel() ? (int)ce_tab.size(0) : 0;
    if (!n_ce) return;
    std::vector<char *> pb(g_late.P, nullptr); std::vector<uint64_t *> ps(g_late.P, nullptr);
    for (int q_ = 0; q_ < g_late.P; ++q_) { pb[q_] = g_late.base + delta_h[q_]; ps[q_] = reinterpret_cast<uint64_t *>(pb[q_] + g_late.sig_off); }
    ce.enqueue(stream, reinterpret_cast<const rt::CeMsg *>(ce_tab.data_ptr<int64_t>()), n_ce, g_late.base, pb, ps, g_late.sig,
               g_late.trig_flag, g_late.epoch, nullptr, 0, nullptr);
}
}  // namespace

struct ReportCtx { bool on; int n_remote, n_rows, NSIG, nl; const int32_t *remote_sig; };
ReportCtx g_rep{};
at::Tensor g_trace, g_tl_sig, g_tl_row, g_tl_ev;
inline int NL_unused() { return 0; }

// DEX_TRACE: worker counters + the scheduler's timeline of the LAST call (synchronises on the kernel: in late mode
// call it only after infllm_fwd_late)
void infllm_fwd_report() {
    if (!g_rep.on) return;
    auto t = g_trace.cpu(); auto *v = t.data_ptr<int64_t>();
    const double nw = 2.0 * (double)std::max<int64_t>(1, v[7]);           // compute warps = 2 x workers (trace[7])
    printf("[pe %d infllm fwd trace] span %.2f ms | per warp: queue wait %.2f, compute %.2f ms | tokens %lld\n",
           nvshmem_my_pe(), (v[4] - v[3]) * 1e-6, v[0] * 1e-6 / nw, v[1] * 1e-6 / nw, (long long)v[2]);
    auto ev = g_tl_ev.cpu(); const int64_t *e = ev.data_ptr<int64_t>(); const double t0 = (double)e[1];
    auto pct = [&](std::vector<double> &x, double q_) { return x.empty() ? 0.0 : x[(size_t)std::min<double>(x.size() - 1, q_ * x.size())]; };
    std::vector<double> in, dis;
    auto ts = g_tl_sig.cpu(); const int64_t *a_ = ts.data_ptr<int64_t>();
    std::vector<int> rs(g_rep.n_remote);
    if (g_rep.n_remote) cudaMemcpy(rs.data(), g_rep.remote_sig, g_rep.n_remote * 4, cudaMemcpyDeviceToHost);
    for (int id : rs) if (a_[id]) in.push_back((a_[id] - t0) * 1e-6);
    auto tr = g_tl_row.cpu(); const int64_t *r = tr.data_ptr<int64_t>();
    for (int i = 0; i < g_rep.n_rows; ++i) if (r[2 * i]) dis.push_back((r[2 * i] - t0) * 1e-6);
    std::sort(in.begin(), in.end()); std::sort(dis.begin(), dis.end());
    printf("[pe %d infllm fwd timeline] from scheduler start (ms): inputs %zu p50 %.1f p90 %.1f last %.1f | rows first dispatched p50 %.1f p90 %.1f last %.1f | STOP %.1f\n",
           nvshmem_my_pe(), in.size(), pct(in, 0.5), pct(in, 0.9), in.empty() ? 0.0 : in.back(), pct(dis, 0.5), pct(dis, 0.9),
           dis.empty() ? 0.0 : dis.back(), e[3] ? (e[3] - t0) * 1e-6 : -1.0);
    printf("[pe %d infllm fwd sched] loop iterations %lld (%lld dispatching) | ms in arrivals %.1f, dispatch %.1f, queue writes %.1f | tasks fast %lld serial %lld | scans %lld\n",
           nvshmem_my_pe(), (long long)e[4], (long long)e[5], e[6] * 1e-6, e[7] * 1e-6, e[8] * 1e-6, (long long)e[9], (long long)e[10], (long long)e[11]);
}

// Late mode, after the host plan: the engine's start messages (msg_src CPU int64 [n, 4] -> the device msg buffer the
// kernel was launched with), n_start and the flag (a side stream: the main stream is held by the kernel), then the
// copy-engine messages.
void infllm_fwd_late(at::Tensor msg_buf, at::Tensor msg_src, at::Tensor late_n_start, at::Tensor late_flag, at::Tensor ce_tab) {
    TORCH_CHECK(g_late.late, "the running call was not launched in late mode");
    TORCH_CHECK(msg_src.dtype() == torch::kInt64 && !msg_src.is_cuda() && msg_buf.is_cuda() && msg_buf.numel() * 4 >= msg_src.numel() * 8);
    if (!g_side) cudaStreamCreateWithFlags(&g_side, cudaStreamNonBlocking);
    auto stream = at::cuda::getCurrentCUDAStream();
    const int n = (int)(msg_src.numel() / 4);
    static int *h_n = nullptr; static unsigned long long *h_flag = nullptr;
    if (!h_n) { cudaMallocHost(&h_n, 64); cudaMallocHost(&h_flag, 64); }
    cudaStreamSynchronize(g_side);                            // the previous call's pinned values were consumed
    *h_n = n; *h_flag = g_late.epoch;
    if (n) cudaMemcpyAsync(msg_buf.data_ptr(), msg_src.data_ptr(), (size_t)n * 32, cudaMemcpyHostToDevice, g_side);   // (pageable: staged now)
    cudaMemcpyAsync(late_n_start.data_ptr(), h_n, 4, cudaMemcpyHostToDevice, g_side);
    cudaMemcpyAsync(late_flag.data_ptr(), h_flag, 8, cudaMemcpyHostToDevice, g_side);
    ce_enqueue(stream, ce_tab);
}


int64_t infllm_fwd_smem() { return infllm_fwd_smem_bytes(); }
int64_t infllm_fwd_grid_size() { return infllm_fwd_grid(); }

// One forward call. base: the symmetric region; offs: byte offsets [sig, kv]; dims: [nb, L, K, n_seg, nrh,
// n_rows, n_remote, n_start, n_pq, NSIG, grid]; tabs: int32 tables (msg, then dex/attn/infllm.py FWD_TABLE_ORDER); scratch int32
// (fwd_scratch_layout); q [T, 2, 128] fp16 (bf16 with INFLLM_BF16; T = parts x L x 16, the parts' duplicated q), o fp16 / bf16 or o32 fp32 [T, 2, 128], lse [2, L * 16] f32, cu_q [n_seg + 1] rows, cu_k zeros, seqused_k [n_seg],
// bm: the official row bitmap int64 [2, parts x L, ceil(nbn / 64)]. The K / V rows this
// PE owns must already be staged into the region. ce_tab: CPU int64 [n, 6] (rt::CeMsg). Returns the epoch.
int64_t infllm_fwd_run(int64_t base, std::vector<int64_t> offs, std::vector<int64_t> dims, std::vector<at::Tensor> tabs,
                       at::Tensor scratch, std::vector<int64_t> scr_offs, at::Tensor pe_delta, at::Tensor pe_inter,
                       at::Tensor q, at::Tensor o, at::Tensor o32, at::Tensor lse, at::Tensor cu_q, at::Tensor cu_k, at::Tensor seqused_k,
                       at::Tensor bm, int64_t max_seqlen_q, int64_t max_seqlen_k, double scale, at::Tensor ce_tab,
                       at::Tensor late_n_start, at::Tensor late_flag) {
    TORCH_CHECK(base && offs.size() == 2 && dims.size() == 11 && scr_offs.size() == 13);
    TORCH_CHECK(q.dtype() == INFLLM_DTYPE_HOST && q.dim() == 3 && q.size(1) == 2 && q.size(2) == 128 && q.is_contiguous());
    TORCH_CHECK(o.is_contiguous() && lse.dtype() == torch::kFloat && lse.is_contiguous() && lse.size(0) == 2 && lse.size(1) == q.size(0));
    const bool f32 = o32.numel() > 0;
    TORCH_CHECK(f32 ? (o32.dtype() == torch::kFloat && o32.sizes() == q.sizes() && o32.is_contiguous()) : o.sizes() == q.sizes());
    TORCH_CHECK(cu_q.dtype() == torch::kInt32 && cu_k.dtype() == torch::kInt32 && seqused_k.dtype() == torch::kInt32 &&
                bm.dtype() == torch::kInt64 && bm.is_contiguous());
    auto stream = at::cuda::getCurrentCUDAStream();
    infllm_fwd::Params p{};
    p.base = reinterpret_cast<char *>(base);
    p.sig = reinterpret_cast<uint64_t *>(p.base + offs[0]); p.kv_off = offs[1];
    p.nb = (int)dims[0]; p.L = (int)dims[1]; p.n_seg = (int)dims[3];           // dims[2]: K (unused)
    p.nrh = (int)dims[4]; p.n_rows = (int)dims[5];
    const int n_remote = (int)dims[6], NSIG = (int)dims[9];
    p.eng.n_start = (int)dims[7]; p.eng.n_pq = (int)dims[8];
    int grid = (int)dims[10];
    if (grid <= 0) grid = infllm_fwd_grid();
    p.n_workers = grid - 1;
    TORCH_CHECK(cu_q.numel() == p.n_seg + 1 && seqused_k.numel() == p.n_seg);
    p.cu_q = cu_q.data_ptr<int>();
    int k = 0;
    auto T = [&]() -> const int32_t * { TORCH_CHECK(k < (int)tabs.size()); TORCH_CHECK(tabs[k].dtype() == torch::kInt32 && tabs[k].is_cuda()); return tabs[k++].data_ptr<int32_t>(); };
    p.eng.msg = reinterpret_cast<const rt::Msg *>(T()); p.eng.trig_ptr = T(); p.eng.trig_lane = T();
    const int64_t n_trig = tabs[k - 1].numel();
    rt::SchedParams &S = p.sch;
    S.remote_sig = T(); S.row_grp_ptr = T(); S.grp_beg = T(); S.grp_end = T(); S.grp_dep = T(); S.grp_need = T(); S.row_qdep = T();
    S.row_order = T(); S.sig_grp_ptr = T(); S.sig_grp = T(); S.sig_row_ptr = T(); S.sig_row = T();
    S.red_need = T(); S.red_kind = T(); S.red_pay = T(); S.red_n = T(); S.sig_red_ptr = T(); S.sig_red = T();
    TORCH_CHECK(k == (int)tabs.size(), "table count mismatch: ", k, " vs ", tabs.size());
    // the engine (inter-node runs) and the copy-engine executor (intra-node runs, relay forwards)
    p.eng.base = p.base; p.eng.sig = p.sig; p.eng.pe_delta = pe_delta.data_ptr<int64_t>(); p.eng.pe_inter = pe_inter.data_ptr<int32_t>();
    p.eng.credit = nullptr; p.eng.call_id = 0; p.eng.trig_first = 0; p.eng.in_trig_first = -1; p.eng.fire_log = nullptr;
    const bool late = late_flag.numel() > 0;              // the start messages are published after the launch (infllm_fwd_late)
    p.eng.late_flag = late ? reinterpret_cast<const unsigned long long *>(late_flag.data_ptr<int64_t>()) : nullptr;
    p.eng.late_n_start = late ? late_n_start.data_ptr<int>() : nullptr;
    const int n_ce = ce_tab.numel() ? (int)ce_tab.size(0) : 0;
    TORCH_CHECK(ce_tab.dtype() == torch::kInt64 && !ce_tab.is_cuda() && (n_ce == 0 || ce_tab.size(1) == 6));
    static unsigned long long *flags = nullptr; static int64_t fcap = 0;
    if (n_trig > fcap) { if (flags) cudaFree(flags); cudaMalloc(&flags, n_trig * 8); cudaMemset(flags, 0, n_trig * 8); fcap = n_trig; }
    p.eng.trig_flag = flags;
    if ((int64_t)pe_delta.numel() != delta_n) { auto h = pe_delta.cpu(); delta_h.assign(h.data_ptr<int64_t>(), h.data_ptr<int64_t>() + h.numel()); delta_n = h.numel(); }
    // per-call scratch (dex/attn/infllm.py fwd_scratch_layout)
    int *s = scratch.data_ptr<int>();
    p.sq = s + scr_offs[0]; p.sq_tail = s + scr_offs[1]; p.tile_counter = s + scr_offs[2]; p.n_enq = s + scr_offs[3];
    p.eng.pq = s + scr_offs[4]; p.eng.pq_tail = s + scr_offs[5]; S.red_cnt = s + scr_offs[6]; S.task_desc = s + scr_offs[7];
    S.row_ntask = s + scr_offs[8]; S.grp_cnt = s + scr_offs[9]; S.rows_done = s + scr_offs[10];
    p.sch_b0 = reinterpret_cast<rt::SchedB0 *>(s + scr_offs[11]);   // scr_offs[12]: total
    // the shared scheduler (rt_sched.cuh, kAvail = 0): rows = (part, head, 8 tokens), one group per row, no reductions
    S.sig = p.sig; S.n_remote = n_remote; S.remote_cnt = nullptr; S.arrived = nullptr;
    S.n_rows = S.n_groups = p.n_rows; S.row_pool = nullptr;                               // one group per row, counted deps
    S.n_red = 0; S.kind_red = 0; S.red_n_all = 0; S.n_enq = p.n_enq;
    S.sq = p.sq; S.sq_tail = p.sq_tail; S.tile_counter = p.tile_counter; S.n_static = 0;
    S.sub = 1; S.gang = 1; S.kind_task = infllm_fwd::kTask; S.kind_stop = infllm_fwd::kStop; S.kind_nop = infllm_fwd::kNop;
    S.n_workers = 2 * p.n_workers;                                  // STOP entries: one per compute warp
    S.sched_low = 4 * p.n_workers;
    S.sched_crit = S.sched_low; S.sched_batch = 64; S.q_work = nullptr;
    S.fwd_trig = 0;                                                 // relay forwards: trigger j of polled signal j (CE lane)
    S.n_units = 0; S.grp_unit = nullptr; S.unit_dep = nullptr; S.grp_work = nullptr;
    S.tl_sig = S.tl_row = S.tl_ev = S.trace = nullptr; S.q_all_in = nullptr; S.q_lim = 0;
    S.grp_max = 0; S.grp_avail_bits = nullptr; S.row_avail = nullptr;
    S.coop_len = 32;                                                // long dependency lists: warp-cooperative arrivals
    S.b0_bytes = rt::sched_b0_bytes(S.n_groups, S.n_rows, S.n_remote, S.sched_batch);
    static void *b0g = nullptr; static int64_t b0cap = 0;
    if (S.b0_bytes <= infllm_fwd_smem_bytes() - 512) S.b0_state = nullptr;
    else {
        if (S.b0_bytes > b0cap) { if (b0g) cudaFree(b0g); cudaMalloc(&b0g, S.b0_bytes); b0cap = S.b0_bytes; }
        S.b0_state = (char *)b0g;
    }
    // the official compute's parameters
    infllm_fwd::FArgs a{};
    a.q = q.data_ptr(); a.o = o.data_ptr(); a.o32 = f32 ? o32.data_ptr() : nullptr; a.q_row_stride = a.o_row_stride = 256;
    a.k = p.base + p.kv_off; a.v = p.base + p.kv_off + 256 * 2; a.k_row_stride = 512;
    a.lse = lse.data_ptr<float>();
    a.cu_q = cu_q.data_ptr<int>(); a.cu_k = cu_k.data_ptr<int>(); a.seqused_k = seqused_k.data_ptr<int>(); a.b = p.n_seg;
    a.blockmask = reinterpret_cast<const uint64_t *>(bm.data_ptr<int64_t>());
    a.total_q = (int)q.size(0); a.num_blocks_m = (a.total_q + 15) / 16; a.num_blocks_n = (int)((max_seqlen_k + 63) / 64);
    TORCH_CHECK(bm.size(0) == 2 && bm.size(1) == a.num_blocks_m && bm.size(2) == (a.num_blocks_n + 63) / 64);
    a.max_seqlen_q = (int)max_seqlen_q; a.max_seqlen_k = (int)max_seqlen_k; a.scale = (float)scale;
    at::Tensor &trace = g_trace, &tl_sig = g_tl_sig, &tl_row = g_tl_row, &tl_ev = g_tl_ev;
    if (getenv("DEX_TRACE") && atoi(getenv("DEX_TRACE"))) {
        if (!trace.defined()) trace = torch::zeros({8}, torch::dtype(torch::kInt64).device(torch::kCUDA));
        trace.zero_(); trace[3] = (int64_t)0x7fffffffffffffffLL; trace[7] = p.n_workers;
        p.trace = (unsigned long long *)trace.data_ptr();
        const auto o64 = torch::dtype(torch::kInt64).device(torch::kCUDA);
        tl_sig = torch::zeros({(int64_t)NSIG + 1}, o64); tl_row = torch::zeros({2 * (int64_t)S.n_rows + 2}, o64); tl_ev = torch::zeros({16}, o64);
        S.tl_sig = (unsigned long long *)tl_sig.data_ptr(); S.tl_row = (unsigned long long *)tl_row.data_ptr(); S.tl_ev = (unsigned long long *)tl_ev.data_ptr();
    }
    cudaMemsetAsync(scratch.data_ptr(), 0, scratch.numel() * 4, stream);
    cudaMemsetAsync(p.sig, 0, (size_t)NSIG * 8, stream);
    p.epoch = ++g_epoch; p.eng.epoch = p.epoch; S.epoch = p.epoch;
    nvshmemx_barrier_all_on_stream(stream);                      // every PE finished the previous call and staged its K / V
    const int P = nvshmem_n_pes();
    ce.ensure(P); ce.begin(stream, p.epoch);                  // (the copies queue behind `go`, recorded here)
    infllm_fwd_launch(a, p, grid, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    g_late = LateCtx{p.base, p.sig, p.eng.trig_flag, p.epoch, (int)offs[0], P, late};
    if (!late) ce_enqueue(stream, ce_tab);
    g_rep = ReportCtx{p.trace != nullptr, n_remote, S.n_rows, NSIG, NL_unused(), S.remote_sig};
    if (!late && p.trace) infllm_fwd_report();
    return (int64_t)p.epoch;
}

// sel int32 [2, L, K] (causal-filtered), is_local u8 [nb] -> the official row bitmap of the local / remote split,
// int64 [2, 2 L, ceil(nbn / 64)]
at::Tensor infllm_split_bitmap(at::Tensor sel, at::Tensor is_local, int64_t nbn) {
    TORCH_CHECK(sel.dtype() == torch::kInt32 && sel.dim() == 3 && sel.size(0) == 2 && sel.is_contiguous() &&
                is_local.dtype() == torch::kUInt8 && is_local.numel() >= nbn && is_local.is_contiguous());
    const int L = (int)sel.size(1), K = (int)sel.size(2), upr = (int)((nbn + 63) / 64);
    auto bm = torch::zeros({2, 2 * (int64_t)L, upr}, sel.options().dtype(torch::kInt64));
    infllm_split_bitmap_launch(sel.data_ptr<int>(), is_local.data_ptr<uint8_t>(), L, K, upr,
                               reinterpret_cast<unsigned long long *>(bm.data_ptr<int64_t>()), at::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return bm;
}

void register_infllm_fwd(pybind11::module &m) {
    m.def("infllm_split_bitmap", &infllm_split_bitmap, "the official row bitmap of the local / remote split");
    m.def("infllm_fwd_run", &infllm_fwd_run, "one InfLLM-v2 forward megakernel call");
    m.def("infllm_fwd_late", &infllm_fwd_late, "late mode: publish the start messages + copy-engine messages after the launch");
    m.def("infllm_fwd_report", &infllm_fwd_report, "DEX_TRACE: print the last call's trace (synchronises)");
    m.def("infllm_fwd_smem", &infllm_fwd_smem); m.def("infllm_fwd_grid", &infllm_fwd_grid_size);
}
