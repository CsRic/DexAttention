// Host side of the 2-D InfLLM-v2 forward megakernel (tables from
// dex/attn/infllm). Derived from csrc/infllm_fwd/infllm_fwd_host.cpp (non-late mode only) with:
// the foreign rows' tables (frow, unit_need), the per-unit official parameter blocks (built here from byte offsets into
// the region), the partial-landing reduction units (no-op entries: the kernel ends only when every partial has landed),
// and the whole signal range of the 2-D layout reset per call. Region: the shared symmetric region (bwd_sym_alloc).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include "infllm2d_fwd_params.h"
#include "../runtime/rt_sched_host.h"
#ifdef INFLLM_BF16
#define INFLLM2D_DTYPE_HOST torch::kBFloat16
#else
#define INFLLM2D_DTYPE_HOST torch::kHalf
#endif
#include "../runtime/rt_host.h"

int infllm2d_fwd_smem_bytes();
int infllm2d_fwd_grid();
int64_t infllm2d_fwd_fparams_bytes();
void infllm2d_fwd_build_units(const infllm2d_fwd::UnitDesc *d, int n, void *dev, cudaStream_t stream);
void infllm2d_fwd_launch(const infllm2d_fwd::FArgs &a, const infllm2d_fwd::Params &p, int grid, cudaStream_t stream);
void infllm2d_split_bitmap_launch(const int *sel, const uint8_t *is_local, int L, int K, int upr, unsigned long long *bm, cudaStream_t stream);
void infllm2d_unit_bitmap_launch(bool col, const int *sel, const int16_t *egpu, int want, int L, int K, int nbn,
                                 unsigned long long *out, cudaStream_t stream);

namespace {
uint64_t g_epoch2f = 0;
rt::CeExec ce2f;
std::vector<int64_t> delta2f_h; int64_t delta2f_n = 0;
at::Tensor g2f_trace, g2f_tl_sig, g2f_tl_row, g2f_tl_ev;
}  // namespace

// One forward call. dims: [nb, L, K, n_seg, nrh, n_rows, n_remote, n_start, n_pq, NSIG, grid, n_rows_own, n_red,
// trig_unit0, n_units]; tabs: int32 [msg] + dex/attn/infllm.py FWD_TABLE_ORDER + [frow, unit_need]; scr_offs: fwd2d_scratch_layout
// (13 + unit_done + total). units: CPU int64 [n_units, 8] = (q_off, bm_off, o32_off, lse_off (region bytes), group, len,
// nbn, max_seqlen_k); ucu: device int32 [n_units, 5] = (0, len * 16, 0, 0, chunk end); fpu: device buffer (>= n_units x
// fparams bytes). NSIG: the whole signal range reset here (static run signals + the 2-D ones). Returns the epoch.
int64_t infllm2d_fwd_run(int64_t base, std::vector<int64_t> offs, std::vector<int64_t> dims, std::vector<at::Tensor> tabs,
                         at::Tensor scratch, std::vector<int64_t> scr_offs, at::Tensor pe_delta, at::Tensor pe_inter,
                         at::Tensor q, at::Tensor o, at::Tensor o32, at::Tensor lse, at::Tensor cu_q, at::Tensor cu_k, at::Tensor seqused_k,
                         at::Tensor bm, int64_t max_seqlen_q, int64_t max_seqlen_k, double scale, at::Tensor ce_tab,
                         at::Tensor units, at::Tensor ucu, at::Tensor fpu) {
    TORCH_CHECK(base && offs.size() == 2 && dims.size() == 15 && scr_offs.size() == 15);
    TORCH_CHECK(q.dtype() == INFLLM2D_DTYPE_HOST && q.dim() == 3 && q.size(1) == 2 && q.size(2) == 128 && q.is_contiguous());
    TORCH_CHECK(o.is_contiguous() && lse.dtype() == torch::kFloat && lse.is_contiguous() && lse.size(0) == 2 && lse.size(1) == q.size(0));
    TORCH_CHECK(o32.dtype() == torch::kFloat && o32.sizes() == q.sizes() && o32.is_contiguous());
    TORCH_CHECK(cu_q.dtype() == torch::kInt32 && cu_k.dtype() == torch::kInt32 && seqused_k.dtype() == torch::kInt32 &&
                bm.dtype() == torch::kInt64 && bm.is_contiguous());
    auto stream = at::cuda::getCurrentCUDAStream();
    infllm2d_fwd::Params p{};
    p.base = reinterpret_cast<char *>(base);
    p.sig = reinterpret_cast<uint64_t *>(p.base + offs[0]); p.kv_off = offs[1];
    p.nb = (int)dims[0]; p.L = (int)dims[1]; p.n_seg = (int)dims[3];
    p.nrh = (int)dims[4]; p.n_rows = (int)dims[5];
    const int n_remote = (int)dims[6], NSIG = (int)dims[9];
    p.eng.n_start = (int)dims[7]; p.eng.n_pq = (int)dims[8];
    int grid = (int)dims[10];
    if (grid <= 0) grid = infllm2d_fwd_grid();
    p.n_workers = grid - 1;
    p.n_rows_own = (int)dims[11];
    const int n_red = (int)dims[12];
    p.trig_unit0 = (int)dims[13];
    const int n_units = (int)dims[14];
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
    p.frow = T(); p.unit_need = T();
    TORCH_CHECK(k == (int)tabs.size(), "table count mismatch: ", k, " vs ", tabs.size());
    p.eng.base = p.base; p.eng.sig = p.sig; p.eng.pe_delta = pe_delta.data_ptr<int64_t>(); p.eng.pe_inter = pe_inter.data_ptr<int32_t>();
    p.eng.credit = nullptr; p.eng.call_id = 0; p.eng.trig_first = 0; p.eng.in_trig_first = -1; p.eng.fire_log = nullptr;
    p.eng.late_flag = nullptr; p.eng.late_n_start = nullptr;
    const int n_ce = ce_tab.numel() ? (int)ce_tab.size(0) : 0;
    TORCH_CHECK(ce_tab.dtype() == torch::kInt64 && !ce_tab.is_cuda() && (n_ce == 0 || ce_tab.size(1) == 6));
    static unsigned long long *flags = nullptr; static int64_t fcap = 0;
    if (n_trig > fcap) { if (flags) cudaFree(flags); cudaMalloc(&flags, n_trig * 8); cudaMemset(flags, 0, n_trig * 8); fcap = n_trig; }
    p.eng.trig_flag = flags;
    if ((int64_t)pe_delta.numel() != delta2f_n) { auto h = pe_delta.cpu(); delta2f_h.assign(h.data_ptr<int64_t>(), h.data_ptr<int64_t>() + h.numel()); delta2f_n = h.numel(); }
    int *s = scratch.data_ptr<int>();
    p.sq = s + scr_offs[0]; p.sq_tail = s + scr_offs[1]; p.tile_counter = s + scr_offs[2]; p.n_enq = s + scr_offs[3];
    p.eng.pq = s + scr_offs[4]; p.eng.pq_tail = s + scr_offs[5]; S.red_cnt = s + scr_offs[6]; S.task_desc = s + scr_offs[7];
    S.row_ntask = s + scr_offs[8]; S.grp_cnt = s + scr_offs[9]; S.rows_done = s + scr_offs[10];
    p.sch_b0 = reinterpret_cast<rt::SchedB0 *>(s + scr_offs[11]);
    p.unit_done = s + scr_offs[13];                                  // scr_offs[12]: (reserved), [14]: total
    S.sig = p.sig; S.n_remote = n_remote; S.remote_cnt = nullptr; S.arrived = nullptr;
    S.n_rows = S.n_groups = p.n_rows; S.row_pool = nullptr;
    S.n_red = n_red; S.kind_red = 0; S.red_n_all = 0; S.n_enq = p.n_enq;
    S.sq = p.sq; S.sq_tail = p.sq_tail; S.tile_counter = p.tile_counter; S.n_static = 0;
    S.sub = 1; S.gang = 1; S.kind_task = infllm2d_fwd::kTask; S.kind_stop = infllm2d_fwd::kStop; S.kind_nop = infllm2d_fwd::kNop;
    S.n_workers = 2 * p.n_workers;                                  // STOP entries: one per compute warp
    S.sched_low = 4 * p.n_workers;
    S.sched_crit = S.sched_low; S.sched_batch = 64; S.q_work = nullptr;
    S.fwd_trig = 0;                                                 // relay forwards: trigger j of polled signal j (CE lane)
    S.n_units = 0; S.grp_unit = nullptr; S.unit_dep = nullptr; S.grp_work = nullptr;
    S.tl_sig = S.tl_row = S.tl_ev = S.trace = nullptr; S.q_all_in = nullptr; S.q_lim = 0;
    S.grp_max = 0; S.grp_avail_bits = nullptr; S.row_avail = nullptr;
    S.coop_len = 32;
    S.b0_bytes = rt::sched_b0_bytes(S.n_groups, S.n_rows, S.n_remote, S.sched_batch);
    static void *b0g = nullptr; static int64_t b0cap = 0;
    if (S.b0_bytes <= infllm2d_fwd_smem_bytes() - 512) S.b0_state = nullptr;
    else {
        if (S.b0_bytes > b0cap) { if (b0g) cudaFree(b0g); cudaMalloc(&b0g, S.b0_bytes); b0cap = S.b0_bytes; }
        S.b0_state = (char *)b0g;
    }
    // the own rows' official parameters (as infllm_fwd_host.cpp)
    infllm2d_fwd::FArgs a{};
    a.q = q.data_ptr(); a.o = o.data_ptr(); a.o32 = o32.data_ptr(); a.q_row_stride = a.o_row_stride = 256;
    a.k = p.base + p.kv_off; a.v = p.base + p.kv_off + 256 * 2; a.k_row_stride = 512;
    a.lse = lse.data_ptr<float>();
    a.cu_q = cu_q.data_ptr<int>(); a.cu_k = cu_k.data_ptr<int>(); a.seqused_k = seqused_k.data_ptr<int>(); a.b = p.n_seg;
    a.blockmask = reinterpret_cast<const uint64_t *>(bm.data_ptr<int64_t>());
    a.total_q = (int)q.size(0); a.num_blocks_m = (a.total_q + 15) / 16; a.num_blocks_n = (int)((max_seqlen_k + 63) / 64);
    TORCH_CHECK(bm.size(0) == 2 && bm.size(1) == a.num_blocks_m && bm.size(2) == (a.num_blocks_n + 63) / 64);
    a.max_seqlen_q = (int)max_seqlen_q; a.max_seqlen_k = (int)max_seqlen_k; a.scale = (float)scale;
    // [2-D] the incoming units' parameter blocks
    TORCH_CHECK(units.dtype() == torch::kInt64 && !units.is_cuda() && (n_units == 0 || (units.size(0) == n_units && units.size(1) == 8)));
    TORCH_CHECK(ucu.dtype() == torch::kInt32 && ucu.is_cuda() && ucu.is_contiguous() && (n_units == 0 || ucu.size(0) == n_units));
    TORCH_CHECK(fpu.is_cuda() && fpu.numel() >= (int64_t)n_units * infllm2d_fwd_fparams_bytes());
    if (n_units) {
        std::vector<infllm2d_fwd::UnitDesc> ud(n_units);
        const int64_t *u = units.data_ptr<int64_t>();
        int *cu = ucu.data_ptr<int>();
        for (int i = 0; i < n_units; ++i, u += 8) {
            infllm2d_fwd::UnitDesc &d = ud[i];
            const int g = (int)u[4];
            d.q = p.base + u[0]; d.bm = p.base + u[1]; d.o32 = p.base + u[2]; d.lse = p.base + u[3];
            d.k = p.base + p.kv_off + g * 128 * 2; d.v = p.base + p.kv_off + 256 * 2 + g * 128 * 2;
            d.cu_q = cu + 5 * i; d.cu_k = cu + 5 * i + 2; d.seqused = cu + 5 * i + 4;
            d.len = (int)u[5]; d.nbn = (int)u[6]; d.max_seqlen_k = (int)u[7]; d.scale = (float)scale;
        }
        infllm2d_fwd_build_units(ud.data(), n_units, fpu.data_ptr(), stream);
    }
    p.fpu = fpu.data_ptr();
    if (getenv("DEX_TRACE") && atoi(getenv("DEX_TRACE"))) {
        if (!g2f_trace.defined()) g2f_trace = torch::zeros({8}, torch::dtype(torch::kInt64).device(torch::kCUDA));
        g2f_trace.zero_(); g2f_trace[3] = (int64_t)0x7fffffffffffffffLL; g2f_trace[7] = p.n_workers;
        p.trace = (unsigned long long *)g2f_trace.data_ptr();
    }
    cudaMemsetAsync(scratch.data_ptr(), 0, scratch.numel() * 4, stream);
    cudaMemsetAsync(p.sig, 0, (size_t)NSIG * 8, stream);
    p.epoch = ++g_epoch2f; p.eng.epoch = p.epoch; S.epoch = p.epoch;
    nvshmemx_barrier_all_on_stream(stream);                      // every PE finished the previous call and staged its inputs
    const int P = nvshmem_n_pes();
    ce2f.ensure(P); ce2f.begin(stream, p.epoch);
    infllm2d_fwd_launch(a, p, grid, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (n_ce) {
        std::vector<char *> pb(P, nullptr); std::vector<uint64_t *> ps(P, nullptr);
        for (int q_ = 0; q_ < P; ++q_) { pb[q_] = p.base + delta2f_h[q_]; ps[q_] = reinterpret_cast<uint64_t *>(pb[q_] + offs[0]); }
        ce2f.enqueue(stream, reinterpret_cast<const rt::CeMsg *>(ce_tab.data_ptr<int64_t>()), n_ce, p.base, pb, ps, p.sig,
                     p.eng.trig_flag, p.epoch, nullptr, 0, nullptr);
    }
    if (p.trace) {
        auto t = g2f_trace.cpu(); auto *v = t.data_ptr<int64_t>();
        const double nw = 2.0 * (double)std::max<int64_t>(1, v[7]);
        printf("[pe %d infllm2d fwd trace] span %.2f ms | per warp: queue wait %.2f, compute %.2f ms | tokens %lld\n",
               nvshmem_my_pe(), (v[4] - v[3]) * 1e-6, v[0] * 1e-6 / nw, v[1] * 1e-6 / nw, (long long)v[2]);
    }
    return (int64_t)p.epoch;
}

// (as infllm_split_bitmap) sel int32 [2, L, K], is_local u8 [nb] -> int64 [2, 2 L, ceil(nbn / 64)]
at::Tensor infllm2d_split_bitmap(at::Tensor sel, at::Tensor is_local, int64_t nbn) {
    TORCH_CHECK(sel.dtype() == torch::kInt32 && sel.dim() == 3 && sel.size(0) == 2 && sel.is_contiguous() &&
                is_local.dtype() == torch::kUInt8 && is_local.numel() >= nbn && is_local.is_contiguous());
    const int L = (int)sel.size(1), K = (int)sel.size(2), upr = (int)((nbn + 63) / 64);
    auto bm = torch::zeros({2, 2 * (int64_t)L, upr}, sel.options().dtype(torch::kInt64));
    infllm2d_split_bitmap_launch(sel.data_ptr<int>(), is_local.data_ptr<uint8_t>(), L, K, upr,
                                 reinterpret_cast<unsigned long long *>(bm.data_ptr<int64_t>()), at::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return bm;
}

// [2-D] restricted bitmap of a token range of ONE group into out (an int64 view of the region, zeroed here):
// col = false: the official row bitmap [L, ceil(nbn / 64)]; col = true: the official column bitmap [nbn, ceil(L / 64)].
// sel int32 [L, K] (causal-filtered), egpu int16 [L, K] (the entry's GPU); only entries with egpu == want.
void infllm2d_unit_bitmap(bool col, at::Tensor sel, at::Tensor egpu, int64_t want, int64_t nbn, at::Tensor out) {
    TORCH_CHECK(sel.dtype() == torch::kInt32 && sel.dim() == 2 && sel.is_contiguous() && egpu.dtype() == torch::kInt16 &&
                egpu.sizes() == sel.sizes() && egpu.is_contiguous() && out.dtype() == torch::kInt64 && out.is_contiguous());
    const int L = (int)sel.size(0), K = (int)sel.size(1);
    const int64_t need = col ? nbn * ((L + 63) / 64) : (int64_t)L * ((nbn + 63) / 64);
    TORCH_CHECK(out.numel() == need, "bitmap size ", out.numel(), " != ", need);
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaMemsetAsync(out.data_ptr(), 0, need * 8, stream);
    infllm2d_unit_bitmap_launch(col, sel.data_ptr<int>(), egpu.data_ptr<int16_t>(), (int)want, L, K, (int)nbn,
                                reinterpret_cast<unsigned long long *>(out.data_ptr<int64_t>()), stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int64_t infllm2d_fwd_grid_size() { return infllm2d_fwd_grid(); }
int64_t infllm2d_fwd_param_bytes() { return infllm2d_fwd_fparams_bytes(); }

void register_infllm2d_fwd(pybind11::module &m) {
    m.def("infllm2d_split_bitmap", &infllm2d_split_bitmap, "2-D: the own rows' local / remote split bitmap");
    m.def("infllm2d_unit_bitmap", &infllm2d_unit_bitmap, "2-D: restricted row / column bitmap of one unit");
    m.def("infllm2d_fwd_run", &infllm2d_fwd_run, "2-D: one InfLLM-v2 forward megakernel call (own + foreign rows)");
    m.def("infllm2d_fwd_grid", &infllm2d_fwd_grid_size);
    m.def("infllm2d_fwd_param_bytes", &infllm2d_fwd_param_bytes);
}
