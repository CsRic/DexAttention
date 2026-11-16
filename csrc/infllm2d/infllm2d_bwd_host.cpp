// Host side of the 2-D InfLLM-v2 backward megakernel (tables from
// dex/attn/infllm). Derived from csrc/infllm_bwd/infllm_bwd_host.cpp's infllm_bwd_run with: the
// incoming units' official parameter blocks, the dQ-landing reduction units (no-op entries after the owner sums /
// gateway aggregations), the unit completion scratch, and the relay-forward trigger range moved after the dQ returns
// (fwd_trig = in_trig_first, so the dQ returns are ordered as outputs). The official preprocess / convert kernels and the
// column bitmap of the own tasks are the existing module functions (infllm_bwd_prep, infllm_bwd_convert_dq, infllm_col_bitmap).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <cstdint>
#include "infllm2d_bwd_params.h"
#include "../runtime/rt_sched_host.h"
#ifdef INFLLM_BF16
#define INFLLM2D_DTYPE_HOST torch::kBFloat16
#else
#define INFLLM2D_DTYPE_HOST torch::kHalf
#endif
#include "../runtime/rt_host.h"

int infllm2d_bwd_smem_bytes();
int infllm2d_bwd_grid();
int64_t infllm2d_bwd_fparams_bytes();
void infllm2d_bwd_build_units(const infllm2d_bwd::UnitDesc *d, int n, void *dev, cudaStream_t stream);
void infllm2d_bwd_launch(const infllm2d_bwd::FArgs &a, const infllm2d_bwd::Params &p, int grid, cudaStream_t stream);

namespace {
uint64_t g_epoch2b = 0;
}

// One backward call. offs: [sig, kv, acc, land]; dims: [nb, n_tasks, n_km, n_ka, n_remote, n_start, n_pq, NSIG, NL, grid,
// n_xred (dQ landings), trig_unit0, n_units, fwd_trig]; tabs: dex/attn/infllm.py BWD_TABLE_ORDER + [unit_need]; scr_offs:
// bwd2d_scratch_layout (13 + unit_done + total). Own tasks' inputs as infllm_bwd_run. units: CPU int64 [n, 10] =
// (q_off, do_off, lse_off, dpsum_off, dqa_off, bm_off (region bytes), group, len, nbn, max_seqlen_k); ucu device int32
// [n, 5] = (0, len * 16, 0, 0, chunk end); fpu: device buffer. The units' dq_accum slots must be zeroed by the caller.
int64_t infllm2d_bwd_run(int64_t base, std::vector<int64_t> offs, std::vector<int64_t> dims, std::vector<at::Tensor> tabs,
                         at::Tensor scratch, std::vector<int64_t> scr_offs, at::Tensor pe_delta, at::Tensor pe_inter,
                         at::Tensor q, at::Tensor o, at::Tensor dout, at::Tensor dq, at::Tensor lse, at::Tensor dpsum,
                         at::Tensor dq_accum, at::Tensor cu_q, at::Tensor cu_k, at::Tensor seqused_k, at::Tensor bm,
                         int64_t max_seqlen_q, int64_t max_seqlen_k, double scale, at::Tensor dk_out, at::Tensor dv_out,
                         at::Tensor ce_tab, at::Tensor units, at::Tensor ucu, at::Tensor fpu) {
    TORCH_CHECK(base && offs.size() == 4 && dims.size() == 14 && scr_offs.size() == 15);
    TORCH_CHECK(cu_k.dtype() == torch::kInt32 && seqused_k.dtype() == torch::kInt32 && bm.dtype() == torch::kInt64 && bm.is_contiguous());
    TORCH_CHECK(dk_out.dtype() == INFLLM2D_DTYPE_HOST && dk_out.is_contiguous() && dv_out.is_contiguous() && dk_out.size(1) == 2 && dk_out.size(2) == 128);
    TORCH_CHECK(q.dtype() == INFLLM2D_DTYPE_HOST && q.dim() == 3 && q.size(1) == 2 && q.size(2) == 128 && q.is_contiguous());
    TORCH_CHECK(o.sizes() == q.sizes() && dout.sizes() == q.sizes() && dq.sizes() == q.sizes() && dout.is_contiguous());
    TORCH_CHECK(cu_q.dtype() == torch::kInt32 && lse.dtype() == torch::kFloat && dpsum.dtype() == torch::kFloat && dq_accum.dtype() == torch::kFloat);
    auto stream = at::cuda::getCurrentCUDAStream();
    infllm2d_bwd::Params p{};
    p.base = reinterpret_cast<char *>(base);
    p.sig = reinterpret_cast<uint64_t *>(p.base + offs[0]); p.kv_off = offs[1]; p.acc_off = offs[2]; p.land_off = offs[3];
    p.nb = (int)dims[0]; p.n_tasks = (int)dims[1]; p.n_km = (int)dims[2]; p.n_ka = (int)dims[3];
    const int n_remote = (int)dims[4], NSIG = (int)dims[7];
    p.eng.n_start = (int)dims[5]; p.eng.n_pq = (int)dims[6];
    int grid = (int)dims[9];
    if (grid <= 0) grid = infllm2d_bwd_grid();
    p.n_workers = grid - 1;
    const int n_xred = (int)dims[10];
    p.trig_unit0 = (int)dims[11];
    const int n_units = (int)dims[12], fwd_trig = (int)dims[13];
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
    p.unit_need = T();
    TORCH_CHECK(k == (int)tabs.size(), "table count mismatch: ", k, " vs ", tabs.size());
    p.dk_out = dk_out.data_ptr(); p.dv_out = dv_out.data_ptr();
    p.eng.base = p.base; p.eng.sig = p.sig; p.eng.pe_delta = pe_delta.data_ptr<int64_t>(); p.eng.pe_inter = pe_inter.data_ptr<int32_t>();
    p.eng.credit = nullptr; p.eng.call_id = 0;
    p.eng.trig_first = 2; p.eng.in_trig_first = fwd_trig; // input forwards fire before partial outputs
    p.eng.trig_flag = nullptr; p.eng.fire_log = nullptr; p.eng.late_flag = nullptr; p.eng.late_n_start = nullptr;
    const int n_ce = ce_tab.numel() ? (int)ce_tab.size(0) : 0;
    TORCH_CHECK(ce_tab.dtype() == torch::kInt64 && !ce_tab.is_cuda() && (n_ce == 0 || ce_tab.size(1) == 6));
    static rt::CeExec ce;
    static unsigned long long *flags = nullptr; static int64_t fcap = 0;
    if (n_trig > fcap) { if (flags) cudaFree(flags); cudaMalloc(&flags, n_trig * 8); cudaMemset(flags, 0, n_trig * 8); fcap = n_trig; }
    p.eng.trig_flag = flags;
    static std::vector<int64_t> delta_h; static int64_t delta_n = 0;
    if ((int64_t)pe_delta.numel() != delta_n) { auto h = pe_delta.cpu(); delta_h.assign(h.data_ptr<int64_t>(), h.data_ptr<int64_t>() + h.numel()); delta_n = h.numel(); }
    int *s = scratch.data_ptr<int>();
    p.sq = s + scr_offs[0]; p.sq_tail = s + scr_offs[1]; p.tile_counter = s + scr_offs[2]; p.ks_done = s + scr_offs[3];
    p.n_enq = s + scr_offs[4]; p.eng.pq = s + scr_offs[5]; p.eng.pq_tail = s + scr_offs[6];
    S.red_cnt = s + scr_offs[7]; S.task_desc = s + scr_offs[8]; S.row_ntask = s + scr_offs[9]; S.grp_cnt = s + scr_offs[10];
    S.rows_done = s + scr_offs[11]; p.sch_b0 = reinterpret_cast<rt::SchedB0 *>(s + scr_offs[12]);
    p.unit_done = s + scr_offs[13];
    S.sig = p.sig; S.n_remote = n_remote; S.remote_cnt = nullptr; S.arrived = nullptr;
    S.n_rows = S.n_groups = p.n_tasks; S.grp_need = nullptr; S.row_pool = nullptr;
    S.n_red = p.n_km + p.n_ka + n_xred; S.kind_red = 0; S.red_n_all = 0; S.n_enq = p.n_enq;
    S.sq = p.sq; S.sq_tail = p.sq_tail; S.tile_counter = p.tile_counter; S.n_static = 0;
    S.sub = 1; S.gang = 1; S.kind_task = infllm2d_bwd::kTask; S.kind_stop = infllm2d_bwd::kStop; S.kind_nop = infllm2d_bwd::kNop;
    S.n_workers = p.n_workers;
    S.sched_low = 2 * p.n_workers;
    S.sched_crit = S.sched_low; S.sched_batch = 64; S.q_work = nullptr;
    S.fwd_trig = fwd_trig;                                   // relay forwards: trigger fwd_trig + j on polled signal j
    S.n_units = 0; S.grp_unit = nullptr; S.unit_dep = nullptr; S.grp_work = nullptr;
    S.tl_sig = S.tl_row = S.tl_ev = S.trace = nullptr; S.q_all_in = nullptr; S.q_lim = 0;
    S.grp_max = 0; S.grp_avail_bits = nullptr; S.row_avail = nullptr;
    S.coop_len = 0;
    S.b0_bytes = rt::sched_b0_bytes(S.n_groups, S.n_rows, S.n_remote, S.sched_batch);
    static void *b0g = nullptr; static int64_t b0cap = 0;
    if (S.b0_bytes <= infllm2d_bwd_smem_bytes() - 512) S.b0_state = nullptr;
    else {
        if (S.b0_bytes > b0cap) { if (b0g) cudaFree(b0g); cudaMalloc(&b0g, S.b0_bytes); b0cap = S.b0_bytes; }
        S.b0_state = (char *)b0g;
    }
    // the own tasks' official parameters (as infllm_bwd_host.cpp fill_q + infllm_bwd_run)
    infllm2d_bwd::FArgs a{};
    const int b = (int)cu_q.numel() - 1;
    a.q = q.data_ptr(); a.o = o.data_ptr(); a.dout = dout.data_ptr(); a.dq = dq.data_ptr();
    a.q_row_stride = a.o_row_stride = a.do_row_stride = a.dq_row_stride = 256;
    a.lse = lse.data_ptr<float>(); a.dpsum = dpsum.data_ptr<float>(); a.dq_accum = dq_accum.data_ptr<float>();
    a.cu_q = cu_q.data_ptr<int>(); a.b = b;
    a.total_q = (int)q.size(0); a.max_seqlen_q = (int)max_seqlen_q; a.scale = (float)scale;
    TORCH_CHECK(cu_k.numel() == cu_q.numel() && seqused_k.numel() == a.b);
    a.k = p.base + p.kv_off; a.v = p.base + p.kv_off + 256 * 2;
    a.k_row_stride = 512;
    a.dk_accum = reinterpret_cast<float *>(p.base + p.acc_off); a.dv_accum = a.dk_accum + 256; a.acc_row_stride = 512;
    a.cu_k = cu_k.data_ptr<int>(); a.seqused_k = seqused_k.data_ptr<int>();
    a.blockmask = reinterpret_cast<const uint64_t *>(bm.data_ptr<int64_t>());
    a.num_blocks_m = a.total_q / 16; a.num_blocks_n = (int)bm.size(1); a.max_seqlen_k = (int)max_seqlen_k;
    TORCH_CHECK(bm.size(0) == 2 && bm.size(2) == (a.num_blocks_m + 63) / 64 && a.num_blocks_n >= (a.max_seqlen_k + 63) / 64);
    // [2-D] the incoming units' parameter blocks
    TORCH_CHECK(units.dtype() == torch::kInt64 && !units.is_cuda() && (n_units == 0 || (units.size(0) == n_units && units.size(1) == 10)));
    TORCH_CHECK(ucu.dtype() == torch::kInt32 && ucu.is_cuda() && ucu.is_contiguous() && (n_units == 0 || ucu.size(0) == n_units));
    TORCH_CHECK(fpu.is_cuda() && fpu.numel() >= (int64_t)n_units * infllm2d_bwd_fparams_bytes());
    if (n_units) {
        std::vector<infllm2d_bwd::UnitDesc> ud(n_units);
        const int64_t *u = units.data_ptr<int64_t>();
        int *cu = ucu.data_ptr<int>();
        float *acc = reinterpret_cast<float *>(p.base + p.acc_off);
        for (int i = 0; i < n_units; ++i, u += 10) {
            infllm2d_bwd::UnitDesc &d = ud[i];
            const int g = (int)u[6];
            d.q = p.base + u[0]; d.dout = p.base + u[1]; d.lse = p.base + u[2]; d.dpsum = p.base + u[3]; d.dq_accum = p.base + u[4];
            d.bm = p.base + u[5];
            d.k = p.base + p.kv_off + g * 128 * 2; d.v = p.base + p.kv_off + 256 * 2 + g * 128 * 2;
            d.dk_acc = acc + g * 128; d.dv_acc = acc + 256 + g * 128;
            d.cu_q = cu + 5 * i; d.cu_k = cu + 5 * i + 2; d.seqused = cu + 5 * i + 4;
            d.len = (int)u[7]; d.nbn = (int)u[8]; d.max_seqlen_k = (int)u[9]; d.scale = (float)scale;
        }
        infllm2d_bwd_build_units(ud.data(), n_units, fpu.data_ptr(), stream);
    }
    p.fpu = fpu.data_ptr();
    static at::Tensor trace;
    if (getenv("DEX_TRACE") && atoi(getenv("DEX_TRACE"))) {
        if (!trace.defined()) trace = torch::zeros({8}, torch::dtype(torch::kInt64).device(torch::kCUDA));
        trace.zero_(); trace[5] = (int64_t)0x7fffffffffffffffLL;
        p.trace = (unsigned long long *)trace.data_ptr();
    }
    cudaMemsetAsync(scratch.data_ptr(), 0, scratch.numel() * 4, stream);
    cudaMemsetAsync(p.sig, 0, (size_t)NSIG * 8, stream);
    cudaMemsetAsync(p.base + p.acc_off, 0, (size_t)p.nb * infllm2d_bwd::kSlotBytes, stream);
    p.epoch = ++g_epoch2b; p.eng.epoch = p.epoch; S.epoch = p.epoch;
    nvshmemx_barrier_all_on_stream(stream);
    const int P = nvshmem_n_pes();
    if (n_ce) { ce.ensure(P); ce.begin(stream, p.epoch); }
    infllm2d_bwd_launch(a, p, grid, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (n_ce) {
        std::vector<char *> pb(P, nullptr); std::vector<uint64_t *> ps(P, nullptr);
        for (int q_ = 0; q_ < P; ++q_) { pb[q_] = p.base + delta_h[q_]; ps[q_] = reinterpret_cast<uint64_t *>(pb[q_] + offs[0]); }
        ce.enqueue(stream, reinterpret_cast<const rt::CeMsg *>(ce_tab.data_ptr<int64_t>()), n_ce, p.base, pb, ps, p.sig,
                   p.eng.trig_flag, p.epoch, nullptr, 0, nullptr);
    }
    if (p.trace) {
        auto t = trace.cpu(); auto *v = t.data_ptr<int64_t>(); const double nw = p.n_workers;
        printf("[pe %d infllm2d bwd trace] span %.2f ms | per worker: queue wait %.2f, compute %.2f, completion %.2f, sums %.2f ms | tasks %lld\n",
               nvshmem_my_pe(), (v[6] - v[5]) * 1e-6, v[0] * 1e-6 / nw, v[1] * 1e-6 / nw, v[2] * 1e-6 / nw, v[3] * 1e-6 / nw, (long long)v[4]);
    }
    return (int64_t)p.epoch;
}

int64_t infllm2d_bwd_grid_size() { return infllm2d_bwd_grid(); }
int64_t infllm2d_bwd_param_bytes() { return infllm2d_bwd_fparams_bytes(); }

void register_infllm2d_bwd(pybind11::module &m) {
    m.def("infllm2d_bwd_run", &infllm2d_bwd_run, "2-D: one InfLLM-v2 backward megakernel call (own + foreign tasks)");
    m.def("infllm2d_bwd_grid", &infllm2d_bwd_grid_size);
    m.def("infllm2d_bwd_param_bytes", &infllm2d_bwd_param_bytes);
}
