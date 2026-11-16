#include <array>
#include <algorithm>
// Host side of the distributed sparse backward (dex/scheduler/bwd.py builds every table).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cmath>
#include "dist_bwd_params.h"
#include "../runtime/rt_host.h"
#include "../runtime/rt_sched_host.h"

void dist_bwd_launch(const dist_bwd::Params &p, int grid, cudaStream_t stream);
int dist_bwd_smem_bytes();

namespace {
void *g_base = nullptr;
int64_t g_bytes = 0;
uint64_t g_epoch = 0;
}

// Collective (every PE, same nbytes): the backward's symmetric region, grown on demand and zeroed once when
// (re)allocated (signal words must start below every epoch). Returns a non-owning uint8 view of this PE's part.
at::Tensor bwd_sym_alloc(int64_t nbytes) {
    if (nbytes > g_bytes) {
        if (g_base) { nvshmem_barrier_all(); nvshmem_free(g_base); }
        g_base = nvshmem_malloc(nbytes);
        TORCH_CHECK(g_base, "nvshmem_malloc failed for the backward region (", nbytes, " bytes)");
        cudaMemset(g_base, 0, nbytes);
        cudaDeviceSynchronize();
        nvshmem_barrier_all();
        g_bytes = nbytes;
    }
    return at::from_blob(g_base, {g_bytes}, at::TensorOptions().dtype(at::kByte).device(at::kCUDA));
}

// The forward shares this region (ring_flash_attn_host.cpp DistSym): its data slots live
// here, its signal words in their own allocation; it barriers at its first call after a backward call (bwd_calls()).
void *bwd_region_base() { return g_base; }
int64_t bwd_region_bytes() { return g_bytes; }
uint64_t bwd_calls() { return g_epoch; }

// per peer: byte delta of its P2P-mapped view of the region (intra-node) and 1 for the NIC path (inter-node)
std::vector<at::Tensor> bwd_pe_info() {
    TORCH_CHECK(g_base);
    const int P = nvshmem_n_pes(), me = nvshmem_my_pe();
    auto delta = torch::zeros({P}, torch::kInt64), inter = torch::zeros({P}, torch::kInt32);
    for (int q = 0; q < P; ++q) {
        void *ptr = q == me ? g_base : nvshmem_ptr(g_base, q);
        if (ptr) delta[q] = (int64_t)((char *)ptr - (char *)g_base);
        else inter[q] = 1;
    }
    return {delta.cuda(), inter.cuda()};
}

// One backward call. offs: byte offsets of [sig, qb, kv, dqa, dkva, dql, dkl] inside the region; sizes: per-slot bytes
// [qb, kv, dqa, dkva]; dims: [NQS, NKS, NDQL, NDKL, h, hk, d, n_tasks, n_init, n_qm, n_km, n_remote, n_start_msgs, -,
// n_xenq, n_pq] (+ rectangle mode, python RECT_ORDER: [rect = 1, n_rect, G1, per, nw, arena ints, n_groups]);
// tabs: int32 tables in the order of dex/scheduler/bwd.py TABLE_ORDER; scratch: int32, zeroed here.
// ce_tab: CPU int64 [n, 6] (rt::CeMsg) -- the intra-node messages that go by copy engine, else empty.

int64_t dist_bwd_run(std::vector<int64_t> offs, std::vector<int64_t> sizes, std::vector<int64_t> dims,
                     std::vector<at::Tensor> tabs, at::Tensor pe_delta, at::Tensor pe_inter, at::Tensor scratch,
                     std::vector<int64_t> scr_offs, at::Tensor dq_out, at::Tensor dk_out, at::Tensor dv_out,
                     double scale, int64_t grid, at::Tensor ce_tab) {
    const bool rect = dims.size() > 16 && dims[16];
    TORCH_CHECK(g_base && offs.size() == 7 && sizes.size() == 4 && (dims.size() == 16 || dims.size() == 23 || dims.size() == 25) &&
                scr_offs.size() == (rect ? 28u : 22u));
    auto stream = at::cuda::getCurrentCUDAStream();
    dist_bwd::Params p{};
    char *b = (char *)g_base;
    p.sig = (uint64_t *)(b + offs[0]); p.qb = b + offs[1]; p.kv = b + offs[2]; p.dqa = b + offs[3];
    p.dkva = b + offs[4]; p.dql = b + offs[5]; p.dkl = b + offs[6];
    p.qb_bytes = sizes[0]; p.kv_bytes = sizes[1]; p.dqa_bytes = sizes[2]; p.dkva_bytes = sizes[3];
    p.NQS = dims[0]; p.NKS = dims[1]; p.NDQL = dims[2]; p.NDKL = dims[3];
    p.h = dims[4]; p.hk = dims[5]; p.d = dims[6];
    p.n_tasks = dims[7]; p.n_init = dims[8]; p.n_qm = dims[9]; p.n_km = dims[10]; p.n_remote = dims[11];
    p.eng.n_start = dims[12]; p.n_xenq = dims[14]; p.eng.n_pq = dims[15];   // dims[13]: unused
    p.scale = scale; p.scale_log2 = scale * M_LOG2E;
    p.dq_out = dq_out.data_ptr(); p.dk_out = dk_out.data_ptr(); p.dv_out = dv_out.data_ptr();
    int k = 0;
    auto T = [&]() -> const int32_t * { TORCH_CHECK(k < (int)tabs.size()); return tabs[k++].data_ptr<int32_t>(); };
    p.task_kv = T(); p.task_ptr = T(); p.items = T(); p.task_qs_ptr = T(); p.task_qs = T(); p.task_need = T();
    p.init_tasks = T(); p.qs_need = T(); p.ks_need = T(); p.qs_push_pe = T(); p.qs_push_dst = T();
    p.ks_push_pe = T(); p.ks_push_dst = T();
    p.qm_need = T(); p.qm_out = T(); p.qm_local = T(); p.qm_ptr = T(); p.qm_land = T();
    p.km_need = T(); p.km_out = T(); p.km_local = T(); p.km_ptr = T(); p.km_land = T();
    p.remote_sig = T(); p.sig_task_ptr = T(); p.sig_task = T(); p.sig_qm = T(); p.sig_km = T();
    p.qs_agg_need = T(); p.qs_agg_ptr = T(); p.qs_agg_land = T(); p.ks_agg_need = T(); p.ks_agg_ptr = T();
    p.ks_agg_land = T(); p.sig_qa = T(); p.sig_ka = T(); p.sig_fwd = T();
    p.eng.msg = reinterpret_cast<const rt::Msg *>(T()); p.eng.trig_ptr = T(); p.eng.trig_lane = T();
    const int64_t n_trig = tabs[k - 1].numel();     // triggers (trig_lane has one entry per trigger)
    rt::SchedParams &S = p.sch;                // the shared scheduler's tables (dex/scheduler/bwd.py)
    S.row_grp_ptr = T(); S.grp_beg = T(); S.grp_end = T(); S.grp_dep = T(); S.grp_need = T(); S.row_qdep = T();
    S.row_order = T(); S.sig_grp_ptr = T(); S.sig_grp = T(); S.sig_row_ptr = T(); S.sig_row = T();
    S.red_need = T(); S.red_kind = T(); S.red_pay = T(); S.red_n = T(); S.sig_red_ptr = T(); S.sig_red = T();
    p.qa_unit = T(); p.ka_unit = T();
    int64_t n_groups = p.n_tasks / 2;
    if (rect) {                                // rectangle tables (dist_bwd_params.h)
        p.rect = 1; p.n_rect = dims[17]; p.G1 = dims[18]; p.per = dims[19]; p.nw = dims[20]; n_groups = dims[22];
        p.hg = p.G1 > 0 ? (int)(n_groups / p.G1) : 1;
        p.g_rect = T(); p.g_col = T(); p.g_k0 = T(); p.g_cnt = T(); p.g_ks = T(); p.rect_pp = T(); p.rect_pair = T();
        p.rect_bk = T(); p.pair_bq = T();
        p.pair_bits = reinterpret_cast<const uint64_t *>(T()); p.pair_dbits = reinterpret_cast<const uint64_t *>(T());
        p.pair_prow = reinterpret_cast<const uint64_t *>(T()); p.pair_pcol = reinterpret_cast<const uint64_t *>(T());
        p.qslot_t = T(); p.kslot_t = T(); p.exp_order = T();
        if (dims.size() > 24 && dims[23]) {      // 128 x 64: the local pairs' half bitmaps (python rect_hbits), nw2 words each
            p.pair_hbits = reinterpret_cast<const uint64_t *>(T()); p.nw2 = (int)dims[24];
        }
        // the per-tile task tables, written by block 0 every call: task_kv, task_ptr, task_end [2 G], items (arena)
        static int32_t *buf = nullptr; static int64_t cap = 0;
        const int64_t need = dims[21] + 6 * n_groups + 128;
        if (need > cap) { if (buf) cudaFree(buf); cudaMalloc(&buf, need * 4); cap = need; }
        p.arena = buf; p.items = buf; p.task_kv = buf + dims[21]; p.task_ptr = p.task_kv + 2 * n_groups + 32;
        p.task_end = p.task_ptr + 2 * n_groups + 32;
    }
    TORCH_CHECK(k == (int)tabs.size(), "table count mismatch: ", k, " vs ", tabs.size());
    if (!rect) p.task_end = p.task_ptr + 1;
    p.pe_delta = pe_delta.data_ptr<int64_t>(); p.pe_inter = pe_inter.data_ptr<int32_t>();
    p.eng.base = b; p.eng.pe_delta = p.pe_delta; p.eng.pe_inter = p.pe_inter;
    p.eng.credit = nullptr; p.eng.call_id = 0;                 // cross-call order: the stream barrier below
    p.eng.trig_first = 2;                                      // 2 = input forwards fire before partial outputs
    p.eng.in_trig_first = p.NQS + p.NKS;                       // relay forwards (inputs): trig_fwd ids
    p.eng.trig_flag = nullptr;
    const int n_ce = ce_tab.numel() ? (int)ce_tab.size(0) : 0;   // rt::CeMsg rows (int64 [n, 6])
    static rt::CeExec ce;
    if (n_ce) {                                // trigger flags: epoch-valued, zeroed once (epochs only grow)
        static unsigned long long *flags = nullptr; static int64_t cap = 0;
        const int64_t need = n_trig;
        if (need > cap) { if (flags) cudaFree(flags); cudaMalloc(&flags, need * 8); cudaMemset(flags, 0, need * 8); cap = need; }
        p.eng.trig_flag = flags;
    }
    int *s = scratch.data_ptr<int>();
    p.sq = s + scr_offs[0]; p.sq_tail = s + scr_offs[1]; p.tile_counter = s + scr_offs[2]; p.task_cnt = s + scr_offs[3];
    p.qs_done = s + scr_offs[4]; p.ks_done = s + scr_offs[5]; p.qm_cnt = s + scr_offs[6]; p.km_cnt = s + scr_offs[7];
    p.n_enq = s + scr_offs[8]; p.quiet_gate = s + scr_offs[9]; p.mail = s + scr_offs[10];
    p.qa_cnt = s + scr_offs[11]; p.ka_cnt = s + scr_offs[12]; p.eng.pq = s + scr_offs[13]; p.eng.pq_tail = s + scr_offs[14];
    p.n_workers = grid - 1;
    {                                          // the shared scheduler (rt_sched.cuh): per-call state in `scratch`
        rt::SchedParams &S = p.sch;
        p.sch_on = true;
        S.sig = p.sig; S.n_remote = p.n_remote; S.remote_sig = p.remote_sig; S.remote_cnt = nullptr; S.arrived = nullptr;
        S.n_rows = S.n_groups = p.n_tasks / 2; S.row_pool = nullptr;
        if (rect) {                            // rows (head, rectangle), one group per task, groups materialised
            S.n_rows = S.n_groups = n_groups; S.grp_max = 1;
            p.avail_bits = reinterpret_cast<unsigned *>(s + scr_offs[22]); p.exp_next = s + scr_offs[23];
            p.done_q = s + scr_offs[24]; p.done_tail = s + scr_offs[25]; p.rowbuf = s + scr_offs[26];
            p.arena_off = s + scr_offs[27]; S.grp_avail_bits = p.avail_bits;
        }
        S.grp_cnt = s + scr_offs[15]; S.red_cnt = s + scr_offs[16]; S.task_desc = s + scr_offs[17];
        S.row_ntask = s + scr_offs[18]; p.row_cnt = s + scr_offs[19]; S.rows_done = s + scr_offs[20];
        p.sch_b0 = reinterpret_cast<rt::SchedB0 *>(s + scr_offs[21]);
        S.n_red = p.n_qm + p.n_km + p.n_xenq; S.kind_red = 0; S.red_n_all = 0; S.n_enq = p.n_enq;
        S.sq = p.sq; S.sq_tail = p.sq_tail; S.tile_counter = p.tile_counter; S.n_static = 0;
        S.sub = 2; S.gang = 2; S.kind_task = 0; S.kind_stop = 3; S.kind_nop = 4; S.n_workers = p.n_workers;
        S.sched_low = 2 * p.n_workers;
        S.sched_crit = S.sched_low; S.sched_batch = 64; S.q_work = nullptr;
        S.fwd_trig = p.NQS + p.NKS;            // relay forwards the engine sends on arrival (trig_lane: engine ones)
        S.b0_bytes = rt::sched_b0_bytes(S.n_groups, S.n_rows, S.n_remote, S.sched_batch);
        static void *b0g = nullptr; static int64_t b0cap = 0;
        if (S.b0_bytes <= dist_bwd_smem_bytes() - 512) S.b0_state = nullptr;   // block 0's shared memory
        else {
            if (S.b0_bytes > b0cap) { if (b0g) cudaFree(b0g); cudaMalloc(&b0g, S.b0_bytes); b0cap = S.b0_bytes; }
            S.b0_state = (char *)b0g;
        }
    }
    {                                          // the compute routine's view of the slots (sparse_bwd_params.h)
        auto &sp = p.sp;
        const int64_t qrow = (int64_t)p.h * p.d, krow = (int64_t)p.hk * p.d;
        sp.q = p.qb; sp.dout = p.qb + 128 * qrow * 2;
        sp.lse = reinterpret_cast<float *>(p.qb + 2 * 128 * qrow * 2); sp.dpsum = sp.lse + 128 * p.h;
        sp.k = p.kv; sp.v = p.kv + 128 * krow * 2;
        sp.dq_accum = reinterpret_cast<float *>(p.dqa);
        sp.dk_accum = reinterpret_cast<float *>(p.dkva); sp.dv_accum = sp.dk_accum + 128 * krow;
        sp.dk = sp.dv = nullptr;
        sp.q_row_stride = qrow; sp.k_row_stride = krow;
        sp.q_slot_stride = p.qb_bytes / 2; sp.k_slot_stride = p.kv_bytes / 2;
        sp.dq_slot_stride = p.dqa_bytes / 4; sp.dkv_slot_stride = p.dkva_bytes / 4;
        sp.lse_slot_stride = p.qb_bytes / 4; sp.lse_head_stride = 128;
        sp.h = p.h; sp.hk = p.hk; sp.d = p.d; sp.scale_softmax = p.scale; sp.scale_softmax_log2 = p.scale_log2;
    }
    p.sched_low = 2 * p.n_workers;
    cudaMemsetAsync(scratch.data_ptr(), 0, scratch.numel() * 4, stream);
    // this PE's signal words: the region is shared by every plan (layouts differ), so words of this call's layout may
    // hold another plan's slot bytes -- garbage >= any epoch would read as "arrived". Clearing here is safe: no peer
    // signals this PE for a call this PE has finished, and this call's signals come after the barrier below.
    cudaMemsetAsync(p.sig, 0, (size_t)(p.NQS + p.NKS + p.NDQL + p.NDKL) * 8, stream);
    cudaMemsetAsync(p.dqa, 0, (size_t)p.NQS * p.dqa_bytes, stream);       // this PE's accumulators only
    cudaMemsetAsync(p.dkva, 0, (size_t)p.NKS * p.dkva_bytes, stream);
    static at::Tensor trace, tl_sig, tl_row, tl_ev;
    if (getenv("DEX_TRACE") && atoi(getenv("DEX_TRACE"))) {      // per-call trace printout
        if (!trace.defined()) trace = torch::zeros({8}, torch::dtype(torch::kInt64).device(torch::kCUDA));
        trace.zero_(); trace[6] = (int64_t)0x7fffffffffffffffLL;
        p.trace = (unsigned long long *)trace.data_ptr();
        // the shared scheduler's timeline: arrival of every signal id, first dispatch per row, events
        const auto o64 = torch::dtype(torch::kInt64).device(torch::kCUDA);
        tl_sig = torch::zeros({(int64_t)(p.NQS + p.NKS + p.NDQL + p.NDKL) + 1}, o64);
        tl_row = torch::zeros({2 * (int64_t)p.sch.n_rows + 2}, o64); tl_ev = torch::zeros({16}, o64);
        p.sch.tl_sig = (unsigned long long *)tl_sig.data_ptr(); p.sch.tl_row = (unsigned long long *)tl_row.data_ptr();
        p.sch.tl_ev = (unsigned long long *)tl_ev.data_ptr();
    }
    p.sig_epoch = ++g_epoch;
    p.eng.sig = p.sig; p.eng.epoch = p.sig_epoch; p.sch.epoch = p.sig_epoch;
    // every PE has finished the previous call (no peer still reads a slot this call's pushes overwrite) and has
    // staged this call's inputs before anyone pushes them
    nvshmemx_barrier_all_on_stream(stream);
    if (n_ce) { ce.ensure(nvshmem_n_pes()); ce.begin(stream, p.sig_epoch); }
    dist_bwd_launch(p, (int)grid, stream);
    if (n_ce) {
        const int P = nvshmem_n_pes();
        static std::vector<char *> pb; static std::vector<uint64_t *> ps;
        pb.assign(P, nullptr); ps.assign(P, nullptr);
        for (int q = 0; q < P; ++q) { pb[q] = (char *)nvshmem_ptr(g_base, q); ps[q] = (uint64_t *)nvshmem_ptr(p.sig, q); }
        ce.enqueue(stream, reinterpret_cast<const rt::CeMsg *>(ce_tab.data_ptr<int64_t>()), n_ce, (char *)g_base, pb, ps,
                   p.sig, p.eng.trig_flag, p.sig_epoch, nullptr, 0, nullptr);
    }
    if (p.trace) {
        auto t = trace.cpu(); auto *v = t.data_ptr<int64_t>(); const double nw = p.n_workers;
        printf("[pe %d bwd trace] span %.2f ms | per worker: queue wait %.2f, compute %.2f, completion+push %.2f, "
               "dQ sums %.2f, dKV sums %.2f ms | tasks %lld\n", nvshmem_my_pe(), (v[7] - v[6]) * 1e-6, v[0] * 1e-6 / nw,
               v[1] * 1e-6 / nw, v[2] * 1e-6 / nw, v[3] * 1e-6 / nw, v[4] * 1e-6 / nw, (long long)v[5]);
        auto ev = tl_ev.cpu(); const int64_t *e = ev.data_ptr<int64_t>(); const double t0 = (double)e[1];
        auto pct = [&](std::vector<double> &x, double q) { return x.empty() ? 0.0 : x[(size_t)std::min<double>(x.size() - 1, q * x.size())]; };
        std::vector<double> arr, dis;
        {
            auto ts = tl_sig.cpu(); const int64_t *a = ts.data_ptr<int64_t>();
            std::vector<int> rs(p.n_remote);
            if (p.n_remote) cudaMemcpy(rs.data(), p.sch.remote_sig, p.n_remote * 4, cudaMemcpyDeviceToHost);
            for (int id : rs) if (a[id]) arr.push_back((a[id] - t0) * 1e-6);
            auto tr = tl_row.cpu(); const int64_t *r = tr.data_ptr<int64_t>();
            for (int i = 0; i < p.sch.n_rows; ++i) if (r[2 * i]) dis.push_back((r[2 * i] - t0) * 1e-6);
        }
        std::sort(arr.begin(), arr.end()); std::sort(dis.begin(), dis.end());
        printf("[pe %d bwd timeline] from scheduler start (ms): inputs arrived %zu/%d p50 %.1f p90 %.1f last %.1f | rows "
               "first dispatched %zu/%d p50 %.1f p90 %.1f last %.1f | STOP queued %.1f\n", nvshmem_my_pe(), arr.size(), p.n_remote,
               pct(arr, 0.5), pct(arr, 0.9), arr.empty() ? 0.0 : arr.back(), dis.size(), p.sch.n_rows, pct(dis, 0.5), pct(dis, 0.9),
               dis.empty() ? 0.0 : dis.back(), e[3] ? (e[3] - t0) * 1e-6 : -1.0);
        printf("[pe %d bwd sched] loop iterations %lld (%lld dispatching) | ms in arrivals %.1f, dispatch %.1f, queue writes %.1f | "
               "tasks fast path %lld, serial %lld | row-chunk scans %lld\n", nvshmem_my_pe(), (long long)e[4], (long long)e[5],
               e[6] * 1e-6, e[7] * 1e-6, e[8] * 1e-6, (long long)e[9], (long long)e[10], (long long)e[11]);
    }
    return (int64_t)g_epoch;
}

void register_dist_bwd(pybind11::module &m) {
    m.def("bwd_sym_alloc", &bwd_sym_alloc, "collective: the backward's symmetric region (non-owning uint8 view)");
    m.def("dist_share_region", []() { return true; }, "the forward's data slots share the backward region");
    m.def("bwd_pe_info", &bwd_pe_info, "per peer: P2P byte delta of the region, inter-node flag");
    m.def("dist_bwd_run", &dist_bwd_run, "one distributed sparse backward megakernel call");
    m.def("dist_bwd_smem_bytes", &dist_bwd_smem_bytes);
}
