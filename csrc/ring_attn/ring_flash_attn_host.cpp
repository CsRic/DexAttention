// PyTorch C++ extension binding (module `dex_cuda`): NVSHMEM bootstrap, the symmetric-heap layout of the
// distributed forward, per-call staging of the local Q / K / V blocks, parameter packing and the intra-node
// copy-engine transfers. The kernel launches are in ring_flash_attn.cu.

#include <torch/extension.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <ATen/cuda/CUDAContext.h>
#include <bootstrap_device_host/nvshmem_uniqueid.h>
#include <cuda.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <stdexcept>
#include <cmath>
#include <cstdlib>
#include <string>
#include <algorithm>

#include "sparse_params.h"
#include "../dist_attn/dist_merge_params.h"
#include "../dist_attn/dist_params.h"
#include "../runtime/rt_host.h"
#include "../runtime/rt_sched_host.h"
#include "../dist_attn/dist_scatter_params.h"   // Scatter_params lives with the merge / scatter device code

using namespace ring_attn;

extern "C" void ring_flash_attn_sparse_launch(const Sparse_fwd_params &params, cudaStream_t stream);
extern "C" void ring_flash_attn_sparse64_launch(const Sparse_fwd_params &p, cudaStream_t stream);
extern "C" void dist_partial_launch(const Sparse_fwd_params &params, cudaStream_t stream);
extern "C" void dist_merge_launch(const Merge_params &params, int max_slots, cudaStream_t stream);
extern "C" void dist_fwd_launch(const Dist_fwd_params &params, cudaStream_t stream);
extern "C" void dist_scatter3_launch(const Scatter_params &q, const Scatter_params &k, const Scatter_params &v,
                                     cudaStream_t stream);

// ── NVSHMEM bootstrap via an existing torch.distributed ProcessGroup ─────────
static bool g_nvshmem_initialized = false;

void ring_attn_init_nvshmem(c10::intrusive_ptr<c10d::ProcessGroup> pg) {
    if (g_nvshmem_initialized) return;

    const int rank       = pg->getRank();
    const int world_size = pg->getSize();

    // Rank 0 generates the uniqueid, broadcasts it through the PG (NCCL).
    nvshmemx_uniqueid_t uid;
    auto options = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);
    torch::Tensor uid_tensor = torch::empty({(int64_t)sizeof(nvshmemx_uniqueid_t)}, options);

    if (rank == 0) {
        nvshmemx_get_uniqueid(&uid);
        cudaMemcpy(uid_tensor.data_ptr(), &uid, sizeof(uid), cudaMemcpyHostToDevice);
    }
    std::vector<at::Tensor> uid_tensors = {uid_tensor};
    pg->broadcast(uid_tensors)->wait();
    cudaMemcpy(&uid, uid_tensor.data_ptr(), sizeof(uid), cudaMemcpyDeviceToHost);

    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    nvshmemx_set_attr_uniqueid_args(rank, world_size, &uid, &attr);
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);

    g_nvshmem_initialized = true;
}

void ring_attn_finalize_nvshmem() {
    if (!g_nvshmem_initialized) return;
    nvshmem_finalize();
    g_nvshmem_initialized = false;
}

// ── Block-sparse forward (single GPU, no NVSHMEM) ────────────────────────────
// The mask arrives already compiled into a CSR over Q tiles (layout: sparse_desc.h). `work_coord` carries the claim
// order, so load balancing is decided on the host.

static std::vector<torch::Tensor> sparse_fwd_impl(bool kv64,
    torch::Tensor q,             // (b, seqlen_q, h,   d) fp16
    torch::Tensor k,             // (b, seqlen_k, h_k, d) fp16
    torch::Tensor v,             // (b, seqlen_k, h_k, d) fp16
    torch::Tensor out,           // (b, seqlen_q, h,   d) pre-allocated
    torch::Tensor lse,           // (b, h, seqlen_q) fp32 pre-allocated
    torch::Tensor kv_indptr,     // (num_m_block + 1) int32 cuda
    torch::Tensor kv_indices,    // (nnz) int32 cuda, ascending within a row, masked last
    torch::Tensor kv_mask_nb,    // (nnz) int32 cuda, the mask's n_block per entry
    torch::Tensor row_n_masked,  // (num_m_block) int32 cuda
    torch::Tensor row_mask_mb,   // (num_m_block) int32 cuda, the mask's m_block per row
    torch::Tensor work_coord,    // (n_work) int32 cuda
    double softmax_scale
) {
    const int b        = q.size(0);
    const int seqlen_q = q.size(1);
    const int h        = q.size(2);
    const int d        = q.size(3);
    const int seqlen_k = k.size(1);
    const int h_k      = k.size(2);

    TORCH_CHECK(q.scalar_type() == torch::kFloat16, "fp16 only");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous());
    TORCH_CHECK(d == 128, "head_dim must be 128");
    TORCH_CHECK(seqlen_q % 128 == 0 && seqlen_k % (kv64 ? 64 : 128) == 0,
                "seqlens must be multiples of the 128 x kBlockN tile");
    TORCH_CHECK(kv_indptr.scalar_type() == torch::kInt32 &&
                kv_indices.scalar_type() == torch::kInt32 &&
                kv_mask_nb.scalar_type() == torch::kInt32 &&
                row_mask_mb.scalar_type() == torch::kInt32 &&
                row_n_masked.scalar_type() == torch::kInt32 &&
                work_coord.scalar_type() == torch::kInt32, "tables must be int32");
    TORCH_CHECK(kv_indptr.is_cuda() && kv_indices.is_cuda() &&
                row_n_masked.is_cuda() && work_coord.is_cuda(), "tables must be on device");

    const int num_m_block = seqlen_q / 128;
    TORCH_CHECK(kv_indptr.numel() == num_m_block + 1, "kv_indptr length mismatch");
    TORCH_CHECK(row_n_masked.numel() == num_m_block, "row_n_masked length mismatch");

    Sparse_fwd_params p{};
    p.q_ptr = q.data_ptr(); p.k_ptr = k.data_ptr(); p.v_ptr = v.data_ptr();
    p.o_ptr = out.data_ptr(); p.softmax_lse_ptr = lse.data_ptr();
    p.q_batch_stride = q.stride(0); p.q_row_stride = q.stride(1); p.q_head_stride = q.stride(2);
    p.o_batch_stride = out.stride(0); p.o_row_stride = out.stride(1); p.o_head_stride = out.stride(2);
    p.b = b; p.h = h; p.h_k = h_k;
    p.seqlen_q = seqlen_q; p.seqlen_k = seqlen_k; p.d = d;
    p.scale_softmax = (float)softmax_scale;
    p.kv_indptr    = kv_indptr.data_ptr<int32_t>();
    p.kv_indices   = kv_indices.data_ptr<int32_t>();
    p.kv_mask_nb   = kv_mask_nb.data_ptr<int32_t>();
    p.row_n_masked = row_n_masked.data_ptr<int32_t>();
    p.row_mask_mb  = row_mask_mb.data_ptr<int32_t>();
    p.work_coord   = work_coord.data_ptr<int32_t>();
    p.n_work       = (int)work_coord.numel();
    p.num_m_block  = num_m_block;

    static int *d_tile_count = nullptr;
    if (!d_tile_count) cudaMalloc(&d_tile_count, sizeof(int));
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaMemsetAsync(d_tile_count, 0, sizeof(int), stream);
    p.tile_counter = d_tile_count;

    if (kv64) ring_flash_attn_sparse64_launch(p, stream); else ring_flash_attn_sparse_launch(p, stream);
    return {out, lse};
}

std::vector<torch::Tensor> sparse_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor out, torch::Tensor lse,
                                      torch::Tensor kv_indptr, torch::Tensor kv_indices, torch::Tensor kv_mask_nb,
                                      torch::Tensor row_n_masked, torch::Tensor row_mask_mb, torch::Tensor work_coord,
                                      double softmax_scale) {
    return sparse_fwd_impl(false, q, k, v, out, lse, kv_indptr, kv_indices, kv_mask_nb, row_n_masked, row_mask_mb, work_coord, softmax_scale);
}

// 128 x 64 tiles: kv_indices = 64-row K / V tile ids (SpargeAttention2's block shape)
std::vector<torch::Tensor> sparse_fwd64(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor out, torch::Tensor lse,
                                        torch::Tensor kv_indptr, torch::Tensor kv_indices, torch::Tensor kv_mask_nb,
                                        torch::Tensor row_n_masked, torch::Tensor row_mask_mb, torch::Tensor work_coord,
                                        double softmax_scale) {
    return sparse_fwd_impl(true, q, k, v, out, lse, kv_indptr, kv_indices, kv_mask_nb, row_n_masked, row_mask_mb, work_coord, softmax_scale);
}

// ── Grow-only device scratch ─────────────────────────────────────────────────
struct DevScratch {
    void  *ptr = nullptr;
    size_t cap = 0;
    void *ensure(size_t bytes) {
        if (bytes > cap) {
            if (ptr) cudaFree(ptr);
            if (cudaMalloc(&ptr, bytes) != cudaSuccess)
                throw std::runtime_error("cudaMalloc failed for device scratch");
            cap = bytes;
        }
        return ptr;
    }
};

// ── Partial-output building blocks, exercised standalone by the numerics test ─
std::vector<torch::Tensor> sparse_fwd_partial(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor o_partial,     // (1, n_q*128, h, d) FP16
    torch::Tensor lse_partial,   // (1, h, n_q*128)    fp32
    torch::Tensor kv_indptr, torch::Tensor kv_indices, torch::Tensor kv_mask_nb,
    torch::Tensor row_n_masked, torch::Tensor row_mask_mb, torch::Tensor work_coord,
    double softmax_scale
) {
    const int b = q.size(0), sq = q.size(1), h = q.size(2), d = q.size(3);
    TORCH_CHECK(q.scalar_type() == torch::kFloat16, "q must be fp16");
    TORCH_CHECK(o_partial.scalar_type() == torch::kFloat16, "partial O must be fp16");
    TORCH_CHECK(d == 128 && sq % 128 == 0 && k.size(1) % 128 == 0, "128-tile geometry");

    Sparse_fwd_params p{};
    p.q_ptr = q.data_ptr(); p.k_ptr = k.data_ptr(); p.v_ptr = v.data_ptr();
    p.o_ptr = o_partial.data_ptr(); p.softmax_lse_ptr = lse_partial.data_ptr();
    p.q_batch_stride = q.stride(0); p.q_row_stride = q.stride(1); p.q_head_stride = q.stride(2);
    p.o_batch_stride = o_partial.stride(0); p.o_row_stride = o_partial.stride(1);
    p.o_head_stride = o_partial.stride(2);
    p.b = b; p.h = h; p.h_k = k.size(2);
    p.seqlen_q = sq; p.seqlen_k = k.size(1); p.d = d;
    p.seqlen_o = (int)o_partial.size(1);
    p.lse_head_stride = lse_partial.stride(1);
    p.scale_softmax = (float)softmax_scale;
    p.kv_indptr    = kv_indptr.data_ptr<int32_t>();
    p.kv_indices   = kv_indices.data_ptr<int32_t>();
    p.kv_mask_nb   = kv_mask_nb.data_ptr<int32_t>();
    p.row_n_masked = row_n_masked.data_ptr<int32_t>();
    p.row_mask_mb  = row_mask_mb.data_ptr<int32_t>();
    p.work_coord   = work_coord.data_ptr<int32_t>();
    p.n_work       = (int)work_coord.numel();
    p.num_m_block  = sq / 128;

    static int *d_tile = nullptr;
    if (!d_tile) cudaMalloc(&d_tile, sizeof(int));
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaMemsetAsync(d_tile, 0, sizeof(int), stream);
    p.tile_counter = d_tile;

    dist_partial_launch(p, stream);
    return {o_partial, lse_partial};
}

std::vector<torch::Tensor> merge_partials(
    torch::Tensor out,           // (1, n_owned*128, h, d) fp16
    torch::Tensor lse,           // (1, h, n_owned*128)    fp32
    torch::Tensor o_partial, torch::Tensor lse_partial,
    torch::Tensor merge_ptr, torch::Tensor merge_slot, torch::Tensor merge_out,
    int64_t max_slots            // ignored: the merge has no slot-count limit
) {
    TORCH_CHECK(out.scalar_type() == torch::kFloat16 && o_partial.scalar_type() == torch::kFloat16);
    Merge_params p{};
    p.o_ptr      = (__half *)out.data_ptr();
    p.lse_ptr    = lse.data_ptr<float>();
    p.part_o     = reinterpret_cast<const __half*>(o_partial.data_ptr());
    p.part_lse   = lse_partial.data_ptr<float>();
    p.merge_ptr  = merge_ptr.data_ptr<int32_t>();
    p.merge_slot = merge_slot.data_ptr<int32_t>();
    p.merge_out  = merge_out.data_ptr<int32_t>();
    p.n_merge    = (int)merge_out.numel();
    p.nheads     = (int)out.size(2);
    // The partial LSE is [h][tokens]; the final LSE is [h][tokens] too, but the
    // two token counts differ, so the merge needs both strides. They coincide
    // only when a PE owns exactly the rows it computes.
    p.lse_stride = (int)lse_partial.size(2);
    p.lse_out_stride = (int)lse.size(2);
    dist_merge_launch(p, (int)max_slots, at::cuda::getCurrentCUDAStream());
    return {out, lse};
}


// ── Distributed forward megakernel ──────────────────────────────────────────
// Symmetric block, sized from GLOBAL layout counts so a slot id means the same
// thing on every PE:
//   [ sig (NQ+NKV+NPART u64, 256B-aligned) | Q | K | V | O_partial | LSE_partial ]
void *bwd_region_base();          // dist_bwd_host.cpp
int64_t bwd_region_bytes();
uint64_t bwd_calls();

struct DistSym {
    void     *base = nullptr;
    uint64_t *sig  = nullptr;
    char *q = nullptr, *k = nullptr, *v = nullptr, *po = nullptr, *pl = nullptr;
    size_t cap = 0, sig_bytes = 0;
    // When the backward's symmetric region exists and is large enough, the data slots live there (the forward and
    // backward never run at once: one region of max(fwd, bwd) instead of both); the signal words stay in their own
    // allocation (never written by the backward, so the epoch rule holds); the caller barriers at the first forward
    // call after a backward call (no peer may push into a region a peer's backward still reads). Otherwise (no
    // backward region, e.g. forward-only benchmarks) the forward uses a private allocation.
    bool shared = false;
    void *own = nullptr, *ssig = nullptr;
    size_t own_cap = 0;

    void ensure(size_t q_b, size_t kv_b, size_t po_b, size_t pl_b, size_t n_sig) {
        size_t sb = ((n_sig * sizeof(uint64_t) + 255) / 256) * 256;
        size_t need = q_b + 2 * kv_b + po_b + pl_b;
        void *rb = bwd_region_base();
        if (rb && (int64_t)need <= bwd_region_bytes()) {        // the same sizes, so the same branch, on every PE
            if (!ssig || sb > sig_bytes) {
                if (ssig) nvshmem_free(ssig);
                sig_bytes = std::max(sig_bytes, sb);
                ssig = nvshmem_malloc(sig_bytes);
                if (!ssig) throw std::runtime_error("nvshmem_malloc failed for the dist signal words");
                cudaMemset(ssig, 0, sig_bytes);                  // once: epochs only grow
                cudaDeviceSynchronize();
                nvshmem_barrier_all();
            }
            if (own) { nvshmem_free(own); own = nullptr; own_cap = 0; }   // (collective as well)
            shared = true;
            base = rb; cap = (size_t)bwd_region_bytes();
            sig = reinterpret_cast<uint64_t *>(ssig);
            q = reinterpret_cast<char *>(rb);
            k = q + q_b; v = k + kv_b; po = v + kv_b; pl = po + po_b;
            return;
        }
        if (shared) { shared = false; base = own; cap = own_cap; }  // (the private allocation, regrown below if needed)
        if (need > cap || sb > sig_bytes || !own) {
            if (own) nvshmem_free(own);            // collective: same shapes everywhere
            cap = std::max(cap, need);
            sig_bytes = std::max(sig_bytes, sb);
            base = own = nvshmem_malloc(sig_bytes + cap); own_cap = cap;
            if (!base) throw std::runtime_error("nvshmem_malloc failed for the dist block");
            // Cleared once, here: a per-call clear would not be ordered against a peer's partial landing in one of
            // this PE's slots. Per call nothing needs clearing: every slot the merge reads is
            // either written by local compute (gated by o_head_done) or by a
            // sender (gated by its signal).
            cudaMemset(base, 0, sig_bytes + cap);
            nvshmem_barrier_all();
        }
        sig = reinterpret_cast<uint64_t *>(base);
        q  = reinterpret_cast<char *>(base) + sig_bytes;
        k  = q + q_b;
        v  = k + kv_b;
        po = v + kv_b;
        pl = po + po_b;
    }
};
static DistSym g_dist;
static uint64_t g_dist_epoch = 0;
static DevScratch g_dist_scratch;   // arrived[] + o_head_done[] + tile_counter
static DevScratch g_dist_pe;        // pe_delta[P] + pe_inter[P]
static DevScratch g_dist_pool;      // dist_sched_kernel: local pool of per-task partials (O fp32 + LSE fp32)
static DevScratch g_acc_ctl;        // bounded pool: per (row, head) lock + merged count
static DevScratch g_row_flag;       // dist_sched_kernel: per compute row, epoch of the call that wrote its partial
static DevScratch g_b0_state;       // dist_sched_kernel: block 0's state when it does not fit in shared memory
static DevScratch g_rect_list;      // dist_sched_kernel, rect: one KV list per block (kMaxRowTiles + 2 entries)

std::vector<torch::Tensor> dist_fwd(
    torch::Tensor q_local,       // (1, n_owned*128, h,   d) fp16, the rows it owns
    torch::Tensor k_local,       // (1, n_owned*128, h_k, d) fp16
    torch::Tensor v_local,
    torch::Tensor out,           // (1, n_owned*128, h, d) fp16, pre-allocated
    torch::Tensor lse,           // (1, h, n_owned*128)    fp32
    std::vector<torch::Tensor> tab,   // dex/scheduler: fwd.tables_blocks + launch.engine_tables, in order
    std::vector<int64_t> dims,        // NQ, NKV, NPART, n_compute, n_merge, n_remote
    double softmax_scale, int64_t num_push_blocks
) {
    TORCH_CHECK(g_nvshmem_initialized, "call init_nvshmem first");
    // the shared transport's tables (dims[17] = index of the first) come LAST: split them off, so every positional /
    // size-based probe below sees exactly the kernel's own table list
    std::vector<torch::Tensor> etab;
    if (dims.size() > 19 && dims[17] > 0) {
        TORCH_CHECK((size_t)dims[17] <= tab.size(), "engine tables: bad offset");
        etab.assign(tab.begin() + dims[17], tab.end()); tab.resize(dims[17]);
    }
    const int b = q_local.size(0), h = q_local.size(2), d = q_local.size(3);
    const int h_k = k_local.size(2);
    TORCH_CHECK(b == 1, "prefill-shaped: batch must be 1");
    TORCH_CHECK(q_local.scalar_type() == torch::kFloat16 && d == 128, "fp16, hdim 128");
    const int NQ = (int)dims[0], NKV = (int)dims[1], NPART = (int)dims[2];
    const int n_compute = (int)dims[3], n_merge = (int)dims[4];

    const size_t q_slot  = (size_t)128 * h   * d * 2;
    const size_t kv_slot = (size_t)128 * h_k * d * 2;
    const size_t po_slot = (size_t)128 * h   * d * 4;   // fp32 partial O
    const size_t pl_slot = (size_t)128 * 4;
    const size_t q_b  = (size_t)NQ    * q_slot;
    const size_t kv_b = (size_t)NKV   * kv_slot;
    // dims[11] (scheduler kernel): combined partial slots [O][LSE h x 128] -> one message per partial
    const bool combined = dims.size() > 11 && dims[11] > 0;
    const size_t comb_slot = po_slot + (size_t)h * 128 * 4;
    const size_t po_b = (size_t)NPART * (combined ? comb_slot : po_slot);
    const size_t pl_b = combined ? 0 : (size_t)h * NPART * 128 * 4;
    const int n_sig = NQ + NKV + NPART;
    g_dist.ensure(q_b, kv_b, po_b, pl_b, (size_t)n_sig);

    auto stream = at::cuda::getCurrentCUDAStream();
    // Staging: this PE's own Q / K / V blocks into its slots (one launch for the three). Every call reuses the same
    // symmetric slots; staging only writes this PE's own slots (peers read only their own), so it may run before the
    // cross-call credit check below.
    {
        auto sp = [](char *dst, const void *src, const torch::Tensor &si, const torch::Tensor &di, size_t sb) {
            Scatter_params p{};
            p.dst = dst; p.src = (const char *)src; p.n = (int)si.numel(); p.slot_bytes = (int64_t)sb;
            p.src_idx = p.n ? si.data_ptr<int32_t>() : nullptr; p.dst_idx = p.n ? di.data_ptr<int32_t>() : nullptr;
            return p;
        };
        dist_scatter3_launch(sp(g_dist.q, q_local.data_ptr(), tab[16], tab[17], q_slot),
                             sp(g_dist.k, k_local.data_ptr(), tab[18], tab[19], kv_slot),
                             sp(g_dist.v, v_local.data_ptr(), tab[18], tab[19], kv_slot), stream);
    }

    // intra vs inter is a property of the edge: nvshmem_ptr() returning non-null
    // IS the definition of "P2P-mapped here". It only changes when the symmetric
    // block is (re)allocated, so it is built once and not per call.
    const int n_pes = nvshmem_n_pes();
    char *pe_buf = (char *)g_dist_pe.ensure(n_pes * (sizeof(int64_t) + sizeof(int32_t)));
    int64_t *d_delta = (int64_t *)pe_buf;
    int32_t *d_inter = (int32_t *)(pe_buf + n_pes * sizeof(int64_t));
    static void *pe_for_base = nullptr;
    if (pe_for_base != g_dist.base) {
        std::vector<int64_t> delta(n_pes, 0);
        std::vector<int32_t> inter(n_pes, 0);
        for (int i = 0; i < n_pes; ++i) {
            if (i == nvshmem_my_pe()) continue;
            void *peer = nvshmem_ptr(g_dist.base, i);
            if (peer) delta[i] = (char *)peer - (char *)g_dist.base;
            else      inter[i] = 1;
        }
        cudaMemcpy(d_delta, delta.data(), n_pes * sizeof(int64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_inter, inter.data(), n_pes * sizeof(int32_t), cudaMemcpyHostToDevice);
        pe_for_base = g_dist.base;
    }

    // + 2 counters, then 16 u64 diagnostic counters (32 ints, 8-byte aligned; reserved)
    const size_t n_trace_off = (((size_t)n_sig + std::max(n_compute, 1) + 2) + 1) & ~(size_t)1;
    const size_t n_groups = (tab.size() > 42) ? (size_t)tab[39].numel() : 0;
    // dims[10] > 0: ready queue; tab[dims[10] .. +2] = dep_need, dep_ptr, dep_item
    const int ready_off = dims.size() > 10 ? (int)dims[10] : 0;
    const size_t n_cwork = (size_t)std::max(n_compute, 1) * h;
    // dims[11] > 0: block-0 task scheduler; tab[dims[11] .. +14] = its tables;
    // dims[12] = pool slots, dims[13] = low-water mark of the task queue (overridden below: thresholds
    // are in tile x head work units (q_work))
    const int sched_off = dims.size() > 11 ? (int)dims[11] : 0;
    // dims[14] > 0: rectangle segments (dist_params.h rect); tab[dims[14] .. +9] = row_brow, row_bi, row_seg_base,
    // brow_seg_ptr, seg_cb, seg_pair, seg_dep (int32), seg_mask, pair_rows, pair_drows (int64); dims[15] = tile rows per
    // block, dims[16] = (row, segment) pairs = the task bound
    const int rect_off = dims.size() > 16 ? (int)dims[14] : 0;
    const int n_grp = sched_off ? (rect_off ? (int)dims[16] : (int)tab[sched_off + 1].numel()) : 0;
    const int n_pool = sched_off ? (int)dims[12] : 0;
    const size_t sq_cap = sched_off ? ((size_t)n_grp + std::max(n_compute, 1) + n_merge) * h + 1024 : 0;
    const size_t n_sched = sched_off ? sq_cap + 1 + 4 * (size_t)n_grp + n_grp + 2 * (size_t)std::max(n_compute, 1)
                                       + std::max(n_merge, 1) + 2 + 4 + 1 + 1
                                       + (size_t)std::max(n_compute, 1) / 4 + 5 : 0;   // +4: int4 alignment, +1 q_all_in, +1 q_work, dx claims + counters
    // dims[17] > 0 (sched kernel): the shared communication engine; tab[dims[17]] = message table (int64 [n, 4] as int32
    // pairs), tab[dims[17] + 1] = trigger CSR (one trigger per compute row); dims[18] = start messages, dims[19] = rows
    // whose partial goes to another node (push-queue entries)
    const int eng_off = dims.size() > 19 ? (int)dims[17] : 0;      // (both kernels: the scheduler one and dist_fwd_kernel)
    const size_t n_eng = eng_off ? (size_t)dims[19] + 2 : 0;
    const size_t n_scratch_pre = n_trace_off + 32 + n_groups + 1    // + blocks_done (cross-call credit)
                           + (ready_off ? (size_t)std::max(n_compute, 1) + n_cwork + 1 : 0)    // dep_cnt, ready_q, tail
                           + n_sched;
    const size_t n_scratch = n_scratch_pre + n_eng;
    int *scratch = (int *)g_dist_scratch.ensure(n_scratch * sizeof(int));
    cudaMemsetAsync(scratch, 0, n_scratch * sizeof(int), stream);

    ++g_dist_epoch;
    // Cross-call ordering: every call reuses the same symmetric slots, and tile signals only order producer ->
    // consumer inside a call. Pairwise credits order consecutive calls: a PE waits only before writing into a PE
    // that may still be reading the previous call (kernel puts check in dist_put_block, copy-engine pushes via
    // cuStreamWaitValue64). Every branch here must be taken by all PEs alike (collectives).
    static uint64_t seen_bwd = 0;                           // shared region: a backward call ran after our last call ->
    const bool kind_switch = g_dist.shared && bwd_calls() != seen_bwd;   // peers may still read it until all are here
    seen_bwd = bwd_calls();
    if (kind_switch) nvshmemx_barrier_all_on_stream(stream);
    static uint64_t *g_credit = nullptr;
    static uint64_t g_call = 0;
    if (!g_credit) {                                         // collective: every PE gets here on its first call
        g_credit = (uint64_t *)nvshmem_malloc(n_pes * sizeof(uint64_t));
        cudaMemset(g_credit, 0, n_pes * sizeof(uint64_t));
        cudaDeviceSynchronize();
        nvshmem_barrier_all();
    }
    ++g_call;                                                // every PE counts every call

    Dist_fwd_params p{};
    p.q_slots = g_dist.q; p.k_slots = g_dist.k; p.v_slots = g_dist.v;
    p.part_o = g_dist.po;
    p.part_lse = combined ? (float *)((char *)g_dist.po + po_slot) : (float *)g_dist.pl;
    p.part_combined = combined ? 1 : 0;
    p.part_o_slot = combined ? (int64_t)(comb_slot / 4) : (int64_t)128 * h * d;   // in fp32 elements
    p.part_lse_slot = combined ? (int64_t)(comb_slot / 4) : (int64_t)128;
    p.o_ptr = out.data_ptr(); p.lse_ptr = lse.data_ptr();
    p.q_row_stride = (int64_t)h * d;   p.q_head_stride = d;
    p.o_row_stride = (int64_t)h * d;   p.o_head_stride = d;
    p.fo_row_stride = out.stride(1);   p.fo_head_stride = out.stride(2);
    p.lse_head_stride  = combined ? (int64_t)128 : (int64_t)NPART * 128;
    p.flse_head_stride = lse.size(2);
    p.b = b; p.h = h; p.h_k = h_k; p.d = d;
    p.seqlen_q = NQ * 128; p.seqlen_k = NKV * 128; p.seqlen_o = NPART * 128;
    p.scale_softmax = (float)softmax_scale;

    auto i32 = [](const torch::Tensor &t) {
        return t.numel() ? t.data_ptr<int32_t>() : nullptr;
    };
    p.kv_indptr = i32(tab[0]);  p.kv_indices = i32(tab[1]); p.kv_mask_nb = i32(tab[2]);
    p.row_n_masked = i32(tab[3]); p.row_mask_mb = i32(tab[4]); p.work_coord = i32(tab[5]);
    p.wait_ptr = i32(tab[6]);   p.wait_sig = i32(tab[7]);
    p.row_nremote = i32(tab[27]); p.row_qwait = i32(tab[28]);
    p.kv_ord_sig = tab[29].numel() ? i32(tab[29]) : nullptr;
    p.opush_pe = i32(tab[8]);   p.opush_slot = i32(tab[9]);
    p.merge_out = i32(tab[10]); p.merge_ptr = i32(tab[11]); p.merge_slot = i32(tab[12]);
    p.mwait_ptr = i32(tab[13]); p.mwait_sig = i32(tab[14]);
    p.merge_self = i32(tab[26]);
    p.remote_sig = i32(tab[15]);
    p.n_remote = (int)tab[15].numel();
    p.pushq_src = i32(tab[20]); p.pushq_pe = i32(tab[21]); p.pushq_dst = i32(tab[22]);
    p.pushkv_src = i32(tab[23]); p.pushkv_pe = i32(tab[24]); p.pushkv_dst = i32(tab[25]);
    p.n_pushq = (int)tab[21].numel(); p.n_pushkv = (int)tab[24].numel();
    p.num_push_blocks = (int)num_push_blocks;
    p.push_order = (tab.size() > 30 && tab[30].numel()) ? i32(tab[30]) : nullptr;
    p.item_q    = (tab.size() > 31 && tab[31].numel()) ? i32(tab[31]) : nullptr;
    p.mself_ptr = (tab.size() > 32 && tab[32].numel()) ? i32(tab[32]) : nullptr;
    p.mself_idx = (tab.size() > 33 && tab[33].numel()) ? i32(tab[33]) : nullptr;
    p.pushq_cnt  = (tab.size() > 37 && tab[36].numel()) ? i32(tab[36]) : nullptr;
    p.pushkv_cnt = (tab.size() > 37 && tab[37].numel()) ? i32(tab[37]) : nullptr;
    const bool grp = tab.size() > 42 && tab[38].numel();
    p.item_group     = grp ? i32(tab[38]) : nullptr;
    p.group_size     = grp ? i32(tab[39]) : nullptr;
    p.group_first    = grp ? i32(tab[40]) : nullptr;
    p.group_dst_pe   = grp ? i32(tab[41]) : nullptr;
    p.group_dst_slot = grp ? i32(tab[42]) : nullptr;
    p.group_done     = grp ? scratch + n_trace_off + 32 : nullptr;
    // dims[9]: global-order KV (dist_params.h). Positional ABI kept: tab[2] (kv_mask_nb, not needed when the
    // slot is the position) = item_direct, tab[7] (wait_sig, unused with the tile gate) = remote_cnt
    // (dex/scheduler/fwd.py).
    if (dims.size() > 9 && dims[9]) {
        p.item_direct = i32(tab[2]); p.remote_cnt = i32(tab[7]);
        p.kv_mask_nb = nullptr; p.wait_sig = nullptr;
    }
    p.push_block_base = dims.size() > 6 ? (int)dims[6] : 0;
    p.put_signal      = dims.size() > 7 ? (int)dims[7] : 0;
    p.sig_base = g_dist.sig; p.sig_epoch = g_dist_epoch;
    p.credit = g_credit;
    p.call_id = g_call;
    p.blocks_done = scratch + n_trace_off + 32 + n_groups;
    p.my_pe = nvshmem_my_pe(); p.n_pes = n_pes;
    p.arrived = scratch;
    p.o_head_done = scratch + n_sig;
    p.tile_counter = scratch + n_sig + std::max(n_compute, 1);
    p.quiet_gate   = scratch + n_sig + std::max(n_compute, 1) + 1;
    p.timeline = nullptr;                                    // (diagnostic buffers: null in normal runs)
    p.hs_pflag = nullptr;
    p.trace = nullptr;
    p.n_compute = n_compute;
    p.n_compute_work = (int)tab[5].numel();
    if (ready_off) {
        TORCH_CHECK((int)tab.size() >= ready_off + 3, "ready queue: dependency tables missing");
        p.dep_need = i32(tab[ready_off]); p.dep_ptr = i32(tab[ready_off + 1]); p.dep_item = i32(tab[ready_off + 2]);
        int *rq = scratch + n_trace_off + 32 + n_groups + 1;
        p.dep_cnt = rq; p.ready_q = rq + std::max(n_compute, 1); p.ready_tail = p.ready_q + n_cwork;
        p.n_init_work = (int)tab[5].numel();              // work_coord = the items with no remote input
        p.n_compute_work = n_compute * h;
    }
    p.n_work = p.n_compute_work + n_merge * h;
    p.n_merge = n_merge;
    if (sched_off) {
        TORCH_CHECK((int)tab.size() >= sched_off + 15, "sched: tables missing");
        const torch::Tensor *t = &tab[sched_off];
        p.row_grp_ptr = i32(t[0]); p.grp_beg = i32(t[1]); p.grp_end = i32(t[2]); p.grp_dep = i32(t[3]);
        p.row_qdep = i32(t[4]); p.sig_grp_ptr = i32(t[5]); p.sig_grp = i32(t[6]); p.sig_row_ptr = i32(t[7]);
        p.sig_row = i32(t[8]); p.sig_merge_ptr = i32(t[9]); p.sig_merge = i32(t[10]); p.row_order = i32(t[11]);
        p.row_pool = i32(t[12]); p.row_merge = i32(t[13]); p.merge_need = i32(t[14]);
        p.n_groups = n_grp;
        p.sched_low = dims.size() > 13 && dims[13] > 0 ? (int)dims[13] : 512;
        p.sched_batch = 32;
        p.sched_crit = 64;                                 // overridden below (tile x head work units, q_work)
        int *s = scratch + n_trace_off + 32 + n_groups + 1 + (ready_off ? (size_t)std::max(n_compute, 1) + n_cwork + 1 : 0);
        p.sq = s;                 s += sq_cap;
        p.sq_tail = s;            s += 1;
        s = (int *)(((uintptr_t)s + 15) & ~(uintptr_t)15);       // task descriptors are read as int4
        p.task_desc = s;          s += 4 * (size_t)n_grp;
        p.task_heads = s;         s += n_grp;
        p.row_tiles_done = s;     s += std::max(n_compute, 1);
        p.row_ntask = s;          s += std::max(n_compute, 1);
        p.merge_cnt = s;          s += std::max(n_merge, 1);
        p.rows_done = s;          s += 1;
        p.merges_enq = s;         s += 1;
        p.q_all_in = nullptr;                              // (slot reserved in the scratch layout)
        s += 1;
        p.q_work = s;                                      // the task queue is throttled by queued work
        if (rect_off) {
            const torch::Tensor *rt = &tab[rect_off];
            auto i64 = [](const torch::Tensor &t) {
                return t.numel() ? reinterpret_cast<const uint64_t *>(t.data_ptr<int64_t>()) : nullptr;
            };
            p.rect = 1; p.per = (int)dims[15]; p.n_rowseg = (int)dims[16];
            p.row_brow = i32(rt[0]); p.row_bi = i32(rt[1]); p.row_seg_base = i32(rt[2]); p.brow_seg_ptr = i32(rt[3]);
            p.seg_cb = i32(rt[4]); p.seg_pair = i32(rt[5]); p.seg_dep = i32(rt[6]);
            p.seg_mask = i64(rt[7]); p.pair_rows = i64(rt[8]); p.pair_drows = i64(rt[9]);
            p.n_segs = (int)rt[4].numel();
            p.rect_list = (int *)g_rect_list.ensure((size_t)1024 * (2048 + 2) * sizeof(int));   // <= 1024 blocks
        }
        {   // block 0's arrays (dist_sched_block0 layout): gst[ng] u8, qok[nr] u8, pad16, rnt[nr] + rem[nr] u16, pad16,
            // sdone[ns/32] u32, batch[sched_batch] int (rect: sready[n_segs] u8 + rst[n_rowseg] u8 in place of gst)
            const int64_t nr_ = std::max(n_compute, 1);
            const int64_t ng_ = rect_off ? (int64_t)std::max(p.n_segs, 1) + std::max(p.n_rowseg, 1) : std::max(n_grp, 1);
            p.b0_bytes = ((((ng_ + nr_ + 15) & ~15LL) + 4 * nr_ + 15) & ~15LL) + 4 * ((int64_t)p.n_remote / 32 + 1)
                         + 4 * (int64_t)std::max(p.sched_batch, 64) + 64;
            p.b0_state = (char *)g_b0_state.ensure((size_t)p.b0_bytes);
        }
        s += 1;
        // dims[23] > 0: block 0 expands the KV lists (dist_params.h dx_on); tab[dims[23] .. +4] = segment first column,
        // mask low / high words, local flag, row -> segments
        const int dx_off = dims.size() > 23 ? (int)dims[23] : 0;
        p.dx_bits = reinterpret_cast<unsigned *>(s); s += std::max(n_compute, 1) / 4 + 1;   // per claim of 4 rows: written
        p.dx_next = s; s += 3;                                 // [0] claim counter, [1] claims published, [2] rows published
        if (dx_off && !rect_off) {
            TORCH_CHECK((int)tab.size() >= dx_off + 5, "dx: segment tables missing");
            p.dx_on = 1; p.dx_seg_cb = i32(tab[dx_off]); p.dx_seg_lo = i32(tab[dx_off + 1]);
            p.dx_seg_hi = i32(tab[dx_off + 2]); p.dx_seg_loc = i32(tab[dx_off + 3]); p.dx_row_seg_ptr = i32(tab[dx_off + 4]);
            if (dims.size() > 25 && dims[25]) p.dx_kslot = i32(tab[dx_off + 5]);   // compact KV
            p.kv_half = dims.size() > 26 ? (int)dims[26] : 0;  // 128 x 64 compute lists (needs compact KV)
            TORCH_CHECK(!p.kv_half || p.dx_kslot, "kv_half needs compact KV slots");
        }
        // dims[24] > 0: compact KV slots -- tab[dims[24]] = the mask geometry list (else the KV list itself)
        p.kv_geom = (dims.size() > 24 && dims[24]) ? i32(tab[dims[24]]) : p.kv_indices;
        // queue thresholds in tile x head work units
        p.sched_low = 16384; p.sched_crit = 2048;
        // tab[sched_off + 15]: CPU int32 [n, 3] intra-node partial returns by copy engine (row, owner, slot)
        if ((int)tab.size() > sched_off + 15 && tab[sched_off + 15].numel() > 0) {
            static size_t flag_cap = 0;
            const size_t need = (size_t)std::max(n_compute, 1) * sizeof(unsigned long long);
            if (need > flag_cap) {                   // epochs only grow, so a fresh zeroed buffer is always behind
                p.row_flag = (unsigned long long *)g_row_flag.ensure(need);
                cudaMemset(p.row_flag, 0, need);
                flag_cap = need;
            }
            p.row_flag = (unsigned long long *)g_row_flag.ptr;
        }
        // tab[sched_off + 16]: CPU int32 [n, 7] relay forwards out of this PE (dst, kind 0 Q / 1 KV, src slot, dst slot,
        // count, dst signal id, wait = this PE's arrival signal id)
        if ((int)tab.size() > sched_off + 16 && tab[sched_off + 16].numel() > 0) {
            static unsigned long long *fw_word = nullptr;
            if (!fw_word) { cudaMalloc(&fw_word, sizeof(unsigned long long)); cudaMemset(fw_word, 0, sizeof(unsigned long long)); }
            p.fw_done = fw_word;
        }
        // bounded pool: one accumulator per row + one scratch slot per worker (dist_acc_merge), so the pool does not
        // grow with the number of tasks a row may be split into
        int n_pool_eff = n_pool;
        {
            int dev_ = 0, n_sm_ = 0;
            cudaGetDevice(&dev_); cudaDeviceGetAttribute(&n_sm_, cudaDevAttrMultiProcessorCount, dev_);
            p.pool_acc = 1; p.pool_scratch0 = std::max(n_compute, 1); p.pool_nws = 2 * n_sm_;   // >= the workers of one wave
            n_pool_eff = p.pool_scratch0 + p.pool_nws;
            const size_t nk = (size_t)std::max(n_compute, 1) * h;
            int *acc = (int *)g_acc_ctl.ensure(2 * nk * sizeof(int));
            cudaMemsetAsync(acc, 0, 2 * nk * sizeof(int), stream);
            p.acc_lock = acc; p.acc_cnt = acc + nk;
        }
        const size_t pool_o_b = (size_t)std::max(n_pool_eff, 1) * 128 * h * d * 4;   // fp32 partial O
        char *pool = (char *)g_dist_pool.ensure(pool_o_b + (size_t)h * std::max(n_pool_eff, 1) * 128 * 4);
        p.pool_o = pool; p.pool_lse = (float *)(pool + pool_o_b);
        p.pool_lse_stride = (int64_t)std::max(n_pool_eff, 1) * 128;
    }
    p.pe_delta = d_delta; p.pe_inter = d_inter;
    p.use_engine = eng_off ? (dims.size() > 20 && dims[20] >= 2 ? (int)dims[20] : 1) : 0;
    if (eng_off) {
        p.eng_grp_trig = dims.size() > 22 ? (int)dims[22] : 0;
        p.eng.base = g_dist.q; p.eng.sig = g_dist.sig; p.eng.epoch = g_dist_epoch;
        p.eng.msg = reinterpret_cast<const rt::Msg *>(etab[0].data_ptr<int32_t>());
        p.eng.trig_ptr = etab[1].data_ptr<int32_t>();
        p.eng.n_start = (int)dims[18]; p.eng.n_pq = (int)dims[19];
        p.eng.pq = scratch + n_scratch_pre; p.eng.pq_tail = p.eng.pq + dims[19] + 1;
        p.eng.pe_delta = d_delta; p.eng.pe_inter = d_inter;
        p.eng.credit = p.credit; p.eng.call_id = p.call_id;
        p.eng.trig_first = 2;                                   // 2 = input forwards fire before partial outputs
        p.eng.in_trig_first = -1;                               // (fwd mode 3: relay forwards go by copy engine)
        p.num_push_blocks = 0;                                  // no push block: block 1 computes from the start
        if (p.use_engine == 2) { p.row_flag = nullptr; p.fw_done = nullptr; }   // no copy-engine traffic at all
        if (p.use_engine == 3) {   // intra-node by the shared copy-engine executor: tab[eng_off + 2] = trigger lanes,
            p.row_flag = nullptr;  // tab[eng_off + 3] = its messages (CPU int64 [n, 6], rt::CeMsg)
            p.eng.trig_lane = etab[2].data_ptr<int32_t>();
            static unsigned long long *flags = nullptr; static int64_t fcap = 0;
            const int64_t need = std::max<int64_t>(etab[2].numel(), 1);
            if (need > fcap) {     // epoch-valued, zeroed once (epochs only grow)
                if (flags) cudaFree(flags);
                cudaMalloc(&flags, need * 8); cudaMemset(flags, 0, need * 8); fcap = need;
            }
            p.eng.trig_flag = flags;
        }
    }
    p.NQ = NQ; p.NKV = NKV; p.NPART = NPART;
    // the shared block-0 scheduler (runtime/rt_sched.cuh)
    // rectangle segments: dims[21] = tab index of the scheduler's group tables (fwd.tables_blocks):
    // row_grp_ptr, grp_beg, grp_end, grp_dep, grp_unit, grp_work -- a group = a (row, non-empty segment) pair
    const int rgrp_off = (p.rect && dims.size() > 21) ? (int)dims[21] : 0;
    p.sp_on = sched_off && (!p.rect || rgrp_off);
    TORCH_CHECK(!p.dx_on || p.sp_on, "dx (block-0 KV-list expansion) needs the shared scheduler");
    if (p.sp_on) {
        rt::SchedParams &s = p.sp;
        s.sig = p.sig_base; s.epoch = p.sig_epoch; s.n_remote = p.n_remote; s.remote_sig = p.remote_sig;
        s.remote_cnt = p.remote_cnt; s.arrived = p.arrived;
        s.n_rows = p.n_compute; s.n_groups = p.n_groups; s.row_grp_ptr = p.row_grp_ptr; s.grp_beg = p.grp_beg;
        s.grp_end = p.grp_end; s.grp_dep = p.grp_dep; s.grp_need = nullptr; s.grp_cnt = nullptr; s.row_qdep = p.row_qdep;
        s.row_order = p.row_order; s.row_pool = p.row_pool;
        s.sig_grp_ptr = p.sig_grp_ptr; s.sig_grp = p.sig_grp; s.sig_row_ptr = p.sig_row_ptr; s.sig_row = p.sig_row;
        s.n_red = p.n_merge; s.red_need = p.merge_need; s.red_kind = nullptr; s.kind_red = 2 /* kSqOwnerMerge */;
        s.red_n_all = p.h; s.sig_red_ptr = p.sig_merge_ptr; s.sig_red = p.sig_merge; s.red_cnt = p.merge_cnt;
        s.n_enq = p.merges_enq; s.rows_done = p.rows_done;
        s.sq = p.sq; s.sq_tail = p.sq_tail; s.tile_counter = p.tile_counter; s.task_desc = p.task_desc;
        s.row_ntask = p.row_ntask; s.sub = p.h; s.gang = 1; s.kind_task = 0; s.kind_stop = 3; s.kind_nop = 7;
        s.sched_low = p.sched_low; s.sched_crit = p.sched_crit; s.sched_batch = p.sched_batch; s.q_work = p.q_work;
        s.fwd_trig = p.use_engine == 2 ? p.n_compute : -1;
        if (rgrp_off) {                            // rectangle segments: groups = (row, segment), units = segments
            s.row_grp_ptr = i32(tab[rgrp_off]); s.grp_beg = i32(tab[rgrp_off + 1]); s.grp_end = i32(tab[rgrp_off + 2]);
            s.grp_dep = i32(tab[rgrp_off + 3]); s.grp_unit = i32(tab[rgrp_off + 4]); s.grp_work = i32(tab[rgrp_off + 5]);
            s.n_groups = (int)tab[rgrp_off + 1].numel(); s.n_units = p.n_segs; s.unit_dep = p.seg_dep;
        }
        s.trace = p.trace; s.q_all_in = p.q_all_in; s.q_lim = NQ;
        if (p.dx_on) s.row_avail = p.dx_next + 2;     // (rt::sched_block0<2>)
        // block 0's arrays: shared memory when they fit (launcher), else this global buffer
        s.b0_bytes = rt::sched_b0_bytes(s.n_groups, s.n_rows, s.n_remote, s.sched_batch, s.grp_unit ? s.n_units : 0);
        s.b0_state = (char *)g_b0_state.ensure((size_t)std::max<int64_t>(s.b0_bytes, p.b0_bytes));
        p.b0_state = s.b0_state;                   // (n_static, n_workers: the launcher, from the grid size)
    }
    p.q_slot_bytes = (int64_t)q_slot; p.kv_slot_bytes = (int64_t)kv_slot;
    p.o_slot_bytes = (int64_t)(combined ? comb_slot : po_slot); p.lse_slot_bytes = (int64_t)pl_slot;

    // ---- copy-engine pushes (intra-node), concurrent with the kernel -------------------------- #
    // tab[34]: CPU int32 [n_runs, 7] = (dst_pe, kind 0=Q 1=KV, src owned index, dst slot, count, sig id,
    // chunk); a run is contiguous at both ends. Everything here runs on COPY ENGINES: the persistent
    // kernel holds every SM and spins, so a signal written by a kernel on a side stream would never run.
    // The signal is an 8-byte CE copy of this call's epoch into the peer's signal word,
    // stream-ordered after the run's data.
    static std::vector<cudaStream_t> ce_streams;
    static std::vector<cudaEvent_t> ce_done;
    static cudaEvent_t ce_go = nullptr;
    static uint64_t *ce_epoch_dev = nullptr;
    const bool ce_part = sched_off && p.row_flag != nullptr;  // intra-node partials ride the copy engines too
    const bool use_ce = p.use_engine < 2 && ((tab.size() > 35 && tab[34].numel() > 0) || ce_part || p.fw_done != nullptr);
    std::vector<int> ce_used;
    if (use_ce) {
        if ((int)ce_streams.size() < n_pes) {
            ce_streams.resize(n_pes, nullptr); ce_done.resize(n_pes, nullptr);
            for (int i = 0; i < n_pes; ++i) {
                cudaStreamCreateWithFlags(&ce_streams[i], cudaStreamNonBlocking);
                cudaEventCreateWithFlags(&ce_done[i], cudaEventDisableTiming);
            }
            cudaEventCreateWithFlags(&ce_go, cudaEventDisableTiming);
            cudaMalloc(&ce_epoch_dev, sizeof(uint64_t));
        }
        const uint64_t ep = g_dist_epoch;
        cudaMemcpyAsync(ce_epoch_dev, &ep, sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
        cudaEventRecord(ce_go, stream);                       // staging, scratch clear, epoch are done
        dist_fwd_launch(p, stream);                           // kernel first: copies are enqueued behind it
        const int32_t *R = tab[34].data_ptr<int32_t>();
        const int nr = (int)tab[34].size(0);
        std::vector<char> started(n_pes, 0);
        uint64_t *sig_peer_cache = nullptr; int sig_peer_pe = -1;
        for (int r = 0; r < nr; ++r) {
            const int32_t *e = R + 7 * r;
            const int dst = e[0], kind = e[1], src = e[2], dslot = e[3], cnt = e[4], sid = e[5];
            cudaStream_t cs = ce_streams[dst];
            if (!started[dst]) {
                cudaStreamWaitEvent(cs, ce_go, 0); started[dst] = 1; ce_used.push_back(dst);
                if (p.credit)                                 // dst must have finished the previous call
                    cuStreamWaitValue64(cs, (CUdeviceptr)(p.credit + dst), (cuuint64_t)(p.call_id - 1),
                                        CU_STREAM_WAIT_VALUE_GEQ);
            }
            // kind 0/1: src = owned index into the local tensors; kind 2/3: src = a slot of this PE's
            // symmetric region (the planner's send region, staged in chunk order -> one copy per chunk)
            const bool from_slot = kind >= 2;
            if ((kind & 1) == 0) {
                char *d = (char *)nvshmem_ptr(g_dist.q, dst) + (size_t)dslot * q_slot;
                const char *sq = from_slot ? g_dist.q : (char *)q_local.data_ptr();
                cudaMemcpyAsync(d, sq + (size_t)src * q_slot, (size_t)cnt * q_slot, cudaMemcpyDeviceToDevice, cs);
            } else {
                char *dk = (char *)nvshmem_ptr(g_dist.k, dst) + (size_t)dslot * kv_slot;
                char *dv = (char *)nvshmem_ptr(g_dist.v, dst) + (size_t)dslot * kv_slot;
                const char *sk = from_slot ? g_dist.k : (char *)k_local.data_ptr();
                const char *sv = from_slot ? g_dist.v : (char *)v_local.data_ptr();
                cudaMemcpyAsync(dk, sk + (size_t)src * kv_slot, (size_t)cnt * kv_slot, cudaMemcpyDeviceToDevice, cs);
                cudaMemcpyAsync(dv, sv + (size_t)src * kv_slot, (size_t)cnt * kv_slot, cudaMemcpyDeviceToDevice, cs);
            }
            if (sig_peer_pe != dst) { sig_peer_cache = (uint64_t *)nvshmem_ptr(g_dist.sig, dst); sig_peer_pe = dst; }
            cudaMemcpyAsync(sig_peer_cache + sid, ce_epoch_dev, sizeof(uint64_t), cudaMemcpyDeviceToDevice, cs);
        }
        if (p.fw_done) {
            // relay forwards: ONE stream (all destinations, in plan order); each waits for its block's arrival here,
            // and, at its destination's first use, for that destination's credit (it finished the previous call)
            static cudaStream_t fw_stream = nullptr;
            static cudaEvent_t fw_ev = nullptr;
            if (!fw_stream) {
                cudaStreamCreateWithFlags(&fw_stream, cudaStreamNonBlocking);
                cudaEventCreateWithFlags(&fw_ev, cudaEventDisableTiming);
            }
            cudaStreamWaitEvent(fw_stream, ce_go, 0);
            const torch::Tensor &T = tab[sched_off + 16];
            const int32_t *E = T.data_ptr<int32_t>();
            std::vector<char> fw_cred(n_pes, 0);
            for (int r = 0; r < (int)T.size(0); ++r) {
                const int32_t *e = E + 7 * r;
                const int dst = e[0], kind = e[1], src = e[2], dslot = e[3], cnt = e[4], sid = e[5], wsig = e[6];
                if (!fw_cred[dst] && p.credit) {
                    cuStreamWaitValue64(fw_stream, (CUdeviceptr)(p.credit + dst), (cuuint64_t)(p.call_id - 1),
                                        CU_STREAM_WAIT_VALUE_GEQ);
                    fw_cred[dst] = 1;
                }
                cuStreamWaitValue64(fw_stream, (CUdeviceptr)(g_dist.sig + wsig), (cuuint64_t)g_dist_epoch,
                                    CU_STREAM_WAIT_VALUE_GEQ);
                if (kind == 0) {
                    cudaMemcpyAsync((char *)nvshmem_ptr(g_dist.q, dst) + (size_t)dslot * q_slot,
                                    g_dist.q + (size_t)src * q_slot, (size_t)cnt * q_slot, cudaMemcpyDeviceToDevice,
                                    fw_stream);
                } else {
                    cudaMemcpyAsync((char *)nvshmem_ptr(g_dist.k, dst) + (size_t)dslot * kv_slot,
                                    g_dist.k + (size_t)src * kv_slot, (size_t)cnt * kv_slot, cudaMemcpyDeviceToDevice,
                                    fw_stream);
                    cudaMemcpyAsync((char *)nvshmem_ptr(g_dist.v, dst) + (size_t)dslot * kv_slot,
                                    g_dist.v + (size_t)src * kv_slot, (size_t)cnt * kv_slot, cudaMemcpyDeviceToDevice,
                                    fw_stream);
                }
                cudaMemcpyAsync((uint64_t *)nvshmem_ptr(g_dist.sig, dst) + sid, ce_epoch_dev, sizeof(uint64_t),
                                cudaMemcpyDeviceToDevice, fw_stream);
            }
            cudaMemcpyAsync(p.fw_done, ce_epoch_dev, sizeof(uint64_t), cudaMemcpyDeviceToDevice, fw_stream);
            cudaEventRecord(fw_ev, fw_stream);
            cudaStreamWaitEvent(stream, fw_ev, 0);                 // next call's staging waits for the reads
        }
        if (ce_part) {
            // each intra-node partial: wait until the kernel marks the row written (this call's epoch), copy the
            // whole slot (O + LSE, contiguous) into the owner's landing slot, signal it -- after the Q/KV runs on
            // the same per-owner stream, so the credit wait above covers these writes too
            const torch::Tensor &T = tab[sched_off + 15];
            const int32_t *E = T.data_ptr<int32_t>();
            for (int r = 0; r < (int)T.size(0); ++r) {
                const int row = E[3 * r], dst = E[3 * r + 1], dslot = E[3 * r + 2];
                cudaStream_t cs = ce_streams[dst];
                if (!started[dst]) {
                    cudaStreamWaitEvent(cs, ce_go, 0); started[dst] = 1; ce_used.push_back(dst);
                    if (p.credit)
                        cuStreamWaitValue64(cs, (CUdeviceptr)(p.credit + dst), (cuuint64_t)(p.call_id - 1),
                                            CU_STREAM_WAIT_VALUE_GEQ);
                }
                cuStreamWaitValue64(cs, (CUdeviceptr)(p.row_flag + row), (cuuint64_t)g_dist_epoch,
                                    CU_STREAM_WAIT_VALUE_GEQ);
                char *d = (char *)nvshmem_ptr(g_dist.po, dst) + (size_t)dslot * (size_t)p.o_slot_bytes;
                cudaMemcpyAsync(d, (char *)g_dist.po + (size_t)row * (size_t)p.o_slot_bytes, (size_t)p.o_slot_bytes,
                                cudaMemcpyDeviceToDevice, cs);
                if (sig_peer_pe != dst) { sig_peer_cache = (uint64_t *)nvshmem_ptr(g_dist.sig, dst); sig_peer_pe = dst; }
                cudaMemcpyAsync(sig_peer_cache + (NQ + NKV + dslot), ce_epoch_dev, sizeof(uint64_t),
                                cudaMemcpyDeviceToDevice, cs);
            }
        }
    }

    if (p.use_engine == 3) {                                  // the shared copy-engine executor (runtime/rt_host.h)
        static rt::CeExec cex;
        const torch::Tensor &ct = etab[3];
        const int n_ce = (int)(ct.numel() / 6);       // rt::CeMsg rows (int64 [n, 6])
        if (n_ce) { cex.ensure(n_pes); cex.begin(stream, g_dist_epoch); }
        dist_fwd_launch(p, stream);
        if (n_ce) {
            static std::vector<char *> pb; static std::vector<uint64_t *> ps;
            pb.assign(n_pes, nullptr); ps.assign(n_pes, nullptr);
            for (int q = 0; q < n_pes; ++q) { pb[q] = (char *)nvshmem_ptr(g_dist.q, q); ps[q] = (uint64_t *)nvshmem_ptr(g_dist.sig, q); }
            cex.enqueue(stream, reinterpret_cast<const rt::CeMsg *>(ct.data_ptr<int64_t>()), n_ce, g_dist.q, pb, ps,
                        g_dist.sig, p.eng.trig_flag, g_dist_epoch, p.credit, p.call_id, p.fw_done);
        }
    } else if (!use_ce) dist_fwd_launch(p, stream);
    for (int dst : ce_used) {                                 // next call may reuse peers' slots
        cudaEventRecord(ce_done[dst], ce_streams[dst]);
        cudaStreamWaitEvent(stream, ce_done[dst], 0);
    }
    return {out, lse};
}

void register_sparse_bwd(pybind11::module &m);   // csrc/dist_bwd (backward)
void register_dist_bwd(pybind11::module &m);
void register_infllm_bwd(pybind11::module &m);
void register_infllm_fwd(pybind11::module &m);
#ifdef DEX_WITH_INFLLM2D
void register_infllm2d_fwd(pybind11::module &m);  // csrc/infllm2d (BF16 build only)
void register_infllm2d_bwd(pybind11::module &m);
#endif
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    register_sparse_bwd(m);
    register_dist_bwd(m);
    register_infllm_bwd(m);
    register_infllm_fwd(m);
#ifdef DEX_WITH_INFLLM2D
    register_infllm2d_fwd(m);
    register_infllm2d_bwd(m);
#endif
    m.def("init_nvshmem", &ring_attn_init_nvshmem, "Bootstrap NVSHMEM from a torch.distributed ProcessGroup");
    m.def("finalize_nvshmem", &ring_attn_finalize_nvshmem, "Finalize NVSHMEM");
    m.def("sparse_fwd", &sparse_fwd, "Block-sparse persistent-kernel forward (single GPU)");
    m.def("sparse_fwd64", &sparse_fwd64, "Block-sparse forward, 128 x 64 tiles (single GPU)");
    m.def("sparse_fwd_partial", &sparse_fwd_partial, "Block-sparse forward emitting partial O + LSE");
    m.def("merge_partials", &merge_partials, "Log-sum-exp merge of per-device partial outputs");
    m.def("dist_fwd", &dist_fwd, "Distributed sparse forward megakernel: one launch runs a plan's transfers and compute");
}
