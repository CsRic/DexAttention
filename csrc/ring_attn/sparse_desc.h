#pragma once
// Block-sparse descriptor consumed by the mainloop.
//
// The planner is block granular:
// one mask cell = one attention tile of `block_tokens` on a side. The kernel's
// tile is kBlockM x kBlockN (128x128 here), so the host expands every plan block
// into kernel tiles and emits, per Q tile row, the list of KV tiles that row
// actually attends to.
//
// Layout, per (row = m_block):
//   kv_indices[indptr[row] .. indptr[row+1])   KV tile ids, masked ones LAST.
//                                              Single-GPU: ascending. dist: the
//                                              planner's order (remote, local,
//                                              masked).
//   n_masked[row]                              how many TRAILING entries of that
//                                              range need elementwise (causal)
//                                              masking
//
// The order is not cosmetic. FA2/FA3's mainloop walks KV downward and applies
// its masked iterations first, so the loop consumes this list back-to-front:
// the LAST stored entries are the FIRST executed, which is why the diagonal
// tiles live at the end and the mask phase boundary is `n_blocks - n_masked`.
// Everything else in the row is a FULL tile and skips the mask entirely.

#include <cstdint>

namespace flash {

// `kv_indices` says WHERE the tile lives (an index into whatever KV buffer the
// kernel was given); `mask_nb` / `mask_m` say WHERE IT IS in the sequence, which
// is what the causal mask needs. On one GPU the two coincide. Across GPUs the KV
// buffer is a compacted per-PE slot region, so the storage index carries no
// positional meaning and the host supplies the geometry separately -- a pair
// whose DIFFERENCE reproduces the diagonal, because the mask only ever reads
// `n * kBlockN - m * kBlockM` plus the two seqlens.
struct SparseRowDesc {
    const int32_t *kv_indices;   // this row's KV tile ids (storage order; masked tail last)
    const int32_t *mask_nb;      // parallel to kv_indices; only the masked tail is read
    int32_t        n_blocks;     // length of the list
    int32_t        n_masked;     // trailing entries needing the causal mask
    int32_t        mask_m;       // the m_block to hand the mask
    // Ordinal window [ord_lo, ord_hi) this call should run; ord_hi == 0 means
    // "the whole row", which is what every caller that does not split passes.
    int32_t        ord_lo;
    int32_t        ord_hi;
    // TILE-GRANULARITY ARRIVAL GATE. `ord_sig` is parallel to kv_indices and
    // holds the signal id a tile has to wait on, or -1 for one already resident.
    // The wait happens where that tile's cp.async is ISSUED, so the row runs as
    // ONE mainloop call: Q is loaded once, the software pipeline is filled once,
    // and a tile that has not landed yet stalls only the stage that needs it.
    // Splitting the row into two mma() calls would cost a second global Q load
    // and a second pipeline prologue.
    const int32_t *ord_sig;      // null -> nothing in this row waits
    const int     *arrived;      // signal id -> nonzero once the tile has landed
};

}  // namespace flash
