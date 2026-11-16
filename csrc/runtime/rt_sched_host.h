#pragma once
// Host helpers of the shared block-0 scheduler (rt_sched.cuh).
#include <algorithm>
#include <cstdint>

namespace rt {

// bytes of block 0's per-group / per-row arrays (rt_sched.cuh sched_block0 layout): gst[ng] u8, qok[nr] u8, ust[nu] u8, pad16,
// rnt[nr] + rem[nr] u16, pad16, sdone[ns / 32] u32, batch[sched_batch] int
inline int64_t sched_b0_bytes(int64_t ng, int64_t nr, int64_t ns, int64_t batch, int64_t nu = 0) {
    ng = std::max<int64_t>(ng, 1); nr = std::max<int64_t>(nr, 1);
    return ((((ng + nr + nu + 15) & ~15LL) + 4 * nr + 15) & ~15LL) + 4 * (ns / 32 + 1) + 4 * std::max<int64_t>(batch, 64) + 64;
}

}  // namespace rt
