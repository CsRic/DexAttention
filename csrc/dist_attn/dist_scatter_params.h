#pragma once
#include <cstdint>
namespace ring_attn {
// Staging descriptor: copy `n` equal-size slots from src[src_idx[i]] to
// dst[dst_idx[i]]. Shared between the host TU and the device code.
struct Scatter_params {
    char *dst;
    const char *src;
    const int32_t *src_idx;
    const int32_t *dst_idx;
    int n;
    int64_t slot_bytes;
};
}  // namespace ring_attn
