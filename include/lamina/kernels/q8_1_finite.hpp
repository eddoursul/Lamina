// include/lamina/kernels/q8_1_finite.hpp - #606: the q8_1 activation block's scale and sum, kept finite.
//
// A q8_1 block stores its scale d = amax / 127 and the sum of its 32 raw activations as fp16. A sum past 65504 (one
// massive-activation dimension among 32 values is enough) rounds to inf, and d does past amax = 8.3M; the dot
// products then read inf * 0 = NaN, and a NaN in the residual ends as one token repeated (ggml-org/llama.cpp#23606
// is the same defect). Clamped to the largest finite half, such a block gives an approximate product instead of NaN,
// and the values are held to the int8 range they then exceed. Every block that was finite before is stored bit for
// bit as before: values in (65504, 65520) round to 65504 anyway, and a finite block's |x / d| is at most 127. A NaN
// stays NaN (the comparisons are false for it), so LAMINA_DBG_NAN still sees one.
#pragma once

#include <cuda_fp16.h>

namespace lamina::kernels {

constexpr float kHalfMax = 65504.0f;

__device__ __forceinline__ float q8_1_finite(const float v) { return fabsf(v) > kHalfMax ? copysignf(kHalfMax, v) : v; }

/// The block's quantized value of `xi` for the (finite) scale `d`; `amax` = 0 is an all-zero block.
__device__ __forceinline__ int8_t q8_1_quant(const float xi, const float d, const float amax) {
    if (amax == 0.0f) return 0;
    const float q = roundf(xi / d);
    return (int8_t) (q > 127.0f ? 127.0f : q < -127.0f ? -127.0f : q);
}

__device__ __forceinline__ half2 q8_1_ds(const float d, const float sum) {
    return make_half2(__float2half(q8_1_finite(d)), __float2half(q8_1_finite(sum)));
}

/// A warp's 32 values (each lane's `x`) as one q8_1 block at at() (a callable: the block's address),
/// quantize_q8_1_kernel's way: the block's absolute maximum and sum in one XOR butterfly, the scale amax / 127 kept
/// finite; the lane's code into qs[slot], the scale and the sum by the lane `first` is true for.  Every lane of the
/// warp calls it; the division is the calling file's (its compile mode).
template <class At>
__device__ __forceinline__ void q8_1_warp_block(const float x, At at, const int slot, const bool first) {
    float amax = fabsf(x), sum = x;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    const float d = q8_1_finite(amax / 127.0f);
    const int8_t q = q8_1_quant(x, d, amax);
    auto* b = at();   // (the block's address after the code, where the copies this replaces computed it)
    b->qs[slot] = q;
    if (first) b->ds = q8_1_ds(d, sum);
}

}   // namespace lamina::kernels
