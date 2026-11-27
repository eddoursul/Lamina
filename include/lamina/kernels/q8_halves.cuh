// include/lamina/kernels/q8_halves.cuh - q8_0 codes as the halves a tensor-core tile reads (K/V cells kept in q8_0).
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace lamina::kernels {

/// Four int8 codes of a q8_0 block (a word) times their block's scale d2 (its FP16 scale twice), as two half2: each
/// code made a half exactly (the byte with its sign bit flipped permuted under 0x64, 1024 + q + 128, less 1152), then
/// one rounding of d * q to FP16, as a float product rounded to FP16 would give.
__device__ __forceinline__ void q8_widen4(uint32_t u, __half2 d2, __half2& lo, __half2& hi) {
    const uint32_t x = u ^ 0x80808080u;
    const uint32_t a = __byte_perm(x, 0x64646464u, 0x4140), b = __byte_perm(x, 0x64646464u, 0x4342);
    const __half2 bias = __float2half2_rn(1152.f);
    lo = __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&a), bias), d2);
    hi = __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&b), bias), d2);
}

}   // namespace lamina::kernels
