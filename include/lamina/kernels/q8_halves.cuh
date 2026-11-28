// include/lamina/kernels/q8_halves.cuh - q8_0 codes as the halves a tensor-core tile reads (K/V cells kept in q8_0).
#pragma once

#include "lamina/kernels/warp.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace lamina::kernels {

/// A warp's 32 values (each lane's `v`) as a q8_0 block, as llama.cpp rounds it: the scale d = the block's amax / 127
/// (into d), the lane's code round(v / d) through 1 / d (0 for an all-zero block).  The division is the calling file's.
__device__ __forceinline__ int8_t q8_0_code(float v, float& d) {
    d = warp_max(fabsf(v)) / 127.f;
    const float id = d != 0.f ? 1.f / d : 0.f;
    return (int8_t) roundf(v * id);
}

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
