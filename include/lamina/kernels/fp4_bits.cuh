// include/lamina/kernels/fp4_bits.cuh - pieces the NVFP4 kernels share: a sub-block's UE4M3 scale, and prmt with its
// selector in a register (a nibble's low three bits pick one of the two words' eight bytes; its top bit replicates
// that byte's sign over the byte instead).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace lamina::kernels {

/// ggml_ue4m3_to_fp32: the scale halved, for the doubled codes (0 and 0x7F: zero)
__device__ __forceinline__ float ue4m3_half(uint8_t x) {
    const uint32_t e = (x >> 3) & 0xF, m = x & 7;
    const float v = e ? __uint_as_float(((e + 119u) << 23) | (m << 20)) : (float) m * 0.0009765625f;
    return x == 0x7F ? 0.0f : v;
}

__device__ __forceinline__ uint32_t prmt_sel(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t d;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
    return d;
}

}  // namespace lamina::kernels
