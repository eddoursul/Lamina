// src/kernels/f16_bits_test.cpp - f16_bits.hpp's conversions against the CPU's own (F16C), exhaustively (CPU only,
// x86):
//   1. f32_from_f16 over all 65,536 fp16 patterns: the same f32 bits as VCVTPH2PS for every non-NaN, a NaN of the
//      same sign for every NaN (F16C sets a signalling NaN's quiet bit, the header keeps the payload as it is);
//   2. f16_from_f32 over all 2^32 f32 patterns: the same fp16 bits as VCVTPS2PH rounding to nearest even for every
//      non-NaN, a quiet NaN of the same sign for every NaN;
//   3. dequant.hpp's fp16_to_fp32 and bf16_to_fp32 (the artifact's dequantizers) the kernels' bits on every pattern.
#include "lamina/artifact/dequant.hpp"
#include "lamina/kernels/bf16_bits.hpp"
#include "lamina/kernels/f16_bits.hpp"

#include <immintrin.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

#if defined(__GNUC__) && !defined(__F16C__)
#define F16C_FN __attribute__((target("f16c")))
#else
#define F16C_FN
#endif

namespace {
F16C_FN uint32_t cpu_f32_bits(uint16_t h) {
    const float f = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}
F16C_FN uint16_t cpu_f16(float f) {
    return (uint16_t) _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(f), _MM_FROUND_TO_NEAREST_INT), 0);
}
uint32_t bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}
bool f32_nan(uint32_t u) { return (u & 0x7fffffffu) > 0x7f800000u; }
bool f16_nan(uint16_t h) { return (h & 0x7fffu) > 0x7c00u; }
}   // namespace

int main() {
    unsigned long long bad = 0;
    for (uint32_t h = 0; h < 65536; ++h) {
        const uint32_t want = cpu_f32_bits((uint16_t) h), got = bits(lamina::kernels::f32_from_f16((uint16_t) h));
        const bool ok = f16_nan((uint16_t) h) ? f32_nan(got) && (got >> 31) == (want >> 31) : got == want;
        if (!ok && bad++ < 4) std::printf("  f32_from_f16(0x%04x) = 0x%08x, F16C 0x%08x\n", h, got, want);
        if (bits(lamina::fp16_to_fp32((uint16_t) h)) != got && bad++ < 8)
            std::printf("  dequant.hpp's fp16_to_fp32(0x%04x) differs\n", h);
        if (bits(lamina::bf16_to_fp32((uint16_t) h)) != bits(lamina::kernels::f32_from_bf16((uint16_t) h)) && bad++ < 8)
            std::printf("  dequant.hpp's bf16_to_fp32(0x%04x) differs\n", h);
    }
    uint32_t u = 0;
    do {
        float f;
        std::memcpy(&f, &u, 4);
        const uint16_t want = cpu_f16(f), got = lamina::kernels::f16_from_f32(f);
        const bool ok = f32_nan(u) ? f16_nan(got) && (got & 0x200u) && (got >> 15) == (want >> 15) : got == want;
        if (!ok && bad++ < 12) std::printf("  f16_from_f32(0x%08x) = 0x%04x, F16C 0x%04x\n", u, got, want);
    } while (++u != 0);
    std::printf("f16_bits_test %s\n", bad ? "FAILED" : "OK");
    return bad ? 1 : 0;
}
