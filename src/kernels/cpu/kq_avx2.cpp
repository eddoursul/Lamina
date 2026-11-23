// src/kernels/cpu/kq_avx2.cpp - Unsloth UD-Q4_K_XL's expert rows (Q4_K gate/up, Q5_1 and Q8_0 down) for several
// tokens at once, BIT-EXACT against ggml-cpu's own dot products.
//
// ggml-cpu is built for AVX2 here (LAMINA_PORTABLE: GGML_AVX2=ON, GGML_AVX512=OFF), so its vec_dot for these formats
// is the __AVX2__ branch of ggml-cpu/arch/x86/quants.c: ggml_vec_dot_q4_K_q8_K, ggml_vec_dot_q5_1_q8_1,
// ggml_vec_dot_q8_0_q8_0.  The functions below are those branches with the loop over tokens moved inside the loop
// over blocks: everything that depends only on the weights (the 4-bit unpacking, the 6-bit scales and mins, the
// fifth bits, the absolute values of the Q8_0 weights, the fp16 scales) is done once per block, and every token then
// runs exactly ggml's integer chain and ggml's float operations in ggml's order - so each token's result is ggml's
// to the bit (native_expert_parity checks it), whatever the number of tokens.  Unlike the i-quant kernels (#152)
// the group size therefore never changes an answer.
//
// The gain is the weight-side work and the weight bytes, read once per verify window instead of once per token.
#include "lamina/kernels/cpu/kq_avx2.hpp"
#include "lamina/kernels/cpu/native_expert.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <cmath>
#include <cstring>

namespace lamina::kernels::cpu {
namespace {

constexpr int MAXT = 8;   // the verify window's tokens per group (kVerifyMaxT)

inline float h2f(ggml_half h) {
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) u)));
}
// the fp16 pair {d, dmin} / {d, m} / {d, s} a block starts with (a union in ggml-common.h's C++ declarations)
inline float half_at(const void* block, int i) {
    ggml_half h;
    std::memcpy(&h, (const uint8_t*) block + 2 * i, 2);
    return h2f(h);
}

// ---- ggml's helpers (ggml-cpu/arch/x86/quants.c, the AVX2 / non-VNNI forms), verbatim
inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}
inline __m256i bytes_from_bits_32(const uint8_t* x) {
    uint32_t x32;
    std::memcpy(&x32, x, sizeof(uint32_t));
    const __m256i shuf_mask = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101,
                                                0x0000000000000000);
    __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32((int) x32), shuf_mask);
    const __m256i bit_mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
    bytes = _mm256_or_si256(bytes, bit_mask);
    return _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1));
}
inline __m256i bytes_from_nibbles_32(const uint8_t* rsi) {
    const __m128i tmp = _mm_loadu_si128((const __m128i*) rsi);
    const __m256i bytes = _mm256_insertf128_si256(_mm256_castsi128_si256(tmp), _mm_srli_epi16(tmp, 4), 1);
    const __m256i lowMask = _mm256_set1_epi8(0xF);
    return _mm256_and_si256(lowMask, bytes);
}
inline __m256 sum_i16_pairs_float(const __m256i x) {
    const __m256i ones = _mm256_set1_epi16(1);
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(ones, x));
}
inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
    return sum_i16_pairs_float(_mm256_maddubs_epi16(ax, sy));
}
alignas(32) const uint8_t k_shuffle[256] = {
     0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
     4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
     8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
    12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15};
inline __m256i get_scale_shuffle_k4(int i) { return _mm256_loadu_si256((const __m256i*) k_shuffle + i); }

// ---- Q4_K x Q8_K (ggml_vec_dot_q4_K_q8_K, __AVX2__)
void q4k_dot(const block_q4_K* x, int nb, const void* const* act, int nt, float* res) {
    static const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m256 acc[MAXT];
    __m128 acc_m[MAXT];
    for (int t = 0; t < nt; ++t) { acc[t] = _mm256_setzero_ps(); acc_m[t] = _mm_setzero_ps(); }
    for (int i = 0; i < nb; ++i) {
        const float xd = half_at(&x[i], 0), xdmin = half_at(&x[i], 1);
        uint32_t utmp[4];
        std::memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32((int) utmp[3], (int) utmp[2],
                                                                           (int) utmp[1], (int) utmp[0]));
        const __m128i mins = _mm256_extracti128_si256(mins_and_scales, 1);
        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = _mm256_insertf128_si256(_mm256_castsi128_si256(sc128), sc128, 1);
        __m256i scale_l[4], scale_h[4], q4l[4], q4h[4];
        const uint8_t* q4 = x[i].qs;
        for (int j = 0; j < QK_K / 64; ++j) {
            scale_l[j] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
            scale_h[j] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));
            const __m256i q4bits = _mm256_loadu_si256((const __m256i*) q4);
            q4 += 32;
            q4l[j] = _mm256_and_si256(q4bits, m4);
            q4h[j] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
        }
        for (int t = 0; t < nt; ++t) {
            const block_q8_K* y = (const block_q8_K*) act[t] + i;
            const float d = y->d * xd;
            const float dmin = -y->d * xdmin;
            const int8_t* q8 = y->qs;
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y->bsums);
            const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
            const __m128i prod = _mm_madd_epi16(mins, q8s);
            acc_m[t] = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m[t]);
            __m256i sumi = _mm256_setzero_si256();
            for (int j = 0; j < QK_K / 64; ++j) {
                const __m256i q8l = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                __m256i p16l = _mm256_maddubs_epi16(q4l[j], q8l);
                p16l = _mm256_madd_epi16(scale_l[j], p16l);
                const __m256i q8h = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                __m256i p16h = _mm256_maddubs_epi16(q4h[j], q8h);
                p16h = _mm256_madd_epi16(scale_h[j], p16h);
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
            }
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc[t]);
        }
    }
    for (int t = 0; t < nt; ++t) {
        __m128 m = acc_m[t];
        m = _mm_add_ps(m, _mm_movehl_ps(m, m));
        m = _mm_add_ss(m, _mm_movehdup_ps(m));
        res[t] = hsum_float_8(acc[t]) + _mm_cvtss_f32(m);
    }
}

// ---- Q5_1 x Q8_1 (ggml_vec_dot_q5_1_q8_1, __AVX2__)
void q5_1_dot(const block_q5_1* x, int nb, const void* const* act, int nt, float* res) {
    __m256 acc[MAXT];
    float summs[MAXT];
    for (int t = 0; t < nt; ++t) { acc[t] = _mm256_setzero_ps(); summs[t] = 0.0f; }
    for (int ib = 0; ib < nb; ++ib) {
        const __m256 dx = _mm256_set1_ps(half_at(&x[ib], 0));
        const float xm = half_at(&x[ib], 1);
        __m256i qx = bytes_from_nibbles_32(x[ib].qs);
        __m256i bxhi = bytes_from_bits_32(x[ib].qh);
        bxhi = _mm256_and_si256(bxhi, _mm256_set1_epi8(0x10));
        qx = _mm256_or_si256(qx, bxhi);
        for (int t = 0; t < nt; ++t) {
            const block_q8_1* y = (const block_q8_1*) act[t] + ib;
            summs[t] += xm * half_at(y, 1);
            const __m256 dy = _mm256_set1_ps(half_at(y, 0));
            const __m256i qy = _mm256_loadu_si256((const __m256i*) y->qs);
            const __m256 q = mul_sum_us8_pairs_float(qx, qy);
            acc[t] = _mm256_fmadd_ps(q, _mm256_mul_ps(dx, dy), acc[t]);
        }
    }
    for (int t = 0; t < nt; ++t) res[t] = hsum_float_8(acc[t]) + summs[t];
}

// ---- Q8_0 x Q8_0 (ggml_vec_dot_q8_0_q8_0, __AVX2__; mul_sum_i8_pairs_float without AVX-VNNI-INT8)
void q8_0_dot(const block_q8_0* x, int nb, const void* const* act, int nt, float* res) {
    __m256 acc[MAXT];
    for (int t = 0; t < nt; ++t) acc[t] = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
        const float xd = h2f(x[ib].d);
        const __m256i qx = _mm256_loadu_si256((const __m256i*) x[ib].qs);
        const __m256i ax = _mm256_sign_epi8(qx, qx);
        for (int t = 0; t < nt; ++t) {
            const block_q8_0* y = (const block_q8_0*) act[t] + ib;
            const __m256 d = _mm256_set1_ps(xd * h2f(y->d));
            const __m256i qy = _mm256_loadu_si256((const __m256i*) y->qs);
            const __m256i sy = _mm256_sign_epi8(qy, qx);
            const __m256 q = mul_sum_us8_pairs_float(ax, sy);
            acc[t] = _mm256_fmadd_ps(d, q, acc[t]);
        }
    }
    for (int t = 0; t < nt; ++t) res[t] = hsum_float_8(acc[t]);
}

// ---- NVFP4 x Q8_0 (ggml_vec_dot_nvfp4_q8_0, __AVX2__): a block of 64 values in four sub-blocks of 16, each with a
// UE4M3 scale (ggml_ue4m3_to_fp32: its table's values, half the E4M3 value), against two Q8_0 blocks.  The weights'
// int8 values (kvalues_fp4 by nibble), their absolute values and the scales once a block; each token ggml's products,
// its scales' products and its two fused adds in ggml's order.
// (a table, as ggml-cpu's GGML_CPU_UE4M3_TO_FP32 on x86: ggml_ue4m3_to_fp32's values)
struct Ue4m3 {
    float v[256];
    Ue4m3() {
        for (int x = 0; x < 256; ++x) {
            const int e = (x >> 3) & 0xF, m = x & 0x7;
            const float raw = e == 0 ? ldexpf((float) m, -9) : ldexpf(1.0f + (float) m / 8.0f, e - 7);
            v[x] = x == 0 || x == 0x7F ? 0.0f : raw * 0.5f;
        }
    }
};
const Ue4m3 k_ue4m3;
inline float ue4m3(uint8_t x) { return k_ue4m3.v[x]; }
void nvfp4_dot(const block_nvfp4* x, int nb, const void* const* act, int nt, float* res) {
    const __m128i values128 = _mm_loadu_si128((const __m128i*) kvalues_fp4);
    const __m128i m4b = _mm_set1_epi8(0x0f);
    const __m256i mone = _mm256_set1_epi16(1);
    __m256 acc[MAXT];
    for (int t = 0; t < nt; ++t) acc[t] = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
        const __m128i q4bits_01 = _mm_loadu_si128((const __m128i*) (x[ib].qs + 0));
        const __m128i q4bits_23 = _mm_loadu_si128((const __m128i*) (x[ib].qs + 16));
        const __m128i q4_01_lo = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_01, m4b));
        const __m128i q4_01_hi = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_01, 4), m4b));
        const __m128i q4_23_lo = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_23, m4b));
        const __m128i q4_23_hi = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_23, 4), m4b));
        const __m256i q4_01 = _mm256_set_m128i(_mm_unpackhi_epi64(q4_01_lo, q4_01_hi), _mm_unpacklo_epi64(q4_01_lo, q4_01_hi));
        const __m256i q4_23 = _mm256_set_m128i(_mm_unpackhi_epi64(q4_23_lo, q4_23_hi), _mm_unpacklo_epi64(q4_23_lo, q4_23_hi));
        const __m256i a01 = _mm256_sign_epi8(q4_01, q4_01), a23 = _mm256_sign_epi8(q4_23, q4_23);
        const float u0 = ue4m3(x[ib].d[0]), u1 = ue4m3(x[ib].d[1]), u2 = ue4m3(x[ib].d[2]), u3 = ue4m3(x[ib].d[3]);
        for (int t = 0; t < nt; ++t) {
            const block_q8_0* y = (const block_q8_0*) act[t] + 2 * ib;
            const __m256i q8_01 = _mm256_loadu_si256((const __m256i*) y[0].qs);
            const __m256i q8_23 = _mm256_loadu_si256((const __m256i*) y[1].qs);
            const __m256i p_1 = _mm256_madd_epi16(_mm256_maddubs_epi16(a01, _mm256_sign_epi8(q8_01, q4_01)), mone);
            const __m256i p_2 = _mm256_madd_epi16(_mm256_maddubs_epi16(a23, _mm256_sign_epi8(q8_23, q4_23)), mone);
            const float dy0 = h2f(y[0].d), dy1 = h2f(y[1].d);
            const float s0 = u0 * dy0, s1 = u1 * dy0, s2 = u2 * dy1, s3 = u3 * dy1;
            const __m256 scales01 = _mm256_set_m128(_mm_set1_ps(s1), _mm_set1_ps(s0));
            const __m256 scales23 = _mm256_set_m128(_mm_set1_ps(s3), _mm_set1_ps(s2));
            acc[t] = _mm256_fmadd_ps(scales01, _mm256_cvtepi32_ps(p_1), acc[t]);
            acc[t] = _mm256_fmadd_ps(scales23, _mm256_cvtepi32_ps(p_2), acc[t]);
        }
    }
    for (int t = 0; t < nt; ++t) res[t] = hsum_float_8(acc[t]);
}

void dot_rows(int type, const uint8_t* row, int n, const void* const* act, int nt, float* res) {
    switch (type) {
        case 12: q4k_dot((const block_q4_K*) row, n / QK_K, act, nt, res); break;
        case 7: q5_1_dot((const block_q5_1*) row, n / QK5_1, act, nt, res); break;
        case 8: q8_0_dot((const block_q8_0*) row, n / QK8_0, act, nt, res); break;
        case 40: nvfp4_dot((const block_nvfp4*) row, n / QK_NVFP4, act, nt, res); break;
        default: break;
    }
}

}  // namespace

void kq256_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                   float* const* ff, int r0, int r1) {
    const bool exact = native_exact_swiglu();
    for (int t0 = 0; t0 < nt; t0 += MAXT) {
        const int m = nt - t0 < MAXT ? nt - t0 : MAXT;
        float g[MAXT], u[MAXT];
        for (int r = r0; r < r1; ++r) {
            dot_rows(type, blob + (size_t) r * gu_row, n, act + t0, m, g);
            dot_rows(type, blob + up_off + (size_t) r * gu_row, n, act + t0, m, u);
            // native_gu_rows' own SwiGLU: the same bits as the per-token ggml path (the exact one compiled there)
            for (int t = 0; t < m; ++t)
                ff[t0 + t][r] = exact ? native_swiglu_exact(g[t], u[t]) : (g[t] / (1.f + std::exp(-g[t]))) * u[t];
        }
    }
}

void nvfp4_dots(const uint8_t* row, int n, const void* const* act, int nt, float* res) {
    for (int t0 = 0; t0 < nt; t0 += MAXT) nvfp4_dot((const block_nvfp4*) row, n / QK_NVFP4, act + t0, nt - t0 < MAXT ? nt - t0 : MAXT, res + t0);
}

void kq256_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                int r0, int r1) {
    for (int t0 = 0; t0 < nt; t0 += MAXT) {
        const int m = nt - t0 < MAXT ? nt - t0 : MAXT;
        float s[MAXT];
        for (int r = r0; r < r1; ++r) {
            dot_rows(type, w + (size_t) r * row_bytes, n, act + t0, m, s);
            for (int t = 0; t < m; ++t) out[t0 + t][r] = s[t];
        }
    }
}

}  // namespace lamina::kernels::cpu
