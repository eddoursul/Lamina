// src/qwen35/q36_lora.cuh - LoRA adapters on the qwen35 path's products: y += s B (A x) for every adapter of a product.
//
// A site is up to three products that read one input (q / k / v, gate / up, GDN alpha / beta) with every adapter
// that has a pair for them: one launch of k_lora_a and one of k_lora_b cover it, whatever the number of adapters.
// A decode batch (T <= kRows rows) runs here; a prompt chunk runs its pairs as GEMMs (the engine, kernels/gemm.hpp).
//   k_lora_a: the partial dot products A_seg x of every segment (the site's adapters, their ranks side by side), a K
//             split a block: its A rows and input loaded at once, one barrier, no atomics.
//   k_lora_b: t = the K splits' partials summed in a fixed order (a row's t does not depend on T or on the other rows,
//             so a verify window's rows are a decode step's bit for bit), then out_m += sum over member m's segments
//             of s_row (B_seg t_seg), the scales read per row from the device table (a batch window's rows may be
//             different requests'), a segment whose adapter no row uses skipped (the same test in both stages).  Its B
//             rows and outputs are loaded before it waits for the partials.
// The scales live in device memory, so one captured graph serves every choice of adapters and scales.
#pragma once

#include <cuda_fp16.h>

#include <cstdint>

namespace q36lora {

constexpr int kRows = 8;          // the decode rows a launch takes (the engine's kVerify)
constexpr int kMaxSeg = 24;       // segments a launch takes (a site with more runs in several)
constexpr int kMaxR = 512;        // their ranks summed, at most
constexpr int kGroup = 8;         // rank rows a block of the A stage computes (a warp each)
constexpr int kChunk = 256;       // input elements a warp covers at once (8 a lane)
constexpr int kSliceMax = 1024;   // a K split's input elements at most (the A stage's staging buffer)
constexpr int kPre = 4;           // 16-byte words of B a thread loads ahead (a rank-32 F16 row)

struct Seg {
    const void* a = nullptr;   // [rank][n_in] row-major (ggml's lora_a [n_in, rank])
    const void* b = nullptr;   // [n_out][rank] row-major (ggml's lora_b [rank, n_out])
    float alpha = 0.f;         // the adapter's alpha (0: none: the scale as given)
    float rank_true = 1.f;     // its rank in the file (the scale is alpha / rank_true)
    int rank = 0;              // a multiple of kGroup (zero rows and columns pad it)
    int r0 = 0;                // its first column in the partials
    int adapter = 0;           // its column of the scale table
    int member = 0;            // the output it adds into
    int f32 = 0;               // A and B in F32 (else F16)
    int g0 = 0;                // its first rank group (the A stage's blocks g0 .. g0 + rank / kGroup - 1)
};

struct Site {
    Seg seg[kMaxSeg];
    int n_seg = 0, n_groups = 0, R = 0;
    // A stage: x [T][ldx]; with u the input is SwiGLU(x, u) (the down product's, as native_swiglu_quantize_q8_1)
    const float* x = nullptr;
    const float* u = nullptr;
    int ldx = 0, n_in = 0, ks = 1, kslice = kSliceMax;   // kslice: a K split's elements (a multiple of kChunk)
    // B stage: member m's outputs out[m] [T][ld_out[m]] (its first n_out[m] columns), blocks ob[m] .. ob[m + 1] - 1
    float* out[3] = {nullptr, nullptr, nullptr};
    int ld_out[3] = {0, 0, 0}, n_out[3] = {0, 0, 0};
    int ob[4] = {0, 0, 0, 0};
    int n_mem = 0;
    int T = 0, n_ad = 0;
    const float* scale = nullptr;   // [kRows][n_ad]: each row's scale of each adapter
    float* part = nullptr;          // [ks][kRows][R]
};

/// llama.cpp's scale of a pair: adapter_scale * alpha / rank, or adapter_scale without an alpha
__host__ __device__ __forceinline__ float pair_scale(float user, float alpha, float rank_true) {
    return alpha != 0.f ? user * alpha / rank_true : user;
}

/// some row of the T uses the adapter: every row's scale loaded at once (the table has kRows rows, zeros past T)
__device__ __forceinline__ bool seg_live(const Site& s, int adapter) {
    float any = 0.f;
#pragma unroll
    for (int r = 0; r < kRows; ++r) any += r < s.T ? fabsf(s.scale[r * s.n_ad + adapter]) : 0.f;
    return any != 0.f;
}

/// 8 floats from 16 bytes of F16
__device__ __forceinline__ void half8(const uint4 v, float w[8]) {
    const __half2* h = (const __half2*) &v;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const float2 f = __half22float2(h[e]);
        w[2 * e] = f.x;
        w[2 * e + 1] = f.y;
    }
}
/// 8 consecutive elements of an F16 or F32 matrix as floats
__device__ __forceinline__ void load8(const void* p, int f32, size_t off, float w[8]) {
    if (f32) {
        const float4* q = (const float4*) ((const float*) p + off);
        const float4 a = q[0], b = q[1];
        w[0] = a.x;
        w[1] = a.y;
        w[2] = a.z;
        w[3] = a.w;
        w[4] = b.x;
        w[5] = b.y;
        w[6] = b.z;
        w[7] = b.w;
    } else {
        half8(*(const uint4*) ((const __half*) p + off), w);
    }
}

/// The A stage: grid (n_groups, ks), 256 threads.  Block (g, k): rank group g's kGroup rows of its segment (a warp
/// each) over K split k: the rows' A words and the split's inputs (SwiGLU applied with u) loaded at once into
/// registers and shared memory, then each row's partial dot product to part.
__global__ void __launch_bounds__(256) k_lora_a(const Site s) {
    __shared__ __align__(16) float xs[kRows][kSliceMax];
    const int g = (int) blockIdx.x, ksp = (int) blockIdx.y;
    const int warp = (int) threadIdx.x >> 5, lane = (int) threadIdx.x & 31;
    int si = 0;
    while (si + 1 < s.n_seg && g >= s.seg[si + 1].g0) ++si;
    const Seg& sg = s.seg[si];
    const int r = (g - sg.g0) * kGroup + warp;   // < sg.rank: the ranks are multiples of kGroup
    const int k0 = ksp * s.kslice, len = min(s.n_in - k0, s.kslice);
    constexpr int NC = kSliceMax / kChunk;
    // every load first (the A words, the inputs, the scales), then the waits: one memory latency, not one a load
    float w[NC][8];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const int e = c * kChunk + lane * 8;
        if (e < len) {
            load8(sg.a, sg.f32, (size_t) r * s.n_in + k0 + e, w[c]);
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) w[c][i] = 0.f;
        }
    }
    float xv[NC][kRows], uv[NC][kRows];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const int i = c * kChunk + (int) threadIdx.x;
#pragma unroll
        for (int row = 0; row < kRows; ++row) {
            const bool in = row < s.T && i < len;
            xv[c][row] = in ? s.x[(size_t) row * s.ldx + k0 + i] : 0.f;
            uv[c][row] = in && s.u ? s.u[(size_t) row * s.ldx + k0 + i] : 0.f;
        }
    }
    if (!seg_live(s, sg.adapter)) return;   // the same for every thread of the block (the B stage skips it too)
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const int i = c * kChunk + (int) threadIdx.x;
#pragma unroll
        for (int row = 0; row < kRows; ++row)
            if (row < s.T && i < len) {
                const float v = xv[c][row];
                xs[row][i] = s.u ? __fmul_rn(__fdividef(v, __fadd_rn(1.0f, __expf(-v))), uv[c][row]) : v;
            }
    }
    __syncthreads();
    for (int row = 0; row < s.T; ++row) {
        float acc = 0.f;
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const int e = c * kChunk + lane * 8;
            if (e >= len) break;   // (n_in is a multiple of 8: whole groups of 8)
            const float4 xa = *(const float4*) &xs[row][e];
            const float4 xb = *(const float4*) &xs[row][e + 4];
            float d = w[c][0] * xa.x;
            d = fmaf(w[c][1], xa.y, d);
            d = fmaf(w[c][2], xa.z, d);
            d = fmaf(w[c][3], xa.w, d);
            d = fmaf(w[c][4], xb.x, d);
            d = fmaf(w[c][5], xb.y, d);
            d = fmaf(w[c][6], xb.z, d);
            d = fmaf(w[c][7], xb.w, d);
            acc += d;
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) s.part[((size_t) ksp * kRows + row) * s.R + sg.r0 + r] = acc;
    }
}

/// d[row] += w . t_row[0 .. 8) for the T rows of t (row stride R) - every index static, the arrays in registers
__device__ __forceinline__ void dot8_rows(const float w[8], const float* t0, int R, int T, float d[kRows]) {
#pragma unroll
    for (int row = 0; row < kRows; ++row)
        if (row < T) {
            const float* tr = t0 + row * R;
            float p = w[0] * tr[0];
#pragma unroll
            for (int e = 1; e < 8; ++e) p = fmaf(w[e], tr[e], p);
            d[row] += p;
        }
}

/// The B stage: grid (ob[n_mem]), 256 threads, a thread per output column of its member.  Its loads before any wait:
/// its B words (its member's first segment's, up to kPre), its outputs, and t of the member's segments from the
/// partials (the K splits in order) into shared memory; then the live segments' sums.
__global__ void __launch_bounds__(256, 2) k_lora_b(const Site s) {
    __shared__ float ts[kRows * kMaxR];
    int m = 0;
    while (m + 1 < s.n_mem && (int) blockIdx.x >= s.ob[m + 1]) ++m;
    const int j = ((int) blockIdx.x - s.ob[m]) * (int) blockDim.x + (int) threadIdx.x;
    const bool col = j < s.n_out[m];
    int first = 0;
    while (first < s.n_seg && s.seg[first].member != m) ++first;
    if (first == s.n_seg) return;
    const Seg& f = s.seg[first];
    const int wpr = f.rank * (f.f32 ? 4 : 2) / 16;   // its 16-byte words a row
    const int npre = col ? min(kPre, wpr) : 0;
    uint4 pre[kPre];
#pragma unroll
    for (int q = 0; q < kPre; ++q)
        pre[q] = q < npre ? ((const uint4*) f.b)[(size_t) j * wpr + q] : make_uint4(0, 0, 0, 0);
    float o[kRows];
#pragma unroll
    for (int row = 0; row < kRows; ++row) o[row] = col && row < s.T ? s.out[m][(size_t) row * s.ld_out[m] + j] : 0.f;
    for (int si = first; si < s.n_seg; ++si) {
        const Seg& sg = s.seg[si];
        if (sg.member != m) continue;   // (a segment no row uses: its columns staged and not read)
        for (int e = (int) threadIdx.x; e < s.T * sg.rank; e += (int) blockDim.x) {
            const int row = e / sg.rank, c = sg.r0 + (e - row * sg.rank);
            const float* pp = s.part + (size_t) row * s.R + c;
            const size_t step = (size_t) kRows * s.R;
            float v = 0.f;   // the K splits in order, their loads four at a time
            int k = 0;
            for (; k + 4 <= s.ks; k += 4) {
                const float a = __ldcg(pp + k * step), b = __ldcg(pp + (k + 1) * step),
                            c2 = __ldcg(pp + (k + 2) * step), d = __ldcg(pp + (k + 3) * step);
                v += a;
                v += b;
                v += c2;
                v += d;
            }
            for (; k < s.ks; ++k) v += __ldcg(pp + k * step);
            ts[row * s.R + c] = v;
        }
    }
    __syncthreads();
    if (!col) return;
    float acc[kRows];
    bool hit[kRows];
#pragma unroll
    for (int i = 0; i < kRows; ++i) {
        acc[i] = 0.f;
        hit[i] = false;
    }
    for (int si = first; si < s.n_seg; ++si) {
        const Seg& sg = s.seg[si];
        if (sg.member != m || !seg_live(s, sg.adapter)) continue;
        float d[kRows];
#pragma unroll
        for (int i = 0; i < kRows; ++i) d[i] = 0.f;
        int k = 0;   // the elements the prefetched words held, then the rest from memory
        if (si == first) {
#pragma unroll
            for (int q = 0; q < kPre; ++q) {
                if (q >= npre) break;
                float w[8];
                if (!sg.f32) {
                    half8(pre[q], w);
                } else {
                    if (q % 2 == 1 || q + 1 >= npre) continue;
                    w[0] = __uint_as_float(pre[q].x);
                    w[1] = __uint_as_float(pre[q].y);
                    w[2] = __uint_as_float(pre[q].z);
                    w[3] = __uint_as_float(pre[q].w);
                    w[4] = __uint_as_float(pre[q + 1].x);
                    w[5] = __uint_as_float(pre[q + 1].y);
                    w[6] = __uint_as_float(pre[q + 1].z);
                    w[7] = __uint_as_float(pre[q + 1].w);
                }
                dot8_rows(w, &ts[sg.r0 + k], s.R, s.T, d);
                k += 8;
            }
        }
        for (; k < sg.rank; k += 8) {
            float w[8];
            load8(sg.b, sg.f32, (size_t) j * sg.rank + k, w);
            dot8_rows(w, &ts[sg.r0 + k], s.R, s.T, d);
        }
#pragma unroll
        for (int row = 0; row < kRows; ++row) {
            if (row >= s.T) break;
            const float u = s.scale[row * s.n_ad + sg.adapter];
            if (u == 0.f) continue;
            acc[row] = hit[row] ? fmaf(pair_scale(u, sg.alpha, sg.rank_true), d[row], acc[row])
                                : pair_scale(u, sg.alpha, sg.rank_true) * d[row];
            hit[row] = true;
        }
    }
#pragma unroll
    for (int row = 0; row < kRows; ++row)
        if (row < s.T && hit[row]) s.out[m][(size_t) row * s.ld_out[m] + j] = o[row] + acc[row];
}

/// x [T][n] (row stride ldx) -> h [T][n] in FP16: a prompt chunk's input for the F16 pairs' GEMMs
__global__ void k_lora_f16(const float* __restrict__ x, int ldx, __half* __restrict__ h, int n, int64_t total) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int64_t row = i / n, col = i - row * n;
    h[i] = __float2half(x[row * ldx + col]);
}

}   // namespace q36lora
