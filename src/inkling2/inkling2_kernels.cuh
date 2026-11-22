// src/inkling2/inkling2_kernels.cuh - Inkling 2's own steps as kernels, for inkling2_engine.cu: the depth mix over the
// block inputs, the loop's messages and gates, the norms with their q8_1 images, the causal convolutions and their
// state, the attention with its distance bias and sink over a window ring or a range of cells, the router, the experts'
// sum.  A forward is T rows (1..kMaxT), consecutive positions of one sequence; row t of a buffer [T][n] is n floats at
// t * n.  The matrix products are the shared kernels' (native_mmvq, native_expert_grouped).
//
// The model as llama.cpp computes it (src/models/inkling2.cpp); the numbers behind the constants are in
// lamina/inkling2/geometry.hpp.
#pragma once

#include "lamina/kernels/q8_1_finite.hpp"
#include "lamina/kernels/q8_1_il16.hpp"
#include "lamina/kernels/warp.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstring>
#include <cstdint>

namespace ik {

using lamina::kernels::q8_1_ds;
using lamina::kernels::q8_1_finite;
using lamina::kernels::q8_1_quant;
using lamina::kernels::cp_async16;
using lamina::kernels::cp_async8;
using lamina::kernels::cp_commit;
using lamina::kernels::cp_wait;
using lamina::kernels::ldsm4;
using lamina::kernels::ldsm4t;
using lamina::kernels::warp_max;
using lamina::kernels::warp_sum;

constexpr int kMaxT = 8;      // rows a forward takes at most (native_mmvq's columns)
constexpr int kBlk = 4;       // block inputs a mix reads at most (the embedding and three blocks' outputs)
constexpr int kTB = 256;      // threads of the row kernels
constexpr int kWide = 1024;   // of those whose time is their threads' loop over a row: the mix, the norm, the bias

/// a q8_1 block as ggml has it: the scale and the values' sum as halves, 32 values
struct Q81 {
    half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(Q81) == 36, "q8_1 block");

// ---------------------------------------------------------------------------------------------- block-wide sums
/// K values a thread summed over a block of kTB threads: every thread gets the K block sums (shared `red`: [8][K])
template <int K> __device__ __forceinline__ void block_sums(float (&v)[K], float (*red)[K]) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#pragma unroll
    for (int k = 0; k < K; ++k) {
        v[k] = warp_sum(v[k]);
        if (lane == 0) red[warp][k] = v[k];
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < K; ++k) {
        float s = 0.f;
        for (int w = 0; w < (int) (blockDim.x >> 5); ++w) s += red[w][k];
        v[k] = s;
    }
    __syncthreads();
}

/// The same in a block of up to 1,024 threads: the warps' sums gathered by the first warp (`red`: [32][K], `tot`: [K])
template <int K> __device__ __forceinline__ void block_sums_wide(float (&v)[K], float (*red)[K], float* tot) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nw = (int) (blockDim.x >> 5);
#pragma unroll
    for (int k = 0; k < K; ++k) {
        v[k] = warp_sum(v[k]);
        if (lane == 0) red[warp][k] = v[k];
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int k = 0; k < K; ++k) {
            const float t = warp_sum(lane < nw ? red[lane][k] : 0.f);
            if (lane == 0) tot[k] = t;
        }
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < K; ++k) v[k] = tot[k];
}

// Four values a load: a row kernel's time is its threads' loads in a row (~0.1 us each, whatever they bring), so a
// thread takes its values 16 bytes at a time and keeps them for the kernel's second step.  Rows are 16-byte aligned
// (their buffers, and widths that are multiples of 4).
__device__ __forceinline__ float4 ld4(const float* p, int i4) { return reinterpret_cast<const float4*>(p)[i4]; }
__device__ __forceinline__ void st4(float* p, int i4, float4 v) { reinterpret_cast<float4*>(p)[i4] = v; }
__device__ __forceinline__ float dot4(float4 a, float4 b) { return (a.x * b.x + a.y * b.y) + (a.z * b.z + a.w * b.w); }
__device__ __forceinline__ float4 mul4(float4 a, float s) { return make_float4(a.x * s, a.y * s, a.z * s, a.w * s); }

// ---------------------------------------------------------------------------------------------- the depth mix
/// Before a layer's attention and before its MoE the input is a softmax mix of the RMS-normed block inputs and the
/// current state, a query a mix: scores = state . query / sqrt(N).  `blk`: the block inputs, normed already ([T][N]
/// each); `cur`: the state, raw; `q`: nq queries of N.  out[c] ([T][N]) the mix of query c; `cur_n`, when not null,
/// the normed state (a block's input-to-be).  nq 0: only cur_n.  The loop's messages are the same mix over the
/// previous pass (blk[0] its first block's output, cur its last).
struct MixIn {
    const float* blk[kBlk];
    const float* cur;
    const float* q;
    float* out;      // [nq][T][N]
    float* cur_n;    // [T][N] or null
    int nb, nq, N, T;
};
/// A block a row and a query; a thread four values (N <= 4 x its threads), loaded once for the sums and the mix.
__global__ void __launch_bounds__(kWide) mix_kernel(MixIn a) {
    const int t = blockIdx.x, c = blockIdx.y, i4 = threadIdx.x;
    __shared__ float red[32][kBlk + 2];
    __shared__ float tot[kBlk + 2];
    const bool on = i4 < a.N / 4;
    const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 x = z, bl[kBlk];
    float v[kBlk + 2];
#pragma unroll
    for (int k = 0; k < kBlk + 2; ++k) v[k] = 0.f;
#pragma unroll
    for (int j = 0; j < kBlk; ++j) bl[j] = z;
    if (on) {
        x = ld4(a.cur + (size_t) t * a.N, i4);
        const float4 q = a.nq > 0 ? ld4(a.q + (size_t) c * a.N, i4) : z;
        v[kBlk] = dot4(x, x);
        v[kBlk + 1] = dot4(x, q);
#pragma unroll
        for (int j = 0; j < kBlk; ++j)
            if (j < a.nb) {
                bl[j] = ld4(a.blk[j] + (size_t) t * a.N, i4);
                v[j] = dot4(bl[j], q);
            }
    }
    block_sums_wide<kBlk + 2>(v, red, tot);
    const float inv = rsqrtf(v[kBlk] / (float) a.N + 1e-6f);
    const float4 xn = mul4(x, inv);
    if (a.nq == 0) {
        if (on) st4(a.cur_n + (size_t) t * a.N, i4, xn);
        return;
    }
    // softmax over the nb block inputs and the state
    const float sc = rsqrtf((float) a.N);
    float s[kBlk + 1], mx = v[kBlk + 1] * inv * sc;
    s[a.nb] = mx;
    for (int j = 0; j < a.nb; ++j) {
        s[j] = v[j] * sc;
        mx = fmaxf(mx, s[j]);
    }
    float sum = 0.f;
    for (int j = 0; j <= a.nb; ++j) {
        s[j] = __expf(s[j] - mx);
        sum += s[j];
    }
    if (!on) return;
    const float r = 1.0f / sum;
    float4 y = mul4(xn, s[a.nb] * r);
#pragma unroll
    for (int j = 0; j < kBlk; ++j)
        if (j < a.nb) {
            const float wj = s[j] * r;
            y = make_float4(y.x + wj * bl[j].x, y.y + wj * bl[j].y, y.z + wj * bl[j].z, y.w + wj * bl[j].w);
        }
    st4(a.out + ((size_t) c * a.T + t) * a.N, i4, y);
    if (c == 0 && a.cur_n) st4(a.cur_n + (size_t) t * a.N, i4, xn);
}

// ---------------------------------------------------------------------------------------------- the loop's gates
/// A looped layer's input u takes the previous pass's message m: (1 - tanh(F h)) u + tanh(G h) m, h = D [rms(u);
/// rms(m)].  D [R][2N], F and G [N][R] (GGUF rows).  n inputs a launch (the attention's three, or the MoE's one).
struct CtrlIn {
    const float* u[3];
    const float* m[3];
    const float* D[3];
    const float* G[3];
    const float* F[3];
    float* out[3];
    float* h;   // [n][T][R]
    int n, N, R;
};
constexpr int kRankMax = 64;
/// h: a block a row, an input and one of the R outputs; a thread four values of each half
__global__ void __launch_bounds__(kWide) ctrl_h_kernel(CtrlIn a) {
    const int r = blockIdx.x, t = blockIdx.y, c = blockIdx.z, i4 = threadIdx.x;
    __shared__ float red[32][4];
    __shared__ float tot[4];
    float v[4] = {0.f, 0.f, 0.f, 0.f};
    if (i4 < a.N / 4) {
        const float4 u = ld4(a.u[c] + (size_t) t * a.N, i4), m = ld4(a.m[c] + (size_t) t * a.N, i4);
        const float* d = a.D[c] + (size_t) r * 2 * a.N;
        v[0] = dot4(u, u);
        v[1] = dot4(m, m);
        v[2] = dot4(ld4(d, i4), u);
        v[3] = dot4(ld4(d + a.N, i4), m);
    }
    block_sums_wide<4>(v, red, tot);
    if (threadIdx.x == 0)
        a.h[((size_t) c * gridDim.y + t) * a.R + r] =
            v[2] * rsqrtf(v[0] / (float) a.N + 1e-6f) + v[3] * rsqrtf(v[1] / (float) a.N + 1e-6f);
}
/// the gated rows: four threads an output, each a quarter of its R products in 16-byte loads (R a multiple of 16; a
/// thread alone took as long as its 2 R loads in a row).  Grid (ceil(4 N / kTB), T, n).
__global__ void __launch_bounds__(kTB) ctrl_apply_kernel(CtrlIn a) {
    const int t = blockIdx.y, c = blockIdx.z;
    const int i = (blockIdx.x * kTB + threadIdx.x) >> 2, part = threadIdx.x & 3;
    __shared__ float h[kRankMax];
    if (threadIdx.x < a.R) h[threadIdx.x] = a.h[((size_t) c * gridDim.y + t) * a.R + threadIdx.x];
    __syncthreads();
    float sg = 0.f, sf = 0.f;
    if (i < a.N) {
        const int per = a.R / 4;   // a thread's products
        const float4* g = reinterpret_cast<const float4*>(a.G[c] + (size_t) i * a.R + part * per);
        const float4* f = reinterpret_cast<const float4*>(a.F[c] + (size_t) i * a.R + part * per);
        const float* hp = h + part * per;
        for (int k = 0; k < per / 4; ++k) {
            const float4 gv = g[k], fv = f[k];
            const float h0 = hp[4 * k], h1 = hp[4 * k + 1], h2 = hp[4 * k + 2], h3 = hp[4 * k + 3];
            sg += gv.x * h0 + gv.y * h1 + gv.z * h2 + gv.w * h3;
            sf += fv.x * h0 + fv.y * h1 + fv.z * h2 + fv.w * h3;
        }
    }
    sg += __shfl_xor_sync(0xffffffffu, sg, 1);
    sg += __shfl_xor_sync(0xffffffffu, sg, 2);
    sf += __shfl_xor_sync(0xffffffffu, sf, 1);
    sf += __shfl_xor_sync(0xffffffffu, sf, 2);
    if (i >= a.N || part != 0) return;
    const size_t at = (size_t) t * a.N + i;
    a.out[c][at] = (1.0f - tanhf(sf)) * a.u[c][at] + tanhf(sg) * a.m[c][at];
}

// ---------------------------------------------------------------------------------------------- norms
/// y = x rsqrt(mean(x^2) + eps) w, and its q8_1 image (N / 32 blocks) where q is not null; n inputs a launch.  w null:
/// no weight.  il: the image's blocks interleaved (q8_1_il16.hpp), as the experts' NVFP4 kernels read them.
struct NormIn {
    const float* x[3];
    const float* w[3];
    float* y[3];
    Q81* q[3];
    int n, N;
    float eps;
    bool il;
};
/// A block a row and an input; a thread four values, eight threads a q8_1 block.
__global__ void __launch_bounds__(kWide) norm_kernel(NormIn a) {
    const int t = blockIdx.x, c = blockIdx.y, i4 = threadIdx.x;
    __shared__ float red[32][1];
    __shared__ float tot[1];
    const bool on = i4 < a.N / 4;
    const float4 x = on ? ld4(a.x[c] + (size_t) t * a.N, i4) : make_float4(0.f, 0.f, 0.f, 0.f);
    float v[1] = {dot4(x, x)};
    block_sums_wide<1>(v, red, tot);
    float4 y = mul4(x, rsqrtf(v[0] / (float) a.N + a.eps));
    if (on && a.w[c]) {
        const float4 w = ld4(a.w[c], i4);
        y = make_float4(y.x * w.x, y.y * w.y, y.z * w.z, y.w * w.w);
    }
    if (on && a.y[c]) st4(a.y[c] + (size_t) t * a.N, i4, y);
    if (a.q[c] == nullptr) return;
    float amax = fmaxf(fmaxf(fabsf(y.x), fabsf(y.y)), fmaxf(fabsf(y.z), fabsf(y.w)));
    float sum = (y.x + y.y) + (y.z + y.w);
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) {   // the block's eight threads
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    if (!on) return;
    const float d = q8_1_finite(amax / 127.0f);
    Q81* q = a.q[c] + (size_t) t * (a.N / 32) + i4 / 8;
    if (a.il) {   // (a thread's four values stand two bytes apart there)
        const int l = (i4 & 7) * 4;
        q->qs[lamina::kernels::q8_1_il16(l)] = q8_1_quant(y.x, d, amax);
        q->qs[lamina::kernels::q8_1_il16(l + 1)] = q8_1_quant(y.y, d, amax);
        q->qs[lamina::kernels::q8_1_il16(l + 2)] = q8_1_quant(y.z, d, amax);
        q->qs[lamina::kernels::q8_1_il16(l + 3)] = q8_1_quant(y.w, d, amax);
    } else {
        const uint32_t codes = (uint32_t) (uint8_t) q8_1_quant(y.x, d, amax) | (uint32_t) (uint8_t) q8_1_quant(y.y, d, amax) << 8 |
                               (uint32_t) (uint8_t) q8_1_quant(y.z, d, amax) << 16 |
                               (uint32_t) (uint8_t) q8_1_quant(y.w, d, amax) << 24;
        *reinterpret_cast<uint32_t*>(q->qs + (i4 & 7) * 4) = codes;
    }
    if ((i4 & 7) == 0) q->ds = q8_1_ds(d, sum);
}

/// rows of floats to q8_1 (n / 32 blocks a row), as native_quantize_q8_1 rounds
__global__ void __launch_bounds__(kTB) quant_kernel(const float* __restrict__ x, Q81* __restrict__ q, int n) {
    const int t = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int b = blockIdx.x * (kTB / 32) + warp;
    if (b >= n / 32) return;
    const float xi = x[(size_t) t * n + b * 32 + lane];
    const float amax = warp_max(fabsf(xi)), sum = warp_sum(xi);
    const float d = q8_1_finite(amax / 127.0f);
    Q81* y = q + (size_t) t * (n / 32) + b;
    y->qs[lane] = q8_1_quant(xi, d, amax);
    if (lane == 0) y->ds = q8_1_ds(d, sum);
}

// ---------------------------------------------------------------------------------------------- small float products
/// y[t][o] = W[o] . x[t] for a float matrix of a few rows (the sink's, a full layer's convolution taps'): a block an
/// output and a row
__global__ void __launch_bounds__(kTB) sproj_kernel(const float* __restrict__ W, const float* __restrict__ x,
                                                    float* __restrict__ y, int N, int n_out) {
    const int o = blockIdx.x, t = blockIdx.y;
    __shared__ float red[8][1];
    const float* w = W + (size_t) o * N;
    const float* xr = x + (size_t) t * N;
    float v[1] = {0.f};
    for (int i = threadIdx.x; i < N; i += kTB) v[0] += w[i] * xr[i];
    block_sums<1>(v, red);
    if (threadIdx.x == 0) y[(size_t) t * n_out + o] = v[0];
}

// ---------------------------------------------------------------------------------------------- the convolutions
/// The causal convolution's input `lag` rows before row t: this forward's rows, or the visit's state (the last C - 1
/// rows before the forward, oldest first)
__device__ __forceinline__ float conv_in(const float* raw, const float* state, int t, int lag, int n, int i, int C) {
    const int r = t - lag;
    return r >= 0 ? raw[(size_t) r * n + i] : state[(size_t) (C - 1 + r) * n + i];
}

/// A visit's state after a forward: the last C - 1 rows of [state | raw].  A thread a channel.
__global__ void conv_state_kernel(float* __restrict__ state, const float* __restrict__ raw, int n, int T, int C) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float keep[8];
    for (int j = 0; j < C - 1; ++j) keep[j] = conv_in(raw, state, T, C - 1 - j, n, i, C);   // rows T - (C-1) .. T - 1
    for (int j = 0; j < C - 1; ++j) state[(size_t) j * n + i] = keep[j];
}

/// Every visit's states after the first `rows` rows of a decode forward of several: conv_state_kernel's for each of
/// a visit's four convolutions, from the rows the forward left the visit (`raw`: [visits][K's | V's | the attention's
/// output | the MoE's], `cap` rows each; `state` as Engine::Visit::conv).  A block 256 of a visit's 2 KV + 2 N channels.
__global__ void __launch_bounds__(256) conv_commit_kernel(float* __restrict__ state, const float* __restrict__ raw, int KV,
                                                          int N, int C, int rows, int cap) {
    const int v = blockIdx.y, c = blockIdx.x * 256 + threadIdx.x, per = 2 * KV + 2 * N;
    if (c >= per) return;
    int n, i, before;   // the convolution's width, the channel in it, the channels of the convolutions before it
    if (c < 2 * KV) {
        n = KV;
        before = c < KV ? 0 : KV;
    } else {
        n = N;
        before = c < 2 * KV + N ? 2 * KV : 2 * KV + N;
    }
    i = c - before;
    float* st = state + (size_t) v * (C - 1) * per + (size_t) (C - 1) * before;
    const float* rw = raw + (size_t) v * cap * per + (size_t) cap * before;
    float keep[8];
    for (int j = 0; j < C - 1; ++j) keep[j] = conv_in(rw, st, rows, C - 1 - j, n, i, C);
    for (int j = 0; j < C - 1; ++j) st[(size_t) j * n + i] = keep[j];
}

/// K and V of T rows from their projections: y = (x + sum over lags of w[C - 1 - lag] x(t - lag) [+ the row's own taps
/// d[lag] x(t - lag), a full layer's]) / sqrt(C + 1 or 2 C + 1); then K's rows clipped to an L2 norm of `clip` a head, V's
/// RMS-normed a head.  Taps `w`: [n][C] (a channel's C values); `dyn`: [T][HK][C] or null.  A block a row, a side and a
/// head; D threads.
struct KvIn {
    const float* raw[2];     // K's and V's projections [T][HK * D]
    const float* state[2];   // their states [C - 1][HK * D]
    const float* w[2];
    const float* dyn[2];
    float* out[2];           // [T][HK * D]
    int HK, D, C;
    float clip, eps;
};
__global__ void kv_conv_kernel(KvIn a) {
    const int t = blockIdx.x, side = blockIdx.y, g = blockIdx.z, i = threadIdx.x;
    __shared__ float part[4];
    const int n = a.HK * a.D, ch = g * a.D + i;
    const float* raw = a.raw[side];
    const float* st = a.state[side];
    const float* w = a.w[side] + (size_t) ch * a.C;
    const float* dyn = a.dyn[side] ? a.dyn[side] + ((size_t) t * a.HK + g) * a.C : nullptr;
    float acc = 0.f;
    for (int lag = 0; lag < a.C; ++lag) {
        const float x = conv_in(raw, st, t, lag, n, ch, a.C);
        acc += (w[a.C - 1 - lag] + (dyn ? dyn[lag] : 0.f)) * x;
    }
    const float y = (raw[(size_t) t * n + ch] + acc) * rsqrtf((float) ((dyn ? 2 : 1) * a.C + 1));
    const float ss = lamina::kernels::block128_sum(y * y, part);
    float f;
    if (side == 0) f = fminf(1.0f, a.clip / (sqrtf(ss) + 1e-6f));
    else f = rsqrtf(ss / (float) a.D + a.eps);
    a.out[side][(size_t) t * n + ch] = y * f;
}

/// q of T rows: its projection times the layer's per-value scale, clipped to an L2 norm of `clip` a head, times tau in a
/// full layer.  A block a row and a head; D threads.
__global__ void q_clip_kernel(const float* __restrict__ raw, const float* __restrict__ scale, const float* __restrict__ tau,
                              float* __restrict__ q, int H, int D, float clip) {
    const int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    __shared__ float part[4];
    const float y = raw[((size_t) t * H + h) * D + i] * scale[(size_t) h * D + i];
    const float ss = lamina::kernels::block128_sum(y * y, part);
    const float f = fminf(1.0f, clip / (sqrtf(ss) + 1e-6f)) * (tau ? tau[t] : 1.0f);
    q[((size_t) t * H + h) * D + i] = y * f;
}

/// tau of T positions: 1 + alpha ln(max(1, (pos + 1) / floor)), a full layer's q and r scale
__global__ void tau_kernel(const int32_t* __restrict__ pos, float* __restrict__ tau, int T, float alpha, float floor_) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < T) tau[t] = 1.0f + alpha * logf(fmaxf(1.0f, ((float) pos[t] + 1.0f) / floor_));
}

/// The distance bias of T rows: r (the projection, R values a head, times the layer's scale, clipped to `clip`, times tau)
/// against the table: bias[t][h][d] = sum over r of table[d][r] r[h][r] - sink[t][h] for d < ext, and -sink at ext (a
/// cell further away, which has no bias).  A block a row and a head.
struct BiasIn {
    const float* raw;     // [T][H][R]
    const float* scale;   // [H][R]
    const float* table;   // [ext][R]
    const float* sink;    // [T][H]
    const float* tau;     // [T] or null
    float* bias;          // [T][H][ext + 1]
    int H, R, ext;
    float clip;
};
constexpr int kRelMax = 32;
__global__ void __launch_bounds__(kWide) bias_kernel(BiasIn a) {
    const int t = blockIdx.x, h = blockIdx.y;
    __shared__ float r[kRelMax];
    if (threadIdx.x < 32) {
        const int k = threadIdx.x;
        const float y = k < a.R ? a.raw[((size_t) t * a.H + h) * a.R + k] * a.scale[(size_t) h * a.R + k] : 0.f;
        const float ss = warp_sum(y * y);
        const float f = fminf(1.0f, a.clip / (sqrtf(ss) + 1e-6f)) * (a.tau ? a.tau[t] : 1.0f);
        if (k < a.R) r[k] = y * f;
    }
    __syncthreads();
    const float sink = a.sink[(size_t) t * a.H + h];
    float* b = a.bias + ((size_t) t * a.H + h) * (a.ext + 1);
    for (int d = threadIdx.x; d < a.ext; d += kWide) {   // (a thread a distance: R / 4 16-byte loads)
        const float* tb = a.table + (size_t) d * a.R;
        float s = 0.f;
        if (a.R % 4 == 0) {
            const float4* t4 = reinterpret_cast<const float4*>(tb);
            for (int k = 0; k < a.R / 4; ++k) {
                const float4 tv = t4[k];
                s += tv.x * r[4 * k] + tv.y * r[4 * k + 1] + tv.z * r[4 * k + 2] + tv.w * r[4 * k + 3];
            }
        } else {
            for (int k = 0; k < a.R; ++k) s += tb[k] * r[k];
        }
        b[d] = s - sink;
    }
    if (threadIdx.x == 0) b[a.ext] = -sink;
}

/// r of a prompt chunk's rows, for bias_ring_kernel: the projection times the layer's scale, clipped to `clip` a
/// head, times tau: rc [T][H][R].  A warp a row and head.
__global__ void __launch_bounds__(32) r_clip_kernel(const float* __restrict__ raw, const float* __restrict__ scale,
                                                    const float* __restrict__ tau, float* __restrict__ rc, int H, int R,
                                                    float clip) {
    const int t = blockIdx.x, h = blockIdx.y, k = threadIdx.x;
    const float y = k < R ? raw[((size_t) t * H + h) * R + k] * scale[(size_t) h * R + k] : 0.f;
    const float ss = warp_sum(y * y);
    const float f = fminf(1.0f, clip / (sqrtf(ss) + 1e-6f)) * (tau ? tau[t] : 1.0f);
    if (k < R) rc[((size_t) t * H + h) * R + k] = y * f;
}
/// A chunk's distance bias by cell, as attn_tc_kernel reads it: ring[t][h][(c - first) mod ext] = table[p - c] . r -
/// sink for the ext cells c at and before the row's position p (r as r_clip_kernel leaves it), and far[t][h] = -sink,
/// the bias of a cell further away.  The sums are bias_kernel's.  A block a row: a thread four cells, whose table
/// rows it holds for every head (R4 quarters of a row's R values; R4 0: any R, the rows read a head).  ext a
/// multiple of 4, H R <= kRingRs.
struct RingIn {
    const float* r;       // [T][H][R]
    const float* table;   // [ext][R]
    const float* sink;    // [T][H]
    const int32_t* pos;   // [T]
    float* ring;          // [T][H][ext]
    float* far;           // [T][H]
    long long first;      // the position of the cells' slot 0
    int H, R, ext;
};
constexpr int kRingRs = 1024;    // a row's r values, every head's
constexpr int kRingHeads = 64;
template <int R4>
__global__ void __launch_bounds__(256) bias_ring_kernel(RingIn a) {
    const int t = blockIdx.x, i = threadIdx.x;
    __shared__ float r[kRingRs];
    __shared__ float sk[kRingHeads];
    for (int k = i; k < a.H * a.R; k += (int) blockDim.x) r[k] = a.r[(size_t) t * a.H * a.R + k];
    if (i < a.H) {
        const float s = a.sink[(size_t) t * a.H + i];
        sk[i] = s;
        a.far[(size_t) t * a.H + i] = -s;
    }
    __syncthreads();
    const int e = (int) ((((long long) a.pos[t] - a.first) % a.ext + a.ext) % a.ext);   // the row's own cell
    for (int j4 = i; j4 < a.ext / 4; j4 += (int) blockDim.x) {
        const float* tb[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            int d = e - (4 * j4 + u);
            if (d < 0) d += a.ext;
            tb[u] = a.table + (size_t) d * a.R;
        }
        if constexpr (R4 > 0) {
            float4 tv[4][R4];
#pragma unroll
            for (int u = 0; u < 4; ++u)
#pragma unroll
                for (int k = 0; k < R4; ++k) tv[u][k] = ld4(tb[u], k);
            for (int h = 0; h < a.H; ++h) {
                const float* rh = r + h * a.R;
                float o[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int k = 0; k < R4; ++k) {
                    const float r0 = rh[4 * k], r1 = rh[4 * k + 1], r2 = rh[4 * k + 2], r3 = rh[4 * k + 3];
#pragma unroll
                    for (int u = 0; u < 4; ++u) o[u] += tv[u][k].x * r0 + tv[u][k].y * r1 + tv[u][k].z * r2 + tv[u][k].w * r3;
                }
                const float s = sk[h];
                st4(a.ring + ((size_t) t * a.H + h) * a.ext, j4, make_float4(o[0] - s, o[1] - s, o[2] - s, o[3] - s));
            }
        } else {
            for (int h = 0; h < a.H; ++h) {
                const float* rh = r + h * a.R;
                float o[4];
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    float s = 0.f;
                    for (int k = 0; k < a.R; ++k) s += tb[u][k] * rh[k];
                    o[u] = s - sk[h];
                }
                st4(a.ring + ((size_t) t * a.H + h) * a.ext, j4, make_float4(o[0], o[1], o[2], o[3]));
            }
        }
    }
}

/// A decode forward's q and bias in one launch (the sink's product, q_clip_kernel, tau_kernel and bias_kernel): q =
/// its projection times the layer's scale, clipped, times tau; bias[d] = table[d] . r - sink for d < ext, -sink at
/// ext, r the distance projection scaled, clipped, times tau, sink = w[h] . x.  A block a row and a head.
struct QbIn {
    const float* qraw;     // [T][H][D]
    const float* qscale;   // [H][D]
    const float* rraw;     // [T][H][R]
    const float* rscale;   // [H][R]
    const float* table;    // [ext][R]
    const float* sinkw;    // [H][N]
    const float* x;        // [T][N]: the q input, normed
    const int32_t* pos;    // [T]
    float* q;              // [T][H][D]
    float* bias;           // [T][H][ext + 1]
    int H, D, R, ext, N, full;
    float qclip, rclip, alpha, floor_;
};
__global__ void __launch_bounds__(kWide) q_bias_kernel(QbIn a) {
    const int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    __shared__ float red[32][3];
    __shared__ float tot[3];
    __shared__ float r[kRelMax];
    float v[3] = {0.f, 0.f, 0.f};
    if (i < a.N / 4) v[0] = dot4(ld4(a.sinkw + (size_t) h * a.N, i), ld4(a.x + (size_t) t * a.N, i));
    float qy = 0.f, ry = 0.f;
    if (i < a.D) {
        qy = a.qraw[((size_t) t * a.H + h) * a.D + i] * a.qscale[(size_t) h * a.D + i];
        v[1] = qy * qy;
    }
    if (i < a.R) {
        ry = a.rraw[((size_t) t * a.H + h) * a.R + i] * a.rscale[(size_t) h * a.R + i];
        v[2] = ry * ry;
    }
    block_sums_wide<3>(v, red, tot);
    const float tau = a.full ? 1.0f + a.alpha * logf(fmaxf(1.0f, ((float) a.pos[t] + 1.0f) / a.floor_)) : 1.0f;
    if (i < a.D) a.q[((size_t) t * a.H + h) * a.D + i] = qy * (fminf(1.0f, a.qclip / (sqrtf(v[1]) + 1e-6f)) * tau);
    if (i < a.R) r[i] = ry * (fminf(1.0f, a.rclip / (sqrtf(v[2]) + 1e-6f)) * tau);
    __syncthreads();
    const float sink = v[0];
    float* b = a.bias + ((size_t) t * a.H + h) * (a.ext + 1);
    for (int d = i; d < a.ext; d += (int) blockDim.x) {
        const float* tb = a.table + (size_t) d * a.R;
        float s = 0.f;
        if (a.R % 4 == 0) {
            for (int k = 0; k < a.R / 4; ++k) {
                const float4 tv = ld4(tb, k);
                s += tv.x * r[4 * k] + tv.y * r[4 * k + 1] + tv.z * r[4 * k + 2] + tv.w * r[4 * k + 3];
            }
        } else {
            for (int k = 0; k < a.R; ++k) s += tb[k] * r[k];
        }
        b[d] = s - sink;
    }
    if (i == 0) b[a.ext] = -sink;
}

// ---------------------------------------------------------------------------------------------- the cells
/// A visit's cells: K and V of each position, HK heads of D values, halves or q8_0 blocks.  A window visit keeps a ring
/// of `cap` cells (position p in slot p % cap, cap the window and a forward's rows); a full visit a range from position
/// `first`.
struct Cells {
    uint8_t* k;      // [cap][HK][D] halves, or [cap][HK][D / 32] q8_0 blocks
    uint8_t* v;
    int64_t first;   // a range's first position (0 for a ring)
    int64_t cap;     // cells
    int ring;        // 1: slot = pos % cap
    int q8;          // 1: q8_0 blocks (34 bytes a 32 values), else halves
};
__device__ __forceinline__ int64_t cell_slot(const Cells& c, int64_t pos) { return c.ring ? pos % c.cap : pos - c.first; }
__device__ __forceinline__ size_t cell_bytes(const Cells& c, int HK, int D) {
    return c.q8 ? (size_t) HK * (D / 32) * 34 : (size_t) HK * D * 2;
}

/// T rows' K and V into their cells (positions pos[t]).  A block a row, a side and a head; a warp a group of 32 values.
__global__ void __launch_bounds__(128) cells_put_kernel(Cells c, const float* __restrict__ k, const float* __restrict__ v,
                                                        const int32_t* __restrict__ pos, int HK, int D) {
    const int t = blockIdx.x, side = blockIdx.y, g = blockIdx.z;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t p = pos[t];
    if (!c.ring && (p < c.first || p >= c.first + c.cap)) return;   // another range's
    const float* src = (side ? v : k) + ((size_t) t * HK + g) * D;
    uint8_t* dst = (side ? c.v : c.k) + (size_t) cell_slot(c, p) * cell_bytes(c, HK, D);
    for (int b = warp; b < D / 32; b += 4) {
        const float x = src[b * 32 + lane];
        if (c.q8) {   // ggml's q8_0: d = amax / 127, round(x / d)
            const float amax = warp_max(fabsf(x));
            const float d = amax / 127.0f, id = d != 0.f ? 1.0f / d : 0.f;
            uint8_t* blk = dst + ((size_t) g * (D / 32) + b) * 34;
            blk[2 + lane] = (uint8_t) (int8_t) roundf(x * id);
            if (lane == 0) *(__half*) blk = __float2half(d);
        } else {
            ((__half*) dst)[(size_t) g * D + b * 32 + lane] = __float2half(x);
        }
    }
}

/// A decode forward's K and V in one launch: kv_conv_kernel's rows into their cells (cells_put_kernel's store), and
/// with `keep` (a forward of one row) the convolutions' states after it (conv_state_kernel's: a thread has its
/// channel's rows in hand).  A block a row, a side and a head; D = 128 threads.
struct KvCellsIn {
    KvIn kv;
    float* keep[2];       // the states to write, or null
    Cells cells;
    const int32_t* pos;   // [T]
};
__global__ void __launch_bounds__(128) kv_cells_kernel(KvCellsIn b) {
    const KvIn& a = b.kv;
    const int t = blockIdx.x, side = blockIdx.y, g = blockIdx.z, i = threadIdx.x;
    __shared__ float part[4];
    const int n = a.HK * a.D, ch = g * a.D + i;
    const float* raw = a.raw[side];
    const float* st = a.state[side];
    const float* w = a.w[side] + (size_t) ch * a.C;
    const float* dyn = a.dyn[side] ? a.dyn[side] + ((size_t) t * a.HK + g) * a.C : nullptr;
    float xs[8];   // the channel's rows, the newest first
    float acc = 0.f;
    for (int lag = 0; lag < a.C; ++lag) {
        xs[lag] = conv_in(raw, st, t, lag, n, ch, a.C);
        acc += (w[a.C - 1 - lag] + (dyn ? dyn[lag] : 0.f)) * xs[lag];
    }
    const float y = (xs[0] + acc) * rsqrtf((float) ((dyn ? 2 : 1) * a.C + 1));
    const float ss = lamina::kernels::block128_sum(y * y, part);
    const float val = y * (side == 0 ? fminf(1.0f, a.clip / (sqrtf(ss) + 1e-6f)) : rsqrtf(ss / (float) a.D + a.eps));
    a.out[side][(size_t) t * n + ch] = val;
    if (b.keep[side])   // (one row: the state's rows shift by one)
        for (int j = 0; j < a.C - 1; ++j) b.keep[side][(size_t) j * n + ch] = xs[a.C - 2 - j];
    const Cells& c = b.cells;
    const int64_t p = b.pos[t];
    if (!c.ring && (p < c.first || p >= c.first + c.cap)) return;   // another range's
    uint8_t* dst = (side ? c.v : c.k) + (size_t) cell_slot(c, p) * cell_bytes(c, a.HK, a.D);
    const int warp = i >> 5, lane = i & 31;
    if (c.q8) {   // ggml's q8_0: d = amax / 127, round(x / d)
        const float amax = warp_max(fabsf(val));
        const float d = amax / 127.0f, id = d != 0.f ? 1.0f / d : 0.f;
        uint8_t* blk = dst + ((size_t) g * (a.D / 32) + warp) * 34;
        blk[2 + lane] = (uint8_t) (int8_t) roundf(val * id);
        if (lane == 0) *(__half*) blk = __float2half(d);
    } else {
        ((__half*) dst)[(size_t) g * a.D + i] = __float2half(val);
    }
}

// ---------------------------------------------------------------------------------------------- the attention
/// A decode forward's rows against a visit's cells, in stretches: attn_rows_kernel (with the chunks' kernel, below)
/// keeps for a row and query head each stretch's largest logit m, the sum of exp(logit - m) and that-weighted sum of
/// V (D values): the softmax's partial sums, which attn_out_kernel joins over the stretches (another card's cells send
/// theirs joined: attn_join_kernel).  logit = q . k / sqrt(D) + bias[min(distance, ext)]; a cell after the row, or
/// outside a window layer's `win` positions, is left out.
struct AttnIn {
    Cells c;
    const float* q;         // [T][H][D]
    const float* bias;      // [T][H][ext + 1]
    const int32_t* pos;     // [T]
    const int32_t* live;    // the cells in use: a range's count (its positions first .. first + *live - 1), a ring's last position + 1
    const int32_t* dyn;     // null: a ring's every slot, `chunk` cells a stretch.  Else the stretches from the device (a
                            // launch captured in a graph serves any number of cells): {cells a stretch, stretches in use,
                            // the first slot} (a ring's go on past its end to its start), or with `canon` {the first cell
                            // in use, stretches in use, the first one's number}
    float* part;            // [T][H][n_chunks][D + 2]: m, s, the D sums
    int H, HK, D, ext, win; // win: a window layer's cells (0: every cell)
    int chunk, n_chunks;
    int canon;              // a range's stretches are stretch_lo's
};
/// A range's stretches that do not depend on how many cells it has (so a row's sums are the same whatever was scored
/// with it, and whenever): stretch i starts at this cell - 32 of 128 cells, 64 of 512, then of 2,048 (590 for 2^20).
__host__ __device__ __forceinline__ long long stretch_lo(int i) {
    return i < 32 ? i * 128LL : i < 96 ? 4096 + (i - 32) * 512LL : 36864 + (i - 96) * 2048LL;
}
constexpr int kQH = 4;      // query heads a K/V head at most

/// The stretches' partial sums joined with the zero sink, times sigmoid(gate): the attention's output of T rows as
/// floats ([T][H * D]) and as its q8_1 image.  A block a row and a query head; D threads (a warp a q8_1 block).
struct AttnOutIn {
    const float* part;    // [T][H][n_chunks][D + 2]
    const float* gate;    // [T][H * D], the gate's projection
    float* out;           // [T][H * D] or null
    Q81* q;               // [T][H * D / 32] or null
    const int32_t* dyn;   // null, or {chunk, stretches in use}
    int H, D, n_chunks;
    const float* extra;   // [n_extra][T][H][D + 2]: other cards' sums over their cells (attn_join_kernel), or null
    int n_extra, T;
};
__global__ void __launch_bounds__(128) attn_out_kernel(AttnOutIn a) {
    const int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    const int used = a.dyn ? a.dyn[1] : a.n_chunks;
    const float* p = a.part + ((size_t) t * a.H + h) * a.n_chunks * (a.D + 2);
    float mx = 0.f;   // the sink's logit
    for (int c = 0; c < used; ++c) mx = fmaxf(mx, p[(size_t) c * (a.D + 2)]);
    for (int e = 0; e < a.n_extra; ++e) {
        const float* pe = a.extra + (((size_t) e * a.T + t) * a.H + h) * (a.D + 2);
        if (pe[1] != 0.f) mx = fmaxf(mx, pe[0]);
    }
    float sum = __expf(-mx), acc = 0.f;
    for (int c = 0; c < used; ++c) {
        const float* pc = p + (size_t) c * (a.D + 2);
        if (pc[1] == 0.f) continue;
        const float w = __expf(pc[0] - mx);
        sum += w * pc[1];
        acc += w * pc[2 + i];
    }
    for (int e = 0; e < a.n_extra; ++e) {
        const float* pe = a.extra + (((size_t) e * a.T + t) * a.H + h) * (a.D + 2);
        if (pe[1] == 0.f) continue;
        const float w = __expf(pe[0] - mx);
        sum += w * pe[1];
        acc += w * pe[2 + i];
    }
    const size_t at = ((size_t) t * a.H + h) * a.D + i;
    const float y = acc / sum / (1.0f + __expf(-a.gate[at]));
    if (a.out) a.out[at] = y;
    if (a.q == nullptr) return;
    const float amax = warp_max(fabsf(y)), s = warp_sum(y);
    const float d = q8_1_finite(amax / 127.0f);
    Q81* q = a.q + at / 32;
    q->qs[i & 31] = q8_1_quant(y, d, amax);
    if ((i & 31) == 0) q->ds = q8_1_ds(d, s);
}

/// A card's stretches joined into one partial sum a row and a query head, for the main card's attn_out_kernel: the
/// largest logit, the sum of exp(logit - it), that-weighted V (no sink, no gate: the main card's).  sum [T][H][D + 2];
/// a head whose cells are none: s = 0.  A block a row and a query head; D threads.
struct JoinIn {
    const float* part;    // [T][H][n_chunks][D + 2]
    float* sum;           // [T][H][D + 2]
    const int32_t* dyn;   // {chunk, stretches in use}
    int H, D, n_chunks;
};
__global__ void __launch_bounds__(128) attn_join_kernel(JoinIn a) {
    const int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    const int used = a.dyn[1];
    const float* p = a.part + ((size_t) t * a.H + h) * a.n_chunks * (a.D + 2);
    float mx = -FLT_MAX;
    for (int c = 0; c < used; ++c)
        if (p[(size_t) c * (a.D + 2) + 1] != 0.f) mx = fmaxf(mx, p[(size_t) c * (a.D + 2)]);
    float sum = 0.f, acc = 0.f;
    for (int c = 0; c < used; ++c) {
        const float* pc = p + (size_t) c * (a.D + 2);
        if (pc[1] == 0.f) continue;
        const float w = __expf(pc[0] - mx);
        sum += w * pc[1];
        acc += w * pc[2 + i];
    }
    float* o = a.sum + ((size_t) t * a.H + h) * (a.D + 2);
    if (i == 0) {
        o[0] = mx;
        o[1] = sum;
    }
    o[2 + i] = acc;
}
/// -sink of each row and head out of its bias row (its last value): what a card's cells take for a bias
__global__ void nsink_kernel(const float* __restrict__ bias, float* __restrict__ ns, int rows, int ext) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows) ns[i] = bias[(size_t) i * (ext + 1) + ext];
}

// ---------------------------------------------------------------------------------------------- after a sublayer
/// A sublayer's output through its causal convolution into the stream: y = (a + conv(a)) / sqrt(C + 1), then x = y (a
/// block's first attention: the stream starts again) or x += y.  `raw` the sublayer's rows [T][N], `state` its visit's.
__global__ void __launch_bounds__(kTB) post_conv_kernel(const float* __restrict__ raw, const float* __restrict__ state,
                                                        const float* __restrict__ w, float* __restrict__ x, int N, int C,
                                                        int fresh) {
    const int t = blockIdx.y, i = blockIdx.x * kTB + threadIdx.x;
    if (i >= N) return;
    float acc = 0.f;
    for (int lag = 0; lag < C; ++lag) acc += w[(size_t) i * C + C - 1 - lag] * conv_in(raw, state, t, lag, N, i, C);
    const float y = (raw[(size_t) t * N + i] + acc) * rsqrtf((float) (C + 1));
    float* o = x + (size_t) t * N + i;
    *o = fresh ? y : *o + y;
}

/// The same for a forward of one row, with the launches around it: with `rows` the sublayer's row is the routed
/// experts' rows by their weights (expert_sum_kernel's, written to `raw` as well); the convolution's state is kept
/// (conv_state_kernel's).  A thread a channel.
__global__ void __launch_bounds__(kTB) post_one_kernel(const float* __restrict__ rows, const float* __restrict__ w,
                                                       float* __restrict__ raw, float* __restrict__ state,
                                                       const float* __restrict__ cw, float* __restrict__ x, int N, int C,
                                                       int top_k, int fresh) {
    const int i = blockIdx.x * kTB + threadIdx.x;
    if (i >= N) return;
    float xs[8];   // the channel's rows, the newest first
    if (rows) {
        float acc = 0.f;
        for (int k = 0; k < top_k; ++k) acc += rows[(size_t) k * N + i] * w[k];
        raw[i] = acc;
        xs[0] = acc;
    } else {
        xs[0] = raw[i];
    }
    for (int lag = 1; lag < C; ++lag) xs[lag] = state[(size_t) (C - 1 - lag) * N + i];
    float acc = 0.f;
    for (int lag = 0; lag < C; ++lag) acc += cw[(size_t) i * C + C - 1 - lag] * xs[lag];
    const float y = (xs[0] + acc) * rsqrtf((float) (C + 1));
    float* o = x + i;
    *o = fresh ? y : *o + y;
    for (int j = 0; j < C - 1; ++j) state[(size_t) j * N + i] = xs[C - 2 - j];
}

// ---------------------------------------------------------------------------------------------- the router
/// T rows' router scores: p = sigmoid(W[e] . x) in floats.  A block a row and an expert.
__global__ void __launch_bounds__(kTB) route_dot_kernel(const float* __restrict__ W, const float* __restrict__ x,
                                                        float* __restrict__ prob, int E, int N) {
    const int e = blockIdx.x, t = blockIdx.y;
    __shared__ float red[8][1];
    const float* w = W + (size_t) e * N;
    const float* xr = x + (size_t) t * N;
    float v[1] = {0.f};
    for (int i = threadIdx.x; i < N; i += kTB) v[0] += w[i] * xr[i];
    block_sums<1>(v, red);
    if (threadIdx.x == 0) prob[(size_t) t * E + e] = 1.0f / (1.0f + __expf(-v[0]));
}

/// T rows' experts picked and grouped, one block (a warp a row, T <= 8): the top_k largest p + bias in that order,
/// their p normalized (/ (sum + 1e-20)) times `scale` (ids, w: [T][top_k]); then the routed entries grouped by expert
/// for native_expert_grouped: a group an expert in use (its blob's address: the layer's slot of it), its entries the
/// rows that route to it; entry (t, k) writes row t * top_k + k.  The groups in the order of their first entry.  An
/// expert with no slot (slot_of < 0) gets no group: its entries are the CPU's (n_miss counts them).
struct PlanIn {
    const float* prob;         // [T][E]
    const float* bias;         // [E] or null
    const int32_t* slot_of;    // [E]
    unsigned long long slots;  // the layer's slots, `stride` apart
    size_t stride;
    int32_t* ids;              // [T][top_k]
    float* w;                  // [T][top_k]
    unsigned long long* grp_ptr;
    int32_t* grp_start;        // [n + 1]
    int32_t* n_groups;
    int32_t* ent_dst;
    int32_t* ent_tok;
    int32_t* n_miss;           // entries without a slot
    int T, E, top_k;
    float scale;
};
constexpr int kExpertMax = 256;
constexpr int kTopMax = 32;
__global__ void __launch_bounds__(kTB) pick_plan_kernel(PlanIn a) {
    __shared__ float sel[kMaxT][kExpertMax];
    __shared__ int32_t sid[kTB];
    __shared__ int32_t lead[kTB], rank_[kTB], cnt[kTB], start[kTB + 1];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = a.T * a.top_k;
    if (warp < a.T) {   // this warp's row
        const float* prob = a.prob + (size_t) warp * a.E;
        for (int e = lane; e < a.E; e += 32) sel[warp][e] = prob[e] + (a.bias ? a.bias[e] : 0.f);
        __syncwarp();
        float sum = 0.f;
        float pw[kTopMax];
        for (int k = 0; k < a.top_k; ++k) {
            float best = -FLT_MAX;
            int bi = 0;
            for (int e = lane; e < a.E; e += 32)
                if (sel[warp][e] > best) {
                    best = sel[warp][e];
                    bi = e;
                }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {   // the largest, the lower index of equals
                const float ob = __shfl_xor_sync(0xffffffffu, best, o);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (ob > best || (ob == best && oi < bi)) {
                    best = ob;
                    bi = oi;
                }
            }
            if (lane == 0) {
                sid[warp * a.top_k + k] = bi;
                sel[warp][bi] = -FLT_MAX;
                pw[k] = prob[bi];
                sum += pw[k];
            }
            __syncwarp();
        }
        if (lane == 0) {
            const float r = a.scale / (sum + 1e-20f);
            for (int k = 0; k < a.top_k; ++k) {
                a.ids[(size_t) warp * a.top_k + k] = sid[warp * a.top_k + k];
                a.w[(size_t) warp * a.top_k + k] = pw[k] * r;
            }
        }
    }
    __syncthreads();
    const int i = threadIdx.x;
    const int e = i < n ? sid[i] : -1;
    const bool held = i < n && a.slot_of[e] >= 0;
    // an entry's leader: the first entry of its expert
    int l = i;
    if (held)
        for (int j = 0; j < i; ++j)
            if (sid[j] == e) {
                l = j;
                break;
            }
    lead[i] = held ? l : -1;
    __syncthreads();
    // a leader's group: the leaders before it; its entries: those it leads
    int g = 0, c = 0, before = 0;
    if (held && l == i) {
        for (int j = 0; j < i; ++j) g += lead[j] == j;
        for (int j = i; j < n; ++j) c += lead[j] == i;
    }
    if (held)
        for (int j = 0; j < i; ++j) before += lead[j] == l;   // this entry's place among its leader's
    rank_[i] = held && l == i ? g : -1;
    cnt[i] = c;
    __syncthreads();
    if (held && l == i) {   // a group's first entry: the entries of the groups before it
        int at = 0;
        for (int j = 0; j < n; ++j) at += rank_[j] >= 0 && rank_[j] < g ? cnt[j] : 0;
        start[g] = at;
        a.grp_start[g] = at;
    }
    if (i == 0) {
        int ng = 0, nm = 0;
        for (int j = 0; j < n; ++j) {
            ng += lead[j] == j;
            nm += lead[j] < 0;
        }
        *a.n_groups = ng;
        *a.n_miss = nm;
        a.grp_start[ng] = n - nm;
    }
    __syncthreads();
    if (!held) return;
    const int gi = rank_[l];
    if (l == i) a.grp_ptr[gi] = a.slots + (unsigned long long) a.slot_of[e] * a.stride;
    const int at = start[gi] + before;
    a.ent_tok[at] = i / a.top_k;
    a.ent_dst[at] = i;
}

/// One row's experts picked and grouped (pick_plan_kernel's result for T = 1, where every routed entry is a group of
/// its own): an expert's rank among the E (the largest p + bias first, the lower index of equals), counted by two
/// threads where 2 E fit the block; the top_k first take their places.
__global__ void __launch_bounds__(kTB) pick_one_kernel(PlanIn a) {
    __shared__ float sel[kExpertMax];
    __shared__ float pw[kTopMax];
    __shared__ int32_t sid[kTopMax], held[kTopMax];
    const int i = threadIdx.x;
    if (i < a.E) sel[i] = a.prob[i] + (a.bias ? a.bias[i] : 0.f);
    if (i < a.top_k) {   // (a row of NaN ranks nothing: its picks stay in range)
        sid[i] = 0;
        pw[i] = 0.f;
        held[i] = 0;
    }
    __syncthreads();
    const int per = 2 * a.E <= kTB ? 2 : 1;
    const int e = i / per, half = (a.E + 1) / 2;
    int rank = 0;
    if (e < a.E) {
        const float me = sel[e];
        const int j0 = per == 2 && (i & 1) ? half : 0, j1 = per == 2 && !(i & 1) ? half : a.E;
        for (int j = j0; j < j1; ++j) {
            const float o = sel[j];
            rank += (o > me || (o == me && j < e)) ? 1 : 0;
        }
    }
    if (per == 2) rank += __shfl_xor_sync(0xffffffffu, rank, 1);
    if (e < a.E && (per == 1 || !(i & 1)) && rank < a.top_k) {
        sid[rank] = e;
        pw[rank] = a.prob[e];
        held[rank] = a.slot_of[e] >= 0 ? 1 : 0;
    }
    __syncthreads();
    if (i >= a.top_k) return;
    float sum = 0.f;
    int before = 0, ng = 0;
    for (int k = 0; k < a.top_k; ++k) {
        sum += pw[k];
        before += k < i ? held[k] : 0;
        ng += held[k];
    }
    a.ids[i] = sid[i];
    a.w[i] = pw[i] * (a.scale / (sum + 1e-20f));
    if (held[i]) {
        a.grp_ptr[before] = a.slots + (unsigned long long) a.slot_of[sid[i]] * a.stride;
        a.grp_start[before] = before;
        a.ent_tok[before] = 0;
        a.ent_dst[before] = i;
    }
    if (i == 0) {
        *a.n_groups = ng;
        *a.n_miss = a.top_k - ng;
        a.grp_start[ng] = ng;
    }
}

/// The MoE's output of T rows: the routed experts' rows by their weights, in the router's order.  rows [T * top_k][N].
/// `dscale` ([E], with `ids`): each expert's scale of its down rows, where the rows come without it (a prompt chunk's).
__global__ void __launch_bounds__(kTB) expert_sum_kernel(const float* __restrict__ rows, const float* __restrict__ w,
                                                         float* __restrict__ y, int N, int top_k,
                                                         const float* __restrict__ dscale, const int32_t* __restrict__ ids) {
    const int t = blockIdx.y, i = blockIdx.x * kTB + threadIdx.x;
    if (i >= N) return;
    float acc = 0.f;
    for (int k = 0; k < top_k; ++k) {
        const size_t e = (size_t) t * top_k + k;
        acc += rows[e * N + i] * (w[e] * (dscale ? dscale[ids[e]] : 1.0f));
    }
    y[(size_t) t * N + i] = acc;
}

// ---------------------------------------------------------------------------------------------- a prompt chunk's
/// A window visit's cells in position order for a chunk at positions p0 ..: range slot j is position p0 - (win - 1) +
/// j.  The win - 1 cells before the chunk come from the ring (a position before 0: left as it is, never read).  A
/// block a cell and a side.
__global__ void __launch_bounds__(256) ring_to_range_kernel(Cells ring, uint8_t* __restrict__ rk, uint8_t* __restrict__ rv,
                                                            long long p0, int win, int bytes) {
    const long long pos = p0 - (win - 1) + blockIdx.x;
    if (pos < 0) return;
    const uint32_t* src = (const uint32_t*) ((blockIdx.y ? ring.v : ring.k) + (size_t) (pos % ring.cap) * bytes);
    uint32_t* dst = (uint32_t*) ((blockIdx.y ? rv : rk) + (size_t) blockIdx.x * bytes);
    for (int i = threadIdx.x; i < bytes / 4; i += blockDim.x) dst[i] = src[i];
}
/// The chunk's last cells back into the ring (its n rows' cells are range slots win - 1 ..): the positions the ring
/// holds after the chunk that are the chunk's own.  A block a cell and a side, the last cell first.
__global__ void __launch_bounds__(256) range_to_ring_kernel(Cells ring, const uint8_t* __restrict__ rk,
                                                            const uint8_t* __restrict__ rv, long long p0, int n, int win,
                                                            int bytes) {
    const long long pos = p0 + n - 1 - blockIdx.x;
    if (pos < p0) return;
    const uint32_t* src = (const uint32_t*) ((blockIdx.y ? rv : rk) + (size_t) (pos - p0 + win - 1) * bytes);
    uint32_t* dst = (uint32_t*) ((blockIdx.y ? ring.v : ring.k) + (size_t) (pos % ring.cap) * bytes);
    for (int i = threadIdx.x; i < bytes / 4; i += blockDim.x) dst[i] = src[i];
}

/// T rows' experts picked, a warp a row, as pick_plan_kernel picks them: ids, w [T][top_k].
struct PickIn {
    float* prob;         // [T][E]; with `logits` the router's products, made the scores here
    const float* bias;   // [E] or null
    int32_t* ids;
    float* w;
    int E, top_k;
    float scale;
    int logits;
};
__global__ void __launch_bounds__(32) pick_kernel(PickIn a) {
    const int t = blockIdx.x, lane = threadIdx.x;
    __shared__ float sel[kExpertMax];
    float* prob = a.prob + (size_t) t * a.E;
    for (int e = lane; e < a.E; e += 32) {
        if (a.logits) prob[e] = 1.0f / (1.0f + __expf(-prob[e]));
        sel[e] = prob[e] + (a.bias ? a.bias[e] : 0.f);
    }
    __syncwarp();
    float sum = 0.f;
    float pw[kTopMax];
    int pid[kTopMax];
    for (int k = 0; k < a.top_k; ++k) {
        float best = -FLT_MAX;
        int bi = 0;
        for (int e = lane; e < a.E; e += 32)
            if (sel[e] > best) {
                best = sel[e];
                bi = e;
            }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {   // the largest, the lower index of equals
            const float ob = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ob > best || (ob == best && oi < bi)) {
                best = ob;
                bi = oi;
            }
        }
        if (lane == 0) {
            pid[k] = bi;
            sel[bi] = -FLT_MAX;
            pw[k] = prob[bi];
            sum += pw[k];
        }
        __syncwarp();
    }
    if (lane == 0) {
        const float r = a.scale / (sum + 1e-20f);
        for (int k = 0; k < a.top_k; ++k) {
            a.ids[(size_t) t * a.top_k + k] = pid[k];
            a.w[(size_t) t * a.top_k + k] = pw[k] * r;
        }
    }
}

/// A chunk's routed entries after their gate/up products: h = beta tanh(g s / beta) sigmoid(g s) * lbeta tanh(u s /
/// lbeta), s the entry's expert's scale (`scale` [E], or null: 1).  gu [n][2 F], the gate first; h [n][F].
__global__ void situ_rows_kernel(const float* __restrict__ gu, const int32_t* __restrict__ ent_exp,
                                 const float* __restrict__ scale, float* __restrict__ h, long long n, int F, float beta,
                                 float lbeta) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * F) return;
    const long long r = i / F;
    const int k = (int) (i % F);
    const float s = scale ? scale[ent_exp[r]] : 1.0f;
    const float g = gu[r * 2 * F + k] * s, u = gu[r * 2 * F + F + k] * s;
    h[i] = beta * tanhf(g / beta) / (1.0f + __expf(-g)) * (lbeta * tanhf(u / lbeta));
}

// ---------------------------------------------------------------------------------------------- a chunk's loop gates
/// The gates' input of T rows for a GEMM: z[t] = [u / rms(u) ; m / rms(m)] (2 N floats).  A block a row.
__global__ void __launch_bounds__(kTB) ctrl_cat_kernel(const float* __restrict__ u, const float* __restrict__ m,
                                                       float* __restrict__ z, int N) {
    const int t = blockIdx.x;
    __shared__ float red[8][2];
    const float* ur = u + (size_t) t * N;
    const float* mr = m + (size_t) t * N;
    float v[2] = {0.f, 0.f};
    for (int i = threadIdx.x; i < N; i += kTB) {
        v[0] += ur[i] * ur[i];
        v[1] += mr[i] * mr[i];
    }
    block_sums<2>(v, red);
    const float iu = rsqrtf(v[0] / (float) N + 1e-6f), im = rsqrtf(v[1] / (float) N + 1e-6f);
    float* zr = z + (size_t) t * 2 * N;
    for (int i = threadIdx.x; i < N; i += kTB) {
        zr[i] = ur[i] * iu;
        zr[N + i] = mr[i] * im;
    }
}
/// out = (1 - tanh(sf)) u + tanh(sg) m over n values (sg, sf: the gain's and the forget's products)
__global__ void ctrl_gate_kernel(const float* __restrict__ u, const float* __restrict__ m, const float* __restrict__ sg,
                                 const float* __restrict__ sf, float* __restrict__ out, long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (1.0f - tanhf(sf[i])) * u[i] + tanhf(sg[i]) * m[i];
}

// ---------------------------------------------------------------------------------------------- a chunk's attention
// On the tensor cores: mma.m16n8k16 (FP16 in, FP32 sums), operands through ldmatrix from shared memory whose rows are
// padded so that ldmatrix's eight rows fall on distinct banks (the Hy4 path's tiles: src/hy4/hy4_prompt.cuh).
__device__ __forceinline__ void mma16816(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned h2(float a, float b) {
    const __half2 v = __floats2half2_rn(a, b);
    return *reinterpret_cast<const unsigned*>(&v);
}
/// four q8_0 codes times their block's scale, as halves
__device__ __forceinline__ uint2 q8_widen4(uint32_t w, float d) {
    return make_uint2(h2((float) (int8_t) (w & 0xff) * d, (float) (int8_t) ((w >> 8) & 0xff) * d),
                      h2((float) (int8_t) ((w >> 16) & 0xff) * d, (float) (int8_t) (w >> 24) * d));
}

constexpr int kTcCells = 32;   // cells a tile
constexpr int kTcPad = 136;    // a row of 128 halves in shared memory (272 bytes)
constexpr int kTcRows = 32;    // a decode forward's rows and query heads a block (attn_rows_kernel)
constexpr int kTcPPad = 40;    // its probability row, halves (80 bytes)

struct __align__(16) TcSmem {
    __half q[kTcRows * kTcPad];
    __half k[kTcCells * kTcPad];
    __half v[kTcCells * kTcPad];
    __half p[kTcRows * kTcPPad];
    float red[2][4][kTcRows];   // [max | sum][the cells' quarter][row]
};

/// A chunk's attention of one query head on the tensor cores, a warp 16 rows: block (b, h, s) takes rows 16 NW b ..
/// of head h over the cells its rows see, in tiles of 32 on the range's own grid (K and V of the head's K/V head as
/// halves in shared memory; with Q8 the q8_0 blocks arrive as stored and are widened there).  A warp scores its rows
/// against the tile (q in registers as the products' row operands, K through ldmatrix), turns the scores into
/// probabilities where they stand - a product's sums lie in the lanes as the next one's row operand - and adds the
/// tile's V rows.  Halves: the next tile arrives behind the products in the other of two buffers, one barrier a tile.
/// The cells are a range in position order (c.first): a full visit's from position 0, a window visit's from win - 1
/// before the chunk; a row sees those at or before it, and within `win` of it.  logit = q . k / sqrt(D) + the bias of
/// the cell's distance: ring[slot mod ext] within ext of the row (bias_ring_kernel), `far` past it; a tile every row
/// of the block sees whole and past the table takes neither a mask nor a load.  An online softmax that starts from
/// the zero sink (m = 0, l = 1); the output, divided by l and times sigmoid(gate), to out [T][H * D].  With a grid of
/// several segments (gridDim.z) a block takes its share of the tiles, for a few rows over many cells: more blocks
/// than the rows and heads make, the segments' partial sums joined by tc_merge_kernel.  D is 128, ext a multiple of
/// 32.  Grid (ceil(T / (16 NW)), H, segments), block 32 NW.
struct TcIn {
    Cells c;
    const float* q;        // [T][H][D]
    const float* ring;     // [T][H][ext]: the bias by cell; not read with ext 0
    const float* far;      // [T][H]: the bias of a cell past the table (-sink)
    const float* gate;     // [T][H * D]
    const int32_t* pos;    // [T]
    float* out;            // [T][H * D]
    int T, H, HK, ext, win;
    long long cells;       // the range's cells in use
    float* part;           // not null: [T][H][D + 2], a row and head's partial sums (the largest logit m, the sum of
                           // exp(logit - m), that-weighted V) instead of `out`: another card's share, or this one's
                           // where others' are joined to it (tc_merge_kernel)
    int no_sink;           // the sums without the zero sink (a card's cells alone)
    long long lo;          // a full visit's first cell on this card (the cells before it are other cards')
    float* seg;            // several segments: [segments - 1][T][H][D + 2], the sums of those after the first (the
                           // first one's to `part`)
};
constexpr int kTcSegs = 8;   // segments at most
template <int NW, bool Q8>
__global__ void __launch_bounds__(NW * 32) attn_tc_kernel(TcIn a) {
    constexpr int D = 128, ROWS = 16 * NW, NT = 32 * NW;
    constexpr int kRawCell = (D / 32) * 34;                                 // a cell's q8_0 blocks of one K/V head
    __shared__ __align__(16) __half tile[Q8 ? 1 : 2][2][kTcCells * kTcPad];   // [buffer][K | V]
    __shared__ __align__(16) uint8_t raw[2][Q8 ? kTcCells * kRawCell : 16];   // Q8: [K | V] as stored
    const int t0 = blockIdx.x * ROWS, h = blockIdx.y, g = h / (a.H / a.HK);
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
    const int li = lane >> 3, lr = lane & 7, tig = lane & 3;
    const int nrows = min(ROWS, a.T - t0);
    const long long p_first = a.pos[t0], p_last = a.pos[t0 + nrows - 1];
    const long long first = (long long) a.c.first;
    // the block's cells: slots [s_true, s_lo + cnt) of the range, its tiles from the grid's slot s_lo at or before
    const long long s_true = max(max(0LL, (a.win ? max(0LL, p_first - a.win + 1) : 0LL) - first), a.lo);
    const long long s_lo = s_true / kTcCells * kTcCells;
    const int c_min = (int) (s_true - s_lo);
    const int cnt = (int) (min(p_last - first + 1, a.cells) - s_lo);
    const size_t cbytes = cell_bytes(a.c, a.HK, D);
    const int n_all = (max(cnt, 0) + kTcCells - 1) / kTcCells, n_seg = (int) gridDim.z, sg = (int) blockIdx.z;
    const int it0 = (int) ((long long) n_all * sg / n_seg), n_tiles = (int) ((long long) n_all * (sg + 1) / n_seg);
    auto fetch = [&](int buf, int base) {
        if constexpr (Q8) {   // 17 chunks of 8 bytes a cell and side
            for (int i = tid; i < 2 * kTcCells * (kRawCell / 8); i += NT) {
                const int side = i >= kTcCells * (kRawCell / 8), j = i - side * kTcCells * (kRawCell / 8);
                const int c = j / (kRawCell / 8), ch = j % (kRawCell / 8);
                uint8_t* d = raw[side] + c * kRawCell + ch * 8;
                if (base + c < cnt)
                    cp_async8(d, (side ? a.c.v : a.c.k) + (size_t) (s_lo + base + c) * cbytes + (size_t) g * kRawCell + ch * 8);
                else
                    *reinterpret_cast<unsigned long long*>(d) = 0ull;
            }
        } else {              // 16 chunks of 16 bytes a cell and side
            for (int i = tid; i < kTcCells * 16; i += NT) {
                const int c = i / 16, ch = i % 16;
                __half* kd = tile[buf][0] + c * kTcPad + ch * 8;
                __half* vd = tile[buf][1] + c * kTcPad + ch * 8;
                if (base + c < cnt) {
                    const size_t at = (size_t) (s_lo + base + c) * cbytes + (size_t) g * D * 2 + ch * 16;
                    cp_async16(kd, a.c.k + at);
                    cp_async16(vd, a.c.v + at);
                } else {
                    *reinterpret_cast<uint4*>(kd) = make_uint4(0, 0, 0, 0);
                    *reinterpret_cast<uint4*>(vd) = make_uint4(0, 0, 0, 0);
                }
            }
        }
        cp_commit();
    };
    if (it0 < n_tiles) fetch(0, it0 * kTcCells);
    const int rA = w * 16 + (lane >> 2), rB = rA + 8;
    const bool inA = rA < nrows, inB = rB < nrows;
    const size_t hA = (size_t) (t0 + (inA ? rA : 0)) * a.H + h, hB = (size_t) (t0 + (inB ? rB : 0)) * a.H + h;
    // q as the products' row operands: step ks is dims 16 ks .. +15
    unsigned qa[8][4];
    {
        const float* qA = a.q + hA * D + 2 * tig;
        const float* qB = a.q + hB * D + 2 * tig;
        const float2 z = make_float2(0.f, 0.f);
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            const float2 a0 = inA ? *reinterpret_cast<const float2*>(qA + 16 * ks) : z;
            const float2 a1 = inB ? *reinterpret_cast<const float2*>(qB + 16 * ks) : z;
            const float2 a2 = inA ? *reinterpret_cast<const float2*>(qA + 16 * ks + 8) : z;
            const float2 a3 = inB ? *reinterpret_cast<const float2*>(qB + 16 * ks + 8) : z;
            qa[ks][0] = h2(a0.x, a0.y);
            qa[ks][1] = h2(a1.x, a1.y);
            qa[ks][2] = h2(a2.x, a2.y);
            qa[ks][3] = h2(a3.x, a3.y);
        }
    }
    const float scale = rsqrtf((float) D);
    // a row's own slot from s_lo (the first row's: the block's nearest), and the bias past the table
    const int sA = inA ? (int) ((long long) a.pos[t0 + rA] - first - s_lo) : 0;
    const int sB = inB ? (int) ((long long) a.pos[t0 + rB] - first - s_lo) : 0;
    const int s_first = (int) (p_first - first - s_lo);
    const float fA = inA ? a.far[hA] : -INFINITY, fB = inB ? a.far[hB] : -INFINITY;
    const float* ringA = a.ring + hA * a.ext + 2 * tig;
    const float* ringB = a.ring + hB * a.ext + 2 * tig;
    int j0 = a.ext > 0 ? (int) ((s_lo + (long long) it0 * kTcCells) % a.ext) : 0;   // the tile's first slot in the ring
    const bool sink = !a.no_sink && sg == 0;
    float mA = sink ? 0.f : -1e30f, mB = mA, lA = sink ? 1.f : 0.f, lB = lA;
    float o[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    for (int it = it0; it < n_tiles; ++it) {
        const int base = it * kTcCells;
        cp_wait<0>();
        __syncthreads();   // the tile has arrived, and the last one's cells were read
        const __half *kt, *vt;
        if constexpr (Q8) {   // the blocks into halves, eight values a step; then the next tile's on their way
            for (int i = tid; i < 2 * kTcCells * 16; i += NT) {
                const int side = i >= kTcCells * 16, j = i - side * kTcCells * 16, c = j / 16, u = j % 16, blk = u / 4;
                const uint8_t* cell = raw[side] + c * kRawCell;
                const float d = __half2float(*reinterpret_cast<const __half*>(cell + blk * 34));
                const int at = blk * 34 + 2 + (u % 4) * 8;   // the codes: at a word, or two bytes into one
                const uint32_t* wp = reinterpret_cast<const uint32_t*>(cell + (at & ~3));
                uint32_t w0 = wp[0], w1 = wp[1];
                if (at & 2) {
                    const uint32_t w2 = wp[2];
                    w0 = (w0 >> 16) | (w1 << 16);
                    w1 = (w1 >> 16) | (w2 << 16);
                }
                const uint2 x0 = q8_widen4(w0, d), x1 = q8_widen4(w1, d);
                *reinterpret_cast<uint4*>(tile[0][side] + c * kTcPad + u * 8) = make_uint4(x0.x, x0.y, x1.x, x1.y);
            }
            __syncthreads();
            if (it + 1 < n_tiles) fetch(0, base + kTcCells);
            kt = tile[0][0];
            vt = tile[0][1];
        } else {
            if (it + 1 < n_tiles) fetch((it + 1 - it0) & 1, base + kTcCells);
            kt = tile[(Q8 ? 0 : it - it0) & 1][0];
            vt = tile[(Q8 ? 0 : it - it0) & 1][1];
        }
        // a tile every row of the block sees whole and past the table: no cell's distance to look at
        const bool far = a.win == 0 && base >= c_min && base + kTcCells <= cnt && s_first - (base + kTcCells - 1) >= a.ext;
        float2 gA[4], gB[4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) gA[nt] = gB[nt] = make_float2(0.f, 0.f);
        if (!far && a.ext > 0) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                gA[nt] = *reinterpret_cast<const float2*>(ringA + j0 + 8 * nt);
                gB[nt] = *reinterpret_cast<const float2*>(ringB + j0 + 8 * nt);
            }
        }
        if (a.ext > 0) j0 = j0 + kTcCells >= a.ext ? j0 + kTcCells - a.ext : j0 + kTcCells;
        float sc[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;
#pragma unroll
        for (int kk = 0; kk < D / 16; kk += 2) {
            unsigned b[4][4];
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) ldsm4(b[nt], kt + (nt * 8 + lr) * kTcPad + li * 8 + kk * 16);
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) mma16816(sc[nt], qa[kk], b[nt][0], b[nt][1]);
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) mma16816(sc[nt], qa[kk + 1], b[nt][2], b[nt][3]);
        }
        float xA = -INFINITY, xB = -INFINITY;
        if (far) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    sc[nt][e] = sc[nt][e] * scale + fA;
                    sc[nt][2 + e] = sc[nt][2 + e] * scale + fB;
                    xA = fmaxf(xA, sc[nt][e]);
                    xB = fmaxf(xB, sc[nt][2 + e]);
                }
        } else {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int col = base + nt * 8 + 2 * tig + e;
                    const int dA = sA - col, dB = sB - col;
                    const bool in = col >= c_min && col < cnt;
                    const bool okA = inA && in && dA >= 0 && (a.win == 0 || dA < a.win);
                    const bool okB = inB && in && dB >= 0 && (a.win == 0 || dB < a.win);
                    const float vA = dA < a.ext ? (e ? gA[nt].y : gA[nt].x) : fA;
                    const float vB = dB < a.ext ? (e ? gB[nt].y : gB[nt].x) : fB;
                    sc[nt][e] = okA ? sc[nt][e] * scale + vA : -INFINITY;
                    sc[nt][2 + e] = okB ? sc[nt][2 + e] * scale + vB : -INFINITY;
                    xA = fmaxf(xA, sc[nt][e]);
                    xB = fmaxf(xB, sc[nt][2 + e]);
                }
        }
        xA = fmaxf(xA, __shfl_xor_sync(0xffffffffu, xA, 1));
        xA = fmaxf(xA, __shfl_xor_sync(0xffffffffu, xA, 2));
        xB = fmaxf(xB, __shfl_xor_sync(0xffffffffu, xB, 1));
        xB = fmaxf(xB, __shfl_xor_sync(0xffffffffu, xB, 2));
        const float nA = fmaxf(mA, xA), nB = fmaxf(mB, xB);
        const float aA = __expf(mA - nA), aB = __expf(mB - nB);
        float sumA[4], sumB[4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                sc[nt][e] = sc[nt][e] == -INFINITY ? 0.f : __expf(sc[nt][e] - nA);
                sc[nt][2 + e] = sc[nt][2 + e] == -INFINITY ? 0.f : __expf(sc[nt][2 + e] - nB);
            }
            sumA[nt] = sc[nt][0] + sc[nt][1];
            sumB[nt] = sc[nt][2] + sc[nt][3];
            sumA[nt] += __shfl_xor_sync(0xffffffffu, sumA[nt], 1);
            sumA[nt] += __shfl_xor_sync(0xffffffffu, sumA[nt], 2);
            sumB[nt] += __shfl_xor_sync(0xffffffffu, sumB[nt], 1);
            sumB[nt] += __shfl_xor_sync(0xffffffffu, sumB[nt], 2);
        }
        lA = lA * aA + ((sumA[0] + sumA[1]) + (sumA[2] + sumA[3]));
        lB = lB * aB + ((sumB[0] + sumB[1]) + (sumB[2] + sumB[3]));
        mA = nA;
        mB = nB;
        if (__any_sync(0xffffffffu, aA != 1.f || aB != 1.f)) {   // (a row's largest logit moved: rare past the start)
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                o[j][0] *= aA;
                o[j][1] *= aA;
                o[j][2] *= aB;
                o[j][3] *= aB;
            }
        }
#pragma unroll
        for (int k = 0; k < 2; ++k) {   // the tile's halves: the probabilities of 16 cells as a row operand
            const unsigned pa[4] = {h2(sc[2 * k][0], sc[2 * k][1]), h2(sc[2 * k][2], sc[2 * k][3]),
                                    h2(sc[2 * k + 1][0], sc[2 * k + 1][1]), h2(sc[2 * k + 1][2], sc[2 * k + 1][3])};
            const __half* vrow = vt + (16 * k + ((li & 1) << 3) + lr) * kTcPad + (li >> 1) * 8;
#pragma unroll
            for (int n4 = 0; n4 < 8; ++n4) {
                unsigned b[4];
                ldsm4t(b, vrow + n4 * 16);
                mma16816(o[2 * n4], pa, b[0], b[1]);
                mma16816(o[2 * n4 + 1], pa, b[2], b[3]);
            }
        }
    }
    if (a.part) {   // the partial sums: a row's m and l by one of its lanes, its values by each lane's
        float* const to = sg == 0 ? a.part : a.seg + (size_t) (sg - 1) * a.T * a.H * (D + 2);
        float* pA = to + hA * (D + 2);
        float* pB = to + hB * (D + 2);
        if (tig == 0) {
            if (inA) *reinterpret_cast<float2*>(pA) = make_float2(mA, lA);
            if (inB) *reinterpret_cast<float2*>(pB) = make_float2(mB, lB);
        }
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int d = j * 8 + 2 * tig;
            if (inA) *reinterpret_cast<float2*>(pA + 2 + d) = make_float2(o[j][0], o[j][1]);
            if (inB) *reinterpret_cast<float2*>(pB + 2 + d) = make_float2(o[j][2], o[j][3]);
        }
        return;
    }
    const float iA = 1.f / lA, iB = 1.f / lB;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int d = j * 8 + 2 * tig;
        if (inA) {
            const size_t at = hA * D + d;
            const float2 gt = *reinterpret_cast<const float2*>(a.gate + at);
            *reinterpret_cast<float2*>(a.out + at) =
                make_float2(o[j][0] * iA / (1.0f + __expf(-gt.x)), o[j][1] * iA / (1.0f + __expf(-gt.y)));
        }
        if (inB) {
            const size_t at = hB * D + d;
            const float2 gt = *reinterpret_cast<const float2*>(a.gate + at);
            *reinterpret_cast<float2*>(a.out + at) =
                make_float2(o[j][2] * iB / (1.0f + __expf(-gt.x)), o[j][3] * iB / (1.0f + __expf(-gt.y)));
        }
    }
}

/// A decode forward's rows (1 .. kMaxT) over a stretch of a visit's cells, on the tensor cores: block (cb, g) takes the
/// rows' query heads of K/V head g (row r: forward row r / QH, its head r % QH of g's; T QH <= 32) over stretch cb, in
/// tiles of 32 cells (K and V of the K/V head, halves, or with Q8 q8_0 blocks widened into halves as they are
/// stashed), so a visit's cells are read once whatever the rows (a full visit's cells are the time of a token deep in
/// a context).  A stretch's sums a row and head: m, the sum of exp(logit - m) and that-weighted V, no sink, into part
/// [T][H][n_chunks][D + 2] (m = -1e30 and a sum of 0 where a row sees none of the cells).  A row's sums do not depend
/// on the rows scored with it: a forward of several rows leaves what a row at a time leaves, to the bit.  A ring's
/// slot stands for the position last written there, and with `dyn` a ring's stretches start at slot dyn[2] and go on
/// past its end to its start (the drafter's rings: the cells in use alone).  The arithmetic is the chunks': q and the
/// probabilities as halves.  Warps: scores m-tile w & 1 x cells (w >> 1) * 8 .. +8; values m-tile w & 1 x dims
/// (w >> 1) * 32 .. +32.  The next tile waits in registers through the current one's products.  Grid (stretches, K/V
/// heads), block 256.
template <bool Q8>
__global__ void __launch_bounds__(256, 1) attn_rows_kernel(AttnIn a, int T) {
    constexpr int D = 128;
    __shared__ TcSmem sm;
    const int cb = blockIdx.x, g = blockIdx.y;
    if (a.dyn && cb >= a.dyn[1]) return;
    const int QH = a.H / a.HK, NQ = T * QH;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
    const int li = lane >> 3, lr = lane & 7, tig = lane & 3;
    const long long live = *a.live, last = live - 1, cap = (long long) a.c.cap, first = (long long) a.c.first;
    // the block's cells: slots [s_lo, s_lo + cnt), a ring's with `dyn` modulo its size
    const bool wrap = !a.canon && a.c.ring && a.dyn != nullptr;
    long long s_lo;
    int cnt;
    if (a.canon) {
        const int i = a.dyn[2] + cb;
        s_lo = max(stretch_lo(i), (long long) a.dyn[0]);
        cnt = (int) (min(stretch_lo(i + 1), live) - s_lo);
    } else {
        const int chunk = a.dyn ? a.dyn[0] : a.chunk;
        s_lo = (a.dyn ? a.dyn[2] : 0) + (long long) cb * chunk;
        cnt = wrap ? chunk : (int) (min(s_lo + chunk, a.c.ring ? cap : live) - s_lo);
    }
    const size_t cbytes = cell_bytes(a.c, a.HK, D);
    auto slot_of = [&](int col) { return wrap ? (s_lo + col) % cap : s_lo + col; };
    // the queries as halves
    for (int i = tid; i < kTcRows * (D / 4); i += 256) {
        const int r = i / (D / 4), c4 = i % (D / 4);
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (r < NQ) v = *reinterpret_cast<const float4*>(a.q + ((size_t) (r / QH) * a.H + g * QH + r % QH) * D + 4 * c4);
        *reinterpret_cast<uint2*>(sm.q + r * kTcPad + 4 * c4) = make_uint2(h2(v.x, v.y), h2(v.z, v.w));
    }
    // a tile: 32 cells of K and of V (attn_tc_kernel's)
    uint4 nk[2], nv[2];
    [[maybe_unused]] float dk = 0.f, dv = 0.f;
    auto fetch = [&](int base) {
        if constexpr (Q8) {
            const int c = tid / 8, pc = tid % 8;
            nk[0] = nv[0] = make_uint4(0, 0, 0, 0);
            dk = dv = 0.f;
            if (base + c < cnt) {
                const size_t at = (size_t) slot_of(base + c) * cbytes + ((size_t) g * (D / 32) + pc / 2) * 34;
                const uint8_t *kb = a.c.k + at, *vb = a.c.v + at;
                dk = __half2float(*reinterpret_cast<const __half*>(kb));
                dv = __half2float(*reinterpret_cast<const __half*>(vb));
                memcpy(&nk[0], kb + 2 + (pc % 2) * 16, 16);
                memcpy(&nv[0], vb + 2 + (pc % 2) * 16, 16);
            }
        } else {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int i = tid + 256 * j, c = i / 16, ch = i % 16;
                if (base + c < cnt) {
                    const size_t at = (size_t) slot_of(base + c) * cbytes + (size_t) g * D * 2 + ch * 16;
                    nk[j] = *reinterpret_cast<const uint4*>(a.c.k + at);
                    nv[j] = *reinterpret_cast<const uint4*>(a.c.v + at);
                } else {
                    nk[j] = nv[j] = make_uint4(0, 0, 0, 0);
                }
            }
        }
    };
    auto stash = [&] {
        if constexpr (Q8) {
            const int c = tid / 8, pc = tid % 8;
            const uint2 k0 = q8_widen4(nk[0].x, dk), k1 = q8_widen4(nk[0].y, dk), k2 = q8_widen4(nk[0].z, dk),
                        k3 = q8_widen4(nk[0].w, dk);
            const uint2 v0 = q8_widen4(nv[0].x, dv), v1 = q8_widen4(nv[0].y, dv), v2 = q8_widen4(nv[0].z, dv),
                        v3 = q8_widen4(nv[0].w, dv);
            uint4* kd = reinterpret_cast<uint4*>(sm.k + c * kTcPad + pc * 16);
            uint4* vd = reinterpret_cast<uint4*>(sm.v + c * kTcPad + pc * 16);
            kd[0] = make_uint4(k0.x, k0.y, k1.x, k1.y);
            kd[1] = make_uint4(k2.x, k2.y, k3.x, k3.y);
            vd[0] = make_uint4(v0.x, v0.y, v1.x, v1.y);
            vd[1] = make_uint4(v2.x, v2.y, v3.x, v3.y);
        } else {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int i = tid + 256 * j, c = i / 16, ch = i % 16;
                *reinterpret_cast<uint4*>(sm.k + c * kTcPad + ch * 8) = nk[j];
                *reinterpret_cast<uint4*>(sm.v + c * kTcPad + ch * 8) = nv[j];
            }
        }
    };
    fetch(0);
    stash();
    const int mt = w & 1, nt = w >> 1;
    const int rA = mt * 16 + (lane >> 2), rB = rA + 8;
    const bool inA = rA < NQ, inB = rB < NQ;
    const int tA = inA ? rA / QH : 0, tB = inB ? rB / QH : 0, hA = inA ? rA % QH : 0, hB = inB ? rB % QH : 0;
    const long long pA = a.pos[tA], pB = a.pos[tB];
    const float* bA = a.bias + ((size_t) tA * a.H + g * QH + hA) * (a.ext + 1);
    const float* bB = a.bias + ((size_t) tB * a.H + g * QH + hB) * (a.ext + 1);
    const float scale = rsqrtf((float) D);
    // nothing yet: a largest logit that any real one passes, finite (no cell, no NaN)
    float mA = -1e30f, mB = mA, lA = 0.f, lB = 0.f;
    float o[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    const __half* qrow = sm.q + (mt * 16 + ((li & 1) << 3) + lr) * kTcPad + (li >> 1) * 8;
    const int n_tiles = (max(cnt, 0) + kTcCells - 1) / kTcCells;
    for (int it = 0; it < n_tiles; ++it) {
        const int base = it * kTcCells;
        __syncthreads();   // the tile (and the queries) in shared memory
        if (it + 1 < n_tiles) fetch(base + kTcCells);
        float sc[4] = {0.f, 0.f, 0.f, 0.f};
        const __half* krow = sm.k + (nt * 8 + lr) * kTcPad + li * 8;
#pragma unroll
        for (int kk = 0; kk < D / 16; kk += 2) {
            unsigned a0[4], a1[4], b[4];
            ldsm4(a0, qrow + kk * 16);
            ldsm4(a1, qrow + kk * 16 + 16);
            ldsm4(b, krow + kk * 16);
            mma16816(sc, a0, b[0], b[1]);
            mma16816(sc, a1, b[2], b[3]);
        }
        float xA = -INFINITY, xB = -INFINITY;
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int col = base + nt * 8 + 2 * tig + e;
            const long long slot = slot_of(col);
            const long long cp = a.c.ring ? last - ((last - slot) % cap + cap) % cap : first + slot;
            const long long dA = pA - cp, dB = pB - cp;
            const bool okA = inA && col < cnt && cp >= 0 && dA >= 0 && (a.win == 0 || dA < a.win);
            const bool okB = inB && col < cnt && cp >= 0 && dB >= 0 && (a.win == 0 || dB < a.win);
            sc[e] = okA ? sc[e] * scale + bA[dA < a.ext ? dA : a.ext] : -INFINITY;
            sc[2 + e] = okB ? sc[2 + e] * scale + bB[dB < a.ext ? dB : a.ext] : -INFINITY;
            xA = fmaxf(xA, sc[e]);
            xB = fmaxf(xB, sc[2 + e]);
        }
        xA = fmaxf(xA, __shfl_xor_sync(0xffffffffu, xA, 1));
        xA = fmaxf(xA, __shfl_xor_sync(0xffffffffu, xA, 2));
        xB = fmaxf(xB, __shfl_xor_sync(0xffffffffu, xB, 1));
        xB = fmaxf(xB, __shfl_xor_sync(0xffffffffu, xB, 2));
        if (tig == 0) {
            sm.red[0][nt][rA] = xA;
            sm.red[0][nt][rB] = xB;
        }
        __syncthreads();
        const float nA = fmaxf(mA, fmaxf(fmaxf(sm.red[0][0][rA], sm.red[0][1][rA]), fmaxf(sm.red[0][2][rA], sm.red[0][3][rA])));
        const float nB = fmaxf(mB, fmaxf(fmaxf(sm.red[0][0][rB], sm.red[0][1][rB]), fmaxf(sm.red[0][2][rB], sm.red[0][3][rB])));
        const float aA = __expf(mA - nA), aB = __expf(mB - nB);
        const float p0A = sc[0] == -INFINITY ? 0.f : __expf(sc[0] - nA), p1A = sc[1] == -INFINITY ? 0.f : __expf(sc[1] - nA);
        const float p0B = sc[2] == -INFINITY ? 0.f : __expf(sc[2] - nB), p1B = sc[3] == -INFINITY ? 0.f : __expf(sc[3] - nB);
        const int pc = nt * 8 + 2 * tig;
        *reinterpret_cast<unsigned*>(sm.p + rA * kTcPPad + pc) = h2(p0A, p1A);
        *reinterpret_cast<unsigned*>(sm.p + rB * kTcPPad + pc) = h2(p0B, p1B);
        float sA = p0A + p1A, sB = p0B + p1B;
        sA += __shfl_xor_sync(0xffffffffu, sA, 1);
        sA += __shfl_xor_sync(0xffffffffu, sA, 2);
        sB += __shfl_xor_sync(0xffffffffu, sB, 1);
        sB += __shfl_xor_sync(0xffffffffu, sB, 2);
        if (tig == 0) {
            sm.red[1][nt][rA] = sA;
            sm.red[1][nt][rB] = sB;
        }
        __syncthreads();
        lA = lA * aA + ((sm.red[1][0][rA] + sm.red[1][1][rA]) + (sm.red[1][2][rA] + sm.red[1][3][rA]));
        lB = lB * aB + ((sm.red[1][0][rB] + sm.red[1][1][rB]) + (sm.red[1][2][rB] + sm.red[1][3][rB]));
        mA = nA;
        mB = nB;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            o[j][0] *= aA;
            o[j][1] *= aA;
            o[j][2] *= aB;
            o[j][3] *= aB;
        }
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            unsigned pa[4];
            ldsm4(pa, sm.p + (mt * 16 + ((li & 1) << 3) + lr) * kTcPPad + (2 * k + (li >> 1)) * 8);
            const __half* vrow = sm.v + (16 * k + ((li & 1) << 3) + lr) * kTcPad + nt * 32 + (li >> 1) * 8;
#pragma unroll
            for (int n2 = 0; n2 < 2; ++n2) {
                unsigned b[4];
                ldsm4t(b, vrow + n2 * 16);
                mma16816(o[2 * n2], pa, b[0], b[1]);
                mma16816(o[2 * n2 + 1], pa, b[2], b[3]);
            }
        }
        __syncthreads();   // the tile's cells and probabilities are read
        if (it + 1 < n_tiles) stash();
    }
    // a row's m and sum by one of its lanes, its values by each lane's eight
    float* oA = a.part + (((size_t) tA * a.H + g * QH + hA) * a.n_chunks + cb) * (D + 2);
    float* oB = a.part + (((size_t) tB * a.H + g * QH + hB) * a.n_chunks + cb) * (D + 2);
    if (nt == 0 && tig == 0) {
        if (inA) *reinterpret_cast<float2*>(oA) = make_float2(mA, lA);
        if (inB) *reinterpret_cast<float2*>(oB) = make_float2(mB, lB);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int d = nt * 32 + j * 8 + 2 * tig;
        if (inA) *reinterpret_cast<float2*>(oA + 2 + d) = make_float2(o[j][0], o[j][1]);
        if (inB) *reinterpret_cast<float2*>(oB + 2 + d) = make_float2(o[j][2], o[j][3]);
    }
}

/// A chunk's attention where cells are on cards: this card's partial sums (with the zero sink) and the cards' joined,
/// divided by their sum and times sigmoid(gate): out [T][H * D].  A block a row and a query head; D threads.
struct TcMergeIn {
    const float* part;    // [T][H][D + 2]
    const float* extra;   // [n_extra][cap][H][D + 2]: the cards'
    const float* seg;     // [n_seg][T][H][D + 2]: this card's segments after the first
    const float* gate;    // [T][H * D]
    float* out;           // [T][H * D]
    int H, D, n_extra, cap, n_seg, T;
};
__global__ void __launch_bounds__(128) tc_merge_kernel(TcMergeIn a) {
    const int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    const float* p = a.part + ((size_t) t * a.H + h) * (a.D + 2);
    auto more = [&](int e) {   // the cards' sums, then the segments'
        return e < a.n_extra ? a.extra + (((size_t) e * a.cap + t) * a.H + h) * (a.D + 2)
                             : a.seg + (((size_t) (e - a.n_extra) * a.T + t) * a.H + h) * (a.D + 2);
    };
    float mx = p[0];
    for (int e = 0; e < a.n_extra + a.n_seg; ++e) {
        const float* pe = more(e);
        if (pe[1] != 0.f) mx = fmaxf(mx, pe[0]);
    }
    const float w0 = __expf(p[0] - mx);
    float sum = w0 * p[1], acc = w0 * p[2 + i];
    for (int e = 0; e < a.n_extra + a.n_seg; ++e) {
        const float* pe = more(e);
        if (pe[1] == 0.f) continue;
        const float w = __expf(pe[0] - mx);
        sum += w * pe[1];
        acc += w * pe[2 + i];
    }
    const size_t at = ((size_t) t * a.H + h) * a.D + i;
    a.out[at] = acc / sum / (1.0f + __expf(-a.gate[at]));
}

// ---------------------------------------------------------------------------------------------- BF16 in a chunk
/// n BF16 values as floats
__global__ void bf16_floats_kernel(const uint16_t* __restrict__ w, float* __restrict__ out, long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __uint_as_float((unsigned) w[i] << 16);
}
/// Rows gathered or scattered by index: dst[r] = src[idx[r]] (`scatter` 0) or dst[idx[r]] = src[r]; n values a row
__global__ void rows_index_kernel(const float* __restrict__ src, const int32_t* __restrict__ idx, float* __restrict__ dst,
                                  int rows, int n, int scatter) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long) rows * n) return;
    const int r = (int) (i / n), c = (int) (i % n);
    if (scatter) dst[(size_t) idx[r] * n + c] = src[i];
    else dst[i] = src[(size_t) idx[r] * n + c];
}

/// With experts in RAM: the entries whose expert has no slot take the CPU's rows, which come packed in the entries'
/// order and without the down rows' scale (`scale` [experts], or null).  A block an entry.
__global__ void __launch_bounds__(kTB) cpu_rows_kernel(const float* __restrict__ src, const int32_t* __restrict__ ids,
                                                       const int32_t* __restrict__ slot_of, const float* __restrict__ scale,
                                                       float* __restrict__ rows, int N) {
    const int e = blockIdx.x;
    if (slot_of[ids[e]] >= 0) return;
    __shared__ int at;
    if (threadIdx.x == 0) {
        int k = 0;   // the CPU's entries before it
        for (int i = 0; i < e; ++i) k += slot_of[ids[i]] < 0 ? 1 : 0;
        at = k;
    }
    __syncthreads();
    const float* s = src + (size_t) at * N;
    float* d = rows + (size_t) e * N;
    const float sc = scale != nullptr ? scale[ids[e]] : 1.0f;
    for (int i = threadIdx.x; i < N; i += kTB) d[i] = s[i] * sc;
}

// ---------------------------------------------------------------------------------------------- checks
/// The first non-finite value's index among n floats into *at (a device int64, -1 before), for LAMINA_INK_NAN
__global__ void nonfinite_kernel(const float* __restrict__ x, long long n, long long* __restrict__ at) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    if (!(fabsf(v) <= FLT_MAX)) atomicMin((unsigned long long*) at, (unsigned long long) i);
}

// ---------------------------------------------------------------------------------------------- the head
/// cap tanh(x / cap) over rows of V logits (cap 0: as they are); bad[row] = 1 where a row holds a non-finite one
/// (zeroed before the launch)
__global__ void softcap_kernel(float* __restrict__ lg, long long n, float cap, int V, int32_t* __restrict__ bad) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = cap > 0.f ? cap * tanhf(lg[i] / cap) : lg[i];
    if (cap > 0.f) lg[i] = v;
    if (!(fabsf(v) <= FLT_MAX)) bad[i / V] = 1;
}

}  // namespace ik
