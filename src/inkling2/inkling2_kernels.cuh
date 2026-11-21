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
#include "lamina/kernels/warp.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

namespace ik {

using lamina::kernels::q8_1_ds;
using lamina::kernels::q8_1_finite;
using lamina::kernels::q8_1_quant;
using lamina::kernels::warp_max;
using lamina::kernels::warp_sum;

constexpr int kMaxT = 8;      // rows a forward takes at most (native_mmvq's columns)
constexpr int kBlk = 4;       // block inputs a mix reads at most (the embedding and three blocks' outputs)
constexpr int kTB = 256;      // threads of the row kernels
constexpr int kSub = 128;     // cells an attention block scores at once (its threads)

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
__global__ void __launch_bounds__(kTB) mix_kernel(MixIn a) {
    const int t = blockIdx.x, c = blockIdx.y;
    __shared__ float red[8][kBlk + 2];
    const float* cur = a.cur + (size_t) t * a.N;
    const float* q = a.nq > 0 ? a.q + (size_t) c * a.N : nullptr;
    float v[kBlk + 2];
#pragma unroll
    for (int k = 0; k < kBlk + 2; ++k) v[k] = 0.f;
    for (int i = threadIdx.x; i < a.N; i += kTB) {
        const float x = cur[i], qi = q ? q[i] : 0.f;
        v[kBlk] += x * x;
        v[kBlk + 1] += x * qi;
        for (int j = 0; j < a.nb; ++j) v[j] += a.blk[j][(size_t) t * a.N + i] * qi;
    }
    block_sums<kBlk + 2>(v, red);
    const float inv = rsqrtf(v[kBlk] / (float) a.N + 1e-6f);
    if (a.nq == 0) {
        for (int i = threadIdx.x; i < a.N; i += kTB) a.cur_n[(size_t) t * a.N + i] = cur[i] * inv;
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
    const float r = 1.0f / sum;
    float* o = a.out + ((size_t) c * a.T + t) * a.N;
    for (int i = threadIdx.x; i < a.N; i += kTB) {
        const float xn = cur[i] * inv;
        float y = s[a.nb] * r * xn;
        for (int j = 0; j < a.nb; ++j) y += s[j] * r * a.blk[j][(size_t) t * a.N + i];
        o[i] = y;
        if (c == 0 && a.cur_n) a.cur_n[(size_t) t * a.N + i] = xn;
    }
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
/// h: a block a row, an input and one of the R outputs
__global__ void __launch_bounds__(kTB) ctrl_h_kernel(CtrlIn a) {
    const int r = blockIdx.x, t = blockIdx.y, c = blockIdx.z;
    __shared__ float red[8][4];
    const float* u = a.u[c] + (size_t) t * a.N;
    const float* m = a.m[c] + (size_t) t * a.N;
    const float* d = a.D[c] + (size_t) r * 2 * a.N;
    float v[4] = {0.f, 0.f, 0.f, 0.f};
    for (int i = threadIdx.x; i < a.N; i += kTB) {
        const float ui = u[i], mi = m[i];
        v[0] += ui * ui;
        v[1] += mi * mi;
        v[2] += d[i] * ui;
        v[3] += d[a.N + i] * mi;
    }
    block_sums<4>(v, red);
    if (threadIdx.x == 0)
        a.h[((size_t) c * gridDim.y + t) * a.R + r] =
            v[2] * rsqrtf(v[0] / (float) a.N + 1e-6f) + v[3] * rsqrtf(v[1] / (float) a.N + 1e-6f);
}
/// the gated rows: a thread an output
__global__ void __launch_bounds__(kTB) ctrl_apply_kernel(CtrlIn a) {
    const int t = blockIdx.y, c = blockIdx.z, i = blockIdx.x * kTB + threadIdx.x;
    __shared__ float h[kRankMax];
    if (threadIdx.x < a.R) h[threadIdx.x] = a.h[((size_t) c * gridDim.y + t) * a.R + threadIdx.x];
    __syncthreads();
    if (i >= a.N) return;
    const float* g = a.G[c] + (size_t) i * a.R;
    const float* f = a.F[c] + (size_t) i * a.R;
    float sg = 0.f, sf = 0.f;
    for (int k = 0; k < a.R; ++k) {
        sg += g[k] * h[k];
        sf += f[k] * h[k];
    }
    const size_t at = (size_t) t * a.N + i;
    a.out[c][at] = (1.0f - tanhf(sf)) * a.u[c][at] + tanhf(sg) * a.m[c][at];
}

// ---------------------------------------------------------------------------------------------- norms
/// y = x rsqrt(mean(x^2) + eps) w, and its q8_1 image (N / 32 blocks) where q is not null; n inputs a launch.  w null:
/// no weight.
struct NormIn {
    const float* x[3];
    const float* w[3];
    float* y[3];
    Q81* q[3];
    int n, N;
    float eps;
};
__global__ void __launch_bounds__(kTB) norm_kernel(NormIn a) {
    const int t = blockIdx.x, c = blockIdx.y;
    __shared__ float red[8][1];
    const float* x = a.x[c] + (size_t) t * a.N;
    float v[1] = {0.f};
    for (int i = threadIdx.x; i < a.N; i += kTB) v[0] += x[i] * x[i];
    block_sums<1>(v, red);
    const float inv = rsqrtf(v[0] / (float) a.N + a.eps);
    const float* w = a.w[c];
    float* y = a.y[c] ? a.y[c] + (size_t) t * a.N : nullptr;
    Q81* q = a.q[c] ? a.q[c] + (size_t) t * (a.N / 32) : nullptr;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int b = warp; b < a.N / 32; b += kTB / 32) {   // a warp a q8_1 block
        const int i = b * 32 + lane;
        const float xi = x[i] * inv * (w ? w[i] : 1.0f);
        if (y) y[i] = xi;
        if (q) {
            const float amax = warp_max(fabsf(xi)), sum = warp_sum(xi);
            const float d = q8_1_finite(amax / 127.0f);
            q[b].qs[lane] = q8_1_quant(xi, d, amax);
            if (lane == 0) q[b].ds = q8_1_ds(d, sum);
        }
    }
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
    const int t = threadIdx.x;
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
__global__ void __launch_bounds__(kTB) bias_kernel(BiasIn a) {
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
    for (int d = threadIdx.x; d < a.ext; d += kTB) {
        const float* tb = a.table + (size_t) d * a.R;
        float s = 0.f;
        for (int k = 0; k < a.R; ++k) s += tb[k] * r[k];
        b[d] = s - sink;
    }
    if (threadIdx.x == 0) b[a.ext] = -sink;
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

// ---------------------------------------------------------------------------------------------- the attention
/// A (row, K/V head)'s query heads against a stretch of cells: a block scores `chunk` cells, `sub` at a time, and keeps
/// for each of its QH query heads the stretch's largest logit m, sum of exp(logit - m) and that-weighted sum of V
/// (D values): the softmax's partial sums, which attn_out_kernel joins over the stretches (and another card's range
/// could send).  logit = q . k / sqrt(D) + bias[min(distance, ext)]; a cell after the row, or outside a window layer's
/// `win` positions, is left out.  `sub` is kSub (a thread a cell's K) or kSub / 4 (four threads a cell: shorter chains
/// of loads where the cells are few and the blocks many).  With `dyn` the stretches' size and count come from the
/// device ({chunk, stretches in use}: a launch captured in a graph serves any number of cells), `n_chunks` then the
/// grid's.
struct AttnIn {
    Cells c;
    const float* q;         // [T][H][D]
    const float* bias;      // [T][H][ext + 1]
    const int32_t* pos;     // [T]
    const int32_t* live;    // the cells in use: a range's count (its positions first .. first + *live - 1), a ring's last position + 1
    const int32_t* dyn;     // null, or {chunk, stretches in use}
    float* part;            // [T][H][n_chunks][D + 2]: m, s, the D sums
    int H, HK, D, ext, win; // win: a window layer's cells (0: every cell)
    int chunk, n_chunks, sub;
};
constexpr int kQH = 4;      // query heads a K/V head at most
__global__ void __launch_bounds__(kSub) attn_part_kernel(AttnIn a) {
    const int cb = blockIdx.x, g = blockIdx.y, t = blockIdx.z, j = threadIdx.x;
    const int chunk = a.dyn ? a.dyn[0] : a.chunk;
    if (a.dyn && cb >= a.dyn[1]) return;
    const int QH = a.H / a.HK;
    __shared__ float sq[kQH][128];
    __shared__ float sw[kQH][kSub];
    __shared__ float red[4][2 * kQH];
    const int64_t p = a.pos[t];
    const int64_t live = *a.live;
    // this block's cells: slots [s0, s1)
    const int64_t s0 = (int64_t) cb * chunk, s1 = min((int64_t) s0 + chunk, a.c.ring ? a.c.cap : live);
    float* out = a.part + (((size_t) t * a.H + (size_t) g * QH) * a.n_chunks + cb) * (a.D + 2);
    const size_t pstride = (size_t) a.n_chunks * (a.D + 2);   // a query head to the next
    if (s0 >= s1) {
        if (j < QH) {
            out[(size_t) j * pstride] = -FLT_MAX;
            out[(size_t) j * pstride + 1] = 0.f;
        }
        return;
    }
    float so[kQH];   // value j of the heads' weighted sums of V
    float m[kQH], s[kQH];
#pragma unroll
    for (int h = 0; h < kQH; ++h) {
        if (h < QH) sq[h][j] = a.q[((size_t) t * a.H + g * QH + h) * a.D + j];
        so[h] = 0.f;
        m[h] = -FLT_MAX;
        s[h] = 0.f;
    }
    __syncthreads();
    const size_t cbytes = cell_bytes(a.c, a.HK, a.D);
    const float scale = rsqrtf((float) a.D);
    const int per = kSub / a.sub;            // threads a cell in the K step
    const int nblk = a.D / 32 / per;         // 32-value groups of K a thread takes
    const int cell = j / per, blk0 = (j % per) * nblk;
    for (int64_t b0 = s0; b0 < s1; b0 += a.sub) {
        // the K step: this thread's share of cell b0 + cell
        const int64_t slot = b0 + cell;
        bool ok = slot < s1;
        int64_t d = 0;
        if (ok) {
            int64_t cp;
            if (a.c.ring) {   // the slot's position: the last one written there
                const int64_t last = live - 1;
                cp = last - ((last - slot) % a.c.cap + a.c.cap) % a.c.cap;
            } else {
                cp = a.c.first + slot;
            }
            d = p - cp;
            ok = cp >= 0 && d >= 0 && (a.win == 0 || d < a.win);
        }
        float dot[kQH];
#pragma unroll
        for (int h = 0; h < kQH; ++h) dot[h] = 0.f;
        if (ok) {
            const uint8_t* kc = a.c.k + (size_t) slot * cbytes;
            if (a.c.q8) {
                const uint8_t* kb = kc + ((size_t) g * (a.D / 32) + blk0) * 34;
                for (int b = 0; b < nblk; ++b) {
                    const float sc = __half2float(*(const __half*) (kb + (size_t) b * 34));
                    const int8_t* qs = (const int8_t*) (kb + (size_t) b * 34 + 2);
                    float acc[kQH];
#pragma unroll
                    for (int h = 0; h < kQH; ++h) acc[h] = 0.f;
                    for (int i = 0; i < 32; ++i) {
                        const float kv = (float) qs[i];
#pragma unroll
                        for (int h = 0; h < kQH; ++h)
                            if (h < QH) acc[h] += sq[h][(blk0 + b) * 32 + i] * kv;
                    }
#pragma unroll
                    for (int h = 0; h < kQH; ++h) dot[h] += acc[h] * sc;
                }
            } else {
                const __half2* kh = (const __half2*) kc + (size_t) g * (a.D / 2) + blk0 * 16;
                for (int i = 0; i < nblk * 16; ++i) {
                    const float2 kv = __half22float2(kh[i]);
#pragma unroll
                    for (int h = 0; h < kQH; ++h)
                        if (h < QH) dot[h] += sq[h][blk0 * 32 + 2 * i] * kv.x + sq[h][blk0 * 32 + 2 * i + 1] * kv.y;
                }
            }
        }
        float l[kQH];
        const int bi = (int) (d < a.ext ? d : a.ext);
#pragma unroll
        for (int h = 0; h < kQH; ++h) {
            if (per == 4) {   // the cell's four threads' shares
                dot[h] += __shfl_xor_sync(0xffffffffu, dot[h], 1);
                dot[h] += __shfl_xor_sync(0xffffffffu, dot[h], 2);
            }
            l[h] = ok && h < QH ? dot[h] * scale + a.bias[((size_t) t * a.H + g * QH + h) * (a.ext + 1) + bi] : -FLT_MAX;
        }
        // the step's largest logits, then the weights against the stretch's largest so far
        float bm[kQH];
#pragma unroll
        for (int h = 0; h < kQH; ++h) {
            bm[h] = warp_max(l[h]);
            if ((j & 31) == 0) red[j >> 5][h] = bm[h];
        }
        __syncthreads();
        float scale_old[kQH];
#pragma unroll
        for (int h = 0; h < kQH; ++h) {
            bm[h] = fmaxf(fmaxf(red[0][h], red[1][h]), fmaxf(red[2][h], red[3][h]));
            const float nm = fmaxf(m[h], bm[h]);
            scale_old[h] = m[h] == -FLT_MAX ? 0.f : __expf(m[h] - nm);
            m[h] = nm;
        }
        __syncthreads();
        const bool mine = ok && j % per == 0;   // one of a cell's threads carries its weight
#pragma unroll
        for (int h = 0; h < kQH; ++h) {
            const float wgt = mine && h < QH ? __expf(l[h] - m[h]) : 0.f;
            if (j % per == 0) sw[h][cell] = wgt;
            const float ws = warp_sum(wgt);
            if ((j & 31) == 0) red[j >> 5][kQH + h] = ws;
        }
        __syncthreads();
#pragma unroll
        for (int h = 0; h < kQH; ++h)
            s[h] = s[h] * scale_old[h] + red[0][kQH + h] + red[1][kQH + h] + red[2][kQH + h] + red[3][kQH + h];
        // the V step: thread j is value j
        float acc[kQH];
#pragma unroll
        for (int h = 0; h < kQH; ++h) acc[h] = 0.f;
        const int nc = (int) min((int64_t) a.sub, s1 - b0);
        if (j < a.D) {
            for (int ci = 0; ci < nc; ++ci) {
                const uint8_t* vc = a.c.v + (size_t) (b0 + ci) * cbytes;
                float vv;
                if (a.c.q8) {
                    const uint8_t* vb = vc + ((size_t) g * (a.D / 32) + j / 32) * 34;
                    vv = __half2float(*(const __half*) vb) * (float) ((const int8_t*) vb)[2 + j % 32];
                } else {
                    vv = __half2float(((const __half*) vc)[(size_t) g * a.D + j]);
                }
#pragma unroll
                for (int h = 0; h < kQH; ++h)
                    if (h < QH) acc[h] += sw[h][ci] * vv;
            }
#pragma unroll
            for (int h = 0; h < kQH; ++h) so[h] = so[h] * scale_old[h] + acc[h];
        }
        __syncthreads();
    }
    for (int h = 0; h < QH; ++h) {
        float* o = out + (size_t) h * pstride;
        if (j == 0) {
            o[0] = m[h];
            o[1] = s[h];
        }
        if (j < a.D) o[2 + j] = so[h];
    }
}

/// The stretches' partial sums joined with the zero sink, times sigmoid(gate): the attention's output of T rows as
/// floats ([T][H * D]) and as its q8_1 image.  A block a row and a query head; D threads (a warp a q8_1 block).
struct AttnOutIn {
    const float* part;    // [T][H][n_chunks][D + 2]
    const float* gate;    // [T][H * D], the gate's projection
    float* out;           // [T][H * D] or null
    Q81* q;               // [T][H * D / 32]
    const int32_t* dyn;   // null, or {chunk, stretches in use}
    int H, D, n_chunks;
};
__global__ void __launch_bounds__(128) attn_out_kernel(AttnOutIn a) {
    const int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    const int used = a.dyn ? a.dyn[1] : a.n_chunks;
    const float* p = a.part + ((size_t) t * a.H + h) * a.n_chunks * (a.D + 2);
    float mx = 0.f;   // the sink's logit
    for (int c = 0; c < used; ++c) mx = fmaxf(mx, p[(size_t) c * (a.D + 2)]);
    float sum = __expf(-mx), acc = 0.f;
    for (int c = 0; c < used; ++c) {
        const float* pc = p + (size_t) c * (a.D + 2);
        if (pc[1] == 0.f) continue;
        const float w = __expf(pc[0] - mx);
        sum += w * pc[1];
        acc += w * pc[2 + i];
    }
    const size_t at = ((size_t) t * a.H + h) * a.D + i;
    const float y = acc / sum / (1.0f + __expf(-a.gate[at]));
    if (a.out) a.out[at] = y;
    const float amax = warp_max(fabsf(y)), s = warp_sum(y);
    const float d = q8_1_finite(amax / 127.0f);
    Q81* q = a.q + at / 32;
    q->qs[i & 31] = q8_1_quant(y, d, amax);
    if ((i & 31) == 0) q->ds = q8_1_ds(d, s);
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
    if (i == 0) {
        int ng = 0, nm = 0;
        for (int j = 0; j < n; ++j) {
            ng += lead[j] == j;
            nm += lead[j] < 0;
        }
        *a.n_groups = ng;
        *a.n_miss = nm;
        int at = 0;
        for (int gi = 0; gi < ng; ++gi)
            for (int j = 0; j < n; ++j)
                if (rank_[j] == gi) {
                    start[gi] = at;
                    at += cnt[j];
                    break;
                }
        start[ng] = at;
        for (int gi = 0; gi <= ng; ++gi) a.grp_start[gi] = start[gi];
    }
    __syncthreads();
    if (!held) return;
    const int gi = rank_[l];
    if (l == i) a.grp_ptr[gi] = a.slots + (unsigned long long) a.slot_of[e] * a.stride;
    const int at = start[gi] + before;
    a.ent_tok[at] = i / a.top_k;
    a.ent_dst[at] = i;
}

/// The MoE's output of T rows: the routed experts' rows by their weights, in the router's order.  rows [T * top_k][N].
__global__ void __launch_bounds__(kTB) expert_sum_kernel(const float* __restrict__ rows, const float* __restrict__ w,
                                                         float* __restrict__ y, int N, int top_k) {
    const int t = blockIdx.y, i = blockIdx.x * kTB + threadIdx.x;
    if (i >= N) return;
    float acc = 0.f;
    for (int k = 0; k < top_k; ++k) acc += rows[((size_t) t * top_k + k) * N + i] * w[(size_t) t * top_k + k];
    y[(size_t) t * N + i] = acc;
}

// ---------------------------------------------------------------------------------------------- the head
/// cap tanh(x / cap) over T rows of logits
__global__ void softcap_kernel(float* __restrict__ lg, long long n, float cap) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) lg[i] = cap * tanhf(lg[i] / cap);
}

}  // namespace ik
