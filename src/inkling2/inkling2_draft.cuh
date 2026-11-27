// src/inkling2/inkling2_draft.cuh - the kernels of Inkling 2's block drafter (DSpark: GGUF architecture dflash with a
// Markov head, unsloth's dspark-*.gguf; llama.cpp's src/models/dspark.cpp, graph_inkling2, is the reference), for
// inkling2_engine.cu beside inkling2_kernels.cuh, whose norms, cells and attention it shares:
//
//   * the target's states of each row (the inputs of the tapped layer applications and the final normed state)
//     mixed a drafter layer, normed, and projected to that layer's context K and V at the row's position;
//   * a block of rows (the last token and masks) through the layers: q and k normed a head and rotated by position
//     (RoPE, pairs i and i + D / 2), attention over the context cells and the block's own without a sink, SwiGLU;
//   * the recurrent head: a state of R values stepped a drafted position, the position's logits the target's head
//     over the block row plus a correction, which is M . latent with M = head x hidden_out made once and kept as
//     signed bytes with a scale a token (a step's latent as bytes too: its values are tanh's).
#pragma once

#include "inkling2_kernels.cuh"
#include "lamina/kernels/sampler.hpp"

namespace ik {

// ---------------------------------------------------------------------------------------------- the context
/// A tapped state of T rows into the drafter's mixes: out[r][t] = (first ? 0 : out[r][t]) + c[r] x[t], r < nr (out
/// [nr][cap][N]).  A thread four values.
struct FuseIn {
    const float* x;   // [T][N]
    float* out;
    float c[4];
    int nr, N, cap, first;
};
__global__ void __launch_bounds__(kTB) fuse_kernel(FuseIn a) {
    const int t = blockIdx.y, i4 = blockIdx.x * kTB + threadIdx.x;
    if (i4 >= a.N / 4) return;
    const float4 x = ld4(a.x + (size_t) t * a.N, i4);
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        if (r >= a.nr) continue;
        float* o = a.out + ((size_t) r * a.cap + t) * a.N;
        float4 y = mul4(x, a.c[r]);
        if (!a.first) {
            const float4 was = ld4(o, i4);
            y = make_float4(was.x + y.x, was.y + y.y, was.z + y.z, was.w + y.w);
        }
        st4(o, i4, y);
    }
}

/// T rows' heads of D values normed (x rsqrt(mean(x^2) + eps) w) and rotated by their row's position: cs [T][D / 2][2],
/// the cosines and sines of pos theta_i (the host's), pairs i and i + D / 2.  y [T][heads][D].  A block a row and a
/// head; D = 128 threads.
__global__ void __launch_bounds__(128) head_rope_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                                        const float* __restrict__ cs, float* __restrict__ y, int heads,
                                                        int D, float eps) {
    const int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    __shared__ float part[4];
    __shared__ float val[128];
    const size_t at = ((size_t) t * heads + h) * D + i;
    float v = x[at];
    const float ss = lamina::kernels::block128_sum(v * v, part);
    val[i] = v * rsqrtf(ss / (float) D + eps) * w[i];
    __syncthreads();
    const int half = D / 2, k = i < half ? i : i - half;
    const float2 r = *reinterpret_cast<const float2*>(cs + ((size_t) t * half + k) * 2);
    const float x0 = val[k], x1 = val[k + half];
    y[at] = i < half ? x0 * r.x - x1 * r.y : x0 * r.y + x1 * r.x;
}

// ---------------------------------------------------------------------------------------------- a block's layers
/// The joined sums of a row and head (attn_join_kernel's: the largest logit, the sum, the weighted V) as the
/// attention's output without a sink: out [T][H * D] and its q8_1 image.  A block a row and a head; D threads.
__global__ void __launch_bounds__(128) attn_avg_kernel(const float* __restrict__ sum, float* __restrict__ out,
                                                       Q81* __restrict__ q, int H, int D) {
    const int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    const float* p = sum + ((size_t) t * H + h) * (D + 2);
    const float y = p[1] != 0.f ? p[2 + i] / p[1] : 0.f;
    const size_t at = ((size_t) t * H + h) * D + i;
    out[at] = y;
    q8_1_store(q + at / 32, y, i & 31);
}

/// x += y over n floats
__global__ void add_kernel(float* __restrict__ x, const float* __restrict__ y, long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

/// h = silu(g) u over rows of n values, and its q8_1 image (n / 32 blocks a row).  A warp a block.
__global__ void __launch_bounds__(kTB) swiglu_kernel(const float* __restrict__ g, const float* __restrict__ u,
                                                     float* __restrict__ h, Q81* __restrict__ q, int n) {
    const int t = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int b = blockIdx.x * (kTB / 32) + warp;
    if (b >= n / 32) return;
    const size_t at = (size_t) t * n + b * 32 + lane;
    const float gv = g[at], x = gv / (1.0f + __expf(-gv)) * u[at];
    h[at] = x;
    q8_1_store(q + (size_t) t * (n / 32) + b, x, lane);
}

// ---------------------------------------------------------------------------------------------- the recurrent head
/// The head's product made once: M[v][c0 + t] = half(y[t][v]), y the target's head over T of hidden_out's columns.
__global__ void head_m_kernel(const float* __restrict__ y, __half* __restrict__ M, int V, int R, int c0, int T) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long) T * V) return;
    const int t = (int) (i / V), v = (int) (i % V);
    M[(size_t) v * R + c0 + t] = __float2half(y[i]);
}
/// The head's product as a file folds it ([V][R] floats, M's own order): M = half(w), n values from where `M` points.
__global__ void half_rows_kernel(const float* __restrict__ w, __half* __restrict__ M, long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) M[i] = __float2half(w[i]);
}
/// Its rows as signed bytes: scale[v] = max |M[v]| / 127 / 127 (a latent's bytes are its values times 127), M8[v] =
/// round(M[v] / (max / 127)).  A block a token; R = 256 threads.
__global__ void __launch_bounds__(256) head_m8_kernel(const __half* __restrict__ M, int8_t* __restrict__ M8,
                                                      float* __restrict__ scale, int R) {
    __shared__ float rv[8];
    const int v = blockIdx.x, i = threadIdx.x;
    const float x = __half2float(M[(size_t) v * R + i]);
    const float wm = warp_max(fabsf(x));
    if ((i & 31) == 0) rv[i >> 5] = wm;
    __syncthreads();
    const float mx =
        fmaxf(fmaxf(fmaxf(rv[0], rv[1]), fmaxf(rv[2], rv[3])), fmaxf(fmaxf(rv[4], rv[5]), fmaxf(rv[6], rv[7])));
    const float d = mx / 127.0f;
    M8[(size_t) v * R + i] = (int8_t) (d > 0.f ? roundf(x / d) : 0.f);
    if (i == 0) scale[v] = d / 127.0f;
}

/// The part of the head's projection that stays over a block: pre[t][o] = W[o][R2 ..] . [h_t | ctx] + b[o] (W
/// [n_out][R2 + 2 N] floats: the state and the last token's row first; h [T][N] the block's rows, ctx [N] the context
/// of the position before the block).  A block an output.
struct JointPreIn {
    const float *W, *b, *h, *ctx;
    float* pre;   // [T][n_out]
    int R2, N, T, n_out;
};
__global__ void __launch_bounds__(kTB) joint_pre_kernel(JointPreIn a) {
    const int o = blockIdx.x;
    __shared__ float red[8][kMaxT + 1];
    const float* w = a.W + (size_t) o * (a.R2 + 2 * a.N) + a.R2;
    float v[kMaxT + 1];
#pragma unroll
    for (int t = 0; t <= kMaxT; ++t) v[t] = 0.f;
    for (int i = threadIdx.x; i < a.N; i += kTB) {
        const float wh = w[i];
#pragma unroll
        for (int t = 0; t < kMaxT; ++t)
            if (t < a.T) v[t] += wh * a.h[(size_t) t * a.N + i];
        v[kMaxT] += w[a.N + i] * a.ctx[i];
    }
    block_sums<kMaxT + 1>(v, red);
    if (threadIdx.x < a.T) a.pre[(size_t) threadIdx.x * a.n_out + o] = v[threadIdx.x] + v[kMaxT] + a.b[o];
}

/// One step of the head: j = W[.][0 .. 2R) . [s | w1[prev]] + pre (3 R values: R = 256); gate = sigmoid(j[0 .. R)),
/// cand = tanh(j[R .. 2R)), s = cand + gate (s - cand), lat = tanh(j[2R .. 3R)).  w1 [V][R] bfloat16.  One block of R
/// threads.
struct StepIn {
    const float* W;     // [3 R][stride]
    const float* pre;   // [3 R], the block row's
    const uint16_t* w1;
    const int32_t* prev;   // the token before
    float* s;              // [R]
    int8_t* lat;           // [R]: the latent's values times 127
    int R, stride;
    const int32_t* stop;   // not 0: the drafts have ended (nothing is done)
};
__global__ void __launch_bounds__(256) head_step_kernel(StepIn a) {
    if (*a.stop) return;
    const int r = threadIdx.x;
    __shared__ __align__(16) float in[512];
    in[r] = a.s[r];
    in[a.R + r] = __uint_as_float((unsigned) a.w1[(size_t) *a.prev * a.R + r] << 16);
    __syncthreads();
    float j[3];
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        const float* w = a.W + (size_t) (c * a.R + r) * a.stride;
        float acc = 0.f;
        for (int i4 = 0; i4 < 2 * a.R / 4; ++i4) acc += dot4(ld4(w, i4), ld4(in, i4));
        j[c] = acc + a.pre[c * a.R + r];
    }
    const float gate = 1.0f / (1.0f + __expf(-j[0])), cand = tanhf(j[1]);
    a.s[r] = cand + gate * (in[r] - cand);
    a.lat[r] = (int8_t) roundf(127.0f * tanhf(j[2]));
}

/// A drafted position's logits over the vocabulary, not kept: l[v] = cap tanh((base[v] + c0[v] + scale[v] M8[v] .
/// lat) / cap) (M8 [V][R] and lat signed bytes, R = 256).  A block's largest (the lowest token of equals) and its sum
/// of exp(l - it): part [blocks][3] {max, token, sum}.  A thread a token.
struct VocabIn {
    const float *base, *c0, *scale;
    const int8_t *M8, *lat;
    float* part;
    int V, R;
    float cap;
    const int32_t* stop;
};
__global__ void __launch_bounds__(256) head_vocab_kernel(VocabIn a) {
    if (*a.stop) return;
    __shared__ __align__(16) int sl[64];
    __shared__ float rv[8];
    __shared__ int ri[8];
    const int i = threadIdx.x, warp = i >> 5, lane = i & 31, v = blockIdx.x * 256 + i;
    if (i < a.R / 4) sl[i] = reinterpret_cast<const int*>(a.lat)[i];
    __syncthreads();
    float l = -FLT_MAX;
    if (v < a.V) {
        const int4* row = reinterpret_cast<const int4*>(a.M8 + (size_t) v * a.R);
        int sumi = 0;
        for (int k = 0; k < a.R / 16; ++k) {
            const int4 w = row[k];
            const int4 x = *reinterpret_cast<const int4*>(sl + 4 * k);
            sumi = __dp4a(w.x, x.x, sumi);
            sumi = __dp4a(w.y, x.y, sumi);
            sumi = __dp4a(w.z, x.z, sumi);
            sumi = __dp4a(w.w, x.w, sumi);
        }
        l = a.cap * tanhf((a.base[v] + a.c0[v] + a.scale[v] * (float) sumi) / a.cap);
    }
    float bv = l;
    int bi = v;
    warp_argmax(bv, bi);
    if (lane == 0) {
        rv[warp] = bv;
        ri[warp] = bi;
    }
    __syncthreads();
    float mx = rv[0];
    int mi = ri[0];
#pragma unroll
    for (int w = 1; w < 8; ++w)
        if (rv[w] > mx || (rv[w] == mx && ri[w] < mi)) {
            mx = rv[w];
            mi = ri[w];
        }
    __syncthreads();
    const float s = warp_sum(v < a.V ? __expf(l - mx) : 0.f);
    if (lane == 0) rv[warp] = s;
    __syncthreads();
    if (i == 0) {
        float* o = a.part + (size_t) blockIdx.x * 3;
        o[0] = mx;
        o[1] = (float) mi;
        o[2] = ((rv[0] + rv[1]) + (rv[2] + rv[3])) + ((rv[4] + rv[5]) + (rv[6] + rv[7]));
    }
}

/// A sampled request's draft j (its probability q under the drafter, the draw u of the position it takes, the
/// temperature T): its chance to be the token that row picks - sigmoid(c + d logit(q)), the same token as the row's
/// likeliest, times sigmoid((a + b logit(q) - logit(u)) / w), u within that token's share (a = a0 + a1 / T,
/// b = b0 + b1 / T, w = exp(w0 + w1 log T)): lamina/core/draft_pass.hpp's model, its constants this drafter's (fitted
/// to 15,940 drafts of the model's DSpark block drafter in Q8_0 at temperatures 0.6-1.2, top_k 64, four prompts), a set
/// for the first draft, the second, and the others.
__device__ __forceinline__ float draft_chance(int j, float q, float u, float T) {
    constexpr float k[3][8] = {{-0.1261f, 0.6805f, -1.2603f, 2.8356f, 0.5522f, 0.1070f, 0.3819f, -0.3074f},
                               {-0.2878f, 0.5908f, -2.2398f, 3.7768f, 0.9437f, -0.1491f, 0.4883f, -0.5104f},
                               {-0.4506f, 0.5293f, -1.3143f, 3.6593f, -0.6258f, 1.7322f, 0.5370f, -0.5615f}};
    const float* c = k[min(j, 2)];
    auto lg = [](float p) {
        p = fminf(fmaxf(p, 1e-6f), 1.0f - 1e-6f);
        return logf(p / (1.0f - p));
    };
    const float lq = lg(q), w = expf(c[6] + c[7] * logf(T));
    const float same = 1.0f / (1.0f + expf(-(c[0] + c[1] * lq)));
    const float share = 1.0f / (1.0f + expf(-((c[2] + c[3] / T + (c[4] + c[5] / T) * lq - lg(u)) / w)));
    return same * share;
}

/// The blocks' parts joined: the position's token (the first of the largest logits) into out[0], its probability's
/// bits into out[1], and the token into *prev for the next step.  The drafts' joint probability so far is in ctl[0]
/// (float bits), the floor in ctl[1], and ctl[2] is set once the joint falls under the floor: the steps after do
/// nothing, and their out[0] stays what it was (-1).  A sampled request (ctl[3]) joins each draft's chance to pass
/// under its position's draw instead (draft_chance: Philox(seed ctl[4..5], the position ctl[6..7] + j), the
/// temperature ctl[8]).  One block; a thread up to four parts.  j: the draft's step.
__global__ void __launch_bounds__(1024) head_pick_kernel(const float* __restrict__ part, int nb,
                                                         int32_t* __restrict__ out, int32_t* __restrict__ prev,
                                                         int32_t* __restrict__ ctl, int j) {
    if (ctl[2]) return;
    __shared__ float rv[32];
    __shared__ int ri[32];
    const int i = threadIdx.x, warp = i >> 5, lane = i & 31;
    float bv = -FLT_MAX;
    int bi = 0x7fffffff;
    for (int b = i; b < nb; b += 1024) {
        const float m = part[(size_t) b * 3];
        const int tk = (int) part[(size_t) b * 3 + 1];
        if (m > bv || (m == bv && tk < bi)) {
            bv = m;
            bi = tk;
        }
    }
    warp_argmax(bv, bi);
    if (lane == 0) {
        rv[warp] = bv;
        ri[warp] = bi;
    }
    __syncthreads();
    float mx = rv[0];
    int mi = ri[0];
    for (int w = 1; w < 32; ++w)
        if (rv[w] > mx || (rv[w] == mx && ri[w] < mi)) {
            mx = rv[w];
            mi = ri[w];
        }
    __syncthreads();
    float s = 0.f;
    for (int b = i; b < nb; b += 1024) s += part[(size_t) b * 3 + 2] * __expf(part[(size_t) b * 3] - mx);
    s = warp_sum(s);
    if (lane == 0) rv[warp] = s;
    __syncthreads();
    if (i == 0) {
        float z = 0.f;
        for (int w = 0; w < 32; ++w) z += rv[w];
        float joint = __int_as_float(ctl[0]) / z;
        if (ctl[3] != 0) {
            const uint64_t seed = (uint64_t) (uint32_t) ctl[4] | (uint64_t) (uint32_t) ctl[5] << 32;
            const uint64_t at = (uint64_t) (uint32_t) ctl[6] | (uint64_t) (uint32_t) ctl[7] << 32;
            joint = __int_as_float(ctl[0]) * draft_chance(j, 1.0f / z,
                                                          lamina::kernels::philox_uniform(seed, at + (uint64_t) j),
                                                          __int_as_float(ctl[8]));
        }
        if (!(joint >= __int_as_float(ctl[1]))) {
            ctl[2] = 1;
            return;
        }
        ctl[0] = __float_as_int(joint);
        out[0] = mi;
        out[1] = __float_as_int(1.0f / z);
        *prev = mi;
    }
}

}   // namespace ik
