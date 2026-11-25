// src/vision/cuda_encoder.cu - the image encoder on the engine's own CUDA kernels (include/lamina/vision/encoder.hpp).
//
// The tower and the adapters of both projector types as products on the tensor cores (kernels/gemm.hpp: FP16 in,
// FP32 sums; FP32 on the float units for the few FP32 matrices) or, for a Q8_0 matrix, through llama.cpp's MMQ, with
// small kernels between them and one tensor-core kernel for the attention.  Activations are FP32 between the
// products; the products' inputs are rounded to FP16 (to q8_1 for MMQ) and summed in FP32.
//
//   patches     host: each 16x16x3 patch a row of 768 (Qwen: in the order of the 2x2 groups); device: one FP32
//               product with the patch kernel (Qwen: its two kernels summed), the bias, the position table added
//               (resampled on the host: projector.hpp)
//   a block     LayerNorm, q k v (one matrix, or three), Qwen: the 2-D rotary positions on q and k, the attention,
//               the output projection, the residual; LayerNorm, up, GELU, down, the residual
//   Qwen        a last LayerNorm, each 2x2 group's four rows as one, two linear layers with GELU between
//   Hy4         an RMS norm, each 2x2 group's four rows as one through the 2x2 convolution's matrix, GELU (erf), the
//               1x1 convolution's matrix, and the newline row after each row of cells
//
// The attention (k_vit_attn) is flash attention on the tensor cores in the form of the qwen35 path's prompt kernel
// (src/qwen35/q36_attn.cuh), without a mask: a block takes 64 query rows of one head and walks all cells in tiles of
// 32, double-buffered through cp.async; q, k and v are FP16 with each head's 72 values padded to 80 (five steps of
// 16), the scores and the outputs FP32 sums, the softmax online.  LAMINA_VISION_ATTN=ref computes it instead as two
// plain products around a softmax over the whole score matrix: the reference the kernel is checked against.
//
// On the device: the weights in one block in the file's types (BF16 turned to FP16; a Q8_0 matrix its blocks, which
// FP16 products dequantize into a scratch a product), sized by a first pass of load(); two buffers of a row a patch
// for the largest picture the projector's limit gives (made again for a picture past it), q, k and v, and a
// product's input and output for kChunk rows.  The adapters leave the rows in one of the two buffers.  All of it is
// the encoder's own allocation or, lent, one block of its caller's (EncoderParams::gpu_arena: the weights at its
// start, the buffers after them): lamina-server's encoder borrows the engine's expert-cache slots that way for a
// picture and loads its weights each time, the file's matrices read into one pinned buffer (0.17-0.31 s).
// LAMINA_VISION_MEM=1 prints the device's free memory and the time after each step of the start.
#include "backends.hpp"

#include "lamina/kernels/gemm.hpp"
#include "lamina/kernels/warp.cuh"
#include "lamina/prefill/moe_mmq.hpp"
#include "lamina/vision/projector.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace lamina::vision {
namespace {

namespace mmq = lamina::prefill::mmq;
using lamina::kernels::cp_async16;
using lamina::kernels::cp_commit;
using lamina::kernels::cp_wait;
using lamina::kernels::ldsm4;
using lamina::kernels::ldsm4t;

// ggml's type ids of the tensors a projector holds
constexpr int kF32 = 0, kF16 = 1, kQ8_0 = 8, kBF16 = 30;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

constexpr int kThreads = 256;
inline unsigned grid1(int64_t n) { return (unsigned) ((n + kThreads - 1) / kThreads); }

// ---- small kernels --------------------------------------------------------------------------------------------------

__global__ void k_to_f16(const float* x, __half* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}

/// BF16 bits to FP16 bits, in place
__global__ void k_bf16_to_f16(uint16_t* w, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned bits = (unsigned) w[i] << 16;
    const __half h = __float2half(__uint_as_float(bits));
    w[i] = *reinterpret_cast<const uint16_t*>(&h);
}

/// Q8_0 blocks (an FP16 scale, 32 int8) to FP16 values
__global__ void k_q8_to_f16(const uint8_t* blocks, __half* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint8_t* b = blocks + (i / 32) * 34;
    const float d = __half2float(*reinterpret_cast<const __half*>(b));
    y[i] = __float2half(d * (float) (int8_t) b[2 + i % 32]);
}

/// the sum of a block's partial sums: `v` this thread's, `part` a double a warp in shared memory
__device__ __forceinline__ double block_sum(double v, double* part) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = v;
    __syncthreads();
    double s = 0.0;
    for (int w = 0; w < (int) blockDim.x / 32; ++w) s += part[w];
    __syncthreads();
    return s;
}

/// LayerNorm as ggml_norm computes it (the mean and the variance in double), times w, plus b: grid (rows), block 128.
/// `rms`: no mean, x / sqrt(mean(x^2) + eps) (ggml_rms_norm), b unused.  y may be x.
__global__ void k_norm(const float* x, const float* w, const float* b, float* y, int E, float eps, bool rms) {
    __shared__ double part[4];
    const float* xr = x + (size_t) blockIdx.x * E;
    float* yr = y + (size_t) blockIdx.x * E;
    float mean = 0.0f;
    if (!rms) {
        double s = 0.0;
        for (int i = (int) threadIdx.x; i < E; i += (int) blockDim.x) s += (double) xr[i];
        mean = (float) (block_sum(s, part) / E);
    }
    double s2 = 0.0;
    for (int i = (int) threadIdx.x; i < E; i += (int) blockDim.x) {
        const float v = xr[i] - mean;
        s2 += (double) (v * v);
    }
    const float variance = (float) (block_sum(s2, part) / E);
    const float scale = 1.0f / sqrtf(variance + eps);
    for (int i = (int) threadIdx.x; i < E; i += (int) blockDim.x) {
        const float v = (xr[i] - mean) * scale * w[i];
        yr[i] = b != nullptr ? v + b[i] : v;
    }
}

__device__ __forceinline__ float gelu_tanh(float x) {   // ggml_gelu
    return 0.5f * x * (1.0f + tanhf(0.79788456080286535587989211986876f * x * (1.0f + 0.044715f * x * x)));
}
__device__ __forceinline__ float gelu_erf(float x) {   // ggml_gelu_erf
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
}

/// y[t][n] (rows `ld` apart) += b[n], then act: 0 none, 1 GELU (tanh), 2 GELU (erf)
__global__ void k_bias_act(float* y, const float* b, int64_t T, int n, int64_t ld, int act) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * n) return;
    const int64_t t = i / n;
    const int c = (int) (i % n);
    float v = y[t * ld + c] + b[c];
    if (act == 1) v = gelu_tanh(v);
    else if (act == 2) v = gelu_erf(v);
    y[t * ld + c] = v;
}

/// x[t][n] += y[t][n] + b[n] (b may be null)
__global__ void k_add(float* x, const float* y, const float* b, int64_t T, int n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * n) return;
    const float v = y[i] + (b != nullptr ? b[i % n] : 0.0f);
    x[i] += v;
}

/// x[t][n] += (y[t][n] + b[n]) * s[n] (b may be null): a block's output through its learned scale (Inkling 2's)
__global__ void k_add_scaled(float* x, const float* y, const float* b, const float* s, int64_t T, int n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * n) return;
    x[i] += (y[i] + (b != nullptr ? b[i % n] : 0.0f)) * s[i % n];
}

constexpr int kHeadPad = 80;   // a head's values as the attention takes them: its 72 (Inkling 2's 64) and zeros to 80
constexpr int kRopeFreqs = 18; // rotary pairs a section at most: Qwen's d_head / 4 (Inkling 2's 16)

/// A chunk's q, k and v rows ([T][3 E] FP32, the three one after the other, plus their bias) into the attention's
/// FP16 layouts: q16 [token][head][80], k16 and v16 [head][token][80], a head's values padded with zeros.  With `cs`
/// (Qwen): the 2-D rotary positions on q and k as ggml's rope of the vision type applies them: of a head's 72 values
/// the pairs (i, i + 36), i < 18 turned by the patch's row and i >= 18 by its column, cs[position][i mod 18] the
/// angle's cosine and sine.  The tokens are in the order of the 2x2 groups: token p is patch (2 my + dy, 2 mx + dx)
/// of group p / 4 = my * (pw / 2) + mx, p mod 4 = 2 dy + dx.  `ink` (Inkling 2, with `cs`): ggml's rope over each
/// half of a head, as mtmd's build_rope_2d asks for it: the pairs (2 i, 2 i + 1), the first half's turned by the
/// token's column and the second's by its row, both counted from 1 (token 0 is the class token, at 0: not turned; the
/// patches follow in the grid's order), cs[position][i] the pair's cosine and sine.  grid (T), block 256.
__global__ void k_qkv(const float* qkv, const float* bias, __half* q16, __half* k16, __half* v16, const float* cs, int t0,
                      int N, int H, int D, int pw, int ink) {
    const int t = (int) blockIdx.x, gt = t0 + t;
    const int E = H * D;
    const float* row = qkv + (size_t) t * 3 * E;
    int prow = 0, pcol = 0;
    if (cs != nullptr && ink == 0) {
        const int cell = gt >> 2, sub = gt & 3, half_w = pw >> 1;
        prow = 2 * (cell / half_w) + (sub >> 1);
        pcol = 2 * (cell % half_w) + (sub & 1);
    } else if (cs != nullptr && gt > 0) {
        prow = (gt - 1) / pw + 1;
        pcol = (gt - 1) % pw + 1;
    }
    const int half = D / 2;   // 36
    for (int i = (int) threadIdx.x; i < H * kHeadPad; i += (int) blockDim.x) {
        const int h = i / kHeadPad, d = i % kHeadPad;
        float q = 0.0f, k = 0.0f, v = 0.0f;
        if (d < D) {
            const int e = h * D + d;
            q = row[e] + bias[e];
            k = row[E + e] + bias[E + e];
            v = row[2 * E + e] + bias[2 * E + e];
            if (cs != nullptr && ink != 0) {
                const int other = (d & 1) != 0 ? e - 1 : e + 1;   // the pair's other value (the halves are even)
                const float* a = cs + ((size_t) (d < half ? pcol : prow) * kRopeFreqs + ((d < half ? d : d - half) >> 1)) * 2;
                const float qo = row[other] + bias[other], ko = row[E + other] + bias[E + other];
                if ((d & 1) == 0) {   // x0 cos - x1 sin
                    q = q * a[0] - qo * a[1];
                    k = k * a[0] - ko * a[1];
                } else {              // x0 sin + x1 cos
                    q = qo * a[1] + q * a[0];
                    k = ko * a[1] + k * a[0];
                }
            } else if (cs != nullptr) {
                const int ic = d < half ? d : d - half;          // the pair's index
                const int other = d < half ? e + half : e - half; // its other value
                const float* a = cs + ((size_t) (ic < kRopeFreqs ? prow : pcol) * kRopeFreqs + ic % kRopeFreqs) * 2;
                const float qo = row[other] + bias[other], ko = row[E + other] + bias[E + other];
                if (d < half) {   // x0 cos - x1 sin
                    q = q * a[0] - qo * a[1];
                    k = k * a[0] - ko * a[1];
                } else {          // x0 sin + x1 cos
                    q = qo * a[1] + q * a[0];
                    k = ko * a[1] + k * a[0];
                }
            }
        }
        q16[((size_t) gt * H + h) * kHeadPad + d] = __float2half(q);
        k16[((size_t) h * N + gt) * kHeadPad + d] = __float2half(k);
        v16[((size_t) h * N + gt) * kHeadPad + d] = __float2half(v);
    }
}

// ---- the attention on the tensor cores ----------------------------------------------------------------------------------

namespace va {

using lamina::kernels::h2;
using lamina::kernels::mma16816;

constexpr int kD = kHeadPad;   // values a head (five steps of 16 in the scores, ten 8-value tiles in the outputs)
constexpr int kDs = 128;       // a cell's stride in a shared-memory tile: 16 chunks of 8 values, for the bank swizzle
constexpr int kBc = 32;        // cells a tile
constexpr int kRows = 64;      // query rows a block

// a K or V tile's cell c, 16-byte chunk j (of the cell's 10), in halves
__device__ __forceinline__ int kv_at(int c, int j) { return c * kDs + ((j ^ (c & 7)) << 3); }
// the probabilities' row r, 16-byte chunk j, in halves
__device__ __forceinline__ int p_at(int r, int j) { return r * kBc + ((j ^ ((r >> 1) & 3)) << 3); }

struct __align__(16) Smem {
    __half k[2][kBc * kDs];
    __half v[2][kBc * kDs];
    __half p[kRows * kBc];
    float red[2][2][kRows];   // [max | sum][the m-tile's warp: cells 0-15 | 16-31][row]
};

// cells [base, base + kBc) of a head's K or V (n of them there: zeros past them, which the mask keeps out) into a tile
__device__ __forceinline__ void load_tile(__half* dst, const __half* src, int n) {
    for (int i = (int) threadIdx.x; i < kBc * (kD / 8); i += (int) blockDim.x) {
        const int c = i / (kD / 8), j = i % (kD / 8);
        __half* d = dst + kv_at(c, j);
        if (c < n) cp_async16(d, src + (size_t) c * kD + j * 8);
        else *reinterpret_cast<uint4*>(d) = make_uint4(0, 0, 0, 0);
    }
}

/// One head's attention over N tokens: q [N][H][80], k and v [H][N][80] (FP16, a head's 72 values and 8 zeros), out
/// [N][H][D] (FP32, D = 72).  grid (ceil(N / 64), H), block 256, shared sizeof(Smem).
/// Warp w: the rows of m-tile w & 3 (16 of the block's 64); in the scores the cells (w >> 2) * 16 .. +16 of a tile,
/// in the outputs the value tiles 0-5 (w >> 2 = 0) or 6-9.
__global__ void __launch_bounds__(256) k_vit_attn(const __half* __restrict__ q, const __half* __restrict__ kc,
                                                  const __half* __restrict__ vc, float* __restrict__ out, int N, int H,
                                                  int D, float scale) {
    extern __shared__ __align__(16) unsigned char raw[];
    Smem& sm = *reinterpret_cast<Smem*>(raw);
    const int w = (int) threadIdx.x >> 5, lane = (int) threadIdx.x & 31, tig = lane & 3;
    const int mt = w & 3, hf = w >> 2;
    const int head = (int) blockIdx.y, t0 = (int) blockIdx.x * kRows;
    // this thread's two rows of the m-tile (a row past the last token sees nothing and is not written)
    const int rA = mt * 16 + (lane >> 2), rB = rA + 8;
    const int tA = t0 + rA, tB = t0 + rB;
    const bool okA = tA < N, okB = tB < N;
    const size_t qA = okA ? ((size_t) tA * H + head) * kD : 0;
    const size_t qB = okB ? ((size_t) tB * H + head) * kD : 0;
    // the queries as A fragments: a0 (row rA, values 16k + 2 tig, +1), a1 (row rB), a2 / a3 the same 8 values on
    unsigned qa[kD / 16][4];
#pragma unroll
    for (int k = 0; k < kD / 16; ++k) {
        const int d = 16 * k + 2 * tig;
        qa[k][0] = okA ? *reinterpret_cast<const unsigned*>(q + qA + d) : 0u;
        qa[k][1] = okB ? *reinterpret_cast<const unsigned*>(q + qB + d) : 0u;
        qa[k][2] = okA ? *reinterpret_cast<const unsigned*>(q + qA + d + 8) : 0u;
        qa[k][3] = okB ? *reinterpret_cast<const unsigned*>(q + qB + d + 8) : 0u;
    }
    const int n_tiles = (N + kBc - 1) / kBc;
    const __half* kh = kc + (size_t) head * N * kD;
    const __half* vh = vc + (size_t) head * N * kD;
    float mA = -INFINITY, mB = -INFINITY, lA = 0.f, lB = 0.f;
    constexpr int kOut = 6;   // value tiles a warp at most
    const int c_first = hf == 0 ? 0 : 6, n_pairs = hf == 0 ? 3 : 2;
    float o[kOut][4];
#pragma unroll
    for (int n = 0; n < kOut; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    load_tile(sm.k[0], kh, min(kBc, N));
    load_tile(sm.v[0], vh, min(kBc, N));
    cp_commit();
    const int li = lane >> 3, lr = lane & 7;   // ldmatrix: this lane's matrix and row in it
    for (int it = 0; it < n_tiles; ++it) {
        const int st = it & 1, base = it * kBc;
        if (it + 1 < n_tiles) {
            const int nb = base + kBc;
            load_tile(sm.k[st ^ 1], kh + (size_t) nb * kD, min(kBc, N - nb));
            load_tile(sm.v[st ^ 1], vh + (size_t) nb * kD, min(kBc, N - nb));
        }
        cp_commit();
        cp_wait<1>();
        __syncthreads();
        // scores: n-tiles 0 and 1 of this warp's 16 cells
        float s[2][4];
#pragma unroll
        for (int n = 0; n < 2; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
        const __half* kt = sm.k[st];
        const int nc0 = hf * 16;
#pragma unroll
        for (int k = 0; k < kD / 16; ++k) {
            unsigned b[4];
            ldsm4(b, kt + kv_at(nc0 + ((li >> 1) << 3) + lr, 2 * k + (li & 1)));
            mma16816(s[0], qa[k], b[0], b[1]);
            mma16816(s[1], qa[k], b[2], b[3]);
        }
        // the cells past the last token out, the tile's row maxima (the four lanes of a row, then the m-tile's other warp)
        float xA = -INFINITY, xB = -INFINITY;
#pragma unroll
        for (int n = 0; n < 2; ++n)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int cell = base + nc0 + 8 * n + 2 * tig + e;
                s[n][e] = cell < N ? s[n][e] * scale : -INFINITY;
                s[n][2 + e] = cell < N ? s[n][2 + e] * scale : -INFINITY;
                xA = fmaxf(xA, s[n][e]);
                xB = fmaxf(xB, s[n][2 + e]);
            }
        xA = fmaxf(xA, __shfl_xor_sync(0xffffffffu, xA, 1));
        xA = fmaxf(xA, __shfl_xor_sync(0xffffffffu, xA, 2));
        xB = fmaxf(xB, __shfl_xor_sync(0xffffffffu, xB, 1));
        xB = fmaxf(xB, __shfl_xor_sync(0xffffffffu, xB, 2));
        if (tig == 0) {
            sm.red[0][hf][rA] = xA;
            sm.red[0][hf][rB] = xB;
        }
        __syncthreads();
        const float nA = fmaxf(mA, fmaxf(sm.red[0][0][rA], sm.red[0][1][rA]));
        const float nB = fmaxf(mB, fmaxf(sm.red[0][0][rB], sm.red[0][1][rB]));
        const float aA = nA == -INFINITY ? 1.f : expf(mA - nA), aB = nB == -INFINITY ? 1.f : expf(mB - nB);
        // the probabilities (FP16 into shared memory) and their row sums
        float sA = 0.f, sB = 0.f;
#pragma unroll
        for (int n = 0; n < 2; ++n) {
            const float p0A = s[n][0] == -INFINITY ? 0.f : expf(s[n][0] - nA);
            const float p1A = s[n][1] == -INFINITY ? 0.f : expf(s[n][1] - nA);
            const float p0B = s[n][2] == -INFINITY ? 0.f : expf(s[n][2] - nB);
            const float p1B = s[n][3] == -INFINITY ? 0.f : expf(s[n][3] - nB);
            sA += p0A + p1A;
            sB += p0B + p1B;
            const int c = nc0 + 8 * n + 2 * tig;   // the pair's first cell in the tile
            *reinterpret_cast<unsigned*>(sm.p + p_at(rA, c >> 3) + (c & 7)) = h2(p0A, p1A);
            *reinterpret_cast<unsigned*>(sm.p + p_at(rB, c >> 3) + (c & 7)) = h2(p0B, p1B);
        }
        sA += __shfl_xor_sync(0xffffffffu, sA, 1);
        sA += __shfl_xor_sync(0xffffffffu, sA, 2);
        sB += __shfl_xor_sync(0xffffffffu, sB, 1);
        sB += __shfl_xor_sync(0xffffffffu, sB, 2);
        if (tig == 0) {
            sm.red[1][hf][rA] = sA;
            sm.red[1][hf][rB] = sB;
        }
        __syncthreads();
        lA = lA * aA + sm.red[1][0][rA] + sm.red[1][1][rA];
        lB = lB * aB + sm.red[1][0][rB] + sm.red[1][1][rB];
        mA = nA;
        mB = nB;
        // outputs: this warp's value tiles of its m-tile's rows
#pragma unroll
        for (int n = 0; n < kOut; ++n) {
            o[n][0] *= aA;
            o[n][1] *= aA;
            o[n][2] *= aB;
            o[n][3] *= aB;
        }
        const __half* vt = sm.v[st];
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            unsigned a[4];
            const int row = mt * 16 + ((li & 1) << 3) + lr;
            ldsm4(a, sm.p + p_at(row, 2 * k + (li >> 1)));
#pragma unroll
            for (int n2 = 0; n2 < 3; ++n2) {
                if (n2 < n_pairs) {
                    unsigned b[4];
                    ldsm4t(b, vt + kv_at(16 * k + ((li & 1) << 3) + lr, c_first + 2 * n2 + (li >> 1)));
                    mma16816(o[2 * n2], a, b[0], b[1]);
                    mma16816(o[2 * n2 + 1], a, b[2], b[3]);
                }
            }
        }
        __syncthreads();   // the stage's K, V and probabilities are read before they are written again
    }
    // the output: divided by the sum; the padding's values are not written
    const float iA = lA > 0.f ? 1.f / lA : 0.f, iB = lB > 0.f ? 1.f / lB : 0.f;
#pragma unroll
    for (int n = 0; n < kOut; ++n) {
        const int d = (c_first + n) * 8 + 2 * tig;
        if (n < 2 * n_pairs && d < D) {
            if (okA) *reinterpret_cast<float2*>(out + ((size_t) tA * H + head) * D + d) = make_float2(o[n][0] * iA, o[n][1] * iA);
            if (okB) *reinterpret_cast<float2*>(out + ((size_t) tB * H + head) * D + d) = make_float2(o[n][2] * iB, o[n][3] * iB);
        }
    }
}

}  // namespace va

// ---- the reference attention's kernels ------------------------------------------------------------------------------------

/// a row of scores to its probabilities, in place: softmax(scale * s).  grid (rows), block 256.
__global__ void k_softmax_rows(float* s, int N, float scale) {
    __shared__ float part[8];
    float* r = s + (size_t) blockIdx.x * N;
    float m = -INFINITY;
    for (int i = (int) threadIdx.x; i < N; i += (int) blockDim.x) m = fmaxf(m, r[i] * scale);
    m = lamina::kernels::warp_max(m);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = m;
    __syncthreads();
    m = part[0];
    for (int w = 1; w < 8; ++w) m = fmaxf(m, part[w]);
    __syncthreads();
    float sum = 0.f;
    for (int i = (int) threadIdx.x; i < N; i += (int) blockDim.x) {
        const float e = expf(r[i] * scale - m);
        r[i] = e;
        sum += e;
    }
    sum = lamina::kernels::warp_sum(sum);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sum;
    __syncthreads();
    sum = 0.f;
    for (int w = 0; w < 8; ++w) sum += part[w];
    const float inv = 1.f / sum;
    for (int i = (int) threadIdx.x; i < N; i += (int) blockDim.x) r[i] *= inv;
}

/// x [rows][cols] in FP16 to its transpose [cols][rows] in FP32
__global__ void k_f16_to_f32_t(const __half* x, float* y, int rows, int cols) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int64_t) rows * cols) y[i] = __half2float(x[(i % rows) * cols + i / rows]);
}

/// a head's outputs o [B][80] into out [N][H][D] at rows t0 ..
__global__ void k_head_out(const float* o, float* out, int t0, int B, int H, int D, int head) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) B * D) return;
    const int b = (int) (i / D), d = (int) (i % D);
    out[((size_t) (t0 + b) * H + head) * D + d] = o[(size_t) b * kHeadPad + d];
}

// ---- the adapters' kernels ---------------------------------------------------------------------------------------------------

/// Hy4: each 2x2 group's four patch rows as one row, for the `T` cells from `cell0` on: g[t][(ky * 2 + kx) * E + c] =
/// x[(2 oy + ky) * pw + 2 ox + kx][c] of the cell cell0 + t = oy * ow + ox (the convolution's matrix is laid out to
/// match: CudaEncoder::load).
__global__ void k_gather_2x2(const float* x, float* g, int pw, int ow, int cell0, int T, int E) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) T * 4 * E) return;
    const int c = (int) (i % E), k = (int) ((i / E) & 3);
    const int cell = cell0 + (int) (i / (4 * (int64_t) E)), oy = cell / ow, ox = cell % ow;
    g[i] = x[((size_t) (2 * oy + (k >> 1)) * pw + 2 * ox + (k & 1)) * E + c];
}

// ---- the encoder -----------------------------------------------------------------------------------------------------------

/// A matrix [n_out][n_in] on the device: FP16 values, FP32 values, or Q8_0 blocks (with a zeroed tail: MMQ's steps
/// read past a matrix's end).
struct Mat {
    void* d = nullptr;
    int type = kF16;
    int64_t n_out = 0, n_in = 0;
    bool fp16 = false;   ///< a Q8_0 matrix whose products are FP16 ones: its values through a scratch a product
};

constexpr int kChunk = 1024;       // rows a product at most: its input and output buffers are this many rows
constexpr size_t kMmqTail = 4096;  // bytes after a Q8_0 matrix

class CudaEncoder final : public Encoder {
public:
    CudaEncoder() = default;
    ~CudaEncoder() override {
        if (dev_ < 0) return;
        cudaSetDevice(dev_);
        mctx_.reset();
        stage_free();
        for (void* p : work_) cudaFree(p);
        if (arena_ != nullptr && !lent_) cudaFree(arena_);
        if (st_ != nullptr) cudaStreamDestroy(st_);
    }
    CudaEncoder(const CudaEncoder&) = delete;
    CudaEncoder& operator=(const CudaEncoder&) = delete;

    /// What open() takes of the device for `p`: the weights' block and the buffers for the projector's largest
    /// picture.  No CUDA call.  throws std::runtime_error
    uint64_t measure(const EncoderParams& p) {
        proj_.open(p);
        settle(p);
        measuring_ = true;
        load();
        measuring_ = false;
        return up(arena_used_) + alloc(first_patches(), true);
    }

    /// throws std::runtime_error
    void open(const EncoderParams& p) {
        proj_.open(p);
        int n_dev = 0;
        if (cudaGetDeviceCount(&n_dev) != cudaSuccess || p.device < 0 || p.device >= n_dev)
            throw std::runtime_error("no GPU " + std::to_string(p.device) + " (CUDA sees " + std::to_string(n_dev) + ")");
        dev_ = p.device;
        ck(cudaSetDevice(dev_), "cudaSetDevice");
        ck(cudaStreamCreateWithFlags(&st_, cudaStreamNonBlocking), "the encoder's stream");
        mem("the context and a stream");
        ck(cudaFuncSetAttribute(va::k_vit_attn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(va::Smem)),
           "the attention kernel's shared memory");
        settle(p);
        // the weights in one block: load() once to measure it, once to fill it
        measuring_ = true;
        load();
        measuring_ = false;
        arena_bytes_ = arena_used_;
        arena_used_ = 0;
        if (p.gpu_arena != nullptr) {   // the caller's memory: the weights at its start, the buffers after them
            lent_ = true;
            const uint64_t need = up(arena_bytes_) + alloc(first_patches(), true);
            if (p.gpu_arena_bytes < need)
                throw std::runtime_error("the memory lent to the encoder (" + std::to_string(p.gpu_arena_bytes >> 20) +
                                         " MiB) is less than it takes (" + std::to_string((need >> 20) + 1) + " MiB)");
            arena_ = p.gpu_arena;
            lent_work_ = (uint8_t*) p.gpu_arena + up(arena_bytes_);
            lent_work_bytes_ = p.gpu_arena_bytes - up(arena_bytes_);
        } else if (cudaMalloc(&arena_, arena_bytes_) != cudaSuccess) {
            (void) cudaGetLastError();
            arena_ = nullptr;
            throw std::runtime_error("the projector's weights (" + std::to_string(arena_bytes_ >> 20) +
                                     " MiB) do not fit the GPU's free memory");
        }
        load();
        mem("the weights");
        alloc(first_patches(), false);
        mem("the buffers");
    }
    /// what the environment and the build settle of `p`
    void settle(const EncoderParams& p) {
        const char* attn = std::getenv("LAMINA_VISION_ATTN");
        ref_attn_ = attn != nullptr && std::string(attn) == "ref";
        // a Q8_0 matrix through MMQ (int8 tensor cores, the activations q8_1, llama.cpp's arithmetic for it) unless
        // the caller asks for FP16 products or MMQ has no tile for it here
        q8_fp16_ = p.q8_fp16 || !mmq::built() || !mmq::supported(kQ8_0);
    }
    /// the rows of the largest picture the projector's limit gives (its patches, and Inkling 2's class token): the
    /// buffers start at this size
    int64_t first_patches() const {
        return proj_.size.max_pixels / (proj_.patch_size * proj_.patch_size) + (proj_.kind == ProjectorKind::inkling2v ? 1 : 0);
    }
    static uint64_t up(uint64_t bytes) { return (bytes + 255) & ~(uint64_t) 255; }
    /// LAMINA_VISION_MEM=1: the device's free memory after a step of the start, and what the step took
    void mem(const char* step) {
        if (std::getenv("LAMINA_VISION_MEM") == nullptr) return;
        size_t fr = 0, total = 0;
        if (cudaMemGetInfo(&fr, &total) != cudaSuccess) return;
        const std::string took =
            mem_last_ == 0 ? "" : " (-" + std::to_string(((long long) mem_last_ - (long long) fr) / 1048576) + " MiB)";
        const auto now = std::chrono::steady_clock::now();
        std::fprintf(stderr, "vision mem: %s: %.0f MiB free%s, %.0f ms\n", step, (double) fr / 1048576.0, took.c_str(),
                     mem_last_ == 0 ? 0.0 : std::chrono::duration<double, std::milli>(now - mem_at_).count());
        mem_at_ = now;
        mem_last_ = fr;
    }
    size_t mem_last_ = 0;
    std::chrono::steady_clock::time_point mem_at_{};

    int n_embd() const override { return proj_.projection_dim; }
    uint64_t gpu_bytes() const override { return arena_bytes_ + work_bytes_; }

    bool encode(const Rgb& picture, Rows& out, std::string& err) override {
        try {
            ck(cudaSetDevice(dev_), "cudaSetDevice");
            encode_or_throw(picture, out);
            mem("a picture");
            return true;
        } catch (const std::exception& e) {
            err = e.what();
            return false;
        }
    }

private:
    // ---- memory ----------------------------------------------------------------------------------------------------
    /// A weight's place in the weights' block, 256 bytes apart at least.  One cudaMalloc for all of them: one each
    /// took 96 MiB more than a Q8_0 projector's 588 MiB (the allocator's granularity, over ~420 tensors).  While
    /// load() measures, the place is counted and nothing is there.
    void* dmalloc(size_t bytes) {
        const size_t at = (arena_used_ + 255) & ~(size_t) 255;
        arena_used_ = at + std::max<size_t>(bytes, 16);
        if (measuring_) return &measure_dummy_;
        if (arena_used_ > arena_bytes_) throw std::logic_error("the encoder's weights passed their measure");
        return (uint8_t*) arena_ + at;
    }
    void free_work() {
        for (void* p : work_) cudaFree(p);
        work_.clear();
        work_bytes_ = 0;
        n_max_ = chunk_ = 0;
    }
    /// Host memory to the device, on the encoder's stream and waited for: a kernel there never meets a buffer whose
    /// copy is still landing (a pageable cudaMemcpy returns before its transfer does).
    void to_device(void* dst, const void* src, size_t bytes) {
        if (measuring_) return;
        ck(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, st_), "a copy to the device");
        ck(cudaStreamSynchronize(st_), "a copy to the device");
    }
    float* upload(const std::vector<float>& v) {
        float* d = (float*) dmalloc(v.size() * sizeof(float));
        to_device(d, v.data(), v.size() * sizeof(float));
        return d;
    }
    float* vec(const std::string& name, int64_t n, bool required = true) {
        if (proj_.find(name) == nullptr) {
            if (required) throw std::runtime_error("the projector has no tensor " + name);
            return nullptr;
        }
        if (measuring_) return (float*) dmalloc((size_t) n * sizeof(float));
        const std::vector<float> v = proj_.floats(name);
        if ((int64_t) v.size() != n) throw std::runtime_error("the tensor " + name + " has another size than the tower's");
        return upload(v);
    }
    /// The matrices' staging buffer: pinned, as large as the largest of them (the measure's count), so each is read
    /// from the file's cache into the same warm pages and copied at the link's rate; pageable where pinning fails.
    uint8_t* stage(size_t bytes) {
        if (stage_ == nullptr && stage_pageable_.empty()) {
            if (cudaHostAlloc(&stage_, std::max(stage_need_, bytes), cudaHostAllocDefault) != cudaSuccess) {
                (void) cudaGetLastError();
                stage_ = nullptr;
            }
        }
        if (stage_ != nullptr && bytes <= stage_need_) return (uint8_t*) stage_;
        if (stage_pageable_.size() < bytes) stage_pageable_.resize(bytes);
        return stage_pageable_.data();
    }
    void stage_free() {
        if (stage_ != nullptr) cudaFreeHost(stage_);
        stage_ = nullptr;
        std::vector<uint8_t>().swap(stage_pageable_);
    }

    /// A matrix as the file has it: F16 as it is, BF16 turned to FP16, Q8_0 as blocks (or, for FP16 products, its
    /// values), F32 as it is.
    Mat mat(const std::string& name, int64_t n_out, int64_t n_in) {
        const TensorInfo* info = proj_.find(name);
        if (info == nullptr) throw std::runtime_error("the projector has no tensor " + name);
        if (info->ne[0] != n_in || info->ne[1] != n_out || info->ne[2] != 1)
            throw std::runtime_error("the tensor " + name + " has another shape than the tower's");
        const size_t bytes = info->bytes;
        stage_need_ = std::max(stage_need_, bytes);
        struct Raw {   // the file's bytes, in the staging buffer (the measure: none)
            const uint8_t* p = nullptr;
            const uint8_t* data() const { return p; }
        } raw;
        if (!measuring_) {
            raw.p = stage(bytes);
            proj_.read(name, *info, (void*) raw.p);
        }
        Mat m;
        m.n_out = n_out;
        m.n_in = n_in;
        const int64_t n = n_out * n_in;
        if (info->type == kF16 || info->type == kBF16) {
            m.type = kF16;
            m.d = dmalloc(bytes);
            to_device(m.d, raw.data(), bytes);
            if (info->type == kBF16 && !measuring_) k_bf16_to_f16<<<grid1(n), kThreads, 0, st_>>>((uint16_t*) m.d, n);
        } else if (info->type == kQ8_0) {
            if (n_in % 32 != 0) throw std::runtime_error("the tensor " + name + " is Q8_0 with rows that are no whole blocks");
            // the blocks as they are; a matrix MMQ has no tile for here is multiplied as FP16 values too
            m.type = kQ8_0;
            m.fp16 = q8_fp16_ || !mmq::fits(kQ8_0, n_out);
            m.d = dmalloc(bytes + kMmqTail);
            to_device(m.d, raw.data(), bytes);
            if (!measuring_) ck(cudaMemsetAsync((uint8_t*) m.d + bytes, 0, kMmqTail, st_), "a weight's tail");
            if (m.fp16) q8_elems_ = std::max(q8_elems_, n);
            else any_mmq_ = true;
        } else if (info->type == kF32 && proj_.kind != ProjectorKind::qwen3vl) {
            // the towers with separate q, k and v multiply FP16 values (mul16): an F32 file's matrices turned to them
            // on the device, through a buffer of the file's bytes
            m.type = kF16;
            m.d = dmalloc(bytes / 2);
            if (!measuring_) {
                struct Staging {   // (freed however the load ends)
                    void* p = nullptr;
                    ~Staging() { cudaFree(p); }
                } f32;
                ck(cudaMalloc(&f32.p, bytes), "an F32 matrix's staging");
                to_device(f32.p, raw.data(), bytes);
                k_to_f16<<<grid1(n), kThreads, 0, st_>>>((const float*) f32.p, (__half*) m.d, n);
                ck(cudaStreamSynchronize(st_), "an F32 matrix's conversion");
            }
        } else if (info->type == kF32) {
            m.type = kF32;
            m.d = dmalloc(bytes);
            to_device(m.d, raw.data(), bytes);
        } else {
            throw std::runtime_error("the tensor " + name + " has a type this encoder does not read (ggml type " +
                                     std::to_string(info->type) + ")");
        }
        return m;
    }
    Mat mat_f32(const std::vector<float>& v, int64_t n_out, int64_t n_in) {
        Mat m;
        m.type = kF32;
        m.n_out = n_out;
        m.n_in = n_in;
        m.d = upload(v);
        return m;
    }
    /// its place alone, while load() measures
    Mat mat_f32_place(int64_t n_out, int64_t n_in) {
        Mat m;
        m.type = kF32;
        m.n_out = n_out;
        m.n_in = n_in;
        m.d = dmalloc((size_t) n_out * (size_t) n_in * sizeof(float));
        return m;
    }

    struct Layer {
        float* ln1_w = nullptr;
        float* ln1_b = nullptr;
        float* ln2_w = nullptr;
        float* ln2_b = nullptr;
        Mat qkv;         // Qwen: one matrix [3 E][E]
        Mat q, k, v;     // Hy4: three
        float* qkv_b = nullptr;   // [3 E]: one bias, or the three one after the other
        Mat o, up, down;
        float* o_b = nullptr;
        float* up_b = nullptr;
        float* down_b = nullptr;
        float* ls1 = nullptr;   // Inkling 2: the attention's and the FFN's outputs times a learned vector
        float* ls2 = nullptr;
    };

    void load() {
        const int E = proj_.n_embd, H = proj_.n_head, D = proj_.d_head();
        const bool qwen = proj_.kind == ProjectorKind::qwen3vl, ink = proj_.kind == ProjectorKind::inkling2v;
        if (E != H * D || (ink ? D > kHeadPad || D % 4 != 0 || D / 4 > kRopeFreqs : D != 72))
            throw std::runtime_error("its heads are not of a width this encoder's attention kernel takes (72 values; "
                                     "Inkling 2's up to 80 in fours)");
        const int P = proj_.patch_size, in = 3 * P * P;
        // the patch kernel as a matrix [E][3 P P] (a patch's values: channel, row, column); Qwen's two summed
        {
            std::vector<float> w = proj_.floats("v.patch_embd.weight");
            if ((int64_t) w.size() != (int64_t) E * in)
                throw std::runtime_error("its patch kernel has another shape than the tower's");
            if (qwen) {
                const std::vector<float> w1 = proj_.floats("v.patch_embd.weight.1");
                if (w1.size() != w.size()) throw std::runtime_error("its second patch kernel has another shape than the first");
                for (size_t i = 0; i < w.size(); ++i) w[i] += w1[i];
            }
            patch_ = mat_f32(w, E, in);
            patch_b_ = vec("v.patch_embd.bias", E, qwen);
        }
        if (!measuring_) {   // the host's: resized for each picture
            pos_host_ = proj_.floats("v.position_embd.weight");
            n_side_ = (int) std::lround(std::sqrt((double) pos_host_.size() / E));
            if ((int64_t) n_side_ * n_side_ * E != (int64_t) pos_host_.size())
                throw std::runtime_error("its position table is not a square grid of the tower's width");
        }
        if (ink) {
            pre_w_ = vec("v.pre_ln.weight", E);
            pre_b_ = vec("v.pre_ln.bias", E);
            if (!measuring_) {   // the host's: the class token's row, its embedding and its position
                cls_ = proj_.floats("v.class_embd");
                const std::vector<float> at = proj_.floats("v.class_position_embd.weight");
                if ((int) cls_.size() != E || at.size() != cls_.size())
                    throw std::runtime_error("its class token is not the tower's width");
                for (size_t i = 0; i < cls_.size(); ++i) cls_[i] += at[i];
            }
        } else if (proj_.find("v.pre_ln.weight") != nullptr) {
            throw std::runtime_error("its tower has a norm before the blocks, which this encoder does not read");
        }
        post_w_ = vec("v.post_ln.weight", E, false);
        post_b_ = post_w_ != nullptr ? vec("v.post_ln.bias", E) : nullptr;
        n_ff_ = 0;
        layers_.resize((size_t) proj_.n_layer);
        for (int il = 0; il < proj_.n_layer; ++il) {
            Layer& l = layers_[(size_t) il];
            const std::string b = "v.blk." + std::to_string(il) + ".";
            l.ln1_w = vec(b + "ln1.weight", E);
            l.ln1_b = vec(b + "ln1.bias", E);
            l.ln2_w = vec(b + "ln2.weight", E);
            l.ln2_b = vec(b + "ln2.bias", E);
            if (qwen) {
                l.qkv = mat(b + "attn_qkv.weight", 3 * E, E);
                l.qkv_b = vec(b + "attn_qkv.bias", 3 * E);
            } else {
                l.q = mat(b + "attn_q.weight", E, E);
                l.k = mat(b + "attn_k.weight", E, E);
                l.v = mat(b + "attn_v.weight", E, E);
                std::vector<float> bias((size_t) 3 * E, 0.0f);
                int at = 0;
                for (const char* n : {"attn_q.bias", "attn_k.bias", "attn_v.bias"}) {
                    if (proj_.find(b + n) != nullptr) {
                        const std::vector<float> one = proj_.floats(b + n);
                        if ((int) one.size() != E)
                            throw std::runtime_error("the tensor " + b + n + " has another size than the tower's");
                        std::copy(one.begin(), one.end(), bias.begin() + (ptrdiff_t) at * E);
                    }
                    ++at;
                }
                l.qkv_b = upload(bias);
            }
            l.o = mat(b + "attn_out.weight", E, E);
            l.o_b = vec(b + "attn_out.bias", E);
            const TensorInfo* up = proj_.find(b + "ffn_up.weight");
            if (up == nullptr) throw std::runtime_error("the projector has no tensor " + b + "ffn_up.weight");
            if (n_ff_ == 0) n_ff_ = (int) up->ne[1];
            l.up = mat(b + "ffn_up.weight", n_ff_, E);
            l.up_b = vec(b + "ffn_up.bias", n_ff_);
            l.down = mat(b + "ffn_down.weight", E, n_ff_);
            l.down_b = vec(b + "ffn_down.bias", E);
            if (ink) {
                l.ls1 = vec(b + "ls1.weight", E);
                l.ls2 = vec(b + "ls2.weight", E);
            }
        }
        const int proj = proj_.projection_dim;
        if (ink) {   // two matrices over a group's four rows, the exact GELU between them, no biases
            mm0_ = mat("mm.1.weight", 4 * E, 4 * E);
            mm1_ = mat("mm.2.weight", proj, 4 * E);
            n_mid_ = 4 * E;
            mm0_b_ = mm1_b_ = upload(std::vector<float>((size_t) std::max(n_mid_, proj), 0.0f));
        } else if (qwen) {
            mm0_ = mat("mm.0.weight", 4 * E, 4 * E);
            mm0_b_ = vec("mm.0.bias", 4 * E);
            mm1_ = mat("mm.2.weight", proj, 4 * E);
            mm1_b_ = vec("mm.2.bias", proj);
            n_mid_ = 4 * E;
        } else {
            // the 2x2 convolution [kx, ky, c, o] as a matrix over a group's four rows one after the other:
            // W[o][(ky * 2 + kx) * E + c]
            const TensorInfo* c0 = proj_.find("mm.0.weight");
            if (c0 == nullptr || c0->ne[0] != 2 || c0->ne[1] != 2 || c0->ne[2] != E)
                throw std::runtime_error("its adapter's first convolution is not 2x2 over the tower's width");
            n_mid_ = (int) c0->ne[3];
            if (measuring_) {
                mm0_ = mat_f32_place(n_mid_, 4 * E);
            } else {
                const std::vector<float> w = proj_.floats("mm.0.weight");
                std::vector<float> m((size_t) n_mid_ * 4 * E);
                for (int o = 0; o < n_mid_; ++o)
                    for (int c = 0; c < E; ++c)
                        for (int k = 0; k < 4; ++k) m[((size_t) o * 4 + k) * E + c] = w[((size_t) o * E + c) * 4 + k];
                mm0_ = mat_f32(m, n_mid_, 4 * E);
            }
            mm0_b_ = vec("mm.0.bias", n_mid_);
            const TensorInfo* c1 = proj_.find("mm.2.weight");
            if (c1 == nullptr || c1->ne[0] != 1 || c1->ne[1] != 1 || c1->ne[2] != n_mid_ || c1->ne[3] != proj)
                throw std::runtime_error("its adapter's second convolution is not 1x1 onto its projection_dim");
            mm1_ = measuring_ ? mat_f32_place(proj, n_mid_) : mat_f32(proj_.floats("mm.2.weight"), proj, n_mid_);
            mm1_b_ = vec("mm.2.bias", proj);
            pre_norm_w_ = vec("mm.pre_norm.weight", E);
            if (!measuring_) {   // the host's: a row after each row of cells
                newline_ = proj_.floats("v.image_newline");
                if ((int) newline_.size() != proj) throw std::runtime_error("its newline row is not its projection_dim wide");
            }
        }
        if (qwen && (mm1_.n_out != proj || mm0_.n_out != 4 * E))
            throw std::runtime_error("its adapter's output is not its projection_dim wide");
        if (!measuring_) {
            ck(cudaStreamSynchronize(st_), "the weights' conversion");
            stage_free();
        }
    }

    /// The buffers for pictures of up to `n` patches.  The projector's limit gives the size they start with; a
    /// picture past it (the sizing's lower limit and its clamp to one merged patch a side can pass the upper limit)
    /// has them made again.  In lent memory they are carved after the weights, each time from its start.  `count`:
    /// nothing is made, their bytes are returned (256 apart, as lent memory holds them).
    uint64_t alloc(int64_t n, bool count) {
        if (!count) free_work();
        const int E = proj_.n_embd, H = proj_.n_head;
        const int64_t n_max = n, chunk = std::min<int64_t>(kChunk, n);
        const int64_t wide = std::max<int64_t>({3 * (int64_t) E, n_ff_, 4 * (int64_t) E, n_mid_, proj_.projection_dim});
        // lent memory holds the buffers of the projector's largest picture; one past it (the sizing's lower limit can
        // pass the upper one) has them from the device's free memory, as an encoder with memory of its own
        const bool in_loan = !count && lent_ && alloc(n, true) <= lent_work_bytes_;
        uint64_t total = 0;
        auto work = [&](size_t bytes) -> void* {
            const uint64_t at = total;
            total += up(std::max<size_t>(bytes, 16));
            if (count) return nullptr;
            if (in_loan) return lent_work_ + at;
            void* p = nullptr;
            if (cudaMalloc(&p, std::max<size_t>(bytes, 16)) != cudaSuccess) {
                (void) cudaGetLastError();
                free_work();   // none are left half made: the next picture asks again
                throw std::runtime_error("the encoder's buffers for a picture of " + std::to_string(n) +
                                         " patches do not fit the GPU's free memory");
            }
            work_.push_back(p);
            return p;
        };
        // the two streams' buffers: a row of the tower's width a patch, or a quarter of an adapter's row where that
        // is wider (the 27B's rows of 5,120): the adapters leave their rows in them, a row a group of four patches
        // (a patch's pixels where those are more: Inkling 2's 20 x 20 patches)
        const int64_t adapter = std::max<int64_t>(n_mid_, proj_.projection_dim);
        const int64_t pixels = (int64_t) 3 * proj_.patch_size * proj_.patch_size;
        const size_t stream = (size_t) n_max * (size_t) std::max<int64_t>({(int64_t) E, (adapter + 3) / 4, pixels}) * 4;
        void* const x = work(stream);
        void* const h = work(stream);
        void* const c32 = work((size_t) chunk * wide * 4);
        const size_t gsc_bytes = std::min<size_t>(lamina::kernels::gemm_scratch_bytes(chunk, wide), (size_t) 8 << 20);
        void* const gsc = work(gsc_bytes);
        void* const x16 = work((size_t) chunk * wide * 2);
        void* const q16 = work((size_t) n_max * H * kHeadPad * 2);
        void* const k16 = work((size_t) n_max * H * kHeadPad * 2);
        void* const v16 = work((size_t) n_max * H * kHeadPad * 2);
        void* const cs = work((size_t) n_max * kRopeFreqs * 2 * 4);
        void* const xq = any_mmq_ ? work(mmq::q8_bytes(chunk, wide)) : nullptr;
        void* const w16 = q8_elems_ > 0 ? work((size_t) q8_elems_ * 2) : nullptr;
        const int64_t ref_b = std::max<int64_t>(1, std::min<int64_t>(n_max, ((int64_t) 1 << 26) / std::max<int64_t>(1, n_max)));
        void* const ref_s = ref_attn_ ? work((size_t) ref_b * n_max * 4) : nullptr;
        void* const ref_v = ref_attn_ ? work((size_t) n_max * kHeadPad * 4) : nullptr;
        void* const ref_o = ref_attn_ ? work((size_t) ref_b * kHeadPad * 4) : nullptr;
        if (count) return total;
        if (any_mmq_ && !mctx_) mctx_ = std::make_unique<mmq::Context>();
        n_max_ = n_max;
        chunk_ = chunk;
        work_bytes_ = total;
        x_ = (float*) x;
        h_ = (float*) h;
        c32_ = (float*) c32;
        gsc_ = {gsc, gsc_bytes};
        x16_ = (__half*) x16;
        q16_ = (__half*) q16;
        k16_ = (__half*) k16;
        v16_ = (__half*) v16;
        cs_ = (float*) cs;
        xq_ = xq;
        w16_ = (__half*) w16;
        ref_b_ = ref_b;
        ref_s_ = (float*) ref_s;
        ref_v_ = (float*) ref_v;
        ref_o_ = (float*) ref_o;
        return total;
    }

    // ---- products --------------------------------------------------------------------------------------------------
    /// y[T][n_out] (rows `ld` apart) = x[T][n_in] (FP32) . w^T
    void mul(const Mat& w, const float* x, int64_t T, float* y, int64_t ld) {
        if (w.type == kF32) {
            lamina::kernels::Gemm32 p;
            p.x = x;
            p.w = (const float*) w.d;
            p.y = y;
            p.T = T;
            p.N = w.n_out;
            p.K = w.n_in;
            p.ldy = ld;
            lamina::kernels::gemm32(p, gsc_, st_);
        } else if (w.type == kF16 || w.fp16) {
            k_to_f16<<<grid1(T * w.n_in), kThreads, 0, st_>>>(x, x16_, T * w.n_in);
            mul16(w, x16_, T, y, ld);
        } else {
            mmq::quantize(x, nullptr, xq_, kQ8_0, w.n_in, w.n_in, T, st_);
            mctx_->run_dense(w.d, kQ8_0, w.n_out, w.n_in, xq_, T, y, ld, st_);
        }
    }
    /// the same from FP16 rows, for an FP16 matrix (several of them over one input)
    void mul16(const Mat& w, const __half* x16, int64_t T, float* y, int64_t ld) {
        const void* values = w.d;
        if (w.type == kQ8_0) {   // its values, 0.06-0.12 ms a matrix
            k_q8_to_f16<<<grid1(w.n_out * w.n_in), kThreads, 0, st_>>>((const uint8_t*) w.d, w16_, w.n_out * w.n_in);
            values = w16_;
        }
        lamina::kernels::Gemm16 p;
        p.x = (const uint16_t*) x16;
        p.w = (const uint16_t*) values;
        p.y = y;
        p.T = T;
        p.N = w.n_out;
        p.K = w.n_in;
        p.ldy = ld;
        lamina::kernels::gemm16(p, gsc_, st_);
    }

    // ---- the attention ---------------------------------------------------------------------------------------------
    void attention(int N) {
        const int H = proj_.n_head, D = proj_.d_head();
        const float scale = 1.0f / sqrtf((float) D);
        if (!ref_attn_) {
            va::k_vit_attn<<<dim3((unsigned) ((N + va::kRows - 1) / va::kRows), (unsigned) H), 256, sizeof(va::Smem), st_>>>(
                q16_, k16_, v16_, h_, N, H, D, scale);
            return;
        }
        // the reference: a head's scores S = Q K^T (FP16 in, FP32 out), a softmax over each row, O = P V in FP32
        const int64_t B = ref_b_;   // query rows at once
        for (int head = 0; head < H; ++head) {
            const __half* kh = k16_ + (size_t) head * N * kHeadPad;
            // V's transpose [80][N]: the value product's rows of weights
            k_f16_to_f32_t<<<grid1((int64_t) N * kHeadPad), kThreads, 0, st_>>>(v16_ + (size_t) head * N * kHeadPad, ref_v_, N,
                                                                               kHeadPad);
            for (int64_t t0 = 0; t0 < N; t0 += B) {
                const int64_t b = std::min<int64_t>(B, N - t0);
                lamina::kernels::Gemm16 sp;   // S [b][N] = Q (its rows H * 80 apart) K^T
                sp.x = (const uint16_t*) (q16_ + ((size_t) t0 * H + head) * kHeadPad);
                sp.ldx = (int64_t) H * kHeadPad;
                sp.w = (const uint16_t*) kh;
                sp.y = ref_s_;
                sp.T = b;
                sp.N = N;
                sp.K = kHeadPad;
                lamina::kernels::gemm16(sp, gsc_, st_);
                k_softmax_rows<<<(unsigned) b, 256, 0, st_>>>(ref_s_, N, scale);
                lamina::kernels::Gemm32 op;   // O [b][80] = P V
                op.x = ref_s_;
                op.w = ref_v_;
                op.y = ref_o_;
                op.T = b;
                op.N = kHeadPad;
                op.K = N;
                lamina::kernels::gemm32(op, gsc_, st_);
                k_head_out<<<grid1(b * D), kThreads, 0, st_>>>(ref_o_, h_, (int) t0, (int) b, H, D, head);
            }
        }
    }

    // ---- a picture -------------------------------------------------------------------------------------------------
    void encode_or_throw(const Rgb& picture, Rows& out) {
        if (picture.w <= 0 || picture.h <= 0 || picture.px.size() != (size_t) picture.w * (size_t) picture.h * 3)
            throw std::runtime_error("an empty picture");
        int tw = 0, th = 0;
        const std::vector<float> pixels = projector_planes(picture, proj_, tw, th);

        const bool qwen = proj_.kind == ProjectorKind::qwen3vl, ink = proj_.kind == ProjectorKind::inkling2v;
        const int E = proj_.n_embd, H = proj_.n_head, D = proj_.d_head(), P = proj_.patch_size;
        // (Inkling 2: the class token's row before the patches')
        const int cls = ink ? 1 : 0, pw = tw / P, ph = th / P, n_patch = pw * ph, N = n_patch + cls, in = 3 * P * P;
        if (N > n_max_) alloc(N, false);

        // the patches as rows (channel, row, column), and the position table, in the tokens' order: Qwen's the 2x2
        // groups' (a group's four patches one after the other), Hy4's the grid's
        auto patch_of = [&](int p, int& py, int& px) {
            if (qwen) {
                const int cell = p >> 2, sub = p & 3;
                py = 2 * (cell / (pw / 2)) + (sub >> 1);
                px = 2 * (cell % (pw / 2)) + (sub & 1);
            } else {
                py = p / pw;
                px = p % pw;
            }
        };
        host_.resize((size_t) N * std::max(in, E));
        const size_t plane = (size_t) tw * th;
        for (int p = 0; p < n_patch; ++p) {
            int py, px;
            patch_of(p, py, px);
            float* row = &host_[(size_t) p * in];
            for (int c = 0; c < 3; ++c)
                for (int y = 0; y < P; ++y) {
                    const float* src = &pixels[c * plane + (size_t) (py * P + y) * tw + (size_t) px * P];
                    std::copy(src, src + P, row + (c * P + y) * P);
                }
        }
        to_device(h_, host_.data(), (size_t) n_patch * in * 4);
        if (ink) ck(cudaMemsetAsync(x_, 0, (size_t) E * 4, st_), "the class token's row");
        for (int64_t t0 = 0; t0 < n_patch; t0 += chunk_) {
            const int64_t T = std::min<int64_t>(chunk_, n_patch - t0);
            mul(patch_, h_ + t0 * in, T, x_ + (t0 + cls) * E, E);
        }
        {
            const std::vector<float> table = qwen  ? qwen_position_table(pos_host_, E, n_side_, pw, ph)
                                             : ink ? qwen_position_table(pos_host_, E, n_side_, pw, ph, false)
                                                   : hy4_position_table(pos_host_, E, n_side_, pw, ph);
            if (ink) std::copy(cls_.begin(), cls_.end(), host_.begin());
            for (int p = 0; p < n_patch; ++p) {
                int py, px;
                patch_of(p, py, px);
                std::copy(&table[((size_t) py * pw + px) * E], &table[((size_t) py * pw + px) * E] + E,
                          &host_[(size_t) (p + cls) * E]);
            }
        }
        to_device(h_, host_.data(), (size_t) N * E * 4);   // after the patch products, which read h_
        // x = patches + bias + positions (Inkling 2's first row: the class token)
        k_add<<<grid1((int64_t) N * E), kThreads, 0, st_>>>(x_, h_, patch_b_, N, E);
        dump("embed", x_, (size_t) N * E);
        if (ink) k_norm<<<(unsigned) N, 128, 0, st_>>>(x_, pre_w_, pre_b_, x_, E, proj_.eps, false);
        const float* cs = nullptr;
        if (ink) {
            // ggml's rope cache over a half of a head: position p's angles p * s^i, s = base^(-2 / (d_head / 2)), the
            // recurrence in float as ggml runs it; the positions 0 (the class token) to the grid's longer side
            const int n_pos = std::max(pw, ph) + 1, n_freq = D / 4;
            const float theta_scale = powf(proj_.rope_theta, -2.0f / (float) (D / 2));
            std::vector<float> table((size_t) n_pos * kRopeFreqs * 2);
            for (int p = 0; p < n_pos; ++p) {
                float theta = (float) p;
                for (int i = 0; i < n_freq; ++i) {
                    table[((size_t) p * kRopeFreqs + i) * 2] = cosf(theta);
                    table[((size_t) p * kRopeFreqs + i) * 2 + 1] = sinf(theta);
                    theta *= theta_scale;
                }
            }
            to_device(cs_, table.data(), table.size() * 4);
            cs = cs_;
        } else if (qwen) {
            // ggml's rope cache for the vision type: position p's angles p * s^i, s = base^(-2 / (d_head / 2)), the
            // recurrence in float as ggml runs it
            const int n_pos = std::max(pw, ph);
            const float theta_scale = powf(10000.0f, -2.0f / (float) (D / 2));
            std::vector<float> table((size_t) n_pos * kRopeFreqs * 2);
            for (int p = 0; p < n_pos; ++p) {
                float theta = (float) p;
                for (int i = 0; i < kRopeFreqs; ++i) {
                    table[((size_t) p * kRopeFreqs + i) * 2] = cosf(theta);
                    table[((size_t) p * kRopeFreqs + i) * 2 + 1] = sinf(theta);
                    theta *= theta_scale;
                }
            }
            to_device(cs_, table.data(), table.size() * 4);
            cs = cs_;
        }

        int il = -1;
        for (const Layer& l : layers_) {
            ++il;
            // attention
            k_norm<<<(unsigned) N, 128, 0, st_>>>(x_, l.ln1_w, l.ln1_b, h_, E, proj_.eps, false);
            if (il == 0) dump("ln0", h_, (size_t) N * E);
            for (int64_t t0 = 0; t0 < N; t0 += chunk_) {
                const int64_t T = std::min<int64_t>(chunk_, N - t0);
                if (qwen) {
                    mul(l.qkv, h_ + t0 * E, T, c32_, 3 * E);
                } else {
                    k_to_f16<<<grid1(T * E), kThreads, 0, st_>>>(h_ + t0 * E, x16_, T * E);
                    mul16(l.q, x16_, T, c32_, 3 * E);
                    mul16(l.k, x16_, T, c32_ + E, 3 * E);
                    mul16(l.v, x16_, T, c32_ + 2 * E, 3 * E);
                }
                k_qkv<<<(unsigned) T, 256, 0, st_>>>(c32_, l.qkv_b, q16_, k16_, v16_, cs, (int) t0, N, H, D, pw, cls);
            }
            attention(N);   // into h_
            if (il == 0) dump("fa0", h_, (size_t) N * E);
            for (int64_t t0 = 0; t0 < N; t0 += chunk_) {
                const int64_t T = std::min<int64_t>(chunk_, N - t0);
                mul(l.o, h_ + t0 * E, T, c32_, E);
                if (il == 0 && t0 == 0) {   // the output projection with its bias, as the ggml encoder's tap has it
                    k_bias_act<<<grid1(T * E), kThreads, 0, st_>>>(c32_, l.o_b, T, E, E, 0);
                    dump("attn0", c32_, (size_t) T * E);
                    if (ink) k_add_scaled<<<grid1(T * E), kThreads, 0, st_>>>(x_ + t0 * E, c32_, nullptr, l.ls1, T, E);
                    else k_add<<<grid1(T * E), kThreads, 0, st_>>>(x_ + t0 * E, c32_, nullptr, T, E);
                    continue;
                }
                if (ink) k_add_scaled<<<grid1(T * E), kThreads, 0, st_>>>(x_ + t0 * E, c32_, l.o_b, l.ls1, T, E);
                else k_add<<<grid1(T * E), kThreads, 0, st_>>>(x_ + t0 * E, c32_, l.o_b, T, E);
            }
            // feed-forward
            k_norm<<<(unsigned) N, 128, 0, st_>>>(x_, l.ln2_w, l.ln2_b, h_, E, proj_.eps, false);
            for (int64_t t0 = 0; t0 < N; t0 += chunk_) {
                const int64_t T = std::min<int64_t>(chunk_, N - t0);
                mul(l.up, h_ + t0 * E, T, c32_, n_ff_);
                k_bias_act<<<grid1(T * n_ff_), kThreads, 0, st_>>>(c32_, l.up_b, T, n_ff_, n_ff_, ink ? 2 : 1);
                mul_into_c32(l.down, T, E);
                if (ink) k_add_scaled<<<grid1(T * E), kThreads, 0, st_>>>(x_ + t0 * E, c32_, l.down_b, l.ls2, T, E);
                else k_add<<<grid1(T * E), kThreads, 0, st_>>>(x_ + t0 * E, c32_, l.down_b, T, E);
            }
            if (il < 2 || il == proj_.n_layer - 1) dump("layer" + std::to_string(il), x_, (size_t) N * E);
        }

        const int proj = proj_.projection_dim;
        int n_rows = 0;
        if (ink) {
            // the patches' rows (not the class token's) into h_, then the projector between the two buffers: each
            // group's four rows side by side through its first matrix into x_, the exact GELU, its second into h_
            const int ow = pw / 2, oh = ph / 2, cells = ow * oh;
            if (post_w_ != nullptr) k_norm<<<(unsigned) N, 128, 0, st_>>>(x_, post_w_, post_b_, x_, E, proj_.eps, false);
            ck(cudaMemcpyAsync(h_, x_ + E, (size_t) n_patch * E * 4, cudaMemcpyDeviceToDevice, st_), "the patches' rows");
            for (int64_t t0 = 0; t0 < cells; t0 += chunk_) {
                const int64_t T = std::min<int64_t>(chunk_, cells - t0);
                k_gather_2x2<<<grid1(T * 4 * E), kThreads, 0, st_>>>(h_, c32_, pw, ow, (int) t0, (int) T, E);
                mul(mm0_, c32_, T, x_ + t0 * n_mid_, n_mid_);
            }
            k_bias_act<<<grid1((int64_t) cells * n_mid_), kThreads, 0, st_>>>(x_, mm0_b_, cells, n_mid_, n_mid_, 2);
            for (int64_t t0 = 0; t0 < cells; t0 += chunk_) {
                const int64_t T = std::min<int64_t>(chunk_, cells - t0);
                mul(mm1_, x_ + t0 * n_mid_, T, h_ + t0 * proj, proj);
            }
            n_rows = cells;
            out.n = n_rows;
            out.nx = n_rows;   // read at text positions: one row of n
            out.ny = 1;
            out.n_embd = proj;
            out.data.resize((size_t) n_rows * proj);
            ck(cudaMemcpyAsync(out.data.data(), h_, out.data.size() * 4, cudaMemcpyDeviceToHost, st_), "the rows' copy");
        } else if (qwen) {
            if (post_w_ != nullptr) k_norm<<<(unsigned) N, 128, 0, st_>>>(x_, post_w_, post_b_, x_, E, proj_.eps, false);
            // the rows into h_, free since the last block (alloc sizes it for them)
            n_rows = N / 4;
            for (int64_t t0 = 0; t0 < n_rows; t0 += chunk_) {
                const int64_t T = std::min<int64_t>(chunk_, n_rows - t0);
                mul(mm0_, x_ + t0 * 4 * E, T, c32_, n_mid_);
                k_bias_act<<<grid1(T * n_mid_), kThreads, 0, st_>>>(c32_, mm0_b_, T, n_mid_, n_mid_, 1);
                mul(mm1_, c32_, T, h_ + t0 * proj, proj);
            }
            k_bias_act<<<grid1((int64_t) n_rows * proj), kThreads, 0, st_>>>(h_, mm1_b_, n_rows, proj, proj, 0);
            out.n = n_rows;
            out.nx = pw / 2;
            out.ny = ph / 2;
            out.n_embd = proj;
            out.data.resize((size_t) n_rows * proj);
            ck(cudaMemcpyAsync(out.data.data(), h_, out.data.size() * 4, cudaMemcpyDeviceToHost, st_), "the rows' copy");
        } else {
            // the adapter between the two free buffers: the cells' rows [cells][n_mid] in x_, [cells][proj] in h_
            // (alloc sizes them for these)
            const int ow = pw / 2, oh = ph / 2, cells = ow * oh;
            k_norm<<<(unsigned) N, 128, 0, st_>>>(x_, pre_norm_w_, nullptr, h_, E, proj_.eps, true);
            for (int64_t t0 = 0; t0 < cells; t0 += chunk_) {
                const int64_t T = std::min<int64_t>(chunk_, cells - t0);
                k_gather_2x2<<<grid1(T * 4 * E), kThreads, 0, st_>>>(h_, c32_, pw, ow, (int) t0, (int) T, E);
                mul(mm0_, c32_, T, x_ + t0 * n_mid_, n_mid_);
            }
            k_bias_act<<<grid1((int64_t) cells * n_mid_), kThreads, 0, st_>>>(x_, mm0_b_, cells, n_mid_, n_mid_, 2);
            for (int64_t t0 = 0; t0 < cells; t0 += chunk_) {
                const int64_t T = std::min<int64_t>(chunk_, cells - t0);
                mul(mm1_, x_ + t0 * n_mid_, T, h_ + t0 * proj, proj);
            }
            k_bias_act<<<grid1((int64_t) cells * proj), kThreads, 0, st_>>>(h_, mm1_b_, cells, proj, proj, 0);
            // the cells' rows, and the newline row after each row of cells
            n_rows = (ow + 1) * oh;
            out.n = n_rows;
            out.nx = n_rows;   // read at text positions: one row of n
            out.ny = 1;
            out.n_embd = proj;
            out.data.resize((size_t) n_rows * proj);
            for (int y = 0; y < oh; ++y) {
                float* row = out.data.data() + (size_t) y * (ow + 1) * proj;
                ck(cudaMemcpyAsync(row, h_ + (size_t) y * ow * proj, (size_t) ow * proj * 4, cudaMemcpyDeviceToHost, st_),
                   "the rows' copy");
                std::copy(newline_.begin(), newline_.end(), row + (size_t) ow * proj);
            }
        }
        ck(cudaStreamSynchronize(st_), "the encoder's kernels");
    }

    /// LAMINA_VISION_DUMP=<folder>: a buffer's floats as <folder>/<name>.f32 (the ggml encoder writes the same stages)
    void dump(const std::string& name, const float* d, size_t n) {
        const char* dir = std::getenv("LAMINA_VISION_DUMP");
        if (dir == nullptr) return;
        std::vector<float> v(n);
        ck(cudaStreamSynchronize(st_), "the encoder's kernels");
        ck(cudaMemcpy(v.data(), d, n * 4, cudaMemcpyDeviceToHost), "a dump's copy");
        if (std::FILE* f = std::fopen((std::string(dir) + "/" + name + ".f32").c_str(), "wb")) {
            std::fwrite(v.data(), 4, n, f);
            std::fclose(f);
        }
    }

    /// c32_ [T][w.n_in] (FP32) through `w` back into c32_ [T][w.n_out]: the input leaves c32_ as FP16 (or q8_1) first
    void mul_into_c32(const Mat& w, int64_t T, int ld) {
        if (w.type == kF16 || w.fp16) {
            k_to_f16<<<grid1(T * w.n_in), kThreads, 0, st_>>>(c32_, x16_, T * w.n_in);
            mul16(w, x16_, T, c32_, ld);
        } else if (w.type == kQ8_0) {
            mmq::quantize(c32_, nullptr, xq_, kQ8_0, w.n_in, w.n_in, T, st_);
            mctx_->run_dense(w.d, kQ8_0, w.n_out, w.n_in, xq_, T, c32_, ld, st_);
        } else {
            throw std::runtime_error("an FP32 matrix where the encoder's buffers take FP16 or Q8_0");
        }
    }

    Projector proj_;
    int dev_ = -1;
    cudaStream_t st_ = nullptr;
    void* arena_ = nullptr;       // the weights' block
    size_t arena_bytes_ = 0, arena_used_ = 0;
    void* stage_ = nullptr;       // load(): the matrices' pinned staging buffer
    size_t stage_need_ = 0;       // the largest matrix's bytes
    std::vector<uint8_t> stage_pageable_;
    bool lent_ = false;           // the caller's memory (EncoderParams::gpu_arena): the weights, then the buffers
    uint8_t* lent_work_ = nullptr;
    uint64_t lent_work_bytes_ = 0;
    bool measuring_ = false;      // load() counts the block's bytes
    uint8_t measure_dummy_ = 0;   // a weight's place while it does
    std::vector<void*> work_;     // the buffers (alloc)
    size_t work_bytes_ = 0;
    bool ref_attn_ = false, q8_fp16_ = false;
    bool any_mmq_ = false;        // a Q8_0 matrix goes through MMQ
    int64_t q8_elems_ = 0;        // the largest Q8_0 matrix multiplied as FP16 values: its values (0: none)
    // the weights
    Mat patch_, mm0_, mm1_;
    float* patch_b_ = nullptr;
    float* post_w_ = nullptr;
    float* post_b_ = nullptr;
    float* pre_w_ = nullptr;    // Inkling 2: the norm before the blocks
    float* pre_b_ = nullptr;
    std::vector<float> cls_;    // its class token's row (the host's)
    float* mm0_b_ = nullptr;
    float* mm1_b_ = nullptr;
    float* pre_norm_w_ = nullptr;
    std::vector<float> newline_;   // Hy4's newline row (the host's)
    std::vector<Layer> layers_;
    std::vector<float> pos_host_;
    int n_side_ = 0, n_ff_ = 0, n_mid_ = 0;
    // the buffers
    int64_t n_max_ = 0, chunk_ = 0, ref_b_ = 0;
    float* x_ = nullptr;      // [n_max][E] the residual stream
    float* h_ = nullptr;      // [n_max][E] a norm's output, the attention's
    float* c32_ = nullptr;    // [chunk][wide] a product's output
    lamina::kernels::GemmScratch gsc_;   // the products' scratch (a K split's partial sums)
    __half* x16_ = nullptr;   // [chunk][wide] a product's input
    __half* q16_ = nullptr;
    __half* k16_ = nullptr;
    __half* v16_ = nullptr;
    float* cs_ = nullptr;
    void* xq_ = nullptr;      // a product's input as q8_1 (MMQ)
    __half* w16_ = nullptr;   // a Q8_0 matrix's values (FP16 products)
    std::unique_ptr<mmq::Context> mctx_;
    float* ref_s_ = nullptr;
    float* ref_v_ = nullptr;
    float* ref_o_ = nullptr;
    std::vector<float> host_;
};

}  // namespace

uint64_t gpu_free_bytes(int device) {
    size_t fr = 0, total = 0;
    if (cudaSetDevice(device) != cudaSuccess || cudaMemGetInfo(&fr, &total) != cudaSuccess) {
        (void) cudaGetLastError();
        return 0;
    }
    return fr;
}

uint64_t cuda_encoder_bytes(const EncoderParams& params, std::string& err) {
    try {
        return CudaEncoder().measure(params);
    } catch (const std::exception& e) {
        err = e.what();
        return 0;
    }
}

std::unique_ptr<Encoder> open_cuda_encoder(const EncoderParams& params, std::string& err) {
    auto enc = std::make_unique<CudaEncoder>();
    try {
        enc->open(params);
    } catch (const std::exception& e) {
        err = e.what();
        return nullptr;
    }
    return enc;
}

}  // namespace lamina::vision
