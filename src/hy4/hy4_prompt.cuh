// src/hy4/hy4_prompt.cuh - Hy4-flash's prompt path kernels, beside hy4_kernels.cuh's (whose per-row kernels a chunk
// runs too): the latent attention and the indexer's scores on the tensor cores, the groups a chunk completes pooled,
// the window layers' cells in position order, and the routed experts' pieces (q8_1 rows gathered by entry, the
// clamped SwiGLU of a [rows][gate | up] product, the routed rows summed per token).  Included by hy4_engine.cu.
//
// Tensor-core tiles: mma.m16n8k16 (FP16 in, FP32 sums), operands through ldmatrix from shared memory whose rows are
// padded so that ldmatrix's eight rows fall on distinct banks.
#pragma once

#include "hy4_kernels.cuh"
#include "lamina/kernels/warp.cuh"

#include <cuda_fp16.h>

#include <cstdint>

namespace hy4p {

using hy4k::kKvRow;
using hy4k::kLatent;
using lamina::kernels::h2;
using lamina::kernels::ldsm4;
using lamina::kernels::ldsm4t;
using lamina::kernels::mma16816;


// ------------------------------------------------------------------------------------------------ the attention
constexpr int kQRows = 32;      // query rows of a block: one row's head group
constexpr int kTileCells = 32;  // cells a tile
constexpr int kPad = 584;       // a query or cell row in shared memory, halves (1,168 bytes)
constexpr int kPPad = 40;       // a probability row, halves (80 bytes)

struct __align__(16) AttnSmem {
    __half q[kQRows * kPad];
    __half k[kTileCells * kPad];
    __half p[kQRows * kPPad];
    float red[2][4][kQRows];    // [max | sum][the cells' quarter][row]
};

/// A chunk's latent attention: block t takes row t's head group of kQRows heads - its absorbed queries qa [T][kQRows]
/// [512] and its roped q_pe at qg [T][kQRows][256] + 192 - over its cells in tiles of 32, GATHER: cells[t][0 .. n[t])
/// of the cache (rows of 576 halves, or with Q8 q8_0 rows widened into halves as they are stashed); else the window,
/// positions max(0, p - n_swa + 1) .. p (p = pos[t]) at kv + (position - lin0) rows of halves.  Scores times `scale`,
/// an online softmax that starts from the head's sink (m = sink, l = 1), the values the cells' first 512; the output,
/// divided by l, replaces the row's queries in qa.  Warps: scores m-tile w & 1 x cells (w >> 1) * 8 .. +8; values
/// m-tile w & 1 x dims (w >> 1) * 128 .. +128.  The next tile waits in registers through the current one's products.
/// Block 256, shared sizeof(AttnSmem).
template <bool GATHER, bool Q8 = false>
__global__ void __launch_bounds__(256, 1) k_attn_prompt(float* qa, const float* __restrict__ qg,
                                                         const void* __restrict__ kv_, const int32_t* __restrict__ cells,
                                                         const int32_t* __restrict__ n, int ld_cells,
                                                         const int32_t* __restrict__ pos, int n_swa, int lin0,
                                                         const float* __restrict__ sinks, float scale) {
    static_assert(GATHER || !Q8, "the window's linear buffer is halves");
    extern __shared__ __align__(16) unsigned char attn_raw[];
    AttnSmem& sm = *reinterpret_cast<AttnSmem*>(attn_raw);
    const __half* kv = reinterpret_cast<const __half*>(kv_);
    const uint8_t* kq = reinterpret_cast<const uint8_t*>(kv_);
    const int t = blockIdx.x, tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
    const int li = lane >> 3, lr = lane & 7, tig = lane & 3;
    int cnt = 0, first = 0;
    const int32_t* cl = nullptr;
    if constexpr (GATHER) {
        cnt = n[t];
        cl = cells + (size_t) t * ld_cells;
    } else {
        const int p = pos[t], a = max(0, p - n_swa + 1);
        cnt = p - a + 1;
        first = a - lin0;
    }
    // the queries as halves: [512 absorbed | 64 roped]
    const float* qt = qa + (size_t) t * kQRows * kLatent;
    for (int i = tid; i < kQRows * (kLatent / 4); i += 256) {
        const int r = i / (kLatent / 4), c4 = i % (kLatent / 4);
        const float4 v = *reinterpret_cast<const float4*>(qt + (size_t) r * kLatent + 4 * c4);
        *reinterpret_cast<uint2*>(sm.q + r * kPad + 4 * c4) = make_uint2(h2(v.x, v.y), h2(v.z, v.w));
    }
    const float* qp = qg + (size_t) t * kQRows * 256 + 192;
    for (int i = tid; i < kQRows * 16; i += 256) {
        const int r = i / 16, c4 = i % 16;
        const float4 v = *reinterpret_cast<const float4*>(qp + (size_t) r * 256 + 4 * c4);
        *reinterpret_cast<uint2*>(sm.q + r * kPad + kLatent + 4 * c4) = make_uint2(h2(v.x, v.y), h2(v.z, v.w));
    }
    // a tile: 32 cells of 72 16-byte chunks, 9 a thread; Q8: 32 cells of 36 16-value pieces (half a block each, with
    // its block's scale), 5 a thread, each widened into two chunks of halves
    constexpr int kQ8Pieces = kKvRow / 16, kQ8PerThread = (kTileCells * kQ8Pieces + 255) / 256;
    uint4 nx[Q8 ? kQ8PerThread : 9];
    [[maybe_unused]] __half2 nd[Q8 ? kQ8PerThread : 1];
    auto fetch = [&](int base) {
        if constexpr (Q8) {
#pragma unroll
            for (int j = 0; j < kQ8PerThread; ++j) {
                const int i = tid + 256 * j, c = i / kQ8Pieces, pc = i % kQ8Pieces;
                if (i < kTileCells * kQ8Pieces && base + c < cnt) {
                    const uint8_t* row = kq + (size_t) cl[base + c] * hy4k::kKvQ8Row;
                    nx[j] = *reinterpret_cast<const uint4*>(row + pc * 16);
                    nd[j] = __half2half2(reinterpret_cast<const __half*>(row + kKvRow)[pc >> 1]);
                } else {
                    nx[j] = make_uint4(0, 0, 0, 0);
                    nd[j] = __float2half2_rn(0.f);
                }
            }
        } else {
#pragma unroll
            for (int j = 0; j < 9; ++j) {
                const int i = tid + 256 * j, c = i / 72, ch = i % 72;
                if (base + c < cnt) {
                    const __half* row = GATHER ? kv + (size_t) cl[base + c] * kKvRow : kv + (size_t) (first + base + c) * kKvRow;
                    nx[j] = *reinterpret_cast<const uint4*>(row + ch * 8);
                } else {
                    nx[j] = make_uint4(0, 0, 0, 0);
                }
            }
        }
    };
    auto stash = [&] {
        if constexpr (Q8) {
#pragma unroll
            for (int j = 0; j < kQ8PerThread; ++j) {
                const int i = tid + 256 * j, c = i / kQ8Pieces, pc = i % kQ8Pieces;
                if (i >= kTileCells * kQ8Pieces) continue;
                const uint32_t qw[4] = {nx[j].x, nx[j].y, nx[j].z, nx[j].w};
                __half2 hv[8];
#pragma unroll
                for (int k = 0; k < 4; ++k) hy4k::q8_widen4(qw[k], nd[j], hv[2 * k], hv[2 * k + 1]);
                unsigned hw[8];
#pragma unroll
                for (int k = 0; k < 8; ++k) hw[k] = *reinterpret_cast<const unsigned*>(&hv[k]);
                uint4* dst = reinterpret_cast<uint4*>(sm.k + c * kPad + pc * 16);
                dst[0] = make_uint4(hw[0], hw[1], hw[2], hw[3]);
                dst[1] = make_uint4(hw[4], hw[5], hw[6], hw[7]);
            }
        } else {
#pragma unroll
            for (int j = 0; j < 9; ++j) {
                const int i = tid + 256 * j, c = i / 72, ch = i % 72;
                *reinterpret_cast<uint4*>(sm.k + c * kPad + ch * 8) = nx[j];
            }
        }
    };
    fetch(0);
    stash();
    const int mt = w & 1, nt = w >> 1;
    const int rA = mt * 16 + (lane >> 2), rB = rA + 8;
    float mA = sinks[rA], mB = sinks[rB], lA = 1.f, lB = 1.f;
    float o[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    const __half* qrow = sm.q + (mt * 16 + ((li & 1) << 3) + lr) * kPad + (li >> 1) * 8;
    const int n_tiles = (cnt + kTileCells - 1) / kTileCells;
    for (int it = 0; it < n_tiles; ++it) {
        const int base = it * kTileCells;
        __syncthreads();   // the tile (and the queries) in shared memory
        if (it + 1 < n_tiles) fetch(base + kTileCells);
        float s[4] = {0.f, 0.f, 0.f, 0.f};
        const __half* krow = sm.k + (nt * 8 + lr) * kPad + li * 8;
#pragma unroll 3
        for (int kk = 0; kk < kKvRow / 16; kk += 2) {
            unsigned a0[4], a1[4], b[4];
            ldsm4(a0, qrow + kk * 16);
            ldsm4(a1, qrow + kk * 16 + 16);
            ldsm4(b, krow + kk * 16);
            mma16816(s, a0, b[0], b[1]);
            mma16816(s, a1, b[2], b[3]);
        }
        float xA = -INFINITY, xB = -INFINITY;
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const bool in = base + nt * 8 + 2 * tig + e < cnt;
            s[e] = in ? s[e] * scale : -INFINITY;
            s[2 + e] = in ? s[2 + e] * scale : -INFINITY;
            xA = fmaxf(xA, s[e]);
            xB = fmaxf(xB, s[2 + e]);
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
        const float aA = expf(mA - nA), aB = expf(mB - nB);
        const float p0A = s[0] == -INFINITY ? 0.f : expf(s[0] - nA), p1A = s[1] == -INFINITY ? 0.f : expf(s[1] - nA);
        const float p0B = s[2] == -INFINITY ? 0.f : expf(s[2] - nB), p1B = s[3] == -INFINITY ? 0.f : expf(s[3] - nB);
        const int pc = nt * 8 + 2 * tig;
        *reinterpret_cast<unsigned*>(sm.p + rA * kPPad + pc) = h2(p0A, p1A);
        *reinterpret_cast<unsigned*>(sm.p + rB * kPPad + pc) = h2(p0B, p1B);
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
        for (int j = 0; j < 16; ++j) {
            o[j][0] *= aA;
            o[j][1] *= aA;
            o[j][2] *= aB;
            o[j][3] *= aB;
        }
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            unsigned a[4];
            ldsm4(a, sm.p + (mt * 16 + ((li & 1) << 3) + lr) * kPPad + (2 * k + (li >> 1)) * 8);
            const __half* vrow = sm.k + (16 * k + ((li & 1) << 3) + lr) * kPad + nt * 128 + (li >> 1) * 8;
#pragma unroll
            for (int n2 = 0; n2 < 8; ++n2) {
                unsigned b[4];
                ldsm4t(b, vrow + n2 * 16);
                mma16816(o[2 * n2], a, b[0], b[1]);
                mma16816(o[2 * n2 + 1], a, b[2], b[3]);
            }
        }
        __syncthreads();   // the tile's cells and probabilities are read
        if (it + 1 < n_tiles) stash();
    }
    const float iA = 1.f / lA, iB = 1.f / lB;
    float* out = qa + (size_t) t * kQRows * kLatent;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int d = nt * 128 + j * 8 + 2 * tig;
        *reinterpret_cast<float2*>(out + (size_t) rA * kLatent + d) = make_float2(o[j][0] * iA, o[j][1] * iA);
        *reinterpret_cast<float2*>(out + (size_t) rB * kLatent + d) = make_float2(o[j][2] * iB, o[j][3] * iB);
    }
}

/// The decode's latent attention in splits (flash decoding): block (s, g, t) takes row t's head group g of kQRows heads
/// (queries q [T][H][576]: absorbed 512 | roped 64) over cells [s * per, min((s + 1) * per, n[t])) of the row's list
/// cells[t][..] (halves, or Q8 q8_0 rows widened as they are stashed), k_attn_prompt's tiles with the online softmax from
/// nothing (the sink is k_mla_merge's): each head's m, l and unnormalized acc[512] into part[t][s][h] (514 floats,
/// k_mla_chunk's layout; m = -inf for a split without cells).  Block 256, shared sizeof(AttnSmem); `per` a multiple of
/// kTileCells.
template <bool Q8>
__global__ void __launch_bounds__(256, 1) k_attn_dec(const float* __restrict__ q, const void* __restrict__ kv_,
                                                      const int32_t* __restrict__ cells, const int32_t* __restrict__ n,
                                                      int ld_cells, int H, int per, float scale, float* __restrict__ part,
                                                      int n_split) {
    extern __shared__ __align__(16) unsigned char attn_raw[];
    AttnSmem& sm = *reinterpret_cast<AttnSmem*>(attn_raw);
    const __half* kv = reinterpret_cast<const __half*>(kv_);
    const uint8_t* kq = reinterpret_cast<const uint8_t*>(kv_);
    const int sp = blockIdx.x, hg = blockIdx.y, t = blockIdx.z;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
    const int li = lane >> 3, lr = lane & 7, tig = lane & 3;
    const int c0 = sp * per, cnt = min(n[t] - c0, per);
    float* pt = part + (((size_t) t * n_split + sp) * H + (size_t) hg * kQRows) * (kLatent + 2);
    if (cnt <= 0) {
        if (tid < kQRows) {
            pt[(size_t) tid * (kLatent + 2)] = -INFINITY;
            pt[(size_t) tid * (kLatent + 2) + 1] = 0.f;
        }
        return;
    }
    const int32_t* cl = cells + (size_t) t * ld_cells + c0;
    const float* qt = q + ((size_t) t * H + (size_t) hg * kQRows) * kKvRow;
    for (int i = tid; i < kQRows * (kKvRow / 4); i += 256) {
        const int r = i / (kKvRow / 4), c4 = i % (kKvRow / 4);
        const float4 v = *reinterpret_cast<const float4*>(qt + (size_t) r * kKvRow + 4 * c4);
        *reinterpret_cast<uint2*>(sm.q + r * kPad + 4 * c4) = make_uint2(h2(v.x, v.y), h2(v.z, v.w));
    }
    constexpr int kQ8Pieces = kKvRow / 16, kQ8PerThread = (kTileCells * kQ8Pieces + 255) / 256;
    uint4 nx[Q8 ? kQ8PerThread : 9];
    [[maybe_unused]] __half2 nd[Q8 ? kQ8PerThread : 1];
    auto fetch = [&](int base) {
        if constexpr (Q8) {
#pragma unroll
            for (int j = 0; j < kQ8PerThread; ++j) {
                const int i = tid + 256 * j, c = i / kQ8Pieces, pc = i % kQ8Pieces;
                if (i < kTileCells * kQ8Pieces && base + c < cnt) {
                    const uint8_t* row = kq + (size_t) cl[base + c] * hy4k::kKvQ8Row;
                    nx[j] = *reinterpret_cast<const uint4*>(row + pc * 16);
                    nd[j] = __half2half2(reinterpret_cast<const __half*>(row + kKvRow)[pc >> 1]);
                } else {
                    nx[j] = make_uint4(0, 0, 0, 0);
                    nd[j] = __float2half2_rn(0.f);
                }
            }
        } else {
#pragma unroll
            for (int j = 0; j < 9; ++j) {
                const int i = tid + 256 * j, c = i / 72, ch = i % 72;
                if (base + c < cnt) nx[j] = *reinterpret_cast<const uint4*>(kv + (size_t) cl[base + c] * kKvRow + ch * 8);
                else nx[j] = make_uint4(0, 0, 0, 0);
            }
        }
    };
    auto stash = [&] {
        if constexpr (Q8) {
#pragma unroll
            for (int j = 0; j < kQ8PerThread; ++j) {
                const int i = tid + 256 * j, c = i / kQ8Pieces, pc = i % kQ8Pieces;
                if (i >= kTileCells * kQ8Pieces) continue;
                const uint32_t qw[4] = {nx[j].x, nx[j].y, nx[j].z, nx[j].w};
                __half2 hv[8];
#pragma unroll
                for (int k = 0; k < 4; ++k) hy4k::q8_widen4(qw[k], nd[j], hv[2 * k], hv[2 * k + 1]);
                unsigned hw[8];
#pragma unroll
                for (int k = 0; k < 8; ++k) hw[k] = *reinterpret_cast<const unsigned*>(&hv[k]);
                uint4* dst = reinterpret_cast<uint4*>(sm.k + c * kPad + pc * 16);
                dst[0] = make_uint4(hw[0], hw[1], hw[2], hw[3]);
                dst[1] = make_uint4(hw[4], hw[5], hw[6], hw[7]);
            }
        } else {
#pragma unroll
            for (int j = 0; j < 9; ++j) {
                const int i = tid + 256 * j, c = i / 72, ch = i % 72;
                *reinterpret_cast<uint4*>(sm.k + c * kPad + ch * 8) = nx[j];
            }
        }
    };
    fetch(0);
    stash();
    const int mt = w & 1, nt = w >> 1;
    const int rA = mt * 16 + (lane >> 2), rB = rA + 8;
    float mA = -INFINITY, mB = -INFINITY, lA = 0.f, lB = 0.f;
    float o[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    const __half* qrow = sm.q + (mt * 16 + ((li & 1) << 3) + lr) * kPad + (li >> 1) * 8;
    const int n_tiles = (cnt + kTileCells - 1) / kTileCells;
    for (int it = 0; it < n_tiles; ++it) {
        const int base = it * kTileCells;
        __syncthreads();
        if (it + 1 < n_tiles) fetch(base + kTileCells);
        float sc[4] = {0.f, 0.f, 0.f, 0.f};
        const __half* krow = sm.k + (nt * 8 + lr) * kPad + li * 8;
#pragma unroll 3
        for (int kk = 0; kk < kKvRow / 16; kk += 2) {
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
            const bool in = base + nt * 8 + 2 * tig + e < cnt;
            sc[e] = in ? sc[e] * scale : -INFINITY;
            sc[2 + e] = in ? sc[2 + e] * scale : -INFINITY;
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
        const float aA = mA == -INFINITY ? 0.f : expf(mA - nA), aB = mB == -INFINITY ? 0.f : expf(mB - nB);
        const float p0A = sc[0] == -INFINITY ? 0.f : expf(sc[0] - nA), p1A = sc[1] == -INFINITY ? 0.f : expf(sc[1] - nA);
        const float p0B = sc[2] == -INFINITY ? 0.f : expf(sc[2] - nB), p1B = sc[3] == -INFINITY ? 0.f : expf(sc[3] - nB);
        const int pc = nt * 8 + 2 * tig;
        *reinterpret_cast<unsigned*>(sm.p + rA * kPPad + pc) = h2(p0A, p1A);
        *reinterpret_cast<unsigned*>(sm.p + rB * kPPad + pc) = h2(p0B, p1B);
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
        for (int j = 0; j < 16; ++j) {
            o[j][0] *= aA;
            o[j][1] *= aA;
            o[j][2] *= aB;
            o[j][3] *= aB;
        }
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            unsigned a[4];
            ldsm4(a, sm.p + (mt * 16 + ((li & 1) << 3) + lr) * kPPad + (2 * k + (li >> 1)) * 8);
            const __half* vrow = sm.k + (16 * k + ((li & 1) << 3) + lr) * kPad + nt * 128 + (li >> 1) * 8;
#pragma unroll
            for (int n2 = 0; n2 < 8; ++n2) {
                unsigned b[4];
                ldsm4t(b, vrow + n2 * 16);
                mma16816(o[2 * n2], a, b[0], b[1]);
                mma16816(o[2 * n2 + 1], a, b[2], b[3]);
            }
        }
        __syncthreads();
        if (it + 1 < n_tiles) stash();
    }
    if (nt == 0 && tig == 0) {
        pt[(size_t) rA * (kLatent + 2)] = mA;
        pt[(size_t) rA * (kLatent + 2) + 1] = lA;
        pt[(size_t) rB * (kLatent + 2)] = mB;
        pt[(size_t) rB * (kLatent + 2) + 1] = lB;
    }
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int d = nt * 128 + j * 8 + 2 * tig;
        *reinterpret_cast<float2*>(pt + (size_t) rA * (kLatent + 2) + 2 + d) = make_float2(o[j][0], o[j][1]);
        *reinterpret_cast<float2*>(pt + (size_t) rB * (kLatent + 2) + 2 + d) = make_float2(o[j][2], o[j][3]);
    }
}

// ------------------------------------------------------------------------------------------------ the indexer
constexpr int kIdxHeads = 16;   // the indexer's heads: one m-tile
constexpr int kIdxDim = 128;
constexpr int kIdxTile = 64;    // pooled keys a tile
constexpr int kIdxPad = 136;    // a key row in shared memory, halves (272 bytes)

/// The indexer's scores of the rows r0 + r (r < R) whose visible groups (pos / pool) outnumber `reach`: scores[r][g] =
/// sum_h w[t][h] relu(q[t][h] . k_g) for g below pos / pool (the others are not written).  A warp a row (its 16 heads
/// one m-tile, the queries as halves in registers); the block's 8 rows over blockIdx.y's share of the groups, `per`
/// of them, in tiles of 64 keys (halves in shared memory).  Grid (ceil(R / 8), shares), block 256.
__global__ void __launch_bounds__(256) k_index_scores_mma(const float* __restrict__ q, const float* __restrict__ w,
                                                           const float* __restrict__ pooled, const int32_t* __restrict__ pos,
                                                           int pool, int reach, int r0, int R, int per,
                                                           float* __restrict__ scores, int ld) {
    __shared__ __align__(16) __half ks[kIdxTile * kIdxPad];
    __shared__ int s_vis[8];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, tig = lane & 3, g8 = lane >> 2;
    const int li = lane >> 3, lr = lane & 7;
    const int r = blockIdx.x * 8 + warp, t = r0 + r;
    const int vis = r < R ? pos[t] / pool : 0;
    const bool need = r < R && vis > reach;
    if (lane == 0) s_vis[warp] = need ? vis : 0;
    __syncthreads();
    int bmax = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) bmax = max(bmax, s_vis[i]);
    const int g0 = blockIdx.y * per, g1 = min(g0 + per, bmax);
    if (g0 >= g1) return;   // the block's every row: uniform
    unsigned qa[kIdxDim / 16][4];
    const float* qt = q + (size_t) t * kIdxHeads * kIdxDim;
#pragma unroll
    for (int kk = 0; kk < kIdxDim / 16; ++kk) {
        const int d = 16 * kk + 2 * tig;
        if (need) {
            const float2 x0 = *reinterpret_cast<const float2*>(qt + (size_t) g8 * kIdxDim + d);
            const float2 x1 = *reinterpret_cast<const float2*>(qt + (size_t) (g8 + 8) * kIdxDim + d);
            const float2 x2 = *reinterpret_cast<const float2*>(qt + (size_t) g8 * kIdxDim + d + 8);
            const float2 x3 = *reinterpret_cast<const float2*>(qt + (size_t) (g8 + 8) * kIdxDim + d + 8);
            qa[kk][0] = h2(x0.x, x0.y);
            qa[kk][1] = h2(x1.x, x1.y);
            qa[kk][2] = h2(x2.x, x2.y);
            qa[kk][3] = h2(x3.x, x3.y);
        } else {
            qa[kk][0] = qa[kk][1] = qa[kk][2] = qa[kk][3] = 0u;
        }
    }
    const float wA = need ? w[(size_t) t * kIdxHeads + g8] : 0.f, wB = need ? w[(size_t) t * kIdxHeads + g8 + 8] : 0.f;
    for (int gb = g0; gb < g1; gb += kIdxTile) {
        __syncthreads();   // the last tile is read
        for (int i = threadIdx.x; i < kIdxTile * (kIdxDim / 4); i += blockDim.x) {
            const int gr = i / (kIdxDim / 4), c4 = i % (kIdxDim / 4), g = gb + gr;
            const float4 v = g < g1 ? *reinterpret_cast<const float4*>(pooled + (size_t) g * kIdxDim + 4 * c4)
                                    : make_float4(0.f, 0.f, 0.f, 0.f);
            *reinterpret_cast<uint2*>(ks + gr * kIdxPad + 4 * c4) = make_uint2(h2(v.x, v.y), h2(v.z, v.w));
        }
        __syncthreads();
        if (!need || gb >= vis) continue;
#pragma unroll 2
        for (int nb = 0; nb < kIdxTile / 8; ++nb) {
            float c[4] = {0.f, 0.f, 0.f, 0.f};
            const __half* krow = ks + (nb * 8 + lr) * kIdxPad + li * 8;
#pragma unroll
            for (int kk = 0; kk < kIdxDim / 16; kk += 2) {
                unsigned b[4];
                ldsm4(b, krow + kk * 16);
                mma16816(c, qa[kk], b[0], b[1]);
                mma16816(c, qa[kk + 1], b[2], b[3]);
            }
            // relu, the heads' weights, the sum over the 16 heads (the rows: lanes of one column across g8)
            float v0 = fmaxf(c[0], 0.f) * wA + fmaxf(c[2], 0.f) * wB;
            float v1 = fmaxf(c[1], 0.f) * wA + fmaxf(c[3], 0.f) * wB;
#pragma unroll
            for (int o = 4; o < 32; o <<= 1) {
                v0 += __shfl_xor_sync(0xffffffffu, v0, o);
                v1 += __shfl_xor_sync(0xffffffffu, v1, o);
            }
            if (g8 == 0) {
                const int g = gb + nb * 8 + 2 * tig;
                if (g < vis) scores[(size_t) r * ld + g] = v0;
                if (g + 1 < vis) scores[(size_t) r * ld + g + 1] = v1;
            }
        }
    }
}

/// The groups a chunk completes (positions p0 .. p0 + T - 1; group g0 + blockIdx.x): pooled from their cells' value
/// and gate rows - a cell at or after p0 from the chunk's rows (val, gate [T][dim]), an earlier one from the tail
/// ([n_tail][value | gate][dim], its slot position % n_tail) - with k_pool_rows' arithmetic, into pooled + g * dim.  A
/// block of dim threads (pool <= 8).
__global__ void k_pool_chunk(const float* val, const float* gate, const float* tail, int n_tail, int p0, int g0, int dim,
                             int pool, const float* ape, const float* norm, float eps, float theta_scale, int n_rot,
                             float* pooled) {
    __shared__ float sh[32];
    __shared__ float kk[256];
    const int grp = g0 + blockIdx.x, c = threadIdx.x;
    if (c < dim) {
        float gv[8], vv[8];
        float mx = -INFINITY;
        for (int m = 0; m < pool; ++m) {
            const int p = grp * pool + m;
            const float* vr = p >= p0 ? val + (size_t) (p - p0) * dim : tail + (size_t) (p % n_tail) * 2 * dim;
            const float* gr = p >= p0 ? gate + (size_t) (p - p0) * dim : tail + (size_t) (p % n_tail) * 2 * dim + dim;
            gv[m] = gr[c] + ape[(size_t) m * dim + c];
            vv[m] = vr[c];
            mx = fmaxf(mx, gv[m]);
        }
        float den = 0.f, num = 0.f;
        for (int m = 0; m < pool; ++m) {
            const float e = expf(gv[m] - mx);
            den += e;
            num += e * vv[m];
        }
        kk[c] = num / den;
    }
    float s = c < dim ? kk[c] * kk[c] : 0.f;
    s = hy4k::block_sum(s, sh);
    const float inv = rsqrtf(s / (float) dim + eps);
    if (c < dim) kk[c] = kk[c] * inv * norm[c];
    __syncthreads();
    const int nope = dim - n_rot;
    if (c < dim) {
        float v = kk[c];
        if (c >= nope) {
            const int i = (c - nope) / 2;
            const float theta = (float) (grp * pool) * powf(theta_scale, (float) i);
            const float cs = cosf(theta), sn = sinf(theta);
            const float x0 = kk[nope + 2 * i], x1 = kk[nope + 2 * i + 1];
            v = ((c - nope) & 1) == 0 ? x0 * cs - x1 * sn : x0 * sn + x1 * cs;
        }
        pooled[(size_t) grp * dim + c] = v;
    }
}

// ------------------------------------------------------------------------------------------------ the window's cells
/// Cells of positions a .. a + gridDim.x - 1 copied between a ring (slot position % mod) and a linear buffer (row
/// position - b): src's row by mod_s (0: linear), dst's by mod_d.  A block a cell.
__global__ void k_copy_cells(const __half* src, int mod_s, __half* dst, int mod_d, int a, int b) {
    const int p = a + blockIdx.x;
    const uint4* s = reinterpret_cast<const uint4*>(src + (size_t) (mod_s > 0 ? p % mod_s : p - b) * kKvRow);
    uint4* d = reinterpret_cast<uint4*>(dst + (size_t) (mod_d > 0 ? p % mod_d : p - b) * kKvRow);
    for (int j = threadIdx.x; j < kKvRow / 8; j += blockDim.x) d[j] = s[j];
}

// ------------------------------------------------------------------------------------------------ the routed experts
/// q8_1 rows of llama.cpp's MMQ layout ([column block][row], blocks of 128 values in 144 bytes) gathered: dst row e =
/// src row map[e] (src has src_rows rows, dst n; n_bc column blocks).  A block per row.
__global__ void k_gather_q8(const uint4* __restrict__ src, uint4* __restrict__ dst, const int32_t* __restrict__ map, int n,
                            int src_rows, int n_bc) {
    const int e = blockIdx.x, s = map[e];
    for (int i = threadIdx.x; i < n_bc * 9; i += blockDim.x) {
        const int bc = i / 9, j = i % 9;
        dst[((size_t) bc * n + e) * 9 + j] = src[((size_t) bc * src_rows + s) * 9 + j];
    }
}

/// h [rows][F] = SwiGLU of gu [rows][gate F | up F], clamped as the native kernels' swiglu_l (L > 0: silu(min(g, L)) *
/// clamp(u, -L, L))
__global__ void k_swiglu_split(const float* gu, float* h, int F, int64_t n, float L) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t r = i / F, c = i % F;
    float g = gu[r * 2 * F + c], u = gu[r * 2 * F + F + c];
    if (L > 0.f) {
        g = fminf(g, L);
        u = fminf(fmaxf(u, -L), L);
    }
    h[i] = (g / (1.0f + __expf(-g))) * u;
}

/// A group's routed rows added to their tokens' sums: run j (token tok[j], the group's entries ent[start[j] ..
/// start[j + 1]) in pick order, weights wt alongside) acc[tok] += sum w_e rows[e] in that order.  A block per run.
__global__ void k_accum_runs(const float* __restrict__ rows, const int32_t* __restrict__ tok,
                             const int32_t* __restrict__ start, const int32_t* __restrict__ ent,
                             const float* __restrict__ wt, float* acc, int n_cols) {
    const int j = blockIdx.x, a = start[j], b = start[j + 1];
    float* dst = acc + (size_t) tok[j] * n_cols;
    for (int c = threadIdx.x; c < n_cols; c += blockDim.x) {
        float s = dst[c];
        for (int i = a; i < b; ++i) s = s + wt[i] * rows[(size_t) ent[i] * n_cols + c];
        dst[c] = s;
    }
}

/// ids[h * T + t] = t * G + h: the per-head products' rows in head order mapped to the [T][G] rows
__global__ void k_head_ids(int32_t* ids, int G, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < G * T) ids[i] = (i % T) * G + i / T;
}

/// bounds[h] = h * T for h <= G
__global__ void k_head_bounds(int32_t* b, int G, int T) {
    const int h = threadIdx.x;
    if (h <= G) b[h] = h * T;
}

/// n floats as halves, saturating at the half's range (a tier's sums crossing the links)
__global__ void k_to_half(const float* __restrict__ x, __half* __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half_rn(fminf(fmaxf(x[i], -65504.f), 65504.f));
}
/// y[i] += b[i], b halves
__global__ void k_add_half(float* __restrict__ y, const __half* __restrict__ b, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += __half2float(b[i]);
}

}  // namespace hy4p
