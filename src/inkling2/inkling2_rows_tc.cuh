// src/inkling2/inkling2_rows_tc.cuh - a decode forward's attention in the chunk kernel's layout (attn_tc_kernel's: a
// warp 16 rows, the probabilities in the lanes between the two products, tiles through cp.async) for
// attn_rows_kernel's work, the same partial sums: an engine's switch (LAMINA_INK_ROWS_TC=1), slower here - a decode
// forward's scan is the cells' bytes, not its products.
#pragma once

#include "inkling2_kernels.cuh"

namespace ik {

/// A decode forward's rows (1 .. kMaxT) over a stretch of a visit's cells, on the tensor cores: block (cb, g) takes the
/// rows' query heads of K/V head g (row r: forward row r / QH, its head r % QH of g's; T QH <= 16 NW) over stretch cb,
/// in tiles of 32 cells, a warp 16 rows: its sums are attn_rows_kernel's, in its order.
template <int NW, bool Q8>
__global__ void __launch_bounds__(NW * 32) attn_rows_tc_kernel(AttnIn a, int T) {
    constexpr int D = 128, NT = 32 * NW;
    constexpr int kRawCell = (D / 32) * 34;
    __shared__ __align__(16) __half tile[Q8 ? 1 : 2][2][kTcCells * kTcPad];   // [buffer][K | V]
    __shared__ __align__(16) uint8_t raw[2][Q8 ? kTcCells * kRawCell : 16];   // Q8: [K | V] as stored
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
    const int n_tiles = (max(cnt, 0) + kTcCells - 1) / kTcCells;
    auto slot_of = [&](int col) { return wrap ? (s_lo + col) % cap : s_lo + col; };
    auto fetch = [&](int buf, int base) {
        if constexpr (Q8) {   // 17 chunks of 8 bytes a cell and side
            for (int i = tid; i < 2 * kTcCells * (kRawCell / 8); i += NT) {
                const int side = i >= kTcCells * (kRawCell / 8), j = i - side * kTcCells * (kRawCell / 8);
                const int c = j / (kRawCell / 8), ch = j % (kRawCell / 8);
                uint8_t* d = raw[side] + c * kRawCell + ch * 8;
                if (base + c < cnt)
                    cp_async8(d, (side ? a.c.v : a.c.k) + (size_t) slot_of(base + c) * cbytes + (size_t) g * kRawCell + ch * 8);
                else
                    *reinterpret_cast<unsigned long long*>(d) = 0ull;
            }
        } else {              // 16 chunks of 16 bytes a cell and side
            for (int i = tid; i < kTcCells * 16; i += NT) {
                const int c = i / 16, ch = i % 16;
                __half* kd = tile[buf][0] + c * kTcPad + ch * 8;
                __half* vd = tile[buf][1] + c * kTcPad + ch * 8;
                if (base + c < cnt) {
                    const size_t at = (size_t) slot_of(base + c) * cbytes + (size_t) g * D * 2 + ch * 16;
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
    if (n_tiles > 0) fetch(0, 0);
    const bool mine = w * 16 < NQ;   // (a warp without rows only brings the tiles)
    const int rA = w * 16 + (lane >> 2), rB = rA + 8;
    const bool inA = rA < NQ, inB = rB < NQ;
    const int tA = inA ? rA / QH : 0, tB = inB ? rB / QH : 0, hA = inA ? rA % QH : 0, hB = inB ? rB % QH : 0;
    const size_t iA = (size_t) tA * a.H + g * QH + hA, iB = (size_t) tB * a.H + g * QH + hB;
    // q as the products' row operands: step ks is dims 16 ks .. +15
    unsigned qa[8][4];
    if (mine) {
        const float* qA = a.q + iA * D + 2 * tig;
        const float* qB = a.q + iB * D + 2 * tig;
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
    const int pA = (int) a.pos[tA], pB = (int) a.pos[tB];
    const float* bA = a.bias + iA * (a.ext + 1);
    const float* bB = a.bias + iB * (a.ext + 1);
    const float fA = inA ? bA[a.ext] : -INFINITY, fB = inB ? bB[a.ext] : -INFINITY;
    const float scale = rsqrtf((float) D);
    // a ring's slot stands for the position last written there: `last` less its distance behind the newest slot
    const int off0 = a.c.ring ? (int) (((last - s_lo) % cap + cap) % cap) : 0;
    const int c0 = (int) (first + s_lo);   // a range's first cell of the block
    const int p_first = (int) a.pos[0];
    // nothing yet: a largest logit that any real one passes, finite (no cell, no NaN)
    float mA = -1e30f, mB = mA, lA = 0.f, lB = 0.f;
    float o[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    for (int it = 0; it < n_tiles; ++it) {
        const int base = it * kTcCells;
        cp_wait<0>();
        __syncthreads();   // the tile has arrived, and the last one's cells were read
        const __half *kt, *vt;
        if constexpr (Q8) {
            for (int i = tid; i < 2 * kTcCells * 16; i += NT) {
                const int side = i >= kTcCells * 16, j = i - side * kTcCells * 16, c = j / 16, u = j % 16, blk = u / 4;
                const uint8_t* cell = raw[side] + c * kRawCell;
                const __half dh = *reinterpret_cast<const __half*>(cell + blk * 34);
                const __half2 d = __halves2half2(dh, dh);
                const int at = blk * 34 + 2 + (u % 4) * 8;
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
            if (it + 1 < n_tiles) fetch((it + 1) & 1, base + kTcCells);
            kt = tile[(Q8 ? 0 : it) & 1][0];
            vt = tile[(Q8 ? 0 : it) & 1][1];
        }
        if (!mine) continue;
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
        // a range's tile every row sees whole and past the table: no cell's distance to look at
        const bool far = !a.c.ring && a.win == 0 && c0 + base >= 0 && base + kTcCells <= cnt &&
                         p_first - (c0 + base + kTcCells - 1) >= a.ext;
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
                    int cp = c0 + col;
                    if (a.c.ring) {
                        int rel = off0 - col;
                        if (rel < 0) rel += (int) cap;
                        cp = (int) last - rel;
                    }
                    const int dA = pA - cp, dB = pB - cp;
                    const bool in = col < cnt && cp >= 0;
                    const bool okA = inA && in && dA >= 0 && (a.win == 0 || dA < a.win);
                    const bool okB = inB && in && dB >= 0 && (a.win == 0 || dB < a.win);
                    const float vA = bA[dA < 0 ? 0 : dA < a.ext ? dA : a.ext], vB = bB[dB < 0 ? 0 : dB < a.ext ? dB : a.ext];
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
        if (__any_sync(0xffffffffu, aA != 1.f || aB != 1.f)) {
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                o[j][0] *= aA;
                o[j][1] *= aA;
                o[j][2] *= aB;
                o[j][3] *= aB;
            }
        }
#pragma unroll
        for (int k = 0; k < 2; ++k) {
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
    if (!mine) return;
    // a row's m and sum by one of its lanes, its values by each lane's
    float* oA = a.part + (iA * a.n_chunks + cb) * (D + 2);
    float* oB = a.part + (iB * a.n_chunks + cb) * (D + 2);
    if (tig == 0) {
        if (inA) *reinterpret_cast<float2*>(oA) = make_float2(mA, lA);
        if (inB) *reinterpret_cast<float2*>(oB) = make_float2(mB, lB);
    }
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int d = j * 8 + 2 * tig;
        if (inA) *reinterpret_cast<float2*>(oA + 2 + d) = make_float2(o[j][0], o[j][1]);
        if (inB) *reinterpret_cast<float2*>(oB + 2 + d) = make_float2(o[j][2], o[j][3]);
    }
}

}  // namespace ik
