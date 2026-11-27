// src/kernels/cuda/gemm_tc.cuh - the products of gemm.cu (include/lamina/kernels/gemm.hpp) as templates over their
// tiles, for gemm.cu's instances and for a bench of others.
//
// Y[T][N] = X[T][K] . W[N][K]^T, X and W halves or BF16 with K along a row, the sums FP32 (mma.m16n8k16: X's rows are
// its row operand and W's rows its column operand as they lie, no transposition).  A block takes BM = 16 MT WM rows of
// X against BN = 16 NT WN rows of W and walks K in slabs of BK through shared memory (cp.async into ST buffers, the
// copies ST - 1 slabs ahead of the products, one barrier a slab), each warp 16 MT x 16 NT sums in registers, K in
// order.  Rows past T or N and values past K arrive as zeros (a copy of no source bytes).  The FP32 product
// (sg_block) has the same frame on the float units: GM x GN threads of TM x TN sums, slabs of 32.
//
// The grid is (tiles, 1, splits).  A split sums its part of K, k in [z kz, z kz + kz), into its own T x N floats (the
// partials, tc_reduce_kernel adds them up in order).  The tiles' order is the launch's choice: blocks start in the
// grid's order, so with X's tiles the inner index the blocks in flight share a tile of W and walk X's tiles, with
// W's tiles inner the other way round; `ci` cuts the inner operand into chunks of that many tiles, each walked
// against every outer tile before the next.  (Measured on an RTX 3090 with both operands past its L2 cache, either
// order and chunks that fit the cache are within 3% of each other: the blocks in flight span many tiles.)
#pragma once

#include "lamina/kernels/warp.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace lamina::kernels::gemmk {

constexpr int kYHalf = 1;    // Y in halves (the sums rounded to nearest)
constexpr int kYPair = 2;    // Y's value pairs are aligned: two a store
constexpr int kYQuad = 8;    // (floats) Y's values in fours are aligned: four a store
constexpr int kTInner = 4;   // X's tiles are the grid's inner index (else W's)

struct TcArgs {
    const uint16_t* x;   // [T][K], rows ldx apart
    const uint16_t* w;   // [N][K]
    void* y;             // [T][N] rows ldy apart; split z's at y + z * zstride values
    long long ldx, ldy, zstride;
    int T, N, K, kz;     // kz: a split's part of K (a multiple of the slab but the last)
    int nI, nO, ci;      // tiles of the inner and the outer operand; the inner operand's tiles a chunk (0: all)
    float alpha, beta;   // y = alpha * sums + beta * y (beta 0: y is not read)
    int flags;
};

/// Block i's tiles of the inner and the outer operand: chunk after chunk of the inner one, each against every outer
/// tile (the last chunk the tiles left).
__device__ __forceinline__ void tile_of(int i, int nI, int nO, int ci, int& inner, int& outer) {
    if (ci <= 0 || ci >= nI) {
        outer = i / nI;
        inner = i % nI;
        return;
    }
    const int full = nI / ci * ci * nO;
    if (i < full) {
        const int c = i / (ci * nO), r = i % (ci * nO);
        outer = r / ci;
        inner = c * ci + r % ci;
    } else {
        const int j = i - full, cl = nI % ci;
        outer = j / cl;
        inner = nI / ci * ci + j % cl;
    }
}

template <bool BF>
__device__ __forceinline__ void tc_mma(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    if constexpr (BF)
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                     "{%0,%1,%2,%3};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    else
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                     "{%0,%1,%2,%3};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
/// 16 bytes to shared memory in the background, `bytes` of them from g and zeros after
__device__ __forceinline__ void tc_cp16(void* s, const void* g, unsigned bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"((unsigned) __cvta_generic_to_shared(s)),
                 "l"(g), "r"(bytes));
}

/// 16 bytes to shared memory in the background
__device__ __forceinline__ void tc_cp16f(void* s, const void* g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned) __cvta_generic_to_shared(s)), "l"(g));
}

/// A buffer holds a slab's rows without padding, a row's granules (16 bytes) in swizzled places: granule q of row r at
/// q ^ tc_swz(r), so that the eight rows an ldmatrix reads at one q lie in eight different bank groups.
template <int BK> __device__ __forceinline__ constexpr int tc_swz(int r) { return BK == 64 ? r & 7 : (r >> 1) & 3; }
template <int MT, int NT, int WM, int WN, int BK, int ST> constexpr int tc_smem_bytes() {
    return ST * (16 * MT * WM + 16 * NT * WN) * BK * 2;
}

/// A block's product: rows t0 .. t0 + BM of x against rows n0 .. n0 + BN of w over k in [kb, ke), into y.
///
/// A tensor unit takes one product of its warp at a time and whatever the warp issues between two products keeps it
/// idle, unless it is one instruction that finishes behind the product before it.  So the slab loop is one flat
/// sequence, a k step's products with its other work spread between them one at a time: the loads (ldmatrix) of the
/// NEXT step's operands into a second set of registers, the copies (cp.async) of the slab ST buffers ahead, and once
/// a slab - in its step before last, after the last step's operands are loaded - the wait for the next slab's copy
/// and the barrier.  After that barrier nothing reads the slab's buffer, and the copies issued from then on fill it
/// with the slab ST ahead: a group of copies runs from a slab's last step through the next slab's steps before its
/// barrier, where it is committed.
/// DBG (a bench's): 1 the copies without the products, 2 the products without the copies after the first buffers,
/// 3 as 2 without the stores, 4 the stores alone.
template <bool BF, int MT, int NT, int WM, int WN, int BK, int ST, bool PL, int DBG>
__device__ __forceinline__ void tc_block(const uint16_t* __restrict__ x, const uint16_t* __restrict__ w, void* yv,
                                         long long ldx, long long ldy, int T, int N, int K, int kb, int ke, int t0,
                                         int n0, float alpha, float beta, int flags) {
    constexpr int BM = 16 * MT * WM, BN = 16 * NT * WN, NTH = 32 * WM * WN, GK = BK / 8, KS = BK / 16;
    constexpr int kBuf = (BM + BN) * BK, kGran = (BM + BN) * GK, NG = (kGran + NTH - 1) / NTH;
    constexpr int M = 2 * MT * NT, NLD = MT + NT;   // a k step's products, and its loads of the next step's operands
    constexpr int CL = NG / KS, CR = NG - CL;       // a group's copies in a slab's last
                                                    // step, and in the next slab's others
    static_assert(BK == 32 || BK == 64, "a slab is 32 or 64 values of K");
    static_assert(ST >= 3 && ST <= 5, "3 to 5 buffers");
    static_assert(NTH % GK == 0, "a thread's granules lie at one place of their rows");
    extern __shared__ __align__(16) uint16_t smem[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp / WN, wn = warp % WN;
    const int nk = (ke - kb + BK - 1) / BK, rem = (ke - kb) % BK, q = tid % GK;
    // a thread's granules (16 bytes: eight values of a row), every slab the same rows: where each starts in x or w at
    // the block's first k, its place in a buffer, and whether its row is inside x or w
    const uint16_t* gsrc[NG];
    unsigned gdst[NG], gmask = 0;
#pragma unroll
    for (int j = 0; j < NG; ++j) {
        const int i = tid + j * NTH, r = min(i, kGran - 1) / GK;
        const bool isx = r < BM;
        const int row = isx ? t0 + r : n0 + r - BM;
        const bool in = i < kGran && row < (isx ? T : N);
        gsrc[j] = in ? (isx ? x + (long long) row * ldx : w + (long long) row * K) + kb + q * 8 : w;
        gdst[j] = (unsigned) (r * BK + (q ^ tc_swz<BK>(r)) * 8);
        gmask |= in ? 1u << j : 0u;
    }
    const bool inside = t0 + BM <= T && n0 + BN <= N;
    if (!inside) {   // rows past x or w: zeros in every buffer, never copied
#pragma unroll
        for (int p = 0; p < ST; ++p)
#pragma unroll
            for (int j = 0; j < NG; ++j)
                if (tid + j * NTH < kGran && !((gmask >> j) & 1))
                    *reinterpret_cast<uint4*>(smem + p * kBuf + gdst[j]) = make_uint4(0, 0, 0, 0);
    }
    // A group: the copies of slab st.  Its last slab may end inside a thread's granule or before it: fewer bytes, and
    // for none a place inside the row to copy nothing from.
    struct Group {
        long long koff;
        unsigned bytes;
        int buf;
        bool live;
    };
    auto group = [&](int st, int buf) {
        Group g;
        const int left = rem != 0 && st == nk - 1 ? rem - q * 8 : 8;
        g.bytes = left >= 8 ? 16u : left > 0 ? 2u * (unsigned) left : 0u;
        g.koff = (long long) st * BK;
        if (g.bytes == 0) g.koff = st > 0 ? g.koff - BK : -(long long) (q * 8);
        g.buf = buf;
        g.live = st < nk;
        if constexpr (DBG == 2 || DBG == 3 || DBG == 4) g.live = g.live && st < ST - 1;
        return g;
    };
    auto copy = [&](int j, const Group& g) {
        if (!(g.live && ((gmask >> j) & 1))) return;
        uint16_t* const dst = smem + g.buf * kBuf + gdst[j];
        const uint16_t* const src = gsrc[j] + g.koff;
        if constexpr (PL) {
#pragma unroll
            for (int e = 0; e < 8; ++e) dst[e] = 2 * e < (int) g.bytes ? src[e] : (uint16_t) 0;
        } else {
            tc_cp16(dst, src, g.bytes);
        }
    };
    float acc[MT][2 * NT][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int j = 0; j < 2 * NT; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[mt][j][i] = 0.0f;
    // a lane's rows of the operands' first 16 and where its granule of each k step lies in them (halves)
    unsigned aoff[KS], boff[KS];
    {
        const int ra = wm * 16 * MT + (lane & 15), rb = BM + wn * 16 * NT + (lane & 7) + ((lane >> 4) << 3);
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
            aoff[ks] = (unsigned) (ra * BK + ((2 * ks + (lane >> 4)) ^ tc_swz<BK>(ra)) * 8);
            boff[ks] = (unsigned) (rb * BK + ((2 * ks + ((lane >> 3) & 1)) ^ tc_swz<BK>(rb)) * 8);
        }
    }
    unsigned af[2][MT][4], bf[2][NT][4];                           // a k step's operands and the next step's
    auto ld = [&](int set, int o, const uint16_t* buf, int ks) {   // operand o of a step: X's tiles, then W's
        if constexpr (DBG == 1 || DBG == 4) return;
        if (o < MT) ldsm4(af[set][o], buf + aoff[ks] + o * 16 * BK);
        else
            ldsm4(bf[set][o - MT], buf + boff[ks] + (o - MT) * 16 * BK);   // [0..1]: columns 0-7 (k
                                                                           // 0-7, 8-15), [2..3]: 8-15
    };
    // the first ST - 1 slabs, a group each, and the first part of the next one's
#pragma unroll
    for (int p = 0; p < ST - 1; ++p) {
        const Group g = group(p, p);
#pragma unroll
        for (int j = 0; j < NG; ++j) copy(j, g);
        if constexpr (!PL) cp_commit();
    }
    Group ga = group(ST - 1, ST - 1);
#pragma unroll
    for (int j = 0; j < CL; ++j) copy(j, ga);
    if constexpr (!PL) cp_wait<ST - 2>();
    __syncthreads();
#pragma unroll
    for (int o = 0; o < NLD; ++o) ld(0, o, smem, 0);
    int b = 0;   // slab s's buffer
    for (int s = 0; s < nk; ++s) {
        const int bn = b == ST - 1 ? 0 : b + 1;
        const uint16_t* const buf = smem + b * kBuf;
        const uint16_t* const bufn = smem + bn * kBuf;
        Group gb = ga;
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
            const int cur = ks & 1, nxt = cur ^ 1;
            // the step's other work: the next step's operands, its copies, and before the last step the group's
            // commit, the wait for the next slab and the barrier
            const int ncp = ks == KS - 1 ? CL : (ks + 1) * CR / (KS - 1) - ks * CR / (KS - 1);
            const int S = NLD + ncp + (ks == KS - 2 ? 3 : 0);
            if (ks == KS - 1) gb = group(s + ST, b);   // (the slab's buffer is free since the barrier)
            auto other = [&](int o) {
                if (o < NLD) {
                    if (ks < KS - 1) ld(nxt, o, buf, ks + 1);
                    else if (s + 1 < nk) ld(nxt, o, bufn, 0);
                } else if (o - NLD < ncp) {
                    if (ks == KS - 1) copy(o - NLD, gb);
                    else copy(CL + ks * CR / (KS - 1) + o - NLD, ga);
                } else if (o - NLD - ncp == 0) {
                    if constexpr (!PL) cp_commit();
                } else if (o - NLD - ncp == 1) {
                    if constexpr (!PL) cp_wait<ST - 2>();
                } else {
                    __syncthreads();
                }
            };
#pragma unroll
            for (int i = 0; i < M; ++i) {
                if constexpr (DBG != 1 && DBG != 4) {
                    const int nt = i / (2 * MT), mt = i % (2 * MT) / 2, h = i & 1;
                    tc_mma<BF>(acc[mt][2 * nt + h], af[cur][mt], bf[cur][nt][2 * h], bf[cur][nt][2 * h + 1]);
                }
#pragma unroll
                for (int o = i * S / M; o < (i + 1) * S / M; ++o) other(o);
            }
        }
        ga = gb;
        b = bn;
    }
    if constexpr (DBG == 3) {
        float sum = 0.0f;
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int j = 0; j < 2 * NT; ++j) sum += acc[mt][j][0] + acc[mt][j][1] + acc[mt][j][2] + acc[mt][j][3];
        if (tid == 0) *((float*) yv + blockIdx.x) = sum;
        return;
    }
    // ---- the sums out: a lane holds rows lane / 4 and lane / 4 + 8 of each 16, two columns of each 8
    const bool half = (flags & kYHalf) != 0, pair = (flags & kYPair) != 0;
    const int tl = t0 + wm * 16 * MT + (lane >> 2), nl = n0 + wn * 16 * NT + (lane & 3) * 2;
    if (inside && !half && pair && alpha == 1.0f && beta == 0.0f) {
        float* const yb = (float*) yv + (long long) tl * ldy + nl;
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                float* const yr = yb + (long long) (mt * 16 + h * 8) * ldy;
#pragma unroll
                for (int j = 0; j < 2 * NT; ++j)
                    *reinterpret_cast<float2*>(yr + j * 8) = make_float2(acc[mt][j][h * 2], acc[mt][j][h * 2 + 1]);
            }
        return;
    }
    if (!half) {
        // floats: a group of 16 rows' values of Y are read before any of them is stored (beta), so the loads are in
        // flight together - a product of a short K is its outputs' bytes, and a load that waits for the store before
        // it costs the block its latency a fragment
        float* const yf = (float*) yv;
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            float2 o[2][2 * NT];
            if (beta != 0.0f) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int t = tl + mt * 16 + h * 8;
#pragma unroll
                    for (int j = 0; j < 2 * NT; ++j) {
                        const int n = nl + j * 8;
                        o[h][j] = make_float2(0.0f, 0.0f);
                        if (t >= T || n >= N) continue;
                        const float* const y = yf + (long long) t * ldy + n;
                        if (n + 1 < N && pair) {
                            o[h][j] = *reinterpret_cast<const float2*>(y);
                        } else {
                            o[h][j].x = y[0];
                            if (n + 1 < N) o[h][j].y = y[1];
                        }
                    }
                }
            }
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int t = tl + mt * 16 + h * 8;
                if (t >= T) continue;
#pragma unroll
                for (int j = 0; j < 2 * NT; ++j) {
                    const int n = nl + j * 8;
                    if (n >= N) continue;
                    float v0 = acc[mt][j][h * 2] * alpha, v1 = acc[mt][j][h * 2 + 1] * alpha;
                    if (beta != 0.0f) {
                        v0 += beta * o[h][j].x;
                        v1 += beta * o[h][j].y;
                    }
                    float* const y = yf + (long long) t * ldy + n;
                    if (n + 1 < N && pair) {
                        *reinterpret_cast<float2*>(y) = make_float2(v0, v1);
                    } else {
                        y[0] = v0;
                        if (n + 1 < N) y[1] = v1;
                    }
                }
            }
        }
        return;
    }
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int t = tl + mt * 16 + h * 8;
            if (t >= T) continue;
#pragma unroll
            for (int j = 0; j < 2 * NT; ++j) {
                const int n = nl + j * 8;
                if (n >= N) continue;
                float v0 = acc[mt][j][h * 2] * alpha, v1 = acc[mt][j][h * 2 + 1] * alpha;
                const bool two = n + 1 < N;
                __half* const y = (__half*) yv + (long long) t * ldy + n;
                if (beta != 0.0f) {
                    v0 += beta * __half2float(y[0]);
                    if (two) v1 += beta * __half2float(y[1]);
                }
                if (two && pair) {
                    *reinterpret_cast<__half2*>(y) = __floats2half2_rn(v0, v1);
                } else {
                    y[0] = __float2half_rn(v0);
                    if (two) y[1] = __float2half_rn(v1);
                }
            }
        }
}

template <bool BF, int MT, int NT, int WM, int WN, int BK, int ST, bool PL = false, int DBG = 0>
__global__ void __launch_bounds__(32 * WM * WN) tc_kernel(const TcArgs a) {
    int ti, to;
    tile_of((int) blockIdx.x, a.nI, a.nO, a.ci, ti, to);
    const int t0 = ((a.flags & kTInner) ? ti : to) * (16 * MT * WM),
              n0 = ((a.flags & kTInner) ? to : ti) * (16 * NT * WN);
    const int kb = blockIdx.z * a.kz, ke = min(a.K, kb + a.kz);
    const long long zoff = (long long) blockIdx.z * a.zstride;
    void* const y = (a.flags & kYHalf) ? (void*) ((__half*) a.y + zoff) : (void*) ((float*) a.y + zoff);
    tc_block<BF, MT, NT, WM, WN, BK, ST, PL, DBG>(a.x, a.w, y, a.ldx, a.ldy, a.T, a.N, a.K, kb, ke, t0, n0, a.alpha,
                                                  a.beta, a.flags);
}

/// Several products of one N and K in one launch (a batch's experts, K whole): a block belongs to the problem whose
/// first block is the last at or before it, and takes its x, w and y from the device arrays.
constexpr int kTcGroupMax = 32;
struct TcGroupArgs {
    const uint16_t* const* x;   // [n] pointers: problem i's x [rows[i]][K], w [N][K] and y [rows[i]][N]
    const uint16_t* const* w;
    float* const* y;
    int first[kTcGroupMax + 1];   // problem i's first block
    int rows[kTcGroupMax];
    int n, N, K;
};
template <bool BF, int MT, int NT, int WM, int WN, int BK, int ST>
__global__ void __launch_bounds__(32 * WM * WN) tc_group_kernel(const TcGroupArgs a) {
    constexpr int BM = 16 * MT * WM, BN = 16 * NT * WN;
    int i = 0;
    for (int j = 1; j < a.n; ++j) i = (int) blockIdx.x >= a.first[j] ? j : i;
    const int local = (int) blockIdx.x - a.first[i], nN = (a.N + BN - 1) / BN;
    tc_block<BF, MT, NT, WM, WN, BK, ST, false, 0>(a.x[i], a.w[i], a.y[i], a.K, a.N, a.rows[i], a.N, a.K, 0, a.K,
                                                   local / nN * BM, local % nN * BN, 1.0f, 0.0f,
                                                   a.N % 2 == 0 ? kYPair : 0);
}

/// y = alpha * (the splits' partials part[z] added in order, T x N floats each, zstride floats apart: a multiple of
/// four) + beta * y; y floats, or halves (HALF).  A thread four values.
template <bool HALF>
__global__ void tc_reduce_kernel(const float* __restrict__ part, void* __restrict__ yv, long long ldy,
                                 long long zstride, int T, int N, int splits, float alpha, float beta) {
    const long long i4 = ((long long) blockIdx.x * blockDim.x + threadIdx.x) * 4, n_all = (long long) T * N;
    if (i4 >= n_all) return;
    float4 s = *reinterpret_cast<const float4*>(part + i4);
    for (int z = 1; z < splits; ++z) {
        const float4 p = *reinterpret_cast<const float4*>(part + z * zstride + i4);
        s.x += p.x;
        s.y += p.y;
        s.z += p.z;
        s.w += p.w;
    }
    const float v[4] = {s.x * alpha, s.y * alpha, s.z * alpha, s.w * alpha};
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const long long i = i4 + e;
        if (i >= n_all) break;
        const long long at = i / N * ldy + i % N;
        if constexpr (HALF) {
            __half* const y = (__half*) yv + at;
            *y = __float2half_rn(beta != 0.0f ? v[e] + beta * __half2float(*y) : v[e]);
        } else {
            float* const y = (float*) yv + at;
            *y = beta != 0.0f ? v[e] + beta * *y : v[e];
        }
    }
}

// ---- A few outputs a row (N <= kRdN: a gate, an inject projection): no tiles.  A row of X goes to `wpr` warps (1, 2,
// 4 or 8 of a block's eight: few rows are then more streams of the card's memory), each a part of K in granules of
// eight values: a lane takes every 32nd granule of its warp's part, loads four of them from the row ahead, then each
// one's granule of every row of W, multiplies in floats (a product of two halves or BF16 values is exact in FP32) and
// adds into two sums an output (the even and the odd values), its granules in order; the lanes' sums are added over the
// warp, the warps' in order.  One launch.  X's and W's rows 16-byte aligned, K a multiple of 8.
constexpr int kRdN = 8;
struct RdArgs {
    const uint16_t* x;
    const uint16_t* w;
    void* y;
    long long ldx, ldy;
    int T, N, K, wpr;
    float alpha, beta;
    int flags;   // kYHalf
};
template <bool BF> __device__ __forceinline__ float rd_f(uint32_t word, int hi) {
    const uint32_t v = hi ? word >> 16 : word & 0xffffu;
    if constexpr (BF) return __uint_as_float(v << 16);
    else return __half2float(__ushort_as_half((unsigned short) v));
}
template <bool BF> __global__ void __launch_bounds__(256) rd_kernel(const RdArgs a) {
    __shared__ float part[8][kRdN];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, rows = 8 / a.wpr;
    const int t = blockIdx.x * rows + warp / a.wpr, sl = warp % a.wpr;
    float acc[kRdN][2];
#pragma unroll
    for (int n = 0; n < kRdN; ++n) acc[n][0] = acc[n][1] = 0.0f;
    if (t < a.T) {
        const uint4* const xr = reinterpret_cast<const uint4*>(a.x + (long long) t * a.ldx);
        const int G = a.K / 8, g1 = (int) ((long long) (sl + 1) * G / a.wpr);
        for (int g = (int) ((long long) sl * G / a.wpr) + lane; g < g1; g += 128) {
            uint4 xv[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) xv[u] = g + 32 * u < g1 ? xr[g + 32 * u] : make_uint4(0, 0, 0, 0);
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                if (g + 32 * u >= g1) break;
                const uint32_t xw[4] = {xv[u].x, xv[u].y, xv[u].z, xv[u].w};
                float xf[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) xf[e] = rd_f<BF>(xw[e >> 1], e & 1);
#pragma unroll
                for (int n = 0; n < kRdN; ++n) {
                    if (n >= a.N) break;
                    const uint4 wv = reinterpret_cast<const uint4*>(a.w + (long long) n * a.K)[g + 32 * u];
                    const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
                    for (int e = 0; e < 8; ++e) acc[n][e & 1] = fmaf(xf[e], rd_f<BF>(ww[e >> 1], e & 1), acc[n][e & 1]);
                }
            }
        }
    }
#pragma unroll
    for (int n = 0; n < kRdN; ++n) {
        if (n >= a.N) break;
        const float v = warp_sum(acc[n][0] + acc[n][1]);
        if (lane == 0) part[warp][n] = v;
    }
    __syncthreads();
    if (t >= a.T || sl != 0 || lane >= a.N) return;   // a row's first warp: lane n its output n
    float v = part[warp][lane];
    for (int s = 1; s < a.wpr; ++s) v += part[warp + s][lane];
    v *= a.alpha;
    const long long at = (long long) t * a.ldy + lane;
    if (a.flags & kYHalf) {
        __half* const y = (__half*) a.y + at;
        *y = __float2half_rn(a.beta != 0.0f ? v + a.beta * __half2float(*y) : v);
    } else {
        float* const y = (float*) a.y + at;
        *y = a.beta != 0.0f ? v + a.beta * *y : v;
    }
}

// ---- FP32: Y[T][N] = X[T][K] . W[N][K]^T on the float units.  A block of GM x GN threads, each TM x TN sums in
// registers: BM = GM TM rows of X against BN = GN TN rows of W, K in slabs of 32 through shared memory as the rows lie
// (cp.async, ST buffers, the copies of the slab ST - 1 ahead between a slab's products, as tc_block), four k a step: a
// thread loads TM + TN vectors of four and makes 4 TM TN multiply-adds, K in order.  A row's granules (four values)
// lie swizzled, so that the rows a warp reads at one k step are in different banks: granule q of X's row r at
// q ^ (r / TM & 7), of W's row r at q ^ (r / TN & 7).  The same splits of K and tiles' order as the tensor-core kernel.
struct SgArgs {
    const float* x;   // [T][K], rows ldx apart
    const float* w;   // [N][K]
    float* y;         // [T][N] rows ldy apart; split z's at y + z * zstride
    long long ldx, ldy, zstride;
    int T, N, K, kz;
    int nI, nO, ci;
    float alpha, beta;
    int flags;   // kTInner, kYQuad
};

constexpr int kSgK = 32;   // a slab
template <int GM, int GN, int TM, int TN, int ST> constexpr int sg_smem_bytes() {
    return ST * (GM * TM + GN * TN) * kSgK * 4;
}

template <int GM, int GN, int TM, int TN, int ST, bool PL>
__device__ __forceinline__ void sg_block(const float* __restrict__ x, const float* __restrict__ w,
                                         float* __restrict__ y, long long ldx, long long ldy, int T, int N, int K,
                                         int kb, int ke, int t0, int n0, float alpha, float beta, bool quad) {
    constexpr int BK = kSgK, BM = GM * TM, BN = GN * TN, NTH = GM * GN, GK = BK / 4;
    constexpr int kBuf = (BM + BN) * BK, kGran = (BM + BN) * GK, NG = (kGran + NTH - 1) / NTH;
    static_assert(ST >= 2 && ST <= 5, "2 to 5 buffers");
    extern __shared__ __align__(16) float smemf[];
    const int tid = threadIdx.x, tm = (tid / GN) * TM, tn = (tid % GN) * TN;
    const int nk = (ke - kb + BK - 1) / BK;
    auto swz = [](int r) { return r < BM ? (r / TM) & 7 : ((r - BM) / TN) & 7; };
    const float* gsrc[NG];
    unsigned gdst[NG];
#pragma unroll
    for (int j = 0; j < NG; ++j) {
        const int i = min(tid + j * NTH, kGran - 1), r = i / GK, q = i % GK;
        const bool isx = r < BM;
        const int row = isx ? t0 + r : n0 + r - BM;
        gsrc[j] = (isx ? x + (long long) row * ldx : w + (long long) row * K) + kb + q * 4;
        gdst[j] = (unsigned) (r * BK + (q ^ swz(r)) * 4);
    }
    const bool inside = t0 + BM <= T && n0 + BN <= N, ragged = (ke - kb) % BK != 0;
    auto copy = [&](int j, int s, int b) {
        if (tid + j * NTH >= kGran) return;
        float* const dst = smemf + b * kBuf + gdst[j];
        const float* const src = gsrc[j] + (long long) s * BK;
        if constexpr (PL) {
            const int i = tid + j * NTH, r = i / GK;
            const bool in = (r < BM ? t0 + r : n0 + r - BM) < (r < BM ? T : N);
            const int left = in ? ke - (kb + s * BK + (i % GK) * 4) : 0;
#pragma unroll
            for (int e = 0; e < 4; ++e) dst[e] = e < left ? src[e] : 0.0f;
        } else {
            if (inside && !(ragged && s == nk - 1)) {
                tc_cp16f(dst, src);
            } else {
                const int i = tid + j * NTH, r = i / GK;
                const bool in = (r < BM ? t0 + r : n0 + r - BM) < (r < BM ? T : N);
                const int left = in ? min(4, ke - (kb + s * BK + (i % GK) * 4)) : 0;
                tc_cp16(dst, left > 0 ? src : w, left > 0 ? 4u * (unsigned) left : 0u);
            }
        }
    };
    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
    const int sa = (tm / TM) & 7, sb = (tn / TN) & 7;   // its rows' swizzle (one for a
                                                        // thread's TM rows, one for its TN)
#pragma unroll
    for (int p = 0; p < ST - 1; ++p) {
        if (p < nk) {
#pragma unroll
            for (int j = 0; j < NG; ++j) copy(j, p, p);
        }
        if constexpr (!PL) cp_commit();
    }
    int b = 0;
    for (int s = 0; s < nk; ++s) {
        if constexpr (!PL) cp_wait<ST - 2>();
        __syncthreads();
        const int sc = s + ST - 1, bc = b == 0 ? ST - 1 : b - 1;
        const bool ahead = sc < nk;
        const float* const buf = smemf + b * kBuf;
#pragma unroll
        for (int kq = 0; kq < GK; ++kq) {
            if (ahead) {   // this step's part of the copies
#pragma unroll
                for (int j = kq * NG / GK; j < (kq + 1) * NG / GK; ++j) copy(j, sc, bc);
            }
            float4 av[TM], bv[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) av[i] = *reinterpret_cast<const float4*>(buf + (tm + i) * BK + (kq ^ sa) * 4);
#pragma unroll
            for (int j = 0; j < TN; ++j)
                bv[j] = *reinterpret_cast<const float4*>(buf + (BM + tn + j) * BK + (kq ^ sb) * 4);
            // k by k, every sum once a k: a sum's next term never waits for its last
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(av[i].x, bv[j].x, acc[i][j]);
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(av[i].y, bv[j].y, acc[i][j]);
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(av[i].z, bv[j].z, acc[i][j]);
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(av[i].w, bv[j].w, acc[i][j]);
        }
        if constexpr (!PL) cp_commit();
        b = b == ST - 1 ? 0 : b + 1;
    }
    if constexpr (TN % 4 == 0) {
        if (quad && inside) {
            // four values a store, a warp's a row's contiguous bytes; with beta a few rows' values of Y read before
            // any is stored (the loads in flight together: a product of a short K is its outputs' bytes)
            constexpr int JQ = TN / 4, RB = 8 / JQ < TM ? (8 / JQ > 0 ? 8 / JQ : 1) : TM;
            float* const yb = y + (long long) (t0 + tm) * ldy + n0 + tn;
#pragma unroll
            for (int i0 = 0; i0 < TM; i0 += RB) {
                float4 o[RB][JQ];
                if (beta != 0.0f) {
#pragma unroll
                    for (int i = 0; i < RB; ++i)
#pragma unroll
                        for (int q = 0; q < JQ; ++q)
                            if (i0 + i < TM)
                                o[i][q] = *reinterpret_cast<const float4*>(yb + (long long) (i0 + i) * ldy + q * 4);
                }
#pragma unroll
                for (int i = 0; i < RB; ++i)
#pragma unroll
                    for (int q = 0; q < JQ; ++q) {
                        if (i0 + i >= TM) continue;
                        float4 v = make_float4(acc[i0 + i][q * 4] * alpha, acc[i0 + i][q * 4 + 1] * alpha,
                                               acc[i0 + i][q * 4 + 2] * alpha, acc[i0 + i][q * 4 + 3] * alpha);
                        if (beta != 0.0f) {
                            v.x += beta * o[i][q].x;
                            v.y += beta * o[i][q].y;
                            v.z += beta * o[i][q].z;
                            v.w += beta * o[i][q].w;
                        }
                        *reinterpret_cast<float4*>(yb + (long long) (i0 + i) * ldy + q * 4) = v;
                    }
            }
            return;
        }
    }
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int t = t0 + tm + i;
        if (t >= T) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int n = n0 + tn + j;
            if (n >= N) continue;
            float* const o = y + (long long) t * ldy + n;
            float v = acc[i][j] * alpha;
            if (beta != 0.0f) v += beta * *o;
            *o = v;
        }
    }
}

template <int GM, int GN, int TM, int TN, int ST, bool PL = false>
__global__ void __launch_bounds__(GM* GN) sg_kernel(const SgArgs a) {
    int ti, to;
    tile_of((int) blockIdx.x, a.nI, a.nO, a.ci, ti, to);
    const int t0 = ((a.flags & kTInner) ? ti : to) * (GM * TM), n0 = ((a.flags & kTInner) ? to : ti) * (GN * TN);
    const int kb = blockIdx.z * a.kz, ke = min(a.K, kb + a.kz);
    sg_block<GM, GN, TM, TN, ST, PL>(a.x, a.w, a.y + (long long) blockIdx.z * a.zstride, a.ldx, a.ldy, a.T, a.N, a.K,
                                     kb, ke, t0, n0, a.alpha, a.beta, (a.flags & kYQuad) != 0);
}

}   // namespace lamina::kernels::gemmk
