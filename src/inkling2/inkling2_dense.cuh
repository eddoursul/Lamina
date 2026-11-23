// src/inkling2/inkling2_dense.cuh - a prompt chunk's dense Q8_0 products by a kernel of the engine's own (an engine's
// switch, LAMINA_INK_DENSE_OWN=1): MMQ's sums on the int8 tensor cores, 21-23 T multiply-adds a second on an RTX 3090
// where MMQ's own reach 38-49 (the optimization repository's scratch/inkling/kern/dense_bench.cu measures both).
#pragma once

#include "lamina/kernels/warp.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ik_dense {

using lamina::kernels::ldsm4;

__device__ __forceinline__ void mma32(int (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

constexpr int kPitch = 144;   // a tile row's bytes: 128 codes + 16 (ldmatrix's eight rows on distinct banks)

// ---------------------------------------------------------------- the candidate: Q8_0 rows against q8_1 images
// Block (row tile, entry tile): 128 rows of the matrix against up to 64 entries, the columns in steps of 128 (four Q8_0
// blocks, an image's block of four).  8 warps of products (16 rows by the 64 entries each: a Q8_0 block a product of
// 16 x 8 x 32, its sums scaled by the row's and the entry's block scales as MMQ does: the same sums in the same order)
// and 4 that meanwhile bring the next step's codes into the other of two tile pairs (the rows' codes realigned: a
// block's 32 start two bytes into its 34; the image is MMQ's: a step's blocks of every row together), one barrier a
// step.
constexpr int kQRows = 128, kQEnt = 64;
constexpr int kQSmem = 2 * (kQRows + kQEnt) * kPitch + 2 * (kQRows + kQEnt) * 16;

__global__ void __launch_bounds__(384) q8_gemm_kernel(const uint8_t* __restrict__ w, size_t row_bytes, int w_rows, int K,
                                                      const uint8_t* __restrict__ xq, int T, float* __restrict__ y,
                                                      size_t ld) {
    extern __shared__ __align__(16) unsigned char smem[];
    int8_t* sw = reinterpret_cast<int8_t*>(smem);
    int8_t* sx = sw + 2 * kQRows * kPitch;
    float* dw = reinterpret_cast<float*>(sx + 2 * kQEnt * kPitch);
    float* dx = dw + 2 * kQRows * 4;
    const int r0 = blockIdx.x * kQRows, e0 = blockIdx.y * kQEnt, n = min(kQEnt, T - e0);
    const int i = threadIdx.x, wp = i >> 5, lane = i & 31;
    const bool stager = wp >= 8;
    const int su = i - 256;
    auto stage = [&](int s) {
        const int buf = s & 1;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int u = su + 128 * j, row = u >> 1, hf = u & 1;
            const int rr = min(r0 + row, w_rows - 1);
            const unsigned long long* p =
                reinterpret_cast<const unsigned long long*>(w + (size_t) rr * row_bytes + (size_t) s * 136) + 8 * hf;
            unsigned long long v[9], o[8];
#pragma unroll
            for (int q = 0; q < 9; ++q) v[q] = p[q];
            unsigned short h0, h1;
            if (hf == 0) {
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    o[q] = (v[q] >> 16) | (v[q + 1] << 48);
                    o[4 + q] = (v[4 + q] >> 32) | (v[5 + q] << 32);
                }
                h0 = (unsigned short) v[0];
                h1 = (unsigned short) (v[4] >> 16);
            } else {
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    o[q] = (v[q] >> 48) | (v[q + 1] << 16);
                    o[4 + q] = v[5 + q];
                }
                h0 = (unsigned short) (v[0] >> 32);
                h1 = (unsigned short) (v[4] >> 48);
            }
            unsigned long long* d = reinterpret_cast<unsigned long long*>(sw + (buf * kQRows + row) * kPitch + 64 * hf);
#pragma unroll
            for (int q = 0; q < 8; ++q) d[q] = o[q];
            float* ds = dw + (buf * kQRows + row) * 4 + 2 * hf;
            ds[0] = __half2float(__ushort_as_half(h0));
            ds[1] = __half2float(__ushort_as_half(h1));
        }
        for (int c = su; c < kQEnt * 9; c += 128) {
            const int e = c / 9, ch = c % 9;
            const uint4 v = *reinterpret_cast<const uint4*>(xq + ((size_t) s * T + e0 + min(e, n - 1)) * 144 + 16 * ch);
            if (ch == 0) *reinterpret_cast<uint4*>(dx + (buf * kQEnt + e) * 4) = v;
            else *reinterpret_cast<uint4*>(sx + (buf * kQEnt + e) * kPitch + 16 * (ch - 1)) = v;
        }
    };
    float f[8][4];
#pragma unroll
    for (int q = 0; q < 8; ++q) f[q][0] = f[q][1] = f[q][2] = f[q][3] = 0.f;
    const int lr = lane & 7, li = lane >> 3, g = lane >> 2, t = lane & 3, ncp = (n + 15) >> 4;
    const int aoff = (16 * (wp & 7) + ((li & 1) << 3) + lr) * kPitch + (li >> 1) * 16;
    const int boff = (((li >> 1) << 3) + lr) * kPitch + (li & 1) * 16;
    if (stager) stage(0);
    __syncthreads();
    const int steps = K / 128;
    for (int s = 0; s < steps; ++s) {
        if (stager) {
            if (s + 1 < steps) stage(s + 1);
        } else {
            const int8_t* tw = sw + (s & 1) * kQRows * kPitch + aoff;
            const int8_t* tx = sx + (s & 1) * kQEnt * kPitch + boff;
            const float* dws = dw + ((s & 1) * kQRows + 16 * wp + g) * 4;
            const float* dxs = dx + (s & 1) * kQEnt * 4;
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                unsigned a[4];
                ldsm4(a, tw + 32 * ks);
                const float w0 = dws[ks], w1 = dws[32 + ks];
#pragma unroll
                for (int cp = 0; cp < 4; ++cp) {
                    if (cp >= ncp) break;
                    unsigned b[4];
                    ldsm4(b, tx + 16 * cp * kPitch + 32 * ks);
                    int p0[4] = {0, 0, 0, 0}, p1[4] = {0, 0, 0, 0};
                    mma32(p0, a, b[0], b[1]);
                    mma32(p1, a, b[2], b[3]);
                    const float* xs = dxs + (16 * cp + 2 * t) * 4 + ks;
                    const float x0 = xs[0], x1 = xs[4], x2 = xs[32], x3 = xs[36];
                    f[2 * cp][0] += (float) p0[0] * w0 * x0;
                    f[2 * cp][1] += (float) p0[1] * w0 * x1;
                    f[2 * cp][2] += (float) p0[2] * w1 * x0;
                    f[2 * cp][3] += (float) p0[3] * w1 * x1;
                    f[2 * cp + 1][0] += (float) p1[0] * w0 * x2;
                    f[2 * cp + 1][1] += (float) p1[1] * w0 * x3;
                    f[2 * cp + 1][2] += (float) p1[2] * w1 * x2;
                    f[2 * cp + 1][3] += (float) p1[3] * w1 * x3;
                }
            }
        }
        __syncthreads();
    }
    if (stager) return;
#pragma unroll
    for (int q = 0; q < 8; ++q)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int e = 8 * q + 2 * t + j;
            if (e >= n) continue;
            float* o = y + (size_t) (e0 + e) * ld + r0 + 16 * wp + g;
            if (r0 + 16 * wp + g < w_rows) o[0] = f[q][j];
            if (r0 + 16 * wp + g + 8 < w_rows) o[8] = f[q][2 + j];
        }
}

/// y[e][r] (row pitch ld) = W x for a Q8_0 matrix of w_rows rows of w_cols (a multiple of 128) against MMQ's image of
/// T rows (MMQ::quantize's for Q8_0): the sums MMQ's run_dense makes, scaled as it scales them.  False: not launched.
inline bool q8_gemm(const uint8_t* w, int w_rows, int w_cols, const void* xq, int T, float* y, size_t ld, cudaStream_t st) {
    if (w_cols % 128 != 0 || (uintptr_t) w % 8 != 0) return false;
    static thread_local int set_on = -1;   // (the kernel's shared memory past 48 KB, once a device)
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    if (set_on != dev) {
        if (cudaFuncSetAttribute(q8_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kQSmem) != cudaSuccess) return false;
        set_on = dev;
    }
    q8_gemm_kernel<<<dim3((unsigned) ((w_rows + kQRows - 1) / kQRows), (unsigned) ((T + kQEnt - 1) / kQEnt)), 384, kQSmem,
                     st>>>(w, (size_t) (w_cols / 32) * 34, w_rows, w_cols, (const uint8_t*) xq, T, y, ld);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace ik_dense
