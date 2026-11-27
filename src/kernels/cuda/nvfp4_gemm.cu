// src/kernels/cuda/nvfp4_gemm.cu - see include/lamina/kernels/nvfp4_gemm.hpp.
#include "lamina/kernels/fp4_bits.cuh"
#include "lamina/kernels/nvfp4_gemm.hpp"
#include "lamina/kernels/warp.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace lamina::kernels {
namespace {

constexpr int kPad = 72;   // a tile row in shared memory, halves (64 + 8: ldmatrix's eight rows on distinct banks)
constexpr int kSmem = 4096 + 2 * (kNvfp4GemmRows + kNvfp4GemmEntries) * kPad * 2;

/// A thread eight halves.  Grid ceil(rows * cols / 8 / 256).
__global__ void halves_kernel(const float* __restrict__ src, const int32_t* __restrict__ map, uint4* __restrict__ dst,
                              long long rows, int cols) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    const int per = cols / 8;
    if (i >= rows * per) return;
    const long long row = i / per;
    const int c = (int) (i % per), g = c >> 1, hf = c & 1;
    const float* s = src + (size_t) (map ? map[row] : row) * cols + 16 * g + 4 * hf;
    const float4 a = *reinterpret_cast<const float4*>(s), b = *reinterpret_cast<const float4*>(s + 8);
    auto h2 = [](float x, float y) {
        const __half2 v = __floats2half2_rn(fminf(fmaxf(x, -65504.f), 65504.f), fminf(fmaxf(y, -65504.f), 65504.f));
        return *reinterpret_cast<const unsigned*>(&v);
    };
    dst[i] = make_uint4(h2(a.x, b.x), h2(a.y, b.y), h2(a.z, b.z), h2(a.w, b.w));
}

/// The eight magnitudes of an NVFP4 sub-block (the doubled codes') times its halved scale as halves, by bytes: x the
/// low bytes of codes 0-3, y of codes 4-7, z and w the high bytes
__device__ __forceinline__ uint4 fp4_table(uint8_t d) {
    const float s = ue4m3_half(d);
    const float m[8] = {0.f, 1.f, 2.f, 3.f, 4.f, 6.f, 8.f, 12.f};
    uint32_t lo[2] = {0, 0}, hi[2] = {0, 0};
#pragma unroll
    for (int c = 0; c < 8; ++c) {
        const uint32_t h = __half_as_ushort(__float2half(m[c] * s));
        lo[c >> 2] |= (h & 0xFF) << (8 * (c & 3));
        hi[c >> 2] |= (h >> 8) << (8 * (c & 3));
    }
    return make_uint4(lo[0], lo[1], hi[0], hi[1]);
}

/// One NVFP4 sub-block's 16 values as halves, interleaved (o: 8 words): a word's half selects the magnitudes' bytes
/// (tb: fp4_table of the sub-block's scale; the sign bit cleared, which would replicate a byte's sign instead), the
/// values' sign bits set after: prmt over 0x80 bytes gives 0xff where the selector's top bit is set and 0x80 where
/// it is not - bit 0 tells them apart.
__device__ __forceinline__ void fp4_halves(uint32_t qa, uint32_t qb, uint4 tb, uint32_t (&o)[8]) {
#pragma unroll
    for (int wd = 0; wd < 2; ++wd) {
        const uint32_t qq = wd ? qb : qa;
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const uint32_t sel = hh ? qq >> 16 : qq, selc = sel & 0x7777u;
            const uint32_t lo4 = prmt_sel(tb.x, tb.y, selc);
            uint32_t hi4 = prmt_sel(tb.z, tb.w, selc);
            hi4 |= (prmt_sel(0x80808080u, 0x80808080u, sel) & 0x01010101u) << 7;
            o[4 * wd + 2 * hh] = prmt_sel(lo4, hi4, 0x5140u);
            o[4 * wd + 2 * hh + 1] = prmt_sel(lo4, hi4, 0x7362u);
        }
    }
}

/// Block (row tile, tile): 128 rows of a group's matrix against up to 64 of its entries, the columns in steps of 64.
/// 16 warps: 8 of products (16 rows by the 64 entries each: 32 sums a lane) and 8 that meanwhile decode the next
/// step's weights (a thread two sub-blocks of its row's block) and copy the entries' halves, into the other of two
/// tile pairs; one barrier a step.  Staged in turn by the same warps, with or without the raw blocks fetched ahead,
/// the block took 6% longer at 1,024 rows a chunk and 7% at 256 (RTX 3090); the card's own rate of these products
/// (17-20 M a ms) is two thirds of the time left.  Dynamic shared memory: kSmem.  Grid (w_rows / 128, tiles).
__global__ void __launch_bounds__(512) nvfp4_gemm_kernel(const unsigned long long* __restrict__ ptr, size_t off,
                                                         size_t row_bytes, int K, const int32_t* __restrict__ bounds,
                                                         const int32_t* __restrict__ tiles,
                                                         const __half* __restrict__ xh, const int32_t* __restrict__ ids,
                                                         float* __restrict__ y, size_t ld) {
    constexpr int BM = kNvfp4GemmRows, BN = kNvfp4GemmEntries;
    extern __shared__ __align__(16) unsigned char smem[];
    uint4* lut = reinterpret_cast<uint4*>(smem);
    __half* sw = reinterpret_cast<__half*>(smem + 4096);
    __half* sx = sw + 2 * BM * kPad;
    const int tile = tiles[blockIdx.y], place = tile >> 16, ct = tile & 0xFFFF;
    const int e0 = bounds[place] + ct * BN, n = min(BN, bounds[place + 1] - e0);
    if (n <= 0) return;
    const int r0 = blockIdx.x * BM, i = threadIdx.x, w = i >> 5, lane = i & 31;
    const bool stager = w >= 8;
    if (!stager) lut[i] = fp4_table((uint8_t) i);
    const int si = i & 255, row = si >> 1, part = si & 1, xe = si >> 2, xp = si & 3;
    const uint32_t* wrow =
        reinterpret_cast<const uint32_t*>((const uint8_t*) ptr[place] + off + (size_t) (r0 + row) * row_bytes);
    __builtin_assume(__isGlobal(wrow));
    const uint4* xrow = reinterpret_cast<const uint4*>(xh + (size_t) (e0 + min(xe, n - 1)) * K) + 2 * xp;
    auto stage = [&](int kt) {
        __half* tw = sw + (kt & 1) * BM * kPad;
        __half* tx = sx + (kt & 1) * BN * kPad;
        const uint32_t* blk = wrow + 9 * kt;   // a block: the four scales' word, eight words of codes
        const uint32_t dd = blk[0] >> (16 * part);
#pragma unroll
        for (int sbi = 0; sbi < 2; ++sbi) {
            const int sb = 2 * part + sbi;
            uint32_t o[8];
            fp4_halves(blk[1 + 2 * sb], blk[2 + 2 * sb], lut[(dd >> (8 * sbi)) & 0xFF], o);
            uint4* d = reinterpret_cast<uint4*>(tw + row * kPad + 16 * sb);
            d[0] = make_uint4(o[0], o[1], o[2], o[3]);
            d[1] = make_uint4(o[4], o[5], o[6], o[7]);
        }
        if (xe < n) {
            const uint4 a = xrow[8 * kt], b = xrow[8 * kt + 1];
            uint4* d = reinterpret_cast<uint4*>(tx + xe * kPad + 16 * xp);
            d[0] = a;
            d[1] = b;
        }
    };
    float c[8][4];
#pragma unroll
    for (int q = 0; q < 8; ++q) c[q][0] = c[q][1] = c[q][2] = c[q][3] = 0.f;
    const int lr = lane & 7, li = lane >> 3, ncp = (n + 15) >> 4;
    const int aoff = (16 * (w & 7) + ((li & 1) << 3) + lr) * kPad + (li >> 1) * 8;
    const int boff = (((li >> 1) << 3) + lr) * kPad + (li & 1) * 8;
    __syncthreads();   // the table
    if (stager) stage(0);
    __syncthreads();
    for (int kt = 0; kt < K / 64; ++kt) {
        if (stager) {
            if (kt + 1 < K / 64) stage(kt + 1);
        } else {
            const __half* arow = sw + (kt & 1) * BM * kPad + aoff;
            const __half* brow = sx + (kt & 1) * BN * kPad + boff;
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                unsigned a[4];
                ldsm4(a, arow + 16 * ks);
#pragma unroll
                for (int cp = 0; cp < 4; ++cp) {   // (a constant index: the sums stay in registers)
                    if (cp >= ncp) break;
                    unsigned b[4];
                    ldsm4(b, brow + 16 * cp * kPad + 16 * ks);
                    mma16816(c[2 * cp], a, b[0], b[1]);
                    mma16816(c[2 * cp + 1], a, b[2], b[3]);
                }
            }
        }
        __syncthreads();
    }
    if (stager) return;
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int q = 0; q < 8; ++q)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int e = 8 * q + 2 * t + j;
            if (e >= n) continue;
            float* o = y + (size_t) (ids ? ids[e0 + e] : e0 + e) * ld + r0 + 16 * w + g;
            o[0] = c[q][j];
            o[8] = c[q][2 + j];
        }
}

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

}   // namespace

void nvfp4_gemm_halves(const float* src, const int32_t* map, void* dst, int64_t rows, int cols, void* stream) {
    if (rows <= 0) return;
    if (cols <= 0 || cols % 16 != 0) {
        std::fprintf(stderr, "nvfp4_gemm_halves: %d columns, not a multiple of 16\n", cols);
        std::exit(1);
    }
    const long long n = rows * (cols / 8);
    halves_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(src, map, (uint4*) dst, rows, cols);
    check("nvfp4_gemm_halves");
}

void nvfp4_gemm(const Nvfp4Gemm& p, void* stream) {
    if (p.n_tiles <= 0) return;
    if (!nvfp4_gemm_fits(p.w_rows, p.w_cols) || p.row_bytes != (size_t) (p.w_cols / 64) * 36) {
        std::fprintf(stderr, "nvfp4_gemm: a matrix of %d rows of %d values (%zu bytes) is not its blocks' shape\n",
                     p.w_rows, p.w_cols, p.row_bytes);
        std::exit(1);
    }
    if (cudaFuncSetAttribute(nvfp4_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem) != cudaSuccess)
        check("nvfp4_gemm: its shared memory");
    nvfp4_gemm_kernel<<<dim3((unsigned) (p.w_rows / kNvfp4GemmRows), (unsigned) p.n_tiles), 512, kSmem,
                        (cudaStream_t) stream>>>(p.ptr, p.off, p.row_bytes, p.w_cols, p.bounds, p.tiles,
                                                 (const __half*) p.x, p.ids, p.y, (size_t) p.ld);
    check("nvfp4_gemm");
}

}   // namespace lamina::kernels
