// src/kernels/nvfp4_gemm_parity.cu - `nvfp4_gemm` (NVFP4 matrices against halves on the FP16 tensor cores) against
// the host's sums in doubles over the same halves: groups of 0 to 150 entries (tiles in part, a group left out), a
// row map, every scale code among the sub-blocks'.  The sums are FP32 in the tiles' order, so the test is a bound on
// the difference, and that rows no tile names stay as they were.
//
//     nvfp4_gemm_parity
#include "lamina/kernels/nvfp4_gemm.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
template <class T> T* dev(const std::vector<T>& h) {
    T* d = nullptr;
    check(cudaMalloc(&d, h.size() * sizeof(T) + 16), "alloc");
    check(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}
double half_value(uint16_t h) {
    const int e = (h >> 10) & 31, m = h & 1023;
    const double v = e == 0 ? std::ldexp((double) m, -24) : std::ldexp(1.0 + m / 1024.0, e - 15);
    return (h & 0x8000) ? -v : v;
}
double scale_half(uint8_t x) {   // ggml_ue4m3_to_fp32, halved
    const int e = (x >> 3) & 0xF, m = x & 7;
    if (x == 0x7F) return 0.0;
    return e ? std::ldexp(1.0 + m / 8.0, e - 8) : m * 0.0009765625;
}

}   // namespace

int main() {
    using namespace lamina::kernels;
    const int W_ROWS = 256, K = 192, G = 6;
    const size_t row_bytes = (size_t) (K / 64) * 36, off = 72, stride = off + (size_t) W_ROWS * row_bytes + 64;
    const int counts[G] = {0, 3, 64, 70, 150, 9};   // (the last group has entries and no tile: left out)
    std::mt19937_64 rng(4);
    std::vector<uint8_t> blobs((size_t) G * stride);
    for (auto& b : blobs) b = (uint8_t) rng();
    for (int g = 0; g < G; ++g)
        for (int r = 0; r < W_ROWS; ++r)
            for (int b = 0; b < K / 64; ++b)
                for (int i = 0; i < 4; ++i) {
                    const uint64_t u = rng();
                    blobs[(size_t) g * stride + off + (size_t) r * row_bytes + (size_t) b * 36 + i] =
                        u % 16 == 0   ? (uint8_t) (u >> 8 & 7)
                        : u % 64 == 1 ? 0x7F
                                      : (uint8_t) (((1 + (u >> 8) % 12) << 3) | (u >> 16 & 7));
                }
    std::vector<int32_t> bounds{0}, tiles;
    for (int g = 0; g < G; ++g) {
        bounds.push_back(bounds.back() + counts[g]);
        if (g < G - 1)
            for (int k = 0; k * kNvfp4GemmEntries < counts[g]; ++k) tiles.push_back(g << 16 | k);
    }
    const int n_ent = bounds.back(), n_src = 40;
    std::vector<float> x((size_t) n_src * K);
    {
        std::mt19937 r2(9);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : x) v = nd(r2);
        x[5] = 1e6f;   // (saturates)
        x[77] = -1e6f;
    }
    std::vector<int32_t> map((size_t) n_ent), ids((size_t) n_ent);
    for (int i = 0; i < n_ent; ++i) {
        map[(size_t) i] = (int32_t) (rng() % (uint64_t) n_src);
        ids[(size_t) i] = n_ent - 1 - i;
    }
    std::vector<unsigned long long> ptr((size_t) G);
    uint8_t* d_blobs = dev(blobs);
    for (int g = 0; g < G; ++g) ptr[(size_t) g] = (unsigned long long) (d_blobs + (size_t) g * stride);
    const float* d_x = dev(x);
    const int32_t *d_map = dev(map), *d_ids = dev(ids), *d_bounds = dev(bounds), *d_tiles = dev(tiles);
    const unsigned long long* d_ptr = dev(ptr);
    void* d_xh = nullptr;
    float* d_y = nullptr;
    check(cudaMalloc(&d_xh, (size_t) n_ent * K * 2), "alloc");
    check(cudaMalloc(&d_y, (size_t) n_ent * W_ROWS * 4), "alloc");
    cudaStream_t s;
    check(cudaStreamCreate(&s), "stream");
    int bad = 0;
    for (int pass = 0; pass < 2; ++pass) {   // with the row map of y, and without
        const std::vector<float> sentinel((size_t) n_ent * W_ROWS, -777.0f);
        check(cudaMemcpy(d_y, sentinel.data(), sentinel.size() * 4, cudaMemcpyHostToDevice), "y");
        nvfp4_gemm_halves(d_x, d_map, d_xh, n_ent, K, s);
        Nvfp4Gemm p;
        p.ptr = d_ptr;
        p.off = off;
        p.row_bytes = row_bytes;
        p.w_rows = W_ROWS;
        p.w_cols = K;
        p.bounds = d_bounds;
        p.tiles = d_tiles;
        p.n_tiles = (int) tiles.size();
        p.x = d_xh;
        p.ids = pass == 0 ? d_ids : nullptr;
        p.y = d_y;
        p.ld = W_ROWS;
        nvfp4_gemm(p, s);
        check(cudaStreamSynchronize(s), "run");
        std::vector<float> y((size_t) n_ent * W_ROWS);
        std::vector<uint16_t> xh((size_t) n_ent * K);
        check(cudaMemcpy(y.data(), d_y, y.size() * 4, cudaMemcpyDeviceToHost), "y");
        check(cudaMemcpy(xh.data(), d_xh, xh.size() * 2, cudaMemcpyDeviceToHost), "x");
        static const double kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
        double worst = 0, rms = 0;
        long long n_sum = 0, wrong = 0, touched = 0, halves_off = 0;
        for (int e = 0; e < n_ent; ++e)   // the halves: the source row's values, saturated, interleaved
            for (int k = 0; k < K; ++k) {
                const float v = x[(size_t) map[(size_t) e] * K + k];
                const double want = v > 65504.f ? 65504.0 : v < -65504.f ? -65504.0 : (double) v;
                const int i16 = k % 16, pos = (k / 16) * 16 + ((i16 & 7) << 1 | (i16 >> 3));
                const double got = half_value(xh[(size_t) e * K + pos]);
                halves_off += std::fabs(got - want) > std::fabs(want) * 0.0005 + 1e-7;
            }
        for (int g = 0; g < G; ++g)
            for (int e = bounds[(size_t) g]; e < bounds[(size_t) g + 1]; ++e) {
                const size_t yrow = (size_t) (pass == 0 ? ids[(size_t) e] : e) * W_ROWS;
                for (int r = 0; r < W_ROWS; ++r) {
                    const float got = y[yrow + r];
                    if (g == G - 1) {   // no tile: untouched (pass 1: its own rows; pass 0: the rows ids give it)
                        touched += got != -777.0f;
                        continue;
                    }
                    const uint8_t* row = blobs.data() + (size_t) g * stride + off + (size_t) r * row_bytes;
                    double sum = 0, mag = 0;
                    for (int k = 0; k < K; ++k) {
                        const uint8_t* blk = row + (k / 64) * 36;
                        const int sb = (k % 64) / 16, i16 = k % 16;
                        const uint8_t byte = blk[4 + 8 * sb + (i16 & 7)];
                        const double wv = kv[i16 < 8 ? byte & 15 : byte >> 4] * scale_half(blk[sb]);
                        const double xv =
                            half_value(xh[(size_t) e * K + (k / 16) * 16 + ((i16 & 7) << 1 | (i16 >> 3))]);
                        sum += wv * xv;
                        mag += std::fabs(wv * xv);
                    }
                    const double d = std::fabs((double) got - sum);
                    wrong += !(d <= mag * 2e-6 + 1e-6);
                    worst = std::max(worst, mag > 0 ? d / mag : d);
                    rms += sum * sum;
                    ++n_sum;
                }
            }
        std::printf(
            "%s: %lld sums (rms %.3g), the largest difference %.2g of a sum's magnitudes, %lld past the bound; %lld "
            "halves off; %lld values written where no tile stands\n",
            pass == 0 ? "with a row map" : "rows in place", n_sum, std::sqrt(rms / (double) n_sum), worst, wrong,
            halves_off, touched);
        bad += wrong != 0 || halves_off != 0 || touched != 0;
    }
    std::printf("nvfp4_gemm_parity: %s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
