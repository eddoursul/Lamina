// src/kernels/mmvq_jobs_parity.cpp - native_mmvq_jobs (several Q8_0 products in one launch) against a native_mmvq
// call a job: bitwise, at 1, 2, 3 and 8 columns, jobs of different row counts and of two activation images (random
// codes with finite scales, normal activations); native_mmvq_il_jobs the same from the images' interleaved copies,
// whose parts lie where native_q8_1_il_view says; and that both decline what they have no kernel for.
//
//     mmvq_jobs_parity
#include "lamina/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}
uint16_t half_bits(float f) {   // a normal value's half (no rounding cases among those used)
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t) (((u >> 16) & 0x8000) | ((((u >> 23) & 0xff) - 112) << 10) | ((u >> 13) & 0x3ff));
}

}  // namespace

int main() {
    using namespace lamina::kernels;
    const int n_in = 2304, n_outs[5] = {2304, 2304, 256, 1024, 1024}, MAXC = 8;
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    std::mt19937_64 rng(8);
    const size_t row = (size_t) n_in / 32 * 34;
    uint8_t* w[5];
    for (int j = 0; j < 5; ++j) {
        std::vector<uint8_t> h(row * (size_t) n_outs[j] + 64);
        for (auto& b : h) b = (uint8_t) rng();
        for (size_t blk = 0; blk < row * (size_t) n_outs[j] / 34; ++blk) {
            const uint16_t d = half_bits((rng() & 1 ? 1.0f : -1.0f) * (0.001f + 0.0005f * (float) (rng() % 32)));
            std::memcpy(&h[blk * 34], &d, 2);
        }
        ck(cudaMalloc(&w[j], h.size()), "alloc");
        ck(cudaMemcpy(w[j], h.data(), h.size(), cudaMemcpyHostToDevice), "upload");
    }
    void* xq[2];
    for (int i = 0; i < 2; ++i) {   // two images of MAXC columns
        std::vector<float> x((size_t) MAXC * n_in);
        std::mt19937 r2(20 + i);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : x) v = nd(r2);
        float* dx;
        ck(cudaMalloc(&dx, x.size() * 4), "alloc");
        ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "upload");
        ck(cudaMalloc(&xq[i], (size_t) MAXC * (n_in / 32) * 36), "alloc");
        native_quantize_q8_1(dx, xq[i], n_in, MAXC, s);
    }
    float *ya[5], *yb[5];
    for (int j = 0; j < 5; ++j) {
        ck(cudaMalloc(&ya[j], (size_t) MAXC * n_outs[j] * 4), "alloc");
        ck(cudaMalloc(&yb[j], (size_t) MAXC * n_outs[j] * 4), "alloc");
    }
    long long checked = 0, diff = 0;
    int bad = 0;
    const int nb = n_in / 32;
    std::vector<uint8_t> img[2];
    for (int i = 0; i < 2; ++i) {
        img[i].resize((size_t) MAXC * nb * 36);
        ck(cudaStreamSynchronize(s), "quantize");
        ck(cudaMemcpy(img[i].data(), xq[i], img[i].size(), cudaMemcpyDeviceToHost), "down");
    }
    for (const int T : {2, 3, 8}) {   // the interleaved copies: their parts, then the products
        void* xil[2];
        long long dv = 0;
        for (int i = 0; i < 2; ++i) {
            const size_t bytes = native_q8_1_il_bytes(n_in, T);
            ck(cudaMalloc(&xil[i], bytes), "alloc");
            ck(cudaMemset(xil[i], 0, bytes), "memset");
            native_q8_1_il_from(xq[i], xil[i], n_in, T, s);
            ck(cudaStreamSynchronize(s), "copy");
            std::vector<uint8_t> h(bytes);
            ck(cudaMemcpy(h.data(), xil[i], bytes, cudaMemcpyDeviceToHost), "down");
            const Q81IlView v = native_q8_1_il_view(xil[i], n_in, T);
            const int cp = native_q8_1_il_cp(T);
            const int* bm = (const int*) (h.data() + ((const uint8_t*) v.bm - (const uint8_t*) xil[i]));
            const int* pm = (const int*) (h.data() + ((const uint8_t*) v.pm - (const uint8_t*) xil[i]));
            const float* d = (const float*) (h.data() + ((const uint8_t*) v.d - (const uint8_t*) xil[i]));
            for (int c = 0; c < T; ++c)
                for (int b = 0; b < nb; ++b) {
                    const uint8_t* blk = &img[i][((size_t) c * nb + b) * 36];
                    for (int p = 0; p < 8; ++p) {
                        int q;
                        std::memcpy(&q, blk + 4 + 4 * p, 4);
                        dv += bm[(size_t) (8 * b + p) * cp + c] != q;
                        dv += pm[(size_t) (p * nb + b) * cp + c] != q;
                    }
                    uint16_t hb;
                    std::memcpy(&hb, blk, 2);
                    dv += half_bits(d[(size_t) b * cp + c]) != hb;
                }
        }
        MmvqJob jobs[5];
        for (int j = 0; j < 5; ++j) {
            jobs[j] = MmvqJob{w[j], nullptr, yb[j], n_outs[j], xil[j < 3 ? 0 : 1]};
            ck(cudaMemset(ya[j], 0xff, (size_t) MAXC * n_outs[j] * 4), "memset");
            ck(cudaMemset(yb[j], 0xee, (size_t) MAXC * n_outs[j] * 4), "memset");
            native_mmvq(8, w[j], xq[j < 3 ? 0 : 1], ya[j], n_in, n_outs[j], T, s);
        }
        if (!native_mmvq_il_jobs(8, jobs, 5, n_in, T, s)) {
            std::printf("%d interleaved columns: declined\n", T);
            ++bad;
            continue;
        }
        ck(cudaStreamSynchronize(s), "run");
        long long d = 0;
        for (int j = 0; j < 5; ++j) {
            std::vector<uint32_t> a((size_t) MAXC * n_outs[j]), b(a.size());
            ck(cudaMemcpy(a.data(), ya[j], a.size() * 4, cudaMemcpyDeviceToHost), "down");
            ck(cudaMemcpy(b.data(), yb[j], b.size() * 4, cudaMemcpyDeviceToHost), "down");
            for (size_t i = 0; i < (size_t) T * n_outs[j]; ++i) d += a[i] != b[i];
            for (size_t i = (size_t) T * n_outs[j]; i < b.size(); ++i) d += b[i] != 0xeeeeeeeeu;
            checked += (long long) T * n_outs[j];
        }
        std::printf("%d interleaved columns, 5 products in a launch: %lld values differ from a launch each; the copy's "
                    "parts: %lld differ from the image\n", T, d, dv);
        diff += d + dv;
        for (int i = 0; i < 2; ++i) ck(cudaFree(xil[i]), "free");
    }
    {   // declined: one column, a K-quant, a job without its copy
        MmvqJob one{w[0], xq[0], yb[0], n_outs[0], xq[0]}, none{w[0], xq[0], yb[0], n_outs[0]};
        const bool c1 = native_mmvq_il_jobs(8, &one, 1, n_in, 1, s), k = native_mmvq_il_jobs(12, &one, 1, n_in, 2, s);
        const bool nc = native_mmvq_il_jobs(8, &none, 1, n_in, 2, s);
        std::printf("declined, interleaved: one column %s, a Q4_K matrix %s, no copy %s\n", c1 ? "NO" : "yes",
                    k ? "NO" : "yes", nc ? "NO" : "yes");
        bad += c1 || k || nc;
    }
    for (const int T : {1, 2, 3, 8}) {
        MmvqJob jobs[5];
        for (int j = 0; j < 5; ++j) {
            jobs[j] = MmvqJob{w[j], xq[j < 3 ? 0 : 1], yb[j], n_outs[j]};
            ck(cudaMemset(ya[j], 0xff, (size_t) MAXC * n_outs[j] * 4), "memset");
            ck(cudaMemset(yb[j], 0xee, (size_t) MAXC * n_outs[j] * 4), "memset");
            native_mmvq(8, w[j], xq[j < 3 ? 0 : 1], ya[j], n_in, n_outs[j], T, s);
        }
        if (!native_mmvq_jobs(8, jobs, 5, n_in, T, s)) {
            std::printf("%d columns: declined\n", T);
            ++bad;
            continue;
        }
        ck(cudaStreamSynchronize(s), "run");
        long long d = 0;
        for (int j = 0; j < 5; ++j) {
            std::vector<uint32_t> a((size_t) MAXC * n_outs[j]), b(a.size());
            ck(cudaMemcpy(a.data(), ya[j], a.size() * 4, cudaMemcpyDeviceToHost), "down");
            ck(cudaMemcpy(b.data(), yb[j], b.size() * 4, cudaMemcpyDeviceToHost), "down");
            for (size_t i = 0; i < (size_t) T * n_outs[j]; ++i) d += a[i] != b[i];
            for (size_t i = (size_t) T * n_outs[j]; i < b.size(); ++i) d += b[i] != 0xeeeeeeeeu;   // (nothing past the columns)
            checked += (long long) T * n_outs[j];
        }
        std::printf("%d columns, 5 products in a launch: %lld values differ from a launch each\n", T, d);
        diff += d;
    }
    {   // declined: a K-quant, more jobs than the table holds, a short row
        MmvqJob one{w[0], xq[0], yb[0], n_outs[0]}, many[kMmvqJobs + 1];
        for (auto& m : many) m = one;
        const bool k = native_mmvq_jobs(12, &one, 1, n_in, 1, s), m = native_mmvq_jobs(8, many, kMmvqJobs + 1, n_in, 1, s);
        const bool sh = native_mmvq_jobs(8, &one, 1, 512, 1, s);
        std::printf("declined: a Q4_K matrix %s, %d jobs %s, a row of 16 blocks %s\n", k ? "NO" : "yes", kMmvqJobs + 1,
                    m ? "NO" : "yes", sh ? "NO" : "yes");
        bad += k || m || sh;
    }
    std::printf("mmvq_jobs_parity: %s (%lld values)\n", diff == 0 && bad == 0 ? "PASS" : "FAIL", checked);
    return diff == 0 && bad == 0 ? 0 : 1;
}
