// src/inkling2/marks_parity.cu - a checkpoint inside an Inkling 2 prompt chunk against the state a chunk that ends at
// its row leaves, bitwise, on synthetic cells and rows: ring_at_kernel's ring against range_to_ring_kernel's of the
// chunk's first `r` rows (halves and q8_0 cells; rows near the chunk's start, where older cells come from the ring,
// around the window's and the ring's lengths, the chunk's end; a sequence's start, whose negative positions no one
// reads), and conv_at_kernel's convolution states against conv_state_kernel's after `r` rows (those before C - 1
// rows: the states from before the chunk).
#include "inkling2_kernels.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

#define CK(x)                                                                                                          \
    do {                                                                                                               \
        const cudaError_t e_ = (x);                                                                                    \
        if (e_ != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "%s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__);           \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

namespace {

std::mt19937 rng(20261128);

template <class T> T* dev(const std::vector<T>& h) {
    T* d = nullptr;
    CK(cudaMalloc(&d, h.size() * sizeof(T)));
    CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <class T> std::vector<T> host(const T* d, size_t n) {
    std::vector<T> h(n);
    CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}
std::vector<uint8_t> bytes_of(size_t n) {
    std::vector<uint8_t> v(n);
    for (uint8_t& b : v) b = (uint8_t) rng();
    return v;
}

/// A window visit's ring of `cap` cells before a chunk of T rows at p0, the chunk's range as the engine builds it, and
/// the ring at each row `r` both ways: the positions [p0 + r - cap, p0 + r) from 0 on must hold the same bytes.
int ring_case(int64_t cap, int win, int bytes, long long p0, int T) {
    const size_t side = (size_t) cap * bytes, range = (size_t) (T + win - 1) * bytes;
    const std::vector<uint8_t> ring_h = bytes_of(2 * side), rows_h = bytes_of(2 * (size_t) T * bytes);
    uint8_t* ring = dev(ring_h);
    uint8_t *rk = dev(bytes_of(range)), *rv = dev(bytes_of(range));   // (slots of negative positions: as they are)
    ik::Cells cells{ring, ring + side, 0, cap, 1, bytes == 2048 ? 0 : 1};
    ik::ring_to_range_kernel<<<dim3((unsigned) (win - 1), 2), 256>>>(cells, rk, rv, p0, win, bytes);
    CK(cudaMemcpy(rk + (size_t) (win - 1) * bytes, rows_h.data(), (size_t) T * bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(rv + (size_t) (win - 1) * bytes, rows_h.data() + (size_t) T * bytes, (size_t) T * bytes,
                  cudaMemcpyHostToDevice));
    uint8_t *want = nullptr, *got = nullptr;
    CK(cudaMalloc(&want, 2 * side));
    CK(cudaMalloc(&got, 2 * side));
    int bad = 0, rows = 0;
    for (const int r :
         {1, 2, 3, 31, 32, 33, 34, win - 1, win, win + 1, (int) cap - 1, (int) cap, (int) cap + 1, T / 2, T - 1, T}) {
        if (r < 1 || r > T) continue;
        ++rows;
        CK(cudaMemcpy(want, ring, 2 * side, cudaMemcpyDeviceToDevice));
        ik::Cells w{want, want + side, 0, cap, 1, cells.q8};
        ik::range_to_ring_kernel<<<dim3((unsigned) std::min<int64_t>(r, cap), 2), 256>>>(w, rk, rv, p0, r, win, bytes);
        ik::ring_at_kernel<<<dim3((unsigned) cap, 2), 256>>>(cells, rk, rv, got, got + side, p0, p0 + r, win, bytes);
        CK(cudaDeviceSynchronize());
        const std::vector<uint8_t> a = host(want, 2 * side), b = host(got, 2 * side);
        for (long long q = std::max<long long>(0, p0 + r - cap); q < p0 + r; ++q)
            for (int s = 0; s < 2; ++s) {
                const size_t at = (size_t) s * side + (size_t) (q % cap) * bytes;
                if (std::memcmp(a.data() + at, b.data() + at, (size_t) bytes) != 0) {
                    if (bad < 5)
                        std::printf("ring: %d-byte cells, p0 %lld, T %d, row %d: position %lld's %c differs\n", bytes,
                                    p0, T, r, q, s ? 'V' : 'K');
                    ++bad;
                }
            }
    }
    CK(cudaFree(ring));
    CK(cudaFree(rk));
    CK(cudaFree(rv));
    CK(cudaFree(want));
    CK(cudaFree(got));
    std::printf("ring: %d-byte cells, p0 %lld, T %d: %d rows %s\n", bytes, p0, T, rows, bad ? "DIFFER" : "the same");
    return bad;
}

/// A visit's four convolutions (K and V of KV channels, the attention's and the MoE's outputs of N) after r rows of a
/// chunk of T: conv_at_kernel's states against conv_state_kernel's on each convolution, from random states and rows.
int conv_case(int KV, int N, int C, int T) {
    const int per = 2 * KV + 2 * N;
    std::uniform_real_distribution<float> u(-4.f, 4.f);
    auto floats = [&](size_t n) {
        std::vector<float> v(n);
        for (float& x : v) x = u(rng);
        return v;
    };
    const std::vector<float> old_h = floats((size_t) (C - 1) * per);
    const std::vector<float> raw_h[4] = {floats((size_t) T * KV), floats((size_t) T * KV), floats((size_t) T * N),
                                         floats((size_t) T * N)};
    float* raw[4];
    for (int i = 0; i < 4; ++i) raw[i] = dev(raw_h[i]);
    float* out = nullptr;
    CK(cudaMalloc(&out, old_h.size() * 4));
    int bad = 0, rows = 0;
    for (const int r : {1, 2, 3, 4, 5, 7, 8, T / 2, T - 1, T}) {
        if (r < 1 || r > T) continue;
        ++rows;
        CK(cudaMemcpy(out, old_h.data(), old_h.size() * 4, cudaMemcpyHostToDevice));
        ik::conv_at_kernel<<<(unsigned) ((per + 255) / 256), 256>>>(out, raw[0], raw[1], raw[2], raw[3], KV, N, C, r);
        CK(cudaDeviceSynchronize());
        const std::vector<float> got = host(out, old_h.size());
        size_t before = 0;
        for (int i = 0; i < 4; ++i) {
            const int n = i < 2 ? KV : N;
            std::vector<float> st(old_h.begin() + (long) ((size_t) (C - 1) * before),
                                  old_h.begin() + (long) ((size_t) (C - 1) * (before + (size_t) n)));
            float* d = dev(st);
            ik::conv_state_kernel<<<(unsigned) ((n + 255) / 256), 256>>>(d, raw[i], n, r, C);
            const std::vector<float> want = host(d, st.size());
            CK(cudaFree(d));
            if (std::memcmp(want.data(), got.data() + (size_t) (C - 1) * before, want.size() * 4) != 0) {
                if (bad < 5) std::printf("conv: C %d, row %d of %d: convolution %d differs\n", C, r, T, i);
                ++bad;
            }
            before += (size_t) n;
        }
    }
    for (float* p : raw) CK(cudaFree(p));
    CK(cudaFree(out));
    std::printf("conv: KV %d, N %d, C %d, T %d: %d rows %s\n", KV, N, C, T, rows, bad ? "DIFFER" : "the same");
    return bad;
}

}   // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("no CUDA device: skipped\n");
        return 77;
    }
    int bad = 0;
    // Inkling 2 Mini's rings: a window of 1,024 cells and a forward's 8 rows in whole tiles of 32 (1,056)
    for (const int bytes : {2048, 1088})
        for (const auto& [p0, T] :
             {std::pair<long long, int>{0, 1024}, {700, 1024}, {5000, 17}, {5000, 1024}, {40000, 2048}, {100000, 4096}})
            bad += ring_case(1056, 1024, bytes, p0, T);
    bad += conv_case(1024, 2304, 4, 1024);   // Inkling 2 Mini's convolutions
    bad += conv_case(96, 200, 8, 37);        // the most taps a state holds, widths off the blocks
    std::printf("%s\n", bad ? "FAILED" : "all the same");
    return bad ? 1 : 0;
}
