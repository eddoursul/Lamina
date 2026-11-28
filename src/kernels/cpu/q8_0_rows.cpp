// src/kernels/cpu/q8_0_rows.cpp - see q8_0_rows.hpp
#include "lamina/kernels/cpu/q8_0_rows.hpp"

#include "lamina/kernels/f16_bits.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>
#include <vector>

namespace lamina::kernels::cpu {
namespace {

/// one block of 32 values: its f16 scale and its codes (quantize_row_q8_0_ref's arithmetic and order)
inline void q8_0_block(const float* x, uint16_t& d16, int8_t* q) {
    float amax = 0.0f;
    for (int j = 0; j < 32; ++j) {   // (ggml's MAX: a NaN value makes amax NaN, as there)
        const float a = std::fabs(x[j]);
        amax = amax > a ? amax : a;
    }
    const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
    d16 = lamina::kernels::f16_from_f32(d);
    for (int j = 0; j < 32; ++j) q[j] = (int8_t) std::roundf(x[j] * id);
}

void rows(const float* x, int64_t r0, int64_t r1, int64_t cols, uint8_t* dst) {
    const int64_t nb = cols / 32;
    for (int64_t r = r0; r < r1; ++r)
        for (int64_t b = 0; b < nb; ++b) {
            uint8_t* blk = dst + ((size_t) r * (size_t) nb + (size_t) b) * 34;
            uint16_t d16 = 0;
            q8_0_block(x + (size_t) r * (size_t) cols + (size_t) b * 32, d16, reinterpret_cast<int8_t*>(blk + 2));
            std::memcpy(blk, &d16, 2);
        }
}

}   // namespace

void quantize_q8_0_rows(const float* x, int64_t n_rows, int64_t cols, uint8_t* dst, int threads) {
    const int parts = (int) std::max<int64_t>(1, std::min<int64_t>(std::max(threads, 1), n_rows));
    if (parts == 1) {
        rows(x, 0, n_rows, cols, dst);
        return;
    }
    std::vector<std::thread> pool;
    for (int i = 0; i < parts; ++i)
        pool.emplace_back([=] { rows(x, n_rows * i / parts, n_rows * (i + 1) / parts, cols, dst); });
    for (std::thread& t : pool) t.join();
}

void quantize_q8_0_row_split(const float* x, int64_t n, int8_t* codes, uint16_t* scales) {
    for (int64_t b = 0; b < n / 32; ++b) q8_0_block(x + b * 32, scales[b], codes + b * 32);
}

}   // namespace lamina::kernels::cpu
