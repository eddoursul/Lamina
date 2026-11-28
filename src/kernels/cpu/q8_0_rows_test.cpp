// src/kernels/cpu/q8_0_rows_test.cpp - quantize_q8_0_rows against ggml's ggml_quantize_chunk(Q8_0), byte for byte (CPU
// only): random rows of several scales, rows with zero blocks, tiny and huge values, f16's subnormal scales, on one
// thread and on several; and quantize_q8_0_row_split's codes and scales against the same blocks.
#include "lamina/kernels/cpu/q8_0_rows.hpp"

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int main() {
    std::mt19937 rng(7);
    int bad = 0, cases = 0;
    auto check = [&](const std::vector<float>& x, int64_t rows, int64_t cols, const char* what) {
        const size_t bytes = (size_t) rows * (size_t) (cols / 32) * 34;
        std::vector<uint8_t> want(bytes), got(bytes), got8(bytes);
        ggml_quantize_chunk(GGML_TYPE_Q8_0, x.data(), want.data(), 0, rows, cols, nullptr);
        lamina::kernels::cpu::quantize_q8_0_rows(x.data(), rows, cols, got.data(), 1);
        lamina::kernels::cpu::quantize_q8_0_rows(x.data(), rows, cols, got8.data(), 8);
        ++cases;
        if (got != want || got8 != want) {
            ++bad;
            size_t i = 0;
            while (i < bytes && got[i] == want[i]) ++i;
            std::printf("  %s: differs from ggml at byte %zu\n", what, i);
            return;
        }
        // the split layout: each row's codes and scales as the blocks hold them
        std::vector<int8_t> codes((size_t) cols);
        std::vector<uint16_t> scales((size_t) (cols / 32));
        for (int64_t r = 0; r < rows; ++r) {
            lamina::kernels::cpu::quantize_q8_0_row_split(x.data() + (size_t) r * (size_t) cols, cols, codes.data(),
                                                          scales.data());
            for (int64_t b = 0; b < cols / 32; ++b) {
                const uint8_t* blk = want.data() + ((size_t) r * (size_t) (cols / 32) + (size_t) b) * 34;
                if (std::memcmp(blk, &scales[(size_t) b], 2) != 0 || std::memcmp(blk + 2, codes.data() + b * 32, 32)) {
                    ++bad;
                    std::printf("  %s: the split row %lld differs at block %lld\n", what, (long long) r, (long long) b);
                    return;
                }
            }
        }
    };
    for (const float scale : {1e-6f, 1e-3f, 0.05f, 1.0f, 30.0f, 3e4f}) {
        std::normal_distribution<float> n(0.0f, scale);
        const int64_t rows = 37, cols = 256;
        std::vector<float> x((size_t) (rows * cols));
        for (float& v : x) v = n(rng);
        for (int64_t i = 0; i < 32; ++i) x[(size_t) (5 * cols + i)] = 0.0f;   // an all-zero block
        x[(size_t) (7 * cols + 3)] = 65504.0f * 127.0f;                       // a scale at f16's top
        x[(size_t) (9 * cols + 1)] = -1e-30f;                                 // a tiny one beside others
        char what[64];
        std::snprintf(what, sizeof what, "normal rows, scale %g", (double) scale);
        check(x, rows, cols, what);
    }
    {   // scales in f16's subnormal range, and halfway values of the codes
        const int64_t rows = 8, cols = 64;
        std::vector<float> x((size_t) (rows * cols));
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        for (int64_t i = 0; i < rows * cols; ++i) x[(size_t) i] = u(rng) * (i < 32 * 4 ? 3e-6f : 1.0f);
        for (int64_t j = 0; j < 32; ++j) x[(size_t) (6 * cols + j)] = (float) (j - 16) * (127.0f / 16.0f) + 0.5f;
        check(x, rows, cols, "subnormal scales, halfway codes");
    }
    std::printf("q8_0_rows_test: %d of %d cases %s\n", cases - bad, cases, bad ? "FAILED" : "OK");
    return bad ? 1 : 0;
}
