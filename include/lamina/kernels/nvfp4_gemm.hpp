// include/lamina/kernels/nvfp4_gemm.hpp - NVFP4 matrices against rows of halves on the FP16 tensor cores: a prompt
// chunk's routed experts on a card without FP4 tensor cores.  The weights are decoded into FP16 tiles in shared memory
// (exact: a code times its sub-block's scale has six significant bits at most), the activations are halves (not
// q8_1: 11 bits a value where q8_1 keeps the block's 8), the sums FP32 (mma.m16n8k16).
#pragma once

#include <cstddef>
#include <cstdint>

namespace lamina::kernels {

constexpr int kNvfp4GemmEntries = 64;   // entries a tile takes
constexpr int kNvfp4GemmRows = 128;     // weight rows a block takes

/// Rows of floats as the halves the product reads: dst row i = src row map[i] (map null: row i), `cols` halves each
/// (a multiple of 16), saturated to the half's range, each 16 interleaved (position 2 i: value i, 2 i + 1: value
/// 8 + i - the order an NVFP4 sub-block's words hold its codes in).
void nvfp4_gemm_halves(const float* src, const int32_t* map, void* dst, int64_t rows, int cols, void* stream);

/// y[ids[e]][r] = W_g[r] . x[e] for every entry e a tile names: group g's [w_rows][w_cols] NVFP4 matrix lies at
/// ptr[g] + off, its rows `row_bytes` apart (GGUF's blocks); its entries are bounds[g] .. bounds[g + 1] of x
/// ([entries][w_cols] halves from nvfp4_gemm_halves); tiles[t] = g << 16 | k stands for the group's entries 64 k ..
/// 64 k + 63 - a caller lists a tile for each started 64 entries of a group, none for a group it leaves out.  ids
/// null: y's row is the entry.  `ld`: floats from one row of y to the next.  ptr, bounds and tiles on the device.
struct Nvfp4Gemm {
    const unsigned long long* ptr = nullptr;
    size_t off = 0, row_bytes = 0;
    int w_rows = 0, w_cols = 0;
    const int32_t* bounds = nullptr;
    const int32_t* tiles = nullptr;
    int n_tiles = 0;
    const void* x = nullptr;
    const int32_t* ids = nullptr;
    float* y = nullptr;
    int64_t ld = 0;
};
void nvfp4_gemm(const Nvfp4Gemm& p, void* stream);

/// The matrices nvfp4_gemm takes: rows in the blocks' 128, columns in NVFP4's blocks of 64
constexpr bool nvfp4_gemm_fits(int64_t w_rows, int64_t w_cols) {
    return w_rows > 0 && w_cols > 0 && w_rows % kNvfp4GemmRows == 0 && w_cols % 64 == 0;
}

}   // namespace lamina::kernels
