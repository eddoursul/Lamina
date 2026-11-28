// include/lamina/kernels/cpu/q8_0_rows.hpp - rows of floats as Q8_0 blocks, as ggml's quantize_row_q8_0_ref writes them
// (ggml_quantize_chunk's): each 32 values' d = amax / 127 stored as f16 (round to nearest even), their codes
// round(x / d) through id = 1 / d (0 for an all-zero block).  Without ggml (a build without the native experts takes
// it too), the rows over `threads` threads: blocks are independent, so any split writes the same bytes.
#pragma once

#include <cstddef>
#include <cstdint>

namespace lamina::kernels::cpu {

/// rows x cols floats (cols a multiple of 32) -> rows x (cols / 32) blocks of 34 bytes at dst
void quantize_q8_0_rows(const float* x, int64_t rows, int64_t cols, uint8_t* dst, int threads = 1);

/// The same for one row of n values into separate codes and f16 scales (HcQ8's layout: n codes, n / 32 scales)
void quantize_q8_0_row_split(const float* x, int64_t n, int8_t* codes, uint16_t* scales);

}   // namespace lamina::kernels::cpu
