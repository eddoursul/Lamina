// include/lamina/kernels/q8_1_il16.hpp - the q8_1 activation block with each 16 of its values interleaved, as the
// NVFP4 expert kernels read it (NativeExpertLayout::x_il): position 2 i of a half holds its value i, position 2 i + 1
// its value 8 + i - the order of the nibbles in an NVFP4 sub-block's words.
#pragma once

#include <cuda_runtime.h>

namespace lamina::kernels {

/// where value l (0..31) of a q8_1 block stands in the interleaved block
__device__ __forceinline__ int q8_1_il16(int l) { return (l & 16) | ((l & 7) << 1) | ((l >> 3) & 1); }

}  // namespace lamina::kernels
