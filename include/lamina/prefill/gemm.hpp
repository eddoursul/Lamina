// include/lamina/prefill/gemm.hpp - plan v0.3 P5: the batched projections of prompt processing.
//
// Every projection of a chunk of T tokens is Y[T, N] = X[T, K] . W[N, K]^T with W row-major (the GGUF / pack layout)
// and FP32 outputs.  Weights are BF16 on the device - either already (the pack's BF16 tensors) or dequantized from
// their native GGUF blocks into a reusable scratch (`dequant_bf16`) right before the product - and activations are
// rounded to BF16, which is also what llama.cpp's batched CUDA path does.  The products are the engine's own
// tensor-core kernels (kernels/gemm.hpp): 16-bit operands, FP32 sums.
#pragma once

#include <cstddef>
#include <cstdint>

namespace lamina::prefill {

class Gemm {
public:
    /// The stream the products run on; set_buffers() before the first product.
    void init(void* stream) { stream_ = stream; }
    /// Caller-owned device buffers (e.g. lent expert-cache slots): the dequantization scratch (`scratch_elems` FP16
    /// elements: the largest weight dequantized at once; may be null for no native() calls) and the products' own
    /// (the partial sums of a product whose K is split: free between two products on the stream).
    void set_buffers(uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes);

    /// Y[T, N] (fp32, row stride ldy) = X[T, K] (bf16, row-major) . W[N, K]^T (bf16, row-major).  `beta` = 1 adds.
    void bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy = 0,
              float beta = 0.0f);

    /// Y = X . W^T with both in FP16 (bits).
    void f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy = 0,
             float beta = 0.0f);

    /// Y_i[rows[i], N] = X_i[rows[i], K] . W_i[N, K]^T (FP16 in, FP32 out) for n problems of one shape but their
    /// own row counts, in one launch; the pointer arrays are DEVICE memory, `rows` host memory.  K a multiple of 8,
    /// every row 16-byte aligned.
    void f16_grouped(const uint16_t* const* X, const uint16_t* const* W, float* const* Y, const int* rows, int n,
                     int64_t N, int64_t K);

    /// W given as native GGUF blocks of `ggml_type`, dequantized to FP16 in the scratch, X in FP16.
    void native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                int64_t ldy = 0, float beta = 0.0f);

    uint16_t* scratch() const { return scratch_; }
    int64_t scratch_elems() const { return scratch_elems_; }
    void* stream() const { return stream_; }

private:
    void* stream_ = nullptr;
    uint16_t* scratch_ = nullptr;
    int64_t scratch_elems_ = 0;
    void* ws_ = nullptr;
    size_t ws_bytes_ = 0;
};

}   // namespace lamina::prefill
