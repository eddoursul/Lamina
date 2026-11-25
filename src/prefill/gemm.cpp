// src/prefill/gemm.cu - see include/lamina/prefill/gemm.hpp.
#include "lamina/prefill/gemm.hpp"
#include "lamina/kernels/dequant_bf16.hpp"
#include "lamina/kernels/gemm.hpp"

#include <cstdio>
#include <cstdlib>

namespace lamina::prefill {

void Gemm::set_buffers(uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes) {
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
    ws_ = workspace;
    ws_bytes_ = ws_bytes;
}

void Gemm::bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                float beta) {
    lamina::kernels::Gemm16 p;
    p.x = X;
    p.w = W;
    p.y = Y;
    p.T = T;
    p.N = N;
    p.K = K;
    p.ldy = ldy;
    p.beta = beta;
    p.bf16 = true;
    lamina::kernels::gemm16(p, {ws_, ws_bytes_}, stream_);
}

void Gemm::f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
               float beta) {
    lamina::kernels::Gemm16 p;
    p.x = X;
    p.w = W;
    p.y = Y;
    p.T = T;
    p.N = N;
    p.K = K;
    p.ldy = ldy;
    p.beta = beta;
    lamina::kernels::gemm16(p, {ws_, ws_bytes_}, stream_);
}

void Gemm::f16_grouped(const uint16_t* const* X, const uint16_t* const* W, float* const* Y, const int* rows, int n,
                       int64_t N, int64_t K) {
    lamina::kernels::gemm16_grouped(X, W, Y, rows, n, N, K, false, stream_);
}

void Gemm::native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                  int64_t ldy, float beta) {
    if (N * K > scratch_elems_) {
        // Too large for the scratch at once: in row slices.
        const int64_t rows = scratch_elems_ / K;
        if (rows <= 0) { std::fprintf(stderr, "prefill gemm: scratch too small for K=%lld\n", (long long) K); std::exit(1); }
        if (ldy <= 0) ldy = N;
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = (N - r0 < rows) ? N - r0 : rows;
            lamina::kernels::dequant_f16(ggml_type, W_blocks, r0, n, K, scratch_, stream_);
            f16(X, scratch_, Y + r0, T, n, K, ldy, beta);
        }
        return;
    }
    lamina::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, scratch_, stream_);
    f16(X, scratch_, Y, T, N, K, ldy, beta);
}

}  // namespace lamina::prefill
