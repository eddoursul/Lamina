// include/lamina/kernels/gemm.hpp - the engine's matrix products of a prompt chunk: Y[T][N] = X[T][K] . W[N][K]^T, the
// weights' rows as the GGUF files hold them (K along a row), nothing transposed.
//
//   gemm16   X and W halves or BF16, the sums FP32 on the tensor cores (mma.m16n8k16), Y floats or halves
//   gemm32   X, W and Y floats, on the float units
//
// A block takes a tile of Y - rows of X against rows of W - and walks K in order through shared memory; a product
// whose tiles are fewer than the card's SMs can use splits K between blocks, each split's sums in the scratch (T x N
// floats a split), then added up in the splits' order: the same values at every run, another rounding than K in one
// pass.  The tiles and the splits follow the shape and the card (its architecture and SM count), so a card of
// another generation may round a product otherwise.  Rows of X or W that are not 16-byte aligned (a K or a row
// distance that is no multiple of 8 halves / 4 floats) take plain loads.
#pragma once

#include <cstddef>
#include <cstdint>

namespace lamina::kernels {

/// Device memory for the splits' partial sums, the caller's, free while the product runs on its stream (none: K is
/// never split, and a product of few rows leaves most of the card idle).
struct GemmScratch {
    void* p = nullptr;
    size_t bytes = 0;
};

/// y = alpha x . w^T + beta y (beta 0: y is not read).  ldx / ldy: values from one row of x / y to the next (0: K / N).
struct Gemm16 {
    const uint16_t* x = nullptr;   // [T][K]
    const uint16_t* w = nullptr;   // [N][K]
    void* y = nullptr;             // [T][N] floats, or halves (y_half)
    int64_t T = 0, N = 0, K = 0, ldx = 0, ldy = 0;
    float alpha = 1.0f, beta = 0.0f;
    bool bf16 = false;   // x and w are BF16 (else halves)
    bool y_half = false;
};
void gemm16(const Gemm16& p, const GemmScratch& scratch, void* stream);

/// What gemm16 runs on the current device for rows of these sizes that are 16-byte aligned, for tools and tests:
/// into `name` the instance as "MT.NT.WM.WN.BK.ST" (a warp's 16 MT x 16 NT sums, WM x WN warps a block, K in slabs of
/// BK through ST buffers), or "rows" for the kernel of one to eight outputs a row; returns the split count.
int gemm16_plan(int64_t T, int64_t N, int64_t K, size_t scratch_bytes, char* name, size_t name_size);

/// For tests: the number of tensor-core instances (and instance `form`'s name into `name`, where it is one), and a
/// product through instance `form` whatever its shape - rows that are 16-byte aligned, K in `splits` parts (1: whole;
/// more need the scratch), X's tiles the grid's inner index or W's.
int gemm16_forms(int form, char* name, size_t name_size);
void gemm16_form(int form, int splits, bool x_inner, const Gemm16& p, const GemmScratch& scratch, void* stream);

/// n products of one N and K in one launch: problem i is x[i] [rows[i]][K] . w[i] [N][K]^T into y[i] [rows[i]][N]
/// (floats); x, w and y are arrays of n pointers in device memory, rows is host memory.  K is not split.
void gemm16_grouped(const uint16_t* const* x, const uint16_t* const* w, float* const* y, const int* rows, int n,
                    int64_t N, int64_t K, bool bf16, void* stream);

struct Gemm32 {
    const float* x = nullptr;   // [T][K]
    const float* w = nullptr;   // [N][K]
    float* y = nullptr;         // [T][N]
    int64_t T = 0, N = 0, K = 0, ldx = 0, ldy = 0;
    float alpha = 1.0f, beta = 0.0f;
};
void gemm32(const Gemm32& p, const GemmScratch& scratch, void* stream);

/// gemm16_plan, gemm16_forms and gemm16_form for the float product: an instance's name is "sGM.GN.TM.TN.ST" (GM x GN
/// threads of TM x TN sums, K in slabs of 32 through ST buffers).
int gemm32_plan(int64_t T, int64_t N, int64_t K, size_t scratch_bytes, char* name, size_t name_size);
int gemm32_forms(int form, char* name, size_t name_size);
void gemm32_form(int form, int splits, bool x_inner, const Gemm32& p, const GemmScratch& scratch, void* stream);

/// The scratch a product of T x N outputs can use at most (bytes).
size_t gemm_scratch_bytes(int64_t T, int64_t N);

}   // namespace lamina::kernels
