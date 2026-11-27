// include/lamina/kernels/native_mma.hpp - the verify window's dense projections of 2-8 tokens on the int8 tensor
// cores (src/kernels/cuda/native_mma.cu): a CTA's 16-row tiles of the GGUF blocks (Q6_K, Q5_K, Q4_K, IQ4_XS; Q8_0 in
// several products a launch, native_mma_q8_0) copied into shared memory a stage ahead and decoded once into mma's A,
// the window's tokens its N.
#pragma once

namespace lamina::kernels {

// y = W x for `ncols` (2-8) columns of n_in quantized by native_quantize_q8_1_il (x_il, its interleaved copy), where a
// table measured on the RTX 3090 puts that format, row count and column count on the tensor cores (each format from a
// column count by its rows); false (nothing launched): the dp4a kernels take it (native_mmvq_il). A column's values
// do not depend on ncols; they are native_mmvq's to float rounding (the integer sums exact, the scales applied per q8_1
// block in another order). Weights: GGUF blocks, Q6_K's rows at 4-byte and the others' at 16-byte multiples (else
// false); y: column j at y + j * n_out.
bool native_mma(int ggml_type, const void* weights, const void* x_il, float* y, int n_in, int n_out, int ncols,
                void* stream);

// Q8_0 matrices of one n_in by 2-8 columns on the int8 tensor cores, up to five a launch: a job's x_il its columns'
// interleaved q8_1 copy, y column j at y + j * n_out. For a caller that measured it on its shapes (native_mma keeps
// Q8_0 on the dp4a kernels): on an RTX 3090 a 2,304 x 2,304 product of 6 / 8 columns takes 10.9 / 10.7 us where
// native_mmvq_il takes 13.1 / 13.8, 2,304 -> 1,024 6.5 / 6.0 against 8.2 / 9.2 (an RTX 5070 Ti 8.5 against 9.5 /
// 12.0, 4.9 / 5.0 against 5.4 / 7.1), and under 6 columns the dp4a kernels are as fast. A column's values do not
// depend on ncols; they are native_mmvq's to float rounding. False (nothing launched): n_in not a multiple of 768, a
// matrix not 16-byte aligned, or more than five jobs.
struct MmaJob {
    const void* weights;
    const void* x_il;
    float* y;
    int n_out;
};
bool native_mma_q8_0(const MmaJob* jobs, int n_jobs, int n_in, int ncols, void* stream);

}   // namespace lamina::kernels
