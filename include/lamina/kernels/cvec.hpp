// include/lamina/kernels/cvec.hpp - a control vector on the residual stream (core/control_vector.hpp loads it):
// llama.cpp's control vectors, applied where its qwen4exp graph calls build_cvec, after a layer's FFN write, with the
// projection mode of the cvec-hooks fork (github.com/Cudila/llama.cpp, branch cvec-hooks) beside its additive one.
//
// Each of the hc hyper-connection streams h of every token becomes
//
//     h + a - (h . u) u
//
// with the layer's two vectors [a, u]: the additive mode's vector in `a` (u zero), or the projection
// h - s (h . v) v of a unit direction v at strength s as u = sqrt(s) v (a zero).  Before it, the layer's pending FFN
// write R += (bo + partial) * 2 sigmoid(inj / hc), with the arithmetic of the read that would otherwise fold it in;
// after it, optionally, the next read's norm as the prompt path's gr_write_norm_rs computes it (the row scales and the
// BF16 image of the normalized rows, by its thread mapping and sums).
#pragma once

#include <cstdint>

namespace lamina::kernels {

struct CvecRows {
    float* R = nullptr;                ///< T tokens' stacks, token-major: hc streams of n_embd each
    int64_t T = 0;
    int64_t n_embd = 0, hc = 0;
    /// The layer's FFN write, applied first; null `bo`: R holds the layer's output already.
    const float* bo = nullptr;         ///< T rows of n_embd
    const uint16_t* partial = nullptr; ///< FP16, T rows of n_embd added to bo (the second GPU's sums), or null
    const float* inj = nullptr;        ///< T rows of `inj_ld`, the first hc of each used
    int64_t inj_ld = 0;
    /// The next read's norm (null `w_norm`: none): rs[t * hc + c] and xn16 = bf16(h * rs * w_norm[c]).
    const float* w_norm = nullptr;     ///< hc rows of n_embd
    float eps = 0.0f;
    float* rs = nullptr;
    uint16_t* xn16 = nullptr;
};

/// The kernel's geometry: n_embd 2560 (the hyper-connection kernels' own).
bool cvec_supported(int64_t n_embd);

/// A layer's vectors `cv` ([a, u], 2 x n_embd floats on this device) on `rows`.  Throws std::runtime_error when the
/// launch fails or the geometry is not supported.
void cvec_apply(const CvecRows& rows, const float* cv, void* stream);

}  // namespace lamina::kernels
