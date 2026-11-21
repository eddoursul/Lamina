// include/lamina/kernels/cpu/native_expert.hpp - plan v0.3 P6: one routed expert in its GGUF form on the CPU.
//
// The IQ2_XS / IQ3_XXS model files keep their experts in i-quant formats (IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
// IQ3_S gate/up; Q2_0 or IQ4_NL down) whose values cannot be re-expressed in the Q2_0 pack form.  A native expert
// blob is the three GGUF slices back to back, [gate rows | up rows | down rows], and the arithmetic is ggml-cpu's
// own (`ggml_get_type_traits_cpu`): the activation is quantized with the weight type's `vec_dot_type` and each row
// is one `vec_dot`, exactly what llama.cpp's CPU backend computes for the same tensor.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace lamina::kernels::cpu {

/// Bytes of the largest quantized activation any native layer uses (2560 values as Q8_K: 10 x 292).
inline constexpr size_t kNativeActBytes = 5120;
/// The widest native expert's hidden (n_ff): the pool's per-token rows (Hy4-flash's 1,536; Flash-Next's 640).
inline constexpr int kNativeFF = 2048;
/// Bytes of the largest quantized down activation (2,048 values as Q8_K: 8 x 292).
inline constexpr size_t kNativeHBytes = 2560;

/// One layer's native expert geometry.
struct NativeFmt {
    int gu_type = -1, d_type = -1;      ///< ggml types of gate/up and of down
    int gu_act = -1, d_act = -1;        ///< their vec_dot_type (the activation formats)
    int64_t n_embd = 0, n_ff = 0;
    size_t gu_row = 0, d_row = 0;       ///< bytes per weight row
    size_t up_off = 0, down_off = 0;    ///< inside the blob
    size_t bytes = 0;                   ///< the whole blob
    size_t act_bytes = 0, h_bytes = 0;  ///< quantized activation sizes (n_embd of gu_act, n_ff of d_act)
    int had_block = 0;                  ///< Hadamard-folded weights (hadamard.hpp): the rotation's block, else 0
    bool had_gu = false, had_d = false; ///< which of the layer's inputs are rotated
    const float* gu_signs = nullptr, *d_signs = nullptr;   ///< their sign vectors (null: identity)
    /// SwiGLU's clamp (Hy4-flash's routed experts: silu(min(g, L)) * clamp(u, -L, L), ggml_swiglu_clamp); 0: none
    float swiglu_limit = 0.f;
    /// 1: Inkling 2's activation in SwiGLU's place, beta tanh(g / beta) sigmoid(g) * lbeta tanh(u / lbeta)
    int act = 0;
    float act_beta = 0.f, act_lbeta = 0.f;
    /// > 0: two floats there in the blob, the scales of its gate/up and of its down rows' products (NVFP4's
    /// per-expert scales, as the GPU's NativeExpertLayout::scale_off)
    size_t scale_off = 0;
};

/// Whether this build has the ggml-cpu path.
bool native_experts_available() noexcept;
/// Fills `f` for a layer; false (with a reason) when ggml-cpu has no dot product for a type.
bool native_fmt(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, NativeFmt& f, std::string& err);

/// x (n_embd floats) -> the gate/up activation (act_bytes), rotated first where the layer is Hadamard-folded.
void native_quant_act(const NativeFmt& f, const float* x, void* dst);
/// h (n_ff floats) -> the down activation (h_bytes), rotated first where the layer is Hadamard-folded.
void native_quant_h(const NativeFmt& f, const float* h, void* dst);

/// From how many tokens native_gu_rows gives this gate/up type to a multi-token kernel (#152; ggml-cpu's per-token dot
/// below that).  1: a token's rows are the same alone and in any group.
int native_gu_mt_min(int gu_type);
/// --expert-math exact: the SwiGLU of the formats the GPU's exact kernels cover (the K-quant gate/up rows) takes
/// `expf_contract`, as those kernels do, instead of the C runtime's expf.  Set once, before any expert work.
void set_native_exact_swiglu(bool on) noexcept;
bool native_exact_swiglu() noexcept;
/// silu(g) * u through `expf_contract`: the exact SwiGLU (swiglu_expf_parity checks it against the GPU's).
float native_swiglu_exact(float g, float u) noexcept;
/// How this build's ggml rounds ggml_vec_dot_q5_K_q8_K's `summs += dmin * total` (C, so the compiler decides): 1 one
/// FMA, 0 a product then a sum, -1 neither.  Probed on two blocks whose quants are zero; --expert-math exact's GPU
/// dots follow it (`set_exact_q5_K_fma`).
int ggml_q5_K_mins_rounding();
/// ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t]) for rows r in [r0, r1), `nt` tokens.
void native_gu_rows(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* ff,
                    int r0, int r1);
/// out[t][r] = down_r . hq[t] for rows r in [r0, r1).
void native_down_rows(const NativeFmt& f, const uint8_t* blob, const void* const* hq, int nt, float* const* out,
                      int r0, int r1);

}  // namespace lamina::kernels::cpu
