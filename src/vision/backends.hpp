// src/vision/backends.hpp - the image encoder's two implementations (include/lamina/vision/encoder.hpp).
#pragma once

#include "lamina/vision/encoder.hpp"

namespace lamina::vision {

/// The CPU's encoder: a ggml graph on ggml's CPU backend (ggml_encoder.cpp).
std::unique_ptr<Encoder> open_ggml_encoder(const EncoderParams& params, std::string& err);
/// A GPU's: the engine's own CUDA kernels (cuda_encoder.cu), on the device params.device.
std::unique_ptr<Encoder> open_cuda_encoder(const EncoderParams& params, std::string& err);
/// what it takes of the device (encoder_gpu_bytes)
uint64_t cuda_encoder_bytes(const EncoderParams& params, std::string& err);

}   // namespace lamina::vision
