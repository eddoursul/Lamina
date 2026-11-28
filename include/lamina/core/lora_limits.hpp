// include/lamina/core/lora_limits.hpp - the LoRA adapters' limits the qwen35 path's kernels hold to (q36_lora.cuh),
// which the file check names (model_check.cpp's lora_file_problem): one place for both.
#pragma once

namespace lamina::core {

constexpr int kLoraGroup = 8;           ///< a pair's rank is padded to a multiple of it (the A stage's rows a block)
constexpr int kLoraMaxRank = 512;       ///< the ranks of a product's pairs summed, at most
constexpr int kLoraInputMultiple = 8;   ///< a product's inputs a multiple of it (the elements a lane reads at once)

}   // namespace lamina::core
