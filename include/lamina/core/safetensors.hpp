// include/lamina/core/safetensors.hpp - a safetensors file's tensors (the checkpoints' drafters: Hy4-flash's DFly2
// shard, Inkling 2's mtp.safetensors).  Host code; the JSON library stays in src/core/safetensors.cpp, out of nvcc.
#pragma once

#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace lamina {

/// A safetensors file's tensors: the header read at open, a tensor's data on request
class SafeTensors {
public:
    struct Tensor {
        std::string dtype;            // "BF16", "F32", ...
        std::vector<int64_t> shape;   // outermost first
        uint64_t begin = 0, end = 0;
    };
    bool open(const std::string& path, std::string& err);
    const Tensor* find(const std::string& name) const;
    const std::map<std::string, Tensor>& tensors() const { return entries_; }
    /// tensor `name` widened to floats (BF16 or F32), its shape outermost first; false when missing or of another type
    bool read(const std::string& name, std::vector<float>& out, std::vector<int64_t>& shape);
    /// tensor `name`'s bytes as the file has them; false when missing
    bool bytes(const std::string& name, std::vector<uint8_t>& out);

private:
    std::map<std::string, Tensor> entries_;
    std::ifstream in_;
    uint64_t data_ = 0;
};

}   // namespace lamina
