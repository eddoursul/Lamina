// include/lamina/core/gguf_loader.hpp - a model's GGUF tensors into a path's memory (the qwen35, Hy4-flash and
// Inkling 2 paths): a tensor found by name (a missing one ends the run), the ones a path wants read ahead in large
// pieces, a tensor's bytes on the device with a zeroed tail, its values as floats on the host or the device; and the
// parallel upload of many tensors' bytes through pinned staging (Flash-Next's dense matrices, the qwen35 path's).
#pragma once

#include "lamina/artifact/gguf_reader.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace lamina::core {

/// Zero bytes after each weight on the device: a kernel's k loop may read past a short last row (zero activations)
constexpr size_t kWeightPad = 4096;

/// A tensor's values as floats (F32 as it is, BF16 / F16 widened, Q8_0 scaled); false for another type
bool tensor_floats(const lamina::TensorInfo& t, const uint8_t* src, std::vector<float>& h);
/// `h` on the current device, in the run's memory (run_owned.hpp's dalloc)
const float* floats_on_device(const std::vector<float>& h);

/// One upload of upload_parallel: `bytes` from `src` (host memory or a mapping) to `dev` on the current device
struct Upload {
    void* dev;
    const uint8_t* src;
    uint64_t bytes;
};
/// The jobs' bytes into the current device's memory from up to 8 threads in 8 MiB pieces through pinned staging: a
/// thread copies a piece from the source (taking a mapping's page faults) into one half of its staging buffer and
/// sends it on its own stream while it fills the other half.  False with why in `err`.
bool upload_parallel(const std::vector<Upload>& jobs, std::string& err);

/// A model's tensors by name, for a path's load
class GgufLoader {
public:
    explicit GgufLoader(const lamina::GgufModel& model) : gm(model) {}
    GgufLoader(const GgufLoader&) = delete;
    GgufLoader& operator=(const GgufLoader&) = delete;

    const lamina::GgufModel& gm;

    /// Every tensor `want` takes, read in large pieces into host memory (read_slices on `threads` threads, past the
    /// OS file cache with `uncached`): a copy out of the mapping faults its pages in a few at a time.  Without the
    /// host memory for them, the mapping's (true); false when a read fails.
    bool read_ahead(const std::function<bool(const std::string&)>& want, int threads, bool uncached, std::string& e);
    /// The read-ahead freed (the mapping's from here)
    void drop_ahead();
    /// The tensor (its shard in *s): a missing one is said and ends the run (Fatal{1})
    const lamina::TensorInfo* find(const std::string& name, size_t* s = nullptr) const;
    bool has(const std::string& name) const { return gm.find(name) != nullptr; }
    /// Its bytes on the host: the read-ahead's, else the mapping's
    const uint8_t* host(const std::string& name, const lamina::TensorInfo** ti = nullptr) const;
    /// Its bytes on the current device with kWeightPad zero bytes after them (the run's memory)
    uint8_t* device_bytes(const std::string& name, const lamina::TensorInfo** ti = nullptr) const;
    /// Its values as floats (tensor_floats; another type is said and ends the run)
    std::vector<float> floats(const std::string& name) const;
    /// ... on the current device
    const float* f32(const std::string& name) const { return floats_on_device(floats(name)); }

private:
    std::map<const lamina::TensorInfo*, const uint8_t*> pre_;   // tensors read ahead
    std::unique_ptr<uint8_t[]> pre_buf_;
};

}   // namespace lamina::core
