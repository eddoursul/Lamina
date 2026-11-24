// include/lamina/vision/projector.hpp - a projector file (mmproj) as the encoders read it: the tower's shape, how a
// picture is sized for it, and where each tensor lies in the file.  Shared by the encoder on ggml's backends and the
// one on the engine's own kernels (encoder.hpp).
#pragma once

#include "lamina/vision/encoder.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_context;
struct gguf_context;

namespace lamina::vision {

enum class ProjectorKind { qwen3vl, hy4, inkling2v };

/// What a projector's header says, and how its pictures are sized.
struct ProjectorParams {
    ProjectorKind kind = ProjectorKind::qwen3vl;
    int n_embd = 0, n_head = 0, n_embd_head = 0, n_head_kv = 0, n_layer = 0, projection_dim = 0;
    int patch_size = 0, n_merge = 2;
    float eps = 0.0f;
    float mean[3] = {}, std[3] = {};
    // the picture's sizing (image.hpp)
    SizeRule size;
    Filter filter = Filter::bicubic;
    bool letterbox = false;
    std::array<uint8_t, 3> pad = {0, 0, 0};
    /// Inkling 2's processor: a picture is not resized (unless it passes the limit) but padded right and below to whole
    /// merged patches (planes_padded)
    bool pad_to_align = false;
    /// its tower's rotary positions: the base of the two dimensions' angles (mtmd's; no projector file names one)
    static constexpr float rope_theta = 10000.0f;

    /// a head's values
    int d_head() const { return n_embd_head > 0 ? n_embd_head : n_embd / n_head; }
};

/// A picture's planes as projector `p` takes them (its size rule, filter and padding: image.hpp) and their size tw x
/// th; std::runtime_error where the picture's size cannot be brought to its grid
std::vector<float> projector_planes(const Rgb& picture, const ProjectorParams& p, int& tw, int& th);

/// A tensor of the file: ggml's type id, its dimensions as ggml has them (ne[0] the fastest) and its bytes.
struct TensorInfo {
    int type = 0;
    int64_t ne[4] = {1, 1, 1, 1};
    uint64_t offset = 0;   ///< in the file
    size_t bytes = 0;
};

class Projector : public ProjectorParams {
public:
    Projector() = default;
    ~Projector();
    Projector(const Projector&) = delete;
    Projector& operator=(const Projector&) = delete;

    /// The header read and checked: one of the projector types above, a tower this code computes (throws
    /// std::runtime_error with what is wrong otherwise).  The caller's token limits go into the sizing.
    void open(const EncoderParams& params);

    const std::string& path() const { return path_; }
    /// the tensor's description; nullptr when the file has none of that name
    const TensorInfo* find(const std::string& name) const;
    /// its bytes from the file into `dst` (info.bytes of them); throws when the file ends inside it
    void read(const std::string& name, const TensorInfo& info, void* dst);
    /// the tensor as floats, from an F32, F16 or BF16 tensor (throws for another type, or a name that is not there)
    std::vector<float> floats(const std::string& name);
    /// ggml's description of every tensor (no data): the context gguf_init_from_file fills
    ggml_context* meta() const { return meta_; }
    gguf_context* gguf() const { return gguf_; }

private:
    std::string path_;
    gguf_context* gguf_ = nullptr;
    ggml_context* meta_ = nullptr;
    std::FILE* file_ = nullptr;
    std::unordered_map<std::string, TensorInfo> tensors_;
};

/// Qwen's learned position table at a grid of width x height patches: bilinear between the table's cells with the
/// corners aligned (ggml's upscale as mtmd asks for it), out[(y * width + x) * n_embd + c].  `table`: n_side x n_side
/// rows of n_embd.  `align_corners` false: Inkling 2's, ggml's upscale without that flag (a sample's place by its
/// pixel centre).
std::vector<float> qwen_position_table(const std::vector<float>& table, int n_embd, int n_side, int width, int height,
                                       bool align_corners = true);

/// Hy4's position table at out_w x out_h (mtmd's encode, HunyuanVL's): bilinear between the table's cells, a sample's
/// place by the scale (target + 0.1) / n_grid, pixel centres.  dst[(y * out_w + x) * n_embd + c].
std::vector<float> hy4_position_table(const std::vector<float>& table, int64_t n_embd, int n_grid, int out_w, int out_h);

}  // namespace lamina::vision
