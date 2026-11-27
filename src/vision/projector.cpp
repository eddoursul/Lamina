// src/vision/projector.cpp - see include/lamina/vision/projector.hpp.
#include "lamina/vision/projector.hpp"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace lamina::vision {
namespace {

int seek64(std::FILE* f, uint64_t at) {
#ifdef _WIN32
    return _fseeki64(f, (long long) at, SEEK_SET);
#else
    return fseeko(f, (off_t) at, SEEK_SET);
#endif
}

struct Keys {
    const gguf_context* g;
    int find(const char* key) const { return (int) gguf_find_key(g, key); }
    /// an integer key's value; `dflt` when the key is not there and not `required`
    int integer(const char* key, bool required, int dflt = 0) const {
        const int i = find(key);
        if (i < 0) {
            if (required) throw std::runtime_error(std::string("the projector has no ") + key);
            return dflt;
        }
        int64_t v = 0;
        switch (gguf_get_kv_type(g, i)) {
            case GGUF_TYPE_UINT8: v = gguf_get_val_u8(g, i); break;
            case GGUF_TYPE_INT8: v = gguf_get_val_i8(g, i); break;
            case GGUF_TYPE_UINT16: v = gguf_get_val_u16(g, i); break;
            case GGUF_TYPE_INT16: v = gguf_get_val_i16(g, i); break;
            case GGUF_TYPE_UINT32: v = gguf_get_val_u32(g, i); break;
            case GGUF_TYPE_INT32: v = gguf_get_val_i32(g, i); break;
            case GGUF_TYPE_UINT64: v = (int64_t) gguf_get_val_u64(g, i); break;
            case GGUF_TYPE_INT64: v = gguf_get_val_i64(g, i); break;
            default: throw std::runtime_error(std::string("the projector's ") + key + " is not an integer");
        }
        if (v < 0 || v > std::numeric_limits<int32_t>::max())
            throw std::runtime_error(std::string("the projector's ") + key + " is out of range");
        return (int) v;
    }
    float f32(const char* key) const {
        const int i = find(key);
        if (i < 0) throw std::runtime_error(std::string("the projector has no ") + key);
        if (gguf_get_kv_type(g, i) != GGUF_TYPE_FLOAT32)
            throw std::runtime_error(std::string("the projector's ") + key + " is not a float");
        return gguf_get_val_f32(g, i);
    }
    bool boolean(const char* key) const {
        const int i = find(key);
        return i >= 0 && gguf_get_kv_type(g, i) == GGUF_TYPE_BOOL && gguf_get_val_bool(g, i);
    }
    std::string str(const char* key) const {
        const int i = find(key);
        if (i < 0 || gguf_get_kv_type(g, i) != GGUF_TYPE_STRING) return "";
        return gguf_get_val_str(g, i);
    }
    void rgb(const char* key, float out[3]) const {
        const int i = find(key);
        if (i < 0 || gguf_get_kv_type(g, i) != GGUF_TYPE_ARRAY || gguf_get_arr_type(g, i) != GGUF_TYPE_FLOAT32 ||
            gguf_get_arr_n(g, i) < 3)
            throw std::runtime_error(std::string("the projector's ") + key + " is not three floats");
        const float* v = (const float*) gguf_get_arr_data(g, i);
        for (int c = 0; c < 3; ++c) out[c] = v[c];
    }
};

}   // namespace

Projector::~Projector() {
    if (file_ != nullptr) std::fclose(file_);
    if (meta_ != nullptr) ggml_free(meta_);
    if (gguf_ != nullptr) gguf_free(gguf_);
}

void Projector::open(const EncoderParams& p) {
    path_ = p.mmproj;
    gguf_init_params gp = {/*.no_alloc =*/true, /*.ctx =*/&meta_};
    gguf_ = gguf_init_from_file(path_.c_str(), gp);
    if (gguf_ == nullptr) throw std::runtime_error("not a GGUF file that can be read");
    const Keys k{gguf_};

    // the tower's shape and the picture's sizing
    if (!k.boolean("clip.has_vision_encoder")) throw std::runtime_error("the file has no vision encoder");
    std::string type = k.str("clip.projector_type");
    if (type.empty()) type = k.str("clip.vision.projector_type");
    if (type == "qwen3vl_merger") kind = ProjectorKind::qwen3vl;
    else if (type == "hy_v4_flash") kind = ProjectorKind::hy4;
    else if (type == "inkling2v") kind = ProjectorKind::inkling2v;
    else
        throw std::runtime_error("its projector type is " + (type.empty() ? std::string("not named") : type) +
                                 ": this encoder reads qwen3vl_merger, hy_v4_flash and inkling2v");
    n_embd = k.integer("clip.vision.embedding_length", true);
    n_head = k.integer("clip.vision.attention.head_count", true);
    n_embd_head = k.integer("clip.vision.attention.head_dim", false);
    n_head_kv = k.integer("clip.vision.attention.head_count_kv", false, n_head);
    n_layer = k.integer("clip.vision.block_count", true);
    projection_dim = k.integer("clip.vision.projection_dim", true);
    eps = k.f32("clip.vision.attention.layer_norm_epsilon");
    patch_size = k.integer("clip.vision.patch_size", true);
    k.rgb("clip.vision.image_mean", mean);
    k.rgb("clip.vision.image_std", std);
    // (Inkling 2's blocks are GELU by the type: its file has no such key)
    if (kind != ProjectorKind::inkling2v && !k.boolean("clip.use_gelu"))
        throw std::runtime_error("its feed-forward activation is not GELU");
    if (n_embd <= 0 || n_head <= 0 || n_layer <= 0 || patch_size <= 0 || n_embd % n_head != 0 || n_head_kv != n_head)
        throw std::runtime_error("its tower's sizes are not ones this encoder reads");
    n_merge = k.integer("clip.vision.spatial_merge_size", false, 2);
    if (n_merge != 2) throw std::runtime_error("its spatial merge is not 2x2");
    const int patch_area = patch_size * patch_size * n_merge * n_merge;
    size.align = patch_size * n_merge;
    if (kind == ProjectorKind::qwen3vl) {
        // 8 to 4,096 tokens a picture unless the caller says otherwise.  The picture is resized straight to its
        // size (bicubic), a side of a whole number of merged patches and a half going to the even one: what
        // transformers' Qwen2VLImageProcessor does (smart_resize rounds with Python's round(), then resizes),
        // and so what the model saw in training.  mtmd at the pinned commit instead rounds a half up and scales
        // the picture to fit inside the size, the rest black (llama.cpp took the processor's way on 2026-10-19,
        // its #30736): a 16:9 screenshot at 1,024 tokens then has 6 px of black above and below.
        // LAMINA_VISION_LETTERBOX=1 keeps that, for comparisons with such a build.
        size.min_pixels = (p.min_tokens > 0 ? p.min_tokens : 8) * patch_area;
        size.max_pixels = (p.max_tokens > 0 ? p.max_tokens : 4096) * patch_area;
        filter = Filter::bicubic;
        const char* lb = std::getenv("LAMINA_VISION_LETTERBOX");
        letterbox = lb != nullptr && std::atoi(lb) != 0;
        size.half_even = !letterbox;
    } else if (kind == ProjectorKind::inkling2v) {
        // up to 4,096 tokens a picture (mtmd's limit for the type; the caller's): a picture keeps its size and is
        // padded to whole merged patches; one whose padded area passes the limit is first resized to fit it (bicubic)
        size.min_pixels = 0;
        size.max_pixels = (p.max_tokens > 0 ? p.max_tokens : 4096) * patch_area;
        filter = Filter::bicubic;
        letterbox = false;
        pad_to_align = true;
    } else {
        // the reference processor's 4 to 4,096 tokens, then the file's own pixel limits over them (the caller's
        // too, as the pull request has it); resized straight to its size, Lanczos
        size.min_pixels = (p.min_tokens > 0 ? p.min_tokens : 4) * patch_area;
        size.max_pixels = (p.max_tokens > 0 ? p.max_tokens : 4096) * patch_area;
        size.min_pixels = k.integer("clip.vision.image_min_pixels", false, size.min_pixels);
        size.max_pixels = k.integer("clip.vision.image_max_pixels", false, size.max_pixels);
        filter = Filter::lanczos;
        letterbox = false;
    }

    // where each tensor lies
    const uint64_t data = gguf_get_data_offset(gguf_);
    const int64_t n = gguf_get_n_tensors(gguf_);
    for (int64_t i = 0; i < n; ++i) {
        const char* name = gguf_get_tensor_name(gguf_, i);
        const ggml_tensor* t = ggml_get_tensor(meta_, name);
        if (t == nullptr) continue;
        TensorInfo info;
        info.type = (int) t->type;
        for (int d = 0; d < 4; ++d) info.ne[d] = t->ne[d];
        info.offset = data + gguf_get_tensor_offset(gguf_, i);
        info.bytes = ggml_nbytes(t);
        tensors_.emplace(name, info);
    }
    file_ = ggml_fopen(path_.c_str(), "rb");
    if (file_ == nullptr) throw std::runtime_error("the file cannot be opened");

    // what mtmd's tower builder would also apply, and these projectors do not have (Inkling 2's has a class token
    // and scales each block's two outputs)
    const bool inkling = kind == ProjectorKind::inkling2v;
    if ((find("v.class_embd") != nullptr) != inkling)
        throw std::runtime_error(inkling ? "its tower has no class embedding"
                                         : "its tower has a class embedding, which this encoder does not read");
    for (int il = 0; il < n_layer; ++il) {
        const std::string blk = "v.blk." + std::to_string(il) + ".";
        for (const char* scale : {"ls1.weight", "ls2.weight"})
            if ((find(blk + scale) != nullptr) != inkling)
                throw std::runtime_error("its tower has " + std::string(inkling ? "no " : "") + blk + scale +
                                         (inkling ? "" : ", which this encoder does not read"));
        for (const char* other : {"attn_q_norm.weight", "attn_k_norm.weight", "ffn_gate.weight",
                                  "attn_post_norm.weight", "ffn_post_norm.weight"})
            if (find(blk + other) != nullptr)
                throw std::runtime_error("its tower has " + blk + other + ", which this encoder does not read");
        if (find("v.deepstack." + std::to_string(il) + ".fc1.weight") != nullptr)
            throw std::runtime_error("its tower has deepstack layers, which the engine's models do not take");
    }
}

const TensorInfo* Projector::find(const std::string& name) const {
    const auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

void Projector::read(const std::string& name, const TensorInfo& info, void* dst) {
    if (seek64(file_, info.offset) != 0 || std::fread(dst, 1, info.bytes, file_) != info.bytes)
        throw std::runtime_error("the file ends inside the tensor " + name);
}

std::vector<float> Projector::floats(const std::string& name) {
    const TensorInfo* info = find(name);
    if (info == nullptr) throw std::runtime_error("the projector has no tensor " + name);
    const int64_t n = info->ne[0] * info->ne[1] * info->ne[2] * info->ne[3];
    std::vector<float> out((size_t) n);
    if (info->type == GGML_TYPE_F32) {
        read(name, *info, out.data());
    } else if (info->type == GGML_TYPE_F16 || info->type == GGML_TYPE_BF16) {
        std::vector<uint16_t> raw((size_t) n);
        read(name, *info, raw.data());
        if (info->type == GGML_TYPE_F16) ggml_fp16_to_fp32_row((const ggml_fp16_t*) raw.data(), out.data(), n);
        else ggml_bf16_to_fp32_row((const ggml_bf16_t*) raw.data(), out.data(), n);
    } else {
        throw std::runtime_error("the tensor " + name + " is not F32, F16 or BF16");
    }
    return out;
}

// ggml's upscale, GGML_SCALE_MODE_BILINEAR with GGML_SCALE_FLAG_ALIGN_CORNERS, over the table as a picture of
// n_side x n_side cells: the output cell (x, y) reads the table at x / sf0, y / sf1, sf = (out - 1) / (in - 1);
// without the flag at (x + 0.5) / sf0 - 0.5, sf = out / in
std::vector<float> qwen_position_table(const std::vector<float>& table, int n_embd, int n_side, int width, int height,
                                       bool align_corners) {
    std::vector<float> out((size_t) width * height * n_embd);
    if (width == n_side && height == n_side) {
        std::copy(table.begin(), table.begin() + (ptrdiff_t) out.size(), out.begin());
        return out;
    }
    float sf0 = (float) width / n_side;
    float sf1 = (float) height / n_side;
    const float pixel_offset = align_corners ? 0.0f : 0.5f;
    if (align_corners) {
        sf0 = width > 1 && n_side > 1 ? (float) (width - 1) / (n_side - 1) : sf0;
        sf1 = height > 1 && n_side > 1 ? (float) (height - 1) / (n_side - 1) : sf1;
    }
    for (int i1 = 0; i1 < height; i1++) {
        const float y = ((float) i1 + pixel_offset) / sf1 - pixel_offset;
        int64_t y0 = (int64_t) floorf(y);
        int64_t y1 = y0 + 1;
        y0 = std::max(int64_t(0), std::min(y0, (int64_t) n_side - 1));
        y1 = std::max(int64_t(0), std::min(y1, (int64_t) n_side - 1));
        float dy = y - (float) y0;
        dy = std::max(0.0f, std::min(dy, 1.0f));
        for (int i0 = 0; i0 < width; i0++) {
            const float x = ((float) i0 + pixel_offset) / sf0 - pixel_offset;
            int64_t x0 = (int64_t) floorf(x);
            int64_t x1 = x0 + 1;
            x0 = std::max(int64_t(0), std::min(x0, (int64_t) n_side - 1));
            x1 = std::max(int64_t(0), std::min(x1, (int64_t) n_side - 1));
            float dx = x - (float) x0;
            dx = std::max(0.0f, std::min(dx, 1.0f));
            const float* a = &table[((size_t) y0 * n_side + x0) * n_embd];
            const float* b = &table[((size_t) y0 * n_side + x1) * n_embd];
            const float* c = &table[((size_t) y1 * n_side + x0) * n_embd];
            const float* d = &table[((size_t) y1 * n_side + x1) * n_embd];
            float* o = &out[((size_t) i1 * width + i0) * n_embd];
            for (int e = 0; e < n_embd; ++e)
                o[e] = a[e] * (1 - dx) * (1 - dy) + b[e] * dx * (1 - dy) + c[e] * (1 - dx) * dy + d[e] * dx * dy;
        }
    }
    return out;
}

std::vector<float> hy4_position_table(const std::vector<float>& src, int64_t n_embd, int n_grid, int out_w, int out_h) {
    std::vector<float> dst((size_t) n_embd * out_h * out_w);

    const float sx = (float) (out_w + 0.1f) / (float) n_grid;
    const float sy = (float) (out_h + 0.1f) / (float) n_grid;

    for (int y = 0; y < out_h; ++y) {
        const float fy = ((float) y + 0.5f) / sy - 0.5f;
        int y0 = (int) std::floor(fy);
        int y1 = y0 + 1;
        y0 = std::clamp(y0, 0, n_grid - 1);
        y1 = std::clamp(y1, 0, n_grid - 1);
        float wy1 = std::clamp(fy - (float) y0, 0.0f, 1.0f);
        const float wy0 = 1.0f - wy1;
        for (int x = 0; x < out_w; ++x) {
            const float fx = ((float) x + 0.5f) / sx - 0.5f;
            int x0 = (int) std::floor(fx);
            int x1 = x0 + 1;
            x0 = std::clamp(x0, 0, n_grid - 1);
            x1 = std::clamp(x1, 0, n_grid - 1);
            float wx1 = std::clamp(fx - (float) x0, 0.0f, 1.0f);
            const float wx0 = 1.0f - wx1;

            const float w00 = wy0 * wx0;
            const float w01 = wy0 * wx1;
            const float w10 = wy1 * wx0;
            const float w11 = wy1 * wx1;

            const float* s00 = &src[((size_t) y0 * n_grid + x0) * n_embd];
            const float* s01 = &src[((size_t) y0 * n_grid + x1) * n_embd];
            const float* s10 = &src[((size_t) y1 * n_grid + x0) * n_embd];
            const float* s11 = &src[((size_t) y1 * n_grid + x1) * n_embd];
            float* d = &dst[((size_t) y * out_w + x) * n_embd];
            for (int c = 0; c < n_embd; ++c) {
                d[c] = w00 * s00[c] + w01 * s01[c] + w10 * s10[c] + w11 * s11[c];
            }
        }
    }
    return dst;
}

std::vector<float> projector_planes(const Rgb& picture, const ProjectorParams& p, int& tw, int& th) {
    if (p.pad_to_align) return planes_padded(picture, p.size, p.filter, p.mean, p.std, tw, th);
    target_size(picture.w, picture.h, p.size, tw, th);
    if (tw <= 0 || th <= 0 || tw % p.size.align != 0 || th % p.size.align != 0)
        throw std::runtime_error("the picture's size cannot be brought to the projector's grid");
    return planes(fit(picture, tw, th, p.filter, p.letterbox, p.pad), p.mean, p.std);
}

}   // namespace lamina::vision
