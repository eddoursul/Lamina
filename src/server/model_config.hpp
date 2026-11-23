// src/server/model_config.hpp - lamina-server -m: the run config built from a model's GGUF: the preset its header
// picks (examples/*.json, built in), with this PC's GPUs and the files beside the model.  --write-config writes it; --check says what a model needs against what this PC has.
#pragma once

#include "config.hpp"
#include "json.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace lamina {

/// A model file's header, as a config is built from it: its first shard's keys, every shard's tensors.
struct ModelFacts {
    std::string path;                  ///< the first shard, absolute
    std::vector<std::string> shards;   ///< every shard, the first first
    std::string arch, name, quantized_by;
    int64_t n_embd = 0;                ///< <arch>.embedding_length: a projector made for it has that projection_dim
    uint64_t bytes = 0;                ///< the shards' size
    uint64_t expert_bytes = 0;         ///< its routed experts' tensors (blk.N.ffn_{gate,up,down,gate_up}_exps)
    /// Inkling 2: a layer's tensors (blk.N.*), the full-attention applications it makes a token (0: a window layer;
    /// a looped layer one a pass), an application's K and V a token in FP16, and the looped layers [first, last]
    std::vector<uint64_t> layer_bytes;
    std::vector<int> layer_full;
    uint64_t cell_pair_bytes = 0;
    int loop_first = -1, loop_last = -1;
    std::string unsupported;           ///< why the engine refuses the file ("" when it reads it)
};
/// `path`'s model (any of its shards names it); std::runtime_error when it is not a readable GGUF or a shard is missing
ModelFacts read_model(const std::string& path);

/// The preset a model's config is built from (examples/<name>.json, built in), by the header's architecture and
/// quantizer: ISTA-DASLab's Qwen3.8-Flash-Next files take iq3_s, the family's others ud-q4_k_xl, Qwen3.6-35B-A3B's
/// qwen36-ud-q4_k_xl, Qwen3.8-27B's qwen38-27b-ud-q6_k_m; "" for another architecture.
std::string preset_name(const std::string& arch, const std::string& quantized_by);
/// that preset's config; std::runtime_error when the build holds no such file
json preset(const std::string& name);

/// The GPU a model runs on: of the cards of compute capability 8.0 or newer (of those `allowed` lists, when it lists
/// any), the one with the most VRAM among those that drive no display, else among all of them, the lower number on a
/// tie; -1 when there is none.  Its position in `gpus`.
int pick_main_gpu(const std::vector<GpuInfo>& gpus, const std::vector<int>& allowed);
/// The image encoder's GPU when the vision section asks for one and names none: of the cards of compute capability
/// 8.0 or newer besides the engine's main one (`main`, its nvidia-smi index, -1 unknown), the one with the least VRAM
/// (where the encoder's ~1.3 GB matter least) among those that drive no display, else among all of them, the higher
/// number on a tie; else the main card itself.  Its nvidia-smi index; -1 when there is none.
int pick_vision_gpu(const std::vector<GpuInfo>& gpus, int main);
/// how a config names `g`: by its name when no other card has it, else by its UUID (a number moves when a card is added)
std::string gpu_spec(const GpuInfo& g, const std::vector<GpuInfo>& gpus);

/// "" when `path` is an image projector made for `m` (a GGUF with a vision encoder whose clip.vision.projection_dim is
/// m's width), else why not
std::string projector_problem(const std::string& path, const ModelFacts& m);
/// The projectors made for `m` in its folder (mmproj*.gguf), the smallest first: the CPU encodes a Q8_0 one sooner
/// than an F16 one, and that one sooner than F32
std::vector<std::string> projectors_beside(const ModelFacts& m);

/// What lamina-server's command line says about -m's config
struct ModelOptions {
    std::string model;               ///< -m: any shard of the model
    std::string mmproj;              ///< --mmproj ("": the projector beside the model, if any)
    bool no_mmproj = false;          ///< --no-mmproj
    std::string written;             ///< the config file that describes the run (--write-config's, else --config's):
                                     ///< the log and the learned profile go beside it, else beside the program
    std::vector<int> allowed_gpus;   ///< --gpu: the cards the engine sees
};
/// A config built from a model, and what the server's window says about it
struct ModelConfig {
    json cfg;
    ModelFacts model;
    std::string preset;
    std::vector<std::string> preset_keys;   ///< the preset's keys: a config written over an earlier one renews them
    std::vector<std::string> notes;   ///< "[lamina] ..." lines: the model and its preset, the GPU, the pictures, the log
};
/// The config for `opts.model`: its preset's, the model's own paths in place of the preset's (--native, the
/// projector, the log, the learned profile), the main GPU pick_main_gpu() chooses, and none of the preset's files the
/// server finds itself (the draft layer: resolve_draft_layer; the expert profile: the engine's built-in one).  Pictures
/// are on with a projector (--mmproj, else the one projectors_beside() lists first) in a build with the image encoder.
/// ConfigError for a model no preset is for, or a --mmproj that is not its projector or has no encoder to run it.
ModelConfig config_from_model(const ModelOptions& opts, const std::vector<GpuInfo>& gpus);

/// `cfg` written to `path` as the Settings view writes it (two-space JSON, LF).  An earlier file is kept as <path>.bak,
/// and its keys the new config lacks are carried over (an api_key, an "images" section), but for those in `renewed`
/// (the preset's: a "vision" section that --no-mmproj leaves out stays out): their names, in its order.  ConfigError
/// when it cannot be written.
std::vector<std::string> write_config(const std::string& path, json cfg, const std::vector<std::string>& renewed = {});

/// --check: this PC (its CPU, RAM and GPUs) and, given a config (built from -m or read), its model, what the model
/// needs and how the config runs it, all said in the window; `preset` the one -m's config came from, `model_error` why
/// -m's model gave none.  False when the model would not start.
bool check_report(const json* cfg, const std::vector<GpuInfo>& gpus, const std::string& preset = "",
                  const std::string& model_error = "");

}  // namespace lamina
