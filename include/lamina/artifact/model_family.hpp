// include/lamina/artifact/model_family.hpp - which model family a GGUF holds, from its general.architecture.
//
// This engine was written for one model (Qwen3.8-Flash-Next, `qwen4exp`).  Qwen3.6-35B-A3B (`qwen35moe`) is a close
// cousin (same tokenizer, the same Gated-DeltaNet + gated-attention 3:1 layer pattern, the same head dims), so the
// reader, the geometry and the layout tables know both.  Hy4-flash (`hy_v4_flash`) shares no layer with them: its
// geometry and layout are lamina/hy4/geometry.hpp's; nor does Inkling 2 (`inkling2`, `inkling2_loop`:
// lamina/inkling2/geometry.hpp).  The ENGINE runs only the families `model_family_runs` names; a
// recognised family it does not run yet is refused with `model_family_not_supported`, not mis-run.
//
// No dependencies: the GGUF reader (the architecture guard) and the core geometry both include this.
#pragma once

#include <string>

namespace lamina {

enum class ModelFamily {
    Unknown = 0,
    Qwen4Exp,    ///< Qwen3.8-Flash-Next and its fine-tunes / pruned variants (Swift 1.5, the Coder)
    Qwen35Moe,   ///< Qwen3.6-35B-A3B (Unsloth's MTP GGUFs): plain pre-norm residual, dense attention, nextn MTP block
    Qwen35,      ///< Qwen3.8-27B: Qwen3.6's layers with a dense FFN in place of the experts (Unsloth's GGUFs)
    HyV4Flash,   ///< Tencent's Hy4-flash: gated MLA, sparse and window
                 ///< layers, iHC, Engram, latent experts (AngelSlim's GGUFs)
    Inkling2,    ///< Thinking Machines Lab's Inkling 2 Mini: looped layers,
                 ///< a distance bias, NVFP4 experts (unsloth's GGUFs)
};

inline ModelFamily model_family_from_arch(const std::string& arch) {
    if (arch == "qwen4exp") return ModelFamily::Qwen4Exp;
    if (arch == "qwen35moe") return ModelFamily::Qwen35Moe;
    if (arch == "qwen35") return ModelFamily::Qwen35;
    if (arch == "hy_v4_flash") return ModelFamily::HyV4Flash;
    // two layouts of the same tensors: unsloth's files and mainline llama.cpp's (lamina/inkling2/geometry.hpp)
    if (arch == "inkling2" || arch == "inkling2_loop") return ModelFamily::Inkling2;
    return ModelFamily::Unknown;
}

/// The GGUF architecture string (also the metadata key prefix: "<arch>.block_count").
inline const char* model_family_arch(ModelFamily f) {
    switch (f) {
        case ModelFamily::Qwen4Exp: return "qwen4exp";
        case ModelFamily::Qwen35Moe: return "qwen35moe";
        case ModelFamily::Qwen35: return "qwen35";
        case ModelFamily::HyV4Flash: return "hy_v4_flash";
        case ModelFamily::Inkling2: return "inkling2";   // unsloth's files; mainline's layout says inkling2_loop
        default: return "";
    }
}

/// The model's name for messages.
inline const char* model_family_title(ModelFamily f) {
    switch (f) {
        case ModelFamily::Qwen4Exp: return "Qwen3.8-Flash-Next";
        case ModelFamily::Qwen35Moe: return "Qwen3.6-35B-A3B";
        case ModelFamily::Qwen35: return "Qwen3.8-27B";
        case ModelFamily::HyV4Flash: return "Hy4-flash";
        case ModelFamily::Inkling2: return "Inkling 2 Mini";
        default: return "an unknown model";
    }
}

/// Whether Flash-Next's engine path runs the family.  Qwen3.6, Hy4-flash and Inkling 2 have engine paths of their own
/// (src/qwen35, src/hy4, src/inkling2), which `lamina-engine` switches to at the top of main (a build with CUDA and
/// native experts); without them the family is refused with `model_family_not_supported`.
inline bool model_family_runs(ModelFamily f) { return f == ModelFamily::Qwen4Exp; }

/// The refusal for a recognised family the engine does not run (empty when it runs it).
inline std::string model_family_not_supported(ModelFamily f) {
    if (model_family_runs(f)) return {};
    if (f == ModelFamily::Unknown) return "this is not a model family Lamina knows";
    return std::string(model_family_title(f)) + " (architecture '" + model_family_arch(f) +
           "') needs its own engine path, which this build of Lamina does not have. Use a Qwen3.8-Flash-Next model "
           "file";
}

}   // namespace lamina
