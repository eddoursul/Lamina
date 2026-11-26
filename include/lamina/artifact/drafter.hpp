// include/lamina/artifact/drafter.hpp - a drafter file as the engine's paths take it: --drafter FILE (its earlier names
// --mtp, --dspark and --dfly still taken, the same option on every path), its kind read from its contents, so that a
// path takes its own model's drafter and names another's.  Header-only, GPU- and JSON-free: the engines' CUDA files
// and the server include it.
#pragma once

#include "lamina/artifact/draft_layer.hpp"
#include "lamina/artifact/gguf_reader.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace lamina {

/// The option, and its earlier names
inline constexpr const char* kDrafterFlag = "--drafter";
inline const std::vector<std::string>& drafter_flags() {
    static const std::vector<std::string> names{"--drafter", "--mtp", "--dspark", "--dfly"};
    return names;
}
/// Whether `flag` with `value` names a drafter file: --drafter or one of its earlier names; not --mtp N (digits alone),
/// the Qwen3.6 and Qwen3.8-27B path's earlier draft count (--drafts-per-step N)
inline bool is_drafter_flag(const std::string& flag, const std::string& value) {
    if (flag == "--mtp")
        return value.empty() || !std::all_of(value.begin(), value.end(), [](unsigned char c) { return std::isdigit(c); });
    for (const std::string& n : drafter_flags())
        if (flag == n) return true;
    return false;
}
/// The value of `args`' last drafter flag (the engines take the last one), "" where none names one; `at`: its flag's
/// index
inline std::string drafter_arg(const std::vector<std::string>& args, size_t* at = nullptr) {
    std::string out;
    for (size_t i = 0; i + 1 < args.size(); ++i)
        if (is_drafter_flag(args[i], args[i + 1])) {
            out = args[i + 1];
            if (at) *at = i;
        }
    return out;
}

/// What a drafter file is
enum class DrafterKind {
    Missing,      ///< no such file or folder
    Other,        ///< no drafter of the kinds below: another GGUF (a model, a projector), another safetensors file
    DraftLayer,   ///< Qwen3.8-Flash-Next's draft layer: the GGUF tools/mtp_rt.py writes, or a folder of its earlier layout
    DSpark,       ///< Inkling 2's block drafter: a dflash GGUF with the Markov head's table, or the checkpoint's mtp.safetensors
    DFly2,        ///< Hy4-flash's DFly2: the tencent/Hy4-flash checkpoint's last shard (model.dfly_layers.*)
};

/// `path`'s safetensors header, "" where it is no safetensors file (the header alone is read)
inline std::string safetensors_header(const std::string& path) {
    std::ifstream in(std::filesystem::path(std::u8string(path.begin(), path.end())), std::ios::binary);
    uint64_t n = 0;
    if (!in || !in.read(reinterpret_cast<char*>(&n), 8) || n < 2 || n > (uint64_t{64} << 20)) return {};
    std::string head((size_t) n, '\0');
    if (!in.read(head.data(), (std::streamsize) n) || head[0] != '{') return {};
    return head;
}

/// `path`'s kind, from its contents: a GGUF's architecture (and a dflash file's Markov table), a safetensors header's
/// tensor names, a folder's files
inline DrafterKind drafter_kind(const std::string& path) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path p(std::u8string(path.begin(), path.end()));
    if (fs::is_directory(p, ec)) return fs::is_regular_file(p / "dense.txt", ec) ? DrafterKind::DraftLayer : DrafterKind::Other;
    if (!fs::is_regular_file(p, ec)) return DrafterKind::Missing;
    if (const std::string h = safetensors_header(path); !h.empty()) {
        if (h.find("\"markov_head.markov_w1.weight\"") != std::string::npos) return DrafterKind::DSpark;
        if (h.find("\"model.dfly_layers.") != std::string::npos) return DrafterKind::DFly2;
        return DrafterKind::Other;
    }
    try {
        const GgufFile f(path);
        const MetaValue* a = f.get("general.architecture");
        if (a == nullptr) return DrafterKind::Other;
        if (a->s == kDraftLayerArch) return DrafterKind::DraftLayer;
        if (a->s == "dflash" && f.find("markov_w1.weight") != nullptr) return DrafterKind::DSpark;
    } catch (const std::exception&) {
    }
    return DrafterKind::Other;
}

/// The model a drafter of that kind is for, and what it is
inline const char* drafter_kind_name(DrafterKind k) {
    switch (k) {
    case DrafterKind::DraftLayer: return "Qwen3.8-Flash-Next's draft layer";
    case DrafterKind::DSpark: return "Inkling 2's block drafter (DSpark)";
    case DrafterKind::DFly2: return "Hy4-flash's drafter (DFly2)";
    case DrafterKind::Missing: return "not there";
    default: return "no drafter this engine reads";
    }
}

/// A path's refusal of the file `flag` names, where it is not there or is another model's drafter: "" where it is of
/// the kind `want`, or of no kind another path takes (the path's own loader says then what is wrong with it).
/// `model`: the path's model; `its`: what its drafter is.
inline std::string drafter_mismatch(const std::string& flag, const std::string& path, DrafterKind want, const char* model,
                                    const char* its) {
    const DrafterKind k = drafter_kind(path);
    if (k == DrafterKind::Missing) return flag + " " + path + " is not there";
    if (k == want || k == DrafterKind::Other) return {};
    return flag + " " + path + " is " + drafter_kind_name(k) + ", another model's: " + model + "'s drafter is " + its;
}

/// A GGUF file that is a drafter rather than a model, and whether this engine reads it
struct GgufDrafter {
    std::string what;    ///< what drafter it is, a phrase for one file or several ("" where the file is none)
    bool read = false;   ///< one this engine's paths read (drafter_kind's)
};
/// Whether the GGUF file at `path` is a drafter, by its header: a kind this engine reads (drafter_kind), another
/// engine's by its architecture (EAGLE-3, DFlash, a DFly drafter, an "-mtp" one), or a model's MTP head - a single file
/// whose layers are its architecture's next-token layers alone (nextn_predict_layers: llama.cpp's mtp-*.gguf); else,
/// for an architecture of no such kind, by llama.cpp's names for a model's side files (mtp-, eagle3-, dflash-,
/// dspark-).  A model's file, a split model's first shard among them, is none.
inline GgufDrafter gguf_drafter(const std::string& path) {
    GgufDrafter d;
    if (const DrafterKind k = drafter_kind(path); k == DrafterKind::DraftLayer || k == DrafterKind::DSpark) {
        d.what = k == DrafterKind::DraftLayer ? "Qwen3.8-Flash-Next's draft layer" : "Inkling 2's DSpark";
        d.read = true;
        return d;
    }
    try {
        const GgufFile f(path);
        const MetaValue* a = f.get("general.architecture");
        if (a == nullptr || a->type != MetaType::STRING) return d;
        const std::string& arch = a->s;
        if (arch == "eagle3" || arch == "eagle") {
            d.what = "EAGLE-3, architecture " + arch;
        } else if (arch == "dflash") {
            d.what = "DFlash, architecture dflash without DSpark's table";
        } else if (arch.ends_with("_dfly") || arch.ends_with("-dfly")) {
            d.what = "DFly, architecture " + arch;
        } else if (arch.ends_with("_mtp") || arch.ends_with("-mtp")) {
            d.what = "MTP, architecture " + arch;
        } else {
            // an MTP head: the next-token layers alone, in a file of its own
            const MetaValue* n = f.get(arch + ".nextn_predict_layers");
            const MetaValue* layers = f.get(arch + ".block_count");
            const MetaValue* split = f.get("split.count");
            if (n != nullptr && layers != nullptr && n->is_num() && layers->is_num() && n->u > 0 && n->u < layers->u &&
                (split == nullptr || split->u <= 1)) {
                bool any = false, trunk = false;
                for (const TensorInfo& t : f.tensors())
                    if (t.name.rfind("blk.", 0) == 0) {
                        any = true;
                        trunk = trunk || std::strtoull(t.name.c_str() + 4, nullptr, 10) < layers->u - n->u;
                    }
                if (any && !trunk) d.what = "MTP head, " + arch + "'s next-token layers alone";
            }
            if (d.what.empty() && model_family_from_arch(arch) == ModelFamily::Unknown) {
                const std::u8string file = std::filesystem::path(std::u8string(path.begin(), path.end())).filename().u8string();
                std::string name(file.begin(), file.end());
                for (char& c : name) c = (char) std::tolower((unsigned char) c);
                for (const char* side : {"mtp-", "eagle3-", "dflash-", "dspark-"})
                    if (name.find(side) != std::string::npos)
                        d.what = "named as llama.cpp names a drafter, architecture " + arch;
            }
        }
    } catch (const std::exception&) {
    }
    return d;
}

}  // namespace lamina
