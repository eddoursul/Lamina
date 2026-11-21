// src/server/model_config.cpp - lamina-server -m's config, --write-config and --check (model_config.hpp).
#include "model_config.hpp"

#include "lamina/artifact/gguf_reader.hpp"
#include "lamina/embedded.hpp"
#include "lamina/kernels/cpu/expert_layout.hpp"
#include "lamina/platform/memory.hpp"
#include "log.hpp"
#include "process.hpp"
#include "runconfig.hpp"
#include "tokenizer.hpp"
#include "vision.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <regex>
#include <thread>
#include <utility>

namespace lamina {
namespace fs = std::filesystem;
namespace {

std::string fixed(double v, int digits = 1) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.*f", digits, v);
    return buf;
}
std::string gib(double bytes) { return fixed(bytes / 1073741824.0) + " GiB"; }
std::string gb(double bytes) { return fixed(bytes / 1e9) + " GB"; }

std::string meta_str(const GgufFile& g, const std::string& key) {
    const MetaValue* v = g.get(key);
    return v != nullptr && v->type == MetaType::STRING ? v->s : "";
}
int64_t meta_int(const GgufFile& g, const std::string& key) {
    const MetaValue* v = g.get(key);
    return v != nullptr && v->is_num() ? (int64_t) v->num() : 0;
}

/// a routed expert tensor: blk.N.ffn_gate_exps, ffn_up_exps, ffn_down_exps or ffn_gate_up_exps
bool expert_tensor(const std::string& name) {
    static const std::regex re(R"(^blk\.\d+\.ffn_(gate|up|down|gate_up)_exps\.weight$)");
    return std::regex_match(name, re);
}

/// the name clients see: the first shard's file name without its shard suffix and .gguf, in lower case
std::string model_name(const std::string& first_shard) {
    std::string n = u8_str(u8_path(first_shard).filename());
    if (ends_with(lower(n), ".gguf")) n.resize(n.size() - 5);
    static const std::regex shard(R"(-\d{5}-of-\d{5}$)");
    return lower(std::regex_replace(n, shard, ""));
}

std::string absolute_path(const std::string& p) {
    std::error_code ec;
    return u8_str(fs::absolute(u8_path(p), ec).lexically_normal());
}

std::string str_of(const json& v) { return v.is_string() ? v.get<std::string>() : py_dumps(v); }

const GpuInfo* gpu_by_index(const std::vector<GpuInfo>& gpus, int index) {
    for (const GpuInfo& g : gpus)
        if (g.index == index) return &g;
    return nullptr;
}

std::string gpu_line(const GpuInfo& g) {
    return "GPU " + std::to_string(g.index) + ", " + g.name;
}

}  // namespace

ModelFacts read_model(const std::string& path) {
    ModelFacts m;
    for (const std::string& s : gguf_split_paths(path)) m.shards.push_back(absolute_path(s));   // a missing shard throws
    m.path = m.shards.front();
    for (const std::string& s : m.shards) {
        const GgufFile f(s);
        if (s == m.path) {
            m.arch = meta_str(f, "general.architecture");
            m.name = meta_str(f, "general.name");
            m.quantized_by = meta_str(f, "general.quantized_by");
            m.n_embd = meta_int(f, m.arch + ".embedding_length");
            m.unsupported = check_architecture(f);
        }
        m.bytes += f.file_size();
        for (const TensorInfo& t : f.tensors())
            if (expert_tensor(t.name)) m.expert_bytes += tensor_payload_bytes(t);
    }
    return m;
}

std::string preset_name(const std::string& arch, const std::string& quantized_by) {
    if (arch == "qwen4exp") return quantized_by == "ISTA DASLab" ? "iq3_s" : "ud-q4_k_xl";
    if (arch == "qwen35moe") return "qwen36-ud-q4_k_xl";
    if (arch == "qwen35") return "qwen38-27b-ud-q6_k_m";
    if (arch == "hy_v4_flash") return "hy4-flash-q4_k_m";
    if (arch == "inkling2" || arch == "inkling2_loop") return "inkling2-mini-ud-iq4_xs";
    return "";
}

json preset(const std::string& name) {
    const EmbeddedFile* f = preset_file(name + ".json");
    json cfg;
    if (f == nullptr || !py_loads(f->data, cfg) || !cfg.is_object())
        throw std::runtime_error("this build holds no preset " + name + " (examples/" + name + ".json)");
    args_to_list(cfg);
    return cfg;
}

int pick_main_gpu(const std::vector<GpuInfo>& gpus, const std::vector<int>& allowed) {
    int best = -1;
    for (size_t i = 0; i < gpus.size(); ++i) {
        const GpuInfo& g = gpus[i];
        // 0: nvidia-smi does not say, and the engine's own check decides
        if ((g.cc != 0 && g.cc < 80) || (!allowed.empty() && std::find(allowed.begin(), allowed.end(), g.index) == allowed.end()))
            continue;
        if (best >= 0) {
            const GpuInfo& b = gpus[(size_t) best];
            if (g.display != b.display ? g.display
                : g.memory_mib != b.memory_mib ? g.memory_mib < b.memory_mib : g.index > b.index)
                continue;
        }
        best = (int) i;
    }
    return best;
}

int pick_vision_gpu(const std::vector<GpuInfo>& gpus, int main) {
    const GpuInfo* best = nullptr;
    for (const bool any : {false, true}) {   // the cards that drive no display first
        for (const GpuInfo& g : gpus) {
            if (g.index == main || (g.cc != 0 && g.cc < 80) || (!any && g.display)) continue;
            if (best != nullptr && (g.memory_mib > best->memory_mib ||
                                    (g.memory_mib == best->memory_mib && g.index < best->index)))
                continue;
            best = &g;
        }
        if (best != nullptr) return best->index;
    }
    for (const GpuInfo& g : gpus)
        if (g.index == main) return main;
    return -1;
}

std::string gpu_spec(const GpuInfo& g, const std::vector<GpuInfo>& gpus) {
    const auto same = std::count_if(gpus.begin(), gpus.end(), [&](const GpuInfo& x) { return x.name == g.name; });
    return same == 1 ? g.name : g.uuid;
}

std::string projector_problem(const std::string& path, const ModelFacts& m) {
    try {
        const GgufFile g(path);
        const MetaValue* enc = g.get("clip.has_vision_encoder");
        if (meta_str(g, "general.architecture") != "clip" || enc == nullptr || enc->u == 0)
            return "not an image projector (a GGUF of the architecture clip with a vision encoder)";
        if (const std::string type = projector_type(path); !projector_type_read(type))
            return "a projector of the type " + (type.empty() ? std::string("it does not name") : type) +
                   ", which the image encoder does not read (it reads qwen3vl_merger and hy_v4_flash)";
        const int64_t dim = meta_int(g, "clip.vision.projection_dim");
        if (m.n_embd > 0 && dim != m.n_embd)
            return "the projector of a model " + std::to_string(dim) + " wide, and this one is " +
                   std::to_string(m.n_embd);
        return "";
    } catch (const std::exception& e) {
        return e.what();
    }
}

std::vector<std::string> projectors_beside(const ModelFacts& m) {
    std::vector<std::pair<uintmax_t, std::string>> found;
    std::error_code ec;
    for (fs::directory_iterator it(u8_path(m.path).parent_path(), ec), end; !ec && it != end; it.increment(ec)) {
        const std::string name = lower(u8_str(it->path().filename()));
        if (!starts_with(name, "mmproj") || !ends_with(name, ".gguf") || !it->is_regular_file(ec)) continue;
        const std::string p = u8_str(it->path().lexically_normal());
        if (projector_problem(p, m).empty()) found.emplace_back(it->file_size(ec), p);
    }
    std::sort(found.begin(), found.end());
    std::vector<std::string> out;
    for (auto& f : found) out.push_back(std::move(f.second));
    return out;
}

ModelConfig config_from_model(const ModelOptions& o, const std::vector<GpuInfo>& gpus) {
    ModelConfig mc;
    try {
        mc.model = read_model(o.model);
    } catch (const std::exception& e) {
        throw ConfigError(o.model + ": " + e.what());
    }
    const ModelFacts& m = mc.model;
    mc.preset = preset_name(m.arch, m.quantized_by);
    if (mc.preset.empty())
        throw ConfigError(m.path + ": not a model Lamina runs (its architecture '" + m.arch + "'; Lamina runs qwen4exp, "
                          "Qwen3.8-Flash-Next, qwen35moe, Qwen3.6-35B-A3B, qwen35, Qwen3.8-27B, hy_v4_flash, Hy4-flash, and "
                          "inkling2 / inkling2_loop, Inkling 2 Mini)");
    if (!m.unsupported.empty()) throw ConfigError(m.path + ": " + m.unsupported);
    json p;
    try {
        p = preset(mc.preset);
    } catch (const std::exception& e) {
        throw ConfigError(e.what());
    }
    const std::string name = model_name(m.path);
    mc.notes.push_back("[lamina] model: " + m.path + " (" + (m.name.empty() ? m.arch : m.name) +
                       (m.quantized_by.empty() ? "" : ", quantized by " + m.quantized_by) + "), preset " + mc.preset);

    const int main_gpu = pick_main_gpu(gpus, o.allowed_gpus);
    if (main_gpu < 0)
        mc.notes.push_back(gpus.empty() ? "[lamina] nvidia-smi lists no GPU: the engine starts on its GPU 0"
                                        : "[lamina] no GPU of compute capability 8.0 or newer: the engine starts on its "
                                          "GPU 0");

    // pictures: the projector (named, else the one beside the model), in a build with the image encoder
    std::string mmproj, why_not;
    if (o.no_mmproj) {
        why_not = "--no-mmproj";
    } else if (!o.mmproj.empty()) {
        mmproj = absolute_path(o.mmproj);
        if (const std::string why = projector_problem(mmproj, m); !why.empty())
            throw ConfigError("--mmproj " + o.mmproj + ": " + why);
        if (!vision_built()) throw ConfigError("--mmproj " + o.mmproj + ": " + vision_not_built());
    } else if (const std::vector<std::string> found = projectors_beside(m); found.empty()) {
        why_not = "no image projector made for it (mmproj*.gguf) beside the model; --mmproj names one";
    } else if (!vision_built()) {
        why_not = vision_not_built();
    } else {
        mmproj = found.front();
    }
    if (mmproj.empty() || !jget(p, "vision").is_object()) {
        mmproj.clear();
        mc.notes.push_back("[lamina] no pictures: " + (why_not.empty() ? "its preset reads none" : why_not));
    } else {
        mc.notes.push_back("[lamina] pictures: " + mmproj + " (--no-mmproj: none)");
    }

    // the files a run writes: beside the written config, else beside the program
    std::error_code ec;
    fs::path dir = u8_path(self_dir());
    std::string stem = name;
    if (!o.written.empty()) {
        const fs::path w = fs::absolute(u8_path(o.written), ec).lexically_normal();
        dir = w.parent_path();
        stem = u8_str(w.stem());
    }

    json args = json::array();
    const json& pa = jget(p, "args");
    const std::optional<std::vector<std::string>> preset_cards = gpu_cards(p);
    bool main_named = preset_cards.has_value();   // the cards name it (below)
    for (size_t i = 0; pa.is_array() && i < pa.size(); ++i) {
        const std::string a = str_of(pa[i]);
        if (a == "--native" && i + 1 < pa.size()) {
            args.push_back(a);
            args.push_back(m.path);
            ++i;
        } else if ((a == "--mtp" || a == "--expert-profile") && i + 1 < pa.size()) {
            // Flash-Next's draft layer is found beside the model (resolve_draft_layer), its profile is the engine's own
            ++i;
        } else if (a == "--main-gpu" && i + 1 < pa.size()) {
            if (main_gpu >= 0) {
                args.push_back(a);
                args.push_back(gpu_spec(gpus[(size_t) main_gpu], gpus));
                main_named = true;
            }
            ++i;
        } else if (a == "--vision") {
            if (!mmproj.empty()) args.push_back(a);
        } else {
            args.push_back(pa[i]);
        }
    }
    if (main_gpu >= 0 && !main_named) {
        args.push_back("--main-gpu");
        args.push_back(gpu_spec(gpus[(size_t) main_gpu], gpus));
    }
    // the preset's cards: the first the main GPU picked here (its settings kept), the others the preset's, or with
    // "gpu" numbers given (the cards it may use) the others of those, in their order
    json cards = json::array();
    if (preset_cards && !preset_cards->empty()) {
        const std::string& first = preset_cards->front();
        const size_t comma = first.find(',');
        // (none picked: the engine starts on its GPU 0, as the note says)
        cards.push_back((main_gpu >= 0 ? gpu_spec(gpus[(size_t) main_gpu], gpus) : std::string("0")) +
                        (comma == std::string::npos ? "" : first.substr(comma)));
        if (!o.allowed_gpus.empty()) {
            for (const int g : o.allowed_gpus)
                if (main_gpu < 0 || g != gpus[(size_t) main_gpu].index)
                    for (const GpuInfo& gi : gpus)
                        if (gi.index == g) cards.push_back(gpu_spec(gi, gpus));
        } else {
            for (size_t i = 1; i < preset_cards->size(); ++i) cards.push_back((*preset_cards)[i]);
        }
    }

    json& cfg = mc.cfg;
    cfg = json::object();
    for (auto it = p.begin(); it != p.end(); ++it) {
        const std::string& k = it.key();
        mc.preset_keys.push_back(k);
        if (k == "args") {
            cfg[k] = args;
        } else if (k == "model") {
            cfg[k] = m.path;
        } else if (k == "gpu" && preset_cards) {
            cfg[k] = cards;
        } else if (k == "cwd" || k == "lib_dirs") {   // the paths are absolute; the release finds its CUDA runtime itself
        } else if (k == "model_name") {
            cfg[k] = name;
        } else if (k == "log") {
            cfg[k] = u8_str(dir / (stem + ".log"));
        } else if (k == "expert_profile_save") {
            cfg[k] = u8_str(dir / (stem + "-learned.bin"));
        } else if (k == "vision") {
            if (mmproj.empty()) continue;
            json v = json::object();
            // the preset's encoder ("gpu", "lend": on the engine's main GPU, its VRAM borrowed from the engine for
            // each new picture) on this PC's cards (no "cuda_device"), its CPU threads this PC's for the pictures
            // that find no VRAM there
            for (auto vt = it->begin(); vt != it->end(); ++vt) {
                const std::string& vk = vt.key();
                if (vk == "exe" || vk == "cuda_device") continue;
                v[vk] = vk == "mmproj" ? json(mmproj) : vk == "model" ? json(m.path)
                      : vk == "threads" ? json((int) std::max(1u, std::thread::hardware_concurrency())) : *vt;
            }
            for (const char* need : {"mmproj", "model"})
                if (!v.contains(need)) v[need] = std::string(need) == "mmproj" ? mmproj : m.path;
            cfg[k] = std::move(v);
        } else {
            cfg[k] = *it;
        }
    }
    if (cfg.contains("log")) mc.notes.push_back("[lamina] the engine's log: " + str_of(cfg["log"]));
    return mc;
}

std::vector<std::string> write_config(const std::string& path, json cfg, const std::vector<std::string>& renewed) {
    std::vector<std::string> kept;
    std::error_code ec;
    try {
        if (fs::exists(u8_path(path), ec)) {   // written again: the earlier file's own keys stay
            const json old = runconfig::load(path);
            for (auto it = old.begin(); it != old.end(); ++it)
                if (!cfg.contains(it.key()) && std::find(renewed.begin(), renewed.end(), it.key()) == renewed.end()) {
                    cfg[it.key()] = *it;
                    kept.push_back(it.key());
                }
            runconfig::save(path, cfg);   // the earlier file kept as <name>.bak
            return kept;
        }
    } catch (const std::exception& e) {
        throw ConfigError(path + ": " + e.what());
    }
    std::ofstream f(u8_path(path), std::ios::binary);
    f << py_dumps(cfg, true, false, 2) << "\n";
    f.close();
    if (!f) throw ConfigError(path + " cannot be written");
    return kept;
}

bool check_report(const json* cfg, const std::vector<GpuInfo>& gpus, const std::string& preset,
                  const std::string& model_error) {
    std::vector<std::string> stops;   // why the model would not start
    if (!model_error.empty()) stops.push_back(model_error);
    const unsigned cores = std::max(1u, std::thread::hardware_concurrency());
    say("Lamina " LAMINA_VERSION ", lamina-server in " + self_dir());
#if defined(__aarch64__) || defined(_M_ARM64)
    say("CPU: arm64, " + std::to_string(cores) + " logical cores");
#else
    const bool avx512 = kernels::cpu::cpu_avx512_ok(), avx2 = kernels::cpu::cpu_avx2_ok();
    say(std::string("CPU: ") + (avx512 ? "AVX-512" : avx2 ? "AVX2 (no AVX-512)" : "no AVX2") + ", " +
        std::to_string(cores) + " logical cores");
    if (const char* missing = kernels::cpu::cpu_floor_missing())   // the start refuses before this (cpu_floor.cpp)
        stops.push_back(std::string("the CPU has no ") + missing + ", which the engine's CPU kernels need");
#endif
    uint64_t total = 0, avail = 0;
    if (platform::total_ram(total) && platform::available_ram(avail))
        say("RAM: " + gib((double) total) + ", " + gib((double) avail) + " available");
    if (gpus.empty()) {
        say("GPUs: nvidia-smi lists none (no NVIDIA driver, or nvidia-smi is not on the PATH)");
        stops.push_back("no NVIDIA GPU");
    } else {
        say("GPUs (NVIDIA driver " + gpus.front().driver + "):");
        for (const GpuInfo& g : gpus)
            say("  " + std::to_string(g.index) + "  " + g.name + ", " + gib((double) g.memory_mib * 1048576.0) +
                (g.cc ? ", compute capability " + std::to_string(g.cc / 10) + "." + std::to_string(g.cc % 10) : "") +
                (g.display ? ", drives a display" : "") +
                (g.cc != 0 && g.cc < 80 ? " (older than the RTX 30 series: the engine leaves it out)" : ""));
        if (pick_main_gpu(gpus, {}) < 0) stops.push_back("no GPU of compute capability 8.0 or newer (the RTX 30 series)");
    }
    if (cfg != nullptr) {
        const json& c = *cfg;
        const std::string path = model_file(c);
        ModelFacts m;
        try {
            m = read_model(path);
        } catch (const std::exception& e) {
            stops.push_back(path + ": " + e.what());
        }
        if (!m.path.empty()) {
            say("Model: " + m.path);
            say("  " + (m.name.empty() ? std::string("(no name)") : m.name) + ", architecture " + m.arch +
                (m.quantized_by.empty() ? "" : ", quantized by " + m.quantized_by) + ": " +
                std::to_string(m.shards.size()) + (m.shards.size() == 1 ? " file, " : " shards, ") + gb((double) m.bytes));
            if (!preset.empty()) say("  preset: " + preset + " (examples/" + preset + ".json)");
            if (!m.unsupported.empty()) stops.push_back(m.path + ": " + m.unsupported);
            try {
                (void) Tokenizer::from_gguf(m.path);
            } catch (const std::exception& e) {
                stops.push_back(std::string("its tokenizer: ") + e.what());
            }
            // where it runs
            std::string gpu_why;
            const std::vector<int> on = engine_gpus(c, gpus, &gpu_why);
            if (!gpu_why.empty() && !gpus.empty()) stops.push_back(gpu_why);   // none listed: said above
            const GpuInfo* main = on.empty() ? nullptr : gpu_by_index(gpus, on.front());
            std::string more;
            for (size_t i = 1; i < on.size(); ++i)
                if (const GpuInfo* g = gpu_by_index(gpus, on[i])) more += (more.empty() ? "" : "; ") + gpu_line(*g);
            const bool flash = m.arch == "qwen4exp";
            const bool inkling = m.arch == "inkling2" || m.arch == "inkling2_loop";
            if (main != nullptr)
                say("  runs on " + gpu_line(*main) +
                    (more.empty() ? "" : (flash ? ", expert tiers: " : inkling ? ", the context's K and V also on: "
                                                                               : ", a layer split may add: ") + more));
            // each card's own settings (--gpu NAME,key=value)
            for (const std::string& card : engine_cards(c)) {
                const size_t comma = card.find(',');
                if (comma == std::string::npos) continue;
                const std::string name(strip(card.substr(0, comma)));
                const int i = name == "auto" ? -1 : find_gpu(name, gpus);
                const GpuInfo* g = i < 0 ? nullptr : gpu_by_index(gpus, i);
                say("  " + (g != nullptr ? gpu_line(*g) : name) + ": " + replace_all(card.substr(comma + 1), ",", ", ") +
                    (flash ? "" : inkling ? " (this model's path takes the cards' order and a later card's clocks)"
                                          : " (this model's path takes the cards' order alone)"));
            }
            // what it takes
            if (flash) {
                const platform::RamFit ram = platform::ram_fit(m.expert_bytes);
                say("  RAM: its experts take " + gib((double) m.expert_bytes) + ", pinned");
                if (ram.too_much()) stops.push_back(ram.say("the model's experts", "a smaller quant of the model takes less"));
                else if (ram.tight()) say("  note: " + ram.say("the model's experts", "close other programs, or use a smaller quant"));
                json probe = c;   // the draft layer as the start finds it
                const std::string line = resolve_draft_layer(probe, m.arch);
                const std::vector<std::string> pa = engine_args(probe);
                const auto mtp = std::find(pa.begin(), pa.end(), "--mtp");
                if (!line.empty()) say("  " + std::string(starts_with(line, "[lamina] ") ? line.substr(9) : line));
                else if (mtp != pa.end() && mtp + 1 != pa.end()) say("  draft layer: " + *(mtp + 1));
            } else if (inkling) {
                // The whole file and ~1.5 GiB beside it (the window layers' cells, a prompt chunk's buffers, the
                // graphs) on the first card; K and V of the 8 full-attention layer applications in what is left of it
                // and on the other cards (which keep ~1 GiB for their own buffers and the reserve).
                const std::vector<std::string> pa = engine_args(c);
                const auto kv = std::find(pa.begin(), pa.end(), "--kv");
                const bool q8 = kv != pa.end() && kv + 1 != pa.end() && *(kv + 1) == "q8_0";
                const double per = (q8 ? 17.0 : 32.0) * 1024.0, gibi = 1073741824.0;
                double room = 0, others = 0;
                if (main != nullptr)
                    room = (double) main->memory_mib * 1048576.0 - (double) m.bytes - (main->display ? 3.0 : 2.2) * gibi;
                for (size_t i = 1; i < on.size(); ++i)
                    if (const GpuInfo* g = gpu_by_index(gpus, on[i]))
                        others += std::max(0.0, (double) g->memory_mib * 1048576.0 - (g->display ? 2.5 : 1.7) * gibi);
                const double tokens = std::min(1048576.0, (std::max(0.0, room) + others) / per);
                say("  VRAM: the whole model (" + gb((double) m.bytes) + ") and about 1.5 GiB beside it on its GPU; the "
                    "context's K and V (" + (q8 ? "17" : "32") + " KiB a token, " + (q8 ? "q8_0" : "FP16; \"kv\": \"q8_0\" 17") +
                    ") in the rest" + (on.size() > 1 ? " and on the other cards" : "") + ": room for about " +
                    std::to_string((long long) (tokens / 1000.0)) + " thousand tokens of the model's 1,048,576");
                if (main != nullptr && room < 32768.0 * per)
                    stops.push_back(gpu_line(*main) + " does not hold the model's " + gb((double) m.bytes) +
                                    " with a context of 32,768 tokens beside it: a smaller quant of the model, or a card "
                                    "with more VRAM first in \"gpu\"");
            } else {
                double vram = 0;
                for (const int i : on)
                    if (const GpuInfo* g = gpu_by_index(gpus, i)) vram += (double) g->memory_mib * 1048576.0;
                say("  VRAM: the model's " + gb((double) m.bytes) + ", its GPUs' " + gib(vram) +
                    " (its layers are split over the fewest that hold them with the context's K/V)");
                if (m.expert_bytes > 0 && (double) m.bytes > vram) {
                    const platform::RamFit ram = platform::ram_fit(m.expert_bytes);
                    say("  RAM: the experts the GPUs cannot hold, up to " + gib((double) m.expert_bytes));
                    if (ram.too_much()) stops.push_back(ram.say("the model's experts", "a smaller quant of the model takes less"));
                }
            }
            // pictures
            const json& v = jget(c, "vision");
            if (!v.is_object()) {
                say("  pictures: off");
            } else {
                const std::string mmproj = jstr(v, "mmproj");
                const std::string why = projector_problem(mmproj, m);
                if (!why.empty()) stops.push_back("the vision section's projector " + mmproj + ": " + why);
                if (!vision_built())
                    stops.push_back(std::string("the config has a \"vision\" section, and ") + vision_not_built());
                say("  pictures: " + mmproj + ", encoded " +
                    (truthy(jget(v, "gpu")) ? "on " + (jget(v, "cuda_device").is_null() ? std::string("a GPU")
                                                                                         : str_of(v["cuda_device"])) +
                                                  (truthy(jget(v, "lend")) ? ", in VRAM borrowed from the engine for each "
                                                                             "new picture where it can lend" : "")
                                            : "on the CPU (" + (jget(v, "threads").is_null() ? std::string("its default")
                                                                                              : str_of(v["threads"])) +
                                                  " threads)"));
            }
            // the engine's default under --serve, each path's (Inkling 2's: what the VRAM holds, said above)
            std::string ctx = flash ? "262144 tokens" : inkling ? "what the VRAM holds" : "32768 tokens";
            const std::vector<std::string> args = engine_args(c);
            for (size_t i = 0; i + 1 < args.size(); ++i)
                if (args[i] == "--max-context") ctx = args[i + 1] + " tokens";
            say("  port " + (jget(c, "port").is_null() ? std::string("8095") : str_of(c["port"])) + ", context " + ctx +
                (jget(c, "log").is_null() ? "" : ", the engine's log " + str_of(c["log"])));
            std::string line = "lamina --serve";
            for (const std::string& a : args) line += " " + (a.find(' ') == std::string::npos ? a : "\"" + a + "\"");
            say("  engine: " + line);
        }
    }
    if (stops.empty()) {
        say(cfg != nullptr ? "It can start." : "This PC runs Lamina; -m MODEL.gguf or --config FILE checks a model.");
        return true;
    }
    say("It would not start:");
    for (const std::string& s : stops) say("  - " + s);
    return false;
}

}  // namespace lamina
