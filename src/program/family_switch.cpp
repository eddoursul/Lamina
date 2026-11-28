// src/program/family_switch.cpp - lamina-engine's family switch: the command line read for its model file's family
// (the GGUF header read once), and a file of Qwen3.6-35B-A3B or Qwen3.8-27B (src/qwen35), Hy4-flash (src/hy4) or
// Inkling 2 (src/inkling2) handed to that path, the --gpu cards written as the path's own flags; Qwen3.8-Flash-Next's
// file, or one not readable here (its own parse says why), stays with generate.cpp's engine.
#include "family_switch.hpp"

#if defined(LAMINA_HAVE_QWEN35)
#include "lamina/artifact/gguf_reader.hpp"
#include "lamina/artifact/gguf_split.hpp"
#include "lamina/artifact/model_family.hpp"
#include "lamina/core/gpu_select.hpp"
#include "lamina/program/gpu_cards.hpp"
#include "lamina/qwen35/engine.hpp"
#if defined(LAMINA_HAVE_HY4)
#include "lamina/hy4/engine.hpp"
#endif
#if defined(LAMINA_HAVE_INKLING2)
#include "lamina/inkling2/engine.hpp"
#endif

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

/// The qwen35 path's switch: the exit code of the engine path of Qwen3.6-35B-A3B
/// and Qwen3.8-27B (src/qwen35), which qwen35moe and qwen35 files (--native's header) run on.  That path runs on
/// --main-gpu (named as for this engine), made current before its first CUDA call, and splits the layers over it and
/// --second-gpu's GPUs where one does not hold them; it gets the command line without either, the GPUs as --stage-gpus.
/// The family switch's reading of a command line: the family of --native's file (Unknown when it is not readable here:
/// the engine below says why), and the line without --main-gpu and --second-gpu, which the switch takes.
struct Qwen35Line {
    lamina::ModelFamily family = lamina::ModelFamily::Unknown;
    std::string main_gpu = "0", second;
    std::string bad;                                          ///< a --gpu it cannot read
    std::string skipped;                                      ///< --gpu cards whose settings this path has no use for
    std::string ink_unused;                                   ///< the cards' settings Inkling 2's path has no use for
    std::vector<std::string> clocks;                          ///< each later --gpu card's clocks= ("" where
                                                              ///< it gives none), as `second` lists them
    std::vector<std::optional<uint64_t>> reserves;            ///< and its reserve=SIZE
    std::vector<std::optional<std::pair<int, int>>> layers;   ///< and its layers=A-B (Inkling 2's layer split)
    std::vector<char> caches;                                 ///< and whether it gives cache= (Inkling
                                                              ///< 2: the card of the RAM experts)
    std::optional<uint64_t> main_cache, main_reserve;         ///< the first --gpu card's cache=SIZE and reserve=SIZE
    std::string main_clocks;                                  ///< and its clocks=
    bool main_slots = false;                                  ///< its cache=N (a number of slots)
    std::vector<char*> rest;
    bool qwen35() const { return family == lamina::ModelFamily::Qwen35Moe || family == lamina::ModelFamily::Qwen35; }
    bool hy4() const { return family == lamina::ModelFamily::HyV4Flash; }
    bool inkling2() const { return family == lamina::ModelFamily::Inkling2; }
};
Qwen35Line qwen35_line(int argc, char** argv) {
    Qwen35Line q;
    std::string model;
    q.rest.push_back(argv[0]);
    int cards = 0;   // --gpu: the first the main GPU, the others the split's in order (their settings are a tier's)
    int autos = 0;
    bool auto_first = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--main-gpu") == 0 && i + 1 < argc) {
            q.main_gpu = argv[++i];
            continue;
        }
        if ((std::strcmp(argv[i], "--second-gpu") == 0 || std::strcmp(argv[i], "--tier-gpus") == 0) && i + 1 < argc) {
            q.second = argv[++i];
            continue;
        }
        if (std::strcmp(argv[i], "--gpu") == 0 && i + 1 < argc) {
            lamina::program::GpuCard c;
            std::string e;
            const char* text = argv[++i];
            if (!lamina::program::parse_gpu_card(text, c, e)) {
                q.bad = "--gpu " + e;
                continue;
            }
            if (lamina::program::gpu_card_text(c) != c.name)
                q.skipped += (q.skipped.empty() ? "" : "; ") + lamina::program::gpu_card_text(c);
            {
                std::string no;
                if (c.has_vram_only()) no += ",vram-only";
                if (c.prefetch) no += ",prefetch";
                if (c.sleep) no += ",sleep";
                if (cards == 0 && c.layers) no += ",layers";
                if (!no.empty()) q.ink_unused += (q.ink_unused.empty() ? "" : "; ") + c.name + " " + no.substr(1);
            }
            autos += c.name == "auto";
            auto_first = auto_first || (cards == 0 && c.name == "auto");
            if (cards == 0) {
                q.main_cache = c.cache_bytes;
                q.main_reserve = c.reserve;
                q.main_clocks = c.clocks.value_or("");
                q.main_slots = c.cache_slots.has_value();
            }
            if (cards++ == 0) q.main_gpu = c.name == "auto" ? "0" : c.name;
            if (c.name == "auto") q.second = "auto";
            else if (cards > 1) q.second += (q.second.empty() ? "" : ",") + c.name;
            if (cards > 1) {
                q.clocks.push_back(c.clocks.value_or(""));
                q.reserves.push_back(c.reserve);
                q.layers.push_back(c.layers);
                q.caches.push_back(c.has_cache() ? 1 : 0);
            }
            continue;
        }
        if ((std::strcmp(argv[i], "--native") == 0 || std::strcmp(argv[i], "--model") == 0) && i + 1 < argc)
            model = argv[i + 1];
        q.rest.push_back(argv[i]);
    }
    if (autos > 0 && cards > (auto_first ? 1 : 2) && q.bad.empty())   // (the cards' settings follow their places)
        q.bad = "--gpu: auto stands for every other GPU: name the cards after the first, or give auto alone after it";
    if (model.empty()) return q;
    try {
        const lamina::GgufFile meta(lamina::gguf_split_paths(model).front());
        q.family = lamina::model_family(meta);
    } catch (const std::exception&) {}
    return q;
}

/// The switch's main GPU (--main-gpu, or the first --gpu card) made current: its CUDA ordinal, or -1 (said)
int switch_main_gpu(const std::string& main_gpu, const std::vector<lamina::core::GpuInfo>& gpus) {
    std::string why;
    const int dev = lamina::core::find_gpu(gpus, main_gpu, why);
    if (dev < 0 || cudaSetDevice(dev) != cudaSuccess) {
        std::fprintf(stderr, "lamina: --main-gpu %s: %s\n", main_gpu.c_str(), dev < 0 ? why.c_str() : "cannot use it");
        return -1;
    }
    if (const std::string no = lamina::core::gpu_unusable(dev); !no.empty()) {
        std::fprintf(stderr, "lamina: --main-gpu %s: GPU %d, %s: %s\n", main_gpu.c_str(), dev,
                     gpus[(size_t) dev].name.c_str(), no.c_str());
        return -1;
    }
    return dev;
}

int qwen35_switch(Qwen35Line& q) {
    if (!q.skipped.empty())
        std::fprintf(stderr,
                     "lamina: %s: this model's path takes the cards' order alone, their settings left out "
                     "(its own flags size its experts' VRAM)\n",
                     q.skipped.c_str());
    const lamina::ModelFamily family = q.family;
    const std::string main_gpu = q.main_gpu, second = q.second;
    std::vector<char*>& rest = q.rest;
    const std::vector<lamina::core::GpuInfo> gpus = lamina::core::list_gpus();
    std::string why;
    const int dev = switch_main_gpu(main_gpu, gpus);
    if (dev < 0) return 2;
    // --second-gpu: the GPUs a layer split may add, in order (names or numbers, comma-separated, a name no visible GPU
    // answers to left out; auto: every other GPU in CUDA's order; -1: none), each one this build's kernels do not run
    // on left out; the engine takes the fewest that hold the model
    std::vector<int> stages{dev};
    auto usable = [&](int d) {   // (the split's GPUs have their contexts made here, as its sizing would)
        const std::string no = lamina::core::gpu_unusable(d);
        if (!no.empty())
            std::fprintf(stderr, "lamina: GPU %d, %s, left out: %s\n", d, gpus[(size_t) d].name.c_str(), no.c_str());
        return no.empty();
    };
    if (second == "auto") {
        for (const lamina::core::GpuInfo& gi : gpus)
            if (gi.ordinal != dev && usable(gi.ordinal)) stages.push_back(gi.ordinal);
    } else if (!second.empty() && second != "-1") {
        for (size_t at = 0; at <= second.size();) {
            const size_t comma = std::min(second.find(',', at), second.size());
            const std::string name = second.substr(at, comma - at);
            at = comma + 1;
            const int d = lamina::core::find_gpu(gpus, name, why);
            const bool twice = d >= 0 && std::find(stages.begin(), stages.end(), d) != stages.end();
            if (d < 0 || twice) {
                std::fprintf(stderr, "lamina: --second-gpu %s left out: %s\n", name.c_str(),
                             d < 0 ? why.c_str() : "the split has that GPU already");
                continue;
            }
            if (usable(d)) stages.push_back(d);
        }
    }
    cudaSetDevice(dev);   // current again for the path's first CUDA call
    std::string list, more, flag = "--stage-gpus";
    for (const int d : stages) {
        list += (list.empty() ? "" : ",") + std::to_string(d);
        if (d != dev)
            more += (more.empty() ? "" : ", ") + std::string("GPU ") + std::to_string(d) + ", " + gpus[(size_t) d].name;
    }
    if (stages.size() > 1) {
        rest.push_back(flag.data());
        rest.push_back(list.data());
    }
    std::fprintf(stderr, "lamina: %s (%s) on GPU %d, %s%s%s\n", lamina::model_family_title(family),
                 lamina::model_family_arch(family), dev, gpus[(size_t) dev].name.c_str(),
                 more.empty() ? "" : "; a layer split may add ", more.c_str());
    return lamina::qwen35::run((int) rest.size(), rest.data());
}

/// The cards after the first as CUDA ordinals, in their order (`auto`: every other usable GPU); one that is not found,
/// is the main GPU, is named twice or cannot run the engine is left out with a line.  `which`: each kept card's place
/// in the list (-1: one of `auto`'s).
std::vector<int> other_cards(const Qwen35Line& q, const std::vector<lamina::core::GpuInfo>& gpus, int dev,
                             std::vector<int>* which = nullptr) {
    std::vector<int> out;
    std::string why;
    if (q.second.empty() || q.second == "-1") return out;
    int at_card = 0;
    for (size_t at = 0; at <= q.second.size(); ++at_card) {
        const size_t comma = std::min(q.second.find(',', at), q.second.size());
        const std::string name = q.second.substr(at, comma - at);
        at = comma + 1;
        if (name == "auto") {
            for (const lamina::core::GpuInfo& gi : gpus)
                if (gi.ordinal != dev && lamina::core::gpu_unusable(gi.ordinal).empty()) {
                    out.push_back(gi.ordinal);
                    if (which) which->push_back(-1);
                }
            continue;
        }
        const int d = lamina::core::find_gpu(gpus, name, why);
        const bool twice = d >= 0 && std::find(out.begin(), out.end(), d) != out.end();
        if (d < 0 || d == dev || twice) {
            std::fprintf(stderr, "lamina: --second-gpu %s left out: %s\n", name.c_str(),
                         d < 0      ? why.c_str()
                         : d == dev ? "it is the main GPU"
                                    : "it is named twice");
            continue;
        }
        if (const std::string no = lamina::core::gpu_unusable(d); !no.empty()) {
            std::fprintf(stderr, "lamina: GPU %d, %s, left out: %s\n", d, gpus[(size_t) d].name.c_str(), no.c_str());
            continue;
        }
        out.push_back(d);
        if (which) which->push_back(at_card);
    }
    return out;
}
std::string ordinals(const std::vector<int>& devs) {
    std::string list;
    for (const int d : devs) list += (list.empty() ? "" : ",") + std::to_string(d);
    return list;
}

#if defined(LAMINA_HAVE_HY4)
/// Hy4-flash's switch: the exit code of src/hy4's path, run on --main-gpu (the
/// first --gpu card's), made current before its first CUDA call;
/// the other cards' GPUs go to it as --tier-gpus (their order).
int hy4_switch(Qwen35Line& q) {
    const std::vector<lamina::core::GpuInfo> gpus = lamina::core::list_gpus();
    const int dev = switch_main_gpu(q.main_gpu, gpus);
    if (dev < 0) return 2;
    std::string list = ordinals(other_cards(q, gpus, dev));
    cudaSetDevice(dev);
    std::vector<char*>& rest = q.rest;
    std::string flag = "--tier-gpus";
    if (!list.empty()) {
        rest.push_back(flag.data());
        rest.push_back(list.data());
    }
    std::fprintf(stderr, "lamina: Hy4-flash (hy_v4_flash) on GPU %d, %s%s%s\n", dev, gpus[(size_t) dev].name.c_str(),
                 list.empty() ? "" : "; expert tiers on GPUs ", list.c_str());
    return lamina::hy4::run((int) rest.size(), rest.data());
}
#endif

#if defined(LAMINA_HAVE_INKLING2)
/// Inkling 2's switch: the exit code of src/inkling2's path, run on --main-gpu (the
/// first --gpu card's), made current before its first CUDA call; the other cards hold the cells the main one has no
/// room for: --cell-gpus (their order), their clocks= as --cell-gpu-clocks, their reserve= as
/// --cell-gpu-reserve-mib.  The first card's cache=SIZE is the VRAM its experts take at most (--expert-cache), its
/// reserve= what it leaves free (--vram-reserve-mib).  A later card with layers=A-B runs those layers (--layer-split,
/// its clocks= as --layer-split-clocks, its reserve= as --layer-split-reserve-mib) instead of being a cell card; a
/// later card with cache= holds the RAM experts' copies and computes a decode forward's entries of them (--expert-gpu);
/// the first card's clocks= (--main-gpu-clocks) locks it while a request runs.
int inkling2_switch(Qwen35Line& q) {
    if (!q.ink_unused.empty())
        std::fprintf(stderr, "lamina: --gpu: this model's path does not take %s: left out\n", q.ink_unused.c_str());
    const std::vector<lamina::core::GpuInfo> gpus = lamina::core::list_gpus();
    const int dev = switch_main_gpu(q.main_gpu, gpus);
    if (dev < 0) return 2;
    std::vector<int> which;
    std::vector<int> others = other_cards(q, gpus, dev, &which);
    std::string split, split_clocks, split_keep;   // the card with layers=: the split's
    for (size_t i = 0; i < others.size(); ++i) {
        const int w = which[i];
        if (w < 0 || (size_t) w >= q.layers.size() || !q.layers[(size_t) w]) continue;
        if (!split.empty()) {
            std::fprintf(
                stderr, "lamina: --gpu: one card takes layers= (this model's path splits its layers over two cards)\n");
            return 2;
        }
        split = std::to_string(others[i]) + ":" + std::to_string(q.layers[(size_t) w]->first) + "-" +
                std::to_string(q.layers[(size_t) w]->second);
        if ((size_t) w < q.clocks.size()) split_clocks = q.clocks[(size_t) w];
        if ((size_t) w < q.reserves.size() && q.reserves[(size_t) w])
            split_keep = std::to_string(*q.reserves[(size_t) w] >> 20);
        others.erase(others.begin() + (long) i);
        which.erase(which.begin() + (long) i);
        --i;
    }
    std::string xgpu;   // the card with cache=: the RAM experts'
    for (size_t i = 0; i < others.size(); ++i) {
        const int w = which[i];
        if (w < 0 || (size_t) w >= q.caches.size() || !q.caches[(size_t) w]) continue;
        if (!xgpu.empty()) {
            std::fprintf(stderr, "lamina: --gpu: one card takes cache= (the RAM experts' copies)\n");
            return 2;
        }
        xgpu = std::to_string(others[i]);
        others.erase(others.begin() + (long) i);
        which.erase(which.begin() + (long) i);
        --i;
    }
    std::string list = ordinals(others), locks, keeps;
    bool any = false, any_keep = false;
    for (const int w : which) {
        const std::string c =
            w >= 0 && (size_t) w < q.clocks.size() && !q.clocks[(size_t) w].empty() ? q.clocks[(size_t) w] : "0";
        any = any || c != "0";
        locks += (locks.empty() ? "" : ",") + c;
        const bool has = w >= 0 && (size_t) w < q.reserves.size() && q.reserves[(size_t) w].has_value();
        any_keep = any_keep || has;
        keeps += (keeps.empty() ? "" : ",") + (has ? std::to_string(*q.reserves[(size_t) w] >> 20) : std::string("-"));
    }
    cudaSetDevice(dev);
    auto given = [&](const char* flag) {
        for (const char* a : q.rest)
            if (std::strcmp(a, flag) == 0) return true;
        return false;
    };
    std::string f_cache = "--expert-cache", v_cache, f_reserve = "--vram-reserve-mib", v_reserve;
    if (q.main_cache && !given("--expert-cache")) {
        v_cache = lamina::program::size_text(*q.main_cache);
        q.rest.push_back(f_cache.data());
        q.rest.push_back(v_cache.data());
    }
    if (q.main_reserve && !given("--vram-reserve-mib")) {
        v_reserve = std::to_string(*q.main_reserve >> 20);
        q.rest.push_back(f_reserve.data());
        q.rest.push_back(v_reserve.data());
    }
    if (q.main_slots)
        std::fprintf(stderr,
                     "lamina: this model's path takes a card's cache= as a size (cache=6G), not a number of slots: "
                     "left out\n");
    std::string f_split = "--layer-split", f_split_clocks = "--layer-split-clocks",
                f_split_keep = "--layer-split-reserve-mib";
    if (!split.empty() && !given("--layer-split")) {
        q.rest.push_back(f_split.data());
        q.rest.push_back(split.data());
        if (!split_clocks.empty() && split_clocks != "0") {
            q.rest.push_back(f_split_clocks.data());
            q.rest.push_back(split_clocks.data());
        }
        if (!split_keep.empty()) {
            q.rest.push_back(f_split_keep.data());
            q.rest.push_back(split_keep.data());
        }
    }
    std::string f_main_clocks = "--main-gpu-clocks";
    if (!q.main_clocks.empty() && q.main_clocks != "0" && !given("--main-gpu-clocks")) {
        q.rest.push_back(f_main_clocks.data());
        q.rest.push_back(q.main_clocks.data());
    }
    std::string f_xgpu = "--expert-gpu";
    if (!xgpu.empty() && !given("--expert-gpu")) {
        q.rest.push_back(f_xgpu.data());
        q.rest.push_back(xgpu.data());
    }
    std::string f_gpus = "--cell-gpus", f_locks = "--cell-gpu-clocks", f_keeps = "--cell-gpu-reserve-mib";
    if (!list.empty()) {
        q.rest.push_back(f_gpus.data());
        q.rest.push_back(list.data());
        if (any) {
            q.rest.push_back(f_locks.data());
            q.rest.push_back(locks.data());
        }
        if (any_keep) {
            q.rest.push_back(f_keeps.data());
            q.rest.push_back(keeps.data());
        }
    }
    std::fprintf(stderr, "lamina: %s (%s) on GPU %d, %s%s%s\n", lamina::model_family_title(q.family),
                 lamina::model_family_arch(q.family), dev, gpus[(size_t) dev].name.c_str(),
                 list.empty()                          ? ""
                 : list.find(',') == std::string::npos ? "; cells also on GPU "
                                                       : "; cells also on GPUs ",
                 list.c_str());
    return lamina::inkling2::run((int) q.rest.size(), q.rest.data());
}
#endif

}   // namespace
#endif

int lamina::program::family_switch(int argc, char** argv) {
#if defined(LAMINA_HAVE_QWEN35)
    Qwen35Line q = qwen35_line(argc, argv);
    if (!q.qwen35() && !q.hy4() && !q.inkling2()) return kOwnFile;
    if (!q.bad.empty()) {
        std::fprintf(stderr, "lamina: %s\n", q.bad.c_str());
        return 2;
    }
#if defined(LAMINA_HAVE_HY4)
    if (q.hy4()) return hy4_switch(q);
#endif
#if defined(LAMINA_HAVE_INKLING2)
    if (q.inkling2()) return inkling2_switch(q);
#endif
    if (q.qwen35()) return qwen35_switch(q);
#else
    (void) argc, (void) argv;
#endif
    return kOwnFile;
}

int lamina::program::family_check(int argc, char** argv) {
#if defined(LAMINA_HAVE_QWEN35)
    Qwen35Line q = qwen35_line(argc, argv);
    if (!q.qwen35() && !q.hy4() && !q.inkling2()) return kOwnFile;
    if (!q.bad.empty()) {
        std::fprintf(stderr, "lamina: %s\n", q.bad.c_str());
        return 2;
    }
#if defined(LAMINA_HAVE_HY4)
    if (q.hy4()) return lamina::hy4::check_args((int) q.rest.size(), q.rest.data());
#endif
#if defined(LAMINA_HAVE_INKLING2)
    if (q.inkling2()) return lamina::inkling2::check_args((int) q.rest.size(), q.rest.data());
#endif
    if (q.qwen35()) return lamina::qwen35::check_args((int) q.rest.size(), q.rest.data());
#else
    (void) argc, (void) argv;
#endif
    return kOwnFile;
}
