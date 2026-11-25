// lamina-server -m's config pieces that need no model and no GPU (src/server/model_config.hpp): the preset by the
// header's architecture and quantizer, the main GPU and the image encoder's, a GPU's name in a config, the cards a
// config's --main-gpu and --second-gpu name (none for a --main-gpu the engine would not find), the command
// line's engine options in place of the config's, image projectors told apart by their width, a config written
// over an earlier one, an earlier config's GPU flags as cards, and the Settings view's reserve on the main card.  The projectors are small GGUF headers written here.  (CPU only.)
#include "config.hpp"
#include "model_config.hpp"
#include "process.hpp"
#include "runconfig.hpp"

#include "../core/gguf_fixture.hpp"
#include "lamina/inkling2/geometry.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using lamina::json;
namespace fs = std::filesystem;

static int fails = 0;
#define CHECK(c)                                                                  \
    do {                                                                          \
        if (!(c)) {                                                               \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);   \
            ++fails;                                                              \
        }                                                                         \
    } while (0)

namespace {

/// a GGUF v3 header without tensors: string, bool and uint32 keys
struct Header {
    std::vector<std::pair<std::string, std::string>> strs;
    std::vector<std::pair<std::string, bool>> bools;
    std::vector<std::pair<std::string, uint32_t>> u32s;
    void write(const fs::path& path, size_t pad_to = 0) const {
        std::string b = "GGUF";
        auto u32 = [&](uint32_t v) { b.append((const char*) &v, 4); };
        auto u64 = [&](uint64_t v) { b.append((const char*) &v, 8); };
        auto str = [&](const std::string& s) {
            u64(s.size());
            b += s;
        };
        u32(3);
        u64(0);
        u64(strs.size() + bools.size() + u32s.size());
        for (const auto& [k, v] : strs) {
            str(k);
            u32(8);
            str(v);
        }
        for (const auto& [k, v] : bools) {
            str(k);
            u32(7);
            b += (char) (v ? 1 : 0);
        }
        for (const auto& [k, v] : u32s) {
            str(k);
            u32(4);
            u32(v);
        }
        while (b.size() % 32 != 0 || b.size() < pad_to) b += '\0';
        std::ofstream(path, std::ios::binary) << b;
    }
};

void projector(const fs::path& path, uint32_t dim, size_t size = 0, const char* type = "qwen3vl_merger") {
    Header h;
    h.strs = {{"general.architecture", "clip"}, {"clip.projector_type", type}};
    h.bools = {{"clip.has_vision_encoder", true}};
    h.u32s = {{"clip.vision.projection_dim", dim}};
    h.write(path, size);
}

lamina::GpuInfo gpu(int index, const char* name, const char* uuid, int64_t mib, int cc, bool display) {
    lamina::GpuInfo g{index, name, "", uuid};
    g.memory_mib = mib;
    g.cc = cc;
    g.display = display;
    return g;
}

std::vector<std::string> args(const json& cfg) {
    std::vector<std::string> out;
    for (const auto& a : cfg["args"]) out.push_back(a.get<std::string>());
    return out;
}

}  // namespace

int main() {
    // the presets: by architecture, and for Flash-Next by quantizer
    CHECK(lamina::preset_name("qwen4exp", "ISTA DASLab") == "iq3_s");
    CHECK(lamina::preset_name("qwen4exp", "Unsloth") == "ud-q4_k_xl");
    CHECK(lamina::preset_name("qwen4exp", "Agention") == "ud-q4_k_xl");
    CHECK(lamina::preset_name("qwen4exp", "") == "ud-q4_k_xl");
    CHECK(lamina::preset_name("qwen35moe", "Unsloth") == "qwen36-ud-q4_k_xl");
    CHECK(lamina::preset_name("qwen35", "") == "qwen38-27b-ud-q6_k_m");
    CHECK(lamina::preset_name("hy_v4_flash", "") == "hy4-flash-q4_k_m");
    CHECK(lamina::preset_name("inkling2", "Unsloth") == "inkling2-mini-ud-iq4_xs");
    CHECK(lamina::preset_name("inkling2_loop", "") == "inkling2-mini-ud-iq4_xs");
    CHECK(lamina::preset_name("llama", "Unsloth").empty());
    for (const char* name : {"iq3_s", "ud-q4_k_xl", "qwen36-ud-q4_k_xl", "qwen38-27b-ud-q6_k_m", "hy4-flash-q4_k_m",
                             "inkling2-mini-ud-iq4_xs"}) {
        const json p = lamina::preset(name);   // built in from examples/
        CHECK(!lamina::model_file(p).empty() && lamina::gpu_cards(p).has_value());   // "model", the "gpu" cards
        const std::vector<std::string> a = lamina::engine_args(p);
        CHECK(std::find(a.begin(), a.end(), "--native") != a.end() && std::find(a.begin(), a.end(), "--gpu") != a.end());
        CHECK(p["port"].is_number_integer());
        // its pictures: on a GPU, in VRAM borrowed from the engine, on the card the PC's choice gives it (none named)
        const json v = p.value("vision", json::object());
        CHECK(v.value("gpu", json()) == true && v.value("lend", json()) == true && !v.contains("cuda_device") &&
              v.contains("threads"));
    }

    // the main GPU: no display first, then the most VRAM, then the lower number; compute capability 8.0 or newer
    const std::vector<lamina::GpuInfo> pc = {gpu(0, "NVIDIA GeForce RTX 5070 Ti", "GPU-aaa", 16303, 120, true),
                                             gpu(1, "NVIDIA GeForce RTX 3090", "GPU-bbb", 24576, 86, false),
                                             gpu(2, "NVIDIA GeForce RTX 3090", "GPU-ccc", 24576, 86, false)};
    CHECK(lamina::pick_main_gpu(pc, {}) == 1);
    CHECK(lamina::pick_main_gpu(pc, {0, 2}) == 2);
    CHECK(lamina::pick_main_gpu(pc, {0}) == 0);   // only the display's card is left: it runs the model
    CHECK(lamina::pick_main_gpu({}, {}) == -1);
    const std::vector<lamina::GpuInfo> old = {gpu(0, "Tesla V100", "GPU-v", 32768, 70, false),
                                              gpu(1, "NVIDIA GeForce RTX 4080", "GPU-x", 16376, 89, true)};
    CHECK(lamina::pick_main_gpu(old, {}) == 1);
    CHECK(lamina::pick_main_gpu({gpu(0, "Tesla V100", "GPU-v", 32768, 70, false)}, {}) == -1);
    CHECK(lamina::gpu_spec(pc[0], pc) == "NVIDIA GeForce RTX 5070 Ti");
    CHECK(lamina::gpu_spec(pc[1], pc) == "GPU-bbb");   // two cards answer to that name

    // the image encoder's GPU: besides the main one, no display first, then the least VRAM, then the higher number
    CHECK(lamina::pick_vision_gpu(pc, 1) == 2);   // this development PC's: the other 3090, not the display's 5070 Ti
    CHECK(lamina::pick_vision_gpu(pc, 2) == 1);
    const std::vector<lamina::GpuInfo> three = {gpu(0, "NVIDIA GeForce RTX 3090", "GPU-a", 24576, 86, false),
                                                gpu(1, "NVIDIA GeForce RTX 4060 Ti", "GPU-b", 16380, 89, false),
                                                gpu(2, "NVIDIA GeForce RTX 4060 Ti", "GPU-c", 16380, 89, false)};
    CHECK(lamina::pick_vision_gpu(three, 0) == 2);   // the least VRAM; of two alike the higher number
    CHECK(lamina::pick_vision_gpu({pc[0], pc[1]}, 1) == 0);   // only a display's card is left besides the main one
    CHECK(lamina::pick_vision_gpu({pc[1]}, 1) == 1);   // one card: the main one
    CHECK(lamina::pick_vision_gpu(old, 1) == 1);   // the V100 is too old: the main card
    CHECK(lamina::pick_vision_gpu(pc, -1) == 2);   // the main card unknown: the least VRAM without a display
    CHECK(lamina::pick_vision_gpu({}, -1) == -1);

    // the cards the engine works on, as it reads --main-gpu: one it cannot find gives none and why, not a guess
    auto on = [&](std::vector<std::string> a, const std::vector<lamina::GpuInfo>& gpus, std::string* why = nullptr) {
        a.insert(a.begin(), {"--native", "m.gguf"});
        return lamina::engine_gpus(json{{"args", a}}, gpus, why);
    };
    std::string why;
    CHECK((on({"--main-gpu", "GPU-bbb", "--second-gpu", "auto"}, pc) == std::vector<int>{1, 0, 2}));
    CHECK((on({"--main-gpu", "RTX 5070", "--second-gpu", "GPU-ccc"}, pc) == std::vector<int>{0, 2}));
    CHECK((on({"--second-gpu", "auto"}, pc) == std::vector<int>{0, 1, 2}));   // the engine's default, GPU 0
    CHECK((on({"--main-gpu", "2", "--second-gpu", "GPU-dead"}, pc, &why) == std::vector<int>{2}) && why.empty());
    CHECK(on({"--main-gpu", "RTX 3090", "--second-gpu", "auto"}, pc, &why).empty());
    CHECK(why == "--main-gpu RTX 3090: several GPUs match RTX 3090 (name one by its UUID): 1 NVIDIA GeForce RTX 3090 "
                 "GPU-bbb; 2 NVIDIA GeForce RTX 3090 GPU-ccc");
    CHECK(on({"--main-gpu", "GPU-dead"}, pc, &why).empty() && why == "--main-gpu GPU-dead: no GPU is GPU-dead");
    CHECK(on({"--main-gpu", "5"}, pc, &why).empty() && why == "--main-gpu 5: no GPU 5 (nvidia-smi lists 3) or GPU named so");
    CHECK((on({"--main-gpu", "1"}, {}) == std::vector<int>{1}));   // nvidia-smi lists nothing: the number as given
    CHECK((lamina::engine_gpus(json{{"gpu", "0,2"}, {"args", {"--main-gpu", "GPU-dead"}}}, pc) == std::vector<int>{0, 2}));

    // the command line's engine options in place of the config's, with their values and the other form's
    json cfg = {{"args", {"--native", "m.gguf", "--kv", "int8", "--kv-grow", "--second-gpu", "-1",
                          "--control-vector-layer-range", "4", "44", "--hc-q8"}}};
    lamina::apply_engine_options(cfg, {{"--kv", "q4_0"}, {"--no-kv-grow"}, {"--control-vector-layer-range", "1", "2"},
                                       {"--second-gpu", "auto"}, {"--max-context", "65536"}});
    CHECK((args(cfg) == std::vector<std::string>{"--native", "m.gguf", "--hc-q8", "--kv", "q4_0", "--no-kv-grow",
                                                  "--control-vector-layer-range", "1", "2", "--second-gpu", "auto",
                                                  "--max-context", "65536"}));
    json flip = {{"args", {"--no-kv-grow", "--spec", "4"}}};
    lamina::apply_engine_options(flip, {{"--kv-grow"}});
    CHECK((args(flip) == std::vector<std::string>{"--spec", "4", "--kv-grow"}));

    // projectors: for the model's width only, the smallest first; not one: another architecture, not a GGUF
    std::error_code ec;
    const fs::path dir = fs::temp_directory_path() / ("lamina-model-config-test-" + std::to_string(std::rand()));
    fs::create_directories(dir);
    lamina::ModelFacts m;
    m.path = lamina::u8_str(dir / "model.gguf");
    m.n_embd = 2560;
    Header model;
    model.strs = {{"general.architecture", "qwen4exp"}};
    model.write(dir / "model.gguf");
    projector(dir / "mmproj-BF16.gguf", 2560, 4096);
    projector(dir / "mmproj-Q8_0.gguf", 2560, 2048);
    projector(dir / "MMPROJ-other.gguf", 2048);
    projector(dir / "vision.gguf", 2560);   // not named as a projector: not looked at
    projector(dir / "mmproj-other-type.gguf", 2560, 1024, "llava_mlp");   // a type the encoder does not read
    std::ofstream(dir / "mmproj-broken.gguf") << "not a GGUF";
    CHECK(lamina::projector_problem(lamina::u8_str(dir / "mmproj-Q8_0.gguf"), m).empty());
    CHECK(!lamina::projector_problem(lamina::u8_str(dir / "MMPROJ-other.gguf"), m).empty());
    CHECK(lamina::projector_problem(lamina::u8_str(dir / "mmproj-other-type.gguf"), m).find("llava_mlp") != std::string::npos);
    CHECK(!lamina::projector_problem(lamina::u8_str(dir / "model.gguf"), m).empty());
    CHECK(!lamina::projector_problem(lamina::u8_str(dir / "mmproj-broken.gguf"), m).empty());
    const std::vector<std::string> found = lamina::projectors_beside(m);
    CHECK(found.size() == 2);
    CHECK(found.size() == 2 && fs::path(lamina::u8_path(found[0])).filename() == "mmproj-Q8_0.gguf");
    CHECK(found.size() == 2 && fs::path(lamina::u8_path(found[1])).filename() == "mmproj-BF16.gguf");

    // a config written, then written again over it: the earlier file's own keys stay, the earlier file as .bak
    const std::string path = lamina::u8_str(dir / "lamina-test.json");
    CHECK(lamina::write_config(path, {{"args", {"--native", "a.gguf"}}, {"port", 8081}}).empty());
    json first;
    {
        std::ifstream f(lamina::u8_path(path), std::ios::binary);
        const std::string text((std::istreambuf_iterator<char>(f)), {});
        CHECK(lamina::py_loads(text, first) && first["port"] == 8081);
        CHECK(text.find('\r') == std::string::npos && !text.empty() && text.back() == '\n');
    }
    first["api_key"] = "secret";
    first["port"] = 9000;
    std::ofstream(lamina::u8_path(path), std::ios::binary) << lamina::py_dumps(first, true, false, 2) << "\n";
    const std::vector<std::string> kept = lamina::write_config(path, {{"args", {"--native", "b.gguf"}}, {"port", 8082}});
    CHECK((kept == std::vector<std::string>{"api_key"}));
    json second;
    {
        std::ifstream f(lamina::u8_path(path), std::ios::binary);
        CHECK(lamina::py_loads(std::string((std::istreambuf_iterator<char>(f)), {}), second));
    }
    CHECK(second["port"] == 8082 && second["api_key"] == "secret" && second["args"][1] == "b.gguf");
    CHECK(fs::is_regular_file(dir / "lamina-test.json.bak"));
    // a key the preset has, left out this time (pictures off), is not taken back from the earlier file
    second["vision"] = {{"mmproj", "p.gguf"}};
    std::ofstream(lamina::u8_path(path), std::ios::binary) << lamina::py_dumps(second, true, false, 2) << "\n";
    CHECK((lamina::write_config(path, {{"args", {"--native", "c.gguf"}}}, {"args", "port", "vision"}) ==
           std::vector<std::string>{"api_key"}));
    json third;
    {
        std::ifstream f(lamina::u8_path(path), std::ios::binary);
        CHECK(lamina::py_loads(std::string((std::istreambuf_iterator<char>(f)), {}), third));
    }
    CHECK(!third.contains("vision") && !third.contains("port") && third["api_key"] == "secret");

    // "args" as an object of flags: the engine's words in its order, and back (a list it cannot say stays a list)
    {
        const json obj = {{"kv", "k8v4"}, {"prompt-split", 26}, {"arena-place", true}, {"hc-q8", false},
                          {"control-vector-layer-range", {4, 44}}, {"spec-min-p", 0.5}, {"mtp", nullptr}};
        const json words = {"--kv", "k8v4", "--prompt-split", "26", "--arena-place", "--no-hc-q8",
                            "--control-vector-layer-range", "4", "44", "--spec-min-p", "0.5"};
        CHECK(lamina::args_as_list(obj) == words);
        json back = obj;
        back.erase("mtp");
        CHECK(lamina::args_as_object(words) == back);
        CHECK(!lamina::args_as_object(json{"--kv", "int8", "--kv", "k8v4"}) && !lamina::args_as_object(json{"x", "--kv"}));
        json c = {{"args", obj}};
        lamina::args_to_list(c);
        CHECK(c["args"] == words);
    }
    // "gpu": cards (names, objects of "id" and settings) or the earlier numbers; engine_args puts a --gpu each in place
    // of the args' own GPU names
    {
        const json carded = {{"gpu", {"GPU-75af3083", "RTX 5070 Ti",
                                      {{"id", "GPU-f26e9ba6"}, {"clocks", "1395-2100/9751"}, {"sleep", "55us"}}}},
                             {"args", {"--native", "m.gguf", "--main-gpu", "0", "--kv", "int8"}}};
        CHECK((lamina::gpu_cards(carded) == std::vector<std::string>{"GPU-75af3083", "RTX 5070 Ti",
                                                                     "GPU-f26e9ba6,clocks=1395-2100/9751,sleep=55us"}));
        CHECK(lamina::gpu_list(carded).empty() && (lamina::gpu_list({{"gpu", "0,2"}}) == std::vector<int>{0, 2}) &&
              !lamina::gpu_cards({{"gpu", {0, 2}}}));
        CHECK((lamina::engine_args(carded) ==
               std::vector<std::string>{"--native", "m.gguf", "--kv", "int8", "--gpu", "GPU-75af3083", "--gpu",
                                        "RTX 5070 Ti", "--gpu", "GPU-f26e9ba6,clocks=1395-2100/9751,sleep=55us"}));
        bool refused = false;
        try {
            (void) lamina::gpu_cards({{"gpu", {{{"id", "RTX 3090"}, {"reserve", 2048}}}}});   // a size without its unit
        } catch (const lamina::ConfigError&) {
            refused = true;
        }
        CHECK(refused);
    }
    // the form --write-config writes: the older GPU flags as cards, the other older names renamed, the options at the
    // engine's default left out, --native as "model", "args" an object
    {
        const json earlier = {
            {"args", {"--native", "m.gguf", "--expert-cache", "auto", "--prefill", "16384", "--spec", "4",
                      "--spec-min-p", "0.5", "--adapt-every", "1", "--max-context", "262144", "--kv", "k8v4",
                      "--kv-grow", "--hc-q8", "--arena-place", "--pcie-mode", "kernel", "--pcie-frac", "0",
                      "--second-gpu", "RTX 5070 Ti,GPU-f26e9ba6", "--second-gpu-clocks", "0,1395-2100/9751",
                      "--second-gpu-clocks-hold", "1", "--second-gpu-clocks-lead", "1.0", "--second-gpu-sleep-us",
                      "0,55", "--main-gpu", "GPU-75af3083", "--prompt-split", "26"}},
            {"port", 8081},
            {"lib_dirs", json::array()}};
        const json written = lamina::modern_config(earlier, true);
        CHECK(written["model"] == "m.gguf");
        CHECK(!written.contains("lib_dirs"));   // taken and not read: left out
        CHECK((written["gpu"] == json{"GPU-75af3083", "RTX 5070 Ti",
                                      {{"id", "GPU-f26e9ba6"}, {"clocks", "1395-2100/9751"}, {"sleep", "55us"}}}));
        CHECK((written["args"] == json{{"kv", "k8v4"}, {"arena-place", true}, {"clocks-hold", "1s"},
                                       {"clocks-lead", "1s"}, {"prompt-split", 26}}));
        CHECK(written["port"] == 8081 && written.begin().key() == "model");
        // the engine reads the same: the cards, the renamed options
        const std::vector<std::string> e = lamina::engine_args(written);
        CHECK(std::find(e.begin(), e.end(), "GPU-f26e9ba6,clocks=1395-2100/9751,sleep=55us") != e.end() &&
              std::find(e.begin(), e.end(), "--clocks-lead") != e.end());
        // another model's path keeps its options (its defaults are its own), and an auto tier its per-tier lists
        const json q = lamina::modern_config({{"args", {"--native", "q.gguf", "--max-context", "262144", "--second-gpu",
                                                        "auto", "--second-gpu-gib", "20"}}}, false);
        CHECK(q["args"]["max-context"] == 262144 && q["args"]["second-gpu"] == "auto" && !q.contains("gpu"));
        // nothing left: an empty object, not none (a config without "args" is the preset's)
        const json e2 = lamina::modern_config({{"args", {"--native", "m.gguf", "--spec", "4"}}}, true);
        CHECK(e2["args"].is_object() && e2["args"].empty());
        // the deprecated options a start names: the older config's, in its order; none in the form written
        CHECK((lamina::deprecated_options(earlier, lamina::engine_args(earlier)) ==
               std::vector<std::string>{
                   "--native: the config's \"model\"",
                   "--expert-cache auto: the main GPU's cache= (N slots; 0: the profile's count)",
                   "--second-gpu \"RTX 5070 Ti,GPU-f26e9ba6\": the later --gpu's cards, or auto (-1: none)",
                   "--second-gpu-clocks 0,1395-2100/9751: each tier's clocks= (0: the driver's)",
                   "--second-gpu-clocks-hold 1: --clocks-hold in seconds",
                   "--second-gpu-clocks-lead 1.0: --clocks-lead in seconds",
                   "--second-gpu-sleep-us 0,55: each tier's sleep= in microseconds (0: none)",
                   "--main-gpu GPU-75af3083: the first --gpu's card"}));
        CHECK(lamina::deprecated_options(written, lamina::engine_args(written)).empty());
        // a switch of an object's args; an option given twice, once
        const json r = {{"model", "m.gguf"}, {"args", {{"second-gpu-adapt", false}}}};
        CHECK((lamina::deprecated_options(r, lamina::engine_args(r)) ==
               std::vector<std::string>{"--no-second-gpu-adapt: --no-tier-adapt"}));
        const json twice = {{"model", "m.gguf"}, {"args", {"--vram-reserve-mib", "800", "--vram-reserve-mib", "800"}}};
        CHECK((lamina::deprecated_options(twice, lamina::engine_args(twice)) ==
               std::vector<std::string>{"--vram-reserve-mib 800: the main GPU's reserve= in MiB"}));
        // its form written beside "gpu" numbers: the older names renamed, the GPU flags kept
        const json rw = lamina::modern_config(
            {{"gpu", {0, 1, 2, 3}}, {"args", {"--native", "m.gguf", "--vram-reserve-mib", "800", "--second-gpu-adapt"}}},
            false);
        CHECK((rw["args"] == json{{"vram-reserve-mib", 800}, {"tier-adapt", true}}) && rw["gpu"] == json({0, 1, 2, 3}));
        // the engine's default switch left out, as the options at their default
        CHECK(lamina::modern_config({{"args", {"--native", "m.gguf", "--second-gpu-adapt"}}}, true)["args"].empty());
    }
    // the GPU flags as cards in each config's form: numbers (the cards the engine sees otherwise) as objects, the main
    // GPU's flags without a GPU named on the engine's "0", into "gpu"'s own cards (a card's own setting wins), kept for
    // an auto card
    {
        const json n = lamina::modern_config({{"args", {"--native", "m.gguf", "--main-gpu", "1", "--second-gpu", "0,2"}}}, true);
        CHECK((n["gpu"] == json{{{"id", "1"}}, {{"id", "0"}}, {{"id", "2"}}}) && n["args"].empty());
        CHECK((lamina::engine_cards(n) == std::vector<std::string>{"1", "0", "2"}) && lamina::gpu_list(n).empty());
        const json mixed = lamina::modern_config({{"args", {"--native", "m.gguf", "--main-gpu", "1", "--second-gpu",
                                                            "RTX 5070 Ti"}}}, true);
        CHECK((mixed["gpu"] == json{"1", "RTX 5070 Ti"}));
        const json flags = lamina::modern_config({{"args", {"--native", "m.gguf", "--vram-reserve-mib", "800",
                                                            "--expert-cache", "512", "--vram-pin-gib", "4"}}}, true);
        CHECK((flags["gpu"] == json{{{"id", "0"}, {"cache", "512"}, {"reserve", "800M"}, {"vram-only", "4G"}}}));
        CHECK(flags["args"].empty() && lamina::deprecated_options(flags, lamina::engine_args(flags)).empty());
        // nothing named, nothing set: the engine's defaults, no "gpu"
        const json none = lamina::modern_config({{"args", {"--native", "m.gguf", "--second-gpu", "-1",
                                                           "--vram-reserve-mib", "700", "--second-gpu-gib", "12"}}}, true);
        CHECK(!none.contains("gpu") && none["args"].empty());
        const json carded = lamina::modern_config(
            {{"gpu", {"GPU-75af3083", "RTX 5070 Ti,clocks=0", {{"id", "GPU-f26e9ba6"}, {"clocks", "1395-2100/9751"}}}},
             {"args", {"--native", "m.gguf", "--main-gpu", "GPU-x", "--vram-reserve-mib", "800", "--second-gpu-clocks",
                       "1-2/3", "--second-gpu-sleep-us", "0,55"}}},
            true);
        CHECK((carded["gpu"] == json{{{"id", "GPU-75af3083"}, {"reserve", "800M"}}, {{"id", "RTX 5070 Ti"}, {"clocks", "0"}},
                                     {{"id", "GPU-f26e9ba6"}, {"clocks", "1395-2100/9751"}, {"sleep", "55us"}}}));
        CHECK(carded["args"].empty());
        // "gpu"'s cards as written when they take nothing; the names beside them (which the engine skips) out
        const json as_is = lamina::modern_config({{"gpu", "GPU-75af3083"}, {"args", {"--native", "m.gguf", "--second-gpu",
                                                                                     "auto"}}}, true);
        CHECK(as_is["gpu"] == "GPU-75af3083" && as_is["args"].empty());
        const json first_auto = lamina::modern_config({{"gpu", {"auto"}}, {"args", {"--native", "m.gguf",
                                                                                    "--vram-reserve-mib", "800"}}}, true);
        CHECK((first_auto["gpu"] == json{"auto"}) && (first_auto["args"] == json{{"vram-reserve-mib", 800}}));
    }
    // the Settings view's reserve: the main card's when "gpu" names cards (it wins over the flag), the flag beside
    // numbers; a change goes to the card and takes the flag out
    {
        namespace rc = lamina::runconfig;
        const json c = {{"gpu", {"GPU-75af3083", "RTX 5070 Ti"}}, {"args", {"--native", "m.gguf", "--vram-reserve-mib", "800"}}};
        CHECK(rc::value_of(c, "vram_reserve_mib") == 800);
        const json set = rc::apply(c, {{"vram_reserve_mib", 900}}).first;
        CHECK((set["gpu"] == json{{{"id", "GPU-75af3083"}, {"reserve", "900M"}}, "RTX 5070 Ti"}));
        CHECK((set["args"] == json{"--native", "m.gguf"}) && rc::value_of(set, "vram_reserve_mib") == 900);
        const json cleared = rc::apply(set, {{"vram_reserve_mib", nullptr}}).first;
        CHECK((cleared["gpu"] == json{"GPU-75af3083", "RTX 5070 Ti"}) && rc::value_of(cleared, "vram_reserve_mib").is_null());
        const json listed = rc::apply({{"gpu", {0, 2}}, {"args", {"--native", "m.gguf"}}}, {{"vram_reserve_mib", 900}}).first;
        CHECK((listed["gpu"] == json{0, 2}) && (listed["args"] == json{"--native", "m.gguf", "--vram-reserve-mib", "900"}));
    }

    {   // Inkling 2's drafter in either layout: unsloth's first (the window as the checkpoint counts it) and mainline
        // llama.cpp's (markov_w2, the window one cell more): the same window, the same rings
        auto drafter = [&](const char* name, uint64_t window, bool folded) {
            std::vector<fixture::Kv> kv = {fixture::str("general.architecture", "dflash"),
                                           fixture::u32("dflash.attention.head_count_kv", 8),
                                           fixture::u32("dflash.attention.key_length", 128),
                                           fixture::u32("dflash.attention.sliding_window", window),
                                           fixture::u32s("dflash.attention.sliding_window_pattern", {1, 0, 1})};
            std::vector<fixture::Tensor> ts = {{"markov_w1.weight", {4, 8}, 0, 1}};
            ts.push_back(folded ? fixture::Tensor{"markov_w2.weight", {4, 8}, 0, 2} : fixture::Tensor{"dspark_hidden_out.weight", {4, 8}, 0, 2});
            fixture::write(dir / name, kv, ts);
            return lamina::GgufFile((dir / name).string());
        };
        const lamina::GgufFile old_f = drafter("dspark-old.gguf", 1024, false), new_f = drafter("dspark-new.gguf", 1025, true);
        CHECK(!lamina::inkling2::drafter_folded(old_f) && lamina::inkling2::drafter_folded(new_f));
        CHECK(lamina::inkling2::drafter_window(old_f) == 1024 && lamina::inkling2::drafter_window(new_f) == 1024);
        for (const bool q8 : {false, true})
            CHECK(lamina::inkling2::drafter_rings_bytes(old_f, 32768, q8) == lamina::inkling2::drafter_rings_bytes(new_f, 32768, q8) &&
                  lamina::inkling2::drafter_rings_bytes(new_f, 32768, q8) > 0);

        // and the checkpoint's own file (mtp.safetensors, the settings its config.json has: those of the GGUFs above):
        // the same rings; its matrices weighed as Q8_0, its Markov table as it is; found beside a model that has no
        // dspark GGUF, after one that has
        const fs::path md = dir / "ink", mtp = md / "mtp.safetensors";
        fs::create_directories(md, ec);
        auto safetensors = [](const fs::path& p, const std::string& head, size_t data) {
            std::ofstream o(p, std::ios::binary);
            const uint64_t n = head.size();
            o.write(reinterpret_cast<const char*>(&n), 8);
            o << head << std::string(data, '\0');
        };
        safetensors(mtp,
                    "{\"__metadata__\":{\"format\":\"pt\"},\"layers.0.self_attn.q_proj.weight\":{\"dtype\":\"BF16\","
                    "\"shape\":[64,32],\"data_offsets\":[0,4096]},\"markov_head.markov_w1.weight\":{\"dtype\":\"BF16\","
                    "\"shape\":[16,64],\"data_offsets\":[4096,6144]}}",
                    6144);
        const std::string ms = mtp.string();
        CHECK(lamina::inkling2::mtp_file(ms) && !lamina::inkling2::mtp_file((dir / "dspark-old.gguf").string()));
        for (const bool q8 : {false, true})
            CHECK(lamina::inkling2::drafter_file_rings_bytes(ms, 32768, q8) == lamina::inkling2::drafter_rings_bytes(old_f, 32768, q8));
        const uint64_t file = (uint64_t) fs::file_size(mtp, ec);
        CHECK(lamina::inkling2::drafter_weights_bytes(ms, file) == 2048 + (file - 2048) * 34 / 64);
        CHECK(lamina::inkling2::drafter_weights_bytes((dir / "dspark-old.gguf").string(), 1000) == 1000);
        CHECK(lamina::inkling2::mtp_tensor_name("blk.2.attn_k_ctx_norm.weight") == "layers.2.self_attn.k_norm_ctx.weight" &&
              lamina::inkling2::mtp_tensor_name("dspark_fuse.weight") == "fuse_weight" &&
              lamina::inkling2::mtp_tensor_name("blk.0.ctx_mix.weight").empty());
        json c = {{"model", (md / "m.gguf").string()}, {"args", json::array()}};
        CHECK(lamina::resolve_drafter(c, "inkling2_loop").find("mtp.safetensors") != std::string::npos &&
              c["args"].size() == 2 && c["args"][1].get<std::string>().find("mtp.safetensors") != std::string::npos);
        fs::copy_file(dir / "dspark-old.gguf", md / "dspark-old.gguf", ec);
        json d = {{"model", (md / "m.gguf").string()}, {"args", json::array()}};
        lamina::resolve_drafter(d, "inkling2_loop");
        CHECK(d["args"].size() == 2 && d["args"][1].get<std::string>().find("dspark-old.gguf") != std::string::npos);
        // (a safetensors file without the Markov head's table is not taken)
        safetensors(mtp, "{\"layers.0.self_attn.q_proj.weight\":{\"dtype\":\"BF16\",\"shape\":[64,32],\"data_offsets\":[0,4096]}}", 4096);
        fs::remove(md / "dspark-old.gguf", ec);
        json e = {{"model", (md / "m.gguf").string()}, {"args", json::array()}};
        CHECK(lamina::resolve_drafter(e, "inkling2_loop").empty() && e["args"].empty());
    }

    fs::remove_all(dir, ec);
    if (fails == 0) std::printf("model_config_test: ok\n");
    return fails == 0 ? 0 : 1;
}
