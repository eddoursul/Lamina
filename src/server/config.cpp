// src/server/config.cpp - the run config as the server reads it (config.hpp).
#include "config.hpp"

#include "lamina/artifact/draft_layer.hpp"
#include "lamina/core/expert_cache.hpp"
#include "lamina/core/gpu_select.hpp"
#include "lamina/program/engine.hpp"
#include "lamina/core/gpu_select.hpp"
#include "lamina/program/gpu_cards.hpp"
#include "log.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <regex>
#include <set>

namespace lamina {
namespace {

constexpr int kParallelMax = 8;   // the engine's batch window holds at most 8 rows (kVerifyMaxT)

void set_env(const std::string& k, const std::string& v) {
#ifdef _WIN32
    _putenv_s(k.c_str(), v.c_str());
#else
    setenv(k.c_str(), v.c_str(), 1);
#endif
}

std::string arg_str(const json& a) { return a.is_string() ? a.get<std::string>() : py_dumps(a); }

std::vector<std::string> args_of(const json& cfg) {
    std::vector<std::string> out;
    for (const auto& a : args_as_list(jget(cfg, "args"))) out.push_back(arg_str(a));
    return out;
}

bool has(const std::vector<std::string>& args, const char* flag) { return std::find(args.begin(), args.end(), flag) != args.end(); }

/// an expert profile's (layers, experts per layer) from its header (tools/make_profile.py's format)
std::optional<std::pair<uint32_t, uint32_t>> profile_shape(const std::string& path) {
    std::string bytes;
    {
        std::ifstream f(u8_path(path), std::ios::binary);
        if (!f) return std::nullopt;
        bytes.assign(std::istreambuf_iterator<char>(f), {});
    }
    const std::streamoff size = (std::streamoff) bytes.size();
    unsigned char head[24];
    if (bytes.size() < sizeof head || std::memcmp(bytes.data(), "STRP", 4) != 0) return std::nullopt;
    std::memcpy(head, bytes.data(), sizeof head);
    auto u32 = [&](int i) { return (uint32_t) head[4 + 4 * i] | (uint32_t) head[5 + 4 * i] << 8 | (uint32_t) head[6 + 4 * i] << 16 | (uint32_t) head[7 + 4 * i] << 24; };
    const uint32_t nl = u32(1), ne = u32(2), n = u32(4);
    if (size < 24 + 4 * (std::streamoff) n) return std::nullopt;
    return std::make_pair(nl, ne);
}

std::string here_path(const json& cfg, const std::string& p) {
    const std::filesystem::path x = u8_path(p);
    if (x.is_absolute() || p == core::kBuiltinExpertProfile) return p;
    const std::string cwd = jstr(cfg, "cwd", ".");
    return u8_str(u8_path(cwd.empty() ? "." : cwd) / x);
}

}  // namespace

json read_config(const std::string& path) {
    std::ifstream f(u8_path(path), std::ios::binary);
    if (!f) throw ConfigError(path + " cannot be read");
    std::string text((std::istreambuf_iterator<char>(f)), {});
    if (starts_with(text, "\xEF\xBB\xBF")) text.erase(0, 3);   // Notepad adds a BOM
    json cfg;
    std::string err;
    if (!py_loads(text, cfg, &err))
        throw ConfigError(path + " is not valid JSON: " + err + ". Check the commas between the entries of \"args\" (each "
                          "option and each value is its own quoted entry, e.g. \"--ple-io\", \"mmap\",) and that no entry is "
                          "followed by a comma at the end of a list.");
    if (!cfg.is_object()) throw ConfigError(path + " is not a JSON object");
    args_to_list(cfg);
    return cfg;
}

json args_as_list(const json& args) {
    json out = json::array();
    if (args.is_array()) return args;
    if (!args.is_object()) return out;
    for (auto it = args.begin(); it != args.end(); ++it) {
        const json& v = it.value();
        if (v.is_null()) continue;
        if (v.is_boolean()) {
            out.push_back((v.get<bool>() ? "--" : "--no-") + it.key());
            continue;
        }
        out.push_back("--" + it.key());
        if (v.is_array())
            for (const json& x : v) out.push_back(arg_str(x));
        else
            out.push_back(arg_str(v));
    }
    return out;
}

std::optional<json> args_as_object(const json& args) {
    if (!args.is_array()) return std::nullopt;
    auto flag = [](const std::string& s) { return s.size() > 2 && starts_with(s, "--"); };
    auto value = [](const std::string& s) -> json {   // a number when the text is one that writes back the same
        json j;
        if (!s.empty() && (std::isdigit((unsigned char) s[0]) || s[0] == '-') && py_loads(s, j) && j.is_number() &&
            py_dumps(j) == s)
            return j;
        return s;
    };
    json out = json::object();
    for (size_t i = 0; i < args.size();) {
        const std::string f = arg_str(args[i]);
        if (!flag(f)) return std::nullopt;   // a value before any flag
        size_t j = i + 1;
        while (j < args.size() && !flag(arg_str(args[j]))) ++j;
        std::string key = f.substr(2);
        json v;
        if (j == i + 1) {
            const bool no = starts_with(key, "no-");
            v = !no;
            if (no) key = key.substr(3);
        } else if (j == i + 2) {
            v = value(arg_str(args[i + 1]));
        } else {
            v = json::array();
            for (size_t k = i + 1; k < j; ++k) v.push_back(value(arg_str(args[k])));
        }
        if (out.contains(key)) return std::nullopt;   // a flag twice: the list says it, the object could not
        out[key] = std::move(v);
        i = j;
    }
    return out;
}

void args_to_list(json& cfg) {
    if (cfg.is_object() && jget(cfg, "args").is_object()) cfg["args"] = args_as_list(cfg["args"]);
}

void split_joined_args(json& cfg) {
    if (!jget(cfg, "args").is_array()) return;
    static const std::regex joined(R"(^(--[A-Za-z0-9][A-Za-z0-9-]*)(?:\s+|=)(\S.*)$)");
    static const std::regex flag(R"(^--[A-Za-z0-9][A-Za-z0-9-]*$)");
    json out = json::array();
    std::vector<std::string> changed;
    for (const auto& a : cfg["args"]) {
        std::smatch m;
        const std::string s = a.is_string() ? a.get<std::string>() : "";
        const bool value_place = !out.empty() && out.back().is_string() && std::regex_match(out.back().get_ref<const std::string&>(), flag);
        if (!a.is_string() || !std::regex_match(s, m, joined) || value_place) {
            out.push_back(a);
            continue;
        }
        out.push_back(m[1].str());
        out.push_back(std::string(strip(m[2].str())));
        changed.push_back(s);
    }
    if (changed.empty()) return;
    cfg["args"] = out;
    std::string list;
    for (const auto& c : changed) list += (list.empty() ? "" : ", ") + py_repr_value(json(c));
    say("[lamina] the config's args hold " + list + " as one element each: split into the flag and its value (write them "
        "as two elements, e.g. \"--ple-io\", \"mmap\")");
}

void lift_reasoning_effort_arg(json& cfg) {
    if (!jget(cfg, "args").is_array()) return;
    json out = json::array(), value;
    const json& args = cfg["args"];
    for (size_t i = 0; i < args.size();) {
        const std::string a = args[i].is_string() ? args[i].get<std::string>() : "";
        if (a == "--reasoning-effort" && i + 1 < args.size()) {
            value = args[i + 1];
            i += 2;
        } else if (starts_with(a, "--reasoning-effort=")) {
            value = a.substr(a.find('=') + 1);
            ++i;
        } else {
            out.push_back(args[i]);
            ++i;
        }
    }
    if (value.is_null()) return;
    cfg["args"] = out;
    if (jget(cfg, "reasoning_effort").is_null()) cfg["reasoning_effort"] = value;
    say("[lamina] '--reasoning-effort' in the config's args is not an engine option: taken out of them. The thinking level "
        "is the config's 'reasoning_effort' key (outside 'args'); it is used as " + py_repr_value(cfg["reasoning_effort"]));
}

namespace {

/// `s` as a shell takes it as one word
std::string shell_word(const std::string& s) { return s.find(' ') == std::string::npos ? s : "\"" + s + "\""; }

/// this platform's script that downloads the draft layer, by its path where it lies (beside the program, as a release
/// has it; in the config's folder or the one above the program's, as in a clone), else by its name
std::string draft_layer_script(const json& cfg) {
    const std::filesystem::path self = u8_path(self_dir());
    std::error_code ec;
    for (const std::filesystem::path& dir : {self, std::filesystem::absolute(u8_path(here_path(cfg, ".")), ec), self.parent_path()}) {
        const std::filesystem::path script = dir / kDraftLayerScript;
        if (std::filesystem::is_regular_file(script, ec)) return shell_word(u8_str(script.lexically_normal()));
    }
#ifdef _WIN32
    return kDraftLayerScript;
#else
    return std::string("./") + kDraftLayerScript;
#endif
}

}  // namespace

std::string resolve_draft_layer(json& cfg, const std::string& arch) {
    if (arch != "qwen4exp" || !jget(cfg, "args").is_array()) return "";
    namespace fs = std::filesystem;
    json& args = cfg["args"];
    std::error_code ec;
    const fs::path model_dir = fs::absolute(u8_path(model_file(cfg)), ec).parent_path().lexically_normal();
    // the warning's parts: why, the folder the download goes to, what the config needs then
    std::string why, into = shell_word(u8_str(model_dir)), then = ": the server takes it from beside the model";
    bool named = false;
    for (size_t i = 0; i + 1 < args.size(); ++i) {
        if (args[i] != "--mtp") continue;
        const std::string name = arg_str(args[i + 1]);
        const fs::path at = fs::absolute(u8_path(here_path(cfg, name)), ec).lexically_normal();
        if (fs::is_regular_file(at, ec)) return "";   // the engine reads it (and refuses a file of another layout)
        const bool folder = fs::is_directory(at, ec);
        std::string gone;   // a folder of the earlier layout: what the engine cannot do without
        for (const char* f : {"dense.txt", "dense.bin", "experts.bin"})
            if (folder && !fs::is_regular_file(at / f, ec)) gone += (gone.empty() ? "" : ", ") + std::string(f);
        if (folder && gone.empty()) return "";
        named = true;
        why = "--mtp " + name + (folder ? " is a folder without " + gone + " (the layer's earlier layout)" : " is not there");
        if (!folder && at.filename() == kDraftLayerFile) {   // download it where the config names it
            into = shell_word(u8_str(at.parent_path()));
            then.clear();
        } else {
            then = ",\n         then take \"--mtp\", \"" + name + "\" out of the config's \"args\": the server takes the file "
                   "from beside the model";
        }
        args.erase(i);   // the engine stops at a file it cannot read
        args.erase(i);
        break;
    }
    if (!named) {   // none named: the layer's file beside the model, else beside the program
        for (const fs::path& dir : {model_dir, u8_path(self_dir())}) {
            const fs::path at = (dir / kDraftLayerFile).lexically_normal();
            if (fs::is_regular_file(at, ec)) {
                args.push_back("--mtp");
                args.push_back(u8_str(at));
                return "[lamina] draft layer: " + u8_str(at);
            }
        }
        why = std::string("no ") + kDraftLayerFile + " beside the model or the program, and the config's \"args\" name no "
              "--mtp";
    }
    return "[lamina] WARNING: no draft layer (" + why + "): generation runs without its drafts, a verify window a "
           "token, and is slower.\n         Download it (0.8 GB) with  " + draft_layer_script(cfg) + " " + into +
           "  (or from\n         " + kDraftLayerUrl + ")" + then;
}

namespace {

/// the "gpu" value's items: numbers, cards' texts (a comma inside one is its settings') or objects; a text of numbers
/// and commas ("0,2") its numbers, a JSON text ("[0, 2]") its items
std::vector<json> gpu_items(const json& g) {
    std::vector<json> out;
    auto text = [&](const std::string& raw) {
        const std::string s(strip(raw));
        if (s.empty()) return;
        if (s[0] == '[' || s[0] == '{') {
            json j;
            if (py_loads(s, j)) {
                if (j.is_array())
                    for (const json& x : j) out.push_back(x);
                else
                    out.push_back(j);
                return;
            }
        }
        const std::vector<std::string> parts = split(s, ",");
        const bool numbers = std::all_of(parts.begin(), parts.end(), [](const std::string& p) {
            const std::string q(strip(p));
            return !q.empty() && std::all_of(q.begin(), q.end(), ::isdigit);
        });
        if (!numbers) {
            out.push_back(s);
            return;
        }
        for (const std::string& p : parts) out.push_back(std::stoi(std::string(strip(p))));
    };
    if (g.is_array()) {
        for (const json& x : g) {
            if (x.is_string()) text(x.get<std::string>());
            else if (!x.is_null()) out.push_back(x);
        }
    } else if (g.is_string()) {
        text(g.get<std::string>());
    } else if (!g.is_null()) {
        out.push_back(g);
    }
    return out;
}

bool numbers_only(const std::vector<json>& items) {
    return std::all_of(items.begin(), items.end(), [](const json& x) { return x.is_number_integer(); });
}

/// a JSON value as a card setting's text: text as it is, a number as JSON writes it
std::string card_value(const json& v) { return v.is_string() ? v.get<std::string>() : py_dumps(v); }

}  // namespace

std::vector<int> gpu_list(const json& cfg) {
    const std::vector<json> items = gpu_items(jget(cfg, "gpu"));
    std::vector<int> out;
    if (!numbers_only(items)) return out;   // cards: gpu_cards
    for (const json& x : items) out.push_back(x.get<int>());
    return out;
}

std::optional<std::vector<std::string>> gpu_cards(const json& cfg) {
    const std::vector<json> items = gpu_items(jget(cfg, "gpu"));
    if (items.empty() || numbers_only(items)) return std::nullopt;
    std::vector<std::string> out;
    for (const json& x : items) {
        std::string text;
        if (x.is_object()) {
            const json& id = jget(x, "id");
            if (id.is_null()) throw ConfigError("\"gpu\": " + py_repr_value(x) + " names no card: {\"id\": \"RTX 3090\", ...}");
            text = card_value(id);
            for (auto it = x.begin(); it != x.end(); ++it)
                if (it.key() != "id") text += "," + it.key() + "=" + card_value(it.value());
        } else {
            text = card_value(x);
        }
        program::GpuCard card;
        std::string why;
        if (!program::parse_gpu_card(text, card, why)) throw ConfigError("\"gpu\": " + why);
        out.push_back(text);
    }
    return out;
}

std::optional<std::vector<json>> card_objects(const json& cfg) {
    const std::optional<std::vector<std::string>> texts = gpu_cards(cfg);
    if (!texts) return std::nullopt;
    std::vector<json> out;
    for (const std::string& t : *texts) {   // "id,key=value,..." as parse_gpu_card took it
        const std::vector<std::string> items = split(t, ",");
        json c = {{"id", std::string(strip(items[0]))}};
        for (size_t i = 1; i < items.size(); ++i) {
            const std::string item(strip(items[i]));
            const size_t eq = item.find('=');
            if (eq != std::string::npos) c[lower(std::string(strip(item.substr(0, eq))))] = std::string(strip(item.substr(eq + 1)));
        }
        out.push_back(std::move(c));
    }
    return out;
}

json cards_json(const std::vector<json>& cards) {
    auto number = [](const json& c) {
        const std::string id = card_value(c["id"]);
        return c.size() == 1 && !id.empty() && std::all_of(id.begin(), id.end(), ::isdigit);
    };
    const bool objects = std::all_of(cards.begin(), cards.end(), number);
    json out = json::array();
    for (const json& c : cards) out.push_back(c.size() == 1 && !objects ? c["id"] : c);
    return out;
}

json gpu_canonical(const json& g) {
    const std::vector<json> items = gpu_items(g);
    if (items.empty()) return g;
    json out = json::array();
    for (const json& x : items) out.push_back(x);
    return out;
}

std::vector<std::string> engine_cards(const json& cfg) {
    if (auto cards = gpu_cards(cfg)) return *cards;
    const std::vector<std::string> args = args_of(cfg);
    std::vector<std::string> out;
    for (size_t i = 0; i + 1 < args.size(); ++i)
        if (args[i] == "--gpu") out.push_back(args[i + 1]);
    if (!out.empty()) return out;
    std::optional<std::string> main;
    std::vector<std::string> tiers;
    for (size_t i = 0; i + 1 < args.size(); ++i) {
        if (args[i] == "--main-gpu") main = args[i + 1];
        if (args[i] == "--second-gpu" || args[i] == "--tier-gpus") {
            tiers.clear();
            for (const std::string& x : split(args[i + 1], ",")) {
                const std::string s(strip(x));
                if (!s.empty() && s[0] != '-') tiers.push_back(s);
            }
        }
    }
    if (!main && tiers.empty()) return out;
    out.push_back(main.value_or("0"));
    out.insert(out.end(), tiers.begin(), tiers.end());
    return out;
}

std::vector<GpuInfo> nvidia_gpus() {
    std::vector<GpuInfo> out;
    try {
        SpawnOptions o;
        o.args = {"nvidia-smi", "--query-gpu=index,name,pci.bus_id,uuid,memory.total,compute_cap,display_active,"
                  "driver_version", "--format=csv,noheader,nounits"};
        o.in = SpawnOptions::null;
        o.out = SpawnOptions::pipe;
        o.err = SpawnOptions::null;
        Process p(o);
        std::string line;
        while (p.read_line(line, 30) == 1) {
            std::vector<std::string> f = split(line, ",");
            for (auto& x : f) x = std::string(strip(x));
            if (f.size() != 8 || f[0].empty() || !std::all_of(f[0].begin(), f[0].end(), ::isdigit)) continue;
            GpuInfo g{std::stoi(f[0]), f[1], f[2], f[3]};
            g.memory_mib = std::atoll(f[4].c_str());
            const size_t dot = f[5].find('.');   // "8.6"; "[N/A]" where the driver does not say
            if (dot != std::string::npos && std::isdigit((unsigned char) f[5][0]))
                g.cc = std::atoi(f[5].c_str()) * 10 + std::atoi(f[5].c_str() + dot + 1);
            g.display = lower(f[6]) == "enabled" || core::gpu_in_desktop_use(f[2]);   // off while a monitor sleeps
            g.driver = f[7];
            out.push_back(std::move(g));
        }
        p.wait(30);
    } catch (const std::exception&) {
    }
    return out;
}

int find_gpu(const std::string& spec_in, const std::vector<GpuInfo>& gpus, std::string* why) {
    // a number (or, when no card has it, a word of a name: "3090"), a UUID or its start, a PCI bus id, or a name or a
    // part of one (a whole name first), as src/core/gpu_select.cpp reads them
    const std::string s(strip_chars(strip(spec_in), "\"'"));
    const std::string low = lower(s);
    const bool num = !s.empty() && std::all_of(s.begin(), s.end(), ::isdigit);
    if (num)
        for (const auto& g : gpus)
            if (g.index == std::stoi(s)) return g.index;
    static const std::regex bus_re(R"(^(?:([0-9a-fA-F]+):)?([0-9a-fA-F]+):([0-9a-fA-F]+)(?:\.[0-9a-fA-F]+)?$)");
    auto bus = [](const std::string& text) -> std::optional<std::tuple<long, long, long>> {
        std::smatch m;
        if (!std::regex_match(text, m, bus_re)) return std::nullopt;
        auto hex = [&](int i) { return m[i].matched && m[i].length() ? std::stol(m[i].str(), nullptr, 16) : 0L; };
        return std::make_tuple(hex(1), hex(2), hex(3));
    };
    std::vector<const GpuInfo*> hits;
    if (starts_with(low, "gpu-")) {
        for (const auto& g : gpus)
            if (starts_with(lower(g.uuid), low)) hits.push_back(&g);
    } else if (bus(s)) {
        for (const auto& g : gpus)
            if (bus(g.bus_id) == bus(s)) hits.push_back(&g);
    } else {
        for (const auto& g : gpus)
            if (lower(g.name) == low) hits.push_back(&g);
        if (hits.empty()) {
            for (const auto& g : gpus) {
                const std::string name = lower(g.name);
                if (num) {   // a number as a word of a name: not a part of a longer number or word
                    for (size_t at = name.find(low); at != std::string::npos; at = name.find(low, at + 1)) {
                        const bool left = at == 0 || !std::isalnum((unsigned char) name[at - 1]);
                        const bool right = at + low.size() >= name.size() || !std::isalnum((unsigned char) name[at + low.size()]);
                        if (left && right) {
                            hits.push_back(&g);
                            break;
                        }
                    }
                } else if (name.find(low) != std::string::npos) {
                    hits.push_back(&g);
                }
            }
        }
    }
    if (hits.size() == 1) return hits[0]->index;
    if (why != nullptr) {   // the engine's words (src/core/gpu_select.cpp), nvidia-smi's list in place of CUDA's
        if (gpus.empty()) {
            *why = "nvidia-smi lists no GPU";
        } else if (hits.empty() && num) {
            *why = "no GPU " + s + " (nvidia-smi lists " + std::to_string(gpus.size()) + ") or GPU named so";
        } else if (hits.empty()) {
            *why = "no GPU is " + s;
        } else {
            *why = "several GPUs match " + s + " (name one by its UUID):";
            for (const GpuInfo* g : hits) *why += " " + std::to_string(g->index) + " " + g->name + " " + g->uuid + ";";
            why->pop_back();
        }
    }
    return -1;
}

std::vector<int> engine_gpus(const json& cfg) {
    const bool named = gpu_list(cfg).empty() && !engine_cards(cfg).empty();
    return engine_gpus(cfg, named ? nvidia_gpus() : std::vector<GpuInfo>{});
}

std::vector<int> engine_gpus(const json& cfg, const std::vector<GpuInfo>& gpus, std::string* why) {
    const std::vector<int> listed = gpu_list(cfg);
    if (!listed.empty()) return listed;
    const std::vector<std::string> cards = engine_cards(cfg);
    if (cards.empty()) return {};
    auto name = [](const std::string& card) { return std::string(strip(card.substr(0, card.find(',')))); };
    auto index = [&](const std::string& spec, std::string* w) {   // a number as given when nvidia-smi lists nothing
        if (gpus.empty() && !spec.empty() && std::all_of(spec.begin(), spec.end(), ::isdigit)) return std::stoi(spec);
        return find_gpu(spec, gpus, w);
    };
    int main_gpu = 0;   // the engine's default
    std::vector<std::string> tiers;
    if (const std::string m = name(cards[0]); m == "auto") {
        tiers.push_back("auto");
    } else {
        std::string w;
        main_gpu = index(m, &w);
        if (main_gpu < 0) {   // the engine stops at it; said with what named it
            const std::vector<std::string> args = args_of(cfg);
            const char* by = gpu_cards(cfg) ? "\"gpu\"" : std::find(args.begin(), args.end(), "--gpu") != args.end()
                                                              ? "--gpu" : "--main-gpu";
            if (why != nullptr) *why = std::string(by) + " " + m + ": " + w;
            return {};
        }
    }
    for (size_t i = 1; i < cards.size(); ++i) tiers.push_back(name(cards[i]));
    std::vector<int> out = {main_gpu};
    if (tiers.size() == 1 && tiers[0] == "auto") {
        for (const auto& g : gpus)
            if (g.index != main_gpu) out.push_back(g.index);
        return out;
    }
    for (const auto& t : tiers) {   // one no card matches the engine leaves out
        const int i = index(t, nullptr);
        if (i >= 0 && std::find(out.begin(), out.end(), i) == out.end() && out.size() <= 3) out.push_back(i);   // kMaxGpuTiers
    }
    return out;
}

std::optional<std::vector<std::string>> effort_end_args(const json& cfg, const Tokenizer& tok) {
    const json& pos = cfg.contains("effort_position") ? cfg["effort_position"] : json("start");
    if (pos != "start" && pos != "end")
        throw ConfigError("\"effort_position\" must be \"start\" (the default) or \"end\", not " + py_repr_value(pos));
    if (pos == "start") return std::nullopt;
    const std::vector<int32_t> role = tok.encode("system", true);
    if (role.size() != 1) {
        say("[lamina] effort_position \"end\" needs the role \"system\" to be one token: the reasoning effort stays at the "
            "top of the prompt");
        return std::nullopt;
    }
    say("[lamina] effort_position end: a request's non-default reasoning effort goes right before the answer, so changing it "
        "keeps the cached conversation");
    return std::vector<std::string>{"--tail-role-token", std::to_string(role[0])};
}

double engine_silence_s(const json& cfg) {
    const json& v = jget(cfg, "engine_silence_s");
    if (v.is_null()) return 300.0;
    if (v.is_boolean() || !v.is_number() || v.get<double>() < 0)
        throw ConfigError("\"engine_silence_s\" must be a number of seconds >= 0 (0 = no limit), not " + py_repr_value(v));
    return v.get<double>();
}

std::vector<std::string> engine_args(const json& cfg) {
    std::vector<std::string> args = args_of(cfg);
    // the config's "model" when its args name none
    if (!has(args, "--native") && !jstr(cfg, "model").empty())
        args.insert(args.begin(), {"--native", here_path(cfg, jstr(cfg, "model"))});
    // "gpu" as cards: a --gpu each, in place of the args' own GPU names
    if (const auto cards = gpu_cards(cfg)) {
        std::vector<std::string> kept;
        for (size_t i = 0; i < args.size(); ++i) {
            const std::string& a = args[i];
            if ((a == "--main-gpu" || a == "--second-gpu" || a == "--tier-gpus" || a == "--gpu") && i + 1 < args.size()) {
                ++i;
                continue;
            }
            kept.push_back(a);
        }
        args = std::move(kept);
        for (const std::string& c : *cards) args.insert(args.end(), {"--gpu", c});
    }
    // the "vision" section is the engine's --vision (#1322: without it the server advertised images and the engine
    // refused every picture)
    if (jget(cfg, "vision").is_object() && !has(args, "--vision")) args.push_back("--vision");
    // #465 (opt-in): "parallel": N - up to N requests decode together in the engine's batch slots
    const json& n = jget(cfg, "parallel");
    if (!has(args, "--batch") && !has(args, "--slots") && !n.is_null() && n != false) {
        if (n.is_boolean() || !n.is_number_integer()) {
            say("[lamina] \"parallel\" must be a whole number of requests (2.." + std::to_string(kParallelMax) + "), not " +
                py_repr_value(n) + ": ignored");
        } else if (n.get<int64_t>() > 1) {
            const char* mtp = std::getenv("LAMINA_BATCH_MTP");
            if (n.get<int64_t>() > kParallelMax && !has(args, "--batch-mtp") && (mtp == nullptr || !*mtp || std::string(mtp) == "0"))
                say("[lamina] \"parallel\": " + std::to_string(n.get<int64_t>()) + " - the engine runs at most " +
                    std::to_string(kParallelMax) + " requests together; it will use " + std::to_string(kParallelMax));
            args.insert(args.end(), {"--batch", std::to_string(n.get<int64_t>())});
        }
    }
    // #477 (opt-in): "expert_profile_save": the engine saves what its adaptive tier learned there, and the next start
    // begins from it when it is a profile of the same model
    const json& save = jget(cfg, "expert_profile_save");
    if (!save.is_string() || strip(save.get<std::string>()).empty() || has(args, "--expert-profile-save")) return args;
    args.insert(args.end(), {"--expert-profile-save", save.get<std::string>()});
    const json& every = jget(cfg, "expert_profile_save_every");
    if (every.is_number() && every.get<double>() >= 0)
        args.insert(args.end(), {"--expert-profile-save-every", every.is_number_integer() ? std::to_string(every.get<int64_t>())
                                                                                           : py_repr(every.get<double>())});
    // the profile the engine would start from: the args' --expert-profile, else (--serve) the built-in one of the
    // model's shape (the one a config's earlier path to the clone's data/expert-profile.bin stands for too)
    auto at = std::find(args.begin(), args.end(), "--expert-profile");
    const bool named = at != args.end() && at + 1 != args.end();
    const auto learned = profile_shape(here_path(cfg, save.get<std::string>()));
    const bool builtin = !named || *(at + 1) == core::kBuiltinExpertProfile ||
                         core::legacy_expert_profile_path(here_path(cfg, *(at + 1)));
    const auto base = !builtin ? profile_shape(here_path(cfg, *(at + 1)))
                      : learned && !core::builtin_expert_profile(learned->first, learned->second).empty() ? learned
                      : std::nullopt;
    if (learned && learned == base) {
        if (named) *(at + 1) = save.get<std::string>();
        else args.insert(args.end(), {"--expert-profile", save.get<std::string>()});
    }
    return args;
}

std::vector<std::string> deprecated_options(const json& file, const std::vector<std::string>& line) {
    std::vector<std::string> out;
    auto add = [&](const std::string& note) {
        if (std::find(out.begin(), out.end(), note) == out.end()) out.push_back(note);
    };
    if (has(args_of(file), "--native")) add("--native: the config's \"model\"");
    auto scan = [&](const std::vector<std::string>& args) {
        for (size_t i = 0; i < args.size(); ++i) {
            const program::DeprecatedOption* d = program::deprecated_option(args[i].c_str());
            if (d == nullptr) continue;
            std::string used = args[i];
            if (*d->value && i + 1 < args.size()) used += " " + shell_word(args[++i]);
            add(used + ": " + d->instead);
        }
    };
    scan(line);
    if (const json& spec = jget(file, "replicas"); spec.is_array())   // an entry's own, added to the config's
        for (const json& entry : spec) {
            std::vector<std::string> own;
            if (entry.is_object())
                for (const json& a : args_as_list(jget(entry, "args"))) own.push_back(arg_str(a));
            scan(own);
        }
    return out;
}

void apply_engine_options(json& cfg, const std::vector<EngineOption>& opts) {
    if (opts.empty()) return;
    auto flag = [](const std::string& s) { return s.size() > 2 && starts_with(s, "--"); };
    std::set<std::string> drop;
    for (const EngineOption& o : opts) {
        drop.insert(o[0]);
        drop.insert(starts_with(o[0], "--no-") ? "--" + o[0].substr(5) : "--no-" + o[0].substr(2));
    }
    const std::vector<std::string> args = args_of(cfg);
    json out = json::array();
    for (size_t i = 0; i < args.size();) {
        if (flag(args[i]) && drop.count(args[i]) > 0) {
            for (++i; i < args.size() && !flag(args[i]);) ++i;   // its values
            continue;
        }
        out.push_back(args[i++]);
    }
    for (const EngineOption& o : opts)
        for (const std::string& w : o) out.push_back(w);
    cfg["args"] = std::move(out);
}

std::string model_file(const json& cfg) {
    const std::vector<std::string> args = args_of(cfg);
    for (size_t i = 0; i + 1 < args.size(); ++i)
        if (args[i] == "--native") return here_path(cfg, args[i + 1]);
    const std::string model = jstr(cfg, "model");   // the config's "model" when its args name none
    return model.empty() ? "" : here_path(cfg, model);
}

const std::vector<std::pair<std::string, std::string>>& offline_env() {
    // what the common runtimes honour (Hugging Face, Transformers, pip, the proxies): an accidental call fails fast
    static const std::vector<std::pair<std::string, std::string>> env = {
        {"HF_HUB_OFFLINE", "1"}, {"TRANSFORMERS_OFFLINE", "1"}, {"HF_HUB_DISABLE_TELEMETRY", "1"}, {"DO_NOT_TRACK", "1"},
        {"PIP_NO_INDEX", "1"}, {"PIP_DISABLE_PIP_VERSION_CHECK", "1"}, {"NO_PROXY", "*"}, {"no_proxy", "*"},
        {"HTTP_PROXY", ""}, {"HTTPS_PROXY", ""}, {"http_proxy", ""}, {"https_proxy", ""}, {"LAMINA_OFFLINE", "1"}};
    return env;
}

void apply_engine_env(const json& cfg, bool offline) {
    const std::vector<int> gpus = gpu_list(cfg);
    if (!gpus.empty()) {   // issue #51: the GPU(s) as nvidia-smi numbers them (CUDA's own order can differ)
        std::string v;
        for (const int g : gpus) v += (v.empty() ? "" : ",") + std::to_string(g);
        set_env("CUDA_DEVICE_ORDER", "PCI_BUS_ID");
        set_env("CUDA_VISIBLE_DEVICES", v);
    }
    const json& env = jget(cfg, "env");   // engine settings the config carries
    if (env.is_object())
        for (auto it = env.begin(); it != env.end(); ++it) set_env(it.key(), arg_str(it.value()));
    if (offline)   // last, so the config's own "env" cannot remove the policy
        for (const auto& [k, v] : offline_env()) set_env(k, v);
}

Environment child_env(const json& cfg, bool offline) {
    Environment env = environment();
    const std::vector<int> gpus = gpu_list(cfg);
    if (!gpus.empty()) {   // issue #51: the GPU(s) as nvidia-smi numbers them
        std::string v;
        for (const int g : gpus) v += (v.empty() ? "" : ",") + std::to_string(g);
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
        env["CUDA_VISIBLE_DEVICES"] = v;
    }
    const json& extra = jget(cfg, "env");
    if (extra.is_object())
        for (auto it = extra.begin(); it != extra.end(); ++it) env[it.key()] = arg_str(it.value());
    if (offline)
        for (const auto& [k, v] : offline_env()) env[k] = v;
    prepend_lib_dirs(cfg, env);
    return env;
}

void prepend_lib_dirs(const json& cfg, Environment& env) {
    std::string dirs;
    const json& lib = jget(cfg, "lib_dirs");
    if (lib.is_array())
        for (const auto& d : lib) {
            std::error_code ec;
            if (d.is_string() && std::filesystem::is_directory(u8_path(d.get<std::string>()), ec))
#ifdef _WIN32
                dirs += (dirs.empty() ? "" : ";") + d.get<std::string>();
#else
                dirs += (dirs.empty() ? "" : ":") + d.get<std::string>();
#endif
        }
    if (!dirs.empty()) {
#ifdef _WIN32
        const char* var = "PATH";
        const char* sep = ";";
#else
        const char* var = "LD_LIBRARY_PATH";
        const char* sep = ":";
#endif
        const auto it = env.find(var);
        env[var] = it == env.end() || it->second.empty() ? dirs : dirs + sep + it->second;
    }
}

void show_vision_gpu(const json& cfg, std::vector<GpuInfo>& gpus) {
    const std::vector<int> engine = gpu_list(cfg);
    const json& v = jget(cfg, "vision");
    const json& dev = jget(v, "cuda_device");
    if (engine.empty() || !v.is_object() || !truthy(jget(v, "gpu")) || dev.is_null()) return;
    const std::string spec(strip(arg_str(dev)));
    if (spec.empty()) return;
    if (gpus.empty()) gpus = nvidia_gpus();
    const int index = find_gpu(spec, gpus);   // -1 where nvidia-smi lists nothing: the name as it is
    const bool number = std::all_of(spec.begin(), spec.end(), ::isdigit);
    const int named = index >= 0 ? index : number ? std::stoi(spec) : -1;
    if (named >= 0 && std::find(engine.begin(), engine.end(), named) != engine.end()) return;   // one of the engine's
    std::string list;
    for (const int g : engine) list += (list.empty() ? "" : ",") + std::to_string(g);
    set_env("CUDA_VISIBLE_DEVICES", list + "," + (named >= 0 ? std::to_string(named) : spec));
    core::engine_gpu_count((int) engine.size());
}

std::vector<std::string> offline_blockers(const json& cfg) {
    std::vector<std::string> needs;
    const json& servers = truthy(jget(cfg, "mcp_servers")) ? cfg["mcp_servers"] : jget(cfg, "mcpServers");
    if (truthy(servers))
        needs.push_back("the config declares " + std::to_string(servers.is_object() || servers.is_array() ? servers.size() : 1) +
                        " MCP server(s), which are started as subprocesses and may fetch tools");
    for (const char* key : {"before_load", "beforeLoad"})
        if (truthy(jget(cfg, key)))
            needs.push_back(std::string("the config declares a ") + key + " command, which runs before the model loads");
    for (const char* key : {"images", "vision"}) {
        const json& section = jget(cfg, key);
        if (!section.is_object()) continue;
        for (const char* field : {"remote_sources", "allowed_hosts", "url"})
            if (truthy(jget(section, field)))
                needs.push_back(std::string("the config's ") + key + "." + field + " permits network sources");
    }
    return needs;
}

namespace {

bool is_flag(const std::string& s) { return s.size() > 2 && starts_with(s, "--"); }

/// an older flag's number with its unit
std::string with_unit(const std::string& number, const char* u) {
    char b[48];
    std::snprintf(b, sizeof b, "%g%s", std::atof(number.c_str()), u);
    return b;
}

/// the tiers' options under their older names in `a` renamed, a value given its unit
void rename_older(std::vector<std::string>& a) {
    struct Rename {
        const char *from, *to, *unit;
    };
    static const Rename kRenames[] = {{"--second-gpu-clocks-hold", "--clocks-hold", "s"},
                                      {"--second-gpu-clocks-lead", "--clocks-lead", "s"},
                                      {"--second-gpu-min-mb", "--tier-min", "M"},
                                      {"--second-gpu-adapt", "--tier-adapt", nullptr},
                                      {"--no-second-gpu-adapt", "--no-tier-adapt", nullptr}};
    for (size_t i = 0; i < a.size(); ++i)
        for (const Rename& r : kRenames) {
            if (a[i] != r.from) continue;
            a[i] = r.to;
            if (r.unit != nullptr && i + 1 < a.size() && !is_flag(a[i + 1]))
                a[i + 1] = std::string(r.unit) == "s" && std::atof(a[i + 1].c_str()) < 0 ? "off" : with_unit(a[i + 1], r.unit);
        }
}

}  // namespace

json modern_config(json cfg, bool flash) {
    std::vector<std::string> a = args_of(cfg);
    auto at = [&](const std::string& f) -> std::optional<size_t> {
        for (size_t i = 0; i < a.size(); ++i)
            if (a[i] == f) return i;
        return std::nullopt;
    };
    auto value = [&](const std::string& f) -> std::optional<std::string> {   // the word after the flag
        const auto i = at(f);
        if (!i || *i + 1 >= a.size() || is_flag(a[*i + 1])) return std::nullopt;
        return a[*i + 1];
    };
    auto erase = [&](const std::string& f) {   // the flag and its values
        if (const auto i = at(f)) {
            size_t j = *i + 1;
            while (j < a.size() && !is_flag(a[j])) ++j;
            a.erase(a.begin() + (std::ptrdiff_t) *i, a.begin() + (std::ptrdiff_t) j);
        }
    };
    // the GPUs as cards: "gpu"'s, else --main-gpu's and --second-gpu's (--tier-gpus') names, the main GPU the
    // engine's "0" when none names it; the older per-card flags their settings where a card has none of its own (its
    // own wins, as in the engine).  Beside a "gpu" of numbers or replicas (the cards the engine sees, a replica's "gpu"
    // its group's numbers) the flags stay, and where a setting would go to an auto card (it takes none as the first, an
    // auto tier's stay the per-tier lists').
    if (gpu_list(cfg).empty() && !truthy(jget(cfg, "replicas"))) {
        auto list = [&](const char* f) {
            std::vector<std::string> v;
            if (const auto s = value(f))
                for (const std::string& x : split(*s, ",")) v.push_back(std::string(strip(x)));
            return v;
        };
        auto pick = [](const std::vector<std::string>& v, size_t t) {
            return v.empty() ? std::string() : v[std::min(t, v.size() - 1)];
        };
        const auto gib = list("--second-gpu-gib"), reserve = list("--second-gpu-reserve-mib"),
                   prefetch = list("--second-gpu-prefetch-mb"), clocks = list("--second-gpu-clocks"),
                   sleep = list("--second-gpu-sleep-us");
        const auto pin = value("--second-gpu-pin-gib");
        const auto cache = value("--expert-cache"), keep = value("--vram-reserve-mib"), only = value("--vram-pin-gib");
        const bool per_tier = !gib.empty() || !reserve.empty() || !prefetch.empty() || !clocks.empty() ||
                              !sleep.empty() || pin;
        std::optional<std::vector<json>> cards = card_objects(cfg);
        const bool given = cards.has_value();
        bool named = given;   // a card named: "gpu"'s, --main-gpu's or a tier
        if (!given) {
            cards.emplace();
            const auto main = value("--main-gpu");
            named = main.has_value();
            cards->push_back(json{{"id", main.value_or("0")}});
            if (const auto t = value("--second-gpu") ? value("--second-gpu") : value("--tier-gpus"))
                for (const std::string& x : split(*t, ",")) {
                    const std::string s(strip(x));
                    if (s.empty() || s[0] == '-') continue;
                    cards->push_back(json{{"id", s}});
                    named = true;
                }
        }
        auto is_auto = [](const json& c) { return lower(card_value(c["id"])) == "auto"; };
        const bool auto_main = is_auto(cards->front());
        const bool auto_tier = auto_main || std::any_of(cards->begin() + 1, cards->end(), is_auto);
        if (!((cache || keep || only) && auto_main) && !(per_tier && auto_tier)) {
            bool changed = false;
            auto set = [&](json& c, const char* k, const std::string& v) {
                if (c.contains(k)) return;
                c[k] = v;
                changed = true;
            };
            json& m = cards->front();
            if (cache && *cache != program::option_default("--expert-cache")) set(m, "cache", *cache);
            if (keep && *keep != program::option_default("--vram-reserve-mib")) set(m, "reserve", with_unit(*keep, "M"));
            if (only && std::atof(only->c_str()) > 0) set(m, "vram-only", with_unit(*only, "G"));
            for (size_t t = 0; t + 1 < cards->size(); ++t) {
                json& c = (*cards)[t + 1];
                if (std::atof(pick(gib, t).c_str()) > 0) set(c, "cache", with_unit(pick(gib, t), "G"));
                if (!reserve.empty()) set(c, "reserve", with_unit(pick(reserve, t), "M"));
                if (!prefetch.empty()) set(c, "prefetch", with_unit(pick(prefetch, t), "M"));
                if (const std::string k = pick(clocks, t); !k.empty() && k != "0") set(c, "clocks", k);
                if (std::atof(pick(sleep, t).c_str()) > 0) set(c, "sleep", with_unit(pick(sleep, t), "us"));
                if (t == 0 && pin && std::atof(pin->c_str()) > 0) set(c, "vram-only", with_unit(*pin, "G"));
            }
            // "gpu" as it was unless a card took a setting; from the flags, unless they leave the engine's default
            if (changed || (!given && named)) cfg["gpu"] = cards_json(*cards);
            // the rest: at their default, a card's own in their place, per-tier lists without a tier, or (beside
            // "gpu"'s cards) the names, which the engine reads no more
            for (const char* f : {"--main-gpu", "--second-gpu", "--tier-gpus", "--expert-cache", "--vram-reserve-mib",
                                  "--vram-pin-gib", "--second-gpu-gib", "--second-gpu-reserve-mib",
                                  "--second-gpu-prefetch-mb", "--second-gpu-clocks", "--second-gpu-sleep-us",
                                  "--second-gpu-pin-gib"})
                erase(f);
        }
    }
    rename_older(a);
    // Qwen3.8-Flash-Next's options at the engine's default (the qwen35 path's defaults are its own)
    for (size_t i = 0; flash && i < a.size();) {
        size_t j = i + 1;
        while (j < a.size() && !is_flag(a[j])) ++j;
        const std::string def = is_flag(a[i]) ? program::option_default(a[i].c_str()) : "";
        bool same = false;
        if (!def.empty() && j == i + 1) {
            same = def == "on";   // a switch the engine has on anyway
        } else if (!def.empty() && j == i + 2) {
            const std::string& v = a[i + 1];
            char* e1 = nullptr;
            char* e2 = nullptr;
            const double x = std::strtod(v.c_str(), &e1), y = std::strtod(def.c_str(), &e2);
            same = v == def || (a[i] == "--prefill" && v == "auto") ||
                   (e1 != v.c_str() && *e1 == '\0' && e2 != def.c_str() && *e2 == '\0' && x == y);
        }
        if (same) a.erase(a.begin() + (std::ptrdiff_t) i, a.begin() + (std::ptrdiff_t) j);
        else i = j;
    }
    // the model as "model", the pictures as their section says
    if (const auto v = value("--native"); v && (jstr(cfg, "model").empty() || jstr(cfg, "model") == *v)) {
        cfg["model"] = *v;
        erase("--native");
    }
    if (jget(cfg, "vision").is_object()) {
        erase("--vision");
        if (!jstr(cfg, "model").empty() && jstr(cfg["vision"], "model") == jstr(cfg, "model")) cfg["vision"].erase("model");
    }
    json list = json::array();
    for (const std::string& w : a) list.push_back(w);
    cfg["args"] = args_as_object(list).value_or(list);   // an empty one stays: a config without "args" is a preset's
    // a replica's own args, the older names renamed (its GPUs are numbers: their flags stay)
    if (cfg.contains("replicas") && cfg["replicas"].is_array())
        for (json& entry : cfg["replicas"]) {
            if (!entry.is_object() || !truthy(jget(entry, "args"))) continue;
            std::vector<std::string> own;
            for (const json& w : args_as_list(entry["args"])) own.push_back(arg_str(w));
            rename_older(own);
            json words = json::array();
            for (const std::string& w : own) words.push_back(w);
            entry["args"] = args_as_object(words).value_or(words);
        }
    // the model, its cards and its options first
    json out = json::object();
    for (const char* k : {"model", "gpu", "args"})
        if (cfg.contains(k)) out[k] = cfg[k];
    for (auto it = cfg.begin(); it != cfg.end(); ++it)
        if (!out.contains(it.key())) out[it.key()] = it.value();
    return out;
}

}  // namespace lamina
