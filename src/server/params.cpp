// src/server/params.cpp - lamina-server's settings table and command line (params.hpp).
#include "params.hpp"

#include "config.hpp"
#include "remote_model.hpp"
#include "lamina/program/options.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iterator>
#include <stdexcept>

namespace lamina {

using K = ParamKind;

// The keys and their options.  Named alike: an option is its key with "-" for "_" (a section's key after the section's
// name: vision.max_tokens is --vision-max-tokens), but for the sampling defaults (llama.cpp's names) and the spellings
// kept from before (--lazy, --idle-unload, --open, --mcp-config).
const std::vector<Param> kParams = {
    {"", "", "", K::Str, "", "The model"},
    {"--model", "-m", "model", K::Path, "GGUF",
     "the model's file (a split one by any shard). Without the config's \"args\" the preset for it (examples/*.json, "
     "picked by the file's header) gives them, with this PC's GPUs and the files beside the model; with them, it takes "
     "the place of their --native. With --check also a URL: a Hugging Face repository's, a folder's or a file's in it, "
     "or a GGUF file's (its files' headers alone are read; several models: a table)"},
    {"--mmproj", "", "vision.mmproj", K::Path, "FILE",
     "pictures: the image projector (with -m by default the mmproj*.gguf made for the model beside it)"},
    {"--no-mmproj", "", "vision", K::Off, "", "no pictures"},
    {"--vision-model", "", "vision.model", K::Path, "GGUF", "taken and not read: the image encoder needs no model file "
     "(earlier versions read its vocabulary)"},
    {"--vision-gpu", "", "vision.gpu", K::Bool, "",
     "the image encoder on a GPU, not the CPU (with -m: on, with --vision-lend; --no-vision-gpu: the CPU)"},
    {"--vision-cuda-device", "", "vision.cuda_device", K::Str, "UUID", "that GPU (nvidia-smi -L lists the UUIDs)"},
    {"--vision-lend", "", "vision.lend", K::Bool, "",
     "the image encoder on a GPU holds no VRAM between pictures: for a new picture it borrows the engine's on its "
     "card (Qwen3.8-Flash-Next: a card's expert cache, as a prompt does; Hy4-flash: the main GPU's; Qwen3.6-35B-A3B "
     "and Qwen3.8-27B: the prompt path's staging buffers, or VRAM the engine leaves free); --no-vision-lend: it "
     "keeps its VRAM"},
    {"--vision-q8", "", "vision.q8", K::Str, "mmq|fp16",
     "a Q8_0 projector on a GPU: its blocks multiplied as llama.cpp multiplies them (mmq, the default: the inputs "
     "rounded to 8 bits), or its values multiplied as FP16 (fp16: nearer the unquantized projector's rows)"},
    {"--vision-threads", "", "vision.threads", K::Int, "N", "the encoder's CPU threads (with -m: the logical cores)"},
    {"--vision-max-tokens", "", "vision.max_tokens", K::Int, "N", "a picture's tokens at most"},
    {"--vision-min-tokens", "", "vision.min_tokens", K::Int, "N", "a picture's tokens at least"},
    {"--model-name", "", "model_name", K::Str, "NAME", "the model's name in the API (with -m: the file's)"},
    {"--aliases", "--alias", "aliases", K::List, "A,B", "other names the model answers to"},
    {"--lora-models", "", "lora_models", K::Value, "JSON",
     "model names that run with the engine's LoRA adapters, {\"NAME\": [{\"id\": 0, \"scale\": 1}], ...}: a request "
     "naming one (and no \"lora\" of its own) takes those scales, the other adapters at 0"},
    {"--arch", "", "arch", K::Str, "ARCH", "the architecture whose model card gives the sampling a request leaves out "
     "(default: the file's)"},
    {"--chat-template-file", "", "chat_template_file", K::Path, "FILE", "a chat template in place of the model file's"},

    {"", "", "", K::Str, "", "Running the engine"},
    {"--gpu", "", "gpu", K::Many, "CARD[,key=value...]",
     "a GPU the engine uses, again for each: the first runs the model, the others hold experts as tiers in their order "
     "(Qwen3.6 and Qwen3.8-27B: the layer split's GPUs; Inkling 2: context cells, layers or the RAM experts' copy). "
     "CARD: a name only one card has, a UUID's start, a PCI bus id, a number or auto (every other GPU), then its "
     "settings, which the engine's options below list; numbers alone (0,1): the cards the engine sees, as nvidia-smi "
     "numbers them"},
    {"--parallel", "", "parallel", K::Int, "N", "requests decoding together in the engine's batch slots"},
    {"--expert-profile-save", "", "expert_profile_save", K::Path, "FILE",
     "Flash-Next and Hy4-flash: what the run learned about the experts, which the next start begins from (with -m: "
     "beside the program)"},
    {"--expert-profile-save-every", "", "expert_profile_save_every", K::Num, "MIN",
     "minutes between those saves (0: at exit only)"},
    {"--engine-silence-s", "", "engine_silence_s", K::Num, "S",
     "a request ends when the engine says nothing this long (default 300; 0: no limit)"},
    {"--effort-position", "", "effort_position", K::Str, "start|end",
     "where a request's non-default reasoning effort goes: end keeps the cached conversation when it changes"},
    {"--log", "--log-file", "log", K::Path, "FILE", "the engine's log (with -m: beside the program)"},
    {"--cwd", "", "cwd", K::Path, "DIR", "the folder the engine runs in, which the config's relative paths start from"},
    {"--env", "", "env", K::Pairs, "NAME=VALUE", "a variable of the engine's environment (again for more)"},
    {"--lib-dirs", "", "lib_dirs", K::List, "D1,D2", "taken and not read: the engine runs in this process, which "
     "loads the CUDA libraries as it starts (earlier versions started the engine as a program of its own)"},

    {"", "", "", K::Str, "", "The network and access"},
    {"--host", "", "host", K::Str, "HOST",
     "the address: 127.0.0.1 this PC only (the default), 0.0.0.0 other devices too (set an API key)"},
    {"--port", "", "port", K::Int, "N", "the port (default 8095, with -m the preset's; 0: one the system picks)"},
    {"--api-key", "", "api_key", K::Str, "KEY",
     "require this key on the API (Authorization: Bearer ... or x-api-key; also $LAMINA_API_KEY); several separated "
     "by commas"},
    {"--cors-origins", "", "cors_origins", K::List, "O1,O2", "web pages' origins allowed to call /v1/* (*: any)"},
    {"--trusted-origins", "", "trusted_origins", K::List, "O1,O2", "origins trusted as this server's own pages"},
    {"--allowed-hosts", "", "allowed_hosts", K::List, "H1,H2",
     "other host names it answers to (*: any; also $LAMINA_ALLOWED_HOSTS)"},
    {"--offline", "", "offline", K::Bool, "", "no image URLs fetched, no MCP servers or before_load commands"},
    {"--api-monitor", "", "api_monitor", K::Bool, "", "the request monitor at /api-monitor (the last 100 requests in "
     "memory)"},
    {"--mcp-servers", "", "mcp_servers", K::Value, "JSON",
     "MCP servers for the web app's chat, in Claude Desktop's form ({\"name\": {\"command\": ...}})"},
    {"--mcp-config", "", "mcp_config", K::Path, "FILE", "MCP servers from a file in Claude Desktop's form "
     "({\"mcpServers\": {...}})"},
    {"--mcp-timeout-s", "", "mcp.timeout_s", K::Num, "S", "an MCP tool call's time limit"},
    {"--mcp-start-timeout-s", "", "mcp.start_timeout_s", K::Num, "S", "an MCP server's time to start"},
    {"--mcp-max-result-chars", "", "mcp.max_result_chars", K::Int, "N", "a tool result's characters at most"},
    {"--mcp-max-rounds", "", "mcp.max_rounds", K::Int, "N", "tool-call rounds of one chat reply at most"},

    {"", "", "", K::Str, "", "Loading and stopping"},
    {"--lazy-load", "--lazy", "lazy_load", K::Bool, "", "start unloaded: the first request loads the model"},
    {"--idle-unload-s", "--idle-unload", "idle_unload_s", K::Num, "S",
     "unload the model after this long without requests (0: never)"},
    {"--min-free-vram-mib", "", "min_free_vram_mib", K::Int, "N",
     "load an unloaded model only when this much VRAM is free, else 503"},
    {"--before-load", "", "before_load", K::Value, "CMD",
     "a command run before the model loads again (a JSON list: the program and its arguments)"},
    {"--before-load-timeout-s", "", "before_load_timeout_s", K::Num, "S", "that command's time limit"},
    {"--before-load-ignore-failure", "", "before_load_ignore_failure", K::Bool, "",
     "load the model even when that command fails"},
    {"--before-load-strict", "", "before_load_strict", K::Bool, "", "answer 503 when that command fails"},
    {"--max-queue", "", "max_queue", K::Int, "N", "inference calls that may wait beyond the slots (0: refuse overload)"},
    {"--queue-timeout", "", "queue_timeout", K::Num, "S",
     "the longest wait for the first engine turn, 0.1..600 (needs max_queue)"},
    {"--shutdown-timeout", "", "shutdown_timeout", K::Num, "S",
     "on a stop, running requests may finish for up to S seconds (0..300)"},
    {"--ready-file", "", "ready_file", K::Str, "FILE",
     "a JSON record of this instance, once it serves (an absolute path, a new file)"},
    {"--open-browser", "--open", "open_browser", K::Bool, "", "open the web app in the browser once the model is ready"},
    {"--prevent-sleep", "", "prevent_sleep", K::Bool, "", "Windows: the PC stays awake while a request runs or waits"},

    {"", "", "", K::Str, "", "Requests (the defaults of those that leave a setting out)"},
    {"--temperature", "--temp", "sampling.temperature", K::Num, "T", "the temperature (0: greedy)"},
    {"--top-p", "", "sampling.top_p", K::Num, "P", "top_p"},
    {"--top-k", "", "sampling.top_k", K::Int, "K", "top_k, 1..64"},
    {"--min-p", "", "sampling.min_p", K::Num, "P", "min_p"},
    {"--presence-penalty", "", "sampling.presence_penalty", K::Num, "X", "presence_penalty"},
    {"--frequency-penalty", "", "sampling.frequency_penalty", K::Num, "X", "frequency_penalty"},
    {"--repetition-penalty", "--repeat-penalty", "sampling.repetition_penalty", K::Num, "X",
     "repetition_penalty (1: off)"},
    {"--penalty-last-n", "--repeat-last-n", "sampling.penalty_last_n", K::Int, "N",
     "the tokens repetition_penalty looks back over"},
    {"--seed", "", "sampling.seed", K::Int, "N", "the seed"},
    {"--reasoning-effort", "", "reasoning_effort", K::Str, "LEVEL", "the thinking level"},
    {"--reasoning-budget-tokens", "--reasoning-budget", "reasoning_budget_tokens", K::Int, "N",
     "a request's thinking at most (0: no cap)"},
    {"--answer-room", "", "answer_room", K::Bool, "",
     "a thinking is closed when a quarter of max_tokens is left, for the answer (default on)"},
    {"--fit-max-tokens", "", "fit_max_tokens", K::Bool, "",
     "max_tokens shortened to the context left (default on; off: a 400, as llama.cpp answers)"},
    {"--anthropic-thinking", "", "anthropic_thinking", K::Str, "model|on_request",
     "an Anthropic request that asks for no thinking: thinks as the model does, or not"},
    {"--tool-call-recovery", "", "tool_call_recovery", K::Bool, "", "also read tool calls written in a few other forms"},
    {"--repeat-stop-tokens", "", "repeat_stop_tokens", K::Int, "N",
     "a reply repeating one token this many times ends (default 256; 0: off)"},
    {"--reasoning-close-retry", "", "reasoning_close_retry", K::Bool, "",
     "a thinking that ended with no answer is closed and the reply goes on once"},
    {"--literal-markers", "", "literal_markers", K::Bool, "",
     "an end-of-turn token drawn where the text quotes it becomes its text (default on)"},
    {"--short-opening-tokens", "", "short_opening_tokens", K::Int, "N",
     "a conversation's only message of up to N tokens is read as at the medium effort"},
    {"--reasoning-loop-recovery", "", "reasoning_loop_recovery", K::Value, "stop|recover|false",
     "a reply whose reasoning repeats whole passages: ended, or restarted"},
    {"--codex-compaction-cache", "", "codex_compaction_cache", K::Bool, "", "Codex's compaction requests from a cache"},
    {"--codex-thread-titles", "", "codex_thread_titles", K::Bool, "",
     "Codex's thread-title requests answered without the engine"},
    {"--env-to-user", "", "env_to_user", K::Bool, "",
     "opencode's <env> block moved to the first user message, so the system turn stays the same"},

    {"", "", "", K::Str, "", "Image generation (the images section: docs/USING.md, Image generation)"},
    {"--images-exe", "", "images.exe", K::Path, "FILE", "sd-cli"},
    {"--images-diffusion-model", "", "images.diffusion_model", K::Path, "FILE", "the denoiser"},
    {"--images-vae", "", "images.vae", K::Path, "FILE", "the VAE"},
    {"--images-llm", "", "images.llm", K::Path, "FILE", "the text encoder"},
    {"--images-model-name", "", "images.model_name", K::Str, "NAME", "the image model's name in the API"},
    {"--images-backend", "", "images.backend", K::Str, "NAME", "sd-cli's backend"},
    {"--images-quant", "", "images.quant", K::Str, "Q", "the denoiser's quant"},
    {"--images-args", "", "images.args", K::List, "A,B", "sd-cli's own options (a JSON list for ones with commas)"},
    {"--images-native", "", "images.native", K::Bool, "", "Lamina's DiT runner inside sd-cli (CUDA builds)"},
    {"--images-gpu", "", "images.gpu", K::Value, "N", "the GPU sd-cli runs on"},
    {"--images-gpu-policy", "", "images.gpu_policy", K::Str, "auto|keep|unload",
     "the text model during an image job: kept loaded, stopped, or by the free VRAM"},
    {"--images-reload", "", "images.reload", K::Str, "lazy|eager",
     "a model stopped for a job loads again with the next request, or at the job's end"},
    {"--images-need-mib", "", "images.need_mib", K::Int, "N", "the VRAM a job takes (auto: by its size)"},
    {"--images-out-dir", "", "images.out_dir", K::Path, "DIR", "where the images go (default: a temporary folder)"},
    {"--images-keep-files", "", "images.keep_files", K::Int, "N", "the newest images kept there (default 64)"},
    {"--images-steps", "", "images.steps", K::Int, "N", "the denoising steps a request leaves out"},
    {"--images-sigmas", "", "images.sigmas", K::Str, "S", "the schedule's sigmas"},
    {"--images-max-n", "", "images.max_n", K::Int, "N", "images a request at most (default 4)"},
    {"--images-max-side", "", "images.max_side", K::Int, "N", "an image's side at most (default 2048)"},
    {"--images-timeout-s", "", "images.timeout_s", K::Num, "S", "a job's time limit (default 3600)"},
    {"--images-lib-dirs", "", "images.lib_dirs", K::List, "D1,D2", "folders searched first for sd-cli's libraries"},
    {"--images-env", "", "images.env", K::Pairs, "NAME=VALUE", "a variable of sd-cli's environment (again for more)"},

    {"", "", "", K::Str, "", "The tests' scripted engine"},
    {"--engine", "", "engine", K::Str, "lamina|mock", "lamina (the default) or mock: the scripted engine"},
    {"--tokenizer", "", "tokenizer", K::Path, "GGUF", "the mock engine's tokenizer and chat template (a model's file)"},
    {"--script", "", "scripts", K::Many, "TEXT", "the mock engine's answer (again: requests get them in turn)"},
    {"--script-cycle", "", "script_cycle", K::Bool, "", "the scripts again from the first, not the last repeated"},
    {"--script-delay", "", "script_delay", K::Num, "S", "seconds between the mock engine's tokens"},
};

namespace {

std::vector<std::string_view> words(std::string_view s) {
    std::vector<std::string_view> out;
    for (size_t at = 0; at < s.size();) {
        const size_t sp = s.find(' ', at);
        const size_t end = sp == std::string_view::npos ? s.size() : sp;
        if (end > at) out.push_back(s.substr(at, end - at));
        at = end + 1;
    }
    return out;
}

bool names(const Param& p, std::string_view flag) {
    if (p.flag.empty()) return false;
    if (p.flag == flag) return true;
    for (const std::string_view a : words(p.aliases))
        if (a == flag) return true;
    return false;
}

[[noreturn]] void bad(const std::string& flag, const std::string& what) {
    throw std::invalid_argument("argument " + flag + ": " + what);
}

json parse_value(const Param& p, const std::string& flag, const std::string& v) {
    switch (p.kind) {
        case K::Int: {
            try {
                size_t used = 0;
                const long long n = std::stoll(v, &used);
                if (used == v.size()) return json((int64_t) n);
            } catch (const std::exception&) {
            }
            bad(flag, "expected a whole number, not '" + v + "'");
        }
        case K::Num: {
            try {
                size_t used = 0;
                const double d = std::stod(v, &used);
                if (used == v.size()) {
                    if (v.find_first_of(".eEnN") == std::string::npos) return json((int64_t) std::stoll(v));
                    return json(d);
                }
            } catch (const std::exception&) {
            }
            bad(flag, "expected a number, not '" + v + "'");
        }
        case K::Bool: {
            const std::string l = lower(v);
            if (l == "true" || l == "1" || l == "yes" || l == "on") return true;
            if (l == "false" || l == "0" || l == "no" || l == "off") return false;
            bad(flag, "expected true or false, not '" + v + "'");
        }
        case K::List: {
            if (starts_with(strip(v), "[")) {
                json a;
                if (!py_loads(v, a) || !a.is_array()) bad(flag, "expected a JSON list or comma-separated values");
                return a;
            }
            json a = json::array();
            for (const auto& x : split(v, ","))
                if (!strip(x).empty()) a.push_back(std::string(strip(x)));
            return a;
        }
        case K::Many: return json::array({v});
        case K::Pairs: {
            const size_t eq = v.find('=');
            if (eq == std::string::npos || eq == 0) bad(flag, "expected NAME=VALUE, not '" + v + "'");
            return json{{v.substr(0, eq), v.substr(eq + 1)}};
        }
        case K::Value: {
            json j;
            if (py_loads(v, j) && !j.is_string()) return j;
            return v;
        }
        case K::Path: return is_url(v) ? v : u8_str(std::filesystem::absolute(u8_path(v)).lexically_normal());   // (-m URL: --check reads it)
        case K::Str:
        case K::Off: break;
    }
    return v;
}

}  // namespace

const Param* find_param(std::string_view flag, bool* negated) {
    if (negated != nullptr) *negated = false;
    for (const Param& p : kParams)
        if (names(p, flag)) return &p;
    if (starts_with(flag, "--no-")) {   // a switch's off form
        const std::string on = "--" + std::string(flag.substr(5));
        for (const Param& p : kParams)
            if (p.kind == K::Bool && names(p, on)) {
                if (negated != nullptr) *negated = true;
                return &p;
            }
    }
    return nullptr;
}

void set_setting(json& cfg, std::string_view key, json v) {
    const size_t dot = key.find('.');
    if (dot == std::string_view::npos) {
        cfg[std::string(key)] = std::move(v);
        return;
    }
    json& section = cfg[std::string(key.substr(0, dot))];
    if (!section.is_object()) section = json::object();
    set_setting(section, key.substr(dot + 1), std::move(v));
}

namespace {

/// the value at `key` in `cfg`, or nullptr
json* setting_at(json& cfg, std::string_view key) {
    const size_t dot = key.find('.');
    const std::string head(key.substr(0, dot));
    if (!cfg.is_object() || !cfg.contains(head)) return nullptr;
    json& v = cfg[head];
    return dot == std::string_view::npos ? &v : setting_at(v, key.substr(dot + 1));
}

}  // namespace

CommandLine parse_command_line(int argc, char** argv) {
    CommandLine cl;
    for (int i = 1; i < argc; ++i) {
        std::string o = argv[i], v;
        const size_t eq = o.find('=');
        const bool inline_value = starts_with(o, "--") && eq != std::string::npos;
        if (inline_value) {
            v = o.substr(eq + 1);
            o = o.substr(0, eq);
        }
        auto value = [&]() -> std::string {
            if (inline_value) return v;
            if (i + 1 >= argc) bad(o, "expected one argument");
            return argv[++i];
        };
        if (o == "-h" || o == "--help") {
            cl.help = true;
            continue;
        }
        if (o == "-v" || o == "--version") {
            cl.version = true;
            continue;
        }
        if (o == "--config") {
            cl.config = value();
            continue;
        }
        if (o == "--write-config") {
            cl.write_config = value();
            continue;
        }
        if (o == "--check") {
            cl.check = true;
            continue;
        }
        static const char* const bench_flags[] = {"--bench",       "--bench-url",   "--bench-runs", "--bench-clients",
                                                  "--bench-long",  "--bench-think", "--bench-out",  "--bench-compare"};
        if (std::find(std::begin(bench_flags), std::end(bench_flags), o) != std::end(bench_flags)) {
            // the benchmark's (src/server/bench.hpp): command-line options, no config keys
            auto whole = [&](int lo, int hi) {
                const std::string v = value();
                try {
                    size_t used = 0;
                    const long long n = std::stoll(v, &used);
                    if (used == v.size() && n >= lo && n <= hi) return (int) n;
                } catch (const std::exception&) {
                }
                bad(o, "expected a whole number in " + std::to_string(lo) + ".." + std::to_string(hi) + ", not '" + v + "'");
            };
            auto absolute = [](const std::string& p) { return u8_str(std::filesystem::absolute(u8_path(p)).lexically_normal()); };
            if (o == "--bench" || o == "--bench-long" || o == "--bench-think") {
                if (inline_value) bad(o, "takes no value");
                (o == "--bench" ? cl.bench : o == "--bench-long" ? cl.bench_long : cl.bench_think) = true;
                cl.bench_options = cl.bench_options || o != "--bench";
                continue;
            }
            if (o == "--bench-url") {
                cl.bench_url = value();
                continue;
            }
            if (o == "--bench-runs" || o == "--bench-clients") {
                (o == "--bench-runs" ? cl.bench_runs : cl.bench_clients) = o == "--bench-runs" ? whole(1, 100) : whole(0, 256);
                cl.bench_options = true;
                continue;
            }
            if (o == "--bench-out") {
                cl.bench_out = absolute(value());
                cl.bench_options = true;
                continue;
            }
            if (o == "--bench-compare") {   // OLD, and NEW when the next word is no option
                cl.bench_compare.push_back(absolute(value()));
                if (!inline_value && i + 1 < argc && !starts_with(argv[i + 1], "--")) cl.bench_compare.push_back(absolute(argv[++i]));
                if (cl.bench_compare.size() > 2) bad(o, "takes one report (OLD) or two (OLD NEW)");
                continue;
            }
        }
        bool negated = false;
        if (const Param* p = find_param(o, &negated)) {
            if (p->kind == K::Off) {
                if (inline_value) bad(o, "takes no value");
                set_setting(cl.settings, p->key, false);
            } else if (p->kind == K::Bool) {
                if (negated && inline_value) bad(o, "takes no value");
                set_setting(cl.settings, p->key, negated ? json(false) : inline_value ? parse_value(*p, o, v) : json(true));
            } else {
                json x = parse_value(*p, o, value());
                json* now = setting_at(cl.settings, p->key);
                if ((p->kind == K::List || p->kind == K::Many) && now != nullptr && now->is_array()) {
                    for (auto& e : x) now->push_back(std::move(e));   // given again: appended
                } else if (p->kind == K::Pairs && now != nullptr && now->is_object()) {
                    for (auto it = x.begin(); it != x.end(); ++it) (*now)[it.key()] = it.value();
                } else {
                    set_setting(cl.settings, p->key, std::move(x));
                }
            }
            continue;
        }
        if (starts_with(o, "--") && o.size() > 2) {   // the engine's, with the values after it
            EngineOption eo{o};
            if (inline_value) eo.push_back(v);
            else
                while (i + 1 < argc && !starts_with(argv[i + 1], "--")) eo.push_back(argv[++i]);
            cl.engine.push_back(std::move(eo));
            continue;
        }
        throw std::invalid_argument("unrecognized arguments: " + std::string(argv[i]));
    }
    if (json* g = setting_at(cl.settings, "gpu")) *g = gpu_canonical(*g);   // --gpu 0,2: [0, 2]; cards: one item each
    return cl;
}

void merge_settings(json& base, const json& over) {
    if (!over.is_object()) return;
    for (auto it = over.begin(); it != over.end(); ++it) {
        json& b = base[it.key()];
        if (it.value().is_object() && b.is_object()) merge_settings(b, it.value());
        else b = it.value();
    }
}

std::vector<std::string> unknown_keys(const json& cfg) {
    std::vector<std::string> out;
    if (!cfg.is_object()) return out;
    for (auto it = cfg.begin(); it != cfg.end(); ++it) {
        const std::string& k = it.key();
        if (k == "args" || k == "mcpServers") continue;
        bool known = false;
        for (const Param& p : kParams) {
            const std::string_view top = p.key.substr(0, p.key.find('.'));
            if (!p.flag.empty() && top == k) {
                known = true;
                break;
            }
        }
        if (!known) out.push_back(k);
    }
    return out;
}

std::string params_help() {
    using program::help_line;
    using program::wrapped;
    constexpr size_t kWidth = 118, kCol = 36;
    std::string out =
        "lamina-server: the OpenAI, Anthropic and Responses APIs and the web app in front of the Lamina engine.\n"
        "\n"
        "  lamina-server -m MODEL.gguf [options] [engine options]\n"
        "  lamina-server --config FILE [options] [engine options]\n"
        "\n";
    out += wrapped("Each of the server's options is a key of the run config too, named in [brackets] (a dot: a key of "
                   "that section), and the other way round; the engine's options, after the server's, are the config's "
                   "\"args\" (\"max-context\": 65536, a switch true or false). A value of the command line takes the "
                   "place of the config file's, and the file's that of the preset -m builds from (--check shows the "
                   "engine options a start gives the model). A switch takes --no-<name> for off. docs/PARAMETERS.md "
                   "describes every option.",
                   0, kWidth, 0);
    out += "\nCommands\n";
    for (const auto& [left, help] : {
             std::pair<const char*, const char*>{"--config FILE", "the run config to start from"},
             {"--write-config FILE", "write the config this command line runs to FILE and stop; an earlier FILE is "
                                     "kept as FILE.bak, and its keys outside the preset stay"},
             {"--check", "say what this PC has and, for the model, what it needs, how it would run and whether it "
                         "starts (exit code 1: it would not); nothing is loaded"},
             {"--bench", "measure the model the way it is used: load it, send a fixed suite through this server's own "
                         "API on a port of its own (short questions, 4K and 16K prompts, a coding agent's session; 3-10 "
                         "minutes), write a report to share into bench-results beside the program, and stop"},
             {"--bench-url URL", "the same against a Lamina server that runs (http://127.0.0.1:8081): nothing is "
                                 "loaded here"},
             {"--bench-runs N", "runs per test (default 3)"},
             {"--bench-clients N", "also N requests at once (the server's --parallel slots run them together)"},
             {"--bench-long", "also a prompt of ~32K tokens"},
             {"--bench-think", "thinking on (the suite turns it off, so the answers are short and comparable)"},
             {"--bench-out DIR", "where the report's folder goes (default: bench-results beside the program)"},
             {"--bench-compare OLD [NEW]", "an earlier report (its folder or results.json) beside this run's; with "
                                           "NEW and no --bench, two reports side by side"},
             {"-h, --help", "this text"},
             {"-v, --version", "the program's version"}})
        out += help_line(left, help, kWidth, kCol);
    for (const Param& p : kParams) {
        if (p.flag.empty()) {
            out += "\n" + std::string(p.help) + "\n";
            continue;
        }
        std::string l;
        for (const std::string_view a : words(p.aliases))
            if (a.size() == 2) l += std::string(a) + ", ";   // a short one first: -m, --model
        l += std::string(p.flag);
        for (const std::string_view a : words(p.aliases))
            if (a.size() > 2) l += ", " + std::string(a);
        if (!p.arg.empty()) l += " " + std::string(p.arg);
        out += help_line(l, "[" + std::string(p.key) + "] " + std::string(p.help), kWidth, kCol);
    }
    out += "\n\nENGINE OPTIONS: the config's \"args\"\n" +
           wrapped("Their defaults are the engine's own; -m's preset sets some of them (lamina-server -m MODEL.gguf "
                   "--check shows those a start gives the model).",
                   0, kWidth, 0);
    out += program::options_help(program::kListServed, true, kWidth, kCol);
    return out;
}

}  // namespace lamina
