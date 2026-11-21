// src/server/main.cpp - lamina-server: the OpenAI, Anthropic and Responses APIs and the web app in front of the engine,
// which runs on a thread of this process (serve/server.py's main).
//
//   lamina-server -m MODEL.gguf                      a model file, its config built from the preset for it
//   lamina-server --config FILE                      the model a run config names
//   lamina-server --engine mock --tokenizer <gguf>   a scripted engine (the tests')
//   lamina-server -m MODEL.gguf --bench              the model measured through this server's API (bench.hpp)
#include "bench.hpp"
#include "config.hpp"
#include "engine.hpp"
#include "frontend.hpp"
#include "http_server.hpp"
#include "images.hpp"
#include "keepawake.hpp"
#include "log.hpp"
#include "mcp.hpp"
#include "model_config.hpp"
#include "params.hpp"
#include "net.hpp"
#include "process.hpp"
#include "prometheus.hpp"
#include "replicas.hpp"
#include "runconfig.hpp"
#include "service.hpp"
#include "telemetry.hpp"
#include "vision.hpp"

#include "lamina/program/engine.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#else
#include <unistd.h>
#endif

using namespace lamina;

namespace {

std::atomic<int> g_stops{0};              // stop requests so far (a second Ctrl+C ends the engine at once)
std::atomic<const char*> g_reason{"sigint"};
std::atomic<bool> g_finished{false};      // the clean stop is over (a Windows console event waits for it)

void request_stop(const char* reason) {
    g_reason = reason;
    if (g_stops.fetch_add(1) >= 1) std::_Exit(130);   // a second Ctrl+C: no waiting for the engine to free its memory
}

#ifdef _WIN32
BOOL WINAPI on_console(DWORD event) {
    switch (event) {
        case CTRL_C_EVENT: request_stop("sigint"); return TRUE;
        case CTRL_BREAK_EVENT: request_stop("sigbreak"); return TRUE;
        case CTRL_CLOSE_EVENT:
        case CTRL_LOGOFF_EVENT:
        case CTRL_SHUTDOWN_EVENT:
            // the system ends the process a few seconds after this returns: wait for the clean stop (up to 4.5 s)
            request_stop(event == CTRL_CLOSE_EVENT ? "console-close" : event == CTRL_LOGOFF_EVENT ? "logoff" : "shutdown");
            for (int i = 0; i < 45 && !g_finished; ++i) Sleep(100);
            return TRUE;
        default: return FALSE;
    }
}
#else
void on_signal(int sig) { request_stop(sig == SIGTERM ? "sigterm" : sig == SIGHUP ? "sighup" : "sigint"); }
#endif

void install_stop_handlers() {
    // every ordinary way of being asked to stop takes Ctrl+C's path (QUIT to the engine, then exit), installed before
    // anything slow starts
#ifdef _WIN32
    SetConsoleCtrlHandler(on_console, TRUE);
#else
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    struct sigaction old {};
    sigaction(SIGHUP, nullptr, &old);
    if (old.sa_handler != SIG_IGN) std::signal(SIGHUP, on_signal);   // nohup: a hangup ignored at start stays ignored
#endif
}

[[noreturn]] void usage_error(const std::string& msg) {
    say_err("usage: lamina-server -m MODEL.gguf | --config FILE [options] [engine options] (--help lists them)");
    say_err("lamina-server: error: " + msg);
    std::exit(2);
}

void open_browser(const std::string& url) {
#ifdef _WIN32
    ShellExecuteW(nullptr, L"open", widen(url).c_str(), nullptr, nullptr, SW_SHOWNORMAL);
#else
    try {
        SpawnOptions o;
        o.args = {"xdg-open", url};
        o.in = SpawnOptions::null;
        o.out = SpawnOptions::null;
        o.err = SpawnOptions::null;
        o.contain = false;
        Process p(o);
        p.wait(-1);
    } catch (const std::exception&) {
    }
#endif
}

long self_pid() {
#ifdef _WIN32
    return (long) GetCurrentProcessId();
#else
    return (long) getpid();
#endif
}

/// a file removed when it goes out of scope, however the run ends (--bench's copy of the expert profile, which the
/// engine saves into at its close: declared before the engine, so removed after it)
struct RemovedAtEnd {
    std::string path;
    RemovedAtEnd() = default;
    RemovedAtEnd(const RemovedAtEnd&) = delete;
    RemovedAtEnd& operator=(const RemovedAtEnd&) = delete;
    ~RemovedAtEnd() {
        if (path.empty()) return;
        std::error_code ec;
        std::filesystem::remove(u8_path(path), ec);
    }
};

std::string join(const std::vector<std::string>& v, const char* sep) {
    std::string out;
    for (const auto& s : v) out += (out.empty() ? "" : sep) + s;
    return out;
}

/// opt-in data-parallel replicas: one `lamina --serve` child per card group, each with its own cards, log and (Linux)
/// CPU cores, started together; the ReplicaEngine in front spreads the requests
std::shared_ptr<Engine> start_replicas(const json& cfg, const std::vector<std::vector<int>>& groups,
                                       const std::vector<std::string>& effort_end, bool lazy, bool offline, double silence) {
    const size_t n = groups.size();
    const auto cpus = cpu_sets(n, jget(cfg, "replica_cpus") != false);
    std::string where;
    for (size_t i = 0; i < n; ++i) {
        std::string g;
        for (const int c : groups[i]) g += (g.empty() ? "" : ",") + std::to_string(c);
        where += (where.empty() ? "" : " | ") + g;
    }
    say("[lamina] " + std::to_string(n) + " replicas on the cards " + where + (cpus ? " (own CPU cores each)" : "") +
        ": requests go to the least busy one, a conversation back to its own");
#ifdef _WIN32
    const std::string exe = u8_str(u8_path(self_dir()) / "lamina.exe");
#else
    const std::string exe = u8_str(u8_path(self_dir()) / "lamina");
#endif
    std::vector<std::shared_ptr<LaminaEngine>> engines(n);
    std::vector<std::string> errors(n);
    auto one = [&](size_t i) {
        const json rc = replica_cfg(cfg, i, groups[i]);
        try {
            std::vector<std::string> args = engine_args(rc);
            args.insert(args.end(), effort_end.begin(), effort_end.end());
            EngineProcess p{exe, jstr(rc, "cwd"), child_env(rc, offline), cpus ? (*cpus)[i] : std::vector<int>{}};
            engines[i] = std::make_shared<LaminaEngine>(args, jstr(rc, "log"), lazy, p);
            engines[i]->silence_s = silence;
        } catch (const std::exception& e) {
            errors[i] = e.what();
        }
    };
    const char* serial = std::getenv("LAMINA_REPLICA_START");
    if (serial != nullptr && std::string(serial) == "serial") {
        for (size_t i = 0; i < n; ++i) one(i);
    } else {
        std::vector<std::thread> ts;
        for (size_t i = 0; i < n; ++i) ts.emplace_back(one, i);
        for (auto& t : ts) t.join();
    }
    for (size_t i = 0; i < n; ++i)
        if (!errors[i].empty()) {
            for (auto& e : engines)
                if (e != nullptr) {
                    try {
                        e->close();
                    } catch (const std::exception&) {
                    }
                }
            throw std::runtime_error("replica " + std::to_string(i) + " did not start: " + errors[i]);
        }
    const json& lt = jget(cfg, "replica_load_tokens");
    return std::make_shared<ReplicaEngine>(engines, truthy(lt) && lt.is_number() ? lt.get<int64_t>() : 3000);
}

std::string server_names(const McpHub& hub) {
    std::string out;
    for (const auto& srv : hub.servers) out += (out.empty() ? "" : ", ") + srv->name;
    return out;
}

/// "k=v, ..." as Python prints a dict's items
std::string items(const json& d) {
    std::string out;
    for (auto it = d.begin(); it != d.end(); ++it)
        out += (out.empty() ? "" : ", ") + it.key() + "=" + (it.value().is_string() ? it.value().get<std::string>() : py_repr_value(it.value()));
    return out;
}

/// the command line's engine options as the engine's own parser reads them, for the family of the config's model:
/// false when it refuses one (said, the engine's line above it)
bool engine_options_ok(const json& cfg, const std::vector<EngineOption>& opts) {
    std::vector<std::string> line{"lamina", "--serve", "--native", model_file(cfg)};
    for (const EngineOption& o : opts) line.insert(line.end(), o.begin(), o.end());
    std::vector<char*> argv;
    for (std::string& w : line) argv.push_back(w.data());
    argv.push_back(nullptr);
    if (lamina::program::check_args((int) line.size(), argv.data()) == 0) return true;
    say_err("[lamina] the engine of this model does not take an option of the command line (above)");
    return false;
}

/// -m: the GPU the engine runs on, as the args name it now (an engine option may have named another)
void say_main_gpu(const json& cfg, const std::vector<GpuInfo>& gpus) {
    const std::vector<int> on = engine_gpus(cfg, gpus);
    for (const GpuInfo& g : gpus)
        if (!on.empty() && g.index == on.front())
            say("[lamina] it runs on GPU " + std::to_string(g.index) + ", " + g.name + " (--gpu names another)");
}

/// a model the file or the command line names for a config with its own engine options: in place of their --native,
/// and of the vision section's model
void use_model(json& cfg, const std::string& model) {
    json args = jget(cfg, "args").is_array() ? cfg["args"] : json::array();
    bool named = false;
    for (size_t i = 0; i + 1 < args.size(); ++i)
        if (args[i] == "--native") {
            args[i + 1] = model;
            named = true;
        }
    if (!named) {
        args.push_back("--native");
        args.push_back(model);
    }
    cfg["args"] = std::move(args);
    if (jget(cfg, "vision").is_object()) cfg["vision"]["model"] = model;
}

/// the pictures as the merged settings have them: "vision" false (--no-mmproj) takes --vision out of the engine's args;
/// a section that names no model file reads the model's, one without a projector is refused
void settle_vision(json& cfg) {
    const json& v = jget(cfg, "vision");
    if (v.is_object()) {
        if (jstr(v, "mmproj").empty())
            usage_error("the vision section (--vision-*) names no image projector: --mmproj FILE, or --no-mmproj");
        if (jstr(v, "model").empty()) cfg["vision"]["model"] = model_file(cfg);
        return;
    }
    if (!cfg.contains("vision") || !jget(cfg, "args").is_array()) return;
    json args = json::array();
    for (const auto& x : cfg["args"])
        if (x != "--vision") args.push_back(x);
    cfg["args"] = std::move(args);
}

/// Why the image encoder cannot borrow the engine's VRAM for its pictures (the vision section's "lend"), "" when it
/// can: the loan is memory of an engine in this process, on a card the engine works on (Flash-Next lends a card's
/// expert cache, the main GPU's or a tier's; Hy4-flash its main GPU's slots; the qwen35 path its prompt path's
/// staging buffers, or VRAM it leaves free - with batch slots too, beside the requests that decode).  `arch`: the
/// model's ("": not read).
std::string no_lending(const json& cfg, const std::string& arch, bool replicas, std::vector<GpuInfo>& gpus) {
    const json& v = jget(cfg, "vision");
    if (!truthy(jget(v, "gpu"))) return "the encoder runs on the CPU";
    if (replicas) return "the replicas' engines are processes of their own";
    const json& dev = jget(v, "cuda_device");
    if (dev.is_null()) return "";
    if (gpus.empty()) gpus = nvidia_gpus();
    const std::vector<int> on = engine_gpus(cfg, gpus);
    const int card = find_gpu(dev.is_string() ? dev.get<std::string>() : py_dumps(dev), gpus);
    if (card < 0 || on.empty()) return "";   // (nvidia-smi lists nothing: the engine answers when asked)
    if (std::find(on.begin(), on.end(), card) == on.end()) return "the engine does not work on the encoder's card";
    if (arch == "hy_v4_flash" && card != on.front()) return "Hy4-flash's engine lends on its main GPU alone";
    if ((arch == "inkling2" || arch == "inkling2_loop") && card != on.front()) return "Inkling 2's engine lends on its own GPU alone";
    return "";
}

/// a vision section that asks for a GPU and names none: the card pick_vision_gpu chooses, by its UUID, said in the
/// window (the encoder would otherwise take CUDA's first); one that names its card by nvidia-smi's number: that
/// card's UUID (the encoder finds its card among this process's CUDA devices, whose order is another).  With
/// "lend", `lend_no` is why the engine cannot lend (no_lending; "": it can), which decides what a section that names
/// no card gets: the engine's main GPU for a loan; where there is none to be had (replicas), the CPU - "lend" asks
/// for an encoder that keeps no VRAM, and one kept on an engine's card would take the model's (replicas are
/// engines in processes of their own, which do not see this process's VRAM when they size their caches).  A section
/// that names its card keeps the encoder there, with its VRAM.
void settle_vision_gpu(json& cfg, std::vector<GpuInfo>& gpus, const std::string& arch, bool replicas, std::string& lend_no) {
    lend_no.clear();
    // "gpu" naming a card (a name, a UUID's start, a PCI bus id, a number): the encoder on it, by its UUID
    if (const json& g = jget(jget(cfg, "vision"), "gpu"); g.is_string() && !strip(g.get<std::string>()).empty()) {
        if (gpus.empty()) gpus = nvidia_gpus();
        std::string why;
        const int i = find_gpu(g.get<std::string>(), gpus, &why);
        if (i < 0) usage_error("vision.gpu " + g.get<std::string>() + ": " + why);
        for (const GpuInfo& x : gpus)
            if (x.index == i) cfg["vision"]["cuda_device"] = x.uuid;
        cfg["vision"]["gpu"] = true;
    }
    const json& v = jget(cfg, "vision");
    const json& dev = jget(v, "cuda_device");
    const bool named = !dev.is_null() && !(dev.is_string() && strip(dev.get<std::string>()).empty());
    if (!v.is_object() || !truthy(jget(v, "gpu"))) return;
    const bool lend_asked = truthy(jget(v, "lend"));
    if (named) {
        const std::string spec(strip(dev.is_string() ? dev.get<std::string>() : py_dumps(dev)));
        if (std::all_of(spec.begin(), spec.end(), ::isdigit)) {
            if (gpus.empty()) gpus = nvidia_gpus();
            for (const GpuInfo& g : gpus)
                if (g.index == std::stoi(spec) && !g.uuid.empty()) cfg["vision"]["cuda_device"] = g.uuid;
        }
        if (lend_asked) lend_no = no_lending(cfg, arch, replicas, gpus);
        return;
    }
    if (gpus.empty()) gpus = nvidia_gpus();
    const std::vector<int> engine = engine_gpus(cfg, gpus);
    if (lend_asked) lend_no = no_lending(cfg, arch, replicas, gpus);
    if (lend_asked && !lend_no.empty()) {
        cfg["vision"]["gpu"] = false;
        say("[lamina] the image encoder on the CPU: it has no VRAM to borrow (" + lend_no + "), and \"lend\" has it keep "
            "none; --no-vision-lend puts it on a GPU with VRAM of its own, --vision-cuda-device names that card");
        return;
    }
    // "lend": the engine's main GPU, where every engine lends; without it a card beside that one
    const bool lend = lend_asked && !engine.empty();
    const int pick = lend ? engine.front() : pick_vision_gpu(gpus, engine.empty() ? -1 : engine.front());
    for (const GpuInfo& g : gpus)
        if (g.index == pick) {
            cfg["vision"]["cuda_device"] = g.uuid;
            say("[lamina] the image encoder on GPU " + std::to_string(g.index) + ", " + g.name +
                (lend ? ", the engine's main one, which lends it VRAM for a picture" : "") +
                " (--vision-cuda-device names another)");
        }
}

int run(int argc, char** argv) {
    no_throttle_self();   // #691: a minimized console window must not slow the engine down
    CommandLine cl;
    try {
        cl = parse_command_line(argc, argv);
    } catch (const std::invalid_argument& e) {
        usage_error(e.what());
    }
    if (cl.help) {
        std::fputs(params_help().c_str(), stdout);
        return 0;
    }
    if (cl.version) {
        std::fputs("lamina-server " LAMINA_VERSION "\n", stdout);
        return 0;
    }
    if (!cl.write_config.empty() && cl.check) usage_error("--write-config or --check: not both");
    const bool attached = !cl.bench_url.empty();   // --bench-url: a server that runs, measured from here
    if (cl.bench_compare.size() == 2) {            // two reports side by side, nothing run
        if (cl.bench || attached || cl.bench_options) usage_error("--bench-compare OLD NEW compares two reports: no --bench with it");
        return bench::compare_only(cl.bench_compare[0], cl.bench_compare[1]);
    }
    if (cl.bench && attached) usage_error("--bench or --bench-url: not both");
    if ((cl.bench_options || !cl.bench_compare.empty()) && !cl.bench && !attached)
        usage_error("the --bench-* options go with --bench (or --bench-url)");
    if ((cl.bench || attached) && (cl.check || !cl.write_config.empty()))
        usage_error("--bench with --check or --write-config: not together");
    bench::Options bench_opts;
    bench_opts.runs = cl.bench_runs;
    bench_opts.clients = cl.bench_clients;
    bench_opts.long_prompt = cl.bench_long;
    bench_opts.think = cl.bench_think;
    bench_opts.out = cl.bench_out;
    bench_opts.compare = cl.bench_compare.empty() ? "" : cl.bench_compare[0];
    bench_opts.command = attached ? "lamina-server --bench-url " + cl.bench_url : "lamina-server --bench";
    if (cl.bench_clients > 0) bench_opts.command += " --bench-clients " + std::to_string(cl.bench_clients);
    if (cl.bench_long) bench_opts.command += " --bench-long";
    if (cl.bench_think) bench_opts.command += " --bench-think";
    if (cl.bench_runs != 3) bench_opts.command += " --bench-runs " + std::to_string(cl.bench_runs);
    for (const EngineOption& eo : cl.engine) {   // the ones the server sets itself
        if (eo[0] == "--native") usage_error("--native: the model is -m MODEL.gguf, or the config's");
        if (eo[0] == "--serve") usage_error("--serve: the server runs the engine in serve mode itself");
        if (eo[0] == "--vision")
            usage_error("--vision: pictures come with a projector (--mmproj; with -m one beside the model, in a config "
                        "its \"vision\" section)");
    }
    install_stop_handlers();
    if (attached) {   // nothing is loaded here: the client alone
        if (!cl.config.empty() || cl.settings.contains("model") || !cl.engine.empty())
            usage_error("--bench-url measures a server that runs: no -m, --config or engine options with it");
        std::string url(rstrip_chars(cl.bench_url, "/"));
        const json given = cl.settings.contains("api_key") ? cl.settings["api_key"] : json(nullptr);
        const char* env_key = std::getenv("LAMINA_API_KEY");
        const std::vector<std::string> keys = key_list(truthy(given) ? given : json(env_key != nullptr ? env_key : ""));
        return bench::run({url, keys.empty() ? "" : keys[0]}, json(nullptr), bench_opts, "", 0,
                          [] { return g_stops.load() > 0; });
    }
    if (!cl.config.empty()) cl.config = u8_str(std::filesystem::absolute(u8_path(cl.config)));   // before the change to its cwd
    json file_cfg = json::object();
    try {
        if (!cl.config.empty()) file_cfg = read_config(cl.config);
    } catch (const ConfigError& e) {
        say_err(std::string("[lamina] ") + e.what());
        return 1;
    }
    for (const std::string& k : unknown_keys(file_cfg))
        say("[lamina] the config's \"" + k + "\" is no key lamina-server reads (lamina-server --help lists them): ignored");
    // the settings: the preset's (with -m, or a config's "model" when it has no "args"), the file's over them, the
    // command line's over both
    json given = file_cfg;
    merge_settings(given, cl.settings);
    const std::string engine_kind = jstr(given, "engine").empty() ? "lamina" : jstr(given, "engine");
    if (engine_kind != "lamina" && engine_kind != "mock")
        usage_error("engine (--engine): lamina or mock, not '" + engine_kind + "'");
    const bool mock = engine_kind == "mock";
    const std::string model = jstr(given, "model");
    if (mock) {
        if (cl.check || !cl.write_config.empty() || !cl.engine.empty())
            usage_error("the mock engine takes no --check, --write-config or engine options");
        if (jstr(given, "tokenizer").empty())
            usage_error("the mock engine needs --tokenizer <a model's .gguf> (its tokenizer and chat template)");
    } else if (model.empty() && !jget(file_cfg, "args").is_array() && !cl.check) {
        usage_error("a model is needed: -m MODEL.gguf (a model file) or --config FILE (a run config)");
    }
    const bool from_preset = !mock && !model.empty() && !jget(file_cfg, "args").is_array();
    json cfg = json::object();
    std::vector<GpuInfo> gpus;   // nvidia-smi's: the preset's main GPU, --check's list
    std::string model_arch;      // the preset's model's general.architecture
    std::string preset;          // the preset
    std::vector<std::string> preset_keys;   // its keys, which --write-config renews
    std::vector<std::string> notes;   // the preset's lines for the window: the model, the pictures, the log
    json preset_base = json::object();   // the preset's settings, under the file's (the Settings view compares)
    if (from_preset || cl.check) gpus = nvidia_gpus();
    try {
        if (from_preset) {
            // a relative path of the file's starts from its cwd (the command line's are absolute already)
            auto files = [&](const std::string& p) {
                const std::string cwd = jstr(file_cfg, "cwd");
                return cwd.empty() || p.empty() || u8_path(p).is_absolute() ? p : u8_str(u8_path(cwd) / u8_path(p));
            };
            ModelOptions mo;
            mo.model = files(model);
            const json& v = jget(given, "vision");
            mo.no_mmproj = v.is_boolean() && !v.get<bool>();
            mo.mmproj = v.is_object() ? files(jstr(v, "mmproj")) : "";
            mo.written = !cl.write_config.empty() ? cl.write_config : cl.config;   // what a run writes goes beside it
            mo.allowed_gpus = gpu_list(given);
            ModelConfig mc = config_from_model(mo, gpus);
            cfg = std::move(mc.cfg);
            preset_base = cfg;
            model_arch = mc.model.arch;
            preset = mc.preset;
            preset_keys = std::move(mc.preset_keys);
            notes = std::move(mc.notes);
        }
        merge_settings(cfg, file_cfg);
        merge_settings(cfg, cl.settings);
        // -m with "gpu" numbers (the cards it may use): the preset's cards as config_from_model picked them among those
        if (from_preset && !gpu_list(cfg).empty() && gpu_cards(preset_base)) cfg["gpu"] = preset_base["gpu"];
        if (!from_preset && !model.empty()) use_model(cfg, model);
    } catch (const ConfigError& e) {
        if (cl.check) return check_report(nullptr, gpus, "", e.what()) ? 0 : 1;
        say_err(std::string("[lamina] ") + e.what());
        return 1;
    }
    split_joined_args(cfg);
    lift_reasoning_effort_arg(cfg);
    if (!cl.engine.empty()) {   // in place of the preset's or the config's, read as the engine reads them
        apply_engine_options(cfg, cl.engine);
        if (!engine_options_ok(cfg, cl.engine)) return 2;
    }
    settle_vision(cfg);
    if (from_preset && !cl.check) {
        for (const std::string& n : notes) say(n);
        say_main_gpu(cfg, gpus);
    }
    // the pictures' card, and with "lend" whether this model's engine can lend (the model's architecture, replicas)
    std::string lend_no;
    {
        const json& v = jget(cfg, "vision");
        std::string arch = model_arch;
        bool replicas = false;
        if (!mock && v.is_object() && truthy(jget(v, "lend"))) {
            try {
                if (arch.empty()) arch = read_model(model_file(cfg)).arch;
            } catch (const std::exception&) {   // (the start says what is wrong with the file)
            }
            try {
                replicas = !gpu_groups(cfg).empty();
            } catch (const std::exception&) {   // (as the start says)
            }
        }
        settle_vision_gpu(cfg, gpus, arch, replicas, lend_no);
    }
    if (!cl.write_config.empty()) {
        json out;
        try {   // the form a config is written in now: its cards, its options past the engine's defaults, an object
            std::string arch = model_arch;
            if (arch.empty()) arch = read_model(model_file(cfg)).arch;
            out = modern_config(cfg, arch == "qwen4exp");
        } catch (const std::exception& e) {
            say_err(std::string("[lamina] ") + e.what());
            return 1;
        }
        try {
            const std::vector<std::string> kept = write_config(cl.write_config, out, preset_keys);
            json probe = cfg;   // the draft layer, or the drafter, as a start would find it
            if (const std::string w = resolve_draft_layer(probe, model_arch.empty() ? "qwen4exp" : model_arch); !w.empty())
                say(w);
            if (const std::string w = resolve_drafter(probe, model_arch); !w.empty()) say(w);
            say("[lamina] wrote " + cl.write_config + (kept.empty() ? "" : " (kept from the earlier file, now " +
                cl.write_config + ".bak: " + join(kept, ", ") + ")") + "; start it with  lamina-server --config " +
                cl.write_config);
        } catch (const std::exception& e) {
            say_err(std::string("[lamina] ") + e.what());
            return 1;
        }
        return 0;
    }
    if (cl.check) return check_report(cl.config.empty() && model.empty() ? nullptr : &cfg, gpus, preset) ? 0 : 1;
    // --bench: the config as the report shows it, then this run's own server: this PC only, on a port of its own (not
    // the one the user's clients use), loaded at once and kept loaded, the expert profile saved into a copy
    json bench_cfg;
    RemovedAtEnd bench_scratch;
    uint64_t bench_log_from = 0;
    if (cl.bench) {
        bench_cfg = cfg;
        const int64_t listed = jget(cfg, "port").is_number_integer() ? cfg["port"].get<int64_t>() : 8095;
        if (listed > 0 && bench::lamina_answers("http://127.0.0.1:" + std::to_string(listed))) {
            say_err("[lamina] a Lamina server answers on port " + std::to_string(listed) + ": the benchmark needs the GPUs "
                    "to itself. Stop that server, or measure it as it runs:  lamina-server --bench-url http://127.0.0.1:" +
                    std::to_string(listed));
            return 2;
        }
        cfg["host"] = "127.0.0.1";
        if (!cl.settings.contains("port")) cfg["port"] = 0;
        cfg["lazy_load"] = false;
        cfg["idle_unload_s"] = 0;
        cfg["open_browser"] = false;
        cfg.erase("ready_file");
        bench_scratch.path = u8_str(u8_path(temp_dir()) / ("lamina-bench-" + std::to_string(self_pid()) + ".profile.bin"));
        cfg = bench::without_profile_save(cfg, bench_scratch.path);
    }
    // the config's relative paths are its cwd's (itself the start directory's), some read before the change to it
    const std::string cfg_cwd = jstr(cfg, "cwd");
    const std::filesystem::path cfg_dir = std::filesystem::absolute(u8_path(cfg_cwd.empty() ? "." : cfg_cwd));
    if (const std::string log = jstr(cfg, "log"); !log.empty() && !u8_path(log).is_absolute())
        cfg["log"] = u8_str(cfg_dir / u8_path(log));   // the engine's stderr goes to it before the change, the rest after
    std::string ready_path;
    if (const std::string rf = jstr(cfg, "ready_file"); !rf.empty()) {
        const std::filesystem::path p = u8_path(rf);
        std::error_code ec;
        if (!p.is_absolute()) usage_error("ready_file (--ready-file) must be an absolute path");
        if (!std::filesystem::is_directory(p.parent_path(), ec)) usage_error("ready_file (--ready-file): its folder must exist");
        if (std::filesystem::exists(std::filesystem::symlink_status(p, ec)))
            usage_error("ready_file (--ready-file) already exists; verify and remove a stale file before reuse");
        ready_path = rf;
    }
    std::optional<double> shutdown_timeout;
    if (const json& st = jget(cfg, "shutdown_timeout"); !st.is_null()) {
        if (st.is_boolean() || !st.is_number() || !(st.get<double>() >= 0 && st.get<double>() <= 300))
            usage_error("shutdown_timeout (--shutdown-timeout) must be in 0..300 seconds");
        shutdown_timeout = st.get<double>();
    }
    const json max_queue = jget(cfg, "max_queue");
    if (!max_queue.is_null() && (!max_queue.is_number_integer() || max_queue.get<int64_t>() < 0 || max_queue.get<int64_t>() > 4096))
        usage_error("max_queue (--max-queue) must be a whole number in 0..4096");
    const json queue_timeout = jget(cfg, "queue_timeout");
    if (!queue_timeout.is_null() && (queue_timeout.is_boolean() || !queue_timeout.is_number() ||
                                     !std::isfinite(queue_timeout.get<double>()) || queue_timeout.get<double>() < 0.1 ||
                                     queue_timeout.get<double>() > 600))
        usage_error("queue_timeout (--queue-timeout) must be finite seconds in 0.1..600");
    if (!queue_timeout.is_null() && max_queue.is_null())
        usage_error("queue_timeout (--queue-timeout) needs max_queue (--max-queue) to bound pending calls");
    if (cl.settings.contains("parallel")) {   // the command line's (a config's is taken as it always was: engine_args)
        const int64_t n = cl.settings["parallel"].get<int64_t>();
        if (n < 1 || n > 1024) usage_error("--parallel must be in 1..1024; the engine may reduce it to the supported count");
        const json& args = jget(cfg, "args");
        if (args.is_array() && std::any_of(args.begin(), args.end(), [](const json& x) { return x == "--batch" || x == "--slots"; }))
            usage_error("--parallel conflicts with an explicit --batch/--slots in the saved engine args");
    }
    const std::string host = jstr(cfg, "host").empty() ? "127.0.0.1" : jstr(cfg, "host");
    int port = 8095;
    if (cfg.contains("port")) {
        if (!cfg["port"].is_number_integer() || cfg["port"].get<int64_t>() < 0 || cfg["port"].get<int64_t>() > 65535)
            usage_error("port (--port) must be a whole number in 0..65535");
        port = (int) cfg["port"].get<int64_t>();
    }
    // the API key: the command line's, else $LAMINA_API_KEY, else the config's
    json key_value = cfg.contains("api_key") ? cfg["api_key"] : json("");
    bool key_given = cl.settings.contains("api_key");
    if (!key_given)
        if (const char* k = std::getenv("LAMINA_API_KEY")) {
            key_value = k;
            key_given = true;
        }
    if (key_given && key_value.is_string() && strip(key_value.get<std::string>()).empty()) {
        say_err("[lamina] an API key was given but it is empty: set a key, or leave it out");
        return 2;
    }
    json api_key;
    try {
        api_key = api_key_of(key_value);
    } catch (const std::invalid_argument& e) {
        say_err(std::string("[lamina] ") + e.what() + ": set a key, or leave it out");
        return 2;
    }
    const bool loopback = host == "127.0.0.1" || host == "localhost" || host == "::1";
    if (!loopback && !truthy(api_key)) {
        say_err("[lamina] a non-loopback bind requires an inference API key");
        return 2;
    }
    if (port != 0) {   // before the minutes of loading: is the port free?
        if (const int code = probe_listen(host, port)) usage_error(listen_problem(host, port, code));
    }
    // the model's file: its tokenizer and chat template (and, for the engine, its weights)
    const std::string gguf = mock ? jstr(cfg, "tokenizer") : model_file(cfg);
    if (gguf.empty()) usage_error("the config's args name no model file (--native <gguf>)");
    std::unique_ptr<Tokenizer> tok;
    try {
        tok = Tokenizer::from_gguf(gguf);
    } catch (const std::exception& e) {
        say_err(std::string("[lamina] the model's tokenizer could not be read from ") + gguf + ": " + e.what());
        return 1;
    }
    if (!mock) {   // Flash-Next's draft layer and Inkling 2's drafter, found beside the model or warned about in this window
        if (const std::string w = resolve_draft_layer(cfg, tok->architecture); !w.empty()) say(w);
        if (const std::string w = resolve_drafter(cfg, tok->architecture); !w.empty()) say(w);
    }
    std::shared_ptr<McpHub> hub;
    try {
        hub = hub_from_config(cfg, jstr(cfg, "mcp_config"));   // before the minutes of loading: a bad entry stops here
    } catch (const std::invalid_argument& e) {
        say_err(std::string("[lamina] ") + e.what());
        return 1;
    }
    const bool offline = jget(cfg, "offline") == true;
    if (offline && (hub != nullptr || truthy(jget(cfg, "before_load"))))
        usage_error("offline (--offline) requires MCP servers and before_load commands to be disabled in this profile");
    if (offline) {
        const std::vector<std::string> blocked = offline_blockers(cfg);
        if (!blocked.empty()) usage_error("offline (--offline) cannot be strict for this profile: " + join(blocked, "; "));
    }
    std::shared_ptr<Engine> engine;
    std::shared_ptr<Vision> vision;
    bool vision_lends = false;   // its section's "lend", where the engine can lend
    json sampling_defaults = json::object();
    std::optional<std::vector<std::string>> effort_end;
    std::vector<std::vector<int>> groups;
    bool lazy = false;
    // the Monitor's cards (#112), read before the engine loads: engine_gpus runs nvidia-smi when the args name a card,
    // and on Linux a child forked once the expert arena is pinned would copy it where madvise refuses MADV_DONTFORK
    // (platform::keep_from_children)
    std::optional<std::vector<int>> monitor_gpus;
    try {
        if (!mock) {
            groups = gpu_groups(cfg);   // replicas: engines of their own, each a child with its cards and its log
            const std::string log = jstr(cfg, "log");
            if (cl.bench && !log.empty()) {   // the report's excerpt: this run's part of the log
                std::error_code ec;
                const auto size = std::filesystem::file_size(u8_path(log), ec);
                bench_log_from = ec ? 0 : (uint64_t) size;
            }
            if (groups.empty()) {
                apply_engine_env(cfg, offline);   // the engine runs in this process: its environment is this one
                show_vision_gpu(cfg, gpus);       // and the image encoder's
                if (!log.empty()) redirect_stderr(log);   // the engine's messages go to its log; the server's stay here
            }
            monitor_gpus = engine_gpus(cfg);   // in the config's "env", before the engine and the image encoder start
            sampling_defaults = sampling_defaults_from_config(cfg);
            if (!sampling_defaults.empty()) say("[lamina] sampling defaults from the config: " + items(sampling_defaults));
            std::vector<std::string> args = engine_args(cfg);
            if (const std::vector<std::string> old = deprecated_options(file_cfg, args); !old.empty()) {
                say("[lamina] deprecated options, still taken (--write-config FILE writes a config with what takes "
                    "their place; a --gpu card is an item of its \"gpu\"):");
                for (const std::string& o : old) say("         " + o);
            }
            lazy = jget(cfg, "lazy_load") == true;
            if (truthy(jget(cfg, "vision"))) {
                if (!vision_built())
                    throw ConfigError(std::string("\"vision\": ") + vision_not_built());
                say(lazy ? "vision encoder unloaded; it starts with the model ..." : "loading the vision encoder ...");
                // in this process, before the engine: what it takes of a GPU is taken when the engine sizes its caches
                json vcfg = cfg["vision"];   // a relative path is the config's cwd's, as for the engine
                if (const std::string f = jstr(vcfg, "mmproj"); !f.empty() && !u8_path(f).is_absolute())
                    vcfg["mmproj"] = u8_str(cfg_dir / u8_path(f));
                // "lend": where the engine can lend VRAM on the encoder's card; else the encoder keeps its own (the
                // CPU's borrows none: said where its card was settled)
                if (truthy(jget(vcfg, "lend")) && !truthy(jget(vcfg, "gpu"))) vcfg["lend"] = false;
                if (truthy(jget(vcfg, "lend")) && !lend_no.empty()) {
                    say("[lamina] the vision section's \"lend\" is left out (" + lend_no + "): the image encoder keeps "
                        "its VRAM");
                    vcfg["lend"] = false;
                }
                vision_lends = truthy(jget(vcfg, "lend"));
                vision = std::make_shared<Vision>(vcfg, lazy);
            }
            say(lazy ? "model unloaded; the first request loads it ..." : "loading the model (the first start takes a minute or two) ...");
            const double silence = engine_silence_s(cfg);   // #481: checked before the (minutes-long) start
            effort_end = effort_end_args(cfg, *tok);        // #458
            if (effort_end) args.insert(args.end(), effort_end->begin(), effort_end->end());
            if (!cfg_cwd.empty()) {
                std::error_code ec;
                std::filesystem::current_path(cfg_dir, ec);   // the engine's relative paths are the config's cwd's
                if (ec) throw ConfigError("\"cwd\": " + cfg_cwd + " is not a directory");
            }
            if (!groups.empty()) {
                engine = start_replicas(cfg, groups, effort_end.value_or(std::vector<std::string>{}), lazy, offline, silence);
            } else {
                auto le = std::make_shared<LaminaEngine>(args, log, lazy);
                le->silence_s = silence;
                engine = le;
            }
            if (vision != nullptr && vision_lends) {   // the encoder's loans for its pictures, and the first one now
                vision->set_lender([engine](int device, uint64_t bytes, std::string& why) { return engine->lend(device, bytes, why); },
                                   [engine] { engine->unlend(); });
                if (!lazy) vision->warm_up();
            }
        } else {
            std::vector<std::string> scripts;
            for (const auto& x : jget(cfg, "scripts")) scripts.push_back(x.is_string() ? x.get<std::string>() : py_dumps(x));
            if (scripts.empty()) scripts.push_back("Thinking about it.</think>\n\nHello from the mock engine.");
            const json& delay = jget(cfg, "script_delay");
            engine = std::make_shared<MockEngine>(*tok, scripts, 32768, delay.is_number() ? delay.get<double>() : 0.0,
                                                  jget(cfg, "script_cycle") == true);
        }
    } catch (const ConfigError& e) {
        say_err(std::string("[lamina] config ") + e.what());
        return 1;
    } catch (const ReplicaConfigError& e) {
        say_err(std::string("[lamina] config ") + e.what());
        return 1;
    } catch (const std::exception& e) {
        say_err(std::string("[lamina] ") + e.what());
        return 1;
    }
    if (const std::string file = jstr(cfg, "chat_template_file"); !file.empty()) {   // in place of the GGUF's
        std::filesystem::path path = u8_path(file);
        if (!path.is_absolute()) path = cfg_dir / path;
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            say_err("[lamina] config \"chat_template_file\": cannot read " + u8_str(path));
            return 1;
        }
        tok->chat_template.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        say("[lamina] chat template: " + u8_str(path));
    }
    auto tpl = std::make_shared<ChatTemplate>(tok->chat_template);
    const bool fit = jget(cfg, "fit_max_tokens") != false;
    Service svc(engine, *tok, tpl, default_model_name(cfg, tok->architecture), vision, sampling_defaults, fit);
    svc.offline = offline;
    if (!max_queue.is_null()) svc.max_queue = max_queue.get<int64_t>();
    if (!queue_timeout.is_null()) svc.queue_timeout_s = queue_timeout.get<double>();
    try {
        if (!mock) {   // the model card's sampling when nothing else is said
            svc.model_arch = jstr(cfg, "arch").empty() ? tok->architecture : jstr(cfg, "arch");
            json requested = jget(cfg, "parallel");
            if (requested.is_null()) {
                const json& args = jget(cfg, "args");
                for (size_t i = 0; args.is_array() && i + 1 < args.size(); ++i)
                    if ((args[i] == "--batch" || args[i] == "--slots") && args[i + 1].is_string()) {
                        try {
                            requested = std::stoi(args[i + 1].get<std::string>());
                        } catch (const std::exception&) {
                        }
                    }
            }
            const int64_t per_replica = requested.is_number_integer() && requested.get<int64_t>() > 0 ? requested.get<int64_t>() : 1;
            svc.requested_slots = groups.empty() ? per_replica : per_replica * (int64_t) groups.size();
        }
        svc.answer_room = jget(cfg, "answer_room") != false;
        svc.reasoning_close_retry = jget(cfg, "reasoning_close_retry") == true;   // #1053: opt-in
        svc.literal_markers = jget(cfg, "literal_markers") != false;
        if (const json& n = jget(cfg, "short_opening_tokens"); !n.is_null()) {
            if (!n.is_number_integer() || n.get<int64_t>() < 0)
                throw ConfigError("\"short_opening_tokens\" must be a whole number, 0 or more, not " + py_repr_value(n));
            svc.short_opening_tokens = n.get<int64_t>();
        }
        if (truthy(jget(cfg, "images"))) {   // image generation (docs/USING.md, Image generation)
            json icfg = cfg["images"];
            for (const char* k : {"exe", "diffusion_model", "vae", "llm", "out_dir"})
                if (jget(icfg, k).is_string() && !u8_path(icfg[k].get<std::string>()).is_absolute())
                    icfg[k] = u8_str(cfg_dir / u8_path(icfg[k].get<std::string>()));
            svc.images = std::make_shared<images::ImageGen>(icfg, jstr(cfg, "log"));
            const std::vector<std::string> gone = svc.images->missing();
            if (!gone.empty())
                say("[lamina] image generation: missing " + join(gone, ", ") + " - requests fail until the images section names files that are there");
            say("[lamina] image generation: " + py_str(svc.images->model) + " " + jstr(icfg, "quant") + " on " +
                (jstr(icfg, "backend").empty() ? "?" : jstr(icfg, "backend")) + " (POST /v1/images/generations; GPU policy " +
                svc.images->policy + ")");
        }
        const json& ps = jget(cfg, "prevent_sleep");
        if (!ps.is_null() && !ps.is_boolean())
            throw ConfigError("\"prevent_sleep\" must be true or false, not " + py_repr_value(ps));
        svc.keep_awake = KeepAwake::create(ps == true);   // #1727: opt-in, Windows only
        if (!jget(cfg, "reasoning_effort").is_null()) {   // #1641: the thinking level of requests that name none
            svc.config_effort = jstr(clean_shared_defaults({{"reasoning_effort", cfg["reasoning_effort"]}}), "reasoning_effort");
            if (!svc.config_effort.empty())
                say("[lamina] thinking level: " + svc.config_effort + " (reasoning_effort in the config; a request, or the Chat "
                    "settings shared with other apps, can set its own)");
        }
        svc.codex_thread_titles = jget(cfg, "codex_thread_titles") == true;       // #923: opt-in
        svc.codex_compaction_cache = jget(cfg, "codex_compaction_cache") == true; // #924: opt-in
        svc.set_aliases(jget(cfg, "aliases"));   // #297: other names the model answers to
        if (!svc.aliases_now().empty()) say("[lamina] model aliases: " + join(svc.aliases_now(), ", "));
        svc.set_lora_models(jget(cfg, "lora_models"));   // names that pick LoRA adapters
        if (const json lm = svc.lora_models_now(); !lm.empty()) {
            std::vector<std::string> names;
            for (auto it = lm.begin(); it != lm.end(); ++it) names.push_back(it.key());
            say("[lamina] LoRA model names: " + join(names, ", "));
        }
        svc.api_key = api_key;
        if (!truthy(svc.api_key) && cfg.contains("api_key"))   // #569: written, but empty: a warning, not a stop
            say_err("[lamina] the api_key in the config is empty: this server has no API key (anyone who can reach it can use "
                    "it); set one, or leave api_key out");
        svc.cors_origins = origins_of(jget(cfg, "cors_origins"), "cors_origins", true);
        svc.trusted_origins = origins_of(jget(cfg, "trusted_origins"), "trusted_origins", false);
        const char* env_hosts = std::getenv("LAMINA_ALLOWED_HOSTS");
        svc.allowed_hosts = allowed_hosts_of(jget(cfg, "allowed_hosts"), env_hosts != nullptr ? env_hosts : "");
        if (!svc.allowed_hosts.empty())
            say(std::find(svc.allowed_hosts.begin(), svc.allowed_hosts.end(), "*") != svc.allowed_hosts.end()
                    ? "[lamina] Host check off: any name reaches this server (allowed_hosts \"*\")"
                    : "[lamina] also answers to the host names " + join(svc.allowed_hosts, ", ") + " (allowed_hosts)");
        svc.api_monitor = jget(cfg, "api_monitor") == true;
        if (svc.api_monitor)
            say(std::string("[lamina] API request monitor on (/api-monitor): the last 100 requests' prompts and answers are kept "
                            "in memory") + (truthy(svc.api_key) ? "" : "; anyone who can reach this server can read them (no API key)"));
        if (!svc.cors_origins.empty())
            say("[lamina] CORS on /v1/* for " + join(svc.cors_origins, ", ") +
                (truthy(svc.api_key) || std::find(svc.cors_origins.begin(), svc.cors_origins.end(), "*") == svc.cors_origins.end()
                     ? "" : " - WARNING: any web page may use the model (no API key)"));
        svc.idle_unload_s = jget(cfg, "idle_unload_s").is_number() ? cfg["idle_unload_s"].get<double>() : 0.0;
        svc.min_free_vram_mib = jget(cfg, "min_free_vram_mib").is_number() ? cfg["min_free_vram_mib"].get<int64_t>() : 0;
        svc.before_load = truthy(jget(cfg, "before_load")) ? cfg["before_load"] : json(nullptr);
        const json& blt = jget(cfg, "before_load_timeout_s");
        if (truthy(blt)) {
            if (blt.is_number()) svc.before_load_timeout_s = blt.get<double>();
            else say("[lamina] \"before_load_timeout_s\" in the config is not a number; using " + py_g(svc.before_load_timeout_s) + " s");
        }
        svc.before_load_ignore_failure = jget(cfg, "before_load_ignore_failure") == true;
        svc.before_load_strict = jget(cfg, "before_load_strict") == true;
        const std::string mode = jget(cfg, "anthropic_thinking").is_string() && !cfg["anthropic_thinking"].get<std::string>().empty()
                                     ? cfg["anthropic_thinking"].get<std::string>() : "model";   // #278
        if (mode != "model" && mode != "on_request")
            throw ConfigError("anthropic_thinking must be \"model\" or \"on_request\", not " + py_repr_value(json(mode)));
        svc.anthropic_think_unasked = mode == "model";
        const json recovery = cfg.contains("tool_call_recovery") ? cfg["tool_call_recovery"] : json(false);
        if (!recovery.is_boolean()) throw ConfigError("\"tool_call_recovery\" must be true or false, not " + py_repr_value(recovery));
        svc.tool_call_recovery = recovery.get<bool>();
        const json etu = cfg.contains("env_to_user") ? cfg["env_to_user"] : json(false);
        if (!etu.is_boolean()) throw ConfigError("\"env_to_user\" must be true or false, not " + py_repr_value(etu));
        g_env_to_user = etu.get<bool>();
        const json rs = cfg.contains("repeat_stop_tokens") ? cfg["repeat_stop_tokens"] : json(256);   // #606: 0 turns it off
        if (!rs.is_number_integer() || rs.get<int64_t>() < 0)
            throw ConfigError("\"repeat_stop_tokens\" must be a whole number >= 0 (0 = off), not " + py_repr_value(rs));
        svc.repeat_stop_tokens = rs.get<int64_t>();
        svc.effort_end = effort_end.has_value();
        if (!jget(cfg, "reasoning_budget_tokens").is_null()) {   // #123: a default thinking budget for every request
            svc.set_reasoning_budget_tokens(cfg["reasoning_budget_tokens"]);
            const auto budget = svc.reasoning_budget(json::object());
            if (budget && *budget)
                say("[lamina] thinking budget: " + std::to_string(*budget) + " tokens (reasoning_budget_tokens; a request can set its own)");
        }
        json loop = cfg.contains("reasoning_loop_recovery") ? cfg["reasoning_loop_recovery"] : json(false);   // #728
        if (loop == true) loop = "recover";
        if (loop != false && loop != "stop" && loop != "recover")
            throw ConfigError("\"reasoning_loop_recovery\" must be false, \"stop\" or \"recover\", not " + py_repr_value(loop));
        svc.reasoning_loop_recovery = loop.is_string() ? loop.get<std::string>() : "";
        // the Monitor reads every card the engine uses (#112), the first its own (#51)
        svc.gpu_indices = monitor_gpus ? *monitor_gpus : engine_gpus(cfg);
        if (!groups.empty()) {               // replicas: every card of every group
            svc.gpu_indices.clear();
            for (const auto& g : groups) svc.gpu_indices.insert(svc.gpu_indices.end(), g.begin(), g.end());
            svc.replica_gpu_groups = groups;
        }
        svc.gpu_index = svc.gpu_indices.empty() ? 0 : svc.gpu_indices[0];
    } catch (const ConfigError& e) {
        say_err(std::string("[lamina] config ") + e.what());
        return 1;
    } catch (const std::invalid_argument& e) {
        say_err(std::string("[lamina] config ") + e.what());
        return 1;
    }
    // #564: the Settings view's settings: this session's (the merged config), the preset's under the file's, the keys
    // the command line set
    svc.live_config = cfg;
    svc.started_config = cfg;
    svc.preset_config = preset_base;
    for (const auto& e : runconfig::kEditable) {
        bool on_line = false;
        if (e.kind == "arg") {
            for (const EngineOption& eo : cl.engine) on_line = on_line || eo[0] == e.rule;
        } else {
            const json* at = &cl.settings;
            for (const std::string& part : split(std::string(e.key), "."))
                at = at != nullptr && at->is_object() && at->contains(part) ? &(*at)[part] : nullptr;
            on_line = at != nullptr;
        }
        if (on_line) svc.command_line_keys.insert(std::string(e.key));
    }
    if (!cl.config.empty()) {
        svc.config_path = cl.config;   // #564: the web page's Settings view saves into it
        // the Chat settings shared with other apps, from last time
        std::filesystem::path sp = u8_path(cl.config);
        sp.replace_extension("");
        svc.shared_path = u8_str(sp) + ".shared-settings.json";
        try {
            std::ifstream f(u8_path(svc.shared_path), std::ios::binary);
            json saved;
            if (f && py_loads(std::string(std::istreambuf_iterator<char>(f), {}), saved)) {
                const json shared = svc.set_shared(saved, false);
                if (!shared.empty()) say("[lamina] other apps use the Chat settings: " + items(shared));
            }
        } catch (const std::exception&) {
            svc.set_shared(json::object(), false);
        }
    }
    if (hub != nullptr) {
        svc.mcp = hub;
        say("[lamina] starting " + std::to_string(hub->servers.size()) + " MCP server" + (hub->servers.size() != 1 ? "s" : "") +
            " for the web app's chat: " + server_names(*hub));
        hub->start();
    }
    // the HTTP side
    svc.host_names = host_names_for(host, svc.allowed_hosts, svc.trusted_origins);
    svc.telemetry = std::make_shared<Telemetry>(
        [&svc] { return json{{"tok_s", svc.tok_s()}, {"tok_s_mean", svc.tok_s_mean()}, {"prefill_tok_s_mean", svc.prefill_tok_s_mean()}}; },
        svc.gpu_index, svc.gpu_indices);
    HttpServer http(svc);
    http.latencies = std::make_shared<prometheus::Latencies>();
    auto lat = http.latencies;
    svc.observe_latencies = [lat](std::optional<double> ttft, int64_t n, double decode_s, double total_s) {
        lat->observe(ttft, n, decode_s, total_s);
    };
    try {
        http.bind(host, port);
    } catch (const std::exception& e) {
        usage_error(e.what());
    }
    http.start();
    svc.start_idle_unload();
    const int bound = http.port();
    const std::string here = host == "0.0.0.0" || host.empty() || host == "::" ? "127.0.0.1" : host;
    say("ready: http://" + here + ":" + std::to_string(bound) + "/v1  (OpenAI: /v1/chat/completions, Anthropic: /v1/messages, "
        "context " + std::to_string(engine->max_context()) + " tokens" + (vision != nullptr ? ", images on" : "") +
        (truthy(svc.api_key) ? ", API key required" : "") + ")");
    if (!cl.bench)
        say("       open http://" + here + ":" + std::to_string(bound) + "/ in a browser to chat; close this window to stop the model");
    if (!loopback) {   // issue #26: reachable from other devices - at which address, and what can still block it
        const std::vector<std::string> ips = lan_addresses();
        for (const auto& ip : ips)
            say("       from other devices: http://" + ip + ":" + std::to_string(bound) + "/   (API: http://" + ip + ":" +
                std::to_string(bound) + "/v1)");
        if (ips.empty()) say("       from other devices: http://<this PC's IP address>:" + std::to_string(bound) + "/");
        if (!truthy(svc.api_key))
            say("       WARNING: no API key - anyone on your network can use this model. Add \"api_key\": \"...\" to the config "
                "(clients send it as their API key; the web page asks for it)");
#ifdef _WIN32
        say("       nothing arrives? Windows Firewall blocks it until allowed: accept its prompt for lamina-server, or run in an "
            "admin PowerShell:\n         New-NetFirewallRule -DisplayName \"Lamina " + std::to_string(bound) + "\" -Direction "
            "Inbound -Protocol TCP -LocalPort " + std::to_string(bound) + " -Action Allow -Profile Private\n       (and set "
            "this network to Private in Windows' network settings)");
#endif
    }
    if (jget(cfg, "open_browser") == true) {   // --open-browser (--open), or the config's
        const std::string url = "http://" + std::string(host == "0.0.0.0" || host.empty() ? "127.0.0.1" : host) + ":" + std::to_string(bound) + "/";
        std::thread([url] { open_browser(url); }).detach();
    }
    if (!ready_path.empty()) {
        const std::string rh = here.find(':') != std::string::npos && !starts_with(here, "[") ? "[" + here + "]" : here;
        json record = {{"api_state", "ready"}, {"api_url", "http://" + rh + ":" + std::to_string(bound)}, {"bound_host", host},
                       {"bound_port", bound}, {"created_at", now_s()}, {"instance_id", svc.instance_id},
                       {"model_id", svc.model}, {"model_state", lazy ? "unloaded" : "ready"}, {"pid", self_pid()},
                       {"schema_version", 1}};   // sorted keys, as json.dumps(sort_keys=True)
        std::FILE* f = std::fopen(ready_path.c_str(), "wx");
        if (f == nullptr || std::fputs((py_dumps(record, false) + "\n").c_str(), f) < 0 || std::fclose(f) != 0) {
            say_err("[lamina] could not write ready file (OSError)");
            http.stop();
            engine->close();
            return 6;
        }
    }
    int bench_rc = 0;
    if (cl.bench) {   // the suite against this server, then the stop as after Ctrl+C
        const std::vector<std::string> keys = key_list(svc.api_key);
        bench_rc = bench::run({"http://127.0.0.1:" + std::to_string(bound), keys.empty() ? "" : keys[0]}, bench_cfg,
                              bench_opts, jstr(cfg, "log"), bench_log_from, [] { return g_stops.load() > 0; });
    } else {
        while (g_stops.load() == 0) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    say("\n[lamina] stopping (Ctrl+C again to end the engine at once) ...");
    bool drained = true;
    if (shutdown_timeout) {
        int64_t pending = svc.begin_drain();
        if (pending) say("[lamina] draining " + std::to_string(pending) + " inference call(s) for up to " + py_g(*shutdown_timeout) + " s ...");
        pending = svc.wait_for_drain(*shutdown_timeout);
        if (pending) {
            const int64_t cancelled = svc.cancel_owned_requests();
            say("[lamina] drain deadline expired; cancelling " + std::to_string(cancelled) + " owned request(s) ...");
            svc.wait_for_drain(5);
            drained = false;
        }
    } else {   // no time given to finish: the running and the queued requests stop now (a console close allows 4.5 s)
        svc.begin_drain();
        if (const int64_t n = svc.cancel_owned_requests()) say("[lamina] stopping " + std::to_string(n) + " request(s) ...");
        svc.wait_for_drain(5);
    }
    http.stop();
    if (svc.telemetry != nullptr) svc.telemetry->close();
    try {
        engine->close();
    } catch (const EngineStuck& e) {
        say_err(std::string("[lamina] ") + e.what());
        std::_Exit(1);
    }
    vision.reset();   // its directory goes with it (#914)
    if (hub != nullptr) hub->close();
    if (!ready_path.empty()) {   // only this instance's own record
        std::ifstream f(u8_path(ready_path), std::ios::binary);
        json rec;
        if (f && py_loads(std::string(std::istreambuf_iterator<char>(f), {}), rec) && jstr(rec, "instance_id") == svc.instance_id &&
            jget(rec, "pid") == self_pid()) {
            f.close();
            std::error_code ec;
            std::filesystem::remove(u8_path(ready_path), ec);
        }
    }
    g_finished = true;
    say("[lamina] stopped");
    if (bench_rc != 0) return bench_rc;
    return drained ? 0 : 8;
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        say_err(std::string("[lamina] ") + e.what());
        return 1;
    }
}
