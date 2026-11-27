// lamina's commands and settings table (src/server/params.hpp): every option names one key and the other way round,
// a command first, the command line's values by their kinds (switches and their --no- forms, numbers, lists given
// again, NAME=VALUE pairs, JSON), the engine's options apart, each command's own options, the errors, a section merged
// key by key, the config's keys the table lacks, and the commands' help naming every option, serve's the engine's a
// server takes too.  (CPU only.)
#include "config.hpp"
#include "lamina/program/options.hpp"
#include "params.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

using lamina::json;

static int fails = 0;
#define CHECK(c)                                                                                                       \
    do {                                                                                                               \
        if (!(c)) {                                                                                                    \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);                                          \
            ++fails;                                                                                                   \
        }                                                                                                              \
    } while (0)

namespace {

lamina::CommandLine parse(std::vector<std::string> words) {
    words.insert(words.begin(), "lamina");
    std::vector<char*> argv;
    for (auto& w : words) argv.push_back(w.data());
    return lamina::parse_command_line((int) argv.size(), argv.data());
}

bool refused(std::vector<std::string> words) {
    try {
        parse(std::move(words));
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

}   // namespace

int main() {
    // one key a flag, one flag a key; none is a command's own
    std::set<std::string> flags, keys;
    for (const lamina::Param& p : lamina::kParams) {
        if (p.flag.empty()) continue;
        CHECK(keys.insert(std::string(p.key)).second);
        CHECK(flags.insert(std::string(p.flag)).second);
        std::string a(p.aliases);
        for (size_t at = 0; at < a.size();) {
            const size_t sp = std::min(a.find(' ', at), a.size());
            CHECK(flags.insert(a.substr(at, sp - at)).second);
            at = sp + 1;
        }
    }
    for (const char* c : {"--config", "--help", "-h", "--version", "-v", "--verbose", "--url", "--runs", "--clients",
                          "--long", "--think", "--out", "--compare"})
        CHECK(flags.count(c) == 0);

    bool neg = false;
    CHECK(lamina::find_param("-m")->key == "model");
    CHECK(lamina::find_param("--temp")->key == "sampling.temperature");
    CHECK(lamina::find_param("--no-lazy", &neg)->key == "lazy_load" && neg);
    CHECK(lamina::find_param("--no-mmproj", &neg)->key == "vision" && !neg);
    CHECK(lamina::find_param("--no-port") == nullptr);
    CHECK(lamina::find_param("--max-context") == nullptr);   // the engine's

    const lamina::CommandLine cl = parse({"serve",
                                          "-m",
                                          "x.gguf",
                                          "--port",
                                          "8090",
                                          "--temp",
                                          "0.6",
                                          "--top-k",
                                          "20",
                                          "--lazy",
                                          "--no-answer-room",
                                          "--aliases",
                                          "a,b",
                                          "--alias",
                                          "c",
                                          "--env",
                                          "A=1",
                                          "--env",
                                          "B=x=y",
                                          "--gpu",
                                          "0,2",
                                          "--before-load",
                                          "[\"a\", \"b\"]",
                                          "--script",
                                          "x, y",
                                          "--script",
                                          "z",
                                          "--max-context",
                                          "65536",
                                          "--kv-grow",
                                          "--second-gpu",
                                          "-1",
                                          "--no-mmproj",
                                          "--mmproj",
                                          "p.gguf",
                                          "--open",
                                          "--idle-unload",
                                          "600",
                                          "--queue-timeout",
                                          "1.5",
                                          "--fit-max-tokens=false",
                                          "--reasoning-loop-recovery",
                                          "false",
                                          "--config",
                                          "c.json"});
    const json& s = cl.settings;
    CHECK(s["model"] == lamina::u8_str(std::filesystem::absolute("x.gguf").lexically_normal()));   // a path
    CHECK(s["port"] == 8090 && s["port"].is_number_integer());
    CHECK(s["sampling"]["temperature"] == 0.6 && s["sampling"]["top_k"] == 20);
    CHECK(s["lazy_load"] == true && s["answer_room"] == false && s["open_browser"] == true);
    CHECK((s["aliases"] == json{"a", "b", "c"}));
    CHECK((s["env"] == json{{"A", "1"}, {"B", "x=y"}}));
    CHECK((s["gpu"] == json{0, 2}));
    {   // --gpu again for each card, a card's settings after its name: one item each, in order
        const char* argv2[] = {"lamina", "serve",       "--gpu", "GPU-75af3083",
                               "--gpu",  "RTX 5070 Ti", "--gpu", "GPU-f26e9ba6,clocks=1395-2100/9751,sleep=55us"};
        const lamina::CommandLine c2 = lamina::parse_command_line(8, (char**) argv2);
        CHECK((c2.settings["gpu"] ==
               json{"GPU-75af3083", "RTX 5070 Ti", "GPU-f26e9ba6,clocks=1395-2100/9751,sleep=55us"}));
        CHECK(
            (lamina::gpu_cards(c2.settings) ==
             std::vector<std::string>{"GPU-75af3083", "RTX 5070 Ti", "GPU-f26e9ba6,clocks=1395-2100/9751,sleep=55us"}));
        CHECK(lamina::gpu_list(c2.settings).empty());
    }
    CHECK((s["before_load"] == json{"a", "b"}));
    CHECK((s["scripts"] == json{"x, y", "z"}));
    CHECK((s["vision"] == json{{"mmproj", lamina::u8_str(std::filesystem::absolute("p.gguf").lexically_normal())}}));
    // (--no-mmproj, then --mmproj: the last)
    CHECK(s["idle_unload_s"] == 600 && s["idle_unload_s"].is_number_integer());
    CHECK(s["queue_timeout"] == 1.5 && s["fit_max_tokens"] == false && s["reasoning_loop_recovery"] == false);
    CHECK(cl.engine.size() == 3 && cl.engine[0] == (std::vector<std::string>{"--max-context", "65536"}) &&
          cl.engine[1] == (std::vector<std::string>{"--kv-grow"}) &&
          cl.engine[2] == (std::vector<std::string>{"--second-gpu", "-1"}));
    CHECK(cl.command == "serve" && cl.config == "c.json" && cl.write_config.empty() && !cl.check && !cl.bench &&
          !cl.help);
    CHECK(parse({"serve", "--mmproj", "p.gguf", "--no-mmproj"}).settings["vision"] == false);
    CHECK(parse({"serve", "--port=9000"}).settings["port"] == 9000);
    CHECK(parse({"serve", "-v"}).version && parse({"check", "--version"}).version && !cl.version);
    CHECK(parse({"bench", "-h"}).help && parse({"serve", "--help"}).help);

    // a command first; each takes the table's options and the engine's
    CHECK(refused({}));
    CHECK(refused({"-m", "x.gguf"}));
    CHECK(refused({"--config", "c.json"}));
    CHECK(refused({"run", "-m", "x.gguf"}));
    {
        const lamina::CommandLine c = parse({"check", "-m", "x.gguf", "--verbose", "--max-context", "8192"});
        CHECK(c.command == "check" && c.check && c.verbose && c.settings.contains("model") && c.engine.size() == 1);
        CHECK(refused({"check", "--verbose=1"}));
        // --verbose is check's alone: another command passes it on to the engine, which refuses it at the start
        const lamina::CommandLine sv = parse({"serve", "--verbose"});
        CHECK(!sv.verbose && sv.engine.size() == 1 && sv.engine[0] == (lamina::EngineOption{"--verbose"}));
        const lamina::CommandLine w = parse({"write-config", "w.json", "-m", "x.gguf", "--port", "8090"});
        CHECK(w.command == "write-config" && w.write_config == "w.json" && w.settings["port"] == 8090 && !w.check);
        CHECK(refused({"write-config", "-m", "x.gguf"}));   // the file first
        CHECK(parse({"write-config", "--help"}).help);
    }

    CHECK(refused({"serve", "--port", "x"}));
    CHECK(refused({"serve", "--port"}));
    CHECK(refused({"serve", "--lazy=maybe"}));
    CHECK(refused({"serve", "--no-lazy=1"}));
    CHECK(refused({"serve", "--env", "NOEQ"}));
    CHECK(refused({"serve", "--aliases", "[1"}));
    CHECK(refused({"serve", "stray"}));
    CHECK(refused({"serve", "-x"}));

    // a section merged key by key, anything else replaced
    json base = {
        {"sampling", {{"temperature", 0.7}, {"top_p", 0.95}}}, {"port", 8081}, {"args", {"a"}}, {"vision", false}};
    lamina::merge_settings(base, {{"sampling", {{"temperature", 0.6}}}, {"args", {"b"}}, {"vision", {{"gpu", true}}}});
    CHECK(base["sampling"]["temperature"] == 0.6 && base["sampling"]["top_p"] == 0.95 && base["port"] == 8081);
    CHECK((base["args"] == json{"b"}) && (base["vision"] == json{{"gpu", true}}));

    CHECK((lamina::unknown_keys({{"port", 1},
                                 {"args", json::array()},
                                 {"mcpServers", json::object()},
                                 {"sampling", json::object()},
                                 {"images", json::object()},
                                 {"portt", 2},
                                 {"backend", "x"}}) == std::vector<std::string>{"portt", "backend"}));

    // bench's own options, no keys; the engine keeps its own --bench-* (the qwen35 path's --bench-pp)
    {
        const lamina::CommandLine b = parse({"bench", "--runs", "2", "--clients=4", "--long", "--think", "--out", "out",
                                             "--compare", "old", "--bench-pp", "512"});
        CHECK(b.bench && b.bench_runs == 2 && b.bench_clients == 4 && b.bench_long && b.bench_think && b.bench_options);
        CHECK(std::filesystem::path(b.bench_out).is_absolute() && b.bench_compare.size() == 1);
        CHECK(b.engine.size() == 1 && b.engine[0] == (lamina::EngineOption{"--bench-pp", "512"}) && b.settings.empty());
        const lamina::CommandLine two = parse({"bench", "--compare", "a", "b"});
        CHECK(two.bench_compare.size() == 2 && !two.bench && !two.bench_options);
        const lamina::CommandLine url = parse({"bench", "--url", "http://127.0.0.1:8081"});
        CHECK(url.bench_url == "http://127.0.0.1:8081" && !url.bench && !url.bench_options);
        CHECK(parse({"serve", "--runs", "2"}).engine.size() == 1);   // another command's: the engine's
        CHECK(refused({"bench", "--runs", "0"}));
        CHECK(refused({"bench", "--runs", "two"}));
        CHECK(refused({"bench", "--long=1"}));
        CHECK(refused({"bench", "--compare", "a", "b", "--compare", "c"}));
    }

    // the commands' help: every command, serve's every option and key, the others' own options
    const std::string commands = lamina::commands_help();
    for (const lamina::Command& c : lamina::kCommands)
        CHECK(commands.find("  " + std::string(c.name) + " ") != std::string::npos);
    const std::string help = lamina::command_help("serve");
    for (const std::string& f : flags) CHECK(help.find(f) != std::string::npos);
    for (const std::string& k : keys) CHECK(help.find("[" + k + "]") != std::string::npos);
    const std::string bench_help = lamina::command_help("bench");
    for (const char* f : {"--url", "--runs", "--clients", "--long", "--think", "--out", "--compare"})
        CHECK(bench_help.find(std::string("  ") + f) != std::string::npos);
    CHECK(lamina::command_help("check").find("  --verbose") != std::string::npos);
    CHECK(lamina::command_help("write-config").find("lamina write-config FILE") != std::string::npos);
    // the engine's options for a server, and its deprecated spellings, in --help; none of them the server's own, which
    // would keep it from the engine
    size_t served = 0;
    for (const lamina::program::OptionHelp& o : lamina::program::engine_options()) {
        if (*o.flag == '\0' || (o.lists & lamina::program::kListServed) == 0) continue;
        ++served;
        CHECK(help.find(std::string("  ") + o.flag + (*o.value ? " " : "") + o.value) != std::string::npos);
        if (o.flag[0] == '-') CHECK(lamina::find_param(o.flag) == nullptr);
    }
    CHECK(served > 80);
    for (const lamina::program::DeprecatedOption& d : lamina::program::deprecated_options()) {
        CHECK(help.find(std::string("  ") + d.flag) != std::string::npos);
        CHECK(lamina::find_param(d.flag) == nullptr);
    }

    if (fails == 0) std::printf("params_test: ok (%zu options)\n", keys.size());
    return fails == 0 ? 0 : 1;
}
