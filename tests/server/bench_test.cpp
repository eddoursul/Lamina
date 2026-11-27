// lamina --bench (src/server/bench.hpp) against a scripted server of its own: the suite's shapes and medians,
// the agent session's reuse, the tests a small context skips, failures reported, a stop, the report written and
// redacted, the comparison, the cards a report names, the expert profile kept from the benchmark's saves, and the word
// lists as Python's random.Random(7) picks them (suite 1's prompts).  (CPU only: no GPU, no engine.)
#include <httplib.h>

#include "bench.hpp"
#include "process.hpp"
#include "sha256.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <set>
#include <string>
#include <thread>

using lamina::json;
namespace B = lamina::bench;
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

/// The parts of the Lamina API the benchmark reads.  The conversation cache is the longest common prefix with the
/// previous prompt, 4 characters a token; every answer is 12 tokens at most, 60 tok/s.
class FakeLamina {
public:
    std::atomic<int64_t> ctx{32768};
    std::atomic<int> fail_after{-1};   ///< the requests after this many fail (-1: none)
    std::string gpu_name = "NVIDIA GeForce RTX 9999";

    FakeLamina() {
        srv_.Get("/health", [](const httplib::Request&, httplib::Response& res) {
            reply(res, {{"status", "ok"}, {"service", "lamina"}, {"loaded", true}});
        });
        srv_.Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
            reply(res, {{"object", "list"}, {"data", {{{"id", "fake"}, {"meta", {{"n_ctx", ctx.load()}}}}}}});
        });
        srv_.Get("/props", [](const httplib::Request&, httplib::Response& res) {
            reply(res, {{"total_slots", 4}, {"model_alias", "fake-model"}, {"build_info", "Lamina 9.9.9"}});
        });
        srv_.Get("/v1/status", [](const httplib::Request&, httplib::Response& res) {
            reply(res, {{"machine", {{"gpu", {{"name", "NVIDIA GeForce RTX 9999"}, {"total_mib", 12288}}}, {"ram", {{"total_gib", 64.0}}}}},
                        {"engine", "9.9.9"}, {"vram", {{"free_mib", 1000}}}});
        });
        srv_.Get("/metrics", [this](const httplib::Request&, httplib::Response& res) {
            std::lock_guard<std::mutex> lk(mu_);
            reply(res, {{"engine", {{"version", "9.9.9"}, {"max_context", ctx.load()}, {"model_path", "C:\\Users\\Someone\\models\\x.gguf"}}},
                        {"hardware_static", {{"gpu_name", gpu_name}, {"cpu_name", "Fake CPU 9"}, {"cores", 8}, {"threads", 16}}}});
        });
        srv_.Post("/v1/chat/completions", [this](const httplib::Request& req, httplib::Response& res) { chat(req, res); });
        port_ = srv_.bind_to_any_port("127.0.0.1");
        th_ = std::thread([this] { srv_.listen_after_bind(); });
        srv_.wait_until_ready();
    }
    ~FakeLamina() {
        srv_.stop();
        th_.join();
    }
    std::string base() const { return "http://127.0.0.1:" + std::to_string(port_); }
    B::Target target() const { return {base(), ""}; }
    void set_gpu_name(const std::string& n) {
        std::lock_guard<std::mutex> lk(mu_);
        gpu_name = n;
    }
    /// as just started: no request counted, nothing cached
    void reset() {
        std::lock_guard<std::mutex> lk(mu_);
        requests_ = 0;
        prev_.clear();
    }

private:
    httplib::Server srv_;
    std::thread th_;
    int port_ = 0;
    std::mutex mu_;
    std::string prev_;
    int requests_ = 0;

    static void reply(httplib::Response& res, const json& j) { res.set_content(j.dump(), "application/json"); }

    void chat(const httplib::Request& req, httplib::Response& res) {
        const json body = json::parse(req.body);
        std::string text;
        for (const json& m : body["messages"]) text += m.value("content", "");
        int n;
        size_t common = 0;
        {
            std::lock_guard<std::mutex> lk(mu_);
            n = ++requests_;
            while (common < prev_.size() && common < text.size() && prev_[common] == text[common]) ++common;
            prev_ = text;
        }
        if (fail_after.load() >= 0 && n > fail_after.load()) {
            res.status = 500;
            res.set_content(R"({"error": "boom"})", "application/json");
            return;
        }
        const int64_t prompt = std::max<int64_t>(1, (int64_t) text.size() / 4);
        const int64_t cached = std::min<int64_t>(prompt, (int64_t) common / 4);
        const int64_t gen = std::min<int64_t>(body.value("max_tokens", 16), 12);
        res.set_chunked_content_provider("text/event-stream", [=](size_t, httplib::DataSink& sink) {
            auto chunk = [&](const json& delta, const json& finish, const json& extra) {
                json d = {{"id", "x"}, {"choices", {{{"index", 0}, {"delta", delta}, {"finish_reason", finish}}}}};
                for (auto it = extra.begin(); it != extra.end(); ++it) d[it.key()] = it.value();
                const std::string s = "data: " + d.dump() + "\n\n";
                sink.write(s.data(), s.size());
            };
            chunk({{"role", "assistant"}}, nullptr, json::object());
            for (int64_t i = 0; i < gen; ++i) chunk({{"content", "word "}}, nullptr, json::object());
            chunk(json::object(), "stop",
                  {{"usage", {{"prompt_tokens", prompt}, {"completion_tokens", gen}, {"total_tokens", prompt + gen},
                              {"prompt_tokens_details", {{"cached_tokens", cached}}}}},
                   {"timings", {{"cache_n", cached}, {"prompt_n", prompt - cached}, {"prompt_ms", 100.0},
                                {"prompt_per_second", 1000.0 + n}, {"predicted_n", gen}, {"predicted_ms", 200.0},
                                {"predicted_per_second", 60.0}}}});
            const std::string done = "data: [DONE]\n\n";
            sink.write(done.data(), done.size());
            sink.done();
            return true;
        });
    }
};

json suite(FakeLamina& srv, int64_t ctx, int runs, int clients = 0, bool long_prompt = false) {
    B::Options o;
    o.runs = runs;
    o.clients = clients;
    o.long_prompt = long_prompt;
    return B::run_suite(srv.target(), ctx, o, [](const std::string&) {}, nullptr);
}

std::string read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

double med(const json& stat) { return stat.is_object() ? stat["median"].get<double>() : -1; }

/// a fresh folder for one test's reports
fs::path scratch(const std::string& name) {
    const fs::path d = lamina::u8_path(lamina::temp_dir()) / ("lamina-bench-test-" + name);
    std::error_code ec;
    fs::remove_all(d, ec);
    fs::create_directories(d);
    return d;
}

std::vector<fs::path> dirs_in(const fs::path& d) {
    std::vector<fs::path> out;
    for (const auto& e : fs::directory_iterator(d)) out.push_back(e.path());
    std::sort(out.begin(), out.end());
    return out;
}

void the_word_lists() {
    // tools/ab_engine.py's long_prompt, as Python 3.12 writes it
    CHECK(B::long_prompt(12) == "Here is a list of words:\norbit copper canyon stone market anchor lantern velvet violet stone "
                                "meadow garden\n\nSummarize what kinds of words appear in the list, in five sentences.");
    const std::string a = B::long_prompt(1500), b = B::long_prompt(3700);
    CHECK(lamina::sha256_hex(a) == "d0bd31d1b3008cedabf5d4bbf218cefd9cae14ab95ab0c8e0a7f29066b1b546f");
    CHECK(lamina::sha256_hex(b) == "bd47e15135f6b5e95ef2492c94fc70bc92cb719eb8e3a5622795423fd5326829");
}

void the_statistics() {
    const json s = B::stat({3, 1, 2, nullptr, "x"});
    CHECK(s["median"] == 2 && s["min"] == 1 && s["max"] == 3 && s["n"] == 3 && s["median"].is_number_integer());
    CHECK(B::stat({1, 2})["median"] == 1.5);
    CHECK(B::stat({0.1234, 0.2})["min"] == 0.123);
    CHECK(B::stat({nullptr}).is_null() && B::stat({}).is_null());
    CHECK(B::pct({1, 2, 3, 4}, .5) == 2.5 && std::abs(B::pct({1, 2, 3, 4}, .95) - 3.85) < 1e-9);
}

void the_full_suite(FakeLamina& srv) {
    const json out = suite(srv, 32768, 3, 3);
    CHECK(out["failures"].empty());
    CHECK(med(out["chat"]["decode_tps"]) == 60.0);
    CHECK(out["chat"]["decode_tps"]["n"] == 9);   // 3 questions x 3 runs
    CHECK(out["prompts"].size() == 2 && out["prompts"].contains("4K") && out["prompts"].contains("16K"));
    for (const auto& v : out["prompts"]) {
        CHECK(v["prompt_tps"]["n"] == 3);
        CHECK(med(v["reused_tokens"]) < med(v["prompt_tokens"]) / 100);   // a fresh tag at the start: next to nothing reused
    }
    const double ratio = med(out["prompts"]["16K"]["prompt_tokens"]) / med(out["prompts"]["4K"]["prompt_tokens"]);
    CHECK(ratio > 3.4 && ratio < 4.6);   // the sizes are calibrated, not guessed
    CHECK(out["clients"]["clients"] == 3 && out["clients"]["errors"] == 0);
    CHECK(out["clients"]["rounds"][0]["tokens"] == 3 * 12);
}

void the_agent_session(FakeLamina& srv) {
    const json ag = suite(srv, 32768, 2)["agent"];
    CHECK(ag["sessions"].size() == 2);
    const json& turns = ag["sessions"][0];
    CHECK(turns.size() == 6);
    for (size_t i = 0; i < turns.size(); ++i) CHECK(turns[i]["turn"] == (int) i);
    CHECK(turns[0]["reused_tokens"] == 0);
    for (size_t i = 1; i < turns.size(); ++i) {
        CHECK(turns[i]["reused_tokens"].get<int64_t>() > 0);
        CHECK(turns[i]["read_tokens"].get<int64_t>() < turns[i]["prompt_tokens"].get<int64_t>());
    }
    CHECK(med(ag["hit_pct"]) > 50);
    CHECK(med(ag["reread_tokens"]) < med(ag["new_tokens"]));   // the new file text dominates, not re-reading
    CHECK(turns[5]["prompt_tokens"].get<int64_t>() > turns[1]["prompt_tokens"].get<int64_t>());   // the conversation grows
}

void a_small_context(FakeLamina& srv) {
    srv.ctx = 8192;
    const json out = suite(srv, 8192, 1, 0, true);
    CHECK(out["prompts"].contains("4K"));
    CHECK(out["skipped"].size() == 3 && out["skipped"].contains("16K") && out["skipped"].contains("32K") &&
          out["skipped"].contains("agent"));
    CHECK(out["skipped"]["agent"].get<std::string>().find("below ~10K") != std::string::npos);
    CHECK(out["agent"].empty());   // no request was sent for the agent session
    srv.ctx = 16384;
    const json fits = suite(srv, 16384, 1);
    CHECK(!fits["skipped"].contains("agent") && !fits["agent"]["sessions"].empty());
    srv.ctx = 32768;
}

void failures_are_reported(FakeLamina& srv) {
    srv.reset();
    srv.fail_after = 3;   // the warm-up and the calibration go through, then every request fails
    const json out = suite(srv, 32768, 1);
    srv.fail_after = -1;
    CHECK(!out["failures"].empty());
    for (const json& f : out["failures"]) CHECK(f["error"].get<std::string>().find("HTTP 500") != std::string::npos);
    CHECK(out["chat"]["decode_tps"].is_null());
}

void a_stop(FakeLamina& srv) {
    std::atomic<int> asked{0};
    bool interrupted = false;
    B::Options o;
    o.runs = 3;
    const json out = B::run_suite(srv.target(), 32768, o, [](const std::string&) {},
                                  [&] { return ++asked > 40; }, &interrupted);
    CHECK(interrupted && !out["prompts"].contains("16K"));
    CHECK(B::run(srv.target(), nullptr, o, "", 0, [] { return true; }) == 130);
}

void the_report(FakeLamina& srv) {
    const fs::path d = scratch("report");
    B::Options o;
    o.runs = 2;
    o.clients = 2;
    o.out = lamina::u8_str(d);
    o.command = "lamina --bench-url " + srv.base() + " --bench-clients 2 --bench-runs 2";
    CHECK(B::run(srv.target(), nullptr, o, "", 0, nullptr) == 0);
    std::vector<fs::path> dirs = dirs_in(d);
    CHECK(dirs.size() == 1);
    if (dirs.size() != 1) return;
    const std::string name = lamina::u8_str(dirs[0].filename());
    CHECK(name.size() > 11 && name.substr(10, 22) == "-community-rtx-9999-fa");
    const std::string readme = read_file(dirs[0] / "README.md");
    for (const char* part : {"# Community benchmark on NVIDIA GeForce RTX 9999", "## Hardware and software",
                             "## Model and configuration", "## Method", "## Results", "## Correctness and limitations",
                             "### Agent session", "### Concurrency: 2 clients",
                             "| Configuration | Actual prompt tokens | Reused tokens |", "`lamina --bench-url "})
        CHECK(readme.find(part) != std::string::npos);
    const std::string raw = read_file(dirs[0] / "results.json");
    const json res = json::parse(raw);
    CHECK(res["schema"] == B::kSchema && res["tool"] == "lamina --bench");
    CHECK(res["machine"]["engine_version"] == "9.9.9" && res["machine"]["gpu"] == json::array({"NVIDIA GeForce RTX 9999"}));
    CHECK(med(res["results"]["chat"]["decode_tps"]) == 60.0);
    CHECK(res["params"]["started_server"] == false && res["params"]["clients"] == 2);
    CHECK((readme + raw).find("Someone") == std::string::npos);   // the engine's model_path never reaches the report
    CHECK(raw.find("127.0.0.1") != std::string::npos);            // not a private address: kept
    // a second run the same day does not overwrite the first; the comparison
    CHECK(B::run(srv.target(), nullptr, o, "", 0, nullptr) == 0);
    dirs = dirs_in(d);
    CHECK(dirs.size() == 2);
    if (dirs.size() != 2) return;
    json older = B::load_results(lamina::u8_str(dirs[0]));
    older["results"]["chat"]["decode_tps"]["median"] = 30.0;
    older["machine"]["engine_version"] = "1.0";
    const std::string table = B::compare_table(older, B::load_results(lamina::u8_str(dirs[1] / "results.json")));
    CHECK(table.find("chat decode tok/s") != std::string::npos);
    CHECK(table.find("+100.0%") != std::string::npos);
    CHECK(table.find("engine: 1.0 -> 9.9.9") != std::string::npos);
    CHECK(B::compare_only(lamina::u8_str(dirs[0]), lamina::u8_str(dirs[1])) == 0);
    CHECK(B::compare_only(lamina::u8_str(d / "none"), lamina::u8_str(dirs[1])) == 2);
    std::error_code ec;
    fs::remove_all(d, ec);
}

void no_server() { CHECK(B::run({"http://127.0.0.1:9", ""}, nullptr, B::Options{}, "", 0, nullptr) == 2); }

void the_cards_a_report_names(FakeLamina& srv) {
    // #1985: only the cards the engine uses
    B::LocalGpus two;
    two.gpus = {{"NVIDIA GeForce RTX 3060", 12288, "560.1"}, {"NVIDIA GeForce RTX 9999", 16384, "560.1"}};
    auto names = [](const json& m) { return m["gpu"]; };
    json m = B::collect_machine(srv.target(), {{"gpu", 1}}, two);
    CHECK(names(m) == json::array({"NVIDIA GeForce RTX 9999"}) && m["gpu_vram_mib"] == 16384);
    m = B::collect_machine(srv.target(), {{"gpu", {0, 1}}}, two);
    CHECK(names(m) == json::array({"NVIDIA GeForce RTX 3060", "NVIDIA GeForce RTX 9999"}) && m["gpu_vram_mib"] == 28672);
    m = B::collect_machine(srv.target(), {{"args", {"--gpu", "0"}}}, two);
    CHECK(names(m) == json::array({"NVIDIA GeForce RTX 3060"}));
    m = B::collect_machine(srv.target(), json::object(), two);   // nothing selected: the card the server names
    CHECK(names(m) == json::array({"NVIDIA GeForce RTX 9999"}) && m["gpu_vram_mib"] == 16384);
    CHECK((B::gpu_selection({{"gpu", "0,2"}}) == std::vector<int>{0, 2}) && B::gpu_selection(json::object()).empty());
    // every card the server names, the main GPU first, with their VRAM
    const std::string a = "NVIDIA GeForce RTX 5070 Ti", b = "NVIDIA GeForce RTX 3090";
    B::LocalGpus pc;
    pc.gpus = {{a, 16303, "621.01"}, {b, 24576, "621.01"}, {b, 24576, "621.01"}};
    const json main_and_tiers = {{"args", {"--main-gpu", "1", "--second-gpu", "auto"}}};
    srv.set_gpu_name(b + " + " + a + " + " + b);
    m = B::collect_machine(srv.target(), main_and_tiers, pc);
    CHECK(names(m) == json::array({b, a, b}) && m["gpu_vram_mib"] == 65455);
    srv.set_gpu_name(b + " + " + b);
    m = B::collect_machine(srv.target(), main_and_tiers, pc);
    CHECK(names(m) == json::array({b, b}) && m["gpu_vram_mib"] == 49152);
    // #2001: a card nvidia-smi does not list is the server's own (the engine on another vendor's card)
    B::LocalGpus nv;
    nv.gpus = {{"NVIDIA GeForce RTX 3080 Ti", 12288, "560.1"}};
    nv.cuda = "12.6";
    srv.set_gpu_name("Intel Arc Pro B70");
    m = B::collect_machine(srv.target(), json::object(), nv);
    CHECK(names(m) == json::array({"Intel Arc Pro B70"}) && m["cuda"].is_null());
    srv.set_gpu_name("NVIDIA GeForce RTX 9999");
    // #2062: mixed cards one by one, identical ones counted
    CHECK(B::gpu_slug({{"gpu", {a}}}) == "rtx-5070-ti");
    CHECK(B::gpu_slug({{"gpu", {a, "NVIDIA RTX A4000"}}}) == "rtx-5070-ti-plus-rtx-a4000");
    CHECK(B::gpu_slug({{"gpu", {"NVIDIA RTX A4000", a, "NVIDIA RTX A4000"}}}) == "2x-rtx-a4000-plus-rtx-5070-ti");
    CHECK(B::gpu_slug({{"gpu", json::array()}}) == "cpu");
    json n = {{"gpu", {"NVIDIA GeForce RTX 3060"}}, {"model", "Qwen3.8 Flash-Next"}, {"quant", "IQ3_XXS"}};
    CHECK(B::results_dir_name(n, "2026-10-10") == "2026-10-10-community-rtx-3060-qwen3-8-flash-next-iq3-xxs");
    n["model"] = "qwen-iq3_xxs";   // a quant already in the name is not added twice
    CHECK(B::results_dir_name(n, "2026-10-10") == "2026-10-10-community-rtx-3060-qwen-iq3-xxs");
    CHECK(B::cuda_of_header("| NVIDIA-SMI 576.40   Driver Version: 576.40   CUDA Version: 12.9     |") == "12.9");
    CHECK(B::cuda_of_header("| NVIDIA-SMI 621.01   KMD Version: 621.01   CUDA UMD Version: 13.5     |") == "13.5");
}

void the_redaction() {
    const B::Redactor red = B::redactor({"Jane Roe Laptop"}, "sk-secret-123456");
    const std::string text = red("C:\\Users\\someone_else\\models\\a.gguf\n/home/bob/lamina/x.json\n/Users/alice/Library\n"
                                 "peer 192.168.1.20 and 10.0.0.7 and 100.101.102.103\nAuthorization: Bearer abcdefghijkl12345\n"
                                 "key sk-secret-123456 here\nname Jane Roe Laptop");
    for (const char* s : {"someone_else", "bob", "alice", "192.168.1.20", "10.0.0.7", "100.101.102.103", "abcdefghijkl12345",
                          "sk-secret-123456", "Jane Roe Laptop"})
        CHECK(text.find(s) == std::string::npos);
    CHECK(text.find("<user>") != std::string::npos && text.find("<ip>") != std::string::npos);
    CHECK(text.find("\n/home/<user>/lamina/x.json\n") != std::string::npos);   // the lines stay lines
    const std::string kept = "127.0.0.1:8080 and version 0.1.40.3 and driver 32.0.15.7";
    CHECK(red(kept) == kept);
    const std::string here = lamina::u8_str(lamina::u8_path(lamina::self_dir()) / "lamina-x.json");
    CHECK(red("loading " + here).find("<lamina>") != std::string::npos);
    CHECK(red("loading " + here).find(lamina::self_dir()) == std::string::npos);
    const json deep = B::redact({{"a", {{{"p", "D:\\Users\\zed\\x"}}}}, {"n", 3}}, red);
    CHECK(deep.dump().find("zed") == std::string::npos && deep["n"] == 3);

    const json cfg = {{"exe", "C:\\Users\\me\\Lamina\\engine\\lamina-engine.exe"},
                      {"args", {"--native", "/home/me/models/m-IQ3_XXS.gguf", "--ctx", "65536"}},
                      {"api_key", "hunter2"}, {"model_name", "Fake Model"}, {"log", "/tmp/x.log"}, {"port", 8080},
                      {"vision", {{"mmproj", "D:/Models/x/mmproj-F16.gguf"}, {"token", "t"}, {"gpu", false}}},
                      {"env", {{"LAMINA_FOO", "/home/me/z"}, {"PATH", "/usr/bin"}, {"LAMINA_TOKEN", "t"}}}};
    const json v = B::config_view(cfg);
    CHECK(v["exe"] == "lamina-engine.exe");
    CHECK(v["args"] == json::array({"--native", "m-IQ3_XXS.gguf", "--ctx", "65536"}));
    CHECK(!v.contains("api_key") && !v.contains("log") && v["port"] == 8080);
    CHECK((v["vision"] == json{{"mmproj", "mmproj-F16.gguf"}, {"gpu", false}}));   // nested: paths cut, secrets out
    CHECK((v["env"] == json{{"LAMINA_FOO", "z"}}));
    CHECK(B::detect_quant(v) == "IQ3_XXS");
    CHECK(v.dump().find("hunter2") == std::string::npos);
    CHECK(B::detect_quant({{"args", {"--native", "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"}}}) == "UD-Q4_K_XL");
    // a value with a slash that is no path stays; one that starts as a path is cut
    CHECK((B::config_view({{"args", {"--second-gpu-clocks", "0,1395-2100/9751", "--control-vector-scaled",
                                     "D:\\Models\\cv\\refusal.gguf:1.0", "--expert-profile", "data/expert-profile.bin",
                                     "--log", "../Lamina-data/x.log"}}})["args"] ==
           json::array({"--second-gpu-clocks", "0,1395-2100/9751", "--control-vector-scaled", "refusal.gguf:1.0",
                        "--expert-profile", "expert-profile.bin", "--log", "x.log"})));
}

void the_profile_stays(const fs::path& d) {
    // what the benchmark's engine saves of the expert profile goes into a copy, which the run starts from
    const fs::path learned = d / "learned.bin", copy = d / "copy.bin";
    {
        std::ofstream f(learned, std::ios::binary);
        f << "profile";
    }
    const json by_args = {{"args", {"--expert-profile", lamina::u8_str(learned), "--expert-profile-save", lamina::u8_str(learned),
                                    "--spec", "4"}}};
    json out = B::without_profile_save(by_args, lamina::u8_str(copy));
    CHECK(out["args"][1] == lamina::u8_str(learned) && out["args"][3] == lamina::u8_str(copy) && out["args"][5] == "4");
    CHECK(read_file(copy) == "profile" && by_args["args"][3] == lamina::u8_str(learned));   // the input as it was
    std::error_code ec;
    fs::remove(copy, ec);
    const json by_key = {{"args", {"--spec", "4"}}, {"expert_profile_save", lamina::u8_str(learned)}, {"expert_profile_save_every", 1}};
    out = B::without_profile_save(by_key, lamina::u8_str(copy));
    CHECK(out["expert_profile_save"] == lamina::u8_str(copy) && out["expert_profile_save_every"] == 1);
    CHECK(read_file(copy) == "profile" && read_file(learned) == "profile");
    const json none = {{"args", {"--spec", "4"}}};
    CHECK(B::without_profile_save(none, lamina::u8_str(copy)) == none);
}

}  // namespace

int main() {
    the_word_lists();
    the_statistics();
    the_redaction();
    {
        FakeLamina srv;
        the_full_suite(srv);
        the_agent_session(srv);
        a_small_context(srv);
        failures_are_reported(srv);
        a_stop(srv);
        the_report(srv);
        the_cards_a_report_names(srv);
    }
    no_server();
    const fs::path d = scratch("profile");
    the_profile_stays(d);
    std::error_code ec;
    fs::remove_all(d, ec);
    if (fails == 0) std::printf("bench_test: ok\n");
    return fails == 0 ? 0 : 1;
}
