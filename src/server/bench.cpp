// src/server/bench.cpp - lamina bench (bench.hpp): the suite, the machine, the report and the comparison, in
// the server so that a release's programs run it as they are.
#include <httplib.h>

#include "bench.hpp"

#include "config.hpp"
#include "log.hpp"
#include "process.hpp"

#include "lamina/embedded.hpp"
#include "lamina/platform/memory.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <optional>
#include <regex>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <pwd.h>
#include <sys/utsname.h>
#include <unistd.h>
#endif

namespace lamina::bench {

namespace fs = std::filesystem;

namespace {

constexpr int kChatMaxTokens = 100;
constexpr int kLongMaxTokens = 24;
constexpr int kAgentMaxTokens = 64;
constexpr int kAgentTurns = 5;
constexpr int kAgentWords = 1500;
constexpr int64_t kAgentMinContext = 10240;   // the session's last turn holds about 10K tokens
constexpr int kBurstMaxTokens = 128;
constexpr int kBurstWords = 400;

// the three questions of tools/ab_engine.py
const char* const kPrompts[] = {
    "Write a Python function that merges two sorted lists into one sorted list, with a docstring and two tests.",
    "Explain in two paragraphs how a refrigerator moves heat from inside to outside.",
    "List ten practical tips for learning a new language as an adult, one sentence each.",
};

// the coding agent of tools/agent_turns_bench.py: its system prompt and tools
const char* const kSystem =
    "You are an autonomous coding agent working in the user's repository. You can read files, search the code, edit "
    "files and run commands with the tools below. Work step by step: read the code you need before changing it, make "
    "small and correct edits, keep the existing style, and explain briefly what you did. Never invent file contents: "
    "read them. When the task is done, give a short summary of the changes and how to test them.";
const char* const kTools = R"([
 {"type": "function", "function": {"name": "read_file", "description": "Read a file of the repository (UTF-8 text).",
  "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "path from the repository root"}},
                 "required": ["path"]}}},
 {"type": "function", "function": {"name": "grep", "description": "Search the repository for a regular expression; returns matching lines.",
  "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]}}},
 {"type": "function", "function": {"name": "edit_file", "description": "Replace an exact string in a file with a new one.",
  "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}},
                 "required": ["path", "old", "new"]}}},
 {"type": "function", "function": {"name": "run", "description": "Run a shell command in the repository and return its output.",
  "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}
])";

// the essays of the burst (tools/serve_load.py)
const char* const kTopics[] = {
    "the history of lighthouses and how their lenses worked",
    "how a mechanical watch keeps time",
    "why bridges are built as arches and as suspension spans",
    "the water cycle and what a drought does to a river valley",
    "how bread rises, from yeast to crust",
    "the design of a medieval castle and why walls were stepped",
    "how a transistor switches, explained for a curious teenager",
    "the life of a honey bee colony through one year",
};

const char* const kShare = "https://github.com/eddoursul/Lamina/issues";

/// Python's random.Random(seed) for a seed below 2**32: MT19937 seeded by init_by_array([seed]), and random.choice's
/// _randbelow (Modules/_randommodule.c, Lib/random.py)
class PyRandom {
public:
    explicit PyRandom(uint32_t seed) {
        init_genrand(19650218U);
        const uint32_t key = seed;
        size_t i = 1;
        for (size_t k = kN; k; --k) {   // max(N, the key's length 1)
            mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1664525U)) + key;   // + key[j] + j, j always 0
            if (++i >= kN) {
                mt_[0] = mt_[kN - 1];
                i = 1;
            }
        }
        for (size_t k = kN - 1; k; --k) {
            mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1566083941U)) - (uint32_t) i;
            if (++i >= kN) {
                mt_[0] = mt_[kN - 1];
                i = 1;
            }
        }
        mt_[0] = 0x80000000U;
    }
    /// _randbelow(n): getrandbits(n.bit_length()) until it is below n
    size_t below(size_t n) {
        int bits = 0;
        while ((n >> bits) != 0) ++bits;
        for (;;) {
            const size_t r = next() >> (32 - bits);
            if (r < n) return r;
        }
    }

private:
    static constexpr size_t kN = 624, kM = 397;
    uint32_t mt_[kN] = {};
    size_t index_ = kN;

    void init_genrand(uint32_t s) {
        mt_[0] = s;
        for (size_t i = 1; i < kN; ++i) mt_[i] = 1812433253U * (mt_[i - 1] ^ (mt_[i - 1] >> 30)) + (uint32_t) i;
        index_ = kN;
    }
    uint32_t next() {
        static const uint32_t mag01[2] = {0x0U, 0x9908b0dfU};
        if (index_ >= kN) {
            size_t kk = 0;
            for (; kk < kN - kM; ++kk) {
                const uint32_t y = (mt_[kk] & 0x80000000U) | (mt_[kk + 1] & 0x7fffffffU);
                mt_[kk] = mt_[kk + kM] ^ (y >> 1) ^ mag01[y & 1U];
            }
            for (; kk < kN - 1; ++kk) {
                const uint32_t y = (mt_[kk] & 0x80000000U) | (mt_[kk + 1] & 0x7fffffffU);
                mt_[kk] = mt_[kk + kM - kN] ^ (y >> 1) ^ mag01[y & 1U];
            }
            const uint32_t y = (mt_[kN - 1] & 0x80000000U) | (mt_[0] & 0x7fffffffU);
            mt_[kN - 1] = mt_[kM - 1] ^ (y >> 1) ^ mag01[y & 1U];
            index_ = 0;
        }
        uint32_t y = mt_[index_++];
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680U;
        y ^= (y << 15) & 0xefc60000U;
        y ^= y >> 18;
        return y;
    }
};

std::string join(const std::vector<std::string>& v, const std::string& sep) {
    std::string out;
    for (size_t i = 0; i < v.size(); ++i) out += (i ? sep : "") + v[i];
    return out;
}

std::string upper(std::string s) {
    for (char& c : s)
        if (c >= 'a' && c <= 'z') c = (char) (c - 'a' + 'A');
    return s;
}

/// format(x, ".Nf")
std::string fixed(double x, int digits) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.*f", digits, x);
    return buf;
}

/// a number of a json value (0 when it is none): Python's `x or 0`
double num(const json& v) { return v.is_number() ? v.get<double>() : 0.0; }
int64_t inum(const json& v) {
    if (v.is_number_integer()) return v.get<int64_t>();
    return v.is_number() ? (int64_t) v.get<double>() : 0;
}

json user_msg(const std::string& text) { return {{"role", "user"}, {"content", text}}; }

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void set_timeouts(httplib::Client& cli, double timeout_s) {
    const time_t sec = (time_t) timeout_s;
    const time_t usec = (time_t) ((timeout_s - (double) sec) * 1e6);
    cli.set_connection_timeout(sec, usec);
    cli.set_read_timeout(sec, usec);
    cli.set_write_timeout(sec, usec);
}

/// GET path as JSON; nullopt when it cannot be read, an error status or not JSON
std::optional<json> get_json(const Target& t, const std::string& path, double timeout_s = 20) {
    try {
        httplib::Client cli(t.base);
        if (!cli.is_valid()) return std::nullopt;
        set_timeouts(cli, timeout_s);
        httplib::Headers h;
        if (!t.key.empty()) h.emplace("Authorization", "Bearer " + t.key);
        const httplib::Result res = cli.Get(path, h);
        if (!res || res->status < 200 || res->status >= 300) return std::nullopt;
        json j;
        if (!py_loads(res->body, j)) return std::nullopt;
        return j;
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

/// `try_json(...) or {}`: an object, else an empty one
json object_at(const Target& t, const std::string& path, double timeout_s = 20) {
    const std::optional<json> j = get_json(t, path, timeout_s);
    return j && j->is_object() && !j->empty() ? *j : json::object();
}

bool is_lamina(const Target& t) {
    const std::optional<json> h = get_json(t, "/health", 5);
    return h && jstr(*h, "service") == "lamina";
}

/// one streamed chat request as a client sees it
struct Reply {
    std::optional<double> first;   ///< seconds to the first content or reasoning piece
    double total = 0;
    std::string text;              ///< the content (not the reasoning)
    json usage, timings;
    std::string err;               ///< "" when it went through
    int64_t pieces = 0;            ///< the chunks that carried a piece
};

Reply stream_chat(const Target& t, const json& body, double timeout_s, const std::function<bool()>& stopped) {
    Reply r;
    const auto t0 = std::chrono::steady_clock::now();
    try {
        httplib::Client cli(t.base);
        if (!cli.is_valid()) throw std::runtime_error("not an http:// address: " + t.base);
        set_timeouts(cli, timeout_s);
        httplib::Request req;
        req.method = "POST";
        req.path = "/v1/chat/completions";
        req.headers.emplace("Content-Type", "application/json");
        if (!t.key.empty()) req.headers.emplace("Authorization", "Bearer " + t.key);
        req.body = py_dumps(body);
        int status = 0;
        std::string buf, err_body;
        bool done = false, halted = false;
        auto line_in = [&](std::string_view raw) {
            const std::string_view line = strip(raw);
            if (!starts_with(line, "data:")) return;
            const std::string_view d = strip(line.substr(5));
            if (d == "[DONE]") {
                done = true;
                return;
            }
            json j;
            if (!py_loads(d, j) || !j.is_object()) return;
            if (truthy(jget(j, "error"))) r.err = py_str(j["error"]).substr(0, 300);
            const json& choices = jget(j, "choices");
            if (choices.is_array())
                for (const json& ch : choices) {
                    const json& dl = jget(ch, "delta");
                    const std::string content = jstr(dl, "content"), reasoning = jstr(dl, "reasoning_content");
                    if (content.empty() && reasoning.empty()) continue;
                    if (!r.first) r.first = seconds_since(t0);
                    ++r.pieces;
                    r.text += content;
                }
            if (truthy(jget(j, "usage"))) r.usage = j["usage"];
            if (truthy(jget(j, "timings"))) r.timings = j["timings"];
        };
        req.response_handler = [&](const httplib::Response& head) {
            status = head.status;
            return true;
        };
        req.content_receiver = [&](const char* data, size_t n, uint64_t, uint64_t) {
            if (stopped && stopped()) {
                halted = true;
                return false;
            }
            if (status < 200 || status >= 300) {
                if (err_body.size() < 200) err_body.append(data, std::min(n, 200 - err_body.size()));
                return true;
            }
            buf.append(data, n);
            for (size_t nl; !done && (nl = buf.find('\n')) != std::string::npos;) {
                line_in(std::string_view(buf).substr(0, nl));
                buf.erase(0, nl + 1);
            }
            return !done;
        };
        const httplib::Result res = cli.send(req);
        if (!done && !buf.empty() && status >= 200 && status < 300) line_in(buf);   // a last line without its end
        if (halted) r.err = "stopped";
        else if (status != 0 && (status < 200 || status >= 300)) r.err = "HTTP " + std::to_string(status) + ": " + err_body;
        else if (!res && !done) r.err = "the request failed: " + httplib::to_string(res.error());
    } catch (const std::exception& e) {
        r.err = e.what();
    }
    r.total = seconds_since(t0);
    return r;
}

json chat_body(const json& messages, int max_tokens, bool think, const json* tools) {
    json body = {{"model", "lamina"}, {"stream", true}, {"max_tokens", max_tokens}, {"temperature", 0}, {"messages", messages}};
    if (tools != nullptr) body["tools"] = *tools;
    if (!think) {
        body["reasoning_effort"] = "none";
        body["chat_template_kwargs"] = {{"enable_thinking", false}};
    }
    return body;
}

/// a request's numbers, the engine's own where it reported them
json digest(const Reply& r) {
    const json u = r.usage.is_object() ? r.usage : json::object();
    const json tm = r.timings.is_object() ? r.timings : json::object();
    const json gen = truthy(jget(u, "completion_tokens")) ? u["completion_tokens"] : json(r.pieces);
    const json prompt_tokens = jget(u, "prompt_tokens");
    json cached = jget(jget(u, "prompt_tokens_details"), "cached_tokens");
    if (cached.is_null()) cached = jget(tm, "cache_n");
    json read = jget(tm, "prompt_n");
    if (read.is_null() && prompt_tokens.is_number() && cached.is_number()) read = inum(prompt_tokens) - inum(cached);
    json dec = jget(tm, "predicted_per_second");
    std::string dsrc = "engine";
    const int64_t g = inum(gen);
    if (dec.is_null() && r.first && g > 1 && r.total > *r.first) {
        dec = py_round((double) (g - 1) / (r.total - *r.first), 1);
        dsrc = "client";
    }
    const json pp = jget(tm, "prompt_per_second");
    return {{"ok", r.err.empty() && r.first.has_value()},
            {"err", r.err.empty() ? json(nullptr) : json(r.err)},
            {"ttft_s", r.first ? json(py_round(*r.first, 3)) : json(nullptr)},
            {"total_s", py_round(r.total, 3)},
            {"prompt_tokens", prompt_tokens},
            {"reused_tokens", cached},
            {"read_tokens", read},
            {"generated_tokens", gen},
            {"prompt_tps", pp},
            {"prompt_tps_source", pp.is_null() ? json(nullptr) : json("engine")},
            {"decode_tps", dec},
            {"decode_tps_source", dec.is_null() ? json(nullptr) : json(dsrc)},
            {"prompt_ms", jget(tm, "prompt_ms")},
            {"decode_ms", jget(tm, "predicted_ms")},
            {"drafts_offered", jget(tm, "draft_n")},
            {"drafts_accepted", jget(tm, "draft_n_accepted")}};
}

/// what a value prints as in the progress lines (Python's str(): None, 95.3, 100)
std::string text_of(const json& v) { return py_str(v); }

/// the five files of the agent session (data/bench-agent-files.json: the clone's README and four docs as they were,
/// the first 1,500 words of each, frozen so that every install and version sends the same session)
std::vector<std::pair<std::string, std::string>> agent_files() {
    std::vector<std::pair<std::string, std::string>> out;
    if (const EmbeddedFile* f = bench_file("bench-agent-files.json")) {
        json files;
        if (py_loads(f->data, files) && files.is_array())
            for (const json& x : files) out.emplace_back(jstr(x, "path"), jstr(x, "text"));
    }
    for (int i = 0; (int) out.size() < kAgentTurns; ++i)
        out.emplace_back("generated/part" + std::to_string(i) + ".txt", long_prompt(kAgentWords));
    out.resize(kAgentTurns);
    return out;
}

class Suite {
public:
    Suite(const Target& t, int64_t ctx, const Options& o, std::function<void(const std::string&)> log,
          std::function<bool()> stopped)
        : t_(t), ctx_(ctx), o_(o), log_(std::move(log)), stopped_(std::move(stopped)),
          nonce_(std::to_string((long long) std::time(nullptr))) {
        if (o_.runs < 1) o_.runs = 1;
        py_loads(kTools, tools_);
        out_ = {{"chat", json::object()}, {"prompts", json::object()}, {"agent", json::object()},
                {"clients", json::object()}, {"skipped", json::object()}, {"failures", json::array()}};
    }

    json run() {
        if (!halt()) warm_up();
        if (!halt()) run_chat();
        if (!halt()) run_prompt(4096, "4K");
        if (!halt()) run_prompt(16384, "16K");
        if (!halt()) run_agent();
        if (o_.clients > 0 && !halt()) run_clients(o_.clients);
        if (o_.long_prompt && !halt()) run_prompt(32768, "32K");
        return out_;
    }

    bool interrupted = false;

private:
    Target t_;
    int64_t ctx_;
    Options o_;
    std::function<void(const std::string&)> log_;
    std::function<bool()> stopped_;
    std::string nonce_;
    json tools_;
    json out_;
    double tokens_per_word_ = 1.1, overhead_ = 40;

    bool halt() {
        if (!interrupted && stopped_ && stopped_()) interrupted = true;
        return interrupted;
    }
    std::string tag(const std::vector<std::string>& parts) const {
        std::vector<std::string> all = parts;
        all.push_back(nonce_);
        return join(all, "-");
    }
    Reply chat(const json& messages, int max_tokens, const json* tools = nullptr) {
        return stream_chat(t_, chat_body(messages, max_tokens, o_.think, tools), 3600, stopped_);
    }
    void fail(const std::string& where, const json& d) {
        const std::string why = truthy(jget(d, "err")) ? jstr(d, "err") : "no token came back";
        out_["failures"].push_back({{"where", where}, {"error", why}});
        log_("    FAILED: " + why);
    }

    /// the words that make a prompt of a wanted size: two small requests (200 and 800 words) give the tokens per word
    /// as a slope, so the template's fixed part does not skew it
    void calibrate() {
        std::vector<json> pts;
        for (const int words : {200, 800})
            pts.push_back(jget(digest(chat(json::array({user_msg("[" + tag({"cal", std::to_string(words)}) + "]\n" +
                                                                 long_prompt(words))}), 1)),
                               "prompt_tokens"));
        std::optional<double> slope;
        if (truthy(pts[0]) && truthy(pts[1]) && num(pts[1]) > num(pts[0])) slope = (num(pts[1]) - num(pts[0])) / 600;
        tokens_per_word_ = slope && *slope != 0 ? *slope : 1.1;
        overhead_ = slope ? std::max(0.0, num(pts[0]) - 200 * tokens_per_word_) : 40;
        log_("  calibrated: about " + fixed(tokens_per_word_, 2) + " tokens per word");
    }
    json prompt_for(int tokens, int run) const {
        const int words = std::max(50, (int) std::trunc(((double) tokens - overhead_) / tokens_per_word_));
        return json::array(
            {user_msg("[bench " + tag({std::to_string(run), std::to_string(tokens), "p"}) + "]\n" + long_prompt(words))});
    }
    void warm_up() {
        log_("warming up (not measured) ...");
        chat(json::array({user_msg(kPrompts[0])}), 24);
        if (halt()) return;
        calibrate();
        if (halt()) return;
        chat(prompt_for(1024, -1), 8);
    }

    void run_chat() {
        log_("chat: 3 short questions x " + std::to_string(o_.runs) + " runs, up to " + std::to_string(kChatMaxTokens) +
             " new tokens");
        json rows = json::array();
        for (int run = 0; run < o_.runs && !halt(); ++run)
            for (int i = 0; i < 3 && !halt(); ++i) {
                json d = digest(chat(json::array({user_msg("[" + tag({std::to_string(run), std::to_string(i), "c"}) +
                                                           "] " + kPrompts[i])}),
                                     kChatMaxTokens));
                d["run"] = run;
                d["prompt"] = i;
                rows.push_back(d);
                if (d["ok"] != true) fail("chat", d);
                else
                    log_("  run " + std::to_string(run + 1) + " q" + std::to_string(i + 1) + ": decode " +
                         text_of(d["decode_tps"]) + " tok/s, first token " + text_of(d["ttft_s"]) + " s, " +
                         text_of(d["generated_tokens"]) + " tokens");
            }
        std::vector<json> dec, ttft, gen, pt;
        for (const json& r : rows)
            if (r["ok"] == true) {
                dec.push_back(r["decode_tps"]);
                ttft.push_back(r["ttft_s"]);
                gen.push_back(r["generated_tokens"]);
                pt.push_back(r["prompt_tokens"]);
            }
        out_["chat"] = {{"requests", rows}, {"decode_tps", stat(dec)}, {"ttft_s", stat(ttft)},
                        {"generated_tokens", stat(gen)}, {"prompt_tokens", stat(pt)}};
    }

    void run_prompt(int tokens, const std::string& label) {
        if (ctx_ > 0 && tokens + kLongMaxTokens + 256 > ctx_) {
            out_["skipped"][label] = "the context is " + std::to_string(ctx_) + " tokens";
            log_(label + " prompt: skipped (the context is " + std::to_string(ctx_) + " tokens)");
            return;
        }
        log_(label + " prompt: " + std::to_string(o_.runs) + " fresh prompts of about " + std::to_string(tokens) +
             " tokens, " + std::to_string(kLongMaxTokens) + " new tokens");
        json rows = json::array();
        for (int run = 0; run < o_.runs && !halt(); ++run) {
            json d = digest(chat(prompt_for(tokens, run), kLongMaxTokens));
            d["run"] = run;
            rows.push_back(d);
            if (d["ok"] != true) fail(label, d);
            else
                log_("  run " + std::to_string(run + 1) + ": " + text_of(d["prompt_tokens"]) + " tokens, prompt " +
                     text_of(d["prompt_tps"]) + " tok/s (" + std::to_string(inum(d["reused_tokens"])) +
                     " reused), first token " + text_of(d["ttft_s"]) + " s");
        }
        std::vector<json> pp, ttft, pt, reused;
        for (const json& r : rows)
            if (r["ok"] == true) {
                pp.push_back(r["prompt_tps"]);
                ttft.push_back(r["ttft_s"]);
                pt.push_back(r["prompt_tokens"]);
                reused.push_back(r["reused_tokens"].is_number() ? r["reused_tokens"] : json(0));
            }
        out_["prompts"][label] = {{"target_tokens", tokens}, {"requests", rows}, {"prompt_tps", stat(pp)},
                                  {"ttft_s", stat(ttft)}, {"prompt_tokens", stat(pt)}, {"reused_tokens", stat(reused)}};
    }

    json agent_session(int run) {
        const auto files = agent_files();
        json msgs = json::array(
            {{{"role", "system"}, {"content", std::string(kSystem) + "\n(session " + tag({std::to_string(run), "a"}) + ")"}},
             user_msg("I will show you files from a project one at a time. After each file, answer in one or two "
                      "sentences what it is for. Do not call tools.")});
        json turns = json::array();
        for (int turn = 0; turn <= kAgentTurns && !halt(); ++turn) {
            if (turn > 0) {
                const auto& [rel, body] = files[(size_t) turn - 1];
                msgs.push_back(user_msg("File " + rel + ":\n" + body + "\n\nWhat is this file for? Two sentences."));
            }
            const Reply r = chat(msgs, kAgentMaxTokens, &tools_);
            json d = digest(r);
            d["run"] = run;
            d["turn"] = turn;
            turns.push_back(d);
            if (d["ok"] != true) {
                fail("agent", d);
                break;
            }
            msgs.push_back({{"role", "assistant"}, {"content", r.text.empty() ? std::string("Understood.") : r.text}});
        }
        return turns;
    }

    void run_agent() {
        if (ctx_ > 0 && ctx_ < kAgentMinContext) {
            out_["skipped"]["agent"] = "context " + std::to_string(ctx_) + " is below ~10K";
            log_("agent: skipped: context " + std::to_string(ctx_) + " is below ~10K");
            return;
        }
        log_("agent: a session of " + std::to_string(kAgentTurns + 1) + " turns with four tools, x " +
             std::to_string(o_.runs) + " runs");
        json sessions = json::array();
        for (int run = 0; run < o_.runs && !halt(); ++run) {
            const json turns = agent_session(run);
            sessions.push_back(turns);
            for (const json& d : turns)
                if (d["ok"] == true)
                    log_("  run " + std::to_string(run + 1) + " turn " + text_of(d["turn"]) + ": prompt " +
                         text_of(d["prompt_tokens"]) + ", reused " + text_of(d["reused_tokens"]) + ", read " +
                         text_of(d["read_tokens"]) + ", first token " + text_of(d["ttft_s"]) + " s");
        }
        std::vector<std::vector<json>> follow;   // each session's turns after the first that came back
        for (const json& s : sessions) {
            std::vector<json> f;
            for (const json& d : s)
                if (d["ok"] == true && inum(d["turn"]) > 0) f.push_back(d);
            follow.push_back(std::move(f));
        }
        std::vector<json> hit, reread, fresh, ttft, last, dec, first;
        for (const auto& s : follow) {
            if (s.empty()) continue;
            double reused = 0, prompt = 0;
            for (const json& d : s) {
                reused += num(d["reused_tokens"]);
                prompt += num(d["prompt_tokens"]);
            }
            hit.push_back(reused / std::max(1.0, prompt) * 100);
            last.push_back(s.back()["prompt_tokens"]);
            for (const json& d : s) {
                ttft.push_back(d["ttft_s"]);
                dec.push_back(d["decode_tps"]);
            }
        }
        // the earlier conversation read again: what the previous turn held (its prompt and reply) less what this turn
        // reused; the rest of what was read is the new file text
        for (const json& all : sessions) {
            std::vector<json> s;
            for (const json& d : all)
                if (d["ok"] == true) s.push_back(d);
            int64_t again = 0, read = 0;
            for (size_t i = 1; i < s.size(); ++i) {
                again += std::max<int64_t>(0, inum(s[i - 1]["prompt_tokens"]) + inum(s[i - 1]["generated_tokens"]) -
                                                  inum(s[i]["reused_tokens"]));
                read += inum(s[i]["read_tokens"]);
            }
            reread.push_back(again);
            fresh.push_back(read - again);
            if (!all.empty() && all[0]["ok"] == true) first.push_back(all[0]["ttft_s"]);
        }
        out_["agent"] = {{"sessions", sessions}, {"hit_pct", stat(hit)}, {"reread_tokens", stat(reread)},
                         {"new_tokens", stat(fresh)}, {"turn_ttft_s", stat(ttft)}, {"last_turn_prompt_tokens", stat(last)},
                         {"turn_decode_tps", stat(dec)}, {"first_turn_ttft_s", stat(first)}};
    }

    void run_clients(int n) {
        log_("concurrency: " + std::to_string(n) + " requests at once x " + std::to_string(o_.runs) + " rounds, " +
             std::to_string(kBurstMaxTokens) + " new tokens each");
        json rows = json::array();
        std::vector<double> firsts, lats;
        int64_t errors = 0;
        std::vector<json> tps;
        for (int rd = 0; rd < o_.runs && !halt(); ++rd) {
            std::vector<Reply> res((size_t) n);
            std::vector<std::thread> th;
            const std::string nonce = " " + tag({"b", std::to_string(n), std::to_string(rd)});
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < n; ++i)
                th.emplace_back([&, i] {
                    const std::string prompt = "[request " + std::to_string(i) + nonce + "] Write a clear, well organised "
                                               "essay of about " + std::to_string(kBurstWords) + " words on " +
                                               kTopics[(size_t) i % (sizeof kTopics / sizeof kTopics[0])] +
                                               ". Use plain prose, no lists.";
                    res[(size_t) i] = stream_chat(t_, chat_body(json::array({user_msg(prompt)}), kBurstMaxTokens, o_.think,
                                                                nullptr),
                                                  900, stopped_);
                });
            for (auto& x : th) x.join();
            const double wall = seconds_since(t0);
            int64_t toks = 0;
            json first_s = json::array(), latency_s = json::array(), errs = json::array();
            int64_t error_count = 0;
            for (const Reply& r : res) {
                toks += truthy(jget(r.usage, "completion_tokens")) ? inum(r.usage["completion_tokens"]) : r.pieces;
                if (r.first) {
                    first_s.push_back(*r.first);
                    firsts.push_back(*r.first);
                }
                latency_s.push_back(r.total);
                lats.push_back(r.total);
                if (!r.err.empty()) {
                    if (errs.size() < 3) errs.push_back(r.err);
                    ++error_count;
                }
            }
            errors += error_count;
            const double rate = (double) toks / wall;
            tps.push_back(py_round(rate, 2));
            rows.push_back({{"round", rd}, {"wall_s", py_round(wall, 2)}, {"tokens", toks}, {"total_tps", py_round(rate, 2)},
                            {"first_s", first_s}, {"latency_s", latency_s}, {"errors", errs}, {"error_count", error_count}});
            log_("  round " + std::to_string(rd + 1) + ": " + std::to_string(toks) + " tokens in " + fixed(wall, 1) +
                 " s = " + fixed(rate, 1) + " tok/s total" +
                 (errs.empty() ? "" : "  ERRORS " + py_repr_value(json::array({errs[0]}))));
        }
        auto q = [](const std::vector<double>& v, double p) { return v.empty() ? json(nullptr) : json(py_round(pct(v, p), 3)); };
        out_["clients"] = {{"clients", n},
                           {"rounds", rows},
                           {"total_tps", stat(tps)},
                           {"ttft_p50_s", q(firsts, .5)},
                           {"ttft_p95_s", q(firsts, .95)},
                           {"latency_p50_s", q(lats, .5)},
                           {"latency_p95_s", q(lats, .95)},
                           {"errors", errors}};
    }
};

// ---- the machine

/// a program's output, its lines joined by "\n"; "" when it cannot run or ends with an error
std::string output_of(const std::vector<std::string>& args, double timeout_s = 15) {
    try {
        SpawnOptions o;
        o.args = args;
        o.in = SpawnOptions::null;
        o.out = SpawnOptions::pipe;
        o.err = SpawnOptions::null;
        Process p(o);
        std::string out, line;
        while (p.read_line(line, timeout_s) == 1) out += line + "\n";
        const std::optional<int> code = p.wait(timeout_s);
        return code && *code == 0 ? out : "";
    } catch (const std::exception&) {
        return "";
    }
}

std::string env_of(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr ? v : "";
}

std::string os_text() {
#ifdef _WIN32
    using RtlGetVersionFn = LONG(WINAPI*)(PRTL_OSVERSIONINFOW);
    RTL_OSVERSIONINFOW v{};
    v.dwOSVersionInfoSize = sizeof v;
    if (HMODULE nt = GetModuleHandleW(L"ntdll.dll"))
        if (auto f = reinterpret_cast<RtlGetVersionFn>(reinterpret_cast<void*>(GetProcAddress(nt, "RtlGetVersion")))) f(&v);
    // Python's platform.system(), release() and version(): "Windows 11 (10.0.26100)"
    const std::string release = v.dwMajorVersion == 10 && v.dwBuildNumber >= 22000 ? "11" : std::to_string(v.dwMajorVersion);
    return "Windows " + release + " (" + std::to_string(v.dwMajorVersion) + "." + std::to_string(v.dwMinorVersion) + "." +
           std::to_string(v.dwBuildNumber) + ")";
#else
    struct utsname u {};
    if (uname(&u) != 0) return "Linux";
    return std::string(u.sysname) + " " + u.release;
#endif
}

std::string user_name() {
#ifdef _WIN32
    return env_of("USERNAME");
#else
    for (const char* k : {"LOGNAME", "USER"})
        if (const std::string v = env_of(k); !v.empty()) return v;
    if (const passwd* pw = getpwuid(geteuid())) return pw->pw_name != nullptr ? pw->pw_name : "";
    return "";
#endif
}

std::vector<std::string> host_names() {
    std::vector<std::string> out;
#ifdef _WIN32
    out.push_back(env_of("COMPUTERNAME"));
    wchar_t buf[256];
    DWORD n = 256;
    if (GetComputerNameExW(ComputerNameDnsHostname, buf, &n)) out.push_back(narrow(std::wstring_view(buf, n)));
#else
    char buf[256] = {};
    if (gethostname(buf, sizeof buf - 1) == 0) out.push_back(buf);
#endif
    return out;
}

std::string home_dir() { return env_of(
#ifdef _WIN32
    "USERPROFILE"
#else
    "HOME"
#endif
); }

/// the commit of the Lamina clone the program was built in (its build folder inside it), else null
json git_commit() {
    const std::string top(strip(output_of({"git", "-C", self_dir(), "rev-parse", "--show-toplevel"})));
    if (top.empty()) return nullptr;
    std::ifstream f(u8_path(top) / "CMakeLists.txt", std::ios::binary);
    const std::string cm((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (cm.find("project(lamina") == std::string::npos) return nullptr;
    const std::string head(strip(output_of({"git", "-C", top, "rev-parse", "--short=8", "HEAD"})));
    return head.empty() ? json(nullptr) : json(head);
}

json ram_gib() {
    uint64_t total = 0;
    if (!platform::total_ram(total) || total == 0) return nullptr;
    return py_round((double) total / 1073741824.0, 1);
}

/// a path cut to its file name (the report's settings): a value that starts as a path does (a drive, a separator, ~,
/// ./ or ../) or names a file of a kind the configs name; another value with a slash stays (--second-gpu-clocks'
/// 0,1395-2100/9751)
json base_if_path(const json& v) {
    if (!v.is_string()) return v;
    static const std::regex ext(R"(\.(gguf|bin|json|pack|txt|jinja|so|dll|exe)$)", std::regex::icase);
    static const std::regex start(R"(^([A-Za-z]:[\\/]|[\\/~]|\.\.?[\\/]))");
    const std::string s = v.get<std::string>();
    const bool path = s.find_first_of("/\\") != std::string::npos && std::regex_search(s, start);
    if (!path && !std::regex_search(s, ext)) return v;
    const std::string t(rstrip_chars(s, "\\/"));
    const size_t cut = t.find_last_of("/\\");
    return cut == std::string::npos ? t : t.substr(cut + 1);
}

bool secret_key(const std::string& k) {
    const std::string l = lower(k);
    for (const char* w : {"key", "token", "secret", "password", "auth"})
        if (l.find(w) != std::string::npos) return true;
    return false;
}

/// the paths of a nested value cut to file names, its secret keys left out
json cut_paths(const json& v) {
    if (v.is_string()) return base_if_path(v);
    if (v.is_array()) {
        json out = json::array();
        for (const json& x : v) out.push_back(cut_paths(x));
        return out;
    }
    if (v.is_object()) {
        json out = json::object();
        for (auto it = v.begin(); it != v.end(); ++it)
            if (!secret_key(it.key())) out[it.key()] = cut_paths(it.value());
        return out;
    }
    return v;
}

std::string launch_line(const json& view) {
    std::vector<std::string> words;
    for (const json& a : jget(view, "args")) {
        const std::string s = py_str(a);
        words.push_back(s.find(' ') != std::string::npos ? "\"" + s + "\"" : s);
    }
    const std::string exe = jget(view, "exe").is_string() ? view["exe"].get<std::string>() : "lamina-engine";
    return std::string(strip(exe + " " + join(words, " ")));
}

std::string model_label(const json& machine) {
    const std::string model = truthy(jget(machine, "model")) ? py_str(machine["model"]) : "model";
    const std::string q = jstr(machine, "quant");
    return q.empty() || slug(model, 200).find(slug(q)) != std::string::npos ? model : model + "-" + q;
}

// ---- the report

/// "median (min-max)" of a stat, or "not measured"
std::string fmt(const json& s, int digits = 1, const std::string& unit = "") {
    if (!s.is_object()) return "not measured";
    return fixed(num(s["median"]), digits) + unit + " (" + fixed(num(s["min"]), digits) + "-" + fixed(num(s["max"]), digits) + ")";
}

std::string row(const std::vector<std::string>& cells) { return "| " + join(cells, " | ") + " |"; }

std::vector<std::string> results_table(const json& out) {
    std::vector<std::string> lines = {
        "| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | "
        "Decode tok/s median and range | TTFT seconds median and range |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |"};
    const json& c = out["chat"];
    if (truthy(c))
        lines.push_back(row({"short chat", fmt(jget(c, "prompt_tokens"), 0), "0", fmt(jget(c, "generated_tokens"), 0),
                             std::to_string(inum(jget(jget(c, "ttft_s"), "n"))), "not measured", fmt(jget(c, "decode_tps")),
                             fmt(jget(c, "ttft_s"), 2)}));
    for (auto it = out["prompts"].begin(); it != out["prompts"].end(); ++it) {
        const json& v = it.value();
        lines.push_back(row({it.key() + " prompt, fresh", fmt(jget(v, "prompt_tokens"), 0), fmt(jget(v, "reused_tokens"), 0),
                             std::to_string(kLongMaxTokens) + " cap", std::to_string(inum(jget(jget(v, "ttft_s"), "n"))),
                             fmt(jget(v, "prompt_tps")), "not measured", fmt(jget(v, "ttft_s"), 2)}));
    }
    const json& a = out["agent"];
    if (truthy(a)) {
        int64_t n = 0;
        for (const json& s : jget(a, "sessions"))
            if (s.size() > 1) ++n;
        lines.push_back(row({"agent session, follow-up turns", fmt(jget(a, "last_turn_prompt_tokens"), 0) + " at the last turn",
                             "see below", "up to " + std::to_string(kAgentMaxTokens), std::to_string(n), "see below",
                             fmt(jget(a, "turn_decode_tps")), fmt(jget(a, "turn_ttft_s"), 2)}));
    }
    return lines;
}

/// the measures --compare and the summary show: {name: stat}
json summary_metrics(const json& out) {
    json m = json::object();
    m["chat decode tok/s"] = jget(jget(out, "chat"), "decode_tps");
    m["chat first token (s)"] = jget(jget(out, "chat"), "ttft_s");
    const json& prompts = jget(out, "prompts");
    for (auto it = prompts.begin(); prompts.is_object() && it != prompts.end(); ++it) {
        m[it.key() + " prompt tok/s"] = jget(it.value(), "prompt_tps");
        m[it.key() + " first token (s)"] = jget(it.value(), "ttft_s");
    }
    const json& a = jget(out, "agent");
    if (truthy(a)) {
        m["agent cache hit %"] = jget(a, "hit_pct");
        m["agent re-read tokens/session"] = jget(a, "reread_tokens");
        m["agent turn first token (s)"] = jget(a, "turn_ttft_s");
        m["agent decode tok/s"] = jget(a, "turn_decode_tps");
    }
    const json& c = jget(out, "clients");
    if (truthy(c)) m[text_of(jget(c, "clients")) + " clients total tok/s"] = jget(c, "total_tps");
    return m;
}

std::string short_table(const json& res) {
    std::string out;
    const json m = summary_metrics(res["results"]);
    for (auto it = m.begin(); it != m.end(); ++it) {
        const json& v = it.value();
        if (!truthy(v)) continue;
        char buf[256];
        std::snprintf(buf, sizeof buf, "  %-32s %10.2f   (%.2f - %.2f, n=%lld)", it.key().c_str(), num(v["median"]),
                      num(v["min"]), num(v["max"]), (long long) inum(v["n"]));
        out += (out.empty() ? "" : "\n") + std::string(buf);
    }
    return out;
}

bool any_ok(const json& out) {
    auto ok = [](const json& rows) {
        for (const json& r : rows)
            if (jget(r, "ok") == true) return true;
        return false;
    };
    if (ok(jget(jget(out, "chat"), "requests"))) return true;
    for (const auto& v : jget(out, "prompts"))
        if (ok(jget(v, "requests"))) return true;
    for (const json& s : jget(jget(out, "agent"), "sessions"))
        if (ok(s)) return true;
    return false;
}

std::string today() {
    const std::time_t now = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &now);
#else
    localtime_r(&now, &tm);
#endif
    char buf[16];
    std::strftime(buf, sizeof buf, "%Y-%m-%d", &tm);
    return buf;
}

fs::path unique_dir(const fs::path& parent, const std::string& name) {
    fs::path p = parent / u8_path(name);
    std::error_code ec;
    for (int n = 2; fs::exists(p, ec); ++n) p = parent / u8_path(name + "-" + std::to_string(n));
    return p;
}

void write_file(const fs::path& p, const std::string& text) {
    std::ofstream f(p, std::ios::binary);
    f << text;
    if (!f) throw std::runtime_error("cannot write " + u8_str(p));
}

/// the engine's log lines worth a report, of this run's part
std::vector<std::string> log_excerpt(const std::string& path, uint64_t from, size_t limit = 80) {
    std::vector<std::string> out;
    if (path.empty()) return out;
    std::ifstream f(u8_path(path), std::ios::binary);
    if (!f) return out;
    f.seekg((std::streamoff) from);
    const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    static const char* const keep[] = {"expert cache", "slots", "vram", "ready", "resident", "kv ", "context", "backend",
                                       "gpu", "prefill", "mtp", "drafter", "dfly", "spec", "batch", "version", "hit"};
    for (const std::string_view line : splitlines(text)) {
        if (out.size() >= limit) break;
        const std::string l = lower(line);
        if (l.find("[lamina] done:") != std::string::npos) continue;
        for (const char* k : keep)
            if (l.find(k) != std::string::npos) {
                out.emplace_back(line);
                break;
            }
    }
    return out;
}

fs::path default_parent() { return u8_path(self_dir()) / "bench-results"; }

}  // namespace

// ---- the pieces

bool lamina_answers(const std::string& base) { return is_lamina({base, ""}); }

std::string long_prompt(int words) {
    static const char* const vocab[] = {"river",  "stone", "market", "lantern", "copper", "winter", "garden",
                                        "signal", "harbor", "needle", "orbit",  "velvet", "canyon", "ember",
                                        "ledger", "quartz", "meadow", "anchor", "violet", "thunder"};
    PyRandom rng(7);
    std::string body;
    for (int i = 0; i < words; ++i) {
        if (i) body += ' ';
        body += vocab[rng.below(sizeof vocab / sizeof vocab[0])];
    }
    return "Here is a list of words:\n" + body + "\n\nSummarize what kinds of words appear in the list, in five sentences.";
}

json stat(const std::vector<json>& values) {
    std::vector<double> v;
    bool ints = true;
    for (const json& x : values)
        if (x.is_number()) {
            v.push_back(x.get<double>());
            ints = ints && x.is_number_integer();
        }
    if (v.empty()) return nullptr;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    // statistics.median: the middle value as it is, or the mean of the two middle ones; round(x, 3) keeps an int
    auto as = [&](double x) { return ints ? json((int64_t) x) : json(py_round(x, 3)); };
    const json median = n % 2 ? as(v[n / 2]) : json(py_round((v[n / 2 - 1] + v[n / 2]) / 2, 3));
    return {{"median", median}, {"min", as(v.front())}, {"max", as(v.back())}, {"n", (int64_t) n}};
}

double pct(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    if (v.empty()) return std::nan("");
    const double k = (double) (v.size() - 1) * p;
    const size_t lo = (size_t) k, hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (v[hi] - v[lo]) * (k - (double) lo);
}

Redactor redactor(const std::vector<std::string>& extra, const std::string& key) {
    struct Sub {
        std::string what, label;
        bool word;
    };
    auto subs = std::make_shared<std::vector<Sub>>();
    auto add = [&](const std::string& value, const std::string& label, bool word = false) {
        if (value.size() < 4) return;
        std::vector<std::string> forms = {value, replace_all(value, "\\", "/"), replace_all(value, "/", "\\")};
        std::sort(forms.begin(), forms.end());
        forms.erase(std::unique(forms.begin(), forms.end()), forms.end());
        for (const std::string& f : forms) subs->push_back({f, label, word});
    };
    add(self_dir(), "<lamina>");
    add(home_dir(), "<home>");
    add(temp_dir(), "<tmp>");
    add(user_name(), "<user>", true);
    for (const std::string& h : host_names()) add(h, "<host>", true);
    for (const std::string& x : extra) add(x, "<private>", true);
    add(key, "<key>");
    std::stable_sort(subs->begin(), subs->end(), [](const Sub& a, const Sub& b) { return a.what.size() > b.what.size(); });
    return [subs](const std::string& text) {
        auto word_char = [](unsigned char c) { return std::isalnum(c) || c == '_' || c == '-' || c >= 0x80; };
        auto same = [](const std::string& s, size_t at, const std::string& w) {
            for (size_t i = 0; i < w.size(); ++i)
                if (std::tolower((unsigned char) s[at + i]) != std::tolower((unsigned char) w[i])) return false;
            return true;
        };
        static const std::regex path_user(R"(([A-Za-z]:)?([\\/])(Users|home)([\\/])(?!<)[^\\/\s"':;,]+)", std::regex::icase);
        static const std::regex bearer(R"((bearer\s+)[A-Za-z0-9._~+/=-]{8,})", std::regex::icase);
        static const std::regex ip4(R"(\b(?:192\.168|10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])|169\.254)\.\d{1,3}\.\d{1,3}\b)");
        std::string out;
        const std::vector<std::string> lines = split(text, "\n");   // a line at a time: short inputs for std::regex
        for (size_t li = 0; li < lines.size(); ++li) {
            std::string s = lines[li];
            for (const Sub& sub : *subs) {
                std::string r;
                const size_t n = sub.what.size();
                for (size_t i = 0; i < s.size();) {
                    if (i + n <= s.size() && same(s, i, sub.what) &&
                        (!sub.word || ((i == 0 || !word_char((unsigned char) s[i - 1])) &&
                                       (i + n == s.size() || !word_char((unsigned char) s[i + n]))))) {
                        r += sub.label;
                        i += n;
                    } else {
                        r += s[i++];
                    }
                }
                s = std::move(r);
            }
            s = std::regex_replace(s, path_user, "$1$2$3$4<user>");
            s = std::regex_replace(s, bearer, "$1<key>");
            s = std::regex_replace(s, ip4, "<ip>");
            if (li) out += '\n';
            out += s;
        }
        return out;
    };
}

json redact(const json& v, const Redactor& red) {
    if (v.is_string()) return red(v.get<std::string>());
    if (v.is_array()) {
        json out = json::array();
        for (const json& x : v) out.push_back(redact(x, red));
        return out;
    }
    if (v.is_object()) {
        json out = json::object();
        for (auto it = v.begin(); it != v.end(); ++it) out[red(it.key())] = redact(it.value(), red);
        return out;
    }
    return v;
}

json config_view(const json& cfg) {
    json out = json::object();
    if (!cfg.is_object()) return out;
    for (auto it = cfg.begin(); it != cfg.end(); ++it) {
        const std::string& k = it.key();
        if (secret_key(k) || k == "log" || k == "cwd" || k == "env") continue;
        out[k] = cut_paths(it.value());
    }
    const json& env = jget(cfg, "env");
    if (env.is_object()) {
        json e = json::object();
        for (auto it = env.begin(); it != env.end(); ++it)
            if ((starts_with(it.key(), "LAMINA_") || starts_with(it.key(), "CUDA_")) && !secret_key(it.key()))
                e[it.key()] = base_if_path(json(py_str(it.value())));
        out["env"] = e;
    }
    return out;
}

std::string detect_quant(const json& view) {
    static const std::regex quant(
        R"((?:^|[-_.])((?:UD-)?(?:IQ\d_[A-Z]+|Q\d_[A-Z0-9_]+|Q\d_K(?:_[A-Z]+)?|IQ\d_?[A-Z]*|BF16|F16|Q\d))(?=[-_.]|$))",
        std::regex::icase);
    static const std::regex file(R"(\.(gguf|pack)$|^[\w.-]+$)");
    std::vector<std::string> names;
    for (const json& a : jget(view, "args"))
        if (a.is_string() && std::regex_search(a.get<std::string>(), file)) names.push_back(a.get<std::string>());
    names.push_back(truthy(jget(view, "model_name")) ? py_str(view["model_name"]) : "");
    for (const std::string& n : names) {
        std::smatch m;
        if (std::regex_search(n, m, quant)) return upper(m[1].str());
    }
    return "";
}

std::vector<int> gpu_selection(const json& cfg) {
    // the cards by number alone: "gpu"'s earlier form, else the args' --gpu numbers (cards by name: none here, the
    // server's own names then pick them)
    std::vector<int> sel;
    try {
        sel = gpu_list(cfg);
    } catch (const std::exception&) {
    }
    const json& args = jget(cfg, "args");
    for (size_t i = 0; sel.empty() && args.is_array() && i + 1 < args.size(); ++i)
        if (py_str(args[i]) == "--gpu") {
            const std::string s(strip(py_str(args[i + 1])));
            if (!s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c); }))
                sel.push_back(std::atoi(s.c_str()));
        }
    return sel;
}

std::string cuda_of_header(const std::string& text) {
    static const std::regex cuda(R"(CUDA (?:UMD )?Version:\s*([\d.]+))");
    std::smatch m;
    return std::regex_search(text, m, cuda) ? m[1].str() : "";
}

LocalGpus local_gpus() {
    LocalGpus out;
    for (const GpuInfo& g : nvidia_gpus()) out.gpus.push_back({g.name, g.memory_mib, g.driver});
    if (!out.gpus.empty()) out.cuda = cuda_of_header(output_of({"nvidia-smi"}));
    return out;
}

json collect_machine(const Target& t, const json& cfg, const LocalGpus& local) {
    const json status = object_at(t, "/v1/status");
    const json metrics = object_at(t, "/metrics", 30);
    const json props = object_at(t, "/props");
    const json models = object_at(t, "/v1/models");
    const json hw = jget(metrics, "hardware_static").is_object() ? metrics["hardware_static"] : json::object();
    const json eng = jget(metrics, "engine").is_object() ? metrics["engine"] : json::object();
    const json view = cfg.is_object() ? config_view(cfg) : json::object();
    // the cards the engine uses: the config's, else those the server names (" + " between them: the main GPU, then
    // those of --second-gpu), each matched to one of nvidia-smi's; not every card the PC has
    std::vector<std::string> served;
    for (const std::string& n : split(jstr(hw, "gpu_name"), " + "))
        if (!n.empty() && n != "?") served.push_back(n);
    std::vector<LocalGpus::Gpu> used = local.gpus;
    if (const std::vector<int> sel = gpu_selection(cfg); !sel.empty()) {
        std::vector<LocalGpus::Gpu> picked;
        for (const int i : sel)
            if (i >= 0 && (size_t) i < local.gpus.size()) picked.push_back(local.gpus[(size_t) i]);
        if (!picked.empty()) used = picked;
    } else if (used.size() > 1 && !jstr(hw, "gpu_name").empty()) {
        std::vector<LocalGpus::Gpu> pool = used, named;
        for (const std::string& n : served) {
            const auto it = std::find_if(pool.begin(), pool.end(), [&](const LocalGpus::Gpu& g) { return g.name == n; });
            if (it != pool.end()) {
                named.push_back(*it);
                pool.erase(it);
            }
        }
        used = named.empty() ? std::vector<LocalGpus::Gpu>{used.front()} : named;
    }
    json gpu_names = json::array();
    for (const auto& g : used) gpu_names.push_back(g.name);
    if (gpu_names.empty() && !jstr(hw, "gpu_name").empty()) gpu_names.push_back(jstr(hw, "gpu_name"));
    json vram = jget(jget(jget(status, "machine"), "gpu"), "total_mib");
    if (!truthy(vram)) vram = nullptr;
    json cuda = local.cuda.empty() ? json(nullptr) : json(local.cuda);
    bool all_local = true;
    for (const std::string& n : served)
        all_local = all_local && std::any_of(local.gpus.begin(), local.gpus.end(), [&](const LocalGpus::Gpu& g) { return g.name == n; });
    if (!served.empty() && !all_local) {   // a card nvidia-smi does not list: the server's names are the right ones
        gpu_names = json::array();
        for (const std::string& n : served) gpu_names.push_back(n);
        used.clear();
        cuda = nullptr;
    } else if (!used.empty() && used.front().vram_mib > 0) {
        int64_t sum = 0;
        for (const auto& g : used) sum += g.vram_mib;
        vram = sum;
    }
    json n_ctx = nullptr;
    for (const json& d : jget(models, "data"))
        if (truthy(jget(jget(d, "meta"), "n_ctx"))) n_ctx = d["meta"]["n_ctx"];
    if (!truthy(n_ctx)) n_ctx = jget(jget(props, "default_generation_settings"), "n_ctx");
    json engine_version = jget(eng, "version");
    if (!truthy(engine_version)) engine_version = jget(status, "engine");
    if (!truthy(engine_version)) {
        const std::string b = replace_all(jstr(props, "build_info"), "Lamina ", "");
        engine_version = b.empty() ? json(nullptr) : json(b);
    }
    json model = jget(view, "model_name");
    if (!truthy(model)) model = jget(status, "model");
    if (!truthy(model)) model = jget(props, "model_alias");
    const std::string quant = view.empty() ? "" : detect_quant(view);
    json engine_info = json::object();
    for (auto it = eng.begin(); it != eng.end(); ++it)
        if (it.key() != "model_path" && (it.value().is_primitive() || it.value().is_object()) && !it.value().is_null())
            engine_info[it.key()] = it.value();
    const json threads = truthy(jget(hw, "threads")) ? hw["threads"] : json((int64_t) std::thread::hardware_concurrency());
    return {{"gpu", gpu_names},
            {"gpu_vram_mib", vram},
            {"gpu_driver", used.empty() ? json(nullptr) : json(used.front().driver)},
            {"cuda", cuda},
            {"cpu", jget(hw, "cpu_name")},
            {"cpu_cores", jget(hw, "cores")},
            {"cpu_threads", threads},
            {"ram_gib", ram_gib()},
            {"os", os_text()},
            {"lamina_version", LAMINA_VERSION},
            {"lamina_commit", git_commit()},
            {"engine_version", engine_version},
            {"model", model},
            {"quant", quant.empty() ? json(nullptr) : json(quant)},
            {"context", truthy(n_ctx) ? n_ctx : jget(eng, "max_context")},
            {"parallel_slots", jget(props, "total_slots")},
            {"config", view},
            {"launch", view.empty() ? json(nullptr) : json(launch_line(view))},
            {"engine_info", engine_info},
            {"vram_after", jget(status, "vram").is_object() ? status["vram"] : json::object()}};
}

std::string slug(const std::string& text, size_t limit) {
    std::string s;
    bool dash = false;
    for (const char c0 : lower(text)) {
        const unsigned char c = (unsigned char) c0;
        if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) {
            if (dash && !s.empty()) s += '-';
            dash = false;
            s += (char) c;
        } else {
            dash = true;
        }
    }
    s = s.substr(0, std::min(limit, s.size()));
    s = std::string(strip_chars(s, "-"));
    return s.empty() ? "unknown" : s;
}

std::string gpu_slug(const json& machine) {
    const json& names = jget(machine, "gpu");
    if (!names.is_array() || names.empty()) return "cpu";
    std::vector<std::pair<std::string, int>> groups;   // the cards in their order, identical ones counted
    for (const json& n : names) {
        const std::string name = py_str(n);
        const auto it = std::find_if(groups.begin(), groups.end(), [&](const auto& g) { return g.first == name; });
        if (it != groups.end()) ++it->second;
        else groups.emplace_back(name, 1);
    }
    static const std::regex vendor(R"(\b(nvidia|geforce|amd|radeon|intel|graphics|corporation)\b)", std::regex::icase);
    std::vector<std::string> parts;
    for (const auto& [name, count] : groups) {
        const std::string n = std::regex_replace(name, vendor, " ");
        parts.push_back(count == 1 ? slug(n, 30) : std::to_string(count) + "x-" + slug(n, 30));
    }
    const std::string s = join(parts, "-plus-");
    return std::string(strip_chars(s.substr(0, std::min<size_t>(80, s.size())), "-"));
}

std::string results_dir_name(const json& machine, const std::string& date) {
    return date + "-community-" + gpu_slug(machine) + "-" + slug(model_label(machine), 40);
}

std::string render_readme(const json& res) {
    const json& mc = res["machine"];
    const json& out = res["results"];
    const json& p = res["params"];
    std::vector<std::string> names;
    for (const json& g : jget(mc, "gpu")) names.push_back(py_str(g));
    const std::string gpu = names.empty() ? "CPU only" : join(names, " + ");
    const std::string vram = truthy(jget(mc, "gpu_vram_mib")) ? fixed(num(mc["gpu_vram_mib"]) / 1024, 0) + " GiB VRAM"
                                                               : "VRAM not read";
    std::string model = truthy(jget(mc, "model")) ? model_label(mc) : jstr(mc, "quant");
    if (model.empty()) model = "model not read";
    auto or_ = [](const json& v, const std::string& dflt) { return truthy(v) ? py_str(v) : dflt; };
    const int64_t runs = inum(jget(p, "runs"));
    const bool think = jget(p, "think") == true;
    std::vector<std::string> L = {
        "# Community benchmark on " + gpu, "",
        "Measured on " + jstr(res, "date") + " with `lamina bench` (suite " + std::to_string(kSchema) + ", " +
            std::to_string(runs) + " runs per test, medians with ranges). Speeds are the engine's own timings; first-token "
            "times are measured by the client over HTTP. Speed only: no correctness check was run.",
        "",
        "## Hardware and software", "",
        "- GPU and VRAM: " + gpu + ", " + vram,
        "- CPU: " + or_(jget(mc, "cpu"), "not read") + " (" + or_(jget(mc, "cpu_cores"), "?") + " cores, " +
            or_(jget(mc, "cpu_threads"), "?") + " threads); installed RAM: " + or_(jget(mc, "ram_gib"), "?") +
            " GiB; storage: not measured",
        "- OS: " + py_str(jget(mc, "os")) + "; GPU driver: " + or_(jget(mc, "gpu_driver"), "not read") + "; CUDA: " +
            or_(jget(mc, "cuda"), "not read"),
        "- Lamina: project " + or_(jget(mc, "lamina_version"), "?") + ", commit " +
            or_(jget(mc, "lamina_commit"), "not a git checkout") + "; engine " + or_(jget(mc, "engine_version"), "not read"),
        "- Background workloads and power limits: not recorded", "",
        "## Model and configuration", "",
        "- Model and quantization: " + model,
        "- Context: " + or_(jget(mc, "context"), "not read") + " tokens; parallel slots: " + or_(jget(mc, "parallel_slots"), "1"),
        std::string("- Settings: the install's own configuration, unchanged; thinking ") + (think ? "on" : "off") +
            " for the suite; temperature 0",
        ""};
    if (truthy(jget(mc, "launch"))) L.insert(L.end(), {"```text", jstr(mc, "launch"), "```", ""});
    const json& env = jget(jget(mc, "config"), "env");
    if (truthy(env)) {
        std::vector<std::string> kv;
        for (auto it = env.begin(); it != env.end(); ++it) kv.push_back("`" + it.key() + "=" + py_str(it.value()) + "`");
        L.insert(L.end(), {"Environment of the config: " + join(kv, ", "), ""});
    }
    const bool long32 = jget(jget(out, "prompts"), "32K").is_object();
    L.insert(L.end(), {
        "## Method", "",
        "Run by `" + jstr(p, "command") + "` against the HTTP API of a Lamina server (" +
            (jget(p, "started_server") == true ? "started by the benchmark with the configuration above" : "already running") +
            "); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with "
            "a unique tag, so the conversation cache cannot help the fresh-prompt tests.",
        "",
        "- short chat: three short questions, up to " + std::to_string(kChatMaxTokens) + " new tokens, " +
            std::to_string(runs) + " runs",
        std::string("- 4K / 16K") + (long32 ? " / 32K" : "") + " prompts: generated word lists sized by a calibration "
            "request, " + std::to_string(kLongMaxTokens) + " new tokens, " + std::to_string(runs) + " fresh prompts each",
        "- agent session: system prompt and four tools, then " + std::to_string(kAgentTurns) + " turns that each add about " +
            std::to_string(kAgentWords) + " words of a project file and continue from the model's own earlier answers (the "
            "way agent clients resend a conversation), " + std::to_string(kAgentMaxTokens) + " new tokens per turn, " +
            std::to_string(runs) + " sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings "
            "(`prompt_n`)",
        "- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, "
        "latency over all requests",
        "",
        "## Results", ""});
    const std::vector<std::string> table = results_table(out);
    L.insert(L.end(), table.begin(), table.end());
    L.insert(L.end(), {"", "Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token."});
    const json& ag = jget(out, "agent");
    if (truthy(ag) && truthy(jget(ag, "sessions"))) {
        L.insert(L.end(), {"", "### Agent session (conversation cache)", "",
                           "Cache hit over the follow-up turns: " + fmt(jget(ag, "hit_pct"), 1, " %") +
                               "; earlier-turn tokens read again per session: " + fmt(jget(ag, "reread_tokens"), 0) +
                               " (earlier turns that the cache did not hold; the " + fmt(jget(ag, "new_tokens"), 0) +
                               " new tokens of the files are read once); first turn (nothing cached) first token: " +
                               fmt(jget(ag, "first_turn_ttft_s"), 2, " s") + ".",
                           "",
                           "| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |",
                           "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"});
        for (const json& d : ag["sessions"][0])
            if (d["ok"] == true)
                L.push_back("| " + py_str(d["turn"]) + " | " + py_str(d["prompt_tokens"]) + " | " + py_str(d["reused_tokens"]) +
                            " | " + py_str(d["read_tokens"]) + " | " + or_(d["prompt_tps"], "-") + " | " + py_str(d["ttft_s"]) +
                            " | " + or_(d["decode_tps"], "-") + " |");
    }
    const json& cl = jget(out, "clients");
    if (truthy(cl))
        L.insert(L.end(), {"", "### Concurrency: " + py_str(cl["clients"]) + " clients", "",
                           "Total " + fmt(jget(cl, "total_tps"), 1, " tok/s") + "; first token p50 " + py_str(jget(cl, "ttft_p50_s")) +
                               " s, p95 " + py_str(jget(cl, "ttft_p95_s")) + " s; request latency p50 " +
                               py_str(jget(cl, "latency_p50_s")) + " s, p95 " + py_str(jget(cl, "latency_p95_s")) +
                               " s; errors " + py_str(jget(cl, "errors")) + ". Server parallel slots: " +
                               or_(jget(mc, "parallel_slots"), "1") + " (requests beyond that wait their turn)."});
    const json& skipped = jget(out, "skipped");
    const json& failures = jget(out, "failures");
    if (truthy(skipped) || truthy(failures)) {
        L.insert(L.end(), {"", "### Skipped and failed", ""});
        for (auto it = skipped.begin(); skipped.is_object() && it != skipped.end(); ++it)
            L.push_back("- skipped " + it.key() + ": " + py_str(it.value()));
        for (const json& f : failures) L.push_back("- failed " + jstr(f, "where") + ": " + jstr(f, "error"));
    }
    L.insert(L.end(), {"", "Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` "
                           "(when this run started the server).",
                       "", "## Correctness and limitations", "",
                       "Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts "
                       "differ a little between runs. Machine details come from the server, `nvidia-smi` and the OS; "
                       "background load and power limits are not recorded. Total time of the suite: " +
                           fixed(num(jget(res, "elapsed_s")), 0) + " s.",
                       ""});
    return join(L, "\n");
}

std::string compare_table(const json& older, const json& newer) {
    const json a = summary_metrics(jget(older, "results")), b = summary_metrics(jget(newer, "results"));
    std::vector<std::string> keys;
    for (const json* m : {&a, &b})
        for (auto it = m->begin(); it != m->end(); ++it)
            if (std::find(keys.begin(), keys.end(), it.key()) == keys.end()) keys.push_back(it.key());
    char buf[256];
    std::snprintf(buf, sizeof buf, "%-32s %12s %12s %9s", "", "old", "new", "change");
    std::vector<std::string> rows = {buf};
    for (const std::string& k : keys) {
        const json& x = jget(jget(a, k), "median");
        const json& y = jget(jget(b, k), "median");
        if (!x.is_number() && !y.is_number()) continue;
        std::string ch;
        if (x.is_number() && num(x) != 0 && y.is_number()) {
            std::snprintf(buf, sizeof buf, "%+.1f%%", (num(y) - num(x)) / num(x) * 100);
            ch = buf;
        }
        std::snprintf(buf, sizeof buf, "%-32s %12s %12s %9s", k.c_str(), x.is_number() ? fixed(num(x), 2).c_str() : "-",
                      y.is_number() ? fixed(num(y), 2).c_str() : "-", ch.c_str());
        rows.push_back(buf);
    }
    const json& ho = jget(older, "machine");
    const json& hn = jget(newer, "machine");
    std::vector<std::string> notes;
    for (const auto& [name, field] : std::vector<std::pair<std::string, std::string>>{
             {"GPU", "gpu"}, {"engine", "engine_version"}, {"model", "model"}, {"quant", "quant"}, {"context", "context"}})
        if (jget(ho, field) != jget(hn, field))
            notes.push_back("  " + name + ": " + py_str(jget(ho, field)) + " -> " + py_str(jget(hn, field)));
    if (!notes.empty()) {
        rows.push_back("");
        rows.push_back("differences:");
        rows.insert(rows.end(), notes.begin(), notes.end());
    }
    return join(rows, "\n");
}

json load_results(const std::string& path) {
    fs::path p = u8_path(path);
    std::error_code ec;
    if (fs::is_directory(p, ec)) p /= "results.json";
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + u8_str(p));
    json j;
    std::string err;
    if (!py_loads(std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>()), j, &err) ||
        !j.is_object())
        throw std::runtime_error(u8_str(p) + " is not a benchmark's results.json" + (err.empty() ? "" : ": " + err));
    return j;
}

json without_profile_save(const json& cfg, const std::string& scratch) {
    json out = cfg;
    auto copy_from = [&](const std::string& src) {
        if (src.empty()) return;
        fs::path from = u8_path(src);
        if (!from.is_absolute() && !jstr(cfg, "cwd").empty()) from = u8_path(jstr(cfg, "cwd")) / from;   // the engine's cwd
        std::error_code ec;
        if (fs::is_regular_file(from, ec)) fs::copy_file(from, u8_path(scratch), fs::copy_options::overwrite_existing, ec);
    };
    json& args = out["args"];
    if (!args.is_array()) {
        out.erase("args");
    } else {
        for (size_t i = 0; i < args.size(); ++i) {
            const std::string a = py_str(args[i]);
            if (a == "--expert-profile-save" && i + 1 < args.size()) {
                copy_from(py_str(args[i + 1]));
                args[i + 1] = scratch;
                ++i;
            } else if (starts_with(a, "--expert-profile-save=")) {
                copy_from(a.substr(22));
                args[i] = "--expert-profile-save=" + scratch;
            }
        }
    }
    if (const json& save = jget(cfg, "expert_profile_save"); save.is_string() && !strip(save.get<std::string>()).empty()) {
        copy_from(save.get<std::string>());
        out["expert_profile_save"] = scratch;
    }
    return out;
}

json run_suite(const Target& t, int64_t ctx, const Options& o, const std::function<void(const std::string&)>& log,
               const std::function<bool()>& stopped, bool* interrupted) {
    Suite s(t, ctx, o, log, stopped);
    json out = s.run();
    if (interrupted != nullptr) *interrupted = s.interrupted;
    return out;
}

int run(const Target& t, const json& cfg, const Options& o, const std::string& log_file, uint64_t log_from,
        const std::function<bool()>& stopped) {
    const bool started = cfg.is_object();
    if (!is_lamina(t)) {
        say_err("No Lamina server answers at " + t.base + ".");
        return 2;
    }
    if (!started) say("Using the Lamina server already running at " + t.base + ".");
    const Redactor red = redactor({}, t.key);
    const auto t0 = std::chrono::steady_clock::now();
    const LocalGpus local = local_gpus();
    const json pre = collect_machine(t, cfg, local);
    Options opts = o;
    opts.runs = std::max(1, o.runs);
    const int64_t slots = std::max<int64_t>(1, inum(jget(pre, "parallel_slots")));
    if (opts.clients > slots)
        say("Note: the server runs " + std::to_string(slots) + " request(s) at a time (--parallel), so " +
            std::to_string(opts.clients) + " clients will partly wait their turn.");
    say("Running the suite (about 3-10 minutes) ...");
    bool interrupted = false;
    const json out = run_suite(t, inum(jget(pre, "context")), opts, [](const std::string& line) { say(line); }, stopped,
                               &interrupted);
    if (interrupted) {
        say("\nThe benchmark was stopped: no report.");
        return 130;
    }
    json machine = collect_machine(t, cfg, local);   // again: VRAM and the cache after the work
    if (!truthy(jget(machine, "gpu"))) machine["gpu"] = pre["gpu"];
    const json res = {{"schema", kSchema},
                      {"tool", "lamina bench"},
                      {"date", today()},
                      {"params", {{"runs", opts.runs}, {"think", opts.think}, {"clients", opts.clients},
                                  {"long", opts.long_prompt}, {"started_server", started}, {"command", opts.command}}},
                      {"elapsed_s", seconds_since(t0)},
                      {"machine", machine},
                      {"results", out}};
    fs::path dir;
    try {
        const fs::path parent = o.out.empty() ? default_parent() : u8_path(o.out);
        dir = unique_dir(parent, results_dir_name(machine, jstr(res, "date")));
        fs::create_directories(dir);
        write_file(dir / "results.json", py_dumps(redact(res, red), false, false, 1) + "\n");
        write_file(dir / "README.md", red(render_readme(res)));
        if (const std::vector<std::string> lines = log_excerpt(log_file, log_from); !lines.empty())
            write_file(dir / "engine-log-excerpt.txt", red(join(lines, "\n")) + "\n");
    } catch (const std::exception& e) {
        say_err(std::string("[lamina] the report could not be written: ") + e.what());
        return 1;
    }
    say("\nResults (median, range over the runs):");
    say(red(short_table(res)));
    const json& failures = out["failures"];
    if (!failures.empty() || !out["skipped"].empty())
        say("  (" + std::to_string(failures.size()) + " failed request(s), " + std::to_string(out["skipped"].size()) +
            " skipped test(s): see the report)");
    if (!o.compare.empty()) {
        try {
            const json older = load_results(o.compare);
            say("\nCompared with " + o.compare + ":");
            say(red(compare_table(older, res)));
        } catch (const std::exception& e) {
            say_err(std::string("[lamina] --compare: ") + e.what());
        }
    }
    say("\nReport: " + u8_str(dir / "README.md") + "\nShare it: paste README.md into an issue at " + kShare +
        ". Home folders, user and host names and IP addresses are already replaced; read it once before you share it.");
    return any_ok(out) || failures.empty() ? 0 : 1;
}

int compare_only(const std::string& older, const std::string& newer) {
    try {
        say(compare_table(load_results(older), load_results(newer)));
        return 0;
    } catch (const std::exception& e) {
        say_err(std::string("[lamina] --compare: ") + e.what());
        return 2;
    }
}

}  // namespace lamina::bench
