// src/server/service.hpp - the server's core between the HTTP routes and the engine (serve/server.py's Service).
//
// One request at a time takes the engine's turn (the FIFO); with the engine's batch slots ("parallel") several run
// at once. run() turns a prompt into events: the parsed output (reasoning, content, tool calls), heartbeats while
// the engine is quiet, and a last "done" with the request's figures. It keeps the server's record of what runs and
// ran (GET /metrics, /v1/status, Prometheus), the guards (one token repeated, a thinking budget and the answer's room,
// a reasoning loop, non-finite logits) and the prompt reuse across turns (PromptEncoder).
#pragma once

#include "engine.hpp"
#include "frontend.hpp"
#include "json.hpp"
#include "tokenizer.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace lamina {

class Vision;
struct VideoJob;
namespace images {
class ImageGen;
}
class McpHub;
class Telemetry;
class KeepAwake;

/// the request's events: kind "event" (ev set), "ping" (a keep-alive), "mcp" (value), "done" (value); false: the
/// client is gone
using Sink = std::function<bool(const std::string& kind, const json& value, const Event* ev)>;

/// (ids, thinking, max_new) of a prepared request
struct Prepared {
    std::vector<int32_t> ids;
    bool thinking = true;
    int64_t max_new = 0;
};

class Service {
public:
    Service(std::shared_ptr<Engine> engine, const Tokenizer& tok, std::shared_ptr<ChatTemplate> tpl, std::string model,
            std::shared_ptr<Vision> vision = nullptr, json sampling_defaults = json::object(),
            bool fit_max_tokens = true);
    ~Service();

    // ---- configuration (main sets these; the Settings view's keys may change while requests run: apply_setting)
    bool answer_room = true, reasoning_close_retry = false, literal_markers = true, codex_compaction_cache = false,
         codex_thread_titles = false;
    int64_t short_opening_tokens = 0;
    bool effort_end = false, offline = false;
    std::atomic<bool> api_monitor{false}, anthropic_think_unasked{true}, fit_max_tokens{true};
    bool tool_call_recovery = false, before_load_strict = false, before_load_ignore_failure = false;
    std::string reasoning_loop_recovery;   ///< "", "stop" or "recover"
    int64_t repeat_stop_tokens = 256;
    std::string model_arch;   ///< the family ("qwen4exp", "qwen35moe", "qwen35"): the model card's sampling
    std::string shared_path, config_path, config_effort;
    /// #564: the run's settings as the Settings view shows them: the merged config as it started (started_config),
    /// as this session has it now (live_config: POST /config's changes), the preset under the file, the keys the
    /// command line set; under config_mu
    json live_config = json::object(), started_config = json::object();
    json preset_config = json::object();   ///< the preset the config file's settings went over (empty: none)
    std::set<std::string> command_line_keys;
    std::vector<std::string> cors_origins, trusted_origins, allowed_hosts;
    json api_key = "";                              ///< a key ("k1,k2": several) or a list of keys; "" none
    int64_t requested_slots = 0;                    ///< the slots the config asked for (0: what the engine has)
    std::vector<int> gpu_indices;                   ///< every card the engine uses (the Monitor's)
    std::mutex config_mu;                           ///< POST /config: one change of the run config at a time
    std::set<std::string> hosted_tools_noted;       ///< /v1/responses' hosted tools said once each
    static constexpr double kUnloadWaitS = 300.0;   ///< POST /unload?wait=1 waits this long for the running turn
    std::set<std::string> host_names;
    std::atomic<double> idle_unload_s{0};
    double before_load_timeout_s = 120, vram_free_wait_s = 15;
    int64_t min_free_vram_mib = 0;
    json before_load;   ///< a command: a string (shell) or a list
    std::optional<int64_t> max_queue;
    std::optional<double> queue_timeout_s;
    int gpu_index = 0;
    std::shared_ptr<images::ImageGen> images;
    std::shared_ptr<McpHub> mcp;
    std::shared_ptr<Telemetry> telemetry;
    std::shared_ptr<KeepAwake> keep_awake;

    // ---- state
    std::shared_ptr<Engine> engine;
    const Tokenizer& tok;
    std::shared_ptr<ChatTemplate> tpl;
    std::string model;
    std::shared_ptr<Vision> vision;
    LiteralTags literals;
    std::set<int32_t> stop_ids;
    std::unique_ptr<PromptEncoder> prompts;
    std::mutex status_mu;
    std::condition_variable admission_changed;
    json status = {{"busy", false}, {"queued", 0}};
    std::deque<std::pair<double, int64_t>> rate;   ///< (time, generated) for the live tok/s window
    std::deque<json> history;                      ///< the last finished requests, newest last
    json totals;
    json last_timings;
    std::optional<double> last_request_at;
    double started_at;
    std::string instance_id;
    std::map<std::string, json> terminal_requests;   ///< queued or running requests, by id
    int64_t admission_reserved = 0, admission_rejected = 0, queue_timeouts = 0, client_disconnect_cancels = 0;
    bool draining = false;
    std::set<Cancel*> active_cancels;
    std::deque<std::shared_ptr<json>> api_requests;   ///< the request monitor (/api-monitor), when on
    std::tuple<int64_t, double, int64_t, double> scrape_bucket{0, 0, 0, 0};
    /// a finished request's latencies for the Prometheus histograms (time to first token, tokens, decode s, total s)
    std::function<void(std::optional<double>, int64_t, double, double)> observe_latencies;

    bool loaded();
    std::optional<int64_t> admission_limit();
    /// one bounded inference call, before response headers: true, false (the queue is full), nullopt (draining)
    std::optional<bool> reserve_inference();
    void release_inference();
    void note_client_disconnect(Cancel& cancel);
    /// shutdown with --shutdown-timeout: no new inference; -> the calls still to drain
    int64_t begin_drain();
    /// waits for the admitted calls; -> those still in flight
    int64_t wait_for_drain(double timeout_s);
    int64_t cancel_owned_requests();
    void set_aliases(const json& value);
    std::vector<std::string> aliases_now() const;
    std::vector<std::string> model_names() const;
    std::string model_for(const json& req) const;
    int64_t reported_ctx();
    std::optional<int64_t> reasoning_budget(const json& req) const;
    /// the config's default thinking budget (0: none)
    void set_reasoning_budget_tokens(json v);
    /// the config's "sampling" block, a copy
    json sampling_defaults_now() const;
    // ---- LoRA adapters (the engine's --lora, llama.cpp's server API)
    /// GET /lora-adapters: [{id, path, scale, task_name, prompt_prefix}] with the scales a request that names none
    /// takes (empty until the engine has started with adapters)
    json lora_adapters();
    /// POST /lora-adapters: [{id, scale}, ...] sets those scales, every adapter left out at 0 (llama.cpp's);
    /// RequestError for a bad body
    void set_lora(const json& body);
    /// a request's "lora" field checked (RequestError: a 400 before anything is sent)
    void check_lora(const json& req);
    /// the scales of every adapter for a request: its "lora" field ([{id, scale}], the others 0), else its model name's
    /// in lora_models, else the server's; empty without adapters
    std::vector<double> lora_for(const json& req);
    /// the config's "lora_models": model names a request picks adapters by, {"NAME": [{"id": N, "scale": X}], ...}
    void set_lora_models(const json& value);
    json lora_models_now() const;
    /// a Settings view key (runconfig::kEditable, one that applies while the server runs) set to `value` (null: its
    /// default) for the requests from now on
    void apply_setting(const std::string& key, const json& value);
    std::optional<int64_t> free_vram_mib();
    void ensure_loaded();
    void load();
    std::string unload(std::optional<double> idle_for = std::nullopt);
    std::string unload_waiting(double wait_s);
    void start_idle_unload();
    /// the Chat settings shared with apps, a copy: POST /settings replaces them while requests read them
    json shared() const;
    /// `save`: into shared_path (the start reads them from there)
    json set_shared(const json& defaults, bool save = true);
    std::string effort_default() const;
    json with_shared(const json& req, const std::string& api) const;
    double tok_s();
    double tok_s_mean();
    json prefill_tok_s_mean();
    /// a request monitor record (null when the monitor is off); this thread's `trace` until end_request
    std::shared_ptr<json> begin_request(const std::string& path, const json& req);
    void end_request();
    json request_records(const std::string& request_id = "");
    json metrics(bool all_requests = false);
    std::string prometheus();
    json v1_status();
    std::string render_prompt(const json& messages, const json& tools, const json& kwargs);
    bool short_opening(const json& messages) const;
    /// the rendered prompt's ids; `videos`: the text each video's template marker becomes (video_text), in order
    std::vector<int32_t> encode_prompt(const json& messages, const json& tools, const json& kwargs, bool store = true,
                                       const std::vector<std::string>* videos = nullptr);
    void forget_prompt(const std::vector<int32_t>& ids);
    Prepared prepare(json& messages, const json& tools, const json& kwargs, std::optional<int64_t> max_new,
                     const std::optional<std::string>& force = std::nullopt, json* req = nullptr,
                     Cancel* cancel = nullptr);
    json resolve_prefix(const json& spec, const json& messages, const json& tools, const json& kwargs,
                        const std::vector<int32_t>& ids, bool has_images);
    json choose(const json& req, Cancel& cancel);
    void drop_embeddings();
    void run(const std::vector<int32_t>& ids, bool thinking, const json& tools, int64_t max_new, json sampling,
             Cancel& cancel, std::optional<std::string> force, const Sink& sink);
    std::string say_died(const std::exception& e);
    /// the engine's single turn (an image encode or job takes it as a request does); a client that leaves while
    /// waiting for it gives its place up (#1999: ClientGone)
    std::unique_lock<std::timed_mutex> turn(Cancel* cancel = nullptr);
    /// the turn if it is free now (owns_lock() false: a request has it)
    std::unique_lock<std::timed_mutex> try_turn() {
        return std::unique_lock<std::timed_mutex>(fifo_, std::try_to_lock);
    }
    /// a request is running in the engine's batch slots
    bool slots_busy();

    /// the current request's image embeddings (GENI): set by prepare, taken by run (per thread)
    struct Embeddings {
        std::string path;
        std::vector<std::string> keys;
    };
    static thread_local Embeddings embeddings;
    /// the request monitor's record of the request on this thread
    static thread_local json* trace;

private:
    void run_inner(const std::vector<int32_t>& ids, bool thinking, const json& tools, int64_t max_new, json sampling,
                   Cancel& cancel, std::optional<std::string> force, const Sink& sink);
    void note(int64_t n, const std::vector<Event>& evs, json& st, std::deque<std::pair<double, int64_t>>& rt);
    double progress_line(double last_print, json& st, double every = 1.0);
    json note_unreadable_tool_images(json& messages, Cancel* cancel);
    /// the request's videos planned, in prompt order (a tool's video that cannot be read becomes a note, as its
    /// picture does); RequestError for a user's
    std::vector<std::shared_ptr<VideoJob>> plan_videos(json& messages, Cancel* cancel);
    /// what a video's template marker becomes in the prompt's text: each group's time and one pad between the
    /// group's tags (Qwen's <|vision_start|> <|vision_end|>; Hy4's picture tags inside its video tags), the pad then
    /// a row a cell as a picture's
    std::string video_text(const VideoJob& v) const;
    double rate_of(const json& s, const std::deque<std::pair<double, int64_t>>& rt);

    std::timed_mutex fifo_;
    std::map<const json*, std::pair<json*, std::deque<std::pair<double, int64_t>>*>> live_reqs_;
    std::thread idle_thread_;
    std::atomic<bool> stopping_{false};
    mutable std::mutex shared_mu_;
    json shared_ = json::object();
    mutable std::mutex live_mu_;   ///< the three below: the Settings view changes them while requests read them
    std::vector<std::string> aliases_;
    json lora_models_ = json::object();
    json reasoning_budget_tokens_ = 0;
    json sampling_defaults_ = json::object();
    /// the adapters the engine reported (its ADAPTER lines) and the scales the server applies to them
    std::mutex lora_mu_;
    json lora_list_ = json::array();
    std::vector<double> lora_scales_;
    /// the engine's adapters, the server's scales reset to their start's when the list changed (a restart with others)
    json lora_sync();
};

// ---- helpers (module functions of serve/server.py)
/// the request's stop strings (OpenAI's stop, Anthropic's stop_sequences); RequestError for a bad value
std::vector<std::string> stop_strings(const json& req);
/// cuts the answer at the first stop string; text that could still start one is held back
class StopMatcher {
public:
    explicit StopMatcher(std::vector<std::string> stops) : stops_(std::move(stops)) {}
    std::string push(const std::string& text);
    std::string flush();
    std::optional<std::string> hit;
    const std::string& held() const { return held_; }

private:
    std::vector<std::string> stops_;
    std::string held_;
};
/// a request's sampling field of the wrong JSON type is a RequestError naming it (numbers sent as strings are read)
void check_request_sampling(json& req);
json clean_shared_defaults(const json& d);
/// the config's "sampling" block; a bad value throws std::runtime_error (the server does not start)
json sampling_defaults_from_config(const json& cfg);
json model_card_sampling(const std::string& arch, bool thinking);
std::string default_model_name(const json& cfg, const std::string& arch);
double reasoning_repeat_coverage(const std::string& text);
std::optional<std::vector<int32_t>> focused_recovery_prompt(const Tokenizer& tok, const std::vector<int32_t>& ids,
                                                            const std::vector<int32_t>& generated);
std::optional<std::string> held_departure(const Tokenizer& tok, const Held& held, const std::vector<int32_t>& ids,
                                          int64_t reused);
int64_t prompt_tokens_seen(int64_t prompt_tokens, const json& last);
json request_timings(int64_t prompt_tokens, int64_t generated, const json& last);
json request_stats(const std::vector<json>& segments);
double now_s();    ///< time.time()
double mono_s();   ///< time.perf_counter(): the request records' clock

inline constexpr int64_t kCtxSlack = 8;   // `lamina-engine --serve` rejects prompt + max_new + 8 > context

}   // namespace lamina
