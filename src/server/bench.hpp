// src/server/bench.hpp - lamina --bench: a model measured the way it is used, through the server's own HTTP API,
// and a report to share without personal details.
//
// The suite, each test three runs (medians and ranges): three short questions (decode tok/s, time to first token),
// fresh prompts of ~4K and ~16K tokens (prompt tok/s), a coding agent's session of six turns with four tools (what the
// conversation cache reused), and on request N requests at once and a ~32K prompt.  Speeds are the engine's own
// timings (the answer's `timings`), first-token times and latencies the client's clock.  Thinking is off and the
// temperature 0; every prompt starts with a tag of its own, so no cache helps a fresh one.  The prompts are fixed, word
// for word (suite 1, which the earlier Python benchmark sent too), so reports compare.
#pragma once

#include "json.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace lamina::bench {

inline constexpr int kSchema = 1;   ///< results.json's; the suite it names

struct Options {
    int runs = 3;           ///< runs per test
    int clients = 0;        ///< N requests at once (0: that test is left out)
    bool long_prompt = false;   ///< also ~32K tokens
    bool think = false;     ///< thinking on
    std::string out;        ///< the reports' folder ("": bench-results beside the program)
    std::string compare;    ///< an earlier report (its folder or results.json) to print beside this run's
    std::string command;    ///< the command line as the report names it
};

/// the server measured
struct Target {
    std::string base;   ///< http://127.0.0.1:8081
    std::string key;    ///< its API key ("": none)
};

/// this PC's NVIDIA cards as nvidia-smi numbers them, and the CUDA version its table's header names
struct LocalGpus {
    struct Gpu {
        std::string name;
        int64_t vram_mib = 0;
        std::string driver;
    };
    std::vector<Gpu> gpus;
    std::string cuda;
};
LocalGpus local_gpus();
/// the CUDA version of nvidia-smi's header: "CUDA Version: 12.9", driver 621's "CUDA UMD Version: 13.5"
std::string cuda_of_header(const std::string& text);

/// whether a Lamina server answers /health at `base` (http://host:port)
bool lamina_answers(const std::string& base);

/// tools/ab_engine.py's word list of `words` words, as Python's random.Random(7) picks them
std::string long_prompt(int words);
/// {median, min, max, n} of the numbers among `values`, rounded to 3 places; null when there are none
json stat(const std::vector<json>& values);
/// the p quantile of `v`, interpolated between neighbours
double pct(std::vector<double> v, double p);

/// Replaces what a report must not show: the program's folder, the home and temporary folders, the user's and the
/// PC's names, private IP addresses, `key` and bearer tokens, and `extra` words.
using Redactor = std::function<std::string(const std::string&)>;
Redactor redactor(const std::vector<std::string>& extra = {}, const std::string& key = "");
/// every string of `v` (keys too) through `red`
json redact(const json& v, const Redactor& red);

/// the config's settings worth reporting: paths cut to their file names, secrets, the log, cwd and env left out (but
/// env's LAMINA_ and CUDA_ variables)
json config_view(const json& cfg);
/// the quantization a model file of the view's args (or its model_name) names: "IQ3_S", "UD-Q4_K_XL"; "" when none
std::string detect_quant(const json& view);
/// the cards (nvidia-smi's numbers) the config gives the engine: its "gpu", else a --gpu in its args
std::vector<int> gpu_selection(const json& cfg);
/// The machine and the server as the report names them: the server's /v1/status, /metrics, /props and /v1/models,
/// `local`, and `cfg`, the run config of a server this run started (null: one it did not).  Only the cards the engine
/// uses: the config's, else those the server names.
json collect_machine(const Target& t, const json& cfg, const LocalGpus& local);

std::string slug(const std::string& text, size_t limit = 40);
std::string gpu_slug(const json& machine);
std::string results_dir_name(const json& machine, const std::string& date);
/// the report, README.md (before the redaction)
std::string render_readme(const json& res);
/// two reports side by side: each measure's medians and the change, then what differs (cards, engine, model)
std::string compare_table(const json& older, const json& newer);
/// a report's results.json: its folder or the file
json load_results(const std::string& path);

/// The run config a benchmark's server starts from: what the engine saves of its expert profile goes into `scratch`
/// (a copy of the user's profile when there is one, which the run then starts from), never over the user's file.
json without_profile_save(const json& cfg, const std::string& scratch);

/// The suite against `t` (`ctx`: the server's context, 0 unknown): the results by test.  `log` takes the progress
/// lines; `stopped` is asked between requests and while one streams (true: the run ends, `*interrupted` set).
json run_suite(const Target& t, int64_t ctx, const Options& o, const std::function<void(const std::string&)>& log,
               const std::function<bool()>& stopped, bool* interrupted = nullptr);

/// The whole benchmark: the machine before and after, the suite, the report in a folder of its own below o.out (or
/// bench-results beside the program), its summary printed.  `cfg`: the config of the server this run started (null:
/// --bench-url's server); `log_file` and `log_from`: the engine's log and where this run's part begins, for the
/// report's excerpt.  The exit code: 0; 1 when no request came back; 2 no Lamina server at t.base; 130 stopped.
int run(const Target& t, const json& cfg, const Options& o, const std::string& log_file, uint64_t log_from,
        const std::function<bool()>& stopped);
/// --bench-compare OLD NEW: the two reports' table; 2 when one cannot be read
int compare_only(const std::string& older, const std::string& newer);

}  // namespace lamina::bench
