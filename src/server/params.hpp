// src/server/params.hpp - lamina's commands and its settings in one table: every key of the run config and the
// command-line option that sets it.  A value of the command line takes the place of the config file's, and the file's
// that of the preset -m builds from (model_config.hpp).  An option the table does not have is the engine's (the
// config's "args").
#pragma once

#include "config.hpp"
#include "json.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace lamina {

enum class ParamKind {
    Bool,    ///< a switch: --x is true, --no-x false, --x=true|false|1|0|yes|no|on|off
    Int,     ///< a whole number
    Num,     ///< a number (written whole when it is)
    Str,     ///< text, as given
    Path,    ///< a file or folder: on the command line made absolute against the start's folder (a config's relative
             ///< paths start from its "cwd")
    List,    ///< comma-separated text or a JSON array; given again: appended
    Many,    ///< text, one item a time: given again, appended (no splitting at commas)
    Pairs,   ///< NAME=VALUE into an object; given again for more
    Value,   ///< JSON when it reads as a number, a list, an object, true, false or null, else text
    Off,     ///< no value: sets the key to false (--no-mmproj: "vision": false)
};

struct Param {
    std::string_view flag;      ///< "--port"
    std::string_view aliases;   ///< other spellings, space-separated ("-m", "--lazy", "--temp")
    std::string_view key;       ///< "port"; a dot names a section's key: "sampling.temperature"
    ParamKind kind;
    std::string_view arg;       ///< the value's name in --help ("N", "FILE")
    std::string_view help;
};
/// The table, in --help's order; a Param with an empty flag starts a group, its help the group's title.
extern const std::vector<Param> kParams;
/// the Param a flag names (an alias, or --no-<flag> of a switch), and whether it was that --no- form; nullptr when none
const Param* find_param(std::string_view flag, bool* negated = nullptr);

/// lamina's commands, in `lamina help`'s order: each takes the table's options and the engine's (but help and
/// version, which main() answers before a command line is read)
struct Command {
    std::string_view name;   ///< "serve"
    std::string_view what;   ///< its line in `lamina help`
};
extern const std::vector<Command> kCommands;

/// lamina's command line: a command, then its options
struct CommandLine {
    std::string command;                 ///< serve, check, bench or write-config
    json settings = json::object();      ///< the options' values by key, a section's nested ({"sampling": {"top_k": 20}})
    std::vector<EngineOption> engine;    ///< the options the table does not have: the engine's
    std::string config;                  ///< --config
    std::string write_config;            ///< write-config's FILE
    bool check = false;                  ///< the command check
    bool help = false;                   ///< -h / --help after the command: its options
    bool version = false;                ///< -v / --version after the command
    bool verbose = false;                ///< check --verbose: a URL's table with what each file is and why not
    // bench (src/server/bench.hpp): the suite against a server of its own, or a running one (--url)
    bool bench = false;                  ///< the command bench, with a model to load (no --url, no --compare OLD NEW)
    std::string bench_url;               ///< bench --url
    int bench_runs = 3;                  ///< bench --runs
    int bench_clients = 0;               ///< bench --clients
    bool bench_long = false;             ///< bench --long
    bool bench_think = false;            ///< bench --think
    std::string bench_out;               ///< bench --out (made absolute)
    std::vector<std::string> bench_compare;   ///< bench --compare OLD [NEW] (made absolute)
    bool bench_options = false;          ///< any of bench's options but --url and --compare
};
/// argv[1] the command, the words after it its options, into a CommandLine (a word that is no option of the command's
/// and starts with -- is the engine's, with the values after it).  std::invalid_argument for a command that is not
/// one, a value its kind does not take, a missing value, or a word that is no option.
CommandLine parse_command_line(int argc, char** argv);

/// `over` into `base` key by key: an object (a section) keeps the keys `over` does not set, anything else is replaced
void merge_settings(json& base, const json& over);
/// `v` at `key` ("a" or "a.b"); a section that is not an object yet becomes one
void set_setting(json& cfg, std::string_view key, json v);

/// the config's top-level keys no Param names (nor "args", "mcpServers"): typos, or keys this server does not read
std::vector<std::string> unknown_keys(const json& cfg);

/// `lamina help`: the commands
std::string commands_help();
/// `lamina <command> --help`: serve's the table's options, then the engine's options a server takes
/// (lamina/program/options.hpp); another command's its own options, then the ones it shares with serve
std::string command_help(std::string_view command);
/// `lamina <command>`'s usage line (a usage error's first line)
std::string command_usage(std::string_view command);

}  // namespace lamina
