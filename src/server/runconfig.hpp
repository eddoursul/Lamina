// src/server/runconfig.hpp - #564: the web page's Settings view of the run config (serve/runconfig.py's
// port).
//
// A short list of documented keys can be read and changed from Lamina's own page (GET / POST /config): the values
// this session runs with (the command line's over the config file's over the preset's), changed for the session
// (Apply: at once; a start-up setting, which lamina reads when it starts, waits for its next start), and
// written to the config file on request (Save configuration; a session started without a config file keeps its
// changes until it stops). Save writes the keys whose session value is not the one the file gives; for a config
// that names a model and has no "args", a value equal to its preset's takes the key out of the file. Everything else
// in the file - the keys --write-config writes, the network and security keys (host, api_key, cors_origins, trusted_origins,
// allowed_hosts), the MCP servers and before_load (which run programs), sampling keys not listed here - is kept as
// it is: a save touches only the keys it names (as --write-config over an earlier file keeps the user's keys, #629). The
// earlier file is kept as <file>.bak.
//
// A value or a file that is not what it must be throws RequestError (frontend.hpp, Python's ValueError: a 400); a
// file that cannot be read or written throws ConfigFileError (Python's OSError).
#pragma once

#include "frontend.hpp"
#include "json.hpp"

#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lamina::runconfig {

/// the run config cannot be read or written (Python's OSError), in Python's words: "[Errno 2] ...: 'path'"
struct ConfigFileError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/// a key the Settings view lists
struct Editable {
    std::string_view key;
    std::string_view kind;   ///< "bool", "int>=0", "num>=0", "enum", "names", "sampling" or "arg"
    /// "sampling": the server's own rule for the key ("num>=0", "0<x<=1", "0<=x<=1", "1..64"); "arg": its flag
    std::string_view rule;
    std::vector<std::string_view> choices;   ///< "enum"
    std::string_view help;
    /// a change applies at once (Service::apply_setting); false: a start-up setting, read when lamina starts
    bool live = true;
    /// "arg": the main GPU card's setting in the flag's place when "gpu" names the cards (a size, its value in MiB)
    std::string_view card = {};
};
/// the keys of the "sampling" block, the top-level keys, and an engine option kept in "args"
extern const std::vector<Editable> kEditable;

/// the config file a session saves into: its path, its content, the settings a start takes from it (for a config
/// that names a model and has no "args", its preset's under its own) and whether it holds engine options ("args")
struct FileSettings {
    std::string path;
    json file = json::object();
    json preset = json::object();   ///< the preset under the file's settings (empty: none)
    json gives = json::object();
    bool engine_options = false;
};

/// a listed key's value in `cfg` (null: not set, the default applies); std::out_of_range for a key not listed
json value_of(const json& cfg, std::string_view key);
/// the listed key `key`, or nullptr
const Editable* editable(std::string_view key);
/// Save configuration can keep `e` in `file` (null: the session has none): an engine option needs an "args" list
bool savable(const Editable& e, const FileSettings* file);
/// GET /config: the editable keys with their values in this session (`live`), each marked "engine_option" (kept in
/// "args"), "live" (a change applies at once; else at lamina's next start), "savable", "command_line" (the
/// command line set it: its value takes the place of the file's at every start), "pending" (a start-up setting
/// changed this session: against `started`) and "unsaved" (a savable key whose value is not the one the file gives);
/// "file" the file's name, null for a session without one (`file` null)
json view(const json& live, const json& started, const std::vector<std::string>& command_line,
          const FileSettings* file);
/// the config file as Save configuration writes it: the savable keys whose value in `live` is not the one the file
/// gives set to it (the preset's value: the key taken out; none over a preset's value: null, which clears it), the
/// keys that changed in `changed`, the keys it cannot keep in `skipped`
json saved(const json& live, const FileSettings& file, std::vector<std::string>& changed,
           std::vector<std::string>& skipped);
/// the value to store for `key`, or a RequestError naming what is expected; null removes the key (its default)
json check(const std::string& key, const json& v, const json& cfg);
/// the config with `changes` ({key: value or null}) made, and the keys that changed. Every value is checked first,
/// so a request with one bad value changes nothing. Call it as runconfig::apply: unqualified, with json arguments,
/// argument-dependent lookup prefers std::apply
std::pair<json, std::vector<std::string>> apply(const json& cfg, const json& changes);
/// the config written whole (a temporary file moved over the old one), the earlier one kept as <name>.bak: its path
std::string save(const std::string& path, const json& cfg);
json load(const std::string& path);

}  // namespace lamina::runconfig
