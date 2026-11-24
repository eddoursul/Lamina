// include/lamina/embedded.hpp - files built into the programs (cmake/embed_files.cmake): the web app, the server's
// presets, the engine's expert profiles.
#pragma once

#include <span>
#include <string_view>

namespace lamina {

struct EmbeddedFile {
    std::string_view path;   ///< below the folder it was taken from ("index.html", "iq3_s.json")
    std::string_view data;
};

/// src/server/web's files (lamina-server)
const EmbeddedFile* web_asset(std::string_view path);
/// examples/*.json, the presets lamina-server builds a config from (src/server/model_config.hpp)
const EmbeddedFile* preset_file(std::string_view path);
/// data/expert-profiles/*.bin, the models' built-in expert profiles ("qwen3.8-flash-next.bin",
/// "inkling-2-mini.bin"; lamina/core/expert_cache.hpp), and every one of them
const EmbeddedFile* profile_file(std::string_view path);
std::span<const EmbeddedFile> profile_file_all();
/// data/bench-agent-files.json, the texts of lamina-server --bench's agent session (src/server/bench.hpp)
const EmbeddedFile* bench_file(std::string_view path);

}  // namespace lamina
