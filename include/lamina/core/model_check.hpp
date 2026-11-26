// include/lamina/core/model_check.hpp - the model files the engine takes, decided from their headers alone.
//
// What each path's load refuses of a file by its header - its architecture and keys, every tensor's name, shape, type
// and place, the shards' sizes - is checked here before anything is allocated or read: each path's load calls its
// part first, and lamina-server --check calls the whole, so the server says of a file (on disk, or the copies of a
// remote file's headers) what a start would say.  Not here: what a file's values hold (a non-finite weight is refused
// as the pack converts it), what the options or this PC's memory and GPUs decide.
#pragma once

#include <string>
#include <vector>

namespace lamina {
class GgufModel;
}

namespace lamina::core {

/// The engine's options that change which files it takes
struct FileCheckOptions {
    std::string ple_gguf;        ///< --ple-gguf FILE (Qwen3.8-Flash-Next): the PLE table read from it, not the model's
    bool hc_q8 = true;           ///< --hc-q8 (the default) / --no-hc-q8 (Qwen3.8-Flash-Next)
    bool expert_exact = false;   ///< --expert-math exact (Qwen3.8-Flash-Next)
    bool mtp = true;             ///< the MTP block loaded: not --no-mtp, no drafter file named (Hy4-flash)
    bool vision = false;         ///< --vision: the vocabulary's token for a picture's rows needed (Hy4-flash, Inkling 2)
};
/// Those of an engine command line (the options after its program's name, as the server passes them)
FileCheckOptions file_check_options(const std::vector<std::string>& args);

/// Why the engine refuses the model `path` names (any of its shards), from its files' headers alone: "" where it reads
/// it.  The path the file's architecture picks (Qwen3.8-Flash-Next, Qwen3.6 / Qwen3.8-27B, Hy4-flash, Inkling 2) is
/// the one asked, and an architecture none of them runs is said so.
std::string model_file_problem(const std::string& path, const FileCheckOptions& o = {});

/// Each path's own part (its load calls it first): Qwen3.8-Flash-Next's from its shards (the PLE table's file too)
std::string flash_file_problem(const std::vector<std::string>& shards, const FileCheckOptions& o);
/// Qwen3.6-35B-A3B's and Qwen3.8-27B's (qwen35moe, qwen35)
std::string qwen35_file_problem(const GgufModel& m);
/// Hy4-flash's (hy_v4_flash)
std::string hy4_file_problem(const GgufModel& m, const FileCheckOptions& o);
/// Inkling 2's (inkling2, inkling2_loop)
std::string inkling2_file_problem(const GgufModel& m, const FileCheckOptions& o = {});

}  // namespace lamina::core
