// include/lamina/qwen35/engine.hpp - the Qwen3.6-35B-A3B (GGUF architecture qwen35moe) engine path as a library:
// `lamina-engine` hands a qwen35moe model to it at the top of main() (the family switch), and lamina-q36 is a thin
// wrapper around the same entry point.  See src/qwen35/q36_engine.cu.
#pragma once

namespace lamina::qwen35 {

/// The engine's command line (lamina-engine's own flag names where they mean the same thing: --native / --model,
/// --max-context, --prefill / --chunk, --serve, --pool-workers, --tokens, --max-new, --no-token-graph; the family's
/// own: --drafts-per-step, --expert-vram, --expert-slots,
/// --hybrid-rows, --cpu-threads).  Returns the process exit code.
int run(int argc, char** argv);

/// run()'s parse of the command line alone, an option it does not have refused even under --serve: 0, else the exit
/// code with the reason on stderr (lamina::program::check_args).
int check_args(int argc, char** argv);

}   // namespace lamina::qwen35
