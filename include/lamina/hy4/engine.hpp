// include/lamina/hy4/engine.hpp - the Hy4-flash (GGUF architecture hy_v4_flash) engine path as a library:
// `lamina-engine` hands a hy_v4_flash model to it at the top of main (the family switch), and lamina-hy4 is a thin
// wrapper around the same entry point.  See src/hy4/hy4_engine.cu.
#pragma once

namespace lamina::hy4 {

/// The engine's command line (lamina-engine's flag names where they mean the same thing: --native / --model,
/// --max-context, --serve, --tokens, --n, --force, --dump-full-logits; the path's own: --cpu-threads).  Returns the
/// process exit code.
int run(int argc, char** argv);

/// run()'s parse of the command line alone: 0, else the exit code with the reason on stderr.
int check_args(int argc, char** argv);

}  // namespace lamina::hy4
