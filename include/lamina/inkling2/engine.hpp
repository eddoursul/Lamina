// include/lamina/inkling2/engine.hpp - the Inkling 2 (GGUF architectures inkling2 and inkling2_loop) engine path as a
// library: `lamina` hands such a model to it at the top of main (the family switch), and lamina-inkling2 is a thin
// wrapper around the same entry point.  See src/inkling2/inkling2_engine.cu.
#pragma once

namespace lamina::inkling2 {

/// The engine's command line (lamina's flag names where they mean the same thing: --native / --model,
/// --max-context, --serve, --tokens, --n, --force, --dump-full-logits, --kv).  Returns the process exit code.
int run(int argc, char** argv);

/// run()'s parse of the command line alone: 0, else the exit code with the reason on stderr.
int check_args(int argc, char** argv);

}  // namespace lamina::inkling2
