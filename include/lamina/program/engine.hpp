// include/lamina/program/engine.hpp - the engine as a function: the `lamina-engine` binary's main, and the engine that
// lamina-server runs on a thread of its own (with in-process channels: serve_io.hpp).
#pragma once

namespace lamina::program {

/// The engine's command line (argv[0] is the program's name). Returns its exit code.
int run(int argc, char** argv);

/// Whether run() would take this command line: its options, read by the parser of the family --native's file is of
/// (Qwen3.8-Flash-Next's, or the qwen35 path's, which then also refuses an option it would skip under --serve).  0, else
/// the exit code with the reason on stderr.  Nothing is loaded and no CUDA call is made: lamina-server checks the
/// engine options of its command line with it before it starts.
int check_args(int argc, char** argv);

/// Qwen3.8-Flash-Next's default for an option under --serve, as its value reads ("4", "0.5", "auto", "kernel"; a
/// switch's "on" or "off"); empty for an option it does not know or one without a default worth writing.  A config that
/// lamina-server --write-config writes leaves out the options at their default.
const char* option_default(const char* flag);

}  // namespace lamina::program
