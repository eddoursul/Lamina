// src/program/family_switch.hpp - lamina-engine's family switch (family_switch.cpp): a model file of another path's
// family handed to that path, before generate.cpp's engine reads the command line.
#pragma once

namespace lamina::program {

/// The switch's answer for a file generate.cpp's engine runs itself (Qwen3.8-Flash-Next's, or one the switch cannot
/// read: that engine's own parse says why)
constexpr int kOwnFile = -1;

/// The command line's model file read for its family: a file of the qwen35 path's, Hy4-flash's or Inkling 2's run
/// on that path (its exit code; a --gpu card it cannot read: 2), else kOwnFile
int family_switch(int argc, char** argv);
/// The same for check mode: the path's check of the command line (no CUDA call), else kOwnFile
int family_check(int argc, char** argv);

}   // namespace lamina::program
