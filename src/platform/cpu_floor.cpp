// src/platform/cpu_floor.cpp - lamina-engine's and lamina's CPU check, before anything else of theirs runs: the
// engine's CPU kernels and ggml-cpu are built for AVX2 (with FMA and F16C), and on a CPU without it a program would
// stop at an illegal instruction somewhere in its start. Its initializer runs ahead of the libraries' (MSVC's lib
// segment, GCC's constructor priority 101), in a file compiled for the x86-64 baseline. A source of each program
// (CMakeLists.txt): a static library's member that nothing calls is not linked in.
#include "lamina/kernels/cpu/expert_layout.hpp"

#include <cstdio>
#include <cstdlib>

#if defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER)
#pragma warning(disable : 4073)   // "initializers put in library initialization area": they are meant to be
#pragma init_seg(lib)
#endif

namespace {

void cpu_floor_check() {
    if (const char* missing = lamina::kernels::cpu::cpu_floor_missing()) {
        std::fprintf(stderr, "[lamina] this CPU has no %s, which Lamina's CPU code needs (or its OS keeps it off)\n",
                     missing);
        std::exit(1);
    }
}

#if defined(_MSC_VER)
const struct CpuFloorCheck {
    CpuFloorCheck() { cpu_floor_check(); }
} cpu_floor_at_start;
#else
__attribute__((constructor(101))) void cpu_floor_at_start() { cpu_floor_check(); }
#endif

}   // namespace
#endif
