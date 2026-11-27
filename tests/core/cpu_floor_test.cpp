// tests/core/cpu_floor_test.cpp - cpu_floor_missing, the programs' CPU check at their start
// (src/platform/cpu_floor.cpp): this CPU meets the build's floor (the development PC and the CI runners have AVX2), and
// LAMINA_FORCE_ISA, which caps the engine's own probes, leaves the answer alone.  (CPU only.)
#include "lamina/kernels/cpu/expert_layout.hpp"

#include <cstdio>
#include <cstdlib>

int main() {
#if defined(_WIN32)
    _putenv_s("LAMINA_FORCE_ISA", "sse");
#else
    setenv("LAMINA_FORCE_ISA", "sse", 1);
#endif
    namespace cpu = lamina::kernels::cpu;
    int fails = 0;
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu::cpu_avx2_ok()) {
        std::printf("FAIL: LAMINA_FORCE_ISA=sse left cpu_avx2_ok true\n");
        ++fails;
    }
#endif
    if (const char* missing = cpu::cpu_floor_missing()) {
        std::printf("FAIL: this CPU has no %s\n", missing);
        ++fails;
    }
    std::printf(fails ? "cpu_floor_test: %d failures\n" : "cpu_floor_test: ok\n", fails);
    return fails ? 1 : 0;
}
