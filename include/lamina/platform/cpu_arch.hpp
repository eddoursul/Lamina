// lamina/platform/cpu_arch.hpp - the CPU architecture the build is for, and what spin loops and the GPU's flags need
// from it.
//
// LAMINA_X86 is 1 on x86-64, 0 on arm64.  The x86 expert kernels (AVX-512, AVX2, AVX: the *_avx*.cpp files and
// expert.cpp) exist on x86-64 only (CMakeLists.txt, LAMINA_CPU_X86); on arm64 the dispatch that would choose them is
// compiled out and the experts run on ggml-cpu's dot products or the scalar references.
#pragma once

#if defined(_M_X64) || defined(__x86_64__)
#define LAMINA_X86 1
#include <immintrin.h>
#elif defined(_M_ARM64) || defined(__aarch64__)
#define LAMINA_X86 0
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#else
#error "Lamina builds for x86-64 and arm64 CPUs"
#endif

namespace lamina::platform {

/// One wait of a spin loop: PAUSE on x86-64 (it yields the pipeline to the sibling hyperthread), YIELD on arm64.
inline void cpu_relax() {
#if LAMINA_X86
    _mm_pause();
#elif defined(_MSC_VER)
    __yield();
#else
    __asm__ __volatile__("yield" ::: "memory");
#endif
}

/// Orders the stores before it ahead of the stores after it for a device reading host memory (a flag a GPU polls
/// behind the data it publishes): SFENCE on x86-64, DMB OSHST on arm64 (Linux's dma_wmb).
inline void device_store_fence() {
#if LAMINA_X86
    _mm_sfence();
#elif defined(_MSC_VER)
    __dmb(0x2);   // OSHST
#else
    __asm__ __volatile__("dmb oshst" ::: "memory");
#endif
}

}   // namespace lamina::platform
