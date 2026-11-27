// src/platform/memory_test.cpp - plan v0.3 P0.1: lock_resident on a 256 MiB region and its undoing, available_ram,
// total_ram and the start's RAM check, RamFit, keep_from_children on Linux (CPU only, no GPU, no model).  On Linux a
// lock that ulimit -l (RLIMIT_MEMLOCK) does not allow is skipped, not failed
// (exit 77): the region cannot be locked there (8 MiB on CI's runners).
#include "lamina/platform/memory.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#else
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cerrno>
#endif

int main() {
    const uint64_t bytes = 256ull << 20;
    void* p = std::malloc(bytes);
    if (p == nullptr) return 2;
    std::memset(p, 1, bytes);
    const lamina::platform::LockResult r = lamina::platform::lock_resident(p, bytes);
    std::printf("lock_resident: ok=%d locked=%llu MiB (%s)\n", (int) r.ok, (unsigned long long) (r.locked_bytes >> 20),
                r.note.c_str());
    bool skipped = false;
#if !defined(_WIN32)
    rlimit rl{};
    if (!r.ok && getrlimit(RLIMIT_MEMLOCK, &rl) == 0 && rl.rlim_cur != RLIM_INFINITY && rl.rlim_cur < bytes) {
        std::printf("lock_resident: SKIPPED, ulimit -l is %llu KiB, below the region's 256 MiB\n",
                    (unsigned long long) (rl.rlim_cur >> 10));
        skipped = true;
    }
#endif
    int fail = !skipped && !(r.ok && r.locked_bytes == bytes);
    // the conversation cache's RAM reserve (--ram-reserve-mib) reads this figure
    uint64_t avail = 0;
    const bool got = lamina::platform::available_ram(avail);
    std::printf("available_ram: ok=%d %llu MiB\n", (int) got, (unsigned long long) (avail >> 20));
    fail |= !got || avail == 0;
    // the start's RAM check (RamFit): the RAM's size, and the verdicts on given figures
    uint64_t total = 0;
    const bool got_total = lamina::platform::total_ram(total);
    std::printf("total_ram: ok=%d %llu MiB\n", (int) got_total, (unsigned long long) (total >> 20));
    fail |= !got_total || total < avail;
    {
        using lamina::platform::RamFit;
        constexpr uint64_t G = 1ull << 30;
        const RamFit over{60 * G, 63 * G, 62 * G};     // 60 + the system's ~4 > 63: refused
        const RamFit tight{47 * G, 63 * G, 50 * G};    // 3 GiB left available: a warning
        const RamFit beyond{47 * G, 63 * G, 40 * G};   // more than is available: a warning
        const RamFit room{47 * G, 63 * G, 60 * G};     // 13 GiB left: nothing said
        const RamFit unknown{47 * G, 0, 0};            // the RAM not read: nothing said
        const std::string so = over.say("x", "y"), st = tight.say("x", "y"), sb = beyond.say("x", "y");
        const bool ok = over.too_much() && !over.tight() && so.rfind("not enough RAM: x take 60.0 GiB", 0) == 0 &&
                        so.find(": y") != std::string::npos && !tight.too_much() && tight.tight() &&
                        st.find("leaving 3.0 GiB of the 50.0 GiB available") != std::string::npos && beyond.tight() &&
                        sb.find("7.0 GiB more than the 40.0 GiB available") != std::string::npos && !room.too_much() &&
                        !room.tight() && room.say("x", "y").empty() && !unknown.too_much() && !unknown.tight() &&
                        unknown.say("x", "y").empty();
        std::printf("RamFit: %s\n  %s\n  %s\n  %s\n", ok ? "ok" : "WRONG", so.c_str(), st.c_str(), sb.c_str());
        fail |= !ok;
    }
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc{};
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc);
    std::printf("working set %llu MiB\n", (unsigned long long) (pmc.WorkingSetSize >> 20));
    fail |= pmc.WorkingSetSize < bytes;
#endif
    lamina::platform::unlock_resident(p, bytes);
#if defined(_WIN32)
    // the working-set minimum the lock raised is lowered by its unlock: three more rounds leave it where it was
    SIZE_T min0 = 0, max0 = 0, min1 = 0, max1 = 0;
    DWORD flags = 0;
    GetProcessWorkingSetSizeEx(GetCurrentProcess(), &min0, &max0, &flags);
    for (int k = 0; k < 3; ++k) {
        const lamina::platform::LockResult again = lamina::platform::lock_resident(p, bytes);
        lamina::platform::unlock_resident(p, again.locked_bytes);
    }
    GetProcessWorkingSetSizeEx(GetCurrentProcess(), &min1, &max1, &flags);
    std::printf("working-set minimum %llu MiB, after three more locks and unlocks %llu MiB\n",
                (unsigned long long) (min0 >> 20), (unsigned long long) (min1 >> 20));
    fail |= min1 != min0;
#endif
#if !defined(_WIN32)
    {   // keep_from_children: a region so advised is not in a forked child; one not advised is (mincore: ENOMEM where
        // nothing is mapped)
        const size_t n = 4ull << 20;
        void* kept = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        void* plain = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        bool ok = kept != MAP_FAILED && plain != MAP_FAILED;
        if (ok) {
            std::memset(kept, 1, n);
            std::memset(plain, 1, n);
            ok = lamina::platform::keep_from_children(kept, n);
        }
        if (ok) {
            std::fflush(stdout);
            const pid_t c = fork();
            if (c == 0) {
                unsigned char v = 0;
                const bool gone = mincore(kept, 4096, &v) != 0 && errno == ENOMEM;
                const bool there = mincore(plain, 4096, &v) == 0;
                _exit(gone && there ? 0 : 1);
            }
            int st = 0;
            ok = c > 0 && waitpid(c, &st, 0) == c && WIFEXITED(st) && WEXITSTATUS(st) == 0;
        }
        std::printf("keep_from_children: %s\n", ok ? "ok, not in a child (an unadvised region is)" : "WRONG");
        fail |= !ok;
        if (kept != MAP_FAILED) munmap(kept, n);
        if (plain != MAP_FAILED) munmap(plain, n);
    }
#endif
    std::free(p);
    std::printf("platform_memory_test: %s\n", fail ? "FAILED" : skipped ? "OK, the lock SKIPPED" : "OK");
    return fail ? 1 : skipped ? 77 : 0;
}
