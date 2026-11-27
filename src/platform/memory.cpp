// src/platform/memory.cpp - see include/lamina/platform/memory.hpp.
#include "lamina/platform/memory.hpp"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <dxgi1_4.h>
#include <cstring>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#endif
#include <cstdio>
#include <cstdlib>

namespace lamina::platform {

#if defined(_WIN32)
namespace {
// Locked pages count against the minimum working set: a lock raises it by the region plus this headroom for the
// rest of the process, and its unlock lowers it by as much (each in-process engine start locks again)
constexpr SIZE_T kLockMargin = (SIZE_T) 512 << 20;
}   // namespace

LockResult lock_resident(void* p, uint64_t bytes) {
    LockResult r;
    if (p == nullptr || bytes == 0) {
        r.note = "nothing to lock";
        return r;
    }
    HANDLE self = GetCurrentProcess();
    SIZE_T min_ws = 0, max_ws = 0;
    DWORD flags = 0;
    if (!GetProcessWorkingSetSizeEx(self, &min_ws, &max_ws, &flags)) {
        r.note = "GetProcessWorkingSetSizeEx failed (error " + std::to_string(GetLastError()) + ")";
        return r;
    }
    // soft limits: the maximum is not enforced, only the minimum is raised
    const SIZE_T new_min = min_ws + (SIZE_T) bytes + kLockMargin;
    const SIZE_T new_max = max_ws > new_min + kLockMargin ? max_ws : new_min + kLockMargin;
    if (!SetProcessWorkingSetSizeEx(self, new_min, new_max,
                                    QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE)) {
        r.note = "SetProcessWorkingSetSizeEx(" + std::to_string((unsigned long long) (new_min >> 20)) +
                 " MiB) failed (error " + std::to_string(GetLastError()) + ")";
        return r;
    }
    const uint64_t chunk = 1ull << 30;
    uint8_t* base = (uint8_t*) p;
    for (uint64_t off = 0; off < bytes; off += chunk) {
        const uint64_t n = bytes - off < chunk ? bytes - off : chunk;
        if (!VirtualLock(base + off, (SIZE_T) n)) {
            r.note = "VirtualLock stopped at " + std::to_string((unsigned long long) (off >> 20)) + " of " +
                     std::to_string((unsigned long long) (bytes >> 20)) + " MiB (error " +
                     std::to_string(GetLastError()) + ")";
            r.ok = off > 0;
            // the minimum holds what was locked (its unlock lowers it by that and the margin), or is as it was
            if (off > 0)
                SetProcessWorkingSetSizeEx(self, min_ws + (SIZE_T) off + kLockMargin, new_max,
                                           QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
            else SetProcessWorkingSetSizeEx(self, min_ws, max_ws, flags);
            return r;
        }
        r.locked_bytes = off + n;
    }
    r.ok = true;
    r.note =
        "locked " + std::to_string((unsigned long long) (bytes >> 20)) + " MiB via working-set minimum + VirtualLock";
    return r;
}

void unlock_resident(void* p, uint64_t bytes) {
    if (p == nullptr || bytes == 0) return;
    const uint64_t chunk = 1ull << 30;
    for (uint64_t off = 0; off < bytes; off += chunk)
        VirtualUnlock((uint8_t*) p + off, (SIZE_T) (bytes - off < chunk ? bytes - off : chunk));
    HANDLE self = GetCurrentProcess();
    SIZE_T min_ws = 0, max_ws = 0;
    DWORD flags = 0;
    const SIZE_T raised = (SIZE_T) bytes + kLockMargin;   // what lock_resident added for them
    if (GetProcessWorkingSetSizeEx(self, &min_ws, &max_ws, &flags) && min_ws > raised)
        SetProcessWorkingSetSizeEx(self, min_ws - raised, max_ws, flags);
}

bool keep_from_children(void*, uint64_t) { return true; }   // CreateProcess shares no memory with the child

bool gpu_shared_memory_budget(const void* luid, uint64_t& budget, uint64_t& usage, std::string& why) {
    budget = usage = 0;
    // dxgi.dll is loaded when asked, not linked: a start that never needs this keeps the imports it had
    HMODULE dxgi = LoadLibraryA("dxgi.dll");
    if (dxgi == nullptr) {
        why = "dxgi.dll not found";
        return false;
    }
    using CreateFactory = HRESULT(WINAPI*)(REFIID, void**);
    const auto create = (CreateFactory) (void*) GetProcAddress(dxgi, "CreateDXGIFactory1");
    IDXGIFactory1* factory = nullptr;
    if (create == nullptr || FAILED(create(__uuidof(IDXGIFactory1), (void**) &factory)) || factory == nullptr) {
        why = "CreateDXGIFactory1 failed";
        FreeLibrary(dxgi);
        return false;
    }
    bool ok = false;
    why = "no DXGI adapter has the CUDA device's LUID";
    for (UINT i = 0; !ok; ++i) {
        IDXGIAdapter1* a = nullptr;
        if (factory->EnumAdapters1(i, &a) == DXGI_ERROR_NOT_FOUND || a == nullptr) break;
        DXGI_ADAPTER_DESC1 d{};
        if (SUCCEEDED(a->GetDesc1(&d)) && std::memcmp(&d.AdapterLuid, luid, sizeof d.AdapterLuid) == 0) {
            IDXGIAdapter3* a3 = nullptr;
            DXGI_QUERY_VIDEO_MEMORY_INFO info{};
            if (SUCCEEDED(a->QueryInterface(__uuidof(IDXGIAdapter3), (void**) &a3)) && a3 != nullptr &&
                SUCCEEDED(a3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &info))) {
                budget = info.Budget;
                usage = info.CurrentUsage;
                ok = budget > 0;
                why = ok ? "" : "the adapter reports no shared-memory budget";
            } else {
                why = "QueryVideoMemoryInfo failed";
            }
            if (a3 != nullptr) a3->Release();
            a->Release();
            break;
        }
        a->Release();
    }
    factory->Release();
    FreeLibrary(dxgi);
    return ok;
}

bool available_ram(uint64_t& bytes) {
    MEMORYSTATUSEX m{};
    m.dwLength = sizeof m;
    bytes = GlobalMemoryStatusEx(&m) ? (uint64_t) m.ullAvailPhys : 0;
    return bytes > 0;
}

bool total_ram(uint64_t& bytes) {
    MEMORYSTATUSEX m{};
    m.dwLength = sizeof m;
    bytes = GlobalMemoryStatusEx(&m) ? (uint64_t) m.ullTotalPhys : 0;
    return bytes > 0;
}

bool read_ahead_enabled() { return false; }
void advise_willneed(const void*, uint64_t) {}
void advise_willneed(int, uint64_t, uint64_t) {}
#else
LockResult lock_resident(void* p, uint64_t bytes) {
    LockResult r;
    if (p == nullptr || bytes == 0) {
        r.note = "nothing to lock";
        return r;
    }
    if (mlock(p, bytes) != 0) {
        r.note = "mlock failed (raise ulimit -l)";
        return r;
    }
    r.ok = true;
    r.locked_bytes = bytes;
    r.note = "mlock";
    return r;
}

void unlock_resident(void* p, uint64_t bytes) {
    if (p != nullptr && bytes != 0) munlock(p, bytes);
}

bool keep_from_children(void* p, uint64_t bytes) {
    return p != nullptr && bytes != 0 && madvise(p, (size_t) bytes, MADV_DONTFORK) == 0;
}

bool gpu_shared_memory_budget(const void*, uint64_t& budget, uint64_t& usage, std::string& why) {
    budget = usage = 0;
    why = "DXGI is Windows-only";
    return false;
}

namespace {
// /proc/meminfo's `key` ("MemAvailable:") in bytes
bool meminfo(const char* key, uint64_t& bytes) {
    bytes = 0;
    FILE* f = std::fopen("/proc/meminfo", "r");
    if (f == nullptr) return false;
    char line[256];
    const size_t n = std::strlen(key);
    unsigned long long kb = 0;
    bool found = false;
    while (!found && std::fgets(line, sizeof line, f))
        found = std::strncmp(line, key, n) == 0 && std::sscanf(line + n, " %llu kB", &kb) == 1;
    std::fclose(f);
    bytes = found ? (uint64_t) kb << 10 : 0;
    return found;
}
}   // namespace

bool available_ram(uint64_t& bytes) { return meminfo("MemAvailable:", bytes); }

bool total_ram(uint64_t& bytes) { return meminfo("MemTotal:", bytes); }

namespace {
constexpr uint64_t kAdviseStep = 128ull << 10;
}

bool read_ahead_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("LAMINA_READ_AHEAD");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
}

void advise_willneed(const void* p, uint64_t bytes) {
    if (p == nullptr || bytes == 0 || !read_ahead_enabled()) return;
    const long ps = sysconf(_SC_PAGE_SIZE);
    const uintptr_t pg = ps > 0 ? (uintptr_t) ps : 4096, end = (uintptr_t) p + bytes;
    for (uintptr_t a = (uintptr_t) p & ~(pg - 1); a < end; a += kAdviseStep)
        (void) madvise((void*) a, (size_t) std::min<uintptr_t>(kAdviseStep, end - a), MADV_WILLNEED);
}

void advise_willneed(int fd, uint64_t offset, uint64_t bytes) {
    if (fd < 0 || !read_ahead_enabled()) return;
    for (uint64_t at = 0; at < bytes; at += kAdviseStep)
        (void) posix_fadvise(fd, (off_t) (offset + at), (off_t) std::min(kAdviseStep, bytes - at), POSIX_FADV_WILLNEED);
}
#endif

namespace {
constexpr uint64_t kSystemRam = 4ull << 30;   // the system's and the engine's other memory beside a model's
constexpr uint64_t kSpareRam = 6ull << 30;    // available once it is taken: --ram-reserve-mib's default

std::string gib(uint64_t bytes) {
    char s[32];
    std::snprintf(s, sizeof s, "%.1f GiB", (double) bytes / 1073741824.0);
    return s;
}
}   // namespace

bool RamFit::too_much() const { return total > 0 && need + kSystemRam > total; }

bool RamFit::tight() const { return total > 0 && !too_much() && need + kSpareRam > available; }

uint64_t RamFit::room() const {
    if (total == 0) return UINT64_MAX;
    const uint64_t all = total > kSystemRam ? total - kSystemRam : 0;
    const uint64_t spare = available > kSpareRam ? available - kSpareRam : 0;
    return std::min(all, spare);
}

std::string RamFit::say(const std::string& what, const std::string& remedy) const {
    if (too_much())
        return "not enough RAM: " + what + " take " + gib(need) + ", and this PC has " + gib(total) +
               ", ~4 GiB of which the system needs: " + remedy;
    if (!tight()) return {};
    std::string s = "RAM is tight: " + what + " take " + gib(need) + ", ";
    s += need <= available ? "leaving " + gib(available - need) + " of the " + gib(available) + " available"
                           : gib(need - available) + " more than the " + gib(available) + " available";
#if defined(_WIN32)
    s += " (this PC has " + gib(total) + "); Windows pages other programs out to disk meanwhile: ";
#else
    s += " (this PC has " + gib(total) + "); Linux may end the engine when the memory runs out: ";
#endif
    return s + remedy;
}

RamFit ram_fit(uint64_t need) {
    RamFit f;
    f.need = need;
    unsigned long long t = 0, a = 0;
    if (const char* v = std::getenv("LAMINA_FORCE_RAM_MIB"); v != nullptr && std::sscanf(v, "%llu,%llu", &t, &a) == 2) {
        f.total = (uint64_t) t << 20;
        f.available = (uint64_t) a << 20;
    } else if (!total_ram(f.total) || !available_ram(f.available)) {
        f.total = 0;
    }
    return f;
}

}   // namespace lamina::platform
