// include/lamina/platform/memory.hpp - plan v0.3 P0.1/P1: keep a large host region resident.
//
// `cudaHostRegister` refuses the 31.6 GiB expert arena on Windows, and an unlocked arena is trimmed under memory
// pressure (the CPU pool then swung 2x between runs). With the n-gram table out of RAM there is headroom to lock
// it instead: raise the process's minimum working set by the region's size, then VirtualLock it (Windows needs
// only SeIncreaseWorkingSetPrivilege, which ordinary accounts hold). Linux: mlock.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace lamina::platform {

struct LockResult {
    bool ok = false;
    uint64_t locked_bytes = 0;   ///< may be less than requested; the rest stays pageable
    std::string note;            ///< what was done or why it failed, for the startup print
};

/// Lock [p, p + bytes) into physical memory. Partial success is reported, not hidden.
LockResult lock_resident(void* p, uint64_t bytes);

/// Undo lock_resident for the same region, `bytes` its LockResult::locked_bytes (best effort): on Windows the
/// working-set minimum goes down by what the lock raised it.
void unlock_resident(void* p, uint64_t bytes);

/// Keeps the page-aligned [p, p + bytes) out of the child processes this one starts: Linux madvise(MADV_DONTFORK),
/// nothing elsewhere (CreateProcess copies no memory).  A region cudaHostRegister pins needs it: fork() gives the child
/// a copy of every pinned page at once instead of sharing it copy-on-write, so a program the server starts once the
/// expert arena is pinned (nvidia-smi, an MCP server, sd-cli) would first take another arena's worth of RAM.  The
/// children exec at once and read none of it.  cudaHostAlloc's memory needs none: the driver maps its own pages, shared
/// with a child.  False when the advice is refused.
bool keep_from_children(void* p, uint64_t bytes);

/// #243, Windows: the GPU's shared (non-local) memory budget and this process's use of it, from DXGI
/// (IDXGIAdapter3::QueryVideoMemoryInfo, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL) for the adapter whose LUID is the
/// 8 bytes at `luid` (cudaDeviceProp::luid).  Page-locked host memory the GPU maps is charged there.  False (and
/// `why` says so) when the query is not possible - always elsewhere than Windows.
bool gpu_shared_memory_budget(const void* luid, uint64_t& budget, uint64_t& usage, std::string& why);

/// The RAM the system can still give without swapping, in `bytes`: MemAvailable on Linux, ullAvailPhys on Windows.
/// False (and 0) when it cannot be read.
bool available_ram(uint64_t& bytes);
/// The RAM the system has, in `bytes`: MemTotal on Linux, ullTotalPhys on Windows.  False (and 0) when it cannot be
/// read.
bool total_ram(uint64_t& bytes);

/// A model's host memory (`need`: its experts in RAM, a table read into it) against the RAM before any of it is taken,
/// in bytes; `total` 0 when the RAM cannot be read, and then nothing is said.
struct RamFit {
    uint64_t need = 0, total = 0, available = 0;
    /// It cannot run here: beside it the system and the engine's other memory (CUDA's, its buffers, the server's) take
    /// ~4 GiB of the RAM.
    bool too_much() const;
    /// It runs, but less than 6 GiB would stay available once it is taken: other programs' memory goes to the page file
    /// (Linux: the engine may be ended), and the conversation cache parks nothing (its --ram-reserve-mib, 6 GiB).
    bool tight() const;
    /// The largest need that is neither too_much() nor tight(), in bytes (UINT64_MAX when the RAM cannot be read).
    uint64_t room() const;
    /// What too_much() or tight() says, `what` the need's name ("the model's experts"), then `remedy`; "" when it fits.
    std::string say(const std::string& what, const std::string& remedy) const;
};
/// `need` against the RAM as it is now; LAMINA_FORCE_RAM_MIB=T,A (tests) stands for RAM of T MiB, A of them available.
RamFit ram_fit(uint64_t need);

/// Whether `advise_willneed` asks the OS for anything: not on Windows, nor with LAMINA_READ_AHEAD=0.
bool read_ahead_enabled();
/// Asks the OS to start reading [p, p + bytes) of a file mapping, without waiting.  Linux reads at most one
/// readahead window per request, so the range is asked for in 128 KiB steps.
void advise_willneed(const void* p, uint64_t bytes);
/// The same for [offset, offset + bytes) of an open file.
void advise_willneed(int fd, uint64_t offset, uint64_t bytes);

}   // namespace lamina::platform
