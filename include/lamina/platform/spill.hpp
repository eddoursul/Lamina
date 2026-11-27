// include/lamina/platform/spill.hpp - the prompt cache's disk tier: a folder of this process's own and files read
// and written past the OS file cache.
//
// A parked conversation is gigabytes of K/V that never need the file cache: unbuffered I/O keeps it out of RAM
// (Windows FILE_FLAG_NO_BUFFERING, Linux O_DIRECT; a file system that refuses O_DIRECT is written through its cache).
// Every engine process writes into `<root>/<pid>-<16 hex digits>` and holds `.lock` there for its life (a handle
// that shares nothing on Windows, flock elsewhere): a start removes the folders of that form whose lock nobody holds
// (an ended or crashed engine's) and nothing else, so two engines can share a root.
#pragma once

#include <cstdint>
#include <string>

namespace lamina::platform {

/// Offsets, lengths and buffers of a SpillFile's reads and writes are multiples of this.
inline constexpr uint64_t kSpillAlign = 4096;
inline uint64_t spill_align_up(uint64_t n) { return (n + kSpillAlign - 1) / kSpillAlign * kSpillAlign; }

/// Page-aligned host memory for such I/O (null: none).
void* spill_alloc(uint64_t bytes);
void spill_free(void* p);
/// Removes the file at `path` (UTF-8), if there is one.
void spill_remove(const std::string& path);

/// One file, read and written synchronously at given offsets, unbuffered.
class SpillFile {
public:
    SpillFile() = default;
    ~SpillFile() { close(); }
    SpillFile(const SpillFile&) = delete;
    SpillFile& operator=(const SpillFile&) = delete;

    /// Opens `path` (UTF-8); `create`: made where it does not exist yet.
    bool open(const std::string& path, bool create, std::string& err);
    void close();
    bool write(uint64_t offset, const void* buf, uint64_t bytes, std::string& err);
    bool read(uint64_t offset, void* buf, uint64_t bytes, std::string& err);

private:
#if defined(_WIN32)
    void* h_ = reinterpret_cast<void*>(~uintptr_t(0));   // INVALID_HANDLE_VALUE
#else
    int fd_ = -1;
#endif
};

/// This process's folder under a root.
class SpillDir {
public:
    SpillDir() = default;
    ~SpillDir() { close(); }
    SpillDir(const SpillDir&) = delete;
    SpillDir& operator=(const SpillDir&) = delete;

    /// Makes `root`, removes the folders of ended processes there (`*cleaned`: how many) and takes a folder of
    /// this process's own.
    bool open(const std::string& root, std::string& err, int* cleaned = nullptr);
    /// Gives the lock back and removes the folder with what it holds.
    void close();
    bool is_open() const { return !dir_.empty(); }
    const std::string& path() const { return dir_; }
    /// The path of file `name` in the folder.
    std::string file(const std::string& name) const;
    /// The bytes free to this process on the folder's disk (0: unknown).
    uint64_t free_bytes() const;

private:
    std::string dir_;
#if defined(_WIN32)
    void* lock_ = reinterpret_cast<void*>(~uintptr_t(0));
#else
    int lock_ = -1;
#endif
};

}   // namespace lamina::platform
