// src/platform/spill.cpp - see include/lamina/platform/spill.hpp.
#include "lamina/platform/spill.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string_view>
#include <system_error>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace lamina::platform {

namespace {

fs::path u8_path(std::string_view s) {
    return fs::path(std::u8string(reinterpret_cast<const char8_t*>(s.data()), s.size()));
}

std::string u8_str(const fs::path& p) {
    const std::u8string u = p.u8string();
    return std::string(u.begin(), u.end());
}

/// `<digits>-<16 hex digits>`: an engine process's folder
bool process_folder(const std::string& n) {
    const size_t dash = n.find('-');
    if (dash == std::string::npos || dash == 0 || n.size() != dash + 17) return false;
    for (size_t i = 0; i < dash; ++i)
        if (n[i] < '0' || n[i] > '9') return false;
    for (size_t i = dash + 1; i < n.size(); ++i)
        if (!((n[i] >= '0' && n[i] <= '9') || (n[i] >= 'a' && n[i] <= 'f'))) return false;
    return true;
}

/// The largest piece one call moves (the Windows calls take a 32-bit length).
constexpr uint64_t kPiece = uint64_t(1) << 30;

#if defined(_WIN32)
HANDLE handle(void* h) { return static_cast<HANDLE>(h); }

std::wstring wide(const std::string& s) { return u8_path(s).wstring(); }

/// The lock of folder `dir`: `.lock` opened sharing nothing, so no other handle can open it while this one is held.
HANDLE take_lock(const fs::path& dir) {
    return CreateFileW((dir / ".lock").wstring().c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_ALWAYS,
                       FILE_ATTRIBUTE_NORMAL, nullptr);
}
bool held(HANDLE h) { return h != INVALID_HANDLE_VALUE; }
void give_lock(HANDLE h) {
    if (held(h)) CloseHandle(h);
}
unsigned long own_pid() { return (unsigned long) GetCurrentProcessId(); }
#else
bool held(int fd) { return fd >= 0; }
int take_lock(const fs::path& dir) {
    const int fd = ::open((dir / ".lock").c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd < 0) return -1;
    if (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
        ::close(fd);
        return -1;
    }
    return fd;
}
void give_lock(int fd) {
    if (fd >= 0) ::close(fd);
}
unsigned long own_pid() { return (unsigned long) ::getpid(); }
#endif

bool aligned(uint64_t offset, const void* buf, uint64_t bytes) {
    return offset % kSpillAlign == 0 && bytes % kSpillAlign == 0 && (uintptr_t) buf % kSpillAlign == 0;
}

}  // namespace

void* spill_alloc(uint64_t bytes) {
#if defined(_WIN32)
    return VirtualAlloc(nullptr, (SIZE_T) bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* p = nullptr;
    return posix_memalign(&p, kSpillAlign, (size_t) bytes) == 0 ? p : nullptr;
#endif
}

void spill_free(void* p) {
    if (p == nullptr) return;
#if defined(_WIN32)
    VirtualFree(p, 0, MEM_RELEASE);
#else
    std::free(p);
#endif
}

void spill_remove(const std::string& path) {
    std::error_code ec;
    fs::remove(u8_path(path), ec);
}

// ------------------------------------------------------------------------------------------------ SpillFile
#if defined(_WIN32)
bool SpillFile::open(const std::string& path, bool create, std::string& err) {
    close();
    h_ = CreateFileW(wide(path).c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr,
                     create ? OPEN_ALWAYS : OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING, nullptr);
    if (handle(h_) == INVALID_HANDLE_VALUE) {
        err = "cannot open " + path + " (error " + std::to_string(GetLastError()) + ")";
        return false;
    }
    return true;
}

void SpillFile::close() {
    if (handle(h_) != INVALID_HANDLE_VALUE) CloseHandle(handle(h_));
    h_ = INVALID_HANDLE_VALUE;
}

bool SpillFile::write(uint64_t offset, const void* buf, uint64_t bytes, std::string& err) {
    if (!aligned(offset, buf, bytes)) {
        err = "an unaligned write";
        return false;
    }
    for (uint64_t at = 0; at < bytes;) {
        const DWORD n = (DWORD) std::min(kPiece, bytes - at);
        OVERLAPPED ov{};   // a synchronous handle: the call returns once the bytes are written at this offset
        ov.Offset = (DWORD) ((offset + at) & 0xFFFFFFFFull);
        ov.OffsetHigh = (DWORD) ((offset + at) >> 32);
        DWORD done = 0;
        if (!WriteFile(handle(h_), (const uint8_t*) buf + at, n, &done, &ov) || done != n) {
            err = "a write failed (error " + std::to_string(GetLastError()) + ")";
            return false;
        }
        at += n;
    }
    return true;
}

bool SpillFile::read(uint64_t offset, void* buf, uint64_t bytes, std::string& err) {
    if (!aligned(offset, buf, bytes)) {
        err = "an unaligned read";
        return false;
    }
    for (uint64_t at = 0; at < bytes;) {
        const DWORD n = (DWORD) std::min(kPiece, bytes - at);
        OVERLAPPED ov{};
        ov.Offset = (DWORD) ((offset + at) & 0xFFFFFFFFull);
        ov.OffsetHigh = (DWORD) ((offset + at) >> 32);
        DWORD done = 0;
        if (!ReadFile(handle(h_), (uint8_t*) buf + at, n, &done, &ov) || done != n) {
            err = "a read failed (error " + std::to_string(GetLastError()) + ", " + std::to_string(done) + " of " +
                  std::to_string(n) + " bytes)";
            return false;
        }
        at += n;
    }
    return true;
}
#else
bool SpillFile::open(const std::string& path, bool create, std::string& err) {
    close();
    const int flags = O_RDWR | O_CLOEXEC | (create ? O_CREAT : 0);
    fd_ = ::open(path.c_str(), flags | O_DIRECT, 0600);
    if (fd_ < 0 && errno == EINVAL) fd_ = ::open(path.c_str(), flags, 0600);   // no O_DIRECT here: through the cache
    if (fd_ < 0) {
        err = "cannot open " + path + " (" + std::strerror(errno) + ")";
        return false;
    }
    return true;
}

void SpillFile::close() {
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
}

bool SpillFile::write(uint64_t offset, const void* buf, uint64_t bytes, std::string& err) {
    if (!aligned(offset, buf, bytes)) {
        err = "an unaligned write";
        return false;
    }
    for (uint64_t at = 0; at < bytes;) {
        const ssize_t n = ::pwrite(fd_, (const uint8_t*) buf + at, (size_t) std::min(kPiece, bytes - at),
                                   (off_t) (offset + at));
        if (n <= 0) {
            err = std::string("a write failed (") + (n < 0 ? std::strerror(errno) : "no progress") + ")";
            return false;
        }
        at += (uint64_t) n;
    }
    return true;
}

bool SpillFile::read(uint64_t offset, void* buf, uint64_t bytes, std::string& err) {
    if (!aligned(offset, buf, bytes)) {
        err = "an unaligned read";
        return false;
    }
    for (uint64_t at = 0; at < bytes;) {
        const ssize_t n = ::pread(fd_, (uint8_t*) buf + at, (size_t) std::min(kPiece, bytes - at), (off_t) (offset + at));
        if (n <= 0) {
            err = std::string("a read failed (") + (n < 0 ? std::strerror(errno) : "the file ends") + ")";
            return false;
        }
        at += (uint64_t) n;
    }
    return true;
}
#endif

// ------------------------------------------------------------------------------------------------ SpillDir
bool SpillDir::open(const std::string& root, std::string& err, int* cleaned) {
    close();
    if (cleaned != nullptr) *cleaned = 0;
    const fs::path r = u8_path(root);
    std::error_code ec;
    fs::create_directories(r, ec);
    if (ec || !fs::is_directory(r, ec) || fs::is_symlink(r, ec)) {
        err = "the folder " + root + " cannot be used" + (ec ? ": " + ec.message() : std::string());
        return false;
    }
    // the folders of ended engines: named as a process's, real folders, and their lock free
    for (fs::directory_iterator it(r, ec), end; !ec && it != end; it.increment(ec)) {
        std::error_code e2;
        const fs::path p = it->path();
        if (it->is_symlink(e2) || !it->is_directory(e2) || !process_folder(u8_str(p.filename()))) continue;
        const auto h = take_lock(p);
        if (!held(h)) continue;   // a running engine's (or one that cannot be checked): kept
        give_lock(h);
        fs::remove_all(p, e2);
        if (!e2 && cleaned != nullptr) ++*cleaned;
    }
    std::random_device rd;
    char name[48];
    std::snprintf(name, sizeof name, "%lu-%08x%08x", own_pid(), (unsigned) rd(), (unsigned) rd());
    const fs::path dir = r / name;
    if (!fs::create_directory(dir, ec) || ec) {
        err = "the folder " + u8_str(dir) + " cannot be made" + (ec ? ": " + ec.message() : std::string());
        return false;
    }
    const auto h = take_lock(dir);
    if (!held(h)) {
        fs::remove_all(dir, ec);
        err = "the folder " + u8_str(dir) + ": its lock cannot be taken";
        return false;
    }
    lock_ = h;
    dir_ = u8_str(fs::path(dir).make_preferred());
    return true;
}

std::string SpillDir::file(const std::string& name) const { return u8_str((u8_path(dir_) / name).make_preferred()); }

void SpillDir::close() {
    if (dir_.empty()) return;
#if defined(_WIN32)
    give_lock(handle(lock_));
    lock_ = INVALID_HANDLE_VALUE;
#else
    give_lock(lock_);
    lock_ = -1;
#endif
    std::error_code ec;
    fs::remove_all(u8_path(dir_), ec);
    dir_.clear();
}

uint64_t SpillDir::free_bytes() const {
    std::error_code ec;
    const fs::space_info s = fs::space(u8_path(dir_), ec);
    return ec ? 0 : (uint64_t) s.available;
}

}  // namespace lamina::platform
