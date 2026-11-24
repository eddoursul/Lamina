// src/platform/slice_read.cpp - see lamina/platform/slice_read.hpp.  A copy out of a mapped file faults its pages in a few at a time (the routed
// experts at 1.2-1.5 GiB/s from this PC's SSD); large reads go at the drive's rate (4.3-4.4 GiB/s), and past the cache
// they leave the file's pages out of RAM the arena needs.
#include "lamina/platform/slice_read.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/uio.h>
#include <unistd.h>
#endif

namespace lamina::platform {
namespace {

struct Piece {
    size_t run = 0;
    int64_t first = 0, n = 0;   // slices
    uint64_t off = 0, len = 0;  // a part of one slice larger than a piece (n = 1); len 0: the slices whole
};

constexpr uint64_t kSector = 4096;   // unbuffered reads: offset, length and buffer in whole sectors

#if defined(_WIN32)
std::wstring widen(const std::string& s) {
    std::wstring w((size_t) MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), (int) w.size());
    if (!w.empty() && w.back() == L'\0') w.pop_back();
    return w;
}
#endif

}  // namespace

bool read_slices(const std::vector<SliceRun>& runs, int threads, bool uncached, uint64_t piece,
                 const std::function<uint8_t*(int)>& base, std::string& err, double* seconds) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<Piece> pieces;
    uint64_t largest = 0;
    for (size_t r = 0; r < runs.size(); ++r) {
        const SliceRun& s = runs[r];
        if (s.bytes == 0 || s.count <= 0) continue;
        if (s.bytes > piece) {   // a slice larger than a piece: in parts
            for (int64_t i = 0; i < s.count; ++i)
                for (uint64_t o = 0; o < s.bytes; o += piece) pieces.push_back({r, i, 1, o, std::min(piece, s.bytes - o)});
            largest = std::max(largest, piece);
            continue;
        }
        const int64_t k = std::max<int64_t>(1, std::min<int64_t>(512, (int64_t) (piece / s.bytes)));
        for (int64_t i = 0; i < s.count; i += k) pieces.push_back({r, i, std::min(k, s.count - i), 0, 0});
        largest = std::max(largest, (uint64_t) k * s.bytes);
    }
    std::atomic<size_t> next{0};
    std::atomic<bool> bad{false};
    std::mutex err_mu;
    auto fail = [&](const std::string& e) {
        std::lock_guard<std::mutex> lk(err_mu);
        if (!bad.exchange(true)) err = e;
    };
    auto worker = [&] {
#if defined(_WIN32)
        std::map<std::string, HANDLE> handles;
        std::map<std::string, FILE*> files;
        const size_t buf_bytes = (size_t) ((largest + 2 * kSector + kSector - 1) / kSector * kSector);
        uint8_t* buf = (uint8_t*) VirtualAlloc(nullptr, buf_bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if (buf == nullptr) { fail("no memory for the read buffer"); return; }
#else
        std::map<std::string, int> fds;
        std::vector<struct iovec> iov;
#endif
        for (size_t p; !bad && (p = next.fetch_add(1)) < pieces.size();) {
            const Piece& pc = pieces[p];
            const SliceRun& s = runs[pc.run];
            uint8_t* const group = base(s.group);
            if (group == nullptr) { fail("no memory for the routed experts"); break; }
            // slice i's place: its own (places) or the run's stride
            auto place = [&](int64_t i) {
                return s.places ? s.places[pc.first + i] + s.dst_off + pc.off
                                : group + s.dst_off + (uint64_t) (pc.first + i) * s.stride + pc.off;
            };
            const uint64_t at = s.offset + (uint64_t) pc.first * s.bytes + pc.off;
            const uint64_t n = pc.len ? pc.len : (uint64_t) pc.n * s.bytes;   // bytes
            const uint64_t slice = pc.len ? pc.len : s.bytes;                  // one slice's (or part's)
#if defined(_WIN32)
            (void) slice;
            const uint8_t* src = nullptr;
            if (uncached) {
                HANDLE& h = handles[s.path];
                if (h == nullptr) {
                    h = CreateFileW(widen(s.path).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                    FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
                    if (h == INVALID_HANDLE_VALUE) { fail("cannot open " + s.path); h = nullptr; break; }
                }
                const uint64_t at0 = at / kSector * kSector, head = at - at0;
                const uint64_t len = (head + n + kSector - 1) / kSector * kSector;
                OVERLAPPED ov{};
                ov.Offset = (DWORD) at0;
                ov.OffsetHigh = (DWORD) (at0 >> 32);
                DWORD got = 0;
                if (!ReadFile(h, buf, (DWORD) len, &got, &ov) || got < head + n) { fail("a read of " + s.path + " failed"); break; }
                src = buf + head;
            } else {
                FILE*& f = files[s.path];
                if (f == nullptr) f = std::fopen(s.path.c_str(), "rb");
                if (f == nullptr) { fail("cannot open " + s.path); break; }
                if (_fseeki64(f, (long long) at, SEEK_SET) != 0 || std::fread(buf, 1, (size_t) n, f) != (size_t) n) {
                    fail("a read of " + s.path + " failed");
                    break;
                }
                src = buf;
            }
            if (pc.n == 1 || (!s.places && s.stride == s.bytes)) {
                std::memcpy(place(0), src, (size_t) n);
            } else {
                for (int64_t i = 0; i < pc.n; ++i) std::memcpy(place(i), src + (uint64_t) i * s.bytes, (size_t) s.bytes);
            }
#else
            int fd = -1;
            if (const auto it = fds.find(s.path); it != fds.end()) {
                fd = it->second;
            } else {
                fd = ::open(s.path.c_str(), O_RDONLY | O_CLOEXEC);
                if (fd < 0) { fail("cannot open " + s.path); break; }
                (void) posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
                fds[s.path] = fd;
            }
            // the slices straight into their places, one forward read (a short read resumed where it stopped)
            iov.resize((size_t) pc.n);
            for (int64_t i = 0; i < pc.n; ++i) iov[(size_t) i] = {place(i), (size_t) slice};
            struct iovec* v = iov.data();
            int left_n = (int) pc.n;
            uint64_t pos = at;
            while (left_n > 0) {
                const ssize_t r = ::preadv(fd, v, left_n, (off_t) pos);
                if (r <= 0) break;
                pos += (uint64_t) r;
                for (size_t left = (size_t) r; left > 0 && left_n > 0;) {
                    if (left >= v->iov_len) {
                        left -= v->iov_len;
                        ++v;
                        --left_n;
                    } else {
                        v->iov_base = (uint8_t*) v->iov_base + left;
                        v->iov_len -= left;
                        left = 0;
                    }
                }
            }
            if (left_n > 0) { fail("a read of " + s.path + " failed"); break; }
            if (uncached) (void) posix_fadvise(fd, (off_t) at, (off_t) n, POSIX_FADV_DONTNEED);
#endif
        }
#if defined(_WIN32)
        for (auto& [p, h] : handles)
            if (h != nullptr) CloseHandle(h);
        for (auto& [p, f] : files)
            if (f != nullptr) std::fclose(f);
        VirtualFree(buf, 0, MEM_RELEASE);
#else
        for (auto& [p, fd] : fds) ::close(fd);
#endif
    };
    std::vector<std::thread> th;
    for (int i = 1; i < std::max(1, threads); ++i) th.emplace_back(worker);
    worker();
    for (auto& t : th) t.join();
    if (seconds) *seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return !bad;
}

}  // namespace lamina::platform
