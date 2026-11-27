// src/core/disk_tier.cpp - see include/lamina/core/disk_tier.hpp.
#include "lamina/core/disk_tier.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <future>

namespace lamina::core {

namespace {
using lamina::platform::kSpillAlign;
using lamina::platform::spill_align_up;

double mib(uint64_t b) { return (double) b / 1048576.0; }

bool failed(const std::string& why) {
    std::fprintf(stderr, "lamina serve: prompt cache: the disk tier: %s\n", why.c_str());
    return false;
}
}   // namespace

DiskTier::~DiskTier() {
    for (int h = 0; h < 2; ++h) {
        if (registered_[h]) cudaHostUnregister(stage_[h]);
        lamina::platform::spill_free(stage_[h]);
    }
}

bool DiskTier::open(const std::string& root, uint64_t budget, uint64_t slot_bytes, std::string& msg) {
    int cleaned = 0;
    if (!dir_.open(root, msg, &cleaned)) return false;
    const uint64_t free = dir_.free_bytes(), room = free > kFloor ? free - kFloor : 0;
    budget_ = std::min(room, budget > 0 ? budget : std::min<uint64_t>(uint64_t(32) << 30, free / 4));
    slot_ = spill_align_up(slot_bytes);
    char b[200];
    if (budget_ < 2 * std::max(slot_, half_)) {
        std::snprintf(b, sizeof b, ": %.0f MiB free on its disk, %.0f MiB of them past the last %.0f MiB", mib(free),
                      mib(room), mib(kFloor));
        msg = "too little room in " + root + b;
        dir_.close();
        return false;
    }
    for (int h = 0; h < 2; ++h) {
        stage_[h] = static_cast<uint8_t*>(lamina::platform::spill_alloc(half_));
        if (stage_[h] == nullptr) {
            msg = "its stage could not be allocated";
            dir_.close();
            return false;
        }
        // pinned for every GPU's copies (portable); pageable where the driver refuses: the copies wait for themselves
        registered_[h] = cudaHostRegister(stage_[h], (size_t) half_, cudaHostRegisterPortable) == cudaSuccess;
        if (!registered_[h]) (void) cudaGetLastError();
    }
    std::snprintf(b, sizeof b, ", up to %.0f MiB (%.0f MiB free on its disk)%s", mib(budget_), mib(free),
                  registered_[0] && registered_[1] ? "" : ", through a pageable stage");
    msg = dir_.path() + b;
    if (cleaned > 0)
        msg += ", " + std::to_string(cleaned) + " folder" + (cleaned == 1 ? "" : "s") + " of ended engines removed";
    return true;
}

uint64_t DiskTier::room() const {
    const uint64_t free = dir_.free_bytes();
    if (free == 0) return UINT64_MAX;   // (unknown: the budget alone)
    return free > kFloor ? free - kFloor : 0;
}

std::unique_ptr<DiskFile> DiskTier::make() {
    auto f = std::make_unique<DiskFile>();
    f->serial = ++serial_;
    f->kv_path = dir_.file(std::to_string(f->serial) + ".kv");
    f->ck_path = dir_.file(std::to_string(f->serial) + ".ckpt");
    return f;
}

void DiskTier::remove(const DiskFile& f) {
    lamina::platform::spill_remove(f.kv_path);
    lamina::platform::spill_remove(f.ck_path);
}

void DiskTier::drop_cells(DiskFile& f) {
    lamina::platform::spill_remove(f.kv_path);
    f.blocks.clear();
    f.cells = 0;
    f.kv_end = f.kv_size = 0;
}

int64_t DiskTier::step(int64_t lo, int64_t hi, int64_t unit, const Bytes& bytes) const {
    if (bytes(lo, hi) <= half_) return std::max<int64_t>(hi - lo, unit);   // (one block)
    // a unit's bytes over a sample of them (the layout's fixed parts spread over it), then fewer until a block fits
    const int64_t probe = std::max<int64_t>(unit, std::min<int64_t>(hi - lo, unit * 1024) / unit * unit);
    const uint64_t per = std::max<uint64_t>(1, bytes(lo, lo + probe) / (uint64_t) (probe / unit));
    int64_t n = std::max<int64_t>(1, (int64_t) ((half_ - (uint64_t(1) << 20)) / per)) * unit;
    while (n > unit && bytes(lo, std::min(hi, lo + n)) > half_) n = std::max(unit, n / unit * 7 / 8 * unit);
    return n;
}

uint64_t DiskTier::size_of(int64_t lo, int64_t hi, int64_t unit, const Bytes& bytes) const {
    if (lo >= hi) return 0;
    const int64_t n = step(lo, hi, unit, bytes);
    uint64_t total = 0;
    for (int64_t c0 = lo; c0 < hi; c0 += n) total += spill_align_up(bytes(c0, std::min(c0 + n, hi)));
    return total;
}

bool DiskTier::append(DiskFile& f, int64_t lo, int64_t hi, int64_t unit, const Bytes& bytes,
                      const std::function<bool(int64_t lo, int64_t hi, uint8_t* host)>& fill) {
    // the blocks from `lo` on hold cells the sequence has rewritten since: the new ones go in their place
    while (!f.blocks.empty() && f.blocks.back().lo >= lo) {
        f.kv_end = f.blocks.back().off;
        f.blocks.pop_back();
    }
    f.cells = std::min(f.cells, lo);
    if (lo >= hi) {
        f.cells = std::max(f.cells, hi);
        return true;
    }
    lamina::platform::SpillFile file;
    std::string err, io_err;
    if (!file.open(f.kv_path, true, err)) return failed(err);
    const int64_t n = step(lo, hi, unit, bytes);
    std::future<bool> io;   // the last block's write, beside the next block's fill
    bool ok = true, filled = true;
    int h = 0;
    for (int64_t c0 = lo; ok && c0 < hi; c0 += n, h ^= 1) {
        const int64_t c1 = std::min(c0 + n, hi);
        const uint64_t b = bytes(c0, c1), padded = spill_align_up(b);
        if (padded > half_) {
            ok = failed("a block of " + std::to_string(c1 - c0) + " cells is past the stage");
            break;
        }
        ok = filled = fill(c0, c1, stage_[h]);   // (half h's last write was waited for a turn ago)
        if (io.valid() && !io.get()) ok = false;
        if (!ok) break;
        const uint64_t off = f.kv_end;
        io = std::async(std::launch::async,
                        [&file, &io_err, p = stage_[h], off, padded] { return file.write(off, p, padded, io_err); });
        f.blocks.push_back({c0, c1, off, b});
        f.kv_end += padded;
    }
    if (io.valid() && !io.get()) ok = false;
    f.kv_size = std::max(f.kv_size, f.kv_end);
    if (!ok) return io_err.empty() ? (filled ? false : failed("a copy from the GPU failed")) : failed(io_err);
    f.cells = hi;
    return true;
}

bool DiskTier::put_cells(DiskFile& f, int64_t lo, int64_t hi, int64_t unit, const Bytes& bytes, const Copy& copy) {
    return append(f, lo, hi, unit, bytes,
                  [&](int64_t c0, int64_t c1, uint8_t* host) { return copy(c0, c1, c1, host, true); });
}

bool DiskTier::put_copy(DiskFile& f, int64_t held, int64_t unit, const Bytes& bytes, const Pack& pack) {
    return append(f, 0, held, unit, bytes, [&](int64_t c0, int64_t c1, uint8_t* host) {
        pack(c0, c1, host);
        return true;
    });
}

bool DiskTier::get_cells(const DiskFile& f, int64_t cells, const Copy& copy) {
    size_t n = 0;
    while (n < f.blocks.size() && f.blocks[n].lo < cells) ++n;
    if (n == 0) return true;
    lamina::platform::SpillFile file;
    std::string err, io_err;
    if (!file.open(f.kv_path, false, err)) return failed(err);
    auto read = [&](size_t i, int h) {
        return file.read(f.blocks[i].off, stage_[h], spill_align_up(f.blocks[i].bytes), io_err);
    };
    bool ok = read(0, 0), copied = true;
    std::future<bool> io;   // the next block's read, beside this one's copy
    for (size_t i = 0; ok && i < n; ++i) {
        const int h = (int) (i % 2);
        if (i + 1 < n) io = std::async(std::launch::async, [&read, i, h] { return read(i + 1, 1 - h); });
        const DiskFile::Block& b = f.blocks[i];
        // each block up to where the next one starts (the newer cells there are the next one's)
        const int64_t end = std::min({b.hi, cells, i + 1 < f.blocks.size() ? f.blocks[i + 1].lo : b.hi});
        if (end > b.lo) ok = copied = copy(b.lo, end, b.hi, stage_[h], false);
        if (io.valid() && !io.get()) ok = false;
    }
    if (io.valid()) (void) io.get();
    if (!ok)
        return copied ? failed(io_err.empty() ? "a block could not be read" : io_err)
                      : failed("a copy to the GPU failed");
    return true;
}

bool DiskTier::put_record(DiskFile& f, int64_t key, const std::vector<Piece>& pieces) {
    uint64_t total = 0;
    for (const Piece& p : pieces) total += p.n;
    if (total > slot_) return failed("a record of " + std::to_string(total) + " bytes is past its slot");
    const bool reuse = !f.free_slots.empty();
    const uint32_t s = reuse ? f.free_slots.back() : f.n_slots;
    const uint64_t at = (uint64_t) s * slot_, len = spill_align_up(total);
    lamina::platform::SpillFile file;
    std::string err;
    if (!file.open(f.ck_path, true, err)) return failed(err);
    bool ok = true;
    if (pieces.size() == 1 && pieces[0].fits && (uintptr_t) pieces[0].p % kSpillAlign == 0) {
        ok = file.write(at, pieces[0].p, len, err);   // (a buffer of whole pages: as it is)
    } else {                                          // through the stage, a half at a time
        size_t pi = 0;
        uint64_t in_piece = 0;
        for (uint64_t w0 = 0; ok && w0 < len; w0 += half_) {
            const uint64_t w = std::min(half_, len - w0);
            uint64_t got = 0;
            while (got < w && pi < pieces.size()) {
                const uint64_t k = std::min(w - got, pieces[pi].n - in_piece);
                std::memcpy(stage_[0] + got, static_cast<const uint8_t*>(pieces[pi].p) + in_piece, (size_t) k);
                got += k;
                in_piece += k;
                if (in_piece == pieces[pi].n) {
                    ++pi;
                    in_piece = 0;
                }
            }
            if (got < w) std::memset(stage_[0] + got, 0, (size_t) (w - got));
            ok = file.write(at + w0, stage_[0], w, err);
        }
    }
    if (!ok) return failed(err);
    f.slots.push_back({key, s});
    if (reuse) f.free_slots.pop_back();
    else ++f.n_slots;
    return true;
}

bool DiskTier::get_record(const DiskFile& f, int64_t key, const std::vector<Into>& pieces) {
    const uint32_t* s = f.slot(key);
    if (s == nullptr) return failed("no record " + std::to_string(key) + " in its file");
    uint64_t total = 0;
    for (const Into& p : pieces) total += p.n;
    const uint64_t at = (uint64_t) *s * slot_, len = spill_align_up(total);
    lamina::platform::SpillFile file;
    std::string err;
    if (!file.open(f.ck_path, false, err)) return failed(err);
    bool ok = true;
    if (pieces.size() == 1 && pieces[0].fits && (uintptr_t) pieces[0].p % kSpillAlign == 0) {
        ok = file.read(at, pieces[0].p, len, err);
    } else {
        size_t pi = 0;
        uint64_t in_piece = 0;
        for (uint64_t w0 = 0; ok && w0 < len; w0 += half_) {
            const uint64_t w = std::min(half_, len - w0);
            ok = file.read(at + w0, stage_[0], w, err);
            uint64_t put = 0;
            while (ok && put < w && pi < pieces.size()) {
                const uint64_t k = std::min(w - put, pieces[pi].n - in_piece);
                std::memcpy(static_cast<uint8_t*>(pieces[pi].p) + in_piece, stage_[0] + put, (size_t) k);
                put += k;
                in_piece += k;
                if (in_piece == pieces[pi].n) {
                    ++pi;
                    in_piece = 0;
                }
            }
        }
    }
    return ok || failed(err);
}

}   // namespace lamina::core
