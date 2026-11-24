// include/lamina/core/disk_tier.hpp - the prompt caches' disk tier (SequenceCache's, Hy4-flash's park): parked
// conversations in files of a folder of this process's own, within a budget, read and written past the file cache
// through a pinned stage of two halves.
//
// A conversation's files: its K/V cells in blocks, each a range of cells laid out as the engine copies that range
// (Copy: the engine's own copy, with what it knows of where its cells are - on several GPUs, on cell cards), and
// records of a fixed size (checkpoints, snapshots) in slots of a second file.  A block is at most a stage half, so it
// is read and written whole, and the next block's copy to or from the GPUs runs beside the last one's file I/O.  A
// conversation that keeps its file while it is live appends a block from where the file stops holding its sequence.
#pragma once

#include "lamina/platform/spill.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace lamina::core {

/// One conversation's files.
struct DiskFile {
    uint64_t serial = 0;
    std::string kv_path, ck_path;
    struct Block {
        int64_t lo, hi;   ///< its cells [lo, hi)
        uint64_t off;     ///< where it starts in the K/V file
        uint64_t bytes;   ///< its layout's bytes
    };
    std::vector<Block> blocks;   ///< lo ascending; a later block's cells over an earlier one's
    int64_t cells = 0;           ///< the blocks hold the sequence's cells [0, cells)
    uint64_t kv_end = 0;         ///< where the next block goes in the K/V file (whole pages)
    uint64_t kv_size = 0;        ///< the K/V file's length
    std::vector<std::pair<int64_t, uint32_t>> slots;   ///< a record's key and its slot
    uint32_t n_slots = 0;
    std::vector<uint32_t> free_slots;   ///< slots of records that went
    const uint32_t* slot(int64_t key) const {
        for (const auto& [k, s] : slots)
            if (k == key) return &s;
        return nullptr;
    }
    /// The record `key` goes: its slot is free for the next one.
    void forget(int64_t key) {
        for (size_t i = 0; i < slots.size(); ++i)
            if (slots[i].first == key) {
                free_slots.push_back(slots[i].second);
                slots.erase(slots.begin() + (std::ptrdiff_t) i);
                return;
            }
    }
    /// Its sequence is rewritten from `at` on: the blocks hold it up to there.
    void cut(int64_t at) { cells = std::min(cells, at); }
};

class DiskTier {
public:
    /// The engine's copy of cells [lo, hi) between its device(s) and `host`, laid out as its copy of cells [lo, held)
    /// (hi <= held; the RAM park's copy is lo = 0).  To the device the ranges come in order, from the first block's.
    using Copy = std::function<bool(int64_t lo, int64_t hi, int64_t held, uint8_t* host, bool to_host)>;
    /// That layout's bytes for cells [lo, held).
    using Bytes = std::function<uint64_t(int64_t lo, int64_t held)>;
    /// Cells [lo, hi) of a copy in RAM into `out`, laid out as Copy lays out [lo, hi).
    using Pack = std::function<void(int64_t lo, int64_t hi, uint8_t* out)>;
    /// A record's pieces, back to back in its slot.  `fits`: the buffer has room for n rounded up to whole pages, so
    /// a page-aligned one alone is read or written as it is; else through the stage.
    struct Piece {
        const void* p;
        uint64_t n;
        bool fits = false;
    };
    struct Into {
        void* p;
        uint64_t n;
        bool fits = false;
    };
    static constexpr uint64_t kFloor = uint64_t(4) << 30;   ///< the disk's free bytes the tier never takes

    DiskTier() = default;
    ~DiskTier();
    DiskTier(const DiskTier&) = delete;
    DiskTier& operator=(const DiskTier&) = delete;

    /// A folder of this process's own under `root` (the ended engines' folders there removed), its budget (0: min(32
    /// GiB, a quarter of the disk's free space)) never into kFloor, records of `slot_bytes`, the stage.  false, with
    /// `msg` saying why; else `msg` says where and how much.
    bool open(const std::string& root, uint64_t budget, uint64_t slot_bytes, std::string& msg);
    uint64_t budget() const { return budget_; }
    uint64_t slot_bytes() const { return slot_; }
    /// A file's bytes on the disk: its K/V file and its records' slots.
    uint64_t bytes(const DiskFile& f) const { return f.kv_size + (uint64_t) f.n_slots * slot_; }
    /// The bytes the disk can still take: its free bytes above kFloor (UINT64_MAX: unknown).
    uint64_t room() const;
    /// A new conversation's files (made when first written).
    std::unique_ptr<DiskFile> make();
    /// Its files go (the object stays the caller's).
    void remove(const DiskFile& f);
    /// Its K/V file goes, its records stay: the next put_cells writes its cells from 0.
    void drop_cells(DiskFile& f);

    /// The bytes put_cells takes in a file for cells [lo, hi): its blocks, each in whole pages.
    uint64_t size_of(int64_t lo, int64_t hi, int64_t unit, const Bytes& bytes) const;
    /// Cells [lo, hi) from the device to the end of `f`, in blocks of whole `unit`s of cells (the last may be short);
    /// `f.cells` is then hi.  `lo` a multiple of `unit`.
    bool put_cells(DiskFile& f, int64_t lo, int64_t hi, int64_t unit, const Bytes& bytes, const Copy& copy);
    /// A copy in RAM of cells [0, held) to the end of `f`, as put_cells lays them out.
    bool put_copy(DiskFile& f, int64_t held, int64_t unit, const Bytes& bytes, const Pack& pack);
    /// The first `cells` cells of `f`'s blocks to the device, in order (each block up to where the next one starts).
    bool get_cells(const DiskFile& f, int64_t cells, const Copy& copy);
    /// A record of `key` from its pieces (at most slot_bytes() in all), in a slot of `f`'s own; and back.
    bool put_record(DiskFile& f, int64_t key, const std::vector<Piece>& pieces);
    bool get_record(const DiskFile& f, int64_t key, const std::vector<Into>& pieces);

private:
    /// Cells per block: whole units whose layout fits a stage half.
    int64_t step(int64_t lo, int64_t hi, int64_t unit, const Bytes& bytes) const;
    bool append(DiskFile& f, int64_t lo, int64_t hi, int64_t unit, const Bytes& bytes,
                const std::function<bool(int64_t lo, int64_t hi, uint8_t* host)>& fill);

    lamina::platform::SpillDir dir_;
    uint64_t budget_ = 0, slot_ = 0, half_ = uint64_t(64) << 20;
    uint8_t* stage_[2] = {nullptr, nullptr};
    bool registered_[2] = {false, false};
    uint64_t serial_ = 0;
};

}  // namespace lamina::core
