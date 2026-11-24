// tests/core/disk_tier_test.cpp - the prompt caches' disk tier (lamina/core/disk_tier.hpp), CPU only: an engine
// stand-in whose "device" is a host array of two arrays (8 and 3 bytes a cell), its copy laid out as a park's:
//
//   1. a sequence's cells to a file in blocks of whole units (a stage half each), and back: every cell as written;
//   2. a prefix of them back (a lent start, a cut): those cells, none after;
//   3. the file cut where the sequence is rewritten and appended from there: the blocks past the cut give their place
//      to the new one, the file holds the new cells, the old ones before the cut stay;
//   4. a copy in RAM (the park's layout) written as the same blocks, read back as the cells;
//   5. records in slots: pieces back to back, a slot freed and reused, a record past its slot refused;
//   6. the folder and its files go with the tier.
#include "lamina/core/disk_tier.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

using lamina::core::DiskFile;
using lamina::core::DiskTier;

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++g_fail;
}

constexpr int64_t kCells = 40000;
const uint64_t kRow[2] = {8, 3};   // the arrays' bytes a cell

/// The stand-in's "device": each array's cells by position; a cell's bytes say its array, its position and `gen`.
struct Device {
    std::vector<uint8_t> a[2];
    Device() {
        for (int k = 0; k < 2; ++k) a[k].assign((size_t) (kCells * kRow[k]), 0);
    }
    void fill(int64_t lo, int64_t hi, uint8_t gen) {
        for (int k = 0; k < 2; ++k)
            for (int64_t c = lo; c < hi; ++c)
                for (uint64_t b = 0; b < kRow[k]; ++b) a[k][(size_t) (c * kRow[k] + b)] = (uint8_t) (c * 7 + b + k * 13 + gen);
    }
    bool same(const Device& o, int64_t lo, int64_t hi) const {
        for (int k = 0; k < 2; ++k)
            if (std::memcmp(a[k].data() + lo * kRow[k], o.a[k].data() + lo * kRow[k], (size_t) ((hi - lo) * kRow[k])) != 0)
                return false;
        return true;
    }
};

/// The engine's copy of cells [lo, hi) laid out as a copy of [lo, held): an array's cells, then the next one's.
DiskTier::Copy copy_of(Device& d) {
    return [&d](int64_t lo, int64_t hi, int64_t held, uint8_t* host, bool to_host) {
        for (int k = 0; k < 2; ++k) {
            uint8_t* dev = d.a[k].data() + lo * kRow[k];
            if (to_host) std::memcpy(host, dev, (size_t) ((hi - lo) * kRow[k]));
            else std::memcpy(dev, host, (size_t) ((hi - lo) * kRow[k]));
            host += (held - lo) * kRow[k];
        }
        return true;
    };
}

uint64_t bytes_of(int64_t lo, int64_t held) { return (uint64_t) (held - lo) * (kRow[0] + kRow[1]); }

}  // namespace

int main() {
    std::printf("disk_tier_test\n");
    std::error_code ec;
    const std::filesystem::path root = std::filesystem::temp_directory_path(ec) / "lamina-disk-tier-test";
    std::filesystem::remove_all(root, ec);
    std::string dir;
    {
        DiskTier tier;
        std::string msg;
        const std::u8string u = root.u8string();
        check(tier.open(std::string(u.begin(), u.end()), uint64_t(1) << 30, 1000, msg), "the tier opens a folder");
        if (tier.budget() == 0) return 1;
        dir = msg.substr(0, msg.find(','));
        Device dev, back;
        dev.fill(0, kCells, 1);
        const DiskTier::Bytes bytes = bytes_of;
        // 1. the whole sequence in blocks and back
        auto f = tier.make();
        check(tier.put_cells(*f, 0, 30000, 4, bytes, copy_of(dev)) && f->cells == 30000, "30,000 cells written");
        check(f->blocks.size() == 1 && f->kv_size == lamina::platform::spill_align_up(bytes_of(0, 30000)),
              "in one block where a stage half holds them, whole pages");
        check(tier.get_cells(*f, 30000, copy_of(back)) && back.same(dev, 0, 30000), "read back cell for cell");
        // 2. a prefix
        Device pre;
        check(tier.get_cells(*f, 12345, copy_of(pre)) && pre.same(dev, 0, 12345) && !pre.same(dev, 12345, 12346),
              "a prefix: its cells, none after");
        // 3. cut and appended
        f->cut(20000);
        dev.fill(20000, 35000, 2);   // the sequence goes on otherwise from 20,000
        const uint64_t end_before = f->kv_end;
        check(tier.put_cells(*f, 20000, 35000, 4, bytes, copy_of(dev)) && f->cells == 35000 && f->blocks.size() == 2,
              "appended from the cut: a block of the new cells");
        check(f->blocks[1].off == end_before, "after the old block, which holds the cells before the cut");
        Device again;
        check(tier.get_cells(*f, 35000, copy_of(again)) && again.same(dev, 0, 35000), "read back: the new cells over the old");
        f->cut(10000);
        dev.fill(10000, 15000, 3);
        check(tier.put_cells(*f, 10000, 15000, 4, bytes, copy_of(dev)) && f->blocks.size() == 2 &&
                  f->blocks[1].lo == 10000 && f->blocks[1].off == end_before,
              "cut before the second block: it gives its place to the new one");
        Device third;
        check(tier.get_cells(*f, 15000, copy_of(third)) && third.same(dev, 0, 15000), "read back as the sequence is now");
        // 4. a copy in RAM, as the park lays it out
        dev.fill(0, kCells, 4);
        std::vector<uint8_t> ram((size_t) bytes_of(0, 25000));
        (void) copy_of(dev)(0, 25000, 25000, ram.data(), true);
        auto g = tier.make();
        check(tier.put_copy(*g, 25000, 4, bytes,
                            [&](int64_t lo, int64_t hi, uint8_t* out) {
                                for (int k = 0; k < 2; ++k) {
                                    const uint8_t* src = ram.data() + (k == 0 ? 0 : 25000 * kRow[0]) + lo * kRow[k];
                                    std::memcpy(out, src, (size_t) ((hi - lo) * kRow[k]));
                                    out += (hi - lo) * kRow[k];
                                }
                            }),
              "a copy in RAM written in blocks");
        Device fromram;
        check(tier.get_cells(*g, 25000, copy_of(fromram)) && fromram.same(dev, 0, 25000), "and read back as the cells");
        // 5. records
        const std::string s1(700, 'a'), s2(300, 'b');
        check(tier.put_record(*g, 100, {{s1.data(), s1.size()}, {s2.data(), s2.size()}}), "a record of two pieces");
        std::string r1(700, 0), r2(300, 0);
        check(tier.get_record(*g, 100, {{r1.data(), r1.size()}, {r2.data(), r2.size()}}) && r1 == s1 && r2 == s2,
              "read back piece for piece");
        check(tier.put_record(*g, 200, {{s2.data(), s2.size()}}) && g->n_slots == 2, "a second slot");
        g->forget(100);
        check(tier.put_record(*g, 300, {{s1.data(), s1.size()}}) && g->n_slots == 2 && *g->slot(300) == 0,
              "a freed slot taken again");
        const std::string big(5000, 'c');   // (a slot is the record's bytes in whole pages: 4,096)
        check(!tier.put_record(*g, 400, {{big.data(), big.size()}}), "a record past its slot refused");
        tier.remove(*f);
        check(!std::filesystem::exists(std::filesystem::path(std::u8string(
                  reinterpret_cast<const char8_t*>(f->kv_path.data()), f->kv_path.size()))),
              "a file's removal takes its files");
    }
    check(!dir.empty() && !std::filesystem::exists(std::filesystem::path(std::u8string(
                              reinterpret_cast<const char8_t*>(dir.data()), dir.size()))),
          "the folder goes with the tier");
    std::filesystem::remove_all(root, ec);
    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
