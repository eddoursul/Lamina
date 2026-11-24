// src/kernels/cpu/test_blobs.hpp - the CPU expert tests' blobs (expert_parity, pool_test): an expert of upstream's pack
// file, or a synthetic one where the file is not here.
#pragma once

#include "lamina/kernels/cpu/expert.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace lamina::kernels::cpu::test {

/// Expert `index`'s blob of the pack file at `path`; false where it has none
inline bool read_blob(const char* path, long long index, std::vector<uint8_t>& out) {
    out.assign(BLOB, 0);
    std::FILE* f = std::fopen(path, "rb");
    if (!f) return false;
#if defined(_MSC_VER)
    if (_fseeki64(f, index * (long long) BLOB, SEEK_SET) != 0) { std::fclose(f); return false; }
#else
    if (fseeko(f, (off_t) index * (off_t) BLOB, SEEK_SET) != 0) { std::fclose(f); return false; }
#endif
    const size_t got = std::fread(out.data(), 1, out.size(), f);
    std::fclose(f);
    return got == out.size();
}

/// An expert without the pack: random codes with finite scales (fp16 in [2^-7, 2^-6)), another one an index.  The
/// kernels and the pool's protocol are under test, not the model's weights.
inline void synth_blob(long long index, std::vector<uint8_t>& out) {
    out.assign(BLOB, 0);
    std::mt19937 rng((uint32_t) (0x9E3779B9u * (uint32_t) (index + 1)));
    for (auto& v : out) v = (uint8_t) rng();
    for (size_t o = O_GU_SCALES; o + 2 <= BLOB; o += 2) {
        const uint16_t h = (uint16_t) (0x2000 | (rng() & 0x3FF));
        std::memcpy(out.data() + o, &h, 2);
    }
}

}  // namespace lamina::kernels::cpu::test
