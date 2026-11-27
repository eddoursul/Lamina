#include "unicode.hpp"

#include <algorithm>
#include <bit>
#include <iterator>
#include <vector>

namespace lamina {
namespace {

struct CategoryRun {
    uint32_t first;
    uint16_t cls;
};
struct CaseMapping {
    uint32_t cp, to;
};
#include "unicode_tables.inc"

constexpr uint32_t kCodePoints = 0x110000;

/// each code point's classes in a byte: its category's bit (0-7), and 8 for White_Space
const uint8_t* class_bytes() {
    static const std::vector<uint8_t> table = [] {
        std::vector<uint8_t> t(kCodePoints);
        const size_t n = std::size(kCategoryRuns);
        for (size_t i = 0; i < n; ++i) {
            const uint32_t end = i + 1 < n ? kCategoryRuns[i + 1].first : kCodePoints;
            std::fill(t.begin() + kCategoryRuns[i].first, t.begin() + end,
                      (uint8_t) std::countr_zero((unsigned) kCategoryRuns[i].cls));
        }
        for (const uint32_t cp : kWhiteSpace) t[cp] |= 8;
        return t;
    }();
    return table.data();
}

template <size_t N> uint32_t mapped(const CaseMapping (&map)[N], uint32_t cp) {
    const auto it =
        std::lower_bound(std::begin(map), std::end(map), cp, [](const CaseMapping& m, uint32_t c) { return m.cp < c; });
    return it != std::end(map) && it->cp == cp ? it->to : cp;
}

}   // namespace

uint16_t cp_class(uint32_t cp) {
    if (cp >= kCodePoints) return kCpUndefined;
    const uint8_t v = class_bytes()[cp];
    return (uint16_t) ((1u << (v & 7)) | ((v & 8u) << 5));
}

uint32_t cp_lower(uint32_t cp) { return mapped(kLowerCase, cp); }

uint32_t cp_upper(uint32_t cp) { return mapped(kUpperCase, cp); }

}   // namespace lamina
