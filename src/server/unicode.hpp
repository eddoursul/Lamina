// src/server/unicode.hpp - code point classes and case mappings: Unicode 15.1's tables (unicode_tables.inc, written by
// tools/unicode_tables.py), the version of the tables llama.cpp's tokenizer splits text with.
#pragma once

#include <cstdint>

namespace lamina {

/// a code point's classes: its general category, one of the first eight bits, and kCpWhitespace for White_Space
enum : uint16_t {
    kCpUndefined = 0x001,     ///< not in the database (Cn), or above 0x10FFFF
    kCpNumber = 0x002,        ///< \p{N}
    kCpLetter = 0x004,        ///< \p{L}
    kCpSeparator = 0x008,     ///< \p{Z}
    kCpMark = 0x010,          ///< \p{M}
    kCpPunctuation = 0x020,   ///< \p{P}
    kCpSymbol = 0x040,        ///< \p{S}
    kCpControl = 0x080,       ///< Cc, Cf, Co and Cs
    kCpWhitespace = 0x100,    ///< White_Space
};

uint16_t cp_class(uint32_t cp);
/// the simple case mappings: a code point without one maps to itself
uint32_t cp_lower(uint32_t cp);
uint32_t cp_upper(uint32_t cp);

}   // namespace lamina
