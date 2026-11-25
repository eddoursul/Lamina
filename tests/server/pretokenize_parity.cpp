// tests/server/pretokenize_parity.cpp - the server's pre-tokenizers (src/server/pretokenize.hpp) and code point tables
// (src/server/unicode.hpp) against llama.cpp's (src/unicode.cpp of the build's llama.cpp), and cases of their own.
//
//   pretokenize_parity                the cases, every code point's classes and case mappings, and 20,000 random
//                                     texts a pre-tokenizer, split both ways
//   pretokenize_parity --texts N      N random texts a pre-tokenizer (--seed S: another set)
//   pretokenize_parity --files F...   each file whole and line by line, split both ways (a part llama.cpp cannot
//                                     split, its std::regex giving up on a long match, is said and left out)
//   pretokenize_parity --bench F...   each pre-tokenizer's speed over each file, both ways
//
// llama.cpp's side is unicode_regex_split with llama-vocab.cpp's expressions for the pre-tokenizer, below as written
// there, after its decoder. The random texts leave out what the two sides read differently on purpose (the cases hold
// this side's): an overlong UTF-8 form or one past 0x10FFFF, which llama.cpp of this build decodes (its later decoder
// reads U+FFFD, as this side), and, where wchar_t is 16 bits, for hy_v4 a code point above 0xFFFF whose low 16 bits
// are kana or CJK (std::wregex sees those bits; this side, the code point).
#include "pretokenize.hpp"
#include "unicode.hpp"

#include "src/unicode-data.h"   // llama.cpp's
#include "src/unicode.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lamina;

namespace {

// llama-vocab.cpp's expressions for the pre-tokenizers (qwen35: LLAMA_VOCAB_PRE_TYPE_QWEN35; hy_v4: _HY_V4, three
// in turn; inkling2: _GPT4O as llama.cpp writes it since it reads Inkling 2, \p{M} in its lookaheads, which an
// earlier llama.cpp's lacks: the expression is passed to its unicode_regex_split as written here)
const char* kQwen35Regex =
    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s"
    "\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const char* const kHyV4Regex[] = {
    "\\p{N}{1,3}",
    "[一-龥぀-ゟ゠-ヿ]+",
    "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
};
const char* kGpt4oRegex =
    "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))*((?=[\\p{L}\\p{M}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL]"
    "[lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))+((?=[\\p{L}\\p{M}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'"
    "[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

struct Kind {
    PreTokenizer kind;
    const char* name;
    std::vector<std::string> regex;
};

const std::vector<Kind>& kinds() {
    static const std::vector<Kind> k = {
        {PreTokenizer::Qwen35, "qwen35", {kQwen35Regex}},
        {PreTokenizer::HyV4, "hy_v4", {std::begin(kHyV4Regex), std::end(kHyV4Regex)}},
        {PreTokenizer::Gpt4o, "inkling2", {kGpt4oRegex}},
    };
    return k;
}

const Kind& kind_of(PreTokenizer p) {
    for (const Kind& k : kinds())
        if (k.kind == p) return k;
    std::abort();
}

/// this side, as the tokenizer runs it
std::vector<std::string> ours(PreTokenizer kind, const std::string& text) {
    const std::optional<std::string> fixed = replace_invalid_utf8(text);
    std::vector<std::string> out;
    for (const std::string_view v : pretokenize(kind, fixed ? std::string_view(*fixed) : std::string_view(text)))
        out.emplace_back(v);
    return out;
}

/// llama.cpp's side, or nothing when it cannot split the text (MSVC's std::regex gives up on a long match: its
/// error_stack, which llama.cpp throws as "Failed to process regex")
std::optional<std::vector<std::string>> theirs(const Kind& k, const std::string& text) {
    try {
        return unicode_regex_split(text, k.regex, false);
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

std::string shown(const std::string& s) {
    std::string out;
    for (const unsigned char c : s) {
        char b[8];
        if (c >= 0x20 && c < 0x7F && c != '|' && c != '\\') {
            out += (char) c;
        } else {
            std::snprintf(b, sizeof b, "\\x%02X", c);
            out += b;
        }
    }
    return out;
}

std::string shown(const std::vector<std::string>& pieces) {
    std::string out;
    for (const std::string& p : pieces) out += (out.empty() ? "" : "|") + shown(p);
    return out;
}

void put(std::string& s, uint32_t cp) {   // UTF-8, surrogates too
    if (cp < 0x80) {
        s += (char) cp;
    } else if (cp < 0x800) {
        s += (char) (0xC0 | (cp >> 6));
        s += (char) (0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += (char) (0xE0 | (cp >> 12));
        s += (char) (0x80 | ((cp >> 6) & 0x3F));
        s += (char) (0x80 | (cp & 0x3F));
    } else {
        s += (char) (0xF0 | (cp >> 18));
        s += (char) (0x80 | ((cp >> 12) & 0x3F));
        s += (char) (0x80 | ((cp >> 6) & 0x3F));
        s += (char) (0x80 | (cp & 0x3F));
    }
}

// ------------------------------------------------------------------------------------------------------------ cases
struct Case {
    PreTokenizer kind;
    const char* text;
    std::vector<std::string> pieces;
    bool oracle = true;   ///< llama.cpp splits it the same
};

const std::vector<Case>& cases() {
    using P = PreTokenizer;
    static const std::vector<Case> c = {
        // qwen35: contractions in either case, letters with their marks, each digit alone, white space before a word
        {P::Qwen35, "Hello world", {"Hello", " world"}},
        {P::Qwen35, "I'm here, they'RE ok", {"I", "'m", " here", ",", " they", "'RE", " ok"}},
        {P::Qwen35, "a  b", {"a", " ", " b"}},
        {P::Qwen35, "x \n\n y", {"x", " \n\n", " y"}},
        {P::Qwen35, "12345", {"1", "2", "3", "4", "5"}},
        {P::Qwen35, "nai\xCC\x88ve", {"nai\xCC\x88ve"}},
        {P::Qwen35, "end  ", {"end", "  "}},
        {P::Qwen35, "a .", {"a", " ."}},
        {P::Qwen35, "\xE3\x80\x80漢字", {"\xE3\x80\x80漢字"}},
        {P::Qwen35, "\xF0\x9F\x98\x80!\r\nok", {"\xF0\x9F\x98\x80!\r\n", "ok"}},
        {P::Qwen35, "a\xED\xA0\x80" "b", {"a", "\xED\xA0\x80" "b"}},   // a surrogate is a code point, Cs
        {P::Qwen35, "ab\xE4\xB8 cd", {"ab", "\xEF\xBF\xBD\xEF\xBF\xBD", " cd"}},   // a cut sequence: U+FFFD a byte
        // an overlong form and one past 0x10FFFF: U+FFFD a byte (llama.cpp of this build decodes them)
        {P::Qwen35, "a\xC0\x80" "b", {"a", "\xEF\xBF\xBD\xEF\xBF\xBD", "b"}, false},
        {P::Qwen35, "x\xF4\x90\x80\x80", {"x", "\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD"}, false},
        // hy_v4: numbers in threes cut the text first, then kana and CJK runs; text no expression takes is one piece
        {P::HyV4, "abc 1234567 x", {"abc", " ", "123", "456", "7", " x"}},
        {P::HyV4, "你好世界hello", {"你好世界", "hello"}},
        {P::HyV4, "カ・カ", {"カ", "・", "カ"}},
        {P::HyV4, ".NET is", {".NET", " is"}},
        {P::HyV4, "ok!!\n\nnext", {"ok", "!!\n\n", "next"}},
        {P::HyV4, "a   b", {"a", "  ", " b"}},
        {P::HyV4, "\xE2\x80\x8B\xE2\x80\x8B", {"\xE2\x80\x8B\xE2\x80\x8B"}},
        // U+25000 is not CJK here (std::wregex with a 16-bit wchar_t sees 0x5000)
        {P::HyV4, "中\xF0\xA5\x80\x80文", {"中", "\xF0\xA5\x80\x80", "文"}, sizeof(wchar_t) > 2},
        // inkling2 (GPT-4o): cased words, a contraction kept, numbers in threes; above 0x7F a letter is either case
        {P::Gpt4o, "HelloWorld", {"Hello", "World"}},
        {P::Gpt4o, "HELLOworld", {"HELLOworld"}},
        {P::Gpt4o, "HELLO world", {"HELLO", " world"}},
        {P::Gpt4o, "don't stop", {"don't", " stop"}},
        {P::Gpt4o, "1234567", {"123", "456", "7"}},
        {P::Gpt4o, "x!\n/", {"x", "!\n/"}},
        {P::Gpt4o, "\xCC\x81" "A", {"\xCC\x81", "A"}},
        {P::Gpt4o, "ÀB", {"À", "B"}},
        {P::Gpt4o, "a  b", {"a", " ", " b"}},
    };
    return c;
}

int check_cases() {
    int bad = 0;
    for (const Case& c : cases()) {
        const std::vector<std::string> got = ours(c.kind, c.text);
        if (got != c.pieces) {
            std::printf("case %s \"%s\": %s, expected %s\n", kind_of(c.kind).name, shown(c.text).c_str(),
                        shown(got).c_str(), shown(c.pieces).c_str());
            ++bad;
        }
        if (c.oracle) {
            const std::optional<std::vector<std::string>> ref = theirs(kind_of(c.kind), c.text);
            if (ref != c.pieces) {
                std::printf("case %s \"%s\": llama.cpp %s, expected %s\n", kind_of(c.kind).name, shown(c.text).c_str(),
                            ref ? shown(*ref).c_str() : "(fails)", shown(c.pieces).c_str());
                ++bad;
            }
        }
    }
    std::printf("cases: %zu, %d wrong\n", cases().size(), bad);
    return bad;
}

// ----------------------------------------------------------------------------------------------------------- tables
int check_tables() {
    int bad = 0;
    auto their_upper = [](uint32_t cp) {
        auto by_first = [](const std::pair<uint32_t, uint32_t>& p, uint32_t c) { return p.first < c; };
        const auto it = std::lower_bound(unicode_map_uppercase.begin(), unicode_map_uppercase.end(), cp, by_first);
        return it != unicode_map_uppercase.end() && it->first == cp ? it->second : cp;
    };
    for (uint32_t cp = 0; cp <= 0x110000; ++cp) {
        const unicode_cpt_flags f = unicode_cpt_flags_from_cpt(cp);
        const uint16_t want = (uint16_t) (f.category_flag() | (f.is_whitespace ? kCpWhitespace : 0));
        const uint16_t got = cp_class(cp);
        const uint32_t lo = unicode_tolower(cp), up = their_upper(cp);
        if (got != want || cp_lower(cp) != lo || cp_upper(cp) != up) {
            if (++bad <= 10)
                std::printf("U+%04X: classes 0x%03X (llama.cpp 0x%03X), lower U+%04X (U+%04X), upper U+%04X (U+%04X)\n",
                            cp, got, want, cp_lower(cp), lo, cp_upper(cp), up);
        }
    }
    std::printf("tables: 0x110001 code points, %d differ\n", bad);
    return bad;
}

// ----------------------------------------------------------------------------------------------------- random texts
// code points by what they are to the expressions
constexpr uint32_t kUpper[] = {'A', 'B', 'M', 'S', 'Z', 0xC0, 0xC9, 0x391, 0x3A9, 0x410, 0x42F, 0x1C5, 0x10400,
                               0x1E900};
constexpr uint32_t kLower[] = {'a', 'd', 'e', 'l', 'm', 'r', 's', 't', 'v', 'z', 0xDF, 0xE9, 0xFF, 0x17F, 0x3B1,
                               0x3C9, 0x430, 0x44F, 0x10428};
constexpr uint32_t kLetters[] = {0x2B0, 0x5D0, 0x627, 0x915, 0xE01, 0xAC00, 0x3400, 0x4E00, 0x4E2D, 0x6F22, 0x9FA5,
                                 0x9FA6, 0x9FFF, 0x3042, 0x3093, 0x309D, 0x30AB, 0x30FC, 0x30FF, 0xF900, 0x20000,
                                 0x2A700, 0x1D400};
constexpr uint32_t kMarks[] = {0x300, 0x301, 0x308, 0x93F, 0x94D, 0xE31, 0x20DD, 0x3099, 0xFE0F, 0x1D167};
constexpr uint32_t kNumbers[] = {'0', '1', '5', '9', 0xB2, 0xBD, 0x660, 0x966, 0x2160, 0x2460, 0x3007, 0xFF11, 0x1D7CE};
constexpr uint32_t kPunct[] = {'!', '"', '#', '%', '&', '\'', '(', ')', '*', ',', '-', '.', '/', ':', ';', '?', '@',
                               '[', '\\', ']', '_', '{', '}', 0xA1, 0xAB, 0xBB, 0xBF, 0x2014, 0x2019, 0x201C, 0x2026,
                               0x3001, 0x3002, 0x30A0, 0x30FB, 0xFF01};
constexpr uint32_t kSymbols[] = {'$', '+', '<', '=', '>', '^', '`', '|', '~', 0xA9, 0xB0, 0x20AC, 0x2190, 0x2211,
                                 0x2605, 0x309B, 0xFFFD, 0x1F3FD, 0x1F44D, 0x1F600};
constexpr uint32_t kSpaces[] = {' ', ' ', ' ', ' ', '\t', '\n', '\n', '\r', 0x0B, 0x0C, 0x85, 0xA0, 0x1680, 0x2000,
                                0x2009, 0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000};
constexpr uint32_t kOthers[] = {0x00, 0x01, 0x07, 0x1B, 0x1C, 0x1F, 0x7F, 0x80, 0x9F, 0xAD, 0x200B, 0x200C, 0x200D,
                                0x2060, 0xFEFF, 0x180E, 0xD800, 0xDFFF, 0xE000, 0xF8FF, 0x378, 0x3040, 0xFFFE,
                                0xE0001, 0x10FFFF};
// above 0xFFFF with kana or CJK in their low 16 bits (Egyptian, Tangut, CJK extension B, private use)
constexpr uint32_t kCut[] = {0x13041, 0x17000, 0x24E00, 0x25000, 0x29FA5, 0xF4E01};
// bytes both decoders read alike (no overlong form can come of them with what follows)
const char* const kInvalid[] = {"\x80", "\xBF", "\xF8", "\xFF", "\xC3", "\xE4\xB8", "\xF0\x9F\x98", "\xED\xA0"};
const char* const kWords[] = {"'s", "'S", "'t", "'re", "'Re", "'rE", "'ve", "'VE", "'m", "'ll", "'lL", "'d", "'x",
                              "'", "\r\n", "\n\n", " \n", "\n ", "\n/", "!/", "McDonald", "iPhone", "HTTPServer",
                              "ÀB", "Àb", "aÀ", ".NET", "-x", "1,234.5", "don't", "I'M"};

struct Texts {
    std::mt19937_64 rng;
    bool cut_ok;   ///< kCut may be used
    size_t pick(size_t n) { return std::uniform_int_distribution<size_t>(0, n - 1)(rng); }
    template <size_t N>
    uint32_t from(const uint32_t (&a)[N]) {
        return a[pick(N)];
    }
    std::string next() {
        std::string s;
        const size_t atoms = 1 + pick(pick(5) == 0 ? 40 : 16);
        for (size_t i = 0; i < atoms; ++i) {
            const size_t run = 1 + pick(pick(4) == 0 ? 9 : 3);
            const size_t what = pick(30);
            for (size_t k = 0; k < run; ++k) {
                if (what < 5) put(s, from(kLower));
                else if (what < 8) put(s, from(kUpper));
                else if (what < 11) put(s, from(kLetters));
                else if (what < 12) put(s, from(kMarks));
                else if (what < 15) put(s, from(kNumbers));
                else if (what < 18) put(s, from(kPunct));
                else if (what < 20) put(s, from(kSymbols));
                else if (what < 24) put(s, from(kSpaces));
                else if (what < 25) put(s, from(kOthers));
                else if (what < 26) put(s, cut_ok ? from(kCut) : from(kLetters));
                else if (what < 27) s += kInvalid[pick(std::size(kInvalid))];
                else s += kWords[pick(std::size(kWords))];
            }
        }
        return s;
    }
};

int check_texts(size_t n, uint64_t seed) {
    int bad = 0;
    for (const Kind& k : kinds()) {
        std::seed_seq seq{(uint32_t) seed, (uint32_t) (seed >> 32), (uint32_t) k.kind};   // a stream per seed and kind
        Texts gen{std::mt19937_64(seq), k.kind != PreTokenizer::HyV4 || sizeof(wchar_t) > 2};
        size_t differ = 0, failed = 0, pieces = 0, bytes = 0;
        for (size_t i = 0; i < n; ++i) {
            const std::string text = gen.next();
            const std::vector<std::string> a = ours(k.kind, text);
            const std::optional<std::vector<std::string>> b = theirs(k, text);
            bytes += text.size();
            if (!b) {
                ++failed;
                continue;
            }
            pieces += b->size();
            if (a != *b && ++differ <= 10)
                std::printf("%s \"%s\":\n  here      %s\n  llama.cpp %s\n", k.name, shown(text).c_str(),
                            shown(a).c_str(), shown(*b).c_str());
        }
        std::printf("%s: %zu random texts (%zu bytes, %zu pieces), %zu differ, %zu llama.cpp cannot split\n", k.name, n,
                    bytes, pieces, differ, failed);
        bad += (int) std::min<size_t>(differ + failed, 1000000);
    }
    return bad;
}

// ------------------------------------------------------------------------------------------------------------ files
std::string read_file(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::printf("cannot read %s\n", path);
        std::exit(2);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int check_files(const std::vector<const char*>& paths) {
    int bad = 0;
    for (const char* path : paths) {
        const std::string text = read_file(path);
        std::vector<std::string> parts{text};
        for (size_t a = 0, b; a < text.size(); a = b + 1) {
            b = text.find('\n', a);
            if (b == std::string::npos) b = text.size();
            parts.push_back(text.substr(a, b - a + (b < text.size())));
        }
        for (const Kind& k : kinds()) {
            size_t differ = 0, failed = 0;
            for (size_t i = 0; i < parts.size(); ++i) {
                const std::optional<std::vector<std::string>> b = theirs(k, parts[i]);
                if (!b) {   // what the comparison cannot cover: said, not counted against this side
                    if (++failed <= 3)
                        std::printf("%s %s: llama.cpp cannot split %s\n", path, k.name,
                                    i == 0 ? "the whole file" : ("line " + std::to_string(i)).c_str());
                    continue;
                }
                if (ours(k.kind, parts[i]) != *b && ++differ <= 3)
                    std::printf("%s %s: \"%s\" differs\n", path, k.name, shown(parts[i].substr(0, 200)).c_str());
            }
            std::printf("%s %s: %zu bytes, %zu lines, %zu differ, %zu llama.cpp cannot split\n", path, k.name,
                        text.size(), parts.size() - 1, differ, failed);
            bad += (int) std::min<size_t>(differ, 1000000);
        }
    }
    return bad;
}

int bench(const std::vector<const char*>& paths) {
    using clock = std::chrono::steady_clock;
    for (const char* path : paths) {
        const std::string text = read_file(path);
        for (const Kind& k : kinds()) {
            double t[2];
            bool failed = false;
            for (int side = 0; side < 2 && !failed; ++side) {
                size_t reps = 0;
                const auto t0 = clock::now();
                double s = 0;
                do {
                    if (side == 0) {
                        ours(k.kind, text);
                    } else if (!theirs(k, text)) {
                        failed = true;
                        break;
                    }
                    ++reps;
                    s = std::chrono::duration<double>(clock::now() - t0).count();
                } while (s < 1.0);
                t[side] = s / (double) std::max<size_t>(reps, 1);
            }
            if (failed)
                std::printf("%s %s: %zu bytes, here %.2f ms (%.1f MB/s), llama.cpp cannot split it\n", path, k.name,
                            text.size(), t[0] * 1e3, (double) text.size() / t[0] / 1e6);
            else
                std::printf("%s %s: %zu bytes, here %.2f ms (%.1f MB/s), llama.cpp %.2f ms (%.1f MB/s): %.1fx\n", path,
                            k.name, text.size(), t[0] * 1e3, (double) text.size() / t[0] / 1e6, t[1] * 1e3,
                            (double) text.size() / t[1] / 1e6, t[1] / t[0]);
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    size_t n = 20000;
    uint64_t seed = 20261125;
    std::vector<const char*> files, benches;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--texts" && i + 1 < argc) {
            n = std::strtoull(argv[++i], nullptr, 10);
        } else if (a == "--seed" && i + 1 < argc) {
            seed = std::strtoull(argv[++i], nullptr, 10);
        } else if (a == "--files") {
            while (i + 1 < argc && argv[i + 1][0] != '-') files.push_back(argv[++i]);
        } else if (a == "--bench") {
            while (i + 1 < argc && argv[i + 1][0] != '-') benches.push_back(argv[++i]);
        } else {
            std::printf("usage: pretokenize_parity [--texts N] [--seed S] [--files F...] [--bench F...]\n");
            return 2;
        }
    }
    if (!benches.empty()) return bench(benches);
    int bad = 0;
    if (!files.empty()) {
        bad += check_files(files);
    } else {
        bad += check_cases();
        bad += check_tables();
        bad += check_texts(n, seed);
    }
    std::printf(bad == 0 ? "OK\n" : "FAILED\n");
    return bad == 0 ? 0 : 1;
}
