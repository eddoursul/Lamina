#include "pretokenize.hpp"

#include "unicode.hpp"

#include <array>
#include <cstdint>

namespace lamina {
namespace {

constexpr uint32_t kReplacement = 0xFFFD;

/// the code point at s[i] and its length, as llama.cpp's decoder reads it: a byte that starts no well-formed sequence
/// is U+FFFD of one byte
uint32_t decode_at(std::string_view s, size_t i, size_t& n) {
    const uint32_t b = (unsigned char) s[i];
    n = 1;
    if (b < 0x80) return b;
    size_t len = 0;
    uint32_t c = 0, least = 0;   // the lead's bits; the smallest value of the length (below it: an overlong form)
    if (b >= 0xC0 && b < 0xE0) {
        len = 2; c = b & 0x1F; least = 0x80;
    } else if (b >= 0xE0 && b < 0xF0) {
        len = 3; c = b & 0x0F; least = 0x800;
    } else if (b >= 0xF0 && b < 0xF8) {
        len = 4; c = b & 0x07; least = 0x10000;
    } else {
        return kReplacement;   // a continuation byte, or 0xF8-0xFF
    }
    for (size_t k = 1; k < len; ++k) {
        if (i + k >= s.size() || ((unsigned char) s[i + k] & 0xC0) != 0x80) return kReplacement;
        c = (c << 6) | ((unsigned char) s[i + k] & 0x3Fu);
    }
    if (c < least || c > 0x10FFFF) return kReplacement;
    n = len;
    return c;
}

// the classes the splits test: \s (White_Space: ASCII's tab to carriage return and the space, which the collapsed
// text's \s matches, and the rest, there \v), the categories (ASCII's own are the collapsed text's classes too), and
// ASCII's upper and lower case letters
enum : uint16_t {
    kWs = 0x001,
    kLet = 0x002,
    kNum = 0x004,
    kMrk = 0x008,
    kPun = 0x010,
    kSym = 0x020,
    kUp = 0x040,   ///< A-Z
    kLo = 0x080,   ///< a-z
};

uint16_t classes(uint32_t c) {
    const uint16_t u = cp_class(c);
    return (uint16_t) ((u & kCpWhitespace ? kWs : 0) | (u & kCpLetter ? kLet : 0) | (u & kCpNumber ? kNum : 0) |
                       (u & kCpMark ? kMrk : 0) | (u & kCpPunctuation ? kPun : 0) | (u & kCpSymbol ? kSym : 0) |
                       (c >= 'A' && c <= 'Z' ? kUp : 0) | (c >= 'a' && c <= 'z' ? kLo : 0));
}

/// a text as code points: each one's value, classes and byte offset (then the text's size)
struct Text {
    std::vector<uint32_t> cp;
    std::vector<uint16_t> cls;
    std::vector<size_t> at;
    size_t size() const { return cp.size(); }
    bool nl(size_t i) const { return cp[i] == '\r' || cp[i] == '\n'; }
};

Text decode(std::string_view s) {
    static const auto ascii = [] {
        std::array<uint16_t, 128> a{};
        for (uint32_t c = 0; c < 128; ++c) a[c] = classes(c);
        return a;
    }();
    Text t;
    t.cp.reserve(s.size());
    t.cls.reserve(s.size());
    t.at.reserve(s.size() + 1);
    for (size_t i = 0, n = 1; i < s.size(); i += n) {
        const uint32_t c = decode_at(s, i, n);
        t.cp.push_back(c);
        t.cls.push_back(c < 128 ? ascii[c] : classes(c));
        t.at.push_back(i);
    }
    t.at.push_back(s.size());
    return t;
}

/// the pieces' bounds in code points: 0, then each piece's end
using Cuts = std::vector<size_t>;

/// llama.cpp's splitter for Qwen3.5's expression (unicode_regex_split_custom_qwen35), step by step:
/// (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|
/// ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
Cuts split_qwen35(const Text& t) {
    const size_t n = t.size();
    auto at = [&](size_t i) -> uint16_t { return i < n ? t.cls[i] : 0; };
    Cuts cuts{0};
    auto cut = [&](size_t end) {
        if (end > cuts.back()) cuts.push_back(end);
    };
    for (size_t pos = 0; pos < n;) {
        const uint32_t c = t.cp[pos];
        const uint16_t k = t.cls[pos];
        // a contraction, its letters compared in lower case
        if (c == '\'' && pos + 1 < n) {
            const uint32_t a = cp_lower(t.cp[pos + 1]);
            if (a == 's' || a == 't' || a == 'm' || a == 'd') {
                cut(pos += 2);
                continue;
            }
            if (pos + 2 < n) {
                const uint32_t b = cp_lower(t.cp[pos + 2]);
                if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) {
                    cut(pos += 3);
                    continue;
                }
            }
        }
        // letters and marks, after a character that is not a line break, letter or number
        if (!t.nl(pos) && !(k & kNum) && ((k & (kLet | kMrk)) || (at(pos + 1) & (kLet | kMrk)))) {
            ++pos;
            while (at(pos) & (kLet | kMrk)) ++pos;
            cut(pos);
            continue;
        }
        if (k & kNum) {
            cut(++pos);
            continue;
        }
        // a space, then the run of others and the line breaks after it (a space at the end is a piece alone)
        if (!((c == ' ' ? at(pos + 1) : k) & (kWs | kLet | kMrk | kNum))) {
            if (c == ' ') ++pos;
            while (pos < n && !(t.cls[pos] & (kWs | kLet | kMrk | kNum))) ++pos;
            while (pos < n && t.nl(pos)) ++pos;
            cut(pos);
            continue;
        }
        // white space: to its last line break; else all of it but its last before a character, or all of it
        size_t w = pos, past_nl = 0;
        for (; w < n && (t.cls[w] & kWs); ++w)
            if (t.nl(w)) past_nl = w + 1;
        if (past_nl > 0) {
            cut(pos = past_nl);
        } else if (w - pos > 1 && w < n) {
            cut(pos = w - 1);
        } else if (w > pos) {
            cut(pos = w);
        } else {
            cut(++pos);   // nothing took it: the character alone
        }
    }
    return cuts;
}

/// std::regex_iterator's split of llama.cpp's general path (unicode_regex_split_stl), over each piece so far: the
/// matches left to right, each the one at the earliest position, and the text between two matches as a piece.
/// `match(t, p, e)` is the end of the match at p in the piece that ends at e (which the expression cannot see past),
/// or p when none starts there.
template <class Match>
Cuts regex_pass(const Text& t, const Cuts& pieces, Match&& match) {
    Cuts out{0};
    for (size_t i = 1; i < pieces.size(); ++i) {
        const size_t e = pieces[i];
        size_t gap = pieces[i - 1];   // the text since the last match
        for (size_t p = gap; p < e;) {
            const size_t q = match(t, p, e);
            if (q == p) {
                ++p;
                continue;
            }
            if (gap < p) out.push_back(p);
            out.push_back(q);
            gap = p = q;
        }
        if (gap < e) out.push_back(e);
    }
    return out;
}

/// \s*[\r\n]+|\s+(?!\S)|\s+
size_t match_spaces(const Text& t, size_t p, size_t e) {
    size_t w = p, past_nl = p;
    for (; w < e && (t.cls[w] & kWs); ++w)
        if (t.nl(w)) past_nl = w + 1;
    if (past_nl > p) return past_nl;            // to the run's last line break
    if (w < e && w - p > 1) return w - 1;       // \s+(?!\S): all but the last before a character
    return w;                                   // all of it (or nothing)
}

/// hy_v4's first: \p{N}{1,3}
size_t match_hy4_numbers(const Text& t, size_t p, size_t e) {
    if (!(t.cls[p] & kNum)) return p;
    size_t q = p + 1;
    while (q < e && q - p < 3 && (t.cls[q] & kNum)) ++q;
    return q;
}

/// hy_v4's second: [一-龥぀-ゟ゠-ヿ]+
size_t match_hy4_cjk(const Text& t, size_t p, size_t e) {
    auto cjk = [&](size_t i) {
        return (t.cp[i] >= 0x3040 && t.cp[i] <= 0x30FF) || (t.cp[i] >= 0x4E00 && t.cp[i] <= 0x9FA5);
    };
    size_t q = p;
    while (q < e && cjk(q)) ++q;
    return q;
}

/// hy_v4's third:
/// [!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|
/// \s*[\r\n]+|\s+(?!\S)|\s+
size_t match_hy4_words(const Text& t, size_t p, size_t e) {
    auto at = [&](size_t i) -> uint16_t { return i < e ? t.cls[i] : 0; };
    const uint16_t k = t.cls[p];
    if (t.cp[p] < 0x80 && (k & (kPun | kSym)) && (at(p + 1) & (kUp | kLo))) {   // ASCII punctuation, ASCII letters
        size_t q = p + 2;
        while (at(q) & (kUp | kLo)) ++q;
        return q;
    }
    const bool lead = !t.nl(p) && !(k & (kLet | kPun | kSym));
    if ((k & (kLet | kMrk)) || (lead && (at(p + 1) & (kLet | kMrk)))) {   // letters and marks, after a lead
        size_t q = p + 1;
        while (at(q) & (kLet | kMrk)) ++q;
        return q;
    }
    const size_t r = t.cp[p] == ' ' && (at(p + 1) & (kPun | kSym)) ? p + 1 : p;
    if (at(r) & (kPun | kSym)) {   // a space, punctuation and symbols, the line breaks after them
        size_t q = r + 1;
        while (at(q) & (kPun | kSym)) ++q;
        while (q < e && t.nl(q)) ++q;
        return q;
    }
    return match_spaces(t, p, e);
}

/// GPT-4o's:
/// [^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))*((?=[\p{L}\p{M}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|
/// '[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))+((?=[\p{L}\p{M}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|
/// '[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
/// Over the collapsed text a letter above 0x7F is both of its cases (not a-z, not A-Z), as is a mark.
size_t match_gpt4o(const Text& t, size_t p, size_t e) {
    auto at = [&](size_t i) -> uint16_t { return i < e ? t.cls[i] : 0; };
    auto upper = [&](size_t i) { return (at(i) & (kLet | kMrk)) && !(at(i) & kLo); };
    auto lower = [&](size_t i) { return (at(i) & (kLet | kMrk)) && !(at(i) & kUp); };
    auto is = [&](size_t i, uint32_t c) { return i < e && (t.cp[i] == c || t.cp[i] == c - 32); };   // either case
    auto contraction = [&](size_t w) -> size_t {
        if (w >= e || t.cp[w] != '\'') return 0;
        if (is(w + 1, 's') || is(w + 1, 't') || is(w + 1, 'm') || is(w + 1, 'd')) return 2;
        const bool e2 = is(w + 2, 'e');
        if ((is(w + 1, 'r') && e2) || (is(w + 1, 'v') && e2) || (is(w + 1, 'l') && is(w + 2, 'l'))) return 3;
        return 0;
    };
    // upper* lower+ from r: the upper run backtracks to the last place a lower one starts (0: none)
    auto upper_lower = [&](size_t r) -> size_t {
        size_t u = r;
        while (upper(u)) ++u;
        for (size_t s = u + 1; s-- > r;) {
            if (!lower(s)) continue;
            size_t w = s + 1;
            while (lower(w)) ++w;
            return w + contraction(w);
        }
        return 0;
    };
    // upper+ lower* from r (0: none)
    auto upper_then = [&](size_t r) -> size_t {
        if (!upper(r)) return 0;
        size_t w = r + 1;
        while (upper(w)) ++w;
        while (lower(w)) ++w;
        return w + contraction(w);
    };
    // the first two alternatives, each with its lead and then without
    const bool lead = !t.nl(p) && !(t.cls[p] & (kLet | kNum));
    size_t q = lead ? upper_lower(p + 1) : 0;
    if (q == 0) q = upper_lower(p);
    if (q == 0 && lead) q = upper_then(p + 1);
    if (q == 0) q = upper_then(p);
    if (q != 0) return q;
    if (t.cls[p] & kNum) {
        q = p + 1;
        while (q < e && q - p < 3 && (t.cls[q] & kNum)) ++q;
        return q;
    }
    auto other = [&](size_t i) { return i < e && !(t.cls[i] & (kWs | kLet | kNum)); };
    const size_t r = t.cp[p] == ' ' && other(p + 1) ? p + 1 : p;
    if (other(r)) {   // a space, a run of others, the line breaks and slashes after it
        q = r + 1;
        while (other(q)) ++q;
        while (q < e && (t.nl(q) || t.cp[q] == '/')) ++q;
        return q;
    }
    return match_spaces(t, p, e);
}

}  // namespace

std::optional<PreTokenizer> pre_tokenizer(std::string_view pre) {
    if (pre == "qwen35" || pre == "qwen2" || pre == "qwen3") return PreTokenizer::Qwen35;
    if (pre == "hy_v4") return PreTokenizer::HyV4;
    if (pre == "inkling2") return PreTokenizer::Gpt4o;
    return std::nullopt;
}

std::vector<std::string_view> pretokenize(PreTokenizer kind, std::string_view text) {
    const Text t = decode(text);
    Cuts cuts;
    switch (kind) {
        case PreTokenizer::Qwen35:
            cuts = split_qwen35(t);
            break;
        case PreTokenizer::HyV4:
            cuts = regex_pass(t, Cuts{0, t.size()}, match_hy4_numbers);
            cuts = regex_pass(t, cuts, match_hy4_cjk);
            cuts = regex_pass(t, cuts, match_hy4_words);
            break;
        case PreTokenizer::Gpt4o:
            cuts = regex_pass(t, Cuts{0, t.size()}, match_gpt4o);
            break;
    }
    std::vector<std::string_view> out;
    out.reserve(cuts.size());
    for (size_t i = 1; i < cuts.size(); ++i)
        out.push_back(text.substr(t.at[cuts[i - 1]], t.at[cuts[i]] - t.at[cuts[i - 1]]));
    return out;
}

std::optional<std::string> replace_invalid_utf8(std::string_view text) {
    size_t i = 0, n = 1;
    while (i < text.size() && !(decode_at(text, i, n) == kReplacement && n == 1)) i += n;
    if (i == text.size()) return std::nullopt;
    std::string out(text.substr(0, i));
    for (; i < text.size(); i += n) {
        if (decode_at(text, i, n) == kReplacement && n == 1) out += "\xEF\xBF\xBD";
        else out.append(text.substr(i, n));
    }
    return out;
}

}  // namespace lamina
