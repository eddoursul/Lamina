// src/server/pretokenize.hpp - the pre-tokenizers: the pieces of a text that the BPE merges then run on, one at a time.
//
// Each splits a text as llama.cpp's unicode_regex_split does with that pre-tokenizer's expressions (llama-vocab.cpp):
//   qwen35  llama.cpp's own splitter for Qwen3.5's expression, which runs in place of the expression;
//   hy_v4 (Hy4-flash)  three expressions in turn, each over the pieces of the one before: \p{N}{1,3}, then the kana
//       and CJK runs [一-龥぀-ゟ゠-ヿ]+, then words, punctuation and white space;
//   GPT-4o's (Inkling 2: inkling2)  cased words with an English contraction, numbers in threes.
// The last two take llama.cpp's std::regex path. It matches over a "collapsed" text, where each code point above 0x7F
// stands as one byte for its class (a letter 0xD2, a number 0xD1, punctuation 0xD3, a mark 0xD4, a symbol 0xD5,
// White_Space \v, any other 0xD0), so an expression's \p{L} and its like are classes of those bytes and of their
// ASCII members. A match is the one ECMAScript finds at the earliest position (the alternatives in order, quantifiers
// greedy and backtracking), and the text between two matches is a piece of its own. The kana and CJK expression
// matches code points: std::wregex on Windows, whose 16-bit wchar_t cuts a code point above 0xFFFF to its low bits,
// has U+24E00 or U+13041 match it as 0x4E00 or 0x3041 would; here they do not, as in llama.cpp on Linux.
//
// The code point classes are Unicode 15.1's (unicode.hpp), as llama.cpp's; tests/server/pretokenize_parity.cpp checks
// the splits against llama.cpp's unicode_regex_split.
#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace lamina {

enum class PreTokenizer {
    Qwen35,   ///< qwen35 (and qwen2, qwen3): letters with their marks, each digit alone
    HyV4,     ///< hy_v4
    Gpt4o,    ///< inkling2
};

/// the pre-tokenizer of a GGUF's tokenizer.ggml.pre, or nothing for one this server does not run
std::optional<PreTokenizer> pre_tokenizer(std::string_view pre);

/// the pieces of a text, in order, as views of it: together they are the text (a byte that starts no well-formed
/// UTF-8 sequence is a character of its own there, as U+FFFD)
std::vector<std::string_view> pretokenize(PreTokenizer kind, std::string_view text);

/// the text with each byte that starts no well-formed UTF-8 sequence as U+FFFD, as llama.cpp's decoder reads it (a
/// continuation byte, a cut sequence, an overlong form, a value past 0x10FFFF; surrogates are code points), or
/// nothing when it holds none
std::optional<std::string> replace_invalid_utf8(std::string_view text);

}   // namespace lamina
