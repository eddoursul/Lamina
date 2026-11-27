// tests/server/tokenizer_test.cpp - the tokenizer on a synthetic vocabulary (CPU only, no model): the 256 byte tokens
// (ids the bytes), "ab" (the one merge, a b) and "abc" (no merge makes it). Under inkling2 a piece that is a token is
// that token, as llama.cpp's ignore_merges has it; under qwen35 the merges' tokens.
#include "tokenizer.hpp"

#include "../core/gguf_fixture.hpp"

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(const std::string& what, const std::vector<int32_t>& got, const std::vector<int32_t>& want) {
    if (got == want) return;
    std::string g, w;
    for (const int32_t x : got) g += std::to_string(x) + " ";
    for (const int32_t x : want) w += std::to_string(x) + " ";
    std::printf("%s: %s, expected %s\n", what.c_str(), g.c_str(), w.c_str());
    ++failures;
}

}   // namespace

int main() {
    // the GPT-2 byte alphabet: bytes 33..126, 161..172 and 174..255 as themselves, the others as 256 + n (below 0x800)
    std::vector<std::string> tokens;
    for (uint32_t b = 0, n = 0; b < 256; ++b) {
        const bool direct = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || b >= 0xAE;
        const uint32_t c = direct ? b : 256 + n++;
        tokens.push_back(c < 0x80 ? std::string(1, (char) c)
                                  : std::string{(char) (0xC0 | (c >> 6)), (char) (0x80 | (c & 0x3F))});
    }
    const int32_t ab = 256, abc = 257, a = 'a', c = 'c', d = 'd', space = ' ';
    tokens.push_back("ab");
    tokens.push_back("abc");
    const std::filesystem::path dir = std::filesystem::temp_directory_path();
    for (const char* name : {"inkling2", "qwen35"}) {
        const std::string pre = name;
        const std::filesystem::path path = dir / ("lamina_tokenizer_test_" + pre + ".gguf");
        fixture::write(path,
                       {fixture::str("tokenizer.ggml.model", "gpt2"), fixture::str("tokenizer.ggml.pre", pre),
                        fixture::strs("tokenizer.ggml.tokens", tokens),
                        fixture::strs("tokenizer.ggml.merges", {"a b"})},
                       {});
        const auto tok = lamina::Tokenizer::from_gguf(path.string());
        std::filesystem::remove(path);
        const bool whole = pre == "inkling2";
        expect(pre + " \"ab\"", tok->encode("ab"), {ab});
        expect(pre + " \"abc\"", tok->encode("abc"), whole ? std::vector<int32_t>{abc} : std::vector<int32_t>{ab, c});
        expect(pre + " \"abc\" again", tok->encode("abc"),
               whole ? std::vector<int32_t>{abc} : std::vector<int32_t>{ab, c});
        expect(pre + " \"abcd\"", tok->encode("abcd"), {ab, c, d});
        expect(pre + " \"a abc abc\"", tok->encode("a abc abc"),
               whole ? std::vector<int32_t>{a, space, ab, c, space, ab, c}
                     : std::vector<int32_t>{a, space, ab, c, space, ab, c});
        expect(pre + " \"abc\\nabc\"", tok->encode("abc\nabc"),
               whole ? std::vector<int32_t>{abc, '\n' + 0, abc} : std::vector<int32_t>{ab, c, '\n' + 0, ab, c});
    }
    std::printf(failures == 0 ? "tokenizer_test: ok\n" : "tokenizer_test: %d failed\n", failures);
    return failures == 0 ? 0 : 1;
}
