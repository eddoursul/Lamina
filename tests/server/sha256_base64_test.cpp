// The server's SHA-256 (src/server/sha256.hpp) against FIPS 180-4's examples (NIST's, one block, two blocks, a million
// 'a's) and the padding's every case (55 bytes left of a block: one block; 56 to 63: two), the digests Python's hashlib
// gives; the base64 encoder (src/server/base64.hpp) against RFC 4648's vectors and bytes above 0x7F. (CPU only.)
#include "base64.hpp"
#include "sha256.hpp"

#include <cstdio>
#include <string>

static int fails = 0;
#define CHECK(c)                                                                  \
    do {                                                                          \
        if (!(c)) {                                                               \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);   \
            ++fails;                                                              \
        }                                                                         \
    } while (0)

int main() {
    using lamina::b64encode;
    using lamina::sha256_hex;

    CHECK(sha256_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK(sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    CHECK(sha256_hex("abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopq"
                     "rstnopqrstu") == "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1");
    CHECK(sha256_hex(std::string(1000000, 'a')) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    const struct {
        size_t n;
        const char* hex;
    } as[] = {
        {55, "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318"},
        {56, "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a"},
        {57, "f13b2d724659eb3bf47f2dd6af1accc87b81f09f59f2b75e5c0bed6589dfe8c6"},
        {63, "7d3e74a05d7db15bce4ad9ec0658ea98e3f06eeecf16b4c6fff2da457ddc2f34"},
        {64, "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb"},
        {65, "635361c48bb9eab14198e76ea8ab7f1a41685d6ad62aa9146d301d4f17eb0ae0"},
        {119, "31eba51c313a5c08226adf18d4a359cfdfd8d2e816b13f4af952f7ea6584dcfb"},
        {120, "2f3d335432c70b580af0e8e1b3674a7c020d683aa5f73aaaedfdc55af904c21c"},
    };
    for (const auto& a : as) CHECK(sha256_hex(std::string(a.n, 'a')) == a.hex);
    std::string bytes;
    for (int b = 0; b < 256; ++b) bytes += (char) b;
    CHECK(sha256_hex(bytes) == "40aff2e9d2d8922e47afd4648e6967497158785fbd1da870e7110266bf944880");

    CHECK(b64encode("").empty());
    CHECK(b64encode("f") == "Zg==");
    CHECK(b64encode("fo") == "Zm8=");
    CHECK(b64encode("foo") == "Zm9v");
    CHECK(b64encode("foob") == "Zm9vYg==");
    CHECK(b64encode("fooba") == "Zm9vYmE=");
    CHECK(b64encode("foobar") == "Zm9vYmFy");
    CHECK(b64encode(std::string(1, '\0')) == "AA==");
    CHECK(b64encode("\xff") == "/w==");
    CHECK(b64encode("\xff\xff") == "//8=");
    CHECK(b64encode("\xfb\xff\xbf") == "+/+/");
    CHECK(b64encode("\xfb\xfc\xfd\xfe\xff") == "+/z9/v8=");
    if (fails) return 1;
    std::printf("sha256_base64_test: ok\n");
    return 0;
}
