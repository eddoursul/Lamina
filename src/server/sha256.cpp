// src/server/sha256.cpp - SHA-256 (FIPS 180-4), see the header.
#include "sha256.hpp"

#include <cstdint>
#include <cstring>

namespace lamina {
namespace {

constexpr uint32_t kRound[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

/// a round with the state's names rotated by the caller instead of its values moved: d becomes the next e, h the next a
inline void step(uint32_t a, uint32_t b, uint32_t c, uint32_t& d, uint32_t e, uint32_t f, uint32_t g, uint32_t& h,
                 uint32_t kw) {
    const uint32_t t1 = h + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + (g ^ (e & (f ^ g))) + kw;
    d += t1;
    h = t1 + (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) | (c & (a | b)));
}

/// one 64-byte block into the state
void compress(uint32_t st[8], const unsigned char* p) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
        w[i] = (uint32_t) p[4 * i] << 24 | (uint32_t) p[4 * i + 1] << 16 | (uint32_t) p[4 * i + 2] << 8 |
               (uint32_t) p[4 * i + 3];
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7];
    for (int i = 0; i < 64; i += 8) {
        step(a, b, c, d, e, f, g, h, kRound[i] + w[i]);
        step(h, a, b, c, d, e, f, g, kRound[i + 1] + w[i + 1]);
        step(g, h, a, b, c, d, e, f, kRound[i + 2] + w[i + 2]);
        step(f, g, h, a, b, c, d, e, kRound[i + 3] + w[i + 3]);
        step(e, f, g, h, a, b, c, d, kRound[i + 4] + w[i + 4]);
        step(d, e, f, g, h, a, b, c, kRound[i + 5] + w[i + 5]);
        step(c, d, e, f, g, h, a, b, kRound[i + 6] + w[i + 6]);
        step(b, c, d, e, f, g, h, a, kRound[i + 7] + w[i + 7]);
    }
    st[0] += a;
    st[1] += b;
    st[2] += c;
    st[3] += d;
    st[4] += e;
    st[5] += f;
    st[6] += g;
    st[7] += h;
}

}   // namespace

std::string sha256_hex(std::string_view data) {
    uint32_t st[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    const auto* p = reinterpret_cast<const unsigned char*>(data.data());
    const size_t n = data.size();
    size_t i = 0;
    for (; n - i >= 64; i += 64) compress(st, p + i);
    // what is left, the 0x80 byte, zeros and the length in bits (big-endian): one block, or two from 56 bytes left on
    unsigned char tail[128] = {};
    const size_t rest = n - i;
    if (rest > 0) std::memcpy(tail, p + i, rest);
    tail[rest] = 0x80;
    const size_t end = rest < 56 ? 64 : 128;
    const uint64_t bits = (uint64_t) n * 8;
    for (int k = 0; k < 8; ++k) tail[end - 1 - k] = (unsigned char) (bits >> (8 * k));
    compress(st, tail);
    if (end == 128) compress(st, tail + 64);
    static const char hex[] = "0123456789abcdef";
    std::string out(64, '0');
    for (int k = 0; k < 32; ++k) {
        const unsigned v = (st[k / 4] >> (24 - 8 * (k % 4))) & 0xff;
        out[2 * k] = hex[v >> 4];
        out[2 * k + 1] = hex[v & 15];
    }
    return out;
}

}   // namespace lamina
