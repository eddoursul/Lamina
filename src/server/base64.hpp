// src/server/base64.hpp - base64 in RFC 4648's standard alphabet, padded with '=': the generated images' b64_json
// (images.cpp) and the Responses API's encrypted_content (responses.cpp, which reads it back strictly); a request's
// data: URL read as Python's b64decode reads it (vision.cpp, video.cpp).
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace lamina {

inline constexpr std::string_view kB64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

inline std::string b64encode(std::string_view in) {
    std::string out((in.size() + 2) / 3 * 4, '=');
    char* o = out.data();
    for (size_t i = 0; i < in.size(); i += 3, o += 4) {
        const size_t left = in.size() - i;
        const uint32_t v = (uint32_t) (unsigned char) in[i] << 16 |
                           (left > 1 ? (uint32_t) (unsigned char) in[i + 1] << 8 : 0u) |
                           (left > 2 ? (uint32_t) (unsigned char) in[i + 2] : 0u);
        o[0] = kB64[v >> 18];
        o[1] = kB64[(v >> 12) & 63];
        if (left > 1) o[2] = kB64[(v >> 6) & 63];
        if (left > 2) o[3] = kB64[v & 63];
    }
    return out;
}

/// base64 as Python's b64decode reads it (characters outside the alphabet are skipped); std::invalid_argument with its
/// words for what it refuses
inline std::string b64decode(std::string_view s) {
    auto val = [](unsigned char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    std::string out;
    out.reserve(s.size() * 3 / 4);
    uint32_t acc = 0;
    int bits = 0;
    size_t n = 0, pad = 0;
    for (const unsigned char c : s) {
        if (c == '=') {
            ++pad;
            continue;
        }
        const int v = val(c);
        if (v < 0) continue;
        if (pad) throw std::invalid_argument("Discontinuous padding not allowed");
        acc = (acc << 6) | (uint32_t) v;
        bits += 6;
        ++n;
        if (bits >= 8) {
            bits -= 8;
            out.push_back((char) ((acc >> bits) & 0xff));
        }
    }
    if (n % 4 == 1)
        throw std::invalid_argument("Invalid base64-encoded string: number of data characters (" + std::to_string(n) +
                                    ") cannot be 1 more than a multiple of 4");
    if (n % 4 != 0 && (n % 4) + pad < 4) throw std::invalid_argument("Incorrect padding");
    return out;
}

}   // namespace lamina
