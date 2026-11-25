// src/server/base64.hpp - base64 in RFC 4648's standard alphabet, padded with '=': the generated images' b64_json
// (images.cpp) and the Responses API's encrypted_content (responses.cpp, which reads it back strictly).
#pragma once

#include <cstdint>
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

}  // namespace lamina
