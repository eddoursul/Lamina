// src/server/sha256.hpp - SHA-256 (FIPS 180-4): the image cache's keys (vision.cpp).
#pragma once

#include <string>
#include <string_view>

namespace lamina {

/// the SHA-256 digest of `data` as 64 lowercase hex digits
std::string sha256_hex(std::string_view data);

}  // namespace lamina
