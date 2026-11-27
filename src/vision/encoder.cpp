// src/vision/encoder.cpp - which encoder a picture's rows come from (include/lamina/vision/encoder.hpp).
#include "backends.hpp"

#include <cstdint>
#include <cstdio>

namespace lamina::vision {

bool write_rows(const std::string& path, const Rows& rows) {
    std::FILE* f = open_utf8(path, true);
    if (f == nullptr) return false;
    const int32_t header[5] = {0x31455653, rows.n, rows.nx, rows.ny, rows.n_embd};
    const size_t floats = (size_t) rows.n * (size_t) rows.n_embd;
    bool ok = rows.data.size() >= floats && std::fwrite(header, sizeof header, 1, f) == 1 &&
              std::fwrite(rows.data.data(), sizeof(float), floats, f) == floats;
    ok = std::fclose(f) == 0 && ok;
    return ok;
}

uint64_t encoder_gpu_bytes(const EncoderParams& params, std::string& err) { return cuda_encoder_bytes(params, err); }

std::unique_ptr<Encoder> open_encoder(const EncoderParams& params, std::string& err) {
    return params.gpu ? open_cuda_encoder(params, err) : open_ggml_encoder(params, err);
}

}   // namespace lamina::vision
