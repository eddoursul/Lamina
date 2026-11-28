// tests/core/pack_embed_test.cpp - the native pack takes a token embedding of every type the engine's embedding reads
// (NativeEmbed::load and model_check: iq_supported) and refuses the others as types without kernels: a one-layer
// model of two Q8_0 experts written for each type (synthetic GGUFs, CPU only).
#include "gguf_fixture.hpp"
#include "lamina/core/pack_image.hpp"
#include "lamina/kernels/iq_kernels.hpp"

#include <cstdio>
#include <filesystem>
#include <string>

namespace {

/// The pack's refusal of a model whose token_embd.weight is of ggml type `type` ("" when it builds)
std::string pack_error(const std::filesystem::path& dir, uint32_t type) {
    const std::filesystem::path p = dir / ("embd-" + std::to_string(type) + ".gguf");
    fixture::Tensor embd;
    embd.name = "token_embd.weight";
    embd.shape = {256, 4};
    embd.type = type;
    std::vector<fixture::Tensor> ts{embd};
    const struct {
        const char* name;
        std::vector<uint64_t> shape;
        uint32_t type;
    } rest[] = {{"blk.0.ffn_gate_inp.weight", {256, 2}, 0},
                {"blk.0.ffn_gate_exps.weight", {256, 32, 2}, 8},
                {"blk.0.ffn_up_exps.weight", {256, 32, 2}, 8},
                {"blk.0.ffn_down_exps.weight", {32, 256, 2}, 8}};
    for (const auto& r : rest) {
        fixture::Tensor t;
        t.name = r.name;
        t.shape = r.shape;
        t.type = r.type;
        ts.push_back(t);
    }
    fixture::write(p, {}, ts);
    lamina::core::PackImage img;
    std::string err;
    if (lamina::core::PackImage::build(p.string(), img, err)) return {};
    return err.empty() ? "(refused without a reason)" : err;
}

}   // namespace

int main() {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "lamina_pack_embed_test";
    std::filesystem::create_directories(dir);
    int bad = 0;
    // the quantized types NativeEmbed dequantizes (F32, F16 and BF16 are the pack's float tensors): Q5_0 6, Q5_1 7,
    // Q8_0 8, Q3_K 11, Q4_K 12, Q5_K 13, Q6_K 14, the i-quants 16-23 and 29, MXFP4 39, NVFP4 40, Q2_0 42, TQP5-7
    // 147-149
    for (const uint32_t t : {6u,  7u,  8u,  11u, 12u, 13u, 14u, 16u, 17u,  18u,  19u,
                             20u, 21u, 22u, 23u, 29u, 39u, 40u, 42u, 147u, 148u, 149u}) {
        if (!lamina::kernels::iq_supported((int) t)) {
            std::printf("FAIL: type %u is not one the embedding reads\n", t);
            ++bad;
            continue;
        }
        const std::string e = pack_error(dir, t);
        if (e.find("token_embd") != std::string::npos) {
            std::printf("FAIL: type %u refused: %s\n", t, e.c_str());
            ++bad;
        }
    }
    // types without a dequantizer for the embedding: Q4_0 2, Q4_1 3
    for (const uint32_t t : {2u, 3u}) {
        const std::string e = pack_error(dir, t);
        if (e.find("token_embd.weight is") == std::string::npos || e.find("no kernels") == std::string::npos) {
            std::printf("FAIL: type %u not refused as a type without kernels: %s\n", t, e.c_str());
            ++bad;
        }
    }
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::printf(bad == 0 ? "pack_embed_test: ok\n" : "pack_embed_test: %d failed\n", bad);
    return bad == 0 ? 0 : 1;
}
