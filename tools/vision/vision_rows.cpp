// tools/vision/vision_rows.cpp - the image encoder (include/lamina/vision/encoder.hpp) as a program over picture
// files, for comparing its rows: two encoders, two builds, a GPU's against the CPU's.  lamina runs the same
// encoder in its own process (src/server/vision.cpp); this is no part of a release.
//
//   vision_rows --mmproj <mmproj.gguf> [--gpu [--device N]] [--threads N] [--max-tokens N] [--min-tokens N]
//               [--flash-attn on|off] [--q8 mmq|fp16]
//   --gpu           the engine's kernels on CUDA device N (default 0, of those CUDA_VISIBLE_DEVICES leaves); without
//                   it the CPU's graph
//   --flash-attn    the CPU's attention: off computes it as two products in FP32, the reference a GPU's rows are
//                   measured against
//   --q8            a GPU, a Q8_0 projector: its products through MMQ (the default) or as FP16 products
//   --lent          a GPU: the encoder in memory this program allocates for it first (encoder_gpu_bytes), as the
//                   server lends it expert-cache slots: the same rows, and what a loan must hold
//   --model <file>  taken and not read (the encoder programs before this one named the text model)
//
// It prints "READY <n_embd>", then per stdin line
//   ENC <image path> <output path>   ->  "OK <n_tokens> <nx> <ny> <ms>"  or  "ERR <message>"
//   FRAMES <output path>\t<frame>[\t<frame>]   ->  the same for a video's group of frames (picture files already
//                                             at its plan's size: include/lamina/vision/video.hpp) through
//                                             encode_frames
//   PLAN <frames> <fps> <width> <height>   ->  "OK {json}": the video plan of a stream of that many frames, rate and
//                                             displayed size (frames, width, height, per_group, group_tokens, times)
//   QUIT
// The output file is  int32 {0x31455653 'SVE1', n_tokens, nx, ny, n_embd}  then float32 [n_tokens][n_embd],
// row i at grid position (x = i % nx, y = i / nx): the record the engine reads a picture from.
#include "lamina/vision/encoder.hpp"
#include "lamina/vision/projector.hpp"
#include "lamina/vision/video.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

// "ENC <image> <out>": the image path ends at the last " " before the output path
bool parse_enc(const std::string& line, std::string& img, std::string& out) {
    if (line.rfind("ENC ", 0) != 0) return false;
    const std::string rest = line.substr(4);
    const size_t sp = rest.rfind(' ');
    if (sp == std::string::npos || sp == 0) return false;
    img = rest.substr(0, sp);
    out = rest.substr(sp + 1);
    return !img.empty() && !out.empty();
}

// "FRAMES <out>\t<first>[\t<second>]"
bool parse_frames(const std::string& line, std::string& out, std::vector<std::string>& frames) {
    if (line.rfind("FRAMES ", 0) != 0) return false;
    std::vector<std::string> f;
    size_t at = 7;
    for (size_t tab; (tab = line.find('\t', at)) != std::string::npos; at = tab + 1)
        f.push_back(line.substr(at, tab - at));
    f.push_back(line.substr(at));
    if (f.size() < 2 || f.size() > 3) return false;
    out = f[0];
    frames.assign(f.begin() + 1, f.end());
    return !out.empty();
}

}   // namespace

int main(int argc, char** argv) {
    using lamina::vision::FlashAttention;
    lamina::vision::EncoderParams params;
    bool lent = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--mmproj") params.mmproj = next();
        else if (a == "--model") (void) next();   // not read
        else if (a == "--gpu") params.gpu = true;
        else if (a == "--device") params.device = std::atoi(next().c_str());
        else if (a == "--threads") params.threads = std::atoi(next().c_str());
        else if (a == "--max-tokens") params.max_tokens = std::atoi(next().c_str());
        else if (a == "--min-tokens") params.min_tokens = std::atoi(next().c_str());
        else if (a == "--q8") params.q8_fp16 = next() == "fp16";
        else if (a == "--lent") lent = true;
        else if (a == "--flash-attn")
            params.flash_attention = next() == "off" ? FlashAttention::off : FlashAttention::on;
        else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    if (params.mmproj.empty()) {
        std::fprintf(stderr,
                     "usage: vision_rows --mmproj <mmproj.gguf> [--gpu [--device N]] [--threads N] [--max-tokens N] "
                     "[--min-tokens N] [--flash-attn on|off] [--q8 mmq|fp16]\n");
        return 2;
    }
    std::string err;
    if (lent && params.gpu) {
        params.gpu_arena_bytes = lamina::vision::encoder_gpu_bytes(params, err);
        if (params.gpu_arena_bytes == 0 || cudaSetDevice(params.device) != cudaSuccess ||
            cudaMalloc(&params.gpu_arena, params.gpu_arena_bytes) != cudaSuccess) {
            std::printf("ERR no memory to lend the encoder (%s)\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "vision_rows: %.0f MiB lent to the encoder\n",
                     (double) params.gpu_arena_bytes / 1048576.0);
    }
    const auto load_t0 = std::chrono::steady_clock::now();
    std::unique_ptr<lamina::vision::Encoder> enc = lamina::vision::open_encoder(params, err);
    if (!enc) {
        std::printf("ERR cannot load the image encoder %s (%s)\n", params.mmproj.c_str(), err.c_str());
        std::fflush(stdout);
        return 1;
    }
    std::fprintf(stderr, "vision_rows: the projector loaded in %.1f s\n",
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - load_t0).count());
    // a GPU: a picture at the projector's limit first, as the server's start encodes one, so the first picture's time
    // is a picture's and not the kernels' first launch
    if (params.gpu) {
        const auto warm_t0 = std::chrono::steady_clock::now();
        lamina::vision::Rgb grey;
        grey.w = grey.h = 2048;
        grey.px.assign((size_t) grey.w * grey.h * 3, 128);
        lamina::vision::Rows rows;
        std::string why;
        if (!enc->encode(grey, rows, why)) rows.n = 0;
        std::fprintf(stderr, "vision_rows: warmed up at %d image tokens in %.1f s%s%s\n", rows.n,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - warm_t0).count(),
                     why.empty() ? "" : ": ", why.c_str());
    }
    std::printf("READY %d\n", enc->n_embd());
    std::fflush(stdout);

    std::string line;
    while (std::getline(std::cin, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (line == "QUIT") break;
        if (line.rfind("PLAN ", 0) == 0) {
            lamina::vision::VideoStream vs;
            if (std::sscanf(line.c_str() + 5, "%lld %lf %d %d", (long long*) &vs.frames, &vs.fps, &vs.width,
                            &vs.height) != 4) {
                std::printf("ERR expected: PLAN <frames> <fps> <width> <height>\n");
                std::fflush(stdout);
                continue;
            }
            lamina::vision::Projector header;
            lamina::vision::VideoPlan plan;
            std::string why;
            try {
                header.open(params);
            } catch (const std::exception& e) {
                why = e.what();
            }
            if (why.empty() && lamina::vision::plan_video(header, vs, params.max_tokens, plan, why)) {
                std::string out = "{\"frames\": [";
                for (size_t i = 0; i < plan.frames.size(); ++i)
                    out += (i ? "," : "") + std::to_string((long long) plan.frames[i]);
                out += "], \"width\": " + std::to_string(plan.width) + ", \"height\": " + std::to_string(plan.height) +
                       ", \"per_group\": " + std::to_string(plan.per_group) +
                       ", \"group_tokens\": " + std::to_string(plan.group_tokens) + ", \"times\": [";
                for (size_t i = 0; i < plan.times.size(); ++i)
                    out += std::string(i ? ",\"" : "\"") + lamina::vision::video_time_text(plan.times[i]) + "\"";
                std::printf("OK %s]}\n", out.c_str());
            } else {
                std::printf("ERR %s\n", why.c_str());
            }
            std::fflush(stdout);
            continue;
        }
        std::string img, out;
        std::vector<std::string> frames;
        const bool video = parse_frames(line, out, frames);
        if (!video && !parse_enc(line, img, out)) {
            std::printf("ERR expected: ENC <image> <output>, or FRAMES <output>\t<frame>[\t<frame>]\n");
            std::fflush(stdout);
            continue;
        }
        if (!video) frames = {img};
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<lamina::vision::Rgb> pictures(frames.size());
        lamina::vision::Rows rows;
        err.clear();
        bool read = true;
        for (size_t f = 0; f < frames.size() && read; ++f)
            if (!lamina::vision::decode_file(frames[f], pictures[f], err)) {
                std::printf("ERR cannot read the image %s (%s)\n", frames[f].c_str(), err.c_str());
                std::fflush(stdout);
                read = false;
            }
        if (!read) continue;
        const bool ok = video ? enc->encode_frames(pictures[0], pictures.size() > 1 ? &pictures[1] : nullptr, rows, err)
                              : enc->encode(pictures[0], rows, err);
        if (!ok && err.empty()) err = "the image encoder failed";
        if (err.empty() && !lamina::vision::write_rows(out, rows)) err = "cannot write " + out;
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (err.empty()) std::printf("OK %d %d %d %.0f\n", rows.n, rows.nx, rows.ny, ms);
        else std::printf("ERR %s\n", err.c_str());
        std::fflush(stdout);
    }
    return 0;
}
