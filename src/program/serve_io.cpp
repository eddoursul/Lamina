#include "lamina/program/serve_io.hpp"

#include <cstdio>
#include <iostream>
#include <mutex>
#include <vector>

namespace lamina::program {
namespace {

ServeIo stdio_io() {
    ServeIo io;
    io.read_line = [](std::string& line) { return static_cast<bool>(std::getline(std::cin, line)); };
    io.write = [](std::string_view text) {
        std::fwrite(text.data(), 1, text.size(), stdout);
        std::fflush(stdout);
    };
    return io;
}

ServeIo& current() {
    static ServeIo io = stdio_io();
    return io;
}

std::mutex& write_mu() {
    static std::mutex mu;
    return mu;
}

}   // namespace

void set_serve_io(ServeIo io) {
    if (!io.read_line || !io.write) io = stdio_io();
    current() = std::move(io);
}

const ServeIo& serve_io() { return current(); }

void serve_vprint(const char* fmt, va_list ap) {
    va_list ap2;
    va_copy(ap2, ap);
    char small[512];
    const int n = std::vsnprintf(small, sizeof small, fmt, ap);
    if (n < 0) {
        va_end(ap2);
        return;
    }
    std::string text;
    if ((size_t) n < sizeof small) {
        text.assign(small, (size_t) n);
    } else {
        std::vector<char> big((size_t) n + 1);
        std::vsnprintf(big.data(), big.size(), fmt, ap2);
        text.assign(big.data(), (size_t) n);
    }
    va_end(ap2);
    std::lock_guard<std::mutex> lk(write_mu());   // a line is written whole, whichever thread writes it
    current().write(text);
}

void serve_print(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    serve_vprint(fmt, ap);
    va_end(ap);
}

void serve_warn(bool serving, const std::string& text) {
    std::fprintf(stderr, "lamina: %s\n", text.c_str());
    if (serving) serve_print("WARN %s\n", text.c_str());
}

std::string read_vision_records(const std::string& path, int64_t n_embd, std::vector<float>& rows,
                                std::vector<VisionImage>& imgs) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return "cannot open " + path;
    std::string ve;
    while (ve.empty()) {
        int32_t hdr[5];
        const size_t got = std::fread(hdr, sizeof(int32_t), 5, f);
        if (got == 0) break;
        if (got != 5 || hdr[0] != 0x31455653 || hdr[1] < 1 || hdr[2] < 1 || hdr[3] < 1 ||
            (int64_t) hdr[2] * hdr[3] != hdr[1] || hdr[4] != (int32_t) n_embd) {
            ve = "bad embeddings file (expected the image encoder's records of width " +
                 std::to_string((long long) n_embd) + ")";
            break;
        }
        const size_t off = rows.size(), cnt = (size_t) hdr[1] * (size_t) hdr[4];
        rows.resize(off + cnt);
        if (std::fread(rows.data() + off, sizeof(float), cnt, f) != cnt) {
            ve = "short embeddings file";
            break;
        }
        imgs.push_back({hdr[1], hdr[2], hdr[3], off});
    }
    std::fclose(f);
    return ve;
}

uint64_t vision_image_key(const VisionImage& im, const uint64_t* key, const float* rows, int64_t n_embd) {
    auto fnv1a = [](const void* data, size_t n, uint64_t h) {
        const uint8_t* p = static_cast<const uint8_t*>(data);
        for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ull;
        return h;
    };
    const int64_t grid[3] = {im.n, im.nx, im.ny};
    const uint64_t h = fnv1a(grid, sizeof grid, 1469598103934665603ull);
    return key != nullptr ? fnv1a(key, sizeof(uint64_t), h)
                          : fnv1a(rows + im.off, (size_t) im.n * (size_t) n_embd * sizeof(float), h);
}

std::string image_cells(const std::string& path, std::vector<int32_t>& ids, int32_t pad, int64_t n_embd,
                        const std::vector<uint64_t>& keys, ImageCells& r) {
    std::vector<VisionImage> imgs;
    if (std::string ve = read_vision_records(path, n_embd, r.rows, imgs); !ve.empty()) return ve;
    const int64_t n = (int64_t) ids.size();
    r.cell_row.assign((size_t) n, -1);
    size_t k = 0;
    for (int64_t i = 0; i < n;) {
        if (ids[(size_t) i] != pad) {
            ++i;
            continue;
        }
        if (k >= imgs.size()) return "the prompt has more images than the embeddings file";
        const VisionImage& im = imgs[k++];
        const uint64_t key =
            vision_image_key(im, keys.size() == imgs.size() ? &keys[k - 1] : nullptr, r.rows.data(), n_embd);
        for (int64_t j = 0; j < im.n; ++j) {
            if (i + j >= n || ids[(size_t) (i + j)] != pad)
                return "image " + std::to_string(k) + " has " + std::to_string((long long) im.n) +
                       " rows but fewer image tokens";
            uint64_t h = key + (uint64_t) j * 0x9E3779B97F4A7C15ull;   // splitmix64 of the key and the row
            h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9ull;
            h = (h ^ (h >> 27)) * 0x94D049BB133111EBull;
            ids[(size_t) (i + j)] = -1 - (int32_t) ((h ^ (h >> 31)) & 0x3fffffff);
            r.cell_row[(size_t) (i + j)] = (int32_t) (im.off / (size_t) n_embd + (size_t) j);
        }
        i += im.n;
    }
    if (k != imgs.size()) return "the embeddings file has more images than the prompt";
    if (n > 0 && r.cell_row.back() >= 0) return "the prompt cannot end in an image";
    return "";
}

}   // namespace lamina::program
