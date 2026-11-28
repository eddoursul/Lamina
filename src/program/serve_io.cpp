#include "lamina/program/serve_io.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
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

ServeInput::ServeInput(bool console_quit) : q_(std::make_shared<Queue>()), stop(q_->stop) {
    const ServeIo& io = serve_io();
    reader_ = std::thread([q = q_, &io] {
        std::string l;
        while (io.read_line(l)) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            if (l == "STOP") {
                q->stop.store(true);
                continue;
            }
            std::lock_guard<std::mutex> lk(q->mu);
            q->lines.push_back(l);
            q->cv.notify_one();
        }
        std::lock_guard<std::mutex> lk(q->mu);
        q->eof = true;
        q->cv.notify_one();
    });
    if (console_quit && !io.in_process)
        console_quit_.emplace([q = q_] {
            q->stop.store(true);
            std::lock_guard<std::mutex> lk(q->mu);
            q->lines.push_front("QUIT");
            q->cv.notify_one();
        });
}

ServeInput::~ServeInput() {
    console_quit_.reset();   // (Ctrl+C ends the process again before the reader goes)
    const ServeIo& io = serve_io();
    if (!io.in_process) {
        reader_.detach();
        return;
    }
    if (io.end_input) io.end_input();
    reader_.join();
}

bool ServeInput::next(std::string& line) {
    std::unique_lock<std::mutex> lk(q_->mu);
    q_->cv.wait(lk, [&] { return q_->eof || !q_->lines.empty(); });
    if (q_->lines.empty()) return false;
    line = std::move(q_->lines.front());
    q_->lines.pop_front();
    return true;
}

bool ServeInput::poll(std::string& line) {
    std::lock_guard<std::mutex> lk(q_->mu);
    if (q_->lines.empty()) return false;
    line = std::move(q_->lines.front());
    q_->lines.pop_front();
    return true;
}

bool ServeInput::ready() {
    std::lock_guard<std::mutex> lk(q_->mu);
    return q_->eof || !q_->lines.empty();
}

bool ServeInput::wait_until(std::chrono::steady_clock::time_point until) {
    std::unique_lock<std::mutex> lk(q_->mu);
    return q_->cv.wait_until(lk, until, [&] { return q_->eof || !q_->lines.empty(); });
}

void ServeInput::quit() {
    std::lock_guard<std::mutex> lk(q_->mu);
    q_->eof = true;
    q_->lines.clear();
}

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

namespace {

/// One file's records (read_vision_records)
std::string read_records(const std::string& path, int64_t n_embd, std::vector<float>& rows,
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

}   // namespace

std::string read_vision_records(const std::string& path, int64_t n_embd, std::vector<float>& rows,
                                std::vector<VisionImage>& imgs) {
    std::vector<std::string> files{path};
    {
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return "cannot open " + path;
        char head[5] = {};
        if (std::fread(head, 1, sizeof head, f) == sizeof head && std::memcmp(head, "SVL1\n", sizeof head) == 0) {
            std::string text;   // a list: its files' records in its order (a file in it is records, no list)
            char buf[4096];
            for (size_t got; (got = std::fread(buf, 1, sizeof buf, f)) > 0;) text.append(buf, got);
            files.clear();
            for (size_t at = 0; at < text.size();) {
                const size_t e = std::min(text.find('\n', at), text.size());
                if (e > at) files.push_back(text.substr(at, e - at));
                at = e + 1;
            }
        }
        std::fclose(f);
    }
    uintmax_t bytes = 0;   // the rows reserved at once: a video's thousand records reallocate nothing
    for (const std::string& p : files) {
        std::error_code ec;
        const uintmax_t n = std::filesystem::file_size(p, ec);
        if (!ec) bytes += n;
    }
    rows.reserve(rows.size() + (size_t) (bytes / sizeof(float)));
    for (const std::string& p : files)
        if (std::string ve = read_records(p, n_embd, rows, imgs); !ve.empty()) return ve;
    return "";
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

std::string parse_gen(const char* p, bool geni, GenRequest& r,
                      const std::function<void(const std::string&, const char*)>& own) {
    r = GenRequest{};
    char* e = nullptr;
    r.max_new = std::strtoll(p, &e, 10);
    p = e;
    for (;;) {
        while (*p == ' ') ++p;
        const char* st0 = p;
        while (*p && *p != ' ') ++p;
        const std::string tk(st0, (size_t) (p - st0));
        const size_t eq = tk.find('=');
        if (tk.empty() || eq == std::string::npos) {
            p = st0;
            break;
        }
        const std::string key = tk.substr(0, eq);
        const char* v = tk.c_str() + eq + 1;
        if (r.smp.set(key, v)) continue;
        if (key == "img_keys") {
            r.img_keys.clear();
            for (const char* q = v; *q;) {
                char* qe = nullptr;
                const unsigned long long x = std::strtoull(q, &qe, 16);
                if (qe == q || (*qe != ',' && *qe != '\0')) {
                    r.img_keys.clear();
                    break;
                }
                r.img_keys.push_back((uint64_t) x);
                q = *qe == ',' ? qe + 1 : qe;
            }
        } else if (own) {
            own(key, v);
        }
    }
    if (geni) {
        while (*p == ' ') ++p;
        const char* e0 = p;
        while (*p && *p != ' ') ++p;
        r.emb.assign(e0, (size_t) (p - e0));
    }
    if (r.max_new < 1) return "bad request: max_new";
    if (geni && r.emb.empty()) return "bad request: GENI names no file of picture rows";
    std::string err;
    if (!parse_id_list(p, r.ids, err)) return "bad request: " + err;
    return "";
}

std::string gen_fits(const GenRequest& r, int64_t n_vocab, int64_t room) {
    for (const int32_t t : r.ids)
        if (t < 0 || t >= n_vocab) return "a token id is outside the vocabulary";
    if ((int64_t) r.ids.size() > room || r.max_new > room - (int64_t) r.ids.size()) {
        char b[160];
        std::snprintf(b, sizeof b, "prompt (%zu tokens) + max_new (%lld) exceeds the context (%lld)", r.ids.size(),
                      r.max_new, (long long) room);
        return b;
    }
    return "";
}

void print_done(const DoneLine& d) {
    serve_print("DONE %lld %zu %.1f %.1f %s %lld %lld %zu %lld %lld 0 0 0.0 %lld 0\n", d.produced, d.prompt,
                d.prompt_ms, d.decode_ms, d.finish, d.accepted, d.offered, d.reused, d.hits, d.lookups, d.read);
}

std::string picture_cells(const std::string& path, const std::vector<int32_t>& ids, const std::vector<int32_t>& pads,
                          int64_t n_embd, const std::vector<uint64_t>& keys, ImageCells& r) {
    std::vector<VisionImage> imgs;
    r.pictures.clear();
    if (std::string ve = read_vision_records(path, n_embd, r.rows, imgs); !ve.empty()) return ve;
    const int64_t n = (int64_t) ids.size();
    r.cell_row.assign((size_t) n, -1);
    r.mrope.assign((size_t) n * 3, 0);
    const auto put = [&](int64_t c, int64_t t, int64_t h, int64_t w) {
        r.mrope[(size_t) c * 3] = (int32_t) t;
        r.mrope[(size_t) c * 3 + 1] = (int32_t) h;
        r.mrope[(size_t) c * 3 + 2] = (int32_t) w;
    };
    int64_t p = 0;   // the next text position
    size_t k = 0;
    for (int64_t i = 0; i < n;) {
        const int32_t pad = ids[(size_t) i];
        if (pad < 0 || std::find(pads.begin(), pads.end(), pad) == pads.end()) {
            put(i, p, p, p);
            ++p;
            ++i;
            continue;
        }
        if (k >= imgs.size()) return "the prompt has more images than the embeddings file";
        const VisionImage& im = imgs[k++];
        r.pictures.push_back(
            {i, im.n,
             vision_image_key(im, keys.size() == imgs.size() ? &keys[k - 1] : nullptr, r.rows.data(), n_embd)});
        for (int64_t j = 0; j < im.n; ++j) {
            if (i + j >= n || ids[(size_t) (i + j)] != pad)
                return "image " + std::to_string(k) + " has " + std::to_string((long long) im.n) +
                       " rows but fewer pad tokens";
            r.cell_row[(size_t) (i + j)] = (int32_t) (im.off / (size_t) n_embd + (size_t) j);
            put(i + j, p, p + j / im.nx, p + j % im.nx);
        }
        i += im.n;
        p += std::max(im.nx, im.ny);
    }
    r.next = p;
    if (k != imgs.size()) return "the embeddings file has more images than the prompt";
    if (n > 0 && r.cell_row.back() >= 0) return "the prompt cannot end in an image";
    return "";
}

std::string picture_cells(const std::string& path, const std::vector<int64_t>& ids, const std::vector<int32_t>& pads,
                          int64_t n_embd, const std::vector<uint64_t>& keys, ImageCells& r) {
    std::vector<int32_t> ids32(ids.size());
    for (size_t i = 0; i < ids.size(); ++i)
        ids32[i] = ids[i] < INT32_MIN || ids[i] > INT32_MAX ? -1 : (int32_t) ids[i];   // (no pad)
    return picture_cells(path, ids32, pads, n_embd, keys, r);
}

std::string image_cells(const std::string& path, std::vector<int32_t>& ids, const std::vector<int32_t>& pads,
                        int64_t n_embd, const std::vector<uint64_t>& keys, ImageCells& r) {
    if (std::string ve = picture_cells(path, ids, pads, n_embd, keys, r); !ve.empty()) return ve;
    for (const ImageCells::Picture& pic : r.pictures)
        for (int64_t j = 0; j < pic.n; ++j) {
            uint64_t h = pic.key + (uint64_t) j * 0x9E3779B97F4A7C15ull;   // splitmix64 of the key and the row
            h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9ull;
            h = (h ^ (h >> 27)) * 0x94D049BB133111EBull;
            ids[(size_t) (pic.cell + j)] = -1 - (int32_t) ((h ^ (h >> 31)) & 0x3fffffff);
        }
    return "";
}

}   // namespace lamina::program
