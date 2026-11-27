// src/server/vision.cpp - the resident image encoder (vision.hpp).
#include "vision.hpp"

#include "base64.hpp"
#include "http_client.hpp"
#include "log.hpp"
#include "process.hpp"
#include "service.hpp"
#include "sha256.hpp"

#include "lamina/artifact/gguf_reader.hpp"
#if LAMINA_VISION
#include "lamina/core/gpu_select.hpp"
#include "lamina/platform/spill.hpp"
#include "lamina/vision/encoder.hpp"
#include "lamina/vision/projector.hpp"
#include "lamina/vision/video.hpp"
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <thread>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#define STBI_FAILURE_USERMSG
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif
#include "stb/stb_image.h"
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <wincodec.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace lamina {
namespace fs = std::filesystem;
namespace {

std::string read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open");
    return std::string(std::istreambuf_iterator<char>(f), {});
}

bool is_jpeg(const std::string& d) {
    return d.size() >= 3 && (unsigned char) d[0] == 0xff && (unsigned char) d[1] == 0xd8 &&
           (unsigned char) d[2] == 0xff;
}
bool passes(const std::string& d) {
    return starts_with(d, std::string_view("\x89PNG\r\n\x1a\n", 8)) || starts_with(d, "BM") ||
           starts_with(d, "GIF87a") || starts_with(d, "GIF89a");
}

/// a JPEG's EXIF Orientation (1 when it has none): stb_image ignores the tag (#1229)
int exif_orientation(const std::string& d) {
    auto u16 = [&](size_t at, bool le) -> uint32_t {
        if (at + 2 > d.size()) return 0;
        const auto a = (unsigned char) d[at], b = (unsigned char) d[at + 1];
        return le ? (uint32_t) (a | b << 8) : (uint32_t) (a << 8 | b);
    };
    auto u32 = [&](size_t at, bool le) -> uint32_t {
        return le ? u16(at, le) | u16(at + 2, le) << 16 : u16(at, le) << 16 | u16(at + 2, le);
    };
    size_t at = 2;
    while (at + 4 <= d.size() && (unsigned char) d[at] == 0xff) {
        const auto marker = (unsigned char) d[at + 1];
        if (marker == 0xd9 || marker == 0xda) break;   // the end, or the image data
        const uint32_t len = u16(at + 2, false);
        if (len < 2) break;
        if (marker == 0xe1 && d.compare(at + 4, 6, std::string("Exif\0\0", 6)) == 0) {
            const size_t tiff = at + 10;
            if (tiff + 8 > d.size()) return 1;
            const bool le = d[tiff] == 'I';
            const size_t ifd = tiff + u32(tiff + 4, le);
            const uint32_t count = u16(ifd, le);
            for (uint32_t i = 0; i < count; ++i) {
                const size_t e = ifd + 2 + 12 * (size_t) i;
                if (e + 12 > d.size()) break;
                if (u16(e, le) == 0x0112) return (int) u16(e + 8, le);
            }
            return 1;
        }
        at += 2 + len;
    }
    return 1;
}

/// a picture over kImagePixelsMax, as Pillow words it
std::string too_many_pixels(uint64_t w, uint64_t h) {
    return "Image size (" + std::to_string(w * h) + " pixels) exceeds limit of " + std::to_string(kImagePixelsMax) +
           " pixels, could be decompression bomb DOS attack.";
}

struct Pixels {
    int w = 0, h = 0;
    std::vector<unsigned char> rgb;   // 3 bytes a pixel, rows top to bottom
};

/// RGBA pixels -> RGB, transparent areas white (not black)
Pixels on_white(const unsigned char* rgba, int w, int h) {
    Pixels p{w, h, std::vector<unsigned char>((size_t) w * h * 3)};
    for (size_t i = 0; i < (size_t) w * h; ++i) {
        const unsigned a = rgba[4 * i + 3];
        for (int c = 0; c < 3; ++c)
            p.rgb[3 * i + c] = (unsigned char) ((rgba[4 * i + c] * a + 255 * (255 - a) + 127) / 255);
    }
    return p;
}

/// Pillow's exif_transpose for orientations 2-8
Pixels upright(const Pixels& s, int orientation) {
    const bool swap = orientation >= 5;
    Pixels o{swap ? s.h : s.w, swap ? s.w : s.h, std::vector<unsigned char>(s.rgb.size())};
    for (int y = 0; y < o.h; ++y)
        for (int x = 0; x < o.w; ++x) {
            int sx = x, sy = y;
            switch (orientation) {
                case 2: sx = s.w - 1 - x; break;
                case 3:
                    sx = s.w - 1 - x;
                    sy = s.h - 1 - y;
                    break;
                case 4: sy = s.h - 1 - y; break;
                case 5:
                    sx = y;
                    sy = x;
                    break;
                case 6:
                    sx = y;
                    sy = s.h - 1 - x;
                    break;
                case 7:
                    sx = s.w - 1 - y;
                    sy = s.h - 1 - x;
                    break;
                case 8:
                    sx = s.w - 1 - y;
                    sy = x;
                    break;
                default: break;
            }
            std::memcpy(&o.rgb[3 * ((size_t) y * o.w + x)], &s.rgb[3 * ((size_t) sy * s.w + sx)], 3);
        }
    return o;
}

/// a 24-bit BMP (the decoder reads it; the converted picture's lossless form)
std::string bmp(const Pixels& p) {
    const size_t row = ((size_t) p.w * 3 + 3) & ~(size_t) 3, size = 54 + row * p.h;
    std::string out(size, '\0');
    auto put32 = [&](size_t at, uint32_t v) {
        for (int i = 0; i < 4; ++i) out[at + i] = (char) (v >> (8 * i));
    };
    out[0] = 'B';
    out[1] = 'M';
    put32(2, (uint32_t) size);
    put32(10, 54);
    put32(14, 40);
    put32(18, (uint32_t) p.w);
    put32(22, (uint32_t) p.h);   // positive: rows bottom to top
    out[26] = 1;
    out[28] = 24;
    put32(34, (uint32_t) (row * p.h));
    for (int y = 0; y < p.h; ++y) {
        char* dst = out.data() + 54 + row * (size_t) (p.h - 1 - y);
        const unsigned char* src = &p.rgb[(size_t) y * p.w * 3];
        for (int x = 0; x < p.w; ++x) {
            dst[3 * x] = (char) src[3 * x + 2];
            dst[3 * x + 1] = (char) src[3 * x + 1];
            dst[3 * x + 2] = (char) src[3 * x];
        }
    }
    return out;
}

#ifdef _WIN32
/// the formats stb_image does not read (WebP, TIFF, ICO, HEIF ...) through Windows' own codecs
std::optional<Pixels> wic_decode(const std::string& data, std::string& why) {
    const HRESULT init = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    struct Uninit {
        bool on;
        ~Uninit() {
            if (on) CoUninitialize();
        }
    } uninit{SUCCEEDED(init)};
    IWICImagingFactory* factory = nullptr;
    IWICStream* stream = nullptr;
    IWICBitmapDecoder* decoder = nullptr;
    IWICBitmapFrameDecode* frame = nullptr;
    IWICBitmapSource* rgba = nullptr;
    auto release = [&] {
        for (IUnknown* u : std::initializer_list<IUnknown*>{rgba, frame, decoder, stream, factory})
            if (u != nullptr) u->Release();
    };
    std::optional<Pixels> out;
    UINT w = 0, h = 0;
    if (SUCCEEDED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory))) &&
        SUCCEEDED(factory->CreateStream(&stream)) &&
        SUCCEEDED(stream->InitializeFromMemory((BYTE*) data.data(), (DWORD) data.size())) &&
        SUCCEEDED(factory->CreateDecoderFromStream(stream, nullptr, WICDecodeMetadataCacheOnDemand, &decoder)) &&
        SUCCEEDED(decoder->GetFrame(0, &frame)) && SUCCEEDED(frame->GetSize(&w, &h)) && w > 0 && h > 0 &&
        (uint64_t) w * h <= kImagePixelsMax &&
        SUCCEEDED(WICConvertBitmapSource(GUID_WICPixelFormat32bppRGBA, frame, &rgba)) &&
        SUCCEEDED(rgba->GetSize(&w, &h)) && w > 0 && h > 0) {
        std::vector<unsigned char> px((size_t) w * h * 4);
        if (SUCCEEDED(rgba->CopyPixels(nullptr, w * 4, (UINT) px.size(), px.data())))
            out = on_white(px.data(), (int) w, (int) h);
    }
    if (!out)
        why = (uint64_t) w * h > kImagePixelsMax ? too_many_pixels(w, h)
                                                 : "not a format this computer's image codecs read";
    release();
    return out;
}
#endif

/// #480: a directory for the pictures' rows whose path has no space (the engine's GENI line splits at spaces), this
/// process's own under `lamina-vision` (lamina/platform/spill.hpp: it holds a lock there for its life, and a start
/// removes the folders of the servers that ended without removing theirs, killed or crashed: each kept the rows of
/// its pictures and videos, up to gigabytes): in the system temp directory while that path has none, then in the
/// working directory, then at the system drive's root; in the temp directory as the last resort
std::string work_dir(lamina::platform::SpillDir& own) {
    std::vector<fs::path> parents = {u8_path(temp_dir())};
    std::error_code ec;
    parents.push_back(fs::current_path(ec));
#ifdef _WIN32
    const char* drive = std::getenv("SystemDrive");
    parents.push_back(u8_path(std::string(drive != nullptr ? drive : "C:") + "\\"));
#endif
    parents.push_back(u8_path(temp_dir()));
    std::string err;
    for (size_t i = 0; i < parents.size(); ++i) {
        const std::string root = u8_str(parents[i] / "lamina-vision");
        if (root.find(' ') != std::string::npos && i + 1 < parents.size()) continue;
        int cleaned = 0;
        if (!own.open(root, err, &cleaned)) continue;
        if (cleaned > 0)
            std::fprintf(stderr, "lamina: %s: the image encoder's folders of ended servers removed (%d)\n",
                         root.c_str(), cleaned);
        return own.path();
    }
    throw std::runtime_error("no directory for the image encoder's files could be made: " + err);
}

}   // namespace

std::string combined_embeddings_path(const std::string& vision_dir, uintmax_t nbytes) {
    const std::string name = "req-" + random_hex(12) + ".sve";
#ifndef _WIN32
    std::error_code ec;
    const fs::path shm = "/dev/shm";
    if (fs::is_directory(shm, ec) && ::access("/dev/shm", W_OK) == 0) {
        const fs::space_info sp = fs::space(shm, ec);
        if (!ec && sp.available > 2 * nbytes + (16u << 20))
            return "/dev/shm/lamina-vision-" + std::to_string(::getpid()) + "-" + name;
    }
#else
    (void) nbytes;
#endif
    return u8_str(u8_path(vision_dir) / name);
}

void write_temporary(const std::string& path, const std::vector<std::string>& parts) {
#ifdef _WIN32
    const std::wstring w = u8_path(path).wstring();
    const HANDLE h = CreateFileW(w.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr, CREATE_ALWAYS,
                                 FILE_ATTRIBUTE_TEMPORARY, nullptr);
    if (h == INVALID_HANDLE_VALUE) throw std::runtime_error("the image embeddings could not be written: " + path);
    struct Close {
        HANDLE h;
        ~Close() { CloseHandle(h); }
    } close{h};
    for (const auto& part : parts) {
        const std::string data = read_file(u8_path(part));
        DWORD n = 0;
        if (!WriteFile(h, data.data(), (DWORD) data.size(), &n, nullptr) || n != data.size())
            throw std::runtime_error("the image embeddings could not be written: " + path);
    }
#else
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0) throw std::runtime_error("the image embeddings could not be written: " + path);
    for (const auto& part : parts) {
        const std::string data = read_file(part);
        for (size_t at = 0; at < data.size();) {
            const ssize_t n = ::write(fd, data.data() + at, data.size() - at);
            if (n <= 0) {
                ::close(fd);
                throw std::runtime_error("the image embeddings could not be written: " + path);
            }
            at += (size_t) n;
        }
    }
    ::close(fd);
#endif
}

std::optional<double> timeout_env(const char* name, double dflt) {
    double v = dflt;
    if (const char* s = std::getenv(name)) {
        try {
            v = std::stod(s);
        } catch (const std::exception&) {
            v = dflt;
        }
    }
    if (v <= 0) return std::nullopt;
    return v;
}

bool device_path(const std::string& path) {
    const std::string p = replace_all(path, "/", "\\");
    if (starts_with(p, "\\??\\")) return true;
    const size_t n = p.find_first_not_of('\\');   // \\.\ and \\?\, after any number of separators
    if (n == std::string::npos || n < 2) return false;
    const std::string_view rest = std::string_view(p).substr(n);
    return rest == "." || rest == "?" || starts_with(rest, ".\\") || starts_with(rest, "?\\");
}

bool share_path(const std::string& path) {
    const size_t n = replace_all(path, "/", "\\").find_first_not_of('\\');
    return n != std::string::npos && n >= 2 && !device_path(path);
}

std::string projector_type(const std::string& mmproj) {
    try {
        const lamina::GgufFile g(mmproj);
        for (const char* key : {"clip.projector_type", "clip.vision.projector_type"}) {
            const lamina::MetaValue* v = g.get(key);
            if (v != nullptr && v->type == lamina::MetaType::STRING && !v->s.empty()) return v->s;
        }
    } catch (const std::exception&) {   // not a GGUF it reads: the encoder says so
    }
    return "";
}

bool projector_type_read(const std::string& type) {
    return type == "qwen3vl_merger" || type == "hy_v4_flash" || type == "inkling2v";
}

std::string projector_refusal(const std::string& mmproj) {
#if LAMINA_VISION
    try {
        vision::EncoderParams p;
        p.mmproj = mmproj;
        vision::Projector header;
        header.open(p);
    } catch (const std::exception& e) {
        return e.what();
    }
#else
    (void) mmproj;
#endif
    return {};
}

bool vision_built() {
#if LAMINA_VISION
    return true;
#else
    return false;
#endif
}

const char* vision_not_built() { return "this lamina was built without the image encoder (LAMINA_BUILD_VISION=OFF)"; }

#if LAMINA_VISION
struct Vision::Loaded {
    std::unique_ptr<vision::Encoder> encoder;
    int device = -1;                // a resident GPU encoder's CUDA device, kept from the engine's resets
    std::function<void()> unlend;   // a lease in the engine's memory: given back after the encoder
    std::string where;              // a lease: what it runs in
    double waited = 0;              // a lease: how long the engine took to lend (it waits for a prompt it reads)
    std::chrono::steady_clock::time_point since = std::chrono::steady_clock::now();
    int pictures = 0;
    ~Loaded() {
        encoder.reset();
        if (unlend) unlend();
        if (device >= 0) core::gpu_keep(device, false);
    }
};
struct Vision::Plan {
    vision::EncoderParams params;
    vision::ProjectorParams proj;       // the projector's header (a video's plan)
    std::string where = "on the CPU";   // "on GPU 1, NVIDIA GeForce RTX 3090"
    uint64_t need = 0;                  // borrowing: what the encoder takes of its GPU
    bool said_free = false, said_cpu = false;
};
namespace {
double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}
}   // namespace
#else
struct Vision::Loaded {};
struct Vision::Plan {};
#endif

Vision::Vision(const json& cfg, bool lazy) : cfg_(cfg) {
    dir = work_dir(own_dir_);
    if (!vision_built())
        throw std::runtime_error(std::string("the config has a \"vision\" section, and ") + vision_not_built());
    if (!lazy) start();
}

Vision::~Vision() {
    close();
    own_dir_.close();   // (the folder, with what it holds)
}

void Vision::set_lender(Lend lend, std::function<void()> unlend) {
    const std::lock_guard<std::mutex> lk(mu_);
    lender_ = std::move(lend);
    unlender_ = std::move(unlend);
}

void Vision::start() {
#if LAMINA_VISION
    using Clock = std::chrono::steady_clock;
    const json& cfg = cfg_;
    auto plan = std::make_unique<Plan>();
    vision::EncoderParams& p = plan->params;
    p.mmproj = jstr(cfg, "mmproj");
    p.gpu = truthy(jget(cfg, "gpu"));
    auto number = [](const json& v) { return v.is_number() ? (int) v.get<double>() : 0; };
    p.threads = number(jget(cfg, "threads"));
    p.max_tokens = number(jget(cfg, "max_tokens"));
    p.min_tokens = number(jget(cfg, "min_tokens"));   // #767
    p.q8_fp16 = lower(jstr(cfg, "q8")) == "fp16";
    {   // the header, for a video's plan (and the file a projector the encoder reads: its load says so otherwise)
        vision::Projector header;
        try {
            header.open(p);
            plan->proj = header;
        } catch (const std::exception&) {}
    }
    const FfmpegTools ff = find_ffmpeg(jstr(cfg, "ffmpeg"));
    const bool lend = p.gpu && truthy(jget(cfg, "lend"));
    auto loaded = std::make_unique<Loaded>();
    if (p.gpu) {
        // its card among this process's CUDA devices: a UUID or its start, a PCI bus id, a name, or a number, which
        // counts the cards in the PCI buses' order as nvidia-smi does
        const std::vector<core::GpuInfo> gpus = core::list_gpus(true);
        const json& dev = jget(cfg, "cuda_device");
        const std::string spec = dev.is_null()     ? ""
                                 : dev.is_string() ? std::string(strip(dev.get<std::string>()))
                                                   : py_dumps(dev);
        std::string why;
        if (spec.empty()) {
            p.device = gpus.empty() ? -1 : 0;
            why = "no CUDA device is visible";
        } else if (std::all_of(spec.begin(), spec.end(), [](unsigned char c) { return std::isdigit(c) != 0; })) {
            std::vector<const core::GpuInfo*> by_bus;
            for (const core::GpuInfo& g : gpus) by_bus.push_back(&g);
            std::stable_sort(by_bus.begin(), by_bus.end(),
                             [](const core::GpuInfo* a, const core::GpuInfo* b) { return a->bus_id < b->bus_id; });
            const size_t i = (size_t) std::stoul(spec);
            p.device = i < by_bus.size() ? by_bus[i]->ordinal : -1;
            why = "no GPU " + spec + " (CUDA sees " + std::to_string(gpus.size()) + ")";
        } else {
            p.device = core::find_gpu(gpus, spec, why);
        }
        if (p.device < 0)
            throw std::runtime_error("the vision encoder did not start: its GPU" +
                                     (spec.empty() ? std::string() : " " + spec) + " is not there (" + why +
                                     "): name a card that is there in the config's vision "
                                     "\"cuda_device\" (nvidia-smi -L lists them), or set \"gpu\" to false");
        plan->where = "on GPU " + std::to_string(p.device) + ", " + gpus[(size_t) p.device].name;
        // the encoder's kernels are the engine's: a card they do not run on (older than an RTX 30, or one this build
        // has no kernels for) is said before anything is loaded
        if (const std::string no = core::gpu_unusable(p.device); !no.empty())
            throw std::runtime_error("the vision encoder did not start " + plan->where + ": " + no +
                                     "; name another card in "
                                     "the config's vision \"cuda_device\", or set \"gpu\" to false (the CPU)");
    }
    std::string err;
    if (lend) {
        // nothing is loaded: the file is read for what a loan must hold, and that it is a projector the encoder reads
        plan->need = vision::encoder_gpu_bytes(p, err);
        if (plan->need == 0)
            throw std::runtime_error("the vision encoder did not start: " + p.mmproj + " (" + err + ")");
        std::fprintf(stderr, "lamina: the image encoder %s borrows %llu MiB of VRAM for a picture: %s\n",
                     plan->where.c_str(), (unsigned long long) (plan->need >> 20), p.mmproj.c_str());
        // said before the engine loads: one without a cache to lend from then sizes itself so that it has the bytes
        core::gpu_loan_ask(p.device, plan->need);
        const std::lock_guard<std::mutex> lk(mu_);
        plan_ = std::move(plan);
        ff_ = ff;
        lend_ = true;
        alive_ = true;
        return;
    }
    if (p.gpu) {
        core::gpu_keep(p.device, true);
        loaded->device = p.device;
    }
    const auto t0 = Clock::now();
    loaded->encoder = vision::open_encoder(p, err);
    if (!loaded->encoder) throw std::runtime_error("the vision encoder did not start: " + p.mmproj + " (" + err + ")");
    const uint64_t vram = loaded->encoder->gpu_bytes();
    std::fprintf(stderr, "lamina: the image encoder %s: %s loaded in %.1f s%s\n", plan->where.c_str(), p.mmproj.c_str(),
                 seconds_since(t0),
                 vram == 0 ? ""
                           : (", " + std::to_string(vram >> 20) + " MiB of VRAM for its weights and buffers").c_str());
    if (p.gpu) {
        // A picture at the projector's limit now: what a picture takes of the GPU (the buffers, the kernels' images)
        // is taken before the engine sizes its caches from what is left, and the first picture's time is a
        // picture's.  (A square well above any limit; the encoder scales it down.)
        const auto w0 = Clock::now();
        vision::Rgb grey;
        grey.w = grey.h = 2048;
        grey.px.assign((size_t) grey.w * grey.h * 3, 128);
        vision::Rows rows;
        if (!loaded->encoder->encode(grey, rows, err))
            throw std::runtime_error("the vision encoder did not start: a picture at its limit failed (" + err + ")");
        std::fprintf(stderr, "lamina: the image encoder warmed up at %d image tokens in %.1f s\n", rows.n,
                     seconds_since(w0));
    }
    const std::lock_guard<std::mutex> lk(mu_);
    plan_ = std::move(plan);
    ff_ = ff;
    lend_ = false;
    loaded_ = std::move(loaded);
    alive_ = true;
#endif
}

void* Vision::encoder_now(std::string& err) {
#if LAMINA_VISION
    if (loaded_ != nullptr) return loaded_->encoder.get();
    if (!lend_ || plan_ == nullptr) {
        err = "the vision encoder is unloaded";
        return nullptr;
    }
    // a lease for this request's pictures: the engine's loan, else the card's free VRAM, else the CPU
    auto lease = std::make_unique<Loaded>();
    vision::EncoderParams p = plan_->params;
    std::string why = "no engine lends";
    // (LAMINA_VISION_NO_LOAN=1: no loan is asked for, for testing what follows a refusal)
    const bool ask = lender_ != nullptr && std::getenv("LAMINA_VISION_NO_LOAN") == nullptr;
    if (!ask && lender_ != nullptr) why = "LAMINA_VISION_NO_LOAN";
    if (ask) {
        const auto asked = std::chrono::steady_clock::now();
        if (void* at = lender_(p.device, plan_->need, why)) {
            lease->unlend = unlender_;
            lease->where = why;   // what the engine lent
            lease->waited = seconds_since(asked);
            p.gpu_arena = at;
            p.gpu_arena_bytes = plan_->need;
            lease->encoder = vision::open_encoder(p, err);
            if (!lease->encoder) {   // (the loan goes back with the lease)
                why = err;
                lease = std::make_unique<Loaded>();
            }
        }
    }
    if (!lease->encoder) {
        // only where the card has the room: an allocation past its free memory does not fail under WDDM, it pages
        const uint64_t free_now = vision::gpu_free_bytes(p.device), room = plan_->need + ((uint64_t) 64 << 20);
        p.gpu_arena = nullptr;
        p.gpu_arena_bytes = 0;
        lease->where = "the card's free VRAM";
        if (free_now >= room) lease->encoder = vision::open_encoder(p, err);
        else err = std::to_string(free_now >> 20) + " MiB free, " + std::to_string(room >> 20) + " needed";
        if (lease->encoder && !plan_->said_free) {
            plan_->said_free = true;
            std::fprintf(stderr, "lamina: the image encoder %s found no loan (%s): in the card's free VRAM\n",
                         plan_->where.c_str(), why.c_str());
        }
    }
    if (!lease->encoder) {
        if (cpu_ == nullptr) {
            auto cpu = std::make_unique<Loaded>();
            p.gpu = false;
            std::string cpu_err;
            cpu->encoder = vision::open_encoder(p, cpu_err);
            if (!cpu->encoder) {
                err = cpu_err;
                return nullptr;
            }
            cpu_ = std::move(cpu);
        }
        if (!plan_->said_cpu) {
            plan_->said_cpu = true;
            std::fprintf(stderr,
                         "lamina: the image encoder %s found no VRAM for a picture (no loan: %s; the card's free "
                         "VRAM: %s): the CPU encodes\n",
                         plan_->where.c_str(), why.c_str(), err.c_str());
        }
        err.clear();
        return cpu_->encoder.get();
    }
    loaded_ = std::move(lease);
    return loaded_->encoder.get();
#else
    err = vision_not_built();
    return nullptr;
#endif
}

void Vision::lease_end() {
#if LAMINA_VISION
    if (!lend_ || loaded_ == nullptr) return;
    const std::string where = loaded_->where;
    const int pictures = loaded_->pictures;
    const auto since = loaded_->since;
    const double used = seconds_since(since), waited = loaded_->waited;
    loaded_.reset();   // the encoder, then the loan (the engine's slots refilled)
    // (an engine with batch slots lends once the prompt a request is reading is read: said when it took a while)
    char wait[96] = "";
    if (waited >= 1.0) std::snprintf(wait, sizeof wait, " (%.2f s of them waiting for the engine to lend)", waited);
    std::fprintf(stderr,
                 "lamina: the image encoder borrowed %s for %d picture%s: %.2f s%s, and %.2f s to give it back\n",
                 where.c_str(), pictures, pictures == 1 ? "" : "s", used, wait, seconds_since(since) - used);
#endif
}

void Vision::warm_up() {
#if LAMINA_VISION
    const std::lock_guard<std::mutex> lk(mu_);
    if (!lend_ || plan_ == nullptr) return;
    std::string err;
    auto* enc = (vision::Encoder*) encoder_now(err);
    if (enc == nullptr) {
        std::fprintf(stderr, "lamina: the image encoder's warm-up found no encoder (%s)\n", err.c_str());
        return;
    }
    if (loaded_ == nullptr) return;   // (the CPU: nothing to warm)
    vision::Rgb grey;
    grey.w = grey.h = 2048;
    grey.px.assign((size_t) grey.w * grey.h * 3, 128);
    vision::Rows rows;
    if (!enc->encode(grey, rows, err))
        std::fprintf(stderr, "lamina: the image encoder's warm-up picture failed (%s)\n", err.c_str());
    loaded_->pictures = 1;
    lease_end();
#endif
}

bool Vision::alive() { return alive_; }

void Vision::unload() { close(); }

void Vision::restart() {
    close();
    start();
}

void Vision::close() {
    const std::lock_guard<std::mutex> lk(mu_);
    alive_ = false;
    loaded_.reset();
    cpu_.reset();
#if LAMINA_VISION
    if (lend_ && plan_ != nullptr) core::gpu_loan_ask(plan_->params.device, 0);   // (the next start says it again)
#endif
}

std::string Vision::load(const std::string& source) {
    if (starts_with(source, "data:")) {
        const size_t comma = source.find(',');
        try {
            if (comma == std::string::npos) throw std::invalid_argument("list index out of range");
            return b64decode(std::string_view(source).substr(comma + 1));
        } catch (const std::invalid_argument& e) {
            throw std::invalid_argument(std::string("the image's data: URL could not be read (") + e.what() + ")");
        }
    }
    if (starts_with(source, "http://") || starts_with(source, "https://")) return download(source);
    const std::string path = media_file_path(source, "an image");
    std::error_code ec;
    if (!path.empty() && fs::is_regular_file(u8_path(path), ec)) {
        if (const uintmax_t size = fs::file_size(u8_path(path), ec); !ec && size > kImageBytesMax)
            throw std::invalid_argument("the image file " + path + " is over " + std::to_string(kImageBytesMax >> 20) +
                                        " MiB");
        try {
            return read_file(u8_path(path));
        } catch (const std::exception& e) {
            throw std::invalid_argument("the image could not be read from " + path + " (" + e.what() + ")");
        }
    }
    throw std::invalid_argument(path.empty() ? "an image must be a data: URL, an http(s) URL or a file's path"
                                             : "the image file " + path + " is not there or cannot be read");
}

std::string media_file_path(const std::string& source, const char* what) {
    const bool file_url = starts_with(source, "file://");
    std::string path = file_url ? source.substr(7) : source;
    // file://host/share/x.png is \\host\share\x.png ("file://C:/x.png", a drive, is a client's spelling of
    // file:///C:/x.png)
    std::string host;
    if (file_url) host = path.substr(0, path.find_first_of("/\\"));
    const bool drive = host.size() == 2 && std::isalpha((unsigned char) host[0]) && host[1] == ':';
    if (file_url) {   // the path as urllib.request.url2pathname reads it: no localhost, %XX decoded, /C:/ a drive's
        if (lower(host) == "localhost") path.erase(0, host.size());
        path = unquote_to_bytes(path);
        if (!host.empty() && !drive && lower(host) != "localhost") path = "//" + path;
#ifdef _WIN32
        if (path.size() >= 3 && (path[0] == '/' || path[0] == '\\') && std::isalpha((unsigned char) path[1]) &&
            path[2] == ':')
            path.erase(0, 1);
#endif
    }
    // checked on the decoded path: file:///%2F%2F.%2Fpipe%2Fx is ///./pipe/x
    if (device_path(path))
        throw std::invalid_argument(std::string(what) +
                                    "'s path names a device (\\\\.\\, \\\\?\\, \\??\\), not a file");
    if (share_path(path)) {
#ifdef _WIN32
        // \\host\share\x.png (///host/share/x.png too), read through Windows' sign-in to that host
        const std::string p = replace_all(path, "/", "\\");
        path = "\\\\" + p.substr(p.find_first_not_of('\\'));
#else
        throw std::invalid_argument(std::string(what) +
                                    " on a network share (\\\\host\\share, //host/share, file://host/...) is read on "
                                    "Windows only: mount the share here and name the file by its path in it");
#endif
    }
    return path;
}

std::string Vision::download(const std::string& url) {
    // at most kImageBytesMax: a huge or endless response is not
    // read whole; one that cannot be read is a 400 that says so
    const std::string too_big = "the image URL's file is over " + std::to_string(kImageBytesMax >> 20) + " MiB";
    HttpReply r;
    try {
        r = http_request("GET", url, {{"User-Agent", "lamina"}}, "", 60, kImageBytesMax);
    } catch (const HttpError& e) {
        if (std::string(e.what()).find("larger than") != std::string::npos) throw std::invalid_argument(too_big);
        throw std::invalid_argument(std::string("the image URL could not be read: ") + e.what());
    }
    if (r.status >= 400)
        throw std::invalid_argument("the image URL could not be read: HTTP Error " + std::to_string(r.status) + ": " +
                                    r.reason);
    const std::string size = r.header("content-length");
    if (!size.empty() && std::all_of(size.begin(), size.end(), ::isdigit) && std::stoull(size) > kImageBytesMax)
        throw std::invalid_argument(too_big);
    return r.body;
}

std::string Vision::normalize(const std::string& data) {
    // a header over the limit is refused before a decoder holds its pixels: this process's, or the encoder's for the
    // formats that pass through
    int w = 0, h = 0, n = 0;
    if (stbi_info_from_memory((const stbi_uc*) data.data(), (int) data.size(), &w, &h, &n) &&
        (uint64_t) w * (uint64_t) h > kImagePixelsMax)
        throw std::invalid_argument("the image could not be read (" + too_many_pixels((uint64_t) w, (uint64_t) h) +
                                    ")");
    const bool jpeg = is_jpeg(data);
    const int orientation = jpeg ? exif_orientation(data) : 1;
    if ((jpeg && (orientation == 0 || orientation == 1 || orientation > 8)) || passes(data)) return data;
    unsigned char* px = stbi_load_from_memory((const stbi_uc*) data.data(), (int) data.size(), &w, &h, &n, 4);
    std::optional<Pixels> p;
    if (px != nullptr) {
        p = on_white(px, w, h);
        stbi_image_free(px);
    } else {
        std::string why = stbi_failure_reason() != nullptr ? stbi_failure_reason() : "unknown format";
#ifdef _WIN32
        p = wic_decode(data, why);
#endif
        if (!p) throw std::invalid_argument("the image could not be read (" + why + ")");
    }
    if (jpeg) *p = upright(*p, orientation);   // the camera's orientation tag, applied (#1229)
    return bmp(*p);
}

std::string Vision::video_refusal() {
#if LAMINA_VISION
    const std::lock_guard<std::mutex> lk(mu_);
    if (plan_ == nullptr) return "the vision encoder is unloaded";
    if (plan_->proj.kind == vision::ProjectorKind::inkling2v || plan_->proj.patch_size <= 0)
        return "this model's projector takes no video";
    return ff_.why;
#else
    return vision_not_built();
#endif
}

std::shared_ptr<VideoJob> Vision::plan_video(const std::string& source, bool is_bytes, double start, double end) {
#if LAMINA_VISION
    if (const std::string no = video_refusal(); !no.empty())
        throw std::invalid_argument("the video could not be read: " + no);
    vision::ProjectorParams proj;
    int max_tokens = 0;
    {
        const std::lock_guard<std::mutex> lk(mu_);
        proj = plan_->proj;
        max_tokens = plan_->params.max_tokens;
    }
    const vision::VideoRange range{start, end};
    auto job = std::make_shared<VideoJob>();
    job->file = open_video(source, is_bytes, dir);
    job->key = job->file->key;   // a part's rows are its own
    if (range.any())
        job->key = sha256_hex(job->key + "@" + std::to_string(start) + "-" + std::to_string(end)).substr(0, 32);
    {
        const std::lock_guard<std::mutex> lk(mu_);
        for (const auto& [k, e] : video_cache_)
            if (k == job->key) {
                const std::shared_ptr<VideoFile> file = job->file;
                *job = e.plan;
                job->file = file;
                job->cached = true;
                return job;
            }
    }
    job->info = probe_video(ff_, job->file->path);
    const vision::VideoStream s{job->info.frames, job->info.fps, job->info.shown_w, job->info.shown_h};
    vision::VideoPlan plan;
    std::string err;
    if (!vision::plan_video(proj, s, max_tokens, plan, err, range))
        throw std::invalid_argument("the video could not be read (" + err + ")");
    job->first = plan.first;
    job->frames = plan.frames;
    job->width = plan.width;
    job->height = plan.height;
    job->per_group = plan.per_group;
    job->group_tokens = plan.group_tokens;
    job->times = plan.times;
    return job;
#else
    (void) source;
    (void) is_bytes;
    (void) start;
    (void) end;
    throw std::invalid_argument(std::string("the video could not be read: ") + vision_not_built());
#endif
}

void Vision::read_video(VideoJob& job, Cancel* cancel) {
#if LAMINA_VISION
    if (job.cached || !job.pixels.empty() || job.frames.empty()) return;
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<int64_t> distinct = job.frames;
    std::sort(distinct.begin(), distinct.end());
    distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
    // ffmpeg's frames at their stored size into a short queue, resized to the plan's by the workers as they come
    std::vector<std::vector<uint8_t>> resized(distinct.size());
    std::mutex qm;
    std::condition_variable qcv;
    std::deque<std::pair<size_t, std::vector<uint8_t>>> queue;
    bool done = false;
    std::string failed;
    const unsigned n_workers = std::max(1u, std::min(8u, std::thread::hardware_concurrency() / 2));
    const VideoInfo info = job.info;
    const int w = job.width, h = job.height;
    std::vector<std::thread> workers;
    for (unsigned i = 0; i < n_workers; ++i)
        workers.emplace_back([&] {
            for (;;) {
                std::pair<size_t, std::vector<uint8_t>> item;
                {
                    std::unique_lock<std::mutex> lk(qm);
                    qcv.wait(lk, [&] { return done || !queue.empty(); });
                    if (queue.empty()) return;
                    item = std::move(queue.front());
                    queue.pop_front();
                }
                qcv.notify_all();
                try {
                    vision::Rgb src;
                    src.w = info.stored_w;
                    src.h = info.stored_h;
                    src.px = std::move(item.second);
                    resized[item.first] = vision::resample(src, w, h, vision::Filter::bicubic).px;
                } catch (const std::exception& e) {
                    const std::lock_guard<std::mutex> lk(qm);
                    if (failed.empty()) failed = e.what();
                }
            }
        });
    struct Join {   // the workers end however the reading ends
        std::vector<std::thread>& ws;
        std::mutex& m;
        std::condition_variable& cv;
        bool& done;
        ~Join() {
            {
                const std::lock_guard<std::mutex> lk(m);
                done = true;
            }
            cv.notify_all();
            for (std::thread& t : ws) t.join();
        }
    } join{workers, qm, qcv, done};
    const size_t got = read_video_frames(
        ff_, job.file->path, info, job.first, distinct,
        [&](size_t k, std::vector<uint8_t>&& px) {
            std::unique_lock<std::mutex> lk(qm);
            qcv.wait(lk, [&] { return queue.size() < 2 * (size_t) n_workers; });
            queue.emplace_back(k, std::move(px));
            lk.unlock();
            qcv.notify_all();
        },
        cancel);
    {
        const std::lock_guard<std::mutex> lk(qm);
        done = true;
    }
    qcv.notify_all();
    for (std::thread& t : workers) t.join();
    workers.clear();
    if (!failed.empty()) throw std::invalid_argument("the video's frames could not be resized (" + failed + ")");
    for (size_t k = got; k < distinct.size(); ++k) resized[k] = resized[got - 1];   // past the stream's end: its last
    job.pixels.resize(job.frames.size());
    for (size_t i = 0; i < job.frames.size(); ++i) {
        const size_t k =
            (size_t) (std::lower_bound(distinct.begin(), distinct.end(), job.frames[i]) - distinct.begin());
        job.pixels[i] = resized[k];
    }
    std::fprintf(stderr, "lamina: a video's %zu frames (%zu read, %dx%d to %dx%d) in %.1f s\n", job.frames.size(), got,
                 info.stored_w, info.stored_h, w, h, seconds_since(t0));
#else
    (void) job;
    (void) cancel;
#endif
}

std::vector<EncodedMedium> Vision::encode_all(const std::vector<Medium>& media, Cancel* cancel) {
    // a borrowing encoder's lease holds for all of the request's new pictures and videos
    struct Hold {
        Vision& v;
        explicit Hold(Vision& vision) : v(vision) {
            const std::lock_guard<std::mutex> lk(v.mu_);
            ++v.holds_;
        }
        ~Hold() {
            const std::lock_guard<std::mutex> lk(v.mu_);
            if (--v.holds_ == 0) v.lease_end();
        }
    } hold(*this);
    std::vector<EncodedMedium> out;
    for (const Medium& m : media) {
        if (cancel != nullptr && cancel->is_set()) throw ClientGone();
        std::vector<std::string> keep;
        for (const EncodedMedium& e : out) keep.push_back(e.file);
        if (m.video != nullptr) {
            out.push_back(encode_video(*m.video, keep));
            continue;
        }
        const auto [file, n] = encode(m.source, m.is_bytes, keep);
        out.push_back({file, {n}, {fs::path(u8_path(file)).stem().string().substr(0, 16)}});
    }
    return out;
}

EncodedMedium Vision::encode_video(VideoJob& job, const std::vector<std::string>& keep) {
#if LAMINA_VISION
    constexpr uintmax_t kVideoCacheBytes = (uintmax_t) 4 << 30;   // the videos' rows kept on the drive at most
    constexpr size_t kVideoCacheItems = 32;
    {
        const std::lock_guard<std::mutex> lk(mu_);
        for (auto it = video_cache_.begin(); it != video_cache_.end(); ++it)
            if (it->first == job.key) {
                auto hit = std::move(*it);   // most recently used last
                video_cache_.erase(it);
                video_cache_.push_back(std::move(hit));
                return video_cache_.back().second.enc;
            }
    }
    // planned from the cache and dropped from it since: read now
    if (job.cached) {
        job.cached = false;
        read_video(job, nullptr);
    }
    std::lock_guard<std::mutex> lk(mu_);
    const auto t0 = std::chrono::steady_clock::now();
    struct LeaseEnd {   // a video encoded outside encode_all: its lease ends with it
        Vision& v;
        ~LeaseEnd() {
            if (v.holds_ == 0) v.lease_end();
        }
    } lease_end_{*this};
    std::string err;
    auto* encoder = (vision::Encoder*) encoder_now(err);
    if (encoder == nullptr) throw std::invalid_argument("the video could not be read: " + err);
    const fs::path out = u8_path(dir) / (job.key + "-video.sve");
    std::FILE* f = vision::open_utf8(u8_str(out), true);
    if (f == nullptr) throw std::invalid_argument("the video's embeddings could not be written to " + u8_str(out));
    struct Close {
        std::FILE*& f;
        ~Close() {
            if (f != nullptr) std::fclose(f);
        }
    } close{f};
    EncodedMedium enc;
    enc.file = u8_str(out);
    const int groups = job.groups();
    if (groups <= 0 || job.pixels.size() != job.frames.size())
        throw std::invalid_argument("the video could not be read: its frames were not read");
    for (int g = 0; g < groups; ++g) {
        vision::Rgb a;
        a.w = job.width;
        a.h = job.height;
        a.px = std::move(job.pixels[(size_t) g * (size_t) job.per_group]);
        vision::Rgb b;
        if (job.per_group == 2) {
            b.w = job.width;
            b.h = job.height;
            b.px = std::move(job.pixels[(size_t) g * 2 + 1]);
        }
        vision::Rows rows;
        if (!encoder->encode_frames(a, job.per_group == 2 ? &b : nullptr, rows, err))
            throw std::invalid_argument("the video could not be read: " + err);
        const int32_t header[5] = {0x31455653, rows.n, rows.nx, rows.ny, rows.n_embd};
        const size_t floats = (size_t) rows.n * (size_t) rows.n_embd;
        if (std::fwrite(header, sizeof header, 1, f) != 1 ||
            std::fwrite(rows.data.data(), sizeof(float), floats, f) != floats)
            throw std::invalid_argument("the video's embeddings could not be written to " + u8_str(out));
        enc.rows.push_back(rows.n);
        enc.keys.push_back(sha256_hex(job.key + "/" + std::to_string(g)).substr(0, 16));
    }
    if (std::fclose(f) != 0) {
        f = nullptr;
        throw std::invalid_argument("the video's embeddings could not be written to " + u8_str(out));
    }
    f = nullptr;
    if (lend_ && loaded_ != nullptr) loaded_->pictures += groups;
    job.pixels.clear();
    job.pixels.shrink_to_fit();
    int64_t tokens = 0;
    for (const int64_t n : enc.rows) tokens += n;
    std::fprintf(stderr, "lamina: a video's %d groups (%lld image tokens) encoded in %.1f s\n", groups,
                 (long long) tokens, seconds_since(t0));
    VideoEntry e;
    e.enc = enc;
    e.plan = job;
    e.plan.file.reset();
    e.plan.cached = false;
    std::error_code ec;
    e.bytes = fs::file_size(out, ec);
    video_cache_.emplace_back(job.key, std::move(e));
    // oldest first, never one this request uses
    uintmax_t total = 0;
    for (const auto& [k, x] : video_cache_) total += x.bytes;
    for (auto it = video_cache_.begin();
         it != video_cache_.end() && (total > kVideoCacheBytes || video_cache_.size() > kVideoCacheItems);) {
        if (it->first != job.key && std::find(keep.begin(), keep.end(), it->second.enc.file) == keep.end()) {
            total -= it->second.bytes;
            fs::remove(u8_path(it->second.enc.file), ec);
            it = video_cache_.erase(it);
        } else {
            ++it;
        }
    }
    return enc;
#else
    (void) job;
    (void) keep;
    throw std::invalid_argument(std::string("the video could not be read: ") + vision_not_built());
#endif
}

std::pair<std::string, int64_t> Vision::encode(const std::string& source, bool is_bytes,
                                               const std::vector<std::string>& keep) {
    const std::string data = normalize(is_bytes ? source : load(source));
    const std::string key = sha256_hex(data).substr(0, 32);
    std::lock_guard<std::mutex> lk(mu_);
    for (auto it = cache_.begin(); it != cache_.end(); ++it)
        if (it->first == key) {
            auto hit = *it;   // most recently used last
            cache_.erase(it);
            cache_.push_back(hit);
            return hit.second;
        }
    const fs::path out = u8_path(dir) / (key + ".sve");
    int64_t ntok = 0;
#if LAMINA_VISION
    vision::Rgb picture;
    vision::Rows rows;
    std::string err;
    if (!vision::decode_memory((const uint8_t*) data.data(), data.size(), picture, err))
        throw std::invalid_argument("the image could not be read (" + err + ")");
    struct LeaseEnd {   // a picture encoded outside encode_all: its lease ends with it
        Vision& v;
        ~LeaseEnd() {
            if (v.holds_ == 0) v.lease_end();
        }
    } lease_end_{*this};
    auto* encoder = (vision::Encoder*) encoder_now(err);
    if (encoder == nullptr) throw std::invalid_argument("the image could not be read: " + err);
    if (!encoder->encode(picture, rows, err)) throw std::invalid_argument("the image could not be read: " + err);
    if (lend_ && loaded_ != nullptr) ++loaded_->pictures;
    if (!vision::write_rows(u8_str(out), rows)) {
        std::error_code ec;
        fs::remove(out, ec);
        throw std::invalid_argument("the image could not be read: its embeddings could not be written to " +
                                    u8_str(out));
    }
    ntok = rows.n;
#else
    throw std::invalid_argument(std::string("the image could not be read: ") + vision_not_built());
#endif
    const std::pair<std::string, int64_t> result{u8_str(out), ntok};
    cache_.push_back({key, result});
    std::vector<std::string> held = keep;
    held.push_back(u8_str(out));
    for (auto it = cache_.begin(); it != cache_.end() && cache_.size() > 64;) {   // oldest first, never one in use
        if (std::find(held.begin(), held.end(), it->second.first) == held.end()) {
            std::error_code ec;
            fs::remove(u8_path(it->second.first), ec);
            it = cache_.erase(it);
        } else {
            ++it;
        }
    }
    return result;
}

}   // namespace lamina
