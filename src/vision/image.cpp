// src/vision/image.cpp - see include/lamina/vision/image.hpp.
#include "lamina/vision/image.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

// stb_image, its functions this file's own (the server has a copy of its own for the formats it converts)
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#if defined(_MSC_VER)
#pragma warning(push, 0)
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#include "stb/stb_image.h"
#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace lamina::vision {

std::FILE* open_utf8(const std::string& path, bool write) {
#ifdef _WIN32
    const int n = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    if (n <= 0) return nullptr;
    std::vector<wchar_t> wide((size_t) n);
    if (MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wide.data(), n) <= 0) return nullptr;
    return _wfopen(wide.data(), write ? L"wb" : L"rb");
#else
    return std::fopen(path.c_str(), write ? "wb" : "rb");
#endif
}

bool decode_memory(const uint8_t* data, size_t n, Rgb& out, std::string& err) {
    int w = 0, h = 0, channels = 0;
    unsigned char* px = n <= (size_t) INT32_MAX ? stbi_load_from_memory(data, (int) n, &w, &h, &channels, 3) : nullptr;
    if (px == nullptr) {
        const char* why = n <= (size_t) INT32_MAX ? stbi_failure_reason() : "too large";
        err = std::string("not a picture this decoder reads (") + (why != nullptr ? why : "unknown format") + ")";
        return false;
    }
    out.w = w;
    out.h = h;
    out.px.assign(px, px + (size_t) w * (size_t) h * 3);
    stbi_image_free(px);
    return true;
}

bool decode_file(const std::string& path, Rgb& out, std::string& err) {
    std::FILE* f = open_utf8(path, false);
    if (f == nullptr) {
        err = "cannot open " + path;
        return false;
    }
    std::vector<uint8_t> buf;
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    bool ok = size >= 0;
    if (ok) {
        buf.resize((size_t) size);
        ok = std::fread(buf.data(), 1, buf.size(), f) == buf.size();
    }
    std::fclose(f);
    if (!ok) {
        err = "cannot read " + path;
        return false;
    }
    return decode_memory(buf.data(), buf.size(), out, err);
}

// Pillow's separable resampling (src/libImaging/Resample.c) as mtmd's img_tool::resize_pillow has it: a horizontal
// pass, then a vertical one, each output pixel a normalized sum over the filter's support in the input, the weights
// in fixed point (22 fractional bits = 32 - 8 for a pixel - 2 of headroom).
Rgb resample(const Rgb& img, int target_width, int target_height, Filter filter) {
    const int PRECISION_BITS = 32 - 8 - 2;

    double filter_support;
    switch (filter) {
        case Filter::bilinear: filter_support = 1.0; break;
        case Filter::bicubic: filter_support = 2.0; break;
        case Filter::lanczos: filter_support = 3.0; break;
        default: throw std::runtime_error("unsupported resize filter");
    }

    // the filter's weight at distance x from the pixel's centre (bicubic: Pillow's a = -0.5)
    auto resample_filter = [filter](double x) -> double {
        if (filter == Filter::lanczos) {
            if (-3.0 <= x && x < 3.0) {
                auto sinc = [](double v) {
                    if (v == 0.0) {
                        return 1.0;
                    }
                    const double pi_v = v * 3.141592653589793238462643383279502884;
                    return std::sin(pi_v) / pi_v;
                };
                return sinc(x) * sinc(x / 3.0);
            }
            return 0.0;
        }

        if (x < 0.0) {
            x = -x;
        }

        if (filter == Filter::bilinear) {
            return x < 1.0 ? 1.0 - x : 0.0;
        }

        constexpr double a = -0.5;
        if (x < 1.0) {
            return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
        }
        if (x < 2.0) {
            return (((x - 5) * x + 8) * x - 4) * a;
        }
        return 0.0;
    };

    auto clip8 = [](int val) -> uint8_t {
        if (val < 0) return 0;
        if (val > 255) return 255;
        return static_cast<uint8_t>(val);
    };

    // One dimension's coefficients: for output pixel xx the input pixels bounds[2 xx] .. + bounds[2 xx + 1] and their
    // weights[xx * ksize + ..]; returns ksize, the most input pixels an output pixel takes.
    auto precompute_weights = [&](int inSize, int outSize, std::vector<int>& bounds,
                                  std::vector<int32_t>& weights) -> int {
        double support, scale, filterscale;
        double center, ww, ss;
        int xx, x, ksize, xmin, xmax;

        filterscale = scale = static_cast<double>(inSize) / outSize;
        if (filterscale < 1.0) {
            filterscale = 1.0;   // enlarging keeps the filter's own width
        }

        support = filter_support * filterscale;
        ksize = static_cast<int>(std::ceil(support)) * 2 + 1;

        std::vector<double> pre_weights((size_t) outSize * ksize);
        bounds.resize((size_t) outSize * 2);

        for (xx = 0; xx < outSize; xx++) {
            center = (xx + 0.5) * scale;
            ww = 0.0;
            ss = 1.0 / filterscale;

            xmin = static_cast<int>(center - support + 0.5);
            if (xmin < 0) {
                xmin = 0;
            }

            xmax = static_cast<int>(center + support + 0.5);
            if (xmax > inSize) {
                xmax = inSize;
            }

            xmax -= xmin;

            for (x = 0; x < xmax; x++) {
                double w = resample_filter((x + xmin - center + 0.5) * ss);
                pre_weights[(size_t) xx * ksize + x] = w;
                ww += w;
            }

            for (x = 0; x < xmax; x++) {
                if (ww != 0.0) {
                    pre_weights[(size_t) xx * ksize + x] /= ww;
                }
            }

            for (; x < ksize; x++) {
                pre_weights[(size_t) xx * ksize + x] = 0;
            }

            bounds[(size_t) xx * 2 + 0] = xmin;
            bounds[(size_t) xx * 2 + 1] = xmax;
        }

        weights.resize((size_t) outSize * ksize);

        const double fxp_scale = std::ldexp(1.0, PRECISION_BITS);

        for (size_t i = 0; i < (size_t) outSize * ksize; i++) {
            // Pillow adds +/- 0.5, then truncates toward zero
            const double rounded = pre_weights[i] * fxp_scale + (pre_weights[i] < 0 ? -0.5 : 0.5);
            weights[i] = static_cast<int32_t>(rounded);
        }

        return ksize;
    };

    auto resample_horizontal = [&](const uint8_t* src, int in_nx, int in_ny, int out_nx, int ksize,
                                   const std::vector<int>& bounds, const std::vector<int32_t>& weights) {
        std::vector<uint8_t> out((size_t) out_nx * in_ny * 3);

        for (int yy = 0; yy < in_ny; yy++) {
            const uint8_t* src_row = src + (size_t) yy * in_nx * 3;
            uint8_t* dst_row = out.data() + (size_t) yy * out_nx * 3;

            for (int xx = 0; xx < out_nx; xx++) {
                const int xmin = bounds[(size_t) xx * 2 + 0];
                const int xcnt = bounds[(size_t) xx * 2 + 1];
                const int32_t* k = &weights[(size_t) xx * ksize];
                const uint8_t* p = src_row + (size_t) xmin * 3;

                // the sums start at a half in fixed point: rounding
                int32_t ss0 = 1 << (PRECISION_BITS - 1);
                int32_t ss1 = 1 << (PRECISION_BITS - 1);
                int32_t ss2 = 1 << (PRECISION_BITS - 1);

                for (int x = 0; x < xcnt; x++) {
                    ss0 += p[0] * k[x];
                    ss1 += p[1] * k[x];
                    ss2 += p[2] * k[x];
                    p += 3;
                }

                dst_row[xx * 3 + 0] = clip8(ss0 >> PRECISION_BITS);
                dst_row[xx * 3 + 1] = clip8(ss1 >> PRECISION_BITS);
                dst_row[xx * 3 + 2] = clip8(ss2 >> PRECISION_BITS);
            }
        }

        return out;
    };

    auto resample_vertical = [&](const uint8_t* src, int in_nx, int out_ny, int ksize, const std::vector<int>& bounds,
                                 const std::vector<int32_t>& weight) {
        const size_t row_elems = (size_t) in_nx * 3;
        std::vector<uint8_t> out(row_elems * out_ny);
        std::vector<int32_t> acc(row_elems);

        for (int yy = 0; yy < out_ny; yy++) {
            const int ymin = bounds[(size_t) yy * 2 + 0];
            const int ycnt = bounds[(size_t) yy * 2 + 1];
            const int32_t* k = &weight[(size_t) yy * ksize];

            std::fill(acc.begin(), acc.end(), 1 << (PRECISION_BITS - 1));

            for (int y = 0; y < ycnt; y++) {
                const uint8_t* src_row = src + (size_t) (ymin + y) * row_elems;
                const int32_t w = k[y];
                for (size_t i = 0; i < row_elems; i++) {
                    acc[i] += src_row[i] * w;
                }
            }

            uint8_t* dst_row = out.data() + (size_t) yy * row_elems;
            for (size_t i = 0; i < row_elems; i++) {
                dst_row[i] = clip8(acc[i] >> PRECISION_BITS);
            }
        }

        return out;
    };

    if (img.w <= 0 || img.h <= 0 || target_width <= 0 || target_height <= 0 || target_width > 65536 ||
        target_height > 65536)
        throw std::runtime_error("a picture of " + std::to_string(img.w) + "x" + std::to_string(img.h) + " to " +
                                 std::to_string(target_width) + "x" + std::to_string(target_height) + ": out of range");

    const int src_width = img.w;
    const int src_height = img.h;

    const bool need_horizontal = (target_width != src_width);
    const bool need_vertical = (target_height != src_height);

    std::vector<int> bounds_horiz, bounds_vert;
    std::vector<int32_t> weights_horiz, weights_vert;
    int ksize_horiz = 0, ksize_vert = 0;

    if (need_horizontal) {
        ksize_horiz = precompute_weights(src_width, target_width, bounds_horiz, weights_horiz);
    }

    if (need_vertical) {
        ksize_vert = precompute_weights(src_height, target_height, bounds_vert, weights_vert);
    }

    Rgb dst;
    const uint8_t* src = img.px.data();
    if (need_horizontal && need_vertical) {
        auto temp =
            resample_horizontal(src, src_width, src_height, target_width, ksize_horiz, bounds_horiz, weights_horiz);
        dst.w = target_width;
        dst.h = target_height;
        dst.px = resample_vertical(temp.data(), target_width, target_height, ksize_vert, bounds_vert, weights_vert);
    } else if (need_horizontal) {
        dst.w = target_width;
        dst.h = src_height;
        dst.px =
            resample_horizontal(src, src_width, src_height, target_width, ksize_horiz, bounds_horiz, weights_horiz);
    } else if (need_vertical) {
        dst.w = src_width;
        dst.h = target_height;
        dst.px = resample_vertical(src, src_width, target_height, ksize_vert, bounds_vert, weights_vert);
    } else {
        dst = img;
    }
    return dst;
}

// mtmd's img_tool::calc_size_preserved_ratio, in its float arithmetic
void target_size(int width, int height, const SizeRule& rule, int& tw, int& th) {
    tw = th = 0;
    if (width <= 0 || height <= 0 || rule.align <= 0) return;

    auto round_by_factor = [f = rule.align, even = rule.half_even](float x) {
        const float q = x / static_cast<float>(f);
        return static_cast<int>(even ? std::nearbyint(q) : std::round(q)) * f;
    };
    auto ceil_by_factor = [f = rule.align](float x) {
        return static_cast<int>(std::ceil(x / static_cast<float>(f))) * f;
    };
    auto floor_by_factor = [f = rule.align](float x) {
        return static_cast<int>(std::floor(x / static_cast<float>(f))) * f;
    };

    int w_bar = std::max(rule.align, round_by_factor(static_cast<float>(width)));
    int h_bar = std::max(rule.align, round_by_factor(static_cast<float>(height)));

    if (rule.max_pixels > 0 && h_bar * w_bar > rule.max_pixels) {
        const auto beta = std::sqrt(static_cast<float>(height) * width / rule.max_pixels);
        h_bar = std::max(rule.align, floor_by_factor(height / beta));
        w_bar = std::max(rule.align, floor_by_factor(width / beta));
    } else if (rule.min_pixels > 0 && h_bar * w_bar < rule.min_pixels) {
        const auto beta = std::sqrt(static_cast<float>(rule.min_pixels) / (static_cast<float>(height) * width));
        h_bar = ceil_by_factor(height * beta);
        w_bar = ceil_by_factor(width * beta);
    }
    tw = w_bar;
    th = h_bar;
}

// mtmd's img_tool::resize: PAD_NONE, or PAD_CEIL with the picture centred on the pad colour
Rgb fit(const Rgb& src, int tw, int th, Filter filter, bool letterbox, const std::array<uint8_t, 3>& pad) {
    if (src.w == tw && src.h == th) return src;
    if (!letterbox) return resample(src, tw, th, filter);

    const float scale_w = static_cast<float>(tw) / src.w;
    const float scale_h = static_cast<float>(th) / src.h;
    const float scale = std::min(scale_w, scale_h);
    const int new_width = std::min(static_cast<int>(std::ceil(src.w * scale)), tw);
    const int new_height = std::min(static_cast<int>(std::ceil(src.h * scale)), th);
    const Rgb resized = resample(src, new_width, new_height, filter);

    Rgb dst;
    dst.w = tw;
    dst.h = th;
    dst.px.resize((size_t) tw * (size_t) th * 3);
    for (size_t i = 0; i < (size_t) tw * (size_t) th; ++i) {
        dst.px[3 * i] = pad[0];
        dst.px[3 * i + 1] = pad[1];
        dst.px[3 * i + 2] = pad[2];
    }
    const int offset_x = (tw - new_width) / 2;
    const int offset_y = (th - new_height) / 2;
    for (int y = 0; y < resized.h; ++y) {
        const int dy = y + offset_y;
        if (dy < 0 || dy >= th) continue;
        for (int x = 0; x < resized.w; ++x) {
            const int dx = x + offset_x;
            if (dx < 0 || dx >= tw) continue;
            std::memcpy(&dst.px[((size_t) dy * tw + dx) * 3], &resized.px[((size_t) y * resized.w + x) * 3], 3);
        }
    }
    return dst;
}

std::vector<float> planes(const Rgb& img, const float mean[3], const float std[3]) {
    const size_t n = (size_t) img.w * (size_t) img.h;
    std::vector<float> out(3 * n);
    for (size_t i = 0; i < n; ++i) {
        for (int c = 0; c < 3; ++c) {
            // mtmd's two steps (clip_image_f32::from_u8, then normalize), a float between them
            const float v = (float) img.px[3 * i + c] / 255.0f;
            out[(size_t) c * n + i] = (v - mean[c]) / std[c];
        }
    }
    return out;
}

std::vector<float> planes_padded(const Rgb& src, const SizeRule& rule, Filter filter, const float mean[3],
                                 const float std[3], int& w, int& h) {
    const int a = rule.align;
    auto up = [a](int v) { return (v + a - 1) / a * a; };
    int sw = src.w, sh = src.h;
    Rgb resized;
    const bool too_large = rule.max_pixels > 0 && (int64_t) up(sw) * up(sh) > rule.max_pixels;
    if (too_large) {
        target_size(src.w, src.h, rule, sw, sh);
        resized = resample(src, sw, sh, filter);
    }
    const Rgb& img = too_large ? resized : src;
    w = up(sw);
    h = up(sh);
    const size_t n = (size_t) w * (size_t) h;
    std::vector<float> out(3 * n);
    for (int c = 0; c < 3; ++c) {
        // mtmd's: the bytes as floats, the padding -1, then one step of (v - mean) / std with both times 255
        const float m = mean[c] * 255.0f, s = std[c] * 255.0f;
        float* plane = out.data() + (size_t) c * n;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const float v =
                    x < sw && y < sh ? (float) img.px[((size_t) y * (size_t) sw + (size_t) x) * 3 + (size_t) c] : -1.0f;
                plane[(size_t) y * (size_t) w + (size_t) x] = (v - m) / s;
            }
    }
    return out;
}

}   // namespace lamina::vision
