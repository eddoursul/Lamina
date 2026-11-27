// include/lamina/vision/image.hpp - a picture on its way into the image encoder: decoded to RGB bytes, brought to the
// size its projector takes, and normalized to float planes.
//
// The sizes and the resampling are what the projectors' reference processors compute, as llama.cpp's mtmd computes
// them (tools/mtmd/mtmd-image.cpp, MIT): the target size is transformers' smart_resize (each side to a multiple of
// the merged patch, the area within the projector's pixel limits), the resampling Pillow's (separable, the filter
// widened when a side shrinks, 22-bit fixed-point weights), so a picture's bytes after it are the same on every
// machine.  The arithmetic is kept expression for expression: an encoder's rows are compared with mtmd's bit for bit.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace lamina::vision {

/// A file by its UTF-8 path, for reading or (made anew) for writing, in binary; nullptr when it cannot be opened.
std::FILE* open_utf8(const std::string& path, bool write);

/// A picture's pixels: 3 bytes each (R, G, B), rows top to bottom.
struct Rgb {
    int w = 0, h = 0;
    std::vector<uint8_t> px;
};

/// The first frame of a picture file as stb_image reads it (PNG, JPEG, BMP, GIF, ...), any alpha dropped.  False
/// with `err` for a file that is not there or not such a picture.
bool decode_file(const std::string& path, Rgb& out, std::string& err);
bool decode_memory(const uint8_t* data, size_t n, Rgb& out, std::string& err);

enum class Filter { bilinear, bicubic, lanczos };

/// Pillow's Image.resize to w x h (its Resample.c: bicubic with a = -0.5, Lanczos over 3 lobes).
Rgb resample(const Rgb& src, int w, int h, Filter filter);

/// How a projector sizes a picture: sides in multiples of `align` pixels, the area from `min_pixels` to `max_pixels`.
struct SizeRule {
    int align = 32;
    int min_pixels = 0;       ///< 0: no lower limit
    int max_pixels = 0;       ///< 0: no upper limit
    bool half_even = false;   ///< a side that is a whole number of aligns and a half rounds to the even one (Python's
                              ///< round() in transformers' smart_resize), not away from zero
};
/// The size a w x h picture is brought to (smart_resize): the sides rounded to `align`, then scaled down to at most
/// max_pixels (the sides floored) or up to at least min_pixels (the sides ceiled), the ratio kept as the grid allows.
void target_size(int w, int h, const SizeRule& rule, int& tw, int& th);

/// `src` at tw x th.  `letterbox`: scaled to fit inside with its ratio kept and centred, the rest `pad`; without it
/// resized straight to that size.
Rgb fit(const Rgb& src, int tw, int th, Filter filter, bool letterbox, const std::array<uint8_t, 3>& pad);

/// The picture as the encoder's input: (v / 255 - mean) / std, a plane a channel (R, G, B), each rows top to bottom.
std::vector<float> planes(const Rgb& img, const float mean[3], const float std[3]);

/// A picture as Inkling 2's processor gives it (mtmd's mtmd_image_preprocessor_inkling2v): at its own size - resized
/// (target_size, `filter`) only where its sides rounded up to `rule.align` pass rule.max_pixels - then padded right
/// and below to whole aligns with a raw value of -1, and normalized as (v - 255 mean) / (255 std).  w x h: the
/// padded size.  Planes as planes().
std::vector<float> planes_padded(const Rgb& src, const SizeRule& rule, Filter filter, const float mean[3],
                                 const float std[3], int& w, int& h);

}   // namespace lamina::vision
