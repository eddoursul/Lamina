// src/vision/video.cpp - see include/lamina/vision/video.hpp.  The processors' arithmetic is kept expression for
// expression (Python's float and integer operations, round() and np.round to the even number at a half), so a video's
// frames and their sizes are the reference's.
#include "lamina/vision/video.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace lamina::vision {
namespace {

int64_t floor_div(int64_t a, int64_t b) { return a / b - ((a % b != 0) && ((a < 0) != (b < 0)) ? 1 : 0); }

bool sizes_ok(const VideoStream& s, std::string& err) {
    if (s.frames <= 0 || !(s.fps > 0) || !std::isfinite(s.fps) || s.width <= 0 || s.height <= 0) {
        err = "the video has no frames, frame rate or size to plan from";
        return false;
    }
    if ((double) std::max(s.width, s.height) / (double) std::min(s.width, s.height) > 200) {
        err = "the video's aspect ratio is over 200";
        return false;
    }
    return true;
}

// transformers' Qwen3VLVideoProcessor: sample_frames, resize (cap_pixels_per_frame) and smart_resize;
// Qwen3VLProcessor's _calculate_timestamps; over the frames `first` to `last`
bool plan_qwen(const ProjectorParams& p, const VideoStream& s, int64_t first, int64_t last, int max_tokens,
               VideoPlan& plan, std::string& err) {
    constexpr double kFps = 2.0;
    constexpr int64_t kMinFrames = 4, kMaxFrames = 768, kTemporal = 2, kFrameTokens = 768;
    constexpr int64_t kMinPixels = 4096, kMaxPixels = 25165824;   // the models' video_preprocessor_config "size"
    const int64_t factor = (int64_t) p.patch_size * p.n_merge, total = last - first + 1;
    // num_frames = min(max(int(total / fps * fps), min_frames), max_frames, total)
    int64_t n = (int64_t) ((double) total / s.fps * kFps);
    n = std::min({std::max(n, kMinFrames), kMaxFrames, total});
    // np.linspace(first, last, n).round() (transformers' from 0 to the video's last frame)
    std::vector<int64_t> idx((size_t) n);
    const double start = (double) first, stop = (double) last, step = n > 1 ? (stop - start) / (double) (n - 1) : 0.0;
    for (int64_t i = 0; i < n; ++i)
        idx[(size_t) i] = (int64_t) std::nearbyint(i + 1 == n && n > 1 ? stop : (double) i * step + start);
    // a frame alone is a pair of it (the processor refuses fewer frames than the temporal factor)
    const int64_t frames = std::max(n, kTemporal);
    // the cap: a frame's pixels at most max_video_tokens patches of the merged size or the budget's even share, at
    // least 1.05 times the smallest clip; the clip's budget that times the frames
    const int64_t cap_tokens = max_tokens > 0 ? std::min<int64_t>(kFrameTokens, max_tokens) : kFrameTokens;
    const int64_t per_frame =
        std::max(std::min(cap_tokens * factor * factor, kMaxPixels / frames), (int64_t) ((double) kMinPixels * 1.05));
    const int64_t max_pixels = per_frame * frames;
    // smart_resize(num_frames, height, width, temporal_factor, factor, min_pixels, max_pixels)
    int64_t height = s.height, width = s.width;
    if (height < factor || width < factor) {
        const double scale = std::max((double) factor / (double) height, (double) factor / (double) width);
        height = (int64_t) ((double) height * scale);
        width = (int64_t) ((double) width * scale);
    }
    int64_t h_bar = (int64_t) std::nearbyint((double) height / (double) factor) * factor;
    int64_t w_bar = (int64_t) std::nearbyint((double) width / (double) factor) * factor;
    const int64_t t_bar = (int64_t) std::nearbyint((double) frames / (double) kTemporal) * kTemporal;
    if (t_bar * h_bar * w_bar > max_pixels) {
        const double beta = std::sqrt((double) (frames * height * width) / (double) max_pixels);
        h_bar = std::max(factor, (int64_t) std::floor((double) height / beta / (double) factor) * factor);
        w_bar = std::max(factor, (int64_t) std::floor((double) width / beta / (double) factor) * factor);
    } else if (t_bar * h_bar * w_bar < kMinPixels) {
        const double beta = std::sqrt((double) kMinPixels / (double) (frames * height * width));
        h_bar = (int64_t) std::ceil((double) height * beta / (double) factor) * factor;
        w_bar = (int64_t) std::ceil((double) width * beta / (double) factor) * factor;
    }
    if (h_bar < factor || w_bar < factor) {
        err = "the video's frames cannot be brought to the projector's grid";
        return false;
    }
    // the frames padded to whole pairs with the last one, a pair's time the mean of its two
    while ((int64_t) idx.size() % kTemporal != 0) idx.push_back(idx.back());
    plan.frames = idx;
    plan.width = (int) w_bar;
    plan.height = (int) h_bar;
    plan.per_group = (int) kTemporal;
    plan.group_tokens = (int) ((h_bar / factor) * (w_bar / factor));
    plan.times.clear();
    for (size_t i = 0; i + 1 < idx.size(); i += 2)
        plan.times.push_back(((double) idx[i] / s.fps + (double) idx[i + 1] / s.fps) / 2.0);
    return true;
}

// the HunYuan team's HYV4VLVideoProcessor (vLLM): sample_frames and adaptive_resize with the checkpoint's
// video_processor section; HYV4VLProcessor's times (a frame a group: its index over the rate); over the frames
// `first` to `last`
bool plan_hy4(const ProjectorParams& p, const VideoStream& s, int64_t first, int64_t last, VideoPlan& plan,
              std::string& err) {
    constexpr double kFps = 2.0;
    constexpr int64_t kMinFrames = 2, kMaxFrames = 2048, kMaxPixels = 61440000, kFrameSide = 1024;
    constexpr int64_t kStampTokens = 6, kStampReserve = 1000;
    const int64_t factor = (int64_t) p.patch_size * p.n_merge, total = last - first + 1;
    // num_frames = min(total, max(min_frames, min(max_frames, floor(total / fps * fps))))
    const int64_t desired = (int64_t) std::floor((double) total / s.fps * kFps);
    const int64_t n = std::min(total, std::max(kMinFrames, std::min(kMaxFrames, desired)));
    // each segment's middle: min((round(segment i) + round(segment (i + 1))) // 2, total - 1)
    const double segment = (double) (total - 1) / (double) n;
    std::vector<int64_t> idx((size_t) n);
    for (int64_t i = 0; i < n; ++i) {
        const int64_t lo = (int64_t) std::nearbyint(segment * (double) i);
        const int64_t hi = (int64_t) std::nearbyint(segment * (double) (i + 1));
        idx[(size_t) i] = first + std::min(floor_div(lo + hi, 2), total - 1);
    }
    // adaptive_resize: a frame's tokens (cells and a newline a row) times the frames within the budget less the
    // timestamps' tokens, else a frame within kFrameSide^2 pixels; never larger
    const int64_t height = s.height, width = s.width;
    const int64_t frame_tokens = height * width / (factor * factor) + std::max(height, width) / factor;
    const int64_t budget = std::max<int64_t>(1, kMaxPixels / (factor * factor) - kStampReserve - kStampTokens * n);
    double scale = 1.0;
    if (frame_tokens * n > budget) scale = std::sqrt((double) budget / (double) (frame_tokens * n));
    else if (height * width > kFrameSide * kFrameSide)
        scale = (double) kFrameSide / std::sqrt((double) (height * width));
    const int64_t h = std::max(factor, (int64_t) ((double) height * scale) / factor * factor);
    const int64_t w = std::max(factor, (int64_t) ((double) width * scale) / factor * factor);
    if (n * h * w > kMaxPixels) {
        err = "the video's frames do not fit the processor's budget";
        return false;
    }
    plan.frames = idx;
    plan.width = (int) w;
    plan.height = (int) h;
    plan.per_group = 1;
    plan.group_tokens = (int) ((h / factor) * (w / factor + 1));
    plan.times.clear();
    for (const int64_t i : idx) plan.times.push_back((double) i / s.fps);
    return true;
}

}   // namespace

bool plan_video(const ProjectorParams& p, const VideoStream& s, int max_tokens, VideoPlan& plan, std::string& err,
                const VideoRange& range) {
    if (p.kind == ProjectorKind::inkling2v) {
        err = "this model's projector takes no video";
        return false;
    }
    if (!sizes_ok(s, err)) return false;
    // qwen-vl-utils' calculate_video_frame_range
    int64_t first = 0, last = s.frames - 1;
    const double duration = (double) s.frames / s.fps;
    if (range.start >= 0) first = (int64_t) std::ceil(std::clamp(range.start, 0.0, duration) * s.fps);
    if (range.end >= 0)
        last = std::min((int64_t) std::floor(std::clamp(range.end, 0.0, duration) * s.fps), s.frames - 1);
    if (range.any() && first >= last) {
        char buf[160];
        if (range.start >= duration)
            std::snprintf(buf, sizeof buf, "its part starts at %g seconds, past its end: it is %.1f s long",
                          range.start, duration);
        else
            std::snprintf(buf, sizeof buf, "its part from %g to %g seconds holds fewer than two frames",
                          std::max(range.start, 0.0), range.end >= 0 ? range.end : duration);
        err = buf;
        return false;
    }
    plan.first = first;
    return p.kind == ProjectorKind::qwen3vl ? plan_qwen(p, s, first, last, max_tokens, plan, err)
                                            : plan_hy4(p, s, first, last, plan, err);
}

std::string video_time_text(double seconds) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "<%.1f seconds>", seconds);
    return buf;
}

}   // namespace lamina::vision
