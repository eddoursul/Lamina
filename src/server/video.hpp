// src/server/video.hpp - a request's video through ffmpeg's programs: its file, its stream's facts, its frames.
//
// A video (a data: URL, an http(s) URL, a local file or a file:// URL) is a file ffmpeg reads: a local file in place,
// the others' bytes in a temporary file of the image encoder's directory.  ffprobe gives its first video stream's
// frames, frame rate and displayed size (the pixel aspect ratio and the rotation applied); ffmpeg decodes it once and
// hands over only the frames a plan names (its select filter: the others are decoded and dropped, never converted), as
// RGB bytes at the stored size, rotated as ffmpeg rotates by default.  The programs: the vision section's "ffmpeg" (its
// path, ffprobe beside it), else those on the PATH; ffmpeg 5.1 or newer (-fps_mode).
#pragma once

#include "sync.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace lamina {

/// a data: or http(s) URL's video is read up to this much (a local file is read by ffmpeg where it is)
inline constexpr size_t kVideoBytesMax = (size_t) 1 << 30;

/// ffmpeg and ffprobe, or why there are none
struct FfmpegTools {
    std::string ffmpeg, ffprobe, why;
    bool ok() const { return why.empty(); }
};
/// `configured`: the vision section's "ffmpeg" ("": the PATH's)
FfmpegTools find_ffmpeg(const std::string& configured);

/// the file ffmpeg reads: a local file, or a temporary copy of a URL's bytes (removed with this)
struct VideoFile {
    std::string path;
    std::string key;   ///< 32 hex digits: the bytes' SHA-256, or a local file's path, size and time
    bool temporary = false;
    VideoFile() = default;
    VideoFile(const VideoFile&) = delete;
    VideoFile& operator=(const VideoFile&) = delete;
    ~VideoFile();
};
/// `source` as such a file (`is_bytes`: the video's bytes themselves); std::invalid_argument when it cannot be read
std::shared_ptr<VideoFile> open_video(const std::string& source, bool is_bytes, const std::string& dir);

/// what ffprobe reports of the first video stream
struct VideoInfo {
    int64_t frames = 0;               ///< the frames it decodes to (the container's count, else its packets')
    double fps = 0;                   ///< its average frame rate
    int stored_w = 0, stored_h = 0;   ///< a decoded frame's size, rotated
    int shown_w = 0, shown_h = 0;     ///< its displayed size (the pixel aspect ratio applied)
};
/// std::invalid_argument when ffprobe cannot read it as a video
VideoInfo probe_video(const FfmpegTools& t, const std::string& path);

/// The frames `frames` (ascending, distinct indices) decoded, each to `on_frame(k, rgb)` in order (k: its place in
/// `frames`; rgb: stored_w x stored_h x 3 bytes) -> how many there were: fewer where the stream ends before the count
/// its container gave (at least one).  std::invalid_argument when ffmpeg fails; ClientGone when `cancel` is set.
size_t read_video_frames(const FfmpegTools& t, const std::string& path, const VideoInfo& info,
                         const std::vector<int64_t>& frames,
                         const std::function<void(size_t, std::vector<uint8_t>&&)>& on_frame, Cancel* cancel);

}   // namespace lamina
