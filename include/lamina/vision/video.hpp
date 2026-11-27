// include/lamina/vision/video.hpp - a video as its model's reference processor takes it: which of its frames, at what
// size, in which groups, at what times.
//
// Two rules, the projectors' (the frames a group goes through the encoder together: encoder.hpp's encode_frames):
//   qwen3vl_merger  transformers' Qwen3VLVideoProcessor and Qwen3VLProcessor with the models' video_preprocessor_config
//                   (a clip of 4,096 to 25,165,824 pixels): 2 frames a second, 4 to 768 of them, np.linspace's indices;
//                   each frame under the per-frame cap of Qwen's own processor (qwen-vl-utils, transformers'
//                   cap_pixels_per_frame: 768 tokens, or the budget's even share) and the clip within the budget
//                   (smart_resize with the temporal factor); two frames a group (a pair through the two patch kernels,
//                   the last frame repeated for an odd count), the group's time the mean of its two frames' times.
//   hy_v4_flash     the HunYuan team's processor for HY4's video (vLLM's transformers_utils/processors/hy_v4.py:
//                   sample_frames, adaptive_resize, get_replacement) with the checkpoint's video_processor section
//                   (61,440,000 pixels, 2 to 2,048 frames, 6 timestamp tokens a frame and 1,000 held back, a frame at
//                   most 1,024 x 1,024 pixels): 2 frames a second, each the middle of its segment; one frame a group.
// Inkling 2 Mini's projector takes no video.
//
// A part of a video (VideoRange: qwen-vl-utils' video_start and video_end) is bounded as qwen-vl-utils'
// calculate_video_frame_range bounds it: its first frame ceil(start * fps), its last floor(end * fps), each within the
// video, at least two frames. Its frames are planned as a whole video of that many frames is, at their own indices
// (Qwen's np.linspace from its first to its last frame, as qwen-vl-utils' readers index a part; Hy4's segments over
// it, from its first), so its times are the video's.
#pragma once

#include "lamina/vision/projector.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace lamina::vision {

/// A video stream as its file describes it.
struct VideoStream {
    int64_t frames = 0;          ///< the frames it decodes to
    double fps = 0;              ///< frames a second (the stream's average rate)
    int width = 0, height = 0;   ///< a frame's displayed size (its pixel aspect ratio and rotation applied)
};

/// A part of a video, in seconds from its start: `start` to `end` (either one < 0: the video's own).
struct VideoRange {
    double start = -1, end = -1;
    bool any() const { return start >= 0 || end >= 0; }
};

/// How a video reaches the model.
struct VideoPlan {
    int64_t first = 0;             ///< the first frame of the range (0 for the whole video), where a reader seeks
    std::vector<int64_t> frames;   ///< the frames to decode, by index, in order (an index can repeat)
    int width = 0, height = 0;     ///< the size each frame is resized to (whole merged patches)
    int per_group = 1;             ///< frames a group: Qwen's 2 (a temporal pair), Hy4's 1
    std::vector<double> times;     ///< each group's time in seconds
    int group_tokens = 0;          ///< each group's rows (Hy4's: its cells and a newline row after each row of them)
    /// the groups: frames.size() / per_group
    int groups() const { return per_group > 0 ? (int) (frames.size() / (size_t) per_group) : 0; }
};

/// The plan for stream `s` (its part `range`) under projector `p`; `max_tokens` (a picture's limit, 0: none) caps
/// Qwen's tokens a group too.  False with `err` for a projector that takes no video, a stream it cannot size, or a
/// range of fewer than two frames.
bool plan_video(const ProjectorParams& p, const VideoStream& s, int max_tokens, VideoPlan& plan, std::string& err,
                const VideoRange& range = {});

/// A group's time as the processors write it before the group: "<1.5 seconds>" (Python's f"<{t:.1f} seconds>")
std::string video_time_text(double seconds);

}   // namespace lamina::vision
