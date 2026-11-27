// src/server/video.cpp - a request's video through ffmpeg's programs (video.hpp).
#include "video.hpp"

#include "base64.hpp"
#include "engine.hpp"
#include "http_client.hpp"
#include "json.hpp"
#include "process.hpp"
#include "sha256.hpp"
#include "vision.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <thread>

namespace lamina {
namespace fs = std::filesystem;
namespace {

void write_file(const fs::path& p, std::string_view data) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    if (f) f.write(data.data(), (std::streamsize) data.size());
    if (!f) throw std::runtime_error("the video could not be written to " + u8_str(p));
}

/// "24000/1001" (or "25") as a number; 0 for "0/0", "N/A" and the like
double rational(const json& v) {
    if (!v.is_string()) return 0;
    const std::string s = v.get<std::string>();
    double num = 0, den = 1;
    if (std::sscanf(s.c_str(), "%lf/%lf", &num, &den) < 1 || !(den > 0) || !(num > 0) || !std::isfinite(num / den))
        return 0;
    return num / den;
}

int64_t integer(const json& v) {
    if (v.is_number_integer()) return v.get<int64_t>();
    if (v.is_string()) {
        try {
            return std::stoll(v.get<std::string>());
        } catch (const std::exception&) {}
    }
    return 0;
}

double number(const json& v) {
    if (v.is_number()) return v.get<double>();
    if (v.is_string()) {
        try {
            return std::stod(v.get<std::string>());
        } catch (const std::exception&) {}
    }
    return 0;
}

/// a program's standard output whole (its lines joined by \n), or std::invalid_argument with its error output
std::string run_output(const std::vector<std::string>& args, const char* what) {
    SpawnOptions o;
    o.args = args;
    o.in = SpawnOptions::null;
    o.out = SpawnOptions::pipe;
    o.err = SpawnOptions::pipe;
    o.no_window = true;
    std::unique_ptr<Process> p;
    try {
        p = std::make_unique<Process>(o);
    } catch (const std::exception& e) {
        throw std::invalid_argument(std::string(what) + " could not start (" + e.what() + ")");
    }
    std::string out, line;
    while (p->read_line(line, -1) == 1) out += line + "\n";
    const std::optional<int> code = p->wait();
    if (!code || *code != 0) {
        const std::string tail = p->stderr_tail();
        throw std::invalid_argument(std::string(what) + " could not read the video" +
                                    (tail.empty() ? std::string() : ": " + tail));
    }
    return out;
}

}   // namespace

FfmpegTools find_ffmpeg(const std::string& configured) {
    FfmpegTools t;
    if (!configured.empty()) {
        std::error_code ec;
        const fs::path p = u8_path(configured);
        if (!fs::is_regular_file(p, ec)) {
            t.why = "the config's vision \"ffmpeg\" is not a file: " + configured;
            return t;
        }
        t.ffmpeg = configured;
        const fs::path probe = p.parent_path() / ("ffprobe" + u8_str(p.extension()));
        t.ffprobe = fs::is_regular_file(probe, ec) ? u8_str(probe) : which("ffprobe");
    } else {
        t.ffmpeg = which("ffmpeg");
        t.ffprobe = which("ffprobe");
    }
    if (t.ffmpeg.empty() || t.ffprobe.empty())
        t.why = std::string("videos are read through ffmpeg and ffprobe (5.1 or newer), and there is no ") +
                (t.ffmpeg.empty() ? "ffmpeg" : "ffprobe") +
                (configured.empty() ? " on this computer's PATH: install ffmpeg, or name it in the config's vision "
                                      "\"ffmpeg\""
                                    : " beside the config's vision \"ffmpeg\" or on the PATH");
    return t;
}

VideoFile::~VideoFile() {
    if (!temporary || path.empty()) return;
    std::error_code ec;
    fs::remove(u8_path(path), ec);
}

std::shared_ptr<VideoFile> open_video(const std::string& source, bool is_bytes, const std::string& dir) {
    auto f = std::make_shared<VideoFile>();
    const auto keep_bytes = [&](const std::string& bytes) {
        if (bytes.empty()) throw std::invalid_argument("the video is empty");
        const fs::path p = u8_path(dir) / ("video-" + random_hex(12) + ".bin");
        f->path = u8_str(p);
        f->temporary = true;   // (removed with f, also when the write fails)
        write_file(p, bytes);
        f->key = sha256_hex(bytes).substr(0, 32);
    };
    if (is_bytes) {
        keep_bytes(source);
    } else if (starts_with(source, "data:")) {
        const size_t comma = source.find(',');
        std::string bytes;
        try {
            if (comma == std::string::npos) throw std::invalid_argument("list index out of range");
            bytes = b64decode(std::string_view(source).substr(comma + 1));
        } catch (const std::invalid_argument& e) {
            throw std::invalid_argument(std::string("the video's data: URL could not be read (") + e.what() + ")");
        }
        keep_bytes(bytes);
    } else if (starts_with(source, "http://") || starts_with(source, "https://")) {
        const std::string too_big = "the video URL's file is over " + std::to_string(kVideoBytesMax >> 20) + " MiB";
        HttpReply r;
        try {
            r = http_request("GET", source, {{"User-Agent", "lamina"}}, "", 600, kVideoBytesMax);
        } catch (const HttpError& e) {
            if (std::string(e.what()).find("larger than") != std::string::npos) throw std::invalid_argument(too_big);
            throw std::invalid_argument(std::string("the video URL could not be read: ") + e.what());
        }
        if (r.status >= 400)
            throw std::invalid_argument("the video URL could not be read: HTTP Error " + std::to_string(r.status) +
                                        ": " + r.reason);
        keep_bytes(r.body);
    } else {
        const std::string path = media_file_path(source, "a video");
        std::error_code ec;
        const fs::path p = u8_path(path);
        if (path.empty()) throw std::invalid_argument("a video must be a data: URL, an http(s) URL or a file's path");
        if (!fs::is_regular_file(p, ec))
            throw std::invalid_argument("the video file " + path + " is not there or cannot be read");
        const uintmax_t size = fs::file_size(p, ec);
        const long long time = (long long) fs::last_write_time(p, ec).time_since_epoch().count();
        f->path = path;
        f->key = sha256_hex(u8_str(fs::absolute(p, ec)) + "\n" + std::to_string(size) + "\n" + std::to_string(time))
                     .substr(0, 32);
    }
    return f;
}

VideoInfo probe_video(const FfmpegTools& t, const std::string& path) {
    const auto probe = [&](const std::vector<std::string>& more) {
        std::vector<std::string> args = {t.ffprobe, "-v", "error", "-select_streams", "v:0"};
        args.insert(args.end(), more.begin(), more.end());
        args.insert(args.end(), {"-of", "json", path});
        json j;
        if (!py_loads(run_output(args, "ffprobe"), j) || !j.is_object())
            throw std::invalid_argument("ffprobe's report on the video could not be read");
        return j;
    };
    const json j = probe({"-show_entries",
                          "stream=width,height,sample_aspect_ratio,avg_frame_rate,r_frame_rate,nb_frames,duration:"
                          "stream_side_data=rotation:format=duration"});
    const json& streams = jget(j, "streams");
    if (!streams.is_array() || streams.empty() || !streams[0].is_object())
        throw std::invalid_argument("the file has no video stream ffprobe reads");
    const json& s = streams[0];
    VideoInfo v;
    v.stored_w = (int) integer(jget(s, "width"));
    v.stored_h = (int) integer(jget(s, "height"));
    double duration = number(jget(s, "duration"));
    if (!(duration > 0)) duration = number(jget(jget(j, "format"), "duration"));
    v.fps = rational(jget(s, "avg_frame_rate"));
    if (!(v.fps > 0)) v.fps = rational(jget(s, "r_frame_rate"));
    v.frames = integer(jget(s, "nb_frames"));
    if (v.frames <= 0) {   // a container that does not count them (Matroska, WebM): its packets, a frame each
        const json c = probe({"-count_packets", "-show_entries", "stream=nb_read_packets"});
        const json& cs = jget(c, "streams");
        if (cs.is_array() && !cs.empty()) v.frames = integer(jget(cs[0], "nb_read_packets"));
    }
    if (v.frames <= 0 && duration > 0 && v.fps > 0) v.frames = (int64_t) std::llround(duration * v.fps);
    if (!(v.fps > 0) && v.frames > 0 && duration > 0) v.fps = (double) v.frames / duration;
    if (v.stored_w <= 0 || v.stored_h <= 0 || v.frames <= 0 || !(v.fps > 0))
        throw std::invalid_argument("ffprobe reports no frames, frame rate or size for the video");
    // the displayed size: the pixel aspect ratio (a plausible one) applied to the width; turned as ffmpeg turns the
    // frames by the stream's rotation
    v.shown_w = v.stored_w;
    v.shown_h = v.stored_h;
    if (const std::string sar = jstr(s, "sample_aspect_ratio"); !sar.empty()) {
        int num = 0, den = 0;
        if (std::sscanf(sar.c_str(), "%d:%d", &num, &den) == 2 && num > 0 && den > 0) {
            const double r = (double) num / den;
            if (r >= 0.1 && r <= 10.0) v.shown_w = std::max(1, (int) std::lround(v.stored_w * r));
        }
    }
    int rotation = 0;
    if (const json& sd = jget(s, "side_data_list"); sd.is_array())
        for (const auto& d : sd)
            if (d.is_object() && d.contains("rotation")) rotation = (int) integer(d["rotation"]);
    if (((rotation % 180) + 180) % 180 == 90) {
        std::swap(v.stored_w, v.stored_h);
        std::swap(v.shown_w, v.shown_h);
    }
    return v;
}

size_t read_video_frames(const FfmpegTools& t, const std::string& path, const VideoInfo& info,
                         const std::vector<int64_t>& frames,
                         const std::function<void(size_t, std::vector<uint8_t>&&)>& on_frame, Cancel* cancel) {
    if (frames.empty()) return 0;
    // the planned frames alone leave the filter graph (its select filter, n counting the decoded frames), at the
    // probed size, one each, as they come (no frame rate's duplicates)
    std::string select = "select=";
    for (size_t k = 0; k < frames.size(); ++k) select += (k ? "+not(n-" : "not(n-") + std::to_string(frames[k]) + ")";
    const std::string scale = "scale=" + std::to_string(info.stored_w) + ":" + std::to_string(info.stored_h);
    const unsigned threads = std::max(1u, std::thread::hardware_concurrency() / 2);   // half the CPU for the rest
    SpawnOptions o;
    o.args = {t.ffmpeg,    "-nostdin",    "-v",
              "error",     "-threads",    std::to_string(threads),
              "-i",        path,          "-map",
              "0:v:0",     "-an",         "-sn",
              "-dn",       "-vf",         select + "," + scale,
              "-fps_mode", "passthrough", "-f",
              "rawvideo",  "-pix_fmt",    "rgb24",
              "pipe:1"};
    o.in = SpawnOptions::null;
    o.out = SpawnOptions::pipe;
    o.raw_out = true;
    o.err = SpawnOptions::pipe;
    o.no_window = true;
    std::unique_ptr<Process> p;
    try {
        p = std::make_unique<Process>(o);
    } catch (const std::exception& e) {
        throw std::invalid_argument(std::string("ffmpeg could not start (") + e.what() + ")");
    }
    const size_t bytes = (size_t) info.stored_w * (size_t) info.stored_h * 3;
    size_t k = 0;
    for (; k < frames.size(); ++k) {
        if (cancel != nullptr && cancel->is_set()) throw ClientGone();
        std::vector<uint8_t> px(bytes);
        if (p->read_raw(px.data(), bytes) != bytes) break;
        on_frame(k, std::move(px));
    }
    const std::optional<int> code = p->wait(k == frames.size() ? 30 : -1);
    if (k == 0 || (code && *code != 0 && k < frames.size())) {
        const std::string tail = p->stderr_tail();
        throw std::invalid_argument("ffmpeg could not decode the video" + (tail.empty() ? std::string() : ": " + tail));
    }
    return k;
}

}   // namespace lamina
