// include/lamina/platform/slice_read.hpp - a model file read in large pieces: runs of equal slices of a file copied to places a stride
// apart, past the OS file cache or through it, on several threads.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace lamina::platform {

/// `count` consecutive slices of `bytes` each from `offset` in the file at `path`, slice i copied to
/// base(group) + dst_off + i * stride, or with `places` to places[i] + dst_off (the group's base asked for all the same:
/// it may wait for the places to be made)
struct SliceRun {
    std::string path;
    uint64_t offset = 0, bytes = 0, dst_off = 0, stride = 0;
    int64_t count = 0;
    int group = 0;
    uint8_t* const* places = nullptr;
};

/// The runs read on `threads` threads in pieces of about `piece` bytes (whole slices), in the runs' order.  `uncached`:
/// past the OS file cache (unbuffered reads on Windows; on Linux each piece's pages dropped from the cache once read),
/// else through it.  `base(group)` is asked before each piece and returns where the group's places start (it may wait
/// for them; null: the read stops and fails).  `seconds`, if set, gets the time the reads took.
bool read_slices(const std::vector<SliceRun>& runs, int threads, bool uncached, uint64_t piece,
                 const std::function<uint8_t*(int)>& base, std::string& err, double* seconds = nullptr);

}  // namespace lamina::platform
