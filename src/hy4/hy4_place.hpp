// src/hy4/hy4_place.hpp - the expert arena's places timed by the RAM's read rate (--arena-place on the hy4 path): every
// core reads one place at a time, as the CPU pool reads an expert (src/core/arena_place.cpp's method for the Flash-Next
// path; on this PC two 32 GB and two 48 GB modules leave 32 GB read from one channel, ~37 GB/s against ~58).
#pragma once

#include <cstdint>
#include <vector>

namespace lamina::hy4 {

/// Each place's read rate in GB/s (`bytes` each), timed once with every physical core reading a share of it, the slow
/// ones (under 85% of the 90th percentile) a second time (the better of two).  `cut` gets that bound.
std::vector<float> time_places(const std::vector<const uint8_t*>& places, uint64_t bytes, float* cut);

}   // namespace lamina::hy4
