// include/lamina/program/sampling.hpp - a request's sampling keys (GEN ... temperature= top_p= ...) as the engine paths
// read them, the seed its draws start from, the device sampler's chain they make (lamina/kernels/sampler.hpp:
// llama.cpp's order), and the id lists of the paths' command lines and requests.  The paths sample their head's rows
// on the GPU (head_sampler.hpp; Flash-Next's verifier after its graph).
#pragma once

#include "lamina/kernels/sampler.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace lamina::program {

/// The seed a request's draws start from: its own, or the clock's when it gave none (0), so an unseeded sampled
/// request sent again draws another text.  Every path's rule.
inline unsigned long long draw_seed(unsigned long long seed) {
    return seed != 0 ? seed : (unsigned long long) std::chrono::steady_clock::now().time_since_epoch().count();
}

/// The penalties' window at most: the history a row's pick counts (the device holds that many tokens a row)
constexpr int kPenaltyWindow = 4096;

struct Sampling {
    float temperature = 0.f, top_p = 1.f, min_p = 0.f, repeat = 1.f, freq = 0.f, present = 0.f;
    int top_k = 20, last_n = 0;
    unsigned long long seed = 0;   // 0: none given (draw_seed: the clock's)
    /// a request's key=value: true when it is one of the sampler's
    bool set(const std::string& key, const char* v) {
        if (key == "temperature") temperature = std::strtof(v, nullptr);
        else if (key == "top_p") top_p = std::strtof(v, nullptr);
        else if (key == "top_k") top_k = std::atoi(v);
        else if (key == "min_p") min_p = std::strtof(v, nullptr);
        else if (key == "penalty_last_n") last_n = std::atoi(v);
        else if (key == "penalty_repeat") repeat = std::strtof(v, nullptr);
        else if (key == "penalty_freq") freq = std::strtof(v, nullptr);
        else if (key == "penalty_present") present = std::strtof(v, nullptr);
        else if (key == "seed") seed = std::strtoull(v, nullptr, 10);
        else return false;
        return true;
    }
};

/// A request's keys as the device sampler's chain, every path's one rule: greedy at a temperature of 0 or below; top_k
/// as given (the chain keeps 1 to 64 candidates: 0 or more than 64 keep 64); min_p within 0 .. 1; the penalties counted
/// over the last penalty_last_n tokens (kPenaltyWindow at most) where one of them is set and the window is above 0
/// (llama.cpp's sampler takes a window below 1 as none); the seed draw_seed's.
inline lamina::kernels::SamplerParams sampler_params(const Sampling& q) {
    lamina::kernels::SamplerParams p;
    p.greedy = q.temperature <= 0.f;
    p.temperature = q.temperature;
    p.top_k = q.top_k;
    p.top_p = q.top_p;
    p.min_p = std::clamp(q.min_p, 0.f, 1.f);
    const bool pen = q.repeat != 1.f || q.freq != 0.f || q.present != 0.f;
    p.penalty_last_n = pen && q.last_n > 0 ? std::min(q.last_n, kPenaltyWindow) : 0;
    p.penalty_repeat = q.repeat;
    p.penalty_freq = q.freq;
    p.penalty_present = q.present;
    p.seed = draw_seed(q.seed);
    return p;
}

/// "1,2,3" (or space-separated) -> ids, up to the first character that is not part of the list
inline std::vector<int32_t> parse_ids(const char* s) {
    std::vector<int32_t> v;
    while (*s) {
        while (*s == ',' || *s == ' ' || *s == '\n' || *s == '\r') ++s;
        if (!*s) break;
        char* e = nullptr;
        const long x = std::strtol(s, &e, 10);
        if (e == s) break;
        v.push_back((int32_t) x);
        s = e;
    }
    return v;
}

}   // namespace lamina::program
