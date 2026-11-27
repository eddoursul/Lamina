// include/lamina/program/host_sampling.hpp - the sampler of the engine paths that sample on the host (Qwen3.6's and
// Hy4-flash's): a request's keys (GEN ... temperature= top_p= ...), one row of logits -> a token, and the id lists of
// their command lines and requests.  Flash-Next's path samples on the GPU (lamina/kernels/sampler.hpp).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace lamina::program {

struct Sampling {
    float temperature = 0.f, top_p = 1.f, min_p = 0.f, repeat = 1.f, freq = 0.f, present = 0.f;
    int top_k = 20, last_n = 0;
    unsigned long long seed = 0;
    bool needs_logits() const { return temperature > 0.f || repeat != 1.f || freq != 0.f || present != 0.f; }
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

/// One token from a row of logits (penalties applied in place): the argmax at temperature 0, else top-k (at most 64),
/// min-p and top-p over the temperature's softmax.  -1 when the row holds a non-finite logit (the request ends
/// "nonfinite").
inline int sample(std::vector<float>& lg, const Sampling& s, const std::vector<int32_t>& hist, std::mt19937_64& rng) {
    if (!std::all_of(lg.begin(), lg.end(), [](float v) { return std::isfinite(v); })) return -1;
    if (s.last_n != 0 && (s.repeat != 1.f || s.freq != 0.f || s.present != 0.f)) {
        std::map<int, int> cnt;
        const size_t from = s.last_n < 0 || (size_t) s.last_n >= hist.size() ? 0 : hist.size() - (size_t) s.last_n;
        for (size_t i = from; i < hist.size(); ++i) ++cnt[hist[i]];
        for (const auto& [id, c] : cnt) {
            float& v = lg[(size_t) id];
            if (s.repeat != 1.f) v = v > 0 ? v / s.repeat : v * s.repeat;
            v -= (float) c * s.freq + s.present;
        }
    }
    if (s.temperature <= 0.f) return (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
    const int kk = std::max(1, std::min(s.top_k > 0 ? s.top_k : 64, (int) lg.size()));
    std::vector<int> idx(lg.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    idx.resize((size_t) kk);
    std::vector<double> p(idx.size());
    const double mx = lg[(size_t) idx[0]];
    double sum = 0;
    for (size_t i = 0; i < idx.size(); ++i) sum += (p[i] = std::exp((lg[(size_t) idx[i]] - mx) / s.temperature));
    size_t keep = idx.size();
    double cum = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (p[i] / sum < s.min_p * (p[0] / sum)) {
            keep = i;
            break;
        }
        cum += p[i] / sum;
        if (cum >= s.top_p) {
            keep = i + 1;
            break;
        }
    }
    keep = std::max<size_t>(1, keep);
    double tot = 0;
    for (size_t i = 0; i < keep; ++i) tot += p[i];
    std::uniform_real_distribution<double> u(0.0, tot);
    double r = u(rng);
    for (size_t i = 0; i < keep; ++i) {
        if ((r -= p[i]) <= 0) return idx[i];
    }
    return idx[keep - 1];
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
