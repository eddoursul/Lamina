// src/hy4/hy4_place.cpp - see hy4_place.hpp.  The crew and the reads are src/core/arena_place.cpp's.
#include "hy4_place.hpp"

#include "lamina/kernels/cpu/pool.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <thread>

namespace lamina::hy4 {
namespace {

using Clock = std::chrono::steady_clock;
std::atomic<uint64_t> g_sink{0};   // the reads' results, so that they happen

// One thread on each physical core (the caller's on the last), all running the same task on a spin barrier
class Crew {
public:
    Crew() {
        for (const auto& core : lamina::kernels::cpu::physical_cores())
            if (!core.empty()) cpus_.push_back(core.front());
        if (cpus_.empty()) cpus_.push_back(-1);
        std::rotate(cpus_.begin(), cpus_.end() - 1, cpus_.end());
        n_ = (int) cpus_.size();
        prev_ = lamina::kernels::cpu::pin_current_thread(cpus_[0]);
        for (int t = 1; t < n_; ++t)
            th_.emplace_back([this, t] {
                (void) lamina::kernels::cpu::pin_current_thread(cpus_[(size_t) t]);
                uint64_t seen = 0;
                for (;;) {
                    const uint64_t g = go_.load(std::memory_order_acquire);
                    if (g == seen) continue;
                    seen = g;
                    if (quit_.load()) return;
                    (*task_)(t);
                    done_.fetch_add(1, std::memory_order_acq_rel);
                }
            });
    }
    ~Crew() {
        quit_.store(true);
        go_.fetch_add(1, std::memory_order_acq_rel);
        for (auto& t : th_) t.join();
        lamina::kernels::cpu::restore_thread_affinity(prev_);
    }
    int size() const { return n_; }
    void run(const std::function<void(int)>& task) {
        task_ = &task;
        done_.store(0, std::memory_order_relaxed);
        go_.fetch_add(1, std::memory_order_acq_rel);
        task(0);
        while (done_.load(std::memory_order_acquire) < n_ - 1) {}
    }

private:
    std::vector<int> cpus_;
    int n_ = 1;
    lamina::kernels::cpu::ThreadAffinity prev_;
    std::vector<std::thread> th_;
    const std::function<void(int)>* task_ = nullptr;
    std::atomic<uint64_t> go_{0};
    std::atomic<int> done_{0};
    std::atomic<bool> quit_{false};
};

void read_words(const uint8_t* p, uint64_t bytes) {
    const uint64_t* w = (const uint64_t*) p;
    const uint64_t n = bytes / 8;
    uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, i = 0;
    for (; i + 4 <= n; i += 4) {
        a0 ^= w[i];
        a1 ^= w[i + 1];
        a2 ^= w[i + 2];
        a3 ^= w[i + 3];
    }
    for (; i < n; ++i) a0 ^= w[i];
    g_sink.fetch_xor(a0 ^ a1 ^ a2 ^ a3, std::memory_order_relaxed);
}

}   // namespace

std::vector<float> time_places(const std::vector<const uint8_t*>& places, uint64_t bytes, float* cut) {
    const size_t n = places.size();
    std::vector<float> gbs(n, 0.f);
    if (n == 0 || bytes < 4096) return gbs;
    Crew crew;
    const int T = crew.size();
    const uint8_t* p = nullptr;
    const std::function<void(int)> read = [&](int t) {
        const uint64_t per = bytes / (uint64_t) T / 64 * 64, from = per * (uint64_t) t;
        read_words(p + from, t == T - 1 ? bytes - from : per);
    };
    auto time_one = [&](size_t q) {
        p = places[q];
        const auto s = Clock::now();
        crew.run(read);
        const double sec = std::chrono::duration<double>(Clock::now() - s).count();
        gbs[q] = std::max(gbs[q], (float) ((double) bytes / sec / 1e9));
    };
    for (size_t q = 0; q < n; ++q) time_one(q);
    std::vector<float> sorted = gbs;
    std::sort(sorted.begin(), sorted.end());
    const float c = 0.85f * sorted[n * 9 / 10];
    for (size_t q = 0; q < n; ++q)
        if (gbs[q] < c) time_one(q);
    if (cut) *cut = c;
    return gbs;
}

}   // namespace lamina::hy4
