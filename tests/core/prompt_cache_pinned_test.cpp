// tests/core/prompt_cache_pinned_test.cpp - the prompt cache's pinned buffers for the checkpoints a run fills inside
// it (SequenceCache::prepare_more, prompt_cache.hpp), on an engine stand-in whose state is a host array:
//
//   1. a parked conversation's checkpoints hold the pinned buffers (the first chain_cap + 1): another conversation's
//      run still gets them, its checkpoints moving to pageable buffers with their bytes, and that conversation comes
//      back from the park with both of its states as they were saved;
//   2. a long conversation whose chain holds them: its checkpoints from before the request give theirs up, the ones
//      the request took itself (a split prompt's stages may still be writing them) do not, and the states stay.
// Needs a CUDA device for the pinned buffers (exit 77 without one).
#include "lamina/core/prompt_cache.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using lamina::core::ImgKey;
using lamina::core::SequenceCache;

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++g_fail;
}

constexpr uint64_t kBytes = 64 << 10;   // a checkpoint
constexpr int64_t kCells = 1 << 16;     // the stand-in's K/V, 8 bytes a cell

class Fake : public SequenceCache {
public:
    explicit Fake(int chain_cap) : SequenceCache(chain_cap, kBytes, 1ull << 30, 0, 0), state(kBytes), kv(kCells * 8) {}
    std::vector<uint8_t> state, kv;
    void fill(uint8_t v) { std::fill(state.begin(), state.end(), v); }
    // a run's checkpoint: the state into a prepared buffer, as the prompt path copies it in
    void filled(void* host) { std::memcpy(host, state.data(), kBytes); }

protected:
    bool state_save(int64_t, void* host, bool filled) override {
        if (!filled) std::memcpy(host, state.data(), kBytes);
        return true;
    }
    bool state_load(int64_t, const void* host) override {
        std::memcpy(state.data(), host, kBytes);
        return true;
    }
    void state_zero() override { fill(0); }
    uint64_t kv_bytes(int64_t cells) const override { return (uint64_t) cells * 8; }
    bool kv_copy(int64_t cells, int64_t, uint8_t* host, bool to_host) override {
        if (to_host) std::memcpy(host, kv.data(), (size_t) cells * 8);
        else std::memcpy(kv.data(), host, (size_t) cells * 8);
        return true;
    }
    void kv_settle() override {}
};

bool pinned(const void* p) {
    unsigned flags = 0;
    const bool ok = p != nullptr && cudaHostGetFlags(&flags, const_cast<void*>(p)) == cudaSuccess;
    (void) cudaGetLastError();
    return ok;
}

bool all(const std::vector<uint8_t>& v, uint8_t x) {
    return std::all_of(v.begin(), v.end(), [&](uint8_t b) { return b == x; });
}

std::vector<int64_t> tokens(int64_t first, int64_t n) {
    std::vector<int64_t> t((size_t) n);
    for (int64_t i = 0; i < n; ++i) t[(size_t) i] = first + i;
    return t;
}

}   // namespace

int main() {
    std::printf("prompt_cache_pinned_test\n");
    void* probe = nullptr;
    if (cudaMallocHost(&probe, 4096) != cudaSuccess) {
        std::printf("  no CUDA device: skipped\n");
        return 77;
    }
    cudaFreeHost(probe);
    const std::vector<ImgKey> none;
    // 1. a parked conversation's pinned buffers
    {
        Fake c(2);   // (three pinned: the first chain_cap + 1)
        const std::vector<int64_t> a = tokens(1, 1200), b = tokens(100000, 1200);
        check(c.begin(a, none) == 0, "conversation A starts empty");
        c.fill(0xA1);
        void* h = c.prepare();
        c.filled(h);
        c.set(0, a.data(), 500);
        c.checkpoint(500, h);
        c.fill(0xA2);
        h = c.prepare();
        c.filled(h);
        c.set(500, a.data() + 500, 500);
        c.checkpoint(1000, h);
        check(c.begin(b, none) == 0, "conversation B starts empty (A parked)");
        void* end = c.prepare();
        void* m1 = c.prepare_more();
        void* m2 = c.prepare_more();
        void* m3 = c.prepare_more();
        check(pinned(end), "B's run: its end's buffer is the third pinned one");
        check(pinned(m1) && pinned(m2) && m1 != m2 && m1 != end && m2 != end,
              "B's run: two more pinned buffers for checkpoints inside it, A's");
        check(m3 == nullptr, "a third: none (no checkpoint holds a pinned one now)");
        c.fill(0xB1);
        c.filled(m1);
        c.set(0, b.data(), 400);
        c.checkpoint(400, m1);
        c.fill(0xB2);
        c.filled(end);
        c.set(400, b.data() + 400, 400);
        c.checkpoint(800, end);
        std::vector<int64_t> a2 = tokens(1, 1000);
        const std::vector<int64_t> more = tokens(500000, 300);
        a2.insert(a2.end(), more.begin(), more.end());
        check(c.begin(a2, none) == 1000, "A continues from its checkpoint at 1,000, back from the park");
        check(all(c.state, 0xA2), "... with the state saved there");
        std::vector<int64_t> a3 = tokens(1, 500);
        const std::vector<int64_t> other = tokens(700000, 300);
        a3.insert(a3.end(), other.begin(), other.end());
        check(c.begin(a3, none) == 500, "A rewound to its checkpoint at 500");
        check(all(c.state, 0xA1), "... with the state saved there");
    }
    // 2. a long conversation's own chain
    {
        Fake c(4);   // (five pinned)
        const std::vector<int64_t> t = tokens(1, 4000);
        check(c.begin(t, none) == 0, "the conversation starts empty");
        int64_t at = 0;
        auto take = [&](int64_t pos, uint8_t v) {
            c.fill(v);
            void* h = c.prepare();
            c.filled(h);
            c.set(at, t.data() + at, pos - at);
            c.checkpoint(pos, h);
            at = pos;
            return h;
        };
        take(500, 0xC1);
        take(1000, 0xC2);
        check(c.begin(t, none) == 1000, "its next request continues from 1,000");
        void* own1 = take(1500, 0xC3);
        void* own2 = take(2000, 0xC4);
        void* end = c.prepare();
        check(pinned(own1) && pinned(own2) && pinned(end), "the request's checkpoints and its run's end: pinned");
        void* m1 = c.prepare_more();
        void* m2 = c.prepare_more();
        void* m3 = c.prepare_more();
        check(pinned(m1) && pinned(m2), "two more pinned buffers: those of the checkpoints at 500 and 1,000");
        check(m3 == nullptr && m1 != own1 && m1 != own2 && m2 != own1 && m2 != own2,
              "none of the request's own (1,500 and 2,000)");
        const std::vector<int64_t> back = tokens(1, 1000);
        std::vector<int64_t> r = back;
        const std::vector<int64_t> other = tokens(900000, 300);
        r.insert(r.end(), other.begin(), other.end());
        check(c.begin(r, none) == 1000, "a request that leaves it at 1,000 continues from there");
        check(all(c.state, 0xC2), "... with the state saved there");
    }
    std::printf(g_fail ? "FAILED: %d\n" : "all ok\n", g_fail);
    return g_fail ? 1 : 0;
}
