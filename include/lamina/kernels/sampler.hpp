// include/lamina/kernels/sampler.hpp - the sampler chain, host-callable (P2.S2).
//
//     penalties -> top_k -> top_p -> min_p -> temperature -> pick
//
// llama.cpp's default chain (issue #53): ONE penalties stage, first, and temperature AFTER the truncation filters;
// top_p cuts before min_p.  `src/kernels/cuda/sampler.cu` has the details and `sampler_parity` pins them.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define LAMINA_SAMPLER_HD __host__ __device__ __forceinline__
#else
#define LAMINA_SAMPLER_HD inline
#endif

namespace lamina::kernels {

/// The uniform number of draw `counter` under `seed`, in [0, 1): Philox 4x32-10 over (counter, seed), the upper 24
/// bits of its first word (so the value never rounds to 1).  Counter-based: a draw is a function of (seed, position)
/// and not of how many came before, so a batch samples in any order, a run repeats, and the host can read the number
/// a position will be drawn with before its window runs (core/draft_pass.hpp).  One definition for the kernels and
/// the host: integer arithmetic and an exact conversion, the same bits on both.
LAMINA_SAMPLER_HD float philox_uniform(uint64_t seed, uint64_t counter) {
    uint32_t c0 = (uint32_t) counter, c1 = (uint32_t) (counter >> 32);
    uint32_t c2 = (uint32_t) seed, c3 = (uint32_t) (seed >> 32);
    for (uint32_t i = 0; i < 10; ++i) {   // round i's key is (i, 0)
        const uint64_t m0 = (uint64_t) 0x9E3779B9u * c0, m1 = (uint64_t) 0xBB67AE85u * c2;
        const uint32_t n0 = (uint32_t) (m1 >> 32) ^ c1 ^ i, n2 = (uint32_t) (m0 >> 32) ^ c3;
        c1 = (uint32_t) m1;
        c3 = (uint32_t) m0;
        c0 = n0;
        c2 = n2;
    }
    return (float) (c0 >> 8) * (1.0f / 16777216.0f);
}

struct SamplerParams {
    int top_k = 20;             // sampled path: 1..64 as given; 0 (off) or more than 64 keep the widest list, 64
    float top_p = 0.95f;        // 1.0 disables the filter
    float min_p = 0.0f;         // 0 disables; keeps tokens with p >= min_p * p_max (a prefix of the top_k list)
    float temperature = 1.0f;   // <= 0 means greedy
    int min_keep = 1;           // top_p keeps at least this many
    int penalty_last_n = 0;     // 0 disables; the window over `history` to count occurrences in
    float penalty_repeat = 1.0f;
    float penalty_freq = 0.0f;
    float penalty_present = 0.0f;
    uint64_t seed = 0;      // drives Philox, which is counter-based on (seed, token index)
    uint64_t counter = 0;   // absolute draw index of row 0; advance across decode calls
    bool greedy = false;
    const float* logit_bias = nullptr;   // optional device array [n_vocab], added BEFORE penalties/filters; -inf bans
};

// logits (n_tokens, n_vocab) -> one sampled token id per row in `out`.
//
// **`logits` AND `out` ARE DEVICE POINTERS.**  This is a CUDA kernel launch, not a host function, and nothing
// in the parameter names or the types says so - `const float* logits` reads exactly like a host buffer.
// Measured: passing host memory for `logits` faults inside the kernel with an ILLEGAL MEMORY ACCESS that is
// reported by whatever synchronising call happens NEXT, which will be somewhere else entirely and will name a
// buffer that has nothing to do with it.  A sticky async fault does not know where it came from.
//
// `history` is (n_tokens, history_len) int32 - ONE ROW PER TOKEN ROW, at a stride of `history_len` - the most
// recent tokens that row's pick follows, with any unused slots set to -1; only the last `p.penalty_last_n` of
// each row are counted, and ids outside [0, n_vocab) are ignored.  Pass nullptr and 0 when no penalties apply.
// A verify window's rows need DIFFERENT histories: row t follows the window's drafts 1..t (`penalty_rows`).
void sample_tokens(const float* logits, int n_tokens, int n_vocab, const int* history, int history_len,
                   const SamplerParams& p, int* out, void* stream);

// The greedy pick (no penalties) on a thread-block cluster of 8 CTAs per row (sm_90+, CUDA; S19): the same token as
// sample_tokens' one-block argmax, which takes it unless LAMINA_ARGMAX_MULTI=0.  False (nothing launched) where it
// cannot run: a card or a build below sm_90.  Device pointers; capturable.
bool sample_greedy_cluster(const float* logits, int n_tokens, int n_vocab, int* out, void* stream);

// The penalty-history rows of a verify window, on the host: row t of `out` (T rows of `h` slots) is the last `h`
// tokens of `tail[0..n_tail)` followed by `window[0..t]`, most recent LAST, -1 in the unused front slots.
// `tail` is what the state consumed before the window, `window[0]` the fed-back token and `window[1..]` the
// drafts: row t is exactly the history plain decode counts when it picks the token after `window[t]`, so drafting
// cannot change which tokens are penalised.  Row 0 alone is the single row the engine staged before 0.1.19.
inline void penalty_rows(const int32_t* tail, int64_t n_tail, const int32_t* window, int T, int h, int32_t* out) {
    for (int t = 0; t < T; ++t) {
        int32_t* row = out + (size_t) t * (size_t) h;
        const int64_t avail = n_tail + t + 1;   // tail + window[0..t]
        const int take = (int) std::min<int64_t>(h, avail);
        std::fill(row, row + (h - take), -1);
        for (int j = 0; j < take; ++j) {
            const int64_t i = avail - take + j;   // index into (tail, window)
            row[h - take + j] = i < n_tail ? tail[i] : window[i - n_tail];
        }
    }
}

}   // namespace lamina::kernels
