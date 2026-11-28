// include/lamina/program/head_sampler.hpp - the rows of a head's logits -> one token each by a request's chain, on the
// GPU, for the engine paths that sample their head's rows themselves (the qwen35 path's, Hy4-flash's, Inkling 2's;
// Flash-Next's verifier samples after its graph): each row's penalty history staged (llama.cpp's: the text before the
// row's token, pictures left out), its draw the Philox number of the position its token takes, so a window's rows
// draw what a row at a time would, and a row that holds a non-finite logit picks nothing.
#pragma once

#include "lamina/core/run_owned.hpp"
#include "lamina/kernels/sampler.hpp"
#include "lamina/program/sampling.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>

namespace lamina::program {

/// Whether a request's chain is the plain argmax (greedy, no penalties, no bias): the token every path's own greedy
/// pick gives, which needs no sampler
inline bool plain_argmax(const lamina::kernels::SamplerParams& p) {
    return p.greedy && p.penalty_last_n <= 0 && p.logit_bias == nullptr;
}

/// Up to `cap` rows a call (a verify window, or a batch window's slots together), its buffers on the GPU current at the
/// first call, the running run's (freed at its end): queue() any number of row spans, then take() their picks.
class HeadSampler {
public:
    explicit HeadSampler(int cap) : cap_(cap) {}
    HeadSampler(const HeadSampler&) = delete;
    HeadSampler& operator=(const HeadSampler&) = delete;

    /// Picks [first, first + rows): rows of `logits` (device, n floats a row) by the chain `p`; row t's token takes
    /// position at + t (its draw's number) and follows tail[0 .. n_tail) and win[0 .. t] (the penalties' history; win
    /// null: one row, the tail's last token its win[0]).  `bad`: the rows' non-finite flags where the caller has them
    /// (device, 1 for such a row), else they are found here.
    void queue(const lamina::kernels::SamplerParams& p, const float* logits, int rows, int n, int64_t at,
               const int32_t* tail, int64_t n_tail, const int32_t* win, cudaStream_t st, int first = 0,
               const int32_t* bad = nullptr) {
        if (rows <= 0) return;
        if (first < 0 || first + rows > cap_ || (win == nullptr && rows != 1)) {
            std::fprintf(stderr, "lamina: the head sampler takes rows %d .. %d of %d\n", first, first + rows - 1, cap_);
            throw lamina::core::Fatal{1};
        }
        on_device();
        const int h = std::clamp(p.penalty_last_n, 0, kPenaltyWindow);   // (a row's region holds kPenaltyWindow)
        const int32_t* hist = nullptr;
        if (h > 0) {
            if (hist_d_ == nullptr) {
                hist_d_ = lamina::core::dalloc<int32_t>((size_t) cap_ * kPenaltyWindow);
                hist_h_ = lamina::core::halloc<int32_t>((size_t) cap_ * kPenaltyWindow);
            }
            int32_t* stage = hist_h_ + (size_t) first * kPenaltyWindow;   // (a span's own: none is reused before take)
            if (win != nullptr) lamina::kernels::penalty_rows(tail, n_tail, win, rows, h, stage);
            else lamina::kernels::penalty_rows(tail, n_tail - 1, tail + n_tail - 1, 1, h, stage);
            int32_t* dst = hist_d_ + (size_t) first * kPenaltyWindow;
            LAMINA_CK(cudaMemcpyAsync(dst, stage, (size_t) rows * (size_t) h * 4, cudaMemcpyHostToDevice, st));
            hist = dst;
        }
        lamina::kernels::SamplerParams q = p;
        q.counter = (uint64_t) at;
        q.penalty_last_n = h;
        lamina::kernels::sample_tokens(logits, rows, n, hist, h, q, pick_d_ + first, st);
        int32_t* flags = pick_d_ + cap_ + first;
        if (bad != nullptr) LAMINA_CK(cudaMemcpyAsync(flags, bad, (size_t) rows * 4, cudaMemcpyDeviceToDevice, st));
        else lamina::kernels::rows_nonfinite(logits, rows, n, flags, st);
    }
    /// Picks [0, rows) into out once the stream has them: each row's token, -1 where its logits hold a non-finite one
    void take(int rows, int32_t* out, cudaStream_t st) {
        on_device();
        LAMINA_CK(cudaMemcpyAsync(pick_h_, pick_d_, (size_t) 2 * cap_ * 4, cudaMemcpyDeviceToHost, st));
        LAMINA_CK(cudaStreamSynchronize(st));
        for (int t = 0; t < rows; ++t) out[t] = pick_h_[cap_ + t] != 0 ? -1 : pick_h_[t];
    }
    /// queue() of rows [0, rows), then take()
    void pick(const lamina::kernels::SamplerParams& p, const float* logits, int rows, int n, int64_t at,
              const int32_t* tail, int64_t n_tail, const int32_t* win, cudaStream_t st, int32_t* out,
              const int32_t* bad = nullptr) {
        queue(p, logits, rows, n, at, tail, n_tail, win, st, 0, bad);
        take(rows, out, st);
    }

private:
    void on_device() {
        int dev = -1;
        LAMINA_CK(cudaGetDevice(&dev));
        if (pick_d_ == nullptr) {
            dev_ = dev;
            pick_d_ = lamina::core::dzero<int32_t>((size_t) 2 * cap_);
            pick_h_ = lamina::core::hzero<int32_t>((size_t) 2 * cap_);
        } else if (dev != dev_) {
            std::fprintf(stderr, "lamina: the head sampler's buffers are on GPU %d, its rows on GPU %d\n", dev_, dev);
            throw lamina::core::Fatal{1};
        }
    }
    int cap_ = 0, dev_ = -1;
    int32_t *pick_d_ = nullptr, *pick_h_ = nullptr;   // [cap] picks, then [cap] non-finite flags
    int32_t *hist_d_ = nullptr, *hist_h_ = nullptr;   // [cap][kPenaltyWindow]: each row's history, staged by span
};

}   // namespace lamina::program
