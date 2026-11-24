// include/lamina/core/draft_pass.hpp - a draft's chance to pass in a sampled request.
//
// A sampled request keeps a draft only when it is the token its verify row draws: the row sorts its chain's tokens by
// logit and takes the one whose interval of the cumulative distribution holds u = Philox(seed, position)
// (kernels/sampler.hpp).  The number is a function of the seed and the position, so it is known when the draft is
// made.  The draft layer offers its likeliest token, which passes when it is the verifier's likeliest too and u
// falls inside its interval there, [0, P0):
//
//     chance = sigmoid(c + d logit(q)) * sigmoid((a + b logit(q) - logit(u)) / w)
//
// q the draft's probability under the draft layer; the first factor is the chance that both pick the same token, the
// second that P0 lies above u.  A lower temperature widens the interval: a = a0 + a1 / T, b = b0 + b1 / T,
// w = exp(w0 + w1 log T).  At one q a draft passes far more often under a small u than under a large one (the first
// draft at q 0.7-0.8: 72% below 0.5, 17% from 0.9 on).
//
// The floor (--spec-min-p) stays a probability under the draft layer: a draft is kept when its chance under its own
// number is at least the chance a draft at the floor has under an average number (`draft_pass_floor`).  So a number
// that leaves the likeliest token room keeps a draft the plain floor would drop, one that has already lost it drops
// a draft the plain floor would keep, and with no number in play (the temperature toward 0) the rule is the plain
// floor again.
//
// The token the number selects in the draft layer's own distribution is no better a draft than its likeliest one (it
// is the token drawn as often), so the draft stays the likeliest token.
//
// The constants: fitted per chain step on dumped rounds of the distilled draft layer (mtp/rt-distilled) under two
// model files (IQ3_S and UD-Q4_K_XL: one set serves both), at top_p 0.95 and top_k 20 over temperatures 0.3-1.5.
// Another draft layer wants its own.
#pragma once

#include <algorithm>
#include <cmath>

namespace lamina::core {

struct DraftPassStep {
    float c, d;        ///< the same token: sigmoid(c + d logit(q))
    float a0, a1;      ///< the interval's edge in log-odds: a0 + a1 / T + (b0 + b1 / T) logit(q)
    float b0, b1;
    float w0, w1;      ///< its spread: exp(w0 + w1 log T)
};

/// Chain steps 1-3; a later step takes the third's.
inline constexpr DraftPassStep kDraftPass[3] = {
    {0.3594f, 0.7245f, -1.2214f, 1.4505f, 0.4971f, 0.3190f, 0.1695f, -0.4200f},
    {0.1142f, 0.7233f, -1.0117f, 1.7660f, 0.5228f, 0.4303f, 0.4723f, -0.4992f},
    {0.0171f, 0.7109f, -0.8102f, 1.9189f, 0.5340f, 0.5118f, 0.6039f, -0.5236f},
};

namespace draft_pass_detail {
inline float logit(float x) {
    x = std::clamp(x, 1e-6f, 1.0f - 1e-6f);
    return std::log(x) - std::log1p(-x);
}
inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
/// The chance's two factors at probability q: the same token, and the interval's edge and spread.
inline void terms(int step, float q, float temperature, float& same, float& edge, float& spread) {
    const DraftPassStep& k = kDraftPass[std::clamp(step, 0, 2)];
    // (below the fitted temperatures the interval's edge runs away from every number, as it should: toward 0 the
    // chance is the same token's alone)
    const float it = 1.0f / std::clamp(temperature, 0.05f, 2.0f);
    const float lq = logit(q);
    same = sigmoid(k.c + k.d * lq);
    edge = k.a0 + k.a1 * it + (k.b0 + k.b1 * it) * lq;
    spread = std::exp(k.w0 - k.w1 * std::log(it));
}
}  // namespace draft_pass_detail

/// The chance that chain step `step`'s draft (0-based), of probability `q` under the draft layer, is the token a
/// request sampling at `temperature` draws with the number `u` in [0, 1).
inline float draft_pass_chance(int step, float q, float u, float temperature) {
    float same, edge, spread;
    draft_pass_detail::terms(step, q, temperature, same, edge, spread);
    return same * draft_pass_detail::sigmoid((edge - draft_pass_detail::logit(u)) / spread);
}

/// The floor `min_p` (a probability under the draft layer) as a chance: what a draft of that probability has under
/// an average number, the mean of draft_pass_chance over u.  0 keeps every draft, above 1 none.
inline float draft_pass_floor(int step, float min_p, float temperature) {
    if (!(min_p > 0.0f)) return 0.0f;
    if (min_p > 1.0f) return 2.0f;
    float same, edge, spread;
    draft_pass_detail::terms(step, min_p, temperature, same, edge, spread);
    constexpr int kPoints = 64;   // the midpoint rule over u: the integrand is smooth and bounded
    float sum = 0.0f;
    for (int i = 0; i < kPoints; ++i)
        sum += draft_pass_detail::sigmoid((edge - draft_pass_detail::logit(((float) i + 0.5f) / kPoints)) / spread);
    return same * sum / kPoints;
}

}  // namespace lamina::core
