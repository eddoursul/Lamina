// tests/core/draft_pass_test.cpp - a sampled request's drafts (core/draft_pass.hpp) and the number they are judged by
// (kernels/sampler.hpp's philox_uniform on the host), CPU only.
#include "lamina/core/draft_pass.hpp"
#include "lamina/kernels/sampler.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>

using lamina::core::draft_pass_chance;
using lamina::core::draft_pass_floor;
using lamina::kernels::philox_uniform;

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++g_fail;
}

// Philox 4x32-10 with the round keys (i, 0), written out from the definition (two 32 x 32 -> 64 products a round):
// the header's loop against it.
float philox_reference(uint64_t seed, uint64_t counter) {
    uint32_t c[4] = {(uint32_t) counter, (uint32_t) (counter >> 32), (uint32_t) seed, (uint32_t) (seed >> 32)};
    for (uint32_t i = 0; i < 10; ++i) {
        const uint64_t p0 = (uint64_t) 0x9E3779B9u * c[0], p1 = (uint64_t) 0xBB67AE85u * c[2];
        const uint32_t hi0 = (uint32_t) (p0 >> 32), lo0 = (uint32_t) p0, hi1 = (uint32_t) (p1 >> 32), lo1 = (uint32_t) p1;
        const uint32_t n[4] = {hi1 ^ c[1] ^ i, lo1, hi0 ^ c[3] ^ 0u, lo0};
        for (int k = 0; k < 4; ++k) c[k] = n[k];
    }
    return (float) (c[0] >> 8) / 16777216.0f;
}

}  // namespace

int main() {
    std::printf("draft_pass_test\n");
    {   // the number: in [0, 1), the definition's, and uniform enough over positions
        bool range = true, same = true;
        double sum = 0;
        int below = 0;
        const int n = 200000;
        for (int i = 0; i < n; ++i) {
            const uint64_t seed = 0x9E3779B97F4A7C15ull * (uint64_t) (i / 1000 + 1), counter = (uint64_t) i * 7919u;
            const float u = philox_uniform(seed, counter);
            range = range && u >= 0.0f && u < 1.0f;
            same = same && u == philox_reference(seed, counter);
            sum += u;
            below += u < 0.25f;
        }
        check(range, "philox_uniform stays in [0, 1)");
        check(same, "philox_uniform is the definition's, bit for bit");
        check(std::fabs(sum / n - 0.5) < 0.005 && std::abs(below - n / 4) < n / 100, "its mean and its lowest quarter");
        check(philox_uniform(1, 0) != philox_uniform(1, 1) && philox_uniform(1, 5) != philox_uniform(2, 5) &&
                  philox_uniform((1ull << 32) + 1, 5) != philox_uniform(1, 5) &&
                  philox_uniform(1, (1ull << 32) + 5) != philox_uniform(1, 5),
              "every half of the seed and the counter counts");
    }
    {   // the chance: a probability, rising with the draft's probability, falling with the number
        bool unit = true, up_q = true, down_u = true, later = true;
        for (int step = 0; step < 4; ++step)
            for (float t : {0.1f, 0.3f, 0.7f, 1.0f, 1.5f, 4.0f})
                for (int iq = 1; iq < 100; ++iq)
                    for (int iu = 0; iu < 100; ++iu) {
                        const float q = (float) iq / 100.0f, u = (float) iu / 100.0f;
                        const float c = draft_pass_chance(step, q, u, t);
                        unit = unit && c >= 0.0f && c <= 1.0f && std::isfinite(c);
                        if (iq > 1) up_q = up_q && c >= draft_pass_chance(step, (float) (iq - 1) / 100.0f, u, t);
                        if (iu > 0) down_u = down_u && c <= draft_pass_chance(step, q, (float) (iu - 1) / 100.0f, t);
                        if (step == 3) later = later && c == draft_pass_chance(2, q, u, t);
                    }
        check(unit, "the chance is in [0, 1]");
        check(up_q, "a likelier draft has the better chance");
        check(down_u, "a larger number leaves the smaller chance");
        check(later, "a step past the third takes the third's constants");
    }
    {   // the ends: a sure draft passes, an unlikely one does not, and a low temperature takes the number out
        check(draft_pass_chance(0, 1.0f, 0.5f, 0.7f) > 0.97f && draft_pass_chance(0, 0.9999f, 0.99f, 0.7f) > 0.9f,
              "a draft the layer is sure of passes");
        check(draft_pass_chance(0, 0.05f, 0.5f, 0.7f) < 0.2f, "an unlikely draft does not");
        check(draft_pass_chance(0, 0.7f, 0.2f, 0.7f) > 0.5f && draft_pass_chance(0, 0.7f, 0.95f, 0.7f) < 0.3f,
              "a draft at 0.7 is kept under a small number and dropped under a large one");
        check(draft_pass_chance(0, 0.7f, 0.9f, 0.3f) > draft_pass_chance(0, 0.7f, 0.9f, 0.7f) &&
                  draft_pass_chance(0, 0.7f, 0.9f, 0.7f) > draft_pass_chance(0, 0.7f, 0.9f, 1.2f),
              "a lower temperature widens the likeliest token's interval");
    }
    {   // the constants: values of the fit's own code
        struct { int step; float q, u, t, want; } c[] = {{0, 0.7f, 0.2f, 0.7f, 0.65417f}, {0, 0.7f, 0.95f, 0.7f, 0.20464f},
                                                         {1, 0.5f, 0.5f, 0.7f, 0.36337f}, {2, 0.9f, 0.99f, 1.0f, 0.28441f},
                                                         {0, 0.4f, 0.05f, 0.3f, 0.49229f}, {5, 0.6f, 0.3f, 1.5f, 0.43514f}};
        struct { int step; float p, t, want; } f[] = {{0, 0.5f, 0.7f, 0.36305f}, {1, 0.5f, 0.7f, 0.34991f},
                                                      {2, 0.5f, 0.7f, 0.34492f}, {0, 0.5f, 1.0f, 0.31496f},
                                                      {0, 0.3f, 0.7f, 0.22100f}, {2, 0.7f, 0.4f, 0.55369f}};
        bool chances = true, floors = true;
        for (const auto& x : c) chances = chances && std::fabs(draft_pass_chance(x.step, x.q, x.u, x.t) - x.want) < 1e-3f;
        for (const auto& x : f) floors = floors && std::fabs(draft_pass_floor(x.step, x.p, x.t) - x.want) < 1e-3f;
        check(chances, "the chance is the fit's");
        check(floors, "the floor is the fit's");
    }
    {   // the floor: the chance of a draft at the floor's probability under an average number
        bool mean = true, rising = true;
        for (int step = 0; step < 3; ++step)
            for (float t : {0.3f, 0.7f, 1.0f, 1.5f}) {
                float last = 0.0f;
                for (int ip = 1; ip < 20; ++ip) {
                    const float min_p = (float) ip / 20.0f;
                    double sum = 0;
                    const int n = 20000;
                    for (int i = 0; i < n; ++i) sum += draft_pass_chance(step, min_p, ((float) i + 0.5f) / n, t);
                    const float f = draft_pass_floor(step, min_p, t);
                    mean = mean && std::fabs(f - (float) (sum / n)) < 2e-3f;
                    rising = rising && f > last;
                    last = f;
                }
            }
        check(mean, "the floor is the chance's mean over the number");
        check(rising, "a higher floor asks for a better chance");
        check(draft_pass_floor(0, 0.0f, 0.7f) == 0.0f && draft_pass_floor(0, -1.0f, 0.7f) == 0.0f,
              "no floor keeps every draft");
        check(draft_pass_floor(0, 1.01f, 0.7f) > 1.0f, "a floor above 1 keeps none");
        // the configs' floor at their temperature: the first draft at 0.5 under a small, a middling and a large number
        const float f = draft_pass_floor(0, 0.5f, 0.7f);
        check(draft_pass_chance(0, 0.5f, 0.1f, 0.7f) > f && draft_pass_chance(0, 0.5f, 0.97f, 0.7f) < f,
              "a draft at the floor is kept under a small number and dropped under a large one");
        check(draft_pass_chance(0, 0.4f, 0.05f, 0.7f) > f && draft_pass_chance(0, 0.7f, 0.97f, 0.7f) < f,
              "the number keeps a draft below the floor and drops one above it");
        // toward a temperature of 0 the number stops counting: the plain floor
        bool plain = true;
        for (int iu = 0; iu < 98; ++iu) {
            const float u = (float) iu / 100.0f, f0 = draft_pass_floor(0, 0.5f, 0.01f);
            plain = plain && draft_pass_chance(0, 0.55f, u, 0.01f) >= f0 && draft_pass_chance(0, 0.45f, u, 0.01f) < f0;
        }
        check(plain, "at a temperature near 0 a draft is kept by its probability alone");
    }
    std::printf(g_fail == 0 ? "draft_pass_test OK\n" : "draft_pass_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
