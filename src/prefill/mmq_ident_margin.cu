// mmq_ident_margin - the MoE identity row map (mmq::Product::ids) must be sized mmq::ids_table_rows(total), not the
// exact total rows: the MMQ kernels read a whole J-wide tile of ids from each expert's first row, and the stream-k
// fixup a tile past the last expert's rows, so an exact-size table is read past its end (the prompt path's carve
// at experts.cpp sized `ident` to exactly ROWS and the last carve in the arena, the page-fault-prone slot; upstream
// proved 311 invalid reads and an Xid-31 illegal access on sm_86-class hardware with the same kernels).
//
// Two independent guarantees, so the fix is red->green checkable:
//
//  1. EXACTNESS (no sanitizer needed, runs in normal CI on a GPU): the same product, run with the padded row map
//     filled by iota (the tail ids point at their own row) vs the tail zeroed, gives byte-identical output.  So the
//     margin's values are inert and the fix cannot change an emitted token.  Skips (77) without a CUDA device.
//
//  2. MEMORY SAFETY (compute-sanitizer, registered only when compute-sanitizer is on PATH): this same margin-mode
//     launch run under `--tool memcheck --error-exitcode 1` must report no invalid read.  It trips the moment the
//     margin shrinks below the kernels' furthest read - the regression gate that keeps experts.cpp's carve padded.
//
// RED negative control (NOT a registered test; it faults on purpose), the one-time proof the bug is real on this
// model/pack:  compute-sanitizer --tool memcheck build/.../mmq_ident_margin --exact
// `--exact` sizes the row map to exactly the total rows; memcheck reports invalid reads of the ids on the first
// launch.  A plain (unsanitized) --exact run is NOT a reliable detector: the fault only lands when the table ends
// an allocation, so it usually reads valid-but-unrelated memory silently - which is why memcheck is the oracle.
#include "lamina/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
namespace mmq = lamina::prefill::mmq;

constexpr size_t TAIL = 4096;              // a zeroed tail after the gathered matrices (their k-loop over-read)

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct Dev {
    void* p = nullptr;
    size_t n = 0;
    explicit Dev(size_t bytes) : n(bytes) { ck(cudaMalloc(&p, bytes), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    uint8_t* b() const { return (uint8_t*) p; }
};

void upload(void* dst, const void* src, size_t n) {
    ck(cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice), "upload");
    ck(cudaDeviceSynchronize(), "upload");
}

// random blocks with sane fp16 scales where the format keeps them
void fill_matrix(uint8_t* m, ggml_type t, int64_t rows, int64_t cols, std::mt19937& rng) {
    const size_t bs = ggml_type_size(t), nb = (size_t) rows * (size_t) (cols / ggml_blck_size(t));
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> sc(0.0004f, 0.0024f);
    for (size_t i = 0; i < nb * bs; ++i) m[i] = (uint8_t) byte(rng);
    for (size_t b = 0; b < nb; ++b) {
        const ggml_fp16_t d = ggml_fp32_to_fp16(sc(rng));
        std::memcpy(m + b * bs, &d, 2);
    }
}

// One MoE group of G experts through the gathered MMQ product, with the row map sized `ids_rows`.  Fills `out` with
// the product's dst bytes.  `tail_zero` zeroes the map past the total rows instead of continuing the identity.
void product(mmq::Context& ctx, ggml_type t, int64_t W_ROWS, int64_t W_COLS, const std::vector<int>& cnt,
             int64_t ids_rows, bool tail_zero, uint32_t seed, cudaStream_t s, std::vector<uint8_t>& out) {
    std::mt19937 rng(seed);   // seeded, not shared: two calls with the same seed see identical weights/activations,
                              // so the ONLY thing that can differ between an identity- and a zero-tail run is the map
    const int n = (int) cnt.size();
    int64_t total = 0, max_rows = 0;
    std::vector<int32_t> bounds(cnt.size() + 1, 0);
    for (size_t i = 0; i < cnt.size(); ++i) {
        total += cnt[i];
        max_rows = std::max<int64_t>(max_rows, cnt[i]);
        bounds[i + 1] = bounds[i] + cnt[i];
    }
    const size_t eb = mmq::matrix_bytes((int) t, W_ROWS, W_COLS);
    const int64_t tr = std::max<int64_t>(1, total);

    Dev x((size_t) tr * W_COLS * 4), xq(mmq::q8_bytes(tr, W_COLS)), ids((size_t) ids_rows * 4), bd(bounds.size() * 4);
    std::vector<float> hx((size_t) tr * W_COLS);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : hx) v = nd(rng);
    upload(x.p, hx.data(), hx.size() * 4);
    upload(bd.p, bounds.data(), bounds.size() * 4);
    if (tail_zero) {
        mmq::iota((int32_t*) ids.p, tr, s);
        ck(cudaMemsetAsync(ids.b() + (size_t) tr * 4, 0, (size_t) (ids_rows - tr) * 4, s), "tail");
    } else {
        mmq::iota((int32_t*) ids.p, ids_rows, s);   // the whole padded map, identity (as experts.cpp fills it)
    }
    if (total > 0) mmq::quantize((const float*) x.p, nullptr, xq.p, (int) t, W_COLS, W_COLS, total, s);
    ck(cudaStreamSynchronize(s), "acts");

    std::vector<std::vector<uint8_t>> w((size_t) n, std::vector<uint8_t>(eb));
    for (auto& m : w) fill_matrix(m.data(), t, W_ROWS, W_COLS, rng);
    Dev grp((size_t) n * eb + TAIL + 4096);
    for (int z = 0; z < n; ++z) upload(grp.b() + (size_t) z * eb, w[(size_t) z].data(), eb);
    ck(cudaMemset(grp.b() + (size_t) n * eb, 0, TAIL), "tail");

    const size_t out_bytes = (size_t) tr * W_ROWS * 4;
    Dev dst(out_bytes);   // NaN-prefilled below: a row the kernel leaves unwritten is not finite

    mmq::Product p;
    p.type = (int) t; p.w_rows = W_ROWS; p.w_cols = W_COLS; p.n = n; p.xq = xq.p; p.bounds = (const int32_t*) bd.p;
    p.ids = (const int32_t*) ids.p; p.total_rows = total; p.max_rows = max_rows; p.ld_dst = W_ROWS;
    p.w = grp.p; p.expert_bytes = eb; p.dst = (float*) dst.p;
    ck(cudaMemset(dst.p, 0x7F, out_bytes), "dst");
    ck(cudaDeviceSynchronize(), "dst");
    ctx.run(p, s);
    ck(cudaStreamSynchronize(s), "run");

    out.assign(out_bytes, 0);
    ck(cudaMemcpy(out.data(), dst.p, out_bytes, cudaMemcpyDeviceToHost), "dtoh");
}

bool all_finite(const std::vector<uint8_t>& b) {
    for (size_t i = 0; i < b.size(); i += 4) {
        float v;
        std::memcpy(&v, b.data() + i, 4);
        if (!(v == v && v - v == 0.0f)) return false;
    }
    return true;
}

// ---- perf probes (the --perf mode): the change's only recurring cost is `iota` over ids_table_rows(n) instead of
// n rows.  Measure that directly, and show the MMQ product launch time is independent of the id-table length (the
// kernel reads only up to its tile bound past the last row, never the whole table).  Min-of-rounds to cut noise.
float time_ms(cudaStream_t s, int iters, const std::function<void()>& fn) {
    fn();
    ck(cudaStreamSynchronize(s), "warm");
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a, s);
    for (int i = 0; i < iters; ++i) fn();
    cudaEventRecord(b, s);
    ck(cudaEventSynchronize(b), "perf");
    float ms = 0.f; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / (float) iters;
}

template <typename F>
float best_ms(cudaStream_t s, int rounds, int iters, const F& fn) {
    float best = 1e30f;
    for (int r = 0; r < rounds; ++r) best = std::min(best, time_ms(s, iters, fn));
    return best;
}

// A live MMQ product group whose ctx.run can be timed repeatedly; the id-table buffer is swapped to show that its
// length does not move the launch time.  Every Dev is sized in the member-init list (Dev has no move/reassign).
static int64_t g_total(const std::vector<int>& c) { int64_t s = 0; for (int v : c) s += v; return s; }
static int64_t g_maxr(const std::vector<int>& c) { int64_t m = 0; for (int v : c) m = std::max(m, (int64_t) v); return m; }

struct PerfGroup {
    const int64_t eb, total, pad_rows, big_rows;
    Dev x, xq, bd, ids_small, ids_big, grp, dst;
    std::vector<int32_t> hb;
    mmq::Product p{};
    PerfGroup(ggml_type t, int64_t W_ROWS, int64_t W_COLS, const std::vector<int>& cnt, cudaStream_t s)
        : eb((int64_t) mmq::matrix_bytes((int) t, W_ROWS, W_COLS)),
          total(g_total(cnt)),
          pad_rows(mmq::ids_table_rows(std::max<int64_t>(1, g_total(cnt)))),
          big_rows(4 * mmq::ids_table_rows(std::max<int64_t>(1, g_total(cnt)))),
          x((size_t) std::max<int64_t>(1, total) * W_COLS * 4),
          xq(mmq::q8_bytes(std::max<int64_t>(1, total), W_COLS)),
          bd((cnt.size() + 1) * 4),
          ids_small((size_t) pad_rows * 4), ids_big((size_t) big_rows * 4),
          grp((size_t) cnt.size() * (size_t) eb + TAIL + 4096),
          dst((size_t) std::max<int64_t>(1, total) * W_ROWS * 4) {
        const int64_t tr = std::max<int64_t>(1, total), max_rows = g_maxr(cnt);
        hb.assign(cnt.size() + 1, 0);
        for (size_t i = 0; i < cnt.size(); ++i) hb[i + 1] = hb[i] + cnt[i];
        std::mt19937 rng(99u);
        std::vector<float> hx((size_t) tr * W_COLS);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : hx) v = nd(rng);
        upload(x.p, hx.data(), hx.size() * 4);
        upload(bd.p, hb.data(), hb.size() * 4);
        mmq::iota((int32_t*) ids_big.p, big_rows, s);
        mmq::iota((int32_t*) ids_small.p, pad_rows, s);   // both a valid identity map past the read bound
        if (total > 0) mmq::quantize((const float*) x.p, nullptr, xq.p, (int) t, W_COLS, W_COLS, total, s);
        std::vector<std::vector<uint8_t>> w(cnt.size(), std::vector<uint8_t>((size_t) eb));
        for (auto& m : w) fill_matrix(m.data(), t, W_ROWS, W_COLS, rng);
        for (size_t z = 0; z < w.size(); ++z) upload(grp.b() + z * (size_t) eb, w[z].data(), (size_t) eb);
        ck(cudaMemset(grp.b() + cnt.size() * (size_t) eb, 0, TAIL), "perf-tail");
        ck(cudaStreamSynchronize(s), "perf-prep");
        p.type = (int) t; p.w_rows = W_ROWS; p.w_cols = W_COLS; p.n = (int) cnt.size(); p.xq = xq.p;
        p.bounds = (const int32_t*) bd.p; p.ids = (const int32_t*) ids_small.p; p.total_rows = total;
        p.max_rows = max_rows; p.ld_dst = W_ROWS; p.w = grp.p; p.expert_bytes = (size_t) eb; p.dst = (float*) dst.p;
    }
};

}  // namespace

int main(int argc, char** argv) {
    // the red negative control (faults on purpose) and the perf probe, both off by default
    const bool exact = argc > 1 && std::strcmp(argv[1], "--exact") == 0;
    const bool perf = argc > 1 && std::strcmp(argv[1], "--perf") == 0;

    int dev = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
        std::printf("mmq_ident_margin: no CUDA device, skipped\n");
        return 77;
    }
    std::printf("mmq_ident_margin on %s (sm_%d%d)%s\n", prop.name, prop.major, prop.minor,
                exact ? " [--exact]" : perf ? " [--perf]" : "");
    try {
        const ggml_type t = GGML_TYPE_IQ4_XS;   // a config's gate/up type, through MMQ on both sm_86 and sm_89
        const int64_t W_ROWS = 1280, W_COLS = 2560;
        if (!mmq::built() || !mmq::supported((int) t) || !mmq::fits((int) t, W_ROWS)) {
            std::printf("mmq_ident_margin: MMQ does not cover %s on this build, skipped\n", ggml_type_name(t));
            return 77;
        }

        // A worst-case group: full G experts, one with zero rows (its tile still read), a large one (wide tile +
        // stream-k grid), and a small LAST expert (so a full tile read from its first row runs past the total).
        std::vector<int> cnt = {7, 0, 64, 1, 255, 8, 33, 1024, 0, 2, 3, 4, 5, 6, 9, 1};
        int64_t total = 0, max_rows = 0;
        for (int c : cnt) { total += c; max_rows = std::max<int64_t>(max_rows, c); }

        cudaStream_t s = nullptr;
        ck(cudaStreamCreate(&s), "stream");
        mmq::Context ctx;

        if (perf) {
            // (1) the fix's only added recurring work: one iota per layer-call over ids_table_rows(ROWS) instead of
            // the old exact capacity ROWS.  That size is fixed capacity, independent of the chunk's token count.
            const int64_t cap = 8192;                        // experts.cpp ROWS
            const int64_t padded = mmq::ids_table_rows(cap); // 16512: the length the fix iota's now
            Dev tbl((size_t) padded * 4);
            ck(cudaMemsetAsync(tbl.p, 0, tbl.n, s), "perf-zero");
            const int iters = 4000, rounds = 5;
            const float us_pre  = best_ms(s, rounds, iters, [&] { mmq::iota((int32_t*) tbl.p, cap, s); }) * 1e3f;
            const float us_post = best_ms(s, rounds, iters, [&] { mmq::iota((int32_t*) tbl.p, padded, s); }) * 1e3f;
            std::printf("PERF iota/layer-call: %lld rows %.3f us | %lld rows %.3f us | delta %+.3f us\n",
                        (long long) cap, us_pre, (long long) padded, us_post, us_post - us_pre);

            // (2) a realistic full-chunk group (total = ROWS rows over 16 experts): does a longer id table slow the
            // MMQ launch?  The kernel reads only its tile bound past the last row, never the whole table, so no.
            // Sustained int8 MMQ work throttles the GPU, so interleave the two arms (alternating which runs first)
            // with a cooldown per block and take the min of each across rounds - that cancels any clock drift.
            const std::vector<int> pcnt = {1320, 1100, 950, 820, 700, 640, 560, 500, 420, 360, 300, 220, 150, 90, 60, 2};
            PerfGroup g(t, W_ROWS, W_COLS, pcnt, s);
            mmq::Product with_pad = g.p;  with_pad.ids = (const int32_t*) g.ids_small.p;  // ids_table_rows(total)
            mmq::Product with_big = g.p;  with_big.ids = (const int32_t*) g.ids_big.p;    // 4x longer, same values
            const int kit = 60, mrounds = 7;
            const auto ms = [&](mmq::Product& pr) { return time_ms(s, kit, [&] { ctx.run(pr, s); }) * 1e3f; };
            auto cool = [] { std::this_thread::sleep_for(std::chrono::milliseconds(300)); };
            float mn_pad = 1e30f, mn_big = 1e30f, first_pad = 0.f, first_big = 0.f;
            for (int r = 0; r < mrounds; ++r) {
                float a, b;
                if (r % 2 == 0) { cool(); a = ms(with_pad); cool(); b = ms(with_big); }
                else            { cool(); b = ms(with_big); cool(); a = ms(with_pad); }
                if (r == 0) { first_pad = a; first_big = b; }   // pad-first, big-second: the old naive order
                mn_pad = std::min(mn_pad, a); mn_big = std::min(mn_big, b);
            }
            std::printf("PERF MMQ launch (interleaved, cooldown, min of %d): ids_table_rows %.3f us | 4x longer "
                        "%.3f us | delta %+.3f us\n", mrounds, mn_pad, mn_big, mn_big - mn_pad);
            std::printf("PERF (for contrast, the naive pad-then-big first round: %.3f us then %.3f us -> a %+.0f us "
                        "gap that is clock drift, not the table)\n", first_pad, first_big, first_big - first_pad);
            cudaStreamDestroy(s);
            return 0;
        }

        if (exact) {
            std::vector<uint8_t> out;
            product(ctx, t, W_ROWS, W_COLS, cnt, total, false, 7u, s, out);   // row map sized to exactly total
            std::printf("mmq_ident_margin --exact: ran with a row map of exactly %lld rows (read past its end; "
                        "run this under compute-sanitizer to see the invalid reads)\n", (long long) total);
            cudaStreamDestroy(s);
            return 0;
        }

        // The margin must bound the furthest read any launch this table serves: total + roundup(max_rows,128) + 128.
        const int64_t want = total + ((max_rows + 127) & ~int64_t(127)) + 128;
        const int64_t rows = mmq::ids_table_rows(total);
        if (rows < want) {
            std::printf("FAIL ids_table_rows(%lld) = %lld < the furthest read %lld\n", (long long) total, (long long) rows,
                        (long long) want);
            cudaStreamDestroy(s);
            return 1;
        }

        std::printf("mmq_ident_margin: ids_table_rows(%lld) = %lld (>= read bound %lld), %zu experts, max_rows %lld\n",
                    (long long) total, (long long) rows, (long long) want, cnt.size(), (long long) max_rows);
        int fail = 0;
        for (uint32_t seed : {11u, 22u, 33u, 44u, 55u}) {
            std::vector<uint8_t> a, b;
            product(ctx, t, W_ROWS, W_COLS, cnt, rows, false, seed, s, a);   // padded map, identity tail (production)
            product(ctx, t, W_ROWS, W_COLS, cnt, rows, true, seed, s, b);    // padded map, zero tail
            if (!all_finite(a)) { std::printf("FAIL seed %u: a row read past the map landed on a non-finite value\n", seed); ++fail; }
            if (a != b) { std::printf("FAIL seed %u: the row map's padding changed the output (it must be inert)\n", seed); ++fail; }
        }
        cudaStreamDestroy(s);
        if (fail) return 1;
        std::printf("mmq_ident_margin: ok, the margin is inert (identity and zero tails byte-identical over 5 seeds)\n");
        return 0;
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
}
