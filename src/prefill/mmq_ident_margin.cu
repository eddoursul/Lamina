// mmq_ident_margin - an MMQ MoE launch reads its row map (mmq::Product::ids) past the launch's rows: the kernels copy
// a tile's ids at a time, the product to the end of each expert's last tile, the stream-k fixup for any tile of the
// launch grid from every expert's first row.  mmq::ids_table_rows names the entries a map takes for that.
//
// Each product's map here has exactly those entries and ends where a mapping ends, the addresses after it reserved
// and not mapped (core::VmmRange), so a read past the map is an illegal access, which the launch's synchronize
// reports.  The gate/up shape [1280, 2560] gathered and in place (the pointer table) and the down shape [2560, 640]
// gathered: groups of 1-16 experts of 0 to 1,000 rows, every tile width, groups that end in experts of few rows or
// none, launches the stream-k grid takes with its fixup and launches it tiles.  Each product runs twice, the entries
// past the rows as the identity goes on and zeroed: the outputs, pre-filled with NaN, must be finite and the same
// bytes (nothing is written through those entries).
//
// mmq_ident_margin [--margin N] [rows of each expert ...]: the given group alone; with --margin N the maps get the
// rows and N entries more, and the run passes when a launch faults (the control: it ends its CUDA context).  0: the
// rows alone, which the product reads past; 128: a tile more, which the fixup reads past.
#include "lamina/core/vmm.hpp"
#include "lamina/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace core = lamina::core;
namespace mmq = lamina::prefill::mmq;

constexpr int G = mmq::kGatherGroupMax;
constexpr int kMostRows = 1024;   // of an expert
constexpr size_t TAIL = 4096;     // zeroed after the gathered matrices: the k loop reads past the last one

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct Dev {
    void* p = nullptr;
    explicit Dev(size_t bytes) { ck(cudaMalloc(&p, bytes), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    uint8_t* b() const { return (uint8_t*) p; }
};

void upload(void* dst, const void* src, size_t n) {
    ck(cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice), "upload");
    ck(cudaDeviceSynchronize(), "upload");
}

// out[0]: the words that differ between two outputs; out[1]: the first one's that are not finite
__global__ void compare_kernel(const uint32_t* __restrict__ a, const uint32_t* __restrict__ b, int64_t n,
                               int32_t* out) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (a[i] != b[i]) atomicAdd(out, 1);
    if ((a[i] & 0x7F800000u) == 0x7F800000u) atomicAdd(out + 1, 1);
}

// The row map's place: one chunk mapped of two reserved, the map's last entry the chunk's last word
struct Guard {
    core::VmmRange range;
    int32_t* end = nullptr;
    bool init() {
        const uint64_t chunk = core::vmm_granularity();
        if (chunk == 0 || !range.reserve(2 * chunk) || !range.map_range(0, 1, [] { return core::VmmChunk(0); }))
            return false;
        end = (int32_t*) (range.base() + chunk);
        return true;
    }
};

struct Shape {
    const char* name;
    ggml_type t;
    int64_t rows, cols;   // an expert's matrix
    bool in_place;        // through the pointer table too
};

// G experts of a shape `eb` apart (random blocks with sane fp16 scales, which both types keep first) and the
// activations of the most rows a group has
struct Experts {
    const Shape sh;
    const size_t eb;
    Dev w, x, xq, bd, a, b, res;
    Experts(const Shape& s, std::mt19937& rng)
        : sh(s),
          eb(mmq::matrix_bytes((int) s.t, s.rows, s.cols)),
          w((size_t) G * eb + TAIL),
          x((size_t) G * kMostRows * (size_t) s.cols * 4),
          xq(mmq::q8_bytes((int64_t) G * kMostRows, s.cols)),
          bd((G + 1) * 4),
          a((size_t) G * kMostRows * (size_t) s.rows * 4),
          b((size_t) G * kMostRows * (size_t) s.rows * 4),
          res(8) {
        const size_t bs = ggml_type_size(s.t);
        std::vector<uint32_t> h(((size_t) G * eb + 3) / 4);
        for (auto& v : h) v = (uint32_t) rng();
        std::uniform_real_distribution<float> scale(0.0004f, 0.0024f);
        for (size_t at = 0; at < (size_t) G * eb; at += bs) {
            const ggml_fp16_t d = ggml_fp32_to_fp16(scale(rng));
            std::memcpy((uint8_t*) h.data() + at, &d, 2);
        }
        upload(w.p, h.data(), (size_t) G * eb);
        ck(cudaMemset(w.b() + (size_t) G * eb, 0, TAIL), "tail");
        std::vector<float> hx((size_t) G * kMostRows * (size_t) s.cols);
        std::uniform_real_distribution<float> act(-1.0f, 1.0f);
        for (auto& v : hx) v = act(rng);
        upload(x.p, hx.data(), hx.size() * 4);
    }
};

struct Run {
    mmq::Context ctx;
    Guard guard;
    cudaStream_t s = nullptr;
    int64_t margin = -1;   // the control's entries past the rows (-1: the maps in ids_table_rows)
    int products = 0, fail = 0;
    ~Run() { if (s) cudaStreamDestroy(s); }
};

// A group's product: gathered, or in place where the shape takes it.  The row map, its last entry at the guard: the
// rows, then `tail` entries as the identity goes on or zeroed.  false: the launch read past it (the context is gone).
bool product(Run& r, const Experts& e, const std::vector<int>& cnt, bool in_place, int64_t tail, bool zeroed,
             float* dst) {
    const int n = (int) cnt.size();
    int64_t total = 0, max_rows = 0;
    for (const int c : cnt) {
        total += c;
        max_rows = std::max<int64_t>(max_rows, c);
    }
    int32_t* map = r.guard.end - (total + tail);
    mmq::iota(map, total + tail, r.s);
    if (zeroed) ck(cudaMemsetAsync(map + total, 0, (size_t) tail * 4, r.s), "tail");
    ck(cudaMemsetAsync(dst, 0xFF, (size_t) total * (size_t) e.sh.rows * 4, r.s), "dst");
    const void* tab[G];
    for (int z = 0; z < n; ++z) tab[z] = e.w.b() + (size_t) z * e.eb;
    mmq::Product p;
    p.type = (int) e.sh.t; p.w_rows = e.sh.rows; p.w_cols = e.sh.cols; p.n = n; p.xq = e.xq.p;
    p.bounds = (const int32_t*) e.bd.p; p.ids = map; p.total_rows = total; p.max_rows = max_rows;
    p.ids_rows = mmq::ids_table_rows(total, max_rows);   // what run() asks for: the control's map is shorter
    p.dst = dst; p.ld_dst = e.sh.rows;
    if (in_place) p.tab = tab;
    else { p.w = e.w.p; p.expert_bytes = e.eb; }
    r.ctx.run(p, r.s);
    const cudaError_t err = cudaStreamSynchronize(r.s);
    if (err == cudaErrorIllegalAddress) return false;
    ck(err, "run");
    ++r.products;
    return true;
}

// A group of experts with cnt[e] rows each through a shape's products.  Ends the process where a launch reads past its
// map: as a failure, or as the control's pass.
void group(Run& r, const Experts& e, const std::vector<int>& cnt) {
    int64_t total = 0, max_rows = 0;
    std::vector<int32_t> bounds(cnt.size() + 1, 0);
    for (size_t i = 0; i < cnt.size(); ++i) {
        total += cnt[i];
        max_rows = std::max<int64_t>(max_rows, cnt[i]);
        bounds[i + 1] = bounds[i] + cnt[i];
    }
    upload(e.bd.p, bounds.data(), bounds.size() * 4);
    mmq::quantize((const float*) e.x.p, nullptr, e.xq.p, (int) e.sh.t, e.sh.cols, e.sh.cols, total, r.s);
    const int64_t reach = mmq::ids_table_rows(total, max_rows) - total, tail = r.margin < 0 ? reach : r.margin;
    for (int in_place = 0; in_place <= (e.sh.in_place ? 1 : 0); ++in_place) {
        const char* how = in_place ? "in place" : "gathered";
        const bool within = product(r, e, cnt, in_place != 0, tail, false, (float*) e.a.p) &&
                            (r.margin >= 0 || product(r, e, cnt, in_place != 0, tail, true, (float*) e.b.p));
        if (!within) {
            std::printf("%s %s %s, %zu experts, %lld rows, %lld an expert: a launch read past its map of the rows and "
                        "%lld entries (ids_table_rows: the rows and %lld)\n", r.margin < 0 ? "FAIL" : "caught:",
                        e.sh.name, how, cnt.size(), (long long) total, (long long) max_rows, (long long) tail,
                        (long long) reach);
            std::fflush(stdout);
            std::_Exit(r.margin < 0 ? 1 : 0);   // no teardown on a faulted context
        }
        if (r.margin >= 0) continue;
        const int64_t words = total * e.sh.rows;
        int32_t res[2] = {0, 0};
        ck(cudaMemsetAsync(e.res.p, 0, 8, r.s), "compare");
        compare_kernel<<<(unsigned) ((words + 255) / 256), 256, 0, r.s>>>(
            (const uint32_t*) e.a.p, (const uint32_t*) e.b.p, words, (int32_t*) e.res.p);
        ck(cudaGetLastError(), "compare");
        ck(cudaMemcpyAsync(res, e.res.p, 8, cudaMemcpyDeviceToHost, r.s), "compare");
        ck(cudaStreamSynchronize(r.s), "compare");
        if (res[0] == 0 && res[1] == 0) continue;
        std::printf("  FAIL %s %s, %zu experts, %lld rows, %lld an expert: %d values differ with the entries past the "
                    "rows zeroed, %d are not finite\n", e.sh.name, how, cnt.size(), (long long) total,
                    (long long) max_rows, res[0], res[1]);
        ++r.fail;
    }
}

}  // namespace

int main(int argc, char** argv) {
    int64_t margin = -1;
    std::vector<int> given;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--margin") == 0 && i + 1 < argc) margin = std::atoll(argv[++i]);
        else given.push_back(std::atoi(argv[i]));
    }
    const bool bad = given.size() > (size_t) G || margin < -1 ||
                     std::any_of(given.begin(), given.end(), [](int c) { return c < 0 || c > kMostRows; }) ||
                     (!given.empty() && *std::max_element(given.begin(), given.end()) == 0);
    if (bad) {
        std::printf("usage: mmq_ident_margin [--margin N] [rows of each expert ...]   (up to %d experts of up to %d "
                    "rows)\n", G, kMostRows);
        return 2;
    }
    int dev = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
        std::printf("mmq_ident_margin: no CUDA device, skipped\n");
        return 77;
    }
    std::printf("mmq_ident_margin on %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    static const Shape kShapes[] = {{"gate/up", GGML_TYPE_IQ4_XS, 1280, 2560, true},
                                    {"down", GGML_TYPE_IQ4_NL, 2560, 640, false}};
    try {
        Run r;
        r.margin = margin;
        if (!r.guard.init()) {
            std::printf("mmq_ident_margin: no CUDA virtual memory on this device, skipped\n");
            return 77;
        }
        ck(cudaStreamCreate(&r.s), "stream");
        std::mt19937 rng(20261113);
        for (Shape sh : kShapes) {
            if (!mmq::built() || !mmq::supported((int) sh.t) || !mmq::fits((int) sh.t, sh.rows)) {
                std::printf("mmq_ident_margin: MMQ does not take %s on this build, skipped\n", ggml_type_name(sh.t));
                return 77;
            }
            sh.in_place = sh.in_place && mmq::tab_takes((int) sh.t, sh.rows, sh.cols) &&
                          mmq::tab_overread((int) sh.t, sh.cols) == 0;
            const Experts e(sh, rng);
            const int p0 = r.products, f0 = r.fail;
            if (!given.empty()) {
                group(r, e, given);
            } else {
                static const int kRows[] = {0, 1, 7, 8, 9, 31, 64, 100, 512, 1000};
                static const int kMax[] = {1, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 64, 65, 80, 81, 96, 97,
                                           112, 113, 128, 129, 200, 256, 512, 1000};
                std::uniform_int_distribution<int> pick(0, 9);
                for (int n = 1; n <= G; ++n) {   // mixed counts
                    std::vector<int> c((size_t) n);
                    for (auto& v : c) v = kRows[pick(rng)];
                    if (*std::max_element(c.begin(), c.end()) == 0) c[0] = 1;
                    group(r, e, c);
                }
                for (const int mx : kMax) {      // every tile width: an expert of mx rows
                    std::uniform_int_distribution<int> fewer(0, mx);
                    std::vector<int> c(5);
                    for (auto& v : c) v = fewer(rng);
                    c[2] = mx;
                    group(r, e, c);                     // among others of fewer
                    group(r, e, {mx, 1, 0});            // before experts of few rows or none
                    group(r, e, {mx, 1, 0, 3, 0, 1});
                }
            }
            if (margin < 0)
                std::printf("%s %-7s %-6s [%lld, %lld] gathered%s: %d products within their maps\n",
                            r.fail == f0 ? "ok  " : "FAIL", sh.name, ggml_type_name(sh.t), (long long) sh.rows,
                            (long long) sh.cols, sh.in_place ? " and in place" : "", r.products - p0);
        }
        if (margin >= 0) {
            std::printf("FAIL: no launch read past a map of its rows and %lld entries\n", (long long) margin);
            return 1;
        }
        std::printf("mmq_ident_margin: %d products within their maps, %d differ with the entries past the rows zeroed\n",
                    r.products, r.fail);
        return r.fail ? 1 : 0;
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
}
