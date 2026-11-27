// lora_parity: the qwen35 path's LoRA kernels (src/qwen35/q36_lora.cuh) on synthetic sites against an FP64 reference.
//   * sites of 1-3 products, adapters of different ranks (a rank below its padding too), F16 and F32 pairs, the
//     SwiGLU input of the down products, 1-8 rows whose scales differ (an adapter some rows leave at 0);
//   * a row's result is the same bits in a launch of 1 row and of 8 (the verify windows' rows are the decode step's);
//   * rows whose every scale is 0 are left as they were, bit for bit.
// --bench: the launch pairs of the 27B's sites (rank 32) timed, 1 and 8 rows.
#include "../qwen35/q36_lora.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

#define CK(x)                                                                                                          \
    do {                                                                                                               \
        cudaError_t e_ = (x);                                                                                          \
        if (e_ != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "%s failed: %s (line %d)\n", #x, cudaGetErrorString(e_), __LINE__);                   \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

struct Pair {   // one adapter's pair at one product, on the host (the padded layout) and the GPU
    int adapter, rank, rank_true, member;
    bool f32;
    float alpha;
    std::vector<float> a, b;   // [rank][n_in], [n_out][rank] (zero padding), the stored values (F16-rounded for F16)
    void* da = nullptr;
    void* db = nullptr;
};

void* upload(const std::vector<float>& v, bool f32) {
    void* d = nullptr;
    if (f32) {
        CK(cudaMalloc(&d, v.size() * 4));
        CK(cudaMemcpy(d, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
    } else {
        std::vector<__half> h(v.size());
        for (size_t i = 0; i < v.size(); ++i) h[i] = __float2half_rn(v[i]);
        CK(cudaMalloc(&d, h.size() * 2));
        CK(cudaMemcpy(d, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
    }
    return d;
}

struct Case {
    std::string name;
    int n_in;
    std::vector<int> n_out;   // per member
    bool swiglu;
    int n_ad;
    // per pair: adapter, rank_true, member, f32
    std::vector<std::tuple<int, int, int, bool>> pairs;
};

int fails = 0;

/// the site of a case, with the rows' scales `scale` [kRows][n_ad], on T rows of x (and u); out [member][kRows][n_out]
/// starts as `base`; returns the outputs after the two launches
std::vector<std::vector<float>> run(const Case& c, std::vector<Pair>& pairs, const std::vector<float>& x,
                                    const std::vector<float>& u, const std::vector<float>& scale,
                                    const std::vector<std::vector<float>>& base, int T, float* d_scale, float* d_part,
                                    const float* d_x, const float* d_u, std::vector<float*>& d_out, int kslice) {
    (void) x;
    (void) u;
    CK(cudaMemcpy(d_scale, scale.data(), scale.size() * 4, cudaMemcpyHostToDevice));
    for (size_t m = 0; m < c.n_out.size(); ++m)
        CK(cudaMemcpy(d_out[m], base[m].data(), base[m].size() * 4, cudaMemcpyHostToDevice));
    q36lora::Site s;
    s.x = d_x;
    s.u = c.swiglu ? d_u : nullptr;
    s.ldx = c.n_in;
    s.n_in = c.n_in;
    s.kslice = kslice;
    s.ks = (c.n_in + kslice - 1) / kslice;
    s.T = T;
    s.n_ad = c.n_ad;
    s.scale = d_scale;
    s.part = d_part;
    s.n_mem = (int) c.n_out.size();
    for (int m = 0; m < s.n_mem; ++m) {
        s.out[m] = d_out[(size_t) m];
        s.ld_out[m] = c.n_out[(size_t) m];
        s.n_out[m] = c.n_out[(size_t) m];
        s.ob[m + 1] = s.ob[m] + (c.n_out[(size_t) m] + 255) / 256;
    }
    for (const Pair& p : pairs) {
        q36lora::Seg& d = s.seg[s.n_seg++];
        d.a = p.da;
        d.b = p.db;
        d.alpha = p.alpha;
        d.rank_true = (float) p.rank_true;
        d.rank = p.rank;
        d.r0 = s.R;
        d.adapter = p.adapter;
        d.member = p.member;
        d.f32 = p.f32;
        d.g0 = s.n_groups;
        s.R += p.rank;
        s.n_groups += p.rank / q36lora::kGroup;
    }
    q36lora::k_lora_a<<<dim3((unsigned) s.n_groups, (unsigned) s.ks), 256>>>(s);
    q36lora::k_lora_b<<<(unsigned) s.ob[s.n_mem], 256>>>(s);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<std::vector<float>> out(c.n_out.size());
    for (size_t m = 0; m < c.n_out.size(); ++m) {
        out[m].resize(base[m].size());
        CK(cudaMemcpy(out[m].data(), d_out[m], out[m].size() * 4, cudaMemcpyDeviceToHost));
    }
    return out;
}

void check_case(const Case& c, std::mt19937& rng) {
    constexpr int R = q36lora::kRows;
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<Pair> pairs;
    for (const auto& [ad, rt, mem, f32] : c.pairs) {
        Pair p;
        p.adapter = ad;
        p.rank_true = rt;
        p.rank = (rt + 7) / 8 * 8;
        p.member = mem;
        p.f32 = f32;
        p.alpha = ad % 2 == 0 ? 2.f * (float) rt : 0.f;   // one with an alpha, one without
        const int n_out = c.n_out[(size_t) mem];
        p.a.assign((size_t) p.rank * c.n_in, 0.f);
        p.b.assign((size_t) n_out * p.rank, 0.f);
        const float sa = 1.f / std::sqrt((float) c.n_in), sb = 0.02f;
        for (int k = 0; k < rt; ++k)
            for (int i = 0; i < c.n_in; ++i) p.a[(size_t) k * c.n_in + i] = sa * nd(rng);
        for (int j = 0; j < n_out; ++j)
            for (int k = 0; k < rt; ++k) p.b[(size_t) j * p.rank + k] = sb * nd(rng);
        if (!f32) {   // the values the GPU holds
            for (float& v : p.a) v = __half2float(__float2half_rn(v));
            for (float& v : p.b) v = __half2float(__float2half_rn(v));
        }
        p.da = upload(p.a, f32);
        p.db = upload(p.b, f32);
        pairs.push_back(std::move(p));
    }
    std::vector<float> x((size_t) R * c.n_in), u((size_t) R * c.n_in);
    for (float& v : x) v = nd(rng);
    for (float& v : u) v = nd(rng);
    // the rows' scales: adapter a at row r is 0 for (r + a) % 3 == 0, else 0.5 + r / 8 + a / 4; row 5 has none at all
    std::vector<float> scale((size_t) R * c.n_ad, 0.f);
    for (int r = 0; r < R; ++r)
        for (int a = 0; a < c.n_ad; ++a)
            scale[(size_t) r * c.n_ad + a] =
                (r + a) % 3 == 0 || r == 5 ? 0.f : 0.5f + (float) r / 8.f + (float) a / 4.f;
    std::vector<std::vector<float>> base(c.n_out.size());
    for (size_t m = 0; m < c.n_out.size(); ++m) {
        base[m].resize((size_t) R * c.n_out[m]);
        for (float& v : base[m]) v = nd(rng);
    }
    float *d_x, *d_u, *d_scale, *d_part;
    CK(cudaMalloc(&d_x, x.size() * 4));
    CK(cudaMalloc(&d_u, u.size() * 4));
    CK(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_u, u.data(), u.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_scale, scale.size() * 4));
    const int ks_max = (c.n_in + q36lora::kChunk - 1) / q36lora::kChunk;
    CK(cudaMalloc(&d_part, (size_t) ks_max * R * q36lora::kMaxR * 4));
    std::vector<float*> d_out(c.n_out.size());
    for (size_t m = 0; m < c.n_out.size(); ++m) CK(cudaMalloc(&d_out[m], base[m].size() * 4));

    // the FP64 reference of every row
    std::vector<double> xin((size_t) R * c.n_in);
    for (size_t i = 0; i < xin.size(); ++i) {
        const float g = x[i];
        xin[i] = c.swiglu ? (double) g / (1.0 + std::exp(-(double) g)) * (double) u[i] : (double) g;
    }
    double worst = 0.0;
    for (int kslice : {1024, 512, 256}) {
        const std::vector<std::vector<float>> got8 =
            run(c, pairs, x, u, scale, base, R, d_scale, d_part, d_x, d_u, d_out, kslice);
        for (size_t m = 0; m < c.n_out.size(); ++m) {
            const int n_out = c.n_out[m];
            for (int r = 0; r < R; ++r) {
                std::vector<double> ref((size_t) n_out, 0.0);
                bool any = false;
                for (const Pair& p : pairs) {
                    if (p.member != (int) m) continue;
                    const float us = scale[(size_t) r * c.n_ad + p.adapter];
                    if (us == 0.f) continue;
                    any = true;
                    const double sc = p.alpha != 0.f ? (double) us * p.alpha / p.rank_true : (double) us;
                    for (int k = 0; k < p.rank_true; ++k) {
                        double t = 0.0;
                        for (int i = 0; i < c.n_in; ++i)
                            t += (double) p.a[(size_t) k * c.n_in + i] * xin[(size_t) r * c.n_in + i];
                        for (int j = 0; j < n_out; ++j)
                            ref[(size_t) j] += sc * (double) p.b[(size_t) j * p.rank + k] * t;
                    }
                }
                double mag = 0.0, err = 0.0;
                for (int j = 0; j < n_out; ++j) {
                    const float b0 = base[m][(size_t) r * n_out + j], g = got8[m][(size_t) r * n_out + j];
                    if (!any) {   // untouched, bit for bit
                        if (std::memcmp(&b0, &g, 4) != 0) {
                            if (fails++ < 10)
                                std::printf("FAIL %s: row %d (no adapter) member %zu column %d changed\n",
                                            c.name.c_str(), r, m, j);
                        }
                        continue;
                    }
                    mag = std::max(mag, std::fabs(ref[(size_t) j]));
                    err = std::max(err, std::fabs(((double) g - (double) b0) - ref[(size_t) j]));
                }
                if (any) {
                    const double rel = mag > 0 ? err / mag : err;
                    worst = std::max(worst, rel);
                    if (rel > 2e-5) {
                        if (fails++ < 10)
                            std::printf(
                                "FAIL %s: row %d member %zu kslice %d: max error %.3g of the delta's %.3g (%.2e)\n",
                                c.name.c_str(), r, m, kslice, err, mag, rel);
                    }
                }
            }
        }
        // each row alone (T = 1, its own row first in x): the same bits as in the 8-row launch
        for (int r = 0; r < R; ++r) {
            std::vector<float> x1(x.begin() + (long) r * c.n_in, x.begin() + (long) (r + 1) * c.n_in),
                u1(u.begin() + (long) r * c.n_in, u.begin() + (long) (r + 1) * c.n_in);
            CK(cudaMemcpy(d_x, x1.data(), x1.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_u, u1.data(), u1.size() * 4, cudaMemcpyHostToDevice));
            std::vector<float> s1(scale.begin() + (long) r * c.n_ad, scale.begin() + (long) (r + 1) * c.n_ad);
            s1.resize((size_t) R * c.n_ad, 0.f);
            std::vector<std::vector<float>> b1(c.n_out.size());
            for (size_t m = 0; m < c.n_out.size(); ++m)
                b1[m].assign(base[m].begin() + (long) r * c.n_out[m], base[m].begin() + (long) (r + 1) * c.n_out[m]);
            const std::vector<std::vector<float>> got1 =
                run(c, pairs, x1, u1, s1, b1, 1, d_scale, d_part, d_x, d_u, d_out, kslice);
            for (size_t m = 0; m < c.n_out.size(); ++m)
                if (std::memcmp(got1[m].data(), got8[m].data() + (size_t) r * c.n_out[m], (size_t) c.n_out[m] * 4) !=
                    0) {
                    if (fails++ < 10)
                        std::printf("FAIL %s: row %d member %zu kslice %d: alone it is not its bits in a window of 8\n",
                                    c.name.c_str(), r, m, kslice);
                }
        }
        CK(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_u, u.data(), u.size() * 4, cudaMemcpyHostToDevice));
    }
    std::printf("%-34s worst error %.2e of the delta%s\n", c.name.c_str(), worst, "");
    for (Pair& p : pairs) {
        cudaFree(p.da);
        cudaFree(p.db);
    }
    for (float* d : d_out) cudaFree(d);
    cudaFree(d_x);
    cudaFree(d_u);
    cudaFree(d_scale);
    cudaFree(d_part);
}

__global__ void spin(long long cycles) {
    const long long t0 = clock64();
    while (clock64() - t0 < cycles) {}
}

/// --bench: the 27B's sites with one rank-32 F16 adapter, 1 and 8 rows, a launch pair's median time
void bench() {
    struct B {
        const char* name;
        int n_in;
        std::vector<int> n_out;
        bool swiglu;
    };
    const B sites[] = {{"q / k / v (5120 -> 12288, 1024, 1024)", 5120, {12288, 1024, 1024}, false},
                       {"attention output (6144 -> 5120)", 6144, {5120}, false},
                       {"gate / up (5120 -> 17408, 17408)", 5120, {17408, 17408}, false},
                       {"down, SwiGLU input (17408 -> 5120)", 17408, {5120}, true}};
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (const B& b : sites) {
        q36lora::Site s;
        std::vector<void*> keep;
        const int rank = 32;
        float *d_x, *d_u, *d_scale, *d_part;
        CK(cudaMalloc(&d_x, (size_t) 8 * b.n_in * 4));
        CK(cudaMalloc(&d_u, (size_t) 8 * b.n_in * 4));
        CK(cudaMemset(d_x, 0, (size_t) 8 * b.n_in * 4));
        CK(cudaMemset(d_u, 0, (size_t) 8 * b.n_in * 4));
        std::vector<float> sc(8, 1.f);
        CK(cudaMalloc(&d_scale, 32));
        CK(cudaMemcpy(d_scale, sc.data(), 32, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d_part, (size_t) 68 * 8 * q36lora::kMaxR * 4));
        s.x = d_x;
        s.u = b.swiglu ? d_u : nullptr;
        s.ldx = s.n_in = b.n_in;
        const char* kv = std::getenv("LAMINA_Q36_LORA_KSLICE");
        s.kslice = std::min(q36lora::kSliceMax,
                            kv ? std::max(1, std::atoi(kv) / q36lora::kChunk) * q36lora::kChunk : q36lora::kSliceMax);
        s.ks = (b.n_in + s.kslice - 1) / s.kslice;
        s.n_ad = 1;
        s.scale = d_scale;
        s.part = d_part;
        s.n_mem = (int) b.n_out.size();
        for (int m = 0; m < s.n_mem; ++m) {
            float* o = nullptr;
            CK(cudaMalloc(&o, (size_t) 8 * b.n_out[(size_t) m] * 4));
            CK(cudaMemset(o, 0, (size_t) 8 * b.n_out[(size_t) m] * 4));
            keep.push_back(o);
            s.out[m] = o;
            s.ld_out[m] = s.n_out[m] = b.n_out[(size_t) m];
            s.ob[m + 1] = s.ob[m] + (b.n_out[(size_t) m] + 255) / 256;
            std::vector<float> av((size_t) rank * b.n_in), bv((size_t) b.n_out[(size_t) m] * rank);
            for (float& v : av) v = 0.01f * nd(rng);
            for (float& v : bv) v = 0.01f * nd(rng);
            q36lora::Seg& d = s.seg[s.n_seg++];
            d.a = upload(av, false);
            d.b = upload(bv, false);
            keep.push_back((void*) d.a);
            keep.push_back((void*) d.b);
            d.alpha = 64.f;
            d.rank_true = (float) rank;
            d.rank = rank;
            d.r0 = s.R;
            d.member = m;
            d.g0 = s.n_groups;
            s.R += rank;
            s.n_groups += rank / q36lora::kGroup;
        }
        for (int T : {1, 8}) {
            s.T = T;
            cudaStream_t st;
            CK(cudaStreamCreate(&st));
            // a graph of 50 launch pairs, as the engine's decode graphs hold them
            cudaGraph_t g;
            cudaGraphExec_t gx;
            CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
            const char* stage = std::getenv("LAMINA_LORA_BENCH_STAGE");   // a, b or both (the default)
            const bool run_a = stage == nullptr || std::strcmp(stage, "b") != 0;
            const bool run_b = stage == nullptr || std::strcmp(stage, "a") != 0;
            for (int i = 0; i < 50; ++i) {
                if (run_a) q36lora::k_lora_a<<<dim3((unsigned) s.n_groups, (unsigned) s.ks), 256, 0, st>>>(s);
                if (run_b) q36lora::k_lora_b<<<(unsigned) s.ob[s.n_mem], 256, 0, st>>>(s);
            }
            CK(cudaStreamEndCapture(st, &g));
            CK(cudaGraphInstantiate(&gx, g, 0));
            for (int w = 0; w < 3; ++w) CK(cudaGraphLaunch(gx, st));
            cudaEvent_t e0, e1;
            CK(cudaEventCreate(&e0));
            CK(cudaEventCreate(&e1));
            std::vector<float> each;
            for (int w = 0; w < 21; ++w) {   // each launch after ~3 ms of a one-thread spin: the clocks of a busy GPU
                spin<<<1, 1, 0, st>>>(5000000);
                CK(cudaEventRecord(e0, st));
                CK(cudaGraphLaunch(gx, st));
                CK(cudaEventRecord(e1, st));
                CK(cudaEventSynchronize(e1));
                float ms = 0.f;
                CK(cudaEventElapsedTime(&ms, e0, e1));
                each.push_back(ms);
            }
            std::sort(each.begin(), each.end());
            std::printf("bench %-40s T=%d: %.2f us a launch pair (ks %d, %d + %d blocks)\n", b.name, T,
                        each[each.size() / 2] * 1000.f / 50, s.ks, s.n_groups * s.ks, s.ob[s.n_mem]);
            cudaGraphExecDestroy(gx);
            cudaGraphDestroy(g);
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);
            cudaStreamDestroy(st);
        }
        for (void* p : keep) cudaFree(p);
        cudaFree(d_x);
        cudaFree(d_u);
        cudaFree(d_scale);
        cudaFree(d_part);
    }
}

}   // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        bench();
        return 0;
    }
    std::mt19937 rng(1234);
    const std::vector<Case> cases = {
        {"q/k/v: 3 products, 2 adapters",
         5120,
         {1536, 256, 256},
         false,
         2,
         {{0, 32, 0, false}, {0, 32, 1, false}, {0, 32, 2, false}, {1, 16, 0, true}, {1, 16, 2, true}}},
        {"gate/up: rank 12 padded to 16", 5120, {2048, 2048}, false, 1, {{0, 12, 0, false}, {0, 12, 1, false}}},
        {"down: SwiGLU input, 3 adapters",
         4352,
         {1024},
         true,
         3,
         {{0, 32, 0, false}, {1, 8, 0, true}, {2, 24, 0, false}}},
        {"one F32 pair, a tail chunk", 2056, {777}, false, 1, {{0, 40, 0, true}}},
    };
    for (const Case& c : cases) check_case(c, rng);
    if (fails) {
        std::printf("lora_parity: %d failures\n", fails);
        return 1;
    }
    std::printf("lora_parity: every site within 2e-5 of the FP64 reference, rows alone bitwise equal to their window, "
                "rows without an adapter untouched\n");
    return 0;
}
