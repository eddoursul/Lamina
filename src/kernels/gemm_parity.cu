// src/kernels/gemm_parity.cu - the matrix products of kernels/gemm.hpp against FP64 on the host, on synthetic operands.
//
// gemm16 (halves and BF16) and gemm32 at shapes that take every form of the kernels: tiles past the rows and the
// outputs, one to eight outputs a row (the row-dot kernel), K whole and split (with and without a scratch), K that ends
// inside a slab and inside a granule, rows that are not 16-byte aligned (an odd K, an odd row distance, an operand at
// an odd address: the plain loads), a row distance of Y past N, alpha and beta, halves out; every tensor-core instance
// by name, whichever of them the card's plan takes (K whole and split, both orders of the tiles); gemm16_grouped at
// several row counts.  Every output is compared with the FP64 sum of the same operands, the difference against the
// sum of the products' magnitudes; each call is made twice and must give the same bits.
//   gemm_parity [device]
#include "lamina/kernels/gemm.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

using lamina::kernels::Gemm16;
using lamina::kernels::Gemm32;
using lamina::kernels::GemmScratch;

bool ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) std::printf("FAIL: %s: %s\n", what, cudaGetErrorString(e));
    return e == cudaSuccess;
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    if (!ck(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)), "cudaMalloc")) std::exit(2);
    return p;
}
uint32_t hash32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}
float value(uint32_t i) { return ((float) (hash32(i * 2654435761u + 977u) >> 8) / 8388608.0f - 1.0f) * 0.75f; }
uint16_t bf16_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
float bf16_value(uint16_t h) {
    const uint32_t u = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
uint16_t half_bits(float f) {   // round to nearest even, the values here are normal halves
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t sign = (u >> 16) & 0x8000u;
    const int e = (int) ((u >> 23) & 255) - 127 + 15;
    uint32_t m = u & 0x7fffffu;
    if (e <= 0) return (uint16_t) sign;   // (below the normal halves: none of the test's values)
    uint32_t h = sign | ((uint32_t) e << 10) | (m >> 13);
    const uint32_t rest = m & 0x1fffu;
    if (rest > 0x1000u || (rest == 0x1000u && (h & 1u))) ++h;
    return (uint16_t) h;
}
float half_value(uint16_t h) {
    const uint32_t s = (h >> 15) & 1u, e = (h >> 10) & 31u, m = h & 1023u;
    float v;
    if (e == 0) v = std::ldexp((float) m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = std::ldexp((float) (m | 1024u), (int) e - 25);
    return s ? -v : v;
}

struct Case {
    int kind;   // 0 halves, 1 BF16, 2 floats
    int T, N, K;
    int ldx_extra = 0, ldy_extra = 0;   // values past K / N to the next row
    int x_shift = 0;                    // elements the operands start past an aligned address
    float alpha = 1.0f, beta = 0.0f;
    bool y_half = false;
    bool scratch = true;
    int form = -1;          // a tensor-core instance by its index (-1: the plan's)
    int splits = 1;         // its K in parts
    bool x_inner = false;   // its tiles' order
};

int g_fail = 0, g_cases = 0;
uint8_t *g_dx, *g_dw, *g_dy, *g_scratch;
constexpr size_t kOperand = 96u << 20, kScratch = 32u << 20;   // bytes

void run(const Case& c, cudaStream_t st) {
    ++g_cases;
    const int es = c.kind == 2 ? 4 : 2, ys = c.y_half ? 2 : 4;
    const int64_t ldx = c.K + c.ldx_extra, ldy = c.N + c.ldy_extra;
    const size_t nx = (size_t) (c.T - 1) * ldx + c.K, nw = (size_t) c.N * c.K, ny = (size_t) (c.T - 1) * ldy + c.N;
    if ((nx + c.x_shift) * es > kOperand || (nw + c.x_shift) * es > kOperand || (ny + 1) * 4 > kOperand) {
        std::printf("FAIL: a case past the test's buffers\n");
        ++g_fail;
        return;
    }
    std::vector<float> xf(nx), wf(nw), y0(ny);
    std::vector<uint8_t> xb(nx * es), wb(nw * es), yb(ny * ys);
    for (size_t i = 0; i < nx; ++i) {
        const float v = value((uint32_t) i * 3u + (uint32_t) g_cases * 7919u);
        if (c.kind == 2) {
            xf[i] = v;
            std::memcpy(&xb[i * 4], &v, 4);
        } else {
            const uint16_t h = c.kind == 1 ? bf16_bits(v) : half_bits(v);
            xf[i] = c.kind == 1 ? bf16_value(h) : half_value(h);
            std::memcpy(&xb[i * 2], &h, 2);
        }
    }
    for (size_t i = 0; i < nw; ++i) {
        const float v = value((uint32_t) i * 5u + 1u + (uint32_t) g_cases * 104729u);
        if (c.kind == 2) {
            wf[i] = v;
            std::memcpy(&wb[i * 4], &v, 4);
        } else {
            const uint16_t h = c.kind == 1 ? bf16_bits(v) : half_bits(v);
            wf[i] = c.kind == 1 ? bf16_value(h) : half_value(h);
            std::memcpy(&wb[i * 2], &h, 2);
        }
    }
    for (size_t i = 0; i < ny; ++i) {   // Y before the product: read where beta is
                                        // not 0, kept between the rows' outputs
        const float v = value((uint32_t) i * 11u + 5u);
        if (c.y_half) {
            const uint16_t h = half_bits(v);
            y0[i] = half_value(h);
            std::memcpy(&yb[i * 2], &h, 2);
        } else {
            y0[i] = v;
            std::memcpy(&yb[i * 4], &v, 4);
        }
    }
    uint8_t* const dx = g_dx + (size_t) c.x_shift * es;
    uint8_t* const dw = g_dw + (size_t) c.x_shift * es;
    cudaMemcpyAsync(dx, xb.data(), xb.size(), cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(dw, wb.data(), wb.size(), cudaMemcpyHostToDevice, st);
    const GemmScratch sc = c.scratch ? GemmScratch{g_scratch, kScratch} : GemmScratch{};
    std::vector<uint8_t> got[2];
    for (int rep = 0; rep < 2; ++rep) {
        cudaMemcpyAsync(g_dy, yb.data(), yb.size(), cudaMemcpyHostToDevice, st);
        if (c.kind == 2) {
            Gemm32 p;
            p.x = (const float*) dx;
            p.w = (const float*) dw;
            p.y = (float*) g_dy;
            p.T = c.T;
            p.N = c.N;
            p.K = c.K;
            p.ldx = ldx;
            p.ldy = ldy;
            p.alpha = c.alpha;
            p.beta = c.beta;
            if (c.form >= 0) lamina::kernels::gemm32_form(c.form, c.splits, c.x_inner, p, sc, st);
            else lamina::kernels::gemm32(p, sc, st);
        } else {
            Gemm16 p;
            p.x = (const uint16_t*) dx;
            p.w = (const uint16_t*) dw;
            p.y = g_dy;
            p.T = c.T;
            p.N = c.N;
            p.K = c.K;
            p.ldx = ldx;
            p.ldy = ldy;
            p.alpha = c.alpha;
            p.beta = c.beta;
            p.bf16 = c.kind == 1;
            p.y_half = c.y_half;
            if (c.form >= 0) lamina::kernels::gemm16_form(c.form, c.splits, c.x_inner, p, sc, st);
            else lamina::kernels::gemm16(p, sc, st);
        }
        if (!ck(cudaStreamSynchronize(st), "a product")) {
            ++g_fail;
            return;
        }
        got[rep].resize(yb.size());
        cudaMemcpyAsync(got[rep].data(), g_dy, yb.size(), cudaMemcpyDeviceToHost, st);
        cudaStreamSynchronize(st);
    }
    const bool same = got[0] == got[1];
    double worst = 0;
    size_t bad = 0, touched = 0;
    for (int t = 0; t < c.T; ++t)
        for (int64_t n = 0; n < ldy; ++n) {
            const size_t at = (size_t) t * ldy + n;
            if (at >= ny) break;
            float g;
            if (c.y_half) {
                uint16_t h;
                std::memcpy(&h, &got[0][at * 2], 2);
                g = half_value(h);
            } else std::memcpy(&g, &got[0][at * 4], 4);
            if (n >= c.N) {   // between two rows' outputs: as before
                touched += std::memcmp(&got[0][at * ys], &yb[at * ys], ys) != 0;
                continue;
            }
            double s = 0, mag = 0;
            const float* xr = &xf[(size_t) t * ldx];
            const float* wr = &wf[(size_t) n * c.K];
            for (int k = 0; k < c.K; ++k) {
                const double pr = (double) xr[k] * (double) wr[k];
                s += pr;
                mag += std::fabs(pr);
            }
            const double want = (double) c.alpha * s + (double) c.beta * (double) y0[at];
            const double room = std::fabs((double) c.alpha) * mag + std::fabs((double) c.beta * (double) y0[at]);
            // FP32 sums: a few ulps of the magnitudes' sum; halves out: their own rounding besides
            const double tol = 3e-6 * room + 1e-30 + (c.y_half ? 6e-4 * std::fabs(want) + 1e-7 : 0.0);
            const double err = std::fabs((double) g - want);
            worst = std::max(worst, err / (room + 1e-30));
            bad += !(err <= tol);
        }
    const bool ok = same && bad == 0 && touched == 0;
    if (!ok) ++g_fail;
    char form[64] = "";
    char name[32];
    if (c.form >= 0) {
        if (c.kind == 2) lamina::kernels::gemm32_forms(c.form, name, sizeof name);
        else lamina::kernels::gemm16_forms(c.form, name, sizeof name);
        std::snprintf(form, sizeof form, " [instance %s x%d%s]", name, c.splits, c.x_inner ? ", X's tiles inner" : "");
    } else {   // (of aligned rows: others take plain loads)
        const size_t room = c.scratch ? kScratch : 0;
        const int z = c.kind == 2 ? lamina::kernels::gemm32_plan(c.T, c.N, c.K, room, name, sizeof name)
                                  : lamina::kernels::gemm16_plan(c.T, c.N, c.K, room, name, sizeof name);
        std::snprintf(form, sizeof form, " [%s x%d]", name, z);
    }
    if (!ok || std::getenv("LAMINA_GEMM_PARITY_VERBOSE"))
        std::printf(
            "%s %s T=%-4d N=%-5d K=%-5d ldx+%d ldy+%d shift %d alpha %.2f beta %.2f%s%s%s: worst |error| / sum of "
            "magnitudes %.2e, %zu outputs off, %zu values outside the outputs changed, two runs %s\n",
            ok ? "ok  " : "FAIL",
            c.kind == 0   ? "f16 "
            : c.kind == 1 ? "bf16"
                          : "f32 ",
            c.T, c.N, c.K, c.ldx_extra, c.ldy_extra, c.x_shift, c.alpha, c.beta, c.y_half ? " halves out" : "",
            c.scratch ? "" : " no scratch", form, worst, bad, touched, same ? "the same bits" : "DIFFER");
}

// gemm16_grouped: n problems of one N and K, their rows side by side in one buffer
void run_grouped(int kind, const std::vector<int>& rows, int N, int K, cudaStream_t st) {
    ++g_cases;
    const int n = (int) rows.size();
    int total = 0;
    for (int r : rows) total += r;
    std::vector<float> xf((size_t) total * K), wf((size_t) n * N * K);
    std::vector<uint16_t> xb(xf.size()), wb(wf.size());
    for (size_t i = 0; i < xf.size(); ++i) {
        xb[i] = kind == 1 ? bf16_bits(value((uint32_t) i * 3u + 17u)) : half_bits(value((uint32_t) i * 3u + 17u));
        xf[i] = kind == 1 ? bf16_value(xb[i]) : half_value(xb[i]);
    }
    for (size_t i = 0; i < wf.size(); ++i) {
        wb[i] = kind == 1 ? bf16_bits(value((uint32_t) i * 5u + 29u)) : half_bits(value((uint32_t) i * 5u + 29u));
        wf[i] = kind == 1 ? bf16_value(wb[i]) : half_value(wb[i]);
    }
    cudaMemcpyAsync(g_dx, xb.data(), xb.size() * 2, cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(g_dw, wb.data(), wb.size() * 2, cudaMemcpyHostToDevice, st);
    std::vector<const uint16_t*> px(n), pw(n);
    std::vector<float*> py(n);
    int at = 0;
    for (int i = 0; i < n; ++i) {
        px[i] = (const uint16_t*) g_dx + (size_t) at * K;
        pw[i] = (const uint16_t*) g_dw + (size_t) i * N * K;
        py[i] = (float*) g_dy + (size_t) at * N;
        at += rows[i];
    }
    const uint16_t** dpx = dalloc<const uint16_t*>(n);
    const uint16_t** dpw = dalloc<const uint16_t*>(n);
    float** dpy = dalloc<float*>(n);
    cudaMemcpyAsync(dpx, px.data(), n * sizeof(void*), cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(dpw, pw.data(), n * sizeof(void*), cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(dpy, py.data(), n * sizeof(void*), cudaMemcpyHostToDevice, st);
    std::vector<float> got[2];
    for (int rep = 0; rep < 2; ++rep) {
        cudaMemsetAsync(g_dy, 0xff, (size_t) total * N * 4, st);
        lamina::kernels::gemm16_grouped(dpx, dpw, dpy, rows.data(), n, N, K, kind == 1, st);
        if (!ck(cudaStreamSynchronize(st), "a grouped product")) {
            ++g_fail;
            return;
        }
        got[rep].resize((size_t) total * N);
        cudaMemcpyAsync(got[rep].data(), g_dy, got[rep].size() * 4, cudaMemcpyDeviceToHost, st);
        cudaStreamSynchronize(st);
    }
    const bool same = std::memcmp(got[0].data(), got[1].data(), got[0].size() * 4) == 0;
    size_t bad = 0;
    double worst = 0;
    at = 0;
    for (int i = 0; i < n; ++i) {
        for (int t = 0; t < rows[i]; ++t)
            for (int c = 0; c < N; ++c) {
                double s = 0, mag = 0;
                for (int k = 0; k < K; ++k) {
                    const double pr =
                        (double) xf[(size_t) (at + t) * K + k] * (double) wf[((size_t) i * N + c) * K + k];
                    s += pr;
                    mag += std::fabs(pr);
                }
                const double err = std::fabs((double) got[0][(size_t) (at + t) * N + c] - s);
                worst = std::max(worst, err / (mag + 1e-30));
                bad += !(err <= 3e-6 * mag + 1e-30);
            }
        at += rows[i];
    }
    const bool ok = same && bad == 0;
    if (!ok) ++g_fail;
    if (!ok || std::getenv("LAMINA_GEMM_PARITY_VERBOSE"))
        std::printf(
            "%s grouped %s, %d problems of %d rows in all, N=%d K=%d: worst |error| / sum of magnitudes %.2e, %zu "
            "outputs off, two runs %s\n",
            ok ? "ok  " : "FAIL", kind == 1 ? "bf16" : "f16 ", n, total, N, K, worst, bad,
            same ? "the same bits" : "DIFFER");
    cudaFree(dpx);
    cudaFree(dpw);
    cudaFree(dpy);
}

}   // namespace

int main(int argc, char** argv) {
    int dev = 0, major = 0;
    if (argc > 1 && cudaSetDevice(std::atoi(argv[1])) != cudaSuccess) {
        std::printf("gemm_parity: no device %s\n", argv[1]);
        return 77;
    }
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) {
        std::printf("gemm_parity: no CUDA device, skipped\n");
        return 77;
    }
    if (major < 8) {
        std::printf("gemm_parity: the products need sm_80+: skipped\n");
        return 77;
    }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);
    std::printf("gemm_parity on %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    cudaStream_t st = nullptr;
    // (the test's copies go through this stream too: a cudaMemcpy from pageable memory may return before its transfer
    // lands, and a kernel of a non-blocking stream does not wait for it)
    if (!ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream")) return 1;
    g_dx = dalloc<uint8_t>(kOperand + 64);
    g_dw = dalloc<uint8_t>(kOperand + 64);
    g_dy = dalloc<uint8_t>(kOperand);
    g_scratch = dalloc<uint8_t>(kScratch);
    for (int kind = 0; kind < 3; ++kind) {
        // the paths' shapes at a few row counts: tiles, splits, the row-dot kernel
        for (const int T : {1, 5, 17, 33, 100, 129, 300})
            for (const auto& nk : {std::pair<int, int>{320, 2560},
                                   {4, 2560},
                                   {1, 640},
                                   {48, 640},
                                   {513, 320},
                                   {2560, 320},
                                   {100, 648},
                                   {7, 328},
                                   {9, 1024},
                                   {640, 96}})
                run({kind, T, nk.first, nk.second}, st);
        run({kind, 700, 1280, 640}, st);
        run({kind, 2048, 48, 256}, st);
        // K: one slab and less, an end inside a slab and inside a granule; without a scratch (K whole)
        for (const int K : {8, 16, 24, 40, 72, 200})
            for (const int T : {3, 70})
                for (const int N : {5, 130}) run({kind, T, N, K}, st);
        for (const int T : {17, 300}) {
            Case c{kind, T, 320, 2560};
            c.scratch = false;
            run(c, st);
        }
        // rows that are not 16-byte aligned: an odd K, an odd row distance, operands at an odd address
        for (const int T : {2, 65})
            for (const int N : {3, 70}) {
                run({kind, T, N, 37}, st);
                run({kind, T, N, 1025}, st);
                Case c{kind, T, N, 256};
                c.ldx_extra = 3;
                run(c, st);
                c.ldx_extra = 0;
                c.x_shift = 1;
                run(c, st);
            }
        // a row distance past K and N, alpha and beta, Y at an odd row distance (single stores)
        for (const int T : {4, 150})
            for (const int N : {6, 96, 321}) {
                Case c{kind, T, N, 512};
                c.ldx_extra = 16;
                c.ldy_extra = 32;
                c.alpha = 0.5f;
                c.beta = 1.0f;
                run(c, st);
                c.ldy_extra = 1;
                c.alpha = -2.0f;
                c.beta = 0.25f;
                run(c, st);
            }
    }
    // halves out (a LoRA pair's first product), with beta too
    for (const int T : {9, 200})
        for (const int N : {8, 64, 100}) {
            Case c{0, T, N, 2048};
            c.y_half = true;
            run(c, st);
            c.beta = 1.0f;
            c.alpha = 0.25f;
            c.ldy_extra = 4;
            run(c, st);
        }
    // every instance: tiles past the rows and the outputs, K ending inside a slab, whole and in three parts; the
    // floats' also with Y read (beta) and where its fours are not aligned
    for (int form = 0, n = lamina::kernels::gemm32_forms(-1, nullptr, 0); form < n; ++form)
        for (const int splits : {1, 3}) {
            Case c{2, 70, 132, 200};
            c.form = form;
            c.splits = splits;
            run(c, st);
            c.x_inner = true;
            c.alpha = 0.5f;
            c.beta = 1.0f;
            run(c, st);
            c = Case{2, 300, 513, 328};
            c.form = form;
            c.splits = splits;
            c.beta = 0.25f;
            run(c, st);
            c = Case{2, 256, 512, 64};
            c.form = form;
            c.ldy_extra = splits == 1 ? 0 : 2;
            c.alpha = -2.0f;
            c.beta = 1.0f;
            run(c, st);
        }
    for (int form = 0, n = lamina::kernels::gemm16_forms(-1, nullptr, 0); form < n; ++form)
        for (int kind = 0; kind < 2; ++kind)
            for (const int splits : {1, 3}) {
                Case c{kind, 70, 130, 200};
                c.form = form;
                c.splits = splits;
                run(c, st);
                c.x_inner = true;
                run(c, st);
                c = Case{kind, 300, 513, 328};
                c.form = form;
                c.splits = splits;
                c.x_inner = kind == 1;
                run(c, st);
                // Y read (beta), its pairs aligned and not
                c = Case{kind, 150, 322, 64};
                c.form = form;
                c.ldy_extra = splits == 1 ? 0 : 1;
                c.alpha = 0.5f;
                c.beta = 1.0f;
                run(c, st);
            }
    for (int kind = 0; kind < 2; ++kind) {
        run_grouped(kind, {1, 37, 64, 0, 200, 65}, 96, 64, st);
        run_grouped(kind, {300, 2, 129}, 320, 136, st);
        std::vector<int> many(40);
        for (int i = 0; i < 40; ++i) many[(size_t) i] = 1 + (i * 7) % 23;
        run_grouped(kind, many, 130, 40, st);
    }
    std::printf("gemm_parity: %d cases, %d failures\n", g_cases, g_fail);
    return g_fail ? 1 : 0;
}
