// src/kernels/cvec_parity.cpp - kernels/cvec.hpp's control vector kernel on synthetic rows:
//
//   1. With zero vectors it is the write it stands in for, bitwise: the prompt path's gr_write_norm_rs (the second
//      GPU's FP16 sums added in) and gr_write, and the verify window's read with the previous half's write folded in
//      (fused_gr_read_multi, apply), whose R' it writes.  So a layer it steers differs only by the vector.
//   2. Its norm is gr_norm_rs's of the rows it wrote, bitwise (the fused pass against the two kernels).
//   3. add, project and both, with and without the write, against FP64: the rows and the norm's row scales.
//
//     cvec_parity --selftest
#include "lamina/kernels/bf16_bits.hpp"
#include "lamina/kernels/cvec.hpp"
#include "lamina/kernels/f16_bits.hpp"
#include "lamina/kernels/fused_gr.hpp"
#include "lamina/prefill/kernels.hpp"
#include "parity.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int N = 2560, HC = 4, LR = 320, D = N * HC;
constexpr float EPS = 1e-6f;
int fails = 0;

void expect(bool ok, const std::string& what) {
    std::printf("  %-74s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++fails;
}
template <typename T> T* up(const std::vector<T>& h) {
    T* d = dev<T>(h.size());
    check(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}
template <typename T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    check(cudaDeviceSynchronize(), "kernel");
    check(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return h;
}
template <typename T> bool same(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

struct Case {
    int T = 0;
    std::vector<float> R, bo, inj, w_norm;
    std::vector<uint16_t> partial;
};
Case make(int T, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    Case c;
    c.T = T;
    c.R.resize((size_t) T * D);
    c.bo.resize((size_t) T * N);
    c.partial.resize((size_t) T * N);
    c.inj.resize((size_t) T * HC);
    c.w_norm.resize((size_t) D);
    for (float& x : c.R) x = 3.0f * nd(rng);
    for (float& x : c.bo) x = 0.5f * nd(rng);
    for (uint16_t& x : c.partial) x = lamina::kernels::f16_from_f32(0.5f * nd(rng));
    for (float& x : c.inj) x = 2.0f * nd(rng);
    for (float& x : c.w_norm) x = 1.0f + 0.1f * nd(rng);
    return c;
}

// the rows after the write and the vector, and their norm's row scales, in FP64 from the same float inputs
void reference(const Case& c, bool write, bool partial, const std::vector<float>& cv, std::vector<double>& rows,
               std::vector<double>& rs) {
    rows.assign((size_t) c.T * D, 0.0);
    rs.assign((size_t) c.T * HC, 0.0);
    for (int t = 0; t < c.T; ++t)
        for (int s = 0; s < HC; ++s) {
            double* x = rows.data() + ((size_t) t * HC + s) * N;
            const double w = 2.0 / (1.0 + std::exp(-(double) c.inj[(size_t) (t * HC + s)] / HC));
            double dot = 0.0;
            for (int d = 0; d < N; ++d) {
                double v = c.R[((size_t) t * HC + s) * N + d];
                if (write) {
                    double b = c.bo[(size_t) t * N + d];
                    if (partial) b += lamina::kernels::f32_from_f16(c.partial[(size_t) t * N + d]);
                    v += b * w;
                }
                x[d] = v;
                dot += v * cv[(size_t) (N + d)];
            }
            double ss = 0.0;
            for (int d = 0; d < N; ++d) {
                x[d] += cv[(size_t) d] - dot * cv[(size_t) (N + d)];
                ss += x[d] * x[d];
            }
            rs[(size_t) (t * HC + s)] = 1.0 / std::sqrt(ss / N + EPS);
        }
}

// the largest error of a row against the reference, over the row's RMS
double row_error(const std::vector<float>& got, const std::vector<double>& want) {
    double worst = 0.0;
    for (size_t r = 0; r < got.size() / N; ++r) {
        double ss = 0.0, e = 0.0;
        for (int d = 0; d < N; ++d) {
            ss += want[r * N + d] * want[r * N + d];
            e = std::max(e, std::fabs((double) got[r * N + d] - want[r * N + d]));
        }
        worst = std::max(worst, e / std::sqrt(ss / N));
    }
    return worst;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || std::strcmp(argv[1], "--selftest") != 0) {
        std::fprintf(stderr, "usage: cvec_parity --selftest\n");
        return 2;
    }
    using lamina::kernels::CvecRows;
    std::mt19937 rng(20261113);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    // the vectors: zero; add; project (u = sqrt(s) v, s 1.78, v unit); both
    // (and the projection at s 1: u = v)
    std::vector<float> zero(2 * N, 0.0f), add(2 * N, 0.0f), proj(2 * N, 0.0f), both(2 * N, 0.0f), unit(2 * N, 0.0f);
    {
        double nv = 0.0;
        std::vector<double> v(N);
        for (double& x : v) { x = nd(rng); nv += x * x; }
        for (int d = 0; d < N; ++d) {
            const float a = 0.25f * nd(rng), u = (float) (std::sqrt(1.78) * v[(size_t) d] / std::sqrt(nv));
            add[(size_t) d] = a;
            proj[(size_t) (N + d)] = u;
            both[(size_t) d] = a;
            both[(size_t) (N + d)] = u;
            unit[(size_t) (N + d)] = (float) (v[(size_t) d] / std::sqrt(nv));
        }
    }
    const float* d_zero = up(zero);
    float* rs1 = dev<float>(8 * HC);
    float* rs2 = dev<float>(8 * HC);
    uint16_t* xn1 = dev<uint16_t>((size_t) 8 * D);
    uint16_t* xn2 = dev<uint16_t>((size_t) 8 * D);

    for (const int T : {1, 5, 8}) {
        std::printf("T = %d\n", T);
        const Case c = make(T, rng);
        float* bo = up(c.bo);
        float* inj = up(c.inj);
        float* wn = up(c.w_norm);
        uint16_t* part = up(c.partial);
        float* R1 = up(c.R);
        float* R2 = up(c.R);
        auto reset = [&] {
            check(cudaMemcpy(R1, c.R.data(), c.R.size() * 4, cudaMemcpyHostToDevice), "reset");
            check(cudaMemcpy(R2, c.R.data(), c.R.size() * 4, cudaMemcpyHostToDevice), "reset");
        };
        auto rows = [&](float* R, bool write, bool partial, bool norm, float* rs, uint16_t* xn) {
            CvecRows r;
            r.R = R; r.T = T; r.n_embd = N; r.hc = HC;
            if (write) { r.bo = bo; r.inj = inj; r.inj_ld = HC; r.partial = partial ? part : nullptr; }
            if (norm) { r.w_norm = wn; r.eps = EPS; r.rs = rs; r.xn16 = xn; }
            return r;
        };

        // ---- 1. zero vectors: the writes it stands in for
        lamina::prefill::gr_write_norm_rs(R1, bo, inj, HC, wn, EPS, rs1, xn1, T, nullptr, part);
        lamina::kernels::cvec_apply(rows(R2, true, true, true, rs2, xn2), d_zero, nullptr);
        expect(same(down(R1, c.R.size()), down(R2, c.R.size())) && same(down(rs1, (size_t) T * HC), down(rs2, (size_t) T * HC)) &&
                   same(down(xn1, (size_t) T * D), down(xn2, (size_t) T * D)),
               "zero vector, write + FP16 sums + norm = gr_write_norm_rs (R, rs, xn16)");
        reset();
        lamina::prefill::gr_write(R1, bo, inj, HC, T, nullptr, part);
        lamina::kernels::cvec_apply(rows(R2, true, true, false, nullptr, nullptr), d_zero, nullptr);
        expect(same(down(R1, c.R.size()), down(R2, c.R.size())), "zero vector, write + FP16 sums = gr_write");
        reset();
        {   // the verify window's read with the write folded in writes R' in place
            std::vector<uint16_t> w_down((size_t) LR * D), w_up((size_t) D * LR), w_inj((size_t) HC * D);
            for (uint16_t& x : w_down) x = lamina::kernels::bf16_from_f32(0.02f * nd(rng));
            for (uint16_t& x : w_up) x = lamina::kernels::bf16_from_f32(0.05f * nd(rng));
            for (uint16_t& x : w_inj) x = lamina::kernels::bf16_from_f32(0.02f * nd(rng));
            uint16_t* dd = up(w_down);
            uint16_t* du = up(w_up);
            uint16_t* di = up(w_inj);
            float* lo = dev<float>((size_t) T * LR);
            float* rsw = dev<float>((size_t) T * HC);
            float* io = dev<float>((size_t) T * HC);
            float* mixed = dev<float>((size_t) T * N);
            float* scratch = (float*) dev<uint8_t>(lamina::kernels::fused_gr_scratch_bytes());
            check(cudaMemset(scratch, 0, lamina::kernels::fused_gr_scratch_bytes()), "scratch");
            lamina::kernels::FusedGrArgs fa[lamina::kernels::kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                lamina::kernels::FusedGrArgs& a = fa[t];
                a.R = R1 + (size_t) t * D; a.R_out = R1 + (size_t) t * D; a.apply = true;
                a.bo_prev = bo + (size_t) t * N; a.inj_prev = inj + t * HC;
                a.w_norm = wn; a.w_down = dd; a.w_up = du; a.w_inject = di; a.eps = EPS;
                a.lo = lo + (size_t) t * LR; a.rs = rsw + t * HC; a.inject_out = io + t * HC; a.mixed = mixed + (size_t) t * N;
            }
            lamina::kernels::fused_gr_read_multi(fa, T, scratch, nullptr);
            lamina::kernels::cvec_apply(rows(R2, true, false, false, nullptr, nullptr), d_zero, nullptr);
            expect(same(down(R1, c.R.size()), down(R2, c.R.size())),
                   "zero vector, write = the verify read's folded write (fused_gr_read_multi)");
            for (void* p : {(void*) dd, (void*) du, (void*) di, (void*) lo, (void*) rsw, (void*) io, (void*) mixed, (void*) scratch})
                cudaFree(p);
            reset();
        }

        // ---- 2. the norm of the rows it wrote, as gr_norm_rs computes it
        const float* d_both = up(both);
        lamina::kernels::cvec_apply(rows(R1, true, true, true, rs1, xn1), d_both, nullptr);
        lamina::kernels::cvec_apply(rows(R2, true, true, false, nullptr, nullptr), d_both, nullptr);
        lamina::prefill::gr_norm_rs(R2, wn, EPS, rs2, xn2, T, nullptr);
        expect(same(down(R1, c.R.size()), down(R2, c.R.size())) && same(down(rs1, (size_t) T * HC), down(rs2, (size_t) T * HC)) &&
                   same(down(xn1, (size_t) T * D), down(xn2, (size_t) T * D)),
               "write + vector + norm in one pass = the pass, then gr_norm_rs");
        reset();

        // ---- 3. against FP64
        const struct { const char* name; const std::vector<float>* cv; } modes[] = {{"add", &add}, {"project", &proj}, {"both", &both}};
        for (const auto& m : modes) {
            const float* d_cv = up(*m.cv);
            for (const bool write : {false, true}) {
                std::vector<double> want, want_rs;
                reference(c, write, write, *m.cv, want, want_rs);
                lamina::kernels::cvec_apply(rows(R1, write, write, true, rs1, xn1), d_cv, nullptr);
                const std::vector<float> got = down(R1, c.R.size()), got_rs = down(rs1, (size_t) T * HC);
                double e_rs = 0.0;
                for (size_t i = 0; i < got_rs.size(); ++i) e_rs = std::max(e_rs, std::fabs(got_rs[i] / want_rs[i] - 1.0));
                const double e = row_error(got, want);
                char what[160];
                std::snprintf(what, sizeof what, "%s%s vs FP64: rows %.2e of their RMS, row scales %.2e", m.name,
                              write ? " after the write" : "", e, e_rs);
                expect(e < 2e-6 && e_rs < 1e-6, what);
                reset();
            }
            cudaFree((void*) d_cv);
        }
        // the projection at s 1 leaves no component along v: (h . v) after it is 0 up to rounding
        {
            const float* d_cv = up(unit);
            lamina::kernels::cvec_apply(rows(R1, true, false, false, nullptr, nullptr), d_cv, nullptr);
            const std::vector<float> got = down(R1, c.R.size());
            double worst = 0.0;
            for (size_t r = 0; r < got.size() / N; ++r) {
                double dot = 0.0, ss = 0.0;
                for (int d = 0; d < N; ++d) {
                    dot += (double) got[r * N + d] * unit[(size_t) (N + d)];
                    ss += (double) got[r * N + d] * got[r * N + d];
                }
                worst = std::max(worst, std::fabs(dot) / std::sqrt(ss));
            }
            char what[160];
            std::snprintf(what, sizeof what, "project at 1: what is left along v, %.2e of the row's norm", worst);
            expect(worst < 1e-6, what);
            cudaFree((void*) d_cv);
            reset();
        }
        cudaFree((void*) d_both);
        for (void* p : {(void*) bo, (void*) inj, (void*) wn, (void*) part, (void*) R1, (void*) R2}) cudaFree(p);
    }
    std::printf("cvec_parity: %s\n", fails == 0 ? "OK" : "FAILED");
    return fails == 0 ? 0 : 1;
}
