// src/kernels/cuda/cvec.cu - see include/lamina/kernels/cvec.hpp.
#include "lamina/kernels/cvec.hpp"
#include "lamina/kernels/bf16_bits.hpp"
#include "lamina/kernels/warp.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace lamina::kernels {
namespace {

constexpr int N = 2560, THREADS = 256, PER = N / THREADS;

__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }

// the prompt path's block_sum (prefill/kernels.cu): its norm's rs is bitwise gr_write_norm_rs's on the same rows
__device__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    float t = (threadIdx.x < nw) ? sh[threadIdx.x] : 0.0f;
    if (w == 0) t = warp_sum(t);
    if (threadIdx.x == 0) sh[0] = t;
    __syncthreads();
    return sh[0];
}

// one block per (token, stream) row, its values in registers between the write, the dot, the update and the norm
__global__ void __launch_bounds__(THREADS) cvec_kernel(CvecRows r, const float* __restrict__ cv) {
    __shared__ float sh[32];
    const int64_t row = blockIdx.x;   // t * hc + c
    const int64_t t = row / r.hc;
    const int c = (int) (row % r.hc);
    float* h = r.R + row * N;
    const float* a = cv;
    const float* u = cv + N;
    float x[PER];
    float dot = 0.0f;
    const float sc = r.bo != nullptr ? 2.0f * sigm(r.inj[t * r.inj_ld + c] / (float) r.hc) : 0.0f;
#pragma unroll
    for (int k = 0; k < PER; ++k) {
        const int d = threadIdx.x + THREADS * k;
        float v = h[d];
        if (r.bo != nullptr) {
            const float b = r.partial ? r.bo[t * N + d] + __half2float(__ushort_as_half(r.partial[t * N + d]))
                                      : r.bo[t * N + d];
            v = fmaf(b, sc, v);
        }
        x[k] = v;
        dot += v * u[d];
    }
    dot = block_sum(dot, sh);
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < PER; ++k) {
        const int d = threadIdx.x + THREADS * k;
        const float v = fmaf(-dot, u[d], x[k] + a[d]);
        h[d] = v;
        x[k] = v;
        ss += v * v;
    }
    if (r.w_norm == nullptr) return;
    const float rs = rsqrtf(block_sum(ss, sh) / (float) N + r.eps);
    if (threadIdx.x == 0) r.rs[row] = rs;
#pragma unroll
    for (int k = 0; k < PER; ++k) {
        const int d = threadIdx.x + THREADS * k;
        r.xn16[row * N + d] = bf16_from_f32(x[k] * rs * r.w_norm[c * N + d]);
    }
}

}  // namespace

bool cvec_supported(int64_t n_embd) { return n_embd == N; }

void cvec_apply(const CvecRows& rows, const float* cv, void* stream) {
    if (rows.T < 1) return;
    if (!cvec_supported(rows.n_embd) || rows.hc < 1 || rows.R == nullptr || cv == nullptr ||
        (rows.bo != nullptr && rows.inj == nullptr) ||
        (rows.w_norm != nullptr && (rows.rs == nullptr || rows.xn16 == nullptr)))
        throw std::runtime_error("cvec_apply: unsupported geometry or missing buffers");
    cvec_kernel<<<(unsigned) (rows.T * rows.hc), THREADS, 0, (cudaStream_t) stream>>>(rows, cv);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("cvec_apply: ") + cudaGetErrorString(e));
}

}  // namespace lamina::kernels
