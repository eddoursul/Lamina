// src/qwen35/q36_engine.cu - lamina-q36: the engine path for Qwen3.6-35B-A3B (GGUF architecture qwen35moe) and
// Qwen3.8-27B (qwen35: the same layers with a dense FFN in place of the experts),
// NEW-MODELS-PLAN.md T1.  A driver of its own beside Flash-Next's (generate/verify/prefill stay untouched, so the
// Flash-Next identity gate cannot move), built from Lamina's kernels:
//
//   * weights straight from the GGUF: dense Q8_0 / K-quant matrices in their blocks (native MMVQ), the 1-D tensors and
//     the F32 projections (ssm_alpha/beta, the router, the shared-expert gate) as f32, the routed experts repacked at
//     load into native expert blobs [gate | up | down] per (layer, expert) for `native_expert_grouped`;
//   * the plain pre-norm residual layer driver: x += mixer(rms(x) * attn_norm); x += moe(rms(x) * post_attention_norm);
//   * GDN: fused conv + L2 and the fused step with the SiLU-gated norm (per token, the state carried);
//   * attention without an indexer: dense, 16 query heads over 2 KV heads, causal; a prompt chunk as FP16 GEMMs, a
//     decode window as flash-decoding over 256-cell chunks (a KV head's query heads and the window's rows read each
//     K / V tile once) with a log-sum-exp merge;
//   * MoE: the native router (256 / top-8), the experts grouped per expert on the device (one weight read per expert
//     for the whole batch), the shared expert with its sigmoid gate, the native combine;
//   * the prompt read in chunks of --chunk tokens through the same driver (batched matmuls, ncols <= 8 per launch);
//   * decode one token at a time from a captured graph (positions on the device), or with --no-graph eagerly.
//
// Modes: --serve (lamina-server's engine protocol: READY / GEN / T / DONE / STOP / PP), --tokens IDS or --tokens-file F
// [--n N] [--force IDS] [--dump-full-logits F] (tools/qwen35/kl.py's reference interface), --bench-pp N / --bench-tg N.
#include "lamina/artifact/gguf_reader.hpp"
#include "lamina/core/cuda_host.hpp"
#include "lamina/core/family_layout.hpp"
#include "lamina/core/gpu_select.hpp"
#include "lamina/core/prompt_cache.hpp"
#include "lamina/kernels/fused_gdn.hpp"
#include "lamina/kernels/iq_kernels.hpp"
#include "lamina/kernels/native_mmvq.hpp"
#include "lamina/kernels/native_moe.hpp"
#include "lamina/kernels/native_router.hpp"
#include "lamina/kernels/warp.cuh"
#include "lamina/prefill/gemm.hpp"
#include "lamina/prefill/moe_mmq.hpp"
#include "lamina/qwen35/engine.hpp"
#include "lamina/platform/memory.hpp"
#include "lamina/program/host_sampling.hpp"
#include "lamina/program/serve_io.hpp"
#include "q36_attn.cuh"
#include "q36_kernels.cuh"
#include "q36_lora.cuh"
#if defined(LAMINA_Q36_CPU_POOL)
#include "lamina/kernels/cpu/native_expert.hpp"
#include "lamina/kernels/cpu/pool.hpp"
#endif

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>

using lamina::program::serve_print;   // the --serve protocol's replies (serve_io.hpp)

namespace {

namespace K = lamina::kernels;
namespace MMQ = lamina::prefill::mmq;

#define CK(x)                                                                                                          \
    do {                                                                                                               \
        cudaError_t e_ = (x);                                                                                          \
        if (e_ != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "lamina: %s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__); \
            throw Fatal{1};                                                                                            \
        }                                                                                                              \
    } while (0)

/// The run's end on an error (CK's, a missing tensor's ...): its message is printed, run() returns `code`.  The engine may
/// run in the server's process (lamina-server), which an exit would end.
struct Fatal {
    int code;
};

/// What a run allocates, freed when it ends (the server starts the engine again in its process after an unload): device,
/// pinned and heap memory, streams, events, graphs and cuBLAS handles, each recorded by the helpers below.  One run at a
/// time per process.
struct Owned {
    std::mutex mu;
    std::vector<void*> dev, pinned, heap;
    std::vector<cudaStream_t> streams;
    std::vector<cudaEvent_t> events;
    std::vector<cudaGraphExec_t> graphs;
    std::vector<cublasHandle_t> blas;
    ~Owned() {
        cudaDeviceSynchronize();
        for (cudaGraphExec_t g : graphs) cudaGraphExecDestroy(g);
        for (cublasHandle_t h : blas) cublasDestroy(h);
        for (cudaEvent_t e : events) cudaEventDestroy(e);
        for (cudaStream_t st : streams) cudaStreamDestroy(st);
        for (void* d : dev) cudaFree(d);
        for (void* h : pinned) cudaFreeHost(h);
        for (void* h : heap) std::free(h);
        (void) cudaGetLastError();
    }
};
Owned* g_owned = nullptr;
template <class V, class T> void own(std::vector<V> Owned::*list, T v) {
    if (g_owned == nullptr) return;
    std::lock_guard<std::mutex> lk(g_owned->mu);
    (g_owned->*list).push_back((V) v);
}
template <class T> cudaError_t dmalloc(T** p, size_t n) {
    const cudaError_t e = cudaMalloc((void**) p, n);
    if (e == cudaSuccess) own(&Owned::dev, (void*) *p);
    return e;
}
template <class T> cudaError_t hmalloc(T** p, size_t n, unsigned flags) {
    const cudaError_t e = cudaHostAlloc((void**) p, n, flags);
    if (e == cudaSuccess) own(&Owned::pinned, (void*) *p);
    return e;
}
cudaError_t dfree(void* p) {
    if (g_owned != nullptr) {
        std::lock_guard<std::mutex> lk(g_owned->mu);
        auto& d = g_owned->dev;
        d.erase(std::remove(d.begin(), d.end(), p), d.end());
    }
    return cudaFree(p);
}
cudaError_t stream_new(cudaStream_t* st, unsigned flags) {
    const cudaError_t e = cudaStreamCreateWithFlags(st, flags);
    if (e == cudaSuccess) own(&Owned::streams, *st);
    return e;
}
cudaError_t event_new(cudaEvent_t* ev, unsigned flags) {
    const cudaError_t e = cudaEventCreateWithFlags(ev, flags);
    if (e == cudaSuccess) own(&Owned::events, *ev);
    return e;
}
cudaError_t graph_new(cudaGraphExec_t* gx, cudaGraph_t graph) {
    const cudaError_t e = cudaGraphInstantiate(gx, graph, 0);
    if (e == cudaSuccess) own(&Owned::graphs, *gx);
    return e;
}
cublasStatus_t blas_new(cublasHandle_t* h) {
    const cublasStatus_t e = cublasCreate(h);
    if (e == CUBLAS_STATUS_SUCCESS) own(&Owned::blas, *h);
    return e;
}
void* heap_new(size_t n) {
    void* h = std::malloc(n);
    if (h != nullptr) own(&Owned::heap, h);
    return h;
}

using Clock = std::chrono::steady_clock;
using lamina::core::ms_since;
using lamina::kernels::block128_sum;
using lamina::kernels::warp_max;
using lamina::kernels::warp_sum;

constexpr int kThreads = 256;
constexpr int kChunkCells = 256;   // attention split-K chunk
constexpr int kConvTile = 16;     // tokens per block of the prompt conv (k_conv_seq tile mode)

// The VRAM a card keeps free once its experts are placed, and a loan.  The server's image encoder, which runs in this
// process, may hold no VRAM of its own between pictures and borrow for each new one (the serve protocol's LEND); it
// says so before the engine loads (lamina::core::gpu_loan_ask).  This engine has no cache to lend from.  In the
// hybrid mode the loan is the prompt path's two staging buffers, idle between prompts (place_experts).  Else it is of
// the VRAM left free, and where the placement has a choice (a layer split's layers, the prompt chunk's size) the
// card keeps the loan and kLoanFree (what the encoder takes beside it: cuBLAS's handle, its kernels' images) free
// past the reserve: the reserve itself is not the loan's, as the engine's later allocations come out of it (the
// 27B's split: 100 MiB in its first request).
constexpr size_t kVramReserve = (size_t) 768 << 20;
constexpr size_t kLoanFree = (size_t) 64 << 20;
/// What `dev` keeps free for the loan asked of it (0: none asked).
size_t loan_keep(int dev) {
    const size_t asked = (size_t) lamina::core::gpu_loan_asked(dev);
    return asked == 0 ? 0 : asked + kLoanFree;
}

// ------------------------------------------------------------------------------------------------ small kernels
__device__ __forceinline__ float silu(float v) { return v / (1.0f + expf(-v)); }
__device__ __forceinline__ float sigm(float v) { return 1.0f / (1.0f + expf(-v)); }
__device__ __forceinline__ float softplus(float v) { return v > 20.f ? v : log1pf(expf(v)); }

__device__ float block_sum(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}

/// y[r] = rms(x[r]) * w, one block per row of n.
__global__ void k_rms(const float* x, const float* w, float* y, int n, float eps) {
    __shared__ float sh[kThreads];
    const float* xr = x + (size_t) blockIdx.x * n;
    float* yr = y + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * r * w[i];
}
/// x[r] += y[r]; h[r] = rms(x[r]) * w (one block per row of n): the residual add and the next norm in one launch.
/// out[r * ld_out + i] = rms(x[r * ldx + ..]) * w (one block per row of n): a norm into a slice of a wider row.
__global__ void k_rms_ld(const float* x, int ldx, const float* w, float* out, int ld_out, int n, float eps) {
    __shared__ float sh[kThreads];
    const float* xr = x + (size_t) blockIdx.x * ldx;
    float* yr = out + (size_t) blockIdx.x * ld_out;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * r * w[i];
}
__global__ void k_add_rms(float* x, const float* y, const float* w, float* h, int n, float eps) {
    __shared__ float sh[kThreads];
    float* xr = x + (size_t) blockIdx.x * n;
    const float* yr = y + (size_t) blockIdx.x * n;
    float* hr = h + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i] + yr[i];
        xr[i] = v;
        s += v * v;
    }
    if (w == nullptr) return;   // the add alone
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) hr[i] = xr[i] * r * w[i];
}
/// The MoE block's end for one token row (block per row): x += sum_k w_k part_k (in pick order, FMA after the first
/// product, as native_moe_combine) + sigmoid(sgate) * shared; then, with `wn`, h = rms(x) * wn (the next layer's
/// attention norm).  Replaces combine + gate + add + rms.
__global__ void k_moe_out(float* x, const float* parts, const int32_t* prow, const float* wts, int K_,
                          const float* shared, const float* sgate, int sg_stride, const float* wn, float* h, int n,
                          float eps) {
    __shared__ float sh[kThreads];
    __shared__ const float* pk[16];
    const int t = blockIdx.x;
    float* xr = x + (size_t) t * n;
    if (threadIdx.x < K_)   // pick k's row: t * K + k, or its sorted row (`prow`, the prompt's MMQ order)
        pk[threadIdx.x] = parts + (size_t) (prow ? prow[t * K_ + threadIdx.x] : t * K_ + threadIdx.x) * n;
    __syncthreads();
    const float* wr = wts + (size_t) t * K_;
    const float gs = sigm(sgate[(size_t) t * sg_stride]);
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float acc = wr[0] * pk[0][i];
        for (int k = 1; k < K_; ++k) acc = fmaf(wr[k], pk[k][i], acc);
        acc += gs * shared[(size_t) t * n + i];
        const float v = xr[i] + acc;
        xr[i] = v;
        s += v * v;
    }
    if (wn == nullptr) return;
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) h[(size_t) t * n + i] = xr[i] * r * wn[i];
}
/// GDN alpha and beta for T <= 8 rows in one launch, their epilogues applied: rows [0, HV) of the output are
/// softplus(alpha . x + dt) * ssm_a, rows [HV, 2 HV) sigmoid(beta . x).  Warp per row (grid (ceil(2 HV / 8), T)).
/// `dab` [T][2 HV]: the adapters' deltas of the products, added before the epilogues (null: none).
__global__ void k_gdn_ab(const float* Wa, const float* Wb, const float* x, const float* dt, const float* ssm_a,
                         float* alpha, float* beta, int n_in, int HV, const float* dab) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int o = blockIdx.x * 8 + warp;
    if (o >= 2 * HV) return;
    const bool isb = o >= HV;
    const int r = isb ? o - HV : o;
    const float4* wr = (const float4*) ((isb ? Wb : Wa) + (size_t) r * n_in);
    const float4* xr = (const float4*) (x + (size_t) blockIdx.y * n_in);
    float acc = 0.f;
    for (int i = lane; i < n_in / 4; i += 32) {
        const float4 a = wr[i], b = xr[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane != 0) return;
    if (dab != nullptr) acc += dab[(size_t) blockIdx.y * 2 * HV + o];
    if (isb) beta[(size_t) blockIdx.y * HV + r] = sigm(acc);
    else alpha[(size_t) blockIdx.y * HV + r] = softplus(acc + dt[r]) * ssm_a[r];
}
__global__ void k_swiglu(const float* g, const float* u, float* h, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = silu(g[i]) * u[i];
}
/// y[t][o] = W[o] . x[t] for f32 W [n_out][n_in], n_in a multiple of 128: one warp per row, float4 loads, 8 rows per
/// block (grid (ceil(n_out / 8), T)).
__global__ void k_gemv_f32(const float* W, const float* x, float* y, int n_in, int n_out) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int o = blockIdx.x * 8 + warp;
    if (o >= n_out) return;
    const float4* wr = (const float4*) (W + (size_t) o * n_in);
    const float4* xr = (const float4*) (x + (size_t) blockIdx.y * n_in);
    float acc = 0.f;
    for (int i = lane; i < n_in / 4; i += 32) {
        const float4 a = wr[i], b = xr[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane == 0) y[(size_t) blockIdx.y * n_out + o] = acc;
}
unsigned rows8(int n) { return (unsigned) ((n + 7) / 8); }
/// GDN gate / beta in place for T rows of HV: alpha -> softplus(alpha + dt) * ssm_a, beta -> sigmoid(beta).
__global__ void k_gdn_gb(float* alpha, const float* dt, const float* ssm_a, float* beta, int HV, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= HV * T) return;
    const int h = i % HV;
    alpha[i] = softplus(alpha[i] + dt[h]) * ssm_a[h];
    beta[i] = sigm(beta[i]);
}

/// Batch rows (several sessions decoded in one window, one token each): row t reads and writes the cache of session
/// slot[t] at cell[t]; its rotary position is cell[t] + delta[t].  Every field null: one session, the rows are its
/// consecutive cells *pos0 + t (the prompt, a verify window, a decode step).
struct RowSel {
    const int32_t* cell = nullptr;   // device [T]
    const int32_t* slot = nullptr;   // device [T]
    const int32_t* delta = nullptr;  // device [T]
    __half* const* kc = nullptr;     // device [n_sessions]: this layer's K / V base of each session (FP16 or q8_0 rows)
    __half* const* vc = nullptr;
};

/// K and V in q8_0 (--kv q8_0): a KV head's cell is a row of its 256 values in int8, then its eight 32-value blocks'
/// scales in FP16 (llama.cpp's q8_0 rounding: d = amax / 127, q = round(x / d)), 272 bytes against FP16's 512.
constexpr int kKvQ8Row = 256 + 8 * 2;
static_assert(kKvQ8Row == q36a::kQ8Row, "the prompt attention reads the same rows");
/// Value i of a q8_0 row, from the lanes of i's warp (each warp a 32-value block)
__device__ __forceinline__ void q8_put(uint8_t* row, int i, float v) {
    const float amax = warp_max(fabsf(v));
    const float d = amax / 127.f, id = d != 0.f ? 1.f / d : 0.f;
    row[i] = (uint8_t) (int8_t) roundf(v * id);
    if ((i & 31) == 0) reinterpret_cast<__half*>(row + 256)[i >> 5] = __float2half(d);
}
/// The attention layer's per-token preparation for T rows at positions *pos0 + t: q (from the [q | gate] pairs) and k
/// RMS-normed per head with their weights, NEOX RoPE over the first n_rot dims, k / v into the cache (FP16, or q8_0
/// rows with `q8`), the gate kept.
/// Rotary position of a cell: the cell itself (text) plus *rdelta (what the pictures before a text run did not use
/// of their cells, see mrope), or, with `mtab` (a request's prompt cells, device int32 [cell][3] = t, h, w), the
/// interleaved M-RoPE position (llama.cpp's IMROPE with sections 11 / 11 / 10: pair p takes stream p % 3).
/// grid (H + HKV, T), block 256 (HD <= 256; 256 with `q8`).
__global__ void k_attn_prep(const float* qg, const float* kx, const float* vx, const float* qw, const float* kw,
                            float* q, float* gate, __half* kc, __half* vc, int H, int HKV, int HD, int n_rot,
                            const int32_t* pos0, int ctx, float base, float eps, const int32_t* rdelta,
                            const int32_t* mtab, RowSel rs, bool q8) {
    __shared__ float sh[kThreads];
    __shared__ float buf[256];
    const int head = blockIdx.x, t = blockIdx.y;
    const int pos = rs.cell ? rs.cell[t] : *pos0 + t;
    const int dl = rs.delta ? rs.delta[t] : (rdelta ? *rdelta : 0);
    if (rs.kc) {
        kc = rs.kc[rs.slot[t]];
        vc = rs.vc[rs.slot[t]];
    }
    const bool isq = head < H;
    const float* src = isq ? qg + ((size_t) t * H + head) * 2 * HD : kx + ((size_t) t * HKV + head - H) * HD;
    const int i = threadIdx.x;
    const float v0 = i < HD ? src[i] : 0.f;
    const float s = block_sum(v0 * v0, sh);
    const float r = rsqrtf(s / (float) HD + eps);
    if (i < HD) buf[i] = v0 * r * (isq ? qw[i] : kw[i]);
    __syncthreads();
    if (i >= HD) return;
    float v = buf[i];
    const int half = n_rot / 2;
    if (i < n_rot) {
        const int p = i < half ? i : i - half;
        const int rp = mtab ? __ldg(mtab + (size_t) pos * 3 + p % 3) : pos + dl;
        const float theta = (float) rp * powf(base, -2.0f * (float) p / (float) n_rot);
        float sn, cs;
        sincosf(theta, &sn, &cs);
        v = i < half ? buf[i] * cs - buf[i + half] * sn : buf[i - half] * sn + buf[i] * cs;
    }
    if (isq) {
        q[((size_t) t * H + head) * HD + i] = v;
        gate[((size_t) t * H + head) * HD + i] = qg[((size_t) t * H + head) * 2 * HD + HD + i];
    } else {
        const int kh = head - H;
        const float vv = vx[((size_t) t * HKV + kh) * HD + i];
        if (q8) {
            q8_put(reinterpret_cast<uint8_t*>(kc) + ((size_t) kh * ctx + pos) * kKvQ8Row, i, v);
            q8_put(reinterpret_cast<uint8_t*>(vc) + ((size_t) kh * ctx + pos) * kKvQ8Row, i, vv);
        } else {
            kc[((size_t) kh * ctx + pos) * HD + i] = __float2half(v);
            vc[((size_t) kh * ctx + pos) * HD + i] = __float2half(vv);
        }
    }
}

/// Decode attention (flash-decoding) over 256-cell chunks, the query heads of a KV head together: one block per (chunk,
/// KV head, row group), its G = blockDim.x / 32 warps the KV head's query heads, each warp the group's NR rows.  The
/// chunk's K and V (FP16, or q8_0 rows with Q8) are read once, kAttnTile cells at a time, into shared memory as FP32: a warp's lanes score a tile's cells
/// (each lane its cell, the dims in an order rotated by the lane, so no two lanes read one bank), then hold 8 dims each
/// of the rows' accumulators.  Row t sees the cells [0, *pos0 + t] (rows_mode: rs.cell[t] of its session's K / V, one
/// row a block).  Writes the chunk's partial accumulator, max and sum per query row t * H + h (t from t0), which
/// k_attn_merge combines; a chunk past a row's cells writes nothing for it (the merge stops before it).
/// grid (n_chunk, HKV, rows_mode ? rows : 1), block 32 * G, dynamic shared attn_dec_smem(G, NR), HD == 256.
constexpr int kAttnTile = 32;
constexpr size_t attn_dec_smem(int G, int NR) { return (size_t) (G * NR + kAttnTile) * 256 * sizeof(float); }
template <int NR, bool Q8>
__global__ void __launch_bounds__(256) k_attn_dec(const float* q, const __half* kc, const __half* vc, const int32_t* pos0,
                                                  int t0, int H, int ctx, float scale, int n_chunk, float* pacc,
                                                  float* pm, float* pl, RowSel rs) {
    constexpr int HD = 256;
    extern __shared__ float smem[];
    const int G = (int) blockDim.x / 32, w = (int) threadIdx.x / 32, lane = (int) threadIdx.x % 32;
    const int c = blockIdx.x, kvh = blockIdx.y, h = kvh * G + w;
    const int r0 = t0 + (int) blockIdx.z;   // the block's first row (rows_mode: its only one)
    int pos[NR];
#pragma unroll
    for (int r = 0; r < NR; ++r) pos[r] = rs.cell ? rs.cell[r0 + r] : *pos0 + r0 + r;
    if (rs.kc) {
        kc = rs.kc[rs.slot[r0]];
        vc = rs.vc[rs.slot[r0]];
    }
    const int p0 = c * kChunkCells;
    const int end = min(p0 + kChunkCells, pos[NR - 1] + 1);   // a later row sees more cells
    if (p0 >= end) return;
    float* qs = smem + w * NR * HD;   // this warp's rows' queries
    float* tile = smem + G * NR * HD;
#pragma unroll
    for (int r = 0; r < NR; ++r)
        for (int d = lane; d < HD; d += 32) qs[r * HD + d] = q[((size_t) (r0 + r) * H + h) * HD + d];
    // FP16: 8 halves (16 bytes) a load, widened into two float4; q8_0: 16 values a load, times their block's scale
    const size_t row = Q8 ? kKvQ8Row : HD * sizeof(__half);
    const uint8_t* kb = reinterpret_cast<const uint8_t*>(kc) + (size_t) kvh * ctx * row;
    const uint8_t* vb = reinterpret_cast<const uint8_t*>(vc) + (size_t) kvh * ctx * row;
    float4* tile4 = reinterpret_cast<float4*>(tile);
    auto widen = [&](const uint8_t* src, int base, int n) {
        if constexpr (Q8) {
            for (int i = threadIdx.x; i < n * HD / 16; i += blockDim.x) {
                const uint8_t* r = src + (size_t) (base + i / (HD / 16)) * kKvQ8Row;
                const int part = i % (HD / 16);
                const uint4 u = *reinterpret_cast<const uint4*>(r + part * 16);
                const float d = __half2float(reinterpret_cast<const __half*>(r + HD)[part >> 1]);
                const int8_t* b = reinterpret_cast<const int8_t*>(&u);
#pragma unroll
                for (int k = 0; k < 4; ++k)
                    tile4[4 * i + k] = make_float4(b[4 * k] * d, b[4 * k + 1] * d, b[4 * k + 2] * d, b[4 * k + 3] * d);
            }
        } else {
            const uint4* s4 = reinterpret_cast<const uint4*>(src);
            for (int i = threadIdx.x; i < n * HD / 8; i += blockDim.x) {
                const uint4 u = s4[(size_t) base * (HD / 8) + i];
                const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
                const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
                const float2 c2 = __half22float2(*reinterpret_cast<const __half2*>(&u.z));
                const float2 d = __half22float2(*reinterpret_cast<const __half2*>(&u.w));
                tile4[2 * i] = make_float4(a.x, a.y, b.x, b.y);
                tile4[2 * i + 1] = make_float4(c2.x, c2.y, d.x, d.y);
            }
        }
    };
    float m[NR], l[NR], acc[NR][8];
#pragma unroll
    for (int r = 0; r < NR; ++r) {
        m[r] = -INFINITY;
        l[r] = 0.f;
#pragma unroll
        for (int k = 0; k < 8; ++k) acc[r][k] = 0.f;
    }
    for (int base = p0; base < end; base += kAttnTile) {
        const int n = min(kAttnTile, end - base);
        __syncthreads();   // the last tile's V is read (and the queries written)
        widen(kb, base, n);
        __syncthreads();
        float s[NR];
#pragma unroll
        for (int r = 0; r < NR; ++r) s[r] = 0.f;
        if (lane < n) {
            const float* kt = tile + lane * HD;
            for (int i = 0; i < HD; ++i) {
                const int d = (i + lane) & (HD - 1);
                const float kv = kt[d];
#pragma unroll
                for (int r = 0; r < NR; ++r) s[r] += qs[r * HD + d] * kv;
            }
        }
        // the tile's scores into each row's running max, sum and accumulator (online softmax)
#pragma unroll
        for (int r = 0; r < NR; ++r) {
            const float sv = lane < n && base + lane <= pos[r] ? s[r] * scale : -INFINITY;
            const float mn = fmaxf(m[r], warp_max(sv));
            const float alpha = mn == -INFINITY ? 1.f : expf(m[r] - mn);
            const float p = sv == -INFINITY ? 0.f : expf(sv - mn);
            l[r] = l[r] * alpha + warp_sum(p);
            m[r] = mn;
#pragma unroll
            for (int k = 0; k < 8; ++k) acc[r][k] *= alpha;
            s[r] = p;
        }
        __syncthreads();   // every warp has scored the K tile
        widen(vb, base, n);
        __syncthreads();
        for (int j = 0; j < n; ++j) {
            const float4 v0 = tile4[j * (HD / 4) + lane * 2], v1 = tile4[j * (HD / 4) + lane * 2 + 1];
#pragma unroll
            for (int r = 0; r < NR; ++r) {
                const float p = __shfl_sync(0xffffffffu, s[r], j);
                acc[r][0] += p * v0.x;
                acc[r][1] += p * v0.y;
                acc[r][2] += p * v0.z;
                acc[r][3] += p * v0.w;
                acc[r][4] += p * v1.x;
                acc[r][5] += p * v1.y;
                acc[r][6] += p * v1.z;
                acc[r][7] += p * v1.w;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < NR; ++r) {
        if (p0 > pos[r]) continue;
        const size_t o = (size_t) ((r0 - t0 + r) * H + h) * n_chunk + c;
        float4* dst = reinterpret_cast<float4*>(pacc + o * HD) + lane * 2;
        dst[0] = make_float4(acc[r][0], acc[r][1], acc[r][2], acc[r][3]);
        dst[1] = make_float4(acc[r][4], acc[r][5], acc[r][6], acc[r][7]);
        if (lane == 0) {
            pm[o] = m[r];
            pl[o] = l[r];
        }
    }
}
using AttnDec = void (*)(const float*, const __half*, const __half*, const int32_t*, int, int, int, float, int, float*,
                        float*, float*, RowSel);
// [q8_0 K / V][rows - 1]
constexpr AttnDec kAttnDec[2][8] = {
    {k_attn_dec<1, false>, k_attn_dec<2, false>, k_attn_dec<3, false>, k_attn_dec<4, false>, k_attn_dec<5, false>,
     k_attn_dec<6, false>, k_attn_dec<7, false>, k_attn_dec<8, false>},
    {k_attn_dec<1, true>, k_attn_dec<2, true>, k_attn_dec<3, true>, k_attn_dec<4, true>, k_attn_dec<5, true>,
     k_attn_dec<6, true>, k_attn_dec<7, true>, k_attn_dec<8, true>}};
using FaPrompt = void (*)(const float*, const __half*, const __half*, const float*, float*, int, int, int, int, float);
// [q8_0 K / V][query heads a KV head - 1]
constexpr FaPrompt kFaPrompt[2][8] = {
    {q36a::k_fa_prompt<1, false>, q36a::k_fa_prompt<2, false>, q36a::k_fa_prompt<3, false>, q36a::k_fa_prompt<4, false>,
     q36a::k_fa_prompt<5, false>, q36a::k_fa_prompt<6, false>, q36a::k_fa_prompt<7, false>, q36a::k_fa_prompt<8, false>},
    {q36a::k_fa_prompt<1, true>, q36a::k_fa_prompt<2, true>, q36a::k_fa_prompt<3, true>, q36a::k_fa_prompt<4, true>,
     q36a::k_fa_prompt<5, true>, q36a::k_fa_prompt<6, true>, q36a::k_fa_prompt<7, true>, q36a::k_fa_prompt<8, true>}};
/// The chunks merged, times sigmoid(gate): grid (n_rows), block HD.
__global__ void k_attn_merge(const float* pacc, const float* pm, const float* pl, const float* gate,
                             const int32_t* pos0, int t0, int H, int n_chunk, float* out, RowSel rs) {
    constexpr int HD = 256;
    const int row = blockIdx.x, t = t0 + row / H, h = row % H, d = threadIdx.x;
    const int npos = (rs.cell ? rs.cell[t] : *pos0 + t) + 1;
    const int nc = (npos + kChunkCells - 1) / kChunkCells;
    float M = -INFINITY;
    for (int c = 0; c < nc; ++c) M = fmaxf(M, pm[(size_t) row * n_chunk + c]);
    float num = 0.f, den = 0.f;
    for (int c = 0; c < nc; ++c) {
        const size_t o = (size_t) row * n_chunk + c;
        const float w = expf(pm[o] - M);
        num += w * pacc[o * HD + d];
        den += w * pl[o];
    }
    const size_t gi = ((size_t) t * H + h) * HD + d;
    out[gi] = num / den * sigm(gate[gi]);
}

/// The routed entries (T tokens x K picks, entry e = t * K + k) grouped per expert, on the device: one group per
/// expert used (ascending id), its blob address, its entries in entry order.  One block.
__global__ void k_group(const int32_t* ids, int n_ent, int K_, int n_expert, const uint8_t* blobs, unsigned long long blob,
                        unsigned long long* grp_ptr, int32_t* grp_start, int32_t* n_groups, int32_t* ent_dst,
                        int32_t* ent_tok) {
    __shared__ int cnt[512], off[512];
    for (int x = threadIdx.x; x < n_expert; x += blockDim.x) cnt[x] = 0;
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x) atomicAdd(&cnt[ids[e]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int g = 0, pos = 0;
        for (int x = 0; x < n_expert; ++x) {
            off[x] = pos;
            if (cnt[x] == 0) continue;
            grp_ptr[g] = (unsigned long long) (blobs + (size_t) x * blob);
            grp_start[g] = pos;
            pos += cnt[x];
            ++g;
        }
        grp_start[g] = pos;
        *n_groups = g;
        for (int e = 0; e < n_ent; ++e) {
            const int p = off[ids[e]]++;
            ent_dst[p] = e;
            ent_tok[p] = e / K_;
        }
    }
}
/// k_group for the hybrid mode: only the entries whose expert has a VRAM slot (slot_of[x] >= 0) are grouped, each
/// group's blob at blobs + slot * blob; the others are the CPU pool's (their rows come from k_merge_cpu).  One block.
__global__ void k_group_hy(const int32_t* ids, int n_ent, int K_, int n_expert, const int32_t* slot_of,
                           const uint8_t* blobs, unsigned long long blob, unsigned long long* grp_ptr, int32_t* grp_start,
                           int32_t* n_groups, int32_t* ent_dst, int32_t* ent_tok) {
    __shared__ int cnt[512], off[512];
    for (int x = threadIdx.x; x < n_expert; x += blockDim.x) cnt[x] = 0;
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x)
        if (slot_of[ids[e]] >= 0) atomicAdd(&cnt[ids[e]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int g = 0, pos = 0;
        for (int x = 0; x < n_expert; ++x) {
            off[x] = pos;
            if (cnt[x] == 0) continue;
            grp_ptr[g] = (unsigned long long) (blobs + (size_t) slot_of[x] * blob);
            grp_start[g] = pos;
            pos += cnt[x];
            ++g;
        }
        grp_start[g] = pos;
        *n_groups = g;
        for (int e = 0; e < n_ent; ++e) {
            if (slot_of[ids[e]] < 0) continue;
            const int p = off[ids[e]]++;
            ent_dst[p] = e;
            ent_tok[p] = e / K_;
        }
    }
}
/// The CPU pool's expert rows into the entries' rows: parts[e] = cpu[e] where miss[e].  grid (n_ent), block 256.
__global__ void k_merge_cpu(float* parts, const float* cpu, const int32_t* miss, int n) {
    const int e = blockIdx.x;
    if (!miss[e]) return;
    for (int i = threadIdx.x; i < n; i += blockDim.x) parts[(size_t) e * n + i] = cpu[(size_t) e * n + i];
}

/// Picture rows over the embeddings of a chunk: x[r] = img[map[r]] for the rows with map[r] >= 0 (the rows of a picture's
/// <|image_pad|> cells).  grid (T), block 256.
/// *p = v (a position for the stream's next kernels, without the pinned staging a later chunk would reuse)
__global__ void k_set_i32(int32_t* p, int32_t v) { *p = v; }
__global__ void k_put_rows(float* x, const float* img, const int32_t* map, int n) {
    const int r = blockIdx.x;
    const int m = map[r];
    if (m < 0) return;
    for (int i = threadIdx.x; i < n; i += blockDim.x) x[(size_t) r * n + i] = img[(size_t) m * n + i];
}

/// The prompt's routed entries sorted by expert for MMQ: bounds[x] .. bounds[x + 1] are expert x's rows; sorted row r
/// holds entry dst[r] (= t * K + k, the row of the per-slot outputs) of token src[r].  The order inside an expert is the
/// atomics' (no output depends on it: every row's products are its own).  cnt / cur: n_expert ints of scratch.
__global__ void k_route_count(const int32_t* ids, int n_ent, int32_t* cnt) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < n_ent) atomicAdd(&cnt[ids[e]], 1);
}
__global__ void k_route_scan(const int32_t* cnt, int n_expert, int32_t* bounds, int32_t* cur, int32_t* max_rows) {
    if (threadIdx.x != 0) return;
    int pos = 0, mx = 0;
    for (int x = 0; x < n_expert; ++x) {
        bounds[x] = pos;
        cur[x] = pos;
        pos += cnt[x];
        mx = max(mx, cnt[x]);
    }
    bounds[n_expert] = pos;
    *max_rows = mx;
}
__global__ void k_route_scatter(const int32_t* ids, int n_ent, int K_, int32_t* cur, int32_t* dst, int32_t* src,
                                int32_t* inv) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_ent) return;
    const int r = atomicAdd(&cur[ids[e]], 1);
    dst[r] = e;
    src[r] = e / K_;
    inv[e] = r;   // entry e's sorted row: where its down product lands
}

/// The probability of the largest of n logits under their softmax, 1 / sum(exp(l - max)): one block (blockDim a
/// multiple of 32, up to 1024)
__global__ void k_pmax(const float* lg, int n, float* out) {
    __shared__ float sm[32], ss[32];
    auto merge = [](float& m, float& s, float m2, float s2) {   // two (max, sum of exp(l - max)) into one
        const float mn = fmaxf(m, m2);
        s = (m == -INFINITY ? 0.f : s * expf(m - mn)) + (m2 == -INFINITY ? 0.f : s2 * expf(m2 - mn));
        m = mn;
    };
    float m = -INFINITY, s = 0.f;
    for (int i = (int) threadIdx.x; i < n; i += (int) blockDim.x) merge(m, s, lg[i], 1.f);
    for (int o = 16; o > 0; o >>= 1) merge(m, s, __shfl_xor_sync(0xffffffffu, m, o), __shfl_xor_sync(0xffffffffu, s, o));
    const int w = (int) threadIdx.x / 32, l = (int) threadIdx.x % 32;
    if (l == 0) {
        sm[w] = m;
        ss[w] = s;
    }
    __syncthreads();
    if (w != 0) return;
    m = l < (int) blockDim.x / 32 ? sm[l] : -INFINITY;
    s = l < (int) blockDim.x / 32 ? ss[l] : 0.f;
    for (int o = 16; o > 0; o >>= 1) merge(m, s, __shfl_xor_sync(0xffffffffu, m, o), __shfl_xor_sync(0xffffffffu, s, o));
    if (l == 0) *out = s > 0.f ? 1.f / s : 0.f;
}

/// The greedy pick of one logits row: out = argmax (lowest id on ties).  Stage 1: `gridDim.x` blocks each write the
/// best of their stride into (pv, pi); stage 2 (one block, n = the stage-1 block count) reads those.
__global__ void k_argmax(const float* x, int n, int32_t* out, float* pv, int32_t* pi, bool stage2) {
    __shared__ float bv[kThreads];
    __shared__ int bi[kThreads];
    // row blockIdx.y: its logits (stage 1, n = the vocabulary) and its kArgBlocks partials
    x += (size_t) blockIdx.y * (stage2 ? 0 : n);
    pv += (size_t) blockIdx.y * 128;
    pi += (size_t) blockIdx.y * 128;
    out += blockIdx.y;
    float v = -INFINITY;
    int id = 0x7fffffff;
    if (stage2) {
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            if (pv[i] > v || (pv[i] == v && pi[i] < id)) { v = pv[i]; id = pi[i]; }
    } else {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
            if (x[i] > v) { v = x[i]; id = i; }
    }
    bv[threadIdx.x] = v;
    bi[threadIdx.x] = id;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) {
            const float ov = bv[threadIdx.x + s];
            const int oi = bi[threadIdx.x + s];
            if (ov > bv[threadIdx.x] || (ov == bv[threadIdx.x] && oi < bi[threadIdx.x])) {
                bv[threadIdx.x] = ov;
                bi[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (stage2) *out = bi[0] == 0x7fffffff ? -1 : bi[0];   // -1: no finite logit (a NaN row)
        else { pv[blockIdx.x] = bv[0]; pi[blockIdx.x] = bi[0]; }
    }
}
constexpr int kArgBlocks = 128;   // (the row stride of k_argmax's partials)

unsigned grid1(int64_t n, int b = kThreads) { return (unsigned) ((n + b - 1) / b); }

/// out[t][o] = W[o] . x[t] in FP32 for a prompt chunk's small F32 projections (the router, the GDN alpha / beta,
/// the shared expert's gate: n_out <= 256): a (16 TM) x 64 output tile per block, K in steps of 16 through shared
/// memory, TM x 4 outputs per thread (TM = 4 for the router, 1 for the 32-wide alpha / beta: more blocks).  hipBLAS
/// picked 8 x 8 / 16 x 16 tiles for these shapes (~10% of a prompt).  K % 4 == 0.
template <int TM>
__global__ void __launch_bounds__(256) k_sgemm_nt(const float* __restrict__ W, const float* __restrict__ x,
                                                  float* __restrict__ out, int K, int n_out, int T, int ldo, int kz) {
    constexpr int BM = 16 * TM, BN = 64, BK = 16;
    const int kb = blockIdx.z * kz, ke = min(K, kb + kz);   // split-K: this block's K range, its own partial
    out += (size_t) blockIdx.z * T * ldo;
    __shared__ float As[BK][BM + 4], Bs[BK][BN + 4];
    const int tid = threadIdx.x;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int tm = (tid / 16) * TM, tn = (tid % 16) * 4;
    const int lr = tid / 4, lk = (tid % 4) * 4;   // this thread's loads: row lr of each tile, k lk .. lk + 3
    float acc[TM][4] = {};
    for (int k0 = kb; k0 < ke; k0 += BK) {
        if (lr < BM) {
            float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
            if (m0 + lr < T && k0 + lk < ke) a = *reinterpret_cast<const float4*>(x + (size_t) (m0 + lr) * K + k0 + lk);
            As[lk][lr] = a.x; As[lk + 1][lr] = a.y; As[lk + 2][lr] = a.z; As[lk + 3][lr] = a.w;
        }
        float4 b = make_float4(0.f, 0.f, 0.f, 0.f);
        if (n0 + lr < n_out && k0 + lk < ke) b = *reinterpret_cast<const float4*>(W + (size_t) (n0 + lr) * K + k0 + lk);
        Bs[lk][lr] = b.x; Bs[lk + 1][lr] = b.y; Bs[lk + 2][lr] = b.z; Bs[lk + 3][lr] = b.w;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float ar[TM];
#pragma unroll
            for (int i = 0; i < TM; ++i) ar[i] = As[k][tm + i];
            const float4 bv = *reinterpret_cast<const float4*>(&Bs[k][tn]);
            const float br[4] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(ar[i], br[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (m0 + tm + i < T && n0 + tn + j < n_out) out[(size_t) (m0 + tm + i) * ldo + n0 + tn + j] = acc[i][j];
}

/// out[i] = the sum of the split-K partials part[z][i] in order (deterministic), n values each.
__global__ void k_sum_splits(const float* part, float* out, int64_t n, int splits) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = 0.f;
    for (int z = 0; z < splits; ++z) a += part[(size_t) z * n + i];
    out[i] = a;
}

/// x f32 -> FP16 bits (the prompt GEMMs' activations).
__global__ void k_to_f16(const float* x, __half* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}

/// The GDN conv for T tokens in ONE launch (the prompt): one block per 128-channel head, thread = channel, the
/// 3-entry history carried in registers across the tokens and written back at the end; SiLU, then the L2 norm of the
/// q/k heads (block < qk_heads) per token - fused_gdn_conv_l2's arithmetic, token after token.  History [channel][3].
__global__ void __launch_bounds__(128) k_conv_seq(float* hist, const float* qkv, const float* w, float* h, int C,
                                                  int qk_heads, float eps, int T, bool write,
                                                  float* const* hist_tab = nullptr, const int32_t* slot_rows = nullptr,
                                                  const int32_t* grow0 = nullptr, const int32_t* gcnt = nullptr) {
    __shared__ float part[4];
    if (hist_tab) {   // batch rows, grid y = group: group g's history is session slot_rows[g]'s; its rows are
                      // grow0[g] .. + gcnt[g] - 1 (one row at row g when there are no group arrays)
        const int g = blockIdx.y, r0 = grow0 ? grow0[g] : g;
        hist = hist_tab[slot_rows[g]];
        if (gcnt) T = gcnt[g];
        qkv += (size_t) r0 * C;
        h += (size_t) r0 * C;
    }
    const int c = blockIdx.x * 128 + threadIdx.x;
    float v0 = hist[c * 3], v1 = hist[c * 3 + 1], v2 = hist[c * 3 + 2];
    // TILE mode (a prompt chunk: grid y > 1, no batch tables): the tiles of kConvTile tokens run side by side.  A tile's history
    // is the three inputs before its first token (exactly the values the sequential loop carries); tile 0 reads the old history,
    // and, as the only writer, stores the new one: the last three of (old history, inputs).  The arithmetic per token is the same.
    int t_first = 0, t_end = T;
    const bool tiled = hist_tab == nullptr && gridDim.y > 1;
    if (tiled) {
        t_first = (int) blockIdx.y * kConvTile;
        t_end = min(T, t_first + kConvTile);
        if (t_first == 0 && write) {
            const float o0 = v0, o1 = v1, o2 = v2;
            const float old3[3] = {o0, o1, o2};
            for (int k = 0; k < 3; ++k) {
                const int idx = T - 3 + k;
                hist[c * 3 + k] = idx >= 0 ? qkv[(size_t) idx * C + c] : old3[idx + 3];
            }
        }
        if (t_first > 0) {
            v0 = qkv[(size_t) (t_first - 3) * C + c];
            v1 = qkv[(size_t) (t_first - 2) * C + c];
            v2 = qkv[(size_t) (t_first - 1) * C + c];
        }
        if (t_first >= T) return;
    }
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    const bool qk = (int) blockIdx.x < qk_heads;
    for (int t = t_first; t < t_end; ++t) {
        const float x = qkv[(size_t) t * C + c];
        const float sum = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
        v0 = v1;
        v1 = v2;
        v2 = x;
        float y = sum / (1.0f + __expf(-sum));
        if (qk) {
            y *= rsqrtf(block128_sum(y * y, part) + eps);
            __syncthreads();   // every thread has read `part` before the next token writes it
        }
        h[(size_t) t * C + c] = y;
    }
    if (!write || tiled) return;   // a verify window: the history is commit()'s to move; tiles: tile 0 stored it
    hist[c * 3] = v0;
    hist[c * 3 + 1] = v1;
    hist[c * 3 + 2] = v2;
}

/// The conv history after the first r tokens of a chunk: the last three of (the history before it, its inputs), which
/// is what k_conv_seq carries past token r - 1.  One block per 128 channels; before k_conv_seq, whose tile 0 stores
/// the chunk's own.
__global__ void __launch_bounds__(128) k_conv_hist(const float* hist, const float* qkv, float* out, int C, int r) {
    const int c = blockIdx.x * 128 + threadIdx.x;
    for (int k = 0; k < 3; ++k) {
        const int idx = r - 3 + k;
        out[c * 3 + k] = idx >= 0 ? qkv[(size_t) idx * C + c] : hist[c * 3 + idx + 3];
    }
}

/// The gated delta rule + SiLU-gated norm for T tokens in ONE launch (the prompt): fused_gdn_step_norm's kernel
/// (block = one V head, 128 columns x 4 row groups, state [row][head][col]) with its 32 state rows per thread kept in
/// registers across the tokens.  h (T, C) = q | k | v, gate / beta (T, HV), z / y (T, ZV).
__global__ void __launch_bounds__(512) k_gdn_seq(float* state, const float* h, int C, const float* gate,
                                                 const float* beta, const float* z, const float* gamma, float eps,
                                                 float* y, int h_k, int h_v, int T, bool write,
                                                 float* const* state_tab = nullptr, const int32_t* slot_rows = nullptr,
                                                 const int32_t* grow0 = nullptr, const int32_t* gcnt = nullptr) {
    constexpr int S = 128, RG = 4, RPG = 32;
    if (state_tab) {   // batch rows, grid y = group (see k_conv_seq)
        const int g = blockIdx.y, r = grow0 ? grow0[g] : g;
        state = state_tab[slot_rows[g]];
        if (gcnt) T = gcnt[g];
        h += (size_t) r * C;
        gate += (size_t) r * h_v;
        beta += (size_t) r * h_v;
        z += (size_t) r * h_v * S;
        if (y) y += (size_t) r * h_v * S;
    }
    __shared__ float sk[S], sq[S];
    __shared__ float red_kv[RG][S];
    __shared__ float red_o[RG][S];
    __shared__ float wsum[S * RG / 32];
    const int head = blockIdx.x, col = threadIdx.x, rg = threadIdx.y, tid = rg * S + col;
    const int qh = head % h_k;
    const int ZV = h_v * S;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    for (int t = 0; t < T; ++t) {
        const float* ht = h + (size_t) t * C;
        if (tid < S) { sk[tid] = ht[h_k * S + qh * S + tid]; sq[tid] = ht[qh * S + tid]; }
        __syncthreads();
        const float g = __expf(gate[(size_t) t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red_kv[rg][col] = kv;
        __syncthreads();
        const float kv_col = red_kv[0][col] + red_kv[1][col] + red_kv[2][col] + red_kv[3][col];
        const float delta = (ht[2 * h_k * S + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        red_o[rg][col] = o;
        __syncthreads();
        float oc = 0.0f, sq_part = 0.0f;
        if (rg == 0) {
            oc = (red_o[0][col] + red_o[1][col] + red_o[2][col] + red_o[3][col]) * rsqrtf((float) S);
            sq_part = oc * oc;
        }
        for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += __shfl_xor_sync(0xffffffffu, sq_part, o2);
        if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
        __syncthreads();
        if (rg == 0 && y != nullptr) {   // (the commit runs without outputs)
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float scale = rsqrtf(ss / (float) S + eps);
            const float zz = z[(size_t) t * ZV + head * S + col];
            y[(size_t) t * ZV + head * S + col] = oc * scale * gamma[col] * (zz / (1.0f + __expf(-zz)));
        }
        __syncthreads();   // sk / sq / the reductions are rewritten by the next token
    }
    if (!write) return;   // a verify window: the state is commit()'s to write
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
}

/// The prompt's gated delta rule with more parallelism: a state column's update needs only that column (kv, delta and
/// the output o all reduce over the rows), so a V head splits into kGdnColGroups blocks of 128 / kGdnColGroups columns,
/// each with 16 row groups of 8 rows (512 threads, 8 state values per thread in registers).  Writes the raw outputs
/// o (T, HV * 128); k_gdn_norm applies the SiLU-gated RMS norm.  grid (HV, kGdnColGroups), block (32, 16).
constexpr int kGdnColGroups = 4;
__global__ void __launch_bounds__(512) k_gdn_seq_cols(float* state, const float* h, int C, const float* gate,
                                                      const float* beta, float* o_raw, int h_k, int h_v, int T) {
    constexpr int S = 128, NC = S / kGdnColGroups, RG = 16, RPG = S / RG;
    // two barriers per token (was three): sk / sq are double-buffered, and the output sum of token t is done after token t + 1's first
    // barrier, off the chain of the recurrence (red_o is read between that barrier and the next one, written after it)
    __shared__ float sk[2][S], sq[2][S];
    __shared__ float red_kv[RG][NC];
    __shared__ float red_o[RG][NC];
    const int head = blockIdx.x, cl = threadIdx.x, rg = threadIdx.y, tid = rg * NC + cl;
    const int col = blockIdx.y * NC + cl;
    const int qh = head % h_k;
    float sv[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) sv[r] = base[r * row_stride];
    // the next token's inputs are loaded while this one is computed (the loads were the head of every iteration)
    float nsk = 0.0f, nsq = 0.0f, nvv = 0.0f, ngate = 0.0f, nbeta = 0.0f;
    auto load = [&](int t) {
        const float* ht = h + (size_t) t * C;
        if (tid < S) { nsk = ht[h_k * S + qh * S + tid]; nsq = ht[qh * S + tid]; }
        nvv = ht[2 * h_k * S + head * S + col];
        ngate = gate[(size_t) t * h_v + head];
        nbeta = beta[(size_t) t * h_v + head];
    };
    if (T > 0) load(0);
    for (int t = 0; t < T; ++t) {
        const int par = t & 1;
        if (tid < S) { sk[par][tid] = nsk; sq[par][tid] = nsq; }
        const float vcur = nvv, gcur = ngate, bcur = nbeta;
        if (t + 1 < T) load(t + 1);
        __syncthreads();
        if (t > 0 && rg == 0) {   // token t - 1's output
            float oc = 0.0f;
#pragma unroll
            for (int i = 0; i < RG; ++i) oc += red_o[i][cl];
            o_raw[(size_t) (t - 1) * h_v * S + head * S + col] = oc * rsqrtf((float) S);
        }
        const float g = __expf(gcur);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(sv[r], sk[par][rg * RPG + r], kv);
        red_kv[rg][cl] = kv;
        __syncthreads();
        float kv_col = 0.0f;
#pragma unroll
        for (int i = 0; i < RG; ++i) kv_col += red_kv[i][cl];
        const float delta = (vcur - g * kv_col) * bcur;
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            sv[r] = fmaf(g, sv[r], sk[par][rg * RPG + r] * delta);
            o = fmaf(sv[r], sq[par][rg * RPG + r], o);
        }
        red_o[rg][cl] = o;
    }
    __syncthreads();
    if (T > 0 && rg == 0) {   // the last token's output
        float oc = 0.0f;
#pragma unroll
        for (int i = 0; i < RG; ++i) oc += red_o[i][cl];
        o_raw[(size_t) (T - 1) * h_v * S + head * S + col] = oc * rsqrtf((float) S);
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * row_stride] = sv[r];
}
/// y = rms(o) * gamma * SiLU(z) per (token, V head): grid (HV, T), block 128.
__global__ void __launch_bounds__(128) k_gdn_norm(const float* o_raw, const float* z, const float* gamma, float eps,
                                                  float* y, int h_v) {
    __shared__ float part[4];
    const size_t i = ((size_t) blockIdx.y * h_v + blockIdx.x) * 128 + threadIdx.x;
    const float oc = o_raw[i];
    const float scale = rsqrtf(block128_sum(oc * oc, part) / 128.0f + eps);
    const float zz = z[i];
    y[i] = oc * scale * gamma[threadIdx.x] * (zz / (1.0f + __expf(-zz)));
}

// ------------------------------------------------------------------------------------------------ the model
/// LoRA adapters (--lora, --lora-scaled: llama.cpp's GGUF adapters; q36_lora.cuh's kernels).  One adapter's pair at one
/// product: A [rank][n_in] and B [n_out][rank] on the product's GPU, the rank padded to a multiple of q36lora::kGroup
/// with zeros; F16 (the file's F16 and Q8_0 pairs) or F32 (its F32 and BF16 ones).
struct LoraSeg {
    int adapter = 0;
    int rank = 0;
    float rank_true = 0.f;
    bool f32 = false;
    void* a = nullptr;
    void* b = nullptr;
};
/// the adapters' pairs at one product
struct LoraMat {
    std::vector<LoraSeg> segs;
};
/// a loaded adapter
struct LoraAdapter {
    std::string path, task_name, prompt_prefix;
    float alpha = 0.f;     // adapter.lora.alpha (0: none, a pair's scale is the adapter's as given)
    int pairs = 0, rank_min = 0, rank_max = 0;
    uint64_t bytes = 0;    // on the GPUs
};

struct Mat {
    const uint8_t* d = nullptr;   // device, GGUF blocks
    int type = -1;
    int n_in = 0, n_out = 0;
    const LoraMat* lora = nullptr;   // the adapters' pairs for this product (null: none)
};

struct Layer {
    bool attn = false;
    const float *attn_norm = nullptr, *post_norm = nullptr;
    // GDN
    Mat qkv, zg, ssm_out;
    const float *alpha = nullptr, *beta = nullptr, *dt = nullptr, *ssm_a = nullptr, *conv = nullptr, *ssm_norm = nullptr;
    const LoraMat *lora_alpha = nullptr, *lora_beta = nullptr;   // the adapters' pairs for the F32 alpha / beta products
    // attention
    Mat wq, wk, wv, wo;
    const float *q_norm = nullptr, *k_norm = nullptr;
    // MoE
    const float* router = nullptr;   // [n_expert][N] f32
    float* router_ext = nullptr;     // the same with the shared expert's gate row appended: [n_expert + 1][N]
    Mat sh_gate, sh_up, sh_down;
    const float* sh_gate_inp = nullptr;   // [N]
    Mat ffn_gate, ffn_up, ffn_down;       // the dense FFN of a qwen35 layer (no router, no experts)
    K::NativeExpertLayout L;
    uint8_t* blobs = nullptr;   // n_expert blobs, `stride` bytes apart (L.bytes rounded up to whole blocks of every
    size_t stride = 0;          // part's type, so MMQ can step over them by blocks), then kMmqTail zero bytes
                                // (hybrid mode: n_slots slots, see place_experts)
    // the GGUF's expert tensors in the mapped file, until place_experts has built the blobs from them
    const uint8_t *src_g = nullptr, *src_u = nullptr, *src_d = nullptr;
    size_t src_gsz = 0, src_dsz = 0;
    uint8_t* hblobs = nullptr;  // hybrid mode: every expert's blob in host memory, `stride` apart
    int n_slots = 0;            // hybrid mode: the experts resident in VRAM
};

struct Model {
    lamina::core::ModelGeometry g;
    bool moe = true;                           // qwen35moe: the routed and shared experts; qwen35: a dense FFN
    std::shared_ptr<lamina::GgufModel> file;   // mapped until the experts are placed (after the engine's buffers)
    bool hybrid = false;                       // not every expert fits in VRAM: slots + the CPU pool
    bool kv_q8 = false;                        // --kv q8_0: K and V in q8_0 rows (kKvQ8Row), else FP16
    /// A KV head's cell of K or V, in bytes
    size_t kv_row() const { return kv_q8 ? (size_t) kKvQ8Row : (size_t) g.head_dim * sizeof(__half); }
    // the GGUF-native MTP block (blk.<n_layers>, drafting with --drafts-per-step): eh_proj over [enorm(e) | hnorm(h)],
    // then an attention + MoE layer, shared_head_norm, the main head
    bool has_mtp = false;
    Layer mtp;
    Mat eh_proj;
    const float *mtp_enorm = nullptr, *mtp_hnorm = nullptr, *mtp_shnorm = nullptr;
    int64_t top_k = 8;
    float eps = 1e-6f, rope_base = 1e7f;
    int n_rot = 64;
    std::vector<Layer> layers;
    const float* out_norm = nullptr;
    Mat head;
    std::vector<LoraAdapter> loras;     // --lora's files, in their order (a request's scales are in it too)
    std::deque<LoraMat> lora_mats;      // the products' pairs (Mat::lora points here)
    const uint8_t* embd = nullptr;
    const uint8_t* embd_mtp = nullptr;   // the MTP block's: embd, or its copy on the last stage's GPU
    int embd_type = -1;
    size_t embd_row = 0;
    int64_t embd_rows = 0;
    // the layer split (--second-gpu): stage s runs the layers below stage_end[s] (from stage_end[s - 1]) on GPU
    // stage_dev[s]; the embedding is on the first stage's GPU, the output norm, the head and the MTP block on the last's
    std::vector<int> stage_dev;
    std::vector<size_t> stage_end;
    int dev_of(int64_t l) const {
        for (size_t i = 0; i < stage_end.size(); ++i)
            if ((size_t) l < stage_end[i]) return stage_dev[i];
        return stage_dev.back();   // the MTP block (index n_layers)
    }
};

constexpr size_t kMmqTail = 4096;
size_t gcd_sz(size_t a, size_t b) { return b ? gcd_sz(b, a % b) : a; }
size_t lcm_sz(size_t a, size_t b) { return a / gcd_sz(a, b) * b; }
size_t block_bytes(int type) {
    int be = 0, bb = 0;
    return lamina::block_geometry((uint32_t) type, be, bb) ? (size_t) bb : 1;
}   // zeroed bytes after the last expert (MMQ reads a little past a matrix)

size_t row_bytes(int type, int64_t n) {
    int be = 0, bb = 0;
    if (!lamina::block_geometry((uint32_t) type, be, bb)) return 0;
    return (size_t) (n / be) * (size_t) bb;
}

float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// IEEE binary16 -> f32 by hand. __ushort_as_half is a __device__ intrinsic and
// upload_f32 runs on the host, so a CUDA toolkit older than 12.5 refuses the
// call; widen the bits here instead, the way bf16_to_f32 does.
float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000) << 16;
    const uint32_t exponent = (h >> 10) & 0x1f;
    const uint32_t mantissa = h & 0x3ff;
    uint32_t u;
    if (exponent == 0) {
        if (mantissa == 0) {
            u = sign;                                               // +-0
        } else {
            uint32_t m = mantissa, shift = 0;                        // subnormal: normalize it
            while ((m & 0x400) == 0) {
                m <<= 1;
                ++shift;
            }
            u = sign | ((uint32_t) (127 - 14 - shift) << 23) | ((m & 0x3ff) << 13);
        }
    } else if (exponent == 0x1f) {
        u = sign | 0x7f800000u | (mantissa << 13);                   // inf / nan
    } else {
        u = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    }
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

/// A tensor's values as f32 on the host (F32 as it is, BF16 / F16 widened, Q8_0 scaled); false for another type.
bool tensor_floats(const lamina::TensorInfo& t, const uint8_t* src, std::vector<float>& h) {
    h.resize((size_t) t.elements());
    if (t.type == 0) std::memcpy(h.data(), src, h.size() * 4);
    else if (t.type == 30)
        for (size_t i = 0; i < h.size(); ++i) h[i] = bf16_to_f32((uint16_t) (src[2 * i] | (src[2 * i + 1] << 8)));
    else if (t.type == 1)   // F16
        for (size_t i = 0; i < h.size(); ++i)
            h[i] = f16_to_f32((uint16_t) (src[2 * i] | (src[2 * i + 1] << 8)));
    else if (t.type == 8 && h.size() % 32 == 0) {
        // Q8_0 (the Q8_0 file stores ssm_alpha / ssm_beta so): blocks of an fp16 scale and 32 int8, widened here
        for (size_t b = 0; b < h.size() / 32; ++b) {
            const uint8_t* q = src + b * 34;
            const float d = f16_to_f32((uint16_t) (q[0] | (q[1] << 8)));
            for (int j = 0; j < 32; ++j) h[b * 32 + j] = d * (float) (int8_t) q[2 + j];
        }
    } else {
        return false;
    }
    return true;
}

/// A tensor as f32 on the device (F32 as it is, BF16 / F16 widened).
float* upload_f32(const lamina::TensorInfo& t, const uint8_t* src) {
    std::vector<float> h;
    if (!tensor_floats(t, src, h)) {
        std::fprintf(stderr, "lamina: %s is %s, expected F32, BF16, F16 or Q8_0\n", t.name.c_str(), t.type_name());
        throw Fatal{1};
    }
    float* d = nullptr;
    CK(dmalloc(&d, h.size() * 4));
    CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    return d;
}

/// About the bytes Engine::init's buffers take per row of a prompt chunk.
uint64_t buffer_bytes_per_row(const Model& m) {
    const auto& g = m.g;
    const uint64_t N = g.n_embd, C = g.ssm_conv_channels, VD = g.ssm_value_dim, HQ = g.n_head * g.head_dim;
    const uint64_t FW = m.moe ? g.n_ff_shexp : g.n_ff, Kp = m.moe ? m.top_k : 1;
    const uint64_t max_in = std::max<uint64_t>({2 * N, VD, HQ, FW});
    const uint64_t floats = 6 * N + 2 * C + 2 * VD + 2 * g.n_head_kv * g.head_dim + 5 * HQ + 3 * FW + (Kp + 2) * N +
                            (g.n_expert + 1);
    uint64_t bytes = 4 * floats + 2 * max_in + 3 * max_in;   // + the FP16 and q8_1 images of a projection's input
    if (m.moe) bytes += Kp * (12 * (uint64_t) g.n_ff + 2 * N + 2 * g.n_ff);   // the prompt experts' MMQ buffers
    return bytes;
}

/// The layer split: one stage on the current GPU without --second-gpu; with `split`, its layer counts; else the fewest
/// of `devs`, in order, that hold the model: the layers in order, each stage taking what its GPU holds (its free VRAM
/// less the engine's buffers at a `chunk`-token prompt chunk and 1.5 GiB: a layer's tensors, the attention layers'
/// K / V at `ctx` cells; the first stage's the embedding, the last's the head and the MTP block) but a layer for each
/// later stage, and the last the rest. A model with experts that no split holds runs on the first GPU alone (hybrid).
/// `lora`: the adapters' bytes per block (the MTP block at n_layers), the head's at n_layers + 1.
bool plan_stages(Model& m, const lamina::GgufModel& gm, const std::vector<int>& devs, const std::vector<int>& split,
                 int ctx, int chunk, const std::vector<uint64_t>& lora, std::string& err) {
    const auto& g = m.g;
    const size_t nl = (size_t) g.n_layers;
    m.stage_dev = devs;
    m.stage_end.clear();
    if (devs.size() == 1) {
        m.stage_end.push_back(nl);
        return true;
    }
    if (!split.empty()) {
        if (split.size() != devs.size()) {
            err = "--layer-split takes a layer count for every GPU (" + std::to_string(devs.size()) + ")";
            return false;
        }
        size_t end = 0;
        for (const int c : split) {
            if (c < 1) {
                err = "--layer-split: every GPU takes at least one layer";
                return false;
            }
            end += (size_t) c;
            m.stage_end.push_back(end);
        }
        if (end != nl) {
            err = "--layer-split: the counts add up to " + std::to_string(end) + " layers, the model has " + std::to_string(nl);
            return false;
        }
        return true;
    }
    std::vector<uint64_t> cost(nl + 1, 0);   // per layer (the MTP block at nl); the rest: the embedding, the head
    uint64_t embd = 0, tail = 0;
    for (size_t si = 0; si < gm.size(); ++si)
        for (const lamina::TensorInfo& t : gm.shard(si).tensors()) {
            const uint64_t b = lamina::tensor_payload_bytes(t);
            if (t.name.rfind("blk.", 0) == 0) {
                const size_t l = (size_t) std::strtoull(t.name.c_str() + 4, nullptr, 10);
                if (l <= nl) cost[l] += b;
            } else if (t.name == "token_embd.weight") {
                embd += b;
            } else {
                tail += b;
            }
        }
    const uint64_t kv = (uint64_t) g.n_head_kv * m.kv_row() * 2 * (uint64_t) ctx;
    for (size_t l = 0; l <= nl; ++l)
        if (l == nl ? g.n_mtp_layers > 0 : lamina::core::is_qsa_layer(g, (int64_t) l)) cost[l] += kv;
    for (size_t l = 0; l < lora.size() && l <= nl + 1; ++l) (l <= nl ? cost[l] : tail) += lora[l];
    tail += cost[nl];
    const uint64_t overhead = ((uint64_t) 3 << 29) + (uint64_t) chunk * buffer_bytes_per_row(m);
    // a loan asked of a card (kVramReserve): it stays free there past the 1.5 GiB, where the same number of GPUs
    // still holds the model (room_loan; else the split is the one without the loan)
    std::vector<uint64_t> room_plain(devs.size()), room_loan(devs.size());
    bool any_loan = false;
    for (size_t i = 0; i < devs.size(); ++i) {
        size_t fr = 0, tot = 0;
        CK(cudaSetDevice(devs[i]));
        CK(cudaMemGetInfo(&fr, &tot));
        const uint64_t keep = overhead + loan_keep(devs[i]);
        room_plain[i] = fr > overhead ? fr - overhead : 0;
        room_loan[i] = fr > keep ? fr - keep : 0;
        any_loan = any_loan || keep != overhead;
    }
    CK(cudaSetDevice(devs[0]));
    const auto take = [&](size_t n, const std::vector<size_t>& ends) {
        m.stage_dev.assign(devs.begin(), devs.begin() + (std::ptrdiff_t) n);
        m.stage_end = ends;
        // the GPUs left out: the context the query made goes
        for (size_t i = n; i < devs.size(); ++i) lamina::core::gpu_reset_unused(devs[i]);
        CK(cudaSetDevice(devs[0]));
    };
    for (size_t n = 1; n <= devs.size(); ++n)
    for (const bool with_loan : {true, false}) {
        if (with_loan && !any_loan) continue;
        const std::vector<uint64_t>& room = with_loan ? room_loan : room_plain;
        std::vector<size_t> ends;
        size_t l = 0;
        bool stuck = false;   // a GPU before the last holds no layer: more GPUs do not help
        for (size_t i = 0; i < n; ++i) {
            const uint64_t fixed = (i == 0 ? embd : 0) + (i + 1 == n ? tail : 0);
            const uint64_t r = room[i] > fixed ? room[i] - fixed : 0;
            const size_t first = l;
            if (i + 1 == n) {
                uint64_t need = 0;
                for (; l < nl; ++l) need += cost[l];
                if (need > r) {
                    err = "the GPUs do not hold the model at a context of " + std::to_string(ctx) + ", the last needing " +
                          std::to_string(need >> 20) + " MiB of its " + std::to_string(r >> 20) + " MiB of room";
                    break;
                }
            } else {
                for (uint64_t used = 0; l + (n - 1 - i) < nl && used + cost[l] <= r; ++l) used += cost[l];
                if (l == first) {
                    err = "GPU " + std::to_string(devs[i]) + " holds no layer";
                    stuck = true;
                    break;
                }
            }
            ends.push_back(l);
        }
        if (ends.size() == n) {
            take(n, ends);
            if (n == 1) std::fprintf(stderr, "lamina: GPU %d holds the model: no layer split\n", devs[0]);
            return true;
        }
        if (stuck && !with_loan) n = devs.size();   // (more GPUs do not help: the outer loop ends)
    }
    if (m.moe) {   // the split needs every expert in VRAM; one GPU runs the rest of them on the CPU
        take(1, {nl});
        std::fprintf(stderr, "lamina: no layer split, %s: GPU %d runs alone, the CPU computing the experts it lacks\n",
                     err.c_str(), devs[0]);
        return true;
    }
    err += "; a shorter --max-context, --layer-split or another GPU";
    return false;
}

// ------------------------------------------------------------------------------------------------ LoRA adapters
/// --lora / --lora-scaled: a file and its scale at the start
struct LoraSpec {
    std::string path;
    float scale = 1.f;
};

/// An adapter's pair: its product's tensor name, its lora_a [n_in, rank] and lora_b [rank, n_out]
struct LoraPair {
    std::string name;
    const lamina::TensorInfo* a = nullptr;
    const lamina::TensorInfo* b = nullptr;
};

/// An adapter file read as llama.cpp reads one: general.type "adapter", adapter.type "lora" (an activated LoRA, with
/// invocation tokens, refused), the architecture `arch` (empty: not checked); its alpha, task name and prompt prefix,
/// and its pairs (each .lora_a with its .lora_b; the norms' vectors left out, as llama.cpp leaves them).
bool lora_read(const lamina::GgufFile& f, const std::string& arch, LoraAdapter& ad, std::vector<LoraPair>& pairs,
               std::string& err) {
    auto str = [&](const char* k) -> std::string {
        const lamina::MetaValue* v = f.get(k);
        return v != nullptr && v->type == lamina::MetaType::STRING ? v->s : "";
    };
    const std::string at = f.path() + ": ";
    if (str("general.type") != "adapter") {
        err = at + "general.type is '" + str("general.type") + "', not 'adapter' (a LoRA adapter's GGUF)";
        return false;
    }
    if (str("adapter.type") != "lora") {
        err = at + "adapter.type is '" + str("adapter.type") + "', not 'lora'";
        return false;
    }
    if (!arch.empty() && str("general.architecture") != arch) {
        err = at + "an adapter of " + str("general.architecture") + ", the model is " + arch;
        return false;
    }
    if (f.get("adapter.alora.invocation_tokens") != nullptr) {
        err = at + "an activated LoRA (aLoRA), which this path does not run";
        return false;
    }
    if (const lamina::MetaValue* v = f.get("adapter.lora.alpha"); v != nullptr && v->is_num()) ad.alpha = (float) v->num();
    ad.task_name = str("adapter.lora.task_name");
    ad.prompt_prefix = str("adapter.lora.prompt_prefix");
    std::map<std::string, LoraPair> by;
    for (const lamina::TensorInfo& t : f.tensors()) {
        const std::string& n = t.name;
        auto ends = [&](const char* s) {
            const size_t k = std::strlen(s);
            return n.size() > k && n.compare(n.size() - k, k, s) == 0;
        };
        if (ends(".lora_a") || ends(".lora_b")) {
            LoraPair& p = by[n.substr(0, n.size() - 7)];
            p.name = n.substr(0, n.size() - 7);
            (ends(".lora_a") ? p.a : p.b) = &t;
        } else if (!ends("_norm.weight")) {   // (norms' vectors: left out)
            err = at + "the tensor " + n + " has an unexpected suffix";
            return false;
        }
    }
    pairs.clear();
    for (auto& [name, p] : by) {
        if (p.a == nullptr || p.b == nullptr) {
            err = at + "the pair of " + name + " lacks its " + (p.a ? "lora_b" : "lora_a");
            return false;
        }
        if (p.a->shape.size() != 2 || p.b->shape.size() != 2 || p.a->shape[1] != p.b->shape[0] || p.a->shape[1] == 0) {
            err = at + name + ": lora_a and lora_b do not pair (an adapter of llama.cpp's old finetune example?)";
            return false;
        }
        for (const lamina::TensorInfo* t : {p.a, p.b})
            if (t->type != 0 && t->type != 1 && t->type != 8 && t->type != 30) {
                err = at + t->name + " is " + t->type_name() + ": F32, F16, BF16 or Q8_0 pairs only";
                return false;
            }
        pairs.push_back(p);
    }
    if (pairs.empty()) {
        err = at + "no LoRA pairs";
        return false;
    }
    return true;
}

int lora_rank_pad(int r) { return (r + q36lora::kGroup - 1) / q36lora::kGroup * q36lora::kGroup; }
/// a pair kept in F32 (a part in F32 or BF16) or F16 (both in F16 or Q8_0)
bool lora_f32(const LoraPair& p) { return p.a->type == 0 || p.a->type == 30 || p.b->type == 0 || p.b->type == 30; }
uint64_t lora_pair_bytes(const LoraPair& p) {
    return (p.a->shape[0] + p.b->shape[1]) * (uint64_t) lora_rank_pad((int) p.a->shape[1]) * (lora_f32(p) ? 4 : 2);
}
/// blk.N's N (the MTP block's too), else -1
int64_t lora_block(const std::string& name) {
    return name.rfind("blk.", 0) == 0 ? std::strtoll(name.c_str() + 4, nullptr, 10) : -1;
}

/// The adapters' bytes on the GPUs per block (the MTP block at n_layers), the head's at n_layers + 1: the split's plan.
bool lora_scan(const std::vector<LoraSpec>& specs, int64_t n_layers, std::vector<uint64_t>& cost, std::string& err) {
    cost.assign((size_t) n_layers + 2, 0);
    for (const LoraSpec& sp : specs) {
        try {
            const lamina::GgufFile f(sp.path);
            LoraAdapter ad;
            std::vector<LoraPair> pairs;
            if (!lora_read(f, "", ad, pairs, err)) return false;
            for (const LoraPair& p : pairs) {
                const int64_t l = p.name == "output.weight" ? n_layers + 1 : lora_block(p.name);
                if (l >= 0 && l <= n_layers + 1) cost[(size_t) l] += lora_pair_bytes(p);
            }
        } catch (const std::exception& e) {
            err = sp.path + ": " + e.what();
            return false;
        }
    }
    return true;
}

/// The product a pair adds to, by its tensor's name: the Mat's pairs slot (or the F32 alpha / beta product's), its GPU
/// and widths; false with the reason for a name this path has no product for (llama.cpp's "does not exist in base
/// model") or does not take an adapter's pair at (the embedding, the router, the routed experts).
bool lora_target(Model& m, const std::string& name, const LoraMat**& slot, int& dev, int& n_in, int& n_out,
                 std::string& why) {
    const auto& g = m.g;
    auto take = [&](Mat& w, int d) {
        if (w.d == nullptr) {
            why = "the model has no such tensor";
            return false;
        }
        slot = &w.lora;
        dev = d;
        n_in = w.n_in;
        n_out = w.n_out;
        return true;
    };
    if (name == "output.weight") return take(m.head, m.stage_dev.back());
    if (name == "token_embd.weight") {
        why = "this path takes no pair for the embedding";
        return false;
    }
    const int64_t l = lora_block(name);
    const size_t dot = name.find('.', 4);
    if (l < 0 || l > g.n_layers || (l == g.n_layers && !m.has_mtp) || dot == std::string::npos) {
        why = "the model has no such tensor";
        return false;
    }
    Layer& L = l == g.n_layers ? m.mtp : m.layers[(size_t) l];
    const int d = m.dev_of(l);
    const std::string s = name.substr(dot + 1);
    if (s == "attn_qkv.weight") return take(L.qkv, d);
    if (s == "attn_gate.weight") return take(L.zg, d);
    if (s == "ssm_out.weight") return take(L.ssm_out, d);
    if (s == "attn_q.weight") return take(L.wq, d);
    if (s == "attn_k.weight") return take(L.wk, d);
    if (s == "attn_v.weight") return take(L.wv, d);
    if (s == "attn_output.weight") return take(L.wo, d);
    if (s == "ffn_gate.weight") return take(L.ffn_gate, d);
    if (s == "ffn_up.weight") return take(L.ffn_up, d);
    if (s == "ffn_down.weight") return take(L.ffn_down, d);
    if (s == "ffn_gate_shexp.weight") return take(L.sh_gate, d);
    if (s == "ffn_up_shexp.weight") return take(L.sh_up, d);
    if (s == "ffn_down_shexp.weight") return take(L.sh_down, d);
    if (s == "nextn.eh_proj.weight" && l == g.n_layers) return take(m.eh_proj, d);
    if ((s == "ssm_alpha.weight" || s == "ssm_beta.weight") && L.alpha != nullptr) {
        slot = s == "ssm_alpha.weight" ? &L.lora_alpha : &L.lora_beta;
        dev = d;
        n_in = (int) g.n_embd;
        n_out = (int) g.ssm_v_heads;
        return true;
    }
    if (s.find("_exps.") != std::string::npos) why = "this path takes no pair for the routed experts";
    else if (s.rfind("ffn_gate_inp", 0) == 0) why = "this path takes no pair for the router or the shared expert's gate";
    else why = "the model has no such tensor";
    return false;
}

/// floats to the current GPU: in F32, or rounded to F16
void* lora_upload(const std::vector<float>& v, bool f32) {
    void* d = nullptr;
    if (f32) {
        CK(dmalloc(&d, v.size() * 4));
        CK(cudaMemcpy(d, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
        return d;
    }
    std::vector<__half> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) h[i] = __float2half_rn(v[i]);
    CK(dmalloc(&d, h.size() * 2));
    CK(cudaMemcpy(d, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
    return d;
}

/// --lora's files onto the loaded model: each pair checked against its product (llama.cpp's checks) and placed on the
/// product's GPU, its rank padded to whole groups.
bool load_loras(Model& m, const std::vector<LoraSpec>& specs, std::string& err) {
    const std::string arch = lamina::model_family_arch(m.g.family);
    std::map<const LoraMat**, LoraMat*> made;   // a product's pairs, the files adding to them in their order
    std::vector<float> fa, fb;
    for (size_t ai = 0; ai < specs.size(); ++ai) {
        const auto t0 = Clock::now();
        try {
            const lamina::GgufFile f(specs[ai].path);
            LoraAdapter ad;
            ad.path = specs[ai].path;
            std::vector<LoraPair> pairs;
            if (!lora_read(f, arch, ad, pairs, err)) return false;
            for (const LoraPair& p : pairs) {
                const LoraMat** slot = nullptr;
                int dev = 0, n_in = 0, n_out = 0;
                std::string why;
                if (!lora_target(m, p.name, slot, dev, n_in, n_out, why)) {
                    err = ad.path + ": " + p.name + ": " + why;
                    return false;
                }
                const int rank = (int) p.a->shape[1], rp = lora_rank_pad(rank);
                if ((int64_t) p.a->shape[0] != n_in || (int64_t) p.b->shape[1] != n_out) {
                    err = ad.path + ": " + p.name + " is " + std::to_string(n_in) + " x " + std::to_string(n_out) +
                          " in the model, its pair " + std::to_string(p.a->shape[0]) + " x " +
                          std::to_string(p.b->shape[1]) +
                          " (an adapter of another model?)";
                    return false;
                }
                if (n_in % 8 != 0 || rp > q36lora::kMaxR) {
                    err = ad.path + ": " + p.name + ": rank " + std::to_string(rank) + " over " + std::to_string(n_in) +
                          " inputs (ranks up to " + std::to_string(q36lora::kMaxR) + ", inputs a multiple of 8)";
                    return false;
                }
                if (!tensor_floats(*p.a, f.tensor_data(*p.a), fa) || !tensor_floats(*p.b, f.tensor_data(*p.b), fb)) {
                    err = ad.path + ": " + p.name + ": an unreadable pair";
                    return false;
                }
                // A [rank][n_in] -> [rp][n_in] (zero rows); B [n_out][rank] -> [n_out][rp] (zero columns)
                std::vector<float> pa((size_t) rp * n_in, 0.f), pb((size_t) n_out * rp, 0.f);
                std::memcpy(pa.data(), fa.data(), (size_t) rank * n_in * 4);
                for (int j = 0; j < n_out; ++j) std::memcpy(&pb[(size_t) j * rp], &fb[(size_t) j * rank], (size_t) rank * 4);
                LoraSeg sg;
                sg.adapter = (int) ai;
                sg.rank = rp;
                sg.rank_true = (float) rank;
                sg.f32 = lora_f32(p);
                CK(cudaSetDevice(dev));
                sg.a = lora_upload(pa, sg.f32);
                sg.b = lora_upload(pb, sg.f32);
                ad.bytes += (pa.size() + pb.size()) * (sg.f32 ? 4 : 2);
                LoraMat*& lm = made[slot];
                if (lm == nullptr) {
                    m.lora_mats.emplace_back();
                    lm = &m.lora_mats.back();
                    *slot = lm;
                }
                lm->segs.push_back(sg);
                ad.rank_min = ad.pairs ? std::min(ad.rank_min, rank) : rank;
                ad.rank_max = std::max(ad.rank_max, rank);
                ++ad.pairs;
            }
            std::fprintf(stderr, "lamina: LoRA adapter %zu: %s: %d pairs, rank %d%s%s, alpha %g, %.1f MiB on the GPUs in "
                         "%.1f s\n", ai, ad.path.c_str(), ad.pairs, ad.rank_min, ad.rank_max != ad.rank_min ? " to " : "",
                         ad.rank_max != ad.rank_min ? std::to_string(ad.rank_max).c_str() : "", (double) ad.alpha,
                         (double) ad.bytes / 1048576.0, ms_since(t0) / 1000.0);
            m.loras.push_back(std::move(ad));
        } catch (const std::exception& e) {
            err = specs[ai].path + ": " + e.what();
            return false;
        }
    }
    CK(cudaSetDevice(m.stage_dev.front()));
    return true;
}

bool load(const std::string& path, Model& m, std::string& err, bool want_mtp, const std::vector<int>& devs,
          const std::vector<int>& split, int ctx, int chunk, const std::vector<LoraSpec>& lora_specs) {
    const auto t0 = Clock::now();
    m.file = std::make_shared<lamina::GgufModel>(lamina::GgufModel::open(path));
    const lamina::GgufModel& gm = *m.file;
    if (!lamina::core::geometry_from_gguf(gm.meta(), m.g, m.top_k, err)) return false;
    if (m.g.family != lamina::ModelFamily::Qwen35Moe && m.g.family != lamina::ModelFamily::Qwen35) {
        err = "this path runs Qwen3.6-35B-A3B (qwen35moe) and Qwen3.8-27B (qwen35) files";
        return false;
    }
    m.moe = m.g.family == lamina::ModelFamily::Qwen35Moe;
    size_t checked = 0;
    if (!lamina::core::check_gguf_layout(gm, m.g, err, &checked)) return false;
    const auto& g = m.g;
    if (g.head_dim != 256 || g.ssm_state_size != 128 || g.ssm_d_conv != 4 || (m.moe && (g.n_expert > 512 || m.top_k > 15)) ||
        g.n_head_kv < 1 || g.n_head % g.n_head_kv != 0 || g.n_head / g.n_head_kv > 8) {
        err = "a geometry this driver's kernels do not take (head 256, state 128, conv 4, <= 512 experts, top-k <= 15, "
              "<= 8 query heads a KV head)";
        return false;
    }
    const std::string p = std::string(lamina::model_family_arch(m.g.family)) + ".";
    if (const lamina::MetaValue* v = gm.meta().get(p + "attention.layer_norm_rms_epsilon")) m.eps = (float) v->num();
    if (const lamina::MetaValue* v = gm.meta().get(p + "rope.freq_base")) m.rope_base = (float) v->num();
    if (const lamina::MetaValue* v = gm.meta().get(p + "rope.dimension_count")) m.n_rot = (int) v->u;
    std::vector<uint64_t> lora_cost;
    if (!lora_scan(lora_specs, m.g.n_layers, lora_cost, err)) return false;
    if (!plan_stages(m, gm, devs, split, ctx, chunk, lora_cost, err)) return false;
    uint64_t bytes = 0;
    auto find = [&](const std::string& name, size_t& s) -> const lamina::TensorInfo* {
        const lamina::TensorInfo* t = gm.find(name, &s);
        if (!t) {
            std::fprintf(stderr, "lamina: missing %s\n", name.c_str());
            throw Fatal{1};
        }
        return t;
    };
    auto f32 = [&](const std::string& name) -> const float* {
        size_t s = 0;
        const lamina::TensorInfo* t = find(name, s);
        bytes += t->elements() * 4;
        return upload_f32(*t, gm.shard(s).tensor_data(*t));
    };
    auto mat = [&](const std::string& name) -> Mat {
        size_t s = 0;
        const lamina::TensorInfo* t = find(name, s);
        Mat w;
        w.type = (int) t->type;
        w.n_in = (int) t->shape[0];
        w.n_out = (int) t->shape[1];
        if (!K::native_mmvq_supported(w.type)) {
            std::fprintf(stderr, "lamina: %s is %s, which the native MMVQ does not take\n", name.c_str(), t->type_name());
            throw Fatal{1};
        }
        const uint64_t nb = lamina::tensor_payload_bytes(*t);
        uint8_t* d = nullptr;
        CK(dmalloc(&d, nb));
        CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
        bytes += nb;
        w.d = d;
        return w;
    };
    auto L = [](int64_t l, const char* s) { return "blk." + std::to_string(l) + "." + s; };
    m.layers.resize((size_t) g.n_layers);
    // the MTP block whenever the file has one, with or without drafts (--drafts-per-step): the VRAM it takes is then
    // the same either way, so the hybrid mode's slot tables - and with them the greedy text - do not depend on drafts
    (void) want_mtp;
    m.has_mtp = g.n_mtp_layers > 0;
    for (int64_t l = 0; l < g.n_layers + (m.has_mtp ? 1 : 0); ++l) {
        const bool is_mtp = l == g.n_layers;
        CK(cudaSetDevice(m.dev_of(l)));   // a layer's tensors on its stage's GPU
        Layer& y = is_mtp ? m.mtp : m.layers[(size_t) l];
        y.attn = is_mtp || lamina::core::is_qsa_layer(g, l);
        y.attn_norm = f32(L(l, "attn_norm.weight"));
        y.post_norm = f32(L(l, "post_attention_norm.weight"));
        if (!y.attn) {
            y.qkv = mat(L(l, "attn_qkv.weight"));
            y.zg = mat(L(l, "attn_gate.weight"));
            y.ssm_out = mat(L(l, "ssm_out.weight"));
            y.alpha = f32(L(l, "ssm_alpha.weight"));
            y.beta = f32(L(l, "ssm_beta.weight"));
            y.dt = f32(L(l, "ssm_dt.bias"));
            y.ssm_a = f32(L(l, "ssm_a"));
            y.conv = f32(L(l, "ssm_conv1d.weight"));
            y.ssm_norm = f32(L(l, "ssm_norm.weight"));
        } else {
            y.wq = mat(L(l, "attn_q.weight"));
            y.wk = mat(L(l, "attn_k.weight"));
            y.wv = mat(L(l, "attn_v.weight"));
            y.wo = mat(L(l, "attn_output.weight"));
            y.q_norm = f32(L(l, "attn_q_norm.weight"));
            y.k_norm = f32(L(l, "attn_k_norm.weight"));
        }
        if (!m.moe) {   // a dense FFN
            y.ffn_gate = mat(L(l, "ffn_gate.weight"));
            y.ffn_up = mat(L(l, "ffn_up.weight"));
            y.ffn_down = mat(L(l, "ffn_down.weight"));
            continue;
        }
        y.router = f32(L(l, "ffn_gate_inp.weight"));
        y.sh_gate = mat(L(l, "ffn_gate_shexp.weight"));
        y.sh_up = mat(L(l, "ffn_up_shexp.weight"));
        y.sh_down = mat(L(l, "ffn_down_shexp.weight"));
        y.sh_gate_inp = f32(L(l, "ffn_gate_inp_shexp.weight"));
        CK(dmalloc(&y.router_ext, (size_t) (g.n_expert + 1) * g.n_embd * 4));
        CK(cudaMemcpy(y.router_ext, y.router, (size_t) g.n_expert * g.n_embd * 4, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(y.router_ext + (size_t) g.n_expert * g.n_embd, y.sh_gate_inp, (size_t) g.n_embd * 4,
                      cudaMemcpyDeviceToDevice));
        bytes += (size_t) (g.n_expert + 1) * g.n_embd * 4;
        // the routed experts: [gate | up | down] per expert, the native expert layout
        size_t sg = 0, su = 0, sd = 0;
        const lamina::TensorInfo* tg = find(L(l, "ffn_gate_exps.weight"), sg);
        const lamina::TensorInfo* tu = find(L(l, "ffn_up_exps.weight"), su);
        const lamina::TensorInfo* td = find(L(l, "ffn_down_exps.weight"), sd);
        if (tg->type != tu->type) {
            err = L(l, "ffn_gate_exps / ffn_up_exps differ in type");
            return false;
        }
        if (!K::native_expert_supported((int) tg->type, (int) td->type, g.n_embd, g.n_ff)) {
            err = L(l, "experts are ") + tg->type_name() + "/" + td->type_name() + ", which the native expert kernels do not take";
            return false;
        }
        y.L = K::native_expert_layout((int) tg->type, (int) td->type, g.n_embd, g.n_ff);
        const size_t gsz = (size_t) g.n_ff * y.L.gu_row, dsz = (size_t) g.n_embd * y.L.d_row;
        // a stride in whole blocks of both parts' types (MMQ steps over the experts by blocks), 16-byte aligned
        const size_t al = lcm_sz(lcm_sz(block_bytes(y.L.gu_type), block_bytes(y.L.d_type)), 16);
        y.stride = (y.L.bytes + al - 1) / al * al;
        y.src_g = gm.shard(sg).tensor_data(*tg);
        y.src_u = gm.shard(su).tensor_data(*tu);
        y.src_d = gm.shard(sd).tensor_data(*td);
        y.src_gsz = gsz;
        y.src_dsz = dsz;
    }
    if (m.has_mtp) {
        const int64_t l = g.n_layers;
        m.eh_proj = mat(L(l, "nextn.eh_proj.weight"));
        m.mtp_enorm = f32(L(l, "nextn.enorm.weight"));
        m.mtp_hnorm = f32(L(l, "nextn.hnorm.weight"));
        m.mtp_shnorm = f32(L(l, "nextn.shared_head_norm.weight"));
    }
    CK(cudaSetDevice(m.stage_dev.back()));
    m.out_norm = f32("output_norm.weight");
    m.head = mat("output.weight");
    CK(cudaSetDevice(m.stage_dev.front()));
    {
        size_t s = 0;
        const lamina::TensorInfo* t = find("token_embd.weight", s);
        m.embd_type = (int) t->type;
        m.embd_rows = (int64_t) t->shape[1];
        if (!K::iq_supported(m.embd_type) || t->shape[0] % 256 != 0) {   // iq_embed_rows' types (it ends the process)
            std::fprintf(stderr, "lamina: token_embd.weight is %s, which the GPU embedding does not take\n", t->type_name());
            throw Fatal{1};
        }
        m.embd_row = row_bytes(m.embd_type, (int64_t) t->shape[0]);
        const uint64_t nb = lamina::tensor_payload_bytes(*t);
        uint8_t* d = nullptr;
        CK(dmalloc(&d, nb));
        CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
        m.embd = m.embd_mtp = d;
        bytes += nb;
        if (m.has_mtp && m.stage_dev.size() > 1) {   // the MTP block embeds its tokens on the last stage's GPU: a copy there
            CK(cudaSetDevice(m.stage_dev.back()));
            CK(dmalloc(&d, nb));
            CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
            m.embd_mtp = d;
            bytes += nb;
            CK(cudaSetDevice(m.stage_dev.front()));
        }
    }
    std::fprintf(stderr, "lamina: %s: %zu tensors (layout checked), %.2f GiB on the GPU in %.1f s (the routed "
                 "experts are placed after the engine's buffers)\n", path.c_str(), checked, (double) bytes / (1 << 30),
                 ms_since(t0) / 1000.0);
    return load_loras(m, lora_specs, err);
}

/// Expert e of layer L as its native blob [gate | up | down] (L.L.bytes) at dst (host memory).
void expert_blob(const Layer& L, int64_t e, uint8_t* dst) {
    std::memcpy(dst, L.src_g + (size_t) e * L.src_gsz, L.src_gsz);
    std::memcpy(dst + L.L.up_off, L.src_u + (size_t) e * L.src_gsz, L.src_gsz);
    std::memcpy(dst + L.L.down_off, L.src_d + (size_t) e * L.src_dsz, L.src_dsz);
}

// ------------------------------------------------------------------------------------------------ the engine
float* dalloc(size_t n) {
    float* p = nullptr;
    CK(dmalloc(&p, n * 4));
    CK(cudaMemset(p, 0, n * 4));
    return p;
}
template <class T> T* dalloc_t(size_t n) {
    T* p = nullptr;
    CK(dmalloc(&p, n * sizeof(T)));
    CK(cudaMemset(p, 0, n * sizeof(T)));
    return p;
}

/// A picture in a prompt: its first cell, its rows, and the hash the conversation cache compares it by.
struct ImgKey {
    int64_t cell = 0, n = 0;
    uint64_t hash = 0;
    bool operator==(const ImgKey& o) const { return cell == o.cell && n == o.n && hash == o.hash; }
};

struct Engine {
    const Model* m = nullptr;
    cudaStream_t st = nullptr;
    int ctx = 32768, tmax = 512;
    int64_t pos = 0;   // cells holding a processed token
    // activations [tmax][...]
    float *x, *h, *y, *qkv, *hc, *z, *alpha, *beta, *gdn_y, *qg, *kx, *vx, *q, *gate, *att, *logits_r, *eparts, *moe_out,
        *sg, *su, *shh, *sy, *sgate, *ewts, *logits;
    float* hmain = nullptr;   // [tmax][N] the main model's final normed hidden of each row (the MTP seeds)
    float* hlast = nullptr;   // [N] that of the last processed position (zeros before the first): the current session's
    void *q8, *xq, *escr;
    // a window's q8_1 rows of xq and q8 interleaved (native_mmvq_il, 2-8 rows), each flag whether its copy is the
    // buffer's rows; il_mode (LAMINA_Q36_IL): 0 native_mmvq, 1 the interleaved kernels (bitwise native_mmvq), 2 and
    // the tensor-core ones where their table takes the matrix (to float rounding); il_quant: quant() writes the copy
    // with the rows (the model has K-quant projections, which take it from 2 or 3 rows), else it is made on first use
    void *xq_il = nullptr, *q8_il = nullptr;
    bool xq_il_ok = false, q8_il_ok = false, il_quant = false;
    const int il_mode = !std::getenv("LAMINA_Q36_IL") ? 2 : std::string(std::getenv("LAMINA_Q36_IL")) == "exact" ? 1
                        : std::atoi(std::getenv("LAMINA_Q36_IL")) ? 2 : 0;
    int32_t *ids, *grp_start, *n_groups, *ent_dst, *ent_tok, *tok, *pos0, *pick;
    int32_t* hpin = nullptr;   // pinned staging: [0] decode token, [1] position, [2..9] picks, [10] max rows, [16..] tokens
    unsigned long long* grp_ptr;
    float *pacc, *pm, *pl, *amv;
    int32_t* ami;
    int n_chunk = 0, att_rows = 0;   // the decode attention's partials: att_rows query rows x n_chunk chunks
    std::vector<float*> conv, S;     // per layer: the CURRENT session's (select)
    std::vector<__half*> kc, vc;     // per layer, FP16; the MTP block's K / V at index n_layers
    // verify windows (T <= kVerify): the GDN inputs of every row per layer, kept for the commit
    static constexpr int kVerify = 8;
    std::vector<float*> v_qkv, v_hc, v_alpha, v_beta;
    // the MTP block (when loaded)
    float *mtp_cat = nullptr, *hmtp = nullptr, *logits_m = nullptr;
    int32_t *tok_m = nullptr, *pos0_m = nullptr, *pick_m = nullptr;
    // graphs: [0] the decode step, [T] a verify window of T rows, mtp1 the MTP draft step
    // (the graphs hold the current session's state pointers as kernel arguments: one set per session, and one more for
    // the passes that apply an adapter, whose LoRA kernels read the rows' scales from the device)
    struct GSet {
        cudaGraphExec_t g1 = nullptr, gv[kVerify + 1] = {}, gm1 = nullptr;
    };
    std::vector<GSet> gsets;
    GSet& gs() { return gsets[(size_t) cur * 2 + (lora_live ? 1 : 0)]; }
    // ---- LoRA adapters (Model::loras): the pass's rows' scales on this stage's GPU, the A stage's partials and t
    int n_ad = 0;
    bool lora_live = false;               // a row of the pass uses an adapter (the LoRA graphs, lora_site runs)
    float* d_lscale = nullptr;            // [kVerify][n_ad] each row's scale of each adapter
    std::vector<float> h_lscale;          // what d_lscale holds
    float *l_part = nullptr, *l_t = nullptr, *l_dab = nullptr;   // the A stage's partials, a prompt GEMM's t, the GDN deltas
    std::vector<float> lora_default;      // stage 0: a request's scales when it names none (--lora's, --lora-scaled's)
    lamina::prefill::Gemm gemm;
    cublasHandle_t blas = nullptr;
    static constexpr int kSplitK = 16;
    float* sk_part = nullptr;          // the split-K partials of the F32 projections: [kSplitK][tmax][the widest]
    // the prompt's MMQ experts
    bool use_mmq = false;
    std::unique_ptr<MMQ::Context> mctx;
    int32_t *rcnt = nullptr, *rcur = nullptr, *rbounds = nullptr, *rmax = nullptr, *ident = nullptr, *rinv = nullptr;
    int64_t ident_rows = 0;   // ident's entries: a chunk's rows and what a launch reads past them
    void *mq_x = nullptr, *mq_hq = nullptr;
    // a prompt chunk's projection input (prep), converted on first use (apply): to q8_1 for MMQ (mq_dx, for the weight
    // type pq; -1: not yet) or to FP16 for the GEMMs (x16, p16)
    const float* px = nullptr;
    int pn = 0, pq = -1;
    bool p16 = false;
    void* mq_dx = nullptr;
    float *mq_gu = nullptr, *mq_h = nullptr;
    __half* x16 = nullptr;
    bool use_graph = true;
    // the layer split (--second-gpu): this stage runs layers [l0, l1) on GPU `dev`; nxt is the next stage (null: this one
    // ends the model - the head, the MTP block, the logits), lst the last; stage 0 drives them all
    int dev = 0;
    size_t l0 = 0, l1 = 0;
    Engine* nxt = nullptr;
    Engine* lst = this;
    bool split = false;
    cudaEvent_t ev_hand = nullptr;   // this stage's layers done: the next stage copies x
    // a later stage's two inputs of prompt chunks (pass_chunk): the stage before fills one while this one runs the
    // other; the stage before's ev_out: filled, this one's ev_free: copied into x (each recorded on its own GPU)
    float* x_in[2] = {nullptr, nullptr};
    cudaEvent_t ev_out[2] = {nullptr, nullptr}, ev_free[2] = {nullptr, nullptr};
    // The conversation cache's checkpoints leave through an image of the recurrent state on this stage's GPU (`snap`,
    // state_bytes_own: each GDN layer's conv history and state, then hlast): written as a chunk passes row snap_row
    // (layer(): the state after the chunk's first snap_row rows) or from the state a chunk leaves (snap_state), then
    // copied to the cache's pinned buffer on a stream of its own (st_snap), so neither the prompt's next chunk nor the
    // first decode steps wait for the copy.  The image is free again when that copy has landed (ev_landed).  A
    // prompt's end has an image of its own (snap_end): its checkpoint comes right behind the turn's, whose copy is
    // then on its way.
    // A buffer of the cache's that is not pinned takes the images through `snap_via`, a pinned one of the engine's
    // (stage 0's, state_bytes; [1]: the prompt's end): a copy to pageable memory would wait for the stage's stream
    // on the host.  snap_flush copies it over once the request has answered (or before the next image needs it).
    uint8_t* snap = nullptr;
    uint8_t* snap_end = nullptr;
    uint8_t* snap_via[2] = {nullptr, nullptr};   // stage 0's
    uint8_t* snap_to[2] = {nullptr, nullptr};    // the cache's buffer snap_via's contents belong in (null: none waits)
    std::vector<size_t> snap_at;   // per layer: where its part of the image starts (the GDN layers')
    int snap_row = 0;              // the chunk being queued takes a checkpoint after this many of its rows (0: none)
    cudaStream_t st_snap = nullptr;
    cudaEvent_t ev_snap = nullptr, ev_landed = nullptr, ev_landed_end = nullptr;
    void on() const {
        if (split) CK(cudaSetDevice(dev));
    }
    template <class F> void for_each(F f) {   // every stage on its GPU, then this one's GPU again
        for (Engine* e = this; e != nullptr; e = e->nxt) {
            e->on();
            f(*e);
        }
        on();
    }
    void sync_all() {
        for_each([](Engine& e) { CK(cudaStreamSynchronize(e.st)); });
    }
    /// every stage's position on its device (pos0)
    void put_pos(int64_t p) {
        for_each([&](Engine& e) {
            e.hpin[1] = (int32_t) p;
            CK(cudaMemcpyAsync(e.pos0, e.hpin + 1, 4, cudaMemcpyHostToDevice, e.st));
        });
    }
    void advance(int64_t n) {
        for (Engine* e = this; e != nullptr; e = e->nxt) e->pos += n;
    }
    /// a picture's rows replace embeddings: the main model's on the first stage, the MTP block's on the last
    bool takes_rows() const { return l0 == 0 || (lst == this && m->has_mtp); }
    int mtp_d = 0;   // MTP drafts per speculative step (0: none; greedy requests only)

    // ---- sessions: the conversation states.  One for a single-request engine; --batch N keeps N, each with its own
    // recurrent state and KV cache, so N requests run at once: their rows share one decode window (forward_rows).  The
    // members conv / S / kc / vc / pos / hlast above are the CURRENT session's (select() swaps them).
    struct Sess {
        std::vector<float*> conv, S;
        std::vector<__half*> kc, vc;
        float* hlast = nullptr;
        int64_t pos = 0;
        int32_t rdelta = 0;                 // rotary offset of the text after the last picture (see k_attn_prep)
        std::vector<int32_t> live;          // the tokens whose state it holds (cell i = live[i])
        std::vector<ImgKey> imgs;           // the pictures among them
        std::vector<float> lora;            // the adapters' scales its state was read with (its request's; stage 0's)
        bool active = false;                // a batch slot is decoding in it
        uint64_t used = 0;                  // when it was last used (the LRU pick)
    };
    std::vector<Sess> sess;
    int n_sess = 1, cur = 0;
    uint64_t use_clock = 0;
    // device tables for batch rows: per state index (layer, then the MTP block) the session pointers
    float **d_conv_tab = nullptr, **d_S_tab = nullptr;
    __half **d_kc_tab = nullptr, **d_vc_tab = nullptr;
    int32_t *d_rcell = nullptr, *d_rslot = nullptr, *d_rdelta = nullptr;   // [kVerify] the rows' cell / session / rotary offset
    // the GDN layers run a slot's rows in sequence: per group (a slot) its session, its first row and its row count
    int32_t *d_gslot = nullptr, *d_grow0 = nullptr, *d_gcnt = nullptr;
    int rows_G = 0;                                                        // groups of the rows being run
    int32_t* hrow = nullptr;                                                // pinned staging for the rows
    bool rows_mode = false;                                                 // layer(): the rows are sessions' (not consecutive cells)
    bool bmtp = false;                                                      // --batch-mtp: batch windows verify MTP proposals
    int64_t rows_pmax = 0;                                                  // (eager) the highest cell of the rows
    // pictures (--vision): the request's rows replace the embeddings of its <|image_pad|> cells; M-RoPE
    bool vision = false;
    int32_t* d_mtab = nullptr;          // [ctx][3] t, h, w of the current request's prompt cells (vision)
    const int32_t* mtab_now = nullptr;  // set for a forward over prompt cells that need it, else null
    int32_t* d_delta = nullptr;         // device scalar: the current session's rotary offset (decode graphs read it)
    float* d_img = nullptr;             // the request's picture rows [n][N]
    size_t img_cap = 0;                 // floats
    int32_t* d_map = nullptr;           // [tmax] a chunk's row -> picture row (or -1)
    int32_t* h_map = nullptr;           // pinned
    bool map_on = false;                // this forward's embeddings take picture rows
    std::vector<int32_t> img_map;       // the request's cell -> picture row (-1: text)
    int64_t img_prompt_cells = 0;       // the request's prompt length: cells below it use the table

    // ---- the hybrid mode (Model::hybrid): per state index (the layers, then the MTP block) the slot tables; a
    // batch of up to hybrid_t rows runs its VRAM-resident experts on the GPU and the others on the CPU pool, side by
    // side; a bigger prompt chunk streams each layer's experts into a staging buffer for MMQ (two, double-buffered)
    struct Hy {
        std::vector<int> slot_of, exp_in_slot;
        std::vector<double> score;   // routing counts, halved at each rebalance
        int32_t* slot_of_d = nullptr;
#if defined(LAMINA_Q36_CPU_POOL)
        lamina::kernels::cpu::NativeFmt fmt;
#endif
    };
    std::vector<Hy> hy;
    int hybrid_t = 32;
    bool main_hybrid = false;   // some main layer is hybrid (else only the MTP block is: the decode graph stays)
    bool mtp_graph = true;      // the MTP draft step may be a graph (not when its layer is hybrid)
    /// state index li's layer runs hybrid (its experts not all in VRAM)
    bool lhy(size_t li) const {
        if (!m->hybrid) return false;
        const Layer& L = li < m->layers.size() ? m->layers[li] : m->mtp;
        return L.n_slots < (int) m->g.n_expert;
    }
#if defined(LAMINA_Q36_CPU_POOL)
    std::unique_ptr<lamina::kernels::cpu::ExpertPool> pool;
    std::unique_ptr<lamina::kernels::cpu::HostPin> host_pin;   // the host loop on the core the pool's workers leave
    std::vector<lamina::kernels::cpu::ExpertJobMulti> jobs;
#endif
    float *h_h = nullptr, *cpu_out = nullptr, *ecpu = nullptr;
    int32_t *h_ids = nullptr, *h_miss = nullptr, *d_miss = nullptr, *h_cnt = nullptr;
    uint8_t* acts = nullptr;   // per token: its quantized activation (kNativeActBytes)
    cudaEvent_t ev_route = nullptr;
    uint8_t* stage[2] = {};   // a streamed prompt's staging buffers: one block (stage_bytes), the second half way in
    size_t stage_bytes = 0;
    /// A batch of T rows streams its hybrid layers' experts through the staging buffers for MMQ: a prompt chunk past
    /// hybrid_t rows.  Every other batch - a decode step, a verify window, the batch slots' window (kVerify rows at
    /// most) - runs them from the VRAM slots and the CPU pool and leaves the staging buffers alone, which is what
    /// lets them be lent while requests decode (Serve::lend).
    bool streamed(int T) const {
        static_assert(kVerify <= kGemmT, "a decode window's rows never stream");
        return main_hybrid && use_mmq && T > hybrid_t && T > kGemmT;
    }
    cudaStream_t cst = nullptr;
    cudaEvent_t ev_copy[2] = {}, ev_used[2] = {};
    std::vector<int> stream_seq;   // the state indices a streamed forward visits, in order
    int stream_pos = 0;
    long long hy_cpu = 0, hy_gpu = 0, hy_swaps = 0;   // entries computed on the CPU / GPU, slot replacements
    double hy_cpu_ms = 0;
    // decode / verify rows: their routing is counted once the rows are kept (commit), so a verify window's rejected
    // drafts never move the slots; the slots follow the counts at every kRebalanceEvery-th position exactly (a
    // window never steps over one), so plain and speculative decoding see the same slot tables at every position
    static constexpr int kRebalanceEvery = 32;
    bool rows_deferred = false;          // set while a decode step or a verify window runs
    std::vector<std::vector<int32_t>> row_ids;   // [main layer][row * top_k + k] of the last such call
    void count_rows(int n) {
        const int Kp = (int) m->top_k;
        for (size_t li = 0; li < row_ids.size() && li < hy.size(); ++li)
            if (!row_ids[li].empty())
                for (int e = 0; e < n * Kp; ++e) hy[li].score[(size_t) row_ids[li][(size_t) e]] += 1.0;
    }
    void after_rows(int n) {   // a decode step / verify window kept n rows: count them, rebalance at the boundary
        if (!m->hybrid) return;
        count_rows(n);
        if (pos % kRebalanceEvery == 0) {
            CK(cudaStreamSynchronize(st));
            rebalance(8);
        }
    }

    void alloc_session(Sess& ss) const {
        const auto& g = m->g;
        const size_t C = (size_t) g.ssm_conv_channels;
        for (size_t l = 0; l < m->layers.size(); ++l) {
            const Layer& L = m->layers[l];
            const bool here = l >= l0 && l < l1;   // another stage's layer: nothing here
            if (L.attn) {
                ss.conv.push_back(nullptr);
                ss.S.push_back(nullptr);
                ss.kc.push_back(here ? dalloc_t<__half>((size_t) g.n_head_kv * ctx * m->kv_row() / 2) : nullptr);
                ss.vc.push_back(here ? dalloc_t<__half>((size_t) g.n_head_kv * ctx * m->kv_row() / 2) : nullptr);
            } else {
                ss.conv.push_back(here ? dalloc(C * (g.ssm_d_conv - 1)) : nullptr);
                ss.S.push_back(here ? dalloc((size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size) : nullptr);
                ss.kc.push_back(nullptr);
                ss.vc.push_back(nullptr);
            }
        }
        if (m->has_mtp) {   // the MTP block's own K / V (an attention layer at index n_layers), on the last stage
            const bool here = lst == this;
            ss.conv.push_back(nullptr);
            ss.S.push_back(nullptr);
            ss.kc.push_back(here ? dalloc_t<__half>((size_t) g.n_head_kv * ctx * m->kv_row() / 2) : nullptr);
            ss.vc.push_back(here ? dalloc_t<__half>((size_t) g.n_head_kv * ctx * m->kv_row() / 2) : nullptr);
        }
        ss.hlast = dalloc((size_t) g.n_embd);
    }

    /// The current session: `i`'s state, cache and position become the engine's (conv / S / kc / vc / pos / hlast).
    void select(int i) {
        for_each([&](Engine& e) { e.select_own(i); });
    }
    void select_own(int i) {
        if (i == cur) return;
        sess[(size_t) cur].pos = pos;
        cur = i;
        load_cur();
    }
    void load_cur() {
        Sess& ss = sess[(size_t) cur];
        conv = ss.conv; S = ss.S; kc = ss.kc; vc = ss.vc;
        hlast = ss.hlast;
        pos = ss.pos;
        ss.used = ++use_clock;
        set_delta_own(ss.rdelta);
    }
    /// The rotary offset the decode graphs read (a device scalar), for the current session.
    void set_delta_own(int32_t d) {
        sess[(size_t) cur].rdelta = d;
        if (!d_delta) return;
        hpin[60] = d;
        CK(cudaMemcpyAsync(d_delta, hpin + 60, 4, cudaMemcpyHostToDevice, st));
        CK(cudaStreamSynchronize(st));   // (the pinned word is reused)
    }

    void init(const Model& model, int ctx_, int tmax_, int n_sessions = 1, bool vision_ = false, bool snapshots = false) {
        m = &model;
        ctx = ctx_;
        n_sess = std::max(1, std::min(n_sessions, kVerify));
        vision = vision_;
        tmax = std::max(tmax_, kVerify);
        on();
        const auto& g = m->g;
        const size_t T = (size_t) tmax, N = (size_t) g.n_embd, Kp = (size_t) std::max<int64_t>(1, m->top_k),
                     FF = (size_t) g.n_ff;
        const int64_t FW = m->moe ? g.n_ff_shexp : g.n_ff;   // the shared expert's width, or the dense FFN's
        const size_t C = (size_t) g.ssm_conv_channels;
        CK(stream_new(&st, cudaStreamNonBlocking));
        x = dalloc(T * N); h = dalloc(T * N); y = dalloc(T * N); hmain = dalloc(T * N);
        qkv = dalloc(T * C); hc = dalloc(T * C); z = dalloc(T * g.ssm_value_dim);
        alpha = dalloc(T * g.ssm_v_heads); beta = dalloc(T * g.ssm_v_heads); gdn_y = dalloc(T * g.ssm_value_dim);
        qg = dalloc(T * 2 * g.n_head * g.head_dim); kx = dalloc(T * g.n_head_kv * g.head_dim);
        vx = dalloc(T * g.n_head_kv * g.head_dim); q = dalloc(T * g.n_head * g.head_dim);
        gate = dalloc(T * g.n_head * g.head_dim); att = dalloc(T * g.n_head * g.head_dim);
        logits_r = dalloc(T * (g.n_expert + 1));
        eparts = dalloc(T * Kp * N); moe_out = dalloc(T * N);
        sg = dalloc(T * FW); su = dalloc(T * FW); shh = dalloc(T * FW); sy = dalloc(T * N);
        sgate = dalloc(T); ewts = dalloc(T * Kp); logits = dalloc((size_t) kVerify * g.n_vocab);
        const int max_in = (int) std::max<int64_t>({2 * g.n_embd, g.ssm_value_dim, g.n_head * g.head_dim, FW});
        CK(dmalloc(&q8, K::native_q8_1_bytes(max_in, 1) * T));
        CK(dmalloc(&xq, K::native_q8_1_bytes((int) N, 1) * T));
        if (il_mode) {
            CK(dmalloc(&q8_il, K::native_q8_1_il_bytes(max_in, 8)));
            CK(dmalloc(&xq_il, K::native_q8_1_il_bytes((int) N, 8)));
            for (const Layer& L : m->layers)
                for (const Mat* w : {&L.qkv, &L.zg, &L.ssm_out, &L.wq, &L.wk, &L.wv, &L.wo, &L.sh_gate, &L.sh_up, &L.sh_down,
                                     &L.ffn_gate, &L.ffn_up, &L.ffn_down})
                    il_quant = il_quant || (w->d != nullptr && w->type != 8 && il_pays(w->type, 3));
        }
        x16 = dalloc_t<__half>(T * (size_t) max_in);
        if (blas_new(&blas) != CUBLAS_STATUS_SUCCESS || cublasSetStream(blas, st) != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "lamina: cuBLAS handle failed\n");
            throw Fatal{1};
        }
        // the F32 projections' widest output: the router with the shared expert's gate (n_expert + 1), or GDN alpha / beta
        sk_part = dalloc((size_t) kSplitK * T * (size_t) std::max<int64_t>(g.n_expert + 1, g.ssm_v_heads));
        const bool mmq_built = MMQ::built() && std::getenv("LAMINA_Q36_NO_MMQ") == nullptr;
        use_mmq = mmq_built && m->moe;   // the prompt's experts on MMQ
        auto mmq_ok = [&](const Layer& L) {
            return MMQ::supported(L.L.gu_type) && MMQ::supported(L.L.d_type) && MMQ::fits(L.L.gu_type, 2 * g.n_ff) &&
                   MMQ::fits(L.L.d_type, g.n_embd);
        };
        for (const Layer& L : m->layers) use_mmq = use_mmq && mmq_ok(L);
        if (m->has_mtp) use_mmq = use_mmq && mmq_ok(m->mtp);
        if (mmq_built) {   // the prompt's projections on MMQ (apply)
            mctx = std::make_unique<MMQ::Context>();
            CK(dmalloc(&mq_dx, MMQ::q8_bytes(T, max_in)));
        }
        if (use_mmq) {
            const int64_t ne = (int64_t) T * Kp;
            rcnt = dalloc_t<int32_t>((size_t) g.n_expert);
            rcur = dalloc_t<int32_t>((size_t) g.n_expert);
            rbounds = dalloc_t<int32_t>((size_t) g.n_expert + 1);
            rmax = dalloc_t<int32_t>(1);
            ident_rows = MMQ::ids_table_rows(ne);
            ident = dalloc_t<int32_t>((size_t) ident_rows);
            rinv = dalloc_t<int32_t>((size_t) ne);
            CK(cudaDeviceSynchronize());   // ident's cudaMemset (legacy stream) before the iota on the non-blocking stream
            MMQ::iota(ident, ident_rows, st);
            CK(dmalloc(&mq_x, MMQ::q8_bytes(ne, g.n_embd)));
            CK(dmalloc(&mq_hq, MMQ::q8_bytes(ne, g.n_ff)));
            mq_gu = dalloc((size_t) ne * 2 * FF);
            mq_h = dalloc((size_t) ne * FF);
        }
        if (m->moe)
            std::fprintf(stderr, "lamina: prompt experts on %s\n", use_mmq ? "MMQ (int8)" : "the native grouped kernels");
        {
            std::string gerr;
            int64_t big = 0;
            auto take = [&](const Layer& L) {
                for (const Mat* w : {&L.qkv, &L.zg, &L.ssm_out, &L.wq, &L.wk, &L.wv, &L.wo, &L.sh_gate, &L.sh_up, &L.sh_down})
                    big = std::max<int64_t>(big, (int64_t) w->n_in * w->n_out);
            };
            for (const Layer& L : m->layers) take(L);
            if (m->has_mtp) {
                take(m->mtp);
                big = std::max<int64_t>(big, (int64_t) m->eh_proj.n_in * m->eh_proj.n_out);
            }
            if (!gemm.init(st, gerr)) {
                std::fprintf(stderr, "lamina: prompt GEMM: %s\n", gerr.c_str());
                throw Fatal{1};
            }
            constexpr size_t kGemmWs = 32u << 20;   // the cuBLAS workspace, as the Flash-Next prompt path's
            uint16_t* gscr = dalloc_t<uint16_t>((size_t) big);
            void* gws = nullptr;
            CK(dmalloc(&gws, kGemmWs));
            gemm.set_buffers(gscr, big, gws, kGemmWs);
        }
        CK(dmalloc(&escr, K::native_expert_scratch_bytes((int64_t) (T * Kp), (int64_t) FF)));
        ids = dalloc_t<int32_t>(T * Kp);
        grp_ptr = dalloc_t<unsigned long long>(std::min<size_t>(T * Kp, (size_t) g.n_expert) + 1);
        grp_start = dalloc_t<int32_t>(std::min<size_t>(T * Kp, (size_t) g.n_expert) + 1);
        n_groups = dalloc_t<int32_t>(1);
        ent_dst = dalloc_t<int32_t>(T * Kp);
        ent_tok = dalloc_t<int32_t>(T * Kp);
        tok = dalloc_t<int32_t>(T);
        pos0 = dalloc_t<int32_t>(1);
        pick = dalloc_t<int32_t>(kVerify);
        amv = dalloc((size_t) kArgBlocks * kVerify);
        ami = dalloc_t<int32_t>((size_t) kArgBlocks * kVerify);
        CK(hmalloc((void**) &hpin, 64 * sizeof(int32_t), cudaHostAllocDefault));
        n_chunk = (ctx + kChunkCells - 1) / kChunkCells;
        for (int r = 1; r <= kVerify; ++r)   // the decode attention's shared memory: past 48 KB from 3 rows
            CK(cudaFuncSetAttribute(kAttnDec[m->kv_q8][r - 1], cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int) attn_dec_smem((int) (g.n_head / g.n_head_kv), r)));
        CK(cudaFuncSetAttribute(kFaPrompt[m->kv_q8][g.n_head / g.n_head_kv - 1], cudaFuncAttributeMaxDynamicSharedMemorySize,
                                (int) sizeof(q36a::FaSmem)));
        att_rows = (int) (g.n_head * std::min<int64_t>((int64_t) T, kGemmT));   // the decode branch's rows
        pacc = dalloc((size_t) att_rows * n_chunk * g.head_dim);
        pm = dalloc((size_t) att_rows * n_chunk);
        pl = dalloc((size_t) att_rows * n_chunk);
        for (size_t l = 0; l < m->layers.size(); ++l) {   // the verify windows' GDN inputs (one set, whichever session runs)
            const Layer& L = m->layers[l];
            if (L.attn || l < l0 || l >= l1) {
                for (auto* v : {&v_qkv, &v_hc, &v_alpha, &v_beta}) v->push_back(nullptr);
            } else {
                v_qkv.push_back(dalloc((size_t) kVerify * C));
                v_hc.push_back(dalloc((size_t) kVerify * C));
                v_alpha.push_back(dalloc((size_t) kVerify * g.ssm_v_heads));
                v_beta.push_back(dalloc((size_t) kVerify * g.ssm_v_heads));
            }
        }
        if (m->has_mtp) {
            mtp_cat = dalloc(T * 2 * N);
            hmtp = dalloc(T * N);
            logits_m = dalloc((size_t) g.n_vocab);
            tok_m = dalloc_t<int32_t>(T);
            pos0_m = dalloc_t<int32_t>(1);
            pick_m = dalloc_t<int32_t>(1);
            p_draft = dalloc(1);
        }
        sess.resize((size_t) n_sess);
        gsets.resize((size_t) n_sess * 2);
        n_ad = (int) m->loras.size();
        if (n_ad > 0) {
            d_lscale = dalloc((size_t) kVerify * n_ad);
            h_lscale.assign((size_t) kVerify * n_ad, 0.f);
            const size_t ks_max = (size_t) (max_in + q36lora::kChunk - 1) / q36lora::kChunk;
            l_part = dalloc(ks_max * kVerify * q36lora::kMaxR);
            l_t = dalloc(T * q36lora::kMaxR);   // a prompt chunk's t (one pair a GEMM)
            l_dab = dalloc((size_t) kVerify * 2 * g.ssm_v_heads);
        }
        {   // the sessions' caches: said before they fail (a slot's K / V at the context, in the --kv format, for the
            // attention layers and the MTP block, and its recurrent state)
            size_t n_attn = m->has_mtp ? 1 : 0, n_gdn = 0;
            for (const Layer& L : m->layers) (L.attn ? n_attn : n_gdn) += 1;
            const size_t per = (size_t) g.n_head_kv * m->kv_row() * 2 * (size_t) ctx * n_attn +
                               (size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size * 4 * n_gdn;
            size_t fr0 = 0, tot0 = 0;
            CK(cudaMemGetInfo(&fr0, &tot0));
            if (n_sess > 1)
                std::fprintf(stderr, "lamina: %d sessions (--batch): %.2f GiB of VRAM each at a context of %d (%.2f GiB in "
                             "all), %.2f GiB free before them\n", n_sess, (double) per / (1 << 30), ctx,
                             (double) per * n_sess / (1 << 30), (double) fr0 / (1 << 30));
            if (n_sess > 1 && per * (size_t) n_sess + ((size_t) 1 << 30) > fr0) {
                std::fprintf(stderr, "lamina: not enough VRAM for %d slots at a context of %d (%.2f GiB each, %.2f GiB "
                             "free): use fewer slots (\"parallel\" / --batch) or a shorter --max-context\n", n_sess, ctx,
                             (double) per / (1 << 30), (double) fr0 / (1 << 30));
                throw Fatal{1};
            }
        }
        for (Sess& ss : sess) alloc_session(ss);
        if (n_sess > 1) {   // the batch rows' tables: per state index the session pointers, on the device
            const size_t ns = sess[0].conv.size();
            std::vector<float*> hcs(ns * (size_t) n_sess), hs(hcs);
            std::vector<__half*> hk(ns * (size_t) n_sess), hv(hk);
            for (size_t li = 0; li < ns; ++li)
                for (int s_ = 0; s_ < n_sess; ++s_) {
                    hcs[li * n_sess + s_] = sess[(size_t) s_].conv[li];
                    hs[li * n_sess + s_] = sess[(size_t) s_].S[li];
                    hk[li * n_sess + s_] = sess[(size_t) s_].kc[li];
                    hv[li * n_sess + s_] = sess[(size_t) s_].vc[li];
                }
            auto upload = [](void** dst, const void* src, size_t bytes) {
                CK(dmalloc(dst, bytes));
                CK(cudaMemcpy(*dst, src, bytes, cudaMemcpyHostToDevice));
            };
            upload((void**) &d_conv_tab, hcs.data(), hcs.size() * sizeof(float*));
            upload((void**) &d_S_tab, hs.data(), hs.size() * sizeof(float*));
            upload((void**) &d_kc_tab, hk.data(), hk.size() * sizeof(__half*));
            upload((void**) &d_vc_tab, hv.data(), hv.size() * sizeof(__half*));
            d_rcell = dalloc_t<int32_t>(kVerify);
            d_rslot = dalloc_t<int32_t>(kVerify);
            d_rdelta = dalloc_t<int32_t>(kVerify);
            d_gslot = dalloc_t<int32_t>(kVerify);
            d_grow0 = dalloc_t<int32_t>(kVerify);
            d_gcnt = dalloc_t<int32_t>(kVerify);
            CK(hmalloc((void**) &hrow, 128 * sizeof(int32_t), cudaHostAllocDefault));
        }
        if (vision) {
            d_mtab = dalloc_t<int32_t>((size_t) ctx * 3);
            d_map = dalloc_t<int32_t>(T);
            CK(hmalloc((void**) &h_map, T * sizeof(int32_t), cudaHostAllocDefault));
        }
        d_delta = dalloc_t<int32_t>(1);
        if (nxt != nullptr) CK(event_new(&ev_hand, cudaEventDisableTiming));
        for (int i = 0; i < 2 && split; ++i) {
            if (nxt != nullptr) CK(event_new(&ev_out[i], cudaEventDisableTiming));
            if (l0 > 0) {
                x_in[i] = dalloc(T * N);
                CK(event_new(&ev_free[i], cudaEventDisableTiming));
            }
        }
        cur = 0;
        load_cur();
        if (snapshots) {   // the conversation cache's image of a checkpoint (the session's that reads a prompt)
            size_t at = 0;
            for (size_t l = 0; l < conv.size(); ++l) {
                snap_at.push_back(at);
                if (conv[l]) at += C * (g.ssm_d_conv - 1) * 4 + (size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size * 4;
            }
            snap = dalloc_t<uint8_t>((size_t) state_bytes_own());
            snap_end = dalloc_t<uint8_t>((size_t) state_bytes_own());
            CK(stream_new(&st_snap, cudaStreamNonBlocking));
            CK(event_new(&ev_snap, cudaEventDisableTiming));
            CK(event_new(&ev_landed, cudaEventDisableTiming));
            CK(event_new(&ev_landed_end, cudaEventDisableTiming));
        }
        CK(cudaDeviceSynchronize());   // every legacy-stream cudaMemset of the buffers above is done
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        std::fprintf(stderr, "lamina: context %d cells, prompt chunk %d%s, %.2f GiB of VRAM free\n", ctx, tmax,
                     m->has_mtp ? ", MTP block loaded" : "", (double) fr / (1 << 30));
    }

    /// The recurrent state and the cache to a fresh sequence.
    void reset() {
        for_each([](Engine& e) { e.reset_own(); });
    }
    void reset_own() {
        const auto& g = m->g;
        for (size_t l = 0; l < conv.size(); ++l) {
            if (conv[l]) CK(cudaMemsetAsync(conv[l], 0, (size_t) g.ssm_conv_channels * (g.ssm_d_conv - 1) * 4, st));
            if (S[l]) CK(cudaMemsetAsync(S[l], 0, (size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size * 4, st));
        }
        CK(cudaMemsetAsync(hlast, 0, (size_t) g.n_embd * 4, st));
        CK(cudaStreamSynchronize(st));
        pos = 0;
        sess[(size_t) cur].pos = 0;
        set_delta_own(0);
    }

    /// y [T][n_out] = W x [T][n_in] on the native MMVQ, 8 columns per launch.
    void mm(const Mat& w, const float* xin, float* out, int T) {
        prep(xin, w.n_in, T, q8);
        apply(w, q8, out, T);
    }
    /// y [T][n_out] = W [n_out][n_in] (f32) x [T][n_in]: a decode batch on the warp-per-row kernel, a prompt chunk on
    /// cuBLAS's FP32 GEMM (the router and the GDN alpha / beta stay in FP32, as llama.cpp keeps them).
    void f32mm(const float* W, const float* xin, float* out, int n_in, int n_out, int T) {
        if (T <= kGemmT) {
            k_gemv_f32<<<dim3(rows8(n_out), T), kThreads, 0, st>>>(W, xin, out, n_in, n_out);
            return;
        }
        static const bool blas_f32 = std::getenv("LAMINA_Q36_BLAS_F32") != nullptr;   // the hipBLAS / cuBLAS SGEMM (A/B)
        if (!blas_f32 && n_in % 4 == 0) {
            // K split so that about 128 blocks run (the partials summed in a fixed order: deterministic); TM = 4 rows per
            // thread for the 256-wide router, 1 for the narrow ones (alpha / beta 32, the gate 1)
            const int tm = n_out > 64 ? 4 : 1;
            const int gx = (n_out + 63) / 64, gy = (T + 16 * tm - 1) / (16 * tm);
            const int want = std::max(1, std::min(kSplitK, 128 / std::max(1, gx * gy)));
            const int kz = (n_in / want + 15) / 16 * 16;
            const int splits = (n_in + kz - 1) / kz;
            float* dst = splits > 1 ? sk_part : out;
            if (tm == 4) k_sgemm_nt<4><<<dim3(gx, gy, splits), 256, 0, st>>>(W, xin, dst, n_in, n_out, T, n_out, kz);
            else k_sgemm_nt<1><<<dim3(gx, gy, splits), 256, 0, st>>>(W, xin, dst, n_in, n_out, T, n_out, kz);
            if (splits > 1)
                k_sum_splits<<<grid1((int64_t) T * n_out), kThreads, 0, st>>>(sk_part, out, (int64_t) T * n_out, splits);
            return;
        }
        const float one = 1.0f, zero = 0.0f;
        if (cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n_out, T, n_in, &one, W, n_in, xin, n_in, &zero, out, n_out) !=
            CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "lamina: cublasSgemm failed\n");
            throw Fatal{1};
        }
    }

    /// A batch above kGemmT tokens (a prompt chunk) runs its projections on llama.cpp's MMQ where it has the weight type
    /// (int8 tensor cores, the activations rounded to q8_1, as llama.cpp's prompt path does: the Q8_0 matrices of
    /// unsloth's files), else as GEMMs (prefill::Gemm::native: the GGUF blocks dequantized to FP16, the activations
    /// FP16); a decode batch on the native MMVQ (q8_1 activations).
    static constexpr int kGemmT = 8;
    void prep(const float* xin, int n, int T, void* xq81) {
        if (T > kGemmT) {
            px = xin;
            pn = n;
            pq = -1;
            p16 = false;
        } else {
            quant(xin, n, T, xq81);
        }
    }
    void apply(const Mat& w, const void* xq81, float* out, int T) {
        if (T <= kGemmT) {
            mmq(w, xq81, out, T);
        } else if (mq_dx != nullptr && MMQ::fits(w.type, w.n_out)) {
            if (pq != w.type) {
                MMQ::quantize(px, nullptr, mq_dx, w.type, pn, pn, T, st);
                pq = w.type;
            }
            mctx->run_dense(w.d, w.type, w.n_out, w.n_in, mq_dx, T, out, w.n_out, st);
        } else {
            if (!p16) {
                k_to_f16<<<grid1((int64_t) pn * T), kThreads, 0, st>>>(px, x16, (int64_t) pn * T);
                p16 = true;
            }
            gemm.native((const uint16_t*) x16, w.type, w.d, out, T, w.n_out, w.n_in);
        }
    }
    /// x [T][n] -> its q8_1 image (T contiguous columns: one vector of T * n), for every matrix that reads x; xq's and
    /// q8's with their interleaved copy for 2-8 rows where il_quant.
    void quant(const float* xin, int n, int T, void* xq81) {
        bool* ok = il_ok(xq81);
        if (ok && il_mode && il_quant && T >= 2 && T <= 8) {
            K::native_quantize_q8_1_il(xin, xq81, il_buf(xq81), n, T, st);
            *ok = true;
            return;
        }
        if (ok) *ok = false;
        K::native_quantize_q8_1(xin, xq81, n * T, 1, st);
    }
    bool* il_ok(const void* p) { return p == xq ? &xq_il_ok : p == q8 ? &q8_il_ok : nullptr; }
    void* il_buf(const void* p) { return p == xq ? xq_il : p == q8 ? q8_il : nullptr; }
    /// A kernel wrote new q8_1 rows into xq or q8: the interleaved copy is not theirs (mmq makes it when it is read).
    void il_stale(const void* p) {
        if (bool* ok = il_ok(p)) *ok = false;
    }
    /// Whether a window of T rows takes a matrix of this format on native_mmvq_il: where that is the faster on the RTX
    /// 3090 (mmvq_il_parity --bench27, Qwen3.8-27B's shapes): Q6_K from 2 rows, Q4_K / Q5_K / IQ4_XS from 3, Q8_0 from 4.
    static bool il_pays(int type, int T) {
        switch (type) {
        case 14: return T >= 2;
        case 12: case 13: case 23: return T >= 3;
        case 8: return T >= 4;
        default: return false;
        }
    }
    // ---- decode kernels of this engine (q36_kernels.cuh), each the old kernel's bits (LAMINA_Q36_VERIFY=1 compares)
    bool fast_k = std::getenv("LAMINA_Q36_OLD_KERNELS") == nullptr;
    bool verify_k = std::getenv("LAMINA_Q36_VERIFY") != nullptr;
    float* vbuf = nullptr;                       // verify scratch (vscratch)
    size_t vbuf_n = 0;
    /// the checks' scratch, of at least n floats: a bigger one when a check needs it (each goes at the end)
    float* vscratch(size_t n) {
        if (n > vbuf_n) {
            CK(dmalloc((void**) &vbuf, n * 4));
            vbuf_n = n;
        }
        return vbuf;
    }
    long long v_calls = 0, v_bad = 0;
    std::map<std::string, long long> v_by;   // mismatches per check
    void verify_cmp(const char* what, const float* a, const float* b, size_t n, const std::string& tag = "") {
        CK(cudaStreamSynchronize(st));
        std::vector<float> ha(n), hb(n);
        CK(cudaMemcpy(ha.data(), a, n * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hb.data(), b, n * 4, cudaMemcpyDeviceToHost));
        ++v_calls;
        if (std::memcmp(ha.data(), hb.data(), n * 4) != 0) {
            ++v_bad;
            size_t i = 0;
            while (i < n && std::memcmp(&ha[i], &hb[i], 4) == 0) ++i;
            if (++v_by[what] <= 3)
                std::fprintf(stderr, "lamina-q36 VERIFY MISMATCH %s%s: first at %zu of %zu: new %g old %g (bits %08x %08x)\n", what,
                             tag.c_str(), i, n, (double) ha[i], (double) hb[i], *(unsigned*) &ha[i], *(unsigned*) &hb[i]);
        }
    }
    /// A Q8_0 matrix of at most 2048 inputs for T <= 8 columns: one warp per row (q36k::k_q8_rows) in place of the
    /// 4-warp block of native_small_mmvq_kernel; the same bits.
    bool q8_small(const Mat& w, const void* xq81, float* out, int T) {
        if (!fast_k || w.type != 8 || T < 1 || T > 8 || w.n_in % 32 != 0) return false;
        const int nb = w.n_in / 32;
        if (nb > 64) return false;
        const void* wp = w.d;
        const half* dpl = nullptr;
        const int NJ = nb <= 32 ? 1 : 2;
#define Q36_Q8(NCV)                                                                                          \
    case NCV:                                                                                                \
        if (NJ == 1) q36k::launch_q8_rows<NCV, 1, false>(wp, dpl, xq81, out, nb, w.n_out, st);                  \
        else q36k::launch_q8_rows<NCV, 2, false>(wp, dpl, xq81, out, nb, w.n_out, st);                          \
        break;
        switch (T) { Q36_Q8(1) Q36_Q8(2) Q36_Q8(3) Q36_Q8(4) Q36_Q8(5) Q36_Q8(6) Q36_Q8(7) Q36_Q8(8) }
#undef Q36_Q8
        if (verify_k) {
            float* vb = vscratch((size_t) T * w.n_out);
            const size_t qb = K::native_q8_1_bytes(w.n_in, 1);
            for (int c0 = 0; c0 < T; c0 += 8) {
                const int nc = std::min(8, T - c0);
                K::native_mmvq(w.type, w.d, (const char*) xq81 + (size_t) c0 * qb, vb + (size_t) c0 * w.n_out, w.n_in, w.n_out,
                               nc, st);
            }
            verify_cmp("q8_small", out, vb, (size_t) T * w.n_out,
                       " n_in " + std::to_string(w.n_in) + " n_out " + std::to_string(w.n_out) + " T " + std::to_string(T));
        }
        return true;
    }
    // ---- the q8_1 images produced by the norm kernels themselves (fast_k, T <= kGemmT): xq_ready = the layer's input h is
    // already quantized in xq (k_moe_out_small of the layer before), xq_post = the MoE input h is (k_add_rms_q)
    bool xq_ready = false, xq_post = false;
    void* vq = nullptr;
    void verify_q8(const char* what, const float* hsrc, int T, const std::string& tag) {
        const int N = (int) m->g.n_embd;
        const size_t qb = K::native_q8_1_bytes(N, 1);
        if (!vq) CK(dmalloc(&vq, qb * 8));
        K::native_quantize_q8_1(hsrc, vq, N * T, 1, st);
        const long long before = v_bad;
        verify_cmp(what, (const float*) xq, (const float*) vq, qb * (size_t) T / 4, tag);
        if (v_bad != before && std::getenv("LAMINA_Q36_VDEBUG") && v_by[what] <= 2) {   // a mismatching block, on the host
            std::vector<uint8_t> a(qb * (size_t) T), b(qb * (size_t) T);
            std::vector<float> hv((size_t) N * T);
            CK(cudaMemcpy(a.data(), xq, a.size(), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(b.data(), vq, b.size(), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hv.data(), hsrc, hv.size() * 4, cudaMemcpyDeviceToHost));
            for (size_t o = 0; o < a.size(); o += 36)
                if (std::memcmp(&a[o], &b[o], 36) != 0) {
                    const size_t blk = o / 36;
                    float t[32];
                    for (int l = 0; l < 32; ++l) t[l] = hv[blk * 32 + (size_t) l];
                    for (int off = 16; off > 0; off >>= 1) {   // the xor butterfly, lane by lane
                        float u[32];
                        for (int l = 0; l < 32; ++l) u[l] = t[l] + t[l ^ off];
                        std::memcpy(t, u, sizeof(u));
                    }
                    unsigned ds_a, ds_b;
                    std::memcpy(&ds_a, &a[o], 4);
                    std::memcpy(&ds_b, &b[o], 4);
                    std::fprintf(stderr, "VDEBUG block %zu: ds new %08x old %08x, qs bytes %s, cpu butterfly sum %.9g (half %04x)\n",
                                 blk, ds_a, ds_b, std::memcmp(&a[o + 4], &b[o + 4], 32) == 0 ? "equal" : "DIFFER", (double) t[0],
                                 (unsigned) __half_as_ushort(__float2half(t[0])));
                    break;
                }
        }
    }
    void prep_in(int T) {
        if (xq_ready && T <= kGemmT) {
            xq_ready = false;
            return;
        }
        xq_ready = false;
        prep(h, (int) m->g.n_embd, T, xq);
    }
    void add_rms_post(const Layer& L, int T) {
        const int N = (int) m->g.n_embd;
        xq_post = false;
        if (fast_k && T <= kGemmT) {
            if (verify_k) {
                if (!vx_) {
                    CK(dmalloc((void**) &vx_, (size_t) 8 * 2048 * 4));
                    CK(dmalloc((void**) &vh_, (size_t) 8 * 2048 * 4));
                }
                CK(cudaMemcpyAsync(vx_, x, (size_t) T * N * 4, cudaMemcpyDeviceToDevice, st));
                k_add_rms<<<T, kThreads, 0, st>>>(vx_, y, L.post_norm, vh_, N, m->eps);
            }
            q36k::k_add_rms_q<<<T, 256, 0, st>>>(x, y, L.post_norm, h, (q36k::QB*) xq, N, m->eps);
            il_stale(xq);
            xq_post = true;
            if (verify_k) {
                verify_cmp("add_rms x", x, vx_, (size_t) T * N, " T " + std::to_string(T));
                verify_cmp("add_rms h", h, vh_, (size_t) T * N, " T " + std::to_string(T));
                verify_q8("add_rms q8", h, T, " T " + std::to_string(T));
            }
            return;
        }
        k_add_rms<<<T, kThreads, 0, st>>>(x, y, L.post_norm, h, N, m->eps);
    }
    /// The shared expert's gate + up + SwiGLU + q8_1 image (into q8) for T <= 4 rows in one launch (q36k::k_sh_gu), else false.
    bool shared_gu(const Layer& L, int T) {
        const int FS = (int) m->g.n_ff_shexp;
        const Mat &a = L.sh_gate, &b = L.sh_up;
        if (!fast_k || T < 1 || T > 4 || a.type != 8 || b.type != 8 || a.n_in != b.n_in || a.n_out != FS || b.n_out != FS ||
            a.n_in % 32 != 0 || a.n_in / 32 > 64 || a.n_in / 32 <= 32 || FS % 32 != 0)
            return false;
        const int nb = a.n_in / 32;
        const dim3 grid((unsigned) (FS / 32));
        const bool pkd = false;
        const uint8_t* wa = a.d;
        const uint8_t* wb = b.d;
        const half* da = nullptr;
        const half* db = nullptr;
#define Q36_SH(NCV)                                                                                          \
    case NCV:                                                                                                \
        if (pkd)                                                                                             \
            q36k::k_sh_gu<NCV, 2, true><<<grid, 1024, 0, st>>>(wa, da, wb, db, (const q36k::QB*) xq,        \
                                                               (q36k::QB*) q8, nb, FS);                      \
        else                                                                                                 \
            q36k::k_sh_gu<NCV, 2, false><<<grid, 1024, 0, st>>>(wa, nullptr, wb, nullptr, (const q36k::QB*) xq, \
                                                                (q36k::QB*) q8, nb, FS);                     \
        break;
        switch (T) { Q36_SH(1) Q36_SH(2) Q36_SH(3) Q36_SH(4) }
        il_stale(q8);
#undef Q36_SH
        if (verify_k) {
            if (!vq) CK(dmalloc(&vq, K::native_q8_1_bytes((int) m->g.n_embd, 1) * 8));
            float* vg = vscratch((size_t) 16 * FS);
            float* vu = vg + (size_t) 8 * FS;
            mmq_ref(a, xq, vg, T);
            mmq_ref(b, xq, vu, T);
            K::native_swiglu_quantize_q8_1(vg, vu, vq, FS, T, st);
            verify_cmp("shared_gu q8", (const float*) q8, (const float*) vq, K::native_q8_1_bytes(FS, 1) * (size_t) T / 4,
                       " T " + std::to_string(T));
        }
        return true;
    }
    /// The old products (native MMVQ), for the checks.
    void mmq_ref(const Mat& w, const void* xq81, float* out, int T) {
        const size_t qb = K::native_q8_1_bytes(w.n_in, 1);
        for (int c0 = 0; c0 < T; c0 += 8) {
            const int nc = std::min(8, T - c0);
            K::native_mmvq(w.type, w.d, (const char*) xq81 + (size_t) c0 * qb, out + (size_t) c0 * w.n_out, w.n_in, w.n_out, nc, st);
        }
    }
    void group_old(const Layer& L, const int32_t* slot_of, int T, unsigned long long* o_ptr, int32_t* o_start, int32_t* o_ng,
                   int32_t* o_dst, int32_t* o_tok) {
        const int Kp = (int) m->top_k, n_ent = T * Kp;
        if (slot_of)
            k_group_hy<<<1, kThreads, 0, st>>>(ids, n_ent, Kp, (int) m->g.n_expert, slot_of, L.blobs, L.stride, o_ptr, o_start,
                                               o_ng, o_dst, o_tok);
        else
            k_group<<<1, kThreads, 0, st>>>(ids, n_ent, Kp, (int) m->g.n_expert, L.blobs, L.stride, o_ptr, o_start, o_ng, o_dst,
                                            o_tok);
    }
    /// VERIFY: the old grouping of the current `ids` into scratch, compared with grp_ptr / grp_start / n_groups / ent_dst / ent_tok.
    void group_verify(const Layer& L, const int32_t* slot_of, int T) {
        if (!vgrp) CK(dmalloc((void**) &vgrp, 2048));
        unsigned long long* o_ptr = (unsigned long long*) vgrp;
        int32_t* o_start = (int32_t*) (o_ptr + 65);
        int32_t* o_ng = o_start + 66;
        int32_t* o_dst = o_ng + 2;
        int32_t* o_tok = o_dst + 64;
        group_old(L, slot_of, T, o_ptr, o_start, o_ng, o_dst, o_tok);
        CK(cudaStreamSynchronize(st));
        int nn = 0, no = 0;
        CK(cudaMemcpy(&nn, n_groups, 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&no, o_ng, 4, cudaMemcpyDeviceToHost));
        ++v_calls;
        bool bad = nn != no;
        if (!bad) {
            std::vector<unsigned long long> a((size_t) nn + 1), b((size_t) nn + 1);
            std::vector<int32_t> sa((size_t) nn + 1), sb((size_t) nn + 1), da(64), db(64), ta(64), tb(64);
            CK(cudaMemcpy(a.data(), grp_ptr, (size_t) nn * 8, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(b.data(), o_ptr, (size_t) nn * 8, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(sa.data(), grp_start, ((size_t) nn + 1) * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(sb.data(), o_start, ((size_t) nn + 1) * 4, cudaMemcpyDeviceToHost));
            const int tot = sa[(size_t) nn];
            CK(cudaMemcpy(da.data(), ent_dst, (size_t) tot * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(db.data(), o_dst, (size_t) tot * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(ta.data(), ent_tok, (size_t) tot * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(tb.data(), o_tok, (size_t) tot * 4, cudaMemcpyDeviceToHost));
            bad = std::memcmp(a.data(), b.data(), (size_t) nn * 8) != 0 ||
                  std::memcmp(sa.data(), sb.data(), ((size_t) nn + 1) * 4) != 0 ||
                  std::memcmp(da.data(), db.data(), (size_t) tot * 4) != 0 ||
                  std::memcmp(ta.data(), tb.data(), (size_t) tot * 4) != 0;
        }
        if (bad) {
            ++v_bad;
            if (v_bad <= 20) std::fprintf(stderr, "lamina-q36 VERIFY MISMATCH group (T %d, groups %d vs %d)\n", T, nn, no);
        }
    }
    bool grouped_by_route = false;   // route_group already filled the group arrays of this layer
    /// The router's top-k and the grouping of the picks in one launch (T <= 8, 256 experts, 8 picks); the logits are in logits_r.
    /// `slot_of`: the hybrid mode's slot table (null: every expert resident).  Returns false when the old kernels must run.
    bool route_group(const Layer& L, const int32_t* slot_of, int T, int ld) {
        const int Kp = (int) m->top_k;
        grouped_by_route = false;
        if (!fast_k || T > 8 || m->g.n_expert != 256 || Kp != 8) return false;
        if (slot_of)
            q36k::k_route_group<true><<<1, dim3(32, 8), 0, st>>>(logits_r, ids, ewts, T, ld, slot_of, L.blobs, L.stride, grp_ptr,
                                                                 grp_start, n_groups, ent_dst, ent_tok);
        else
            q36k::k_route_group<false><<<1, dim3(32, 8), 0, st>>>(logits_r, ids, ewts, T, ld, nullptr, L.blobs, L.stride, grp_ptr,
                                                                  grp_start, n_groups, ent_dst, ent_tok);
        grouped_by_route = true;
        if (verify_k) {
            // the old top-k into scratch, the ids / weights compared, then the old grouping of the new ids
            float* vs = vscratch((size_t) 2 * T * Kp);
            int32_t* vid = (int32_t*) vs;
            float* vw = vs + (size_t) T * Kp;
            K::native_router_topk(logits_r, vid, vw, 256, Kp, T, st);
            verify_cmp("route ids", (const float*) ids, (const float*) vid, (size_t) T * Kp, " T " + std::to_string(T));
            verify_cmp("route weights", ewts, vw, (size_t) T * Kp, " T " + std::to_string(T));
            group_verify(L, slot_of, T);
        }
        return true;
    }
    /// The routed entries of T rows grouped per expert (k_group; with slot_of the hybrid mode's k_group_hy).
    void group_entries(const Layer& L, const int32_t* slot_of, int T) {
        if (grouped_by_route) {   // route_group did it
            grouped_by_route = false;
            return;
        }
        const int Kp = (int) m->top_k, n_ent = T * Kp;
        if (fast_k && n_ent <= 64) {
            if (slot_of)
                q36k::k_group_small<true><<<1, 64, 0, st>>>(ids, n_ent, Kp, slot_of, L.blobs, L.stride, grp_ptr, grp_start,
                                                            n_groups, ent_dst, ent_tok);
            else
                q36k::k_group_small<false><<<1, 64, 0, st>>>(ids, n_ent, Kp, nullptr, L.blobs, L.stride, grp_ptr, grp_start,
                                                             n_groups, ent_dst, ent_tok);
            if (verify_k) group_verify(L, slot_of, T);
            return;
        }
        group_old(L, slot_of, T, grp_ptr, grp_start, n_groups, ent_dst, ent_tok);
    }
    void* vgrp = nullptr;
    float *vx_ = nullptr, *vh_ = nullptr;
    /// The MoE block's end (k_moe_out) for T rows.
    void moe_end(int T, const int32_t* prow, const float* shared, const float* sgate_in, const float* wn, float* hout, int sgs = 1) {
        const int N = (int) m->g.n_embd, Kp = (int) m->top_k;
        const bool small = fast_k && T <= 8 && N <= 2048 && N % 256 == 0 && Kp <= 15;
        if (small && verify_k) {
            if (!vx_) {
                CK(dmalloc((void**) &vx_, (size_t) 8 * 2048 * 4));
                CK(dmalloc((void**) &vh_, (size_t) 8 * 2048 * 4));
            }
            CK(cudaMemcpyAsync(vx_, x, (size_t) T * N * 4, cudaMemcpyDeviceToDevice, st));
            k_moe_out<<<T, kThreads, 0, st>>>(vx_, eparts, prow, ewts, Kp, shared, sgate_in, sgs, wn, vh_, N, m->eps);
        }
        if (small) {
            const bool mkq = wn != nullptr && hout == h;   // the next layer's input, quantized here
            q36k::k_moe_out_small<<<T, 1024, 0, st>>>(x, eparts, prow, ewts, Kp, shared, sgate_in, sgs, wn, hout, N, m->eps,
                                                      mkq ? (q36k::QB*) xq : nullptr);
            if (mkq) il_stale(xq);
            xq_ready = mkq;
            if (verify_k) {
                if (mkq) verify_q8("moe_out q8", hout, T, " T " + std::to_string(T));
                verify_cmp("moe_out x", x, vx_, (size_t) T * N, " T " + std::to_string(T));
                if (wn) verify_cmp("moe_out h", hout, vh_, (size_t) T * N, " T " + std::to_string(T));
            }
            return;
        }
        xq_ready = false;
        k_moe_out<<<T, kThreads, 0, st>>>(x, eparts, prow, ewts, Kp, shared, sgate_in, sgs, wn, hout, N, m->eps);
    }
    /// y [T][n_out] = W x from x's q8_1 image, 8 columns per launch.
    void mmq(const Mat& w, const void* xq81, float* out, int T) {
        if (q8_small(w, xq81, out, T)) return;
        if (bool* ok = il_ok(xq81); ok && il_mode && T >= 2 && T <= 8 && il_pays(w.type, T)) {
            if (!*ok) {   // these rows' interleaved copy, on its first use
                K::native_q8_1_il_from(xq81, il_buf(xq81), w.n_in, T, st);
                *ok = true;
            }
            K::native_mmvq_il(w.type, w.d, xq81, il_buf(xq81), out, w.n_in, w.n_out, T, st, il_mode == 2);
            return;
        }
        const size_t qb = K::native_q8_1_bytes(w.n_in, 1);
        for (int c0 = 0; c0 < T; c0 += 8) {
            const int nc = std::min(8, T - c0);
            K::native_mmvq(w.type, w.d, (const char*) xq81 + (size_t) c0 * qb, out + (size_t) c0 * w.n_out, w.n_in,
                           w.n_out, nc, st);
        }
    }

    // ---- LoRA adapters
    /// The scales of the pass's rows on every stage: row r's are session rows[r]'s (its request's), every row's the current
    /// session's when `rows` is null; lora_live when a row uses an adapter.  Before the pass (a table that changed is an
    /// upload, not a graph's node).
    void lora_rows(const int* rows, int T) {
        if (n_ad == 0) return;
        std::vector<float> tab((size_t) kVerify * n_ad, 0.f);
        bool live = false;
        for (int r = 0; r < kVerify; ++r) {
            const int s_ = rows == nullptr ? cur : r < T ? rows[r] : -1;
            if (s_ < 0) continue;
            const std::vector<float>& sc = sess[(size_t) s_].lora;
            for (int a = 0; a < n_ad && a < (int) sc.size(); ++a) {
                tab[(size_t) r * n_ad + a] = sc[(size_t) a];
                live = live || sc[(size_t) a] != 0.f;
            }
        }
        for_each([&](Engine& e) {
            e.lora_live = live;
            if (tab == e.h_lscale) return;
            e.h_lscale = tab;
            CK(cudaMemcpyAsync(e.d_lscale, e.h_lscale.data(), tab.size() * 4, cudaMemcpyHostToDevice, e.st));
        });
    }
    /// one product of a site: its pairs (null: none) and its output [T][ld] (its first n columns)
    struct LoraOut {
        const LoraMat* lm;
        float* y;
        int ld, n;
    };
    /// The adapters of a site - up to three products that read one input, x [T][ldx] of n_in (with u: SwiGLU(x, u), the
    /// down product's decode input) - added into the products' outputs, after them: a decode batch's launch pairs (the A
    /// stage, then the B stage), a prompt chunk's GEMMs.  `assign` (decode only): the outputs take the sum alone, zeros
    /// where no pair is (the GDN alpha / beta deltas).  Nothing without an adapter in the pass.
    void lora_site(const float* xin, const float* u, int ldx, int n_in, int T, std::initializer_list<LoraOut> outs,
                   bool assign = false) {
        if (!lora_live || T <= 0) return;
        bool any = false;
        for (const LoraOut& o : outs) any = any || o.lm != nullptr;
        if (!any) return;
        if (T > kGemmT) {
            lora_gemms(xin, ldx, n_in, T, outs);
            return;
        }
        static const int kslice = [] {   // a K split's elements (LAMINA_Q36_LORA_KSLICE: A/B, up to kSliceMax)
            const char* v = std::getenv("LAMINA_Q36_LORA_KSLICE");
            const int k = v ? std::atoi(v) : q36lora::kSliceMax;
            return std::min(q36lora::kSliceMax, std::max(1, k / q36lora::kChunk) * q36lora::kChunk);
        }();
        q36lora::Site s;
        s.x = xin;
        s.u = u;
        s.ldx = ldx;
        s.n_in = n_in;
        s.kslice = kslice;
        s.ks = (n_in + kslice - 1) / kslice;
        s.T = T;
        s.n_ad = n_ad;
        s.scale = d_lscale;
        s.part = l_part;
        auto launch = [&] {
            for (int m_ = 0; m_ < s.n_mem; ++m_) s.ob[m_ + 1] = s.ob[m_] + (s.n_out[m_] + 255) / 256;
            q36lora::k_lora_a<<<dim3((unsigned) s.n_groups, (unsigned) s.ks), 256, 0, st>>>(s);
            q36lora::k_lora_b<<<(unsigned) s.ob[s.n_mem], 256, 0, st>>>(s);
            s.n_seg = s.n_groups = s.R = s.n_mem = 0;
        };
        if (assign)   // the deltas alone: zeros, the pairs then added
            for (const LoraOut& o : outs)
                CK(cudaMemset2DAsync(o.y, (size_t) o.ld * 4, 0, (size_t) o.n * 4, (size_t) T, st));
        // the members in order, a launch taking their pairs while kMaxSeg pairs, kMaxR ranks and 3 members hold them
        for (const LoraOut& o : outs) {
            if (o.lm == nullptr) continue;
            const std::vector<LoraSeg>& segs = o.lm->segs;
            for (size_t i = 0; i < segs.size();) {
                if (s.n_mem == 3 || s.n_seg == q36lora::kMaxSeg || s.R + segs[i].rank > q36lora::kMaxR) launch();
                const int mi = s.n_mem++;
                s.out[mi] = o.y;
                s.ld_out[mi] = o.ld;
                s.n_out[mi] = o.n;
                for (; i < segs.size() && s.n_seg < q36lora::kMaxSeg && s.R + segs[i].rank <= q36lora::kMaxR; ++i) {
                    const LoraSeg& pr = segs[i];
                    q36lora::Seg& d = s.seg[s.n_seg++];
                    d.a = pr.a;
                    d.b = pr.b;
                    d.alpha = m->loras[(size_t) pr.adapter].alpha;
                    d.rank_true = pr.rank_true;
                    d.rank = pr.rank;
                    d.r0 = s.R;
                    d.adapter = pr.adapter;
                    d.member = mi;
                    d.f32 = pr.f32 ? 1 : 0;
                    d.g0 = s.n_groups;
                    s.R += pr.rank;
                    s.n_groups += pr.rank / q36lora::kGroup;
                }
            }
        }
        if (s.n_mem > 0) launch();
    }
    /// A prompt chunk's pairs as GEMMs (cuBLAS, FP32 accumulation): t = A x per pair, then out += s B t; the scales of the
    /// table's first row (a prompt is one request's: every row holds them).
    void lora_gemms(const float* xin, int ldx, int n_in, int T, std::initializer_list<LoraOut> outs) {
        const float* sc = h_lscale.data();
        bool f16_ready = false;
        auto ok = [](cublasStatus_t e) {
            if (e != CUBLAS_STATUS_SUCCESS) {
                std::fprintf(stderr, "lamina: a LoRA GEMM failed (cuBLAS status %d)\n", (int) e);
                throw Fatal{1};
            }
        };
        for (const LoraOut& o : outs) {
            if (o.lm == nullptr) continue;
            for (const LoraSeg& pr : o.lm->segs) {
                const float us = sc[(size_t) pr.adapter];
                if (us == 0.f) continue;
                const float scale = q36lora::pair_scale(us, m->loras[(size_t) pr.adapter].alpha, pr.rank_true);
                const float one = 1.f, zero = 0.f;
                if (pr.f32) {
                    ok(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, pr.rank, T, n_in, &one, (const float*) pr.a, n_in,
                                   xin, ldx, &zero, l_t, pr.rank));
                    ok(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, o.n, T, pr.rank, &scale, (const float*) pr.b,
                                   pr.rank, l_t, pr.rank, &one, o.y, o.ld));
                    continue;
                }
                if (!f16_ready) {   // the input in FP16 once for the site's F16 pairs (x16: the prompt GEMMs' buffer)
                    if (!(xin == px && p16 && pn == n_in)) {
                        const int64_t total = (int64_t) T * n_in;
                        q36lora::k_lora_f16<<<(unsigned) ((total + 255) / 256), 256, 0, st>>>(xin, ldx, x16, n_in, total);
                        p16 = xin == px && pn == n_in && ldx == n_in;   // x16 holds the projections' input again, or not
                    }
                    f16_ready = true;
                }
                __half* t16 = (__half*) l_t;
                ok(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, pr.rank, T, n_in, &one, pr.a, CUDA_R_16F, n_in, x16,
                                CUDA_R_16F, n_in, &zero, t16, CUDA_R_16F, pr.rank, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                ok(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, o.n, T, pr.rank, &scale, pr.b, CUDA_R_16F, pr.rank, t16,
                                CUDA_R_16F, pr.rank, &one, o.y, CUDA_R_32F, o.ld, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            }
        }
    }

    /// One layer for T rows: x (the residual) += mixer(h); then MoE; x += moe; `hout` = rms(x) * wn (the next norm).
    /// `li` indexes the layer's state; positions *dpos .. + T - 1 (device), p0 the same on the host (eager grids);
    /// `verify`: a verify window - the GDN state is read, not written (commit() writes it for the kept rows).
    void layer(const Layer& L, size_t li, int T, int64_t p0, const int32_t* dpos, bool graph_mode, bool verify,
               const float* wn, float* hout) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, Kp = (int) m->top_k, FF = (int) g.n_ff, FS = (int) g.n_ff_shexp;
        const int HK = (int) g.ssm_k_heads, HV = (int) g.ssm_v_heads;
        const int C = (int) g.ssm_conv_channels;
        const int H = (int) g.n_head, HKV = (int) g.n_head_kv, HD = (int) g.head_dim;
        if (!L.attn) {
            float* qkv_b = verify ? v_qkv[li] : qkv;
            float* hc_b = verify ? v_hc[li] : hc;
            float* al_b = verify ? v_alpha[li] : alpha;
            float* be_b = verify ? v_beta[li] : beta;
            prep_in(T);
            apply(L.qkv, xq, qkv_b, T);
            apply(L.zg, xq, z, T);
            lora_site(h, nullptr, N, N, T, {{L.qkv.lora, qkv_b, L.qkv.n_out, L.qkv.n_out},
                                            {L.zg.lora, z, L.zg.n_out, L.zg.n_out}});
            if (T <= kGemmT) {
                const float* dab = nullptr;   // the adapters' alpha / beta deltas, added before the epilogues
                if (lora_live && (L.lora_alpha != nullptr || L.lora_beta != nullptr)) {
                    lora_site(h, nullptr, N, N, T,
                              {{L.lora_alpha, l_dab, 2 * HV, HV}, {L.lora_beta, l_dab + HV, 2 * HV, HV}}, true);
                    dab = l_dab;
                }
                k_gdn_ab<<<dim3(rows8(2 * HV), T), kThreads, 0, st>>>(L.alpha, L.beta, h, L.dt, L.ssm_a, al_b, be_b, N,
                                                                      HV, dab);
            } else {
                f32mm(L.alpha, h, al_b, N, HV, T);
                f32mm(L.beta, h, be_b, N, HV, T);
                lora_site(h, nullptr, N, N, T, {{L.lora_alpha, al_b, HV, HV}, {L.lora_beta, be_b, HV, HV}});
                k_gdn_gb<<<grid1((int64_t) HV * T), kThreads, 0, st>>>(al_b, L.dt, L.ssm_a, be_b, HV, T);
            }
            if (rows_mode) {   // batch rows: a slot's rows (one, or a verify window's two) in sequence on its own state
                k_conv_seq<<<dim3(C / 128, rows_G), 128, 0, st>>>(nullptr, qkv_b, L.conv, hc_b, C, 2 * HK, m->eps, 1,
                                                                  !verify, d_conv_tab + li * n_sess, d_gslot, d_grow0, d_gcnt);
                k_gdn_seq<<<dim3(HV, rows_G), dim3(128, 4), 0, st>>>(nullptr, hc_b, C, al_b, be_b, z, L.ssm_norm, m->eps,
                                                                     gdn_y, HK, HV, 1, !verify, d_S_tab + li * n_sess,
                                                                     d_gslot, d_grow0, d_gcnt);
            } else {
                // one token, a window or a chunk: the same kernels (every token in one launch each), so a decode step and a
                // verify window compute bitwise the same rows and the MTP's greedy text is the plain greedy text.  (Lamina's
                // fused single-token kernels are not used here: fused_gdn_conv_l2's 4-tap sum contracts into FMAs
                // differently from k_conv_seq's, a 1-ulp difference in h that the Q8_1 activation rounding can turn into a
                // flipped near-tie later; with them the KL is the same, 1.05x / 0.81x the floor, but MTP would not be exact.)
                // a prompt chunk: tiles of kConvTile tokens side by side (tile mode); else one block per head walks the tokens
                const int tiles = (fast_k && T > kGemmT && !verify) ? (T + kConvTile - 1) / kConvTile : 1;
                // a checkpoint inside the chunk (snap_row): the state after its first `sr` rows goes into the image.  The
                // scan walks the tokens one after the other, so it is two launches with the state copied between them,
                // the same arithmetic as one; the conv history there is the last three inputs before that row.
                const int sr = !verify && snap_row > 0 && snap_row < T ? snap_row : 0;
                float* const image = sr ? reinterpret_cast<float*>(snap + snap_at[li]) : nullptr;
                const size_t cb = (size_t) C * (g.ssm_d_conv - 1), sb = (size_t) HV * g.ssm_state_size * g.ssm_state_size;
                const size_t ZV = (size_t) HV * g.ssm_state_size;   // a row of z / gdn_y
                if (sr) k_conv_hist<<<C / 128, 128, 0, st>>>(conv[li], qkv_b, image, C, sr);
                k_conv_seq<<<dim3(C / 128, tiles), 128, 0, st>>>(conv[li], qkv_b, L.conv, hc_b, C, 2 * HK, m->eps, T, !verify);
                if (T > kGemmT && !verify) {   // a prompt chunk: the column-split delta rule, then the norm
                    const dim3 grid(HV, kGdnColGroups), block(128 / kGdnColGroups, 16);
                    if (sr) {
                        k_gdn_seq_cols<<<grid, block, 0, st>>>(S[li], hc_b, C, al_b, be_b, gdn_y, HK, HV, sr);
                        CK(cudaMemcpyAsync(image + cb, S[li], sb * 4, cudaMemcpyDeviceToDevice, st));
                        k_gdn_seq_cols<<<grid, block, 0, st>>>(S[li], hc_b + (size_t) sr * C, C, al_b + (size_t) sr * HV,
                                                               be_b + (size_t) sr * HV, gdn_y + (size_t) sr * ZV, HK, HV,
                                                               T - sr);
                    } else {
                        k_gdn_seq_cols<<<grid, block, 0, st>>>(S[li], hc_b, C, al_b, be_b, gdn_y, HK, HV, T);
                    }
                    k_gdn_norm<<<dim3(HV, T), 128, 0, st>>>(gdn_y, z, L.ssm_norm, m->eps, gdn_y, HV);
                } else if (sr) {   // a short chunk (the decode kernel), around the checkpoint's row
                    k_gdn_seq<<<HV, dim3(128, 4), 0, st>>>(S[li], hc_b, C, al_b, be_b, z, L.ssm_norm, m->eps, gdn_y, HK, HV,
                                                           sr, true);
                    CK(cudaMemcpyAsync(image + cb, S[li], sb * 4, cudaMemcpyDeviceToDevice, st));
                    k_gdn_seq<<<HV, dim3(128, 4), 0, st>>>(S[li], hc_b + (size_t) sr * C, C, al_b + (size_t) sr * HV,
                                                           be_b + (size_t) sr * HV, z + (size_t) sr * ZV, L.ssm_norm, m->eps,
                                                           gdn_y + (size_t) sr * ZV, HK, HV, T - sr, true);
                } else {   // decode, verify windows and commits: one kernel, so a window's rows are a decode step's rows
                    k_gdn_seq<<<HV, dim3(128, 4), 0, st>>>(S[li], hc_b, C, al_b, be_b, z, L.ssm_norm, m->eps, gdn_y, HK, HV,
                                                           T, !verify);
                }
            }
            mm(L.ssm_out, gdn_y, y, T);
            lora_site(gdn_y, nullptr, L.ssm_out.n_in, L.ssm_out.n_in, T, {{L.ssm_out.lora, y, N, N}});
        } else {
            prep_in(T);
            apply(L.wq, xq, qg, T);
            apply(L.wk, xq, kx, T);
            apply(L.wv, xq, vx, T);
            lora_site(h, nullptr, N, N, T, {{L.wq.lora, qg, L.wq.n_out, L.wq.n_out}, {L.wk.lora, kx, L.wk.n_out, L.wk.n_out},
                                            {L.wv.lora, vx, L.wv.n_out, L.wv.n_out}});
            RowSel rs;
            if (rows_mode) {
                rs.cell = d_rcell;
                rs.slot = d_rslot;
                rs.delta = d_rdelta;
                rs.kc = d_kc_tab + li * n_sess;
                rs.vc = d_vc_tab + li * n_sess;
            }
            k_attn_prep<<<dim3(H + HKV, T), kThreads, 0, st>>>(qg, kx, vx, L.q_norm, L.k_norm, q, gate, rows_mode ? nullptr : kc[li],
                                                              rows_mode ? nullptr : vc[li], H, HKV, HD, m->n_rot, dpos, ctx,
                                                              m->rope_base, m->eps, d_delta, mtab_now, rs, m->kv_q8);
            if (T > kGemmT) {
                // a prompt chunk: flash attention on the tensor cores (q36_attn.cuh), the output gated
                const int G = H / HKV, tt = q36a::kRows / G;
                kFaPrompt[m->kv_q8][G - 1]<<<dim3((T + tt - 1) / tt, HKV), 256, sizeof(q36a::FaSmem), st>>>(
                    q, kc[li], vc[li], gate, att, T, H, ctx, (int) p0, 1.0f / sqrtf((float) HD));
            } else {
                // flash-decoding over 256-cell chunks (k_attn_dec); the chunk grid covers every cell a row can see
                // (graph mode: the whole context, past-the-end chunks return at once)
                const int rows_per = att_rows / H;
                for (int t0 = 0; t0 < T; t0 += rows_per) {
                    const int nt = std::min(rows_per, T - t0);
                    const int nc = graph_mode ? n_chunk
                                              : (int) std::min<int64_t>(n_chunk, (p0 + t0 + nt + kChunkCells - 1) / kChunkCells);
                    const int nr = rows_mode ? 1 : nt, G = H / HKV;
                    kAttnDec[m->kv_q8][nr - 1]<<<dim3(nc, HKV, rows_mode ? nt : 1), 32 * G, attn_dec_smem(G, nr), st>>>(
                        q, rows_mode ? nullptr : kc[li], rows_mode ? nullptr : vc[li], dpos, t0, H, ctx,
                        1.0f / sqrtf((float) HD), n_chunk, pacc, pm, pl, rs);
                    k_attn_merge<<<nt * H, HD, 0, st>>>(pacc, pm, pl, gate, dpos, t0, H, n_chunk, att, rs);
                }
            }
            mm(L.wo, att, y, T);
            lora_site(att, nullptr, L.wo.n_in, L.wo.n_in, T, {{L.wo.lora, y, N, N}});
        }
        add_rms_post(L, T);
        if (!m->moe) {
            dense_ffn(L, T, wn, hout);
            return;
        }
        // routed experts (one token: the router and the shared expert's gate in one product, the gate its last row)
        const bool one = T == 1;
        // a decode batch of up to 8 rows takes the router and the shared expert's gate in one product too (rows of 257) when the fused
        // router follows; the prompt chunks and the checks keep the separate products
        const bool ext = one || (fast_k && std::getenv("LAMINA_Q36_NO_EXT") == nullptr && !verify_k && T <= kGemmT && g.n_expert == 256 && Kp == 8);
        if (ext) f32mm(L.router_ext, h, logits_r, N, (int) g.n_expert + 1, T);
        else f32mm(L.router, h, logits_r, N, (int) g.n_expert, T);
        // the hybrid rows path: every batch of a hybrid layer but a prompt chunk past hybrid_t that MMQ streams (main
        // layers: when only the MTP block is hybrid it has no staging buffers); the all-resident path below would read
        // its slot-indexed blobs by expert id (--hybrid-rows under 8)
        const bool hy_rows = lhy(li) && !streamed(T);
        const bool mmq_rows = !hy_rows && T > kGemmT && use_mmq;
        if (!(!mmq_rows && route_group(L, hy_rows ? hy[li].slot_of_d : nullptr, T, ext ? (int) g.n_expert + 1 : (int) g.n_expert)))
            K::native_router_topk(logits_r, ids, ewts, (int) g.n_expert, Kp, T, st);
        const int64_t n_ent = (int64_t) T * Kp;
        const int32_t* prow = nullptr;
        if (hy_rows) {
            moe_hybrid_gpu(L, li, T);
        } else if (T > kGemmT && use_mmq) {
            // a prompt chunk: the experts on llama.cpp's MMQ (int8, the weights read once per expert): the entries
            // sorted by expert on the device, gate+up as one [2 FF] product per expert, SwiGLU, down into rows the
            // combine reads through entry -> sorted row
            const int ne = (int) g.n_expert;
            CK(cudaMemsetAsync(rcnt, 0, (size_t) ne * 4, st));
            k_route_count<<<grid1(n_ent), kThreads, 0, st>>>(ids, (int) n_ent, rcnt);
            k_route_scan<<<1, 32, 0, st>>>(rcnt, ne, rbounds, rcur, rmax);
            k_route_scatter<<<grid1(n_ent), kThreads, 0, st>>>(ids, (int) n_ent, Kp, rcur, ent_dst, ent_tok, rinv);
            CK(cudaMemcpyAsync(hpin + 10, rmax, 4, cudaMemcpyDeviceToHost, st));
            if (lhy(li)) CK(cudaMemcpyAsync(h_cnt + li * g.n_expert, rcnt, (size_t) ne * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            if (lhy(li))
                for (int e = 0; e < ne; ++e) hy[li].score[(size_t) e] += h_cnt[li * g.n_expert + e];
            const int64_t maxr = hpin[10];
            const uint8_t* wbase = lhy(li) ? stream_take((int) li) : L.blobs;
            MMQ::quantize(h, ent_tok, mq_x, L.L.gu_type, N, N, n_ent, st);
            MMQ::Product gu;
            gu.w = wbase; gu.type = L.L.gu_type; gu.w_rows = 2 * FF; gu.w_cols = N; gu.expert_bytes = L.stride;
            gu.n = ne; gu.xq = mq_x; gu.bounds = rbounds; gu.ids = ident; gu.ids_rows = ident_rows;
            gu.total_rows = n_ent; gu.max_rows = maxr;
            gu.dst = mq_gu; gu.ld_dst = 2 * FF;
            mctx->run(gu, st);
            MMQ::swiglu(mq_gu, mq_h, n_ent, FF, false, st);
            MMQ::quantize(mq_h, nullptr, mq_hq, L.L.d_type, FF, FF, n_ent, st);
            MMQ::Product dn;
            dn.w = wbase + L.L.down_off; dn.type = L.L.d_type; dn.w_rows = N; dn.w_cols = FF;
            dn.expert_bytes = L.stride; dn.n = ne; dn.xq = mq_hq; dn.bounds = rbounds; dn.ids = ident;
            dn.ids_rows = ident_rows; dn.total_rows = n_ent; dn.max_rows = maxr; dn.dst = eparts; dn.ld_dst = N;
            mctx->run(dn, st);
            if (lhy(li)) stream_done();
            prow = rinv;
        } else {
            group_entries(L, nullptr, T);
            if (!xq_post) quant(h, N, T, xq);   // the experts' and the shared expert's input
            K::native_expert_grouped(L.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
                                     std::min<int64_t>(n_ent, g.n_expert), n_ent, xq, (int64_t) (N / 32) * 36, escr,
                                     eparts, st);
        }
        // the shared expert, scaled by sigmoid(w . h)
        if (T > kGemmT) prep(h, N, T, xq);
        if (T <= kGemmT) {
            // (an adapter's pairs need the gate and up products themselves: no fused gate + up + SwiGLU then)
            const bool sh_lora = lora_live && (L.sh_gate.lora || L.sh_up.lora || L.sh_down.lora);
            if (sh_lora || !shared_gu(L, T)) {
                apply(L.sh_gate, xq, sg, T);
                apply(L.sh_up, xq, su, T);
                lora_site(h, nullptr, N, N, T, {{L.sh_gate.lora, sg, FS, FS}, {L.sh_up.lora, su, FS, FS}});
                K::native_swiglu_quantize_q8_1(sg, su, q8, FS, T, st);   // SwiGLU and its q8_1 image in one launch
                il_stale(q8);
            }
            mmq(L.sh_down, q8, sy, T);
            lora_site(sg, su, FS, FS, T, {{L.sh_down.lora, sy, N, N}});
        } else {
            apply(L.sh_gate, xq, sg, T);
            apply(L.sh_up, xq, su, T);
            lora_site(h, nullptr, N, N, T, {{L.sh_gate.lora, sg, FS, FS}, {L.sh_up.lora, su, FS, FS}});
            k_swiglu<<<grid1((int64_t) T * FS), kThreads, 0, st>>>(sg, su, shh, (int64_t) T * FS);
            mm(L.sh_down, shh, sy, T);
            lora_site(shh, nullptr, FS, FS, T, {{L.sh_down.lora, sy, N, N}});
        }
        if (!ext) f32mm(L.sh_gate_inp, h, sgate, N, 1, T);
        if (hy_rows) moe_hybrid_cpu(li, T);   // the CPU pool's experts (while the GPU runs the rest), merged
        // the combine, the shared expert's gate, the residual add and the next norm in one launch
        moe_end(T, prow, sy, ext ? logits_r + g.n_expert : sgate, wn, hout, ext ? (int) g.n_expert + 1 : 1);
    }

    /// A qwen35 layer's dense FFN over h: x += down(silu(gate . h) * (up . h)), then h = rms(x) * wn (none: a stage's
    /// last layer hands x over).
    void dense_ffn(const Layer& L, int T, const float* wn, float* hout) {
        const int N = (int) m->g.n_embd, FF = (int) m->g.n_ff;
        if (T <= kGemmT) {
            if (!xq_post) quant(h, N, T, xq);
            apply(L.ffn_gate, xq, sg, T);
            apply(L.ffn_up, xq, su, T);
            lora_site(h, nullptr, N, N, T, {{L.ffn_gate.lora, sg, FF, FF}, {L.ffn_up.lora, su, FF, FF}});
            K::native_swiglu_quantize_q8_1(sg, su, q8, FF, T, st);   // SwiGLU and its q8_1 image in one launch
            il_stale(q8);   // SwiGLU(gate, up), as the product
            mmq(L.ffn_down, q8, sy, T);
            lora_site(sg, su, FF, FF, T, {{L.ffn_down.lora, sy, N, N}});   // SwiGLU(gate, up), as the product
        } else {
            prep(h, N, T, xq);
            apply(L.ffn_gate, xq, sg, T);
            apply(L.ffn_up, xq, su, T);
            lora_site(h, nullptr, N, N, T, {{L.ffn_gate.lora, sg, FF, FF}, {L.ffn_up.lora, su, FF, FF}});
            k_swiglu<<<grid1((int64_t) T * FF), kThreads, 0, st>>>(sg, su, shh, (int64_t) T * FF);
            mm(L.ffn_down, shh, sy, T);
            lora_site(shh, nullptr, FF, FF, T, {{L.ffn_down.lora, sy, N, N}});
        }
        xq_ready = false;
        k_add_rms<<<T, kThreads, 0, st>>>(x, sy, wn, hout, N, m->eps);
    }

    /// The main model over T rows (device `tok`, positions p0 .. p0 + T - 1, also at *pos0): hmain gets every row's
    /// final normed hidden; `heads` rows from the end get logits (rows [T - heads, T): logits row r - (T - heads)) and
    /// their greedy picks (`pick`).  `verify`: a verify window (the GDN state left for commit()).
    void forward(int T, int64_t p0, int heads, bool graph_mode, bool verify) {
        const auto& g = m->g;
        const int N = (int) g.n_embd;
        if (streamed(T)) {   // (not a window of more rows than a small --hybrid-rows: its layers take no stream)
            std::vector<int> seq;
            for (size_t l = l0; l < l1; ++l)
                if (lhy(l)) seq.push_back((int) l);
            stream_begin(seq);
        }
        if (l0 == 0) {
            K::iq_embed_rows(m->embd_type, m->embd, m->embd_row, m->embd_rows, tok, T, N, x, st);
            if (map_on) k_put_rows<<<T, kThreads, 0, st>>>(x, d_img, d_map, N);   // a picture's cells take its rows
        }
        // the stage's first layer's norm (later layers: from k_moe_out; a later stage: over the residual handed in), its
        // q8_1 image made by the layer
        k_rms<<<T, kThreads, 0, st>>>(x, m->layers[l0].attn_norm, h, N, m->eps);
        xq_ready = false;
        for (size_t l = l0; l < l1; ++l) {
            const bool last = l + 1 == m->layers.size();
            // the next layer's norm fused into this one's end; a stage's last layer hands x over (no norm: the next GPU's)
            const float* wn = last ? m->out_norm : l + 1 < l1 ? m->layers[l + 1].attn_norm : nullptr;
            layer(m->layers[l], l, T, p0, pos0, graph_mode, verify, wn, last ? hmain : h);
        }
        if (heads > 0 && l1 == m->layers.size()) head_rows(hmain + (size_t) (T - heads) * N, heads, logits, pick);
    }
    /// The main model over every stage (one stage: its forward): each stage's layers - in the stage's graph that `gx`
    /// picks when `graph` - then the residual x copied to the next stage's GPU.
    template <class Gx> void pass(Gx gx, bool graph, int T, int64_t p0, int heads, bool verify) {
        for (Engine* e = this; e != nullptr; e = e->nxt) {
            e->on();
            auto body = [&] { e->forward(T, p0, heads, graph && e->use_graph, verify); };
            if (graph) e->graph_run(gx(*e), body);
            else body();
            if (e->nxt != nullptr) {
                CK(cudaEventRecord(e->ev_hand, e->st));
                e->nxt->on();
                CK(cudaStreamWaitEvent(e->nxt->st, e->ev_hand, 0));
                CK(cudaMemcpyPeerAsync(e->nxt->x, e->nxt->dev, e->x, e->dev, (size_t) T * m->g.n_embd * 4, e->nxt->st));
            }
        }
        on();
    }
    /// A prompt chunk over every stage, nothing waited for on the host: each stage's layers, then the residual into the
    /// next stage's input `slot` (chunks alternate the two), which that stage copies into x before its layers - so a stage
    /// runs the next chunk while the one after it runs this one.
    void pass_chunk(int T, int64_t p0, int heads, int slot) {
        const size_t bytes = (size_t) T * m->g.n_embd * 4;
        for (Engine *e = this, *prev = nullptr; e != nullptr; prev = e, e = e->nxt) {
            e->on();
            if (prev != nullptr) {
                CK(cudaStreamWaitEvent(e->st, prev->ev_out[slot], 0));
                CK(cudaMemcpyAsync(e->x, e->x_in[slot], bytes, cudaMemcpyDeviceToDevice, e->st));
                CK(cudaEventRecord(e->ev_free[slot], e->st));
            }
            e->forward(T, p0, heads, false, false);
            if (Engine* n = e->nxt) {   // the input the next stage copied two chunks ago is free again
                CK(cudaStreamWaitEvent(e->st, n->ev_free[slot], 0));
                CK(cudaMemcpyPeerAsync(n->x_in[slot], n->dev, e->x, e->dev, bytes, e->st));
                CK(cudaEventRecord(e->ev_out[slot], e->st));
            }
        }
        on();
    }
    /// logits [n][V] = W_out hn rows, and each row's greedy pick.
    void head_rows(const float* hn, int n, float* lg, int32_t* pk) {
        const auto& g = m->g;
        mm(m->head, hn, lg, n);
        lora_site(hn, nullptr, (int) g.n_embd, (int) g.n_embd, n, {{m->head.lora, lg, (int) g.n_vocab, (int) g.n_vocab}});
        k_argmax<<<dim3(kArgBlocks, n), kThreads, 0, st>>>(lg, (int) g.n_vocab, pk, amv, ami, false);
        k_argmax<<<dim3(1, n), kThreads, 0, st>>>(lg, kArgBlocks, pk, amv, ami, true);
    }

    /// The draft head: the MTP block's pick over the first `draft_n` rows of the output matrix only (the lowest token ids are
    /// the most frequent ones).  A draft only proposes: the verify window decides, so a token outside the subset is a
    /// rejected draft and nothing else - the answer is the plain greedy answer whatever draft_n is.  0 = the whole vocabulary.
    int draft_n = 0;
    // --spec-min-p: a draft whose probability under the draft head (p_draft, from head_draft) is below this ends the
    // step's drafts before it (0: every step makes --drafts-per-step drafts)
    float draft_min_p = 0.f;
    float* p_draft = nullptr;
    void head_draft(const float* hn) {
        const auto& g = m->g;
        const int N = (int) g.n_embd;
        const Mat& hd = m->head;
        if (draft_n <= 0 || draft_n >= g.n_vocab) {
            head_rows(hn, 1, logits_m, pick_m);
            if (draft_min_p > 0.f) k_pmax<<<1, 1024, 0, st>>>(logits_m, (int) g.n_vocab, p_draft);
            return;
        }   // (B's first draft_n rows)
        quant(hn, N, 1, q8);
        if (fast_k && draft_n % 8 == 0 && hd.type == 8 && hd.n_in == N && N / 32 == 64) {
            q36k::launch_q8_rows<1, 2, false>(hd.d, nullptr, q8, logits_m, 64, draft_n, st);
        } else {   // the head's first draft_n rows: the matrix is row-major, so they are its first bytes
            Mat sub = hd;
            sub.n_out = draft_n;
            mmq(sub, q8, logits_m, 1);
        }
        lora_site(hn, nullptr, N, N, 1, {{hd.lora, logits_m, draft_n, draft_n}});   // (B's first draft_n rows)
        k_argmax<<<dim3(kArgBlocks, 1), kThreads, 0, st>>>(logits_m, draft_n, pick_m, amv, ami, false);
        k_argmax<<<dim3(1, 1), kThreads, 0, st>>>(logits_m, kArgBlocks, pick_m, amv, ami, true);
        if (draft_min_p > 0.f) k_pmax<<<1, 1024, 0, st>>>(logits_m, draft_n, p_draft);
    }

    /// The MTP block over T rows: row t reads token tok_m[t] at position p0 + t (also at *pos0_m) and the seed hidden
    /// seeds[t] (the main model's normed hidden of the position before, or the block's own for a chained draft);
    /// hmtp gets each row's normed output.  `draft`: logits of the last row and its pick (`pick_m`).
    void forward_mtp(int T, int64_t p0, const float* seeds, bool draft, bool graph_mode) {
        const auto& g = m->g;
        const int N = (int) g.n_embd;
        if (streamed(T) && lhy(m->layers.size())) stream_begin({(int) m->layers.size()});
        K::iq_embed_rows(m->embd_type, m->embd_mtp, m->embd_row, m->embd_rows, tok_m, T, N, y, st);
        if (map_on) k_put_rows<<<T, kThreads, 0, st>>>(y, d_img, d_map, N);
        k_rms_ld<<<T, kThreads, 0, st>>>(y, N, m->mtp_enorm, mtp_cat, 2 * N, N, m->eps);              // [e | ...]
        k_rms_ld<<<T, kThreads, 0, st>>>(seeds, N, m->mtp_hnorm, mtp_cat + N, 2 * N, N, m->eps);      // [... | h]
        mm(m->eh_proj, mtp_cat, x, T);
        lora_site(mtp_cat, nullptr, 2 * N, 2 * N, T, {{m->eh_proj.lora, x, N, N}});
        k_rms<<<T, kThreads, 0, st>>>(x, m->mtp.attn_norm, h, N, m->eps);
        layer(m->mtp, m->layers.size(), T, p0, pos0_m, graph_mode, false, m->mtp_shnorm, hmtp);
        if (draft) head_draft(hmtp + (size_t) (T - 1) * N);
    }

    /// After a verify window of T rows: the GDN state takes the first n_keep rows (conv history and recurrence run
    /// again over the rows' kept inputs).
    void commit(int n_keep) {
        for_each([&](Engine& e) { e.commit_own(n_keep); });
    }
    void commit_own(int n_keep) {
        const auto& g = m->g;
        const int HK = (int) g.ssm_k_heads, HV = (int) g.ssm_v_heads, C = (int) g.ssm_conv_channels;
        for (size_t l = l0; l < l1; ++l) {
            const Layer& L = m->layers[l];
            if (L.attn) continue;
            k_conv_seq<<<C / 128, 128, 0, st>>>(conv[l], v_qkv[l], L.conv, hc, C, 2 * HK, m->eps, n_keep, true);
            k_gdn_seq<<<HV, dim3(128, 4), 0, st>>>(S[l], v_hc[l], C, v_alpha[l], v_beta[l], z, L.ssm_norm, m->eps,
                                                   nullptr, HK, HV, n_keep, true);
        }
    }

    // ---- the hybrid mode
    const Layer& layer_at(int li) const { return li < (int) m->layers.size() ? m->layers[(size_t) li] : m->mtp; }

    /// A streamed forward over the state indices `seq`: the first two layers' experts start copying now.
    void stream_begin(const std::vector<int>& seq) {
        stream_seq = seq;
        stream_pos = 0;
        for (int i = 0; i < 2 && i < (int) seq.size(); ++i) stream_kick(i);
    }
    void stream_kick(int i) {   // the i-th layer of the sequence into staging buffer i % 2, after its last reader
        const int b = i % 2;
        const Layer& L = layer_at(stream_seq[(size_t) i]);
        CK(cudaStreamWaitEvent(cst, ev_used[b], 0));
        // the experts with a VRAM slot come from it (device to device, at VRAM speed), only the others over PCIe; runs
        // of consecutive experts from the same source in one copy each.  LAMINA_Q36_STREAM_ALL=1: the whole layer from
        // host memory (the first version: PCIe-bound)
        static const bool all = std::getenv("LAMINA_Q36_STREAM_ALL") != nullptr;
        const int ne = (int) m->g.n_expert;
        const Hy& H = hy[(size_t) stream_seq[(size_t) i]];
        // fewer than half the experts resident (a 12 GB card): the one whole-layer copy is faster than many small ones
        // (RTX 3060, UD-Q4_K_XL at 37 of 256: pp2048 1970 whole vs 1540 in runs)
        if (all || 2 * L.n_slots < ne) {
            CK(cudaMemcpyAsync(stage[b], L.hblobs, L.stride * (size_t) ne, cudaMemcpyHostToDevice, cst));
        } else {
            for (int ea = 0; ea < ne;) {
                const bool held = H.slot_of[(size_t) ea] >= 0;
                int eb = ea + 1;
                if (held)   // consecutive slots holding consecutive experts
                    while (eb < ne && H.slot_of[(size_t) eb] == H.slot_of[(size_t) eb - 1] + 1) ++eb;
                else
                    while (eb < ne && H.slot_of[(size_t) eb] < 0) ++eb;
                const size_t n = (size_t) (eb - ea) * L.stride;
                if (held)
                    CK(cudaMemcpyAsync(stage[b] + (size_t) ea * L.stride, L.blobs + (size_t) H.slot_of[(size_t) ea] * L.stride,
                                       n, cudaMemcpyDeviceToDevice, cst));
                else
                    CK(cudaMemcpyAsync(stage[b] + (size_t) ea * L.stride, L.hblobs + (size_t) ea * L.stride, n,
                                       cudaMemcpyHostToDevice, cst));
                ea = eb;
            }
        }
        CK(cudaEventRecord(ev_copy[b], cst));
    }
    const uint8_t* stream_take(int li) {
        if (stream_pos >= (int) stream_seq.size() || stream_seq[(size_t) stream_pos] != li) {
            std::fprintf(stderr, "lamina: the expert stream is out of step (layer %d)\n", li);
            throw Fatal{1};
        }
        const int b = stream_pos % 2;
        CK(cudaStreamWaitEvent(st, ev_copy[b], 0));
        return stage[b];
    }
    void stream_done() {
        const int b = stream_pos % 2;
        CK(cudaEventRecord(ev_used[b], st));
        if (stream_pos + 2 < (int) stream_seq.size()) stream_kick(stream_pos + 2);
        ++stream_pos;
    }

    /// The GPU half of a hybrid MoE: the routing to the host, the resident experts' entries grouped and run.
    void moe_hybrid_gpu(const Layer& L, size_t li, int T) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, Kp = (int) m->top_k;
        const int64_t n_ent = (int64_t) T * Kp;
        CK(cudaMemcpyAsync(h_ids, ids, (size_t) n_ent * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaMemcpyAsync(h_h, h, (size_t) T * N * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaEventRecord(ev_route, st));
        group_entries(L, hy[li].slot_of_d, T);
        if (!xq_post) quant(h, N, T, xq);   // the experts' and the shared expert's input
        K::native_expert_grouped(L.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
                                 std::min<int64_t>(n_ent, g.n_expert), n_ent, xq, (int64_t) (N / 32) * 36, escr, eparts,
                                 st);
    }
    /// The CPU half: the entries without a slot on the pool (ggml-cpu's arithmetic for the layer's formats), their
    /// rows uploaded and merged into the entries' rows.
    void moe_hybrid_cpu(size_t li, int T) {
#if defined(LAMINA_Q36_CPU_POOL)
        namespace C = lamina::kernels::cpu;
        const auto& g = m->g;
        const int N = (int) g.n_embd, Kp = (int) m->top_k;
        const int n_ent = T * Kp;
        Hy& H = hy[li];
        const Layer& L = layer_at((int) li);
        while (cudaEventQuery(ev_route) == cudaErrorNotReady) {
        }
        const auto t0 = Clock::now();
        // the missed entries per expert (up to MAXT tokens per job)
        jobs.clear();
        std::vector<char> quantized((size_t) T, 0);
        int n_miss = 0;
        std::map<int, size_t> open;   // expert -> its job with room
        const bool defer = rows_deferred && li < m->layers.size();
        if (defer) {
            if (row_ids.size() < m->layers.size()) row_ids.resize(m->layers.size());
            row_ids[li].assign(h_ids, h_ids + n_ent);
        }
        for (int e = 0; e < n_ent; ++e) {
            const int ex = h_ids[e];
            if (!defer) H.score[(size_t) ex] += 1.0;
            h_miss[e] = H.slot_of[(size_t) ex] < 0;
            if (!h_miss[e]) continue;
            ++n_miss;
            const int t = e / Kp;
            if (!quantized[(size_t) t]) {
                C::native_quant_act(H.fmt, h_h + (size_t) t * N, acts + (size_t) t * C::kNativeActBytes);
                quantized[(size_t) t] = 1;
            }
            auto it = open.find(ex);
            if (it == open.end() || jobs[it->second].nt == C::MAXT) {
                C::ExpertJobMulti j;
                j.blob = L.hblobs + (size_t) ex * L.stride;
                jobs.push_back(j);
                it = open.insert_or_assign(ex, jobs.size() - 1).first;
            }
            C::ExpertJobMulti& j = jobs[it->second];
            j.nact[j.nt] = acts + (size_t) t * C::kNativeActBytes;
            j.out[j.nt] = cpu_out + (size_t) e * N;
            ++j.nt;
        }
        hy_cpu += n_miss;
        hy_gpu += n_ent - n_miss;
        if (n_miss == 0) return;
        pool->run_split_multi_native(H.fmt, jobs.data(), (int) jobs.size());
        hy_cpu_ms += ms_since(t0);
        CK(cudaMemcpyAsync(ecpu, cpu_out, (size_t) n_ent * N * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_miss, h_miss, (size_t) n_ent * 4, cudaMemcpyHostToDevice, st));
        k_merge_cpu<<<n_ent, kThreads, 0, st>>>(eparts, ecpu, d_miss, N);
#else
        (void) li;
        (void) T;
#endif
    }

    /// The slots to the experts the routing used most (the counts halved each time): at most max_swaps replacements
    /// over all layers, between forwards (the copies are on the engine's stream).
    void rebalance(int max_swaps) {
        static const bool off = std::getenv("LAMINA_Q36_FIXED_SLOTS") != nullptr;   // the slots as placed (tests)
        if (!m->hybrid || off || warming) return;
        const int64_t ne = m->g.n_expert;
        int left = max_swaps;
        for (size_t li = 0; li < hy.size() && left > 0; ++li) {
            Hy& H = hy[li];
            const Layer& L = layer_at((int) li);
            const int n_slots = L.n_slots;
            if (n_slots <= 0 || n_slots >= ne) continue;
            std::vector<int> order((size_t) ne);
            for (int ex = 0; ex < ne; ++ex) order[(size_t) ex] = ex;
            std::sort(order.begin(), order.end(), [&](int a, int b) { return H.score[(size_t) a] > H.score[(size_t) b]; });
            // residents outside the top n_slots, lowest score first: the slots to give away
            std::vector<int> victims;
            for (int s = 0; s < n_slots; ++s) victims.push_back(s);
            std::sort(victims.begin(), victims.end(), [&](int a, int b) {
                return H.score[(size_t) H.exp_in_slot[(size_t) a]] < H.score[(size_t) H.exp_in_slot[(size_t) b]];
            });
            size_t vi = 0;
            bool changed = false;
            for (int i = 0; i < n_slots && left > 0; ++i) {
                const int ex = order[(size_t) i];
                if (H.slot_of[(size_t) ex] >= 0) continue;
                while (vi < victims.size()) {
                    const int s = victims[vi];
                    const int ey = H.exp_in_slot[(size_t) s];
                    if (H.score[(size_t) ey] >= H.score[(size_t) ex]) { vi = victims.size(); break; }
                    // ey must not be in the top n_slots itself
                    ++vi;
                    H.slot_of[(size_t) ey] = -1;
                    H.slot_of[(size_t) ex] = s;
                    H.exp_in_slot[(size_t) s] = ex;
                    CK(cudaMemcpyAsync(L.blobs + (size_t) s * L.stride, L.hblobs + (size_t) ex * L.stride, L.L.bytes,
                                       cudaMemcpyHostToDevice, st));
                    --left;
                    ++hy_swaps;
                    changed = true;
                    break;
                }
                if (vi >= victims.size()) break;
            }
            if (changed)
                CK(cudaMemcpyAsync(H.slot_of_d, H.slot_of.data(), (size_t) ne * 4, cudaMemcpyHostToDevice, st));
            for (double& v : H.score) v *= 0.5;
        }
        CK(cudaStreamSynchronize(st));   // slot_of's host copy is pageable: the upload is done before it changes
    }


    template <class F> void graph_run(cudaGraphExec_t& gx, F body) {
        if (!use_graph) { body(); return; }
        if (!gx) {
            cudaGraph_t graph = nullptr;
            CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
            body();
            CK(cudaStreamEndCapture(st, &graph));
            CK(graph_new(&gx, graph));
            CK(cudaGraphDestroy(graph));
        }
        CK(cudaGraphLaunch(gx, st));
    }

    // ---- pictures (--vision).  A request with pictures hands the engine its rows (the encoder's, in prompt order), the
    // cell -> row map of its <|image_pad|> cells and the M-RoPE table of its prompt cells; the cells below the prompt
    // length take their rotary positions from the table, the text after it continues at cell + delta (what the pictures
    // did not use of their cells: llama.cpp's mtmd advances the text position by max(nx, ny) per picture).
    bool img_active = false;
    void set_images(const std::vector<float>& rows, const std::vector<int32_t>& cell_row, const std::vector<int32_t>& mrope,
                    int64_t n_prompt, int32_t delta) {
        for_each([&](Engine& e) { e.set_images_own(rows, cell_row, mrope, n_prompt, delta); });
    }
    void set_images_own(const std::vector<float>& rows, const std::vector<int32_t>& cell_row,
                        const std::vector<int32_t>& mrope, int64_t n_prompt, int32_t delta) {
        const size_t need = takes_rows() ? rows.size() : 0;
        if (need > img_cap) {
            CK(cudaDeviceSynchronize());
            if (d_img) CK(dfree(d_img));
            CK(dmalloc((void**) &d_img, need * 4));
            img_cap = need;
        }
        if (need > 0) CK(cudaMemcpy(d_img, rows.data(), need * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_mtab, mrope.data(), mrope.size() * 4, cudaMemcpyHostToDevice));
        img_map = cell_row;
        img_prompt_cells = n_prompt;
        img_active = true;
        set_delta_own(delta);
    }
    void clear_images() {
        for_each([](Engine& e) { e.clear_images_own(); });
    }
    void clear_images_own() {
        img_map.clear();
        img_active = false;
        img_prompt_cells = 0;
        mtab_now = nullptr;
        map_on = false;
        set_delta_own(0);
    }
    /// every stage's pictures for a forward over [c0, c0 + T), and none after it
    void pictures(int64_t c0, int T) {
        for_each([&](Engine& e) { e.chunk_pictures(c0, T); });
    }
    void pictures_off() {
        for (Engine* e = this; e != nullptr; e = e->nxt) {
            e->mtab_now = nullptr;
            e->map_on = false;
        }
    }
    /// A forward over the cells [c0, c0 + T): the picture rows of its cells, and the table when it is a prompt's.
    void chunk_pictures(int64_t c0, int T) {
        mtab_now = nullptr;
        map_on = false;
        if (!img_active) return;
        if (c0 < img_prompt_cells) mtab_now = d_mtab;
        bool any = false;
        for (int j = 0; j < T; ++j) {
            const int64_t c = c0 + j;
            const int32_t v = c < (int64_t) img_map.size() ? img_map[(size_t) c] : -1;
            h_map[j] = v;
            any = any || v >= 0;
        }
        if (any && takes_rows()) {
            CK(cudaMemcpyAsync(d_map, h_map, (size_t) T * 4, cudaMemcpyHostToDevice, st));
            map_on = true;
        }
    }

    /// The tokens at the current position: a prompt in chunks (eager), one token from the decode graph.  The last
    /// token's logits stay in `logits` (its greedy pick in `pick`), hlast its normed hidden.  With the MTP block the
    /// prompt also runs through it (its K / V for every position: row q reads h_{q-1} and token q).  on_chunk(done) runs
    /// after every chunk; false ends the read there (a cancelled request): the tokens processed are returned.  `plan`
    /// (the conversation cache's): asked once for the chunk that starts at each token, it gives the chunk's rows (tmax
    /// at most: a chunk ends where a checkpoint is taken from its callback) and a checkpoint inside it (Chunk).
    struct Chunk {
        int T = 0;                 // its rows
        int snap = 0;              // > 0: the state after its first `snap` rows is a checkpoint ...
        uint8_t* host = nullptr;   // ... which goes into this buffer of state_bytes()
        bool pinned = true;        // (a pageable one: through the engine's pinned buffer)
    };
    template <class F>
    int64_t process(const int32_t* toks, int64_t n, F on_chunk_, const std::function<Chunk(int64_t)>* plan = nullptr) {
        const int N = (int) m->g.n_embd;
        auto chunk_at = [&](int64_t i) {   // the chunk that starts at token i
            const int most = (int) std::min<int64_t>(tmax, n - i);
            Chunk c = plan != nullptr ? (*plan)(i) : Chunk{};
            if (c.T <= 0 || c.T > most) c = Chunk{most};
            if (snap == nullptr || c.host == nullptr || c.snap <= 0 || c.snap >= c.T) c.snap = 0;
            return c;
        };
        lora_rows(nullptr, 0);   // the current session's adapters
        auto on_chunk = [&](int64_t done) -> bool {   // a callback returning void never cancels
            if constexpr (std::is_void_v<decltype(on_chunk_(done))>) {
                on_chunk_(done);
                return true;
            } else {
                return on_chunk_(done);
            }
        };
        Engine& L = *lst;   // the head, the MTP block and hlast are the last stage's
        auto g1 = [](Engine& e) -> cudaGraphExec_t& { return e.gs().g1; };
        if (n == 1) {
            sync_all();   // the pinned staging is free again
            hpin[0] = toks[0];
            CK(cudaMemcpyAsync(tok, hpin, 4, cudaMemcpyHostToDevice, st));
            put_pos(pos);
            pictures(pos, 1);
            rows_deferred = true;
            // a prompt cell of a picture request (its rotary position is in the table, its row a picture's) runs eagerly
            bool eager = false;
            for (Engine* e = this; e != nullptr; e = e->nxt) eager = eager || e->mtab_now || e->map_on;
            pass(g1, !eager, 1, pos, 1, false);
            pictures_off();
            rows_deferred = false;
            L.on();
            CK(cudaMemcpyAsync(L.hlast, L.hmain, (size_t) N * 4, cudaMemcpyDeviceToDevice, L.st));
            on();
            advance(1);
            after_rows(1);
            on_chunk(1);
            return 1;
        }
        static const bool serial = std::getenv("LAMINA_Q36_SERIAL_CHUNKS") != nullptr;   // A/B: the stages in turn
        const Chunk first = chunk_at(0);
        if (split && first.T < n && !img_active && !serial) {
            // several chunks over a layer split, no picture: the stages overlap (pass_chunk); a chunk goes in once the
            // callback of the one before is done, so a cancel leaves every stage at the same chunk
            pictures_off();
            sync_all();
            int64_t i = 0;
            for (int slot = 0; i < n; slot ^= 1) {
                const Chunk ck = i == 0 ? first : chunk_at(i);
                const int T = ck.T;
                CK(cudaMemcpyAsync(tok, toks + i, (size_t) T * 4, cudaMemcpyHostToDevice, st));
                for_each([&](Engine& e) { k_set_i32<<<1, 1, 0, e.st>>>(e.pos0, (int32_t) pos); });
                snap_rows(ck.snap);
                pass_chunk(T, pos, i + T == n ? 1 : 0, slot);
                L.on();
                if (m->has_mtp && mtp_d > 0) {
                    CK(cudaMemcpyAsync(L.mtp_seed(), L.hlast, (size_t) N * 4, cudaMemcpyDeviceToDevice, L.st));
                    if (T > 1)
                        CK(cudaMemcpyAsync(L.mtp_seed() + N, L.hmain, (size_t) (T - 1) * N * 4, cudaMemcpyDeviceToDevice, L.st));
                    CK(cudaMemcpyAsync(L.tok_m, toks + i, (size_t) T * 4, cudaMemcpyHostToDevice, L.st));
                    k_set_i32<<<1, 1, 0, L.st>>>(L.pos0_m, (int32_t) pos);
                    L.forward_mtp(T, pos, L.mtp_seed(), false, false);
                }
                CK(cudaMemcpyAsync(L.hlast, L.hmain + (size_t) (T - 1) * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, L.st));
                on();
                if (ck.snap > 0) snap_send(ck.host, ck.snap, ck.pinned);
                advance(T);
                i += T;
                if (!on_chunk(i)) break;
            }
            sync_all();
            return i;
        }
        for (int64_t i = 0; i < n;) {
            const Chunk ck = i == 0 ? first : chunk_at(i);
            const int T = ck.T;
            sync_all();
            CK(cudaMemcpyAsync(tok, toks + i, (size_t) T * 4, cudaMemcpyHostToDevice, st));
            put_pos(pos);
            pictures(pos, T);
            snap_rows(ck.snap);
            pass(g1, false, T, pos, i + T == n ? 1 : 0, false);
            L.on();
            if (m->has_mtp && mtp_d > 0) {
                // seeds: h_{q-1} for each row q: hlast, then this chunk's rows 0 .. T - 2 (in `y`, free here)
                CK(cudaMemcpyAsync(L.mtp_seed(), L.hlast, (size_t) N * 4, cudaMemcpyDeviceToDevice, L.st));
                if (T > 1)
                    CK(cudaMemcpyAsync(L.mtp_seed() + N, L.hmain, (size_t) (T - 1) * N * 4, cudaMemcpyDeviceToDevice, L.st));
                CK(cudaMemcpyAsync(L.tok_m, toks + i, (size_t) T * 4, cudaMemcpyHostToDevice, L.st));
                CK(cudaMemcpyAsync(L.pos0_m, L.hpin + 1, 4, cudaMemcpyHostToDevice, L.st));
                L.forward_mtp(T, pos, L.mtp_seed(), false, false);
            }
            CK(cudaMemcpyAsync(L.hlast, L.hmain + (size_t) (T - 1) * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, L.st));
            on();
            if (ck.snap > 0) snap_send(ck.host, ck.snap, ck.pinned);
            advance(T);
            sync_all();
            pictures_off();
            i += T;
            if (!on_chunk(i)) {
                if (m->hybrid) rebalance(hybrid_swaps_after_prompt);
                return i;
            }
        }
        if (m->hybrid) rebalance(hybrid_swaps_after_prompt);
        return n;
    }

    // ---- batch rows: one decode window for the next token of several sessions
    struct Row { int slot; int32_t tok; };
    long long win_count = 0;
    std::map<int, cudaGraphExec_t> row_graphs;   // by (rows, groups, kind): graphs of the batch windows
    /// A slot's rows of one window: `n` consecutive cells from the session's position (1: the next token; 2: also an MTP
    /// proposal the window verifies).
    struct Group { int slot; int n; int32_t tok[2]; };
    int32_t rows_pick[kVerify] = {};
    int rows_G_last = 0;   // groups of the window run last (commit_rows takes their keep counts)

    void run_graph_key(int key, const std::function<void()>& body) {
        if (!use_graph) { body(); return; }
        graph_run(row_graphs[key + (lora_live ? 100000 : 0)], body);   // (the LoRA kernels: graphs of their own)
    }

    /// One window over the groups' rows.  `verify`: a verify window - the GDN state is read, not written (commit_rows writes
    /// it for the kept rows) and the sessions' positions stay.  The greedy picks end up in rows_pick (one per row), every
    /// row's logits in `logits`.  Without verify the sessions' positions advance by their rows.
    void run_rows(const Group* gr, int G, bool verify) {
        sess[(size_t) cur].pos = pos;
        CK(cudaStreamSynchronize(st));   // the pinned staging is free again
        int64_t pmax = 0;
        int T = 0;
        int row_sess[kVerify];
        for (int g = 0; g < G; ++g) {
            const Sess& ss = sess[(size_t) gr[g].slot];
            hrow[40 + g] = gr[g].slot;
            hrow[48 + g] = T;
            hrow[56 + g] = gr[g].n;
            for (int j = 0; j < gr[g].n; ++j) {
                hrow[T] = gr[g].tok[j];
                hrow[8 + T] = (int32_t) (ss.pos + j);
                hrow[16 + T] = gr[g].slot;
                hrow[24 + T] = ss.rdelta;
                row_sess[T] = gr[g].slot;
                ++T;
            }
            pmax = std::max<int64_t>(pmax, ss.pos + gr[g].n - 1);
        }
        lora_rows(row_sess, T);   // each row its own request's adapters
        CK(cudaMemcpyAsync(tok, hrow, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_rcell, hrow + 8, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_rslot, hrow + 16, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_rdelta, hrow + 24, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_gslot, hrow + 40, (size_t) G * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_grow0, hrow + 48, (size_t) G * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_gcnt, hrow + 56, (size_t) G * 4, cudaMemcpyHostToDevice, st));
        mtab_now = nullptr;
        map_on = false;
        rows_mode = true;
        rows_G = G;
        rows_deferred = true;
        run_graph_key((verify ? 1000 : 0) + G * 10 + (T - G), [&] { forward(T, pmax, T, use_graph, verify); });
        rows_deferred = false;
        rows_mode = false;
        CK(cudaMemcpyAsync(hrow + 32, pick, (size_t) T * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        for (int r = 0; r < T; ++r) rows_pick[r] = hrow[32 + r];
        if (!verify)
            for (int g = 0; g < G; ++g) sess[(size_t) gr[g].slot].pos += gr[g].n;
        pos = sess[(size_t) cur].pos;
        rows_G_last = G;
        if (m->hybrid) {
            count_rows(T);
            if (++win_count % kRebalanceEvery == 0) rebalance(8);
        }
    }
    /// One window for the next token of each of the rows' sessions (no proposals): the greedy picks in rows_pick.
    void step_rows(const Row* rows, int nb) {
        Group gr[kVerify];
        for (int r = 0; r < nb; ++r) gr[r] = {rows[r].slot, 1, {rows[r].tok, 0}};
        run_rows(gr, nb, false);
    }
    /// After run_rows(.., verify = true): the GDN state of group g takes its first keep[g] rows (the conv history and the
    /// recurrence run again over the rows' kept inputs, as commit() does) and the session advances by keep[g].
    void commit_rows(const Group* gr, const int* keep, int G) {
        const auto& g = m->g;
        const int HK = (int) g.ssm_k_heads, HV = (int) g.ssm_v_heads, C = (int) g.ssm_conv_channels;
        sess[(size_t) cur].pos = pos;
        for (int i = 0; i < G; ++i) hrow[56 + i] = keep[i];   // (hrow's slot / first-row words are the window's, unchanged)
        CK(cudaMemcpyAsync(d_gcnt, hrow + 56, (size_t) G * 4, cudaMemcpyHostToDevice, st));
        rows_mode = true;
        rows_G = G;
        run_graph_key(2000 + G, [&] {
            for (size_t l = 0; l < m->layers.size(); ++l) {
                const Layer& L = m->layers[l];
                if (L.attn) continue;
                k_conv_seq<<<dim3(C / 128, G), 128, 0, st>>>(nullptr, v_qkv[l], L.conv, hc, C, 2 * HK, m->eps, 1, true,
                                                             d_conv_tab + l * n_sess, d_gslot, d_grow0, d_gcnt);
                k_gdn_seq<<<dim3(HV, G), dim3(128, 4), 0, st>>>(nullptr, v_hc[l], C, v_alpha[l], v_beta[l], z, L.ssm_norm,
                                                                m->eps, nullptr, HK, HV, 1, true, d_S_tab + l * n_sess,
                                                                d_gslot, d_grow0, d_gcnt);
            }
        });
        rows_mode = false;
        CK(cudaStreamSynchronize(st));
        for (int i = 0; i < G; ++i) sess[(size_t) gr[i].slot].pos += keep[i];
        pos = sess[(size_t) cur].pos;
    }

    /// The MTP block over rows of several sessions: row r reads token tok at cell `cell` of session `slot` and the seed
    /// hidden at `seed` (device, n_embd floats); its draft - the block's greedy pick after that row - ends up in
    /// mtp_pick[r], and the block's K / V cells are written.
    struct MtpRow { int slot; int32_t tok; int64_t cell; const float* seed; };
    int32_t mtp_pick[kVerify] = {};
    float* logits_mr = nullptr;
    int32_t* pick_mr = nullptr;
    void mtp_rows(const MtpRow* rows, int T) {
        const auto& g = m->g;
        const int N = (int) g.n_embd;
        if (!logits_mr) {
            logits_mr = dalloc((size_t) kVerify * g.n_vocab);
            pick_mr = dalloc_t<int32_t>(kVerify);
        }
        sess[(size_t) cur].pos = pos;
        CK(cudaStreamSynchronize(st));
        int64_t pmax = 0;
        int row_sess[kVerify];
        for (int r = 0; r < T; ++r) row_sess[r] = rows[r].slot;
        lora_rows(row_sess, T);
        for (int r = 0; r < T; ++r) {
            hrow[r] = rows[r].tok;
            hrow[8 + r] = (int32_t) rows[r].cell;
            hrow[16 + r] = rows[r].slot;
            hrow[24 + r] = sess[(size_t) rows[r].slot].rdelta;
            pmax = std::max<int64_t>(pmax, rows[r].cell);
            CK(cudaMemcpyAsync(mtp_seed() + (size_t) r * N, rows[r].seed, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
        }
        CK(cudaMemcpyAsync(tok_m, hrow, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_rcell, hrow + 8, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_rslot, hrow + 16, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_rdelta, hrow + 24, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        mtab_now = nullptr;
        map_on = false;
        rows_mode = true;
        rows_G = T;
        rows_deferred = true;
        // a hybrid MTP layer's host part (the CPU pool's experts) is no graph's: run as it is
        const bool graphed = use_graph && mtp_graph;
        auto body = [&] {
            K::iq_embed_rows(m->embd_type, m->embd, m->embd_row, m->embd_rows, tok_m, T, N, y, st);
            k_rms_ld<<<T, kThreads, 0, st>>>(y, N, m->mtp_enorm, mtp_cat, 2 * N, N, m->eps);                 // [e | ...]
            k_rms_ld<<<T, kThreads, 0, st>>>(mtp_seed(), N, m->mtp_hnorm, mtp_cat + N, 2 * N, N, m->eps);   // [... | h]
            mm(m->eh_proj, mtp_cat, x, T);
            lora_site(mtp_cat, nullptr, 2 * N, 2 * N, T, {{m->eh_proj.lora, x, N, N}});
            k_rms<<<T, kThreads, 0, st>>>(x, m->mtp.attn_norm, h, N, m->eps);
            layer(m->mtp, m->layers.size(), T, pmax, pos0_m, graphed, false, m->mtp_shnorm, hmtp);
            mm(m->head, hmtp, logits_mr, T);
            lora_site(hmtp, nullptr, N, N, T, {{m->head.lora, logits_mr, (int) g.n_vocab, (int) g.n_vocab}});
            k_argmax<<<dim3(kArgBlocks, T), kThreads, 0, st>>>(logits_mr, (int) g.n_vocab, pick_mr, amv, ami, false);
            k_argmax<<<dim3(1, T), kThreads, 0, st>>>(logits_mr, kArgBlocks, pick_mr, amv, ami, true);
        };
        if (graphed) run_graph_key(3000 + T, body);
        else body();
        rows_deferred = false;
        rows_mode = false;
        CK(cudaMemcpyAsync(hrow + 32, pick_mr, (size_t) T * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        for (int r = 0; r < T; ++r) mtp_pick[r] = hrow[32 + r];
    }
    /// Row r's logits of the last step_rows, on the host.
    void host_logits_row(int r, std::vector<float>& out) {
        out.resize((size_t) m->g.n_vocab);
        CK(cudaMemcpyAsync(out.data(), logits + (size_t) r * m->g.n_vocab, out.size() * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
    }

    /// Session dst takes session src's state: the recurrent state, the cache's first src.pos cells, the position, the
    /// pictures' offset (for a conversation read in one session that continues in another).
    void copy_session(int src, int dst) {
        for_each([&](Engine& e) { e.copy_session_own(src, dst); });
    }
    void copy_session_own(int src, int dst) {
        if (src == dst) return;
        sess[(size_t) cur].pos = pos;
        const auto& g = m->g;
        const Sess& a = sess[(size_t) src];
        Sess& b = sess[(size_t) dst];
        const size_t C = (size_t) g.ssm_conv_channels;
        const size_t row = (size_t) ctx * m->kv_row(), used = (size_t) a.pos * m->kv_row();   // a KV head's cells
        for (size_t l = 0; l < a.conv.size(); ++l) {
            if (a.conv[l]) {
                CK(cudaMemcpyAsync(b.conv[l], a.conv[l], C * (g.ssm_d_conv - 1) * 4, cudaMemcpyDeviceToDevice, st));
                CK(cudaMemcpyAsync(b.S[l], a.S[l], (size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size * 4,
                                   cudaMemcpyDeviceToDevice, st));
            }
            if (a.kc[l] && a.pos > 0) {
                CK(cudaMemcpy2DAsync(b.kc[l], row, a.kc[l], row, used, (size_t) g.n_head_kv, cudaMemcpyDeviceToDevice, st));
                CK(cudaMemcpy2DAsync(b.vc[l], row, a.vc[l], row, used, (size_t) g.n_head_kv, cudaMemcpyDeviceToDevice, st));
            }
        }
        CK(cudaMemcpyAsync(b.hlast, a.hlast, (size_t) g.n_embd * 4, cudaMemcpyDeviceToDevice, st));
        CK(cudaStreamSynchronize(st));
        b.pos = a.pos;
        b.rdelta = a.rdelta;
        b.live = a.live;
        b.imgs = a.imgs;
        b.lora = a.lora;
        if (dst == cur) load_cur();
    }

    // ---- the conversation cache's view of the current session (lamina/core/prompt_cache.hpp; stage 0 drives).  A
    // checkpoint is the recurrent state of each stage's layers and its hlast, stage after stage; a parked copy the first
    // cells of every attention layer's K and V (the MTP block's too), each KV head's cells back to back.  The K / V cells
    // are indexed by position, so the state of a checkpoint at `p` continues the sequence from there over the cells it
    // finds.  A copy that fails is said by its false (the cache drops what it was for): not fatal.
    static bool soft(cudaError_t e) {
        if (e == cudaSuccess) return true;
        (void) cudaGetLastError();
        return false;
    }
    /// The last stage's image ends with the engine's output at a prompt's end - the logits, the pick - and whether it
    /// holds one (two words): a checkpoint of a prompt's end serves the same prompt again with nothing to read.
    size_t out_bytes() const { return lst == this ? (size_t) m->g.n_vocab * 4 + 8 : 0; }
    uint64_t state_bytes_own() const {
        const auto& g = m->g;
        uint64_t n = (uint64_t) g.n_embd * 4 + out_bytes();
        for (size_t l = 0; l < conv.size(); ++l)
            if (conv[l])
                n += (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1) * 4 +
                     (uint64_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size * 4;
        return n;
    }
    uint64_t state_bytes() const {
        uint64_t n = 0;
        for (const Engine* e = this; e != nullptr; e = e->nxt) n += e->state_bytes_own();
        return n;
    }
    bool state_copy_own(uint8_t* host, bool to_host) {
        const auto& g = m->g;
        bool ok = true;
        auto part = [&](void* d, size_t n) {
            ok = ok && soft(to_host ? cudaMemcpyAsync(host, d, n, cudaMemcpyDeviceToHost, st)
                                    : cudaMemcpyAsync(d, host, n, cudaMemcpyHostToDevice, st));
            host += n;
        };
        for (size_t l = 0; l < conv.size(); ++l)
            if (conv[l]) {
                part(conv[l], (size_t) g.ssm_conv_channels * (g.ssm_d_conv - 1) * 4);
                part(S[l], (size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size * 4);
            }
        part(hlast, (size_t) g.n_embd * 4);
        if (out_bytes() > 0) {   // the last stage: an image loaded brings its output back; a state saved here has none
            const size_t V4 = (size_t) g.n_vocab * 4;
            int32_t held = 0;
            if (!to_host) std::memcpy(&held, host + V4 + 4, 4);
            if (held == 1) {
                part(logits, V4);
                part(pick, 4);
            }
            ok = soft(cudaStreamSynchronize(st)) && ok;
            if (to_host) std::memset(host + V4, 0, 8);
            return ok;
        }
        return soft(cudaStreamSynchronize(st)) && ok;   // (a pageable buffer's copy has landed too)
    }
    /// The current session's recurrent state into `host` (state_bytes), or from it (a prompt's end: with its output).
    bool state_copy(uint8_t* host, bool to_host) {
        bool ok = true;
        for (Engine* e = this; e != nullptr; e = e->nxt) {
            e->on();
            ok = e->state_copy_own(host, to_host) && ok;
            host += e->state_bytes_own();
        }
        on();
        return ok;
    }
    /// The chunk about to be queued takes a checkpoint after its first `r` rows (0: none; layer() writes every stage's
    /// image as the chunk passes that row).  An image whose copy of the checkpoint before is still on its way is
    /// waited for on the stage's stream, not on the host.
    void snap_rows(int r) {
        for (Engine* e = this; e != nullptr; e = e->nxt) {
            e->snap_row = r;
            if (r > 0) {
                e->on();
                CK(cudaStreamWaitEvent(e->st, e->ev_landed, 0));
            }
        }
        if (r > 0) on();
    }
    /// Every stage's image to `host`, the cache's buffer of state_bytes(): once the stage's stream has passed the work
    /// it holds now, on the stage's snapshot stream.  With `row` > 0 the images are a chunk's (snap_rows), and hlast's
    /// part is still to come: the hidden state of the row before, which the last stage has.  `end`: a prompt's end -
    /// the last stage's image takes the logits and the pick the read left.  A `host` that is not pinned gets the
    /// images through snap_via (snap_flush).
    bool snap_send(uint8_t* host, int row, bool pinned, bool end = false) {
        const size_t N4 = (size_t) m->g.n_embd * 4, V4 = (size_t) m->g.n_vocab * 4;
        bool ok = true;
        if (!pinned) {
            if (snap_to[end] != nullptr) snap_wait();   // (the one before is in its buffer first)
            snap_to[end] = host;
            host = snap_via[end];
        }
        for (Engine* e = this; e != nullptr; e = e->nxt) {
            e->on();
            uint8_t* const image = end ? e->snap_end : e->snap;
            const size_t bytes = (size_t) e->state_bytes_own(), out = bytes - e->out_bytes();   // (`out`: where its output starts)
            if (row > 0)
                ok = ok && soft(cudaMemcpyAsync(image + out - N4, e == lst ? e->hmain + (size_t) (row - 1) * m->g.n_embd : e->hlast,
                                                N4, cudaMemcpyDeviceToDevice, e->st));
            if (e->out_bytes() > 0) {
                if (end)
                    ok = ok && soft(cudaMemcpyAsync(image + out, e->logits, V4, cudaMemcpyDeviceToDevice, e->st)) &&
                         soft(cudaMemcpyAsync(image + out + V4, e->pick, 4, cudaMemcpyDeviceToDevice, e->st));
                k_set_i32<<<1, 1, 0, e->st>>>(reinterpret_cast<int32_t*>(image + out + V4 + 4), end ? 1 : 0);
            }
            ok = ok && soft(cudaEventRecord(e->ev_snap, e->st)) && soft(cudaStreamWaitEvent(e->st_snap, e->ev_snap, 0)) &&
                 soft(cudaMemcpyAsync(host, image, bytes, cudaMemcpyDeviceToHost, e->st_snap)) &&
                 soft(cudaEventRecord(end ? e->ev_landed_end : e->ev_landed, e->st_snap));
            host += bytes;
            e->snap_row = 0;
        }
        on();
        return ok;
    }
    /// A checkpoint of the state the queued work leaves (a chunk's end; `end`: the prompt's, with the engine's output):
    /// each stage's state into its image behind that work, then as snap_send.  Nothing is waited for: the stages go on
    /// with the prompt's next chunk, or the first decode step.
    bool snap_state(uint8_t* host, bool pinned, bool end = false) {
        bool ok = true;
        for (Engine* e = this; e != nullptr; e = e->nxt) {
            e->on();
            ok = ok && soft(cudaStreamWaitEvent(e->st, end ? e->ev_landed_end : e->ev_landed, 0));
            uint8_t* at = end ? e->snap_end : e->snap;
            auto part = [&](const void* d, size_t n) {
                ok = ok && soft(cudaMemcpyAsync(at, d, n, cudaMemcpyDeviceToDevice, e->st));
                at += n;
            };
            const auto& g = m->g;
            for (size_t l = 0; l < e->conv.size(); ++l)
                if (e->conv[l]) {
                    part(e->conv[l], (size_t) g.ssm_conv_channels * (g.ssm_d_conv - 1) * 4);
                    part(e->S[l], (size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size * 4);
                }
            part(e->hlast, (size_t) g.n_embd * 4);
        }
        on();
        return snap_send(host, 0, pinned, end) && ok;
    }
    /// The images' copies have landed, a pageable buffer's contents are in it (before the cache reads, reuses or
    /// frees a buffer).
    void snap_wait() {
        if (snap == nullptr) return;
        for (Engine* e = this; e != nullptr; e = e->nxt) {
            if (e->split) (void) cudaSetDevice(e->dev);
            (void) cudaStreamSynchronize(e->st_snap);
        }
        if (split) (void) cudaSetDevice(dev);
        (void) cudaGetLastError();
        for (int i = 0; i < 2; ++i) {
            if (snap_to[i] != nullptr) std::memcpy(snap_to[i], snap_via[i], (size_t) state_bytes());
            snap_to[i] = nullptr;
        }
    }
    /// What snap_via holds for a pageable buffer goes there (after a request's answer; before the next image).
    void snap_flush() {
        if (snap_to[0] != nullptr || snap_to[1] != nullptr) snap_wait();
    }
    /// The current session holds `p` cells (after a state_copy from a checkpoint there).
    void set_cells(int64_t p) {
        for (Engine* e = this; e != nullptr; e = e->nxt) {
            e->pos = p;
            e->sess[(size_t) e->cur].pos = p;
        }
    }
    uint64_t kv_bytes(int64_t cells) const {
        uint64_t n = 0;
        for (const Engine* e = this; e != nullptr; e = e->nxt)
            for (size_t l = 0; l < e->kc.size(); ++l)
                if (e->kc[l]) n += 2 * (uint64_t) m->g.n_head_kv * (uint64_t) cells * m->kv_row();
        return n;
    }
    /// The current session's first `cells` K / V cells between the GPUs and `host`, a copy of `held` >= cells cells.
    bool kv_copy(int64_t cells, int64_t held, uint8_t* host, bool to_host) { return kv_range(0, cells, held, host, to_host); }
    /// The current session's cells [lo, hi) between the GPUs and `host`, laid out as a copy of cells [lo, held) (kv_copy's
    /// for lo 0): each stage's attention layers, a layer's K then its V, a KV head's cells each.
    bool kv_range(int64_t lo, int64_t hi, int64_t held, uint8_t* host, bool to_host) {
        const size_t H = (size_t) m->g.n_head_kv, kr = m->kv_row(), w = (size_t) (hi - lo) * kr,
                     mine = (size_t) (held - lo) * kr;
        bool ok = true;
        for (Engine* e = this; e != nullptr; e = e->nxt) {
            e->on();
            const size_t row = (size_t) e->ctx * kr;   // a KV head's cells on the device
            for (size_t l = 0; l < e->kc.size(); ++l) {
                if (!e->kc[l]) continue;
                for (__half* d : {e->kc[l], e->vc[l]}) {
                    uint8_t* const at = reinterpret_cast<uint8_t*>(d) + (size_t) lo * kr;
                    if (w > 0)
                        ok = ok && soft(to_host ? cudaMemcpy2DAsync(host, mine, at, row, w, H, cudaMemcpyDeviceToHost, e->st)
                                                : cudaMemcpy2DAsync(at, row, host, mine, w, H, cudaMemcpyHostToDevice, e->st));
                    host += H * mine;
                }
            }
            ok = soft(cudaStreamSynchronize(e->st)) && ok;
        }
        on();
        return ok;
    }
    /// Cells [lo, hi) of a copy of `held` cells (kv_copy's layout) into `out`, laid out as kv_range lays them out.
    void kv_pack(int64_t lo, int64_t hi, const uint8_t* copy, int64_t held, uint8_t* out) const {
        const size_t kr = m->kv_row(), w = (size_t) (hi - lo) * kr, mine = (size_t) held * kr;
        size_t rows = 0;   // a KV head's cells of a layer's K or V
        for (const Engine* e = this; e != nullptr; e = e->nxt)
            for (size_t l = 0; l < e->kc.size(); ++l)
                if (e->kc[l]) rows += 2 * (size_t) m->g.n_head_kv;
        for (size_t r = 0; r < rows; ++r) std::memcpy(out + r * w, copy + r * mine + (size_t) lo * kr, w);
    }
    /// The current session's first `cells` K / V cells from session `src`'s, which keeps them: the start a
    /// conversation lends another one in a session of its own (the cells below a position do not change while the
    /// lender decodes on).
    bool kv_from(int src, int64_t cells) {
        const size_t H = (size_t) m->g.n_head_kv, w = (size_t) cells * m->kv_row();
        bool ok = true;
        for (Engine* e = this; e != nullptr; e = e->nxt) {
            e->on();
            const Sess& a = e->sess[(size_t) src];
            const size_t row = (size_t) e->ctx * m->kv_row();
            for (size_t l = 0; l < e->kc.size() && src != e->cur && w > 0; ++l) {
                if (!e->kc[l]) continue;
                ok = ok && soft(cudaMemcpy2DAsync(e->kc[l], row, a.kc[l], row, w, H, cudaMemcpyDeviceToDevice, e->st)) &&
                     soft(cudaMemcpy2DAsync(e->vc[l], row, a.vc[l], row, w, H, cudaMemcpyDeviceToDevice, e->st));
            }
            ok = soft(cudaStreamSynchronize(e->st)) && ok;
        }
        on();
        return ok;
    }
    int hybrid_swaps_after_prompt = 2048;
    /// the MTP seed rows of a prompt chunk: a buffer the block's own forward does not use (`moe_out`, [tmax][N])
    float* mtp_seed() { return moe_out; }

    /// One speculative step with D MTP drafts (greedy): `x` is the next token (decided, not processed); `fix` the
    /// tokens of the rows the block must redo first (the drafts the last window accepted, seeded by hmain rows
    /// 0 .. fix.size() - 1 of that window).  Drafts d1 .. dD, then the window [x, d1 .. dD] through the main model;
    /// returns the accepted count a and writes the picks of rows 0 .. a (row a's pick is the token after them).
    /// `cancel`: a stop asked for before the state moves keeps only x (the caller then emits no draft), so that a request
    /// stopped mid-answer leaves the state at the tokens it has shown: a conversation that continues from them (the server's
    /// hand-over of a stopped request to a batch slot) reuses it instead of reading them again.
    int spec_step(int x_tok, const std::vector<int32_t>& fix, int D, std::vector<int32_t>& drafts, int32_t* picks,
                  const std::atomic<bool>* cancel = nullptr) {
        const int N = (int) m->g.n_embd;
        Engine& L = *lst;   // the MTP block, the picks and hlast are the last stage's
        lora_rows(nullptr, 0);
        if (main_hybrid) D = std::min<int>(D, (int) (kRebalanceEvery - pos % kRebalanceEvery) - 1);
        const int T0 = (int) fix.size() + 1;   // the redo rows and the row of x
        sync_all();
        L.on();
        // rows of the first MTP call: (hmain row i, fix[i]) at positions pos - fix.size() + i, then (hlast, x) at pos
        for (int i = 0; i < (int) fix.size(); ++i) L.hpin[16 + i] = fix[(size_t) i];
        L.hpin[16 + fix.size()] = x_tok;
        L.hpin[1] = (int32_t) (pos - (int64_t) fix.size());
        CK(cudaMemcpyAsync(L.tok_m, L.hpin + 16, (size_t) T0 * 4, cudaMemcpyHostToDevice, L.st));
        CK(cudaMemcpyAsync(L.pos0_m, L.hpin + 1, 4, cudaMemcpyHostToDevice, L.st));
        // seeds: hmain rows 0 .. fix - 1 are the h of the positions before the fixed rows; hlast is that before x
        float* seeds = L.mtp_seed();
        if (!fix.empty())
            CK(cudaMemcpyAsync(seeds, L.hmain, fix.size() * (size_t) N * 4, cudaMemcpyDeviceToDevice, L.st));
        CK(cudaMemcpyAsync(seeds + fix.size() * (size_t) N, L.hlast, (size_t) N * 4, cudaMemcpyDeviceToDevice, L.st));
        L.forward_mtp(T0, pos - (int64_t) fix.size(), seeds, D > 0, false);
        drafts.assign((size_t) D, 0);
        // a draft's pick, and with --spec-min-p whether the drafts so far are together sure enough (the product of
        // their probabilities under the draft head) to keep it, else the drafts end before it
        float joint = 1.f;
        auto take = [&](int d) {
            CK(cudaMemcpyAsync(L.hpin + 2, L.pick_m, 4, cudaMemcpyDeviceToHost, L.st));
            if (draft_min_p > 0.f) CK(cudaMemcpyAsync(L.hpin + 11, L.p_draft, 4, cudaMemcpyDeviceToHost, L.st));
            CK(cudaStreamSynchronize(L.st));
            drafts[(size_t) d] = L.hpin[2];
            if (draft_min_p <= 0.f) return;
            float p = 0.f;
            std::memcpy(&p, L.hpin + 11, 4);
            joint *= p;
            if (joint < draft_min_p) D = d;
        };
        if (D > 0) take(0);
        int last_row = T0 - 1;   // the row of the previous MTP call whose output seeds the next draft
        for (int d = 1; d < D; ++d) {   // chained: (the block's own output, the last draft) at the next position
            L.hpin[0] = drafts[(size_t) d - 1];
            L.hpin[1] = (int32_t) (pos + d);
            CK(cudaMemcpyAsync(L.tok_m, L.hpin, 4, cudaMemcpyHostToDevice, L.st));
            CK(cudaMemcpyAsync(L.pos0_m, L.hpin + 1, 4, cudaMemcpyHostToDevice, L.st));
            CK(cudaMemcpyAsync(seeds, L.hmtp + (size_t) last_row * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, L.st));
            last_row = 0;
            if (L.mtp_graph) L.graph_run(L.gs().gm1, [&] { L.forward_mtp(1, pos + d, seeds, true, L.use_graph); });
            else L.forward_mtp(1, pos + d, seeds, true, false);
            take(d);
        }
        drafts.resize((size_t) D);
        on();
        // the verify window
        const int T = D + 1;
        hpin[16] = x_tok;
        for (int d = 0; d < D; ++d) hpin[17 + d] = drafts[(size_t) d];
        CK(cudaMemcpyAsync(tok, hpin + 16, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        put_pos(pos);
        rows_deferred = true;
        pass([T](Engine& e) -> cudaGraphExec_t& { return e.gs().gv[T]; }, true, T, pos, T, true);
        rows_deferred = false;
        L.on();
        CK(cudaMemcpyAsync(L.hpin + 2, L.pick, (size_t) T * 4, cudaMemcpyDeviceToHost, L.st));
        CK(cudaStreamSynchronize(L.st));
        int a = 0;
        while (a < D && L.hpin[2 + a] == drafts[(size_t) a]) ++a;
        if (cancel != nullptr && cancel->load()) a = 0;
        for (int i = 0; i <= a; ++i) picks[i] = L.hpin[2 + i];
        on();
        commit(a + 1);
        L.on();
        CK(cudaMemcpyAsync(L.hlast, L.hmain + (size_t) a * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, L.st));
        on();
        advance(a + 1);
        after_rows(a + 1);
        return a;
    }

    /// The first prompt after load ran at about half speed (pp4096 rep 0: ~1770 tok/s against ~3290): the GPU's code
    /// objects, the BLAS handles' kernels and the decode graph are made on first use.  A throw-away prompt of one
    /// chunk, a decode step and (with drafts) a speculative step at load do that before the first request; then the
    /// state, the cache position and the hybrid counters are as after load.
    bool warming = false;
    void warmup() {
        const auto t0 = Clock::now();
        warming = true;
        const int32_t filler = 198;   // a newline: any token works
        std::vector<int32_t> toks((size_t) tmax, filler);
        // with adapters, the steps twice: without them and with them (their graphs are others; the scales --lora gave,
        // or 1 each where they start at 0)
        const std::vector<float> keep = sess[(size_t) cur].lora;
        for (int round = 0; round < (n_ad > 0 ? 2 : 1); ++round) {
            if (n_ad > 0) {
                std::vector<float> w((size_t) n_ad, 0.f);
                if (round == 1)
                    for (int a = 0; a < n_ad; ++a)
                        w[(size_t) a] = lora_default[(size_t) a] != 0.f ? lora_default[(size_t) a] : 1.f;
                sess[(size_t) cur].lora = w;
                reset();
            }
            process(toks.data(), (int64_t) toks.size(), [](int64_t) {});
            if (tmax > kGemmT + 1) process(toks.data(), kGemmT + 1, [](int64_t) {});   // the smallest GEMM chunk too
            process(&filler, 1, [](int64_t) {});
            if (mtp_d > 0) {
                std::vector<int32_t> fix, drafts;
                int32_t picks[16];
                spec_step(filler, fix, mtp_d, drafts, picks);
            }
        }
        sess[(size_t) cur].lora = keep;
        sync_all();
        warming = false;
        for (Hy& H : hy) std::fill(H.score.begin(), H.score.end(), 0.0);
        hy_cpu = hy_gpu = hy_swaps = 0;
        hy_cpu_ms = 0;
        reset();
        std::fprintf(stderr, "lamina: warmed up in %.1f s (one prompt chunk, a decode step%s%s)\n",
                     ms_since(t0) / 1000.0, mtp_d > 0 ? ", a speculative step" : "",
                     n_ad > 0 ? "; without the LoRA adapters and with them" : "");
    }

    int greedy() {
        Engine& L = *lst;
        L.on();
        CK(cudaMemcpyAsync(L.hpin + 2, L.pick, 4, cudaMemcpyDeviceToHost, L.st));
        CK(cudaStreamSynchronize(L.st));
        on();
        return L.hpin[2];
    }
    void host_logits(std::vector<float>& out) {
        Engine& L = *lst;
        out.resize((size_t) m->g.n_vocab);
        L.on();
        CK(cudaMemcpyAsync(out.data(), L.logits, out.size() * 4, cudaMemcpyDeviceToHost, L.st));
        CK(cudaStreamSynchronize(L.st));
        on();
    }
};

// ------------------------------------------------------------------------------------------------ expert placement
/// The routed experts after the engine's buffers: all in VRAM when they fit in `vram_gib` (< 0: the free VRAM less a
/// reserve), else the hybrid mode - every expert in host memory, the same number of VRAM slots per layer, a decode
/// batch's other experts on the CPU pool and a prompt chunk's streamed through two staging buffers.  `serve`: --serve
/// (a warning that RAM is tight goes to the server's window too).
bool place_experts(Model& m, Engine& E, double vram_gib, int threads, int hybrid_t, int slots, bool serve,
                   std::string& err) {
    if (!m.moe) return true;   // a dense FFN: no experts to place
    const auto t0 = Clock::now();
    const auto& g = m.g;
    const int64_t ne = g.n_expert;
    std::vector<Layer*> all;   // the stage's layers (and the MTP block on the last)
    for (size_t l = E.l0; l < E.l1; ++l) all.push_back(&m.layers[l]);
    if (m.has_mtp && E.lst == &E) all.push_back(&m.mtp);
    size_t total = 0, per_slot = 0, max_layer = 0;
    for (Layer* L : all) {
        total += L->stride * (size_t) ne + kMmqTail;
        per_slot += L->stride;
        max_layer = std::max(max_layer, L->stride * (size_t) ne);
    }
    size_t fr = 0, tot = 0;
    CK(cudaMemGetInfo(&fr, &tot));
    const size_t reserve = kVramReserve;
    // A loan asked of this card (kVramReserve, above) never changes where the experts go: an engine pushed into the
    // hybrid mode by the encoder's memory would write all its text slower for the pictures' sake.  With every
    // expert in VRAM the loan is of the VRAM left free, as much as that is.  In the hybrid mode it is the streamed
    // prompt path's staging buffers, or, without that path, VRAM the slots leave free past the reserve (loan_room).
    const size_t loan = (size_t) lamina::core::gpu_loan_asked(E.dev);
    const size_t loan_room = vram_gib >= 0 ? 0 : loan_keep(E.dev);
    const size_t budget = vram_gib >= 0 ? (size_t) (vram_gib * (double) (1ull << 30))
                                        : (fr > reserve ? fr - reserve : 0);
    size_t budget2 = vram_gib >= 0 ? budget : (fr > reserve ? fr - reserve : 0);
    // the MAIN layers' experts fit, the MTP block's not all (UD-Q6_K_XL on 32 GB): the main layers all in VRAM - the
    // text is the all-in-VRAM text, the decode graph stays - and the MTP block's experts on slots + the CPU pool (its
    // drafts only).  When that needs it: the reserve at 256 MiB instead of 768 (said in the log; --expert-vram sets the
    // budget instead).  LAMINA_Q36_NO_MTP_CPU=1: the plain hybrid mode for every layer.
    size_t total_main = 0;
    for (size_t l = E.l0; l < E.l1; ++l) total_main += m.layers[l].stride * (size_t) ne + kMmqTail;
    bool mtp_only = false;
    if (slots <= 0 && vram_gib < 0 && m.has_mtp && total > budget2 && std::getenv("LAMINA_Q36_NO_MTP_CPU") == nullptr) {
        const size_t small_reserve = (size_t) 256 << 20;
        const size_t room = fr > small_reserve ? fr - small_reserve : 0;
        if (total_main + ((size_t) 64 << 20) <= budget2) {
            mtp_only = true;
        } else if (total_main + ((size_t) 64 << 20) <= room) {
            budget2 = room;
            mtp_only = true;
            std::fprintf(stderr, "lamina: the experts nearly fit: the VRAM reserve at 256 MiB (instead of 768) so that "
                         "every main layer's experts fit; a driver or another program needing more VRAM later can make it "
                         "fail - --expert-vram GiB sets the experts' budget instead\n");
        }
    }
    if (slots <= 0 && total <= budget2 && total + ((size_t) 256 << 20) <= fr) {
        std::vector<uint8_t> blob;
        for (Layer* L : all) {
            CK(dmalloc(&L->blobs, L->stride * (size_t) ne + kMmqTail));
            CK(cudaMemset(L->blobs, 0, L->stride * (size_t) ne + kMmqTail));
            blob.assign(L->L.bytes, 0);
            for (int64_t e = 0; e < ne; ++e) {
                expert_blob(*L, e, blob.data());
                CK(cudaMemcpy(L->blobs + (size_t) e * L->stride, blob.data(), L->L.bytes, cudaMemcpyHostToDevice));
            }
            L->n_slots = (int) ne;
        }
        std::fprintf(stderr, "lamina: every routed expert in VRAM (%.2f GiB) in %.1f s\n", (double) total / (1 << 30),
                     ms_since(t0) / 1000.0);
        return true;
    }
#if !defined(LAMINA_Q36_CPU_POOL)
    (void) threads;
    (void) hybrid_t;
    (void) serve;
    (void) loan;
    (void) loan_room;
    err = "the routed experts (" + std::to_string(total >> 20) + " MiB) do not fit in VRAM, and this build has no CPU "
          "expert pool (LAMINA_NATIVE_EXPERTS=OFF)";
    return false;
#else
    namespace C = lamina::kernels::cpu;
    m.hybrid = true;
    E.main_hybrid = !mtp_only;
    if (E.main_hybrid) E.use_graph = false;   // the host takes part in every layer
    E.mtp_graph = false;                      // the MTP block's layer is hybrid either way
    // (MMQ takes a batch of more than kGemmT rows: a --hybrid-rows under that streams from there too)
    E.hybrid_t = std::max(Engine::kGemmT, std::min(hybrid_t, E.tmax));
    if (!E.use_mmq) E.hybrid_t = E.tmax;   // no streamed prompt path: every batch on slots + the pool
    // the streamed prompt path's two staging buffers, a layer's experts each, in one block: idle between prompts, so
    // the block is what the image encoder borrows, as large as its loan where two layers are less
    const size_t stage_one = (max_layer + kMmqTail + 255) & ~(size_t) 255;
    size_t staging = E.use_mmq && E.main_hybrid ? 2 * stage_one : 0;
    if (loan > 0 && E.main_hybrid) {
        if (staging > 0) staging = std::max(staging, (loan + 255) & ~(size_t) 255);
        else if (budget2 > loan_room) budget2 -= loan_room;
    }
    size_t slot_bytes = budget2 > staging ? budget2 - staging : 0;
    if (slot_bytes + staging + ((size_t) 256 << 20) > fr) slot_bytes = fr > staging + ((size_t) 512 << 20) ? fr - staging - ((size_t) 512 << 20) : 0;
    // --expert-slots: that many per layer (tests: the same split with and without the MTP block)
    int S = slots > 0 ? (int) std::min<int64_t>(ne, slots) : (int) std::min<size_t>((size_t) ne, slot_bytes / per_slot);
    if (mtp_only) {   // the main layers all, the MTP block what is left
        const size_t left = budget2 > total_main ? budget2 - total_main : 0;
        S = (int) std::min<size_t>((size_t) ne, left / m.mtp.stride);
    }
    {   // what the hybrid mode allocates next (the staging pair, the CPU part's outputs, the slots) against the VRAM the
        // sessions and the context left: said with the numbers instead of a cudaMalloc failure (8 slots on a 12 GB card)
        const size_t Tn = (size_t) E.tmax, Nn = (size_t) g.n_embd, Kn = (size_t) m.top_k;
        size_t need = staging + Tn * Kn * Nn * 4 + Tn * Kn * 4;
        for (Layer* L : all) {
            if (mtp_only && L != &m.mtp) need += L->stride * (size_t) ne + kMmqTail;
            else need += (S > 0 ? (size_t) S * L->stride + kMmqTail : 0);
        }
        if (need > fr) {
            err = "not enough VRAM left for the hybrid experts' buffers and slots (" + std::to_string((need >> 20)) +
                  " MiB needed, " + std::to_string(fr >> 20) + " MiB free after the weights, the context and the sessions): use "
                  "fewer slots (\"parallel\" / --batch), a shorter --max-context" +
                  (slots > 0 ? ", or fewer --expert-slots" : "");
            return false;
        }
    }
    // host memory: every expert, pinned where the driver allows it (the prompt's stream and the slot copies), first
    // checked against the RAM
    size_t host = 0;
    for (Layer* L : all)
        if (!mtp_only || L == &m.mtp) host += L->stride * (size_t) ne + kMmqTail;
    {
        const lamina::platform::RamFit ram = lamina::platform::ram_fit(host);
        const std::string what = mtp_only ? "the MTP block's routed experts" : "the routed experts (in RAM, every one)";
        if (ram.total > 0)
            std::fprintf(stderr, "lamina: RAM: %s take %.1f GiB, %.1f of this PC's %.1f GiB available\n", what.c_str(),
                         (double) host / 1073741824.0, (double) ram.available / 1073741824.0,
                         (double) ram.total / 1073741824.0);
        if (ram.too_much()) {
            err = ram.say(what, "with the layers over more GPUs (--second-gpu) or a shorter --max-context, every expert "
                                "fits in VRAM and none takes RAM");
            return false;
        }
        if (ram.tight())
            lamina::program::serve_warn(serve, ram.say(what, "close other programs, or fit every expert in VRAM (more "
                                                             "GPUs through --second-gpu, a shorter --max-context)"));
    }
    uint8_t* arena = nullptr;
    bool pinned = hmalloc((void**) &arena, host, cudaHostAllocDefault) == cudaSuccess;
    if (!pinned) {
        (void) cudaGetLastError();
        arena = (uint8_t*) heap_new(host);
        if (!arena) {
            err = "no host memory for the experts (" + std::to_string(host >> 20) + " MiB)";
            return false;
        }
    }
    size_t off = 0;
    E.hy.resize(all.size());
    for (size_t li = 0; li < all.size(); ++li) {
        Layer& L = *all[li];
        if (mtp_only && &L != &m.mtp) {   // a main layer: every expert in VRAM, as in the all-in-VRAM placement
            CK(dmalloc(&L.blobs, L.stride * (size_t) ne + kMmqTail));
            CK(cudaMemset(L.blobs, 0, L.stride * (size_t) ne + kMmqTail));
            std::vector<uint8_t> blob(L.L.bytes, 0);
            for (int64_t e = 0; e < ne; ++e) {
                expert_blob(L, e, blob.data());
                CK(cudaMemcpy(L.blobs + (size_t) e * L.stride, blob.data(), L.L.bytes, cudaMemcpyHostToDevice));
            }
            L.n_slots = (int) ne;
            continue;
        }
        L.hblobs = arena + off;
        off += L.stride * (size_t) ne + kMmqTail;
        std::memset(L.hblobs, 0, L.stride * (size_t) ne + kMmqTail);
        for (int64_t e = 0; e < ne; ++e) expert_blob(L, e, L.hblobs + (size_t) e * L.stride);
        L.n_slots = S;
        if (S > 0) {
            CK(dmalloc(&L.blobs, L.stride * (size_t) S + kMmqTail));
            CK(cudaMemset(L.blobs, 0, L.stride * (size_t) S + kMmqTail));
            CK(cudaMemcpy(L.blobs, L.hblobs, L.stride * (size_t) S, cudaMemcpyHostToDevice));
        }
        Engine::Hy& H = E.hy[li];
        H.slot_of.assign((size_t) ne, -1);
        H.exp_in_slot.assign((size_t) S, 0);
        for (int s = 0; s < S; ++s) {
            H.slot_of[(size_t) s] = s;
            H.exp_in_slot[(size_t) s] = s;
        }
        H.score.assign((size_t) ne, 0.0);
        CK(dmalloc(&H.slot_of_d, (size_t) ne * 4));
        CK(cudaMemcpy(H.slot_of_d, H.slot_of.data(), (size_t) ne * 4, cudaMemcpyHostToDevice));
        if (!C::native_fmt(L.L.gu_type, L.L.d_type, g.n_embd, g.n_ff, H.fmt, err)) return false;
        if (H.fmt.up_off != L.L.up_off || H.fmt.down_off != L.L.down_off || H.fmt.bytes != L.L.bytes) {
            err = "the CPU and GPU expert layouts differ";
            return false;
        }
    }
    const size_t T = (size_t) E.tmax, N = (size_t) g.n_embd, Kp = (size_t) m.top_k;
    CK(hmalloc((void**) &E.h_h, T * N * 4, cudaHostAllocDefault));
    CK(hmalloc((void**) &E.cpu_out, T * Kp * N * 4, cudaHostAllocDefault));
    CK(hmalloc((void**) &E.h_ids, T * Kp * 4, cudaHostAllocDefault));
    CK(hmalloc((void**) &E.h_miss, T * Kp * 4, cudaHostAllocDefault));
    CK(hmalloc((void**) &E.h_cnt, all.size() * (size_t) ne * 4, cudaHostAllocDefault));
    CK(dmalloc(&E.ecpu, T * Kp * N * 4));
    CK(dmalloc(&E.d_miss, T * Kp * 4));
    E.acts = (uint8_t*) heap_new(T * C::kNativeActBytes + 64);
    CK(event_new(&E.ev_route, cudaEventDisableTiming));
    if (staging) {
        CK(stream_new(&E.cst, cudaStreamNonBlocking));
        CK(dmalloc(&E.stage[0], staging));
        CK(cudaMemset(E.stage[0], 0, staging));
        E.stage[1] = E.stage[0] + stage_one;
        E.stage_bytes = staging;
        for (int b = 0; b < 2; ++b) {
            CK(event_new(&E.ev_copy[b], cudaEventDisableTiming));
            CK(event_new(&E.ev_used[b], cudaEventDisableTiming));
            CK(cudaEventRecord(E.ev_used[b], E.st));
        }
    }
    E.pool = std::make_unique<C::ExpertPool>(threads);
    E.host_pin = std::make_unique<C::HostPin>(C::CorePlacement(false).host());
    CK(cudaDeviceSynchronize());
    if (mtp_only)
        std::fprintf(stderr, "lamina: every main layer's routed experts in VRAM (%.2f GiB); the MTP block's: %d of "
                     "%lld in VRAM, the rest on the CPU pool (its drafts only)\n", (double) total_main / (1 << 30), S,
                     (long long) ne);
    else
    std::fprintf(stderr, "lamina: hybrid experts: %d of %lld per layer in VRAM (%.2f GiB), all %.2f GiB in %s host "
                 "memory, the rest of a batch of <= %d rows on %d CPU threads, bigger prompt chunks streamed (%.2f GiB "
                 "staging); %.1f s\n", S, (long long) ne, (double) S * per_slot / (1 << 30), (double) host / (1 << 30),
                 pinned ? "pinned" : "pageable", E.hybrid_t, E.pool->workers() + (E.pool->host_works() ? 1 : 0),
                 (double) staging / (1 << 30), ms_since(t0) / 1000.0);
    return true;
#endif
}

// ------------------------------------------------------------------------------------------------ sampling
using lamina::program::Sampling;
using lamina::program::parse_ids;
using lamina::program::sample;

bool is_eos(int t) { return t == 248044 || t == 248046; }

// ------------------------------------------------------------------------------------------------ the conversation cache
constexpr int32_t kImStart = 248045, kImEnd = 248046;   // <|im_start|>, <|im_end|>: a turn's first and last token

/// --serve's prompt cache (lamina/core/prompt_cache.hpp, Flash-Next's options): a request continues from the longest
/// start of its prompt the engine holds - a session's conversation, one of its checkpoints, or a conversation another
/// request replaced, parked in RAM with its K / V cells.  With batch slots every session holds a conversation, each
/// with checkpoints of its own, and the park takes the ones the sessions do not hold.
struct CacheOpts {
    int chain = 16;                  // --prompt-cache: checkpoints a conversation keeps (0: off)
    int64_t every = 16384;           // --prompt-cache-every: a long prompt's checkpoints, in cells
    int64_t park_mib = 16384;        // --conversation-cache-mib: the parked conversations' RAM
    int64_t park_slots = 0;          // --conversation-cache-slots: their number at most (0: no limit)
    int64_t min_tokens = 1024;       // --conversation-cache-min-tokens: shorter ones are not parked
    int64_t ram_reserve_mib = 6144;  // --ram-reserve-mib: they take no RAM below this much available to the system
    int64_t tail_role = -1;          // --tail-role-token: a trailing turn of this role stays after the turn's checkpoint
    std::string spill_dir;           // --conversation-spill-dir: the disk tier's folder (empty: none)
    int64_t spill_mib = -1;          // --conversation-spill-mib: its files' MiB at most (-1: its default; 0: none)
};

/// The engine's state behind the cache's policy: the current session's (Engine::state_copy, kv_copy), which the
/// policy selects for each request (session_select).
struct Cache final : lamina::core::SequenceCache {
    Engine& E;
    Cache(Engine& e, const CacheOpts& o)
        : SequenceCache(o.chain, e.state_bytes(), (uint64_t) std::max<int64_t>(0, o.park_mib) << 20,
                        (size_t) std::max<int64_t>(0, o.park_slots), o.min_tokens, e.n_sess),
          E(e) {}
    ~Cache() override {   // (no copy still on its way to a buffer the cache frees)
        for (Engine* e = &E; e != nullptr; e = e->nxt) {
            if (e->split) (void) cudaSetDevice(e->dev);
            (void) cudaStreamSynchronize(e->st);
        }
        E.snap_wait();
    }

protected:
    bool state_save(int64_t, void* host, bool filled) override {
        if (filled) return true;   // the engine's image is on its way (Engine::snap_send)
        E.sync_all();
        E.snap_wait();   // (the buffer may be one an image's copy is still landing in)
        return E.state_copy(static_cast<uint8_t*>(host), true);
    }
    bool state_load(int64_t pos, const void* host) override {
        E.sync_all();
        E.snap_wait();
        if (!E.state_copy(const_cast<uint8_t*>(static_cast<const uint8_t*>(host)), false)) return false;
        E.set_cells(pos);
        return true;
    }
    void state_zero() override { E.reset(); }
    uint64_t kv_bytes(int64_t cells) const override { return E.kv_bytes(cells); }
    bool kv_copy(int64_t cells, int64_t held, uint8_t* host, bool to_host) override {
        E.sync_all();
        return E.kv_copy(cells, held, host, to_host);
    }
    void session_select(int s) override {
        E.sync_all();
        E.select(s);
    }
    bool kv_from(int from, int64_t cells) override {
        E.sync_all();
        return E.kv_from(from, cells);
    }
    // the disk tier: the current session's cells a range at a time, from every GPU of a split
    bool disk_ready() const override { return true; }
    bool kv_range(int64_t lo, int64_t hi, int64_t held, uint8_t* host, bool to_host) override {
        E.sync_all();
        return E.kv_range(lo, hi, held, host, to_host);
    }
    void kv_pack(int64_t lo, int64_t hi, const uint8_t* copy, int64_t held, uint8_t* out) const override {
        E.kv_pack(lo, hi, copy, held, out);
    }
    void kv_settle() override {
        E.sync_all();
        E.snap_wait();   // (an image's copy into a checkpoint buffer)
    }
};

// ------------------------------------------------------------------------------------------------ pictures
constexpr int32_t kImagePad = 248056;   // <|image_pad|>

/// A request's pictures: the encoder's rows in prompt order, each picture's cell run, and the positions.
struct ImgReq {
    std::vector<float> rows;
    std::vector<ImgKey> imgs;
    std::vector<int32_t> cell_row;   // per prompt cell: its row in `rows`, or -1 (text)
    std::vector<int32_t> mrope;      // [cell][3] t, h, w of every prompt cell (llama.cpp's mtmd positions)
    int32_t delta = 0;               // the text after the prompt: rotary position = cell + delta
};

/// Reads a lamina-vision file (one or more records: int32 'SVE1', n, nx, ny, n_embd, then n x n_embd floats, in prompt
/// order) and lays the pictures over the prompt's <|image_pad|> runs.  Positions as mtmd's: text counts up by one; a
/// picture's cells take (t, h, w) = (p, p + row, p + column) and the text after it continues at p + max(nx, ny).  "" or
/// the error.  `keys`: the server's key for each picture (img_keys=), else the embeddings' hash.
std::string build_images(const std::string& path, const std::vector<int32_t>& ids, int64_t n_embd,
                         const std::vector<uint64_t>& keys, ImgReq& r) {
    std::vector<lamina::program::VisionImage> imgs;
    if (std::string ve = lamina::program::read_vision_records(path, n_embd, r.rows, imgs); !ve.empty()) return ve;
    const int64_t n = (int64_t) ids.size();
    r.cell_row.assign((size_t) n, -1);
    r.mrope.assign((size_t) n * 3, 0);
    auto put = [&](int64_t c, int64_t t, int64_t h, int64_t w) {
        r.mrope[(size_t) c * 3] = (int32_t) t;
        r.mrope[(size_t) c * 3 + 1] = (int32_t) h;
        r.mrope[(size_t) c * 3 + 2] = (int32_t) w;
    };
    int64_t p = 0, i = 0;
    size_t k = 0;
    while (i < n) {
        if (ids[(size_t) i] != kImagePad) { put(i, p, p, p); ++p; ++i; continue; }
        if (k >= imgs.size()) return "the prompt has more images than the embeddings file";
        const lamina::program::VisionImage& im = imgs[k++];
        const uint64_t h = lamina::program::vision_image_key(im, keys.size() == imgs.size() ? &keys[k - 1] : nullptr,
                                                             r.rows.data(), n_embd);
        r.imgs.push_back({i, im.n, h});
        for (int64_t j = 0; j < im.n; ++j)
            if (i + j >= n || ids[(size_t) (i + j)] != kImagePad)
                return "image " + std::to_string(k) + " has " + std::to_string((long long) im.n) +
                       " rows but fewer <|image_pad|> tokens";
        for (int64_t j = 0; j < im.n; ++j) {
            const int64_t y = j / im.nx, x = j % im.nx;
            put(i + j, p, p + y, p + x);
            r.cell_row[(size_t) (i + j)] = (int32_t) ((im.off / (size_t) n_embd) + (size_t) j);
        }
        i += im.n;
        p += std::max(im.nx, im.ny);
    }
    if (k != imgs.size()) return "the embeddings file has more images than the prompt";
    if (n > 0 && ids[(size_t) (n - 1)] == kImagePad) return "the prompt cannot end in an image";
    r.delta = (int32_t) (p - n);
    return "";
}

// ------------------------------------------------------------------------------------------------ serve
/// A batch slot: a session decoding (its next token is `x`, not yet in the state).
/// s as the inside of a JSON string
std::string json_escape(const std::string& s) {
    std::string o;
    for (const unsigned char c : s) {
        if (c == '"' || c == '\\') {
            o += '\\';
            o += (char) c;
        } else if (c < 0x20) {
            char b[8];
            std::snprintf(b, sizeof b, "\\u%04x", c);
            o += b;
        } else {
            o += (char) c;
        }
    }
    return o;
}

struct Slot {
    bool active = false, stop = false;
    int sess = 0;                 // the session it decodes in (the prompt cache's choice; without the cache its own number)
    int32_t x = 0;
    int64_t produced = 0, max_new = 0;
    Sampling smp;
    std::mt19937_64 rng;
    std::vector<int32_t> hist;
    Clock::time_point t0;
    int32_t draft = 0;            // --batch-mtp: the MTP block's proposal for the token after `x`
    bool draft_ready = false;
    long long offered = 0, accepted = 0;
};

/// The engine protocol (lamina-server's).  One request at a time on the control lines: GEN / GENI (alone: speculative
/// steps with the MTP block), or with --batch N BGEN / BGENI <slot> (the request's prompt is read into a session -
/// the decode windows of the other slots go on between its chunks - then its tokens come as BT lines, one window for
/// every slot decoding, until BDONE).  BSTOP <slot> ends a slot at its next window; STOP the request on the control
/// lines.  A slot is the server's name for a request's stream.  Its session is the prompt cache's choice: the one
/// that holds the request's conversation, else one for a new conversation (Serve::request); without the cache
/// (--prompt-cache 0) slot b decodes in session b.
struct Serve {
    Engine& E;
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
    std::atomic<bool> stop{false};
    std::vector<Slot> slots;
    int n_active = 0;
    std::vector<float> lg;
    // decode windows of the other slots between two chunks of a prompt read: LAMINA_Q36_READ_WINDOWS (default 1; a chunk
    // takes ~0.15 s at 512 tokens, a window ~0.012 s: more windows give the decoding slots more of the time a read takes)
    int read_windows = 1;
    // the conversation cache (null: off)
    CacheOpts copt;
    std::unique_ptr<Cache> cache;
    std::vector<int32_t> prev_prompt;               // the last text request's prompt: where the next one leaves it
    std::vector<std::vector<float>> scale_sets;     // the adapters' scales requests came with: a conversation's tag
    bool poisoned = false;                          // non-finite logits: whatever the cache holds may be bad
    // LAMINA_Q36_CKPT_CUT=1 (A/B): every checkpoint at a chunk's end, as where the cache has no buffer to give - a
    // turn's read is then two chunks - and none of a prompt's end: the same request again reads its last tokens
    const bool ckpt_cut = std::getenv("LAMINA_Q36_CKPT_CUT") != nullptr && std::string(std::getenv("LAMINA_Q36_CKPT_CUT")) != "0";
    // LAMINA_Q36_CKPT_VIA=1 (a test's): every checkpoint through the engine's pinned buffer, as into a pageable one
    const bool ckpt_via = std::getenv("LAMINA_Q36_CKPT_VIA") != nullptr && std::string(std::getenv("LAMINA_Q36_CKPT_VIA")) != "0";

    /// A buffer of the cache's for a checkpoint the engine brings there itself (`end`: a prompt's end), and whether it
    /// is pinned; null: none.
    void* checkpoint_buffer(bool& pinned, bool end = false) {
        bool pageable = false;
        void* const h = E.snap != nullptr ? cache->prepare(&pageable, end) : nullptr;
        pinned = !pageable && !ckpt_via;
        return h;
    }
    /// After a request's answer: what its checkpoints still owe a pageable buffer, and the cache's next pinned one.
    void cache_idle() {
        if (!cache) return;
        E.snap_flush();
        cache->spare();
    }

    Serve(Engine& e, const CacheOpts& co) : E(e), slots((size_t) e.n_sess), copt(co) {
        if (const char* v = std::getenv("LAMINA_Q36_READ_WINDOWS")) read_windows = std::max(0, std::min(64, std::atoi(v)));
        if (copt.chain <= 0) {
            if (E.n_sess > 1)
                std::fprintf(stderr, "lamina serve: prompt cache: off (each slot keeps its conversation, which a request "
                             "continues when it extends it)\n");
            return;
        }
        cache = std::make_unique<Cache>(E, copt);
        for (int i = 0; i < 2 && E.snap != nullptr; ++i) CK(hmalloc(&E.snap_via[i], (size_t) E.state_bytes(), cudaHostAllocDefault));
        cache->whole_prompt = E.snap != nullptr && !ckpt_cut;   // (a prompt's end is a checkpoint with its output)
        uint64_t ram_now = 0;
        const bool ram_known = copt.ram_reserve_mib > 0 && lamina::platform::available_ram(ram_now);
        if (ram_known)   // the experts a hybrid placement pins cannot be swapped: the park is what gives way
            cache->ram_spare = [reserve = copt.ram_reserve_mib << 20]() -> int64_t {
                uint64_t avail = 0;
                if (!lamina::platform::available_ram(avail)) return lamina::core::ConversationCache::kNoRamLimit;
                return (int64_t) avail - reserve;
            };
        std::fprintf(stderr, "lamina serve: prompt cache on: up to %d checkpoints of %.1f MiB per conversation%s, %lld MiB "
                     "for the conversations requests replace%s%s\n", copt.chain,
                     (double) cache->ckpt_bytes() / 1048576.0,
                     E.n_sess > 1 ? (", one in each of the " + std::to_string(E.n_sess) + " sessions").c_str() : "",
                     (long long) copt.park_mib,
                     copt.park_slots > 0 ? (", at most " + std::to_string(copt.park_slots) + " of them").c_str() : "",
                     ram_known ? (", above a RAM reserve of " + std::to_string(copt.ram_reserve_mib) + " MiB (" +
                                  std::to_string(ram_now >> 20) + " MiB available now)").c_str() : "");
        cache->open_disk(copt.spill_dir, copt.spill_mib);   // the disk tier below the park
    }

    /// The CACHE line (the server's Conversation cache card): at READY, then after each request before its DONE.
    void print_cache() {
        if (!cache) return;
        const auto cs = cache->state();
        serve_print("CACHE live=%lld checkpoints=%zu parked=%zu parked_bytes=%llu parked_budget=%llu slots=%zu "
                    "evictions=%zu held=%zu sessions=%d disk=%zu disk_bytes=%llu disk_budget=%llu\n",
                    (long long) cs.live_cells, cs.checkpoints, cs.parked, (unsigned long long) cs.parked_bytes,
                    (unsigned long long) cs.parked_budget, cs.slots, cs.evictions, cs.held, E.n_sess, cs.disk,
                    (unsigned long long) cs.disk_bytes, (unsigned long long) cs.disk_budget);
    }
    /// Session `s` took token `t` (a decode step): its tokens, and the cache's record of them.
    void fed(int s, int32_t t) {
        E.sess[(size_t) s].live.push_back(t);
        if (cache) cache->append(s, t);
    }
    /// The slot decoding in session `s` ended.  `bad`: with non-finite logits - its state is not continued, nor is
    /// anything the cache holds (forgotten at the next request).
    void slot_ended(int s, bool bad) {
        Engine::Sess& ss = E.sess[(size_t) s];
        ss.active = false;
        --n_active;
        if (bad) {
            ss.live.clear();
            ss.imgs.clear();
            poisoned = poisoned || cache != nullptr;
        }
        if (!cache) return;
        if (bad) cache->drop(s);
        else cache->touch(s);
    }

    /// The cells a request's prompt continues from and the session it runs in, the engine's state set there: the end
    /// of the conversation a session holds, a checkpoint of it, a parked conversation taken back, or a start borrowed
    /// from another conversation (a session's, also one that decodes: copied from its K / V; or a parked one's); 0: an
    /// empty sequence.  A new conversation takes the session that holds nothing, else the one whose conversation is
    /// the smallest, which is parked.  -1: every session decodes.  A conversation is continued by requests with its
    /// adapters' scales only.
    int64_t cache_begin(const std::vector<int32_t>& ids, const std::vector<ImgKey>& imgs, const std::vector<float>& lora,
                        int& session) {
        E.snap_wait();   // (the last request's checkpoints are in their buffers before the cache moves or frees any)
        uint64_t busy = 0;
        for (int i = 0; i < E.n_sess; ++i) busy |= (uint64_t) (E.sess[(size_t) i].active ? 1 : 0) << i;
        if (poisoned) {
            cache->clear();
            for (int i = 0; i < E.n_sess; ++i)   // (the sessions that decode go on: their tokens stay known)
                if (E.sess[(size_t) i].active)
                    for (const int32_t t : E.sess[(size_t) i].live) cache->append(i, t);
            poisoned = false;
        }
        size_t tag = 0;
        while (tag < scale_sets.size() && scale_sets[tag] != lora) ++tag;
        if (tag == scale_sets.size()) scale_sets.push_back(lora);
        std::vector<lamina::core::ImgKey> keys;
        for (const ImgKey& k : imgs)   // (a picture is its rows' hash and their count)
            keys.push_back({k.cell, k.hash ^ (0x9e3779b97f4a7c15ull * (uint64_t) (k.n + 1))});
        return cache->begin(std::vector<int64_t>(ids.begin(), ids.end()), keys, (uint64_t) tag, busy, &session);
    }

    /// Where a prompt's read takes a checkpoint besides its chunks' ends, ascending, each in (reuse, n) with two cells or
    /// more to its neighbours and the ends (a read of one token goes through the decode graph, past the MTP block's
    /// K / V), none inside a picture.  The recurrent state cannot step back, so the places a later prompt may part from this one are
    /// saved as the read passes them: before the prompt's last <|im_start|> (the next turn's prompt re-renders what
    /// follows: the answer without its reasoning, or without the empty block a request with thinking off ends its
    /// prompt with), where this prompt leaves the one before (agents send one long start again with another tail), a
    /// few cells before the end of a last message that grows between requests.  A long prompt's checkpoints every
    /// --prompt-cache-every cells are the read's own, at its chunks' ends, and the prompt's end is one with the
    /// engine's output (request): the same prompt again reads nothing.  (LAMINA_Q36_CKPT_CUT: none of the end, and one
    /// two cells before it where the turn's is not near.)
    std::vector<int64_t> checkpoint_marks(const std::vector<int32_t>& ids, int64_t reuse, const std::vector<ImgKey>& imgs,
                                          bool geni) {
        const int64_t n = (int64_t) ids.size();
        int64_t q = 0;
        for (int64_t i = n - 2; i > reuse; --i)
            if (ids[(size_t) i] == kImStart) { q = i; break; }
        if (q > 0 && copt.tail_role >= 0)   // a trailing effort turn stays out of the checkpoint
            for (int64_t i = q - 1; i > reuse; --i)
                if (ids[(size_t) i] == kImStart) {
                    if (ids[(size_t) i + 1] == copt.tail_role) q = i;
                    break;
                }
        int64_t fork = 0, msg_end = 0;
        {
            const int64_t lim = std::min<int64_t>((int64_t) prev_prompt.size(), n - 1);
            int64_t d = 0;
            while (d < lim && prev_prompt[(size_t) d] == ids[(size_t) d]) ++d;
            if (d > reuse + 256 && d < n - 1 && d != q) fork = d;
            for (int64_t i = (q > 0 ? q : n - 1) - 1; i > reuse; --i)
                if (ids[(size_t) i] == kImEnd) {
                    int64_t start = i;   // the message's <|im_start|>; then its role and a newline
                    while (start > 0 && ids[(size_t) start] != kImStart) --start;
                    if (d > start + 3 && i - 4 > reuse + 256 && i - 4 != fork) msg_end = i - 4;
                    break;
                }
        }
        if (!geni) prev_prompt = ids;
        std::vector<int64_t> want{q, fork, msg_end};
        if (ckpt_cut && n - 2 - std::max(q, reuse) > kEndNear) want.push_back(n - 2);
        std::sort(want.begin(), want.end());
        std::vector<int64_t> marks;
        for (const int64_t p : want) {
            if (p < reuse + 2 || p > n - 2 || (!marks.empty() && p < marks.back() + 2)) continue;
            bool inside = false;
            for (const ImgKey& k : imgs) inside = inside || (k.cell < p && p < k.cell + k.n);
            if (!inside) marks.push_back(p);
        }
        return marks;
    }
    static constexpr int64_t kEndNear = 64;   // a turn's checkpoint this near the prompt's end serves as its end's too

    /// BSTOP / BYIELD lines among the queued ones: applied now (a window is a moment away at most).
    void poll_control() {
        std::lock_guard<std::mutex> lk(mu);
        for (auto it = lines.begin(); it != lines.end();) {
            if (it->rfind("BSTOP ", 0) == 0) {
                const int b = std::atoi(it->c_str() + 6);
                if (b >= 0 && b < (int) slots.size()) slots[(size_t) b].stop = true;
                it = lines.erase(it);
            } else if (it->rfind("BYIELD", 0) == 0) {   // (a read here does not give way: the server then goes on)
                it = lines.erase(it);
            } else {
                ++it;
            }
        }
    }

    /// One decode window with MTP proposals (--batch-mtp): every active slot's next token, and for up to kVerify - S of the
    /// slots (the greedy ones, in turn) also the proposal its MTP block made: the window verifies both at once.  A slot whose
    /// proposal the main model confirms moves two tokens in the window.  The proposals for the next window come right after
    /// it, from the hidden states the window just produced (the first one of a slot from the state its prompt left).
    int mtp_turn = 0;
    void window_mtp() {
        poll_control();
        const int N = (int) E.m->g.n_embd;
        std::vector<int> act;
        for (int b = 0; b < E.n_sess; ++b)
            if (slots[(size_t) b].active) act.push_back(b);
        const int S = (int) act.size();
        if (S == 0) return;
        // who verifies a proposal this window: the greedy slots in turn, as many as fit beside one row per slot
        const int room = Engine::kVerify - S;
        std::vector<char> cand((size_t) S, 0);
        int nc = 0;
        for (int k = 0; k < S && nc < room; ++k) {
            const int i = (mtp_turn + k) % S;
            if (!slots[(size_t) act[(size_t) i]].smp.needs_logits()) { cand[(size_t) i] = 1; ++nc; }
        }
        ++mtp_turn;
        // a candidate without a proposal gets its first one from the state its last row left
        std::vector<Engine::MtpRow> first;
        std::vector<int> first_of;
        for (int i = 0; i < S; ++i) {
            Slot& sl = slots[(size_t) act[(size_t) i]];
            if (cand[(size_t) i] && !sl.draft_ready) {
                Engine::Sess& ss = E.sess[(size_t) sl.sess];
                first.push_back({sl.sess, sl.x, ss.pos, ss.hlast});
                first_of.push_back(i);
            }
        }
        if (!first.empty()) {
            E.mtp_rows(first.data(), (int) first.size());
            for (size_t k = 0; k < first.size(); ++k) {
                Slot& sl = slots[(size_t) act[(size_t) first_of[k]]];
                sl.draft = E.mtp_pick[k];
                sl.draft_ready = true;
            }
        }
        std::vector<Engine::Group> gr((size_t) S);
        std::vector<int> row0((size_t) S);
        int T = 0;
        for (int i = 0; i < S; ++i) {
            const Slot& sl = slots[(size_t) act[(size_t) i]];
            gr[(size_t) i] = {sl.sess, cand[(size_t) i] ? 2 : 1, {sl.x, sl.draft}};
            row0[(size_t) i] = T;
            T += gr[(size_t) i].n;
        }
        E.run_rows(gr.data(), S, true);
        // the picks: emit, decide how many rows each slot keeps
        std::vector<int> keep((size_t) S, 1);
        std::vector<int32_t> last((size_t) S, 0);
        std::vector<char> alive((size_t) S, 0);
        for (int i = 0; i < S; ++i) {
            const int b = act[(size_t) i];
            Slot& sl = slots[(size_t) b];
            Engine::Sess& ss = E.sess[(size_t) sl.sess];
            const int n = gr[(size_t) i].n, r0 = row0[(size_t) i];
            int y0 = E.rows_pick[r0];
            if (sl.smp.needs_logits()) {
                E.host_logits_row(r0, lg);
                y0 = sample(lg, sl.smp, sl.hist, sl.rng);
            }
            bool two = n == 2 && y0 == sl.draft && !is_eos(y0) && !sl.stop && sl.produced + 2 <= sl.max_new &&
                       ss.pos + 3 <= E.ctx;   // (ss.pos: the position of the slot's next token: the window did not move it)
            if (n == 2) { ++sl.offered; if (two) ++sl.accepted; }
            const int rows_kept = two ? 2 : 1;
            keep[(size_t) i] = rows_kept;
            const int32_t row_tok[2] = {sl.x, sl.draft};
            bool done = false;
            int y = y0;
            for (int j = 0; j < rows_kept; ++j) {
                if (j == 1) y = E.rows_pick[r0 + 1];
                if (y < 0) {   // no finite logit: the slot ends, and its state is not continued
                    serve_print("BDONE %d %lld nonfinite %.1f\n", b, (long long) sl.produced, ms_since(sl.t0));
                    std::fprintf(stderr, "lamina batch: slot %d: non-finite logits: the request ends\n", b);
                    sl.active = false;
                    slot_ended(sl.sess, true);
                    done = true;
                    break;
                }
                fed(sl.sess, row_tok[j]);
                serve_print("BT %d %d\n", b, y);
                ++sl.produced;
                sl.hist.push_back(y);
                const char* fin = is_eos(y) ? "stop" : sl.stop ? "cancel" : sl.produced >= sl.max_new ? "length"
                                  : ss.pos + j + 2 > E.ctx ? "length" : nullptr;
                if (fin) {
                    serve_print("BDONE %d %lld %s %.1f\n", b, (long long) sl.produced, fin, ms_since(sl.t0));
                    if (sl.offered > 0)
                        std::fprintf(stderr, "lamina batch: slot %d: %lld of %lld MTP proposals accepted\n", b, sl.accepted,
                                     sl.offered);
                    sl.active = false;
                    slot_ended(sl.sess, false);
                    done = true;
                    break;
                }
            }
            last[(size_t) i] = y;
            alive[(size_t) i] = !done;
            if (!done) sl.x = y;
        }
        E.commit_rows(gr.data(), keep.data(), S);
        // the hidden state after each slot's last kept row (the seed of a later first proposal), and the next proposals
        std::vector<Engine::MtpRow> nx;
        std::vector<int> nx_of, nx_last;
        for (int i = 0; i < S; ++i) {
            Slot& sl = slots[(size_t) act[(size_t) i]];
            const int b = sl.sess;   // (the rows are the session's)
            Engine::Sess& ss = E.sess[(size_t) b];
            const int r0 = row0[(size_t) i];
            CK(cudaMemcpyAsync(ss.hlast, E.hmain + (size_t) (r0 + keep[(size_t) i] - 1) * N, (size_t) N * 4,
                               cudaMemcpyDeviceToDevice, E.st));
            sl.draft_ready = false;
            if (!alive[(size_t) i] || !cand[(size_t) i]) continue;
            const int64_t p_new = ss.pos;   // commit_rows moved it: the cell of the token the next window feeds
            if (keep[(size_t) i] == 2) {
                nx.push_back({b, sl.draft, p_new - 1, E.hmain + (size_t) r0 * N});
                nx.push_back({b, last[(size_t) i], p_new, E.hmain + (size_t) (r0 + 1) * N});
                nx_last.push_back((int) nx.size() - 1);
            } else {
                nx.push_back({b, last[(size_t) i], p_new, E.hmain + (size_t) r0 * N});
                nx_last.push_back((int) nx.size() - 1);
            }
            nx_of.push_back(i);
        }
        if (!nx.empty()) {
            // (more rows than the window holds: the proposals come in two calls)
            for (size_t at = 0, k = 0; at < nx.size();) {
                size_t take = std::min<size_t>(nx.size() - at, Engine::kVerify);
                while (take > 0 && at + take < nx.size() && nx[at + take].slot == nx[at + take - 1].slot) --take;   // a slot's rows together
                E.mtp_rows(nx.data() + at, (int) take);
                for (; k < nx_last.size() && (size_t) nx_last[k] < at + take; ++k) {
                    Slot& sl = slots[(size_t) act[(size_t) nx_of[k]]];
                    sl.draft = E.mtp_pick[(size_t) nx_last[k] - at];
                    sl.draft_ready = true;
                }
                at += take;
            }
        }
    }

    /// One decode window over every active slot: one token each.
    void window() {
        poll_control();
        if (E.bmtp && E.mtp_d > 0) {   // proposals need a row to spare and a slot that decodes greedily
            int act = 0, greedy = 0;
            for (const Slot& sl : slots)
                if (sl.active) { ++act; greedy += !sl.smp.needs_logits(); }
            if (greedy > 0 && Engine::kVerify - act > 0) { window_mtp(); return; }
        }
        window_plain();
    }

    /// One token of every active slot, no proposals.
    void window_plain() {
        std::vector<Engine::Row> rows;
        std::vector<int> who;   // each row's slot
        for (int b = 0; b < E.n_sess; ++b)
            if (slots[(size_t) b].active) {
                rows.push_back({slots[(size_t) b].sess, slots[(size_t) b].x});
                who.push_back(b);
            }
        if (rows.empty()) return;
        E.step_rows(rows.data(), (int) rows.size());
        const size_t Nb = (size_t) E.m->g.n_embd * 4;
        for (size_t r = 0; r < rows.size(); ++r) {
            const int b = who[r];
            Slot& sl = slots[(size_t) b];
            Engine::Sess& ss = E.sess[(size_t) sl.sess];
            if (E.bmtp)   // (the hidden state a later first proposal starts from)
                CK(cudaMemcpyAsync(ss.hlast, E.hmain + r * (size_t) E.m->g.n_embd, Nb, cudaMemcpyDeviceToDevice, E.st));
            sl.draft_ready = false;
            fed(sl.sess, sl.x);
            int y;
            if (sl.smp.needs_logits()) {
                E.host_logits_row((int) r, lg);
                y = sample(lg, sl.smp, sl.hist, sl.rng);
            } else {
                y = E.rows_pick[r];
            }
            serve_print("BT %d %d\n", b, y);
            ++sl.produced;
            sl.hist.push_back(y);
            const char* fin = is_eos(y) ? "stop" : sl.stop ? "cancel" : sl.produced >= sl.max_new ? "length"
                              : ss.pos + 1 > E.ctx ? "length" : nullptr;
            if (fin) {
                serve_print("BDONE %d %lld %s %.1f\n", b, (long long) sl.produced, fin, ms_since(sl.t0));
                sl.active = false;
                slot_ended(sl.sess, false);
            } else {
                sl.x = y;
            }
        }
    }

    /// The length of session ss's held tokens when they are a proper start of `ids` with the same pictures, read with the
    /// same adapters' scales (`lora`), else 0.
    size_t match_len(const Engine::Sess& ss, const std::vector<int32_t>& ids, const std::vector<ImgKey>& imgs,
                     const std::vector<float>& lora) const {
        const size_t L = ss.live.size();
        if (L == 0 || L >= ids.size() || ss.lora != lora || !std::equal(ss.live.begin(), ss.live.end(), ids.begin()))
            return 0;
        std::vector<ImgKey> in_live;
        for (const ImgKey& k : imgs) {
            if (k.cell + k.n <= (int64_t) L) in_live.push_back(k);
            else if (k.cell < (int64_t) L) return 0;   // the held tokens end inside a picture
        }
        return in_live == ss.imgs ? L : 0;
    }

    /// The session a request alone takes: the one holding the longest start of its prompt, else an empty one, else the
    /// least recently used (never a decoding slot).
    int pick_session(const std::vector<int32_t>& ids, const std::vector<ImgKey>& imgs,
                     const std::vector<float>& lora) const {
        int best = -1, free_empty = -1, lru = -1;
        size_t bl = 0;
        for (int i = 0; i < E.n_sess; ++i) {
            const Engine::Sess& ss = E.sess[(size_t) i];
            if (ss.active) continue;
            const size_t l = match_len(ss, ids, imgs, lora);
            if (l > bl) { bl = l; best = i; }
            if (ss.live.empty() && free_empty < 0) free_empty = i;
            if (lru < 0 || ss.used < E.sess[(size_t) lru].used) lru = i;
        }
        return best >= 0 ? best : free_empty >= 0 ? free_empty : lru;
    }

    // The server's image encoder, which runs in this process, holds no VRAM of its own between pictures where the
    // server has it borrow (its "vision" section's "lend"): for a new picture it asks for memory on its card, loads
    // its weights there, encodes and gives it back, all between two requests.  This engine has no expert cache to
    // lend from.  In the hybrid mode it lends the streamed prompt path's staging buffers, which hold nothing
    // between prompts (place_experts makes them one block, as large as the loan the server announced); else VRAM
    // allocated of what the card has free, which the planner left that large where the model let it (kVramReserve).
    // LEND <device> <bytes> -> LENT <address> <bytes> 0 0 <what it is> or ERR; UNLEND -> UNLENT.
    // Only a prompt's chunks use the staging buffers (Engine::streamed), so a loan is made wherever none is read:
    // between requests, between the batch slots' decode windows (run), and between the decode steps of the request
    // on the control lines (poll_loan) - the encoder works beside the requests that decode.  A request that comes
    // while a loan is out waits for it (await_unlend): its prompt is read once the buffers are back.
    void* enc_at = nullptr;      // the loan that is out
    Engine* enc_on = nullptr;    // the stage whose GPU it is on
    bool enc_staging = false;    // that stage's staging buffers; else an allocation of its own
    bool in_process = false;     // the server and the engine in one process (run): the only loan there is
    static constexpr double kLoanWaitMs = 300000.0;   // how long a request waits for a loan to come back
    void enc_back() {
        if (enc_at == nullptr) return;
        enc_on->on();
        CK(cudaDeviceSynchronize());   // the encoder's last kernels
        if (enc_staging) {
            // as the start left them: a stream writes a layer's experts and no more, and MMQ reads a little past
            // them, where the bytes must be quants' (a float's bytes read as a block's scale can be a NaN)
            CK(cudaMemset(enc_on->stage[0], 0, enc_on->stage_bytes));
            CK(cudaDeviceSynchronize());
        } else {
            CK(dfree(enc_at));
        }
        enc_at = nullptr;
        enc_on = nullptr;
        enc_staging = false;
        E.on();
    }
    void lend(const std::string& line) {
        long long dev = -1;
        unsigned long long bytes = 0;
        std::string why, what;
        if (std::sscanf(line.c_str() + 5, "%lld %llu", &dev, &bytes) != 2 || bytes == 0) {
            why = "expected LEND <device> <bytes>";
        } else if (!in_process) {
            why = "a loan is memory of the engine's own process";
        } else if (enc_at != nullptr) {
            why = "a loan is out already";
        } else {
            Engine* S = nullptr;
            for (Engine* e = &E; e != nullptr && S == nullptr; e = e->nxt)
                if (e->dev == (int) dev) S = e;
            if (S == nullptr) {
                why = "the engine has no layers on GPU " + std::to_string(dev);
            } else {
                S->on();
                if (S->stage[0] != nullptr && S->stage_bytes >= bytes) {
                    CK(cudaStreamSynchronize(S->cst));   // nothing of a prompt still copies into them, or reads them
                    CK(cudaStreamSynchronize(S->st));
                    enc_at = S->stage[0];
                    enc_staging = true;
                    what = "the engine's staging buffers for a prompt's experts";
                } else {
                    // asked of the driver first: an allocation past the free memory does not fail under WDDM, it pages
                    size_t fr = 0, tot = 0;
                    CK(cudaMemGetInfo(&fr, &tot));
                    void* at = nullptr;
                    if (fr < bytes + kLoanFree) {
                        why = "GPU " + std::to_string(dev) + " has " + std::to_string(fr >> 20) + " MiB of VRAM free beside "
                              "the model, " + std::to_string((bytes + kLoanFree) >> 20) + " needed";
                    } else if (dmalloc(&at, (size_t) bytes) != cudaSuccess) {
                        (void) cudaGetLastError();
                        why = "GPU " + std::to_string(dev) + " gave no " + std::to_string(bytes >> 20) + " MiB";
                    } else {
                        enc_at = at;
                        enc_staging = false;
                        what = "VRAM the engine leaves free (" + std::to_string(fr >> 20) + " MiB now)";
                    }
                }
                if (enc_at != nullptr) enc_on = S;
                E.on();
            }
        }
        if (why.empty())
            serve_print("LENT %llu %llu 0 0 %s\n", (unsigned long long) (uintptr_t) enc_at, bytes, what.c_str());
        else
            serve_print("ERR no loan: %s\n", why.c_str());
    }
    void loan_line(const std::string& line) {   // LEND <device> <bytes>, or UNLEND
        if (line == "UNLEND") {
            enc_back();
            serve_print("UNLENT\n");
        } else {
            lend(line);
        }
    }
    /// The encoder's lines that are next among the queued ones (a slot's BSTOP / BYIELD apart), answered now: called
    /// between a request's decode steps, where no prompt is being read.
    void poll_loan() {
        for (;;) {
            std::string l;
            {
                std::lock_guard<std::mutex> lk(mu);
                auto it = lines.begin();
                while (it != lines.end() && (it->rfind("BSTOP ", 0) == 0 || it->rfind("BYIELD", 0) == 0)) ++it;
                if (it == lines.end() || !(*it == "UNLEND" || it->rfind("LEND ", 0) == 0)) return;
                l = std::move(*it);
                lines.erase(it);
            }
            loan_line(l);
        }
    }
    /// A request's line with a loan out: the loan comes back first (its UNLEND is a picture's encoding away), the
    /// slots decoding meanwhile.  One that does not come in kLoanWaitMs is taken back.
    void await_unlend() {
        if (enc_at == nullptr) return;
        const auto t0 = Clock::now();
        bool taken = false;
        while (enc_at != nullptr) {
            bool got = false, give_up = false;
            {
                std::unique_lock<std::mutex> lk(mu);
                const auto it = std::find(lines.begin(), lines.end(), std::string("UNLEND"));
                if (it != lines.end()) {
                    lines.erase(it);
                    got = true;
                } else if (eof || ms_since(t0) > kLoanWaitMs) {
                    give_up = true;
                } else if (n_active == 0) {
                    cv.wait_for(lk, std::chrono::milliseconds(20));
                }
            }
            if (got) {
                loan_line("UNLEND");
            } else if (give_up) {
                enc_back();
                taken = true;
            } else if (n_active > 0) {
                window();
            }
        }
        std::fprintf(stderr, "lamina serve: a request waited %.0f ms for the image encoder's loan%s\n", ms_since(t0),
                     taken ? ", which did not come back: taken back" : "");
    }

    void request(const std::string& line) {
        stop.store(false);   // a STOP that came between requests is stale; one from here on is this request's
        await_unlend();      // (a loan that is out: the prompt is read once the staging buffers are back)
        const bool adm = line[0] == 'B';
        const bool geni = line.rfind(adm ? "BGENI " : "GENI ", 0) == 0;
        if (!adm && !geni && line.rfind("GEN ", 0) != 0) {
            serve_print("ERR expected: GEN <max_new> [key=value ...] <id,id,...> or GENI <max_new> [keys] <file> <id,id,...>\n");
            return;
        }
        char* p = const_cast<char*>(line.c_str()) + (adm ? (geni ? 6 : 5) : (geni ? 5 : 4));
        int slot = -1;
        if (adm) {
            char* e1 = nullptr;
            const long b = std::strtol(p, &e1, 10);
            p = e1;
            slot = (int) b;
            if (e1 == line.c_str() || b < 0 || b >= (long) slots.size() || slots[(size_t) b].active || E.n_sess < 2) {
                serve_print("ERR BGEN: no such free slot (--batch %d) or a bad max_new\n", E.n_sess < 2 ? 0 : E.n_sess);
                return;
            }
        }
        const long long max_new = std::strtoll(p, &p, 10);
        Sampling smp;
        std::vector<uint64_t> img_keys;
        std::vector<float> lsc = E.lora_default;   // the adapters' scales (lora=: the request's, in their order)
        bool bad_lora = false;
        for (;;) {
            while (*p == ' ') ++p;
            const char* st0 = p;
            while (*p && *p != ' ') ++p;
            const std::string tk(st0, (size_t) (p - st0));
            const size_t eq = tk.find('=');
            if (tk.empty() || eq == std::string::npos) { p = const_cast<char*>(st0); break; }
            const std::string key = tk.substr(0, eq);
            const char* v = tk.c_str() + eq + 1;
            if (key == "temperature") smp.temperature = std::strtof(v, nullptr);
            else if (key == "top_p") smp.top_p = std::strtof(v, nullptr);
            else if (key == "top_k") smp.top_k = std::atoi(v);
            else if (key == "min_p") smp.min_p = std::strtof(v, nullptr);
            else if (key == "penalty_last_n") smp.last_n = std::atoi(v);
            else if (key == "penalty_repeat") smp.repeat = std::strtof(v, nullptr);
            else if (key == "penalty_freq") smp.freq = std::strtof(v, nullptr);
            else if (key == "penalty_present") smp.present = std::strtof(v, nullptr);
            else if (key == "seed") smp.seed = std::strtoull(v, nullptr, 10);
            else if (key == "lora") {   // every adapter's scale, comma-separated in --lora's order (the ones left out: 0)
                lsc.assign((size_t) E.n_ad, 0.f);
                size_t k = 0;
                for (const char* q = v; *q;) {
                    char* qe = nullptr;
                    const float s_ = std::strtof(q, &qe);
                    if (qe == q || k >= lsc.size() || !std::isfinite(s_)) { bad_lora = true; break; }
                    lsc[k++] = s_;
                    q = *qe == ',' ? qe + 1 : qe;
                    if (*qe != ',' && *qe != '\0') { bad_lora = true; break; }
                }
            }
            else if (key == "img_keys") {
                const char* q = v;
                img_keys.clear();
                while (*q) {
                    char* qe = nullptr;
                    const unsigned long long x = std::strtoull(q, &qe, 16);
                    if (qe == q) { img_keys.clear(); break; }
                    img_keys.push_back((uint64_t) x);
                    q = *qe == ',' ? qe + 1 : qe;
                    if (*qe != ',' && *qe != '\0') { img_keys.clear(); break; }
                }
            }
        }
        std::string emb;
        if (geni) {
            while (*p == ' ') ++p;
            const char* e = p;
            while (*e && *e != ' ') ++e;
            emb.assign(p, (size_t) (e - p));
            p = const_cast<char*>(e);
        }
        const std::vector<int32_t> ids = parse_ids(p);
        if (max_new < 1 || ids.empty() || (geni && emb.empty())) {
            serve_print("ERR bad request: max_new or the ids\n");
            return;
        }
        if (geni && !E.vision) {
            serve_print("ERR this engine was started without --vision\n");
            return;
        }
        if ((int64_t) ids.size() > E.ctx || max_new > (long long) E.ctx - (long long) ids.size()) {
            serve_print("ERR prompt (%zu tokens) + max_new (%lld) exceeds the context (%d)\n", ids.size(), max_new, E.ctx);
            return;
        }
        bool bad = false;
        for (int32_t t : ids) bad |= t < 0 || t >= (int32_t) E.m->g.n_vocab;
        if (bad) { serve_print("ERR a token id is outside the vocabulary\n"); return; }
        if (bad_lora) {
            serve_print("ERR lora=: a scale for each of the %d LoRA adapters at most, comma-separated\n", E.n_ad);
            return;
        }
        ImgReq ir;
        if (geni) {
            const std::string ve = build_images(emb, ids, E.m->g.n_embd, img_keys, ir);
            if (!ve.empty()) { serve_print("ERR %s\n", ve.c_str()); return; }
        }
        // ---- the session: the held state continues when the request extends it (the recurrent state cannot step back);
        // with the conversation cache also from a checkpoint of it, or from a conversation parked in RAM - and the
        // session is then the cache's choice, whichever slot the server named: the one that holds the conversation
        int s = adm ? slot : -1;
        size_t reuse = 0;
        double begin_ms = 0.0;   // the cache's own work: a conversation parked or taken back (the prompt's time has it)
        if (cache) {
            const auto tb = Clock::now();
            const int64_t held = cache_begin(ids, ir.imgs, lsc, s);
            begin_ms = ms_since(tb);
            if (held < 0) s = -1;
            else reuse = (size_t) held;
        } else if (!adm) {
            s = pick_session(ids, ir.imgs, lsc);
        }
        if (s < 0) {   // every session decodes in a slot: a request alone has none to run in
            serve_print("ERR every session is busy (batch slots decoding)\n");
            return;
        }
        Engine::Sess& ss = E.sess[(size_t) s];
        if (cache) {
            ss.live.assign(ids.begin(), ids.begin() + (long) reuse);
            ss.imgs.clear();
        } else {
            reuse = match_len(ss, ids, ir.imgs, lsc);
        }
        if (adm && !cache) {   // another session holding more of this prompt (the conversation was read alone): copy its state
            int src = -1;
            size_t best = reuse;
            for (int i = 0; i < E.n_sess; ++i) {
                if (i == s || E.sess[(size_t) i].active) continue;
                const size_t l = match_len(E.sess[(size_t) i], ids, ir.imgs, lsc);
                if (l > best) { best = l; src = i; }
            }
            if (src >= 0) {
                E.copy_session(src, s);
                reuse = best;
            }
        }
        E.select(s);
        if (reuse == 0 && !cache) {   // (the cache has set the state: an empty sequence where it found none)
            E.reset();
            ss.live.clear();
            ss.imgs.clear();
        }
        ss.lora = lsc;   // the scales its state is read with from here (the same ones when it continues)
        if (geni) E.set_images(ir.rows, ir.cell_row, ir.mrope, (int64_t) ids.size(), ir.delta);
        else E.clear_images();
        const auto t0 = Clock::now();
        const int64_t fresh = (int64_t) (ids.size() - reuse);
        bool cancelled = false;
        // The cache's checkpoints are taken as the read passes them: at its marks, and in a long prompt at a chunk's end
        // once --prompt-cache-every cells came since the last (not where a mark follows within a chunk, nor inside a
        // picture).  A mark inside a chunk is the state after the rows before it, which the chunk writes to the engine's
        // image as it passes (Engine::Chunk): the turn's read stays one chunk.  One a chunk: a second mark ends it, and
        // is the checkpoint of the state it leaves, as a long prompt's are.  The image goes to a buffer of the cache's
        // on a stream of its own, so nothing waits for the copy (Engine::snap_send); where the cache has no buffer
        // to give, the chunk ends at the mark and the state is copied then, the stages' queues empty first.
        std::vector<int64_t> marks;
        if (cache) marks = checkpoint_marks(ids, (int64_t) reuse, ir.imgs, geni);
        const int64_t every = std::max<int64_t>(1, copt.every / E.tmax) * E.tmax;
        int64_t noted = (int64_t) reuse, saved = (int64_t) reuse;   // the cache has the tokens up to `noted`
        auto note = [&](int64_t upto) {
            if (!cache || upto <= noted) return;
            cache->set(noted, ids.data() + noted, upto - noted);
            noted = upto;
        };
        auto checkpoint = [&](int64_t here) {   // of the state the queued chunk leaves
            note(here);
            saved = here;
            bool pinned = true;
            void* const h = checkpoint_buffer(pinned);
            if (h == nullptr) cache->checkpoint(here);
            else if (E.snap_state(static_cast<uint8_t*>(h), pinned)) cache->checkpoint(here, h);
        };
        const std::function<Engine::Chunk(int64_t)> plan = [&](int64_t i) -> Engine::Chunk {
            const int64_t c0 = (int64_t) reuse + i;
            Engine::Chunk c;
            c.T = (int) std::min<int64_t>(E.tmax, (int64_t) ids.size() - c0);
            const auto mark = std::upper_bound(marks.begin(), marks.end(), c0);
            if (mark == marks.end() || *mark >= c0 + c.T) return c;   // none inside it (one at its end: the callback's)
            void* const h = ckpt_cut ? nullptr : checkpoint_buffer(c.pinned);
            if (h == nullptr) {
                c.T = (int) (*mark - c0);
                return c;
            }
            if (mark + 1 != marks.end() && *(mark + 1) < c0 + c.T) c.T = (int) (*(mark + 1) - c0);
            note(*mark);   // the checkpoint is the cache's from here: its buffer fills as the chunk runs
            cache->checkpoint(*mark, h);
            saved = *mark;
            c.snap = (int) (*mark - c0);
            c.host = static_cast<uint8_t*>(h);
            return c;
        };
        // (a prompt whose end the cache held: nothing to read, the engine's output there is back with its state)
        const int64_t got = fresh == 0 ? 0 : E.process(ids.data() + reuse, fresh, [&](int64_t done) -> bool {
            if (cache && done < fresh) {
                const int64_t here = (int64_t) reuse + done;
                const auto next = std::lower_bound(marks.begin(), marks.end(), here);
                bool take = next != marks.end() && *next == here;
                if (!take && here - saved >= every &&
                    (next != marks.end() ? *next : (int64_t) ids.size()) - here >= E.tmax) {
                    take = true;
                    for (const ImgKey& k : ir.imgs) take = take && !(k.cell < here && here < k.cell + k.n);
                }
                if (take) checkpoint(here);
            }
            serve_print("PP %lld %lld %.0f %.1f\n", (long long) done, (long long) fresh, ms_since(t0),
                        (double) done / std::max(1e-9, ms_since(t0) / 1000.0));
            if (n_active > 0 && done < fresh) {
                poll_control();
                // the slots decoding go on between a read's chunks (the last chunk's
                for (int w = 0; w < read_windows && n_active > 0; ++w) window();   // logits must stay for the first token)
            } else {
                poll_control();
            }
            if (stop.load() && done < fresh) { cancelled = true; return false; }
            return true;
        }, cache ? &plan : nullptr);
        note((int64_t) reuse + got);
        if (cache && cache->whole_prompt && fresh > 1 && got == fresh && !cancelled) {
            // the prompt's end: its state and the logits the read left, before the first decode step moves them (not
            // after a read of one token - a stopped answer going on, as the server hands a request to a slot and
            // back: the conversation keeps the end of the prompt it was asked with)
            bool pinned = true;
            void* const h = checkpoint_buffer(pinned, true);
            if (h != nullptr && E.snap_state(static_cast<uint8_t*>(h), pinned, true)) cache->checkpoint((int64_t) ids.size(), h, true);
        }
        ss.live.insert(ss.live.end(), ids.begin() + (long) reuse, ids.begin() + (long) reuse + (long) got);
        ss.imgs.clear();
        bool cut = false;
        for (const ImgKey& k : ir.imgs) {
            if (k.cell + k.n <= (int64_t) ss.live.size()) ss.imgs.push_back(k);
            else if (k.cell < (int64_t) ss.live.size()) cut = true;
        }
        if (cancelled && cut) {   // stopped inside a picture: the state cannot be continued, the next read starts over
            ss.live.clear();
            ss.imgs.clear();
            if (cache) cache->set(0, static_cast<const int32_t*>(nullptr), 0);
        }
        E.sess[(size_t) E.cur].pos = E.pos;
        const double prompt_ms = begin_ms + ms_since(t0);
        if (cancelled) {
            if (cache) {
                E.snap_wait();
                cache->shed();
            }
            print_cache();
            serve_print("DONE 0 %zu %.1f 0.0 cancel 0 0 %zu 0 0 0 0 0.0 %lld 0\n", ids.size(), prompt_ms, reuse, (long long) got);
            if (adm) serve_print("BADM %d 0\n", slot);
            cache_idle();
            return;
        }
        std::mt19937_64 rng(smp.seed);
        std::vector<int32_t> hist = ids;
        const auto d0 = Clock::now();
        long long produced = 0, offered = 0, accepted = 0;
        const char* finish = "length";
        // greedy requests alone with the MTP block: speculative steps of mtp_d drafts (verified, so the text is the greedy
        // text); sampled ones, and requests beside others, one token at a time
        const bool spec = E.mtp_d > 0 && !smp.needs_logits() && !adm;
        std::vector<int32_t> fix, drafts;
        int32_t picks[16];
        int nx = smp.needs_logits() ? -1 : E.greedy();
        if (adm) {
            if (smp.needs_logits()) {
                E.host_logits(lg);
                nx = sample(lg, smp, hist, rng);
            }
            if (nx < 0) {   // no finite logit: the slot is not taken, and its state is not continued
                ss.live.clear();
                ss.imgs.clear();
                if (cache) {   // (nor is anything the cache holds: forgotten at the next request)
                    cache->drop(s);
                    poisoned = true;
                }
                std::fprintf(stderr, "lamina batch: slot %d: non-finite logits after the prompt: the request ends\n", slot);
                serve_print("DONE 0 %zu %.1f %.1f nonfinite 0 0 %zu 0 0 0 0 0.0 %lld 0\n", ids.size(), prompt_ms,
                            ms_since(d0), reuse, (long long) fresh);
                serve_print("BADM %d 0\n", slot);
                return;
            }
            serve_print("T %d\n", nx);
            ++produced;
            hist.push_back(nx);
            const char* fin = is_eos(nx) ? "stop" : stop.load() ? "cancel" : max_new <= 1 ? "length"
                              : E.pos + 1 > E.ctx ? "length" : nullptr;
            const bool cont = fin == nullptr;
            print_cache();
            serve_print("DONE %lld %zu %.1f %.1f %s 0 0 %zu 0 0 0 0 0.0 %lld 0\n", produced, ids.size(), prompt_ms,
                        ms_since(d0), cont ? "length" : fin, reuse, (long long) fresh);
            if (cont) {
                Slot& sl = slots[(size_t) slot];
                sl = Slot{};
                sl.active = true;
                sl.sess = s;
                sl.x = nx;
                sl.produced = 1;
                sl.max_new = max_new;
                sl.smp = smp;
                sl.rng = rng;
                sl.hist = hist;
                sl.t0 = Clock::now();
                ss.active = true;
                ++n_active;
            }
            serve_print("BADM %d %d\n", slot, cont ? 1 : 0);
            std::fprintf(stderr, "lamina batch: slot %d (session %d) takes a prompt of %zu tokens = %zu reused + %lld read in "
                         "%.0f ms (%.1f tok/s)%s\n", slot, s, ids.size(), reuse, (long long) fresh, prompt_ms,
                         prompt_ms > 0 ? 1000.0 * (double) fresh / prompt_ms : 0.0, cont ? "" : ", done at once");
            if (cache) std::fprintf(stderr, "lamina serve: prompt cache: %s\n", cache->summary().c_str());
            cache_idle();
            return;
        }
        for (;;) {
            poll_loan();   // (the encoder's loan beside this request's decoding: its prompt is read)
            if (smp.needs_logits()) {
                E.host_logits(lg);
                nx = sample(lg, smp, hist, rng);
            }
            auto emit = [&](int t) {   // false: the answer ends here
                if (t < 0) { finish = "nonfinite"; return false; }   // no finite logit: nothing of the row goes out
                serve_print("T %d\n", t);
                ++produced;
                hist.push_back(t);
                if (is_eos(t)) { finish = "stop"; return false; }
                if (produced >= max_new) return false;
                if (stop.load()) { finish = "cancel"; return false; }
                return true;
            };
            if (!emit(nx)) break;
            if (n_active > 0) window();
            if (spec && n_active == 0 && E.pos + E.mtp_d + 1 <= E.ctx) {
                const int a = E.spec_step(nx, fix, E.mtp_d, drafts, picks, &stop);
                offered += (long long) drafts.size();   // the drafts made (--spec-min-p can end them early)
                accepted += a;
                // the state now holds x and every accepted draft, whether or not the answer ends inside them (an end of
                // answer, max_new): the held tokens say so, so a later request that continues this conversation matches them
                // (a prompt that has the same tokens next reuses the state, any other reads from the start) and never reads
                // a token twice
                ss.live.push_back(nx);
                for (int i = 0; i < a; ++i) ss.live.push_back(drafts[(size_t) i]);
                fix.assign(drafts.begin(), drafts.begin() + a);
                bool ended = false;
                for (int i = 0; i < a && !ended; ++i)
                    if (!emit(drafts[(size_t) i])) ended = true;
                if (ended) break;
                nx = picks[a];
                continue;
            }
            if (E.pos + 1 > E.ctx) break;
            const int32_t one = nx;
            E.process(&one, 1, [](int64_t) {});
            ss.live.push_back(nx);
            fix.clear();
            if (!smp.needs_logits()) nx = E.greedy();
        }
        if (std::strcmp(finish, "nonfinite") == 0) {   // the state holds the non-finite rows: not continued
            ss.live.clear();
            ss.imgs.clear();
            poisoned = true;   // (nor is anything the cache holds: forgotten at the next request)
            if (cache) cache->set(0, static_cast<const int32_t*>(nullptr), 0);
            std::fprintf(stderr, "lamina serve: non-finite logits: the request ends, and the next one reads its prompt from "
                         "the start\n");
        } else if (cache && ss.live.size() > ids.size()) {   // the tokens the state took while it decoded
            cache->set((int64_t) ids.size(), ss.live.data() + ids.size(), (int64_t) (ss.live.size() - ids.size()));
        }
        E.sess[(size_t) E.cur].pos = E.pos;
        const double decode_ms = ms_since(d0);
        if (cache) {
            cache->touch(s);
            E.snap_wait();   // (shed may free a buffer)
            cache->shed();
        }
        print_cache();
        serve_print("DONE %lld %zu %.1f %.1f %s %lld %lld %zu 0 0 0 0 0.0 %lld 0\n", produced, ids.size(), prompt_ms,
                    decode_ms, finish, accepted, offered, reuse, (long long) fresh);
        std::fprintf(stderr, "lamina serve: session %d: prompt %zu tokens = %zu reused + %lld read in %.0f ms (%.1f tok/s), "
                     "%lld generated in %.0f ms (%.1f tok/s), drafts accepted %lld of %lld (%s)\n", s, ids.size(), reuse,
                     (long long) fresh, prompt_ms, prompt_ms > 0 ? 1000.0 * (double) fresh / prompt_ms : 0.0, produced,
                     decode_ms, decode_ms > 0 ? 1000.0 * (double) produced / decode_ms : 0.0, accepted, offered, finish);
        if (cache) std::fprintf(stderr, "lamina serve: prompt cache: %s\n", cache->summary().c_str());
        cache_idle();
    }

    int run() {
        if (E.n_sess > 1) serve_print("INFO batch_slots=%d slot_cache=1\n", E.n_sess);
        for (size_t a = 0; a < E.m->loras.size(); ++a) {   // the adapters, for the server's /lora-adapters
            const LoraAdapter& ad = E.m->loras[a];
            serve_print("ADAPTER {\"id\":%zu,\"path\":\"%s\",\"scale\":%.9g,\"alpha\":%.9g,\"rank\":%d,\"pairs\":%d,"
                        "\"task_name\":\"%s\",\"prompt_prefix\":\"%s\"}\n", a, json_escape(ad.path).c_str(),
                        (double) E.lora_default[a], (double) ad.alpha, ad.rank_max, ad.pairs,
                        json_escape(ad.task_name).c_str(), json_escape(ad.prompt_prefix).c_str());
        }
        print_cache();
        serve_print("READY %d stop\n", E.ctx);
        const lamina::program::ServeIo& io = lamina::program::serve_io();
        in_process = io.in_process;
        std::thread reader([&] {
            std::string l;
            while (io.read_line(l)) {
                if (!l.empty() && l.back() == '\r') l.pop_back();
                if (l == "STOP") { stop.store(true); continue; }
                std::lock_guard<std::mutex> lk(mu);
                lines.push_back(l);
                cv.notify_one();
            }
            std::lock_guard<std::mutex> lk(mu);
            eof = true;
            cv.notify_one();
        });
        lamina::program::ReaderEnd reader_end{reader, io};
        for (;;) {
            std::string line;
            bool have = false;
            {
                std::unique_lock<std::mutex> lk(mu);
                if (n_active == 0) cv.wait(lk, [&] { return eof || !lines.empty(); });
                if (!lines.empty()) {
                    line = lines.front();
                    lines.pop_front();
                    have = true;
                } else if (n_active == 0) {
                    return 0;
                }
            }
            if (!have) { window(); continue; }
            if (line.empty()) continue;
            if (line.rfind("BSTOP ", 0) == 0) {
                const int b = std::atoi(line.c_str() + 6);
                if (b >= 0 && b < (int) slots.size()) slots[(size_t) b].stop = true;
            } else if (line.rfind("BYIELD", 0) == 0) {
                // (a read that has ended)
            } else if (line.rfind("GEN", 0) == 0 || line.rfind("BGEN", 0) == 0) {
                request(line);
            } else if (line.rfind("LEND ", 0) == 0 || line == "UNLEND") {
                loan_line(line);
            } else if (line == "QUIT") {   // as the input's end: the running requests finish (STOP ends them first)
                std::lock_guard<std::mutex> lk(mu);
                eof = true;
                lines.clear();
            } else {
                serve_print("ERR expected: GEN <max_new> [key=value ...] <id,id,...>\n");
            }
            if (n_active > 0) window();
        }
    }
};

int serve(Engine& E, const CacheOpts& cache) {
    Serve sv(E, cache);
    return sv.run();
}

}  // namespace

/// The engine's run, or with `check_only` its command line's parse alone (no CUDA call, nothing loaded).
static int run_engine(int argc, char** argv, bool check_only) {
    std::string model, tokens, force, dump_full, emb_file, kv = "fp16";
    int ctx = 32768, chunk = 0, n_gen = 64, bench_pp = 0, bench_tg = 0, mtp_d = 0, cpu_threads = 0, hybrid_rows = 128;
    double expert_vram = -1.0;
    bool warm = true;
    int expert_slots = 0, batch = 0, draft_vocab = -1;
    float spec_min_p = 0.f;
    std::vector<int> stage_gpus, layer_split;   // the layer split: the GPUs (the first the main one) and their layer counts
    std::vector<LoraSpec> lora_specs;           // --lora / --lora-scaled, in their order
    bool lora_no_apply = false;                 // --lora-init-without-apply: every adapter at scale 0 to start
    auto int_list = [](const std::string& v) {
        std::vector<int> out;
        for (size_t i = 0; i < v.size();) {
            const size_t j = v.find(',', i);
            out.push_back(std::atoi(v.substr(i, j == std::string::npos ? std::string::npos : j - i).c_str()));
            if (j == std::string::npos) break;
            i = j + 1;
        }
        return out;
    };
    bool serve_mode = false, graph = true, vision = false, batch_mtp = false;
    CacheOpts cache_opts;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "lamina: %s needs a value\n", a.c_str()); throw Fatal{2}; }
            return argv[++i];
        };
        // lamina's names where they mean the same thing (the family switch passes lamina's command line on)
        if (a == "--model" || a == "--native") model = next();
        else if (a == "--max-new") n_gen = std::atoi(next().c_str());
        else if (a == "--no-token-graph") graph = false;
        else if (a == "--pool-workers") cpu_threads = std::atoi(next().c_str());
        else if (a == "--gpu") (void) next();   // applied by lamina before the switch (CUDA/HIP_VISIBLE_DEVICES)
        else if (a == "--greedy") {}            // --tokens decodes greedily here anyway
        else if (a == "--serve") serve_mode = true;
        else if (a == "--max-context" || a == "--ctx") ctx = std::atoi(next().c_str());
        else if (a == "--chunk" || a == "--prefill") { const std::string v = next(); chunk = v == "auto" ? 0 : std::atoi(v.c_str()); }
        else if (a == "--tokens") tokens = next();
        else if (a == "--tokens-file") {   // the prompt's ids in a file (commas or whitespace): past a command line's length
            const std::string path = next();
            FILE* f = std::fopen(path.c_str(), "rb");
            if (f == nullptr) { std::fprintf(stderr, "lamina: cannot read %s\n", path.c_str()); return 2; }
            char buf[65536];
            tokens.clear();
            for (size_t n; (n = std::fread(buf, 1, sizeof buf, f)) > 0;) tokens.append(buf, n);
            std::fclose(f);
        }
        else if (a == "--n") n_gen = std::atoi(next().c_str());
        else if (a == "--force") force = next();
        else if (a == "--emb") emb_file = next();   // --tokens with pictures: a lamina-vision file (the ids carry <|image_pad|> runs)
        else if (a == "--dump-full-logits") dump_full = next();
        else if (a == "--bench-pp") bench_pp = std::atoi(next().c_str());
        else if (a == "--bench-tg") bench_tg = std::atoi(next().c_str());
        else if (a == "--no-graph") graph = false;
        else if (a == "--no-warmup") warm = false;
        else if (a == "--drafts-per-step") mtp_d = std::atoi(next().c_str());   // the GGUF's own MTP block's drafts
        else if (a == "--mtp") {
            // Flash-Next's draft layer file, which this family has no use for; a number is the earlier releases'
            // --drafts-per-step, taken with a warning so their configs still start
            const std::string v = next();
            if (!v.empty() && std::isdigit((unsigned char) v[0])) {
                mtp_d = std::atoi(v.c_str());
                std::fprintf(stderr, "lamina: --mtp %s is --drafts-per-step %s on this path (--mtp names Flash-Next's "
                                     "draft layer)\n", v.c_str(), v.c_str());
            } else {
                std::fprintf(stderr, "lamina: ignoring --mtp %s (this family drafts with the GGUF's own MTP block: "
                                     "--drafts-per-step N)\n", v.c_str());
            }
        }
        else if (a == "--draft-vocab") draft_vocab = std::atoi(next().c_str());   // the MTP drafts pick among the first N token ids
        else if (a == "--kv") kv = next();                                        // the K / V cache's format: fp16 or q8_0
        else if (a == "--spec-min-p") spec_min_p = (float) std::atof(next().c_str());   // drafts while this sure (0: all of them)
        else if (a == "--vision") vision = true;                                  // pictures (GENI): the encoder is lamina-vision's
        else if (a == "--batch" || a == "--slots") batch = std::atoi(next().c_str());   // sessions decoding together
        else if (a == "--batch-mtp") batch_mtp = true;                                  // ... each verifying an MTP proposal
        else if (a == "--expert-vram") expert_vram = std::atof(next().c_str());   // GiB for the routed experts
        else if (a == "--cpu-threads") cpu_threads = std::atoi(next().c_str());
        else if (a == "--expert-slots") expert_slots = std::atoi(next().c_str());
        else if (a == "--stage-gpus") stage_gpus = int_list(next());     // CUDA ordinals: the family switch's --second-gpu
        else if (a == "--layer-split") layer_split = int_list(next());   // a layer count per GPU (default: fill the first)
        else if (a == "--hybrid-rows") hybrid_rows = std::atoi(next().c_str());
        else if (a == "--lora" || a == "--lora-scaled") {
            // llama.cpp's: --lora F[,F] at scale 1, --lora-scaled F:S[,F:S] (the scale after the last ':', so that a
            // drive's colon stays the path's)
            const std::string v = next();
            for (size_t at = 0; at <= v.size();) {
                const size_t comma = std::min(v.find(',', at), v.size());
                const std::string item = v.substr(at, comma - at);
                at = comma + 1;
                if (item.empty()) continue;
                LoraSpec sp;
                sp.path = item;
                if (a == "--lora-scaled") {
                    const size_t c = item.rfind(':');
                    char* e = nullptr;
                    const float s_ = c == std::string::npos ? 0.f : std::strtof(item.c_str() + c + 1, &e);
                    if (c == std::string::npos || c == 0 || e == item.c_str() + c + 1 || *e != '\0' || !std::isfinite(s_)) {
                        std::fprintf(stderr, "lamina: --lora-scaled takes FNAME:SCALE[,FNAME:SCALE], not '%s'\n",
                                     item.c_str());
                        return 2;
                    }
                    sp.path = item.substr(0, c);
                    sp.scale = s_;
                }
                lora_specs.push_back(sp);
            }
        }
        else if (a == "--lora-init-without-apply") lora_no_apply = true;   // loaded at scale 0: a request names its own
        // --serve's prompt cache (Flash-Next's options)
        else if (a == "--prompt-cache") cache_opts.chain = std::max(0, std::atoi(next().c_str()));
        else if (a == "--prompt-cache-every") cache_opts.every = std::max<int64_t>(1, std::atoll(next().c_str()));
        else if (a == "--conversation-cache-mib") cache_opts.park_mib = std::max<int64_t>(0, std::atoll(next().c_str()));
        else if (a == "--conversation-cache-slots") cache_opts.park_slots = std::max<int64_t>(0, std::atoll(next().c_str()));
        else if (a == "--conversation-cache-min-tokens") cache_opts.min_tokens = std::max<int64_t>(0, std::atoll(next().c_str()));
        else if (a == "--ram-reserve-mib") cache_opts.ram_reserve_mib = std::max<int64_t>(0, std::atoll(next().c_str()));
        else if (a == "--conversation-spill-dir") cache_opts.spill_dir = next();
        else if (a == "--conversation-spill-mib") cache_opts.spill_mib = std::max<int64_t>(0, std::atoll(next().c_str()));
        else if (a == "--tail-role-token") cache_opts.tail_role = std::atoll(next().c_str());
        else if (a.rfind("--control-vector", 0) == 0 || a.rfind("--cvec-", 0) == 0) {   // not a skip: it steers
            std::fprintf(stderr, "lamina: %s: control vectors run on Qwen3.8-Flash-Next's engine only\n", a.c_str());
            return 2;
        } else if (serve_mode && !check_only && a.rfind("--", 0) == 0) {
            // serve passes the config's args plus a few of its own: one this engine does not have is said and skipped
            // (with its value), not fatal
            const bool val = i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0;
            std::fprintf(stderr, "lamina: ignoring %s%s%s (not an option of this engine)\n", a.c_str(),
                         val ? " " : "", val ? argv[i + 1] : "");
            if (val) ++i;
        } else {
            std::fprintf(stderr, "lamina: unknown argument for Qwen3.6-35B-A3B and Qwen3.8-27B: %s\n", a.c_str());
            std::fprintf(stderr,
                         "usage: lamina-q36 --model GGUF [--max-context 32768] [--chunk 512] [--no-graph]\n"
                         "       (--serve | --tokens IDS [--n 64] [--force IDS] [--dump-full-logits F] | --bench-pp N "
                         "--bench-tg N)\n");
            return 2;
        }
    }
    if (model.empty()) { std::fprintf(stderr, "lamina: --model is required\n"); return 2; }
    if (mtp_d < 0 || mtp_d > 7) { std::fprintf(stderr, "lamina: --drafts-per-step takes 0 .. 7\n"); return 2; }
    if (chunk < 0 || chunk > 4096 || ctx < 256) { std::fprintf(stderr, "lamina: bad --chunk / --max-context\n"); return 2; }
    if (kv != "fp16" && kv != "q8_0") {
        std::fprintf(stderr, "lamina: --kv takes fp16 or q8_0 on this path\n");
        return 2;
    }
    for (const LoraSpec& sp : lora_specs) {   // the adapters' headers now (no GPU yet): a wrong file fails a check too
        std::string e;
        try {
            const lamina::GgufFile f(sp.path);
            LoraAdapter ad;
            std::vector<LoraPair> pairs;
            if (!lora_read(f, "", ad, pairs, e)) {
                std::fprintf(stderr, "lamina: --lora %s\n", e.c_str());
                return 2;
            }
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "lamina: --lora %s: %s\n", sp.path.c_str(), ex.what());
            return 2;
        }
    }
    if (check_only) return 0;
#if !defined(_WIN32)
    setenv("CUDA_MODULE_LOADING", "EAGER", 0);   // every kernel loaded with the context, not in the first prompt
#else
    if (std::getenv("CUDA_MODULE_LOADING") == nullptr) _putenv_s("CUDA_MODULE_LOADING", "EAGER");
#endif
    if (stage_gpus.empty()) {
        int cur = 0;
        CK(cudaGetDevice(&cur));
        stage_gpus.push_back(cur);
    }
    if (stage_gpus.size() > 1 && batch > 1) {
        std::fprintf(stderr, "lamina: the batch slots (\"parallel\" / --batch) run on one GPU: drop --second-gpu or "
                     "--batch\n");
        return 2;
    }
    Model m;
    m.kv_q8 = kv == "q8_0";
    std::string err;
    try {
        if (!load(model, m, err, mtp_d > 0, stage_gpus, layer_split, ctx, chunk > 0 ? chunk : 2048, lora_specs)) {
            std::fprintf(stderr, "lamina: %s\n", err.c_str());
            return 1;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "lamina: %s\n", e.what());
        return 1;
    }
    // a stage per GPU, chained; stage 0 drives them
    std::vector<std::unique_ptr<Engine>> stages;
    for (size_t i = 0; i < m.stage_dev.size(); ++i) stages.push_back(std::make_unique<Engine>());
    for (size_t i = 0; i < stages.size(); ++i) {
        Engine& S = *stages[i];
        S.dev = m.stage_dev[i];
        S.l0 = i == 0 ? 0 : m.stage_end[i - 1];
        S.l1 = m.stage_end[i];
        S.split = stages.size() > 1;
        S.nxt = i + 1 < stages.size() ? stages[i + 1].get() : nullptr;
        S.lst = stages.back().get();
        S.use_graph = graph;
    }
    Engine& E = *stages[0];
    if (E.split) {
        std::string where;
        for (size_t i = 0; i < stages.size(); ++i)
            where += std::string(i ? ", " : "") + "layers " + std::to_string(stages[i]->l0) + "-" + std::to_string(stages[i]->l1 - 1) +
                     " on GPU " + std::to_string(stages[i]->dev);
        std::fprintf(stderr, "lamina: the layer split: %s (the head and the MTP block on the last)\n", where.c_str());
        if (chunk == 0) chunk = 2048;   // every expert in VRAM: MMQ's rows at a 2048-token chunk
    }
    if (chunk == 0) {
        // auto: the largest chunk (4096 .. 1024) that fits beside every expert in VRAM; 2048 when they will not all fit (a prompt chunk streams every layer's
        // experts over PCIe once, so a bigger chunk pays that once for more tokens)
        size_t fr = 0, tot = 0, experts = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        for (const Layer& L : m.layers) experts += L.stride * (size_t) m.g.n_expert;
        // the engine's buffers at 512 (~1.1 GiB on gfx1201 at 8K, less the scores buffer's shrink) + the small reserve, and
        // what the sessions' K / V take past 8K cells: per cell a session's K / V and the decode attention's partials
        const double GiB = (double) (1ull << 30);
        const int ns = std::max(1, batch);
        int kv_layers = m.has_mtp ? 1 : 0;   // the attention layers' K / V and the MTP block's
        for (const Layer& L : m.layers) kv_layers += L.attn ? 1 : 0;
        const double kv_one = (double) kv_layers * (double) m.g.n_head_kv * (double) m.kv_row() * 2.0, kv_shared = 512.0;
        const double sess_extra = std::max(0.0, (double) ctx * (kv_one * ns + kv_shared) - 8192.0 * (kv_one + kv_shared));
        const double want = (double) experts + 1.1 * GiB + sess_extra;
        const bool fits = expert_slots <= 0 && (expert_vram < 0 ? want <= (double) fr
                                                               : (double) experts <= expert_vram * GiB);
        chunk = fits ? 512 : 2048;
        if (fits && expert_vram < 0) {
            // a bigger prompt chunk feeds the MMQ expert kernels more rows per expert (R9700, UD-Q4_K_XL, pp4096: 512 3390, 1024
            // 4495, 2048 5353, 4096 5750 tok/s): the largest that fits beside the experts.  Its buffers grow by about 0.5 MiB per
            // token (measured 0.47: 1.66 GiB more at 4096 than at 512), the sessions' K / V caches by 50 KB per cell and session.
            // With several slots a long prompt would hold the other slots' windows for a whole chunk: 1024 at most there.
            // (measured on a 32 GB R9700 at 8192 cells, 512 chunk: 1.77 GiB of buffers + K / V; the free room after init must still hold
            // every expert and the 0.77 GiB reserve, or place_experts goes hybrid)
            // (a loan asked of the card, kVramReserve: the chunk leaves it free as well)
            const int cap = ns > 1 ? 1024 : 4096;
            const double loan_room = (double) loan_keep(E.dev);
            for (int c : {4096, 2048, 1024})
                if (c <= cap && (double) experts + 2.75 * GiB + (double) (c - 512) * 0.5 * 1048576.0 + sess_extra + loan_room <= (double) fr) {
                    chunk = c;
                    break;
                }
        } else if (!fits) {
            // the hybrid mode: 2048 (a chunk streams every layer's experts over PCIe once); a small card, where the chunk's buffers
            // would take the slots' room, falls back to less
            const double fr_gib = (double) fr / GiB;
            chunk = fr_gib >= 7.5 ? 2048 : fr_gib >= 5.5 ? 1024 : 512;
        }
        if (const char* cv = std::getenv("LAMINA_Q36_CHUNK")) chunk = std::max(256, std::min(4096, std::atoi(cv)));   // force a chunk (512: the old default)
        if (std::getenv("LAMINA_Q36_PLACE_DEBUG"))
            std::fprintf(stderr, "lamina: chunk rule: free %.2f GiB, main experts %.2f GiB, chunk %d\n",
                         fr / 1073741824.0, experts / 1073741824.0, chunk);
    }
    for (auto& S : stages) {
        S->init(m, ctx, chunk, batch, vision, serve_mode && cache_opts.chain > 0);
        if (!place_experts(m, *S, expert_vram, cpu_threads, hybrid_rows, expert_slots, serve_mode, err)) {
            std::fprintf(stderr, "lamina: %s\n", err.c_str());
            return 1;
        }
        if (E.split && m.hybrid) {
            std::fprintf(stderr, "lamina: the layers of GPU %d do not fit its VRAM beside their K / V: --layer-split moves "
                         "layers, a shorter --max-context frees VRAM\n", S->dev);
            return 1;
        }
    }
    for (auto& S : stages)   // a loan the server's image encoder asked of a stage's card (kVramReserve): what it will be
        if (const size_t loan = (size_t) lamina::core::gpu_loan_asked(S->dev); loan > 0) {
            size_t fr = 0, tot = 0;
            S->on();
            CK(cudaMemGetInfo(&fr, &tot));
            if (S->stage_bytes >= loan)
                std::fprintf(stderr, "lamina: the image encoder borrows GPU %d's staging buffers for a picture (%.2f GiB; "
                             "its %zu MiB fit)\n", S->dev, (double) S->stage_bytes / (1 << 30), loan >> 20);
            else
                std::fprintf(stderr, "lamina: the image encoder borrows %zu MiB of GPU %d's free VRAM for a picture: %zu "
                             "MiB are free now%s\n", (loan + kLoanFree) >> 20, S->dev, fr >> 20,
                             fr >= loan + kLoanFree ? "" : ", too little (the model's placement comes first): the server "
                                                           "encodes elsewhere");
        }
    E.on();
    // every tensor is in VRAM or the pinned arena: the file's mapping goes (its pages stay in the OS's file cache, out
    // of the process's working set)
    m.mtp.src_g = m.mtp.src_u = m.mtp.src_d = nullptr;
    for (Layer& L : m.layers) L.src_g = L.src_u = L.src_d = nullptr;
    m.file.reset();
    E.on();
    // the adapters' scales a request that names none takes (every session starts with them)
    E.lora_default.assign(lora_specs.size(), 0.f);
    for (size_t a = 0; a < lora_specs.size(); ++a) E.lora_default[a] = lora_no_apply ? 0.f : lora_specs[a].scale;
    for (Engine::Sess& ss : E.sess) ss.lora = E.lora_default;
    struct Report {
        Engine& E;
        ~Report() {
            if (E.verify_k) std::fprintf(stderr, "lamina-q36 VERIFY: %lld comparisons, %lld mismatches\n", E.v_calls, E.v_bad);
            if (E.verify_k)
                for (const auto& kv : E.v_by)
                    std::fprintf(stderr, "lamina-q36 VERIFY   %s: %lld mismatches\n", kv.first.c_str(), kv.second);
            if (!E.m->hybrid) return;
            const long long n = E.hy_cpu + E.hy_gpu;
            std::fprintf(stderr, "lamina: hybrid experts: %lld of %lld batch entries on the GPU (%.1f%%), %lld on the "
                         "CPU pool in %.0f ms, %lld slot replacements\n", E.hy_gpu, n, n ? 100.0 * E.hy_gpu / n : 0.0,
                         E.hy_cpu, E.hy_cpu_ms, E.hy_swaps);
        }
    } report{E};
    E.reset();
    if (draft_vocab < 0 && std::getenv("LAMINA_Q36_DRAFT_VOCAB") != nullptr) draft_vocab = std::atoi(std::getenv("LAMINA_Q36_DRAFT_VOCAB"));
    for (auto& S : stages) {
        S->mtp_d = m.has_mtp ? mtp_d : 0;
        S->draft_n = std::max(0, draft_vocab);
        S->draft_min_p = spec_min_p;
    }
    E.bmtp = (batch_mtp || (std::getenv("LAMINA_BATCH_MTP") != nullptr && std::string(std::getenv("LAMINA_BATCH_MTP")) != "0")) && E.n_sess > 1 && E.mtp_d > 0;
    if (batch_mtp && !E.bmtp) std::fprintf(stderr, "lamina: --batch-mtp needs --batch N (2 or more) and --drafts-per-step N: ignored\n");
    if (warm) E.warmup();
    if (serve_mode) return serve(E, cache_opts);
    if (bench_pp > 0 || bench_tg > 0) {
        // llama-bench's shapes: a prompt of bench_pp tokens from a fresh state, then bench_tg tokens one at a time
        if (bench_tg < 0 || (int64_t) std::max(bench_pp, 1) + bench_tg + 8 > E.ctx) {
            std::fprintf(stderr, "lamina: --bench-pp %d and --bench-tg %d exceed the context (%d)\n", bench_pp, bench_tg, E.ctx);
            return 2;
        }
        std::vector<int32_t> toks((size_t) std::max(bench_pp, 1));
        std::mt19937 rng(1234);
        for (auto& t : toks) t = (int32_t) (rng() % 200000u);
        for (int rep = 0; rep < 3; ++rep) {
            E.reset();
            const auto t0 = Clock::now();
            E.process(toks.data(), (int64_t) toks.size(), [](int64_t) {});
            E.sync_all();
            const double pp_ms = ms_since(t0);
            const auto t1 = Clock::now();
            int nx = E.greedy();
            for (int i = 0; i < bench_tg; ++i) {
                const int32_t one = nx;
                E.process(&one, 1, [](int64_t) {});
                nx = E.greedy();
            }
            const double tg_ms = ms_since(t1);
            serve_print("bench %d: pp%d %.1f tok/s, tg%d %.1f tok/s\n", rep, (int) toks.size(),
                        (double) toks.size() / (pp_ms / 1000.0), bench_tg,
                        bench_tg > 0 ? (double) bench_tg / (tg_ms / 1000.0) : 0.0);
        }
        return 0;
    }
    const std::vector<int32_t> prompt = parse_ids(tokens.c_str()), forced = parse_ids(force.c_str());
    if (prompt.empty()) { std::fprintf(stderr, "lamina: --tokens, --serve or --bench-pp is required\n"); return 2; }
    if (!forced.empty()) n_gen = (int) forced.size() + 1;
    for (const std::vector<int32_t>* v : {&prompt, &forced})   // as a request's: the vocabulary's ids, the context
        for (const int32_t t : *v)
            if (t < 0 || t >= m.g.n_vocab) {
                std::fprintf(stderr, "lamina: the token %d is outside the vocabulary (%lld)\n", (int) t, (long long) m.g.n_vocab);
                return 2;
            }
    if (n_gen < 0 || (int64_t) prompt.size() + n_gen + 8 > E.ctx) {
        std::fprintf(stderr, "lamina: the prompt (%zu tokens) and %d generated exceed the context (%d)\n", prompt.size(),
                     n_gen, E.ctx);
        return 2;
    }
    FILE* ff = dump_full.empty() ? nullptr : std::fopen(dump_full.c_str(), "wb");
    std::vector<float> lg;
    std::vector<int32_t> gen;
    if (!emb_file.empty()) {
        if (!vision) { std::fprintf(stderr, "lamina: --emb needs --vision\n"); return 2; }
        ImgReq ir;
        const std::string ve = build_images(emb_file, prompt, m.g.n_embd, {}, ir);
        if (!ve.empty()) { std::fprintf(stderr, "lamina: %s\n", ve.c_str()); return 2; }
        E.set_images(ir.rows, ir.cell_row, ir.mrope, (int64_t) prompt.size(), ir.delta);
        std::fprintf(stderr, "lamina: %zu picture(s), %zu rows, text after them at cell + %d\n", ir.imgs.size(),
                     ir.rows.size() / (size_t) m.g.n_embd, (int) ir.delta);
    }
    const auto t0 = Clock::now();
    E.process(prompt.data(), (int64_t) prompt.size(), [](int64_t) {});
    const double pp_ms = ms_since(t0);
    const auto t1 = Clock::now();
    if (E.mtp_d > 0 && forced.empty() && !ff) {
        long long offered = 0, accepted = 0;
        std::vector<int32_t> fix, drafts;
        int32_t picks[16];
        int nx = E.greedy();
        while ((int) gen.size() < n_gen) {
            gen.push_back(nx);
            if ((int) gen.size() >= n_gen) break;
            const int a = E.spec_step(nx, fix, E.mtp_d, drafts, picks);
            offered += (long long) drafts.size();
            accepted += a;
            fix.assign(drafts.begin(), drafts.begin() + a);
            for (int i = 0; i < a && (int) gen.size() < n_gen; ++i) gen.push_back(drafts[(size_t) i]);
            nx = picks[a];
        }
        gen.resize((size_t) n_gen);
        serve_print("gen:");
        for (int t : gen) serve_print(" %d", t);
        serve_print("\n");
        std::fprintf(stderr, "lamina: prompt %zu tokens in %.1f ms, %zu generated in %.1f ms (%.1f tok/s), MTP %d drafts: "
                     "%lld of %lld accepted (%.0f%%)\n", prompt.size(), pp_ms, gen.size(), ms_since(t1),
                     gen.size() / (ms_since(t1) / 1000.0), E.mtp_d, accepted, offered,
                     offered ? 100.0 * accepted / offered : 0.0);
        return 0;
    }
    for (int i = 0; i < n_gen; ++i) {
        const int a = E.greedy();
        gen.push_back(a);
        if (ff) {
            E.host_logits(lg);
            std::fwrite(lg.data(), 4, lg.size(), ff);
        }
        if (i + 1 == n_gen) break;
        const int32_t nx = i < (int) forced.size() ? forced[(size_t) i] : a;
        E.process(&nx, 1, [](int64_t) {});
    }
    if (ff) std::fclose(ff);
    serve_print("gen:");
    for (int t : gen) serve_print(" %d", t);
    serve_print("\n");
    std::fprintf(stderr, "lamina: prompt %zu tokens in %.1f ms (%.1f tok/s), %zu generated in %.1f ms (%.1f tok/s)\n",
                 prompt.size(), pp_ms, prompt.size() / (pp_ms / 1000.0), gen.size(), ms_since(t1),
                 gen.size() / (ms_since(t1) / 1000.0));
    return 0;
}

int lamina::qwen35::run(int argc, char** argv) {
    // what the run allocates goes when it ends, on an error too (Fatal: its message is printed)
    Owned owned;
    g_owned = &owned;
    struct Unset {
        ~Unset() { g_owned = nullptr; }
    } unset;
    try {
        return run_engine(argc, argv, false);
    } catch (const Fatal& f) {
        return f.code;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "lamina: %s\n", e.what());
        return 1;
    }
}

int lamina::qwen35::check_args(int argc, char** argv) {
    try {
        return run_engine(argc, argv, true);
    } catch (const Fatal& f) {
        return f.code;
    }
}
