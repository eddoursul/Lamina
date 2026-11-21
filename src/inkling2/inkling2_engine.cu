// src/inkling2/inkling2_engine.cu - the engine path of Inkling 2 (GGUF architectures inkling2 and inkling2_loop:
// Thinking Machines Lab's Inkling 2 Mini), a driver of its own beside Flash-Next's, Qwen3.6's and Hy4-flash's, built
// from the engine's kernels and src/inkling2/inkling2_kernels.cuh:
//
//   * the model as lamina/inkling2/geometry.hpp reads it (llama.cpp's src/models/inkling2.cpp is the reference); the
//     dense weights on the GPU in their GGUF blocks (native_mmvq and iq_mmvq against q8_1 activations, BF16 through the
//     MMVF GEMV against floats), the small float matrices and vectors as floats; the experts as native blobs
//     [gate | up | down | scales] in slots on the GPU (native_expert_grouped); the embedding table in host memory;
//   * T <= kMaxT rows at consecutive positions through every layer application (a "visit": the looped layers run
//     n_pass times a token, each pass with cells and convolution state of its own): the depth mix over the block inputs,
//     the loop's messages and gates, attention with the distance bias and the sink over a window visit's ring of cells
//     or a full visit's every cell (FP16 or q8_0), the experts, the causal convolutions after each;
//   * generate mode: --tokens IDS [--n N] [--force IDS] [--dump-full-logits F] (bench/parity.py's interface through
//     scratch/q36/logits_to_parity.py); --serve: lamina-server's line protocol.
#include "inkling2_kernels.cuh"

#include "lamina/artifact/gguf_reader.hpp"
#include "lamina/core/cuda_host.hpp"
#include "lamina/core/gpu_select.hpp"
#include "lamina/core/prompt_cache.hpp"
#include "lamina/core/vmm.hpp"
#include "lamina/inkling2/engine.hpp"
#include "lamina/inkling2/geometry.hpp"
#include "lamina/kernels/bf16_gemv.hpp"
#include "lamina/kernels/iq_kernels.hpp"
#include "lamina/kernels/native_mmvq.hpp"
#include "lamina/platform/console_quit.hpp"
#include "lamina/platform/memory.hpp"
#include "lamina/platform/slice_read.hpp"
#include "lamina/prefill/moe_mmq.hpp"
#include "lamina/program/host_sampling.hpp"
#include "lamina/program/serve_io.hpp"

#include "ggml.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
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
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using lamina::program::serve_print;

namespace {

namespace K = lamina::kernels;
namespace MMQ = lamina::prefill::mmq;
using ik::kMaxT;
using lamina::core::ms_since;
using lamina::inkling2::Geometry;
using Clock = std::chrono::steady_clock;

#define CK(x)                                                                                                          \
    do {                                                                                                               \
        cudaError_t e_ = (x);                                                                                          \
        if (e_ != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "lamina: %s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__); \
            throw Fatal{1};                                                                                            \
        }                                                                                                              \
    } while (0)

/// A CUDA call that may fail without ending the run (a checkpoint's copy): false then, the error cleared
inline bool soft(cudaError_t e) {
    if (e == cudaSuccess) return true;
    (void) cudaGetLastError();
    return false;
}

/// The run's end on an error: its message is printed, run() returns `code` (the engine may run in lamina-server's
/// process, which an exit would end).
struct Fatal {
    int code;
};

/// What a run allocates, freed when it ends: device and pinned memory, streams.
struct Owned {
    std::mutex mu;
    std::vector<void*> dev, pinned;
    std::vector<cudaStream_t> streams;
    std::vector<cudaGraphExec_t> graphs;
    ~Owned() {
        cudaDeviceSynchronize();
        for (cudaGraphExec_t g : graphs) cudaGraphExecDestroy(g);
        for (cudaStream_t st : streams) cudaStreamDestroy(st);
        for (void* d : dev) cudaFree(d);
        for (void* h : pinned) cudaFreeHost(h);
        (void) cudaGetLastError();
    }
};
Owned* g_owned = nullptr;
template <class V, class T> void own(std::vector<V> Owned::*list, T v) {
    if (g_owned == nullptr) return;
    std::lock_guard<std::mutex> lk(g_owned->mu);
    (g_owned->*list).push_back((V) v);
}
template <class T> T* dalloc(size_t n) {
    void* p = nullptr;
    CK(cudaMalloc(&p, std::max<size_t>(n * sizeof(T), 256)));
    own(&Owned::dev, p);
    return (T*) p;
}
template <class T> T* dzero(size_t n) {
    T* p = dalloc<T>(n);
    CK(cudaMemset(p, 0, std::max<size_t>(n * sizeof(T), 256)));
    return p;
}
template <class T> T* halloc(size_t n) {   // pinned
    void* p = nullptr;
    CK(cudaHostAlloc(&p, std::max<size_t>(n * sizeof(T), 256), cudaHostAllocPortable));
    own(&Owned::pinned, p);
    return (T*) p;
}

constexpr size_t kWeightPad = 4096;       // zero bytes after each weight: a kernel may read past a short last row
constexpr int kChunks = 256;              // stretches of cells an attention launch splits a full visit's cells into at most
constexpr int64_t kGrow = 4096;           // cells a full visit's storage grows by
constexpr int64_t kContextFloor = 32768;  // a context not set is what fits, not under this
// the vocabulary's message tokens: where a prompt's read places the cache's checkpoints
constexpr int32_t kMsgFirst = 200000, kMsgLast = 200003;   // <|message_user|> .. <|message_tool|>
constexpr int32_t kEndMessage = 200010;                    // <|end_message|>
constexpr int kRingSub = ik::kSub / 4;    // cells a step of a window visit's attention (many short blocks: 1,024 cells)

// ------------------------------------------------------------------------------------------------ the model
struct Mat {
    const uint8_t* d = nullptr;   // device, GGUF blocks
    int type = -1;
    int n_in = 0, n_out = 0;
};

struct Layer {
    int il = 0;
    bool full = false, looped = false, first = false;   // first: its block's first layer
    int nq = 0;    // the attention mix's queries: 0 (the first layer: no mix), 1 (a window layer), 3 (a full one)
    int ext = 0;   // distances its bias table has
    Mat wq, wk, wv, wrel, wgate, wo;
    const float *sink = nullptr, *k_dyn = nullptr, *v_dyn = nullptr;
    const float *q_scale = nullptr, *rel_scale = nullptr, *rel_table = nullptr;
    const float *k_conv = nullptr, *v_conv = nullptr, *a_conv = nullptr, *f_conv = nullptr;
    const float* norm[3] = {};   // the attention inputs' norm weights: q, k, v (a window layer: one for the three)
    const float *attn_query = nullptr, *ffn_query = nullptr, *ffn_norm = nullptr;
    // the loop: the messages' queries [4][N] (q, k, v, ffn) and each one's gate
    const float* msg_query = nullptr;
    const float *cD[4] = {}, *cG[4] = {}, *cF[4] = {};
    // the MoE
    const float *router = nullptr, *route_bias = nullptr;
    K::NativeExpertLayout gl;
    const float *gu_scale = nullptr, *d_scale = nullptr;   // [n_expert]: the scales of an expert's rows' products, or none
    size_t stride = 0;             // a blob's bytes in a slot: a multiple of both formats' blocks
    uint8_t* slots = nullptr;      // device: n_slots blobs
    int n_slots = 0;
    int32_t* slot_of = nullptr;    // device [n_expert]: its slot or -1
};

struct Model {
    Geometry g;
    std::vector<Layer> layers;
    // the embedding table in host memory, as the file has it
    std::unique_ptr<uint8_t[]> embd;
    int embd_type = -1;
    size_t embd_row = 0;
    std::vector<float> tok_norm;
    const float *out_norm = nullptr, *out_query = nullptr;
    Mat head;
    int32_t eos = -1;
    int64_t trained_ctx = 0;
    double dense_gib = 0, expert_gib = 0, load_s = 0;
};

struct Loader {
    const lamina::GgufModel& gm;
    std::map<const lamina::TensorInfo*, const uint8_t*> pre;   // tensors read ahead, else the mapping's
    std::unique_ptr<uint8_t[]> pre_buf;
    /// Every tensor `want` takes, read in large pieces into host memory: a copy out of the mapping faults its pages in a
    /// few at a time
    bool read_ahead(const std::function<bool(const std::string&)>& want, int threads, bool uncached, std::string& e) {
        std::vector<lamina::platform::SliceRun> runs;
        std::vector<std::pair<const lamina::TensorInfo*, uint64_t>> at;
        uint64_t total = 0;
        for (size_t s = 0; s < gm.size(); ++s)
            for (const lamina::TensorInfo& t : gm.shard(s).tensors()) {
                const uint64_t n = lamina::tensor_payload_bytes(t);
                if (n == 0 || !gm.in_bounds(t, s) || !want(t.name)) continue;
                runs.push_back({gm.shard(s).path(), gm.shard(s).data_start() + t.offset, n, total, n, 1, 0});
                at.emplace_back(&t, total);
                total += (n + 63) / 64 * 64;
            }
        pre_buf.reset(new (std::nothrow) uint8_t[std::max<uint64_t>(total, 64)]);
        if (!pre_buf) return true;   // the mapping's, then
        uint8_t* b = pre_buf.get();
        if (!lamina::platform::read_slices(runs, threads, uncached, (uint64_t) 32 << 20, [b](int) { return b; }, e))
            return false;
        for (const auto& [t, o] : at) pre[t] = b + o;
        return true;
    }
    const lamina::TensorInfo* find(const std::string& name, size_t* s = nullptr) {
        const lamina::TensorInfo* t = gm.find(name, s);
        if (!t) {
            std::fprintf(stderr, "lamina: missing %s\n", name.c_str());
            throw Fatal{1};
        }
        return t;
    }
    bool has(const std::string& name) const { return gm.find(name) != nullptr; }
    const uint8_t* host(const std::string& name, const lamina::TensorInfo** ti = nullptr) {
        size_t s = 0;
        const lamina::TensorInfo* t = find(name, &s);
        if (ti) *ti = t;
        if (const auto it = pre.find(t); it != pre.end()) return it->second;
        return gm.shard(s).tensor_data(*t);
    }
    /// a matrix [n_in, n_out] in its GGUF blocks on the device: a format native_mmvq or iq_mmvq reads, or BF16
    Mat mat(const std::string& name) {
        const lamina::TensorInfo* t = nullptr;
        const uint8_t* src = host(name, &t);
        Mat m;
        m.type = (int) t->type;
        m.n_in = (int) t->shape[0];
        uint64_t rows = 1;
        for (size_t i = 1; i < t->shape.size(); ++i) rows *= t->shape[i];
        m.n_out = (int) rows;
        const bool takes = m.type == 30 ? m.n_in % 2 == 0
                                        : (K::native_mmvq_supported(m.type) || K::iq_supported(m.type)) &&
                                              K::iq_row_bytes(m.type, m.n_in) * rows == lamina::tensor_payload_bytes(*t);
        if (!takes) {
            std::fprintf(stderr, "lamina: %s is %s with rows of %d values, which this path's kernels do not read\n",
                         name.c_str(), t->type_name(), m.n_in);
            throw Fatal{1};
        }
        const uint64_t n = lamina::tensor_payload_bytes(*t);
        uint8_t* d = dalloc<uint8_t>(n + kWeightPad);
        CK(cudaMemcpy(d, src, n, cudaMemcpyHostToDevice));
        CK(cudaMemset(d + n, 0, kWeightPad));
        m.d = d;
        return m;
    }
    /// a tensor's values as floats, whatever its format (ggml's dequantizers)
    std::vector<float> floats(const std::string& name) {
        const lamina::TensorInfo* t = nullptr;
        const uint8_t* src = host(name, &t);
        const size_t n = (size_t) t->elements();
        std::vector<float> h(n);
        if (t->type == 0) {
            std::memcpy(h.data(), src, n * 4);
            return h;
        }
        const ggml_type_traits* tr = t->type < GGML_TYPE_COUNT ? ggml_get_type_traits((ggml_type) t->type) : nullptr;
        if (tr == nullptr || tr->to_float == nullptr) {
            std::fprintf(stderr, "lamina: %s is %s, which has no dequantizer here\n", name.c_str(), t->type_name());
            throw Fatal{1};
        }
        tr->to_float(src, h.data(), (int64_t) n);
        return h;
    }
    const float* f32(const std::string& name) {
        const std::vector<float> h = floats(name);
        float* d = dalloc<float>(h.size());
        CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
        return d;
    }
};

const float* f32_dev(const std::vector<float>& h) {
    float* d = dalloc<float>(h.size());
    CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    return d;
}

/// What the kernels take of a geometry; "" or the first limit it passes
std::string kernel_limits(const Geometry& g) {
    auto no = [](const std::string& what) { return "Inkling 2's path does not run " + what; };
    if (g.head_dim != 128) return no("a head of " + std::to_string(g.head_dim) + " values (its kernels take 128)");
    if (g.n_head / g.n_head_kv > ik::kQH) return no("more than 4 query heads a K/V head");
    if (g.n_rel > ik::kRelMax) return no("a distance bias of more than 32 values a head");
    if (g.n_expert > ik::kExpertMax || g.top_k * kMaxT > ik::kTB) return no("that many experts a layer or a token");
    if (g.n_pass > 1 && (g.gate_rank > ik::kRankMax || ik::kTB % g.gate_rank != 0)) return no("a loop gate of that rank");
    if ((g.n_layers + g.res_block - 1) / g.res_block > ik::kBlk) return no("more than 4 blocks of layers");
    if (g.n_embd % 256 != 0 || g.q_width() % 256 != 0) return no("rows that are no multiple of 256 values");
    return {};
}

bool load(const std::string& path, Model& m, int threads, bool file_cache, size_t reserve, std::string& err) {
    const auto t0 = Clock::now();
    const lamina::GgufModel gm = lamina::GgufModel::open(path);
    if (std::string e = lamina::inkling2::read_geometry(gm.meta(), m.g); !e.empty()) { err = e; return false; }
    if (std::string e = lamina::inkling2::check_layout(gm, m.g); !e.empty()) { err = e; return false; }
    if (std::string e = kernel_limits(m.g); !e.empty()) { err = e; return false; }
    const Geometry& g = m.g;
    if (const lamina::MetaValue* v = gm.meta().get("tokenizer.ggml.eos_token_id"); v && v->is_num()) m.eos = (int32_t) v->u;
    if (const lamina::MetaValue* v = gm.meta().get(g.arch + ".context_length"); v && v->is_num()) m.trained_ctx = (int64_t) v->u;
    const int64_t N = g.n_embd, E = g.n_expert;

    // the experts' layouts, and whether the model fits the GPU: the dense weights, every expert, the window visits'
    // cells and the rows' buffers (the full visits' cells take what is left)
    m.layers.resize((size_t) g.n_layers);
    uint64_t dense = 0, experts = 0, stage_max = 0;
    for (size_t s = 0; s < gm.size(); ++s)
        for (const lamina::TensorInfo& t : gm.shard(s).tensors())
            if (t.name.find("_exps.weight") == std::string::npos && t.name != "token_embd.weight")
                dense += lamina::tensor_payload_bytes(t) + kWeightPad;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        Layer& y = m.layers[(size_t) l];
        const std::string p = "blk." + std::to_string(l) + ".";
        const lamina::TensorInfo* tg = gm.find(p + "ffn_gate_up_exps.weight");
        const lamina::TensorInfo* td = gm.find(p + "ffn_down_exps.weight");
        if (!K::native_expert_supported((int) tg->type, (int) td->type, N, g.n_ff)) {
            err = p + ": experts of " + tg->type_name() + " gate/up and " + td->type_name() + " down rows, which the "
                  "expert kernels do not read at this geometry";
            return false;
        }
        y.gl = K::native_expert_layout((int) tg->type, (int) td->type, N, g.n_ff);
        if (lamina::tensor_payload_bytes(*tg) != (uint64_t) E * y.gl.down_off ||
            lamina::tensor_payload_bytes(*td) != (uint64_t) E * (y.gl.bytes - y.gl.down_off)) {
            err = p + ": the experts' tensors are not " + std::to_string(E) + " blobs of the layout's rows";
            return false;
        }
        y.gl.act = 1;
        y.gl.act_beta = g.situ_beta;
        y.gl.act_lbeta = g.situ_lbeta;
        const bool scaled = gm.find(p + "ffn_gate_up_exps.scale") || gm.find(p + "ffn_down_exps.scale");
        // A prompt chunk's MMQ reads past the down matrix's last row (its steps of 256 values against rows of n_ff,
        // meeting zero activations): zeros follow the matrix for as far as it reads, the scales after them.  It walks
        // from one expert's matrix to the next in the format's blocks: a blob apart is whole blocks of both formats.
        const size_t over = MMQ::built() && MMQ::supported((int) td->type) ? MMQ::tab_overread((int) td->type, g.n_ff) : 0;
        y.gl.scale_off = scaled ? (y.gl.bytes + over + 7) / 8 * 8 : 0;
        int eg = 0, bg = 0, ed = 0, bd = 0;
        lamina::block_geometry(tg->type, eg, bg);
        lamina::block_geometry(td->type, ed, bd);
        const size_t unit = std::lcm(std::lcm((size_t) std::max(bg, 1), (size_t) std::max(bd, 1)), (size_t) 8);
        y.stride = ((scaled ? y.gl.scale_off + 8 : y.gl.bytes + over) + unit - 1) / unit * unit;
        experts += (uint64_t) E * y.stride + kWeightPad;
        stage_max = std::max<uint64_t>(stage_max, (uint64_t) E * y.stride);
    }
    m.dense_gib = (double) dense / (double) (1ull << 30);
    m.expert_gib = (double) experts / (double) (1ull << 30);
    {
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        const uint64_t floor_cells = (uint64_t) kContextFloor * (uint64_t) g.n_full_visits() * 2 * (uint64_t) g.kv_width();   // q8_0's bytes, about
        if (dense + experts + floor_cells + reserve > fr) {
            char b[512];
            std::snprintf(b, sizeof b, "the model's %.2f GiB of dense weights and %.2f GiB of experts, with the cells of "
                          "%lld tokens and the %.1f GiB left free, pass this GPU's %.2f GiB of free VRAM", m.dense_gib,
                          m.expert_gib, (long long) kContextFloor, (double) reserve / (double) (1ull << 30),
                          (double) fr / (double) (1ull << 30));
            err = b;
            return false;
        }
    }

    Loader L{gm, {}, {}};
    if (!L.read_ahead([](const std::string& n) { return n.find("_exps.weight") == std::string::npos && n != "token_embd.weight"; },
                      threads, !file_cache, err))
        return false;
    {   // the embedding table: host memory, its rows dequantized as tokens come
        size_t s = 0;
        const lamina::TensorInfo* te = L.find("token_embd.weight", &s);
        const uint64_t n = lamina::tensor_payload_bytes(*te);
        const ggml_type_traits* tr = te->type < GGML_TYPE_COUNT ? ggml_get_type_traits((ggml_type) te->type) : nullptr;
        if (te->type != 0 && (tr == nullptr || tr->to_float == nullptr)) {
            err = std::string("token_embd.weight is ") + te->type_name() + ", which has no dequantizer here";
            return false;
        }
        m.embd_type = (int) te->type;
        m.embd_row = (size_t) (n / (uint64_t) g.n_vocab_padded);
        m.embd.reset(new (std::nothrow) uint8_t[n]);
        if (!m.embd) { err = "no memory for the embedding table"; return false; }
        uint8_t* b = m.embd.get();
        const std::vector<lamina::platform::SliceRun> run{{gm.shard(s).path(), gm.shard(s).data_start() + te->offset, n, 0, n, 1, 0}};
        if (!lamina::platform::read_slices(run, threads, !file_cache, (uint64_t) 32 << 20, [b](int) { return b; }, err))
            return false;
        m.tok_norm = L.floats("token_embd_norm.weight");
    }
    m.out_norm = L.f32("output_norm.weight");
    m.out_query = L.f32("output_res_query.weight");
    m.head = L.mat("output.weight");
    m.head.n_out = (int) g.n_vocab;   // the padded rows are not computed

    for (int64_t l = 0; l < g.n_layers; ++l) {
        Layer& y = m.layers[(size_t) l];
        const std::string p = "blk." + std::to_string(l) + ".";
        y.il = (int) l;
        y.full = g.full(l);
        y.looped = g.looped(l);
        y.first = l % g.res_block == 0;
        y.nq = l == 0 ? 0 : y.full ? 3 : 1;
        y.ext = (int) (y.full ? g.rel_extent : g.n_swa);
        y.wq = L.mat(p + "attn_q.weight");
        y.wk = L.mat(p + "attn_k.weight");
        y.wv = L.mat(p + "attn_v.weight");
        y.wrel = L.mat(p + "attn_rel.weight");
        y.wgate = L.mat(p + "attn_gate.weight");
        y.wo = L.mat(p + "attn_output.weight");
        y.sink = L.f32(p + "attn_sink_proj.weight");
        y.q_scale = L.f32(p + "attn_q_clip.weight");
        y.rel_scale = L.f32(p + "attn_rel_clip.weight");
        y.rel_table = L.f32(p + "attn_rel_bias.weight");
        y.k_conv = L.f32(p + "attn_k_conv.weight");
        y.v_conv = L.f32(p + "attn_v_conv.weight");
        y.a_conv = L.f32(p + "attn_post_conv.weight");
        y.f_conv = L.f32(p + "ffn_post_conv.weight");
        if (y.full) {
            y.norm[0] = L.f32(p + "attn_q_inp_norm.weight");
            y.norm[1] = L.f32(p + "attn_k_inp_norm.weight");
            y.norm[2] = L.f32(p + "attn_v_inp_norm.weight");
            y.k_dyn = L.f32(p + "attn_k_conv_proj.weight");
            y.v_dyn = L.f32(p + "attn_v_conv_proj.weight");
        } else {
            y.norm[0] = y.norm[1] = y.norm[2] = L.f32(p + "attn_norm.weight");
        }
        if (l > 0) y.attn_query = L.f32(p + "attn_res_query.weight");
        y.ffn_query = L.f32(p + "ffn_res_query.weight");
        y.ffn_norm = L.f32(p + "ffn_norm.weight");
        y.router = L.f32(p + "ffn_gate_inp.weight");
        if (L.has(p + "exp_probs_b.bias")) y.route_bias = L.f32(p + "exp_probs_b.bias");
        if (y.looped) {
            std::vector<float> q((size_t) (4 * N));
            for (int c = 0; c < lamina::inkling2::kCtrlCount; ++c) {
                const auto [name, idx] = lamina::inkling2::msg_query_at(g, l, c);
                const std::vector<float> all = L.floats(name);
                std::memcpy(q.data() + (size_t) c * (size_t) N, all.data() + (size_t) idx * (size_t) N, (size_t) N * 4);
                y.cD[c] = L.f32(lamina::inkling2::ctrl_name(g, l, c, "down"));
                y.cG[c] = L.f32(lamina::inkling2::ctrl_name(g, l, c, "gain"));
                y.cF[c] = L.f32(lamina::inkling2::ctrl_name(g, l, c, "forget"));
            }
            y.msg_query = f32_dev(q);
        }
    }

    // the experts: every one in a slot of its layer, read in large pieces through a pinned stage
    uint8_t* stage = nullptr;
    const bool stage_pinned = cudaHostAlloc((void**) &stage, stage_max, cudaHostAllocDefault) == cudaSuccess;
    std::unique_ptr<uint8_t[]> stage_heap;
    if (!stage_pinned) {
        (void) cudaGetLastError();
        stage_heap.reset(new (std::nothrow) uint8_t[stage_max]);
        stage = stage_heap.get();
        if (!stage) { err = "no memory to stage a layer's experts"; return false; }
    }
    struct StageFree {
        uint8_t* p;
        bool pinned;
        ~StageFree() {
            if (pinned) cudaFreeHost(p);
        }
    } stage_free{stage, stage_pinned};
    for (int64_t l = 0; l < g.n_layers; ++l) {
        Layer& y = m.layers[(size_t) l];
        const std::string p = "blk." + std::to_string(l) + ".";
        size_t sg = 0, sd = 0;
        const lamina::TensorInfo* tg = L.find(p + "ffn_gate_up_exps.weight", &sg);
        const lamina::TensorInfo* td = L.find(p + "ffn_down_exps.weight", &sd);
        const std::vector<lamina::platform::SliceRun> runs{
            {gm.shard(sg).path(), gm.shard(sg).data_start() + tg->offset, y.gl.down_off, 0, y.stride, E, 0},
            {gm.shard(sd).path(), gm.shard(sd).data_start() + td->offset, y.gl.bytes - y.gl.down_off, y.gl.down_off, y.stride, E, 0}};
        if (!lamina::platform::read_slices(runs, threads, !file_cache, (uint64_t) 32 << 20, [stage](int) { return stage; }, err))
            return false;
        // the bytes between one blob and the next: zeros (a prompt chunk's kernels read past a matrix's last row,
        // against zero activations: what they read there must be finite in the row's format)
        for (int64_t e = 0; e < E; ++e)
            std::memset(stage + (size_t) e * y.stride + y.gl.bytes, 0, y.stride - y.gl.bytes);
        if (y.gl.scale_off) {
            std::vector<float> sgu((size_t) E, 1.0f), sdn((size_t) E, 1.0f);
            if (L.has(p + "ffn_gate_up_exps.scale")) sgu = L.floats(p + "ffn_gate_up_exps.scale");
            if (L.has(p + "ffn_down_exps.scale")) sdn = L.floats(p + "ffn_down_exps.scale");
            for (int64_t e = 0; e < E; ++e) {
                const float two[2] = {sgu[(size_t) e], sdn[(size_t) e]};
                std::memcpy(stage + (size_t) e * y.stride + y.gl.scale_off, two, 8);
            }
            y.gu_scale = f32_dev(sgu);
            y.d_scale = f32_dev(sdn);
        }
        y.n_slots = (int) E;
            y.slots = dalloc<uint8_t>((size_t) E * y.stride + kWeightPad);
        CK(cudaMemcpy(y.slots, stage, (size_t) E * y.stride, cudaMemcpyHostToDevice));
        CK(cudaMemset(y.slots + (size_t) E * y.stride, 0, kWeightPad));
        std::vector<int32_t> so((size_t) E);
        for (int64_t e = 0; e < E; ++e) so[(size_t) e] = (int32_t) e;
        y.slot_of = dalloc<int32_t>((size_t) E);
        CK(cudaMemcpy(y.slot_of, so.data(), (size_t) E * 4, cudaMemcpyHostToDevice));
    }
    m.load_s = ms_since(t0) / 1000.0;
    std::fprintf(stderr, "lamina: Inkling 2 (%s): %lld layers, %lld applications a token (layers %lld-%lld %lld times), "
                 "%lld experts a layer; %.2f GiB of dense weights and %.2f GiB of experts on the GPU, loaded in %.1f s\n",
                 g.arch.c_str(), (long long) g.n_layers, (long long) g.n_visits(), (long long) g.loop_start,
                 (long long) g.loop_end - 1, (long long) g.n_pass, (long long) E, m.dense_gib, m.expert_gib, m.load_s);
    return true;
}

// ------------------------------------------------------------------------------------------------ the engine
/// The rows' buffers of a forward: the decode's (kMaxT rows), or a prompt chunk's (`cap` rows, `batch`: its matrix
/// products through llama.cpp's MMQ, its routed entries grouped by expert, a window visit's cells in position order).
struct Rows {
    int cap = 0;
    bool batch = false;
    float *x = nullptr, *prelude = nullptr, *pass_out = nullptr, *sblk = nullptr, *vblk = nullptr, *mixed = nullptr,
          *ctl = nullptr, *msg = nullptr, *nrm = nullptr;
    ik::Q81 *img = nullptr, *aimg = nullptr;
    float *qraw = nullptr, *graw = nullptr, *rraw = nullptr, *sink = nullptr, *kraw = nullptr, *vraw = nullptr,
          *dynk = nullptr, *dynv = nullptr, *tau = nullptr, *q = nullptr, *bias = nullptr, *kc = nullptr, *vc = nullptr,
          *part = nullptr, *aout = nullptr, *oraw = nullptr, *fraw = nullptr, *w = nullptr, *rows = nullptr,
          *logits = nullptr, *ctl_h = nullptr, *probs = nullptr;
    // the step record: the rows' positions, then the cells in use, the full visits' {stretch, stretches in use} and
    // a chunk's window range's cells
    int32_t *step_d = nullptr, *pos_d = nullptr, *live_d = nullptr, *dyn_d = nullptr, *wlive_d = nullptr;
    int32_t *ids = nullptr, *grp_start = nullptr, *n_groups = nullptr, *n_miss = nullptr, *ent_dst = nullptr,
            *ent_tok = nullptr;
    unsigned long long* grp_ptr = nullptr;
    void* scratch = nullptr;
    // pinned host rows
    float* emb_h = nullptr;
    int32_t* step_h = nullptr;
    // a chunk's: the inputs' and the attention output's MMQ images a layout, the routed entries by expert (their
    // token rows, their experts, each expert's bounds), their gate/up rows and hidden rows, a window visit's cells
    uint8_t *xq[3][3] = {}, *aq[3] = {}, *tq = nullptr, *hq = nullptr, *wk = nullptr, *wv = nullptr;
    bool have[4][3] = {};
    float *gu = nullptr, *h = nullptr;
    float *cz = nullptr, *cs1 = nullptr, *cs2 = nullptr;   // the loop gates' normed pair [T][2 N] and their two products
    int32_t *ent_row = nullptr, *ent_exp = nullptr, *bounds = nullptr, *ident = nullptr;
    int64_t ids_rows = 0;
    int32_t *ids_h = nullptr, *ent_row_h = nullptr, *ent_dst_h = nullptr, *ent_exp_h = nullptr, *bounds_h = nullptr;
    int head_rows() const { return batch ? 1 : cap; }
};

struct Engine {
    const Model* m = nullptr;
    cudaStream_t st = nullptr;
    int64_t ctx = 0;        // positions a sequence may take
    bool kv_q8 = false;
    bool grow = true;       // the full visits' cells mapped as the context grows (CUDA virtual memory)
    size_t reserve = 0;     // VRAM left free
    bool prof = false;      // LAMINA_INK_PROF: the steps timed (a wait after each)
    bool use_graphs = true; // a forward of T rows replayed from its captured graph
    std::map<int, cudaGraphExec_t> graphs;   // by rows * 16 + head rows
    int prefill = 1024;     // rows a prompt chunk takes at most (0: prompts through the decode's forwards)
    static constexpr int kChunkMin = 16;   // a shorter prompt part goes through the decode's forwards
    int feed_rows = kMaxT;                  // rows a decode forward of a prompt part takes (LAMINA_INK_ROWS: fewer, a test's)

    // a visit: one application of a layer
    struct Visit {
        int layer = 0, pass = 0;
        float* conv = nullptr;   // [C - 1] rows of: K's projection | V's | the attention's output | the MoE's
        ik::Cells cells{};
        std::unique_ptr<lamina::core::VmmRange> k_vm, v_vm;
    };
    std::vector<Visit> visits;
    float* conv_all = nullptr;       // every visit's convolution state
    size_t conv_floats = 0;          // a visit's
    uint8_t* rings = nullptr;        // the window visits' cells
    size_t ring_bytes = 0;           // all of them
    int64_t ring_cap = 0;
    size_t cell_bytes = 0;           // a cell's K (and its V)
    int64_t mapped = 0;              // cells the full visits have memory for

    // the sequence: the positions whose cells the state holds
    int64_t n_pos = 0;
    int64_t pos() const { return n_pos; }

    Rows D, P;                       // the decode's rows, a prompt chunk's
    Rows* lg_rows = nullptr;         // whose logits the last head wrote
    float* logits_h = nullptr;       // pinned
    std::unique_ptr<MMQ::Context> mmq;
    bool lay_used[3] = {};           // the MMQ activation layouts the dense formats take
    cublasHandle_t blas = nullptr;   // a chunk's float products (the loop gates, the router)
    ~Engine() {
        if (blas != nullptr) cublasDestroy(blas);
    }

    // ---- what a checkpoint of the prompt cache keeps (lamina/core/prompt_cache.hpp), the full visits' cells apart,
    // which stay where they are: [the window visits' rings | every visit's convolution state | the logits a prompt's
    // read left | int32 1 where they are held, int32 0].  A checkpoint the run takes as it goes leaves through an image
    // on the device: copied there behind the queued work, sent to the cache's buffer on a stream of its own, so neither
    // a prompt's next chunk nor the first decode step waits for the bus.  A buffer that is not pinned takes the image
    // through `snap_via`.
    bool cache_on = false;             // the image is made (init)
    uint8_t* snap = nullptr;           // the image
    uint8_t* snap_via = nullptr;       // pinned
    uint8_t* snap_to = nullptr;        // the buffer snap_via's contents belong in (null: none waits)
    cudaStream_t st_snap = nullptr;
    cudaEvent_t ev_snap = nullptr, ev_landed = nullptr;
    bool logits_held = false;          // the state came from a prompt's end: its logits are the decode rows' row 0
    size_t conv_bytes() const { return visits.size() * conv_floats * 4; }
    size_t state_bytes() const { return ring_bytes + conv_bytes() + (size_t) m->g.n_vocab * 4 + 8; }

    // timing
    std::map<std::string, double> prof_ms;
    Clock::time_point prof_t;

    int nblocks() const { return (int) ((m->g.n_layers + m->g.res_block - 1) / m->g.res_block); }
    /// a block whose input differs from pass to pass: one inside the loop after its first
    bool varying(int b) const {
        const Geometry& g = m->g;
        return g.n_pass > 1 && b * g.res_block > g.loop_start && b * g.res_block < g.loop_end;
    }
    float* blk_at(const Rows& R, int b, int pass) const {
        const size_t row = (size_t) R.cap * (size_t) m->g.n_embd;
        return varying(b) ? R.vblk + ((size_t) pass * ik::kBlk + (size_t) b) * row : R.sblk + (size_t) b * row;
    }

    void make_rows(Rows& R, int cap, bool batch) {
        const Geometry& g = m->g;
        R.cap = cap;
        R.batch = batch;
        const size_t N = (size_t) g.n_embd, H = (size_t) g.n_head, D_ = (size_t) g.head_dim, KV = (size_t) g.kv_width(),
                     Q = (size_t) g.q_width(), C = (size_t) g.n_conv, T = (size_t) cap, Rr = (size_t) g.n_rel,
                     F = (size_t) g.n_ff, E = (size_t) g.n_expert;
        const size_t ext_max = (size_t) std::max(g.rel_extent, g.n_swa);
        const size_t Th = (size_t) R.head_rows();   // rows the head and the q8_1 images serve
        R.x = dzero<float>(T * N);
        R.prelude = dzero<float>(T * N);
        R.pass_out = dzero<float>(T * N);
        R.sblk = dzero<float>(ik::kBlk * T * N);
        R.vblk = dzero<float>((size_t) g.n_pass * ik::kBlk * T * N);
        R.mixed = dzero<float>(4 * T * N);
        R.ctl = dzero<float>(3 * T * N);
        R.msg = dzero<float>(4 * T * N);
        R.nrm = dzero<float>(3 * T * N);
        R.img = dzero<ik::Q81>(3 * Th * (N / 32));
        R.qraw = dzero<float>(T * Q);
        R.graw = dzero<float>(T * Q);
        R.rraw = dzero<float>(T * H * Rr);
        R.sink = dzero<float>(T * H);
        R.kraw = dzero<float>(T * KV);
        R.vraw = dzero<float>(T * KV);
        R.dynk = dzero<float>(T * (size_t) g.n_head_kv * C);
        R.dynv = dzero<float>(T * (size_t) g.n_head_kv * C);
        R.tau = dzero<float>(T);
        R.q = dzero<float>(T * Q);
        R.bias = dzero<float>(T * H * (ext_max + 1));
        R.kc = dzero<float>(T * KV);
        R.vc = dzero<float>(T * KV);
        if (!batch) R.part = dzero<float>(T * H * (size_t) kChunks * (D_ + 2));
        R.aout = dzero<float>(T * Q);
        R.aimg = dzero<ik::Q81>(Th * (Q / 32));
        R.oraw = dzero<float>(T * N);
        R.fraw = dzero<float>(T * N);
        const size_t n_ent = T * (size_t) g.top_k;
        R.ids = dzero<int32_t>(n_ent);
        R.w = dzero<float>(n_ent);
        R.rows = dzero<float>(n_ent * N);
        R.probs = dzero<float>(T * E);
        R.ctl_h = dzero<float>(3 * T * (size_t) std::max<int64_t>(g.gate_rank, 1));
        R.logits = dzero<float>(Th * (size_t) g.n_vocab);
        R.step_d = dzero<int32_t>(T + 4);
        R.pos_d = R.step_d;
        R.live_d = R.step_d + T;
        R.dyn_d = R.step_d + T + 1;
        R.wlive_d = R.step_d + T + 3;
        R.emb_h = halloc<float>(T * N);
        R.step_h = halloc<int32_t>(T + 4);
        if (!batch) {
            R.grp_ptr = dzero<unsigned long long>(n_ent + 1);
            R.grp_start = dzero<int32_t>(n_ent + 1);
            R.n_groups = dzero<int32_t>(1);
            R.n_miss = dzero<int32_t>(1);
            R.ent_dst = dzero<int32_t>(n_ent);
            R.ent_tok = dzero<int32_t>(n_ent);
            R.scratch = dzero<uint8_t>(K::native_expert_scratch_bytes((int64_t) n_ent, g.n_ff));
            return;
        }
        for (int lay = 0; lay < 3; ++lay) {
            if (!lay_used[lay]) continue;
            for (int c = 0; c < 3; ++c) R.xq[c][lay] = dzero<uint8_t>(MMQ::q8_bytes((int64_t) T, (int64_t) N));
            R.aq[lay] = dzero<uint8_t>(MMQ::q8_bytes((int64_t) T, (int64_t) Q));
        }
        R.tq = dzero<uint8_t>(MMQ::q8_bytes((int64_t) n_ent, (int64_t) N));
        R.hq = dzero<uint8_t>(MMQ::q8_bytes((int64_t) n_ent, (int64_t) F));
        R.gu = dzero<float>(n_ent * 2 * F);
        R.h = dzero<float>(n_ent * F);
        R.cz = dzero<float>(T * 2 * N);
        R.cs1 = dzero<float>(T * N);
        R.cs2 = dzero<float>(T * N);
        R.ids_rows = MMQ::ids_table_rows((int64_t) n_ent);
        R.ent_dst = dzero<int32_t>((size_t) R.ids_rows);
        R.ident = dzero<int32_t>((size_t) R.ids_rows);
        MMQ::iota(R.ident, R.ids_rows, st);
        R.ent_row = dzero<int32_t>(n_ent);
        R.ent_exp = dzero<int32_t>(n_ent);
        R.bounds = dzero<int32_t>(E + 1);
        R.wk = dzero<uint8_t>((T + (size_t) g.n_swa) * cell_bytes);
        R.wv = dzero<uint8_t>((T + (size_t) g.n_swa) * cell_bytes);
        R.ids_h = halloc<int32_t>(n_ent);
        R.ent_row_h = halloc<int32_t>(n_ent);
        R.ent_dst_h = halloc<int32_t>(n_ent);
        R.ent_exp_h = halloc<int32_t>(n_ent);
        R.bounds_h = halloc<int32_t>(E + 1);
    }

    /// Whether a prompt chunk's products have their MMQ kernels on this GPU: "" or why not
    std::string batch_problem() {
        if (!MMQ::built()) return "this build has no MMQ kernels";
        const Geometry& g = m->g;
        for (const Layer& L : m->layers) {
            for (const Mat* W : {&L.wq, &L.wk, &L.wv, &L.wrel, &L.wgate, &L.wo}) {
                if (W->type == 30) continue;   // BF16: the GEMV, 8 rows at a time
                if (!MMQ::fits(W->type, W->n_out)) return "a dense matrix's format has no MMQ kernel here";
                lay_used[MMQ::ds_layout(W->type)] = true;
            }
            for (const int t : {L.gl.gu_type, L.gl.d_type})
                if (MMQ::native_fp4(t)) return "this GPU runs NVFP4 on its FP4 tensor cores, which the prompt kernels do not feed yet";
            if (!MMQ::fits(L.gl.gu_type, 2 * g.n_ff) || !MMQ::fits(L.gl.d_type, g.n_embd))
                return "an expert format has no MMQ kernel here";
            // gate/up's rows end on the kernels' step, so nothing is read past them into the down rows
            if (MMQ::tab_overread(L.gl.gu_type, g.n_embd) != 0) return "the experts' gate/up rows end inside an MMQ step";
        }
        return {};
    }

    void init(const Model& model, int64_t max_ctx, bool q8, bool grow_cells, size_t reserve_bytes) {
        m = &model;
        kv_q8 = q8;
        reserve = reserve_bytes;
        prof = std::getenv("LAMINA_INK_PROF") != nullptr;
        nan_probe = std::getenv("LAMINA_INK_NAN") != nullptr;
        xrow_on = std::getenv("LAMINA_INK_XROW") != nullptr;
        if (const char* r = std::getenv("LAMINA_INK_ROWS")) feed_rows = std::max(1, std::min(kMaxT, std::atoi(r)));
        if (prof || nan_probe || xrow_on) use_graphs = false;
        if (nan_probe) nan_at = dzero<long long>(1);
        const Geometry& g = m->g;
        const size_t N = (size_t) g.n_embd, D_ = (size_t) g.head_dim, KV = (size_t) g.kv_width(), C = (size_t) g.n_conv;
        CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        own(&Owned::streams, st);
        cell_bytes = kv_q8 ? (size_t) g.n_head_kv * (D_ / 32) * 34 : KV * 2;
        make_rows(D, kMaxT, false);
        logits_h = halloc<float>((size_t) g.n_vocab);
        if (prefill > 0) {
            if (const std::string why = batch_problem(); !why.empty()) {
                std::fprintf(stderr, "lamina: prompts go through the decode's forwards, %d rows at a time: %s\n", kMaxT,
                             why.c_str());
                prefill = 0;
            } else {
                mmq = std::make_unique<MMQ::Context>();
                if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS || cublasSetStream(blas, st) != CUBLAS_STATUS_SUCCESS) {
                    std::fprintf(stderr, "lamina: cuBLAS could not start\n");
                    throw Fatal{1};
                }
                make_rows(P, prefill, true);
                gemm(m->layers[0].router, (int) g.n_expert, (int) g.n_embd, P.nrm, 1, P.probs);   // cuBLAS's start, not a first prompt's
            }
        }

        // the visits: convolution state, cells
        conv_floats = (C - 1) * (2 * KV + 2 * N);
        const int64_t nv = g.n_visits();
        conv_all = dzero<float>((size_t) nv * conv_floats);
        ring_cap = (g.n_swa + kMaxT + kRingSub - 1) / kRingSub * kRingSub;
        int n_win = 0;
        for (int64_t v = 0; v < nv; ++v) n_win += !g.full(g.visit_layer(v));
        ring_bytes = (size_t) n_win * 2 * (size_t) ring_cap * cell_bytes;
        rings = dzero<uint8_t>(ring_bytes);
        visits.resize((size_t) nv);
        int wi = 0;
        for (int64_t v = 0; v < nv; ++v) {
            Visit& V = visits[(size_t) v];
            V.layer = (int) g.visit_layer(v);
            V.pass = (int) g.visit_pass(v);
            V.conv = conv_all + (size_t) v * conv_floats;
            V.cells.q8 = kv_q8 ? 1 : 0;
            if (!g.full(V.layer)) {
                V.cells.ring = 1;
                V.cells.cap = ring_cap;
                V.cells.k = rings + (size_t) (2 * wi) * (size_t) ring_cap * cell_bytes;
                V.cells.v = rings + (size_t) (2 * wi + 1) * (size_t) ring_cap * cell_bytes;
                ++wi;
            }
        }
        if (cache_on) {   // (before the context is sized: the image is VRAM the cells do not get)
            snap = dalloc<uint8_t>(state_bytes());
            snap_via = halloc<uint8_t>(state_bytes());
            CK(cudaStreamCreateWithFlags(&st_snap, cudaStreamNonBlocking));
            own(&Owned::streams, st_snap);
            CK(cudaEventCreateWithFlags(&ev_snap, cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&ev_landed, cudaEventDisableTiming));
            CK(cudaEventRecord(ev_landed, st_snap));
        }

        // the context: what was asked, else what the free VRAM holds of the trained one (the full visits' cells)
        const uint64_t per_pos = (uint64_t) g.n_full_visits() * 2 * cell_bytes;
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        const int64_t fits = (int64_t) ((fr > reserve ? fr - reserve : 0) / per_pos) / kGrow * kGrow;
        const int64_t trained = m->trained_ctx > 0 ? m->trained_ctx : 1048576;
        if (max_ctx > 0) {
            ctx = max_ctx;
            if (fits < ctx)
                std::fprintf(stderr, "lamina: --max-context %lld is past the %lld tokens this GPU's free VRAM holds cells "
                             "for: a sequence ends there\n", (long long) ctx, (long long) fits);
        } else {
            ctx = std::min(trained, fits);
            if (ctx < kContextFloor) {
                std::fprintf(stderr, "lamina: this GPU's free VRAM holds the cells of %lld tokens, under the %lld a context "
                             "that is not set starts at (--max-context N sets one)\n", (long long) fits,
                             (long long) kContextFloor);
                throw Fatal{1};
            }
        }
        grow = grow_cells && lamina::core::vmm_available();
        for (Visit& V : visits) {
            if (V.cells.ring) continue;
            V.cells.cap = ctx;
            const uint64_t bytes = (uint64_t) ctx * cell_bytes;
            if (grow) {
                V.k_vm = std::make_unique<lamina::core::VmmRange>();
                V.v_vm = std::make_unique<lamina::core::VmmRange>();
                if (!V.k_vm->reserve(bytes) || !V.v_vm->reserve(bytes)) {
                    std::fprintf(stderr, "lamina: the cells' address range could not be reserved\n");
                    throw Fatal{1};
                }
                V.cells.k = V.k_vm->base();
                V.cells.v = V.v_vm->base();
            } else {
                V.cells.k = dalloc<uint8_t>(bytes);
                V.cells.v = dalloc<uint8_t>(bytes);
            }
        }
        mapped = grow ? 0 : ctx;
        if (!ensure(std::min<int64_t>(ctx, kGrow))) {
            std::fprintf(stderr, "lamina: no VRAM for the first cells\n");
            throw Fatal{1};
        }
        std::fprintf(stderr, "lamina: a context of %lld tokens (%s cells: %.1f KiB a token over %lld full-attention "
                     "applications, %.0f MiB of window cells), %s; prompts %d rows a chunk\n", (long long) ctx,
                     kv_q8 ? "q8_0" : "FP16", (double) per_pos / 1024.0, (long long) g.n_full_visits(),
                     (double) ring_bytes / 1048576.0, grow ? "mapped as it grows" : "allocated whole",
                     prefill > 0 ? prefill : kMaxT);
    }

    /// Memory for the full visits' cells of positions [0, n): false when the VRAM above the reserve does not hold them
    bool ensure(int64_t n) {
        if (n <= mapped) return true;
        if (n > ctx) return false;
        const int64_t target = std::min<int64_t>(ctx, (n + kGrow - 1) / kGrow * kGrow);
        const uint64_t G = lamina::core::vmm_granularity();
        auto chunks = [&](int64_t cells) { return (int64_t) (((uint64_t) cells * cell_bytes + G - 1) / G); };
        int64_t ranges = 0;
        for (const Visit& V : visits) ranges += V.k_vm ? 2 : 0;
        const int64_t need = (chunks(target) - chunks(mapped)) * ranges;
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        if ((uint64_t) need * G + reserve > fr) return false;
        CK(cudaStreamSynchronize(st));
        auto none = [] { return (lamina::core::VmmChunk) 0; };
        for (Visit& V : visits)
            if (V.k_vm && (!V.k_vm->map_range(0, chunks(target), none) || !V.v_vm->map_range(0, chunks(target), none)))
                return false;
        mapped = target;
        return true;
    }

    // ---- the steps
    bool nan_probe = false, nan_found = false;   // LAMINA_INK_NAN: the first non-finite value named
    long long* nan_at = nullptr;
    int nan_visit = 0;
    void probe(const char* what, const float* x, size_t rows, size_t width) {
        if (!nan_probe || nan_found) return;
        const long long n = (long long) (rows * width), none = -1;
        CK(cudaMemcpyAsync(nan_at, &none, 8, cudaMemcpyHostToDevice, st));
        ik::nonfinite_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, st>>>(x, n, nan_at);
        long long at = -1;
        CK(cudaMemcpyAsync(&at, nan_at, 8, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        if (at < 0) return;
        nan_found = true;
        std::fprintf(stderr, "lamina: a non-finite value: visit %d (layer %d, pass %d), %s, row %lld, value %lld of %zu\n",
                     nan_visit, visits[(size_t) nan_visit].layer, visits[(size_t) nan_visit].pass, what,
                     at / (long long) width, at % (long long) width, width);
    }
    // LAMINA_INK_XROW: each visit's values under llama-eval-callback's names, for a comparison with llama.cpp's dump
    // of the same forward: the stream after the attention (ffn_inp) and after the MoE (l_out), its last row's first and
    // last three values and the rows' sum; q, K, V and (in a chunk) r, the first and last three of each dimension
    bool xrow_on = false;
    std::vector<float> xrow_h;
    void xrow(const char* what, int v, const float* x, int T) {
        if (!xrow_on) return;
        const size_t N = (size_t) m->g.n_embd, n = (size_t) T * N;
        xrow_h.resize(n);
        CK(cudaMemcpyAsync(xrow_h.data(), x, n * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        double sum = 0;
        for (const float f : xrow_h) sum += (double) f;
        const float* r = xrow_h.data() + (size_t) (T - 1) * N;
        std::fprintf(stderr, "xrow %s-%d rows %d: %.4f %.4f %.4f ... %.4f %.4f %.4f sum %.6f\n", what, v, T, r[0], r[1], r[2],
                     r[N - 3], r[N - 2], r[N - 1], sum);
    }
    void xten(const char* what, int v, const float* x, int T, int H, int W) {
        if (!xrow_on) return;
        const size_t n = (size_t) T * H * W;
        xrow_h.resize(n);
        CK(cudaMemcpyAsync(xrow_h.data(), x, n * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        auto edge = [](int n_, std::vector<int>& out) {
            out.clear();
            for (int i = 0; i < n_; ++i)
                if (n_ <= 6 || i < 3 || i >= n_ - 3) out.push_back(i);
        };
        std::vector<int> ts, hs, ds;
        edge(T, ts);
        edge(H, hs);
        edge(W, ds);
        std::fprintf(stderr, "xten %s-%d %d %d %d:", what, v, T, H, W);
        for (const int t : ts)
            for (const int h : hs)
                for (const int d : ds) std::fprintf(stderr, " %.4f", xrow_h[((size_t) t * H + h) * W + d]);
        std::fprintf(stderr, "\n");
    }
    void tick(const char* what) {
        if (!prof) return;
        CK(cudaStreamSynchronize(st));
        const auto now = Clock::now();
        prof_ms[what] += std::chrono::duration<double, std::milli>(now - prof_t).count();
        prof_t = now;
    }
    void copy(float* dst, const float* src, size_t n) {
        CK(cudaMemcpyAsync(dst, src, n * 4, cudaMemcpyDeviceToDevice, st));
    }
    /// y[t] = W x[t] for T rows: x as q8_1 images (xq), or as floats (xf) for a BF16 matrix
    void mm(const Mat& W, const ik::Q81* xq, const float* xf, float* y, int T) {
        if (W.type == 30) K::bf16_gemv_fp32_mmvf_multi(xf, (const uint16_t*) W.d, y, W.n_in, W.n_out, T, st);
        else if (K::native_mmvq_supported(W.type)) K::native_mmvq(W.type, W.d, xq, y, W.n_in, W.n_out, T, st);
        else K::iq_mmvq(W.type, W.d, xq, y, W.n_in, W.n_out, T, st);
    }
    /// The same for a chunk's T rows: MMQ over the rows' image for W's layout, made once an input (`input` 0-2 the
    /// attention's inputs, 3 its output)
    void mmb(Rows& R, const Mat& W, int input, const float* xf, float* y, int T) {
        if (W.type == 30) {
            for (int t0 = 0; t0 < T; t0 += kMaxT)
                K::bf16_gemv_fp32_mmvf_multi(xf + (size_t) t0 * W.n_in, (const uint16_t*) W.d, y + (size_t) t0 * W.n_out,
                                             W.n_in, W.n_out, std::min(kMaxT, T - t0), st);
            return;
        }
        const int lay = MMQ::ds_layout(W.type);
        uint8_t* xq = input == 3 ? R.aq[lay] : R.xq[input][lay];
        if (!R.have[input][lay]) {
            MMQ::quantize(xf, nullptr, xq, W.type, W.n_in, W.n_in, T, st);
            R.have[input][lay] = true;
        }
        mmq->run_dense(W.d, W.type, W.n_out, W.n_in, xq, T, y, W.n_out, st);
    }
    void mix(const float* const* blks, int nb, const float* cur, const float* queries, int nq, float* out, float* cur_n,
             int T) {
        ik::MixIn a{};
        for (int j = 0; j < nb; ++j) a.blk[j] = blks[j];
        a.cur = cur;
        a.q = queries;
        a.out = out;
        a.cur_n = cur_n;
        a.nb = nb;
        a.nq = nq;
        a.N = (int) m->g.n_embd;
        a.T = T;
        ik::mix_kernel<<<dim3((unsigned) T, (unsigned) std::max(nq, 1)), ik::kTB, 0, st>>>(a);
    }
    /// C [T][m] = B's rows [T][k] against A's rows [m][k]: c[t][i] = A[i] . b[t] (cuBLAS, a chunk's float products)
    /// (`ld`: floats from one row of C to the next, m when not given)
    void gemm(const float* A, int m_, int k, const float* B, int T, float* C, int ld = 0) {
        const float one = 1.f, zero = 0.f;
        if (cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, m_, T, k, &one, A, k, B, k, &zero, C, ld > 0 ? ld : m_) !=
            CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "lamina: a cuBLAS product failed\n");
            throw Fatal{1};
        }
    }
    /// A chunk's loop gate of one input: out = (1 - tanh(F h)) u + tanh(G h) m, h = D [u / rms(u); m / rms(m)]
    void ctrl_batch(Rows& R, const float* u, const float* msg, const float* Dm, const float* G, const float* F, float* out,
                    int T) {
        const int N = (int) m->g.n_embd, rank = (int) m->g.gate_rank;
        ik::ctrl_cat_kernel<<<(unsigned) T, ik::kTB, 0, st>>>(u, msg, R.cz, N);
        gemm(Dm, rank, 2 * N, R.cz, T, R.ctl_h);
        gemm(G, N, rank, R.ctl_h, T, R.cs1);
        gemm(F, N, rank, R.ctl_h, T, R.cs2);
        const long long cnt = (long long) T * N;
        ik::ctrl_gate_kernel<<<(unsigned) ((cnt + 255) / 256), 256, 0, st>>>(u, msg, R.cs1, R.cs2, out, cnt);
    }
    void conv_keep(float* state, const float* raw, int n, int T) {
        ik::conv_state_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, st>>>(state, raw, n, T, (int) m->g.n_conv);
    }

    /// A chunk's routed entries through their experts: R.rows [T * top_k][N] without the down rows' scales
    void experts_batch(Rows& R, const Layer& L, int T) {
        const Geometry& g = m->g;
        const int N = (int) g.n_embd, E = (int) g.n_expert, top_k = (int) g.top_k, F = (int) g.n_ff;
        const int n_ent = T * top_k;
        // the entries by expert, on the host: each expert's bounds, an entry's token row and its row of the output
        CK(cudaMemcpyAsync(R.ids_h, R.ids, (size_t) n_ent * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        std::fill(R.bounds_h, R.bounds_h + E + 1, 0);
        for (int i = 0; i < n_ent; ++i) ++R.bounds_h[R.ids_h[i] + 1];
        int max_rows = 0;
        for (int e = 0; e < E; ++e) {
            max_rows = std::max(max_rows, R.bounds_h[e + 1]);
            R.bounds_h[e + 1] += R.bounds_h[e];
        }
        std::vector<int32_t> at(R.bounds_h, R.bounds_h + E);
        for (int i = 0; i < n_ent; ++i) {
            const int e = R.ids_h[i], p = at[(size_t) e]++;
            R.ent_row_h[p] = i / top_k;
            R.ent_dst_h[p] = i;
            R.ent_exp_h[p] = e;
        }
        CK(cudaMemcpyAsync(R.bounds, R.bounds_h, (size_t) (E + 1) * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(R.ent_row, R.ent_row_h, (size_t) n_ent * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(R.ent_dst, R.ent_dst_h, (size_t) n_ent * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(R.ent_exp, R.ent_exp_h, (size_t) n_ent * 4, cudaMemcpyHostToDevice, st));
        MMQ::quantize(R.nrm, R.ent_row, R.tq, L.gl.gu_type, N, N, n_ent, st);
        MMQ::Product p{};
        p.w = L.slots;
        p.type = L.gl.gu_type;
        p.w_rows = 2 * F;
        p.w_cols = N;
        p.expert_bytes = L.stride;
        p.n = E;
        p.xq = R.tq;
        p.bounds = R.bounds;
        p.ids = R.ident;
        p.ids_rows = R.ids_rows;
        p.total_rows = n_ent;
        p.max_rows = max_rows;
        p.dst = R.gu;
        p.ld_dst = 2 * F;
        mmq->run(p, st);
        const long long cnt = (long long) n_ent * F;
        ik::situ_rows_kernel<<<(unsigned) ((cnt + 255) / 256), 256, 0, st>>>(R.gu, R.ent_exp, L.gu_scale, R.h, n_ent, F,
                                                                              g.situ_beta, g.situ_lbeta);
        MMQ::quantize(R.h, nullptr, R.hq, L.gl.d_type, F, F, n_ent, st);
        p.w = L.slots + L.gl.down_off;
        p.type = L.gl.d_type;
        p.w_rows = N;
        p.w_cols = F;
        p.xq = R.hq;
        p.ids = R.ent_dst;
        p.dst = R.rows;
        p.ld_dst = N;
        mmq->run(p, st);
    }

    /// Visit v over T rows of R at positions p0 ..: x in, x out
    void visit(int v, int T, Rows& R, int64_t p0) {
        const Geometry& g = m->g;
        Visit& V = visits[(size_t) v];
        const Layer& L = m->layers[(size_t) V.layer];
        const bool batch = R.batch;
        const int l = V.layer, pass = V.pass, res = (int) g.res_block;
        const int N = (int) g.n_embd, H = (int) g.n_head, HK = (int) g.n_head_kv, D_ = (int) g.head_dim, C = (int) g.n_conv;
        const int KV = HK * D_, Rr = (int) g.n_rel, E = (int) g.n_expert, top_k = (int) g.top_k, win = (int) g.n_swa;
        const size_t TN = (size_t) T * (size_t) N, TB = (size_t) T * (size_t) (N / 32);
        const bool msgs = pass > 0;
        float *ck = V.conv, *cv = ck + (size_t) (C - 1) * KV, *ca = cv + (size_t) (C - 1) * KV, *cf = ca + (size_t) (C - 1) * N;

        // a pass of the loop starts from the state before the loop; the pass before it speaks through messages
        if (L.looped && l == g.loop_start) {
            if (pass == 0) {
                copy(R.prelude, R.x, TN);
            } else {
                copy(R.pass_out, R.x, TN);
                copy(R.x, R.prelude, TN);
            }
        }
        const int use_pass = l >= g.loop_end ? (int) g.n_pass - 1 : pass;
        const float* blks[ik::kBlk] = {};
        const int nb = (l + res - 1) / res;
        for (int j = 0; j < nb; ++j) blks[j] = blk_at(R, j, use_pass);
        float* newblk = L.first ? blk_at(R, l / res, use_pass) : nullptr;
        if (msgs) {
            const float* bank[ik::kBlk] = {};
            int nbank = 0;
            for (int b = 0; b < nblocks(); ++b)
                if (varying(b)) bank[nbank++] = blk_at(R, b, pass - 1);
            mix(bank, nbank, R.pass_out, L.msg_query, 4, R.msg, nullptr, T);
        }

        // ---- attention: its inputs
        const float* u[3];
        if (L.nq == 0) {
            mix(blks, nb, R.x, nullptr, 0, nullptr, newblk, T);
            u[0] = u[1] = u[2] = R.x;
        } else {
            mix(blks, nb, R.x, L.attn_query, L.nq, R.mixed, newblk, T);
            for (int c = 0; c < 3; ++c) u[c] = R.mixed + (size_t) (L.nq == 3 ? c : 0) * TN;
        }
        const int n_in = msgs || L.full ? 3 : 1;
        if (msgs && batch) {
            for (int c = 0; c < 3; ++c) {
                float* out = R.ctl + (size_t) c * TN;
                ctrl_batch(R, u[c], R.msg + (size_t) c * TN, L.cD[c], L.cG[c], L.cF[c], out, T);
                u[c] = out;
            }
        } else if (msgs) {
            ik::CtrlIn a{};
            for (int c = 0; c < 3; ++c) {
                a.u[c] = u[c];
                a.m[c] = R.msg + (size_t) c * TN;
                a.D[c] = L.cD[c];
                a.G[c] = L.cG[c];
                a.F[c] = L.cF[c];
                a.out[c] = R.ctl + (size_t) c * TN;
                u[c] = a.out[c];
            }
            a.h = R.ctl_h;
            a.n = 3;
            a.N = N;
            a.R = (int) g.gate_rank;
            ik::ctrl_h_kernel<<<dim3((unsigned) a.R, (unsigned) T, 3), ik::kTB, 0, st>>>(a);
            ik::ctrl_apply_kernel<<<dim3((unsigned) ((N + ik::kTB - 1) / ik::kTB), (unsigned) T, 3), ik::kTB, 0, st>>>(a);
        }
        {
            ik::NormIn a{};
            for (int c = 0; c < n_in; ++c) {
                a.x[c] = u[c];
                a.w[c] = L.norm[c];
                a.y[c] = R.nrm + (size_t) c * TN;
                a.q[c] = batch ? nullptr : R.img + (size_t) c * TB;
            }
            a.n = n_in;
            a.N = N;
            a.eps = g.eps;
            ik::norm_kernel<<<dim3((unsigned) T, (unsigned) n_in), ik::kTB, 0, st>>>(a);
        }
        tick("attention inputs");
        nan_visit = v;
        probe("the visit's input x", R.x, (size_t) T, (size_t) N);
        probe("the attention inputs' norms", R.nrm, (size_t) T * (size_t) n_in, (size_t) N);
        const int ki = n_in == 3 ? 1 : 0, vi = n_in == 3 ? 2 : 0;
        const float *fq = R.nrm, *fk = R.nrm + (size_t) ki * TN, *fv = R.nrm + (size_t) vi * TN;
        if (batch) {
            std::memset(R.have, 0, sizeof R.have);
            mmb(R, L.wq, 0, fq, R.qraw, T);
            mmb(R, L.wgate, 0, fq, R.graw, T);
            mmb(R, L.wrel, 0, fq, R.rraw, T);
            mmb(R, L.wk, ki, fk, R.kraw, T);
            mmb(R, L.wv, vi, fv, R.vraw, T);
        } else {
            const ik::Q81 *iq = R.img, *ikk = R.img + (size_t) ki * TB, *iv = R.img + (size_t) vi * TB;
            mm(L.wq, iq, fq, R.qraw, T);
            mm(L.wgate, iq, fq, R.graw, T);
            mm(L.wrel, iq, fq, R.rraw, T);
            mm(L.wk, ikk, fk, R.kraw, T);
            mm(L.wv, iv, fv, R.vraw, T);
        }
        ik::sproj_kernel<<<dim3((unsigned) H, (unsigned) T), ik::kTB, 0, st>>>(L.sink, fq, R.sink, N, H);
        if (L.full) {
            ik::sproj_kernel<<<dim3((unsigned) (HK * C), (unsigned) T), ik::kTB, 0, st>>>(L.k_dyn, fk, R.dynk, N, HK * C);
            ik::sproj_kernel<<<dim3((unsigned) (HK * C), (unsigned) T), ik::kTB, 0, st>>>(L.v_dyn, fv, R.dynv, N, HK * C);
            ik::tau_kernel<<<(unsigned) ((T + 31) / 32), 32, 0, st>>>(R.pos_d, R.tau, T, g.log_alpha, (float) g.log_floor);
        }
        tick("attention projections");
        probe("q's projection", R.qraw, (size_t) T, (size_t) H * D_);
        probe("the gate's projection", R.graw, (size_t) T, (size_t) H * D_);
        probe("the distance projection", R.rraw, (size_t) T, (size_t) H * Rr);
        probe("k's projection", R.kraw, (size_t) T, (size_t) KV);
        probe("v's projection", R.vraw, (size_t) T, (size_t) KV);
        ik::q_clip_kernel<<<dim3((unsigned) T, (unsigned) H), (unsigned) D_, 0, st>>>(R.qraw, L.q_scale, L.full ? R.tau : nullptr,
                                                                                     R.q, H, D_, g.qk_clip);
        if (batch) {   // the table's products as one GEMM: r clipped in place, the sink taken off after
            ik::r_clip_kernel<<<dim3((unsigned) T, (unsigned) H), 32, 0, st>>>(R.rraw, L.rel_scale, L.full ? R.tau : nullptr,
                                                                              R.rraw, H, Rr, g.rel_clip);
            gemm(L.rel_table, L.ext, Rr, R.rraw, T * H, R.bias, L.ext + 1);
            const long long cnt = (long long) T * H * (L.ext + 1);
            ik::bias_fix_kernel<<<(unsigned) ((cnt + 255) / 256), 256, 0, st>>>(R.bias, R.sink, (long long) T * H, L.ext);
        } else {
            ik::BiasIn a{R.rraw, L.rel_scale, L.rel_table, R.sink, L.full ? R.tau : nullptr, R.bias, H, Rr, L.ext, g.rel_clip};
            ik::bias_kernel<<<dim3((unsigned) T, (unsigned) H), ik::kTB, 0, st>>>(a);
        }
        {
            ik::KvIn a{};
            a.raw[0] = R.kraw;
            a.raw[1] = R.vraw;
            a.state[0] = ck;
            a.state[1] = cv;
            a.w[0] = L.k_conv;
            a.w[1] = L.v_conv;
            a.dyn[0] = L.full ? R.dynk : nullptr;
            a.dyn[1] = L.full ? R.dynv : nullptr;
            a.out[0] = R.kc;
            a.out[1] = R.vc;
            a.HK = HK;
            a.D = D_;
            a.C = C;
            a.clip = g.qk_clip;
            a.eps = g.eps;
            ik::kv_conv_kernel<<<dim3((unsigned) T, 2, (unsigned) HK), (unsigned) D_, 0, st>>>(a);
        }
        conv_keep(ck, R.kraw, KV, T);
        conv_keep(cv, R.vraw, KV, T);
        // the cells the rows attend: the visit's own, or for a chunk's window visit a range in position order (the
        // win - 1 cells before the chunk out of the ring, then the chunk's)
        ik::Cells cells = V.cells;
        const bool slide = batch && V.cells.ring;
        if (slide) {
            ik::ring_to_range_kernel<<<dim3((unsigned) (win - 1), 2), 256, 0, st>>>(V.cells, R.wk, R.wv, (long long) p0, win,
                                                                                   (int) cell_bytes);
            cells.k = R.wk;
            cells.v = R.wv;
            cells.first = p0 - (win - 1);
            cells.cap = T + win - 1;
            cells.ring = 0;
        }
        ik::cells_put_kernel<<<dim3((unsigned) T, 2, (unsigned) HK), 128, 0, st>>>(cells, R.kc, R.vc, R.pos_d, HK, D_);
        tick("q, k, v, bias");
        xten("Qcur", v, R.q, T, H, D_);
        xten("Kcur", v, R.kc, T, HK, D_);
        xten("Vcur", v, R.vc, T, HK, D_);
        if (batch) xten("Rcur", v, R.rraw, T, H, Rr);
        probe("q", R.q, (size_t) T, (size_t) H * D_);
        probe("the bias", R.bias, (size_t) T, (size_t) H * (L.ext + 1));
        probe("k", R.kc, (size_t) T, (size_t) KV);
        probe("v", R.vc, (size_t) T, (size_t) KV);
        // ---- attention over the cells
        ik::AttnIn at{};
        at.c = cells;
        at.q = R.q;
        at.bias = R.bias;
        at.pos = R.pos_d;
        at.live = slide ? R.wlive_d : R.live_d;
        at.part = R.part;
        at.H = H;
        at.HK = HK;
        at.D = D_;
        at.ext = L.ext;
        at.win = L.full ? 0 : win;
        if (batch) {   // on the tensor cores: 32 rows of a query head a block, over the cells its rows see
            ik::TcIn a{cells, R.q, R.bias, R.graw, R.pos_d, R.aout, T, H, HK, L.ext, L.full ? 0 : win,
                       slide ? (long long) (T + win - 1) : (long long) (p0 + T)};
            const dim3 grid((unsigned) ((T + ik::kTcRows - 1) / ik::kTcRows), (unsigned) H);
            if (kv_q8) ik::attn_tc_kernel<true><<<grid, 256, 0, st>>>(a);
            else ik::attn_tc_kernel<false><<<grid, 256, 0, st>>>(a);
        } else {
            if (V.cells.ring) {
                at.sub = at.chunk = kRingSub;
                at.n_chunks = (int) (ring_cap / kRingSub);
            } else {   // the stretches from the step record: the same launch for any number of cells
                at.dyn = R.dyn_d;
                at.sub = ik::kSub;
                at.n_chunks = kChunks;
            }
            ik::attn_part_kernel<<<dim3((unsigned) at.n_chunks, (unsigned) HK, (unsigned) T), ik::kSub, 0, st>>>(at);
            ik::AttnOutIn a{R.part, R.graw, L.wo.type == 30 ? R.aout : nullptr, R.aimg, at.dyn, H, D_, at.n_chunks};
            ik::attn_out_kernel<<<dim3((unsigned) T, (unsigned) H), (unsigned) D_, 0, st>>>(a);
        }
        if (slide)
            ik::range_to_ring_kernel<<<dim3((unsigned) std::min<int64_t>(T, ring_cap), 2), 256, 0, st>>>(
                V.cells, R.wk, R.wv, (long long) p0, T, win, (int) cell_bytes);
        tick(L.full ? "attention, full" : "attention, window");
        if (batch || L.wo.type == 30) probe("the attention's output", R.aout, (size_t) T, (size_t) H * D_);
        if (batch) mmb(R, L.wo, 3, R.aout, R.oraw, T);
        else mm(L.wo, R.aimg, R.aout, R.oraw, T);
        const dim3 gN((unsigned) ((N + ik::kTB - 1) / ik::kTB), (unsigned) T);
        ik::post_conv_kernel<<<gN, ik::kTB, 0, st>>>(R.oraw, ca, L.a_conv, R.x, N, C, L.first ? 1 : 0);
        conv_keep(ca, R.oraw, N, T);
        tick("attention output");
        probe("the output projection", R.oraw, (size_t) T, (size_t) N);
        probe("x after the attention", R.x, (size_t) T, (size_t) N);
        xrow("ffn_inp", v, R.x, T);

        // ---- the MoE
        const int nb2 = l / res + 1;
        if (L.first) blks[nb2 - 1] = newblk;
        mix(blks, nb2, R.x, L.ffn_query, 1, R.mixed, nullptr, T);
        const float* fu = R.mixed;
        if (msgs && batch) {
            ctrl_batch(R, R.mixed, R.msg + (size_t) 3 * TN, L.cD[3], L.cG[3], L.cF[3], R.ctl, T);
            fu = R.ctl;
        } else if (msgs) {
            ik::CtrlIn a{};
            a.u[0] = R.mixed;
            a.m[0] = R.msg + (size_t) 3 * TN;
            a.D[0] = L.cD[3];
            a.G[0] = L.cG[3];
            a.F[0] = L.cF[3];
            a.out[0] = R.ctl;
            a.h = R.ctl_h;
            a.n = 1;
            a.N = N;
            a.R = (int) g.gate_rank;
            ik::ctrl_h_kernel<<<dim3((unsigned) a.R, (unsigned) T, 1), ik::kTB, 0, st>>>(a);
            ik::ctrl_apply_kernel<<<dim3((unsigned) ((N + ik::kTB - 1) / ik::kTB), (unsigned) T, 1), ik::kTB, 0, st>>>(a);
            fu = R.ctl;
        }
        {
            ik::NormIn a{};
            a.x[0] = fu;
            a.w[0] = L.ffn_norm;
            a.y[0] = R.nrm;
            a.q[0] = batch ? nullptr : R.img;
            a.n = 1;
            a.N = N;
            a.eps = g.eps;
            ik::norm_kernel<<<dim3((unsigned) T, 1), ik::kTB, 0, st>>>(a);
        }
        const int n_ent = T * top_k;
        const float w_scale = g.w_scale != 0.f ? g.w_scale : 1.0f;
        if (batch) {
            gemm(L.router, E, N, R.nrm, T, R.probs);
            ik::PickIn a{R.probs, L.route_bias, R.ids, R.w, E, top_k, w_scale, 1};
            ik::pick_kernel<<<(unsigned) T, 32, 0, st>>>(a);
            tick("MoE input and router");
            probe("the MoE's input", R.nrm, (size_t) T, (size_t) N);
            probe("the router's weights", R.w, (size_t) T, (size_t) top_k);
            experts_batch(R, L, T);
            tick("experts");
            probe("the experts' gate/up rows", R.gu, (size_t) n_ent, (size_t) 2 * g.n_ff);
            probe("the experts' hidden rows", R.h, (size_t) n_ent, (size_t) g.n_ff);
            probe("the experts' rows", R.rows, (size_t) n_ent, (size_t) N);
            ik::expert_sum_kernel<<<gN, ik::kTB, 0, st>>>(R.rows, R.w, R.fraw, N, top_k, L.d_scale, R.ids);
        } else {
            ik::route_dot_kernel<<<dim3((unsigned) E, (unsigned) T), ik::kTB, 0, st>>>(L.router, R.nrm, R.probs, E, N);
            ik::PlanIn a{R.probs, L.route_bias, L.slot_of, (unsigned long long) (uintptr_t) L.slots, L.stride, R.ids, R.w,
                         R.grp_ptr, R.grp_start, R.n_groups, R.ent_dst, R.ent_tok, R.n_miss, T, E, top_k, w_scale};
            ik::pick_plan_kernel<<<1, ik::kTB, 0, st>>>(a);
            tick("MoE input and router");
            K::native_expert_grouped(L.gl, R.grp_ptr, R.grp_start, R.n_groups, R.ent_dst, R.ent_tok, n_ent, n_ent, R.img,
                                     (int64_t) (N / 32) * 36, R.scratch, R.rows, st);
            tick("experts");
            ik::expert_sum_kernel<<<gN, ik::kTB, 0, st>>>(R.rows, R.w, R.fraw, N, top_k, nullptr, nullptr);
        }
        ik::post_conv_kernel<<<gN, ik::kTB, 0, st>>>(R.fraw, cf, L.f_conv, R.x, N, C, 0);
        conv_keep(cf, R.fraw, N, T);
        tick("MoE output");
        probe("the MoE's output", R.fraw, (size_t) T, (size_t) N);
        probe("x after the MoE", R.x, (size_t) T, (size_t) N);
        xrow("l_out", v, R.x, T);
    }

    /// The logits of rows [r0, r0 + n) of R's last forward, into its logits rows [0, n)
    void head(Rows& R, int r0, int n) {
        const Geometry& g = m->g;
        const int N = (int) g.n_embd;
        const float* blks[ik::kBlk] = {};
        for (int b = 0; b < nblocks(); ++b) blks[b] = blk_at(R, b, (int) g.n_pass - 1) + (size_t) r0 * N;
        mix(blks, nblocks(), R.x + (size_t) r0 * N, m->out_query, 1, R.mixed, nullptr, n);
        ik::NormIn a{};
        a.x[0] = R.mixed;
        a.w[0] = m->out_norm;
        a.y[0] = R.nrm;
        a.q[0] = R.img;
        a.n = 1;
        a.N = N;
        a.eps = g.eps;
        ik::norm_kernel<<<dim3((unsigned) n, 1), ik::kTB, 0, st>>>(a);
        mm(m->head, R.img, R.nrm, R.logits, n);
        if (g.softcap > 0) {
            const long long cnt = (long long) n * g.n_vocab;
            ik::softcap_kernel<<<(unsigned) ((cnt + 255) / 256), 256, 0, st>>>(R.logits, cnt, g.softcap);
        }
        lg_rows = &R;
        tick("head");
    }

    /// T tokens' embedding rows as the first layer reads them (normed by the embedding's weight), in R.emb_h
    void embed(Rows& R, const int32_t* tok, int T) {
        const Geometry& g = m->g;
        const size_t N = (size_t) g.n_embd;
        const ggml_type_traits* tr = m->embd_type != 0 ? ggml_get_type_traits((ggml_type) m->embd_type) : nullptr;
        for (int t = 0; t < T; ++t) {
            float* row = R.emb_h + (size_t) t * N;
            const uint8_t* src = m->embd.get() + (size_t) tok[t] * m->embd_row;
            if (tr) tr->to_float(src, row, (int64_t) N);
            else std::memcpy(row, src, N * 4);
            double ss = 0;
            for (size_t i = 0; i < N; ++i) ss += (double) row[i] * (double) row[i];
            const float inv = 1.0f / std::sqrt((float) (ss / (double) N) + g.eps);
            for (size_t i = 0; i < N; ++i) row[i] = row[i] * inv * m->tok_norm[i];
        }
    }

    /// The step record of T rows at p0 ..
    void steps(Rows& R, int64_t p0, int T) {
        const int64_t live = p0 + T;
        for (int t = 0; t < R.cap; ++t) R.step_h[t] = (int32_t) (p0 + std::min(t, T - 1));
        const int64_t per = (live + kChunks - 1) / kChunks;
        const int64_t chunk = std::max<int64_t>(ik::kSub, (per + ik::kSub - 1) / ik::kSub * ik::kSub);
        R.step_h[R.cap] = (int32_t) live;
        R.step_h[R.cap + 1] = (int32_t) chunk;
        R.step_h[R.cap + 2] = (int32_t) ((live + chunk - 1) / chunk);
        R.step_h[R.cap + 3] = (int32_t) (T + m->g.n_swa - 1);
    }
    /// A forward's launches: the rows and the step record from their pinned copies, every visit, the head's rows
    void launches(Rows& R, int T, int n_head, int64_t p0) {
        CK(cudaMemcpyAsync(R.x, R.emb_h, (size_t) T * (size_t) m->g.n_embd * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(R.step_d, R.step_h, (size_t) (R.cap + 4) * 4, cudaMemcpyHostToDevice, st));
        tick("embedding");
        for (int v = 0; v < (int) visits.size(); ++v) visit(v, T, R, p0);
        if (n_head > 0) head(R, T - n_head, n_head);
    }
    /// T tokens at the sequence's end through every visit; the logits of the last n_head rows.  False: no cells left.
    bool forward(const int32_t* tok, int T, int n_head) {
        const int64_t p0 = pos();
        if (p0 + T > ctx || !ensure(p0 + T)) return false;
        CK(cudaStreamSynchronize(st));   // the last forward has read the pinned rows
        if (prof) prof_t = Clock::now();
        embed(D, tok, T);
        steps(D, p0, T);
        if (!use_graphs) {
            launches(D, T, n_head, p0);
        } else {
            cudaGraphExec_t& exec = graphs[T * 16 + n_head];
            if (exec == nullptr) {
                cudaGraph_t graph = nullptr;
                CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
                launches(D, T, n_head, p0);
                CK(cudaStreamEndCapture(st, &graph));
                CK(cudaGraphInstantiate(&exec, graph, 0));
                CK(cudaGraphDestroy(graph));
                own(&Owned::graphs, exec);
            }
            CK(cudaGraphLaunch(exec, st));
            if (n_head > 0) lg_rows = &D;
        }
        n_pos += T;
        logits_held = false;
        return true;
    }
    /// T tokens of a prompt as one chunk (T <= prefill); the last one's logits with `logits`.  False: no cells left.
    bool chunk(const int32_t* tok, int T, bool logits) {
        const int64_t p0 = pos();
        if (p0 + T > ctx || !ensure(p0 + T)) return false;
        CK(cudaStreamSynchronize(st));
        if (prof) prof_t = Clock::now();
        embed(P, tok, T);
        steps(P, p0, T);
        launches(P, T, logits ? 1 : 0, p0);
        n_pos += T;
        logits_held = false;
        return true;
    }

    /// `n` prompt tokens; the last one's logits are row 0.  A chunk (or a decode forward) ends at each of `cuts`
    /// (ascending, counted in this part's tokens).  0, -1 when `progress` said stop, -2 without cells.
    int feed(const int32_t* tok, int64_t n, const std::function<bool(int64_t)>& progress,
             const std::vector<int64_t>* cuts = nullptr) {
        size_t ci = 0;
        for (int64_t at = 0; at < n;) {
            while (cuts != nullptr && ci < cuts->size() && (*cuts)[ci] <= at) ++ci;
            const int64_t left = (cuts != nullptr && ci < cuts->size() ? std::min(n, (*cuts)[ci]) : n) - at;
            int T;
            bool ok;
            if (prefill > 0 && left >= kChunkMin) {
                T = (int) std::min<int64_t>(prefill, left);
                ok = chunk(tok + at, T, at + T == n);
            } else {
                T = (int) std::min<int64_t>(feed_rows, left);
                ok = forward(tok + at, T, at + T == n ? 1 : 0);
            }
            if (!ok) return -2;
            at += T;
            if (progress) {
                CK(cudaStreamSynchronize(st));
                if (!progress(at)) return -1;
            }
        }
        return 0;
    }

    void host_logits(int row, std::vector<float>& lg) {
        const size_t V = (size_t) m->g.n_vocab;
        CK(cudaMemcpyAsync(logits_h, lg_rows->logits + (size_t) row * V, V * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        lg.assign(logits_h, logits_h + V);
    }

    // ---- the sequence's start, and the state to and from the prompt cache's buffers
    void reset() {
        CK(cudaStreamSynchronize(st));
        CK(cudaMemsetAsync(conv_all, 0, conv_bytes(), st));
        n_pos = 0;
        logits_held = false;
    }
    /// A checkpoint of the state the queued work leaves (a chunk's end; `end`: the prompt's, with the logits its read
    /// left): into the image behind that work, then to `host` (state_bytes()) on the snapshot stream.  Nothing is
    /// waited for but an image still on its way, by the stream.  A `host` that is not pinned gets the image through
    /// snap_via (snap_wait).
    bool snap_state(uint8_t* host, bool pinned, bool end) {
        const size_t V4 = (size_t) m->g.n_vocab * 4, body = ring_bytes + conv_bytes();
        if (!pinned) {
            if (snap_to != nullptr) snap_wait();   // (the one before is in its buffer first)
            snap_to = host;
            host = snap_via;
        }
        bool ok = soft(cudaStreamWaitEvent(st, ev_landed, 0)) &&
                  soft(cudaMemcpyAsync(snap, rings, ring_bytes, cudaMemcpyDeviceToDevice, st)) &&
                  soft(cudaMemcpyAsync(snap + ring_bytes, conv_all, conv_bytes(), cudaMemcpyDeviceToDevice, st));
        if (end) ok = ok && soft(cudaMemcpyAsync(snap + body, lg_rows->logits, V4, cudaMemcpyDeviceToDevice, st));
        const int32_t held[2] = {end ? 1 : 0, 0};
        std::memcpy(host + body + V4, held, 8);   // (past what the copy below writes)
        return ok && soft(cudaEventRecord(ev_snap, st)) && soft(cudaStreamWaitEvent(st_snap, ev_snap, 0)) &&
               soft(cudaMemcpyAsync(host, snap, body + (end ? V4 : 0), cudaMemcpyDeviceToHost, st_snap)) &&
               soft(cudaEventRecord(ev_landed, st_snap));
    }
    /// The images' copies have landed, a pageable buffer's contents are in it (before the cache reads, reuses or
    /// frees a buffer).
    void snap_wait() {
        if (snap == nullptr) return;
        (void) cudaStreamSynchronize(st_snap);
        (void) cudaGetLastError();
        if (snap_to != nullptr) std::memcpy(snap_to, snap_via, state_bytes());
        snap_to = nullptr;
    }
    /// The state into `host` (state_bytes(), no logits with it) or from it, now.  One loaded from a prompt's end brings
    /// its logits back (logits_held).
    bool state_copy(uint8_t* host, bool to_host) {
        const size_t V4 = (size_t) m->g.n_vocab * 4, body = ring_bytes + conv_bytes();
        if (to_host) {
            const bool ok = soft(cudaMemcpyAsync(host, rings, ring_bytes, cudaMemcpyDeviceToHost, st)) &&
                            soft(cudaMemcpyAsync(host + ring_bytes, conv_all, conv_bytes(), cudaMemcpyDeviceToHost, st));
            std::memset(host + body + V4, 0, 8);
            return soft(cudaStreamSynchronize(st)) && ok;
        }
        int32_t held = 0;
        std::memcpy(&held, host + body + V4, 4);
        bool ok = soft(cudaMemcpyAsync(rings, host, ring_bytes, cudaMemcpyHostToDevice, st)) &&
                  soft(cudaMemcpyAsync(conv_all, host + ring_bytes, conv_bytes(), cudaMemcpyHostToDevice, st));
        if (held == 1) ok = ok && soft(cudaMemcpyAsync(D.logits, host + body, V4, cudaMemcpyHostToDevice, st));
        ok = soft(cudaStreamSynchronize(st)) && ok;
        logits_held = ok && held == 1;
        if (logits_held) lg_rows = &D;
        return ok;
    }
    /// A parked conversation's bytes for `cells` positions: the full visits' K and V
    uint64_t kv_bytes(int64_t cells) const {
        return (uint64_t) m->g.n_full_visits() * 2 * (uint64_t) cells * (uint64_t) cell_bytes;
    }
    /// The full visits' first `cells` cells between the device and `host`, a copy of `held` >= cells cells (a visit's
    /// K, then its V, each `held` cells long)
    bool kv_copy(int64_t cells, int64_t held, uint8_t* host, bool to_host) {
        const size_t w = (size_t) cells * cell_bytes, mine = (size_t) held * cell_bytes;
        bool ok = true;
        for (Visit& V : visits) {
            if (V.cells.ring) continue;
            for (uint8_t* d : {V.cells.k, V.cells.v}) {
                if (w > 0)
                    ok = ok && soft(to_host ? cudaMemcpyAsync(host, d, w, cudaMemcpyDeviceToHost, st)
                                            : cudaMemcpyAsync(d, host, w, cudaMemcpyHostToDevice, st));
                host += mine;
            }
        }
        return soft(cudaStreamSynchronize(st)) && ok;
    }

    void prof_print(const char* what) {
        if (!prof || prof_ms.empty()) return;
        double total = 0;
        for (const auto& [k, v] : prof_ms) total += v;
        std::fprintf(stderr, "lamina: %s, %.1f ms in the steps:\n", what, total);
        for (const auto& [k, v] : prof_ms) std::fprintf(stderr, "lamina:   %-24s %9.1f ms %5.1f%%\n", k.c_str(), v, 100.0 * v / total);
        prof_ms.clear();
    }
    std::string vram_free() const {
        size_t fr = 0, tot = 0;
        if (cudaMemGetInfo(&fr, &tot) != cudaSuccess) return "?";
        return std::to_string(fr >> 20);
    }
};

int argmax(const std::vector<float>& lg) {
    int best = -1;
    float bv = -INFINITY;
    for (size_t i = 0; i < lg.size(); ++i)
        if (lg[i] > bv) {
            bv = lg[i];
            best = (int) i;
        }
    return best;
}

// ------------------------------------------------------------------------------------------------ serve
/// --serve's prompt cache (lamina/core/prompt_cache.hpp, the other paths' options): a request continues from the
/// longest start of its prompt the engine holds - the conversation's end, one of its checkpoints, or a conversation
/// another request replaced, parked in RAM with the full visits' cells.
struct CacheOpts {
    int chain = 16;                  // --prompt-cache: checkpoints a conversation keeps (0: off)
    int64_t every = 16384;           // --prompt-cache-every: a long prompt's checkpoints, in cells
    int64_t park_mib = 16384;        // --conversation-cache-mib: the parked conversations' RAM
    int64_t park_slots = 0;          // --conversation-cache-slots: their number at most (0: no limit)
    int64_t min_tokens = 1024;       // --conversation-cache-min-tokens: shorter ones are not parked
    int64_t ram_reserve_mib = 6144;  // --ram-reserve-mib: they take no RAM below this much available to the system
};

/// The engine's state behind the cache's policy
struct Cache final : lamina::core::SequenceCache {
    Engine& E;
    Cache(Engine& e, const CacheOpts& o)
        : SequenceCache(o.chain, e.state_bytes(), (uint64_t) std::max<int64_t>(0, o.park_mib) << 20,
                        (size_t) std::max<int64_t>(0, o.park_slots), o.min_tokens),
          E(e) {}
    ~Cache() override { E.snap_wait(); }   // (no copy still on its way to a buffer the cache frees)

protected:
    bool state_save(int64_t, void* host, bool filled) override {
        if (filled) return true;   // the engine's image is on its way (Engine::snap_state)
        E.snap_wait();             // (the buffer may be one an image's copy is still landing in)
        return E.state_copy(static_cast<uint8_t*>(host), true);
    }
    bool state_load(int64_t pos, const void* host) override {
        E.snap_wait();
        if (!E.state_copy(const_cast<uint8_t*>(static_cast<const uint8_t*>(host)), false)) return false;
        E.n_pos = pos;
        return true;
    }
    void state_zero() override { E.reset(); }
    uint64_t kv_bytes(int64_t cells) const override { return E.kv_bytes(cells); }
    bool kv_copy(int64_t cells, int64_t held, uint8_t* host, bool to_host) override {
        return E.kv_copy(cells, held, host, to_host);
    }
};

/// lamina-server's engine protocol, one request at a time: GEN <max_new> [key=value ...] <ids> -> RESUME (the tokens the
/// state already held), PP lines while the prompt is read, a T line a token, CACHE (the cache's state), DONE; STOP ends
/// a request at its next step, QUIT the loop.
struct Serve {
    Engine& E;
    int32_t eos = -1;
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
    std::atomic<bool> stop{false};
    CacheOpts copt;
    std::unique_ptr<Cache> cache;      // null: off
    std::vector<int32_t> live;         // the tokens the state holds
    std::vector<int32_t> prev_prompt;  // the last request's prompt: where the next one leaves it
    bool poisoned = false;             // non-finite logits: whatever the cache holds may be bad

    Serve(Engine& e, int32_t eos_id, const CacheOpts& co) : E(e), eos(eos_id), copt(co) {
        if (E.snap == nullptr) {
            std::fprintf(stderr, "lamina serve: prompt cache: off (the held conversation continues when a request extends "
                         "it)\n");
            return;
        }
        cache = std::make_unique<Cache>(E, copt);
        cache->whole_prompt = true;   // (a prompt's end is a checkpoint with its logits)
        cache->keep_ends = true;      // (and a turn's end: it stays when the next prompt's comes)
        cache->ensure_kv = [this](int64_t cells) { return E.ensure(cells); };
        uint64_t ram_now = 0;
        const bool ram_known = copt.ram_reserve_mib > 0 && lamina::platform::available_ram(ram_now);
        if (ram_known)
            cache->ram_spare = [reserve = copt.ram_reserve_mib << 20]() -> int64_t {
                uint64_t avail = 0;
                if (!lamina::platform::available_ram(avail)) return lamina::core::ConversationCache::kNoRamLimit;
                return (int64_t) avail - reserve;
            };
        std::fprintf(stderr, "lamina serve: prompt cache on: up to %d checkpoints of %.1f MiB per conversation, %lld MiB "
                     "for the conversations requests replace (%.1f KiB a token)%s%s\n", copt.chain,
                     (double) cache->ckpt_bytes() / 1048576.0, (long long) copt.park_mib,
                     (double) E.kv_bytes(1) / 1024.0,
                     copt.park_slots > 0 ? (", at most " + std::to_string(copt.park_slots) + " of them").c_str() : "",
                     ram_known ? (", above a RAM reserve of " + std::to_string(copt.ram_reserve_mib) + " MiB (" +
                                  std::to_string(ram_now >> 20) + " MiB available now)").c_str() : "");
    }

    /// The CACHE line (the server's Conversation cache card): at READY, then after each request before its DONE.
    void print_cache() {
        if (!cache) return;
        const auto cs = cache->state();
        serve_print("CACHE live=%lld checkpoints=%zu parked=%zu parked_bytes=%llu parked_budget=%llu slots=%zu "
                    "evictions=%zu held=%zu sessions=1\n", (long long) cs.live_cells, cs.checkpoints, cs.parked,
                    (unsigned long long) cs.parked_bytes, (unsigned long long) cs.parked_budget, cs.slots, cs.evictions,
                    cs.held);
    }

    /// Where a prompt's read takes a checkpoint besides a long prompt's periodic ones and its end (which stays in the
    /// chain: a prompt ends where a turn does, so a request that edits the next message continues from there),
    /// ascending, each in (reuse, n) with two cells or more to its neighbours and the ends: where this prompt leaves
    /// the one before (agents send one long start again with another tail: the next such request continues from
    /// there), and a few cells before the end of a last message that grows between requests.
    std::vector<int64_t> checkpoint_marks(const std::vector<int32_t>& ids, int64_t reuse) {
        const int64_t n = (int64_t) ids.size();
        int64_t fork = 0, msg_end = 0;
        const int64_t lim = std::min<int64_t>((int64_t) prev_prompt.size(), n - 1);
        int64_t d = 0;
        while (d < lim && prev_prompt[(size_t) d] == ids[(size_t) d]) ++d;
        if (d > reuse + 256 && d < n - 1) fork = d;
        for (int64_t i = n - 1; i > reuse; --i)
            if (ids[(size_t) i] == kEndMessage) {
                int64_t start = i;   // the message's first token
                while (start > 0 && (ids[(size_t) start] < kMsgFirst || ids[(size_t) start] > kMsgLast)) --start;
                if (d > start + 3 && i - 4 > reuse + 256 && i - 4 != fork) msg_end = i - 4;
                break;
            }
        prev_prompt = ids;
        std::vector<int64_t> want{fork, msg_end};
        std::sort(want.begin(), want.end());
        std::vector<int64_t> marks;
        for (const int64_t p : want)
            if (p >= reuse + 2 && p <= n - 2 && (marks.empty() || p >= marks.back() + 2)) marks.push_back(p);
        return marks;
    }

    void request(const std::string& line) {
        const Geometry& g = E.m->g;
        const char* p = line.c_str() + 4;
        char* e = nullptr;
        const long long max_new = std::strtoll(p, &e, 10);
        p = e;
        lamina::program::Sampling smp;
        for (;;) {
            while (*p == ' ') ++p;
            const char* st0 = p;
            while (*p && *p != ' ') ++p;
            const std::string tk(st0, (size_t) (p - st0));
            const size_t eq = tk.find('=');
            if (tk.empty() || eq == std::string::npos) { p = st0; break; }
            smp.set(tk.substr(0, eq), tk.c_str() + eq + 1);   // keys of other paths are left
        }
        std::vector<int32_t> ids = lamina::program::parse_ids(p);
        if (max_new < 1 || ids.empty()) {
            serve_print("ERR bad request: max_new or the ids\n");
            return;
        }
        for (const int32_t t : ids)
            if (t < 0 || t >= g.n_vocab) { serve_print("ERR a token id is outside the vocabulary\n"); return; }
        if ((int64_t) ids.size() + max_new > E.ctx) {
            serve_print("ERR prompt (%zu tokens) + max_new (%lld) exceeds the context (%lld)\n", ids.size(), max_new,
                        (long long) E.ctx);
            return;
        }
        stop.store(false);
        // ---- what the state continues from: with the cache the longest start of the prompt it holds (the
        // conversation's end, a checkpoint of it, a conversation parked in RAM), the engine's state set there; without
        // it the held sequence where the prompt extends it (the windows' state cannot step back)
        std::vector<float> lg;
        size_t reuse = 0;
        double begin_ms = 0.0;   // the cache's own work: a conversation parked or taken back (the prompt's time has it)
        if (cache) {
            const auto tb = Clock::now();
            E.snap_wait();   // (the last request's checkpoints are in their buffers before the cache moves or frees any)
            if (poisoned) {
                cache->clear();
                poisoned = false;
            }
            const int64_t held = cache->begin(std::vector<int64_t>(ids.begin(), ids.end()), {});
            reuse = (size_t) std::max<int64_t>(0, held);
            if (reuse == ids.size() && !E.logits_held) {   // (a whole prompt without its logits: read again)
                cache->set(0, static_cast<const int32_t*>(nullptr), 0);
                E.reset();
                reuse = 0;
            }
            begin_ms = ms_since(tb);
        } else {
            size_t common = 0;
            while (common < live.size() && common < ids.size() && live[common] == ids[common]) ++common;
            if (common > 0 && common == live.size() && common < ids.size()) reuse = common;
            else E.reset();
        }
        live.assign(ids.begin(), ids.begin() + (long) reuse);
        if (reuse > 0) serve_print("RESUME %zu\n", reuse);
        const auto t0 = Clock::now();
        const int64_t fresh = (int64_t) (ids.size() - reuse);
        // The cache's checkpoints are taken as the read passes them: at its marks (a chunk ends there), and in a long
        // prompt at a chunk's end once --prompt-cache-every cells came since the last; the prompt's end is one with the
        // logits its read left.  Each is an image on the device, sent to the cache's buffer behind the run.
        std::vector<int64_t> marks, cuts;
        if (cache) marks = checkpoint_marks(ids, (int64_t) reuse);
        for (const int64_t at : marks) cuts.push_back(at - (int64_t) reuse);
        const int64_t rows = E.prefill > 0 ? E.prefill : kMaxT;
        const int64_t every = std::max<int64_t>(1, copt.every / rows) * rows;
        int64_t noted = (int64_t) reuse, saved = (int64_t) reuse;   // the cache has the tokens up to `noted`
        auto note = [&](int64_t upto) {
            if (!cache || upto <= noted) return;
            cache->set(noted, ids.data() + noted, upto - noted);
            noted = upto;
        };
        auto checkpoint = [&](int64_t here, bool end) {   // of the state the queued work leaves
            note(here);
            saved = here;
            bool pageable = false;
            void* const h = cache->prepare(&pageable, end);
            if (h == nullptr) {
                if (!end) cache->checkpoint(here);   // (copied now; an end without a buffer keeps no logits: none)
            } else if (E.snap_state(static_cast<uint8_t*>(h), !pageable, end)) {
                cache->checkpoint(here, h, end);
            }
        };
        int64_t done = 0;
        auto last_pp = Clock::now();
        const int fed = E.feed(ids.data() + reuse, fresh, [&](int64_t d) {
            done = d;
            if (cache && done < fresh) {
                const int64_t here = (int64_t) reuse + done;
                const auto next = std::lower_bound(marks.begin(), marks.end(), here);
                const bool mark = next != marks.end() && *next == here;
                if (mark || (here - saved >= every && (next != marks.end() ? *next : (int64_t) ids.size()) - here >= rows))
                    checkpoint(here, false);
            }
            if (done == fresh || ms_since(last_pp) > 250) {
                serve_print("PP %lld %lld %.0f %.1f\n", (long long) done, (long long) fresh, ms_since(t0),
                            (double) done / std::max(1e-9, ms_since(t0) / 1000.0));
                last_pp = Clock::now();
            }
            return !stop.load();
        }, &cuts);
        E.prof_print("the prompt");
        note((int64_t) reuse + done);
        live.insert(live.end(), ids.begin() + (long) reuse, ids.begin() + (long) reuse + (long) done);
        const double prompt_ms = begin_ms + ms_since(t0);
        if (fed < 0) {
            if (fed == -2) std::fprintf(stderr, "lamina: no VRAM for the prompt's cells past %lld tokens\n", (long long) E.pos());
            if (cache) {
                E.snap_wait();
                cache->shed();
            }
            print_cache();
            serve_print("DONE 0 %zu %.1f 0.0 %s 0 0 %zu 0 0 0 0 0.0 %lld 0\n", ids.size(), prompt_ms,
                        fed == -1 ? "cancel" : "length", reuse, (long long) done);
            return;
        }
        E.host_logits(0, lg);   // the read's, or the ones a prompt's end brought back
        // the prompt's end, before the first decode step moves the state (not after a read of one token - a stopped
        // answer going on: the conversation keeps the end of the prompt it was asked with)
        if (cache && fresh > 1) checkpoint((int64_t) ids.size(), true);
        std::mt19937_64 rng(smp.seed);
        std::vector<int32_t> hist(ids.begin(), ids.end());   // the penalties' history: the prompt's text
        const auto d0 = Clock::now();
        long long produced = 0;
        const char* finish = "length";
        while (true) {
            const int nx = lamina::program::sample(lg, smp, hist, rng);
            if (nx < 0) { finish = "nonfinite"; break; }
            serve_print("T %d\n", nx);
            ++produced;
            hist.push_back(nx);
            if (nx == eos) { finish = "stop"; break; }
            if (stop.load()) { finish = "cancel"; break; }
            if (produced >= max_new) break;
            const int32_t one = nx;
            if (!E.forward(&one, 1, 1)) break;   // the context's end, or no VRAM for more cells
            live.push_back(nx);
            if (cache) cache->append(0, nx);
            E.host_logits(0, lg);
        }
        if (std::strcmp(finish, "nonfinite") == 0) {   // the state holds the non-finite rows: not continued
            live.clear();
            poisoned = cache != nullptr;   // (nor is anything the cache holds: forgotten at the next request)
            if (cache) cache->set(0, static_cast<const int32_t*>(nullptr), 0);
            std::fprintf(stderr, "lamina serve: non-finite logits: the request ends, and the next one reads its prompt from "
                         "the start\n");
        }
        const double decode_ms = ms_since(d0);
        E.prof_print("generation");
        if (cache) {
            cache->touch(0);
            E.snap_wait();   // (shed may free a buffer)
            cache->shed();
        }
        print_cache();
        serve_print("DONE %lld %zu %.1f %.1f %s 0 0 %zu 0 0 0 0 0.0 %lld 0\n", produced, ids.size(), prompt_ms, decode_ms,
                    finish, reuse, (long long) fresh);
        std::fprintf(stderr, "lamina serve: prompt %zu tokens = %zu reused + %lld read in %.1f ms (%.1f tok/s), %lld "
                     "generated in %.1f ms (%.2f tok/s), %s; MiB of VRAM free: %s\n", ids.size(), reuse, (long long) fresh,
                     prompt_ms, prompt_ms > 0 ? 1000.0 * (double) fresh / prompt_ms : 0.0, produced, decode_ms,
                     decode_ms > 0 ? 1000.0 * (double) produced / decode_ms : 0.0, finish, E.vram_free().c_str());
        if (cache) {
            std::fprintf(stderr, "lamina serve: prompt cache: %s\n", cache->summary().c_str());
            cache->spare();
        }
    }

    int run() {
        serve_print("READY %lld stop\n", (long long) E.ctx);
        print_cache();
        const lamina::program::ServeIo& io = lamina::program::serve_io();
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
        // Ctrl+C in the console ends the engine as QUIT does, before any request queued (the running one stops at its
        // next token).  In-process the host handles it.
        std::optional<lamina::platform::ConsoleQuit> console_quit;
        if (!io.in_process)
            console_quit.emplace([&] {
                stop.store(true);
                std::lock_guard<std::mutex> lk(mu);
                lines.push_front("QUIT");
                cv.notify_one();
            });
        for (;;) {
            std::string line;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return eof || !lines.empty(); });
                if (lines.empty()) return 0;
                line = lines.front();
                lines.pop_front();
            }
            if (line.empty()) continue;
            if (line.rfind("GEN ", 0) == 0) {
                request(line);
            } else if (line.rfind("LEND ", 0) == 0) {
                serve_print("ERR no loan: Inkling 2's engine lends no VRAM\n");
            } else if (line == "UNLEND") {
                serve_print("UNLENT\n");
            } else if (line == "QUIT") {
                std::lock_guard<std::mutex> lk(mu);
                eof = true;
                lines.clear();
            } else {
                serve_print("ERR expected: GEN <max_new> [key=value ...] <id,id,...>\n");
            }
        }
    }
};

int run_engine(int argc, char** argv, bool check_only) {
    std::string model, tokens, force, dump_full, kv = "fp16";
    int64_t max_ctx = 0;
    int n_gen = 64, threads = 8;
    bool serve = false, file_cache = true, kv_grow = true, graphs = true;
    int prefill = 1024;
    std::optional<uint64_t> reserve;
    CacheOpts copt;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "lamina: %s needs a value\n", a.c_str());
                throw Fatal{2};
            }
            return argv[++i];
        };
        if (a == "--native" || a == "--model") model = next();
        else if (a == "--no-graph") graphs = false;   // every launch issued each forward
        else if (a == "--prefill") prefill = std::max(0, std::atoi(next()));   // rows a prompt chunk (0: the decode's forwards)
        else if (a == "--max-context") max_ctx = std::atoll(next());
        else if (a == "--tokens") tokens = next();
        else if (a == "--tokens-file") {
            const char* f = next();
            FILE* fp = std::fopen(f, "rb");
            if (!fp) { std::fprintf(stderr, "lamina: cannot open %s\n", f); return 2; }
            char buf[1 << 16];
            size_t n;
            while ((n = std::fread(buf, 1, sizeof buf, fp)) > 0) tokens.append(buf, n);
            std::fclose(fp);
        } else if (a == "--n") n_gen = std::atoi(next());
        else if (a == "--force") force = next();
        else if (a == "--dump-full-logits") dump_full = next();
        else if (a == "--serve") serve = true;
        else if (a == "--kv") kv = next();   // the cells: fp16 or q8_0
        else if (a == "--kv-grow") kv_grow = true;
        else if (a == "--no-kv-grow") kv_grow = false;   // the full visits' cells allocated whole at the start
        else if (a == "--file-cache") file_cache = true;
        else if (a == "--no-file-cache") file_cache = false;   // the model file read past the OS file cache
        else if (a == "--load-threads") threads = std::max(1, std::atoi(next()));
        else if (a == "--vram-reserve-mib") reserve = (uint64_t) std::max(0, std::atoi(next())) << 20;
        else if (a == "--prompt-cache") copt.chain = std::max(0, std::atoi(next()));
        else if (a == "--prompt-cache-every") copt.every = std::max<int64_t>(1, std::atoll(next()));
        else if (a == "--conversation-cache-mib") copt.park_mib = std::max<int64_t>(0, std::atoll(next()));
        else if (a == "--conversation-cache-slots") copt.park_slots = std::max<int64_t>(0, std::atoll(next()));
        else if (a == "--conversation-cache-min-tokens") copt.min_tokens = std::max<int64_t>(0, std::atoll(next()));
        else if (a == "--ram-reserve-mib") copt.ram_reserve_mib = std::max<int64_t>(0, std::atoll(next()));
        else {
            std::fprintf(stderr, "lamina: Inkling 2's path does not take %s\n", a.c_str());
            return 2;
        }
    }
    if (model.empty()) { std::fprintf(stderr, "lamina: --native MODEL is required\n"); return 2; }
    if (kv != "fp16" && kv != "q8_0") { std::fprintf(stderr, "lamina: --kv takes fp16 or q8_0 on this path\n"); return 2; }
    if (max_ctx < 0) { std::fprintf(stderr, "lamina: --max-context takes a number of tokens\n"); return 2; }
    if (check_only) return 0;
    if (!reserve) {   // a card the desktop draws on keeps more free
        int dev = 0;
        CK(cudaGetDevice(&dev));
        bool known = false, desktop = false;
        for (const lamina::core::GpuInfo& gi : lamina::core::list_gpus(true))
            if (gi.ordinal == dev) desktop = lamina::core::gpu_in_desktop_use(gi.bus_id, known);
        reserve = (uint64_t) (known && !desktop ? 700 : 1536) << 20;
    }
    Model m;
    std::string err;
    if (!load(model, m, threads, file_cache, (size_t) *reserve, err)) {
        std::fprintf(stderr, "lamina: %s\n", err.c_str());
        return 1;
    }
    Engine E;
    E.use_graphs = graphs;
    E.cache_on = serve && copt.chain > 0;
    E.prefill = prefill;
    E.init(m, max_ctx, kv == "q8_0", kv_grow, (size_t) *reserve);
    if (serve) {
        Serve sv(E, m.eos, copt);
        return sv.run();
    }
    std::vector<int32_t> prompt = lamina::program::parse_ids(tokens.c_str());
    const std::vector<int32_t> forced = lamina::program::parse_ids(force.c_str());
    if (prompt.empty()) { std::fprintf(stderr, "lamina: --tokens is required\n"); return 2; }
    if (!forced.empty()) n_gen = (int) forced.size() + 1;
    for (const std::vector<int32_t>* v : {(const std::vector<int32_t>*) &prompt, &forced})
        for (const int32_t t : *v)
            if (t < 0 || t >= m.g.n_vocab) {
                std::fprintf(stderr, "lamina: the token %d is outside the vocabulary (%lld)\n", (int) t, (long long) m.g.n_vocab);
                return 2;
            }
    if ((int64_t) prompt.size() + n_gen > E.ctx) {
        std::fprintf(stderr, "lamina: the prompt (%zu tokens) and %d generated exceed the context (%lld)\n", prompt.size(),
                     n_gen, (long long) E.ctx);
        return 2;
    }
    FILE* ff = dump_full.empty() ? nullptr : std::fopen(dump_full.c_str(), "wb");
    const auto t0 = Clock::now();
    if (E.feed(prompt.data(), (int64_t) prompt.size(), nullptr) != 0) {
        std::fprintf(stderr, "lamina: no VRAM for the prompt's cells\n");
        return 1;
    }
    std::vector<float> lg;
    E.host_logits(0, lg);
    const double pp_ms = ms_since(t0);
    E.prof_print("the prompt");
    const auto t1 = Clock::now();
    std::vector<int32_t> gen;
    for (int i = 0; i < n_gen; ++i) {
        const int a = argmax(lg);
        gen.push_back(a);
        if (ff) std::fwrite(lg.data(), 4, lg.size(), ff);
        if (i + 1 == n_gen) break;
        const int32_t nx = i < (int) forced.size() ? forced[(size_t) i] : a;
        if (nx < 0 || !E.forward(&nx, 1, 1)) break;
        E.host_logits(0, lg);
    }
    const double tg_ms = ms_since(t1);
    E.prof_print("generation");
    if (ff) std::fclose(ff);
    serve_print("gen:");
    for (int t : gen) serve_print(" %d", t);
    serve_print("\n");
    std::fprintf(stderr, "lamina: prompt %zu tokens in %.1f ms (%.1f tok/s), %zu generated in %.1f ms (%.2f tok/s); MiB of "
                 "VRAM free: %s\n", prompt.size(), pp_ms, prompt.size() / (pp_ms / 1000.0), gen.size(), tg_ms,
                 gen.size() / (tg_ms / 1000.0), E.vram_free().c_str());
    return 0;
}

}  // namespace

int lamina::inkling2::run(int argc, char** argv) {
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

int lamina::inkling2::check_args(int argc, char** argv) {
    try {
        return run_engine(argc, argv, true);
    } catch (const Fatal& f) {
        return f.code;
    }
}
