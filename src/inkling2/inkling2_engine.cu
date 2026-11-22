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
#include "inkling2_draft.cuh"
#include "inkling2_kernels.cuh"

#include "lamina/artifact/gguf_reader.hpp"
#include "lamina/core/cuda_host.hpp"
#include "lamina/core/expert_cache.hpp"
#include "lamina/core/gpu_select.hpp"
#include "lamina/core/prompt_cache.hpp"
#include "lamina/core/vmm.hpp"
#include "lamina/inkling2/engine.hpp"
#include "lamina/inkling2/geometry.hpp"
#include "lamina/kernels/bf16_gemv.hpp"
#include "lamina/kernels/cpu/native_expert.hpp"
#include "lamina/kernels/cpu/pool.hpp"
#include "lamina/kernels/iq_kernels.hpp"
#include "lamina/kernels/native_mmvq.hpp"
#include "lamina/kernels/nvfp4_gemm.hpp"
#include "lamina/kernels/sampler.hpp"
#include "lamina/platform/console_quit.hpp"
#include "lamina/platform/cpu_arch.hpp"
#include "lamina/platform/memory.hpp"
#include "lamina/platform/slice_read.hpp"
#include "lamina/prefill/moe_mmq.hpp"
#include "lamina/program/host_sampling.hpp"
#include "lamina/program/serve_io.hpp"
#include "lamina/spec/suffix_drafter.hpp"

#include "ggml.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
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
namespace C = lamina::kernels::cpu;
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
template <class T> T* halloc(size_t n) {   // pinned, zeroed (pages given back and taken again are not)
    void* p = nullptr;
    CK(cudaHostAlloc(&p, std::max<size_t>(n * sizeof(T), 256), cudaHostAllocPortable));
    std::memset(p, 0, std::max<size_t>(n * sizeof(T), 256));
    own(&Owned::pinned, p);
    return (T*) p;
}

constexpr int64_t kGrow = 4096;           // cells a full visit's storage grows by
constexpr int64_t kContextFloor = 32768;  // a context not set is what fits, not under this
// the vocabulary's message tokens: where a prompt's read places the cache's checkpoints
constexpr int32_t kMsgFirst = 200000, kMsgLast = 200003;   // <|message_user|> .. <|message_tool|>
constexpr int32_t kEndMessage = 200010;                    // <|end_message|>
constexpr int kRingSub = ik::kTcCells;    // a window visit's ring: a multiple of an attention tile's cells
constexpr int32_t kFar = 1 << 30;         // a row's position as a card sees it: every cell there is before it

// ------------------------------------------------------------------------------------------------ the model
constexpr size_t kWeightPad = 4096;       // zero bytes after each weight: a kernel may read past a short last row

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
    uint8_t* slots = nullptr;      // device: the slots, an expert's blob each
    int n_slots = 0;               // those filled at the load: every expert's, or (Model::in_ram) the ones in VRAM alone
    int32_t* slot_of = nullptr;    // device [n_expert]: its slot or -1
    // The layer's experts in the ranking's order, and an expert's place in it: where the card holds them all, each
    // expert's own number, its slot.  With experts in RAM the first n_slots of the order are in VRAM alone, in the
    // slots of their places; the others' blobs are in RAM (blob_h, in the order's order) and the engine says which
    // of them the card has and where (Engine::slot_at).  fmt: the layer as the CPU pool computes it; slot_vm: the
    // slots' addresses, mapped as far as slots are in use.
    std::vector<int32_t> order, slot_at;
    std::vector<const uint8_t*> blob_h;
    C::NativeFmt fmt;
    std::unique_ptr<lamina::core::VmmRange> slot_vm;
};
/// The chunks a layer's first n slots take of its range (with the bytes a kernel may read past the last)
int64_t slot_chunks(const Layer& L, int n) {
    const uint64_t G = lamina::core::vmm_granularity();
    return n > 0 ? (int64_t) (((uint64_t) n * L.stride + kWeightPad + G - 1) / G) : 0;
}

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
    int32_t image_pad = -1;   // the token the chat template writes for a picture's rows (after <|content_image|>), or -1
    int64_t trained_ctx = 0;
    double dense_gib = 0, expert_gib = 0, load_s = 0;
    // A card smaller than the model and the context asked of it: the experts are in RAM (`arena`, but the ranking's
    // first `n_fixed`, which RAM had no room for: VRAM alone holds those), and the card has the ones used most, as
    // many as its VRAM holds beside the cells - fewer as the context grows (Engine: experts in RAM).  The CPU
    // computes the others where a token routes them.
    bool in_ram = false;
    std::vector<std::pair<int32_t, int32_t>> ranked;   // every (layer, expert), the most used first
    int64_t n_fixed = 0;
    uint8_t* arena = nullptr;
    std::unique_ptr<uint8_t[]> arena_heap;   // (where the driver did not pin it)
    uint64_t arena_bytes = 0, fixed_bytes = 0;
    uint64_t expert_cap = 0;                 // --expert-cache: the experts' VRAM at most (0: no limit)
    std::string ranking;                     // what the ranking is, for the log
};

/// What a start asks of the loader beside the file
struct LoadOpts {
    int threads = 8;
    bool file_cache = true;
    size_t reserve = 0;
    uint64_t expert_vram = 0;    // --expert-cache
    std::string profile;         // --expert-profile: the ranking (empty: the one built into the programs)
    int64_t max_ctx = 0;         // --max-context
    bool q8 = false, cache = false, cell_cards = false;
    int prefill = 1024;
    uint64_t ram_reserve = 0;    // the experts take no RAM below this much available to the system
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
    if (g.n_head > ik::kRingHeads || g.n_head * g.n_rel > ik::kRingRs) return no("that many heads");
    if (g.n_expert > ik::kExpertMax || g.top_k * kMaxT > ik::kTB) return no("that many experts a layer or a token");
    if (g.n_pass > 1 && (g.gate_rank > ik::kRankMax || ik::kTB % g.gate_rank != 0 || g.gate_rank % 16 != 0))
        return no("a loop gate of that rank");
    if ((g.n_layers + g.res_block - 1) / g.res_block > ik::kBlk) return no("more than 4 blocks of layers");
    if (g.n_embd % 256 != 0 || g.q_width() % 256 != 0) return no("rows that are no multiple of 256 values");
    if (g.n_embd > 4 * ik::kWide) return no("rows of more than 4,096 values");
    if (g.rel_extent % ik::kTcCells != 0 || g.n_swa % ik::kTcCells != 0)
        return no("a window or a table of distances that is no multiple of 32 cells");
    return {};
}

bool load(const std::string& path, Model& m, const LoadOpts& o, std::string& err) {
    const int threads = o.threads;
    const bool file_cache = o.file_cache;
    const size_t reserve = o.reserve;
    const auto t0 = Clock::now();
    const lamina::GgufModel gm = lamina::GgufModel::open(path);
    if (std::string e = lamina::inkling2::read_geometry(gm.meta(), m.g); !e.empty()) { err = e; return false; }
    if (std::string e = lamina::inkling2::check_layout(gm, m.g); !e.empty()) { err = e; return false; }
    if (std::string e = kernel_limits(m.g); !e.empty()) { err = e; return false; }
    const Geometry& g = m.g;
    if (const lamina::MetaValue* v = gm.meta().get("tokenizer.ggml.eos_token_id"); v && v->is_num()) m.eos = (int32_t) v->u;
    if (const lamina::MetaValue* v = gm.meta().get(g.arch + ".context_length"); v && v->is_num()) m.trained_ctx = (int64_t) v->u;
    if (const lamina::MetaValue* v = gm.meta().get("tokenizer.chat_template"); v != nullptr) {
        // (the template's picture block: <|content_image|>, then the token that stands for the rows)
        const std::string open = "<|content_image|><|";
        if (const size_t at = v->s.find(open); at != std::string::npos) {
            const size_t from = at + open.size() - 2, to = v->s.find("|>", from);
            if (to != std::string::npos && to - from < 64)
                m.image_pad = (int32_t) gm.meta().find_string("tokenizer.ggml.tokens", v->s.substr(from, to + 2 - from));
        }
    }
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
        y.gl.x_il = K::native_expert_il(y.gl);   // (a decode forward's image of the MoE's input follows it)
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
    // The experts' place.  A card that holds the dense weights, every expert and what a context takes beside them -
    // its cells (the floor's; those of the context asked for where this card alone holds it), the window rings, the
    // rows of the decode and of a chunk at its fewest - has them all, each in its own slot.  Else they are in RAM
    // and the card has the ranking's first.
    std::vector<int> fixed((size_t) g.n_layers, 0);   // (a layer's experts that VRAM alone holds)
    {
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        const double gib = (double) (1ull << 30);
        const uint64_t cell = o.q8 ? (uint64_t) g.n_head_kv * (uint64_t) (g.head_dim / 32) * 34 : (uint64_t) g.kv_width() * 2;
        const uint64_t per_pos = (uint64_t) g.n_full_visits() * 2 * cell;
        const int64_t trained = m.trained_ctx > 0 ? m.trained_ctx : 1048576;
        const uint64_t ring = (uint64_t) (g.n_visits() - g.n_full_visits()) * 2 * cell *
                              (uint64_t) ((g.n_swa + kMaxT + kRingSub - 1) / kRingSub * kRingSub);
        const uint64_t beside = ring * (o.cache ? 2 : 1) + (o.prefill > 0 ? 230ull << 20 : 0) + (96ull << 20);
        const int64_t least = o.cell_cards ? 4 * kGrow : o.max_ctx > 0 ? std::min(o.max_ctx, kContextFloor) : kContextFloor;
        const int64_t asked = o.max_ctx > 0 && !o.cell_cards ? std::min(o.max_ctx, trained) : least;
        m.expert_cap = o.expert_vram;
        const bool all = dense + experts + (uint64_t) asked * per_pos + beside + reserve <= fr &&
                         (o.expert_vram == 0 || o.expert_vram >= experts);
        if (!all) {
            if (!lamina::core::vmm_available()) {
                char b[512];
                std::snprintf(b, sizeof b, "the model's %.2f GiB of dense weights and %.2f GiB of experts, with the cells of "
                              "%lld tokens and the %.1f GiB left free, pass this GPU's %.2f GiB of free VRAM, and experts in "
                              "RAM need CUDA's virtual memory, which this driver does not have", m.dense_gib, m.expert_gib,
                              (long long) asked, (double) reserve / gib, (double) fr / gib);
                err = b;
                return false;
            }
            // the ranking: the profile's, the pairs it leaves out after it in their order; a layer the CPU has no
            // kernels for first of all (VRAM alone holds its experts)
            int64_t slots = 0;
            std::string perr;
            m.ranking = o.profile.empty() ? "the built-in ranking" : o.profile;
            if (!lamina::core::read_expert_profile(o.profile.empty() ? lamina::core::kBuiltinExpertProfile : o.profile, g.n_layers,
                                                   E, m.ranked, slots, perr)) {
                if (!o.profile.empty()) {
                    err = perr;
                    return false;
                }
                std::fprintf(stderr, "lamina: no ranking of this model's experts is built in (%s): they are placed in "
                             "their layers' order (--expert-profile FILE names a ranking)\n", perr.c_str());
                m.ranked.clear();
                m.ranking = "no ranking";
            }
            std::vector<char> seen((size_t) (g.n_layers * E), 0), cpu_ok((size_t) g.n_layers, 1);
            {
                std::vector<std::pair<int32_t, int32_t>> r;
                for (const auto& [l, e] : m.ranked)
                    if (l >= 0 && l < g.n_layers && e >= 0 && e < E && !seen[(size_t) (l * E + e)]) {
                        seen[(size_t) (l * E + e)] = 1;
                        r.emplace_back(l, e);
                    }
                for (int64_t e = 0; e < E; ++e)
                    for (int64_t l = 0; l < g.n_layers; ++l)
                        if (!seen[(size_t) (l * E + e)]) r.emplace_back((int32_t) l, (int32_t) e);
                int64_t n_must = 0;
                for (int64_t l = 0; l < g.n_layers; ++l) {
                    Layer& y = m.layers[(size_t) l];
                    std::string why;
                    cpu_ok[(size_t) l] = C::native_fmt(y.gl.gu_type, y.gl.d_type, N, g.n_ff, y.fmt, why) ? 1 : 0;
                    if (!cpu_ok[(size_t) l]) {
                        std::fprintf(stderr, "lamina: layer %lld's experts stay in VRAM (%s)\n", (long long) l, why.c_str());
                        n_must += E;
                    }
                    y.fmt.act = 1;
                    y.fmt.act_beta = g.situ_beta;
                    y.fmt.act_lbeta = g.situ_lbeta;
                    y.fmt.scale_off = y.gl.scale_off;
                }
                std::stable_partition(r.begin(), r.end(), [&](const auto& le) { return !cpu_ok[(size_t) le.first]; });
                m.ranked = std::move(r);
                m.n_fixed = n_must;
            }
            // RAM: the experts from the ranking's end, as many as it holds above what it leaves free
            uint64_t avail = 0;
            const bool known = lamina::platform::available_ram(avail);
            const uint64_t ram = !known ? UINT64_MAX : avail > o.ram_reserve ? avail - o.ram_reserve : 0;
            int64_t a0 = (int64_t) m.ranked.size();
            while (a0 > m.n_fixed) {
                const uint64_t b = m.layers[(size_t) m.ranked[(size_t) (a0 - 1)].first].stride;
                if (m.arena_bytes + b > ram) break;
                m.arena_bytes += b;
                --a0;
            }
            m.n_fixed = a0;
            for (int64_t r = 0; r < a0; ++r) ++fixed[(size_t) m.ranked[(size_t) r].first];
            const uint64_t G = lamina::core::vmm_granularity();
            for (int64_t l = 0; l < g.n_layers; ++l)
                m.fixed_bytes += (uint64_t) slot_chunks(m.layers[(size_t) l], fixed[(size_t) l]) * G;
            if (dense + m.fixed_bytes + (uint64_t) least * per_pos + beside + reserve > fr ||
                (o.expert_vram > 0 && m.fixed_bytes > o.expert_vram)) {
                char b[640];
                std::snprintf(b, sizeof b, "the model's %.2f GiB of experts fit neither this GPU nor the RAM: beside the %.2f "
                              "GiB of dense weights, the cells of %lld tokens and the %.1f GiB left free the GPU's %.2f GiB of "
                              "free VRAM%s hold under the %.2f GiB of them that the RAM has no room for (%.1f GiB available, "
                              "%.1f GiB of it left free: --ram-reserve-mib).  A smaller quant of the model fits",
                              m.expert_gib, m.dense_gib, (long long) least, (double) reserve / gib, (double) fr / gib,
                              o.expert_vram > 0 ? " (and --expert-cache)" : "", (double) m.fixed_bytes / gib,
                              (double) avail / gib, (double) o.ram_reserve / gib);
                err = b;
                return false;
            }
            m.in_ram = true;
            if (m.arena_bytes > 0) {
                if (cudaHostAlloc((void**) &m.arena, m.arena_bytes, cudaHostAllocPortable) == cudaSuccess) {
                    own(&Owned::pinned, m.arena);
                } else {
                    (void) cudaGetLastError();
                    m.arena_heap.reset(new (std::nothrow) uint8_t[m.arena_bytes]);
                    m.arena = m.arena_heap.get();
                    if (m.arena == nullptr) {
                        err = "no memory for the experts in RAM";
                        return false;
                    }
                }
            }
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
    uint8_t* arena_at = m.arena;
    std::vector<std::vector<int32_t>> by_rank((size_t) g.n_layers);   // (with experts in RAM: a layer's, the ranking's order)
    for (const auto& [l, e] : m.ranked) by_rank[(size_t) l].push_back(e);
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
        std::vector<int32_t> so((size_t) E);
        y.slot_of = dalloc<int32_t>((size_t) E);
        if (!m.in_ram) {   // every expert in its own number's slot
            y.n_slots = (int) E;
            y.slots = dalloc<uint8_t>((size_t) E * y.stride + kWeightPad);
            CK(cudaMemcpy(y.slots, stage, (size_t) E * y.stride, cudaMemcpyHostToDevice));
            CK(cudaMemset(y.slots + (size_t) E * y.stride, 0, kWeightPad));
            for (int64_t e = 0; e < E; ++e) so[(size_t) e] = (int32_t) e;
            y.order = so;
            y.slot_at = so;
        } else {   // the slots in the ranking's order: the ones VRAM alone holds now, the others' blobs into RAM
            y.order = by_rank[(size_t) l];
            y.slot_at.assign((size_t) E, 0);
            for (int64_t k = 0; k < E; ++k) y.slot_at[(size_t) y.order[(size_t) k]] = (int32_t) k;
            y.n_slots = fixed[(size_t) l];
            y.slot_vm = std::make_unique<lamina::core::VmmRange>();
            const int64_t nc = slot_chunks(y, y.n_slots);
            if (!y.slot_vm->reserve((uint64_t) E * y.stride + kWeightPad) ||
                !y.slot_vm->map_range(0, nc, [] { return (lamina::core::VmmChunk) 0; })) {
                err = "no VRAM for the experts that RAM has no room for";
                return false;
            }
            y.slots = y.slot_vm->base();
            y.blob_h.assign((size_t) E, nullptr);
            for (int64_t k = 0; k < E; ++k) {
                const uint8_t* const blob = stage + (size_t) y.order[(size_t) k] * y.stride;
                if (k < y.n_slots) {
                    CK(cudaMemcpy(y.slots + (size_t) k * y.stride, blob, y.stride, cudaMemcpyHostToDevice));
                } else {
                    std::memcpy(arena_at, blob, y.stride);
                    y.blob_h[(size_t) y.order[(size_t) k]] = arena_at;
                    arena_at += y.stride;
                }
            }
            if (nc > 0)
                CK(cudaMemset(y.slots + (size_t) y.n_slots * y.stride, 0,
                              (size_t) nc * lamina::core::vmm_granularity() - (size_t) y.n_slots * y.stride));
            for (int64_t e = 0; e < E; ++e) so[(size_t) e] = y.slot_at[(size_t) e] < y.n_slots ? y.slot_at[(size_t) e] : -1;
        }
        CK(cudaMemcpy(y.slot_of, so.data(), (size_t) E * 4, cudaMemcpyHostToDevice));
    }
    m.load_s = ms_since(t0) / 1000.0;
    char where[320];
    if (m.in_ram)
        std::snprintf(where, sizeof where, "%.2f GiB of dense weights on the GPU, %.2f GiB of experts in RAM%s%s (%s)",
                      m.dense_gib, (double) m.arena_bytes / (double) (1ull << 30), m.arena_heap ? ", not pinned" : "",
                      m.n_fixed > 0 ? (", the " + std::to_string(m.n_fixed) + " that RAM has no room for on the GPU alone").c_str() : "",
                      m.ranking.c_str());
    else
        std::snprintf(where, sizeof where, "%.2f GiB of dense weights and %.2f GiB of experts on the GPU", m.dense_gib,
                      m.expert_gib);
    std::fprintf(stderr, "lamina: Inkling 2 (%s): %lld layers, %lld applications a token (layers %lld-%lld %lld times), "
                 "%lld experts a layer; %s, loaded in %.1f s\n",
                 g.arch.c_str(), (long long) g.n_layers, (long long) g.n_visits(), (long long) g.loop_start,
                 (long long) g.loop_end - 1, (long long) g.n_pass, (long long) E, where, m.load_s);
    return true;
}

// ------------------------------------------------------------------------------------------------ pictures
/// A request's pictures (--vision): the image encoder's rows in prompt order and each prompt cell's row (-1: text)
struct ImgReq {
    std::vector<float> rows;
    std::vector<int32_t> cell_row;
    int32_t row(int64_t p) const { return p >= 0 && p < (int64_t) cell_row.size() ? cell_row[(size_t) p] : -1; }
};

/// Reads the image encoder's file (records 'SVE1': n, nx, ny, n_embd, then n x n_embd floats, a picture each in prompt
/// order) and lays the pictures over the prompt's runs of `pad`, a row a cell; each such cell's id becomes negative, from
/// the picture's key (the server's, img_keys=, else the rows' hash) and its row.  "" or the error.
std::string build_images(const std::string& path, std::vector<int32_t>& ids, int32_t pad, int64_t n_embd,
                         const std::vector<uint64_t>& keys, ImgReq& r) {
    std::vector<lamina::program::VisionImage> imgs;
    if (std::string ve = lamina::program::read_vision_records(path, n_embd, r.rows, imgs); !ve.empty()) return ve;
    const int64_t n = (int64_t) ids.size();
    r.cell_row.assign((size_t) n, -1);
    size_t k = 0;
    for (int64_t i = 0; i < n;) {
        if (ids[(size_t) i] != pad) {
            ++i;
            continue;
        }
        if (k >= imgs.size()) return "the prompt has more images than the embeddings file";
        const lamina::program::VisionImage& im = imgs[k++];
        const uint64_t key = lamina::program::vision_image_key(im, keys.size() == imgs.size() ? &keys[k - 1] : nullptr,
                                                               r.rows.data(), n_embd);
        for (int64_t j = 0; j < im.n; ++j) {
            if (i + j >= n || ids[(size_t) (i + j)] != pad)
                return "image " + std::to_string(k) + " has " + std::to_string((long long) im.n) + " rows but fewer image tokens";
            uint64_t h = key + (uint64_t) j * 0x9E3779B97F4A7C15ull;   // splitmix64 of the key and the row
            h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9ull;
            h = (h ^ (h >> 27)) * 0x94D049BB133111EBull;
            ids[(size_t) (i + j)] = -1 - (int32_t) ((h ^ (h >> 31)) & 0x3fffffff);
            r.cell_row[(size_t) (i + j)] = (int32_t) (im.off / (size_t) n_embd + (size_t) j);
        }
        i += im.n;
    }
    if (k != imgs.size()) return "the embeddings file has more images than the prompt";
    if (n > 0 && r.cell_row.back() >= 0) return "the prompt cannot end in an image";
    return "";
}

// ------------------------------------------------------------------------------------------------ the engine
/// The rows' buffers of a forward: the decode's (kMaxT rows), or a prompt chunk's (`cap` rows, `batch`: its matrix
/// products through llama.cpp's MMQ, its routed entries grouped by expert, a window visit's cells in position order).
// ------------------------------------------------------------------------------------------------ the drafter
/// The model's block drafter as its file has it: DSpark (GGUF architecture dflash with a Markov head; unsloth's
/// dspark-*.gguf beside the model).  A few dense layers over a block of rows - the last token and masks - whose
/// attention reads, beside the block, context cells made from the target's own states of every position; then a
/// recurrent head a drafted position.  llama.cpp's src/models/dspark.cpp (graph_inkling2) is the reference.
struct SparkLayer {
    Mat wq, wk, wv, wo, wk_ctx, wv_ctx, gate, up, down;
    const float *attn_norm = nullptr, *ffn_norm = nullptr, *ctx_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr,
                *k_ctx_norm = nullptr;
    bool window = false;   // its rows see the last n_swa context cells (else every one the engine keeps: --dspark-reach)
};
struct Spark {
    int n_layer = 0, N = 0, H = 0, HK = 0, D = 0, F = 0, R = 0, block = 0, n_swa = 0, V = 0;
    float eps = 1e-6f, cap = 30.f;
    double rope_base = 1e6;
    std::vector<int> taps;      // the target's layer applications whose inputs it reads; the last: the final normed state
    std::vector<SparkLayer> layers;
    std::vector<float> fuse;    // [n_layer + 1][taps]: each layer's weights of the taps, then the head's context's
    const float *out_norm = nullptr, *mask = nullptr, *rnn_norm = nullptr, *joint_w = nullptr, *joint_b = nullptr;
    const uint16_t* w1 = nullptr;     // [the vocabulary, padded][R] bfloat16: a token's row for the head
    std::vector<float> hidden_cols;   // hidden_out's columns [R][N], then its bias [N]: the head's product is made of them
    double gib = 0, load_s = 0;
};

/// The drafter of model `m` from `path`.  False with `err`: not such a file, or not one this path's kernels run.
bool load_spark(const std::string& path, const Model& m, Spark& s, std::string& err) {
    const auto t0 = Clock::now();
    const lamina::GgufModel gm = lamina::GgufModel::open(path);
    const auto& f = gm.meta();
    if (const lamina::MetaValue* a = f.get("general.architecture"); a == nullptr || a->s != "dflash") {
        err = path + " is not a block drafter (general.architecture dflash)";
        return false;
    }
    auto num = [&](const char* k, double def) {
        const lamina::MetaValue* v = f.get(std::string("dflash.") + k);
        return v && v->is_num() ? v->num() : def;
    };
    s.n_layer = (int) num("block_count", 0);
    s.N = (int) num("embedding_length", 0);
    s.F = (int) num("feed_forward_length", 0);
    s.H = (int) num("attention.head_count", 0);
    s.HK = (int) num("attention.head_count_kv", 0);
    s.D = (int) num("attention.key_length", 0);
    s.R = (int) num("dspark.markov_rank", 0);
    s.block = (int) num("block_size", 0);
    s.n_swa = (int) num("attention.sliding_window", 0);
    s.V = (int) num("vocab_size_unpadded", 0);
    s.eps = (float) num("attention.layer_norm_rms_epsilon", 1e-6);
    s.cap = (float) num("final_logit_softcapping", 0);
    s.rope_base = num("rope.freq_base", 10000.0);
    const int Vp = (int) num("vocab_size", 0);
    std::vector<uint64_t> tl, pat;
    f.numbers("dflash.target_layers", &tl);
    f.numbers("dflash.attention.sliding_window_pattern", &pat);
    const Geometry& g = m.g;
    auto no = [&](const std::string& what) {
        err = "the drafter " + path + ": " + what;
        return false;
    };
    if (s.R != 256 || gm.find("markov_w1.weight") == nullptr) return no("no Markov head of rank 256 (the model's own DSpark has one)");
    if (s.block != kMaxT) return no("a block of " + std::to_string(s.block) + " rows, where the kernels take " + std::to_string(kMaxT));
    if (s.N != g.n_embd || s.V != g.n_vocab || Vp < s.V) return no("another model's (its rows or its vocabulary are not this model's)");
    if (s.D != 128 || s.HK <= 0 || s.H % s.HK != 0 || s.H / s.HK > ik::kQH || s.H * s.D % 256 != 0)
        return no("heads the kernels do not take");
    if (s.n_layer < 1 || s.n_layer > 3 || (int) pat.size() != s.n_layer || s.n_swa <= 0 || s.F % 32 != 0 || !(s.cap > 0))
        return no("layers the kernels do not take");
    if (tl.size() < 2 || tl.size() > 4 || (int64_t) tl.back() != g.n_visits()) return no("states of the target this path does not tap");
    for (size_t i = 0; i < tl.size(); ++i) {
        if (i > 0 && tl[i] <= tl[i - 1]) return no("states of the target out of order");
        s.taps.push_back((int) tl[i]);
    }
    for (const char* k : {"dflash.attention.causal", "dflash.sample_from_anchor"})
        if (const lamina::MetaValue* v = f.get(k); v && v->u != 0) return no(std::string(k) + " set");
    Loader L{gm, {}, nullptr};
    size_t fr0 = 0, tot = 0;
    CK(cudaMemGetInfo(&fr0, &tot));
    const size_t nt = s.taps.size();
    {   // each row of the mix: the softmax of its weights
        const std::vector<float> w = L.floats("dspark_fuse.weight");
        if (w.size() != nt * (size_t) (s.n_layer + 1)) return no("a mix of the target's states of another shape");
        s.fuse.resize(w.size());
        for (int r = 0; r <= s.n_layer; ++r) {
            double mx = -1e30, sum = 0;
            for (size_t k = 0; k < nt; ++k) mx = std::max(mx, (double) w[(size_t) r * nt + k]);
            for (size_t k = 0; k < nt; ++k) sum += std::exp((double) w[(size_t) r * nt + k] - mx);
            for (size_t k = 0; k < nt; ++k) s.fuse[(size_t) r * nt + k] = (float) (std::exp((double) w[(size_t) r * nt + k] - mx) / sum);
        }
    }
    s.out_norm = L.f32("output_norm.weight");
    s.mask = L.f32("mask_embd.weight");
    s.rnn_norm = L.f32("dspark_norm.weight");
    s.joint_b = L.f32("joint_proj.bias");
    {
        const lamina::TensorInfo* t = L.find("joint_proj.weight");
        if (t->shape.size() != 2 || (int64_t) t->shape[0] != 2 * s.R + 2 * s.N || (int64_t) t->shape[1] != 3 * s.R)
            return no("a head projection of another shape");
        s.joint_w = L.f32("joint_proj.weight");
    }
    {   // hidden_out [R -> N] by columns, its bias after them
        const lamina::TensorInfo* t = L.find("dspark_hidden_out.weight");
        if (t->shape.size() != 2 || (int64_t) t->shape[0] != s.R || (int64_t) t->shape[1] != s.N) return no("a head output of another shape");
        const std::vector<float> w = L.floats("dspark_hidden_out.weight"), b = L.floats("dspark_hidden_out.bias");
        s.hidden_cols.resize((size_t) (s.R + 1) * s.N);
        for (int o = 0; o < s.N; ++o) {
            for (int r = 0; r < s.R; ++r) s.hidden_cols[(size_t) r * s.N + o] = w[(size_t) o * s.R + r];
            s.hidden_cols[(size_t) s.R * s.N + o] = b[(size_t) o];
        }
    }
    {
        const lamina::TensorInfo* t = nullptr;
        const uint8_t* src = L.host("markov_w1.weight", &t);
        if (t->type != 30 || t->shape.size() != 2 || (int64_t) t->shape[0] != s.R || (int64_t) t->shape[1] != Vp)
            return no("a Markov table that is not [rank, vocabulary] bfloat16");
        uint16_t* d = dalloc<uint16_t>((size_t) Vp * s.R);
        CK(cudaMemcpy(d, src, (size_t) Vp * s.R * 2, cudaMemcpyHostToDevice));
        s.w1 = d;
    }
    s.layers.resize((size_t) s.n_layer);
    for (int i = 0; i < s.n_layer; ++i) {
        SparkLayer& y = s.layers[(size_t) i];
        const std::string b = "blk." + std::to_string(i) + ".";
        y.window = pat[(size_t) i] != 0;
        y.wq = L.mat(b + "attn_q.weight");
        y.wk = L.mat(b + "attn_k.weight");
        y.wv = L.mat(b + "attn_v.weight");
        y.wo = L.mat(b + "attn_output.weight");
        y.wk_ctx = L.mat(b + "attn_k_ctx.weight");
        y.wv_ctx = L.mat(b + "attn_v_ctx.weight");
        y.gate = L.mat(b + "ffn_gate.weight");
        y.up = L.mat(b + "ffn_up.weight");
        y.down = L.mat(b + "ffn_down.weight");
        const int Q = s.H * s.D, KV = s.HK * s.D;
        if (y.wq.n_in != s.N || y.wq.n_out != Q || y.wk.n_out != KV || y.wv.n_out != KV || y.wo.n_in != Q || y.wo.n_out != s.N ||
            y.wk_ctx.n_in != s.N || y.wk_ctx.n_out != KV || y.wv_ctx.n_out != KV || y.gate.n_out != s.F || y.up.n_out != s.F ||
            y.down.n_in != s.F || y.down.n_out != s.N)
            return no("a layer's matrices of another shape");
        for (const Mat* W : {&y.wk_ctx, &y.wv_ctx})
            if (W->type != 30 && !(MMQ::built() && MMQ::fits(W->type, W->n_out) && !MMQ::native_fp4(W->type)))
                return no("context projections in a format a prompt chunk's kernels do not read");
        y.attn_norm = L.f32(b + "attn_norm.weight");
        y.ffn_norm = L.f32(b + "ffn_norm.weight");
        y.ctx_norm = L.f32(b + "dflare_norm.weight");
        y.q_norm = L.f32(b + "attn_q_norm.weight");
        y.k_norm = L.f32(b + "attn_k_norm.weight");
        y.k_ctx_norm = L.f32(b + "attn_k_ctx_norm.weight");
    }
    size_t fr1 = 0;
    CK(cudaMemGetInfo(&fr1, &tot));
    s.gib = (double) (fr0 - fr1) / (double) (1ull << 30);
    s.load_s = ms_since(t0) / 1000.0;
    return true;
}

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
    float* crows = nullptr;   // with experts in RAM: the CPU's rows of a visit, packed in the entries' order
    // the step record: the rows' positions, then the cells in use, the full visits' {stretch, stretches in use} and
    // a chunk's window range's cells
    int32_t *step_d = nullptr, *pos_d = nullptr, *live_d = nullptr, *dyn_d = nullptr, *wlive_d = nullptr;
    // with cells on cards: -sink a row and head, the cards' sums [cards][T][H][D + 2], a chunk's own partial sums
    float *ns = nullptr, *rpart = nullptr, *tpart = nullptr, *spart = nullptr;
    int32_t *ids = nullptr, *grp_start = nullptr, *n_groups = nullptr, *n_miss = nullptr, *ent_dst = nullptr,
            *ent_tok = nullptr;
    unsigned long long* grp_ptr = nullptr;
    void* scratch = nullptr;
    // the head rows' tokens by the request's sampler, and 1 where a row's logits hold a non-finite one
    int32_t *pick = nullptr, *bad = nullptr;
    // pinned host rows
    float* emb_h = nullptr;
    int32_t *step_h = nullptr, *pick_h = nullptr;
    // a chunk's: the inputs' and the attention output's MMQ images a layout, the routed entries by expert (their
    // token rows, their experts, each expert's bounds), their gate/up rows and hidden rows, a window visit's cells
    uint8_t *xq[3][3] = {}, *aq[3] = {}, *tq = nullptr, *hq = nullptr, *wk = nullptr, *wv = nullptr;
    bool have[4][3] = {};
    float *gu = nullptr, *h = nullptr;
    float* wf = nullptr;   // a BF16 matrix widened to floats for a GEMM
    float *tqs = nullptr, *hqs = nullptr;   // the entries' row scales where the experts run on FP4 tensor cores
    float *cz = nullptr, *cs1 = nullptr, *cs2 = nullptr;   // the loop gates' normed pair [T][2 N] and their two products
    // with the drafter: the rows' mixes of the target's states [its layers + 1][cap][N] (normed in place), a layer's
    // context K and V projections and K rotated, the rows' rotations [cap][D / 2][2] (their pinned source), the mixes'
    // q8_1 images (a decode forward's) or one's MMQ image (a chunk's)
    float *f_fused = nullptr, *f_kc = nullptr, *f_vc = nullptr, *f_kn = nullptr, *f_cs = nullptr, *f_cs_h = nullptr;
    ik::Q81* f_img = nullptr;
    uint8_t* f_q = nullptr;
    int32_t *ent_row = nullptr, *ent_exp = nullptr, *bounds = nullptr, *ident = nullptr;
    int64_t ids_rows = 0;
    int32_t *ids_h = nullptr, *ent_row_h = nullptr, *ent_dst_h = nullptr, *ent_exp_h = nullptr, *bounds_h = nullptr;
    // the experts on the FP16 tensor cores (nvfp4_gemm): each place's blob, the tiles of 64 entries
    unsigned long long *pptr = nullptr, *pptr_h = nullptr;
    int32_t *tiles = nullptr, *tiles_h = nullptr;
    int n_part = 0;   // stretches `part` holds a row and head
    int head_rows() const { return batch ? 1 : cap; }
};

struct Engine {
    const Model* m = nullptr;
    cudaStream_t st = nullptr;
    int64_t ctx = 0;        // positions a sequence may take
    bool kv_q8 = false;
    bool grow = true;       // the full visits' cells mapped as the context grows (CUDA virtual memory)
    size_t reserve = 0;     // VRAM left free
    bool prof = false;      // LAMINA_INK_PROF: the steps timed (a wait after each); 2: every launch, by events
    int prof_level = 0;
    bool use_graphs = true; // a forward of T rows replayed from its captured graph
    std::map<int, cudaGraphExec_t> graphs;   // by rows * 16 + head rows
    int prefill = 1024;     // rows a prompt chunk takes (0: prompts through the decode's forwards)
    // A chunk grows to prefill_max rows for a part that long while the VRAM free above the reserve holds its larger
    // buffers beside the cells it will write (~0.75 MiB a row), and is back at prefill's when it does not, and after
    // a request: the VRAM a request leaves is what it was.  Not with experts in RAM, which take free VRAM first.
    int prefill_max = 4096;
    int chunk_cap = 0;      // the rows the chunk's buffers hold now
    std::vector<void*> ck_dev, ck_pin;   // those buffers (in the run's lists too)
    int chunk_min = 17;     // a shorter prompt part goes through the decode's forwards (LAMINA_INK_CHUNK_MIN: a test's)
    bool fuse_products = true;   // a decode forward's five attention projections in one launch (LAMINA_INK_FUSE=0: apart)
    // A chunk's NVFP4 experts on the FP16 tensor cores (nvfp4_gemm) instead of MMQ: where the card has no FP4 tensor
    // cores of its own, from tc_min rows a chunk (LAMINA_INK_TC: 0 never, 1 on any card; LAMINA_INK_TC_MIN: tests')
    bool tc_on = false;
    int tc_min = 48;
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
    float* win_raw = nullptr;        // a decode forward of several rows: every visit's convolution inputs [kMaxT] rows
    int64_t win_p0 = -1;             // a window's first position while its rows wait for keep() (-1: none)
    int win_T = 0;
    // ---- the full visits' cells over other cards (--cell-gpus).  The main card keeps the newest [lo, pos); the
    // positions before `lo` are on the cards, in units of `unit` positions in the order they left the main card
    // (unit_card: each unit's card).  A card's cells are all older than any row by more than the bias table reaches,
    // so it scores them with q and -sink alone (no position), and returns each row and head's partial sums.
    struct Card {
        int dev = -1;
        std::string name;
        cudaStream_t st = nullptr;
        cudaEvent_t ev = nullptr;         // behind its queued scan
        std::vector<std::unique_ptr<lamina::core::VmmRange>> k_vm, v_vm;   // a full visit each
        int64_t cap = 0, cells = 0;       // cells it has room for, and holds
        int n_part = 0;                   // stretches its `part` holds a row and head
        float *q = nullptr, *ns = nullptr, *part = nullptr, *sum = nullptr;   // a decode forward's rows on the card
        float *cq = nullptr, *cns = nullptr, *csum = nullptr;                 // a chunk's
        int32_t* cpos = nullptr;          // its rows' positions (kFar)
        float* csum_h = nullptr;          // pinned
        int32_t* step = nullptr;          // [kMaxT] positions (kFar) | cells | {stretch, stretches in use, 0}
        int32_t* step_h = nullptr;        // pinned
        float* sum_h = nullptr;           // pinned: its sums on their way to the main card
        std::vector<void*> dev_mem, host_mem;
    };
    std::vector<Card> cards;
    std::vector<int> cell_devs;        // --cell-gpus
    std::vector<std::string> cell_clocks;   // --cell-gpu-clocks: a card's lock while a request runs (the last for the rest)
    lamina::core::GpuClocks clocks;
    double clocks_hold = 1.0;          // seconds a lock outlasts a request
    int main_dev = 0;
    int64_t unit = kGrow;              // positions a move takes: whole mapping chunks, of K and V alike
    int64_t lo = 0;                    // the main card's first cell
    int64_t main_limit = 0;            // --main-cells (a test's): the cells the main card keeps at most
    int64_t spread = 0;                // --cell-spread: tokens past which the cards share the cells (0: when it is full)
    static constexpr int kBalance = 6; // units a growth step moves to even the cards out
    std::vector<int> unit_card;        // unit u -> its card
    std::vector<int> full_of;          // a visit -> its index among the full visits, or -1
    float *xq_h = nullptr, *xs_h = nullptr;   // pinned: a forward's q and -sink on their way to the cards
    float *pq_h = nullptr, *ps_h = nullptr;   // a chunk's
    uint8_t* stage_h[2] = {};          // pinned: a unit of one array's cells on its way to a card
    bool far_mode = false;             // the forward under way scores cells on cards
    // a forward with cells on cards, as graph segments: after each (but the last) the cards score full visit `fi`
    // while the main card runs its own scan `at`, issued apart
    struct Seg {
        cudaGraphExec_t exec = nullptr;
        int fi = -1;      // after it the cards score full visit `fi`
        ik::AttnIn at{};
        int moe = -1;     // after it the host computes visit `moe`'s routed experts that the card has no slot for
    };
    std::map<int, std::vector<Seg>> seg_graphs;   // by rows * 16 + head rows
    std::vector<Seg>* cur_segs = nullptr;         // the one being captured
    bool capturing = false;

    // ---- the drafter (--dspark): its weights, a ring of context cells a layer (K and V of every position the target
    // ran, made from the target's states in that forward: positions as slots, so a block's own rows at the sequence's
    // end go in after them and are written over by the rows that stay), the block's buffers, the head's product.
    const Spark* sp = nullptr;
    int64_t sp_reach = 32768;        // context cells a layer without a window keeps (--dspark-reach)
    static constexpr int64_t kSparkSlack = 4096;   // cells a ring keeps past its layer's: a state may step back that far
    struct SparkRing {
        ik::Cells cells{};
        int64_t see = 0;             // context cells its layer's rows see
        int chunk = 0, n_chunks = 0; // a scan's stretches: cells each, how many at most
    };
    std::vector<SparkRing> sp_ring;
    std::vector<double> sp_theta;    // a pair's angle a position
    float* sp_wf = nullptr;          // a BF16 context projection widened to floats (a chunk's GEMM)
    // the context the rings hold of the state's text: positions [sp_vlo, sp_vhi), written since without a break, the
    // highest written since then one before sp_whi (a ring of `cap` slots has lost what is `cap` below that)
    int64_t sp_vlo = 0, sp_vhi = 0, sp_whi = 0;
    float* frnn = nullptr;           // [kMaxT][N]: the head's context of the last forward's rows (a chunk: its last row's)
    int64_t frnn_p0 = -1;            // the first row's position
    int frnn_n = 0;                  // rows of it that are the state's
    struct SparkBlock {              // a block's rows
        float *x = nullptr, *x0 = nullptr, *xn = nullptr, *qr = nullptr, *q = nullptr, *kr = nullptr, *vr = nullptr,
              *kn = nullptr, *cs = nullptr, *zero = nullptr, *part = nullptr, *sum = nullptr, *ao = nullptr, *o = nullptr,
              *g = nullptr, *u = nullptr, *h = nullptr, *d = nullptr, *hn = nullptr, *base = nullptr, *pre = nullptr,
              *state = nullptr, *vpart = nullptr, *ctx = nullptr, *c0 = nullptr, *mscale = nullptr;
        ik::Q81 *img = nullptr, *aimg = nullptr, *himg = nullptr, *hnimg = nullptr;
        int8_t *M8 = nullptr, *lat = nullptr;   // [V][R]: the target's head over hidden_out's columns, a scale a token; a step's latent
        // the step record: the rows' position for the scan (the block's last: every row sees the block), the cells in
        // use, the rows' own positions, a layer's stretches {cells, in use, first slot}, the token before the drafts,
        // then the head's {the drafts' joint probability, the floor, 1 once they ended}
        int32_t *step = nullptr, *out = nullptr;
        float *emb_h = nullptr, *cs_h = nullptr;
        int32_t *step_h = nullptr, *out_h = nullptr;
        int n_step = 0;
    } sb;
    cudaGraphExec_t sp_graph = nullptr;         // a block's layers and its first draft
    std::map<int, cudaGraphExec_t> sp_graphs;   // the drafts after the first: by their count
    long long sp_blocks = 0;
    double sp_ms = 0;

    // ---- experts in RAM (Model::in_ram).  The card has n_held of them, a layer's in its first on_card[] slots: mapped
    // chunks of the layers' address ranges, taken while VRAM is free above the reserve and given up to the cells as
    // the context grows.  A decode forward that may route an expert the card does not have is a graph segment a
    // visit: behind the routing the host reads which experts it took and the MoE's input (a bell rings when both
    // are in pinned memory), the CPU pool computes the entries without a slot while the card computes the others,
    // and their rows go back before the experts' rows are summed.  A chunk sends the RAM experts it routes to a
    // scratch on the card.
    //
    // Which experts are on the card follows the decode forwards' routing.  An entry adds to its expert's heat, a
    // count that fades (a forward makes the next entries weigh 1 / kHeatKeep more) and starts from the ranking's
    // order; a looped layer's expert collects three times a token.  The card has the hottest: it gives up the
    // coldest and takes the hottest whatever their layers (give_way, place: when the cells grow, and after a request
    // the slots move between layers), and between forwards a layer's hottest expert off the card changes places with
    // its coldest one on it once it leads by kSwapLead (adapt: its blob sent into that slot, kSwaps a forward at
    // most).  A replayed agent session at 6 GiB of experts on the card: 27.5% of the entries the CPU's as the
    // built-in ranking places them, 15.1% with the swaps in a layer, 13.9% with slots moving between layers (13.4%
    // by a ranking of the session's own counts); ~1,000 copies in 8,700 tokens.
    std::unique_ptr<C::ExpertPool> pool;
    int cpu_threads = 0;             // --cpu-threads (0: a worker a core)
    std::vector<int> on_card;           // a layer's slots in use
    int64_t n_held = 0;              // their sum
    std::vector<std::vector<int32_t>> slot_exp, slot_at;   // [layer]: the expert in a slot; an expert's slot or -1
    std::vector<std::vector<float>> heat;                  // [layer][expert]
    float heat_w = 1.f;                                    // what an entry adds now
    static constexpr float kHeatKeep = 0.999f, kSwapLead = 1.25f;
    static constexpr int kSwaps = 2;
    int32_t* swap_h = nullptr;       // pinned: the slot numbers a forward's swaps send
    long long swaps = 0;
    std::string profile_save;        // --expert-profile-save: the ranking the heat makes, written there
    double profile_every = 60.0;     // --expert-profile-save-every (seconds here)
    Clock::time_point profile_at = Clock::now();
    bool profile_new = false;        // routing counted since the last save
    std::vector<lamina::core::VmmChunk> spare;   // chunks unmapped and not yet mapped elsewhere
    bool moe_cuts = false;           // the forward under way sends its routing to the host
    int32_t* moe_h = nullptr;        // pinned: a visit's routed experts [kMaxT * top_k], then its MoE input [kMaxT][N]
    int32_t* bell_h = nullptr;       // pinned: -1, then the visit's entries without a slot once those have arrived
    float* rows_h = nullptr;         // pinned: the CPU's rows, packed
    std::vector<uint8_t> acts;       // the rows' activations as the layer's dots take them
    std::vector<C::ExpertJobMulti> jobs;
    std::unique_ptr<lamina::core::VmmRange> x_vm;   // a chunk's scratch: a layer's RAM experts
    int64_t x_chunks = 0;            // its chunks mapped
    // A chunk's RAM experts with the fewest rows are the CPU's while the others' blobs are on their way to the card:
    // as many as take the CPU no longer than the uploads left (the rates measured at the start)
    static constexpr int kChunkCpu = 1024;   // entries a visit at most
    float *nrm_p = nullptr, *crow_p = nullptr;   // pinned: a chunk visit's MoE input rows; the CPU's rows, packed
    int32_t* cdst_p = nullptr;       // pinned: their entries
    float* x_rows = nullptr;         // device: the packed rows
    int32_t* x_dst = nullptr;        // device: their entries
    std::vector<uint8_t> cacts;      // a chunk's rows' activations, made when a job first takes a row
    std::vector<int32_t> cact_of;    // the visit that made a row's (a count of visits)
    int32_t cvisit = 0;
    double up_ms_mb = 0.12;          // an upload's ms a MB
    std::vector<float> cpu_job_ms, cpu_row_ms;   // [layer]: a CPU job of one row, and each row more
    long long ck_entries = 0, ck_cpu = 0, ck_sent = 0;   // a chunk's RAM entries, the CPU's of them, experts uploaded
    long long ex_entries = 0, ex_ram = 0;   // routed entries of the decode forwards, and those the CPU computed
    double ex_ms = 0;                // the CPU's part of them
    // LAMINA_INK_ROUTE=FILE: the routing the host sees (a chunk's, and a decode forward's with experts in RAM), for
    // placement simulations: "INKR", layers, experts, top_k, each layer's blob bytes (int64); then records of
    // {kind (0 a chunk's visit, 1 a decode forward's, 2 a request's end), layer, n, n experts}, int32 each
    FILE* route_f = nullptr;
    void route_dump(int kind, int layer, int n, const int32_t* ids) {
        if (route_f == nullptr) return;
        const int32_t head[3] = {kind, layer, n};
        std::fwrite(head, 4, 3, route_f);
        if (n > 0) std::fwrite(ids, 4, (size_t) n, route_f);
    }

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
    bool bf16_chunks = false;        // a BF16 matrix among a chunk's products
    cublasHandle_t blas = nullptr;   // a chunk's float products (the loop gates, the router)
    ~Engine() {
        if (route_f != nullptr) std::fclose(route_f);
        if (blas != nullptr) cublasDestroy(blas);
        for (const lamina::core::VmmChunk h : spare) lamina::core::vmm_chunk_free(h);
        for (Card& c : cards) {   // a card's memory is freed with the card current
            if (cudaSetDevice(c.dev) != cudaSuccess) continue;
            (void) cudaStreamSynchronize(c.st);
            c.k_vm.clear();
            c.v_vm.clear();
            for (void* d : c.dev_mem) cudaFree(d);
            for (void* h : c.host_mem) cudaFreeHost(h);
            cudaEventDestroy(c.ev);
            cudaStreamDestroy(c.st);
        }
        if (!cards.empty()) (void) cudaSetDevice(main_dev);
        (void) cudaGetLastError();
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
        const size_t n_ent = T * (size_t) g.top_k;
        if (batch) {
            R.nrm = dzero<float>(3 * T * N);
            R.ids = dzero<int32_t>(n_ent);
        } else {   // (the decode's: the routing and the MoE's input in a row, one copy to the host with experts in RAM)
            R.ids = dzero<int32_t>(n_ent + 3 * T * N);
            R.nrm = (float*) (R.ids + n_ent);
        }
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
        R.aout = dzero<float>(T * Q);
        R.aimg = dzero<ik::Q81>(Th * (Q / 32));
        R.oraw = dzero<float>(T * N);
        R.fraw = dzero<float>(T * N);
        R.w = dzero<float>(n_ent);
        R.rows = dzero<float>(n_ent * N);
        R.probs = dzero<float>(T * E);
        R.ctl_h = dzero<float>(3 * T * (size_t) std::max<int64_t>(g.gate_rank, 1));
        R.logits = dzero<float>(Th * (size_t) g.n_vocab);
        R.pick = dzero<int32_t>(2 * Th);
        R.bad = R.pick + Th;
        R.pick_h = halloc<int32_t>(2 * Th);
        if (sp != nullptr) {
            const size_t L = (size_t) sp->n_layer, KVs = (size_t) sp->HK * (size_t) sp->D;
            R.f_fused = dzero<float>((L + 1) * T * N);
            R.f_kc = dzero<float>(T * KVs);
            R.f_vc = dzero<float>(T * KVs);
            R.f_kn = dzero<float>(T * KVs);
            R.f_cs = dzero<float>(T * (size_t) sp->D);
            R.f_cs_h = halloc<float>(T * (size_t) sp->D);
            if (batch) R.f_q = dzero<uint8_t>(MMQ::q8_bytes((int64_t) T, (int64_t) N));
            else R.f_img = dzero<ik::Q81>(L * T * (N / 32));
        }
        R.step_d = dzero<int32_t>(T + 5);
        R.pos_d = R.step_d;
        R.live_d = R.step_d + T;
        R.dyn_d = R.step_d + T + 1;   // {the first cell in use, the stretches in use, the first one's number}
        R.wlive_d = R.step_d + T + 4;
        R.emb_h = halloc<float>(T * N);
        R.step_h = halloc<int32_t>(T + 5);
        R.ns = dzero<float>(T * H);
        if (batch) R.spart = dzero<float>((size_t) seg_cap() * H * (D_ + 2));
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
        R.tqs = dzero<float>(n_ent);
        R.hqs = dzero<float>(n_ent);
        R.hq = dzero<uint8_t>(MMQ::q8_bytes((int64_t) n_ent, (int64_t) F));
        R.gu = dzero<float>(n_ent * 2 * F);
        R.h = dzero<float>(n_ent * F);
        R.cz = dzero<float>(T * 2 * N);
        if (bf16_chunks) R.wf = dzero<float>(std::max(Q * N, 2 * F * N));
        R.cs1 = dzero<float>(T * N);
        R.cs2 = dzero<float>(T * N);
        R.ids_rows = MMQ::ids_table_rows((int64_t) n_ent);
        R.ent_dst = dzero<int32_t>((size_t) R.ids_rows);
        R.ident = dzero<int32_t>((size_t) R.ids_rows);
        CK(cudaDeviceSynchronize());   // (the zeroing is not the stream's: it would land on the map)
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
        const size_t n_tiles = n_ent / (size_t) K::kNvfp4GemmEntries + E;
        R.pptr = dzero<unsigned long long>(E);
        R.pptr_h = halloc<unsigned long long>(E);
        R.tiles = dzero<int32_t>(n_tiles);
        R.tiles_h = halloc<int32_t>(n_tiles);
    }

    // ---- a chunk's attention (attn_tc_kernel): its blocks, and the segments its cells are scored in
    int main_sms_ = 0, tc_segs = 0;   // (LAMINA_INK_TC_SEGS: that many segments, a test's)
    /// this card's SMs (called with it current)
    int main_sms() {
        if (main_sms_ == 0) {
            cudaDeviceProp prop{};
            int dev = 0;
            CK(cudaGetDevice(&dev));
            CK(cudaGetDeviceProperties(&prop, dev));
            main_sms_ = std::max(1, prop.multiProcessorCount);
        }
        return main_sms_;
    }
    static constexpr int kTcWarps = 4;   // warps a block, 16 rows each (8 gained q8_0 cells a tenth on the RTX 3090
                                         // from 2,048 rows and lost as much on the RTX 5070 Ti)
    /// rows of partial sums the segments of a chunk write at most
    int seg_cap() { return 640 * main_sms() / (int) m->g.n_head + 64; }
    /// The segments T rows' cells are scored in: more blocks than the rows and heads make, so that a few rows fill
    /// the card too - five blocks an SM over 16,384 cells or more (the blocks end apart: the last ones alone would
    /// leave SMs idle), one and a quarter over 4,096 or fewer (a block's start and end cost a short range more), a
    /// segment eight tiles at least
    int tc_segments(int T, long long cells) {
        const int rows = 16 * kTcWarps, blocks = (T + rows - 1) / rows * (int) m->g.n_head;
        const double fill = std::min(5.0, std::max(1.25, 5.0 * (double) cells / 16384.0));
        const long long want = tc_segs > 0 ? tc_segs : ((long long) (fill * main_sms()) + blocks - 1) / blocks;
        return (int) std::max(1LL, std::min({want, (long long) ik::kTcSegs, cells / (8 * ik::kTcCells), (long long) (seg_cap() / T)}));
    }
    void tc_launch(const ik::TcIn& a, int n_seg, cudaStream_t s) {
        const dim3 grid((unsigned) ((a.T + 16 * kTcWarps - 1) / (16 * kTcWarps)), (unsigned) a.H, (unsigned) n_seg);
        if (kv_q8) ik::attn_tc_kernel<kTcWarps, true><<<grid, 32 * kTcWarps, 0, s>>>(a);
        else ik::attn_tc_kernel<kTcWarps, false><<<grid, 32 * kTcWarps, 0, s>>>(a);
    }

    bool elastic() const { return prefill > 0 && prefill_max > prefill && !m->in_ram; }
    int chunk_most() const { return elastic() ? prefill_max : prefill; }
    /// VRAM a chunk's buffers take a row, as measured (896 MiB at 1,024 rows with cuBLAS's and MMQ's 125 beside
    /// them; with the drafter the rows' mixes of the target's states and their context projections too)
    uint64_t chunk_bytes(int rows) const { return (uint64_t) rows * ((753ull + (sp ? 52 : 0)) << 10); }
    /// A chunk's partial sums of the cells on cards (the cards are known by then)
    void chunk_parts() {
        const size_t d0 = g_owned ? g_owned->dev.size() : 0;
        const size_t prow = (size_t) chunk_cap * (size_t) m->g.n_head, D_ = (size_t) m->g.head_dim;
        P.rpart = dzero<float>(cards.size() * prow * (D_ + 2));
        P.tpart = dzero<float>(prow * (D_ + 2));
        if (g_owned) ck_dev.insert(ck_dev.end(), g_owned->dev.begin() + (long) d0, g_owned->dev.end());
    }
    /// The chunk's buffers for `rows` rows
    void chunk_alloc(int rows) {
        const size_t d0 = g_owned ? g_owned->dev.size() : 0, h0 = g_owned ? g_owned->pinned.size() : 0;
        P = Rows{};
        make_rows(P, rows, true);
        if (g_owned) {
            ck_dev.assign(g_owned->dev.begin() + (long) d0, g_owned->dev.end());
            ck_pin.assign(g_owned->pinned.begin() + (long) h0, g_owned->pinned.end());
        }
        chunk_cap = rows;
        if (!cards.empty()) chunk_parts();
        CK(cudaDeviceSynchronize());   // (the buffers' zeroing has run: the stream does not wait for it)
    }
    /// They are freed (the stream idle); a prompt's logits they held move to the decode rows' first
    void chunk_free() {
        CK(cudaStreamSynchronize(st));
        if (lg_rows == &P) {
            CK(cudaMemcpy(D.logits, P.logits, (size_t) m->g.n_vocab * 4, cudaMemcpyDeviceToDevice));
            CK(cudaDeviceSynchronize());   // (a copy on the device may return before it ran)
            lg_rows = &D;
        }
        if (g_owned) {
            std::lock_guard<std::mutex> lk(g_owned->mu);
            auto drop = [](std::vector<void*>& all, const std::vector<void*>& gone) {
                all.erase(std::remove_if(all.begin(), all.end(), [&](void* q) {
                    return std::find(gone.begin(), gone.end(), q) != gone.end();
                }), all.end());
            };
            drop(g_owned->dev, ck_dev);
            drop(g_owned->pinned, ck_pin);
        }
        for (void* q : ck_dev) CK(cudaFree(q));
        for (void* q : ck_pin) CK(cudaFreeHost(q));
        ck_dev.clear();
        ck_pin.clear();
        P = Rows{};
        chunk_cap = 0;
    }
    void chunk_set(int rows) {
        if (rows == chunk_cap) return;
        chunk_free();
        chunk_alloc(rows);
    }
    /// ensure(n) would find its VRAM on this card with `extra` more bytes of it taken (fewer: freed)
    bool cells_fit(int64_t n, int64_t extra) {
        const int64_t G = (int64_t) lamina::core::vmm_granularity();
        auto chunks = [&](int64_t cells) { return ((int64_t) ((uint64_t) cells * cell_bytes) + G - 1) / G; };
        int64_t need = 0;
        if (n > mapped) {
            int64_t ranges = 0;
            for (const Visit& V : visits) ranges += V.k_vm ? 2 : 0;
            need = (chunks(std::min<int64_t>(ctx, (n + kGrow - 1) / kGrow * kGrow)) - chunks(mapped)) * ranges;
        }
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        const int64_t above = (int64_t) fr - (int64_t) reserve - extra;
        return above >= 0 && need <= (int64_t) spare.size() + above / G;
    }
    /// Before a chunk of a part with `left` tokens to read: the rows its buffers hold (chunk_cap)
    void chunk_fit(int64_t left) {
        if (!elastic()) return;
        const int64_t p0 = pos();
        auto fits = [&](int rows) {   // the buffers of `rows` rows beside the cells of a chunk that long
            const int64_t more = (int64_t) chunk_bytes(rows) - (int64_t) chunk_bytes(chunk_cap);
            return cells_fit(std::min<int64_t>(ctx, p0 + std::min<int64_t>(rows, left)), more > 0 ? more + (256ll << 20) : more);
        };
        for (int rows = prefill_max; rows > chunk_cap; rows /= 2)
            if (rows > prefill && left > rows / 2 && fits(rows)) {
                chunk_set(rows);
                return;
            }
        if (chunk_cap > prefill && !fits(chunk_cap)) chunk_set(prefill);
    }
    /// After a request: the chunk's buffers are prefill's again
    void chunk_base() {
        if (chunk_cap > prefill) chunk_set(prefill);
    }

    /// Whether a prompt chunk's products have their MMQ kernels on this GPU: "" or why not
    std::string batch_problem() {
        if (!MMQ::built()) return "this build has no MMQ kernels";
        const Geometry& g = m->g;
        for (const Layer& L : m->layers) {
            for (const Mat* W : {&L.wq, &L.wk, &L.wv, &L.wrel, &L.wgate, &L.wo}) {
                if (W->type == 30) {   // BF16: widened and through a GEMM
                    bf16_chunks = true;
                    continue;
                }
                if (!MMQ::fits(W->type, W->n_out)) return "a dense matrix's format has no MMQ kernel here";
                lay_used[MMQ::ds_layout(W->type)] = true;
            }
            if (L.gl.gu_type == 30 || L.gl.d_type == 30) bf16_chunks = true;   // (BF16 experts: a GEMM an expert)
            if ((L.gl.gu_type != 30 && !MMQ::fits(L.gl.gu_type, 2 * g.n_ff)) || (L.gl.d_type != 30 && !MMQ::fits(L.gl.d_type, g.n_embd)))
                return "an expert format has no MMQ kernel here";
            // (a gate/up row that ends inside a kernel's step - FP4 rows on an RTX 50, whose step is 512 values - is
            // read on into the next row, the last one into the blob's down rows, against the zeros that pad the
            // activations: products of zero)
        }
        return {};
    }

    void init(const Model& model, int64_t max_ctx, bool q8, bool grow_cells, size_t reserve_bytes) {
        m = &model;
        kv_q8 = q8;
        reserve = reserve_bytes;
        prof = std::getenv("LAMINA_INK_PROF") != nullptr;
        prof_level = prof ? std::max(1, std::atoi(std::getenv("LAMINA_INK_PROF"))) : 0;
        if (const char* d = std::getenv("LAMINA_INK_DUP")) {
            const std::string t = d;
            const size_t c = t.rfind(':');
            dup_name = t.substr(0, c);
            dup_n = c == std::string::npos ? 1 : std::max(0, std::atoi(t.c_str() + c + 1));
        }
        nan_probe = std::getenv("LAMINA_INK_NAN") != nullptr;
        xrow_on = std::getenv("LAMINA_INK_XROW") != nullptr;
        if (const char* r = std::getenv("LAMINA_INK_ROWS")) feed_rows = std::max(1, std::min(kMaxT, std::atoi(r)));
        if (const char* r = std::getenv("LAMINA_INK_CHUNK_MIN")) chunk_min = std::max(1, std::atoi(r));
        if (const char* r = std::getenv("LAMINA_INK_FUSE")) fuse_products = std::atoi(r) != 0;
        tc_on = MMQ::built() && !MMQ::native_fp4(40);
        if (const char* r = std::getenv("LAMINA_INK_TC")) tc_on = std::atoi(r) != 0;
        if (const char* r = std::getenv("LAMINA_INK_TC_MIN")) tc_min = std::max(0, std::atoi(r));
        if (const char* r = std::getenv("LAMINA_INK_TC_SEGS")) tc_segs = std::max(0, std::atoi(r));
        if (prof || nan_probe || xrow_on) use_graphs = false;
        if (nan_probe) nan_at = dzero<long long>(1);
        const Geometry& g = m->g;
        const size_t N = (size_t) g.n_embd, D_ = (size_t) g.head_dim, KV = (size_t) g.kv_width(), C = (size_t) g.n_conv;
        CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        own(&Owned::streams, st);
        cell_bytes = kv_q8 ? (size_t) g.n_head_kv * (D_ / 32) * 34 : KV * 2;
        make_rows(D, kMaxT, false);
        logits_h = halloc<float>((size_t) g.n_vocab);
        if (sp != nullptr) spark_init();
        if (const char* f = std::getenv("LAMINA_INK_ROUTE")) {
            route_f = std::fopen(f, "wb");
            if (route_f != nullptr) {
                const int32_t head[3] = {(int32_t) g.n_layers, (int32_t) g.n_expert, (int32_t) g.top_k};
                std::fwrite("INKR", 1, 4, route_f);
                std::fwrite(head, 4, 3, route_f);
                for (const Layer& L : m->layers) {
                    const int64_t b = (int64_t) L.stride;
                    std::fwrite(&b, 8, 1, route_f);
                }
            }
        }
        if (m->in_ram) {
            const size_t nl = m->layers.size(), ne = (size_t) g.n_expert;
            on_card.resize(nl);
            slot_exp.resize(nl);
            slot_at.assign(nl, std::vector<int32_t>(ne, -1));
            heat.assign(nl, std::vector<float>(ne, 0.f));
            for (size_t l = 0; l < nl; ++l) {
                const Layer& L = m->layers[l];
                on_card[l] = L.n_slots;
                slot_exp[l].assign(L.order.begin(), L.order.begin() + L.n_slots);
                for (int k = 0; k < L.n_slots; ++k) slot_at[l][(size_t) L.order[(size_t) k]] = k;
            }
            // (the start's heat: the ranking's order, a few entries' worth)
            for (size_t r = 0; r < m->ranked.size(); ++r)
                heat[(size_t) m->ranked[r].first][(size_t) m->ranked[r].second] = 8.f * (1.f - (float) r / (float) m->ranked.size());
            n_held = m->n_fixed;
            swap_h = halloc<int32_t>(2 * kSwaps);
            pool = std::make_unique<C::ExpertPool>(cpu_threads);
            const size_t n_ent = (size_t) kMaxT * (size_t) g.top_k;
            moe_h = halloc<int32_t>(n_ent + (size_t) kMaxT * N);
            bell_h = halloc<int32_t>(1);
            rows_h = halloc<float>(n_ent * N);
            D.crows = dzero<float>(n_ent * N);
            acts.resize((size_t) kMaxT * C::kNativeActBytes);
            size_t most = 0;
            for (const Layer& L : m->layers) most = std::max(most, (size_t) g.n_expert * L.stride);
            x_vm = std::make_unique<lamina::core::VmmRange>();
            if (!x_vm->reserve(most + kWeightPad)) {
                std::fprintf(stderr, "lamina: the address range of a chunk's experts could not be reserved\n");
                throw Fatal{1};
            }
            rates();
        }
        if (prefill > 0) {
            if (const std::string why = batch_problem(); !why.empty()) {
                std::fprintf(stderr, "lamina: prompts go through the decode's forwards, %d rows at a time: %s\n", kMaxT,
                             why.c_str());
                prefill = 0;
            } else {
                // A chunk's buffers are VRAM the cells do not get (~0.75 MiB a row): where this card alone holds the
                // context and would hold less than the one asked for (not set: the floor) with them, fewer rows.
                if (cell_devs.empty()) {
                    const uint64_t per_pos = (uint64_t) g.n_full_visits() * 2 * cell_bytes;
                    // (with experts in RAM they give way to a longer context: the floor's cells are what must fit)
                    const int64_t want_ctx = max_ctx > 0 ? std::min<int64_t>(max_ctx, m->trained_ctx > 0 ? m->trained_ctx : 1048576)
                                                         : kContextFloor;
                    const uint64_t want = (uint64_t) (m->in_ram ? std::min(want_ctx, kContextFloor) : want_ctx) * per_pos;
                    // (still to come beside the cells: the window rings and the cache's image of them; a chunk's rows
                    // with cuBLAS and MMQ beside them, as measured: 896 MiB at 1,024 rows)
                    const uint64_t ring = (uint64_t) (g.n_visits() - g.n_full_visits()) * 2 * cell_bytes *
                                          (uint64_t) ((g.n_swa + kMaxT + kRingSub - 1) / kRingSub * kRingSub);
                    const uint64_t rest = ring * (cache_on ? 2 : 1) + (32ull << 20);
                    // (with the drafter: the rows' mixes of the target's states and their context projections)
                    auto rows_bytes = [&](int rows) { return chunk_bytes(rows) + (125ull << 20); };
                    size_t fr = 0, tot = 0;
                    CK(cudaMemGetInfo(&fr, &tot));
                    const int asked = prefill;
                    while (prefill >= 128 && rows_bytes(prefill) + rest + want + reserve > fr) prefill /= 2;
                    if (prefill < 128) prefill = 0;
                    if (prefill != asked)
                        std::fprintf(stderr, "lamina: prompts %s: this GPU's VRAM beside the model holds a context of %lld tokens "
                                     "with no more\n", prefill > 0 ? ("in chunks of " + std::to_string(prefill) + " rows, not " +
                                     std::to_string(asked)).c_str() : "through the decode's forwards, 8 rows at a time",
                                     (long long) (want / per_pos));
                }
            }
            if (prefill > 0) {
                mmq = std::make_unique<MMQ::Context>();
                if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS || cublasSetStream(blas, st) != CUBLAS_STATUS_SUCCESS) {
                    std::fprintf(stderr, "lamina: cuBLAS could not start\n");
                    throw Fatal{1};
                }
                chunk_alloc(prefill);
                gemm(m->layers[0].router, (int) g.n_expert, (int) g.n_embd, P.nrm, 1, P.probs);   // cuBLAS's start, not a first prompt's
                if (m->in_ram) {
                    nrm_p = halloc<float>((size_t) prefill * N);
                    crow_p = halloc<float>((size_t) kChunkCpu * N);
                    cdst_p = halloc<int32_t>(kChunkCpu);
                    x_rows = dzero<float>((size_t) kChunkCpu * N);
                    x_dst = dzero<int32_t>(kChunkCpu);
                    cacts.resize((size_t) prefill * C::kNativeActBytes);
                    cact_of.assign((size_t) prefill, -1);
                }
            }
        }

        // the visits: convolution state, cells
        conv_floats = (C - 1) * (2 * KV + 2 * N);
        const int64_t nv = g.n_visits();
        conv_all = dzero<float>((size_t) nv * conv_floats);
        win_raw = dzero<float>((size_t) nv * kMaxT * (size_t) (2 * KV + 2 * N));
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

        // the context: what was asked, else what the free VRAM holds of the trained one (the full visits' cells), on
        // this card and on the cell cards
        const uint64_t per_pos = (uint64_t) g.n_full_visits() * 2 * cell_bytes;
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        // (with experts in RAM what is free now is the cells' once every expert has given way: but a chunk's scratch
        // then holds a layer's)
        size_t x_most = 0;
        if (m->in_ram && prefill > 0) {
            for (const Layer& L : m->layers) x_most = std::max(x_most, (size_t) (g.n_expert - L.n_slots) * L.stride + kWeightPad);
            fr -= std::min(fr, x_most);
        }
        {   // (first the decode rows' partial sums, a stretch each of the cells this card may hold: VRAM too)
            const int64_t want = max_ctx > 0 ? max_ctx : (m->trained_ctx > 0 ? m->trained_ctx : 1048576);
            const int64_t cells = std::min<int64_t>(want, (int64_t) ((fr > reserve ? fr - reserve : 0) / per_pos));
            D.n_part = stretches(cells + 2 * 32768) + 2;
            D.part = dzero<float>((size_t) kMaxT * (size_t) g.n_head * (size_t) D.n_part * (D_ + 2));
            CK(cudaMemGetInfo(&fr, &tot));
            fr -= std::min(fr, x_most);
        }
        int64_t fits = (int64_t) ((fr > reserve ? fr - reserve : 0) / per_pos) / kGrow * kGrow;
        grow = grow_cells && lamina::core::vmm_available();
        CK(cudaGetDevice(&main_dev));
        full_of.assign(visits.size(), -1);
        int n_full = 0;
        for (size_t v = 0; v < visits.size(); ++v)
            if (!visits[v].cells.ring) full_of[v] = n_full++;
        {   // a unit: positions whose K (and V) are whole mapping chunks
            const uint64_t G = grow ? lamina::core::vmm_granularity() : 1;
            unit = (int64_t) (std::lcm<uint64_t>(G, cell_bytes) / cell_bytes);
            unit = (kGrow + unit - 1) / unit * unit;
        }
        if (!cell_devs.empty() && !grow) {
            std::fprintf(stderr, "lamina: --cell-gpus left out: cells move between cards only where they are mapped as the "
                         "context grows (--kv-grow, CUDA virtual memory)\n");
            cell_devs.clear();
        }
        if (main_limit > 0) fits = std::min(fits, std::max(3 * unit, main_limit / unit * unit));
        int64_t on_cards = 0;
        for (const int dev : cell_devs) cards_add(dev, per_pos, on_cards);
        CK(cudaSetDevice(main_dev));
        if (!cards.empty() && fits < 3 * unit) {
            std::fprintf(stderr, "lamina: the cell cards are left out: this GPU keeps under %lld cells itself\n",
                         (long long) (3 * unit));
            cards.clear();
            on_cards = 0;
        }
        const int64_t trained = m->trained_ctx > 0 ? m->trained_ctx : 1048576;
        const int64_t all = fits + on_cards;
        if (max_ctx > 0) {
            ctx = max_ctx;
            if (all < ctx)
                std::fprintf(stderr, "lamina: --max-context %lld is past the %lld tokens the free VRAM holds cells for: a "
                             "sequence ends there\n", (long long) ctx, (long long) all);
        } else {
            ctx = std::min(trained, all);
            if (ctx < kContextFloor) {
                std::fprintf(stderr, "lamina: the free VRAM holds the cells of %lld tokens, under the %lld a context that is "
                             "not set starts at (--max-context N sets one)\n", (long long) all, (long long) kContextFloor);
                throw Fatal{1};
            }
        }
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
        if (!cards.empty()) {
            const size_t rows = (size_t) kMaxT * (size_t) g.n_head;
            CK(cudaHostAlloc((void**) &xq_h, rows * D_ * 4, cudaHostAllocPortable));
            CK(cudaHostAlloc((void**) &xs_h, rows * 4, cudaHostAllocPortable));
            own(&Owned::pinned, xq_h);
            own(&Owned::pinned, xs_h);
            D.rpart = dzero<float>(cards.size() * rows * (D_ + 2));
            if (prefill > 0) {   // a chunk's rows (the cards' and the pinned ones for the most a chunk grows to)
                const size_t prow = (size_t) chunk_most() * (size_t) g.n_head;
                CK(cudaHostAlloc((void**) &pq_h, prow * D_ * 4, cudaHostAllocPortable));
                CK(cudaHostAlloc((void**) &ps_h, prow * 4, cudaHostAllocPortable));
                own(&Owned::pinned, pq_h);
                own(&Owned::pinned, ps_h);
                chunk_parts();
                for (Card& c : cards) {
                    CK(cudaSetDevice(c.dev));
                    auto dev_alloc = [&](size_t bytes) {
                        void* p = nullptr;
                        CK(cudaMalloc(&p, bytes));
                        CK(cudaMemset(p, 0, bytes));
                        c.dev_mem.push_back(p);
                        return p;
                    };
                    c.cq = (float*) dev_alloc(prow * D_ * 4);
                    c.cns = (float*) dev_alloc(prow * 4);
                    c.csum = (float*) dev_alloc(prow * (D_ + 2) * 4);
                    c.cpos = (int32_t*) dev_alloc((size_t) chunk_most() * 4);
                    const std::vector<int32_t> away((size_t) chunk_most(), kFar);
                    CK(cudaMemcpy(c.cpos, away.data(), away.size() * 4, cudaMemcpyHostToDevice));
                    void* h = nullptr;
                    CK(cudaHostAlloc(&h, prow * (D_ + 2) * 4, cudaHostAllocPortable));
                    c.host_mem.push_back(h);
                    c.csum_h = (float*) h;
                }
                CK(cudaSetDevice(main_dev));
            }
            for (uint8_t*& h : stage_h) {
                CK(cudaHostAlloc((void**) &h, (size_t) unit * cell_bytes, cudaHostAllocPortable));
                own(&Owned::pinned, h);
            }
        }
        mapped = grow ? 0 : ctx;
        if (!ensure(std::min<int64_t>(ctx, kGrow))) {
            std::fprintf(stderr, "lamina: no VRAM for the first cells\n");
            throw Fatal{1};
        }
        if (m->in_ram) {
            if (std::getenv("LAMINA_INK_POOLBENCH")) pool_bench();
            place();
            std::fprintf(stderr, "lamina: %lld of the %zu experts on the GPU (%.2f GiB), the others the CPU's (%d threads); "
                         "they give way to the cells as the context grows%s\n", (long long) n_held, m->ranked.size(),
                         (double) slot_bytes() / (double) (1ull << 30), pool->workers() + 1,
                         m->n_fixed > 0 ? (", down to the " + std::to_string(m->n_fixed) + " that RAM has no room for").c_str() : "");
        }
        std::string with;
        for (const Card& c : cards)
            with += ", " + std::to_string(c.cap) + " on GPU " + std::to_string(c.dev) + " (" + c.name + ")";
        std::fprintf(stderr, "lamina: a context of %lld tokens (%s cells: %.1f KiB a token over %lld full-attention "
                     "applications, %.0f MiB of window cells), %s%s%s; prompts %d rows a chunk%s\n", (long long) ctx,
                     kv_q8 ? "q8_0" : "FP16", (double) per_pos / 1024.0, (long long) g.n_full_visits(),
                     (double) ring_bytes / 1048576.0, grow ? "mapped as it grows" : "allocated whole",
                     cards.empty() ? "" : (": " + std::to_string(fits) + " on this GPU").c_str(), with.c_str(),
                     prefill > 0 ? prefill : kMaxT,
                     elastic() ? (", " + std::to_string(prefill_max) + " while this GPU's free VRAM holds their buffers").c_str() : "");
    }

    /// A cell card: its stream, the address ranges of the cells it has room for, a decode forward's buffers.  Left out
    /// (with a line) where it has no room for a unit.
    void cards_add(int dev, uint64_t per_pos, int64_t& on_cards) {
        const Geometry& g = m->g;
        const size_t D_ = (size_t) g.head_dim, rows = (size_t) kMaxT * (size_t) g.n_head;
        if (cudaSetDevice(dev) != cudaSuccess) {
            (void) cudaGetLastError();
            std::fprintf(stderr, "lamina: --cell-gpus: GPU %d cannot be used: left out\n", dev);
            return;
        }
        Card c;
        c.dev = dev;
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess) c.name = prop.name;
        bool known = false, desktop = false;
        for (const lamina::core::GpuInfo& gi : lamina::core::list_gpus(true))
            if (gi.ordinal == dev) desktop = lamina::core::gpu_in_desktop_use(gi.bus_id, known);
        const uint64_t keep = (uint64_t) (known && !desktop ? 700 : 1536) << 20;
        // (its own buffers: a decode forward's rows and stretches, a chunk's rows, and room to spare)
        const uint64_t own_bytes = rows * (D_ + 1 + (size_t) stretches(1 << 20) * (D_ + 2) + (D_ + 2)) * 4 +
                                   (uint64_t) std::max(chunk_most(), 0) * (uint64_t) g.n_head * (2 * D_ + 4) * 4 + (64ull << 20);
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        c.cap = fr > keep + own_bytes ? (int64_t) ((fr - keep - own_bytes) / per_pos) / unit * unit : 0;
        if (c.cap < unit || !lamina::core::vmm_available()) {
            std::fprintf(stderr, "lamina: --cell-gpus: GPU %d (%s) has no room for cells: left out\n", dev, c.name.c_str());
            return;
        }
        CK(cudaStreamCreateWithFlags(&c.st, cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&c.ev, cudaEventDisableTiming));
        if (!cell_clocks.empty()) {
            const std::string& spec = cell_clocks[std::min(cards.size(), cell_clocks.size() - 1)];
            std::string why;
            if (spec != "0" && !clocks.add(dev, spec, why))
                std::fprintf(stderr, "lamina: --cell-gpu-clocks %s: %s: GPU %d keeps the driver's clocks\n", spec.c_str(),
                             why.c_str(), dev);
        }
        for (int64_t i = 0; i < g.n_full_visits(); ++i) {
            c.k_vm.push_back(std::make_unique<lamina::core::VmmRange>());
            c.v_vm.push_back(std::make_unique<lamina::core::VmmRange>());
            if (!c.k_vm.back()->reserve((uint64_t) c.cap * cell_bytes) || !c.v_vm.back()->reserve((uint64_t) c.cap * cell_bytes)) {
                std::fprintf(stderr, "lamina: --cell-gpus: GPU %d: the cells' address range could not be reserved\n", dev);
                throw Fatal{1};
            }
        }
        auto dev_alloc = [&](size_t bytes) {
            void* p = nullptr;
            CK(cudaMalloc(&p, bytes));
            CK(cudaMemset(p, 0, bytes));
            c.dev_mem.push_back(p);
            return p;
        };
        auto host_alloc = [&](size_t bytes) {
            void* p = nullptr;
            CK(cudaHostAlloc(&p, bytes, cudaHostAllocPortable));
            c.host_mem.push_back(p);
            return p;
        };
        c.q = (float*) dev_alloc(rows * D_ * 4);
        c.ns = (float*) dev_alloc(rows * 4);
        c.n_part = stretches(c.cap);
        c.part = (float*) dev_alloc(rows * (size_t) c.n_part * (D_ + 2) * 4);
        c.sum = (float*) dev_alloc(rows * (D_ + 2) * 4);
        c.step = (int32_t*) dev_alloc((kMaxT + 4) * 4);
        c.step_h = (int32_t*) host_alloc((kMaxT + 4) * 4);
        c.sum_h = (float*) host_alloc(rows * (D_ + 2) * 4);
        card_empty(c);
        on_cards += c.cap;
        cards.push_back(std::move(c));
    }
    /// A card that scores nothing: its sums say so (s = 0)
    void card_empty(Card& c) {
        const size_t rows = (size_t) kMaxT * (size_t) m->g.n_head, w = (size_t) m->g.head_dim + 2;
        for (size_t r = 0; r < rows; ++r) {
            std::fill(c.sum_h + r * w, c.sum_h + (r + 1) * w, 0.f);
            c.sum_h[r * w] = -FLT_MAX;
        }
    }

    /// The main card's oldest unit of cells to a card: the one whose cells take the least to score (the fewest: the
    /// cards score side by side), of those with room.  False: no card has room, or the main card would keep cells
    /// closer to a row than the bias table reaches.
    bool evict() {
        const Geometry& g = m->g;
        if (cards.empty() || lo + unit + std::max(g.rel_extent, g.n_swa) > pos()) return false;
        int best = -1;
        for (size_t i = 0; i < cards.size(); ++i)
            if (cards[i].cells + unit <= cards[i].cap && (best < 0 || cards[i].cells < cards[(size_t) best].cells)) best = (int) i;
        if (best < 0) return false;
        Card& c = cards[(size_t) best];
        const uint64_t G = lamina::core::vmm_granularity();
        const uint64_t bytes = (uint64_t) unit * cell_bytes;
        const int64_t c0 = (int64_t) ((uint64_t) c.cells * cell_bytes / G), c1 = (int64_t) (((uint64_t) c.cells * cell_bytes + bytes) / G);
        CK(cudaStreamSynchronize(st));
        CK(cudaSetDevice(c.dev));
        auto none = [] { return (lamina::core::VmmChunk) 0; };
        bool ok = true;
        for (size_t i = 0; i < c.k_vm.size() && ok; ++i) ok = c.k_vm[i]->map_range(c0, c1, none) && c.v_vm[i]->map_range(c0, c1, none);
        CK(cudaSetDevice(main_dev));
        if (!ok) {   // (its VRAM went elsewhere since the start: it takes no more)
            c.cap = c.cells;
            return evict();
        }
        const int64_t m0 = (int64_t) ((uint64_t) lo * cell_bytes / G), m1 = (int64_t) (((uint64_t) lo * cell_bytes + bytes) / G);
        // through pinned memory, two arrays under way at a time (a copy between cards' mapped ranges is refused)
        int n = 0;
        for (Visit& V : visits) {
            if (V.cells.ring) continue;
            const size_t fi = (size_t) full_of[(size_t) (&V - visits.data())];
            for (int side = 0; side < 2; ++side, ++n) {
                uint8_t* const via = stage_h[n & 1];
                if (n >= 2) {
                    CK(cudaSetDevice(c.dev));
                    CK(cudaStreamSynchronize(c.st));
                    CK(cudaSetDevice(main_dev));
                }
                CK(cudaMemcpyAsync(via, (side ? V.cells.v : V.cells.k) + (size_t) lo * cell_bytes, bytes, cudaMemcpyDeviceToHost, st));
                CK(cudaStreamSynchronize(st));
                CK(cudaSetDevice(c.dev));
                CK(cudaMemcpyAsync((side ? c.v_vm[fi] : c.k_vm[fi])->base() + (size_t) c.cells * cell_bytes, via, bytes,
                                   cudaMemcpyHostToDevice, c.st));
                CK(cudaSetDevice(main_dev));
            }
        }
        CK(cudaSetDevice(c.dev));
        CK(cudaStreamSynchronize(c.st));
        CK(cudaSetDevice(main_dev));
        for (Visit& V : visits) {
            if (V.cells.ring) continue;
            for (int64_t i = m0; i < m1; ++i) {
                lamina::core::vmm_chunk_free(V.k_vm->unmap(i));
                lamina::core::vmm_chunk_free(V.v_vm->unmap(i));
            }
        }
        unit_card.push_back(best);
        c.cells += unit;
        lo += unit;
        clocks.hold();   // (cells on cards: their clocks' locks, where they have some, until the request ends)
        return true;
    }
    /// A card's slot of unit u: the units of that card before it
    int64_t unit_slot(size_t u) const {
        int64_t n = 0;
        for (size_t i = 0; i < u; ++i) n += unit_card[i] == unit_card[u];
        return n * unit;
    }
    /// `bytes` of one array between a card's range (at `cell`) and pinned or pageable host memory, waited for
    void card_copy(Card& c, size_t fi, int side, int64_t cell, uint8_t* host, size_t bytes, bool to_host) {
        uint8_t* const d = (side ? c.v_vm[fi] : c.k_vm[fi])->base() + (size_t) cell * cell_bytes;
        CK(cudaSetDevice(c.dev));
        if (to_host) CK(cudaMemcpyAsync(host, d, bytes, cudaMemcpyDeviceToHost, c.st));
        else CK(cudaMemcpyAsync(d, host, bytes, cudaMemcpyHostToDevice, c.st));
        CK(cudaStreamSynchronize(c.st));
        CK(cudaSetDevice(main_dev));
    }
    /// The sequence is to continue from `pos`: where the cells within the bias table's reach before it are on cards,
    /// their units come back to the main card, and the cards drop every unit from there on (the main card's cells past
    /// `pos` are stale either way).
    void cards_rewind(int64_t pos) {
        const int64_t reach = std::max(m->g.rel_extent, m->g.n_swa);
        if (lo == 0 || pos - reach >= lo) return;
        const int64_t u0 = std::max<int64_t>(0, pos - reach) / unit;   // the first unit the main card holds after
        CK(cudaStreamSynchronize(st));
        const uint64_t G = lamina::core::vmm_granularity();
        const size_t bytes = (size_t) unit * cell_bytes;
        for (Visit& V : visits) {   // the main card's own cells: all past `pos`
            if (V.cells.ring) continue;
            for (int64_t i = (int64_t) ((uint64_t) lo * cell_bytes / G); i < V.k_vm->chunks() && V.k_vm->mapped(i); ++i) {
                lamina::core::vmm_chunk_free(V.k_vm->unmap(i));
                lamina::core::vmm_chunk_free(V.v_vm->unmap(i));
            }
        }
        const int64_t top = lo;
        lo = u0 * unit;
        mapped = lo;
        auto none = [] { return (lamina::core::VmmChunk) 0; };
        const int64_t c0 = (int64_t) ((uint64_t) lo * cell_bytes / G), c1 = (int64_t) ((uint64_t) top * cell_bytes / G);
        for (Visit& V : visits)
            if (!V.cells.ring && (!V.k_vm->map_range(c0, c1, none) || !V.v_vm->map_range(c0, c1, none))) {
                std::fprintf(stderr, "lamina: no VRAM for the cells a restored state brings back from the cards\n");
                throw Fatal{1};
            }
        mapped = top;
        while ((int64_t) unit_card.size() > u0) {   // the cards' last units, the newest first
            const size_t u = unit_card.size() - 1;
            Card& c = cards[(size_t) unit_card[u]];
            c.cells -= unit;
            if ((int64_t) u * unit < pos)   // (cells the sequence keeps)
                for (Visit& V : visits) {
                    if (V.cells.ring) continue;
                    const size_t fi = (size_t) full_of[(size_t) (&V - visits.data())];
                    for (int side = 0; side < 2; ++side) {
                        card_copy(c, fi, side, c.cells, stage_h[0], bytes, true);
                        CK(cudaMemcpyAsync((side ? V.cells.v : V.cells.k) + (size_t) u * bytes, stage_h[0], bytes,
                                           cudaMemcpyHostToDevice, st));
                        CK(cudaStreamSynchronize(st));
                    }
                }
            if (c.cells == 0) card_empty(c);
            unit_card.pop_back();
        }
    }
    /// The cards hold nothing, the main card's cells start at position 0 again (a new sequence)
    void cards_clear() {
        if (lo == 0) return;
        clocks.release_after(clocks_hold);
        CK(cudaStreamSynchronize(st));
        const uint64_t G = lamina::core::vmm_granularity();
        for (Visit& V : visits) {
            if (V.cells.ring) continue;
            for (int64_t i = (int64_t) ((uint64_t) lo * cell_bytes / G); i < V.k_vm->chunks(); ++i) {
                if (!V.k_vm->mapped(i)) break;
                lamina::core::vmm_chunk_free(V.k_vm->unmap(i));
                lamina::core::vmm_chunk_free(V.v_vm->unmap(i));
            }
        }
        for (Card& c : cards) {
            c.cells = 0;
            card_empty(c);
        }
        unit_card.clear();
        lo = 0;
        mapped = 0;
        if (!ensure(std::min<int64_t>(ctx, kGrow))) {
            std::fprintf(stderr, "lamina: no VRAM for the first cells\n");
            throw Fatal{1};
        }
    }
    /// A decode forward's q and -sink (pinned) to every card that holds cells, its scan of full visit `fi` queued there
    /// and its sums on their way to pinned memory
    void cards_scan(int fi, int T) {
        const Geometry& g = m->g;
        const int H = (int) g.n_head, HK = (int) g.n_head_kv, D_ = (int) g.head_dim;
        for (Card& c : cards) {
            if (c.cells == 0) continue;
            CK(cudaSetDevice(c.dev));
            for (int t = 0; t < kMaxT; ++t) c.step_h[t] = kFar;
            c.step_h[kMaxT] = (int32_t) c.cells;
            c.step_h[kMaxT + 1] = 0;
            c.step_h[kMaxT + 2] = (int32_t) stretches(c.cells);
            c.step_h[kMaxT + 3] = 0;
            CK(cudaMemcpyAsync(c.step, c.step_h, (kMaxT + 4) * 4, cudaMemcpyHostToDevice, c.st));
            CK(cudaMemcpyAsync(c.q, xq_h, (size_t) T * H * D_ * 4, cudaMemcpyHostToDevice, c.st));
            CK(cudaMemcpyAsync(c.ns, xs_h, (size_t) T * H * 4, cudaMemcpyHostToDevice, c.st));
            ik::AttnIn at{};
            at.c.k = c.k_vm[(size_t) fi]->base();
            at.c.v = c.v_vm[(size_t) fi]->base();
            at.c.first = 0;
            at.c.cap = c.cap;
            at.c.ring = 0;
            at.c.q8 = kv_q8 ? 1 : 0;
            at.q = c.q;
            at.bias = c.ns;   // (a bias row of one value: -sink)
            at.pos = c.step;
            at.live = c.step + kMaxT;
            at.dyn = c.step + kMaxT + 1;
            at.part = c.part;
            at.H = H;
            at.HK = HK;
            at.D = D_;
            at.ext = 0;
            at.win = 0;
            at.n_chunks = c.n_part;
            at.canon = 1;
            scan(at, T, c.st);
            ik::JoinIn j{c.part, c.sum, at.dyn, H, D_, c.n_part};
            ik::attn_join_kernel<<<dim3((unsigned) T, (unsigned) H), (unsigned) D_, 0, c.st>>>(j);
            CK(cudaMemcpyAsync(c.sum_h, c.sum, (size_t) T * H * (D_ + 2) * 4, cudaMemcpyDeviceToHost, c.st));
            CK(cudaEventRecord(c.ev, c.st));
            (void) cudaStreamQuery(c.st);   // submit now, not at the next driver call: the cards score side by side
        }
        CK(cudaSetDevice(main_dev));
    }
    /// The same for a chunk of T rows: its q and -sink to the cards, each one's partial sums over its cells queued
    void cards_scan_chunk(int fi, int T) {
        const Geometry& g = m->g;
        const int H = (int) g.n_head, HK = (int) g.n_head_kv, D_ = (int) g.head_dim;
        for (Card& c : cards) {
            if (c.cells == 0) continue;
            CK(cudaSetDevice(c.dev));
            CK(cudaMemcpyAsync(c.cq, pq_h, (size_t) T * H * D_ * 4, cudaMemcpyHostToDevice, c.st));
            CK(cudaMemcpyAsync(c.cns, ps_h, (size_t) T * H * 4, cudaMemcpyHostToDevice, c.st));
            ik::Cells cc{};
            cc.k = c.k_vm[(size_t) fi]->base();
            cc.v = c.v_vm[(size_t) fi]->base();
            cc.cap = c.cap;
            cc.q8 = kv_q8 ? 1 : 0;
            ik::TcIn a{cc, c.cq, nullptr, c.cns, nullptr, c.cpos, nullptr, T, H, HK, 0, 0, (long long) c.cells, c.csum, 1, 0,
                       nullptr};
            tc_launch(a, 1, c.st);
            CK(cudaMemcpyAsync(c.csum_h, c.csum, (size_t) T * H * (D_ + 2) * 4, cudaMemcpyDeviceToHost, c.st));
            CK(cudaEventRecord(c.ev, c.st));
            (void) cudaStreamQuery(c.st);
        }
        CK(cudaSetDevice(main_dev));
    }
    /// Every card's sums are in pinned memory: one spin over their events
    void cards_wait() {
        for (bool all = false; !all;) {
            all = true;
            for (Card& c : cards) {
                if (c.cells == 0) continue;
                CK(cudaSetDevice(c.dev));
                const cudaError_t q = cudaEventQuery(c.ev);
                if (q == cudaErrorNotReady) all = false;
                else CK(q);
            }
            if (!all) lamina::platform::cpu_relax();
        }
        (void) cudaGetLastError();   // (a query that was not ready leaves its code behind)
        CK(cudaSetDevice(main_dev));
    }
    /// A forward with cells on cards, at a full visit: the launches so far run (as a graph segment of their own while
    /// one is captured), the rows' q and -sink are in pinned memory, and the cards start their scans
    void seg_end(int fi, int T) {
        if (capturing) {
            cudaGraph_t graph = nullptr;
            Seg sg;
            CK(cudaStreamEndCapture(st, &graph));
            CK(cudaGraphInstantiate(&sg.exec, graph, 0));
            CK(cudaGraphDestroy(graph));
            own(&Owned::graphs, sg.exec);
            sg.fi = fi;
            cur_segs->push_back(sg);
            CK(cudaGraphLaunch(sg.exec, st));
        }
        CK(cudaStreamSynchronize(st));
        cards_scan(fi, T);
    }
    /// A range's stretches that cover its first `cells` cells (ik::stretch_lo), and the one that holds cell `cell`
    static int stretches(int64_t cells) {
        if (cells <= 4096) return (int) ((cells + 127) / 128);
        if (cells <= 36864) return 32 + (int) ((cells - 4096 + 511) / 512);
        return 96 + (int) ((cells - 36864 + 2047) / 2048);
    }
    static int stretch_of(int64_t cell) { return stretches(cell + 1) - 1; }
    /// A decode forward's scan of a visit's cells on stream `s`: every row's in one launch on the tensor cores
    /// (attn_rows_kernel), each row's sums what they are when it is scored alone
    void scan(const ik::AttnIn& at, int T, cudaStream_t s) {
        const dim3 grid((unsigned) at.n_chunks, (unsigned) at.HK);
        if (kv_q8) ik::attn_rows_kernel<true><<<grid, 256, 0, s>>>(at, T);
        else ik::attn_rows_kernel<false><<<grid, 256, 0, s>>>(at, T);
    }
    /// The cards' sums are in pinned memory: the next segment takes them
    void seg_begin() {
        cards_wait();
        if (capturing) CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    }

    /// Memory for the full visits' cells up to position n: where the main card has none left above its reserve, its
    /// oldest cells move to a card first.  False: neither holds them.
    bool ensure(int64_t n) {
        if (n <= mapped) return true;
        if (n > ctx) return false;
        const int64_t target = std::min<int64_t>(ctx, (n + kGrow - 1) / kGrow * kGrow);
        const uint64_t G = lamina::core::vmm_granularity();
        auto chunks = [&](int64_t cells) { return (int64_t) (((uint64_t) cells * cell_bytes + G - 1) / G); };
        int64_t ranges = 0;
        for (const Visit& V : visits) ranges += V.k_vm ? 2 : 0;
        const int64_t need = (chunks(target) - chunks(mapped)) * ranges;
        for (;;) {
            size_t fr = 0, tot = 0;
            CK(cudaMemGetInfo(&fr, &tot));
            const int64_t have = (int64_t) spare.size() + (fr > reserve ? (int64_t) ((fr - reserve) / G) : 0);
            const bool room = need <= have;
            if (room && (main_limit <= 0 || target - lo <= std::max(3 * unit, main_limit))) break;
            if (!evict()) {
                if (room) break;   // (the test's limit alone: the card has the VRAM)
                if (m->in_ram && give_way(need - have)) continue;   // (the experts the ranking puts last)
                return false;
            }
        }
        // cells on cards (or a sequence past --cell-spread): this card keeps no more than they do, a few units a step
        if (!cards.empty() && (lo > 0 || (spread > 0 && n > spread)))
            for (int i = 0; i < kBalance; ++i) {
                int64_t least = INT64_MAX;
                for (const Card& c : cards)
                    if (c.cells + unit <= c.cap) least = std::min(least, c.cells);
                if (least == INT64_MAX || pos() - lo - unit < least + unit || !evict()) break;
            }
        CK(cudaStreamSynchronize(st));
        auto take = [&] { return take_spare(); };
        const int64_t from = (int64_t) ((uint64_t) lo * cell_bytes / G);
        for (Visit& V : visits)
            if (V.k_vm && (!V.k_vm->map_range(from, chunks(target), take) || !V.v_vm->map_range(from, chunks(target), take)))
                return false;
        mapped = target;
        return true;
    }

    // ---- experts in RAM: the slots
    lamina::core::VmmChunk take_spare() {
        if (spare.empty()) return 0;
        const lamina::core::VmmChunk h = spare.back();
        spare.pop_back();
        return h;
    }
    /// The slots' VRAM: the chunks mapped
    uint64_t slot_bytes() const {
        uint64_t n = 0;
        for (size_t l = 0; l < on_card.size(); ++l) n += (uint64_t) slot_chunks(m->layers[l], on_card[l]);
        return n * lamina::core::vmm_granularity();
    }
    /// Layer l's slots of the card as the device reads them
    void slots_say(size_t l) { CK(cudaMemcpy(m->layers[l].slot_of, slot_at[l].data(), slot_at[l].size() * 4, cudaMemcpyHostToDevice)); }
    /// Layer l keeps `to` of its slots: the coldest of its experts that RAM has leave, and one that stays moves out of
    /// a slot past them.  The moves are queued: the chunks go once the stream has run them (layer_chunks).
    void layer_shrink(size_t l, int to) {
        const Layer& L = m->layers[l];
        std::vector<int32_t> out;
        for (const int32_t e : slot_exp[l])
            if (L.blob_h[(size_t) e] != nullptr) out.push_back(e);
        const size_t drop = (size_t) (on_card[l] - to);
        std::partial_sort(out.begin(), out.begin() + (std::ptrdiff_t) drop, out.end(),
                          [&](int32_t a, int32_t b) { return heat[l][(size_t) a] < heat[l][(size_t) b]; });
        std::vector<int32_t> holes;   // the slots that stay which they leave
        for (size_t i = 0; i < drop; ++i) {
            const int32_t k = slot_at[l][(size_t) out[i]];
            slot_at[l][(size_t) out[i]] = -1;
            if (k < to) holes.push_back(k);
        }
        for (int k = to; k < on_card[l]; ++k) {
            const int32_t e = slot_exp[l][(size_t) k];
            if (slot_at[l][(size_t) e] < 0) continue;
            const int32_t h = holes.back();
            holes.pop_back();
            CK(cudaMemcpyAsync(L.slots + (size_t) h * L.stride, L.slots + (size_t) k * L.stride, L.stride,
                               cudaMemcpyDeviceToDevice, st));
            slot_exp[l][(size_t) h] = e;
            slot_at[l][(size_t) e] = h;
        }
        slot_exp[l].resize((size_t) to);
    }
    /// Layer l's range is mapped as far as `to` slots reach (the chunks past them spare), and the device reads its
    /// slots as they are now.  The stream is idle.
    void layer_chunks(size_t l, int to) {
        const Layer& L = m->layers[l];
        for (int64_t c = slot_chunks(L, to); c < L.slot_vm->chunks(); ++c)
            if (L.slot_vm->mapped(c)) spare.push_back(L.slot_vm->unmap(c));
        n_held += to - on_card[l];
        on_card[l] = to;
        slots_say(l);
    }
    /// Layer l takes slots up to `to` (its range is mapped that far): the hottest of its experts off the card, their
    /// blobs from RAM, those in a row there in one copy
    void layer_grow(size_t l, int to) {
        const Layer& L = m->layers[l];
        std::vector<int32_t> in;
        for (const int32_t e : L.order)
            if (slot_at[l][(size_t) e] < 0) in.push_back(e);
        const size_t n = (size_t) (to - on_card[l]);
        std::stable_sort(in.begin(), in.end(), [&](int32_t a, int32_t b) { return heat[l][(size_t) a] > heat[l][(size_t) b]; });
        for (size_t i = 0; i < n;) {
            size_t j = i + 1;
            while (j < n && L.blob_h[(size_t) in[j]] == L.blob_h[(size_t) in[j - 1]] + L.stride) ++j;
            CK(cudaMemcpyAsync(L.slots + ((size_t) on_card[l] + i) * L.stride, L.blob_h[(size_t) in[i]], (j - i) * L.stride,
                               cudaMemcpyHostToDevice, st));
            i = j;
        }
        for (size_t i = 0; i < n; ++i) {
            slot_at[l][(size_t) in[i]] = on_card[l] + (int32_t) i;
            slot_exp[l].push_back(in[i]);
        }
        // (what is mapped past the slots: zeros, a kernel may read past a short last row)
        CK(cudaMemsetAsync(L.slots + (size_t) to * L.stride, 0,
                           (size_t) slot_chunks(L, to) * lamina::core::vmm_granularity() - (size_t) to * L.stride, st));
    }
    /// The experts RAM has, as (heat, layer * experts + expert): those on the card, or those off it
    std::vector<std::pair<float, int32_t>> by_heat(bool there) const {
        const int32_t E = (int32_t) m->g.n_expert;
        std::vector<std::pair<float, int32_t>> v;
        for (size_t l = 0; l < on_card.size(); ++l)
            for (int32_t e = 0; e < E; ++e)
                if (m->layers[l].blob_h[(size_t) e] != nullptr && (slot_at[l][(size_t) e] >= 0) == there)
                    v.emplace_back(heat[l][(size_t) e], (int32_t) l * E + e);
        return v;
    }
    /// The coldest experts on the card leave it, whatever their layers, until `chunks` more chunks are spare.  False,
    /// and nothing changes: the ones left are in VRAM alone.
    bool give_way(int64_t chunks) {
        const int32_t E = (int32_t) m->g.n_expert;
        std::vector<std::pair<float, int32_t>> c = by_heat(true);
        std::sort(c.begin(), c.end());
        std::vector<int> to = on_card;
        int64_t got = 0;
        for (size_t i = 0; i < c.size() && got < chunks; ++i) {
            const size_t l = (size_t) (c[i].second / E);
            got += slot_chunks(m->layers[l], to[l]) - slot_chunks(m->layers[l], to[l] - 1);
            --to[l];
        }
        if (got < chunks) return false;
        for (size_t l = 0; l < on_card.size(); ++l)
            if (to[l] < on_card[l]) layer_shrink(l, to[l]);
        CK(cudaStreamSynchronize(st));   // (the forwards before, and the moves: a chunk is unmapped once nothing reads it)
        for (size_t l = 0; l < on_card.size(); ++l)
            if (to[l] < on_card[l]) layer_chunks(l, to[l]);
        return true;
    }
    /// The card has the hottest experts its VRAM holds: those its ranges have mapped, the spare chunks and the VRAM
    /// free above the reserve (--expert-cache at most).  At the start it fills the card; after a request the chunks
    /// the cells gave back take experts again, and slots move from a layer's cold experts to another's hotter ones
    /// (one on the card counts kSwapLead times its heat: it stays unless clearly colder).
    void place() {
        if (!m->in_ram) return;
        CK(cudaStreamSynchronize(st));
        const uint64_t G = lamina::core::vmm_granularity();
        const int32_t E = (int32_t) m->g.n_expert;
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        int64_t budget = (int64_t) (slot_bytes() / G) + (int64_t) spare.size() + (fr > reserve ? (int64_t) ((fr - reserve) / G) : 0);
        if (m->expert_cap > 0) budget = std::min(budget, (int64_t) (m->expert_cap / G));
        // each layer's count: the ones in VRAM alone, then the hottest of the others while their chunks fit
        std::vector<std::pair<float, int32_t>> c = by_heat(true);
        for (auto& x : c) x.first *= kSwapLead;
        {
            const std::vector<std::pair<float, int32_t>> off = by_heat(false);
            c.insert(c.end(), off.begin(), off.end());
        }
        std::stable_sort(c.begin(), c.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        std::vector<int> to(on_card.size());
        for (size_t l = 0; l < on_card.size(); ++l) {
            to[l] = m->layers[l].n_slots;
            budget -= slot_chunks(m->layers[l], to[l]);
        }
        for (const auto& x : c) {
            const size_t l = (size_t) (x.second / E);
            const int64_t more = slot_chunks(m->layers[l], to[l] + 1) - slot_chunks(m->layers[l], to[l]);
            if (more > budget) break;
            budget -= more;
            ++to[l];
        }
        // the layers that give slots first; then the others' chunks (a layer that does not get its own - the VRAM
        // went elsewhere meanwhile - keeps what it has) and their experts
        bool any = false;
        for (size_t l = 0; l < on_card.size(); ++l)
            if (to[l] < on_card[l]) {
                layer_shrink(l, to[l]);
                any = true;
            }
        if (any) CK(cudaStreamSynchronize(st));
        for (size_t l = 0; l < on_card.size(); ++l)
            if (to[l] < on_card[l]) layer_chunks(l, to[l]);
        for (size_t l = 0; l < on_card.size(); ++l) {
            if (to[l] <= on_card[l]) continue;
            const Layer& L = m->layers[l];
            if (!L.slot_vm->map_range(0, slot_chunks(L, to[l]), [&] { return take_spare(); })) {
                layer_chunks(l, on_card[l]);
                continue;
            }
            layer_grow(l, to[l]);
            any = true;
        }
        if (!any) return;
        CK(cudaStreamSynchronize(st));
        for (size_t l = 0; l < on_card.size(); ++l)
            if (to[l] > on_card[l] && (int) slot_exp[l].size() == to[l]) layer_chunks(l, to[l]);
    }
    /// Between forwards: where a layer's hottest expert off the card leads its coldest one on it (of those RAM has), the
    /// two change places - the slot is no one's while the blob is on its way.  The layers with the largest leads first.
    void adapt() {
        heat_w /= kHeatKeep;   // (a forward: the entries before it fade)
        if (heat_w > 1e18f) {
            for (std::vector<float>& h : heat)
                for (float& x : h) x /= heat_w;
            heat_w = 1.f;
        }
        struct Swap {
            float lead;
            int l, out, in;
        } best[kSwaps];
        int n = 0;
        const int E = (int) m->g.n_expert;
        for (size_t l = 0; l < on_card.size(); ++l) {
            if (on_card[l] == 0 || on_card[l] == E) continue;
            const Layer& L = m->layers[l];
            const float* const h = heat[l].data();
            int out = -1, in = -1;
            for (int e = 0; e < E; ++e) {
                if (slot_at[l][(size_t) e] < 0) {
                    if (in < 0 || h[e] > h[in]) in = e;
                } else if (L.blob_h[(size_t) e] != nullptr && (out < 0 || h[e] < h[out])) {
                    out = e;
                }
            }
            if (in < 0 || out < 0 || h[in] <= h[out] * kSwapLead + 0.5f * heat_w) continue;
            const Swap sw{h[in] - h[out], (int) l, out, in};
            int at = n;
            if (n == kSwaps) {
                if (best[n - 1].lead >= sw.lead) continue;
                at = n - 1;
            } else {
                ++n;
            }
            best[at] = sw;
            for (; at > 0 && best[at].lead > best[at - 1].lead; --at) std::swap(best[at], best[at - 1]);
        }
        for (int i = 0; i < n; ++i) {
            const size_t l = (size_t) best[i].l;
            const Layer& L = m->layers[l];
            const int32_t slot = slot_at[l][(size_t) best[i].out];
            swap_h[2 * i] = -1;
            swap_h[2 * i + 1] = slot;
            CK(cudaMemcpyAsync(L.slot_of + best[i].out, swap_h + 2 * i, 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(L.slots + (size_t) slot * L.stride, L.blob_h[(size_t) best[i].in], L.stride, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(L.slot_of + best[i].in, swap_h + 2 * i + 1, 4, cudaMemcpyHostToDevice, st));
            slot_at[l][(size_t) best[i].out] = -1;
            slot_at[l][(size_t) best[i].in] = slot;
            slot_exp[l][(size_t) slot] = best[i].in;
            ++swaps;
        }
    }
    /// The ranking the heat makes, written to --expert-profile-save: at a request's end once the time has passed (`now`:
    /// whenever routing was counted).  A pair's score is its share of the heat plus its share by the ranking the start
    /// had (1 / (its place + 1) over the places' sum: as Flash-Next's learned profile blends them, the experts routed
    /// in general stay near the front after a run on one subject).
    void profile_write(bool now) {
        if (profile_save.empty() || !profile_new || !m->in_ram) return;
        if (!now && std::chrono::duration<double>(Clock::now() - profile_at).count() < profile_every) return;
        const size_t nl = m->layers.size(), E = (size_t) m->g.n_expert, R = m->ranked.size();
        double all = 0, harmonic = 0;
        for (const std::vector<float>& h : heat)
            for (const float x : h) all += x;
        for (size_t r = 0; r < R; ++r) harmonic += 1.0 / (double) (r + 1);
        std::vector<std::pair<double, int32_t>> score(R);   // (score, the start's place)
        for (size_t r = 0; r < R; ++r) {
            const auto [l, e] = m->ranked[r];
            score[r] = {(all > 0 ? (double) heat[(size_t) l][(size_t) e] / all : 0.0) + 1.0 / (double) (r + 1) / harmonic, (int32_t) r};
        }
        std::stable_sort(score.begin(), score.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        std::vector<std::pair<int32_t, int32_t>> ranked;
        for (const auto& x : score) ranked.push_back(m->ranked[(size_t) x.second]);
        std::string err;
        if (!lamina::core::write_expert_profile(profile_save, (int64_t) nl, (int64_t) E, ranked, err))
            std::fprintf(stderr, "lamina: the experts' ranking was not saved: %s\n", err.c_str());
        profile_at = Clock::now();
        profile_new = false;
    }
    /// The start's rates for a chunk's split: an upload's time a MB (64 MB of the RAM experts to the card), and each
    /// layer's CPU job of one row and of kMaxT (a few jobs of experts in RAM, random rows)
    void rates() {
        const Geometry& g = m->g;
        const int N = (int) g.n_embd;
        cpu_job_ms.assign(m->layers.size(), 0.12f);
        cpu_row_ms.assign(m->layers.size(), 0.035f);
        if (m->arena_bytes >= ((size_t) 64 << 20)) {
            void* d = nullptr;
            if (cudaMalloc(&d, (size_t) 64 << 20) == cudaSuccess) {
                up_ms_mb = 1e9;
                for (int i = 0; i < 4; ++i) {   // (the first: the driver's own start)
                    const auto t0 = Clock::now();
                    CK(cudaMemcpy(d, m->arena, (size_t) 64 << 20, cudaMemcpyHostToDevice));
                    if (i > 0) up_ms_mb = std::min(up_ms_mb, ms_since(t0) / 64.0);
                }
                cudaFree(d);
            }
            (void) cudaGetLastError();
        }
        std::vector<float> x((size_t) kMaxT * N), out((size_t) 8 * kMaxT * N);
        std::mt19937 rng(1);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (float& v : x) v = nd(rng);
        for (size_t li = 0; li < m->layers.size(); ++li) {
            const Layer& L = m->layers[li];
            std::vector<const uint8_t*> blobs;
            for (const uint8_t* b : L.blob_h)
                if (b != nullptr) blobs.push_back(b);
            if (blobs.size() < 32) continue;
            for (int t = 0; t < kMaxT; ++t) C::native_quant_act(L.fmt, x.data() + (size_t) t * N, acts.data() + (size_t) t * C::kNativeActBytes);
            double ms[2] = {1e9, 1e9};   // (the best of three sets of other experts each)
            for (int pass = 0; pass < 6; ++pass) {
                const int nt = pass & 1 ? kMaxT : 1;
                jobs.clear();
                for (int e = 0; e < 8; ++e) {
                    C::ExpertJobMulti j;
                    j.blob = blobs[(size_t) ((pass * 8 + e) * 7) % blobs.size()];
                    for (int t = 0; t < nt; ++t) {
                        j.nact[t] = acts.data() + (size_t) t * C::kNativeActBytes;
                        j.out[t] = out.data() + ((size_t) e * kMaxT + (size_t) t) * N;
                    }
                    j.nt = nt;
                    jobs.push_back(j);
                }
                const auto t0 = Clock::now();
                pool->run_split_multi_native(L.fmt, jobs.data(), 8);
                ms[pass & 1] = std::min(ms[pass & 1], ms_since(t0) / 8.0);
            }
            cpu_job_ms[li] = (float) ms[0];
            cpu_row_ms[li] = (float) std::max(0.0, (ms[1] - ms[0]) / (kMaxT - 1));
        }
        std::fprintf(stderr, "lamina: a RAM expert's upload %.2f ms (%.1f GB/s), the CPU's job of it %.3f ms and %.3f ms a row "
                     "more (layer 0)\n", up_ms_mb * (double) m->layers[0].stride / 1e6, 1.0 / up_ms_mb, cpu_job_ms[0], cpu_row_ms[0]);
    }
    /// LAMINA_INK_POOLBENCH=1: the CPU pool on a layer's experts in RAM, n experts of t rows each at a time (random
    /// rows): the time a set takes, an entry's share of it, the blobs' bytes a second
    void pool_bench() {
        const Geometry& g = m->g;
        const int N = (int) g.n_embd;
        std::vector<float> x((size_t) kMaxT * N), out((size_t) 64 * kMaxT * N);
        std::mt19937 rng(1);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (float& v : x) v = nd(rng);
        for (const size_t li : {(size_t) 0, m->layers.size() - 1}) {
            const Layer& L = m->layers[li];
            std::vector<const uint8_t*> blobs;
            for (const uint8_t* b : L.blob_h)
                if (b != nullptr) blobs.push_back(b);
            if (blobs.size() < 64) continue;
            for (int t = 0; t < kMaxT; ++t) C::native_quant_act(L.fmt, x.data() + (size_t) t * N, acts.data() + (size_t) t * C::kNativeActBytes);
            for (const int ne : {1, 2, 4, 8, 16, 64})
                for (const int nt : {1, 2, 4, 8}) {
                    const int reps = std::max(2, 256 / (ne * nt));
                    size_t at = 0;
                    const auto t0 = Clock::now();
                    for (int r = 0; r < reps; ++r) {
                        jobs.clear();
                        for (int e = 0; e < ne; ++e) {
                            C::ExpertJobMulti j;
                            j.blob = blobs[at++ % blobs.size()];   // (other experts each time: none in a cache)
                            for (int t = 0; t < nt; ++t) {
                                j.nact[t] = acts.data() + (size_t) t * C::kNativeActBytes;
                                j.out[t] = out.data() + ((size_t) e * kMaxT + (size_t) t) * N;
                            }
                            j.nt = nt;
                            jobs.push_back(j);
                        }
                        pool->run_split_multi_native(L.fmt, jobs.data(), ne);
                    }
                    const double ms = ms_since(t0) / reps;
                    std::fprintf(stderr, "lamina: pool, layer %zu (types %d/%d, %.2f MB an expert): %2d experts x %d rows %7.3f ms, "
                                 "%6.1f us an entry, %5.1f GB/s\n", li, L.gl.gu_type, L.gl.d_type, (double) L.stride / 1e6, ne, nt,
                                 ms, 1000.0 * ms / (ne * nt), (double) ne * (double) L.stride / 1e6 / ms);
                }
        }
    }
    /// `need` more chunks can be mapped: spare ones, VRAM free above the reserve, else the experts give way
    bool chunks_room(int64_t need) {
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        const uint64_t G = lamina::core::vmm_granularity();
        const int64_t have = (int64_t) spare.size() + (fr > reserve ? (int64_t) ((fr - reserve) / G) : 0);
        return have >= need || give_way(need - have);
    }
    /// A chunk's scratch holds the RAM experts of any one layer
    bool scratch_fit() {
        const uint64_t G = lamina::core::vmm_granularity();
        for (;;) {
            const int64_t want = scratch_chunks();
            if (want <= x_chunks) return true;
            const int64_t before = n_held;
            if (!chunks_room(want - x_chunks)) return false;
            if (n_held != before) continue;   // (the experts gave way: a layer may have more in RAM now)
            if (!x_vm->map_range(0, want, [&] { return take_spare(); })) return false;
            CK(cudaMemset(x_vm->base() + (size_t) x_chunks * G, 0, (size_t) (want - x_chunks) * G));
            x_chunks = want;
            return true;
        }
    }
    int64_t scratch_chunks() const {
        const uint64_t G = lamina::core::vmm_granularity();
        uint64_t most = 0;
        for (size_t l = 0; l < on_card.size(); ++l)
            most = std::max<uint64_t>(most, (uint64_t) (m->g.n_expert - on_card[l]) * m->layers[l].stride);
        return most ? (int64_t) ((most + kWeightPad + G - 1) / G) : 0;
    }
    /// After a request: the cells' chunks past what the state needs (two steps of room) and the scratch's past a
    /// layer's RAM experts go back to the experts; chunks none takes are freed
    void relax() {
        route_dump(2, 0, 0, nullptr);
        if (route_f != nullptr) std::fflush(route_f);
        if (!m->in_ram) return;
        const uint64_t G = lamina::core::vmm_granularity();
        CK(cudaStreamSynchronize(st));
        if (grow && lo == 0) {
            const int64_t target = std::min<int64_t>(ctx, (pos() + 3 * kGrow - 1) / kGrow * kGrow);
            auto chunks = [&](int64_t cells) { return (int64_t) (((uint64_t) cells * cell_bytes + G - 1) / G); };
            if (mapped > target) {
                for (Visit& V : visits) {
                    if (!V.k_vm) continue;
                    for (int64_t c = chunks(target); c < chunks(mapped); ++c) {
                        if (V.k_vm->mapped(c)) spare.push_back(V.k_vm->unmap(c));
                        if (V.v_vm->mapped(c)) spare.push_back(V.v_vm->unmap(c));
                    }
                }
                mapped = target;
            }
        }
        for (int pass = 0; pass < 2; ++pass) {   // (the experts that come back leave the scratch less to hold)
            place();
            for (const int64_t want = scratch_chunks(); x_chunks > want; --x_chunks)
                if (x_vm->mapped(x_chunks - 1)) spare.push_back(x_vm->unmap(x_chunks - 1));
            if (spare.empty()) break;
        }
        for (const lamina::core::VmmChunk h : spare) lamina::core::vmm_chunk_free(h);
        spare.clear();
        profile_write(false);
    }
    /// What the CPU took of the decode forwards' routed entries since the last call, for a request's line
    std::string ram_summary() {
        if (!m->in_ram) return {};
        char b[320];
        std::snprintf(b, sizeof b, "; %lld of %zu experts on the GPU, the CPU's %.1f%% of %lld entries (%.1f ms), %lld swaps; "
                      "the chunks' %lld RAM entries: %.1f%% the CPU's, %lld experts sent",
                      (long long) n_held, m->ranked.size(), ex_entries ? 100.0 * (double) ex_ram / (double) ex_entries : 0.0,
                      ex_entries, ex_ms, swaps, ck_entries, ck_entries ? 100.0 * (double) ck_cpu / (double) ck_entries : 0.0, ck_sent);
        ex_entries = ex_ram = swaps = ck_entries = ck_cpu = ck_sent = 0;
        ex_ms = 0;
        return b;
    }
    /// A decode forward with experts in RAM, behind visit v's experts: the launches so far run (as a graph segment of
    /// their own while one is captured), and the host computes the entries the card has no slot for
    void moe_cut(int v, int T) {
        if (capturing) {
            cudaGraph_t graph = nullptr;
            Seg sg;
            CK(cudaStreamEndCapture(st, &graph));
            CK(cudaGraphInstantiate(&sg.exec, graph, 0));
            CK(cudaGraphDestroy(graph));
            own(&Owned::graphs, sg.exec);
            sg.moe = v;
            cur_segs->push_back(sg);
            *bell_h = -1;
            CK(cudaGraphLaunch(sg.exec, st));
        }
        moe_host(v, T);
        if (capturing) CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    }
    /// The bell's value once the copy behind a visit's routing has brought it (bell_h was -1 when that was queued)
    int bell_wait() {
        (void) cudaStreamQuery(st);   // (submitted now, not at the next call into the driver)
        const volatile int32_t* const bell = bell_h;
        for (unsigned spins = 1; *bell < 0; ++spins) {
            lamina::platform::cpu_relax();
            if ((spins & 0xfffff) != 0) continue;
            const cudaError_t q = cudaStreamQuery(st);   // (a stream that stopped says why: no wait without an end)
            if (q == cudaErrorNotReady) {
                (void) cudaGetLastError();
                continue;
            }
            CK(q);
            if (*bell < 0) {
                std::fprintf(stderr, "lamina: a forward's routing did not reach the host\n");
                throw Fatal{1};
            }
        }
        return *bell;
    }
    /// Visit v's routed entries of T rows whose experts the card does not have: once the routing is in pinned memory
    /// (the bell), their rows by the CPU pool - the card computes the others meanwhile - and to the card, packed
    void moe_host(int v, int T) {
        const Geometry& g = m->g;
        const size_t li = (size_t) visits[(size_t) v].layer;
        const Layer& L = m->layers[li];
        const int N = (int) g.n_embd, top_k = (int) g.top_k, n_ent = T * top_k;
        const int n_miss = bell_wait();
        ex_entries += n_ent;
        route_dump(1, (int) li, n_ent, moe_h);
        for (int e = 0; e < n_ent; ++e) heat[li][(size_t) moe_h[e]] += heat_w;
        profile_new = true;
        if (n_miss == 0) return;
        const auto t0 = Clock::now();
        const int32_t* const ids = moe_h;
        const float* const nrm = (const float*) (moe_h + (size_t) kMaxT * (size_t) top_k);
        jobs.clear();
        int job_of[ik::kExpertMax];
        std::fill(job_of, job_of + g.n_expert, -1);
        bool quantized[kMaxT] = {};
        int k = 0;
        for (int e = 0; e < n_ent; ++e) {
            const int ex = ids[e], t = e / top_k;
            if (slot_at[li][(size_t) ex] >= 0) continue;
            uint8_t* const act = acts.data() + (size_t) t * C::kNativeActBytes;
            if (!quantized[t]) {
                C::native_quant_act(L.fmt, nrm + (size_t) t * N, act);
                quantized[t] = true;
            }
            if (job_of[ex] < 0) {
                C::ExpertJobMulti j;
                j.blob = L.blob_h[(size_t) ex];
                job_of[ex] = (int) jobs.size();
                jobs.push_back(j);
            }
            C::ExpertJobMulti& j = jobs[(size_t) job_of[ex]];
            j.nact[j.nt] = act;
            j.out[j.nt] = rows_h + (size_t) k * N;
            ++j.nt;
            ++k;
        }
        if (k != n_miss) {
            std::fprintf(stderr, "lamina: the card counts %d routed entries without a slot, the host %d\n", n_miss, k);
            throw Fatal{1};
        }
        pool->run_split_multi_native(L.fmt, jobs.data(), (int) jobs.size());
        CK(cudaMemcpyAsync(D.crows, rows_h, (size_t) k * N * 4, cudaMemcpyHostToDevice, st));
        ex_ram += k;
        ex_ms += ms_since(t0);
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
    // LAMINA_INK_PROF=2: an event behind each launch, read once the forward is through (prof_collect): the device's own
    // times, nothing waited for in between
    std::vector<cudaEvent_t> prof_ev;
    std::vector<const char*> prof_name;
    size_t prof_n = 0;
    void prof_mark(const char* what) {
        if (prof_n == prof_ev.size()) {
            cudaEvent_t e = nullptr;
            CK(cudaEventCreate(&e));
            prof_ev.push_back(e);
            prof_name.push_back(what);
        }
        prof_name[prof_n] = what;
        CK(cudaEventRecord(prof_ev[prof_n++], st));
    }
    void prof_start() {
        if (!prof) return;
        prof_t = Clock::now();
        if (prof_level >= 2) {
            prof_n = 0;
            prof_mark("");
        }
    }
    void prof_collect() {
        if (prof_level < 2 || prof_n == 0) return;
        CK(cudaStreamSynchronize(st));
        for (size_t i = 1; i < prof_n; ++i) {
            float ms = 0.f;
            CK(cudaEventElapsedTime(&ms, prof_ev[i - 1], prof_ev[i]));
            prof_ms[prof_name[i]] += (double) ms;
        }
        prof_n = 0;
    }
    // LAMINA_INK_DUP="name:n": the launch of that name issued n more times (its cost in the graph, by difference)
    std::string dup_name;
    int dup_n = 0;
    /// A named launch (LAMINA_INK_PROF=2 times it)
    template <class F> void run(const char* what, F&& launch) {
        launch();
        if (dup_n > 0 && dup_name == what)
            for (int i = 0; i < dup_n; ++i) launch();
        if (prof_level >= 2) prof_mark(what);
    }
    /// A step's launches are queued
    void tick(const char* what) {
        if (!prof) return;
        if (prof_level >= 2) {
            prof_mark(what);
            return;
        }
        CK(cudaStreamSynchronize(st));
        const auto now = Clock::now();
        prof_ms[what] += std::chrono::duration<double, std::milli>(now - prof_t).count();
        prof_t = now;
    }
    /// A row kernel's threads: one a four values of the stream's row
    unsigned row4() const { return (unsigned) ((m->g.n_embd / 4 + 31) / 32 * 32); }
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
        if (W.type == 30) {   // BF16: its floats, then a GEMM
            const long long n = (long long) W.n_out * W.n_in;
            ik::bf16_floats_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, st>>>((const uint16_t*) W.d, R.wf, n);
            gemm(R.wf, W.n_out, W.n_in, xf, T, y);
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
        ik::mix_kernel<<<dim3((unsigned) T, (unsigned) std::max(nq, 1)), row4(), 0, st>>>(a);
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
        // The experts in the launches' order: the layer's slots (every expert's own where the card holds them all),
        // then, with experts in RAM, those of the others this chunk routes, their blobs sent to the scratch - slots in
        // a row in one copy (their blobs are in a row in RAM too).
        route_dump(0, L.il, n_ent, R.ids_h);
        const int ns = m->in_ram ? on_card[(size_t) L.il] : E;
        const int32_t* const at_of = m->in_ram ? slot_at[(size_t) L.il].data() : L.slot_at.data();
        int rows_of[ik::kExpertMax] = {}, place[ik::kExpertMax];
        for (int i = 0; i < n_ent; ++i) ++rows_of[R.ids_h[i]];
        for (int e = 0; e < E; ++e) place[e] = at_of[e];
        // The CPU's of the RAM experts this chunk routes: those with the fewest rows, while their jobs take it no
        // longer than the uploads of the others take the card.  The MoE's input rows go to the host for them.
        bool cpus[ik::kExpertMax] = {};
        int n_cpu = 0;   // their entries
        if (ns < E) {
            int cand[ik::kExpertMax], nc = 0;
            double t_up = 0;
            const double up = up_ms_mb * (double) L.stride / 1e6;
            for (int e = 0; e < E; ++e)
                if (place[e] < 0 && rows_of[e] > 0) {
                    cand[nc++] = e;
                    t_up += up;
                    ck_entries += rows_of[e];
                }
            std::sort(cand, cand + nc, [&](int a, int b) { return rows_of[a] < rows_of[b]; });
            double t_cpu = 0;
            for (int i = 0; i < nc; ++i) {
                const int r = rows_of[cand[i]], nj = (r + kMaxT - 1) / kMaxT;
                const double c = (double) cpu_job_ms[(size_t) L.il] * nj + (double) cpu_row_ms[(size_t) L.il] * (r - nj);
                if (n_cpu + r > kChunkCpu || t_cpu + c > t_up - up) break;
                t_cpu += c;
                t_up -= up;
                cpus[cand[i]] = true;
                n_cpu += r;
            }
            ck_cpu += n_cpu;
            if (n_cpu > 0) {
                *bell_h = -1;
                CK(cudaMemcpyAsync(nrm_p, R.nrm, (size_t) T * N * 4, cudaMemcpyDeviceToHost, st));
                CK(cudaMemcpyAsync(bell_h, R.ids, 4, cudaMemcpyDeviceToHost, st));
            }
        }
        int n_place = ns;
        uint8_t* const xs = x_vm ? x_vm->base() : nullptr;
        auto sent = [&](int k) {
            const int e = k < E ? L.order[(size_t) k] : 0;
            return k < E && place[e] < 0 && rows_of[e] > 0 && !cpus[e];
        };
        for (int k = L.n_slots; ns < E && k < E;) {   // (the order's experts after those in VRAM alone: their blobs' order)
            if (!sent(k)) {
                ++k;
                continue;
            }
            const int k0 = k, p0 = n_place;
            for (; sent(k); ++k) place[L.order[(size_t) k]] = n_place++;
            CK(cudaMemcpyAsync(xs + (size_t) (p0 - ns) * L.stride, L.blob_h[(size_t) L.order[(size_t) k0]],
                               (size_t) (k - k0) * L.stride, cudaMemcpyHostToDevice, st));
            ck_sent += k - k0;
        }
        const int n_gpu = n_place;   // the places the card's products take; the CPU's experts after them (no launch reads those)
        for (int e = 0; e < E; ++e)
            if (cpus[e]) place[e] = n_place++;
        std::fill(R.bounds_h, R.bounds_h + E + 1, 0);
        int max_rows = 0, max_x = 0;
        for (int e = 0; e < E; ++e) {
            if (place[e] < 0) continue;   // (an expert in RAM that this chunk does not route)
            R.bounds_h[place[e] + 1] = rows_of[e];
            if (cpus[e]) continue;
            int& most = place[e] < ns ? max_rows : max_x;
            most = std::max(most, rows_of[e]);
        }
        for (int i = 0; i < E; ++i) R.bounds_h[i + 1] += R.bounds_h[i];
        std::vector<int32_t> at(R.bounds_h, R.bounds_h + E);
        for (int i = 0; i < n_ent; ++i) {
            const int e = R.ids_h[i], p = at[(size_t) place[e]]++;
            R.ent_row_h[p] = i / top_k;
            R.ent_dst_h[p] = i;
            R.ent_exp_h[p] = e;
        }
        auto blob_at = [&](int i) { return i < ns ? L.slots + (size_t) i * L.stride : xs + (size_t) (i - ns) * L.stride; };
        CK(cudaMemcpyAsync(R.bounds, R.bounds_h, (size_t) (E + 1) * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(R.ent_row, R.ent_row_h, (size_t) n_ent * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(R.ent_dst, R.ent_dst_h, (size_t) n_ent * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(R.ent_exp, R.ent_exp_h, (size_t) n_ent * 4, cudaMemcpyHostToDevice, st));
        MMQ::Product p{};
        p.expert_bytes = L.stride;
        p.ids_rows = R.ids_rows;
        p.total_rows = n_ent;
        // a product's launches: over the slots, then over the scratch's experts (`off`: the matrix in a blob)
        auto launch = [&](size_t off) {
            p.w = L.slots + off;
            p.n = ns;
            p.bounds = R.bounds;
            p.max_rows = max_rows;
            mmq->run(p, st);
            if (n_gpu == ns) return;
            p.w = xs + off;
            p.n = n_gpu - ns;
            p.bounds = R.bounds + ns;
            p.max_rows = max_x;
            mmq->run(p, st);
        };
        // An NVFP4 matrix on the FP16 tensor cores: the card's places in tiles of 64 entries, the entries' rows as
        // halves (x: their floats, map: an entry's row of them, xh: room for the halves)
        const bool tc = tc_on && T >= tc_min;
        const bool tc_gu = tc && L.gl.gu_type == 40 && K::nvfp4_gemm_fits(2 * F, N);
        const bool tc_d = tc && L.gl.d_type == 40 && K::nvfp4_gemm_fits(N, F);
        int n_tiles = 0;
        if (tc_gu || tc_d) {
            for (int i = 0; i < n_gpu; ++i) {
                R.pptr_h[i] = (unsigned long long) (uintptr_t) blob_at(i);
                const int n = R.bounds_h[i + 1] - R.bounds_h[i];
                for (int k = 0; k * K::kNvfp4GemmEntries < n; ++k) R.tiles_h[n_tiles++] = i << 16 | k;
            }
            CK(cudaMemcpyAsync(R.pptr, R.pptr_h, (size_t) n_gpu * 8, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(R.tiles, R.tiles_h, (size_t) std::max(n_tiles, 1) * 4, cudaMemcpyHostToDevice, st));
        }
        auto tc_product = [&](size_t off, size_t row_bytes, int w_rows, int w_cols, const float* x, const int32_t* map,
                              void* xh, const int32_t* ids, float* y) {
            K::nvfp4_gemm_halves(x, map, xh, n_ent, w_cols, st);
            K::Nvfp4Gemm q;
            q.ptr = R.pptr;
            q.off = off;
            q.row_bytes = row_bytes;
            q.w_rows = w_rows;
            q.w_cols = w_cols;
            q.bounds = R.bounds;
            q.tiles = R.tiles;
            q.n_tiles = n_tiles;
            q.x = xh;
            q.ids = ids;
            q.y = y;
            q.ld = w_rows;
            K::nvfp4_gemm(q, st);
        };
        // a BF16 matrix (MMQ has none): an expert's rows widened to floats, a GEMM over the rows routed to it
        auto bf16_product = [&](size_t off, int w_rows, int w_cols, const float* x, float* y) {
            for (int i = 0; i < n_gpu; ++i) {
                const int b0 = R.bounds_h[i], n = R.bounds_h[i + 1] - b0;
                if (n == 0) continue;
                const long long cnt = (long long) w_rows * w_cols;
                ik::bf16_floats_kernel<<<(unsigned) ((cnt + 255) / 256), 256, 0, st>>>((const uint16_t*) (blob_at(i) + off), R.wf, cnt);
                gemm(R.wf, w_rows, w_cols, x + (size_t) b0 * w_cols, n, y + (size_t) b0 * w_rows);
            }
        };
        if (L.gl.gu_type == 30) {   // the entries' inputs in their experts' order first
            const long long cnt = (long long) n_ent * N;
            ik::rows_index_kernel<<<(unsigned) ((cnt + 255) / 256), 256, 0, st>>>(R.nrm, R.ent_row, R.rows, n_ent, N, 0);
            bf16_product(0, 2 * F, N, R.rows, R.gu);
        } else if (tc_gu) {   // (the halves in the rows' buffer, which the down product fills after)
            tc_product(0, L.gl.gu_row, 2 * F, N, R.nrm, R.ent_row, R.rows, nullptr, R.gu);
        } else {
            MMQ::quantize(R.nrm, R.ent_row, R.tq, L.gl.gu_type, N, N, n_ent, st, R.tqs);
            p.type = L.gl.gu_type;
            p.w_rows = 2 * F;
            p.w_cols = N;
            p.xq = R.tq;
            p.scale = R.tqs;
            p.ids = R.ident;
            p.dst = R.gu;
            p.ld_dst = 2 * F;
            launch(0);
        }
        const long long cnt = (long long) n_ent * F;
        ik::situ_rows_kernel<<<(unsigned) ((cnt + 255) / 256), 256, 0, st>>>(R.gu, R.ent_exp, L.gu_scale, R.h, n_ent, F,
                                                                              g.situ_beta, g.situ_lbeta);
        if (L.gl.d_type == 30) {   // (gu is free again: the down rows in the experts' order, then to their entries)
            bf16_product(L.gl.down_off, N, F, R.h, R.gu);
            const long long all = (long long) n_ent * N;
            ik::rows_index_kernel<<<(unsigned) ((all + 255) / 256), 256, 0, st>>>(R.gu, R.ent_dst, R.rows, n_ent, N, 1);
        } else if (tc_d) {   // (h's halves in the gate/up product's MMQ image, free by now)
            tc_product(L.gl.down_off, L.gl.d_row, N, F, R.h, nullptr, R.tq, R.ent_dst, R.rows);
        } else {
            MMQ::quantize(R.h, nullptr, R.hq, L.gl.d_type, F, F, n_ent, st, R.hqs);
            p.type = L.gl.d_type;
            p.w_rows = N;
            p.w_cols = F;
            p.xq = R.hq;
            p.scale = R.hqs;
            p.ids = R.ent_dst;
            p.dst = R.rows;
            p.ld_dst = N;
            launch(L.gl.down_off);
        }
        if (n_cpu == 0) return;
        // the CPU's experts, while the card works: jobs of up to kMaxT of an expert's rows, their rows packed in the
        // jobs' order, then to their entries behind the card's own
        (void) bell_wait();
        ++cvisit;
        jobs.clear();
        int k = 0;
        for (int e = 0; e < E; ++e) {
            if (!cpus[e]) continue;
            for (int i = R.bounds_h[place[e]]; i < R.bounds_h[place[e] + 1]; ++i) {
                const int row = R.ent_row_h[i];
                uint8_t* const act = cacts.data() + (size_t) row * C::kNativeActBytes;
                if (cact_of[(size_t) row] != cvisit) {
                    C::native_quant_act(L.fmt, nrm_p + (size_t) row * N, act);
                    cact_of[(size_t) row] = cvisit;
                }
                if (jobs.empty() || jobs.back().blob != L.blob_h[(size_t) e] || jobs.back().nt == kMaxT) {
                    C::ExpertJobMulti j;
                    j.blob = L.blob_h[(size_t) e];
                    jobs.push_back(j);
                }
                C::ExpertJobMulti& j = jobs.back();
                j.nact[j.nt] = act;
                j.out[j.nt] = crow_p + (size_t) k * N;
                ++j.nt;
                cdst_p[k++] = R.ent_dst_h[i];
            }
        }
        pool->run_split_multi_native(L.fmt, jobs.data(), (int) jobs.size());
        CK(cudaMemcpyAsync(x_rows, crow_p, (size_t) k * N * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(x_dst, cdst_p, (size_t) k * 4, cudaMemcpyHostToDevice, st));
        const long long all = (long long) k * N;
        ik::rows_index_kernel<<<(unsigned) ((all + 255) / 256), 256, 0, st>>>(x_rows, x_dst, R.rows, k, N, 1);
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
        // the convolutions' inputs: a decode forward of several rows leaves them a visit, and the states are taken
        // from them once it is known how many rows stay (conv_commit)
        float* const wr = !batch && T > 1 ? win_raw + (size_t) v * kMaxT * (size_t) (2 * KV + 2 * N) : nullptr;
        float* const kraw = wr ? wr : R.kraw;
        float* const vraw = wr ? wr + (size_t) kMaxT * KV : R.vraw;
        float* const oraw = wr ? wr + (size_t) kMaxT * 2 * KV : R.oraw;
        float* const fraw = wr ? oraw + (size_t) kMaxT * N : R.fraw;

        // a pass of the loop starts from the state before the loop; the pass before it speaks through messages
        if (L.looped && l == g.loop_start) {
            if (pass == 0) {
                copy(R.prelude, R.x, TN);
            } else {
                copy(R.pass_out, R.x, TN);
                copy(R.x, R.prelude, TN);
            }
        }
        if (sp != nullptr)
            for (size_t k = 0; k + 1 < sp->taps.size(); ++k)
                if (sp->taps[k] == v) run("d: tap", [&] { spark_tap(R, (int) k, T, R.x); });
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
            run("a: mix of the messages", [&] { mix(bank, nbank, R.pass_out, L.msg_query, 4, R.msg, nullptr, T); });
        }

        // ---- attention: its inputs
        const float* u[3];
        if (L.nq == 0) {
            run("a: mix", [&] { mix(blks, nb, R.x, nullptr, 0, nullptr, newblk, T); });
            u[0] = u[1] = u[2] = R.x;
        } else {
            run("a: mix", [&] { mix(blks, nb, R.x, L.attn_query, L.nq, R.mixed, newblk, T); });
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
            run("a: gates' h", [&] { ik::ctrl_h_kernel<<<dim3((unsigned) a.R, (unsigned) T, 3), row4(), 0, st>>>(a); });
            run("a: gates applied", [&] {
                ik::ctrl_apply_kernel<<<dim3((unsigned) ((4 * N + ik::kTB - 1) / ik::kTB), (unsigned) T, 3), ik::kTB, 0, st>>>(a);
            });
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
            run("a: norm", [&] { ik::norm_kernel<<<dim3((unsigned) T, (unsigned) n_in), row4(), 0, st>>>(a); });
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
            mmb(R, L.wk, ki, fk, kraw, T);
            mmb(R, L.wv, vi, fv, vraw, T);
        } else {
            const ik::Q81 *iq = R.img, *ikk = R.img + (size_t) ki * TB, *iv = R.img + (size_t) vi * TB;
            // the five in one launch where their format has that kernel (the same values: a launch's own time once)
            const K::MmvqJob five[5] = {{L.wq.d, iq, R.qraw, (int) L.wq.n_out},     {L.wgate.d, iq, R.graw, (int) L.wgate.n_out},
                                        {L.wrel.d, iq, R.rraw, (int) L.wrel.n_out}, {L.wk.d, ikk, kraw, (int) L.wk.n_out},
                                        {L.wv.d, iv, vraw, (int) L.wv.n_out}};
            bool as_one = fuse_products;
            for (const Mat* W : {&L.wq, &L.wgate, &L.wrel, &L.wk, &L.wv}) as_one = as_one && W->type == L.wq.type && W->n_in == N;
            if (as_one) {
                bool took = false;
                run("a: the projections", [&] { took = K::native_mmvq_jobs((int) L.wq.type, five, 5, N, T, st); });
                as_one = took;
            }
            if (!as_one) {
                run("a: q's product", [&] { mm(L.wq, iq, fq, R.qraw, T); });
                run("a: the gate's product", [&] { mm(L.wgate, iq, fq, R.graw, T); });
                run("a: r's product", [&] { mm(L.wrel, iq, fq, R.rraw, T); });
                run("a: k's product", [&] { mm(L.wk, ikk, fk, kraw, T); });
                run("a: v's product", [&] { mm(L.wv, iv, fv, vraw, T); });
            }
        }
        // a decode forward takes the sink's product, tau, q's clip and the bias in one launch (q_bias_kernel)
        if (batch)
            run("a: sink", [&] { ik::sproj_kernel<<<dim3((unsigned) H, (unsigned) T), ik::kTB, 0, st>>>(L.sink, fq, R.sink, N, H); });
        if (L.full) {
            run("a: taps", [&] {
                ik::sproj_kernel<<<dim3((unsigned) (HK * C), (unsigned) T), ik::kTB, 0, st>>>(L.k_dyn, fk, R.dynk, N, HK * C);
                ik::sproj_kernel<<<dim3((unsigned) (HK * C), (unsigned) T), ik::kTB, 0, st>>>(L.v_dyn, fv, R.dynv, N, HK * C);
            });
            if (batch)
                run("a: tau", [&] {
                    ik::tau_kernel<<<(unsigned) ((T + 31) / 32), 32, 0, st>>>(R.pos_d, R.tau, T, g.log_alpha,
                                                                             (float) g.log_floor);
                });
        }
        tick("attention projections");
        probe("q's projection", R.qraw, (size_t) T, (size_t) H * D_);
        probe("the gate's projection", R.graw, (size_t) T, (size_t) H * D_);
        probe("the distance projection", R.rraw, (size_t) T, (size_t) H * Rr);
        probe("k's projection", kraw, (size_t) T, (size_t) KV);
        probe("v's projection", vraw, (size_t) T, (size_t) KV);
        // a chunk's window visit scores a range in position order (the win - 1 cells before the chunk out of the ring,
        // then the chunk's)
        const bool slide = batch && V.cells.ring;
        if (batch) {   // r clipped in place; the bias by cell of the range the rows score, and -sink
            run("a: q's clip", [&] {
                ik::q_clip_kernel<<<dim3((unsigned) T, (unsigned) H), (unsigned) D_, 0, st>>>(
                    R.qraw, L.q_scale, L.full ? R.tau : nullptr, R.q, H, D_, g.qk_clip);
            });
            ik::r_clip_kernel<<<dim3((unsigned) T, (unsigned) H), 32, 0, st>>>(R.rraw, L.rel_scale, L.full ? R.tau : nullptr,
                                                                              R.rraw, H, Rr, g.rel_clip);
            ik::RingIn rb{R.rraw, L.rel_table, R.sink, R.pos_d, R.bias, R.ns,
                          slide ? (long long) p0 - (win - 1) : (long long) V.cells.first, H, Rr, L.ext};
            run("a: bias by cell", [&] {
                if (Rr == 16) ik::bias_ring_kernel<4><<<(unsigned) T, 256, 0, st>>>(rb);
                else ik::bias_ring_kernel<0><<<(unsigned) T, 256, 0, st>>>(rb);
            });
        } else {
            ik::QbIn a{R.qraw, L.q_scale, R.rraw, L.rel_scale, L.rel_table, L.sink, fq, R.pos_d, R.q, R.bias, H, D_, Rr, L.ext,
                       N, L.full ? 1 : 0, g.qk_clip, g.rel_clip, g.log_alpha, (float) g.log_floor};
            run("a: q, bias", [&] { ik::q_bias_kernel<<<dim3((unsigned) T, (unsigned) H), ik::kWide, 0, st>>>(a); });
        }
        // the cells the rows attend: the visit's own, or for a chunk's window visit the range
        ik::Cells cells = V.cells;
        {
            ik::KvIn a{};
            a.raw[0] = kraw;
            a.raw[1] = vraw;
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
            if (batch) {
                run("a: k, v convolved", [&] {
                    ik::kv_conv_kernel<<<dim3((unsigned) T, 2, (unsigned) HK), (unsigned) D_, 0, st>>>(a);
                });
            } else {   // with the cells' store, and a forward of one row its states
                ik::KvCellsIn b{a, {T == 1 ? ck : nullptr, T == 1 ? cv : nullptr}, cells, R.pos_d};
                run("a: k, v convolved, stored", [&] {
                    ik::kv_cells_kernel<<<dim3((unsigned) T, 2, (unsigned) HK), (unsigned) D_, 0, st>>>(b);
                });
            }
        }
        if (batch)
            run("a: k, v states kept", [&] {
                conv_keep(ck, kraw, KV, T);
                conv_keep(cv, vraw, KV, T);
            });
        if (slide) {
            ik::ring_to_range_kernel<<<dim3((unsigned) (win - 1), 2), 256, 0, st>>>(V.cells, R.wk, R.wv, (long long) p0, win,
                                                                                   (int) cell_bytes);
            cells.k = R.wk;
            cells.v = R.wv;
            cells.first = p0 - (win - 1);
            cells.cap = T + win - 1;
            cells.ring = 0;
        }
        if (batch)
            run("a: cells stored", [&] {
                ik::cells_put_kernel<<<dim3((unsigned) T, 2, (unsigned) HK), 128, 0, st>>>(cells, R.kc, R.vc, R.pos_d, HK, D_);
            });
        tick("q, k, v, bias");
        xten("Qcur", v, R.q, T, H, D_);
        xten("Kcur", v, R.kc, T, HK, D_);
        xten("Vcur", v, R.vc, T, HK, D_);
        if (batch) xten("Rcur", v, R.rraw, T, H, Rr);
        probe("q", R.q, (size_t) T, (size_t) H * D_);
        probe("the bias", R.bias, (size_t) T, (size_t) H * (L.ext + (batch ? 0 : 1)));
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
        if (batch) {   // on the tensor cores: 64 rows of a query head a block, over the cells its rows see
            // cells on cards: the chunk's q and -sink go to them, each scores its cells while this card scores its
            // own from `lo`, and the partial sums are joined; so are this card's own where it scores a few rows'
            // cells in segments
            const bool carded = L.full && lo > 0;
            if (carded) {
                CK(cudaMemcpyAsync(pq_h, R.q, (size_t) T * H * D_ * 4, cudaMemcpyDeviceToHost, st));
                CK(cudaMemcpyAsync(ps_h, R.ns, (size_t) T * H * 4, cudaMemcpyDeviceToHost, st));
                CK(cudaStreamSynchronize(st));
                cards_scan_chunk(full_of[(size_t) v], T);
            }
            const long long in_use = slide ? (long long) (T + win - 1) : (long long) (p0 + T);
            const long long from = L.full ? (long long) lo : 0LL;
            const int n_seg = tc_segments(T, slide ? (long long) win : in_use - from);
            float* const part = carded ? R.tpart : n_seg > 1 ? R.spart : nullptr;
            float* const seg = n_seg > 1 ? R.spart + (carded ? 0 : (size_t) T * H * (D_ + 2)) : nullptr;
            ik::TcIn a{cells, R.q, R.bias, R.ns, R.graw, R.pos_d, R.aout, T, H, HK, L.ext, L.full ? 0 : win, in_use, part, 0,
                       from, seg};
            tc_launch(a, n_seg, st);
            if (carded) {
                (void) cudaStreamQuery(st);
                cards_wait();
                const size_t one = (size_t) R.cap * H * (D_ + 2);
                for (size_t ci = 0; ci < cards.size(); ++ci) {   // (a card without cells: sums of nothing)
                    const size_t bytes = (size_t) T * H * (D_ + 2) * 4;
                    if (cards[ci].cells > 0) CK(cudaMemcpyAsync(R.rpart + ci * one, cards[ci].csum_h, bytes, cudaMemcpyHostToDevice, st));
                    else CK(cudaMemsetAsync(R.rpart + ci * one, 0, bytes, st));
                }
            }
            if (part != nullptr) {
                ik::TcMergeIn mg{part, R.rpart, seg, R.graw, R.aout, H, D_, carded ? (int) cards.size() : 0, R.cap, n_seg - 1, T};
                ik::tc_merge_kernel<<<dim3((unsigned) T, (unsigned) H), (unsigned) D_, 0, st>>>(mg);
            }
        } else {
            if (V.cells.ring) {   // three tiles a block, where the ring divides so
                at.chunk = ring_cap % (3 * kRingSub) == 0 ? 3 * kRingSub : kRingSub;
                at.n_chunks = (int) (ring_cap / at.chunk);
            } else {   // the stretches from the step record: the same launch for any number of cells
                at.dyn = R.dyn_d;
                at.n_chunks = R.n_part;
                at.canon = 1;
            }
            // cells on cards: the rows' q and -sink go to them (pinned memory), they score their cells while this card
            // scores its own, and their sums come back for the join
            const bool carded = far_mode && L.full;
            if (carded) {
                ik::nsink_kernel<<<(unsigned) ((T * H + 255) / 256), 256, 0, st>>>(R.bias, R.ns, T * H, L.ext);
                CK(cudaMemcpyAsync(xq_h, R.q, (size_t) T * H * D_ * 4, cudaMemcpyDeviceToHost, st));
                CK(cudaMemcpyAsync(xs_h, R.ns, (size_t) T * H * 4, cudaMemcpyDeviceToHost, st));
                seg_end(full_of[(size_t) v], T);
                if (capturing) cur_segs->back().at = at;
            }
            run(L.full ? "a: attention's sums, full" : "a: attention's sums, window", [&] { scan(at, T, st); });
            ik::AttnOutIn a{R.part, R.graw, L.wo.type == 30 ? R.aout : nullptr, R.aimg, at.dyn, H, D_, at.n_chunks};
            if (carded) {
                (void) cudaStreamQuery(st);   // (this card's scan runs beside theirs)
                seg_begin();
                const size_t one = (size_t) kMaxT * H * (D_ + 2);
                for (size_t ci = 0; ci < cards.size(); ++ci)
                    CK(cudaMemcpyAsync(R.rpart + ci * one, cards[ci].sum_h, (size_t) T * H * (D_ + 2) * 4,
                                       cudaMemcpyHostToDevice, st));
                a.extra = R.rpart;
                a.n_extra = (int) cards.size();
                a.T = kMaxT;
            }
            run("a: attention's output", [&] {
                ik::attn_out_kernel<<<dim3((unsigned) T, (unsigned) H), (unsigned) D_, 0, st>>>(a);
            });
        }
        if (slide)
            ik::range_to_ring_kernel<<<dim3((unsigned) std::min<int64_t>(T, ring_cap), 2), 256, 0, st>>>(
                V.cells, R.wk, R.wv, (long long) p0, T, win, (int) cell_bytes);
        tick(L.full ? "attention, full" : "attention, window");
        if (batch || L.wo.type == 30) probe("the attention's output", R.aout, (size_t) T, (size_t) H * D_);
        if (batch) mmb(R, L.wo, 3, R.aout, oraw, T);
        else run("a: the output's product", [&] { mm(L.wo, R.aimg, R.aout, oraw, T); });
        const dim3 gN((unsigned) ((N + ik::kTB - 1) / ik::kTB), (unsigned) T);
        const bool one = !batch && T == 1;   // a forward of one row: a sublayer's end is one launch (post_one_kernel)
        if (one) {
            run("a: convolved into the stream", [&] {
                ik::post_one_kernel<<<gN.x, ik::kTB, 0, st>>>(nullptr, nullptr, oraw, ca, L.a_conv, R.x, N, C, top_k,
                                                             L.first ? 1 : 0);
            });
        } else {
            run("a: convolved into the stream", [&] {
                ik::post_conv_kernel<<<gN, ik::kTB, 0, st>>>(oraw, ca, L.a_conv, R.x, N, C, L.first ? 1 : 0);
            });
            if (batch) run("a: output's state kept", [&] { conv_keep(ca, oraw, N, T); });
        }
        tick("attention output");
        probe("the output projection", oraw, (size_t) T, (size_t) N);
        probe("x after the attention", R.x, (size_t) T, (size_t) N);
        xrow("ffn_inp", v, R.x, T);

        // ---- the MoE
        const int nb2 = l / res + 1;
        if (L.first) blks[nb2 - 1] = newblk;
        run("f: mix", [&] { mix(blks, nb2, R.x, L.ffn_query, 1, R.mixed, nullptr, T); });
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
            run("f: gate's h", [&] { ik::ctrl_h_kernel<<<dim3((unsigned) a.R, (unsigned) T, 1), row4(), 0, st>>>(a); });
            run("f: gate applied", [&] {
                ik::ctrl_apply_kernel<<<dim3((unsigned) ((4 * N + ik::kTB - 1) / ik::kTB), (unsigned) T, 1), ik::kTB, 0, st>>>(a);
            });
            fu = R.ctl;
        }
        {
            ik::NormIn a{};
            a.x[0] = fu;
            a.w[0] = L.ffn_norm;
            a.y[0] = R.nrm;
            a.q[0] = batch ? nullptr : R.img;
            a.il = L.gl.x_il;
            a.n = 1;
            a.N = N;
            a.eps = g.eps;
            run("f: norm", [&] { ik::norm_kernel<<<dim3((unsigned) T, 1), row4(), 0, st>>>(a); });
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
            ik::expert_sum_kernel<<<gN, ik::kTB, 0, st>>>(R.rows, R.w, fraw, N, top_k, L.d_scale, R.ids);
        } else {
            run("f: router's products", [&] {
                ik::route_dot_kernel<<<dim3((unsigned) E, (unsigned) T), ik::kTB, 0, st>>>(L.router, R.nrm, R.probs, E, N);
            });
            ik::PlanIn a{R.probs, L.route_bias, L.slot_of, (unsigned long long) (uintptr_t) L.slots, L.stride, R.ids, R.w,
                         R.grp_ptr, R.grp_start, R.n_groups, R.ent_dst, R.ent_tok, R.n_miss, T, E, top_k, w_scale};
            run("f: experts picked", [&] {
                if (T == 1) ik::pick_one_kernel<<<1, ik::kTB, 0, st>>>(a);
                else ik::pick_plan_kernel<<<1, ik::kTB, 0, st>>>(a);
            });
            tick("MoE input and router");
            if (moe_cuts) {   // the routing and the MoE's input to the host, the bell behind them
                if (!capturing) *bell_h = -1;
                run("f: routing to the host", [&] {
                    CK(cudaMemcpyAsync(moe_h, R.ids, ((size_t) kMaxT * top_k + TN) * 4, cudaMemcpyDeviceToHost, st));
                    CK(cudaMemcpyAsync(bell_h, R.n_miss, 4, cudaMemcpyDeviceToHost, st));
                });
            }
            run("f: experts", [&] {
                K::native_expert_grouped(L.gl, R.grp_ptr, R.grp_start, R.n_groups, R.ent_dst, R.ent_tok, n_ent, n_ent, R.img,
                                         (int64_t) (N / 32) * 36, R.scratch, R.rows, st);
            });
            if (moe_cuts) {
                moe_cut(v, T);
                run("f: the CPU's rows", [&] {
                    ik::cpu_rows_kernel<<<(unsigned) n_ent, ik::kTB, 0, st>>>(R.crows, R.ids, L.slot_of, L.d_scale, R.rows, N);
                });
            }
            tick("experts");
            if (!one)
                run("f: experts' rows summed", [&] {
                    ik::expert_sum_kernel<<<gN, ik::kTB, 0, st>>>(R.rows, R.w, fraw, N, top_k, nullptr, nullptr);
                });
        }
        if (one) {
            run("f: summed, convolved into the stream", [&] {
                ik::post_one_kernel<<<gN.x, ik::kTB, 0, st>>>(R.rows, R.w, fraw, cf, L.f_conv, R.x, N, C, top_k, 0);
            });
        } else {
            run("f: convolved into the stream", [&] {
                ik::post_conv_kernel<<<gN, ik::kTB, 0, st>>>(fraw, cf, L.f_conv, R.x, N, C, 0);
            });
            if (batch) run("f: output's state kept", [&] { conv_keep(cf, fraw, N, T); });
        }
        tick("MoE output");
        probe("the MoE's output", fraw, (size_t) T, (size_t) N);
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
        ik::norm_kernel<<<dim3((unsigned) n, 1), row4(), 0, st>>>(a);
        mm(m->head, R.img, R.nrm, R.logits, n);
        CK(cudaMemsetAsync(R.bad, 0, (size_t) n * 4, st));
        const long long cnt = (long long) n * g.n_vocab;
        ik::softcap_kernel<<<(unsigned) ((cnt + 255) / 256), 256, 0, st>>>(R.logits, cnt, g.softcap, (int) g.n_vocab, R.bad);
        lg_rows = &R;
        tick("head");
    }

    /// With the drafter: every row's final normed state (the last of the states it reads) and the rows' context cells,
    /// then the logits of the last n rows
    void head_all(Rows& R, int T, int n) {
        const Geometry& g = m->g;
        const int N = (int) g.n_embd;
        const float* blks[ik::kBlk] = {};
        for (int b = 0; b < nblocks(); ++b) blks[b] = blk_at(R, b, (int) g.n_pass - 1);
        mix(blks, nblocks(), R.x, m->out_query, 1, R.mixed, nullptr, T);
        ik::NormIn a{};
        a.x[0] = R.mixed;
        a.w[0] = m->out_norm;
        a.y[0] = R.nrm;
        a.q[0] = R.batch ? nullptr : R.img;
        a.n = 1;
        a.N = N;
        a.eps = g.eps;
        ik::norm_kernel<<<dim3((unsigned) T, 1), row4(), 0, st>>>(a);
        spark_tap(R, (int) sp->taps.size() - 1, T, R.nrm);
        spark_inject(R, T);
        if (n > 0) {
            const int r0 = T - n;
            const ik::Q81* img = R.img + (size_t) r0 * (size_t) (N / 32);
            if (R.batch) {   // (a chunk's rows have no images: its last row's)
                ik::quant_kernel<<<dim3((unsigned) ((N / 32 + 7) / 8), (unsigned) n), ik::kTB, 0, st>>>(R.nrm + (size_t) r0 * N,
                                                                                                    R.img, N);
                img = R.img;
            }
            mm(m->head, img, R.nrm + (size_t) r0 * N, R.logits, n);
            CK(cudaMemsetAsync(R.bad, 0, (size_t) n * 4, st));
            const long long cnt = (long long) n * g.n_vocab;
            ik::softcap_kernel<<<(unsigned) ((cnt + 255) / 256), 256, 0, st>>>(R.logits, cnt, g.softcap, (int) g.n_vocab, R.bad);
            lg_rows = &R;
        }
        tick("head");
    }

    /// A token's embedding row as the first layer reads it (normed by the embedding's weight)
    void embed_row(int32_t tok, float* row) {
        const Geometry& g = m->g;
        const size_t N = (size_t) g.n_embd;
        const ggml_type_traits* tr = m->embd_type != 0 ? ggml_get_type_traits((ggml_type) m->embd_type) : nullptr;
        const uint8_t* src = m->embd.get() + (size_t) tok * m->embd_row;
        if (tr) tr->to_float(src, row, (int64_t) N);
        else std::memcpy(row, src, N * 4);
        double ss = 0;
        for (size_t i = 0; i < N; ++i) ss += (double) row[i] * (double) row[i];
        const float inv = 1.0f / std::sqrt((float) (ss / (double) N) + g.eps);
        for (size_t i = 0; i < N; ++i) row[i] = row[i] * inv * m->tok_norm[i];
    }
    /// T tokens' rows at positions p0 .., in R.emb_h: a token's embedding, normed; a picture's cell its row as the
    /// image encoder gave it
    void embed(Rows& R, const int32_t* tok, int T, int64_t p0) {
        const size_t N = (size_t) m->g.n_embd;
        for (int t = 0; t < T; ++t) {
            const int32_t r = img != nullptr ? img->row(p0 + t) : -1;
            if (r >= 0) std::memcpy(R.emb_h + (size_t) t * N, img->rows.data() + (size_t) r * N, N * 4);
            else embed_row(tok[t], R.emb_h + (size_t) t * N);
        }
    }
    // pictures (--vision): the request's rows while its prompt is read
    bool vision = false;
    const ImgReq* img = nullptr;

    // ---- the drafter
    /// A pair's cosine and sine for rows at positions p0, p0 + 1, ..: cs_h [T][D / 2][2] (in doubles: a position past
    /// a million leaves a float no angle)
    void rope_rows(float* cs_h, int64_t p0, int T) {
        const size_t half = sp_theta.size();
        for (int t = 0; t < T; ++t)
            for (size_t k = 0; k < half; ++k) {
                const double th = std::fmod((double) (p0 + t) * sp_theta[k], 6.283185307179586476925);
                cs_h[((size_t) t * half + k) * 2] = (float) std::cos(th);
                cs_h[((size_t) t * half + k) * 2 + 1] = (float) std::sin(th);
            }
    }
    /// The drafter's buffers, its rings and the head's product (before the context is sized: VRAM the cells do not get)
    void spark_init() {
        const Spark& s = *sp;
        const size_t N = (size_t) s.N, Q = (size_t) s.H * s.D, KV = (size_t) s.HK * s.D, F = (size_t) s.F, T = kMaxT,
                     V = (size_t) s.V, R_ = (size_t) s.R;
        for (int k = 0; k < s.D / 2; ++k) sp_theta.push_back(std::pow(s.rope_base, -2.0 * k / (double) s.D));
        // the rings: a layer's cells, the block's after them, and the slack; whole stretches of whole tiles
        sp_ring.resize((size_t) s.n_layer);
        int max_ch = 1;
        for (int l = 0; l < s.n_layer; ++l) {
            SparkRing& r = sp_ring[(size_t) l];
            r.see = s.layers[(size_t) l].window ? s.n_swa : std::max<int64_t>(sp_reach, s.n_swa);
            // (three tiles a stretch at least, thirty at most, about 64 stretches between)
            r.chunk = (int) std::max<int64_t>(3, std::min<int64_t>(30, (r.see + (int64_t) T + 64 * ik::kTcCells - 1) / (64 * ik::kTcCells))) *
                      ik::kTcCells;
            r.n_chunks = (int) ((r.see + T + r.chunk - 1) / r.chunk);
            r.cells.ring = 1;
            r.cells.q8 = 0;
            r.cells.cap = (r.see + (int64_t) T + kSparkSlack + 2 * r.chunk - 1) / r.chunk * r.chunk;
            r.cells.k = dzero<uint8_t>((size_t) r.cells.cap * KV * 2);
            r.cells.v = dzero<uint8_t>((size_t) r.cells.cap * KV * 2);
            max_ch = std::max(max_ch, r.n_chunks);
        }
        frnn = dzero<float>(T * N);
        for (const SparkLayer& y : s.layers)
            if ((y.wk_ctx.type == 30 || y.wv_ctx.type == 30) && sp_wf == nullptr) sp_wf = dzero<float>(KV * N);
        SparkBlock& b = sb;
        b.x = dzero<float>(T * N);
        b.x0 = dzero<float>(T * N);
        b.xn = dzero<float>(T * N);
        b.img = dzero<ik::Q81>(T * (N / 32));
        b.qr = dzero<float>(T * Q);
        b.q = dzero<float>(T * Q);
        b.kr = dzero<float>(T * KV);
        b.vr = dzero<float>(T * KV);
        b.kn = dzero<float>(T * KV);
        b.cs = dzero<float>(T * (size_t) s.D);
        b.zero = dzero<float>(T * (size_t) s.H);
        b.part = dzero<float>(T * (size_t) s.H * (size_t) max_ch * (size_t) (s.D + 2));
        b.sum = dzero<float>(T * (size_t) s.H * (size_t) (s.D + 2));
        b.ao = dzero<float>(T * Q);
        b.aimg = dzero<ik::Q81>(T * (Q / 32));
        b.o = dzero<float>(T * N);
        b.g = dzero<float>(T * F);
        b.u = dzero<float>(T * F);
        b.h = dzero<float>(T * F);
        b.himg = dzero<ik::Q81>(T * (F / 32));
        b.d = dzero<float>(T * N);
        b.hn = dzero<float>(T * N);
        b.hnimg = dzero<ik::Q81>(T * (N / 32));
        b.base = dzero<float>(T * V);
        b.pre = dzero<float>(T * 3 * R_);
        b.state = dzero<float>(R_);
        b.lat = dzero<int8_t>(R_);
        b.vpart = dzero<float>((V + 255) / 256 * 3);
        b.ctx = dzero<float>(N);
        b.c0 = dzero<float>(V);
        b.M8 = dzero<int8_t>(V * R_);
        b.mscale = dzero<float>(V);
        b.n_step = 2 * (int) T + 5 + 3 * s.n_layer;
        b.step = dzero<int32_t>((size_t) b.n_step);
        b.out = dzero<int32_t>(2 * T);
        b.emb_h = halloc<float>(N);
        b.cs_h = halloc<float>(T * (size_t) s.D);
        b.step_h = halloc<int32_t>((size_t) b.n_step);
        b.out_h = halloc<int32_t>(2 * T);
        // a block's rows before its layers: the token's row comes into the first, the others are the mask
        for (size_t t = 1; t < T; ++t) CK(cudaMemcpy(b.x0 + t * N, s.mask, N * 4, cudaMemcpyDeviceToDevice));
        // the head's product: the target's head over hidden_out's columns, 8 a pass; over its bias; then as bytes
        float* cols = nullptr;
        __half* M = nullptr;
        CK(cudaMalloc(&cols, s.hidden_cols.size() * 4));
        CK(cudaMalloc(&M, V * R_ * 2));
        CK(cudaMemcpy(cols, s.hidden_cols.data(), s.hidden_cols.size() * 4, cudaMemcpyHostToDevice));
        for (size_t c = 0; c <= R_; c += T) {
            const int n = (int) std::min(T, R_ + 1 - c);
            ik::quant_kernel<<<dim3((unsigned) ((N / 32 + 7) / 8), (unsigned) n), ik::kTB, 0, st>>>(cols + c * N, b.img, (int) N);
            mm(m->head, b.img, cols + c * N, b.base, n);
            const int nm = (int) std::min<size_t>((size_t) n, R_ - std::min(c, R_));
            if (nm > 0) {
                const long long cnt = (long long) nm * (long long) V;
                ik::head_m_kernel<<<(unsigned) ((cnt + 255) / 256), 256, 0, st>>>(b.base, M, (int) V, (int) R_, (int) c, nm);
            }
            if (c + (size_t) n > R_)   // the bias's row is this pass's last
                CK(cudaMemcpyAsync(b.c0, b.base + (size_t) (n - 1) * V, V * 4, cudaMemcpyDeviceToDevice, st));
        }
        ik::head_m8_kernel<<<(unsigned) V, 256, 0, st>>>(M, b.M8, b.mscale, (int) R_);
        CK(cudaStreamSynchronize(st));
        CK(cudaFree(M));
        CK(cudaFree(cols));
    }
    /// A tapped state of T rows into the mixes of R
    void spark_tap(Rows& R, int k, int T, const float* x) {
        const Spark& s = *sp;
        ik::FuseIn a{x, R.f_fused, {}, s.n_layer + 1, s.N, R.cap, k == 0 ? 1 : 0};
        for (int r = 0; r <= s.n_layer; ++r) a.c[r] = s.fuse[(size_t) r * s.taps.size() + (size_t) k];
        ik::fuse_kernel<<<dim3((unsigned) ((s.N / 4 + ik::kTB - 1) / ik::kTB), (unsigned) T), ik::kTB, 0, st>>>(a);
    }
    /// The rows' context cells of every drafter layer from their mixes, and the head's context of each row (a chunk:
    /// of its last)
    void spark_inject(Rows& R, int T) {
        const Spark& s = *sp;
        const int N = s.N, L = s.n_layer;
        const size_t TN = (size_t) R.cap * (size_t) N;
        ik::NormIn a{};
        for (int l = 0; l < L; ++l) {
            a.x[l] = a.y[l] = R.f_fused + (size_t) l * TN;
            a.w[l] = s.layers[(size_t) l].ctx_norm;
            a.q[l] = R.batch ? nullptr : R.f_img + (size_t) l * (size_t) R.cap * (size_t) (N / 32);
        }
        a.n = L;
        a.N = N;
        a.eps = s.eps;
        run("d: mixes normed", [&] { ik::norm_kernel<<<dim3((unsigned) T, (unsigned) L), row4(), 0, st>>>(a); });
        {
            const int r0 = R.batch ? T - 1 : 0, n = R.batch ? 1 : T;
            ik::NormIn b{};
            b.x[0] = R.f_fused + (size_t) L * TN + (size_t) r0 * N;
            b.w[0] = s.rnn_norm;
            b.y[0] = frnn;
            b.n = 1;
            b.N = N;
            b.eps = s.eps;
            run("d: head's context", [&] { ik::norm_kernel<<<dim3((unsigned) n, 1), row4(), 0, st>>>(b); });
        }
        for (int l = 0; l < L; ++l) {
            const SparkLayer& y = s.layers[(size_t) l];
            const float* cx = R.f_fused + (size_t) l * TN;
            if (R.batch) {   // MMQ over the rows' image, made once a layout; a BF16 matrix widened, then a GEMM
                int have = -1;
                for (const auto& [W, out] : {std::pair<const Mat*, float*>{&y.wk_ctx, R.f_kc}, {&y.wv_ctx, R.f_vc}}) {
                    if (W->type == 30) {
                        const long long cnt = (long long) W->n_out * W->n_in;
                        ik::bf16_floats_kernel<<<(unsigned) ((cnt + 255) / 256), 256, 0, st>>>((const uint16_t*) W->d, sp_wf, cnt);
                        gemm(sp_wf, W->n_out, W->n_in, cx, T, out);
                        continue;
                    }
                    if (const int lay = MMQ::ds_layout(W->type); lay != have) {
                        MMQ::quantize(cx, nullptr, R.f_q, W->type, N, N, T, st);
                        have = lay;
                    }
                    mmq->run_dense(W->d, W->type, W->n_out, N, R.f_q, T, out, W->n_out, st);
                }
            } else {
                const ik::Q81* img = R.f_img + (size_t) l * (size_t) R.cap * (size_t) (N / 32);
                run("d: context k's product", [&] { mm(y.wk_ctx, img, cx, R.f_kc, T); });
                run("d: context v's product", [&] { mm(y.wv_ctx, img, cx, R.f_vc, T); });
            }
            run("d: context cells", [&] {
                ik::head_rope_kernel<<<dim3((unsigned) T, (unsigned) s.HK), 128, 0, st>>>(R.f_kc, y.k_ctx_norm, R.f_cs, R.f_kn, s.HK,
                                                                                         s.D, s.eps);
                ik::cells_put_kernel<<<dim3((unsigned) T, 2, (unsigned) s.HK), 128, 0, st>>>(sp_ring[(size_t) l].cells, R.f_kn,
                                                                                            R.f_vc, R.pos_d, s.HK, s.D);
            });
        }
        tick("drafter's context");
    }
    /// A forward or a chunk wrote the context cells of positions [p0, p0 + T)
    void spark_wrote(int64_t p0, int T, bool is_chunk) {
        if (sp == nullptr) return;
        if (p0 > sp_vhi || p0 < sp_vlo) {   // not on from what the rings hold: they start again here
            sp_vlo = p0;
            sp_whi = 0;
        }
        sp_vhi = p0 + T;
        sp_whi = std::max(sp_whi, p0 + T);
        frnn_p0 = is_chunk ? p0 + T - 1 : p0;
        frnn_n = is_chunk ? 1 : T;
    }
    /// The state's text is another one from position `at` on
    void spark_cut(int64_t at) {
        sp_vhi = std::min(sp_vhi, at);
        sp_vlo = std::min(sp_vlo, sp_vhi);
        frnn_n = 0;
    }
    /// The head's steps for the block's rows [i0, i1]: the target's head over them, then a draft a row while the
    /// drafts' joint probability stays at the floor (out: a token and its probability's bits a step, -1 from where
    /// they ended)
    void spark_steps(int i0, int i1) {
        const Spark& s = *sp;
        SparkBlock& b = sb;
        const int N = s.N, V = s.V, R_ = s.R, T = kMaxT, nb = (V + 255) / 256;
        int32_t* prev = b.step + 2 * T + 1 + 3 * s.n_layer;
        int32_t* ctl = prev + 1;
        mm(m->head, b.hnimg + (size_t) i0 * (size_t) (N / 32), b.hn + (size_t) i0 * N, b.base + (size_t) i0 * (size_t) V, i1 - i0 + 1);
        tick("drafter: the target's head");
        for (int i = i0; i <= i1; ++i) {
            ik::StepIn si{s.joint_w, b.pre + (size_t) i * 3 * (size_t) R_, s.w1, prev, b.state, b.lat, R_, 2 * R_ + 2 * N, ctl + 2};
            ik::head_step_kernel<<<1, 256, 0, st>>>(si);
            ik::VocabIn vi{b.base + (size_t) i * (size_t) V, b.c0, b.mscale, b.M8, b.lat, b.vpart, V, R_, s.cap, ctl + 2};
            ik::head_vocab_kernel<<<(unsigned) nb, 256, 0, st>>>(vi);
            ik::head_pick_kernel<<<1, 1024, 0, st>>>(b.vpart, nb, b.out + 2 * (i - 1), prev, ctl);
        }
        tick("drafter: the head's steps");
        CK(cudaMemcpyAsync(b.out_h, b.out, (size_t) 2 * T * 4, cudaMemcpyDeviceToHost, st));
    }
    /// A block's launches: [the token, masks] through the layers, then the first draft
    void spark_launches() {
        const Spark& s = *sp;
        SparkBlock& b = sb;
        const int T = kMaxT, N = s.N, F = s.F, R_ = s.R;
        const size_t TN = (size_t) T * (size_t) N;
        CK(cudaMemcpyAsync(b.x, b.x0, TN * 4, cudaMemcpyDeviceToDevice, st));
        CK(cudaMemcpyAsync(b.x, b.emb_h, (size_t) N * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(b.step, b.step_h, (size_t) b.n_step * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(b.cs, b.cs_h, (size_t) T * (size_t) s.D * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemsetAsync(b.state, 0, (size_t) R_ * 4, st));
        CK(cudaMemsetAsync(b.out, 0xff, (size_t) 2 * T * 4, st));
        const int32_t *pos_scan = b.step, *live = b.step + T, *pos_own = b.step + T + 1, *dyn0 = b.step + 2 * T + 1;
        const dim3 gN((unsigned) ((TN + 255) / 256));
        auto norm = [&](const float* x, const float* w, float* y, ik::Q81* q) {
            ik::NormIn a{};
            a.x[0] = x;
            a.w[0] = w;
            a.y[0] = y;
            a.q[0] = q;
            a.n = 1;
            a.N = N;
            a.eps = s.eps;
            ik::norm_kernel<<<dim3((unsigned) T, 1), row4(), 0, st>>>(a);
        };
        for (int l = 0; l < s.n_layer; ++l) {
            const SparkLayer& y = s.layers[(size_t) l];
            const SparkRing& r = sp_ring[(size_t) l];
            norm(b.x, y.attn_norm, b.xn, b.img);
            mm(y.wq, b.img, b.xn, b.qr, T);
            mm(y.wk, b.img, b.xn, b.kr, T);
            mm(y.wv, b.img, b.xn, b.vr, T);
            ik::head_rope_kernel<<<dim3((unsigned) T, (unsigned) s.H), 128, 0, st>>>(b.qr, y.q_norm, b.cs, b.q, s.H, s.D, s.eps);
            ik::head_rope_kernel<<<dim3((unsigned) T, (unsigned) s.HK), 128, 0, st>>>(b.kr, y.k_norm, b.cs, b.kn, s.HK, s.D, s.eps);
            ik::cells_put_kernel<<<dim3((unsigned) T, 2, (unsigned) s.HK), 128, 0, st>>>(r.cells, b.kn, b.vr, pos_own, s.HK, s.D);
            ik::AttnIn at{};
            at.c = r.cells;
            at.q = b.q;
            at.bias = b.zero;
            at.pos = pos_scan;
            at.live = live;
            at.dyn = dyn0 + 3 * l;
            at.part = b.part;
            at.H = s.H;
            at.HK = s.HK;
            at.D = s.D;
            at.ext = 0;
            at.win = (int) (r.see + T);
            at.chunk = r.chunk;
            at.n_chunks = r.n_chunks;
            ik::attn_rows_kernel<false><<<dim3((unsigned) r.n_chunks, (unsigned) s.HK), 256, 0, st>>>(at, T);
            ik::JoinIn jn{b.part, b.sum, at.dyn, s.H, s.D, r.n_chunks};
            ik::attn_join_kernel<<<dim3((unsigned) T, (unsigned) s.H), (unsigned) s.D, 0, st>>>(jn);
            ik::attn_avg_kernel<<<dim3((unsigned) T, (unsigned) s.H), (unsigned) s.D, 0, st>>>(b.sum, b.ao, b.aimg, s.H, s.D);
            mm(y.wo, b.aimg, b.ao, b.o, T);
            ik::add_kernel<<<gN, 256, 0, st>>>(b.x, b.o, (long long) TN);
            tick("drafter: attention");
            norm(b.x, y.ffn_norm, b.xn, b.img);
            mm(y.gate, b.img, b.xn, b.g, T);
            mm(y.up, b.img, b.xn, b.u, T);
            ik::swiglu_kernel<<<dim3((unsigned) ((F / 32 + 7) / 8), (unsigned) T), ik::kTB, 0, st>>>(b.g, b.u, b.h, b.himg, F);
            mm(y.down, b.himg, b.h, b.d, T);
            ik::add_kernel<<<gN, 256, 0, st>>>(b.x, b.d, (long long) TN);
            tick("drafter: the MLP");
        }
        norm(b.x, s.out_norm, b.hn, b.hnimg);
        ik::JointPreIn jp{s.joint_w, s.joint_b, b.hn, b.ctx, b.pre, 2 * R_, N, T, 3 * R_};
        ik::joint_pre_kernel<<<(unsigned) (3 * R_), ik::kTB, 0, st>>>(jp);
        spark_steps(1, 1);
    }
    /// Up to `n` drafts after x, the token the state does not hold yet (at pos()): a block through the drafter, a draft
    /// while the drafts' joint probability under its head stays at or above `min_p`.  None where the rings do not hold
    /// the context up to the position before (a state brought back from a checkpoint or the park: the next forward
    /// writes it).
    std::vector<int32_t> spark_draft(int32_t x, int n, float min_p) {
        std::vector<int32_t> out;
        const int64_t P = pos();
        n = std::min(n, kMaxT - 1);
        if (sp == nullptr || n <= 0 || sp_vhi != P || P < 1 || frnn_n <= 0 || P - 1 < frnn_p0 || P - 1 >= frnn_p0 + frnn_n) return out;
        const auto t0 = Clock::now();
        const Spark& s = *sp;
        SparkBlock& b = sb;
        const int T = kMaxT;
        CK(cudaStreamSynchronize(st));   // (the last block has read the pinned rows)
        if (prof) prof_t = Clock::now();
        CK(cudaMemcpyAsync(b.ctx, frnn + (size_t) (P - 1 - frnn_p0) * (size_t) s.N, (size_t) s.N * 4, cudaMemcpyDeviceToDevice, st));
        embed_row(x, b.emb_h);
        rope_rows(b.cs_h, P, T);
        for (int t = 0; t < T; ++t) {
            b.step_h[t] = (int32_t) (P + T - 1);
            b.step_h[T + 1 + t] = (int32_t) (P + t);
        }
        b.step_h[T] = (int32_t) (P + T);
        for (int l = 0; l < s.n_layer; ++l) {
            const SparkRing& r = sp_ring[(size_t) l];
            // the context cells in use: within the layer's reach, of the text the rings hold, not yet written over
            const int64_t lo = std::min(P, std::max({(int64_t) 0, P - r.see, sp_vlo, sp_whi - r.cells.cap + r.chunk}));
            int32_t* d = b.step_h + 2 * T + 1 + 3 * l;
            d[0] = r.chunk;
            d[1] = (int32_t) ((P + T - lo + r.chunk - 1) / r.chunk);
            d[2] = (int32_t) (lo % r.cells.cap);
        }
        {
            int32_t* ctl = b.step_h + 2 * T + 1 + 3 * s.n_layer;
            const float one = 1.0f;
            ctl[0] = x;
            std::memcpy(&ctl[1], &one, 4);
            std::memcpy(&ctl[2], &min_p, 4);
            ctl[3] = 0;
        }
        // the layers and the first draft; the others (their rows of the target's head first) where it is at the floor
        auto issue = [&](cudaGraphExec_t& exec, const std::function<void()>& launches) {
            if (!use_graphs) {
                launches();
            } else {
                if (exec == nullptr) {
                    cudaGraph_t graph = nullptr;
                    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
                    launches();
                    CK(cudaStreamEndCapture(st, &graph));
                    CK(cudaGraphInstantiate(&exec, graph, 0));
                    CK(cudaGraphDestroy(graph));
                    own(&Owned::graphs, exec);
                }
                CK(cudaGraphLaunch(exec, st));
            }
            CK(cudaStreamSynchronize(st));
        };
        issue(sp_graph, [&] { spark_launches(); });
        if (n > 1 && b.out_h[0] >= 0) issue(sp_graphs[n], [&] { spark_steps(2, n); });
        sp_whi = std::max(sp_whi, P + T);
        for (int i = 0; i < n && b.out_h[2 * i] >= 0 && b.out_h[2 * i] < s.V; ++i) out.push_back(b.out_h[2 * i]);
        ++sp_blocks;
        sp_ms += ms_since(t0);
        return out;
    }

    /// The step record of T rows at p0 ..
    void steps(Rows& R, int64_t p0, int T) {
        const int64_t live = p0 + T;
        for (int t = 0; t < R.cap; ++t) R.step_h[t] = (int32_t) (p0 + std::min(t, T - 1));
        // the main card's cells, from `lo`: {the first, the stretches in use, the first one's number}
        R.step_h[R.cap] = (int32_t) live;
        R.step_h[R.cap + 1] = (int32_t) lo;
        R.step_h[R.cap + 2] = (int32_t) (stretches(live) - stretch_of(lo));
        R.step_h[R.cap + 3] = (int32_t) stretch_of(lo);
        R.step_h[R.cap + 4] = (int32_t) (T + m->g.n_swa - 1);
    }
    /// A forward's launches: the rows and the step record from their pinned copies, every visit, the head's rows; with
    /// `commit` the convolutions' states after a decode forward of several rows (without: keep() takes them)
    void launches(Rows& R, int T, int n_head, int64_t p0, bool commit) {
        CK(cudaMemcpyAsync(R.x, R.emb_h, (size_t) T * (size_t) m->g.n_embd * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(R.step_d, R.step_h, (size_t) (R.cap + 5) * 4, cudaMemcpyHostToDevice, st));
        tick("embedding");
        if (sp != nullptr) CK(cudaMemcpyAsync(R.f_cs, R.f_cs_h, (size_t) T * (size_t) sp->D * 4, cudaMemcpyHostToDevice, st));
        for (int v = 0; v < (int) visits.size(); ++v) visit(v, T, R, p0);
        if (sp != nullptr) head_all(R, T, n_head);
        else if (n_head > 0) head(R, T - n_head, n_head);
        if (commit && !R.batch && T > 1) run("states kept", [&] { conv_commit(T); });
    }
    /// Every visit's convolution states after the first `rows` rows of the last decode forward of several
    void conv_commit(int rows) {
        const Geometry& g = m->g;
        const int KV = (int) g.kv_width(), N = (int) g.n_embd, per = 2 * KV + 2 * N;
        ik::conv_commit_kernel<<<dim3((unsigned) ((per + 255) / 256), (unsigned) visits.size()), 256, 0, st>>>(
            conv_all, win_raw, KV, N, (int) g.n_conv, rows, kMaxT);
    }
    /// The window's first `a` rows stay: the sequence ends after them
    void keep(int a) {
        if (win_p0 < 0) return;
        if (win_T > 1) conv_commit(a);
        n_pos = win_p0 + a;
        win_p0 = -1;
        if (sp != nullptr) {   // (the rows after: their context cells are written over by the next forward's)
            sp_vhi = n_pos;
            frnn_n = std::min(frnn_n, a);
        }
    }
    /// T tokens at the sequence's end through every visit; the logits of the last n_head rows.  False: no cells left.
    /// `window`: the rows stay as far as keep() says (a token and its drafts).
    bool forward(const int32_t* tok, int T, int n_head, bool window = false) {
        const int64_t p0 = pos();
        // (the cells grow, and move to the cards, where a forward of kMaxT rows would make them: at the same positions
        // whatever the forwards' rows)
        if (p0 + T > ctx || !ensure(std::min<int64_t>(p0 + kMaxT, ctx))) return false;
        const bool waits = window && T > 1;   // (a window of one row: its states as any forward's)
        const int key = T * 16 + n_head + (waits ? 256 : 0);
        CK(cudaStreamSynchronize(st));   // the last forward has read the pinned rows
        prof_start();
        embed(D, tok, T, p0);
        steps(D, p0, T);
        if (sp != nullptr) rope_rows(D.f_cs_h, p0, T);
        far_mode = lo > 0;
        moe_cuts = m->in_ram && n_held < (int64_t) m->ranked.size();
        if (moe_cuts) adapt();
        if (!use_graphs) {
            launches(D, T, n_head, p0, !waits);
            prof_collect();
        } else if (far_mode || moe_cuts) {   // cells on cards, experts in RAM: the graph in segments around their work
            std::vector<Seg>& segs = seg_graphs[key + (far_mode ? 512 : 0) + (moe_cuts ? 1024 : 0)];
            if (segs.empty()) {
                cur_segs = &segs;
                capturing = true;
                CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
                launches(D, T, n_head, p0, !waits);
                cudaGraph_t graph = nullptr;
                Seg last;
                CK(cudaStreamEndCapture(st, &graph));
                CK(cudaGraphInstantiate(&last.exec, graph, 0));
                CK(cudaGraphDestroy(graph));
                own(&Owned::graphs, last.exec);
                capturing = false;
                segs.push_back(last);
                CK(cudaGraphLaunch(last.exec, st));
            } else {
                for (Seg& sg : segs) {
                    if (sg.moe >= 0) *bell_h = -1;
                    CK(cudaGraphLaunch(sg.exec, st));
                    if (sg.moe >= 0) moe_host(sg.moe, T);
                    if (sg.fi < 0) continue;
                    CK(cudaStreamSynchronize(st));
                    cards_scan(sg.fi, T);
                    scan(sg.at, T, st);
                    (void) cudaStreamQuery(st);   // (this card's scan runs beside theirs)
                    cards_wait();
                }
            }
            if (n_head > 0) lg_rows = &D;
        } else {
            cudaGraphExec_t& exec = graphs[key];
            if (exec == nullptr) {
                cudaGraph_t graph = nullptr;
                CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
                launches(D, T, n_head, p0, !waits);
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
        win_p0 = window ? p0 : -1;
        win_T = T;
        spark_wrote(p0, T, false);
        return true;
    }
    /// T tokens of a prompt as one chunk (T <= chunk_cap); the last one's logits with `logits`.  False: no cells left.
    bool chunk(const int32_t* tok, int T, bool logits) {
        const int64_t p0 = pos();
        if (p0 + T > ctx || !ensure(p0 + T)) return false;
        if (m->in_ram && !scratch_fit()) return false;
        CK(cudaStreamSynchronize(st));
        prof_start();
        embed(P, tok, T, p0);
        steps(P, p0, T);
        if (sp != nullptr) rope_rows(P.f_cs_h, p0, T);
        far_mode = false;
        launches(P, T, logits ? 1 : 0, p0, true);
        prof_collect();
        n_pos += T;
        logits_held = false;
        spark_wrote(p0, T, true);
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
            if (prefill > 0 && left >= chunk_min) {
                chunk_fit(left);
                T = (int) std::min<int64_t>(chunk_cap, left);
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

    // ---- the request's sampler: Flash-Next's chain on the device (lamina/kernels/sampler.hpp), a token from each
    // head row.  A row's draw is Philox(seed, the position its token takes), so a window's rows draw what a row at a
    // time would.
    lamina::kernels::SamplerParams smp;
    static constexpr int kHistCap = 4096;   // the penalties' window at most
    int hist_n = 0;                         // its tokens (0: no penalties)
    int32_t* hist_d = nullptr;              // [kMaxT][kHistCap], made by the first request with penalties
    std::vector<int32_t> hist_stage;
    void sampling(const lamina::program::Sampling& q) {
        smp = {};
        smp.top_k = q.top_k;
        smp.top_p = q.top_p;
        smp.min_p = q.min_p;
        smp.temperature = q.temperature;
        smp.greedy = q.temperature <= 0.f;
        smp.penalty_repeat = q.repeat;
        smp.penalty_freq = q.freq;
        smp.penalty_present = q.present;
        const bool pen = q.repeat != 1.f || q.freq != 0.f || q.present != 0.f;
        hist_n = !pen || q.last_n == 0 ? 0 : q.last_n < 0 ? kHistCap : std::min(q.last_n, kHistCap);
        smp.penalty_last_n = hist_n;
        smp.seed = q.seed ? q.seed : (unsigned long long) Clock::now().time_since_epoch().count();
        if (hist_n > 0 && hist_d == nullptr) {
            hist_d = dzero<int32_t>((size_t) kMaxT * kHistCap);
            hist_stage.assign((size_t) kMaxT * kHistCap, -1);
        }
    }
    /// A token from each of the last head's first `rows` rows, into out: row t's is the one at position `at + t` (its
    /// draw's number) and follows tail[0..n_tail), then win[0..t] (the penalties' history; `win` null: the tail alone,
    /// one row).  -1 where a row's logits hold a non-finite one.
    void pick(int rows, int64_t at, const int32_t* tail, int64_t n_tail, const int32_t* win, int32_t* out) {
        Rows& R = *lg_rows;
        const int32_t* hist = nullptr;
        if (hist_n > 0) {
            if (win != nullptr) lamina::kernels::penalty_rows(tail, n_tail, win, rows, hist_n, hist_stage.data());
            else lamina::kernels::penalty_rows(tail, n_tail - 1, tail + n_tail - 1, 1, hist_n, hist_stage.data());
            CK(cudaMemcpyAsync(hist_d, hist_stage.data(), (size_t) rows * (size_t) hist_n * 4, cudaMemcpyHostToDevice, st));
            hist = hist_d;
        }
        smp.counter = (uint64_t) at;
        lamina::kernels::sample_tokens(R.logits, rows, (int) m->g.n_vocab, hist, hist_n, smp, R.pick, st);
        const int Th = R.head_rows();
        CK(cudaMemcpyAsync(R.pick_h, R.pick, (size_t) 2 * Th * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        for (int t = 0; t < rows; ++t) out[t] = R.pick_h[Th + t] != 0 ? -1 : R.pick_h[t];
    }

    // ---- the sequence's start, and the state to and from the prompt cache's buffers
    void reset() {
        CK(cudaStreamSynchronize(st));
        cards_clear();
        CK(cudaMemsetAsync(conv_all, 0, conv_bytes(), st));
        n_pos = 0;
        logits_held = false;
        sp_vlo = sp_vhi = sp_whi = 0;
        frnn_n = 0;
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
        if (held == 1)
            ok = ok && soft(cudaMemcpyAsync(D.logits, host + body, V4, cudaMemcpyHostToDevice, st)) &&
                 soft(cudaMemsetAsync(D.bad, 0, 4, st));
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
        const size_t mine = (size_t) held * cell_bytes;
        if (cards.empty()) {
            const size_t w = (size_t) cells * cell_bytes;
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
        // with cell cards.  Out: each unit from the card that holds it, the rest from this card.  In: a unit at a time
        // through this card as a sequence that grows (ensure moves what it has no room for to the cards again).
        auto array = [&](const Visit& V, int side) {   // an array's place in the copy
            return host + ((size_t) full_of[(size_t) (&V - visits.data())] * 2 + (size_t) side) * mine;
        };
        CK(cudaStreamSynchronize(st));
        if (to_host) {
            for (size_t u = 0; u < unit_card.size() && (int64_t) u * unit < cells; ++u) {
                Card& c = cards[(size_t) unit_card[u]];
                const int64_t slot = unit_slot(u);
                const size_t n = (size_t) std::min<int64_t>(unit, cells - (int64_t) u * unit) * cell_bytes;
                for (Visit& V : visits)
                    if (!V.cells.ring)
                        for (int side = 0; side < 2; ++side)
                            card_copy(c, (size_t) full_of[(size_t) (&V - visits.data())], side, slot,
                                      array(V, side) + (size_t) u * unit * cell_bytes, n, true);
            }
            bool ok = true;
            if (cells > lo)
                for (Visit& V : visits)
                    if (!V.cells.ring)
                        for (int side = 0; side < 2; ++side)
                            ok = ok && soft(cudaMemcpyAsync(array(V, side) + (size_t) lo * cell_bytes,
                                                            (side ? V.cells.v : V.cells.k) + (size_t) lo * cell_bytes,
                                                            (size_t) (cells - lo) * cell_bytes, cudaMemcpyDeviceToHost, st));
            return soft(cudaStreamSynchronize(st)) && ok;
        }
        cards_clear();
        const int64_t was = n_pos;
        bool ok = true;
        for (int64_t c0 = 0; c0 < cells && ok; c0 += unit) {
            const int64_t n = std::min<int64_t>(unit, cells - c0);
            n_pos = c0;
            ok = ensure(c0 + n);
            for (Visit& V : visits)
                if (ok && !V.cells.ring)
                    for (int side = 0; side < 2; ++side)
                        ok = ok && soft(cudaMemcpyAsync((side ? V.cells.v : V.cells.k) + (size_t) c0 * cell_bytes,
                                                        array(V, side) + (size_t) c0 * cell_bytes, (size_t) n * cell_bytes,
                                                        cudaMemcpyHostToDevice, st));
            ok = soft(cudaStreamSynchronize(st)) && ok;
        }
        n_pos = ok ? cells : was;
        return ok;
    }

    void prof_print(const char* what) {
        if (!prof || prof_ms.empty()) return;
        double total = 0;
        for (const auto& [k, v] : prof_ms) total += v;
        std::fprintf(stderr, "lamina: %s, %.1f ms in the steps:\n", what, total);
        for (const auto& [k, v] : prof_ms) std::fprintf(stderr, "lamina:   %-32s %9.1f ms %5.1f%%\n", k.c_str(), v, 100.0 * v / total);
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

// ------------------------------------------------------------------------------------------------ drafts
/// A window's drafts: the tokens that may follow x, the token the state does not hold yet.  Prompt lookup
/// (--spec-lookup N, lamina/spec/suffix_drafter.hpp): where the text's last N tokens or more occurred before, the
/// tokens that followed them there.
struct Drafts {
    Engine* E = nullptr;   // with a drafter: its drafts where the lookup has none
    int per_step = 0;      // the drafter's drafts a window at most (--drafts-per-step)
    float min_p = 0.f;     // while their joint probability under its head stays at or above this (--spec-min-p)
    int lookup = 0;   // prompt lookup's shortest match (0: none)
    // LAMINA_INK_WINDOW=N (a test's): every window N drafts that are the token itself, so a window of N + 1 rows is
    // timed at any context (its drafts stay only where the text repeats a token)
    int fake = std::getenv("LAMINA_INK_WINDOW") ? std::max(0, std::atoi(std::getenv("LAMINA_INK_WINDOW"))) : 0;
    std::unique_ptr<lamina::spec::SuffixDrafter> sfx;
    size_t sfx_ok = 0;   // the lookup's history is the state's tokens, its first sfx_ok of them right
    // a run's count: windows, those with drafts, those from the lookup, the drafts offered and kept
    long long windows = 0, drafted = 0, looked = 0, offered = 0, kept = 0;
    bool on() const { return lookup > 0 || fake > 0 || (E != nullptr && per_step > 0); }
    void init(int lookup_min, int64_t ctx) {
        lookup = lookup_min;
        if (lookup > 0) sfx = std::make_unique<lamina::spec::SuffixDrafter>(lookup, 32, (size_t) std::max<int64_t>(4096, ctx));
    }
    long long blocks0 = 0;
    double ms0 = 0;
    void begin() {
        windows = drafted = looked = offered = kept = 0;
        if (E != nullptr) {
            blocks0 = E->sp_blocks;
            ms0 = E->sp_ms;
        }
    }
    /// Up to `n` drafts after x, which follows `seq` (the state's tokens)
    std::vector<int32_t> propose(const std::vector<int32_t>& seq, int32_t x, int n) {
        std::vector<int32_t> out;
        ++windows;
        if (fake > 0) {
            out.assign((size_t) std::min(n, fake), x);
            offered += (long long) out.size();
            return out;
        }
        if (n <= 0) return out;
        if (sfx) {
            // the history up to where it and seq agree: seq's tokens before sfx_ok are the ones it was given
            sfx_ok = std::min(sfx_ok, seq.size());
            if (sfx->size() > sfx_ok) sfx->truncate(sfx_ok);
            if (seq.size() > sfx_ok) {
                sfx->append(seq.data() + sfx_ok, seq.size() - sfx_ok);
                sfx_ok = seq.size();
            }
            sfx->append(x);   // not in the state yet: the next call takes it out
            out.resize((size_t) n);
            out.resize((size_t) std::max(0, sfx->propose(n, out.data())));
            if (!out.empty()) {
                ++drafted;
                ++looked;
                offered += (long long) out.size();
                return out;
            }
        }
        if (E != nullptr && per_step > 0) {
            out = E->spark_draft(x, std::min(n, per_step), min_p);
            if (!out.empty()) {
                ++drafted;
                offered += (long long) out.size();
            }
        }
        return out;
    }
    /// The state's tokens are another text from `from` on (a request's start)
    void rewound(size_t from) { sfx_ok = std::min(sfx_ok, from); }
    std::string summary() const {
        if (!on() || windows == 0) return {};
        char b[300];
        int at = std::snprintf(b, sizeof b, "; %lld windows, %lld with drafts (%lld from the prompt lookup), %lld of %lld "
                               "drafts kept", windows, drafted, looked, kept, offered);
        if (E != nullptr && E->sp_blocks > blocks0)
            std::snprintf(b + at, sizeof b - (size_t) at, ", %lld blocks of the drafter at %.2f ms", E->sp_blocks - blocks0,
                          (E->sp_ms - ms0) / (double) (E->sp_blocks - blocks0));
        return b;
    }
};

/// A run of windows after a prompt's read: the first token from the logits the read left, then each window the last
/// token and its drafts, every row's token picked by the request's sampler and a draft kept while it is the token
/// picked.  `emit(token)` takes each token in order: false, and the token ends the run (it stays out of the state).
/// `seq`: the state's tokens, appended to as rows stay (`stays` is told each); `hist`: the penalties' history, the
/// prompt's text and every token emitted.  `fixed` (a test's): the run's tokens are these whatever the sampler picks,
/// so two settings' windows are timed along one text.  Returns "" where emit ended it, else why it ended: "nonfinite"
/// (the state holds the row), "length" (no cells left, or the fixed text's end).
template <class Emit>
const char* run_windows(Engine& E, Drafts& dr, std::vector<int32_t>& seq, std::vector<int32_t>& hist, long long max_new,
                        Emit&& emit, const std::function<void(int32_t)>& stays, const std::vector<int32_t>* fixed = nullptr) {
    int32_t picks[kMaxT];
    long long produced = 0;
    auto the_text = [&](int rows) {   // row t's token is the text's token `produced + t`
        if (fixed == nullptr) return;
        for (int t = 0; t < rows; ++t)
            picks[t] = produced + t < (long long) fixed->size() ? (*fixed)[(size_t) (produced + t)] : -2;
    };
    E.pick(1, E.pos(), hist.data(), (int64_t) hist.size(), nullptr, picks);
    the_text(1);
    int32_t x = picks[0];
    for (;;) {
        if (x == -2) return "length";
        if (x < 0) return "nonfinite";
        ++produced;
        hist.push_back(x);
        if (!emit(x)) return "";
        // the window: x and the drafts the run has room for
        const int room = (int) std::min<long long>({(long long) kMaxT - 1, max_new - produced, (long long) (E.ctx - E.pos() - 1)});
        std::vector<int32_t> win{x};
        const std::vector<int32_t> drafts = dr.on() ? dr.propose(seq, x, room) : std::vector<int32_t>();
        win.insert(win.end(), drafts.begin(), drafts.end());
        const int T = (int) win.size();
        const int64_t p0 = E.pos();
        if (!E.forward(win.data(), T, T, true)) return "length";
        E.pick(T, p0 + 1, hist.data(), (int64_t) hist.size() - 1, win.data(), picks);   // (the tail: the text before x)
        the_text(T);
        seq.push_back(x);
        if (stays) stays(x);
        int rows = 1;   // the window's rows that stay
        bool ended = false;
        int32_t next = picks[0];
        for (int k = 0; k < (int) drafts.size(); ++k) {
            if (picks[k] != drafts[(size_t) k]) break;   // (a non-finite row's -1 is no draft: the run ends on it)
            // draft k is the token picked: emitted, and its row gives the next one
            ++dr.kept;
            ++produced;
            hist.push_back(drafts[(size_t) k]);
            if (!emit(drafts[(size_t) k])) {
                ended = true;
                break;
            }
            seq.push_back(drafts[(size_t) k]);
            if (stays) stays(drafts[(size_t) k]);
            ++rows;
            next = picks[k + 1];
        }
        E.keep(rows);
        if (ended) return "";
        x = next;
    }
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
        E.cards_rewind(pos);
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

    Drafts& dr;

    Serve(Engine& e, int32_t eos_id, const CacheOpts& co, Drafts& d) : E(e), eos(eos_id), copt(co), dr(d) {
        if (E.snap == nullptr) {
            std::fprintf(stderr, "lamina serve: prompt cache: off (the held conversation continues when a request extends "
                         "it)\n");
            return;
        }
        cache = std::make_unique<Cache>(E, copt);
        cache->whole_prompt = true;   // (a prompt's end is a checkpoint with its logits)
        cache->keep_ends = true;      // (and a turn's end: it stays when the next prompt's comes)
        // (with cell cards a restored conversation finds its room a unit at a time: Engine::kv_copy)
        cache->ensure_kv = [this](int64_t cells) { return E.cards.empty() ? E.ensure(cells) : cells <= E.ctx; };
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

    // The server's image encoder runs in this process and, where the server has it borrow (its "vision" section's
    // "lend"), holds no VRAM of its own between pictures: for a new picture it asks for memory of the engine's card,
    // mapped here from the spare chunks and the VRAM free above the reserve, else from the experts (those in RAM
    // give way, as to the cells); given back, the experts take it again.
    std::unique_ptr<lamina::core::VmmRange> enc_vm;
    void enc_back() {
        if (!enc_vm) return;
        CK(cudaDeviceSynchronize());   // the encoder's last kernels
        for (int64_t c = 0; c < enc_vm->chunks(); ++c)
            if (enc_vm->mapped(c)) E.spare.push_back(enc_vm->unmap(c));
        enc_vm.reset();
        E.place();
        for (const lamina::core::VmmChunk h : E.spare) lamina::core::vmm_chunk_free(h);
        E.spare.clear();
    }
    void lend(const std::string& line, bool in_process) {
        long long dev = -1;
        unsigned long long bytes = 0;
        std::string why;
        const int64_t before = E.n_held;
        if (std::sscanf(line.c_str() + 5, "%lld %llu", &dev, &bytes) != 2 || bytes == 0) {
            why = "expected LEND <device> <bytes>";
        } else if (!in_process) {
            why = "a loan is memory of the engine's own process";
        } else if (enc_vm) {
            why = "a loan is out already";
        } else if (dev != E.main_dev) {
            why = "Inkling 2's engine lends on its own GPU alone (GPU " + std::to_string(E.main_dev) + ")";
        } else if (!lamina::core::vmm_available()) {
            why = "this driver has no CUDA virtual memory";
        } else {
            CK(cudaStreamSynchronize(E.st));
            auto vm = std::make_unique<lamina::core::VmmRange>();
            if (!vm->reserve(bytes)) {
                why = "no address range of " + std::to_string(bytes >> 20) + " MiB";
            } else if (!E.chunks_room(vm->chunks())) {
                why = "the GPU cannot spare " + std::to_string(bytes >> 20) + " MiB";
            } else if (vm->map_range(0, vm->chunks(), [&] { return E.take_spare(); })) {
                enc_vm = std::move(vm);
            } else {
                for (int64_t c = 0; c < vm->chunks(); ++c)
                    if (vm->mapped(c)) E.spare.push_back(vm->unmap(c));
                E.place();
                why = "the GPU has no " + std::to_string(bytes >> 20) + " MiB to map";
            }
        }
        if (why.empty()) {
            const long long out = (long long) (before - E.n_held);
            serve_print("LENT %llu %llu %lld %lld\n", (unsigned long long) (uintptr_t) enc_vm->base(),
                        (unsigned long long) enc_vm->chunks() * lamina::core::vmm_granularity(), out, out);
        } else {
            serve_print("ERR no loan: %s\n", why.c_str());
        }
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
    /// there), a few cells before the end of a last message that grows between requests, and the end of the block of
    /// the prompt's last picture.
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
        int64_t pic = 0;   // behind the prompt's last picture (its block's end): another question about it starts there
        for (int64_t i = n - 1; i > reuse; --i)
            if (ids[(size_t) i] < 0) {
                pic = i + 2;
                break;
            }
        std::vector<int64_t> want{fork, msg_end, pic};
        std::sort(want.begin(), want.end());
        std::vector<int64_t> marks;
        for (const int64_t p : want)
            if (p >= reuse + 2 && p <= n - 2 && (marks.empty() || p >= marks.back() + 2)) marks.push_back(p);
        return marks;
    }

    void request(const std::string& line, bool geni) {
        const Geometry& g = E.m->g;
        if (enc_vm) {   // a request while a loan is out: the chunks come back first
            std::fprintf(stderr, "lamina: a request came while the image encoder's loan was out: taken back\n");
            enc_back();
        }
        const char* p = line.c_str() + (geni ? 5 : 4);
        char* e = nullptr;
        const long long max_new = std::strtoll(p, &e, 10);
        p = e;
        lamina::program::Sampling smp;
        std::vector<uint64_t> img_keys;   // img_keys=<hex>,..: each picture's key from the server, in prompt order
        for (;;) {
            while (*p == ' ') ++p;
            const char* st0 = p;
            while (*p && *p != ' ') ++p;
            const std::string tk(st0, (size_t) (p - st0));
            const size_t eq = tk.find('=');
            if (tk.empty() || eq == std::string::npos) { p = st0; break; }
            if (tk.compare(0, eq, "img_keys") == 0) {
                for (const char* q = tk.c_str() + eq + 1; *q;) {
                    char* qe = nullptr;
                    const unsigned long long x = std::strtoull(q, &qe, 16);
                    if (qe == q || (*qe != ',' && *qe != '\0')) { img_keys.clear(); break; }
                    img_keys.push_back((uint64_t) x);
                    q = *qe == ',' ? qe + 1 : qe;
                }
            } else {
                smp.set(tk.substr(0, eq), tk.c_str() + eq + 1);   // keys of other paths are left
            }
        }
        std::string emb;   // GENI: the request's file of picture rows before the ids
        if (geni) {
            while (*p == ' ') ++p;
            const char* e0 = p;
            while (*p && *p != ' ') ++p;
            emb.assign(e0, (size_t) (p - e0));
        }
        std::vector<int32_t> ids = lamina::program::parse_ids(p);
        if (max_new < 1 || ids.empty() || (geni && emb.empty())) {
            serve_print("ERR bad request: max_new or the ids\n");
            return;
        }
        for (const int32_t t : ids)
            if (t < 0 || t >= g.n_vocab) { serve_print("ERR a token id is outside the vocabulary\n"); return; }
        ImgReq ir;
        if (geni) {
            if (!E.vision) { serve_print("ERR this engine was started without --vision\n"); return; }
            if (const std::string ve = build_images(emb, ids, E.m->image_pad, g.n_embd, img_keys, ir); !ve.empty()) {
                serve_print("ERR %s\n", ve.c_str());
                return;
            }
        }
        struct Pictures {   // the engine reads this request's rows until it ends
            Engine& e;
            ~Pictures() { e.img = nullptr; }
        } pictures{E};
        E.img = geni ? &ir : nullptr;
        if ((int64_t) ids.size() + max_new > E.ctx) {
            serve_print("ERR prompt (%zu tokens) + max_new (%lld) exceeds the context (%lld)\n", ids.size(), max_new,
                        (long long) E.ctx);
            return;
        }
        stop.store(false);
        if (E.lo > 0) E.clocks.hold();   // (cells on cards: their clocks, where they are locked for a request)
        struct Release {
            Engine& e;
            ~Release() {
                if (e.lo > 0) e.clocks.release_after(e.clocks_hold);
            }
        } release{E};
        // ---- what the state continues from: with the cache the longest start of the prompt it holds (the
        // conversation's end, a checkpoint of it, a conversation parked in RAM), the engine's state set there; without
        // it the held sequence where the prompt extends it (the windows' state cannot step back)
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
        {   // (the lookup's history follows the state's tokens: right as far as the last ones go on in these)
            size_t same = 0;
            while (same < live.size() && same < reuse && live[same] == ids[same]) ++same;
            dr.rewound(same);
            if (same != live.size() || reuse != live.size()) E.spark_cut((int64_t) same);   // (else the state goes on as it is)
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
        // the prompt's end, before the first decode step moves the state (not after a read of one token - a stopped
        // answer going on: the conversation keeps the end of the prompt it was asked with); its logits are the read's,
        // or the ones a prompt's end brought back
        if (cache && fresh > 1) checkpoint((int64_t) ids.size(), true);
        E.sampling(smp);
        dr.begin();
        std::vector<int32_t> hist;   // the penalties' history: the prompt's text (not a picture's cells)
        for (const int32_t t : ids)
            if (t >= 0) hist.push_back(t);
        const auto d0 = Clock::now();
        long long produced = 0;
        const char* finish = "length";
        const char* const why = run_windows(E, dr, live, hist, max_new, [&](int32_t t) {
            serve_print("T %d\n", t);
            ++produced;
            if (t == eos) finish = "stop";
            else if (stop.load()) finish = "cancel";
            else return produced < max_new;
            return false;
        }, [&](int32_t t) {
            if (cache) cache->append(0, t);
        });
        if (*why) finish = why;   // "nonfinite", or "length": the context's end, or no VRAM for more cells
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
        serve_print("DONE %lld %zu %.1f %.1f %s %lld %lld %zu 0 0 0 0 0.0 %lld 0\n", produced, ids.size(), prompt_ms,
                    decode_ms, finish, dr.kept, dr.offered, reuse, (long long) fresh);
        std::fprintf(stderr, "lamina serve: %zu prompt tokens (%zu reused) in %.1f ms (%.1f tok/s), %lld generated in %.1f "
                     "ms (%.2f tok/s), %s%s%s; MiB of VRAM free: %s\n", ids.size(), reuse, prompt_ms,
                     prompt_ms > 0 ? 1000.0 * (double) fresh / prompt_ms : 0.0, produced, decode_ms,
                     decode_ms > 0 ? 1000.0 * (double) produced / decode_ms : 0.0, finish, dr.summary().c_str(),
                     E.ram_summary().c_str(), E.vram_free().c_str());
        if (cache) {
            std::fprintf(stderr, "lamina serve: prompt cache: %s\n", cache->summary().c_str());
            cache->spare();
        }
        E.chunk_base();
        E.relax();
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
                request(line, false);
            } else if (line.rfind("GENI ", 0) == 0) {
                request(line, true);
            } else if (line.rfind("LEND ", 0) == 0) {
                lend(line, io.in_process);
            } else if (line == "UNLEND") {
                enc_back();
                serve_print("UNLENT\n");
            } else if (line == "QUIT") {
                std::lock_guard<std::mutex> lk(mu);
                eof = true;
                lines.clear();
            } else {
                serve_print("ERR expected: GEN <max_new> [key=value ...] <id,id,...>, or GENI <max_new> [key=value ...] "
                            "<file> <id,id,...>\n");
            }
        }
    }
};

int run_engine(int argc, char** argv, bool check_only) {
    std::string model, tokens, force, dump_full, emb_file, kv = "fp16";
    bool vision = false;
    int64_t max_ctx = 0;
    int n_gen = 64, threads = 8;
    bool serve = false, file_cache = true, kv_grow = true, graphs = true;
    int prefill = 1024, prefill_max = -1;
    std::optional<uint64_t> reserve;
    std::string cell_gpus, cell_clocks;
    double clocks_hold = 1.0;
    int64_t main_cells = 0, cell_spread = 0;
    int lookup = 16, drafts = -1, cpu_threads = 0;
    uint64_t expert_vram = 0;
    std::string expert_profile, expert_save;
    double expert_save_every = 1.0;
    float spec_min_p = 0.7f;
    std::string dspark;
    int64_t dspark_reach = 32768;
    lamina::program::Sampling gen_smp;   // generate mode's sampling (greedy)
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
        else if (a == "--prefill-max") prefill_max = std::max(0, std::atoi(next()));   // rows it grows to while VRAM is free
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
        else if (a == "--vision") vision = true;    // pictures: GENI requests (their rows the image encoder's)
        else if (a == "--emb") emb_file = next();   // generate mode: --tokens' pictures, a file of the encoder's rows
        else if (a == "--kv") kv = next();   // the cells: fp16 or q8_0
        else if (a == "--kv-grow") kv_grow = true;
        else if (a == "--no-kv-grow") kv_grow = false;   // the full visits' cells allocated whole at the start
        else if (a == "--file-cache") file_cache = true;
        else if (a == "--no-file-cache") file_cache = false;   // the model file read past the OS file cache
        else if (a == "--load-threads") threads = std::max(1, std::atoi(next()));
        else if (a == "--vram-reserve-mib") reserve = (uint64_t) std::max(0, std::atoi(next())) << 20;
        else if (a == "--cell-gpus") cell_gpus = next();   // cards for the full visits' cells (CUDA ordinals)
        else if (a == "--cell-gpu-clocks") cell_clocks = next();   // their clocks locked while a request runs
        else if (a == "--clocks-hold") clocks_hold = std::max(0.0, std::atof(next()));
        else if (a == "--main-cells") main_cells = std::max<int64_t>(0, std::atoll(next()));   // (tests: fewer on this card)
        else if (a == "--cell-spread") cell_spread = std::max<int64_t>(0, std::atoll(next()));
        else if (a == "--spec-lookup") lookup = std::max(0, std::atoi(next()));   // prompt lookup's match (0: none)
        else if (a == "--dspark") dspark = next();   // the model's block drafter (dspark-*.gguf)
        else if (a == "--expert-vram") expert_vram = (uint64_t) (std::max(0.0, std::atof(next())) * 1073741824.0);   // (in GiB, as the hy4 path's)
        else if (a == "--expert-cache") {   // the experts' VRAM at most: auto (what the card has), or a size (6G, 9500M)
            const std::string v = next();
            if (v != "auto") {
                char* end = nullptr;
                const double x = std::strtod(v.c_str(), &end);
                const char u = end != nullptr ? (char) std::toupper((unsigned char) *end) : 0;
                const double unit = u == 'G' ? 1073741824.0 : u == 'M' ? 1048576.0 : u == 'K' ? 1024.0 : u == 'B' ? 1.0 : 0.0;
                if (end == v.c_str() || x <= 0 || unit == 0) {
                    std::fprintf(stderr, "lamina: --expert-cache takes auto or a size (6G, 9500M) on this path\n");
                    return 2;
                }
                expert_vram = (uint64_t) (x * unit);
            }
        }
        else if (a == "--expert-profile") expert_profile = next();   // the ranking the experts are placed by
        else if (a == "--expert-profile-save") expert_save = next();   // what the routing made of it, written there
        else if (a == "--expert-profile-save-every") expert_save_every = std::max(0.0, std::atof(next()));   // minutes (0: at the end)
        else if (a == "--cpu-threads" || a == "--pool-workers") cpu_threads = std::max(0, std::atoi(next()));
        else if (a == "--drafts-per-step") drafts = std::max(0, std::min(kMaxT - 1, std::atoi(next())));
        else if (a == "--spec-min-p") spec_min_p = (float) std::atof(next());
        else if (a == "--dspark-reach") dspark_reach = std::max<int64_t>(0, std::atoll(next()));
        else if (a == "--temperature") gen_smp.temperature = (float) std::atof(next());
        else if (a == "--top-p") gen_smp.top_p = (float) std::atof(next());
        else if (a == "--top-k") gen_smp.top_k = std::atoi(next());
        else if (a == "--min-p") gen_smp.min_p = (float) std::atof(next());
        else if (a == "--seed") gen_smp.seed = std::strtoull(next(), nullptr, 10);
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
    LoadOpts lo;
    lo.threads = threads;
    lo.file_cache = file_cache;
    // (what the server's image encoder will borrow on this card for a picture: experts in RAM give way to it; a card
    // that holds every expert keeps it free of cells)
    int main_dev = 0;
    CK(cudaGetDevice(&main_dev));
    const uint64_t loan = lamina::core::gpu_loan_asked(main_dev);
    lo.reserve = (size_t) (*reserve + loan);
    lo.expert_vram = expert_vram;
    lo.profile = expert_profile;
    lo.max_ctx = max_ctx;
    lo.q8 = kv == "q8_0";
    lo.cache = serve && copt.chain > 0;
    lo.cell_cards = !cell_gpus.empty();
    lo.prefill = prefill;
    lo.ram_reserve = (uint64_t) copt.ram_reserve_mib << 20;
    if (!load(model, m, lo, err)) {
        std::fprintf(stderr, "lamina: %s\n", err.c_str());
        return 1;
    }
    Spark spark;
    if (!dspark.empty() && drafts != 0) {
        // The drafter is VRAM the cells do not get (its file, and ~0.35 GiB of cells and buffers): where this card
        // alone holds the context and would hold under the floor with it, it is left out.
        bool room = true;
        if (cell_gpus.empty()) {
            size_t fr = 0, tot = 0;
            CK(cudaMemGetInfo(&fr, &tot));
            std::error_code ec;
            const uint64_t file = (uint64_t) std::filesystem::file_size(std::filesystem::u8path(dspark), ec);
            const Geometry& g = m.g;
            const uint64_t cell = kv == "q8_0" ? (uint64_t) g.n_head_kv * (uint64_t) (g.head_dim / 32) * 34 : (uint64_t) g.kv_width() * 2;
            const uint64_t ring = (uint64_t) (g.n_visits() - g.n_full_visits()) * 2 * cell * (uint64_t) (g.n_swa + kMaxT + kRingSub);
            const uint64_t want = (uint64_t) (max_ctx > 0 ? std::min<int64_t>(max_ctx, kContextFloor) : kContextFloor) *
                                  (uint64_t) g.n_full_visits() * 2 * cell;
            // (a chunk of 128 rows at least, the rings and the cache's image of them, the graphs)
            const uint64_t rest = ring * (serve && copt.chain > 0 ? 2 : 1) + (prefill > 0 ? 230ull << 20 : 0) + (64ull << 20);
            room = ec || file + (350ull << 20) + want + rest + *reserve <= fr;
            if (!room)
                std::fprintf(stderr, "lamina: the drafter is left out: beside it this GPU's VRAM would hold a context under "
                             "%lld tokens\n", (long long) (want / ((uint64_t) g.n_full_visits() * 2 * cell)));
        }
        if (room) {
            if (!load_spark(dspark, m, spark, err)) {
                std::fprintf(stderr, "lamina: %s\n", err.c_str());
                return 1;
            }
            std::fprintf(stderr, "lamina: the drafter (%d layers, a block of %d rows): %.2f GiB on the GPU, loaded in %.1f s\n",
                         spark.n_layer, spark.block, spark.gib, spark.load_s);
        }
    }
    Engine E;
    E.vision = vision || !emb_file.empty();
    if (E.vision && m.image_pad < 0) {
        std::fprintf(stderr, "lamina: the model's chat template names no token for a picture's rows, so --vision has no "
                             "cells to fill\n");
        return 2;
    }
    E.cpu_threads = cpu_threads;
    E.profile_save = expert_save;
    E.profile_every = expert_save_every > 0 ? expert_save_every * 60.0 : 1e18;
    if (!spark.layers.empty()) E.sp = &spark;
    E.sp_reach = dspark_reach;
    E.use_graphs = graphs;
    E.cache_on = serve && copt.chain > 0;
    for (const int d : lamina::program::parse_ids(cell_gpus.c_str())) E.cell_devs.push_back(d);
    E.main_limit = main_cells;
    E.spread = cell_spread;
    E.clocks_hold = clocks_hold;
    for (size_t at = 0; !cell_clocks.empty() && at <= cell_clocks.size();) {
        const size_t comma = std::min(cell_clocks.find(',', at), cell_clocks.size());
        E.cell_clocks.push_back(cell_clocks.substr(at, comma - at));
        at = comma + 1;
    }
    E.prefill = prefill;
    if (prefill_max >= 0) E.prefill_max = prefill_max;
    if (loan > 0 && !m.in_ram)
        std::fprintf(stderr, "lamina: %llu MiB of this GPU stay free of cells for the image encoder's loan\n",
                     (unsigned long long) (loan >> 20));
    E.init(m, max_ctx, kv == "q8_0", kv_grow, (size_t) (*reserve + (m.in_ram ? 0 : loan)));
    Drafts dr;
    dr.init(lookup, E.ctx);
    if (E.sp != nullptr) {
        dr.E = &E;
        dr.per_step = drafts < 0 ? kMaxT - 1 : drafts;
        dr.min_p = spec_min_p;
    }
    if (serve) {
        Serve sv(E, m.eos, copt, dr);
        const int rc = sv.run();
        E.profile_write(true);
        return rc;
    }
    std::vector<int32_t> prompt = lamina::program::parse_ids(tokens.c_str());
    const std::vector<int32_t> forced = lamina::program::parse_ids(force.c_str());
    if (prompt.empty()) { std::fprintf(stderr, "lamina: --tokens is required\n"); return 2; }
    if (!forced.empty()) n_gen = (int) forced.size() + 1;
    ImgReq ir;
    if (!emb_file.empty()) {
        if (const std::string ve = build_images(emb_file, prompt, m.image_pad, m.g.n_embd, {}, ir); !ve.empty()) {
            std::fprintf(stderr, "lamina: --emb: %s\n", ve.c_str());
            return 2;
        }
        E.img = &ir;
    }
    for (const std::vector<int32_t>* v : {(const std::vector<int32_t>*) &prompt, &forced})
        for (const int32_t t : *v)
            if ((t < 0 && (v != &prompt || ir.cell_row.empty())) || t >= m.g.n_vocab) {
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
    CK(cudaStreamSynchronize(E.st));
    const double pp_ms = ms_since(t0);
    E.prof_print("the prompt");
    const auto t1 = Clock::now();
    std::vector<int32_t> gen;
    const bool along = !forced.empty() && ff == nullptr && dr.on();   // the windows timed along a fixed text
    if (!along && (ff != nullptr || !forced.empty())) {   // a reference's run: its tokens a row at a time, each row's logits kept
        std::vector<float> lg;
        E.host_logits(0, lg);
        for (int i = 0; i < n_gen; ++i) {
            const int a = argmax(lg);
            gen.push_back(a);
            if (ff) std::fwrite(lg.data(), 4, lg.size(), ff);
            if (i + 1 == n_gen) break;
            const int32_t nx = i < (int) forced.size() ? forced[(size_t) i] : a;
            if (nx < 0) break;
            if (dr.fake > 0) {   // (a test's: the token as a window's first row, its drafts dropped)
                const std::vector<int32_t> win((size_t) std::min<int64_t>(dr.fake + 1, E.ctx - E.pos()), nx);
                if (win.empty() || !E.forward(win.data(), (int) win.size(), (int) win.size(), true)) break;
                E.host_logits(0, lg);
                E.keep(1);
                continue;
            }
            if (!E.forward(&nx, 1, 1)) break;
            E.host_logits(0, lg);
        }
    } else {   // as a request's: the sampler's tokens (greedy unless --temperature), windows of a token and its drafts
        E.sampling(gen_smp);
        dr.begin();
        std::vector<int32_t> seq = prompt, hist;   // (the penalties' history: the prompt's text, not a picture's cells)
        for (const int32_t t : prompt)
            if (t >= 0) hist.push_back(t);
        run_windows(E, dr, seq, hist, along ? (long long) forced.size() : n_gen, [&](int32_t t) {
            gen.push_back(t);
            return (int) gen.size() < (along ? (int) forced.size() : n_gen);
        }, nullptr, along ? &forced : nullptr);
    }
    const double tg_ms = ms_since(t1);
    E.prof_print("generation");
    if (ff) std::fclose(ff);
    serve_print("gen:");
    for (int t : gen) serve_print(" %d", t);
    serve_print("\n");
    std::fprintf(stderr, "lamina: prompt %zu tokens in %.1f ms (%.1f tok/s), %zu generated in %.1f ms (%.2f tok/s)%s%s; MiB of "
                 "VRAM free: %s\n", prompt.size(), pp_ms, prompt.size() / (pp_ms / 1000.0), gen.size(), tg_ms,
                 gen.size() / (tg_ms / 1000.0), dr.summary().c_str(), E.ram_summary().c_str(), E.vram_free().c_str());
    E.profile_write(true);
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
