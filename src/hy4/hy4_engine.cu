// src/hy4/hy4_engine.cu - the engine path of Hy4-flash (GGUF architecture hy_v4_flash), a driver of its own beside
// Flash-Next's and Qwen3.6's (no layer is theirs), built from the engine's kernels and src/hy4/hy4_kernels.cuh:
//
//   * the model as lamina/hy4/geometry.hpp reads it (llama.cpp's PR #32517 is the reference: done 2026-11-18 Hy4-flash
//     support assessed); dense weights on the main GPU in their GGUF blocks (Q8_0 through native_mmvq, the BF16 head
//     through the MMVF GEMV, F32 vectors and small matrices as they are); the routed experts in host memory as native
//     blobs [gate | up | down] for the CPU pool (ggml-cpu's arithmetic, llama.cpp's CPU backend's);
//   * T <= kMaxT rows at consecutive positions through every layer: the four iHC streams, Engram's rows (the table
//     read from the SSD by PleReader) before its layers, gated MLA with sinks over the window's ring of cells or, on
//     the sparse layers, over the cells the indexer picks, the dense FFN of layer 0, the latent MoE elsewhere; the
//     head after the streams' reduce;
//   * generate mode: --tokens IDS [--n N] [--force IDS] [--dump-full-logits F] (parity runs against llama.cpp).
#include "hy4_dfly_file.hpp"
#include "hy4_kernels.cuh"
#include "hy4_place.hpp"
#include "hy4_prompt.cuh"

#include "lamina/artifact/drafter.hpp"
#include "lamina/artifact/gguf_reader.hpp"
#include "lamina/artifact/gguf_split.hpp"
#include "lamina/core/conversation_cache.hpp"
#include "lamina/core/cuda_host.hpp"
#include "lamina/core/disk_tier.hpp"
#include "lamina/core/gpu_select.hpp"
#include "lamina/core/model_check.hpp"
#include "lamina/core/run_owned.hpp"
#include "lamina/core/vmm.hpp"
#include "lamina/hy4/engine.hpp"
#include "lamina/hy4/geometry.hpp"
#include "lamina/kernels/bf16_gemv.hpp"
#include "lamina/kernels/cpu/native_expert.hpp"
#include "lamina/kernels/cpu/pool.hpp"
#include "lamina/kernels/gemm.hpp"
#include "lamina/kernels/iq_kernels.hpp"
#include "lamina/kernels/native_mmvq.hpp"
#include "lamina/ngram/ple_reader.hpp"
#include "lamina/platform/memory.hpp"
#include "lamina/platform/slice_read.hpp"
#include "lamina/prefill/moe_mmq.hpp"
#include "lamina/program/host_sampling.hpp"
#include "lamina/program/serve_io.hpp"
#include "lamina/spec/suffix_drafter.hpp"

#include "ggml.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
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
#include <random>
#include <string>
#include <thread>
#include <vector>

using lamina::program::serve_print;

namespace {

namespace K = lamina::kernels;
namespace C = lamina::kernels::cpu;
using lamina::core::ms_since;
using Clock = std::chrono::steady_clock;

#define CK(x) LAMINA_CK(x)
using lamina::core::dalloc;
using lamina::core::dmalloc;
using lamina::core::event_new;
using lamina::core::Fatal;
using lamina::core::graph_end;
using lamina::core::halloc;
using lamina::core::heap_new;
using lamina::core::hmalloc;
using lamina::core::hmapped;
using lamina::core::soft;
using lamina::core::stream_new;

constexpr int kMaxT = 8;                       // rows a forward takes at most (native_mmvq's columns)
constexpr int kTail = 16;                      // the indexer's per-cell value / gate rows kept (four groups of 4)
constexpr int kScoreBlocks = 128;              // the indexer scores' grid (a grid-stride loop over the groups)
constexpr float kEngramEps = 1.1920929e-07f;   // the reference's unset RMSNorm eps: FLT_EPSILON
constexpr size_t kWeightPad = 4096;            // zero bytes after each dense weight

// ------------------------------------------------------------------------------------------------ the model
struct Mat {
    const uint8_t* d = nullptr;   // device, GGUF blocks
    int type = -1;
    int n_in = 0, n_out = 0;
};

/// A GPU's slots of one MoE layer
struct SlotSet {
    uint8_t* slots = nullptr;   // n blobs, the layer's stride apart (device)
    int n = 0;
    std::vector<int32_t> slot_of;       // [n_expert]: its slot or -1 (host)
    std::vector<int32_t> slot_expert;   // [n]: its expert or -1
    int32_t* slot_of_d = nullptr;       // slot_of on that GPU
};

struct Layer {
    int il = 0;   // its layer (a DFly layer: the trunk layer it reads)
    bool mtp = false, swa = false, has_idx = false, dense = false;
    int eng = -1;   // its Engram module, or -1
    int H = 0;
    float theta = 0;   // rope theta_scale = base^(-2 / n_rot)
    const float *attn_norm = nullptr, *sinks = nullptr, *q_a_norm = nullptr, *kv_a_norm = nullptr, *post_attn = nullptr,
                *ffn_norm = nullptr, *post_ffw = nullptr;
    Mat q_a, q_b, kv_a, k_b, v_b, wo, gate;
    const float *hca_fn = nullptr, *hca_base = nullptr, *hca_scale = nullptr;
    const float *hcf_fn = nullptr, *hcf_base = nullptr, *hcf_scale = nullptr;
    // a DFly layer's cells (Engine::dfly_draft's: the trunk layer's picks and the block), ld apart
    const int32_t *ext_cells = nullptr, *ext_n = nullptr;
    int ext_ld = 0;
    // the indexer
    Mat iq_b, ik, icv, icg;
    const float *ik_norm = nullptr, *ape = nullptr, *ipool_norm = nullptr, *iproj = nullptr;
    // Engram
    Mat e_kv;
    const float *e_kv_b = nullptr, *e_gate = nullptr, *e_lnorm = nullptr, *e_gamma = nullptr, *e_hcs = nullptr;
    // the dense FFN (layer 0)
    Mat f_gate, f_up, f_down;
    // the latent MoE
    const float *router = nullptr, *bias = nullptr;
    Mat r_down, r_up, s_gate, s_up, s_down;
    C::NativeFmt fmt;
    // the routed experts' blobs in host memory, `stride` bytes each: expert e's at blob[e], in pinned memory or not; a
    // layer's blobs in two blocks (its hottest experts by the start profile pageable, the rest pinned, as the pin
    // budget allows)
    std::vector<uint8_t*> blob;
    std::vector<char> blob_pinned;
    size_t stride = 0;
    // the experts in VRAM: each GPU's slots (Engine::gpus: [0] the main GPU, then the expert tiers)
    K::NativeExpertLayout gl;   // the GPUs' view of a blob
    std::vector<SlotSet> held;
    std::vector<float> use;     // routing counts, halved at each rebalance
    std::vector<char> moving;   // [n_expert]: a copy into some GPU's slot in flight (Engine::issue)
    std::vector<float> total;   // the run's routing counts (the learned profile's)
    std::vector<float> start;   // the start profile's shares (empty: none)
    // caches
    uint8_t* kv = nullptr;   // latent rows: the context's cells (sparse) or the window's ring
    int64_t kv_cells = 0;
    bool kv_q8 = false;   // its rows in q8_0 (--kv q8_0: the sparse layers'), else halves
    size_t kv_row() const { return kv_q8 ? (size_t) hy4k::kKvQ8Row : (size_t) hy4k::kKvRow * sizeof(__half); }
    // the K/V grown with the context (Engine::ensure): the sparse layer's cells and pooled keys; the main GPU's slots
    std::shared_ptr<lamina::core::VmmRange> kv_vm, pooled_vm, slot_vm;
    float* pooled = nullptr;   // the indexer's pooled keys [ctx / pool][idx_dim]
    float* tail = nullptr;     // the indexer's last kTail cells' value and gate rows [kTail][2][idx_dim]
};

struct Model {
    lamina::hy4::Geometry g;
    std::vector<Layer> layers;   // the trunk
    // the MTP block (blk.<n_layers>, --no-mtp: not loaded): a sparse layer on one stream after eh_proj
    bool has_mtp = false;
    Layer mtp;
    Mat eh_proj;
    const float *mtp_enorm = nullptr, *mtp_hnorm = nullptr, *mtp_shnorm = nullptr;
    // DFly2 (--drafter FILE; the MTP block is then not loaded): layers as the trunk's sparse ones with dense FFNs, each
    // on the K/V and the index picks of its trunk layer (target); the block's first row fused with the trunk's state,
    // each draft's state corrected with the last draft's embedding
    struct DFly {
        bool on = false;
        std::vector<Layer> layers;
        std::vector<int> target;
        Mat eh_proj, hc_gate, hc_up, hc_down;
        const float *enorm = nullptr, *hnorm = nullptr, *fgate = nullptr, *hc_hnorm = nullptr, *hc_enorm = nullptr,
                    *norm = nullptr;
        int32_t mask = -1;
        int width = 0;   // the block's rows: the bonus token and width - 1 masks
    } dfly;
    const uint8_t* embd = nullptr;
    int embd_type = -1;
    size_t embd_row = 0;
    const float *out_norm = nullptr, *ohc_fn = nullptr, *ohc_base = nullptr, *ohc_scale = nullptr;
    const uint16_t* head = nullptr;   // BF16 [n_vocab][n_embd]
    std::string table_path;
    uint64_t table_off = 0;
    double arena_gib = 0;
    size_t pinned = 0;        // the arena's bytes in pinned memory
    int32_t eos = -1;         // tokenizer.ggml.eos_token_id: the end of an answer
    int32_t video_pad = -1;   // <｜hy_video▁pad｜>: a video frame's cells, as a picture's
    int32_t image_pad = -1;   // <｜hy_image▁pad｜>: a picture's cells in a
                              // prompt (the template writes one, the server n)
    // a message: <｜hy_start｜>, its role, <｜hy_middle｜>, its text, <｜hy_end｜> (-1: not in the vocabulary)
    int32_t turn_start = -1, turn_middle = -1, turn_end = -1;
};

/// Rows of floats as a Q8_0 matrix on the device (ggml's quantizer; kWeightPad zero bytes after it)
Mat q8_mat(const float* w, int64_t rows, int64_t cols) {
    Mat m;
    m.type = 8;
    m.n_in = (int) cols;
    m.n_out = (int) rows;
    const size_t bytes = (size_t) rows * (size_t) (cols / 32) * 34;
    std::vector<uint8_t> q(bytes);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, w, q.data(), 0, rows, cols, nullptr);
    uint8_t* d = dalloc<uint8_t>(bytes + kWeightPad);
    CK(cudaMemcpy(d, q.data(), bytes, cudaMemcpyHostToDevice));
    CK(cudaMemset(d + bytes, 0, kWeightPad));
    m.d = d;
    return m;
}
const float* f32_dev(const std::vector<float>& h) {
    float* d = dalloc<float>(h.size());
    CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    return d;
}

struct Loader {
    const lamina::GgufModel& gm;
    std::string err;
    std::map<const lamina::TensorInfo*, const uint8_t*> pre;   // tensors read ahead (read_ahead), else the mapping's
    std::unique_ptr<uint8_t[]> pre_buf;
    /// Every tensor `want` takes, read in large pieces into host memory: a copy out of the mapping faults its pages in
    /// a few at a time
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
    void drop_ahead() {
        pre.clear();
        pre_buf.reset();
    }
    const lamina::TensorInfo* find(const std::string& name, size_t* s) {
        const lamina::TensorInfo* t = gm.find(name, s);
        if (!t) {
            std::fprintf(stderr, "lamina: missing %s\n", name.c_str());
            throw Fatal{1};
        }
        return t;
    }
    const uint8_t* host(const std::string& name, const lamina::TensorInfo** ti = nullptr) {
        size_t s = 0;
        const lamina::TensorInfo* t = find(name, &s);
        if (ti) *ti = t;
        if (const auto it = pre.find(t); it != pre.end()) return it->second;
        return gm.shard(s).tensor_data(*t);
    }
    /// the tensor's GGUF bytes on the device
    const uint8_t* raw(const std::string& name, const lamina::TensorInfo** ti = nullptr) {
        const lamina::TensorInfo* t = nullptr;
        const uint8_t* src = host(name, &t);
        if (ti) *ti = t;
        const uint64_t n = lamina::tensor_payload_bytes(*t);
        uint8_t* d = dalloc<uint8_t>(n + kWeightPad);   // MMQ's k loop reads past a short last row (zero activations)
        CK(cudaMemcpy(d, src, n, cudaMemcpyHostToDevice));
        CK(cudaMemset(d + n, 0, kWeightPad));
        return d;
    }
    /// a matrix [n_in, n_out...] as Q8_0 rows on the device: Q8_0 as it is, BF16 or F32 quantized here (ggml's Q8_0;
    /// the MTP block's matrices are BF16)
    Mat mat(const std::string& name) {
        const lamina::TensorInfo* t = nullptr;
        const uint8_t* src = host(name, &t);
        Mat m;
        m.type = 8;
        m.n_in = (int) t->shape[0];
        uint64_t rows = 1;
        for (size_t i = 1; i < t->shape.size(); ++i) rows *= t->shape[i];
        m.n_out = (int) rows;
        if (t->type == 8) {
            m.d = raw(name);
            return m;
        }
        if ((t->type != 30 && t->type != 0) || m.n_in % 32 != 0) {
            std::fprintf(stderr,
                         "lamina: %s is %s; this path reads its dense matrices as Q8_0 (or BF16 / F32 made "
                         "Q8_0)\n",
                         name.c_str(), t->type_name());
            throw Fatal{1};
        }
        const size_t n = (size_t) rows * (size_t) m.n_in;
        std::vector<float> f(n);
        if (t->type == 0) std::memcpy(f.data(), src, n * 4);
        else ggml_bf16_to_fp32_row(reinterpret_cast<const ggml_bf16_t*>(src), f.data(), (int64_t) n);
        return q8_mat(f.data(), (int64_t) rows, m.n_in);
    }
    /// an F32 tensor on the device (BF16 / F16 widened)
    const float* f32(const std::string& name) {
        const lamina::TensorInfo* t = nullptr;
        const uint8_t* src = host(name, &t);
        const size_t n = (size_t) t->elements();
        std::vector<float> h(n);
        if (t->type == 0) std::memcpy(h.data(), src, n * 4);
        else if (t->type == 30)
            for (size_t i = 0; i < n; ++i) {
                const uint32_t u = (uint32_t) (src[2 * i] | (src[2 * i + 1] << 8)) << 16;
                std::memcpy(&h[i], &u, 4);
            }
        else {
            std::fprintf(stderr, "lamina: %s is %s, expected F32 or BF16\n", name.c_str(), t->type_name());
            throw Fatal{1};
        }
        return f32_dev(h);
    }
};

/// A BF16 matrix's rows into Q8_0 rows (ggml's quantizer)
void bf16_rows_q8_0(const uint8_t* bf16, uint8_t* dst, int64_t rows, int64_t cols) {
    std::vector<float> x((size_t) (rows * cols));
    ggml_bf16_to_fp32_row(reinterpret_cast<const ggml_bf16_t*>(bf16), x.data(), rows * cols);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, x.data(), dst, 0, rows, cols, nullptr);
}

/// The start profile's order of each MoE layer's experts (the learned one first, as load_profile takes them): the most
/// used first; empty when there is none of this model's shape
std::vector<std::vector<int32_t>> profile_order(const std::vector<std::string>& paths, int64_t n_layers,
                                                int64_t n_dense, int64_t n_expert) {
    for (const std::string& path : paths) {
        if (path.empty()) continue;
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) continue;
        int32_t hdr[3] = {};
        bool ok = std::fread(hdr, 4, 3, f) == 3 && hdr[0] == 0x50345948 && hdr[1] == (int32_t) n_layers &&
                  hdr[2] == (int32_t) n_expert;
        std::vector<std::vector<int32_t>> out;
        std::vector<float> sh((size_t) n_expert);
        for (int64_t l = n_dense; ok && l < n_layers; ++l) {
            ok = std::fread(sh.data(), 4, sh.size(), f) == sh.size();
            std::vector<int32_t> order((size_t) n_expert);
            for (int64_t e = 0; e < n_expert; ++e) order[(size_t) e] = (int32_t) e;
            std::stable_sort(order.begin(), order.end(),
                             [&](int32_t a, int32_t b) { return sh[(size_t) a] > sh[(size_t) b]; });
            out.push_back(std::move(order));
        }
        std::fclose(f);
        if (ok) return out;
    }
    return {};
}

bool load(const std::string& path, Model& m, int threads, double pin_gib, bool want_mtp, bool file_cache,
          const std::vector<std::string>& profiles, bool serving, std::string& err) {
    const auto t0 = Clock::now();
    const lamina::GgufModel gm = lamina::GgufModel::open(path);
    {   // what the file must hold for this path, before anything is read (lamina check says the same of it)
        lamina::core::FileCheckOptions fo;
        fo.mtp = want_mtp;
        if (std::string e = lamina::core::hy4_file_problem(gm, fo); !e.empty()) {
            err = e;
            return false;
        }
    }
    if (std::string e = lamina::hy4::read_geometry(gm.meta(), m.g); !e.empty()) {
        err = e;
        return false;
    }
    if (const lamina::MetaValue* v = gm.meta().get("tokenizer.ggml.eos_token_id"); v && v->is_num())
        m.eos = (int32_t) v->u;
    m.image_pad = (int32_t) gm.meta().find_string("tokenizer.ggml.tokens", "<\xEF\xBD\x9Chy_image\xE2\x96\x81pad");
    m.video_pad = (int32_t) gm.meta().find_string("tokenizer.ggml.tokens", "<\xEF\xBD\x9Chy_video\xE2\x96\x81pad");
    m.turn_start = (int32_t) gm.meta().find_string("tokenizer.ggml.tokens", "<\xEF\xBD\x9Chy_start");
    m.turn_middle = (int32_t) gm.meta().find_string("tokenizer.ggml.tokens", "<\xEF\xBD\x9Chy_middle");
    m.turn_end = (int32_t) gm.meta().find_string("tokenizer.ggml.tokens", "<\xEF\xBD\x9Chy_end");
    const auto& g = m.g;
    Loader L{gm, {}, {}, {}};
    {   // the dense tensors: all but the routed experts, the Engram table and the MTP block when it is not loaded
        const std::string mtp = "blk." + std::to_string(g.n_layers) + ".";
        const bool keep_mtp = want_mtp && g.n_mtp > 0;
        if (!L.read_ahead(
                [&](const std::string& n) {
                    return n.find("_exps.") == std::string::npos && n != "per_layer_token_embd.weight" &&
                           (keep_mtp || n.rfind(mtp, 0) != 0);
                },
                threads, !file_cache, err))
            return false;
    }
    {
        size_t s = 0;
        const lamina::TensorInfo* t = gm.find("per_layer_token_embd.weight", &s);
        if (t->type != 43) {
            err = "per_layer_token_embd.weight is " + std::string(t->type_name()) + ", expected F8_E4M3";
            return false;
        }
        m.table_path = gm.shard(s).path();
        m.table_off = gm.shard(s).data_start() + t->offset;
    }
    const lamina::TensorInfo* te = nullptr;
    m.embd = L.raw("token_embd.weight", &te);
    m.embd_type = (int) te->type;
    m.embd_row = (size_t) (lamina::tensor_payload_bytes(*te) / te->shape[1]);
    if (!K::iq_supported(m.embd_type)) {
        err = std::string("token_embd.weight is ") + te->type_name() + ", which iq_embed_rows does not read";
        return false;
    }
    m.out_norm = L.f32("output_norm.weight");
    m.ohc_fn = L.f32("output_hc_fn.weight");
    m.ohc_base = L.f32("output_hc_base.weight");
    m.ohc_scale = L.f32("output_hc_scale.weight");
    {
        const lamina::TensorInfo* th = nullptr;
        m.head = reinterpret_cast<const uint16_t*>(L.raw("output.weight", &th));
        if (th->type != 30) {
            err = std::string("output.weight is ") + th->type_name() + ", expected BF16";
            return false;
        }
    }
    m.layers.resize((size_t) g.n_layers);
    m.has_mtp = want_mtp && g.n_mtp > 0;
    for (int64_t l = 0; l < g.n_layers + (m.has_mtp ? 1 : 0); ++l) {
        const bool mtp = l >= g.n_layers;   // the MTP block: one stream, no Engram, its own head norms
        Layer& y = mtp ? m.mtp : m.layers[(size_t) l];
        const std::string p = "blk." + std::to_string(l) + ".";
        y.il = (int) l;
        y.swa = g.swa[(size_t) l];
        y.has_idx = g.has_indexer(l);
        if (!y.swa && !y.has_idx) {
            err = p + ": a sparse layer without an indexer, which this path does not run";
            return false;
        }
        y.mtp = mtp;
        y.dense = !mtp && l < g.n_dense;
        y.eng = mtp ? -1 : g.engram_index(l);
        y.H = (int) g.n_head[(size_t) l];
        y.theta = powf((float) g.base(l), -2.0f / (float) g.n_rot);
        y.attn_norm = L.f32(p + "attn_norm.weight");
        y.sinks = L.f32(p + "attn_sinks.weight");
        y.q_a = L.mat(p + "attn_q_a.weight");
        y.q_a_norm = L.f32(p + "attn_q_a_norm.weight");
        y.q_b = L.mat(p + "attn_q_b.weight");
        y.kv_a = L.mat(p + "attn_kv_a_mqa.weight");
        y.kv_a_norm = L.f32(p + "attn_kv_a_norm.weight");
        y.k_b = L.mat(p + "attn_k_b.weight");
        y.v_b = L.mat(p + "attn_v_b.weight");
        y.wo = L.mat(p + "attn_output.weight");
        y.gate = L.mat(p + "attn_gate.weight");
        y.post_attn = L.f32(p + "post_attention_norm.weight");
        y.ffn_norm = L.f32(p + "ffn_norm.weight");
        y.post_ffw = L.f32(p + "post_ffw_norm.weight");
        if (mtp) {
            m.eh_proj = L.mat(p + "nextn.eh_proj.weight");
            m.mtp_enorm = L.f32(p + "nextn.enorm.weight");
            m.mtp_hnorm = L.f32(p + "nextn.hnorm.weight");
            m.mtp_shnorm = L.f32(p + "nextn.shared_head_norm.weight");
        } else {
            y.hca_fn = L.f32(p + "hc_attn_fn.weight");
            y.hca_base = L.f32(p + "hc_attn_base.weight");
            y.hca_scale = L.f32(p + "hc_attn_scale.weight");
            y.hcf_fn = L.f32(p + "hc_ffn_fn.weight");
            y.hcf_base = L.f32(p + "hc_ffn_base.weight");
            y.hcf_scale = L.f32(p + "hc_ffn_scale.weight");
        }
        if (y.has_idx) {
            y.iq_b = L.mat(p + "indexer.attn_q_b.weight");
            y.ik = L.mat(p + "indexer.attn_k.weight");
            y.ik_norm = L.f32(p + "indexer.k_norm.weight");
            y.icv = L.mat(p + "indexer_compressor_value.weight");
            y.icg = L.mat(p + "indexer_compressor_gate.weight");
            y.ape = L.f32(p + "indexer_compressor_ape.weight");
            y.ipool_norm = L.f32(p + "indexer_compressor_norm.weight");
            y.iproj = L.f32(p + "indexer.proj.weight");
        }
        if (y.eng >= 0) {
            y.e_kv = L.mat(p + "engram_kv.weight");
            y.e_kv_b = L.f32(p + "engram_kv_b.weight");
            y.e_gate = L.f32(p + "engram_gate.weight");
            y.e_lnorm = L.f32(p + "engram_lookup_norm.weight");
            y.e_gamma = L.f32(p + "engram_pre_gamma.weight");
            y.e_hcs = L.f32(p + "engram_hc_scale.weight");
        }
        if (y.dense) {
            y.f_gate = L.mat(p + "ffn_gate.weight");
            y.f_up = L.mat(p + "ffn_up.weight");
            y.f_down = L.mat(p + "ffn_down.weight");
        } else {
            y.router = L.f32(p + "ffn_gate_inp.weight");
            y.bias = L.f32(p + "exp_probs_b.bias");
            y.r_down = L.mat(p + "ffn_routed_down.weight");
            y.r_up = L.mat(p + "ffn_routed_up.weight");
            y.s_gate = L.mat(p + "ffn_gate_shexp.weight");
            y.s_up = L.mat(p + "ffn_up_shexp.weight");
            y.s_down = L.mat(p + "ffn_down_shexp.weight");
        }
    }
    CK(cudaDeviceSynchronize());
    L.drop_ahead();
    const double dense_s = ms_since(t0) / 1000.0;
    // the routed experts: native blobs [gate rows | up rows | down rows] in host memory, read from the file in large
    // pieces by `threads` threads (a copy out of the mapping faults its pages in a few at a time; through the OS file
    // cache, which reads ahead, or past it), each layer's block allocated on a thread of its own as the reads come to
    // it; BF16 experts made Q8_0 from the mapping after
    struct Src {
        Layer* y;
        const uint8_t *g, *u, *d;
        size_t gsz, dsz;
        bool bf16 = false;        // BF16 experts (the MTP block's): each matrix quantized to Q8_0 into the blob
        std::string gp, up, dp;   // the shards that hold the three tensors, and their places in them
        uint64_t go = 0, uo = 0, dof = 0;
    };
    std::vector<Src> src;
    size_t total = 0;
    std::vector<Layer*> moe;
    for (Layer& y : m.layers) moe.push_back(&y);
    if (m.has_mtp) moe.push_back(&m.mtp);
    auto file_at = [&](const std::string& name, std::string& path) {
        size_t s = 0;
        const lamina::TensorInfo* t = L.find(name, &s);
        path = gm.shard(s).path();
        return gm.shard(s).data_start() + t->offset;
    };
    for (Layer* yp : moe) {
        Layer& y = *yp;
        if (y.dense) continue;
        const std::string p = "blk." + std::to_string(y.il) + ".";
        const lamina::TensorInfo *tg = nullptr, *tu = nullptr, *td = nullptr;
        Src s{&y,
              L.host(p + "ffn_gate_exps.weight", &tg),
              L.host(p + "ffn_up_exps.weight", &tu),
              L.host(p + "ffn_down_exps.weight", &td),
              0,
              0,
              false};
        s.go = file_at(p + "ffn_gate_exps.weight", s.gp);
        s.uo = file_at(p + "ffn_up_exps.weight", s.up);
        s.dof = file_at(p + "ffn_down_exps.weight", s.dp);
        if (tg->type != tu->type) {
            err = p + "ffn_gate_exps and ffn_up_exps differ in type";
            return false;
        }
        s.bf16 = tg->type == 30 && td->type == 30;   // quantized here, to Q8_0
        const int gt = s.bf16 ? 8 : (int) tg->type, dt = s.bf16 ? 8 : (int) td->type;
        if (!C::native_fmt(gt, dt, g.n_latent, g.n_ff_exp, y.fmt, err)) {
            err = p + "routed experts: " + err;
            return false;
        }
        y.fmt.swiglu_limit = g.swiglu_limit[(size_t) y.il];
        s.gsz = lamina::tensor_payload_bytes(*tg) / (size_t) g.n_expert;
        s.dsz = lamina::tensor_payload_bytes(*td) / (size_t) g.n_expert;
        if (!s.bf16 && (s.gsz != y.fmt.up_off || 2 * s.gsz + s.dsz != y.fmt.bytes)) {
            err = p + "routed experts: unexpected blob layout";
            return false;
        }
        {   // whole blocks of both types (MMQ reads expert e at e * stride), 64-byte aligned
            int be = 0, bg = 0, bd = 0;
            lamina::block_geometry((uint32_t) gt, be, bg);
            lamina::block_geometry((uint32_t) dt, be, bd);
            const size_t unit = std::lcm(std::lcm((size_t) 64, (size_t) bg), (size_t) bd);
            y.stride = (y.fmt.bytes + unit - 1) / unit * unit;
        }
        total += y.stride * (size_t) g.n_expert;
        src.push_back(s);
    }
    m.arena_gib = (double) total / (double) (1ull << 30);
    {   // the RAM: every routed expert in host memory, weighed before any is taken
        const lamina::platform::RamFit ram = lamina::platform::ram_fit(total);
        const std::string what = "the routed experts (in RAM, every one)";
        if (ram.total > 0)
            std::fprintf(stderr, "lamina: RAM: %s take %.1f GiB, %.1f of this PC's %.1f GiB available\n", what.c_str(),
                         (double) total / 1073741824.0, (double) ram.available / 1073741824.0,
                         (double) ram.total / 1073741824.0);
        if (ram.too_much()) {
            err = ram.say(what, "a smaller quant of the model takes less");
            return false;
        }
        if (ram.tight())
            lamina::program::serve_warn(serving,
                                        ram.say(what, "close other programs, or use a smaller quant of the model"));
    }
    // each layer's blocks, in order, on a thread of its own.  Pinned while the limit allows: the GPUs copy from it at
    // their links' full rate, the host not waiting.  With a start profile each layer keeps its `hot` most used experts
    // (the GPUs' at the start, rarely streamed to a GPU) in a pageable block, as many in every layer as the pin budget
    // leaves out, and the rest pinned, so what a prompt streams is pinned in every layer; without one, whole layers
    // pinned until the next would pass the budget
    const std::vector<std::vector<int32_t>> order = profile_order(profiles, g.n_layers, g.n_dense, g.n_expert);
    int64_t hot = 0;
    {
        size_t trunk = 0, per_expert = 0;
        int n_moe = 0;
        for (const Src& s : src)
            if (!s.y->mtp) {
                trunk += s.y->stride * (size_t) g.n_expert;
                per_expert += s.y->stride;
                ++n_moe;
            }
        const double budget = pin_gib * (double) (1ull << 30);
        if (!order.empty() && (int) order.size() == n_moe && (double) trunk > budget && per_expert > 0 &&
            std::getenv("LAMINA_HY4_ARENA_LAYERS") == nullptr)
            hot = std::min<int64_t>(g.n_expert, (int64_t) (((double) trunk - budget) / (double) per_expert) + 1);
    }
    std::vector<uint8_t*> bases(src.size(), nullptr);
    std::vector<char> pinned_at(src.size(), 0);
    for (Src& s : src) {
        s.y->blob.assign((size_t) g.n_expert, nullptr);
        s.y->blob_pinned.assign((size_t) g.n_expert, 0);
    }
    std::mutex mu;
    std::condition_variable cv;
    size_t made = 0;
    bool alloc_end = false;
    double alloc_s = 0;
    int dev = 0;
    CK(cudaGetDevice(&dev));
    const bool serial = std::getenv("LAMINA_HY4_LOAD_SERIAL") != nullptr;
    auto make_blocks = [&] {
        const auto ta = Clock::now();
        if (!serial) (void) cudaSetDevice(dev);   // no context made on another GPU for this thread
        size_t pinned = 0;
        auto pin = [&](size_t bytes) -> uint8_t* {   // pinned within the budget, else null
            void* h = nullptr;
            if (bytes == 0 || (double) (pinned + bytes) > pin_gib * (double) (1ull << 30)) return nullptr;
            if (!soft(hmalloc(&h, bytes, cudaHostAllocPortable))) return nullptr;
            pinned += bytes;
            return (uint8_t*) h;
        };
        for (size_t i = 0; i < src.size(); ++i) {
            Layer& y = *src[i].y;
            const size_t st = y.stride;
            // the hot experts (the profile's first `hot` of this layer; none for the MTP block) and the cold ones
            std::vector<char> is_hot((size_t) g.n_expert, 0);
            const size_t li = (size_t) (y.il - g.n_dense);
            if (hot > 0 && !y.mtp && li < order.size())
                for (int64_t k = 0; k < hot; ++k) is_hot[(size_t) order[li][(size_t) k]] = 1;
            size_t n_hot = 0;
            for (const char c : is_hot) n_hot += c ? 1 : 0;
            const size_t n_cold = (size_t) g.n_expert - n_hot;
            uint8_t* cold = pin(n_cold * st);
            const bool cold_pinned = cold != nullptr;
            // a hot block pinned only while the later layers' cold blocks still fit the budget
            size_t later_cold = 0;
            for (size_t j = i + 1; j < src.size(); ++j)
                later_cold += src[j].y->stride * (size_t) (g.n_expert - (src[j].y->mtp ? 0 : hot));
            uint8_t* hotb = n_hot && (double) (pinned + n_hot * st + later_cold) <= pin_gib * (double) (1ull << 30)
                                ? pin(n_hot * st)
                                : nullptr;
            const bool hot_pinned = hotb != nullptr;
            if (!cold) cold = (uint8_t*) heap_new(std::max<size_t>(n_cold * st, 64));
            if (n_hot && !hotb) hotb = (uint8_t*) heap_new(n_hot * st);
            bool ok = cold != nullptr && (n_hot == 0 || hotb != nullptr);
            if (ok) {
                size_t ic = 0, ih = 0;
                for (int64_t e = 0; e < g.n_expert; ++e) {
                    const bool h = is_hot[(size_t) e] != 0;
                    y.blob[(size_t) e] = h ? hotb + (ih++) * st : cold + (ic++) * st;
                    y.blob_pinned[(size_t) e] = h ? (char) hot_pinned : (char) cold_pinned;
                }
            }
            std::lock_guard<std::mutex> lk(mu);
            bases[i] = ok ? y.blob[0] : nullptr;
            pinned_at[i] = cold_pinned && (n_hot == 0 || hot_pinned);
            made = i + 1;
            if (!ok) break;
            if (i + 1 == src.size()) m.pinned = pinned;
            cv.notify_all();
        }
        std::lock_guard<std::mutex> lk(mu);
        alloc_end = true;
        alloc_s = ms_since(ta) / 1000.0;
        cv.notify_all();
    };
    std::thread alloc;
    if (serial) make_blocks();
    else alloc = std::thread(make_blocks);
    auto base = [&](int i) -> uint8_t* {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return made > (size_t) i || alloc_end; });
        return made > (size_t) i ? bases[(size_t) i] : nullptr;
    };
    std::vector<lamina::platform::SliceRun> runs;
    uint64_t read_bytes = 0;
    for (size_t i = 0; i < src.size(); ++i) {
        const Src& s = src[i];
        if (s.bf16) continue;
        const uint64_t st = s.y->stride;
        uint8_t* const* pl = s.y->blob.data();   // filled by the allocation before base(i) returns
        runs.push_back({s.gp, s.go, s.gsz, 0, st, g.n_expert, (int) i, pl});
        runs.push_back({s.up, s.uo, s.gsz, s.gsz, st, g.n_expert, (int) i, pl});
        runs.push_back({s.dp, s.dof, s.dsz, 2 * s.gsz, st, g.n_expert, (int) i, pl});
        read_bytes += (uint64_t) (2 * s.gsz + s.dsz) * (uint64_t) g.n_expert;
    }
    double read_s = 0;
    const bool read_ok =
        lamina::platform::read_slices(runs, std::max(1, threads), !file_cache, (uint64_t) 32 << 20, base, err, &read_s);
    for (size_t i = 0; i < src.size(); ++i) base((int) i);   // every block made (or the allocation's end)
    if (alloc.joinable()) alloc.join();
    if (!read_ok) return false;
    for (size_t i = 0; i < src.size(); ++i)
        if (bases[i] == nullptr) {
            err = "out of host memory for the routed experts";
            return false;
        }
    // BF16 experts (the MTP block's): each matrix through float into Q8_0 rows, gate, up (n_ff rows of n_latent), down
    std::atomic<size_t> next{0};
    std::vector<size_t> bf;
    for (size_t i = 0; i < src.size(); ++i)
        if (src[i].bf16) bf.push_back(i);
    const size_t jobs = bf.size() * (size_t) g.n_expert;
    auto convert = [&] {
        for (size_t j; (j = next.fetch_add(1)) < jobs;) {
            const Src& s = src[bf[j / (size_t) g.n_expert]];
            const size_t e = j % (size_t) g.n_expert;
            uint8_t* dst = s.y->blob[e];
            bf16_rows_q8_0(s.g + e * s.gsz, dst, g.n_ff_exp, g.n_latent);
            bf16_rows_q8_0(s.u + e * s.gsz, dst + s.y->fmt.up_off, g.n_ff_exp, g.n_latent);
            bf16_rows_q8_0(s.d + e * s.dsz, dst + s.y->fmt.down_off, g.n_latent, g.n_ff_exp);
        }
    };
    if (jobs > 0) {
        std::vector<std::thread> th;
        for (int i = 0; i < std::max(1, threads); ++i) th.emplace_back(convert);
        for (auto& t : th) t.join();
    }
    std::fprintf(stderr,
                 "lamina: Hy4-flash: %lld layers%s, dense weights on the GPU in %.1f s, the routed experts "
                 "(%.1f GiB, %.1f of it pinned%s) in host memory in %.1f s (read %s the file cache at %.1f GiB/s; "
                 "the blocks made in %.1f s beside)\n",
                 (long long) g.n_layers, m.has_mtp ? " and the MTP block" : "", dense_s, m.arena_gib,
                 (double) m.pinned / (double) (1ull << 30),
                 hot > 0 ? (", each layer's " + std::to_string(hot) + " most used experts pageable").c_str() : "",
                 ms_since(t0) / 1000.0 - dense_s, file_cache ? "through" : "past",
                 read_s > 0 ? (double) read_bytes / read_s / (double) (1ull << 30) : 0.0, alloc_s);
    return true;
}

// ------------------------------------------------------------------------------------------------ DFly2
/// DFly2 from the checkpoint's model.dfly_layers (the safetensors file at `path`, its config.json beside it): its
/// layers as the trunk's sparse layers (MLA with the gate, the sinks and the post-projection norm; kv_b split per
/// head into k_b, each head's transposed, and v_b, as llama.cpp's converter splits the trunk's) with dense FFNs, each
/// reading the K/V and the index picks of its trunk layer; the fusion of the block's first row, the hidden
/// correction, the final norm.  Matrices made Q8_0 as the MTP block's.
bool load_dfly(const std::string& path, Model& m, std::string& err) {
    const auto t0 = Clock::now();
    const auto& g = m.g;
    lamina::hy4::DFlyConfig c;
    lamina::SafeTensors st;
    if (!lamina::hy4::read_dfly_config(std::filesystem::path(path).parent_path().string(), c, err) ||
        !st.open(path, err))
        return false;
    if (c.top_k != g.idx_top_k || c.width < 2 || c.width > kMaxT || c.mask < 0 || c.mask >= g.n_vocab) {
        err = "DFly: its top-k " + std::to_string(c.top_k) + " (the trunk's " + std::to_string(g.idx_top_k) +
              "), its block of " + std::to_string(c.width) + " rows or its mask token " + std::to_string(c.mask) +
              " does not fit this model";
        return false;
    }
    Model::DFly& d = m.dfly;
    d.target = c.kv_layers;
    d.mask = c.mask;
    d.width = c.width;
    const std::string P = "model.dfly_layers.";
    std::vector<float> v;
    std::vector<int64_t> sh;
    auto read = [&](const std::string& name, size_t dims) {
        if (!st.read(P + name, v, sh) || sh.size() != dims)
            throw std::runtime_error("DFly: " + name + " is missing, or not BF16 / F32 of " + std::to_string(dims) +
                                     " dimensions");
    };
    auto vec = [&](const std::string& name) -> const float* {
        read(name, 1);
        return f32_dev(v);
    };
    auto mat = [&](const std::string& name) -> Mat {
        read(name, 2);
        return q8_mat(v.data(), sh[0], sh[1]);
    };
    try {
        d.layers.resize(d.target.size());
        for (size_t i = 0; i < d.target.size(); ++i) {
            const int t = d.target[i];
            if (t < 0 || t >= g.n_layers || g.swa[(size_t) t] || !g.has_indexer(t)) {
                err = "DFly: its target layer " + std::to_string(t) + " is not a sparse layer with an indexer";
                return false;
            }
            Layer& y = d.layers[i];
            const Layer& tl = m.layers[(size_t) t];
            const std::string p = std::to_string(i) + ".";
            y.il = t;   // the trunk layer it reads
            y.H = tl.H;
            y.theta = tl.theta;
            y.attn_norm = vec(p + "input_layernorm.weight");
            y.ffn_norm = vec(p + "post_attention_layernorm.weight");
            y.q_a = mat(p + "self_attn.q_a_proj.weight");
            y.q_a_norm = vec(p + "self_attn.q_a_layernorm.weight");
            y.q_b = mat(p + "self_attn.q_b_proj.weight");
            y.kv_a = mat(p + "self_attn.kv_a_proj_with_mqa.weight");
            y.kv_a_norm = vec(p + "self_attn.kv_a_layernorm.weight");
            y.wo = mat(p + "self_attn.o_proj.weight");
            y.gate = mat(p + "self_attn.linear_gate.weight");
            y.sinks = vec(p + "self_attn.learnable_sink_param");
            y.post_attn = vec(p + "self_attn.post_proj_norm.weight");
            y.f_gate = mat(p + "mlp.gate_proj.weight");
            y.f_up = mat(p + "mlp.up_proj.weight");
            y.f_down = mat(p + "mlp.down_proj.weight");
            // kv_b [H (nope + v), latent] -> k_b [H][latent][nope] (each head's block transposed), v_b [H][v][latent]
            read(p + "self_attn.kv_b_proj.weight", 2);
            const int64_t H = y.H, nope = g.nope(), vh = g.v_head, lat = hy4k::kLatent;
            if (sh[0] != H * (nope + vh) || sh[1] != lat) {
                err = "DFly: layer " + std::to_string(i) + "'s kv_b_proj is not [" + std::to_string(H * (nope + vh)) +
                      ", " + std::to_string(lat) + "]";
                return false;
            }
            std::vector<float> kb((size_t) (H * lat * nope)), vb((size_t) (H * vh * lat));
            for (int64_t h = 0; h < H; ++h) {
                const float* blk = v.data() + (size_t) (h * (nope + vh) * lat);
                for (int64_t j = 0; j < lat; ++j)
                    for (int64_t k = 0; k < nope; ++k)
                        kb[(size_t) ((h * lat + j) * nope + k)] = blk[(size_t) (k * lat + j)];
                std::memcpy(vb.data() + (size_t) (h * vh * lat), blk + (size_t) (nope * lat), (size_t) (vh * lat) * 4);
            }
            y.k_b = q8_mat(kb.data(), H * lat, nope);
            y.v_b = q8_mat(vb.data(), H * vh, lat);
        }
        d.eh_proj = mat("mtp_fusion.eh_proj.weight");
        d.enorm = vec("mtp_fusion.enorm.weight");
        d.hnorm = vec("mtp_fusion.hnorm.weight");
        d.fgate = vec("mtp_fusion.gate");
        d.hc_gate = mat("hidden_correction.gate_proj.weight");
        d.hc_up = mat("hidden_correction.up_proj.weight");
        d.hc_down = mat("hidden_correction.down_proj.weight");
        d.hc_hnorm = vec("hidden_correction.hidden_norm.weight");
        d.hc_enorm = vec("hidden_correction.embed_norm.weight");
        d.norm = vec("final_norm.weight");
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    CK(cudaDeviceSynchronize());
    d.on = true;
    std::fprintf(stderr,
                 "lamina: DFly2: %zu layers on the trunk's layers %d .. %d, its FFN %d, a block of %d rows, in "
                 "%.1f s\n",
                 d.layers.size(), d.target.front(), d.target.back(), d.layers[0].f_gate.n_out, d.width,
                 ms_since(t0) / 1000.0);
    return true;
}

// ------------------------------------------------------------------------------------------------ Engram's hash
/// The 16 table rows of each Engram module for the token at `pos` of `seq` (llama.cpp's engram_hash): ids [modules]
/// [16], `valid` 0 for a token outside the map (an image position), else 1.
void engram_ids(const lamina::hy4::Geometry& g, const std::vector<int32_t>& seq, int64_t pos, uint32_t* ids,
                float& valid) {
    const int64_t n_gram = g.ngram, n_head = g.heads_per_ngram, n_cols = g.eng_heads();
    const int64_t pad = g.eng_map[(size_t) g.eng_pad];
    auto cid = [&](int64_t p) -> int64_t {
        if (p < 0) return -1;
        const int32_t t = seq[(size_t) p];
        return t < 0 || t >= (int32_t) g.eng_map.size() ? -1 : g.eng_map[(size_t) t];
    };
    const int64_t c0 = cid(pos);
    valid = c0 < 0 ? 0.f : 1.f;
    uint64_t toks[8];
    toks[0] = (uint64_t) (c0 < 0 ? pad : c0);
    for (int64_t s = 1; s < n_gram; ++s) {
        const int64_t c = cid(pos - s);
        toks[s] = (uint64_t) (c < 0 ? pad : c);
    }
    for (size_t e = 0; e < g.eng_layers.size(); ++e) {
        const uint64_t* mult = g.eng_mult.data() + e * (size_t) n_gram;
        const uint64_t* prime = g.eng_prime.data() + e * (size_t) n_cols;
        const uint64_t* off = g.eng_off.data() + e * (size_t) n_cols;
        uint64_t rolling = toks[0] * mult[0];
        for (int64_t o = 1; o < n_gram; ++o) {
            rolling ^= toks[o] * mult[o];
            for (int64_t h = 0; h < n_head; ++h) {
                const int64_t col = (o - 1) * n_head + h;
                ids[e * (size_t) n_cols + (size_t) col] = (uint32_t) (rolling % prime[col] + off[col]);
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ pictures
using ImgReq = lamina::program::ImageCells;   // a request's pictures (--vision)

// ------------------------------------------------------------------------------------------------ the engine
/// A planned move: expert e into slot k of GPU gi's slots of layer li (its resident `out` leaving it, or none)
struct Planned {
    float gain;
    int li, gi, k;
    int32_t e, out;
};
/// A slot's copy from the arena: expert e into slot k of GPU gi's slots of layer li, recorded by `ev` on that GPU's
/// copy stream once `issued` (its queue's thread issues it)
struct SlotMove {
    int li = 0, gi = 0, k = 0;
    int32_t e = -1;
    uint8_t* dst = nullptr;
    const uint8_t* src = nullptr;
    size_t bytes = 0;
    bool pinned = true;   // src in pinned memory, else staged through the queue's own pinned buffers
    cudaEvent_t ev = nullptr;
    std::atomic<bool> issued{false};
    std::atomic<bool> ok{true};
};
/// A GPU's slot copies, issued in order on its copy stream by a thread of its own: a copy from pageable memory holds
/// the thread that issues it until its bytes are staged, and the decode's thread must not wait for that
struct CopyQueue {
    int dev = 0;
    cudaStream_t st = nullptr;
    std::mutex mu;
    std::condition_variable cv;
    std::deque<SlotMove*> q;
    bool quit = false;
    // a copy from pageable memory staged here (the driver's own staging of one holds its API for the copy's time):
    // kStage buffers in turn, each reused once its last copy has gone
    static constexpr int kStage = 4;
    uint8_t* stage[kStage] = {};
    cudaEvent_t staged[kStage] = {};
    size_t stage_bytes = 0;
    int stage_at = 0;
    bool bounce = true;   // the staging buffers made
    // a copy in pieces: a small transfer of a forward (a tier's routed input) waits behind one piece, not an expert
    // (the edit text's tiers' rows 8.3 ms a forward after the CPU's at 512 KB, 2.6 at 128, 0.8-1.0 without copies)
    static constexpr size_t piece = (size_t) 256 << 10;
    /// dst <- src in pieces on the queue's stream
    bool copy(uint8_t* dst, const uint8_t* src, size_t n) {
        for (size_t o = 0; o < n; o += piece)
            if (!soft(cudaMemcpyAsync(dst + o, src + o, std::min(piece, n - o), cudaMemcpyHostToDevice, st)))
                return false;
        return true;
    }
    std::thread th;
    CopyQueue(int d, cudaStream_t s, size_t blob) : dev(d), st(s), stage_bytes(blob) {
        th = std::thread([this] { run(); });
    }
    ~CopyQueue() {
        {
            std::lock_guard<std::mutex> lk(mu);
            quit = true;
        }
        cv.notify_all();
        th.join();
        for (int i = 0; i < kStage; ++i) {
            if (staged[i] != nullptr) {
                (void) cudaEventSynchronize(staged[i]);
                (void) cudaEventDestroy(staged[i]);
            }
            if (stage[i] != nullptr) (void) cudaFreeHost(stage[i]);
        }
    }
    void push(SlotMove* m) {
        {
            std::lock_guard<std::mutex> lk(mu);
            q.push_back(m);
        }
        cv.notify_one();
    }
    void run() {
        (void) cudaSetDevice(dev);
        if (bounce)
            for (int i = 0; i < kStage; ++i)
                if (!soft(cudaHostAlloc((void**) &stage[i], stage_bytes, cudaHostAllocPortable)) ||
                    !soft(cudaEventCreateWithFlags(&staged[i], cudaEventDisableTiming)))
                    bounce = false;
        for (;;) {
            SlotMove* m = nullptr;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return quit || !q.empty(); });
                if (quit) return;
                m = q.front();
                q.pop_front();
            }
            const uint8_t* src = m->src;
            if (!m->pinned && bounce && m->bytes <= stage_bytes) {
                const int b = stage_at;
                stage_at = (stage_at + 1) % kStage;
                (void) cudaEventSynchronize(staged[b]);   // its last copy gone
                std::memcpy(stage[b], m->src, m->bytes);
                src = stage[b];
                if (!copy(m->dst, src, m->bytes) || !soft(cudaEventRecord(staged[b], st)) ||
                    !soft(cudaEventRecord(m->ev, st)))
                    m->ok.store(false);
                m->issued.store(true, std::memory_order_release);
                continue;
            }
            if (!copy(m->dst, src, m->bytes) || !soft(cudaEventRecord(m->ev, st))) m->ok.store(false);
            m->issued.store(true, std::memory_order_release);
        }
    }
};

// A parked conversation's file: closing it takes it off the disk tier.  Outside Engine: nested there, its default
// member initializer is not usable before Engine is complete, so GCC and Clang find it not default-constructible and
// delete the default constructors of every class that holds one.
struct FileGone {
    lamina::core::DiskTier* tier = nullptr;
    void operator()(lamina::core::DiskFile* f) const {
        if (f == nullptr) return;
        if (tier != nullptr) tier->remove(*f);
        delete f;
    }
};

struct Engine {
    Model* m = nullptr;
    int64_t ctx = 0;    // cells of the sparse layers' caches
    int64_t ring = 0;   // cells of a window layer's ring
    cudaStream_t st = nullptr;
    std::unique_ptr<C::ExpertPool> pool;
    lamina::ngram::PleReader reader;
    std::vector<int32_t> seq;   // the tokens in the state, in order

    // buffers (kMaxT rows)
    float *X = nullptr, *xh = nullptr, *xn = nullptr, *y = nullptr, *sy = nullptr, *emb = nullptr, *post = nullptr,
          *hc_part = nullptr;
    uint8_t *xq = nullptr, *qrq = nullptr, *qnq = nullptr, *attq = nullptr, *oq = nullptr, *hq = nullptr,
            *moeq = nullptr, *erq = nullptr, *ikq = nullptr;
    float *qa = nullptr, *q = nullptr, *qnope = nullptr, *qabs = nullptr, *kva = nullptr, *gt = nullptr,
          *part = nullptr, *att = nullptr, *o = nullptr;
    float *fg = nullptr, *fu = nullptr, *fh = nullptr;
    float *logits_r = nullptr, *rin = nullptr, *sg = nullptr, *su = nullptr, *sh = nullptr, *parts = nullptr,
          *moe = nullptr, *wts = nullptr;
    int32_t* ids = nullptr;
    uint8_t* eraw = nullptr;
    float *erows = nullptr, *ekv = nullptr, *evalid = nullptr;
    float *iq = nullptr, *ik = nullptr, *ivg = nullptr, *iw = nullptr, *scores = nullptr;
    int32_t *cells_swa = nullptr, *n_swa = nullptr, *cells_sel = nullptr, *n_sel = nullptr;
    float* logits = nullptr;
    int32_t *tok_d = nullptr, *pos_d = nullptr;
    // host (pinned; h_out and h_miss mapped: the next segment reads the CPU's rows through the mapping)
    float *h_rin = nullptr, *h_out = nullptr, *h_out_d = nullptr, *h_valid = nullptr;
    int32_t *h_ids = nullptr, *h_tok = nullptr, *h_pos = nullptr;
    uint8_t* h_raw = nullptr;
    uint32_t* h_rows = nullptr;
    uint8_t* acts = nullptr;
    std::vector<C::ExpertJobMulti> jobs;
    cudaEvent_t ev = nullptr;
    // the GPU's share of the routed experts: the entries grouped by slot, their input as q8_1, the kernels' scratch
    unsigned long long* grp_ptr = nullptr;
    int32_t *grp_start = nullptr, *n_groups = nullptr, *ent_dst = nullptr, *ent_tok = nullptr;
    int32_t *h_miss = nullptr, *h_miss_d = nullptr;
    uint8_t *rinq = nullptr, *escr = nullptr;
    int64_t hits = 0, lookups = 0;
    std::string profile_path, save_path;   // the profile started from; where the learned one goes
    double save_every_min = 1.0;
    Clock::time_point last_save = Clock::now();
    // counters
    double cpu_ms = 0, table_ms = 0;
    int64_t forwards = 0;
    // LAMINA_HY4_TRACE=1: the host's time of the decode's forwards by part, printed by trace_print (2: the main GPU is
    // also waited for after each MoE layer's host part, which tells its own experts' tail from the next segment's work;
    // 3: no graphs, the main GPU waited for after each phase of a layer: its time by phase, the wall times no longer
    // the engine's)
    const int trace = [] {
        const char* v = std::getenv("LAMINA_HY4_TRACE");
        return v ? std::atoi(v) : 0;
    }();
    static constexpr int kPhases = 14;
    struct Trace {   // sums by a forward's rows
        struct Part {
            double ms = 0, table = 0, route = 0, scan = 0, tier_launch = 0, pool = 0, tier_wait = 0, main_tail = 0,
                   launch = 0;
            int64_t n = 0, layers = 0, pool_layers = 0, tier_layers = 0, cpu_experts = 0, cpu_entries = 0;
        } by[kMaxT + 1];
        double draft = 0, logits = 0;
        double phase[kMaxT + 1][kPhases] = {};
        int64_t l_ent[64] = {}, l_cpu[64] = {};   // by layer: the decode's routed entries, the CPU's of them
    } tr;
    // LAMINA_HY4_ROUTE_DUMP=FILE: the routing as int32 records (a placement policy replayed over a session's): kind,
    // layer, rows, then rows * top_k expert ids.  Kinds: 0 a decode forward's MoE layer, 1 a prompt chunk's, 2 a
    // request's start, 3 its generation's start, 4 a rebalance (its copy budget in place of the layer); 2-4 have no ids
    std::FILE* route_dump = [] {
        const char* v = std::getenv("LAMINA_HY4_ROUTE_DUMP");
        return v ? std::fopen(v, "wb") : nullptr;
    }();
    void dump_route(int32_t kind, int32_t layer, int32_t rows, const int32_t* ids_) {
        if (!route_dump) return;
        const int32_t hdr[3] = {kind, layer, rows};
        std::fwrite(hdr, 4, 3, route_dump);
        if (rows > 0) std::fwrite(ids_, 4, (size_t) rows * (size_t) m->g.top_k, route_dump);
        if (kind >= 2) std::fflush(route_dump);
    }
    static constexpr const char* kPhaseNames[kPhases] = {"the embedding",
                                                         "Engram",
                                                         "the streams' mixes and the FFN input",
                                                         "q_a, q_b, kv_a and the gate",
                                                         "k_b and the cell",
                                                         "the indexer",
                                                         "the attention",
                                                         "v_b, the gate and the output",
                                                         "the router, routed_down "
                                                         "and the copies",
                                                         "its routed experts",
                                                         "the shared expert",
                                                         "the MoE's end",
                                                         "layer 0's FFN",
                                                         "the head"};
    Clock::time_point ph_t;
    int ph_rows = 0;   // the forward's rows (0: not a trunk forward, nothing is marked)
    /// trace level 3: the stream waited for, the time since the last mark added to phase k of this forward's rows
    void ph(int k) {
        if (trace < 3 || ph_rows == 0) return;
        CK(cudaStreamSynchronize(st));
        tr.phase[ph_rows][k] += ms_since(ph_t);
        ph_t = Clock::now();
    }
    static double ms_between(Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    }
    /// The trace since the last print: a line for each row count's forwards (ms a forward by part) and one for all,
    /// then zeroed
    void trace_print(const char* what) {
        if (!trace) return;
        Trace::Part all;
        auto line = [&](const char* rows, const Trace::Part& x) {
            const double f = 1.0 / (double) x.n, L = x.layers ? 1.0 / (double) x.layers : 0.0;
            std::fprintf(
                stderr,
                "lamina: trace (%s): %s: %lld forwards, %.2f ms each: the table %.2f, the waits for a "
                "layer's routing %.2f, the holders %.2f, the tiers' launches %.2f, the pool %.2f (in %.0f%% of "
                "the layers, %.1f experts, %.1f entries), the tiers' rows after it %.2f (a tier in %.0f%%), the "
                "main GPU after them %.2f, the segments' launches %.2f\n",
                what, rows, (long long) x.n, x.ms * f, x.table * f, x.route * f, x.scan * f, x.tier_launch * f,
                x.pool * f, 100.0 * (double) x.pool_layers * L, (double) x.cpu_experts * f, (double) x.cpu_entries * f,
                x.tier_wait * f, 100.0 * (double) x.tier_layers * L, x.main_tail * f, x.launch * f);
        };
        for (int t = 1; t <= kMaxT; ++t) {
            const Trace::Part& x = tr.by[t];
            if (x.n == 0) continue;
            line((std::to_string(t) + (t == 1 ? " row" : " rows")).c_str(), x);
            all.ms += x.ms, all.table += x.table, all.route += x.route, all.scan += x.scan;
            all.tier_launch += x.tier_launch, all.pool += x.pool, all.tier_wait += x.tier_wait;
            all.main_tail += x.main_tail, all.launch += x.launch;
            all.n += x.n, all.layers += x.layers, all.pool_layers += x.pool_layers, all.tier_layers += x.tier_layers;
            all.cpu_experts += x.cpu_experts, all.cpu_entries += x.cpu_entries;
        }
        if (all.n > 0) {
            line("all", all);
            if (trace >= 3)
                for (int t = 1; t <= kMaxT; ++t) {
                    if (tr.by[t].n == 0) continue;
                    std::string txt;
                    double sum = 0;
                    for (int k = 0; k < kPhases; ++k) {
                        char b[96];
                        std::snprintf(b, sizeof b, "%s%s %.2f", k ? ", " : "", kPhaseNames[k],
                                      tr.phase[t][k] / (double) tr.by[t].n);
                        txt += b;
                        sum += tr.phase[t][k];
                    }
                    std::fprintf(stderr, "lamina: trace (%s): the main GPU's ms a forward of %d row%s, %.2f: %s\n",
                                 what, t, t == 1 ? "" : "s", sum / (double) tr.by[t].n, txt.c_str());
                }
            std::fprintf(stderr, "lamina: trace (%s): a forward's logits waited for %.2f ms, its drafts took %.2f\n",
                         what, tr.logits / (double) all.n, tr.draft / (double) all.n);
            std::string by_layer;
            for (int i = 0; i < 64; ++i) {
                if (tr.l_ent[i] == 0) continue;
                char b[32];
                std::snprintf(b, sizeof b, " %d:%.0f", i, 100.0 * (double) tr.l_cpu[i] / (double) tr.l_ent[i]);
                by_layer += b;
            }
            std::fprintf(stderr, "lamina: trace (%s): the CPU's %% of the routed entries by layer:%s\n", what,
                         by_layer.c_str());
        }
        tr = Trace{};
    }

    int n_chunks_max() const {
        const auto chunks = [](int64_t cells) { return (int) ((cells + hy4k::kAttnChunk - 1) / hy4k::kAttnChunk); };
        return std::max({chunks(m->g.idx_top_k), chunks(m->g.n_swa), chunks(dfly_ld)});
    }

    void init(Model& model, int64_t max_ctx, int threads) {
        m = &model;
        if (m->dfly.on)
            dfly_ld =
                (int) ((m->g.idx_top_k + m->dfly.width + hy4k::kAttnChunk - 1) / hy4k::kAttnChunk * hy4k::kAttnChunk);
        const auto& g = m->g;
        ctx = (max_ctx + g.idx_pool - 1) / g.idx_pool * g.idx_pool;
        ring = g.n_swa + kMaxT;
        st = stream_new();
        ev = event_new();
        ev_mtp = event_new();
        for (cudaStream_t* b : {&sb1, &sb2}) {
            *b = stream_new();
        }
        for (cudaEvent_t* e : {&ev_fork, &ev_j1, &ev_j2, &ev_q}) {
            *e = event_new();
        }
        ev_moe = ev;
        const size_t T = kMaxT, N = (size_t) g.n_embd, hc = (size_t) g.hc;
        size_t hmax = 0;
        for (const Layer& ly : m->layers) hmax = std::max<size_t>(hmax, (size_t) ly.H);
        const size_t QH = hmax * (size_t) g.k_head, VH = hmax * (size_t) g.v_head;
        const size_t q81 = 36;   // bytes of a q8_1 block of 32
        const size_t widest = std::max<size_t>({N, (size_t) g.n_ff, QH, VH});
        X = dalloc<float>(T * hc * N);
        xh = dalloc<float>(T * N);
        xn = dalloc<float>(T * N);
        y = dalloc<float>(T * N);
        sy = dalloc<float>(T * N);
        emb = dalloc<float>(T * N);
        post = dalloc<float>(T * hc);
        hc_part = dalloc<float>(T * hy4k::kHcSlices * 9);
        xq = dalloc<uint8_t>(T * widest / 32 * q81);
        qa = dalloc<float>(T * (size_t) g.q_lora);
        qrq = dalloc<uint8_t>(T * (size_t) g.q_lora / 32 * q81);
        q = dalloc<float>(T * QH);
        qnope = dalloc<float>(T * hmax * (size_t) g.nope());
        qnq = dalloc<uint8_t>(T * hmax * (size_t) g.nope() / 32 * q81);
        qabs = dalloc<float>(T * hmax * (size_t) hy4k::kKvRow);
        kva = dalloc<float>(T * (size_t) g.kv_row());
        gt = dalloc<float>(T * VH);
        part = dalloc<float>(T * (size_t) n_chunks_max() * hmax * (hy4k::kLatent + 2));
        att = dalloc<float>(T * hmax * hy4k::kLatent);
        attq = dalloc<uint8_t>(T * hmax * hy4k::kLatent / 32 * q81);
        o = dalloc<float>(T * VH);
        oq = dalloc<uint8_t>(T * VH / 32 * q81);
        fg = dalloc<float>(T * (size_t) g.n_ff);
        fu = dalloc<float>(T * (size_t) g.n_ff);
        fh = dalloc<float>(T * (size_t) g.n_ff);
        hq = dalloc<uint8_t>(T * (size_t) std::max(g.n_ff, g.n_ff_exp) / 32 * q81);
        logits_r = dalloc<float>(T * (size_t) g.n_expert);
        ids = dalloc<int32_t>(T * (size_t) g.top_k);
        wts = dalloc<float>(T * (size_t) g.top_k);
        rin = dalloc<float>(T * (size_t) g.n_latent);
        sg = dalloc<float>(T * (size_t) g.n_ff_exp);
        su = dalloc<float>(T * (size_t) g.n_ff_exp);
        sh = dalloc<float>(T * (size_t) g.n_ff_exp);
        parts = dalloc<float>(T * (size_t) g.top_k * (size_t) g.n_latent);
        moe = dalloc<float>(T * (size_t) g.n_latent);
        moeq = dalloc<uint8_t>(T * (size_t) g.n_latent / 32 * q81);
        const size_t n_mod = g.eng_layers.size(), eh = (size_t) g.eng_heads(), ed = (size_t) g.eng_dim;
        eraw = dalloc<uint8_t>(n_mod * T * eh * ed);
        erows = dalloc<float>(T * eh * ed);
        erq = dalloc<uint8_t>(T * eh * ed / 32 * q81);
        ekv = dalloc<float>(T * (hc + 1) * N);
        evalid = dalloc<float>(T);
        const size_t ih = (size_t) g.idx_heads, idd = (size_t) g.idx_dim;
        iq = dalloc<float>(T * ih * idd);
        ik = dalloc<float>(T * idd);
        ikq = dalloc<uint8_t>(T * idd / 32 * q81);
        ivg = dalloc<float>(T * 2 * idd);
        iw = dalloc<float>(T * ih);
        scores = dalloc<float>(T * (size_t) (ctx / g.idx_pool));
        cells_swa = dalloc<int32_t>(T * (size_t) g.n_swa);
        cells_sel = dalloc<int32_t>(T * (size_t) g.idx_top_k);
        n_swa = dalloc<int32_t>(T);
        n_sel = dalloc<int32_t>(T);
        logits = dalloc<float>(T * (size_t) g.n_vocab);
        tok_d = dalloc<int32_t>(T);
        pos_d = dalloc<int32_t>(T);
        h_rin = halloc<float>(T * (size_t) g.n_latent);
        h_out = hmapped<float>(T * (size_t) g.top_k * (size_t) g.n_latent, &h_out_d);
        h_valid = halloc<float>(T);
        h_ids = halloc<int32_t>(T * (size_t) g.top_k);
        h_tok = halloc<int32_t>(T);
        pick_d = dalloc<int32_t>(2);
        h_pick = halloc<int32_t>(2);
        h_pos = halloc<int32_t>(T);
        if (vision) {
            img_rows_d = dalloc<float>(T * N);
            img_slot_d = dalloc<int32_t>(T);
            h_img = halloc<float>(T * N);
            h_slot = halloc<int32_t>(T);
            mrope_d = dalloc<int32_t>((size_t) ctx * 3);
            mr.s1 = (int) g.mrope[1];
            mr.s2 = (int) g.mrope[2];
            mrope_identity = false;
            mrope_set(nullptr);
            mr.tab = mrope_d;
        }
        for (size_t li = 0; li < m->layers.size(); ++li)
            if (!m->layers[li].dense) moe_layers.push_back((int) li);
        h_raw = halloc<uint8_t>(n_mod * T * eh * ed);
        h_rows = halloc<uint32_t>(n_mod * T * eh);
        acts = (uint8_t*) heap_new(T * C::kNativeActBytes);
        const size_t n_ent = T * (size_t) g.top_k;
        grp_ptr = dalloc<unsigned long long>(n_ent + 1);
        grp_start = dalloc<int32_t>(n_ent + 2);
        n_groups = dalloc<int32_t>(1);
        ent_dst = dalloc<int32_t>(n_ent);
        ent_tok = dalloc<int32_t>(n_ent);
        h_miss = hmapped<int32_t>(n_ent, &h_miss_d);
        rinq = dalloc<uint8_t>(T * (size_t) g.n_latent / 32 * q81);
        escr = dalloc<uint8_t>(K::native_expert_scratch_bytes((int64_t) n_ent, g.n_ff_exp));
        // the caches: the window layers' rings; the sparse layers' cells and pooled keys for the whole context, or
        // (grown) address ranges for it that ensure() maps as the positions come
        elastic = elastic && lamina::core::vmm_available();
        cl.assign(m->layers.begin(), m->layers.end());
        if (m->has_mtp) cl.push_back(std::ref(m->mtp));
        size_t kv_bytes = 0;
        for (Layer& l : cl) {
            l.kv_cells = l.swa ? ring : ctx;
            l.kv_q8 = kv_q8 && !l.swa;   // the window layers' rings stay halves (512 cells each)
            const size_t kb = (size_t) l.kv_cells * l.kv_row(), pb = (size_t) (ctx / g.idx_pool) * idd * 4;
            if (!l.swa && elastic) {
                l.kv_vm = std::make_shared<lamina::core::VmmRange>();
                l.pooled_vm = std::make_shared<lamina::core::VmmRange>();
                if (!l.kv_vm->reserve(kb) || !l.pooled_vm->reserve(pb)) {
                    std::fprintf(stderr, "lamina: the K/V's address range could not be reserved\n");
                    throw Fatal{1};
                }
                l.kv = l.kv_vm->base();
                l.pooled = (float*) l.pooled_vm->base();
            } else {
                l.kv = dalloc<uint8_t>(kb);
                if (l.has_idx) l.pooled = dalloc<float>((size_t) (ctx / g.idx_pool) * idd);
            }
            kv_bytes += kb + (l.has_idx ? pb : 0);
            if (l.has_idx) l.tail = dalloc<float>((size_t) kTail * 2 * idd);
        }
        if (elastic) ensure(std::min<int64_t>(ctx, kKvStep));
        CK(cudaFuncSetAttribute(hy4p::k_attn_dec<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                (int) sizeof(hy4p::AttnSmem)));
        CK(cudaFuncSetAttribute(hy4p::k_attn_dec<false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                (int) sizeof(hy4p::AttnSmem)));
        if (m->dfly.on) {
            const size_t nl = m->dfly.layers.size(), W = (size_t) m->dfly.width;
            snap_cells = dalloc<int32_t>(nl * T * (size_t) g.idx_top_k);
            snap_n = dalloc<int32_t>(nl * T);
            dfly_cells = dalloc<int32_t>(nl * W * (size_t) dfly_ld);
            dfly_n = dalloc<int32_t>(nl * W);
            dfly_h = dalloc<float>(W * N);
            dfly_res = dalloc<float>(W * N);
            dfly_out = dalloc<float>(W * N);
            dfly_cat = dalloc<float>(2 * N);
            dfly_catq = dalloc<uint8_t>(2 * N / 32 * q81);
            dfly_d = dalloc<float>(N);
            dfly_lg = dalloc<float>((size_t) g.n_vocab);
            dfly_prm_d = dalloc<int32_t>(4);
            h_dfly_prm = halloc<int32_t>(4);
            dfly_ring = dalloc<float>(N);
            for (size_t i = 0; i < nl; ++i) {   // the K/V of its trunk layer; its cells from the block's pass
                Layer& d = m->dfly.layers[i];
                d.kv = m->layers[(size_t) m->dfly.target[i]].kv;
                d.kv_q8 = m->layers[(size_t) m->dfly.target[i]].kv_q8;
                d.ext_cells = dfly_cells + i * W * (size_t) dfly_ld;
                d.ext_n = dfly_n + i * W;
                d.ext_ld = dfly_ld;
            }
        }
        if (m->has_mtp || m->dfly.on) {
            hring = dalloc<float>((size_t) kHRing * N);
            CK(cudaMemset(hring, 0, (size_t) kHRing * N * 4));
        }
        if (m->has_mtp) {
            mtp_cat = dalloc<float>(T * 2 * N);
            mtp_catq = dalloc<uint8_t>(T * 2 * N / 32 * q81);
            mtp_x = dalloc<float>(T * N);
            mtp_hs = dalloc<float>(T * N);
            mtp_h = dalloc<float>(T * N);
            mtp_lg = dalloc<float>((size_t) g.n_vocab);
            mtp_src = dalloc<int32_t>(T);
            h_src = halloc<int32_t>(T);
            m->mtp.use.assign((size_t) g.n_expert, 0.f);
            m->mtp.total.assign((size_t) g.n_expert, 0.f);
        }
        std::fprintf(stderr, "lamina: %lld cells of context: K/V (%s) and pooled keys %.2f GiB%s\n", (long long) ctx,
                     kv_q8 ? "q8_0" : "fp16", (double) kv_bytes / (double) (1ull << 30),
                     elastic ? ", mapped as the context grows (the main GPU's expert slots give way)" : "");
        pool = std::make_unique<C::ExpertPool>(threads);
        std::string err;
        if (!reader.open(m->table_path, m->table_off, m->g.eng_rows, 256, (uint64_t) 1 << 20, err, true,
                         (uint32_t) m->g.eng_dim)) {
            std::fprintf(stderr, "lamina: the Engram table: %s\n", err.c_str());
            throw Fatal{1};
        }
        reader.set_keepalive(100.0, 30.0);
    }

    int64_t pos() const { return (int64_t) seq.size(); }

    // ---- the K/V grown with the context: the sparse layers' cells mapped up to kv_mapped (a multiple of kKvStep), the
    // physical chunks taken from the free VRAM above kReserve, else from the main GPU's slots (shrink_main)
    bool elastic = true;
    bool kv_q8 = false;   // --kv q8_0: the sparse layers' cells in q8_0 rows (Layer::kv_q8)
    static constexpr int64_t kKvStep = 16384;
    // the main GPU's VRAM left free (the slots and the K/V's growth stop there): 1.5 GiB on a GPU a display or a
    // graphics program uses (NVML; also when it cannot say), 700 MiB on one without (a session's requests left 40 MiB
    // of the 1.5 GiB used: graphs, the prompt path's first run)
    size_t kReserve = (size_t) 1536 << 20;
    int64_t kv_mapped = 0;
    std::vector<lamina::core::VmmChunk> spare;
    std::vector<std::reference_wrapper<Layer>> cl;   // the layers with caches: the trunk, then the MTP block
    // the drafters' input: the trunk's normed final hidden rows of the last kHRing positions; the MTP block's buffers
    static constexpr int kHRing = 16;
    float *hring = nullptr, *mtp_cat = nullptr, *mtp_x = nullptr, *mtp_hs = nullptr, *mtp_h = nullptr,
          *mtp_lg = nullptr;
    uint8_t* mtp_catq = nullptr;
    int32_t *mtp_src = nullptr, *h_src = nullptr;
    bool drafts_on = false;   // drafted windows: the MTP block kept up with the trunk, or DFly's inputs kept (m->dfly)
    // DFly2: the trunk's picks of its target layers for the rows of the last pass (snap_rows of them from position
    // snap_base; -1: none), its block's cells, rows and buffers
    int dfly_ld = 0;   // a block row's cells: the trunk's top-k picks and the block, in whole attention chunks
    int32_t *snap_cells = nullptr, *snap_n = nullptr, *dfly_cells = nullptr, *dfly_n = nullptr;
    float *dfly_h = nullptr, *dfly_res = nullptr, *dfly_out = nullptr, *dfly_cat = nullptr, *dfly_d = nullptr,
          *dfly_lg = nullptr;
    uint8_t* dfly_catq = nullptr;
    int64_t snap_base = -1;
    int snap_rows = 0;
    /// The DFly snapshot slot of trunk layer il, or -1
    int dfly_slot(int il) const {
        for (size_t i = 0; i < m->dfly.target.size(); ++i)
            if (m->dfly.target[i] == il) return (int) i;
        return -1;
    }
    float spec_min_p = 0.f;                           // a window's drafts while their joint probability
                                                      // under the drafter's head stays >= this
    cudaEvent_t ev_moe = nullptr, ev_mtp = nullptr;   // the MoE block's routing event: ev (the decode's), ev_mtp
    ~Engine() {                                       // (the ranges free their own chunks with the model)
        if (route_dump) std::fclose(route_dump);
        for (const lamina::core::VmmChunk h : spare) lamina::core::vmm_chunk_free(h);
    }
    int s_max = 0;   // the main GPU's slots at the start
    /// Chunks the sparse layers' ranges need mapped for `cells` cells that they do not have
    int64_t kv_need(int64_t cells) const {
        const uint64_t G = lamina::core::vmm_granularity();
        int64_t n = 0;
        for (const Layer& l : cl) {
            if (!l.kv_vm) continue;
            const int64_t kc = (int64_t) (((uint64_t) cells * l.kv_row() + G - 1) / G);
            const int64_t pc =
                (int64_t) (((uint64_t) (cells / m->g.idx_pool) * (uint64_t) m->g.idx_dim * 4 + G - 1) / G);
            for (int64_t c = 0; c < std::min(kc, l.kv_vm->chunks()); ++c) n += l.kv_vm->mapped(c) ? 0 : 1;
            for (int64_t c = 0; c < std::min(pc, l.pooled_vm->chunks()); ++c) n += l.pooled_vm->mapped(c) ? 0 : 1;
        }
        return n;
    }
    /// Room for `cells` cells (a step at a time): chunks from the free VRAM above the reserve, then the main GPU's
    /// slots'.  The GPU is idle first (its slots may be read).
    void ensure(int64_t cells) {
        if (!elastic || cells <= kv_mapped) return;
        const int64_t target = std::min<int64_t>(ctx, (cells + kKvStep - 1) / kKvStep * kKvStep);
        const int64_t need = kv_need(target);
        CK(cudaStreamSynchronize(st));
        const uint64_t G = lamina::core::vmm_granularity();
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        const int64_t free_chunks = fr > kReserve ? (int64_t) ((fr - kReserve) / G) : 0;
        if ((int64_t) spare.size() + free_chunks < need) {
            settle(true);   // no copy in flight into a slot that moves or goes
            shrink_main(need - (int64_t) spare.size() - free_chunks);
        }
        auto take = [&]() -> lamina::core::VmmChunk {
            if (spare.empty()) return 0;
            const lamina::core::VmmChunk h = spare.back();
            spare.pop_back();
            return h;
        };
        for (Layer& l : cl) {
            if (!l.kv_vm) continue;
            const int64_t kc = (int64_t) (((uint64_t) target * l.kv_row() + G - 1) / G);
            const int64_t pc =
                (int64_t) (((uint64_t) (target / m->g.idx_pool) * (uint64_t) m->g.idx_dim * 4 + G - 1) / G);
            if (!l.kv_vm->map_range(0, kc, take) || !l.pooled_vm->map_range(0, pc, take)) {
                std::fprintf(stderr, "lamina: no VRAM for %lld cells of context\n", (long long) target);
                throw Fatal{1};
            }
        }
        kv_mapped = target;
    }
    /// The main GPU's slots fewer by enough for `chunks` chunks: each MoE layer keeps its most used residents in the
    /// slots that stay (moved down there), the others go; the chunks past the slots that stay go to `spare`.
    void shrink_main(int64_t chunks) {
        if (gpus.empty() || gpus[0].S <= 0) {
            std::fprintf(stderr, "lamina: no VRAM left for the context\n");
            throw Fatal{1};
        }
        const uint64_t G = lamina::core::vmm_granularity();
        int S = gpus[0].S;
        auto freed = [&](int s2) {   // chunks past s2 slots in every layer
            int64_t n = 0;
            for (const Layer& l : m->layers)
                if (l.slot_vm)
                    for (int64_t c = (int64_t) (((uint64_t) s2 * l.stride + G - 1) / G); c < l.slot_vm->chunks(); ++c)
                        n += l.slot_vm->mapped(c) ? 1 : 0;
            return n;
        };
        int s2 = S;
        while (s2 > 0 && freed(s2) < chunks) --s2;
        if (freed(s2) < chunks) {
            std::fprintf(stderr, "lamina: no VRAM left for the context\n");
            throw Fatal{1};
        }
        int moved = 0, gone = 0;
        for (Layer& l : m->layers) {
            if (l.dense || l.held.empty() || !l.slot_vm) continue;
            SlotSet& sl = l.held[0];
            std::vector<int32_t> res;   // the residents, most used first
            for (int k = 0; k < sl.n; ++k)
                if (sl.slot_expert[(size_t) k] >= 0) res.push_back(sl.slot_expert[(size_t) k]);
            std::stable_sort(res.begin(), res.end(),
                             [&](int32_t a, int32_t b) { return l.use[(size_t) a] > l.use[(size_t) b]; });
            std::vector<char> keep(l.use.size(), 0);
            for (size_t i = 0; i < res.size() && (int) i < s2; ++i) keep[(size_t) res[i]] = 1;
            std::vector<int> holes;   // slots that stay, not holding a kept expert
            for (int k = 0; k < s2; ++k) {
                const int32_t e = sl.slot_expert[(size_t) k];
                if (e < 0 || !keep[(size_t) e]) holes.push_back(k);
            }
            size_t hi = 0;
            for (int k = s2; k < sl.n; ++k) {   // a kept expert above the line moves into a hole
                const int32_t e = sl.slot_expert[(size_t) k];
                if (e < 0) continue;
                if (keep[(size_t) e] && hi < holes.size()) {
                    const int h = holes[hi++];
                    if (sl.slot_expert[(size_t) h] >= 0) {
                        sl.slot_of[(size_t) sl.slot_expert[(size_t) h]] = -1;
                        ++gone;
                    }
                    CK(cudaMemcpyAsync(sl.slots + (size_t) h * l.stride, sl.slots + (size_t) k * l.stride, l.fmt.bytes,
                                       cudaMemcpyDeviceToDevice, st));
                    sl.slot_expert[(size_t) h] = e;
                    sl.slot_of[(size_t) e] = h;
                    ++moved;
                } else {
                    sl.slot_of[(size_t) e] = -1;
                    ++gone;
                }
                sl.slot_expert[(size_t) k] = -1;
            }
            sl.n = s2;
            sl.slot_expert.resize((size_t) s2);
            upload_table(l, 0);
        }
        CK(cudaStreamSynchronize(st));
        for (Layer& l : m->layers)
            if (l.slot_vm)
                for (int64_t c = (int64_t) (((uint64_t) s2 * l.stride + G - 1) / G); c < l.slot_vm->chunks(); ++c)
                    if (l.slot_vm->mapped(c)) spare.push_back(l.slot_vm->unmap(c));
        gpus[0].S = s2;
        std::fprintf(stderr,
                     "lamina: the context takes the main GPU's expert slots %d -> %d a layer (%d experts moved "
                     "down, %d gone)\n",
                     S, s2, moved, gone);
    }
    /// the layers whose experts the main GPU's slots hold (a slot a layer given up is this many slots)
    int lent_layers() const {
        int n = 0;
        for (const Layer& l : m->layers) n += l.slot_vm ? 1 : 0;
        return n;
    }
    /// n chunks for a buffer the prompt path maps for a while: the spare ones, the free VRAM above kReserve (new
    /// chunks), the rest from the main GPU's last slots (shrink_main puts theirs in spare)
    bool lend_chunks(int64_t n) {
        if (!elastic) return false;
        if ((int64_t) spare.size() >= n) return true;
        const uint64_t G = lamina::core::vmm_granularity();
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        const int64_t free_chunks = fr > kReserve ? (int64_t) ((fr - kReserve) / G) : 0;
        const int64_t short_by = n - (int64_t) spare.size() - free_chunks;
        if (short_by <= 0) return true;
        if (gpus.empty() || gpus[0].S <= 0) return false;
        {   // the slots' chunks past their first quarter at most (a smaller card keeps some of its slots, or declines)
            int64_t can = 0;
            for (const Layer& l : m->layers)
                if (l.slot_vm)
                    for (int64_t c = (int64_t) (((uint64_t) (gpus[0].S / 4) * l.stride + G - 1) / G);
                         c < l.slot_vm->chunks(); ++c)
                        can += l.slot_vm->mapped(c) ? 1 : 0;
            if (short_by > can) return false;
        }
        CK(cudaStreamSynchronize(st));
        settle(true);                             // no copy in flight into a slot that moves or goes
        lent_from.assign(m->layers.size(), {});   // the residents, for queue_restore
        for (size_t li = 0; li < m->layers.size(); ++li) {
            const Layer& l = m->layers[li];
            if (l.dense || l.held.empty()) continue;
            for (const int32_t e : l.held[0].slot_expert)
                if (e >= 0) lent_from[li].push_back(e);
        }
        shrink_main(short_by);
        return true;
    }
    /// The main GPU's residents before a lend (lend_chunks), and the copies that put the ones it displaced back into
    /// the slots that came back (queue_restore), served by pump ahead of the plan's
    std::vector<std::vector<int32_t>> lent_from;
    std::vector<Planned> restore;
    size_t restore_next = 0;
    /// The residents a lend displaced, queued back into the main GPU's empty slots (the most used first): the state the
    /// answer had before the prompt, not the prompt's routing's
    int queue_restore() {
        restore.clear();
        restore_next = 0;
        for (size_t li = 0; li < lent_from.size() && li < m->layers.size(); ++li) {
            Layer& l = m->layers[li];
            if (lent_from[li].empty() || l.held.empty()) continue;
            std::vector<int32_t> back;
            for (const int32_t e : lent_from[li]) {
                bool held = l.moving[(size_t) e] != 0;
                for (const SlotSet& hs : l.held) held = held || (hs.n > 0 && hs.slot_of[(size_t) e] >= 0);
                if (!held) back.push_back(e);
            }
            std::stable_sort(back.begin(), back.end(),
                             [&](int32_t a, int32_t b) { return l.use[(size_t) a] > l.use[(size_t) b]; });
            const SlotSet& s0 = l.held[0];
            size_t bi = 0;
            for (int k = 0; k < s0.n && bi < back.size(); ++k)
                if (s0.slot_expert[(size_t) k] == -1) restore.push_back({1e9f, (int) li, 0, k, back[bi++], -1});
        }
        lent_from.clear();
        const int n = (int) restore.size();
        pump();
        if (!adapt) drain();
        return n;
    }
    /// The chunks the main GPU's slots of every layer lack for n slots a layer
    int64_t slot_chunks_lacking(int n) const {
        const uint64_t G = lamina::core::vmm_granularity();
        int64_t k = 0;
        for (const Layer& l : m->layers)
            if (l.slot_vm) {
                const int64_t hi = (int64_t) (((uint64_t) n * l.stride + G - 1) / G);
                for (int64_t c = 0; c < std::min(hi, l.slot_vm->chunks()); ++c) k += l.slot_vm->mapped(c) ? 0 : 1;
            }
        return k;
    }
    /// The main GPU's slots up to s2 a layer, their chunks the spare ones first, else new: they come back empty, the
    /// mover's next plan fills them.  A map that fails leaves every layer at the count it had (what it mapped stays,
    /// for a later growth).
    void grow_slots(int s2) {
        const int S = gpus[0].S;
        const uint64_t G = lamina::core::vmm_granularity();
        auto take = [&]() -> lamina::core::VmmChunk {
            if (spare.empty()) return 0;
            const lamina::core::VmmChunk h = spare.back();
            spare.pop_back();
            return h;
        };
        for (Layer& l : m->layers) {
            if (!l.slot_vm) continue;
            const int64_t hi = (int64_t) (((uint64_t) s2 * l.stride + G - 1) / G);
            if (!l.slot_vm->map_range(0, hi, take)) {
                std::fprintf(stderr, "lamina: the main GPU's slots could not grow back past %d\n", S);
                return;
            }
        }
        for (Layer& l : m->layers) {
            if (!l.slot_vm) continue;
            l.held[0].n = s2;
            l.held[0].slot_expert.resize((size_t) s2, -1);
        }
        gpus[0].S = s2;
        std::fprintf(stderr, "lamina: the main GPU's expert slots back %d -> %d a layer\n", S, s2);
    }
    /// The main GPU's slots back toward s_max from the spare chunks and the free VRAM above kReserve (after the prompt
    /// path's lent buffers went)
    void slots_back() {
        const int top = s_max + (base_lent ? lend_slots : 0);
        if (!elastic || gpus.empty() || gpus[0].S >= top) return;
        const uint64_t G = lamina::core::vmm_granularity();
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        const int64_t avail = (int64_t) spare.size() + (fr > kReserve ? (int64_t) ((fr - kReserve) / G) : 0);
        const int S = gpus[0].S;
        int s2 = S;
        while (s2 < top && slot_chunks_lacking(s2 + 1) <= avail) ++s2;
        if (s2 > S) grow_slots(s2);
    }
    /// After a request: the K/V's chunks past what the state needs (two steps of room) go back to the main GPU's slots,
    /// which come back empty (the rebalance fills them); chunks no slot takes are freed.
    void relax() {
        if (!elastic || gpus.empty()) return;
        const uint64_t G = lamina::core::vmm_granularity();
        const int64_t target = std::min<int64_t>(ctx, (pos() + 2 * kKvStep + kKvStep - 1) / kKvStep * kKvStep);
        if (kv_mapped <= target || gpus[0].S >= s_max) return;
        CK(cudaDeviceSynchronize());
        for (Layer& l : cl) {
            if (!l.kv_vm) continue;
            const int64_t kc = (int64_t) (((uint64_t) target * l.kv_row() + G - 1) / G);
            const int64_t pc =
                (int64_t) (((uint64_t) (target / m->g.idx_pool) * (uint64_t) m->g.idx_dim * 4 + G - 1) / G);
            for (int64_t c = kc; c < l.kv_vm->chunks(); ++c)
                if (l.kv_vm->mapped(c)) spare.push_back(l.kv_vm->unmap(c));
            for (int64_t c = pc; c < l.pooled_vm->chunks(); ++c)
                if (l.pooled_vm->mapped(c)) spare.push_back(l.pooled_vm->unmap(c));
        }
        kv_mapped = target;
        // the slots back while the spare chunks cover a whole slot more in every layer
        const int S = gpus[0].S;
        int s2 = S;
        while (s2 < s_max && slot_chunks_lacking(s2 + 1) <= (int64_t) spare.size()) ++s2;
        if (s2 > S) grow_slots(s2);
        for (const lamina::core::VmmChunk h : spare) lamina::core::vmm_chunk_free(h);
        spare.clear();
    }

    // ---- a prompt part: chunks through the prompt path (Prompt, when it is set up and the part has chunk_min tokens
    // or more: pieces as even as chunk_cap allows), else forwards of kMaxT rows
    std::function<void(const int32_t*, int)> chunk_fn;
    int64_t chunk_cap = 0, chunk_min = 0;
    // --prefill-pass: a part of two chunks or more in chunks of up to pass_cap rows while pass_fn(true) has their
    // buffers (pass_fn(false) after the part gives them back)
    int64_t pass_cap = 0;
    std::function<bool(bool)> pass_fn;
    /// n tokens into the state; `on(done)` after each piece (false: stop there).  Returns the logits row of the last
    /// token (host_logits), -1 when stopped first.
    /// Why a snapshot is kept (keep_snap): a prompt's end (the last kSnaps), a multiple of kSnapEvery a read passed
    /// (all), behind a prompt's last picture or video (the last kMediaSnaps), where a later prompt may part from this
    /// one (the last kPartSnaps), before a prompt's last message (the last kLastSnaps)
    enum class Kept : uint8_t { end, every, media, part, last };
    std::vector<int64_t> chunk_snaps;   // the snapshots inside the next chunk: the multiples of kSnapEvery, `wanted`'s
    // the next read's snapshots besides the multiples of kSnapEvery (request()), ascending, where the prompt path's
    // pieces pass them: inside a piece or at a piece's end
    std::vector<std::pair<int64_t, Kept>> wanted;
    /// a snapshot's kind at p: a wanted one's, else a multiple's
    Kept kept_at(int64_t p) const {
        for (const auto& [at, k] : wanted)
            if (at == p) return k;
        return Kept::every;
    }
    int feed(const int32_t* toks, int64_t n, const std::function<bool(int64_t)>& on) {
        int64_t done = 0;
        int last = 0;
        while (done < n) {
            // the prompt path: the rest in pieces as even as chunk_cap allows, a piece that crosses a multiple of
            // kSnapEvery taking its snapshot there; the decode's forwards: a segment up to the next multiple (a
            // snapshot there) or the end
            const int64_t p = pos();
            const bool chunks = chunk_fn && n - done >= chunk_min;
            if (chunks && base_lent && base_fn) base_fn(false);
            const int64_t seg = chunks ? n - done : std::min<int64_t>(n - done, (p / kSnapEvery + 1) * kSnapEvery - p);
            const bool passes = chunks && pass_fn && pass_cap > chunk_cap && seg >= 2 * chunk_cap && pass_fn(true);
            const int64_t step = passes   ? pass_cap
                                 : chunks ? chunk_cap
                                          : kMaxT,
                          pieces = (seg + step - 1) / step, size = (seg + pieces - 1) / pieces;
            struct PassEnd {   // the pass's buffers back however the part ends
                const std::function<bool(bool)>* fn;
                ~PassEnd() {
                    if (fn) (*fn)(false);
                }
            } pass_end{passes ? &pass_fn : nullptr};
            for (int64_t sd = 0; sd < seg;) {
                const int T = (int) std::min<int64_t>(size, seg - sd);
                settle(false);
                if (moves_waiting()) {   // the slots it empties idle first: the last piece's work done
                    CK(cudaStreamSynchronize(st));
                    pump();
                }
                if (chunks) {
                    const int64_t p0 = pos();
                    chunk_snaps.clear();
                    for (int64_t mk = (p0 / kSnapEvery + 1) * kSnapEvery; mk < p0 + T; mk += kSnapEvery)
                        chunk_snaps.push_back(mk);
                    for (const auto& w : wanted)
                        if (w.first > p0 && w.first < p0 + T && w.first % kSnapEvery != 0)
                            chunk_snaps.push_back(w.first);
                    std::sort(chunk_snaps.begin(), chunk_snaps.end());
                    chunk_fn(toks + done + sd, T);
                    chunk_snaps.clear();
                    // the piece ends at a multiple, or where a wanted one is
                    if ((pos() % kSnapEvery == 0 || kept_at(pos()) != Kept::every) && done + sd + T < n) {
                        std::vector<float> lg;
                        host_logits(0, lg);
                        snapshot(lg, kept_at(pos()));
                    }
                } else {
                    const int64_t p0 = pos();
                    forward(toks + done + sd, T);
                    if (drafts_on && m->has_mtp) mtp_catch_up(toks + done + sd, T, p0);
                }
                sd += T;
                last = chunks ? 0 : T - 1;
                if (on && !on(done + sd) && done + sd < n) return -1;
            }
            done += seg;
            if (!chunks && pos() % kSnapEvery == 0 && done < n) {
                std::vector<float> lg;
                host_logits(last, lg);
                snapshot(lg, Kept::every);
            }
        }
        return last;
    }
    /// A snapshot taken inside a chunk at `p` (Prompt::run): its bytes as snapshot() lays them out, no logits
    void add_snapshot(int64_t p, std::vector<uint8_t>&& bytes, Kept kept) {
        if (snap_at(p) != nullptr) return;
        Snap sn;
        sn.pos = p;
        sn.kept = kept;
        sn.bytes = std::move(bytes);
        keep_snap(std::move(sn));
    }

    // ---- going back.  The sparse layers' cells and the pooled keys are by position: what a shorter state keeps is
    // already right.  The window layers' rings and the indexer's tail hold the last cells only: going back up to
    // kMaxT tokens leaves them whole (the ring has kMaxT cells to spare, the tail 16); further back takes a snapshot of
    // both taken at or before the point (snapshot(): at each prompt's end), else the state starts over.  A read keeps
    // one at each multiple of kSnapEvery it passes, and at the places a later prompt may part from this one: behind
    // its last picture or video (another question about it in a new conversation continues there), where it leaves
    // the request before (agents send one long start again with another tail), a few cells before the end of a last
    // message that grows between requests, and before its last message (a client that sends that message again after
    // each new turn leaves the prompt there, a sibling with another first message after one system prompt too).  Each
    // at the last multiple of the indexer's pool at or before its place, where a snapshot inside a chunk needs no tail
    // (it keeps the tails as the chunk leaves them).
    struct Snap {
        int64_t pos = 0;
        std::vector<uint8_t> bytes;   // every window layer's ring, then every indexer layer's tail
        std::vector<float> logits;    // the last row's (the next token's distribution there)
        Kept kept = Kept::end;
    };
    std::vector<Snap> snaps;
    static constexpr size_t kSnaps = 8;           // prompt ends kept
    static constexpr size_t kMediaSnaps = 4;      // snapshots behind pictures kept
    static constexpr size_t kPartSnaps = 4;       // snapshots where prompts part kept
    static constexpr size_t kLastSnaps = 2;       // snapshots before a prompt's last message kept (the next request's)
    static constexpr int64_t kSnapEvery = 8192;   // a read keeps a snapshot at each multiple of this it passes
    /// s into snaps, the oldest of its kind out where it has its count already (all but the multiples')
    void keep_snap(Snap&& s) {
        if (s.kept != Kept::every) {
            const size_t cap = s.kept == Kept::end     ? kSnaps
                               : s.kept == Kept::media ? kMediaSnaps
                               : s.kept == Kept::last  ? kLastSnaps
                                                       : kPartSnaps;
            const auto kind = [&](const Snap& x) { return x.kept == s.kept; };
            if ((size_t) std::count_if(snaps.begin(), snaps.end(), kind) >= cap)
                snaps.erase(std::find_if(snaps.begin(), snaps.end(), kind));
        }
        snaps.push_back(std::move(s));
    }
    size_t snap_bytes() const {
        size_t n = 0;
        for (const Layer& l : cl) {
            if (l.swa) n += (size_t) l.kv_cells * l.kv_row();
            if (l.has_idx) n += (size_t) kTail * 2 * (size_t) m->g.idx_dim * 4;
        }
        if (hring) n += (size_t) kHRing * (size_t) m->g.n_embd * 4;   // the MTP block's states
        return n;
    }
    void snapshot(const std::vector<float>& logits_now, Kept kept = Kept::end) {
        if (snap_at(pos()) != nullptr) return;
        Snap s;
        s.pos = pos();
        s.kept = kept;
        s.logits = logits_now;
        s.bytes.resize(snap_bytes());
        size_t at = 0;
        for (const Layer& l : cl) {
            if (l.swa) {
                const size_t n = (size_t) l.kv_cells * l.kv_row();
                CK(cudaMemcpyAsync(s.bytes.data() + at, l.kv, n, cudaMemcpyDeviceToHost, st));
                at += n;
            }
            if (l.has_idx) {
                const size_t n = (size_t) kTail * 2 * (size_t) m->g.idx_dim * 4;
                CK(cudaMemcpyAsync(s.bytes.data() + at, l.tail, n, cudaMemcpyDeviceToHost, st));
                at += n;
            }
        }
        if (hring)
            CK(cudaMemcpyAsync(s.bytes.data() + at, hring, (size_t) kHRing * m->g.n_embd * 4, cudaMemcpyDeviceToHost,
                               st));
        CK(cudaStreamSynchronize(st));
        keep_snap(std::move(s));
    }
    /// The snapshot at exactly `p`, or null
    const Snap* snap_at(int64_t p) const {
        for (const Snap& s : snaps)
            if (s.pos == p) return &s;
        return nullptr;
    }
    /// The state back to (at most) its first `keep` tokens: the cells left whole, a snapshot's, or none.  Returns the
    /// tokens kept.
    int64_t rewind(int64_t keep, bool from_snap = false) {
        keep = std::min<int64_t>(keep, pos());
        if (pos() - keep <= kMaxT && !from_snap) {
            seq.resize((size_t) keep);
            sfx_ok = std::min(sfx_ok, seq.size());
        } else {
            const Snap* best = nullptr;
            for (const Snap& s : snaps)
                if (s.pos <= keep && (!best || s.pos > best->pos)) best = &s;
            if (best == nullptr) {
                seq.clear();
                sfx_ok = 0;
            } else {
                size_t at = 0;
                for (const Layer& l : cl) {
                    if (l.swa) {
                        const size_t n = (size_t) l.kv_cells * l.kv_row();
                        CK(cudaMemcpyAsync(l.kv, best->bytes.data() + at, n, cudaMemcpyHostToDevice, st));
                        at += n;
                    }
                    if (l.has_idx) {
                        const size_t n = (size_t) kTail * 2 * (size_t) m->g.idx_dim * 4;
                        CK(cudaMemcpyAsync(l.tail, best->bytes.data() + at, n, cudaMemcpyHostToDevice, st));
                        at += n;
                    }
                }
                if (hring)
                    CK(cudaMemcpyAsync(hring, best->bytes.data() + at, (size_t) kHRing * m->g.n_embd * 4,
                                       cudaMemcpyHostToDevice, st));
                CK(cudaStreamSynchronize(st));
                seq.resize((size_t) best->pos);
                sfx_ok = std::min(sfx_ok, seq.size());
            }
            snap_base = -1;   // DFly's picks are not a snapshot's
        }
        snaps.erase(std::remove_if(snaps.begin(), snaps.end(), [&](const Snap& s) { return s.pos > pos(); }),
                    snaps.end());
        if (live_file) {   // (the file it came back from: its cells and snapshots past here are no longer its own)
            live_file->cut(pos());
            for (size_t i = live_file->slots.size(); i-- > 0;)
                if (live_file->slots[i].first > pos()) live_file->forget(live_file->slots[i].first);
            if (pos() == 0) live_file.reset();
        }
        return pos();
    }

    // ---- parked conversations: one a request leaves goes to RAM (its sparse layers' cells and pooled keys up to its
    // end, its snapshots and one at its end), for a later request that continues it; the least recently used leave
    // when the park passes its budget or the RAM available to the system falls under the reserve, only where the one
    // coming in then fits, and none while it lends its start.  A request starts from the longest start of its prompt
    // that the live state or a parked conversation continues from (reusable()); a parked one's own next request
    // (lamina::core::superseded) takes it back, any other request borrows its start and leaves it parked whole
    // (borrow()).  With a disk tier (--conversation-spill-dir, lamina/core/disk_tier.hpp) one the RAM does not hold,
    // or that leaves it, goes to files instead - its cells and pooled keys from the device a range at a time, its
    // snapshots as records - and comes back from them; one that comes back keeps its file while it is live, so its
    // next park writes only what it lacks.
    using File = std::unique_ptr<lamina::core::DiskFile, FileGone>;
    struct SnapRec {   // a snapshot in a file: its record is its bytes, then its logits
        int64_t pos;
        Kept kept;
        size_t n_logits;
    };
    struct Parked {
        std::vector<int32_t> seq;
        std::vector<uint8_t> cells;   // per sparse layer: K/V rows [0, pos), then pooled keys [0, pos / pool)
        std::vector<Snap> snaps;
        uint64_t used = 0;
        File file;   // on the disk tier: its cells and snapshots there (`cells` and `snaps` empty)
        std::vector<SnapRec> recs;
        bool pin = false;        // lends its start to the live state: it stays as it is meanwhile
        size_t bytes() const {   // (its RAM)
            size_t n = cells.size();
            for (const Snap& x : snaps) n += x.bytes.size() + x.logits.size() * 4;
            return n;
        }
    };
    std::unique_ptr<lamina::core::DiskTier> disk;   // (before the files: they go first)
    std::vector<Parked> park;
    uint64_t park_clock = 0;
    size_t park_budget = (size_t) 16384 << 20, ram_reserve = (size_t) 6144 << 20;
    static constexpr int64_t kParkMin = 1024;   // shorter conversations are not parked
    File live_file;                             // the live state's, back from the disk tier: its
                                                // cells up to its `cells`, its snapshots' records
    size_t parked_bytes() const {
        size_t n = 0;
        for (const Parked& c : park) n += c.bytes();
        return n;
    }
    struct SnapAt {   // a snapshot's place, and whether it has its last token's logits
        int64_t pos;
        bool logits;
    };
    static std::vector<SnapAt> snaps_at(const std::vector<Snap>& v) {
        std::vector<SnapAt> r;
        for (const Snap& x : v) r.push_back({x.pos, !x.logits.empty()});
        return r;
    }
    static std::vector<SnapAt> snaps_at(const Parked& c) {
        if (!c.file) return snaps_at(c.snaps);
        std::vector<SnapAt> r;
        for (const SnapRec& x : c.recs) r.push_back({x.pos, x.n_logits > 0});
        return r;
    }
    /// The start of the prompt `ids` that a conversation of tokens `s` and snapshots `at` continues from (rewind()'s
    /// rule): the start they share - less its last token where that is all of the prompt and no snapshot there has its
    /// logits - where `s` ends within kMaxT tokens of it, else the last snapshot at or before it
    static int64_t reusable(const std::vector<int32_t>& s, const std::vector<SnapAt>& at,
                            const std::vector<int32_t>& ids) {
        const int64_t L = (int64_t) s.size(), n = (int64_t) ids.size();
        int64_t c = 0;
        while (c < L && c < n && s[(size_t) c] == ids[(size_t) c]) ++c;
        if (c == n && std::none_of(at.begin(), at.end(), [&](const SnapAt& x) { return x.pos == n && x.logits; })) --c;
        if (L - c <= kMaxT) return std::max<int64_t>(c, 0);
        int64_t r = 0;
        for (const SnapAt& x : at)
            if (x.pos <= c) r = std::max(r, x.pos);
        return r;
    }
    /// --conversation-spill-dir / --conversation-spill-mib (-1: the default budget, 0: none)
    void open_disk(const std::string& dir, int64_t mib) {
        if (dir.empty() || mib == 0) return;
        auto d = std::make_unique<lamina::core::DiskTier>();
        std::string msg;
        if (d->open(dir, mib > 0 ? (uint64_t) mib << 20 : 0, snap_bytes() + (size_t) m->g.n_vocab * 4, msg)) {
            disk = std::move(d);
            std::fprintf(stderr, "lamina: parked conversations the RAM does not hold go to %s\n", msg.c_str());
        } else {
            std::fprintf(stderr, "lamina: no disk tier for parked conversations: %s\n", msg.c_str());
        }
    }
    File make_file() { return File(disk->make().release(), FileGone{disk.get()}); }
    /// The sparse layers' cells [lo, hi) and their pooled keys between the device and `host`, laid out as a copy of
    /// cells [lo, held) (the park's for lo 0: a layer's K/V rows, then its pooled keys); lo a whole number of pools
    bool cells_range(int64_t lo, int64_t hi, int64_t held, uint8_t* host, bool to_host) {
        const int64_t per = m->g.idx_pool, g0 = lo / per, gn = hi / per - g0, gheld = held / per - g0;
        const size_t pk = (size_t) m->g.idx_dim * 4;
        const cudaMemcpyKind kind = to_host ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
        bool ok = true;
        for (const Layer& l : cl) {
            if (l.swa) continue;
            const size_t kr = l.kv_row(), kb = (size_t) (hi - lo) * kr;
            uint8_t* const dk = l.kv + (size_t) lo * kr;
            uint8_t* const dp = reinterpret_cast<uint8_t*>(l.pooled) + (size_t) g0 * pk;
            if (kb > 0) ok = ok && soft(cudaMemcpyAsync(to_host ? host : dk, to_host ? dk : host, kb, kind, st));
            host += (size_t) (held - lo) * kr;
            if (gn > 0)
                ok = ok && soft(cudaMemcpyAsync(to_host ? host : dp, to_host ? dp : host, (size_t) gn * pk, kind, st));
            host += (size_t) gheld * pk;
        }
        return soft(cudaStreamSynchronize(st)) && ok;
    }
    uint64_t cells_bytes(int64_t lo, int64_t held) const {
        const int64_t per = m->g.idx_pool;
        uint64_t n = 0;
        for (const Layer& l : cl)
            if (!l.swa)
                n += (uint64_t) (held - lo) * l.kv_row() + (uint64_t) (held / per - lo / per) * m->g.idx_dim * 4;
        return n;
    }
    /// Cells [lo, hi) of a park's copy of `held` cells into `out`, laid out as cells_range lays them out
    void cells_pack(int64_t lo, int64_t hi, const uint8_t* copy, int64_t held, uint8_t* out) const {
        const int64_t per = m->g.idx_pool, gn = hi / per - lo / per;
        const size_t pk = (size_t) m->g.idx_dim * 4;
        for (const Layer& l : cl) {
            if (l.swa) continue;
            const size_t kr = l.kv_row();
            std::memcpy(out, copy + (size_t) lo * kr, (size_t) (hi - lo) * kr);
            out += (size_t) (hi - lo) * kr;
            copy += (size_t) held * kr;
            std::memcpy(out, copy + (size_t) (lo / per) * pk, (size_t) gn * pk);
            out += (size_t) gn * pk;
            copy += (size_t) (held / per) * pk;
        }
    }
    lamina::core::DiskTier::Bytes disk_bytes_of() const {
        return [this](int64_t lo, int64_t held) { return cells_bytes(lo, held); };
    }
    lamina::core::DiskTier::Copy disk_copy() {
        return [this](int64_t lo, int64_t hi, int64_t held, uint8_t* host, bool out) {
            return cells_range(lo, hi, held, host, out);
        };
    }
    uint64_t files_bytes() const {   // the parked ones'
        uint64_t n = 0;
        for (const Parked& c : park)
            if (c.file) n += disk->bytes(*c.file);
        return n;
    }
    /// Room on the disk tier for `need` bytes more beside its files (and `also`, a file in hand that a park appends
    /// to): the live state's file gives its cells back first (its next park writes them whole), then the least
    /// recently used parked files leave
    bool disk_room(uint64_t need, uint64_t also) {
        uint64_t phys = disk->room(), live = live_file ? disk->bytes(*live_file) : 0;
        auto short_of = [&] { return files_bytes() + live + also + need > disk->budget() || need > phys; };
        if (short_of() && live_file && live_file->kv_size > 0) {
            const uint64_t b = live_file->kv_size;
            disk->drop_cells(*live_file);
            live -= b;
            if (phys != UINT64_MAX) phys += b;
        }
        while (short_of()) {
            auto lru = park.end();
            for (auto it = park.begin(); it != park.end(); ++it)
                if (it->file && !it->pin && (lru == park.end() || it->used < lru->used)) lru = it;
            if (lru == park.end()) return false;
            const uint64_t b = disk->bytes(*lru->file);
            std::fprintf(stderr, "lamina: a parked conversation of %zu tokens leaves the disk (%.0f MiB)\n",
                         lru->seq.size(), (double) b / (double) (1 << 20));
            park.erase(lru);
            if (phys != UINT64_MAX) phys += b;
        }
        return true;
    }
    /// A parked conversation in RAM to the disk tier (its RAM freed); false: no room, or a write failed
    bool demote(Parked& c) {
        const auto t0 = std::chrono::steady_clock::now();
        const int64_t p = (int64_t) c.seq.size(), per = m->g.idx_pool;
        const uint64_t need = disk->size_of(0, p, per, disk_bytes_of()) + c.snaps.size() * disk->slot_bytes();
        if (!disk_room(need, 0)) return false;
        File f = make_file();
        bool ok = disk->put_copy(*f, p, per, disk_bytes_of(), [&](int64_t lo, int64_t hi, uint8_t* out) {
            cells_pack(lo, hi, c.cells.data(), p, out);
        });
        std::vector<SnapRec> recs;
        for (const Snap& x : c.snaps) {
            ok = ok && disk->put_record(*f, x.pos,
                                        {{x.bytes.data(), x.bytes.size()}, {x.logits.data(), x.logits.size() * 4}});
            recs.push_back({x.pos, x.kept, x.logits.size()});
        }
        if (!ok) return false;
        c.cells = std::vector<uint8_t>();   // (its RAM freed: `= {}` keeps a vector's capacity)
        c.snaps = std::vector<Snap>();
        c.recs = std::move(recs);
        c.file = std::move(f);
        std::fprintf(stderr, "lamina: a parked conversation of %lld tokens moved to the disk (%.0f MiB in %.0f ms)\n",
                     (long long) p, (double) disk->bytes(*c.file) / (double) (1 << 20), ms_since(t0));
        return true;
    }
    /// The live conversation into the park on the disk tier (the RAM does not hold it, or it came back from there):
    /// its cells from where its file stops holding them, the snapshots the file lacks; the live state keeps copies
    /// of its snapshots up to `keep`
    void park_to_disk(int64_t keep) {
        const auto t0 = std::chrono::steady_clock::now();
        const int64_t p = pos(), per = m->g.idx_pool;
        Parked c;
        const uint64_t had = live_file ? disk->bytes(*live_file) : 0;
        c.file = live_file ? std::move(live_file) : make_file();
        const int64_t from = std::min<int64_t>(c.file->cells, p) / per * per;
        size_t fresh = 0;
        for (const Snap& x : snaps) fresh += c.file->slot(x.pos) == nullptr;
        const uint64_t cells = disk->size_of(from, p, per, disk_bytes_of());
        const uint64_t need =
            cells + (fresh > c.file->free_slots.size() ? fresh - c.file->free_slots.size() : 0) * disk->slot_bytes();
        bool ok = disk_room(need, had) && disk->put_cells(*c.file, from, p, per, disk_bytes_of(), disk_copy());
        for (size_t i = c.file->slots.size(); ok && i-- > 0;) {   // (records of snapshots it no longer has)
            const int64_t k = c.file->slots[i].first;
            if (std::none_of(snaps.begin(), snaps.end(), [&](const Snap& x) { return x.pos == k; })) c.file->forget(k);
        }
        for (const Snap& x : snaps) {
            if (ok && c.file->slot(x.pos) == nullptr)
                ok = disk->put_record(*c.file, x.pos,
                                      {{x.bytes.data(), x.bytes.size()}, {x.logits.data(), x.logits.size() * 4}});
            c.recs.push_back({x.pos, x.kept, x.logits.size()});
        }
        if (!ok) {
            std::fprintf(stderr,
                         "lamina: a conversation of %lld tokens is not parked: the disk tier has no room for it, "
                         "or its file could not be written\n",
                         (long long) p);
            return;
        }
        c.seq = seq;
        std::vector<Snap> kept;
        for (Snap& x : snaps)
            if (x.pos <= keep) kept.push_back(std::move(x));
        snaps = std::move(kept);
        c.used = ++park_clock;
        std::fprintf(stderr,
                     "lamina: a conversation of %lld tokens parked on the disk (%.0f MiB written in %.0f ms, its "
                     "file %.0f MiB, from token %lld)\n",
                     (long long) p, (double) (cells + fresh * disk->slot_bytes()) / (double) (1 << 20), ms_since(t0),
                     (double) disk->bytes(*c.file) / (double) (1 << 20), (long long) from);
        park.push_back(std::move(c));
    }
    /// Non-finite logits: the live state, its snapshots and every parked conversation (their files off the disk tier)
    /// forgotten, so nothing continues from them - the next request reads its prompt from the start
    void forget() {
        rewind(0, true);   // (no snapshot at 0: the sequence and its snapshots go, and the live state's file)
        park.clear();
    }
    /// The live conversation into the park, with a snapshot at its end (the logits of its last token unknown: empty);
    /// the live state keeps copies of its snapshots up to `keep` (the start the next request shares with it)
    void park_live(int64_t keep) {
        const auto& g = m->g;
        const int64_t p = pos(), groups = p / g.idx_pool;
        if (p < kParkMin) {
            live_file.reset();
            return;
        }
        snapshot({}, Kept::end);
        if (disk && live_file) {   // back from the disk tier: its file holds most of it, the rest goes there
            park_to_disk(keep);
            return;
        }
        Parked c;
        size_t n = 0;
        for (const Layer& l : cl)
            if (!l.swa) n += (size_t) p * l.kv_row() + (size_t) groups * (size_t) g.idx_dim * 4;
        const size_t need = n + [&] {
            size_t k = 0;
            for (const Snap& x : snaps) k += x.bytes.size() + x.logits.size() * 4;
            return k;
        }();
        // the budget counts the snapshots too; the RAM, the cells alone (the snapshots are in RAM already)
        uint64_t avail = 0;
        const bool known = lamina::platform::available_ram(avail);
        auto short_of = [&] { return parked_bytes() + need > park_budget || (known && avail < n + ram_reserve); };
        // the least recently used in RAM leave it, to the disk tier where there is one - only where this one then fits
        // (with every one in RAM that lends nothing gone)
        size_t movable = 0;
        for (const Parked& x : park)
            if (!x.file && !x.pin) movable += x.bytes();
        const bool fits =
            parked_bytes() - movable + need <= park_budget && (!known || avail + movable >= n + ram_reserve);
        while (fits && short_of()) {
            auto lru = park.end();
            for (auto it = park.begin(); it != park.end(); ++it)
                if (!it->file && !it->pin && (lru == park.end() || it->used < lru->used)) lru = it;
            if (lru == park.end()) break;
            Parked out = std::move(*lru);   // (out of the park first: the disk tier's room may take others)
            park.erase(lru);
            if (disk && demote(out)) {
                park.push_back(std::move(out));   // (with its use as it was)
            } else {
                std::fprintf(stderr, "lamina: a parked conversation of %zu tokens leaves the park (%.0f MiB)\n",
                             out.seq.size(), (double) out.bytes() / (double) (1 << 20));
            }
            if (known) lamina::platform::available_ram(avail);
        }
        if (short_of()) {
            if (disk) {   // the RAM does not hold it even so: the disk tier does
                park_to_disk(keep);
                return;
            }
            std::fprintf(stderr,
                         "lamina: a conversation of %lld tokens is not parked: %.0f MiB, over the park's budget "
                         "or the RAM reserve\n",
                         (long long) p, (double) need / (double) (1 << 20));
            return;
        }
        c.cells.resize(n);
        size_t at = 0;
        for (const Layer& l : cl) {
            if (l.swa) continue;
            const size_t kb = (size_t) p * l.kv_row(), pb = (size_t) groups * (size_t) g.idx_dim * 4;
            CK(cudaMemcpyAsync(c.cells.data() + at, l.kv, kb, cudaMemcpyDeviceToHost, st));
            at += kb;
            if (pb) CK(cudaMemcpyAsync(c.cells.data() + at, l.pooled, pb, cudaMemcpyDeviceToHost, st));
            at += pb;
        }
        CK(cudaStreamSynchronize(st));
        c.seq = seq;
        c.snaps = std::move(snaps);
        snaps.clear();
        for (const Snap& x : c.snaps)
            if (x.pos <= keep) snaps.push_back(x);
        c.used = ++park_clock;
        std::fprintf(stderr, "lamina: a conversation of %lld tokens parked (%.0f MiB; the park %zu, %.0f MiB)\n",
                     (long long) p, (double) c.bytes() / (double) (1 << 20), park.size() + 1,
                     (double) (parked_bytes() + c.bytes()) / (double) (1 << 20));
        park.push_back(std::move(c));
    }
    /// A parked conversation (taken out of the park) as the live state: its cells and snapshots back, the state at its
    /// last snapshot (its end); one from the disk tier keeps its file
    void unpark(Parked c) {
        const auto& g = m->g;
        const auto t0 = std::chrono::steady_clock::now();
        const int64_t p = (int64_t) c.seq.size(), groups = p / g.idx_pool;
        ensure(p);
        if (c.file) {
            bool ok = disk->get_cells(*c.file, p, disk_copy());
            snaps.clear();
            for (const SnapRec& r : c.recs) {
                Snap s;
                s.pos = r.pos;
                s.kept = r.kept;
                s.bytes.resize(snap_bytes());
                s.logits.resize(r.n_logits);
                ok = ok && disk->get_record(*c.file, r.pos,
                                            {{s.bytes.data(), s.bytes.size()}, {s.logits.data(), s.logits.size() * 4}});
                snaps.push_back(std::move(s));
            }
            if (!ok) {   // (the state starts over)
                std::fprintf(stderr,
                             "lamina: a parked conversation of %lld tokens could not be read back from the disk\n",
                             (long long) p);
                snaps.clear();
                seq.clear();
                sfx_ok = 0;
                return;
            }
            live_file = std::move(c.file);
            seq = std::move(c.seq);
            sfx_ok = 0;
            rewind(pos(), true);   // the rings and tails from its end's snapshot
            // its file within what the budget leaves beside the parked ones: its cells go first
            if (files_bytes() + disk->bytes(*live_file) > disk->budget()) disk->drop_cells(*live_file);
            std::fprintf(stderr, "lamina: a parked conversation of %lld tokens resumed from the disk (%.0f ms)\n",
                         (long long) p, ms_since(t0));
            return;
        }
        size_t at = 0;
        for (const Layer& l : cl) {
            if (l.swa) continue;
            const size_t kb = (size_t) p * l.kv_row(), pb = (size_t) groups * (size_t) g.idx_dim * 4;
            CK(cudaMemcpyAsync(l.kv, c.cells.data() + at, kb, cudaMemcpyHostToDevice, st));
            at += kb;
            if (pb) CK(cudaMemcpyAsync(l.pooled, c.cells.data() + at, pb, cudaMemcpyHostToDevice, st));
            at += pb;
        }
        CK(cudaStreamSynchronize(st));
        seq = std::move(c.seq);
        sfx_ok = 0;
        snaps = std::move(c.snaps);
        rewind(pos(), true);   // the rings and tails from its end's snapshot
        std::fprintf(stderr, "lamina: a parked conversation of %lld tokens resumed\n", (long long) p);
    }
    /// The first `r` tokens of the parked conversation `c` (at one of its snapshots) as the live state, for another
    /// conversation that starts as it does: its cells up to there, copies of its snapshots up to there, the state at
    /// that snapshot; `c` stays parked as it is
    void borrow(Parked& c, int64_t r) {
        const auto t0 = std::chrono::steady_clock::now();
        const int64_t p = (int64_t) c.seq.size();
        ensure(r);
        std::vector<Snap> got;
        bool ok = true;
        if (c.file) {
            ok = disk->get_cells(*c.file, r, disk_copy());
            for (const SnapRec& x : c.recs) {
                if (!ok || x.pos > r) continue;
                Snap s;
                s.pos = x.pos;
                s.kept = x.kept;
                s.bytes.resize(snap_bytes());
                s.logits.resize(x.n_logits);
                ok = disk->get_record(*c.file, x.pos,
                                      {{s.bytes.data(), s.bytes.size()}, {s.logits.data(), s.logits.size() * 4}});
                got.push_back(std::move(s));
            }
        } else {
            ok = cells_range(0, r, p, c.cells.data(), false);   // (the park's copy: lo 0, `p` cells)
            for (const Snap& x : c.snaps)
                if (x.pos <= r) got.push_back(x);
        }
        c.used = ++park_clock;
        live_file.reset();
        sfx_ok = 0;
        if (!ok) {   // (the state starts over)
            std::fprintf(stderr, "lamina: the start of a parked conversation of %lld tokens could not be read\n",
                         (long long) p);
            snaps.clear();
            seq.clear();
            return;
        }
        seq.assign(c.seq.begin(), c.seq.begin() + r);
        snaps = std::move(got);
        rewind(r, true);   // the rings and tails from its snapshot there
        std::fprintf(stderr, "lamina: the first %lld tokens of a parked conversation of %lld lent%s (%.0f ms)\n",
                     (long long) r, (long long) p, c.file ? " from the disk" : "", ms_since(t0));
    }

    void quant(const float* x, int64_t rows, int64_t n, uint8_t* out, cudaStream_t s_ = nullptr) {
        K::quantize_q8_1_rows(x, rows, n, out, s_ ? s_ : st);
    }
    // the decode's branches: sb1 and sb2 forked from st at a layer's points and joined (fork(): the branch streams wait
    // for st; join(): st waits for them), events made at the start and used again by every layer and capture
    bool branches = [] {
        const char* v = std::getenv("LAMINA_HY4_BRANCHES");
        return v == nullptr || std::atoi(v) != 0;
    }();
    cudaStream_t sb1 = nullptr, sb2 = nullptr;
    // the decode's chain with fewer kernels (quantizations in the kernel before them, the MoE's end and the post norm
    // with iHC's post each one kernel): the same bytes (LAMINA_HY4_FUSE=0: the kernels apart)
    bool fuse = [] {
        const char* v = std::getenv("LAMINA_HY4_FUSE");
        return v == nullptr || std::atoi(v) != 0;
    }();
    // the pre-mix's second pass quantizing xn too: its one block a row made it slower (an A/B, off)
    bool fuse_pre = std::getenv("LAMINA_HY4_FUSE_PRE") != nullptr;
    cudaEvent_t ev_fork = nullptr, ev_j1 = nullptr, ev_j2 = nullptr, ev_q = nullptr;
    bool br_on() const { return branches && sb1 != nullptr && trace < 3; }
    void fork(cudaStream_t a, cudaStream_t b = nullptr) {
        CK(cudaEventRecord(ev_fork, st));
        CK(cudaStreamWaitEvent(a, ev_fork, 0));
        if (b) CK(cudaStreamWaitEvent(b, ev_fork, 0));
    }
    void join(cudaStream_t a, cudaStream_t b = nullptr) {
        CK(cudaEventRecord(ev_j1, a));
        CK(cudaStreamWaitEvent(st, ev_j1, 0));
        if (b) {
            CK(cudaEventRecord(ev_j2, b));
            CK(cudaStreamWaitEvent(st, ev_j2, 0));
        }
    }
    void mmvq(const Mat& w, const uint8_t* xq_, float* y_, int T, cudaStream_t s_) {
        K::native_mmvq(w.type, w.d, xq_, y_, w.n_in, w.n_out, T, s_);
    }
    void mmvq(const Mat& w, const uint8_t* xq_, float* y_, int T) {
        K::native_mmvq(w.type, w.d, xq_, y_, w.n_in, w.n_out, T, st);
    }
    unsigned blocks(int64_t n) const { return (unsigned) ((n + hy4k::kThreads - 1) / hy4k::kThreads); }

    // ---- a forward.  Its GPU work runs in segments between the CPU's parts: segment 0 is the embedding and the layers
    // up to the first MoE layer's routing; segment s > 0 the s-th MoE layer's end (the CPU's rows merged, the weighted
    // sum, routed_up, the FFN's norm and iHC post) and the layers after it up to the next MoE layer's routing, the last
    // one the head.  Each (segment, rows) pair is captured once as a CUDA graph: what changes from one forward to the
    // next is in device memory (the tokens, the positions, the Engram rows, the slots' table) or in mapped host memory
    // (the CPU's rows), and every decision a row's position makes (its cells, its group's pooling, the indexer's pick)
    // is the GPU's.
    std::vector<int> moe_layers;
    std::map<int, cudaGraphExec_t> graphs;
    bool use_graphs = true;
    // pictures (--vision): the request's (feed's) rows; a forward's image cells staged for its first segment
    bool vision = false;
    const ImgReq* img = nullptr;
    float *img_rows_d = nullptr, *h_img = nullptr;
    int32_t *img_slot_d = nullptr, *h_slot = nullptr;
    // the cells' M-RoPE positions (--vision: every rope reads them; a CUDA graph keeps the table's address, its
    // contents change): a request's with pictures, else each cell's index (mrope_identity)
    hy4k::MRope mr;
    int32_t* mrope_d = nullptr;
    std::vector<int32_t> mrope_h;
    bool mrope_identity = false;

    /// The table for a request (`ir`: its pictures' cells and positions, the cells after its prompt counting on from
    /// ir->next; null: the identity, written only where a request with pictures left another).
    void mrope_set(const ImgReq* ir) {
        if (mrope_d == nullptr || (ir == nullptr && mrope_identity)) return;
        mrope_h.resize((size_t) ctx * 3);
        const int64_t n = ir != nullptr ? (int64_t) ir->mrope.size() / 3 : 0;
        if (ir != nullptr) std::copy(ir->mrope.begin(), ir->mrope.end(), mrope_h.begin());
        for (int64_t c = n; c < ctx; ++c) {
            const int32_t p = (int32_t) (ir != nullptr ? ir->next + (c - n) : c);
            mrope_h[(size_t) c * 3] = mrope_h[(size_t) c * 3 + 1] = mrope_h[(size_t) c * 3 + 2] = p;
        }
        CK(cudaMemcpyAsync(mrope_d, mrope_h.data(), mrope_h.size() * 4, cudaMemcpyHostToDevice, st));
        CK(cudaStreamSynchronize(st));
        mrope_identity = ir == nullptr;
    }

    /// T rows (tokens toks, at positions pos() ..) through the model; logits of every row into `logits`.
    void forward(const int32_t* toks, int T) {
        const auto& g = m->g;
        ensure(pos() + T);
        const int64_t p0 = pos();
        snap_base = p0;
        snap_rows = T;
        for (int t = 0; t < T; ++t) seq.push_back(toks[t]);
        // Engram's rows: hashed now, read from the SSD while this forward's inputs go to the GPU
        const auto tt = Clock::now();
        const size_t n_mod = g.eng_layers.size(), eh = (size_t) g.eng_heads(), ed = (size_t) g.eng_dim;
        for (int t = 0; t < T; ++t) {
            std::vector<uint32_t> r(n_mod * eh);
            engram_ids(g, seq, p0 + t, r.data(), h_valid[t]);
            for (size_t e = 0; e < n_mod; ++e)
                std::memcpy(h_rows + (e * (size_t) T + (size_t) t) * eh, r.data() + e * eh, eh * 4);
            h_pos[t] = (int32_t) (p0 + t);
            h_tok[t] = toks[t];
        }
        const lamina::ngram::PleReader::Ticket ticket = reader.issue(h_rows, n_mod * (size_t) T * eh, h_raw);
        CK(cudaMemcpyAsync(tok_d, h_tok, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(pos_d, h_pos, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        if (vision) {   // the forward's image cells: their rows staged, the others' slots -1
            const int N = (int) g.n_embd;
            int any = 0;
            for (int t = 0; t < T; ++t) {
                const int32_t r = img ? img->row(p0 + t) : -1;
                h_slot[t] = r >= 0 ? t : -1;
                if (r >= 0) {
                    std::memcpy(h_img + (size_t) t * N, img->rows.data() + (size_t) r * N, (size_t) N * 4);
                    any = t + 1;
                }
            }
            CK(cudaMemcpyAsync(img_slot_d, h_slot, (size_t) T * 4, cudaMemcpyHostToDevice, st));
            if (any) CK(cudaMemcpyAsync(img_rows_d, h_img, (size_t) any * N * 4, cudaMemcpyHostToDevice, st));
        }
        std::string err;
        if (!reader.collect(ticket, err)) {
            std::fprintf(stderr, "lamina: the Engram table: %s\n", err.c_str());
            throw Fatal{1};
        }
        table_ms += ms_since(tt);
        if (trace) tr.by[T].table += ms_since(tt);
        CK(cudaMemcpyAsync(eraw, h_raw, n_mod * (size_t) T * eh * ed, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(evalid, h_valid, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        const int n_seg = (int) moe_layers.size() + 1;
        if (trace >= 3) {
            CK(cudaStreamSynchronize(st));
            ph_rows = T;
            ph_t = Clock::now();
        }
        const bool doors = tier_doorbell && use_graphs && trace < 3 && gpus.size() > 1 && n_seg <= 128;
        if (doors) door_launch(n_seg, T);
        for (int s = 0; s < n_seg; ++s) {
            if (s > 0) {
                door_seg = doors ? s : -1;
                cpu_part(m->layers[(size_t) moe_layers[(size_t) s - 1]], T);
                door_seg = -1;
                ph_t = Clock::now();   // the host's part is not the GPU's time
            }
            const auto tl = Clock::now();
            run_segment(s, T);
            if (trace) tr.by[T].launch += ms_since(tl);
        }
        ++forwards;
        ph(13);
        ph_rows = 0;
        if (trace) {
            ++tr.by[T].n;
            tr.by[T].ms += ms_since(tt);
        }
    }

    void run_segment(int s, int T) {
        if (!use_graphs) {
            enqueue_segment(s, T);
            return;
        }
        const int key = s * 16 + T;
        auto it = graphs.find(key);
        if (it == graphs.end()) {
            CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
            enqueue_segment(s, T);
            cudaGraphExec_t ex = graph_end(st);
            it = graphs.emplace(key, ex).first;
        }
        CK(cudaGraphLaunch(it->second, st));
    }

    void enqueue_segment(int s, int T) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, hc = (int) g.hc;
        size_t li = 0;
        if (s == 0) {
            K::iq_embed_rows(m->embd_type, m->embd, m->embd_row, g.n_vocab, tok_d, T, N, emb, st);
            if (vision) hy4k::k_image_rows<<<T, 256, 0, st>>>(emb, img_rows_d, img_slot_d, N);
            hy4k::k_streams_init<<<T, hy4k::kThreads, 0, st>>>(emb, X, hc, N);
            hy4k::k_swa_cells<<<T, 256, 0, st>>>(pos_d, (int) g.n_swa, (int) ring, cells_swa, n_swa);
            ph(0);
        } else {
            const int lm = moe_layers[(size_t) s - 1];
            moe_end(m->layers[(size_t) lm], T);
            ph(11);
            li = (size_t) lm + 1;
        }
        for (; li < m->layers.size(); ++li) {
            Layer& l = m->layers[li];
            front(l, T);
            if (!l.dense) {
                moe_front(l, T);
                return;
            }
            dense_ffn(l, T);
            ph(12);
        }
        // the head: the streams reduced, output_norm, the BF16 head
        hc_pre(m->ohc_fn, m->ohc_scale, m->ohc_base, hc, 0.f, nullptr, m->out_norm, T);
        K::bf16_gemv_fp32_mmvf_multi(xn, m->head, logits, N, g.n_vocab, T, st);
        if (hring) hy4k::k_ring_store<<<T, 256, 0, st>>>(xn, hring, pos_d, kHRing, N);
    }

    /// Engram, the attention, and the FFN's input (xn, its q8_1 rows in xq; iHC's post weights in `post`)
    void front(Layer& l, int T) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, hc = (int) g.hc;
        if (l.eng >= 0) {
            const int eh = (int) g.eng_heads(), ed = (int) g.eng_dim;
            hy4k::k_engram_rows<<<dim3(eh, T), 128, 0, st>>>(eraw + (size_t) l.eng * T * eh * ed, l.e_lnorm, evalid,
                                                             erows, eh, ed, kEngramEps);
            quant(erows, T, (int64_t) eh * ed, erq);
            mmvq(l.e_kv, erq, ekv, T);
            hy4k::k_engram_mix<<<T, 256, 0, st>>>(X, ekv, l.e_kv_b, l.e_gate, l.e_gamma, l.e_hcs, evalid, hc, N,
                                                  kEngramEps);
            ph(1);
        }
        const bool xq_made = hc_pre(l.hca_fn, l.hca_scale, l.hca_base, 2 * hc, g.hc_mag, post, l.attn_norm, T, true);
        ph(2);
        const bool post_fused = fuse && trace < 3;
        attend(l, T, xq_made, post_fused);
        if (post_fused)   // the post-attention norm and iHC's post in one
            hy4k::k_rms_post<<<dim3((unsigned) ((N + 255) / 256), T), 256, 0, st>>>(y, nullptr, l.post_attn, X, post,
                                                                                    hc, N, g.eps);
        else hc_post(T);
        if (!hc_pre(l.hcf_fn, l.hcf_scale, l.hcf_base, 2 * hc, g.hc_mag, post, l.ffn_norm, T, true))
            quant(xn, T, N, xq);
        ph(2);
    }

    /// A layer's attention over T rows: its input normed in xn, its output (wo, the post-attention norm) in y
    /// `xq_made`: xn's q8_1 rows are in xq already; `post_fused`: y left before the post-attention norm (the caller's
    /// k_rms_post does it with iHC's post)
    void attend(Layer& l, int T, bool xq_made = false, bool post_fused = false) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, H = l.H;
        const bool fz = fuse && trace < 3;
        if (!xq_made) quant(xn, T, N, xq);
        // branches: sb1 kv_a and the cell; sb2 the gate, then (a sparse layer) the indexer once qrq is made; st q_a ..
        // k_b; joined before the attention
        const bool br = br_on();
        cudaStream_t s1 = br ? sb1 : st, s2 = br ? sb2 : st;
        if (br) fork(sb1, sb2);
        mmvq(l.kv_a, xq, kva, T, s1);
        (l.kv_q8 ? hy4k::k_kv_store<true> : hy4k::k_kv_store<false>) <<<T, hy4k::kThreads, 0, s1>>>(
            kva, l.kv_a_norm, l.kv, pos_d, l.swa ? (int) ring : 0, 0, l.theta, g.eps, mr);
        mmvq(l.gate, xq, gt, T, s2);
        mmvq(l.q_a, xq, qa, T);
        if (fz && g.q_lora % 256 == 0) {
            hy4k::k_rms_q<<<T, 256, 0, st>>>(qa, (int) g.q_lora, l.q_a_norm, reinterpret_cast<hy4k::Q81*>(qrq),
                                             (int) g.q_lora, g.eps);
        } else {
            hy4k::k_rms<<<T, hy4k::kThreads, 0, st>>>(qa, (int) g.q_lora, l.q_a_norm, qa, (int) g.q_lora,
                                                      (int) g.q_lora, g.eps);
            quant(qa, T, g.q_lora, qrq);
        }
        const int32_t* cells = cells_swa;
        const int32_t* counts = n_swa;
        int ld = (int) g.n_swa;
        if (l.ext_cells) {   // a DFly layer
            cells = l.ext_cells;
            counts = l.ext_n;
            ld = l.ext_ld;
        } else if (!l.swa) {
            if (br) {   // the indexer reads qrq
                CK(cudaEventRecord(ev_q, st));
                CK(cudaStreamWaitEvent(s2, ev_q, 0));
            }
            indexer(l, T, s2);
            cells = cells_sel;
            counts = n_sel;
            ld = (int) g.idx_top_k;
            if (const int sl = m->dfly.on ? dfly_slot(l.il) : -1; sl >= 0) {   // a layer DFly reads: its picks kept
                CK(cudaMemcpyAsync(snap_cells + (size_t) sl * kMaxT * g.idx_top_k, cells_sel,
                                   (size_t) T * g.idx_top_k * 4, cudaMemcpyDeviceToDevice, s2));
                CK(cudaMemcpyAsync(snap_n + (size_t) sl * kMaxT, n_sel, (size_t) T * 4, cudaMemcpyDeviceToDevice, s2));
            }
            ph(5);
        }
        mmvq(l.q_b, qrq, q, T);
        ph(3);
        if (fz && g.nope() % 32 == 0 && g.n_rot <= 64) {
            hy4k::k_qsplit_rope_q<<<dim3(H, T), 256, 0, st>>>(q, reinterpret_cast<hy4k::Q81*>(qnq), qabs, H,
                                                              (int) g.k_head, (int) g.nope(), (int) g.n_rot, pos_d,
                                                              l.theta, mr);
        } else {
            hy4k::k_q_split<<<dim3(H, T), 256, 0, st>>>(q, qnope, qabs, H, (int) g.k_head, (int) g.nope());
            hy4k::k_rope_norm<<<dim3(H, T), 32, 0, st>>>(qabs, H, hy4k::kKvRow, hy4k::kLatent, (int) g.n_rot, pos_d, 0,
                                                         l.theta, mr);
            quant(qnope, (int64_t) T * H, g.nope(), qnq);
        }
        hy4k::k_head_q8_0_w2<<<(unsigned) ((H * hy4k::kLatent + 15) / 16), 256, 0, st>>>(
            l.k_b.d, qnq, qabs, H, (int) g.nope(), hy4k::kLatent, H * hy4k::kKvRow, hy4k::kKvRow, T);
        if (br) join(sb1, sb2);
        ph(4);
        if (H % hy4p::kQRows == 0) {   // on the tensor cores, the cells in splits of `per`
            const int per = std::max(64, (ld / 16 + 31) / 32 * 32), n_split = (ld + per - 1) / per;
            (l.kv_q8
                 ? hy4p::k_attn_dec<true>
                 : hy4p::k_attn_dec<false>) <<<dim3(n_split, H / hy4p::kQRows, T), 256, sizeof(hy4p::AttnSmem), st>>>(
                qabs, l.kv, cells, counts, ld, H, per, 1.0f / sqrtf((float) g.k_head), part, n_split);
            if (fz)
                hy4k::k_mla_merge_q<<<dim3(H, T), 128, 0, st>>>(part, l.sinks, reinterpret_cast<hy4k::Q81*>(attq), H,
                                                                n_split);
            else hy4k::k_mla_merge<<<dim3(H, T), 128, 0, st>>>(part, l.sinks, att, H, n_split);
        } else {
            const int n_chunks = (ld + hy4k::kAttnChunk - 1) / hy4k::kAttnChunk;
            const size_t smem = (size_t) hy4k::kAttnChunk * hy4k::kKvRowPad * sizeof(__half);
            (l.kv_q8 ? hy4k::k_mla_chunk<true> : hy4k::k_mla_chunk<false>) <<<dim3(n_chunks, T), 256, smem, st>>>(
                qabs, l.kv, cells, counts, ld, H, 1.0f / sqrtf((float) g.k_head), part, n_chunks);
            hy4k::k_mla_merge<<<dim3(H, T), 128, 0, st>>>(part, l.sinks, att, H, n_chunks);
        }
        ph(6);
        if (!fz || H % hy4p::kQRows != 0) quant(att, (int64_t) T * H, hy4k::kLatent, attq);
        hy4k::k_head_q8_0_w2<<<(unsigned) ((H * g.v_head + 15) / 16), 256, 0, st>>>(
            l.v_b.d, attq, o, H, hy4k::kLatent, (int) g.v_head, H * (int) g.v_head, (int) g.v_head, T);
        if (fz && (H * g.v_head) % 32 == 0) {
            hy4k::k_gate_mul_q<<<blocks((int64_t) T * H * g.v_head), hy4k::kThreads, 0, st>>>(
                o, gt, reinterpret_cast<hy4k::Q81*>(oq), (int64_t) T * H * g.v_head);
        } else {
            hy4k::k_gate_mul<<<blocks((int64_t) T * H * g.v_head), hy4k::kThreads, 0, st>>>(o, gt,
                                                                                            (int64_t) T * H * g.v_head);
            quant(o, T, (int64_t) H * g.v_head, oq);
        }
        mmvq(l.wo, oq, y, T);
        if (!post_fused) hy4k::k_rms_inplace<<<T, hy4k::kThreads, 0, st>>>(y, l.post_attn, N, g.eps);
        ph(7);
    }

    /// The sparse layer's indexer: the rows' index keys into the tail, the groups they complete pooled, and each row's
    /// cells: every cell while its visible groups (those before its own) are within the reach, else the top groups'.
    void indexer(Layer& l, int T, cudaStream_t is) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, ih = (int) g.idx_heads, idd = (int) g.idx_dim, pool_n = (int) g.idx_pool;
        mmvq(l.iq_b, qrq, iq, T, is);
        hy4k::k_rope_norm<<<dim3(ih, T), 32, 0, is>>>(iq, ih, idd, idd - (int) g.n_rot, (int) g.n_rot, pos_d, 0,
                                                      l.theta, mr);
        mmvq(l.ik, xq, ik, T, is);
        hy4k::k_rms<<<T, 128, 0, is>>>(ik, idd, l.ik_norm, ik, idd, idd, g.eps);
        quant(ik, T, idd, ikq, is);
        mmvq(l.icv, ikq, ivg, T, is);
        mmvq(l.icg, ikq, ivg + (size_t) kMaxT * idd, T, is);
        hy4k::k_tail_store<<<T, 128, 0, is>>>(ivg, ivg + (size_t) kMaxT * idd, l.tail, pos_d, kTail, idd);
        hy4k::k_pool_rows<<<T, 128, 0, is>>>(l.tail, kTail, idd, pool_n, l.ape, l.ipool_norm, g.eps, l.theta,
                                             (int) g.n_rot, pos_d, l.pooled, mr);
        const int reach = (int) (g.idx_top_k / pool_n - (g.idx_tail ? 1 : 0));
        hy4k::k_gemv_f32<<<dim3((unsigned) ((ih + 7) / 8), T), 256, 0, is>>>(l.iproj, xn, iw, N, ih,
                                                                             1.0f / sqrtf((float) (idd * ih)));
        const size_t smem = ((size_t) ih * idd + ih) * sizeof(float);
        hy4k::k_index_scores_pos<<<dim3(kScoreBlocks, T), 256, smem, is>>>(iq, iw, l.pooled, pos_d, pool_n, reach, ih,
                                                                           idd, scores, (int) (ctx / pool_n));
        hy4k::k_select_cells<<<T, 1024, 0, is>>>(scores, (int) (ctx / pool_n), pos_d, pool_n, reach, cells_sel,
                                                 (int) g.idx_top_k, n_sel);
    }

    void dense_ffn(Layer& l, int T) {
        const auto& g = m->g;
        const int N = (int) g.n_embd;
        mmvq(l.f_gate, xq, fg, T);
        mmvq(l.f_up, xq, fu, T);
        hy4k::k_swiglu<<<blocks((int64_t) T * g.n_ff), hy4k::kThreads, 0, st>>>(fg, fu, fh, (int64_t) T * g.n_ff);
        quant(fh, T, g.n_ff, hq);
        mmvq(l.f_down, hq, y, T);
        hy4k::k_rms_inplace<<<T, hy4k::kThreads, 0, st>>>(y, l.post_ffw, N, g.eps);
        hc_post(T);
    }

    /// The MoE layer's GPU work before the CPU's: the routing, the routed input to the host (an event when it is
    /// there), the experts with a VRAM slot, the shared expert.
    void moe_front(Layer& l, int T) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, E = (int) g.n_expert, Kk = (int) g.top_k, Ld = (int) g.n_latent,
                  F = (int) g.n_ff_exp;
        const bool br = br_on();
        cudaStream_t s1 = br ? sb1 : st;
        if (br) fork(sb1);   // the router and its picks beside routed_down
        hy4k::k_gemv_f32<<<dim3((unsigned) ((E + 7) / 8), T), 256, 0, s1>>>(l.router, xn, logits_r, N, E, 1.f);
        if (E <= 1024 && Kk <= 32)
            hy4k::k_route_w<<<(unsigned) ((T + 7) / 8), 256, 0, s1>>>(logits_r, l.bias, E, Kk, g.w_norm, g.w_scale, T,
                                                                      ids, wts);
        else
            hy4k::k_route<<<T, std::max(32, ((E + 31) / 32) * 32), 0, s1>>>(logits_r, l.bias, E, Kk, g.w_norm,
                                                                            g.w_scale, ids, wts);
        mmvq(l.r_down, xq, rin, T);
        if (br) join(sb1);
        CK(cudaMemcpyAsync(h_rin, rin, (size_t) T * Ld * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaMemcpyAsync(h_ids, ids, (size_t) T * Kk * 4, cudaMemcpyDeviceToHost, st));
        // a graph's node the host waits on; launched (no graphs, the MTP block's event) a plain record
        if (ev_moe == ev && use_graphs) CK(cudaEventRecordWithFlags(ev, st, cudaEventRecordExternal));
        else CK(cudaEventRecord(ev_moe, st));
        ph(8);
        if (!l.held.empty() && l.held[0].n > 0) {
            quant(rin, T, Ld, rinq);
            group_slots(ids, T * Kk, Kk, E, l.held[0].slot_of_d, l.held[0].slots, l.stride, grp_ptr, grp_start,
                        n_groups, ent_dst, ent_tok, st);
            K::native_expert_grouped(l.gl, grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
                                     std::min<int64_t>((int64_t) T * Kk, E), (int64_t) T * Kk, rinq,
                                     (int64_t) (Ld / 32) * 36, escr, parts, st);
            ph(9);
        }
        mmvq(l.s_gate, xq, sg, T);
        mmvq(l.s_up, xq, su, T);
        hy4k::k_swiglu<<<blocks((int64_t) T * F), hy4k::kThreads, 0, st>>>(sg, su, sh, (int64_t) T * F);
        quant(sh, T, F, hq);
        mmvq(l.s_down, hq, sy, T);
        ph(10);
    }

    /// The host's part of an MoE layer, between two segments: each entry to its holder (the main GPU's slots, which the
    /// segment before ran; a tier's, whose share runs now on that GPU; else the CPU pool), the tiers' and the CPU's
    /// rows into mapped memory (h_out, with h_miss), which the next segment merges.
    void cpu_part(Layer& l, int T) {
        const auto& g = m->g;
        const int Kk = (int) g.top_k, Ld = (int) g.n_latent;
        const auto tw = Clock::now();
        CK(cudaEventSynchronize(ev_moe));
        const auto t0 = Clock::now();
        if (!l.mtp) dump_route(0, l.il, T, h_ids);
        std::fill(tier_used.begin(), tier_used.end(), (char) 0);
        int n_miss = 0;
        for (int e = 0; e < T * Kk; ++e) {
            const int ex = h_ids[e];
            l.use[(size_t) ex] += 1.f;
            l.total[(size_t) ex] += 1.f;
            int holder = -1;
            for (size_t gi = 0; gi < l.held.size() && holder < 0; ++gi)
                if (l.held[gi].n > 0 && l.held[gi].slot_of[(size_t) ex] >= 0) holder = (int) gi;
            h_miss[e] = holder != 0;
            if (holder >= 0) ++gpus[(size_t) holder].hits;
            if (holder > 0) tier_used[(size_t) holder] = 1;
            if (holder < 0) ++n_miss;
        }
        const auto t1 = Clock::now();
        for (size_t gi = 1; gi < tier_used.size(); ++gi)
            if (door_seg > 0) {   // every tier's share is queued: its doorbell (its inputs are in)
                std::atomic_thread_fence(std::memory_order_seq_cst);
                reinterpret_cast<volatile uint32_t*>(gpus[gi].door)[door_seg] = 1u;
            } else if (tier_used[gi]) {
                tier_launch((int) gi, l, T);
            }
        const auto t2 = Clock::now();
        jobs.clear();
        std::map<int, size_t> open;
        std::vector<char> quantized((size_t) T, 0);
        for (int e = 0; e < T * Kk; ++e) {
            const int ex = h_ids[e], t = e / Kk;
            bool held = false;
            for (const SlotSet& s : l.held) held = held || (s.n > 0 && s.slot_of[(size_t) ex] >= 0);
            if (held) continue;
            if (!quantized[(size_t) t]) {
                C::native_quant_act(l.fmt, h_rin + (size_t) t * Ld, acts + (size_t) t * C::kNativeActBytes);
                quantized[(size_t) t] = 1;
            }
            auto it = open.find(ex);
            if (it == open.end() || jobs[it->second].nt == C::MAXT) {
                C::ExpertJobMulti j;
                j.blob = l.blob[(size_t) ex];
                jobs.push_back(j);
                it = open.insert_or_assign(ex, jobs.size() - 1).first;
            }
            C::ExpertJobMulti& j = jobs[it->second];
            j.nact[j.nt] = acts + (size_t) t * C::kNativeActBytes;
            j.out[j.nt] = h_out + (size_t) e * Ld;
            ++j.nt;
        }
        lookups += T * Kk;
        hits += T * Kk - n_miss;
        if (n_miss > 0) pool->run_split_multi_native(l.fmt, jobs.data(), (int) jobs.size());
        const auto t3 = Clock::now();
        bool tiers = false;
        for (size_t gi = 1; gi < tier_used.size(); ++gi)
            if (tier_used[gi] || door_seg > 0) {   // a rung share reads the routing's host copies: done before the next
                lamina::core::DeviceScope ds(gpus[gi].dev, main_dev);
                CK(cudaEventSynchronize(door_seg > 0 ? gpus[gi].door_ev[(size_t) door_seg] : gpus[gi].ev));
                tiers = tiers || tier_used[gi];
            }
        cpu_ms += ms_since(t0);
        if (trace) {
            const auto t4 = Clock::now();
            Trace::Part& x = tr.by[T];
            x.route += ms_between(tw, t0);
            x.scan += ms_between(t0, t1);
            x.tier_launch += ms_between(t1, t2);
            x.pool += ms_between(t2, t3);
            x.tier_wait += ms_between(t3, t4);
            ++x.layers;
            x.pool_layers += n_miss > 0;
            x.tier_layers += tiers;
            x.cpu_experts += (int64_t) jobs.size();
            x.cpu_entries += n_miss;
            if (l.il < 64) {
                tr.l_ent[l.il] += T * Kk;
                tr.l_cpu[l.il] += n_miss;
            }
            if (trace > 1) {
                CK(cudaStreamSynchronize(st));
                x.main_tail += ms_since(t4);
            }
        }
    }

    /// The MoE layer's end: the CPU's rows merged (read through the mapping), the weighted sum, routed_up, the shared
    /// expert added, the FFN's norm, iHC's post.
    void moe_end(Layer& l, int T) {
        const bool post_fused = fuse && trace < 3;
        moe_out(l, T, post_fused);
        if (post_fused)   // y + the shared expert's, the FFN's norm and iHC's post in one
            hy4k::k_rms_post<<<dim3((unsigned) ((m->g.n_embd + 255) / 256), T), 256, 0, st>>>(
                y, sy, l.post_ffw, X, post, (int) m->g.hc, (int) m->g.n_embd, m->g.eps);
        else hc_post(T);
    }
    /// iHC's pre-mix of the decode's T rows from X into xh (and xn with `wn`), `m_mix` mixes (2 hc for a layer: post
    /// weights into `post_out`; hc for the head), over many blocks
    /// With `with_q`, xn's q8_1 rows into xq as well (true: made; else the caller quantizes)
    bool hc_pre(const float* fn, const float* scale, const float* base, int m_mix, float mag, float* post_out,
                const float* wn, int T, bool with_q = false) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, hc = (int) g.hc;
        const bool fq = with_q && fuse_pre && wn != nullptr && N % 256 == 0;
        hy4k::k_hc_pre1<<<dim3(hy4k::kHcSlices, T), 256, 0, st>>>(X, fn, hc * N, m_mix, hc_part);
        hy4k::k_hc_pre2<<<T, 256, 0, st>>>(X, hc_part, scale, base, hc, N, m_mix, g.eps, g.hc_eps, mag, post_out, xh,
                                           wn, xn, g.eps, fq ? (void*) xq : nullptr);
        return fq;
    }
    /// The entries the slots hold grouped by expert (k_group_slots_p, or the one-thread form for more experts)
    static void group_slots(const int32_t* ids_, int n_ent, int K_, int E_, const int32_t* slot_of,
                            const uint8_t* slots, size_t stride, unsigned long long* gptr, int32_t* gstart,
                            int32_t* ngroups, int32_t* edst, int32_t* etok, cudaStream_t s_) {
        if (E_ <= 512 && n_ent <= 128)
            hy4k::k_group_slots_p<<<1, 512, 0, s_>>>(ids_, n_ent, K_, E_, slot_of, slots, stride, gptr, gstart, ngroups,
                                                     edst, etok);
        else
            hy4k::k_group_slots<<<1, 512, 0, s_>>>(ids_, n_ent, K_, E_, slot_of, slots, stride, gptr, gstart, ngroups,
                                                   edst, etok);
    }
    /// iHC's post of the decode's T rows: X += y * post, a thread a column of every stream
    void hc_post(int T) {
        const int N = (int) m->g.n_embd;
        hy4k::k_hc_post_c<<<dim3((unsigned) ((N + 255) / 256), T), 256, 0, st>>>(X, y, post, (int) m->g.hc, N);
    }
    /// The MoE layer's output in y: the CPU's rows merged, the weighted sum, routed_up, the shared expert added, the
    /// norm `post_fused`: y left as routed_up's (the caller adds the shared expert's and norms it with iHC's post)
    void moe_out(Layer& l, int T, bool post_fused = false) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, Kk = (int) g.top_k, Ld = (int) g.n_latent;
        if (fuse && trace < 3 && Ld % 256 == 0) {
            hy4k::k_moe_end_q<<<dim3((unsigned) (Ld / 256), T), 256, 0, st>>>(
                parts, h_out_d, h_miss_d, wts, reinterpret_cast<hy4k::Q81*>(moeq), Kk, Ld);
        } else {
            hy4k::k_merge_cpu<<<T * Kk, hy4k::kThreads, 0, st>>>(parts, h_out_d, h_miss_d, Ld);
            hy4k::k_moe_sum_c<<<dim3((unsigned) ((Ld + 255) / 256), T), 256, 0, st>>>(parts, wts, moe, Kk, Ld);
            quant(moe, T, Ld, moeq);
        }
        mmvq(l.r_up, moeq, y, T);
        if (post_fused) return;
        hy4k::k_add<<<blocks((int64_t) T * N), hy4k::kThreads, 0, st>>>(y, sy, (int64_t) T * N);
        hy4k::k_rms_inplace<<<T, hy4k::kThreads, 0, st>>>(y, l.post_ffw, N, g.eps);
    }

    // ---- the MTP block (blk.<n_layers>): row t takes token toks[t] at position p0 + t and the state src[t] (the
    // ring's slot of the position before, -1 - k for row k of the block's own last output, kNoState for zeros); eh_proj
    // of [enorm(embedding) | hnorm(state)] is its residual, a sparse layer's attention and latent MoE on it (no iHC),
    // the shared head's norm, the trunk's head.  Its attention runs over every row (their K/V, pooled keys and tail);
    // the FFN and the head over the last row alone (`head`: its logits into mtp_lg, its output state into mtp_h row 0).
    void mtp_forward(const int32_t* toks, const int32_t* src, int T, int64_t p0, bool head) {
        const auto& g = m->g;
        const int N = (int) g.n_embd;
        Layer& l = m->mtp;
        ensure(p0 + T);
        for (int t = 0; t < T; ++t) {
            h_tok[t] = toks[t];
            h_pos[t] = (int32_t) (p0 + t);
            h_src[t] = src[t];
        }
        CK(cudaMemcpyAsync(tok_d, h_tok, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(pos_d, h_pos, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(mtp_src, h_src, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        hy4k::k_gather_states<<<T, 256, 0, st>>>(hring, mtp_h, mtp_src, mtp_hs, N);
        K::iq_embed_rows(m->embd_type, m->embd, m->embd_row, g.n_vocab, tok_d, T, N, emb, st);
        hy4k::k_rms<<<T, hy4k::kThreads, 0, st>>>(emb, N, m->mtp_enorm, mtp_cat, 2 * N, N, g.eps);
        hy4k::k_rms<<<T, hy4k::kThreads, 0, st>>>(mtp_hs, N, m->mtp_hnorm, mtp_cat + N, 2 * N, N, g.eps);
        quant(mtp_cat, T, 2 * N, mtp_catq);
        mmvq(m->eh_proj, mtp_catq, mtp_x, T);
        hy4k::k_rms<<<T, hy4k::kThreads, 0, st>>>(mtp_x, N, l.attn_norm, xn, N, N, g.eps);
        attend(l, T);
        if (!head) return;
        // the last row on: the residual, the FFN, the head
        float* xl = mtp_x + (size_t) (T - 1) * N;
        hy4k::k_add<<<blocks(N), hy4k::kThreads, 0, st>>>(xl, y + (size_t) (T - 1) * N, N);
        hy4k::k_rms<<<1, hy4k::kThreads, 0, st>>>(xl, N, l.ffn_norm, xn, N, N, g.eps);
        quant(xn, 1, N, xq);
        ev_moe = ev_mtp;
        moe_front(l, 1);
        cpu_part(l, 1);
        ev_moe = ev;
        moe_out(l, 1);
        hy4k::k_add<<<blocks(N), hy4k::kThreads, 0, st>>>(xl, y, N);
        hy4k::k_rms<<<1, hy4k::kThreads, 0, st>>>(xl, N, m->mtp_shnorm, mtp_h, N, N, g.eps);
        K::bf16_gemv_fp32_mmvf_multi(mtp_h, m->head, mtp_lg, N, g.n_vocab, 1, st);
    }
    /// DFly2's greedy drafts after `bonus`, the token at `anchor` (the state's end): one pass of its layers over
    /// the block [bonus, masks] at anchor .., row 0 fused with the trunk's state of anchor - 1, each layer's cells the
    /// trunk layer's picks of that row and the block's own (its K/V written into the trunk layer's cells past the
    /// state); then draft k from row k + 1's state corrected with the embedding of draft k - 1 (the bonus token's for
    /// the first), through the trunk's head.  None when the last pass did not end at anchor - 1 (a restored state),
    /// fewer than n past --spec-min-p.
    // the block pass's window values on the device ([0] the source row, [1] the anchor, [2] the ring's row), the
    // trunk's state row copied out of the ring, the pass's graphs by block size (LAMINA_HY4_DFLY_GRAPH=0: none)
    int32_t *dfly_prm_d = nullptr, *h_dfly_prm = nullptr;
    float* dfly_ring = nullptr;
    std::map<int, cudaGraphExec_t> dfly_graphs;
    bool dfly_graph_on = [] {
        const char* v = std::getenv("LAMINA_HY4_DFLY_GRAPH");
        return v == nullptr || std::atoi(v) != 0;
    }();
    /// The block's W rows through DFly's layers into dfly_out: the cells, the embeddings fused with the trunk's state
    /// (row 0), the layers, the final norm; the window's values from dfly_prm_d.  `dump`: the embeddings to the host
    /// (not under capture)
    void dfly_block(int W, bool dump) {
        const auto& g = m->g;
        const auto& d = m->dfly;
        const int N = (int) g.n_embd;
        for (size_t i = 0; i < d.layers.size(); ++i)
            hy4k::k_dfly_cells_d<<<W, 256, 0, st>>>(snap_cells, snap_n, dfly_prm_d, (int) i, kMaxT, (int) g.idx_top_k,
                                                    dfly_cells + i * d.width * (size_t) dfly_ld, dfly_n + i * d.width,
                                                    dfly_ld);
        // the block's rows; row 0 += gate * eh_proj([enorm(row 0) | hnorm(the trunk's state of anchor - 1)])
        K::iq_embed_rows(m->embd_type, m->embd, m->embd_row, g.n_vocab, tok_d, W, N, dfly_h, st);
        if (dump) {
            dfly_dump_emb.resize((size_t) W * N);
            CK(cudaMemcpyAsync(dfly_dump_emb.data(), dfly_h, dfly_dump_emb.size() * 4, cudaMemcpyDeviceToHost, st));
        }
        hy4k::k_rms<<<1, hy4k::kThreads, 0, st>>>(dfly_h, N, d.enorm, dfly_cat, 2 * N, N, g.eps);
        hy4k::k_ring_row<<<(N + 255) / 256, 256, 0, st>>>(hring, dfly_prm_d, dfly_ring, N);
        hy4k::k_rms<<<1, hy4k::kThreads, 0, st>>>(dfly_ring, N, d.hnorm, dfly_cat + N, 2 * N, N, g.eps);
        quant(dfly_cat, 1, 2 * N, dfly_catq);
        mmvq(d.eh_proj, dfly_catq, dfly_d, 1);
        hy4k::k_gated_add<<<blocks(N), hy4k::kThreads, 0, st>>>(dfly_h, d.fgate, dfly_d, N);
        // the layers (pre-norm, one residual stream): h = attention(norm(h)) + h, h = FFN(norm(h)) + h; the sum normed
        const int64_t WN = (int64_t) W * N;
        for (size_t i = 0; i < d.layers.size(); ++i) {
            Layer& l = m->dfly.layers[i];
            if (i > 0) hy4k::k_add<<<blocks(WN), hy4k::kThreads, 0, st>>>(dfly_h, dfly_res, WN);
            CK(cudaMemcpyAsync(dfly_res, dfly_h, (size_t) WN * 4, cudaMemcpyDeviceToDevice, st));
            hy4k::k_rms<<<W, hy4k::kThreads, 0, st>>>(dfly_h, N, l.attn_norm, xn, N, N, g.eps);
            attend(l, W);
            hy4k::k_add<<<blocks(WN), hy4k::kThreads, 0, st>>>(dfly_res, y, WN);
            hy4k::k_rms<<<W, hy4k::kThreads, 0, st>>>(dfly_res, N, l.ffn_norm, xn, N, N, g.eps);
            quant(xn, W, N, xq);
            const int F = l.f_gate.n_out;
            mmvq(l.f_gate, xq, fg, W);
            mmvq(l.f_up, xq, fu, W);
            hy4k::k_swiglu<<<blocks((int64_t) W * F), hy4k::kThreads, 0, st>>>(fg, fu, fh, (int64_t) W * F);
            quant(fh, W, F, hq);
            mmvq(l.f_down, hq, dfly_h, W);
        }
        hy4k::k_add<<<blocks(WN), hy4k::kThreads, 0, st>>>(dfly_h, dfly_res, WN);
        hy4k::k_rms<<<W, hy4k::kThreads, 0, st>>>(dfly_h, N, d.norm, dfly_out, N, N, g.eps);
    }

    std::vector<int32_t> dfly_draft(int32_t bonus, int64_t anchor, int n) {
        const auto& g = m->g;
        const auto& d = m->dfly;
        const int N = (int) g.n_embd;
        std::vector<int32_t> out;
        n = std::min(n, d.width - 1);
        const int W = n + 1;   // the block's rows needed (causal: row k + 1 sees rows 0 .. k + 1 alone)
        const int64_t src = anchor - 1 - snap_base;
        if (n < 1 || snap_base < 0 || src < 0 || src >= snap_rows || anchor + W > ctx) return out;
        ensure(anchor + W);
        for (int t = 0; t < W; ++t) {
            h_tok[t] = t == 0 ? bonus : d.mask;
            h_pos[t] = (int32_t) (anchor + t);
        }
        CK(cudaMemcpyAsync(tok_d, h_tok, (size_t) W * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(pos_d, h_pos, (size_t) W * 4, cudaMemcpyHostToDevice, st));
        h_dfly_prm[0] = (int32_t) src;
        h_dfly_prm[1] = (int32_t) anchor;
        h_dfly_prm[2] = (int32_t) ((anchor - 1) % kHRing);
        CK(cudaMemcpyAsync(dfly_prm_d, h_dfly_prm, 3 * 4, cudaMemcpyHostToDevice, st));
        const bool dump = dfly_dump_path != nullptr;
        if (use_graphs && dfly_graph_on && !dump && trace < 3) {   // the block pass's graph for W rows
            auto it = dfly_graphs.find(W);
            if (it == dfly_graphs.end()) {
                CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
                dfly_block(W, false);
                cudaGraphExec_t ex = graph_end(st);
                it = dfly_graphs.emplace(W, ex).first;
            }
            CK(cudaGraphLaunch(it->second, st));
        } else {
            dfly_block(W, dump);
        }
        // the drafts: row k + 1's state + down(silu(gate) * up([hidden_norm(it) | embed_norm(the last token)]))
        double joint = 1.0;
        for (int k = 0; k < n; ++k) {
            const float* hk = dfly_out + (size_t) (k + 1) * N;
            if (k > 0) {   // the last draft's embedding (tok_d row 0 held the bonus token for the first)
                h_tok[0] = out.back();
                CK(cudaMemcpyAsync(tok_d, h_tok, 4, cudaMemcpyHostToDevice, st));
            }
            K::iq_embed_rows(m->embd_type, m->embd, m->embd_row, g.n_vocab, tok_d, 1, N, emb, st);
            hy4k::k_rms<<<1, hy4k::kThreads, 0, st>>>(hk, N, d.hc_hnorm, dfly_cat, 2 * N, N, g.eps);
            hy4k::k_rms<<<1, hy4k::kThreads, 0, st>>>(emb, N, d.hc_enorm, dfly_cat + N, 2 * N, N, g.eps);
            quant(dfly_cat, 1, 2 * N, dfly_catq);
            const int F = d.hc_gate.n_out;
            mmvq(d.hc_gate, dfly_catq, fg, 1);
            mmvq(d.hc_up, dfly_catq, fu, 1);
            hy4k::k_swiglu<<<blocks(F), hy4k::kThreads, 0, st>>>(fg, fu, fh, F);
            quant(fh, 1, F, hq);
            mmvq(d.hc_down, hq, dfly_d, 1);
            hy4k::k_add<<<blocks(N), hy4k::kThreads, 0, st>>>(dfly_d, hk, N);
            if (dump && k == 0) dfly_dump(anchor, bonus, W, src);
            K::bf16_gemv_fp32_mmvf_multi(dfly_d, m->head, dfly_lg, N, g.n_vocab, 1, st);
            const int32_t tok = pick_draft(dfly_lg, joint);
            if (tok < 0) break;
            out.push_back(tok);
        }
        return out;
    }
    // LAMINA_HY4_DFLY_DUMP=FILE: DFly's first pass written for a check against a reference from the BF16 weights: the
    // sizes, the trunk's state, the block's embeddings, each layer's picks and its trunk layer's K/V rows
    // up to the block's end (the block's own written by the pass), the block's final states, the first draft's state
    const char* dfly_dump_path = std::getenv("LAMINA_HY4_DFLY_DUMP");
    std::vector<float> dfly_dump_emb;
    void dfly_dump(int64_t anchor, int32_t bonus, int W, int64_t src) {
        const auto& g = m->g;
        const auto& d = m->dfly;
        const int N = (int) g.n_embd;
        CK(cudaStreamSynchronize(st));
        std::FILE* f = std::fopen(dfly_dump_path, "wb");
        dfly_dump_path = nullptr;
        if (!f) return;
        const int32_t head[12] = {N,
                                  W,
                                  (int32_t) d.layers.size(),
                                  d.layers[0].H,
                                  (int32_t) g.nope(),
                                  (int32_t) g.n_rot,
                                  (int32_t) g.v_head,
                                  hy4k::kLatent,
                                  (int32_t) anchor,
                                  bonus,
                                  d.mask,
                                  hy4k::kKvRow};
        std::fwrite(head, 4, 12, f);
        std::vector<float> h((size_t) N);
        CK(cudaMemcpy(h.data(), hring + (size_t) ((anchor - 1) % kHRing) * N, (size_t) N * 4, cudaMemcpyDeviceToHost));
        std::fwrite(h.data(), 4, h.size(), f);
        std::fwrite(dfly_dump_emb.data(), 4, dfly_dump_emb.size(), f);
        for (size_t i = 0; i < d.layers.size(); ++i) {
            int32_t n = 0;
            CK(cudaMemcpy(&n, snap_n + i * kMaxT + src, 4, cudaMemcpyDeviceToHost));
            std::vector<int32_t> picks((size_t) n);
            CK(cudaMemcpy(picks.data(), snap_cells + (i * kMaxT + (size_t) src) * g.idx_top_k, (size_t) n * 4,
                          cudaMemcpyDeviceToHost));
            const Layer& dl = d.layers[i];
            const size_t cells = (size_t) (anchor + W);
            std::vector<uint8_t> raw(cells * dl.kv_row());
            CK(cudaMemcpy(raw.data(), dl.kv, raw.size(), cudaMemcpyDeviceToHost));
            std::vector<__half> kv(cells * hy4k::kKvRow);   // the rows as halves (q8_0 ones
                                                            // widened as the attention does)
            if (!dl.kv_q8) std::memcpy(kv.data(), raw.data(), raw.size());
            else
                for (size_t c = 0; c < cells; ++c) {
                    const uint8_t* r = raw.data() + c * hy4k::kKvQ8Row;
                    for (int j = 0; j < hy4k::kKvRow; ++j) {
                        __half dh;
                        std::memcpy(&dh, r + hy4k::kKvRow + 2 * (j / 32), 2);
                        kv[c * hy4k::kKvRow + (size_t) j] = __float2half((float) (int8_t) r[j] * __half2float(dh));
                    }
                }
            const int32_t t = d.target[i];
            std::fwrite(&t, 4, 1, f);
            std::fwrite(&n, 4, 1, f);
            std::fwrite(picks.data(), 4, picks.size(), f);
            std::fwrite(kv.data(), 2, kv.size(), f);
        }
        std::vector<float> rows((size_t) W * N);
        CK(cudaMemcpy(rows.data(), dfly_out, rows.size() * 4, cudaMemcpyDeviceToHost));
        std::fwrite(rows.data(), 4, rows.size(), f);
        CK(cudaMemcpy(h.data(), dfly_d, (size_t) N * 4, cudaMemcpyDeviceToHost));
        std::fwrite(h.data(), 4, h.size(), f);
        std::fclose(f);
        std::fprintf(stderr, "lamina: DFly's pass at %lld written for its reference\n", (long long) anchor);
    }

    /// The state source of position p's MTP row: the ring's slot of p - 1, zeros at 0
    static int32_t ring_src(int64_t p) { return p == 0 ? hy4k::kNoState : (int32_t) ((p - 1) % kHRing); }
    /// The MTP block caught up with the trunk over n tokens at
    /// positions p0 .. (their states in the ring): its attention
    void mtp_catch_up(const int32_t* toks, int64_t n, int64_t p0) {
        std::vector<int32_t> src;
        for (int64_t i = 0; i < n; i += kMaxT) {
            const int T = (int) std::min<int64_t>(kMaxT, n - i);
            src.resize((size_t) T);
            for (int t = 0; t < T; ++t) src[(size_t) t] = ring_src(p0 + i + t);
            mtp_forward(toks + i, src.data(), T, p0 + i, false);
        }
    }
    /// One drafted window after x, the token at pos() (`rows`: the tokens the MTP block has not caught up with, x
    /// last): up to n drafts (the MTP block's or DFly's), the window [x, drafts] through the trunk, row k's target
    /// token from `pick` (its logits; the history of the tokens kept so far) while it matches draft k; the state then
    /// ends after the kept drafts.
    /// `rows` becomes the window's kept tokens (x and the drafts kept), `next` the first target token not a draft (or
    /// -1: sample it from `lg`, the window's last row's logits), and `kept` the drafts kept, each passed to `keep`
    /// first (false: stop there, the state up to it).
    struct Window {
        std::vector<int32_t> drafts;
        int kept = 0;
        int32_t next = -1;
        bool looked = false;   // the drafts the context's own continuation (prompt lookup)
    };
    // prompt lookup (--spec-lookup N): when the text's last N tokens or more occurred before and the tokens that
    // followed there outnumber the drafter's drafts, the window takes up to kMaxT - 1 of them instead; the drafter's
    // history is the state's tokens, its first sfx_ok of them right (rewind and unpark lower it)
    int lookup_min = 16;
    std::unique_ptr<lamina::spec::SuffixDrafter> sfx;
    size_t sfx_ok = 0;
    int64_t looked_windows = 0;
    /// The context's continuation after x (the token at pos()) when it is longer than n tokens, else none
    std::vector<int32_t> look_up(int32_t x, int n) {
        std::vector<int32_t> out;
        if (!sfx || n <= 0) return out;
        if (sfx->size() > sfx_ok) sfx->truncate(sfx_ok);
        if (seq.size() > sfx_ok) {
            sfx->append(seq.data() + sfx_ok, seq.size() - sfx_ok);
            sfx_ok = seq.size();
        }
        sfx->append(x);   // not in the state yet: the next sync takes it out
        out.resize(kMaxT - 1);
        const int k = sfx->propose(kMaxT - 1, out.data());
        out.resize(k > n ? (size_t) k : 0);
        return out;
    }
    template <class Pick, class Keep>
    Window drafted(std::vector<int32_t>& rows, int n, std::vector<float>& lg, Pick pick, Keep keep) {
        Window w;
        const int64_t P = pos();
        const auto td = Clock::now();
        w.drafts = look_up(rows.back(), n);
        if (!w.drafts.empty()) {   // the drafter's pass skipped; the MTP block over the rows before x (x comes later)
            w.looked = true;
            ++looked_windows;
            if (!m->dfly.on && m->has_mtp && rows.size() > 1)
                mtp_catch_up(rows.data(), (int64_t) rows.size() - 1, P - (int64_t) rows.size() + 1);
        } else {
            w.drafts = m->dfly.on ? dfly_draft(rows.back(), P, n) : draft(rows, P - (int64_t) rows.size() + 1, n);
        }
        if (trace) tr.draft += ms_since(td);
        std::vector<int32_t> win{rows.back()};
        win.insert(win.end(), w.drafts.begin(), w.drafts.end());
        forward(win.data(), (int) win.size());
        bool stopped = false;
        for (size_t k = 0; k <= w.drafts.size(); ++k) {
            host_logits((int) k, lg);
            if (k == w.drafts.size()) break;   // every draft kept: lg is the window's last row
            const int32_t t = pick(lg);
            if (t != w.drafts[k]) {
                w.next = t;
                break;
            }
            ++w.kept;
            if (!keep(t)) {
                stopped = true;
                break;
            }
        }
        rewind(P + w.kept + 1);   // the window's rows after the kept drafts out (fewer than kMaxT: no snapshot)
        rows.assign(win.begin(), win.begin() + 1 + w.kept);
        if (stopped) w.next = -2;
        return w;
    }

    /// Greedy drafts after `rows` (tokens at positions p0 .., the last the token they follow, every row's state in the
    /// ring): one pass for the rows and the first draft, then each next draft from the last one's output state.
    /// A draft from a row of the drafter's logits on the device: its argmax and its probability under them
    /// (1 / sum exp(l - max)) by k_pick, 8 bytes to the host; -1 once the drafts' joint probability falls below
    /// --spec-min-p (`joint` the product so far)
    int32_t pick_draft(const float* lg_d, double& joint) {
        hy4k::k_pick<<<1, 1024, 0, st>>>(lg_d, (int) m->g.n_vocab, pick_d);
        CK(cudaMemcpyAsync(h_pick, pick_d, 8, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        if (spec_min_p > 0.f) {
            float z = 0.f;
            std::memcpy(&z, &h_pick[1], 4);
            joint /= (double) z;
            if (joint < spec_min_p) return -1;
        }
        return h_pick[0];
    }
    int32_t *pick_d = nullptr, *h_pick = nullptr;
    std::vector<int32_t> draft(const std::vector<int32_t>& rows, int64_t p0, int n) {
        std::vector<int32_t> out, src(rows.size());
        for (size_t i = 0; i < rows.size(); ++i) src[i] = ring_src(p0 + (int64_t) i);
        mtp_forward(rows.data(), src.data(), (int) rows.size(), p0, true);
        int64_t at = p0 + (int64_t) rows.size();
        double joint = 1.0;
        for (int i = 0; i < n; ++i) {
            const int32_t pick = pick_draft(mtp_lg, joint);
            if (pick < 0) break;
            out.push_back(pick);
            if (i + 1 == n) break;
            const int32_t tok = out.back(), own = -1;   // mtp_h row 0
            mtp_forward(&tok, &own, 1, at++, true);
        }
        return out;
    }

    // ---- the routed experts in VRAM: the main GPU's slots (holder 0) and the expert tiers' (the other GPUs, holders
    // 1..), every MoE layer the same count a GPU.  One ranking of a layer's experts by its routing counts places them:
    // the most used on the main GPU, the next on the first tier and so on, the rest on the CPU; an expert is held by
    // one GPU at most.
    struct Gpu {
        int dev = 0;
        std::string name;
        cudaStream_t st = nullptr;
        cudaEvent_t ev = nullptr;
        int S = 0;   // slots a layer
        // a tier's own buffers on its GPU: the routing and the routed input, its entries grouped, their rows
        int32_t* ids_d = nullptr;
        float* rin_d = nullptr;
        uint8_t *rinq_d = nullptr, *escr = nullptr;
        unsigned long long* grp_ptr = nullptr;
        int32_t *grp_start = nullptr, *n_groups = nullptr, *ent_dst = nullptr, *ent_tok = nullptr;
        float* rows_d = nullptr;
        float** out_d = nullptr;       // h_out's address (the mapping's, the same on every GPU)
        uint32_t** flag_d = nullptr;   // null: native_expert_rows_out raises no flag
        unsigned* counter_d = nullptr;
        std::map<int, cudaGraphExec_t> graphs;   // layer * 16 + rows
        // --tier-doorbell: the doorbells (mapped, a segment each), the shares' events, the forward's graphs by rows
        uint32_t *door = nullptr, *door_d = nullptr;
        std::vector<cudaEvent_t> door_ev;
        std::map<int, cudaGraphExec_t> door_graphs;
        int64_t hits = 0;
        // the slots' copies from the arena: their stream, the copies in flight in order, their events, the queue
        cudaStream_t cst = nullptr;
        std::deque<SlotMove> moves;
        std::vector<cudaEvent_t> free_ev;
        std::unique_ptr<CopyQueue> mover;
        double h2d_gbps = 6.0;          // its copies from the arena (the slots'
                                        // first fill), for the copies kept in flight
        std::vector<Planned> backlog;   // the last plan's moves for it, the largest
                                        // gain first; `next` the first not started
        size_t next = 0;
    };
    std::vector<Gpu> gpus;                    // [0]: the main GPU
    static constexpr int32_t kPending = -2;   // slot_expert of a slot whose copy is in flight
    // the moves: when an answer starts, the slots toward the ranking with the prompt's routing counted, every copy it
    // takes (LAMINA_HY4_ANSWER_MOVE=0: none); every 128 generated tokens up to move_budget copies
    int move_at_start = [] {
        const char* v = std::getenv("LAMINA_HY4_ANSWER_MOVE");
        return v ? std::max(0, std::atoi(v)) : 512;
    }();
    int move_budget = [] {
        const char* v = std::getenv("LAMINA_HY4_MOVE_BUDGET");
        return v ? std::max(0, std::atoi(v)) : 32;
    }();
    // --adapt-every 0: the start's placement stays - no moves planned (at an answer's start, during it, after it), the
    // residents a lend displaced back before the next forward - so every process writes the same output
    bool adapt = true;
    int main_dev = 0;
    std::vector<char> tier_used;   // a forward's layer: the tiers given entries

    /// The main GPU's slots (as many a layer as `main_gib` GiB hold; negative: its free VRAM less a reserve) and each
    /// of `tier_devs`' (its free VRAM less a reserve), filled by the profile's ranking (else experts in order).
    void place_slots(double main_gib, const std::vector<int>& tier_devs) {
        const auto& g = m->g;
        CK(cudaGetDevice(&main_dev));
        size_t row = 0;
        for (const Layer& l : m->layers)
            if (!l.dense) row += l.stride;
        auto slots_for = [&](double gib, size_t reserve) {
            size_t fr = 0, tot = 0;
            CK(cudaMemGetInfo(&fr, &tot));
            const size_t budget = gib >= 0 ? (size_t) (gib * (double) (1ull << 30)) : (fr > reserve ? fr - reserve : 0);
            return (int) std::min<size_t>((size_t) g.n_expert, row ? budget / row : 0);
        };
        gpus.clear();
        Gpu g0;
        g0.dev = main_dev;
        g0.st = st;
        g0.ev = ev;
        {
            char bus[32] = {};
            bool known = false;
            if (cudaDeviceGetPCIBusId(bus, sizeof bus, main_dev) == cudaSuccess &&
                !lamina::core::gpu_in_desktop_use(bus, known) && known)
                kReserve = (size_t) 700 << 20;
        }
        g0.S = slots_for(main_gib, kReserve);
        cudaDeviceProp pr{};
        CK(cudaGetDeviceProperties(&pr, main_dev));
        g0.name = pr.name;
        gpus.push_back(std::move(g0));
        for (const int d : tier_devs) {
            lamina::core::DeviceScope ds(d, main_dev);
            Gpu t;
            t.dev = d;
            CK(cudaGetDeviceProperties(&pr, d));
            t.name = pr.name;
            t.st = stream_new();
            t.ev = event_new();
            const size_t n_ent = (size_t) kMaxT * (size_t) g.top_k;
            t.ids_d = dalloc<int32_t>(n_ent);
            t.rin_d = dalloc<float>((size_t) kMaxT * (size_t) g.n_latent);
            t.rinq_d = dalloc<uint8_t>((size_t) kMaxT * (size_t) g.n_latent / 32 * 36);
            t.grp_ptr = dalloc<unsigned long long>(n_ent + 1);
            t.grp_start = dalloc<int32_t>(n_ent + 2);
            t.n_groups = dalloc<int32_t>(1);
            t.ent_dst = dalloc<int32_t>(n_ent);
            t.ent_tok = dalloc<int32_t>(n_ent);
            t.escr = dalloc<uint8_t>(K::native_expert_scratch_bytes((int64_t) n_ent, g.n_ff_exp));
            t.rows_d = dalloc<float>(n_ent * (size_t) g.n_latent);
            t.out_d = dalloc<float*>(1);
            t.flag_d = dalloc<uint32_t*>(1);
            t.counter_d = dalloc<unsigned>(1);
            float* out = h_out_d;
            uint32_t* none = nullptr;
            CK(cudaMemcpy(t.out_d, &out, sizeof out, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(t.flag_d, &none, sizeof none, cudaMemcpyHostToDevice));
            CK(cudaMemset(t.counter_d, 0, sizeof(unsigned)));
            t.S = slots_for(-1, (size_t) 1280 << 20);
            gpus.push_back(std::move(t));
        }
        for (Gpu& gp : gpus) {   // the slots' copies: a stream and an issuing thread a GPU
            lamina::core::DeviceScope ds(gp.dev, main_dev, gp.dev != main_dev);
            gp.cst = stream_new();
            size_t blob = 0;
            for (const Layer& l : m->layers)
                if (!l.dense) blob = std::max(blob, l.fmt.bytes);
            gp.mover = std::make_unique<CopyQueue>(gp.dev, gp.cst, blob);
        }
        if (lend_bytes > 0 && elastic && !gpus.empty()) {   // the slots the prompt path's buffers make when lent
            size_t row_bytes = 0;
            for (const Layer& l : m->layers)
                if (!l.dense) row_bytes += l.stride;
            lend_slots = row_bytes ? (int) (lend_bytes / row_bytes) : 0;
        }
        load_profile();
        const auto fill0 = Clock::now();
        for (Layer& l : m->layers) {
            if (l.dense) continue;
            l.moving.assign((size_t) g.n_expert, 0);
            l.gl = K::native_expert_layout(l.fmt.gu_type, l.fmt.d_type, g.n_latent, g.n_ff_exp);
            l.gl.swiglu_limit = l.fmt.swiglu_limit;
            if (l.use.empty()) l.use.assign((size_t) g.n_expert, 0.f);
            l.total.assign((size_t) g.n_expert, 0.f);
            const bool ok = K::native_expert_supported(l.fmt.gu_type, l.fmt.d_type, g.n_latent, g.n_ff_exp);
            l.held.assign(gpus.size(), SlotSet{});
            for (size_t gi = 0; gi < gpus.size(); ++gi) {
                lamina::core::DeviceScope ds(gpus[gi].dev, main_dev, gi > 0);
                SlotSet& s = l.held[gi];
                s.slot_of.assign((size_t) g.n_expert, -1);
                s.slot_of_d = dalloc<int32_t>((size_t) g.n_expert);
                if (!ok || gpus[gi].S <= 0) continue;
                s.n = gpus[gi].S;
                if (gi == 0 && elastic) {   // a range the K/V can take the last slots' chunks from
                    l.slot_vm = std::make_shared<lamina::core::VmmRange>();
                    const uint64_t G = lamina::core::vmm_granularity();
                    const int64_t mapped = (int64_t) (((uint64_t) s.n * l.stride + kWeightPad + G - 1) / G);
                    if (!l.slot_vm->reserve((uint64_t) (s.n + lend_slots) * l.stride + kWeightPad) ||
                        !l.slot_vm->map_range(0, mapped, [] { return (lamina::core::VmmChunk) 0; })) {
                        std::fprintf(stderr, "lamina: the main GPU's expert slots could not be mapped\n");
                        throw Fatal{1};
                    }
                    s.slots = l.slot_vm->base();
                } else {
                    s.slots = dalloc<uint8_t>((size_t) s.n * l.stride);
                }
                s.slot_expert.assign((size_t) s.n, -1);
            }
            const std::vector<int32_t> order = ranking(l);
            size_t at = 0;
            for (size_t gi = 0; gi < gpus.size(); ++gi)
                for (int k = 0; k < l.held[gi].n && at < order.size(); ++k) put_slot(l, (int) gi, k, order[at++]);
            for (size_t gi = 0; gi < gpus.size(); ++gi) upload_table(l, (int) gi);
        }
        {   // each GPU's fill as one stretch: its copy rate from the arena
            size_t blob = 0;
            for (const Layer& l : m->layers)
                if (!l.dense) blob += l.fmt.bytes;
            for (Gpu& gp : gpus) {
                lamina::core::DeviceScope ds(gp.dev, main_dev, gp.dev != main_dev);
                CK(cudaStreamSynchronize(gp.st));
                const double sec = ms_since(fill0) / 1000.0;
                if (gp.S > 0 && sec > 0.05) gp.h2d_gbps = std::max(1.0, (double) gp.S * (double) blob / sec / 1e9);
                CK(cudaDeviceSynchronize());
            }
        }
        tier_used.assign(gpus.size(), 0);
        s_max = gpus.empty() ? 0 : std::max(gpus[0].S, 0);
        for (size_t gi = 0; gi < gpus.size(); ++gi)
            std::fprintf(
                stderr, "lamina: GPU %d, %s: %d of %lld routed experts a layer (%.2f GiB)%s\n", gpus[gi].dev,
                gpus[gi].name.c_str(), std::max(gpus[gi].S, 0), (long long) g.n_expert,
                (double) std::max(gpus[gi].S, 0) * (double) row / (double) (1ull << 30),
                gi == 0
                    ? (profile_loaded ? ", ranked by " + profile_from : std::string(", in order (no profile)")).c_str()
                    : "");
    }
    /// --arena-place: the pinned places of each layer's experts timed (every core reading one at a time); within a
    /// layer the experts the CPU reads (on no GPU), the most used first, take its fast places, swapped with the slow
    /// places' experts (those on a GPU, then the CPU's least used).  `dry`: timed and reported only.
    void place_arena(bool dry) {
        const auto t0 = Clock::now();
        std::vector<const uint8_t*> places;
        std::vector<std::pair<int, int32_t>> owner;   // (layer, expert)
        size_t bytes = SIZE_MAX, widest = 0;          // the places timed over the smallest
                                                      // blob; a swap's buffer the widest
        for (size_t li = 0; li < m->layers.size(); ++li) {
            const Layer& l = m->layers[li];
            if (l.dense || l.blob.empty()) continue;
            bytes = std::min(bytes, l.fmt.bytes);
            widest = std::max(widest, l.fmt.bytes);
            for (size_t e = 0; e < l.blob.size(); ++e)
                if (l.blob_pinned[e]) {
                    places.push_back(l.blob[e]);
                    owner.emplace_back((int) li, (int32_t) e);
                }
        }
        if (places.empty()) {
            std::fprintf(stderr, "lamina: --arena-place: no pinned places\n");
            return;
        }
        float cut = 0.f;
        const std::vector<float> gbs = lamina::hy4::time_places(places, (uint64_t) bytes, &cut);
        auto on_gpu = [&](const Layer& l, int32_t e) {
            for (const SlotSet& s : l.held)
                if (s.n > 0 && s.slot_of[(size_t) e] >= 0) return true;
            return false;
        };
        size_t slow = 0, cpu = 0, cpu_slow = 0;
        double sum_gbs = 0;
        for (size_t pq = 0; pq < places.size(); ++pq) {
            const Layer& l = m->layers[(size_t) owner[pq].first];
            const bool s = gbs[pq] < cut, c = !on_gpu(l, owner[pq].second);
            slow += s;
            cpu += c;
            cpu_slow += s && c;
            sum_gbs += gbs[pq];
        }
        // the swaps: within a layer, its pinned places' experts by key (the CPU's, the most used first; then the
        // GPUs'), the first as many as there are fast places into them
        size_t swaps = 0;
        if (!dry) {
            std::vector<uint8_t> tmp(widest);
            std::map<int, std::vector<size_t>> by_layer;
            for (size_t pq = 0; pq < places.size(); ++pq) by_layer[owner[pq].first].push_back(pq);
            for (auto& [li, qs] : by_layer) {
                Layer& l = m->layers[(size_t) li];
                std::vector<int32_t> rank_of(l.use.size(), 0);
                {
                    const std::vector<int32_t> order = ranking(l);
                    for (size_t r = 0; r < order.size(); ++r) rank_of[(size_t) order[r]] = (int32_t) r;
                }
                auto key = [&](int32_t e) {
                    return on_gpu(l, e) ? (int64_t) 1 << 40 | rank_of[(size_t) e] : rank_of[(size_t) e];
                };
                std::vector<size_t> by_key = qs;   // places by their expert's key
                std::sort(by_key.begin(), by_key.end(),
                          [&](size_t a, size_t b) { return key(owner[a].second) < key(owner[b].second); });
                size_t nf = 0;
                for (const size_t pq : qs) nf += gbs[pq] >= cut;
                std::vector<size_t> out, in;   // the first nf experts in slow places, the others in fast places
                for (size_t k = 0; k < by_key.size(); ++k) {
                    if (k < nf && gbs[by_key[k]] < cut) out.push_back(by_key[k]);
                    if (k >= nf && gbs[by_key[k]] >= cut) in.push_back(by_key[k]);
                }
                for (size_t k = 0; k < out.size() && k < in.size(); ++k) {
                    const size_t a = out[k], b = in[k];
                    const int32_t ea = owner[a].second, eb = owner[b].second;
                    uint8_t* pa = l.blob[(size_t) ea];
                    uint8_t* pb = l.blob[(size_t) eb];
                    std::memcpy(tmp.data(), pa, l.fmt.bytes);
                    std::memcpy(pa, pb, l.fmt.bytes);
                    std::memcpy(pb, tmp.data(), l.fmt.bytes);
                    std::swap(l.blob[(size_t) ea], l.blob[(size_t) eb]);
                    std::swap(owner[a].second, owner[b].second);
                    ++swaps;
                }
            }
        }
        std::fprintf(stderr,
                     "lamina: --arena-place: %zu pinned places timed in %.1f s (mean %.1f GB/s, slow under %.1f): "
                     "%zu slow (%.1f GiB), %zu of the CPU's %zu experts in them; %zu swapped%s\n",
                     places.size(), ms_since(t0) / 1000.0, sum_gbs / (double) places.size(), cut, slow,
                     (double) slow * (double) bytes / (double) (1ull << 30), cpu_slow, cpu, swaps, dry ? " (dry)" : "");
    }
    /// A layer's experts, the most used first (ties by id)
    std::vector<int32_t> ranking(const Layer& l) const {
        std::vector<int32_t> order(l.use.size());
        for (size_t e = 0; e < order.size(); ++e) order[e] = (int32_t) e;
        std::stable_sort(order.begin(), order.end(),
                         [&](int32_t a, int32_t b) { return l.use[(size_t) a] > l.use[(size_t) b]; });
        return order;
    }
    void upload_table(Layer& l, int gi) {
        lamina::core::DeviceScope ds(gpus[(size_t) gi].dev, main_dev, gi > 0);
        SlotSet& s = l.held[(size_t) gi];
        for (size_t at = 0; at < s.slot_of.size(); at += 512) {   // by a kernel: the copy engine has the slots' copies
            hy4k::SlotTable t;
            const int n = (int) std::min<size_t>(512, s.slot_of.size() - at);
            std::memcpy(t.v, s.slot_of.data() + at, (size_t) n * 4);
            hy4k::k_slot_table<<<1, 512, 0, gpus[(size_t) gi].st>>>(s.slot_of_d + at, t, n);
        }
        CK(cudaGetLastError());
    }
    /// Expert e into GPU gi's slot k (its blob from the RAM arena), out of any other GPU's slot
    void put_slot(Layer& l, int gi, int k, int32_t e) {
        SlotSet& s = l.held[(size_t) gi];
        for (size_t oi = 0; oi < l.held.size(); ++oi) {
            SlotSet& so = l.held[oi];
            if (so.slot_of.empty() || so.slot_of[(size_t) e] < 0) continue;
            so.slot_expert[(size_t) so.slot_of[(size_t) e]] = -1;
            so.slot_of[(size_t) e] = -1;
        }
        if (s.slot_expert[(size_t) k] >= 0) s.slot_of[(size_t) s.slot_expert[(size_t) k]] = -1;
        {
            lamina::core::DeviceScope ds(gpus[(size_t) gi].dev, main_dev, gi > 0);
            CK(cudaMemcpyAsync(s.slots + (size_t) k * l.stride, l.blob[(size_t) e], l.fmt.bytes, cudaMemcpyHostToDevice,
                               gpus[(size_t) gi].st));
        }
        s.slot_expert[(size_t) k] = e;
        s.slot_of[(size_t) e] = k;
    }
    /// The GPUs' slots toward the ranking: in each layer the experts the GPUs hold together should be the most used, as
    /// many as they have slots; each of those on the CPU takes an empty slot, else the slot of the least used resident
    /// outside them while that one is used less (by 1.25x).  Which GPU holds an expert is left as it is (a tier's rows
    /// come beside the CPU's).  The moves, the largest gain first, at most `max_copies` (and no fewer than the moves
    /// still waiting from the last plan), replace those: pump() copies them a GPU's link at a time, a resident leaving
    /// its slot when its copy starts, an expert switched in when its copy has landed (settle).  With `halve` the counts
    /// are halved after.
    int plan_moves(int max_copies, bool halve) {
        dump_route(4, halve ? max_copies : -max_copies, 0, nullptr);   // negative: the counts not halved
        std::vector<Planned> cand;
        for (size_t li = 0; li < m->layers.size(); ++li) {
            Layer& l = m->layers[li];
            if (l.dense || l.held.empty()) continue;
            const std::vector<int32_t> order = ranking(l);
            int total = 0;
            for (const SlotSet& s : l.held) total += s.n;
            std::vector<char> top(order.size(), 0);
            for (int i = 0; i < total && i < (int) order.size(); ++i) top[(size_t) order[(size_t) i]] = 1;
            auto on_gpu = [&](int32_t e) {
                for (const SlotSet& s : l.held)
                    if (s.n > 0 && s.slot_of[(size_t) e] >= 0) return true;
                return false;
            };
            std::vector<int32_t> in;   // the most used first
            for (int i = 0; i < total && i < (int) order.size(); ++i) {
                const int32_t e = order[(size_t) i];
                if (!on_gpu(e) && !l.moving[(size_t) e]) in.push_back(e);
            }
            if (in.empty()) continue;
            struct Slot {
                int gi, k;
                int32_t e;
            };
            std::vector<Slot> free, out;
            for (size_t gi = 0; gi < l.held.size(); ++gi) {
                const SlotSet& s = l.held[gi];
                for (int k = 0; k < s.n; ++k) {
                    const int32_t e = s.slot_expert[(size_t) k];
                    if (e == -1) free.push_back({(int) gi, k, -1});
                    else if (e >= 0 && !top[(size_t) e] && !l.moving[(size_t) e]) out.push_back({(int) gi, k, e});
                }
            }
            std::stable_sort(out.begin(), out.end(),
                             [&](const Slot& a, const Slot& b) { return l.use[(size_t) a.e] < l.use[(size_t) b.e]; });
            size_t fi = 0, oi = 0;
            for (const int32_t e : in) {
                if (fi < free.size()) {
                    const Slot& f = free[fi++];
                    cand.push_back({l.use[(size_t) e], (int) li, f.gi, f.k, e, -1});
                    continue;
                }
                if (oi >= out.size() || l.use[(size_t) e] <= 1.25f * l.use[(size_t) out[oi].e] + 0.5f) break;
                const Slot& r = out[oi++];
                cand.push_back({l.use[(size_t) e] - l.use[(size_t) r.e], (int) li, r.gi, r.k, e, r.e});
            }
        }
        std::stable_sort(cand.begin(), cand.end(), [](const Planned& a, const Planned& b) { return a.gain > b.gain; });
        size_t waiting = 0;
        for (const Gpu& gp : gpus) waiting += gp.backlog.size() - gp.next;
        const size_t keep = std::max<size_t>((size_t) std::max(0, max_copies), waiting);
        if (cand.size() > keep) cand.resize(keep);
        for (Gpu& gp : gpus) {
            gp.backlog.clear();
            gp.next = 0;
        }
        for (const Planned& c : cand) gpus[(size_t) c.gi].backlog.push_back(c);
        if (halve)
            for (Layer& l : m->layers)
                for (float& u : l.use) u *= 0.5f;
        pump();
        return (int) cand.size();
    }
    /// The planned moves started while each GPU's copies in flight are under about a window's worth of its link
    /// (`kMoveWindowMs`), each checked against the slots as they are now: its resident out of the slot first.  Between
    /// forwards only (the slots it empties are not being read).
    static constexpr double kMoveWindowMs = 100.0;
    int pump() {
        std::vector<std::vector<char>> dirty;
        int started = 0;
        for (size_t gi = 0; gi < gpus.size(); ++gi) {
            Gpu& gp = gpus[gi];
            if (gi == 0 && restore_next < restore.size()) {   // the residents a lend displaced, ahead of the plan
                size_t blob = 0;
                for (const Layer& l : m->layers)
                    if (!l.dense) blob = std::max(blob, l.fmt.bytes);
                const size_t cap = std::max<size_t>(
                    8, (size_t) (gp.h2d_gbps * 1e9 * kMoveWindowMs / 1000.0 / (double) std::max<size_t>(blob, 1)));
                while (gp.moves.size() < cap && restore_next < restore.size()) {
                    const Planned c = restore[restore_next++];
                    Layer& l = m->layers[(size_t) c.li];
                    const SlotSet& s0 = l.held[0];
                    if (c.k >= s0.n || s0.slot_expert[(size_t) c.k] != -1 || l.moving[(size_t) c.e]) continue;
                    bool held = false;
                    for (const SlotSet& so : l.held) held = held || (so.n > 0 && so.slot_of[(size_t) c.e] >= 0);
                    if (held) continue;
                    issue(c.li, 0, c.k, c.e);
                    ++started;
                }
                if (restore_next >= restore.size()) {
                    restore.clear();
                    restore_next = 0;
                }
            }
            if (gp.next >= gp.backlog.size()) continue;
            size_t blob = 0;
            for (const Layer& l : m->layers)
                if (!l.dense) blob = std::max(blob, l.fmt.bytes);
            const size_t cap = std::max<size_t>(
                8, (size_t) (gp.h2d_gbps * 1e9 * kMoveWindowMs / 1000.0 / (double) std::max<size_t>(blob, 1)));
            while (gp.moves.size() < cap && gp.next < gp.backlog.size()) {
                const Planned c = gp.backlog[gp.next++];
                Layer& l = m->layers[(size_t) c.li];
                SlotSet& s = l.held[gi];
                if (c.k >= s.n || l.moving[(size_t) c.e]) continue;
                bool held = false;
                for (const SlotSet& so : l.held) held = held || (so.n > 0 && so.slot_of[(size_t) c.e] >= 0);
                if (held) continue;
                if (c.out >= 0) {
                    if (s.slot_expert[(size_t) c.k] != c.out || l.moving[(size_t) c.out]) continue;
                    s.slot_of[(size_t) c.out] = -1;
                    s.slot_expert[(size_t) c.k] = -1;
                    if (dirty.empty()) dirty.assign(m->layers.size(), std::vector<char>(gpus.size(), 0));
                    dirty[(size_t) c.li][gi] = 1;
                } else if (s.slot_expert[(size_t) c.k] != -1) {
                    continue;
                }
                issue(c.li, (int) gi, c.k, c.e);
                ++started;
            }
            if (gp.next >= gp.backlog.size()) {
                gp.backlog.clear();
                gp.next = 0;
            }
        }
        for (size_t li = 0; li < dirty.size(); ++li)
            for (size_t gi = 0; gi < gpus.size(); ++gi)
                if (dirty[li][gi]) upload_table(m->layers[li], (int) gi);
        return started;
    }
    /// Between forwards: the copies that have landed switched in, more started
    void step_moves() {
        settle(false);
        pump();
    }
    bool moves_waiting() const {
        if (restore_next < restore.size()) return true;
        for (const Gpu& gp : gpus)
            if (gp.next < gp.backlog.size()) return true;
        return false;
    }
    /// Every copy queued landed and switched in (adapt off: the placement as the requests leave it, not as the copies'
    /// timing does).  Between forwards only.
    void drain() {
        do {
            pump();
            settle(true);
        } while (moves_waiting());
    }
    /// Expert e's copy into slot k (empty) of GPU gi's slots of layer li, queued on that GPU's copy thread
    void issue(int li, int gi, int k, int32_t e) {
        Layer& l = m->layers[(size_t) li];
        Gpu& gp = gpus[(size_t) gi];
        SlotSet& s = l.held[(size_t) gi];
        s.slot_expert[(size_t) k] = kPending;
        l.moving[(size_t) e] = 1;
        if (gp.free_ev.empty()) {
            lamina::core::DeviceScope ds(gp.dev, main_dev, gi > 0);
            cudaEvent_t made = nullptr;
            made = event_new();
            gp.free_ev.push_back(made);
        }
        gp.moves.emplace_back();
        SlotMove& mv = gp.moves.back();
        mv.li = li;
        mv.gi = gi;
        mv.k = k;
        mv.e = e;
        mv.dst = s.slots + (size_t) k * l.stride;
        mv.src = l.blob[(size_t) e];
        mv.bytes = l.fmt.bytes;
        mv.pinned = l.blob_pinned[(size_t) e] != 0;
        mv.ev = gp.free_ev.back();
        gp.free_ev.pop_back();
        gp.mover->push(&mv);
    }
    /// The slot copies that have landed (`wait`: all of them, waited for) switched into their GPUs' tables: the expert
    /// into its new slot and out of a slot it had on another GPU.  Between forwards only.  Returns the copies switched.
    int64_t moves_landed = 0;   // copies switched in since the start
    int settle(bool wait) {
        std::vector<std::vector<char>> dirty;
        int n = 0;
        for (size_t gi = 0; gi < gpus.size(); ++gi) {
            Gpu& gp = gpus[gi];
            while (!gp.moves.empty()) {   // in order: a copy that has not landed has none after it that has
                SlotMove& mv = gp.moves.front();
                if (!mv.issued.load(std::memory_order_acquire)) {
                    if (!wait) break;
                    while (!mv.issued.load(std::memory_order_acquire)) std::this_thread::yield();
                }
                if (mv.ok.load()) {
                    lamina::core::DeviceScope ds(gp.dev, main_dev, gi > 0);
                    if (wait) {
                        CK(cudaEventSynchronize(mv.ev));
                    } else {
                        const cudaError_t qr = cudaEventQuery(mv.ev);
                        if (qr == cudaErrorNotReady) break;
                        CK(qr);
                    }
                }
                Layer& l = m->layers[(size_t) mv.li];
                if (dirty.empty()) dirty.assign(m->layers.size(), std::vector<char>(gpus.size(), 0));
                SlotSet& s = l.held[gi];
                if (mv.ok.load() && mv.k < s.n && s.slot_expert[(size_t) mv.k] == kPending) {
                    for (size_t oi = 0; oi < l.held.size(); ++oi) {   // out of another GPU's slot
                        SlotSet& so = l.held[oi];
                        if (oi == gi || so.slot_of.empty() || so.slot_of[(size_t) mv.e] < 0) continue;
                        so.slot_expert[(size_t) so.slot_of[(size_t) mv.e]] = -1;
                        so.slot_of[(size_t) mv.e] = -1;
                        dirty[(size_t) mv.li][oi] = 1;
                    }
                    s.slot_expert[(size_t) mv.k] = mv.e;
                    s.slot_of[(size_t) mv.e] = mv.k;
                    dirty[(size_t) mv.li][gi] = 1;
                    ++n;
                    ++moves_landed;
                } else if (mv.k < s.n && s.slot_expert[(size_t) mv.k] == kPending) {
                    s.slot_expert[(size_t) mv.k] = -1;   // a copy that failed: the slot empty again
                }
                l.moving[(size_t) mv.e] = 0;
                gp.free_ev.push_back(mv.ev);
                gp.moves.pop_front();
            }
        }
        for (size_t li = 0; li < dirty.size(); ++li)
            for (size_t gi = 0; gi < gpus.size(); ++gi)
                if (dirty[li][gi]) upload_table(m->layers[li], (int) gi);
        return n;
    }
    /// Each GPU's free VRAM, for the log ("GPU 1 812, GPU 0 1210, ..." MiB)
    std::string vram_free() {
        std::string out;
        for (const Gpu& gp : gpus) {
            lamina::core::DeviceScope ds(gp.dev, main_dev, gp.dev != main_dev);
            size_t fr = 0, tot = 0;
            if (cudaMemGetInfo(&fr, &tot) != cudaSuccess) continue;
            char b[48];
            std::snprintf(b, sizeof b, "%sGPU %d %zu", out.empty() ? "" : ", ", gp.dev, fr >> 20);
            out += b;
        }
        return out;
    }
    /// A tier's share of an MoE layer, on its GPU (current): the routing and the routed input from the host, its
    /// entries grouped and run, their rows into the mapped h_out (the next segment merges them as the CPU's)
    void tier_enqueue(Gpu& gp, int gi, Layer& l, int T, cudaEvent_t done = nullptr) {
        const auto& g = m->g;
        const int E = (int) g.n_expert, Kk = (int) g.top_k, Ld = (int) g.n_latent;
        SlotSet& s = l.held[(size_t) gi];
        CK(cudaMemcpyAsync(gp.ids_d, h_ids, (size_t) T * Kk * 4, cudaMemcpyHostToDevice, gp.st));
        CK(cudaMemcpyAsync(gp.rin_d, h_rin, (size_t) T * Ld * 4, cudaMemcpyHostToDevice, gp.st));
        K::quantize_q8_1_rows(gp.rin_d, T, Ld, gp.rinq_d, gp.st);
        group_slots(gp.ids_d, T * Kk, Kk, E, s.slot_of_d, s.slots, l.stride, gp.grp_ptr, gp.grp_start, gp.n_groups,
                    gp.ent_dst, gp.ent_tok, gp.st);
        K::native_expert_grouped(l.gl, gp.grp_ptr, gp.grp_start, gp.n_groups, gp.ent_dst, gp.ent_tok,
                                 std::min<int64_t>((int64_t) T * Kk, E), (int64_t) T * Kk, gp.rinq_d,
                                 (int64_t) (Ld / 32) * 36, gp.escr, gp.rows_d, gp.st);
        K::native_expert_rows_out(gp.rows_d, gp.n_groups, gp.grp_start, gp.ent_dst, Ld, (int64_t) T * Kk, gp.out_d,
                                  gp.flag_d, nullptr, gp.counter_d, gp.st);
        if (use_graphs) CK(cudaEventRecordWithFlags(done ? done : gp.ev, gp.st, cudaEventRecordExternal));
        else CK(cudaEventRecord(done ? done : gp.ev, gp.st));
    }
    // --tier-doorbell: the forward's segment whose MoE layer the host's part serves (-1: none, or the MTP block's)
    bool tier_doorbell = false;
    int door_seg = -1;
    /// Every tier's shares of a forward of T rows queued (a graph a tier and a row count), each behind its doorbell
    void door_launch(int n_seg, int T) {
        for (size_t gi = 1; gi < gpus.size(); ++gi) {
            Gpu& gp = gpus[gi];
            lamina::core::DeviceScope ds(gp.dev, main_dev);
            if (!gp.door) gp.door = hmapped<uint32_t>(128, &gp.door_d);
            CK(cudaStreamSynchronize(gp.st));   // the last forward's shares done: their doorbells down
            for (int s = 0; s < n_seg; ++s) reinterpret_cast<volatile uint32_t*>(gp.door)[s] = 0u;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            auto it = gp.door_graphs.find(T);
            if (it == gp.door_graphs.end()) {
                while ((int) gp.door_ev.size() < n_seg) {
                    cudaEvent_t e = nullptr;
                    e = event_new();
                    gp.door_ev.push_back(e);
                }
                CK(cudaStreamBeginCapture(gp.st, cudaStreamCaptureModeThreadLocal));
                for (int s = 1; s < n_seg; ++s) {
                    hy4k::k_wait_flag<<<1, 1, 0, gp.st>>>(gp.door_d + s, 10000000000ull);
                    tier_enqueue(gp, (int) gi, m->layers[(size_t) moe_layers[(size_t) s - 1]], T,
                                 gp.door_ev[(size_t) s]);
                }
                cudaGraphExec_t ex = graph_end(gp.st);
                it = gp.door_graphs.emplace(T, ex).first;
            }
            CK(cudaGraphLaunch(it->second, gp.st));
        }
    }
    void tier_launch(int gi, Layer& l, int T) {
        Gpu& gp = gpus[(size_t) gi];
        lamina::core::DeviceScope ds(gp.dev, main_dev);
        if (!use_graphs) {
            tier_enqueue(gp, gi, l, T);
            return;
        }
        const int key = l.il * 16 + T;
        auto it = gp.graphs.find(key);
        if (it == gp.graphs.end()) {
            CK(cudaStreamBeginCapture(gp.st, cudaStreamCaptureModeThreadLocal));
            tier_enqueue(gp, gi, l, T);
            cudaGraphExec_t ex = graph_end(gp.st);
            it = gp.graphs.emplace(key, ex).first;
        }
        CK(cudaGraphLaunch(it->second, gp.st));
    }
    // a profile: 'HY4P', n_layers, n_expert, then each MoE layer's shares of its routing (float, summing to 1)
    bool profile_loaded = false;
    std::string profile_from;
    /// The start: the learned profile when it holds this model's shape, else --expert-profile's.  Its shares rank the
    /// experts and count as 1,000 entries a layer of routing until the rebalances halve them.
    void load_profile() {
        const auto& g = m->g;
        auto read = [&](const std::string& path) {
            if (path.empty()) return false;
            FILE* f = std::fopen(path.c_str(), "rb");
            if (!f) return false;
            int32_t hdr[3] = {};
            bool ok = std::fread(hdr, 4, 3, f) == 3 && hdr[0] == 0x50345948 && hdr[1] == (int32_t) g.n_layers &&
                      hdr[2] == (int32_t) g.n_expert;
            for (Layer& l : m->layers) {
                if (l.dense || !ok) continue;
                l.start.assign((size_t) g.n_expert, 0.f);
                ok = std::fread(l.start.data(), 4, l.start.size(), f) == l.start.size();
                double sum = 0;
                for (const float v : l.start) sum += v > 0 ? v : 0;
                for (float& v : l.start) v = sum > 0 && v > 0 ? (float) (v / sum) : 0.f;
            }
            std::fclose(f);
            if (!ok)
                for (Layer& l : m->layers) l.start.clear();
            return ok;
        };
        if (read(save_path)) profile_from = save_path;
        else if (read(profile_path)) profile_from = profile_path;
        profile_loaded = !profile_from.empty();
        if (!profile_loaded) return;
        for (Layer& l : m->layers) {
            if (l.dense) continue;
            l.use.resize(l.start.size());
            for (size_t e = 0; e < l.start.size(); ++e) l.use[e] = 1000.f * l.start[e];
        }
    }
    /// The learned profile: each layer's share of the run's routing and the start's share at equal weight (the run's
    /// alone without a start); written once `save_every_min` minutes passed since the last (`force`: now)
    void save_profile(bool force) {
        if (save_path.empty()) return;
        if (!force && (save_every_min <= 0 || ms_since(last_save) < save_every_min * 60000.0)) return;
        last_save = Clock::now();
        const std::string tmp = save_path + ".tmp";
        FILE* f = std::fopen(tmp.c_str(), "wb");
        if (!f) return;
        const auto& g = m->g;
        const int32_t hdr[3] = {0x50345948, (int32_t) g.n_layers, (int32_t) g.n_expert};
        bool ok = std::fwrite(hdr, 4, 3, f) == 3;
        std::vector<float> share((size_t) g.n_expert);
        for (const Layer& l : m->layers) {
            if (l.dense) continue;
            double sum = 0;
            for (const float v : l.total) sum += v;
            const bool run = sum > 0, base = !l.start.empty();
            for (size_t e = 0; e < share.size(); ++e) {
                const double r = run ? l.total[e] / sum : 0.0, b = base ? l.start[e] : 0.0;
                share[e] = (float) (run && base ? 0.5 * (r + b) : run ? r : b);
            }
            ok = ok && std::fwrite(share.data(), 4, share.size(), f) == share.size();
        }
        ok = std::fclose(f) == 0 && ok;
        if (ok) {
            std::remove(save_path.c_str());
            std::rename(tmp.c_str(), save_path.c_str());
        }
    }

    // --prefill-lend: an answer that has run lend_after tokens lends the prompt path's buffers (lend_bytes) to the main
    // GPU's slots, which may then grow lend_slots past s_max; the next prompt part takes them back (base_fn)
    int64_t lend_after = 0;
    size_t lend_bytes = 0;
    int lend_slots = 0;
    bool base_lent = false;
    std::function<void(bool)> base_fn;   // true: give (an answer's tokens), false: take (a prompt part)
    void maybe_lend(int64_t produced) {
        if (lend_after > 0 && !base_lent && produced >= lend_after && base_fn) base_fn(true);
    }
    // --warm-ms: a request after a pause keeps every GPU running that long from its start (warm_up), on a stream a GPU
    int warm_ms = 500;
    std::vector<cudaStream_t> warm_st;
    Clock::time_point last_end{};
    bool served = false;
    int64_t warmed = 0;
    /// The warm-up's streams (the lowest priority, a GPU), made at the start: a request does not pay for them
    void warm_streams() {
        if (warm_ms <= 0) return;
        warm_st.assign(gpus.size(), nullptr);
        for (size_t gi = 0; gi < gpus.size(); ++gi) {
            lamina::core::DeviceScope ds(gpus[gi].dev, main_dev, gi > 0);
            int lo = 0, hi = 0;
            CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
            warm_st[gi] = stream_new(cudaStreamNonBlocking, lo);
            hy4k::k_keep<<<1, 32, 0, warm_st[gi]>>>(1000ull);   // its first launch's setup now too
        }
    }
    void warm_up() {
        if (warm_ms <= 0 || warm_st.size() < gpus.size()) return;
        if (served && ms_since(last_end) < 1000.0) return;   // back to back: at speed already
        for (size_t gi = 0; gi < gpus.size(); ++gi) {
            lamina::core::DeviceScope ds(gpus[gi].dev, main_dev, gi > 0);
            hy4k::k_keep<<<1, 32, 0, warm_st[gi]>>>((unsigned long long) warm_ms * 1000000ull);
        }
        ++warmed;
    }
    /// row r's logits on the host
    void host_logits(int r, std::vector<float>& out) {
        const auto& g = m->g;
        const auto t0 = Clock::now();
        out.resize((size_t) g.n_vocab);
        CK(cudaMemcpyAsync(out.data(), logits + (size_t) r * g.n_vocab, (size_t) g.n_vocab * 4, cudaMemcpyDeviceToHost,
                           st));
        CK(cudaStreamSynchronize(st));
        if (trace) tr.logits += ms_since(t0);
    }
};

int argmax(const std::vector<float>& v) { return (int) (std::max_element(v.begin(), v.end()) - v.begin()); }

// ------------------------------------------------------------------------------------------------ the prompt path
namespace MMQ = lamina::prefill::mmq;
constexpr size_t kQ8Block = 144;   // bytes of a q8_1 block of MMQ's layout (128 values)

/// A GPU's share of a prompt chunk's routed experts: the entries of the experts in its slots and, on the streaming GPU,
/// of the experts no GPU holds (copied from the arena into two staging buffers in turn); their rows summed per token
/// into `acc`, which a tier hands to the main GPU through pinned memory.
struct MoeDev {
    int gi = 0, dev = 0;
    cudaStream_t st = nullptr, cst = nullptr;
    cudaEvent_t done = nullptr, copied[2] = {}, used[2] = {};
    std::unique_ptr<MMQ::Context> mmq;
    uint8_t *tq = nullptr, *eq = nullptr, *hq = nullptr;
    float *gu = nullptr, *h = nullptr, *rows = nullptr, *acc = nullptr;
    int32_t *ident = nullptr, *plan = nullptr, *plan_h = nullptr;
    int64_t ident_rows = 0;
    float* acc_h = nullptr;
    __half* acc16 = nullptr;   // a tier's sums as halves (Prompt::sum_f16)
    // the second half's (Prompt::run's halves): its plan's host copy, its sums' end on a tier
    int32_t* plan_h2 = nullptr;
    cudaEvent_t done2 = nullptr;
    uint8_t* stage[2] = {};
};

/// The prompt path: a chunk of up to `cap` rows a forward.  The main GPU runs the projections on llama.cpp's MMQ (the
/// Q8_0 weights read once a chunk), the attention a head group of 32 at a time on the tensor cores (the window layers
/// over their cells in position order, the sparse ones over the indexer's pick, its scores on the tensor cores), the
/// router and the shared expert; the routed experts run on every GPU that holds some (MMQ over its slots), the ones no
/// GPU holds on the GPU with the fastest link, copied there from the arena.  The state it leaves is the decode's.
struct Prompt {
    Engine& E;
    Model* m = nullptr;
    int cap = 0;    // rows a chunk's dense steps take at once (a sub-chunk of a pass)
    int pass = 0;   // rows the MoE takes at once (--prefill-pass; cap when off): a chunk's rows
    // a sub-chunk's views of the pass's buffers (front, attend, indexer, dense_ffn): its streams, post weights,
    // positions, Engram rows (eng_T the pass's rows: a module's stride) and their valid flags
    float *Xc = nullptr, *postc = nullptr, *evalidc = nullptr;
    int32_t* posc = nullptr;
    uint8_t* erawc = nullptr;
    int eng_T = 0, hids_T = -1;
    // the main GPU's buffers a chunk's MoE and streams take whole: --prefill's rows (base, kept), the pass's (lent:
    // allocated by pass_begin for a part, freed by pass_end); the running set in the members below
    struct Bufs {
        float *X = nullptr, *y = nullptr, *sy = nullptr, *post = nullptr, *evalid = nullptr, *wts = nullptr,
              *rin = nullptr, *tmp = nullptr, *acc = nullptr;
        uint8_t *eraw = nullptr, *mq = nullptr, *tq = nullptr;
        int32_t *ids = nullptr, *tok_d = nullptr, *pos_d = nullptr;
    };
    Bufs base, lent;
    // fr: the prompt path's own buffers on the main GPU, one VMM range (lend_base: --prefill-lend), planned by lalloc
    // while init runs, made by base_commit; base_give lends its
    // chunks to the main GPU's slots, base_take maps them back
    bool lend_base = false;
    std::vector<std::pair<void**, size_t>> lend_plan;
    std::unique_ptr<lamina::core::VmmRange> base_vm;
    bool base_mapped = true;
    template <class T> void lalloc(T*& p, size_t n) {
        if (!lend_base) {
            p = dalloc<T>(n);
            return;
        }
        lend_plan.emplace_back(reinterpret_cast<void**>(&p), (std::max<size_t>(n * sizeof(T), 256) + 511) / 512 * 512);
    }
    void base_commit() {
        if (!lend_base) return;
        size_t total = 0;
        for (const auto& [pp, z] : lend_plan) total += z;
        base_vm = std::make_unique<lamina::core::VmmRange>();
        if (!base_vm->reserve(total) ||
            !base_vm->map_range(0, base_vm->chunks(), [] { return (lamina::core::VmmChunk) 0; })) {
            std::fprintf(stderr, "lamina: the prompt path's buffers could not be mapped\n");
            throw Fatal{1};
        }
        size_t at = 0;
        for (const auto& [pp, z] : lend_plan) {
            *pp = base_vm->base() + at;
            at += z;
        }
        E.lend_bytes = (size_t) base_vm->chunks() * lamina::core::vmm_granularity();
    }
    /// Between prompts (an answer N tokens long): the buffers' chunks to the main GPU's slots, which grow into them
    void base_give() {
        if (!base_vm || !base_mapped) return;
        CK(cudaStreamSynchronize(st));
        for (int64_t c = 0; c < base_vm->chunks(); ++c)
            if (base_vm->mapped(c)) E.spare.push_back(base_vm->unmap(c));
        base_mapped = false;
        E.base_lent = true;
        E.slots_back();
        if (E.adapt)
            E.plan_moves(512, false);   // the new slots filled by the ranking now, not 32 copies per 128 tokens
    }
    /// A prompt part: the buffers mapped again, the slots past what is left giving way
    void base_take() {
        if (!base_vm || base_mapped) return;
        const int64_t n = base_vm->chunks();
        E.base_lent = false;
        if (!E.lend_chunks(n)) {
            std::fprintf(stderr, "lamina: the prompt path's buffers could not be taken back from the slots\n");
            throw Fatal{1};
        }
        auto take = [&]() -> lamina::core::VmmChunk {
            if (E.spare.empty()) return 0;
            const lamina::core::VmmChunk h = E.spare.back();
            E.spare.pop_back();
            return h;
        };
        if (!base_vm->map_range(0, n, take)) {
            std::fprintf(stderr, "lamina: the prompt path's buffers could not be mapped again\n");
            throw Fatal{1};
        }
        base_mapped = true;
    }
    bool lent_live = false;
    int rows_now = 0;                         // the running set's rows
    int64_t pass_p0 = 0;                      // the running pass's first position
    int min_rows = 32;                        // a shorter part goes through the decode's forwards
    static constexpr int kG = hy4p::kQRows;   // heads a pass
    static constexpr int kEmax = 4096;        // a group's entries at most
    static constexpr int kStage = 32;         // experts a staging buffer
    static constexpr int kSub = 512;          // rows a pass of Engram's and the dense FFN's products
    int R = 0;                                // the indexer's rows a pass
    int streamer = 0;                         // the devs index that runs the experts no GPU holds
    uint8_t* gsc = nullptr;                   // the router product's scratch (a K split's
                                              // partial sums: few rows fill the card that way)
    size_t gsc_bytes = 0;
    cudaStream_t st = nullptr;
    cudaEvent_t routed = nullptr, routed2 = nullptr;   // a half's routing on the host (routed2: the second half's)
    // the main GPU's rows
    float *X = nullptr, *xh = nullptr, *xn = nullptr, *y = nullptr, *sy = nullptr, *emb = nullptr, *post = nullptr;
    uint8_t *xq = nullptr, *qrq = nullptr, *hdq = nullptr, *oq = nullptr, *ikq = nullptr, *subq = nullptr,
            *fhq = nullptr, *shq = nullptr, *mq = nullptr, *eraw = nullptr;
    float *qa = nullptr, *kva = nullptr, *qg = nullptr, *ql = nullptr, *gt = nullptr;
    float *iq = nullptr, *ik = nullptr, *iv = nullptr, *ig = nullptr, *iw = nullptr, *scores = nullptr;
    float *erows = nullptr, *ekv = nullptr, *evalid = nullptr, *fg = nullptr, *fu = nullptr, *fh = nullptr;
    float *logits_r = nullptr, *wts = nullptr, *rin = nullptr, *sg = nullptr, *su = nullptr, *sh = nullptr,
          *tmp = nullptr;
    // the tiers' sums cross the links as halves (LAMINA_HY4_SUM_F16=0: floats): half the bytes, an error well under
    // the q8_1 rounding of routed_up's input
    bool sum_f16 = true;
    // a chunk of two sub-chunks or more in two halves overlapped (LAMINA_HY4_HALVES=0: its MoE in one piece)
    bool halves_on = true;
    int32_t *ids = nullptr, *cells = nullptr, *ncells = nullptr, *hids = nullptr, *hb = nullptr, *tok_d = nullptr,
            *pos_d = nullptr;
    int64_t hids_rows = 0;
    __half* lin = nullptr;
    // the host's (pinned)
    int32_t *h_ids = nullptr, *h_tok = nullptr, *h_pos = nullptr;
    float *h_wts = nullptr, *h_valid = nullptr;
    uint8_t *tq_h = nullptr, *tq_h2 = nullptr, *h_raw = nullptr;   // tq_h2: the second half's tiers' input
    uint32_t* h_rows = nullptr;
    std::vector<MoeDev> devs;   // [0] the main GPU, then the tiers in Engine::gpus' order
    // the snapshots inside a chunk (Engine::chunk_snaps, one buffer each): their bytes on the main GPU as snapshot()
    // lays them out, each window layer's ring at ring_off[il], each indexer layer's tail at tail_off[il] (of the cache
    // layers, the MTP block's at [n_layers]), the drafters' ring at hring_off
    std::vector<uint8_t*> snap_devs;
    bool snap_dev_failed = false;   // (more_snap_dev's note given)
    std::vector<size_t> ring_off, tail_off;
    size_t hring_off = 0, snap_size = 0;
    size_t stage_bytes = 0;
    bool ok = false;
    double ms_moe = 0, ms_copy = 0;
    // LAMINA_HY4_PP_TRACE=1: every GPU waited for after each phase, the phases' sums printed after a chunk
    const bool trace = std::getenv("LAMINA_HY4_PP_TRACE") != nullptr;
    double ph[6] = {};
    Clock::time_point ph_t;
    void mark(int k) {
        if (!trace) return;
        for (const MoeDev& d : devs) {
            lamina::core::DeviceScope ds(d.dev, main_dev, d.gi > 0);
            CK(cudaDeviceSynchronize());
        }
        ph[k] += ms_since(ph_t);
        ph_t = Clock::now();
    }

    explicit Prompt(Engine& e) : E(e), m(e.m) {}

    /// The buffers of a chunk of `rows` on the main GPU and on each of `tier_devs` (before their slots take the rest);
    /// false (with a note) where MMQ cannot run one of the model's products.
    bool init(int rows, const std::vector<int>& tier_devs, int pass_rows = 0) {
        const auto& g = m->g;
        if (const char* v = std::getenv("LAMINA_HY4_SUM_F16")) sum_f16 = std::atoi(v) != 0;
        if (const char* v = std::getenv("LAMINA_HY4_HALVES")) halves_on = std::atoi(v) != 0;
        lend_base = E.lend_after > 0 && E.elastic;
        cap = std::max(kMaxT + 1, rows);
        pass = std::max(cap, pass_rows);
        std::vector<const Layer*> checked;
        for (const Layer& l : m->layers) checked.push_back(&l);
        if (m->has_mtp) checked.push_back(&m->mtp);
        if (m->has_mtp && !MMQ::fits(8, m->eh_proj.n_out)) {
            std::fprintf(stderr, "lamina: no MMQ for the MTP block's eh_proj: prompts read %d rows a forward\n", kMaxT);
            return false;
        }
        for (const Layer* lp : checked) {
            const Layer& l = *lp;
            // every product's (type, rows) as the chunk runs it: the head-group slices, the per-head matrices
            std::vector<std::pair<int, int64_t>> need{{l.q_a.type, l.q_a.n_out},    {l.q_b.type, kG * g.k_head},
                                                      {l.kv_a.type, l.kv_a.n_out},  {l.wo.type, l.wo.n_out},
                                                      {l.gate.type, kG * g.v_head}, {l.k_b.type, hy4k::kLatent},
                                                      {l.v_b.type, g.v_head}};
            for (const Mat* w : {&l.iq_b, &l.ik, &l.icv, &l.icg, &l.e_kv, &l.f_gate, &l.f_up, &l.f_down, &l.r_down,
                                 &l.r_up, &l.s_gate, &l.s_up, &l.s_down})
                if (w->d) need.emplace_back(w->type, w->n_out);
            for (const auto& [t, r] : need)
                if (!MMQ::fits(t, r)) {
                    std::fprintf(stderr,
                                 "lamina: no MMQ for a [%lld rows] product of type %d: prompts read %d rows a "
                                 "forward\n",
                                 (long long) r, t, kMaxT);
                    return false;
                }
            if (!l.dense && (!MMQ::fits(l.fmt.gu_type, 2 * g.n_ff_exp) || !MMQ::fits(l.fmt.d_type, g.n_latent))) {
                std::fprintf(stderr,
                             "lamina: no MMQ for the routed experts' types (%d, %d): prompts read %d rows a "
                             "forward\n",
                             l.fmt.gu_type, l.fmt.d_type, kMaxT);
                return false;
            }
        }
        CK(cudaGetDevice(&main_dev));
        st = E.st;
        for (cudaEvent_t* e : {&routed, &routed2}) {
            *e = event_new();
        }
        const size_t T = (size_t) cap, N = (size_t) g.n_embd, hc = (size_t) g.hc, Ld = (size_t) g.n_latent,
                     F = (size_t) g.n_ff_exp, Ex = (size_t) g.n_expert, K = (size_t) g.top_k;
        const size_t PR = (size_t) pass;   // a pass's rows: the streams' buffers and the MoE pass's
        // a tier's rows at once: a half's (Prompt::run's halves, one after the other on its stream), else the pass's
        const size_t TR =
            pass > cap && halves_on && !trace ? (size_t) (((pass + cap - 1) / cap + 1) / 2) * (size_t) cap : PR;
        size_t hmax = 0;
        for (const Layer& l : m->layers) hmax = std::max<size_t>(hmax, (size_t) l.H);
        lalloc(X, T * hc * N);
        lalloc(xh, T * N);
        lalloc(xn, T * N);
        lalloc(y, T * N);
        lalloc(sy, T * N);
        lalloc(emb, T * N);
        lalloc(post, T * hc);
        lalloc(xq, MMQ::q8_bytes((int64_t) T, (int64_t) N));
        lalloc(qa, T * (size_t) g.q_lora);
        lalloc(qrq, MMQ::q8_bytes((int64_t) T, g.q_lora));
        lalloc(kva, T * (size_t) g.kv_row());
        lalloc(qg, T * kG * (size_t) g.k_head);
        lalloc(ql, T * kG * hy4k::kLatent);
        lalloc(gt, T * kG * (size_t) g.v_head);
        lalloc(hdq, MMQ::q8_bytes((int64_t) (kG * T), hy4k::kLatent));
        lalloc(oq, MMQ::q8_bytes((int64_t) T, (int64_t) (hmax * g.v_head)));
        hids_rows = MMQ::ids_table_rows((int64_t) (kG * T), (int64_t) T);
        hids = dalloc<int32_t>((size_t) hids_rows);
        CK(cudaMemset(hids, 0, (size_t) hids_rows * 4));
        hb = dalloc<int32_t>(kG + 1);
        lalloc(lin, ((size_t) g.n_swa + T) * hy4k::kKvRow);
        const size_t ih = (size_t) g.idx_heads, idd = (size_t) g.idx_dim, ld = (size_t) (E.ctx / g.idx_pool);
        lalloc(iq, T * ih * idd);
        lalloc(ik, T * idd);
        lalloc(ikq, MMQ::q8_bytes((int64_t) T, (int64_t) idd));
        lalloc(iv, T * idd);
        lalloc(ig, T * idd);
        lalloc(iw, T * ih);
        R = (int) std::max<size_t>(8, std::min<size_t>(T, ((size_t) 128 << 20) / (ld * 4)));
        lalloc(scores, (size_t) R * ld);
        lalloc(cells, T * (size_t) g.idx_top_k);
        lalloc(ncells, T);
        const size_t n_mod = g.eng_layers.size(), eh = (size_t) g.eng_heads(), ed = (size_t) g.eng_dim;
        lalloc(eraw, n_mod * T * eh * ed);
        lalloc(erows, T * eh * ed);
        lalloc(evalid, T);
        const size_t sub = std::min<size_t>(T, kSub);
        size_t widest = std::max<size_t>(eh * ed, N);
        lalloc(subq, MMQ::q8_bytes((int64_t) sub, (int64_t) std::max<size_t>(widest, (size_t) g.n_ff)));
        lalloc(ekv, sub * (hc + 1) * N);
        lalloc(fg, sub * (size_t) g.n_ff);
        lalloc(fu, sub * (size_t) g.n_ff);
        lalloc(fh, sub * (size_t) g.n_ff);
        lalloc(logits_r, T * Ex);
        gsc_bytes = std::min<size_t>(lamina::kernels::gemm_scratch_bytes((int64_t) T, (int64_t) Ex), (size_t) 8 << 20);
        lalloc(gsc, gsc_bytes);
        lalloc(ids, T * K);
        lalloc(wts, T * K);
        lalloc(rin, T * Ld);
        lalloc(sg, T * F);
        lalloc(su, T * F);
        lalloc(sh, T * F);
        lalloc(shq, MMQ::q8_bytes((int64_t) T, (int64_t) F));
        lalloc(mq, MMQ::q8_bytes((int64_t) T, (int64_t) Ld));
        lalloc(tmp, T * Ld);
        lalloc(tok_d, T);
        lalloc(pos_d, T);
        h_ids = halloc<int32_t>(PR * K);
        h_wts = halloc<float>(PR * K);
        h_tok = halloc<int32_t>(PR);
        h_pos = halloc<int32_t>(PR);
        h_valid = halloc<float>(PR);
        h_rows = halloc<uint32_t>(n_mod * PR * eh);
        h_raw = halloc<uint8_t>(n_mod * PR * eh * ed);
        tq_h = halloc<uint8_t>(MMQ::q8_bytes((int64_t) PR, (int64_t) Ld));
        if (pass > cap) tq_h2 = halloc<uint8_t>(MMQ::q8_bytes((int64_t) PR, (int64_t) Ld));
        for (const Layer& l : m->layers) stage_bytes = std::max(stage_bytes, l.stride);
        {
            ring_off.assign((size_t) g.n_layers + 1, 0);
            tail_off.assign((size_t) g.n_layers + 1, 0);
            size_t at = 0;
            for (const Layer& l : E.cl) {
                const size_t li = l.mtp ? (size_t) g.n_layers : (size_t) l.il;
                if (l.swa) {
                    ring_off[li] = at;
                    at += (size_t) l.kv_cells * l.kv_row();
                }
                if (l.has_idx) {
                    tail_off[li] = at;
                    at += (size_t) kTail * 2 * (size_t) g.idx_dim * 4;
                }
            }
            hring_off = at;
            if (E.hring) at += (size_t) Engine::kHRing * (size_t) g.n_embd * 4;
            snap_size = at;
            if (snap_size != E.snap_bytes()) {
                std::fprintf(stderr, "lamina: the snapshot's layout differs from the engine's\n");
                return false;
            }
            // the multiples of kSnapEvery a chunk can hold inside it
            for (int64_t k = 0; k < ((int64_t) PR - 1) / Engine::kSnapEvery + 1; ++k)
                snap_devs.push_back(dalloc<uint8_t>(snap_size));
        }
        base_commit();
        // the GPUs' shares: the main GPU, then the tiers
        std::vector<int> all{main_dev};
        all.insert(all.end(), tier_devs.begin(), tier_devs.end());
        const size_t plan_n = PR * K * 6 + 16384;
        for (size_t i = 0; i < all.size(); ++i) {
            lamina::core::DeviceScope ds(all[i], main_dev, i > 0);
            MoeDev d;
            d.gi = (int) i;
            d.dev = all[i];
            if (i == 0) {
                d.st = st;
            } else {
                d.st = stream_new();
            }
            d.cst = stream_new();
            for (cudaEvent_t* e : {&d.done, &d.done2, &d.copied[0], &d.copied[1], &d.used[0], &d.used[1]}) {
                *e = event_new();
            }
            CK(cudaEventRecord(d.used[0], d.st));
            CK(cudaEventRecord(d.used[1], d.st));
            d.mmq = std::make_unique<MMQ::Context>();
            d.tq = dalloc<uint8_t>(MMQ::q8_bytes((int64_t) (i == 0 ? T : TR), (int64_t) Ld));
            d.eq = dalloc<uint8_t>(MMQ::q8_bytes(kEmax, (int64_t) Ld));
            d.gu = dalloc<float>((size_t) kEmax * 2 * F);
            d.h = dalloc<float>((size_t) kEmax * F);
            d.hq = dalloc<uint8_t>(MMQ::q8_bytes(kEmax, (int64_t) F));
            d.rows = dalloc<float>((size_t) kEmax * Ld);
            d.acc = dalloc<float>((i == 0 ? T : TR) * Ld);
            d.ident_rows = MMQ::ids_table_rows(kEmax);
            d.ident = dalloc<int32_t>((size_t) d.ident_rows);
            MMQ::iota(d.ident, d.ident_rows, d.st);
            d.plan = dalloc<int32_t>(plan_n);
            d.plan_h = halloc<int32_t>(plan_n);
            if (pass > cap) d.plan_h2 = halloc<int32_t>(plan_n);
            if (i > 0) d.acc_h = halloc<float>(PR * Ld);
            if (i > 0 && sum_f16) d.acc16 = dalloc<__half>(TR * Ld);
            devs.push_back(std::move(d));
        }
        // the experts no GPU holds go to the GPU whose link copies fastest (core::probe_gpus: the links woken first)
        double best = 0;
        if (devs.size() > 1) {
            std::vector<int> ords;
            for (const MoeDev& d : devs) ords.push_back(d.dev);
            const std::vector<lamina::core::GpuProbe> pr = lamina::core::probe_gpus(ords, main_dev);
            for (size_t i = 0; i < pr.size(); ++i)
                if (pr[i].h2d_gbps > best) {
                    best = pr[i].h2d_gbps;
                    streamer = (int) i;
                }
        }
        {
            lamina::core::DeviceScope ds(devs[(size_t) streamer].dev, main_dev, streamer > 0);
            for (uint8_t*& s : devs[(size_t) streamer].stage) s = dalloc<uint8_t>((size_t) kStage * stage_bytes + 4096);
        }
        for (const MoeDev& d : devs) {
            lamina::core::DeviceScope ds(d.dev, main_dev, d.gi > 0);
            CK(cudaDeviceSynchronize());
        }
        CK(cudaFuncSetAttribute(hy4p::k_attn_prompt<true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                (int) sizeof(hy4p::AttnSmem)));
        CK(cudaFuncSetAttribute(hy4p::k_attn_prompt<true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                (int) sizeof(hy4p::AttnSmem)));
        CK(cudaFuncSetAttribute(hy4p::k_attn_prompt<false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                (int) sizeof(hy4p::AttnSmem)));
        base = take();
        rows_now = cap;
        if (pass > cap && E.elastic) {
            lent_vm = std::make_unique<lamina::core::VmmRange>();
            if (!lent_vm->reserve(lent_total())) lent_vm.reset();
        }
        if (pass > cap && !lent_vm) pass = cap;   // no VMM: no passes
        char lend_note[192] = "";
        if (pass > cap)
            std::snprintf(lend_note, sizeof lend_note,
                          " (a part of two or more: of up to %d, the experts' pass over them "
                          "all, %.2f GiB of the main GPU's slots lent to it)",
                          pass, (double) lent_total() / (double) (1ull << 30));
        std::fprintf(stderr, "lamina: prompts in chunks of up to %d rows%s; the experts no GPU holds on GPU %d%s\n",
                     cap, lend_note, devs[(size_t) streamer].dev,
                     devs.size() > 1 ? (" (" + std::to_string((int) best) + " GB/s from pinned memory)").c_str() : "");
        ok = true;
        return true;
    }
    int main_dev = 0;
    std::unique_ptr<lamina::core::VmmRange> lent_vm;   // the lent set's addresses (mapped while a part runs in passes)
    double lend_ms = 0;

    Bufs take() const {
        Bufs b;
        b.X = X;
        b.y = y;
        b.sy = sy;
        b.post = post;
        b.evalid = evalid;
        b.wts = wts;
        b.rin = rin;
        b.tmp = tmp;
        b.acc = devs[0].acc;
        b.eraw = eraw;
        b.mq = mq;
        b.tq = devs[0].tq;
        b.ids = ids;
        b.tok_d = tok_d;
        b.pos_d = pos_d;
        return b;
    }
    void use(const Bufs& b, int rows) {
        X = b.X;
        y = b.y;
        sy = b.sy;
        post = b.post;
        evalid = b.evalid;
        wts = b.wts;
        rin = b.rin;
        tmp = b.tmp;
        devs[0].acc = b.acc;
        eraw = b.eraw;
        mq = b.mq;
        devs[0].tq = b.tq;
        ids = b.ids;
        tok_d = b.tok_d;
        pos_d = b.pos_d;
        rows_now = rows;
    }
    /// The lent set's buffers' bytes, in Bufs' order (each rounded to 512 in the allocation)
    std::vector<size_t> lent_sizes() const {
        const auto& g = m->g;
        const size_t Rp = (size_t) pass, N = (size_t) g.n_embd, hc = (size_t) g.hc, Ld = (size_t) g.n_latent,
                     K = (size_t) g.top_k, n_mod = g.eng_layers.size(), eh = (size_t) g.eng_heads(),
                     ed = (size_t) g.eng_dim;
        return {Rp * hc * N * 4,
                Rp * N * 4,
                Rp * N * 4,
                Rp * hc * 4,
                Rp * 4,
                Rp * K * 4,
                Rp * Ld * 4,
                Rp * Ld * 4,
                Rp * Ld * 4,
                n_mod * Rp * eh * ed,
                MMQ::q8_bytes((int64_t) Rp, (int64_t) Ld),
                MMQ::q8_bytes((int64_t) Rp, (int64_t) Ld),
                Rp * K * 4,
                Rp * 4,
                Rp * 4};
    }
    size_t lent_total() const {
        size_t t = 0;
        for (const size_t z : lent_sizes()) t += (z + 511) / 512 * 512;
        return t;
    }
    /// The pass's buffers for a part, mapped from the chunks the main GPU's last slots give up (Engine::lend_chunks)
    /// and its free VRAM; false: no VMM or too few slots (the part runs in --prefill's chunks)
    bool pass_begin() {
        if (pass <= cap || !lent_vm) return false;
        if (lent_live) return true;
        const auto t0 = Clock::now();
        const int64_t n = lent_vm->chunks();
        if (!E.lend_chunks(n)) return false;
        auto take = [&]() -> lamina::core::VmmChunk {
            if (E.spare.empty()) return 0;
            const lamina::core::VmmChunk h = E.spare.back();
            E.spare.pop_back();
            return h;
        };
        if (!lent_vm->map_range(0, n, take)) {
            for (int64_t c = 0; c < n; ++c)
                if (lent_vm->mapped(c)) E.spare.push_back(lent_vm->unmap(c));
            E.slots_back();
            return false;
        }
        const std::vector<size_t> z = lent_sizes();
        size_t at = 0;
        auto carve = [&](size_t i) {
            uint8_t* q = lent_vm->base() + at;
            at += (z[i] + 511) / 512 * 512;
            return q;
        };
        lent.X = (float*) carve(0);
        lent.y = (float*) carve(1);
        lent.sy = (float*) carve(2);
        lent.post = (float*) carve(3);
        lent.evalid = (float*) carve(4);
        lent.wts = (float*) carve(5);
        lent.rin = (float*) carve(6);
        lent.tmp = (float*) carve(7);
        lent.acc = (float*) carve(8);
        lent.eraw = carve(9);
        lent.mq = carve(10);
        lent.tq = carve(11);
        lent.ids = (int32_t*) carve(12);
        lent.tok_d = (int32_t*) carve(13);
        lent.pos_d = (int32_t*) carve(14);
        use(lent, pass);
        lent_live = true;
        lend_ms = ms_since(t0);
        return true;
    }
    /// The pass's buffers' chunks back to the main GPU's slots
    /// after a part (they come back empty, the mover fills them)
    void pass_end() {
        if (!lent_live) return;
        CK(cudaStreamSynchronize(st));
        use(base, cap);
        const auto t0 = Clock::now();
        for (int64_t c = 0; c < lent_vm->chunks(); ++c)
            if (lent_vm->mapped(c)) E.spare.push_back(lent_vm->unmap(c));
        lent_live = false;
        E.slots_back();
        const int q = E.queue_restore();
        std::fprintf(stderr,
                     "lamina: the prompt's pass buffers %.2f GiB lent by the main GPU's slots (%.0f ms, back %.0f "
                     "ms, %d residents queued back)\n",
                     (double) lent_total() / (double) (1ull << 30), lend_ms, ms_since(t0), q);
    }

    unsigned blocks(int64_t n) const { return (unsigned) ((n + hy4k::kThreads - 1) / hy4k::kThreads); }
    void dense(const Mat& w, const uint8_t* x, int T, float* out, int64_t rows = 0, size_t row0 = 0, int64_t ld = 0) {
        const int64_t r = rows ? rows : w.n_out;
        const size_t row_bytes = (size_t) w.n_in / 32 * 34;
        devs[0].mmq->run_dense(w.d + row0 * row_bytes, w.type, r, w.n_in, x, T, out, ld ? ld : r, st);
    }
    void q8(const float* x, int64_t rows, int64_t cols, uint8_t* out, int type = 8, const int32_t* map = nullptr,
            int64_t ld = 0) {
        MMQ::quantize(x, map, out, type, cols, ld ? ld : cols, rows, st);
    }

    /// A chunk that holds wanted snapshots beside its multiples of kSnapEvery: their buffers more, of the VRAM the
    /// slots leave (the GPU's plan stays as it is; a buffer kept for the chunks after); one with no buffer is not taken
    void more_snap_devs() {
        auto& v = E.chunk_snaps;
        while (v.size() > snap_devs.size()) {
            void* b = nullptr;
            if (soft(dmalloc(&b, snap_size))) {
                snap_devs.push_back(static_cast<uint8_t*>(b));
                continue;
            }
            const auto w = std::find_if(v.rbegin(), v.rend(), [](int64_t p) { return p % Engine::kSnapEvery != 0; });
            if (w == v.rend()) break;   // (init() made the multiples' buffers)
            v.erase(std::next(w).base());
            if (!snap_dev_failed)
                std::fprintf(stderr, "lamina: no VRAM for a snapshot inside a chunk (%.1f MiB): not taken there\n",
                             (double) snap_size / (double) (1 << 20));
            snap_dev_failed = true;
        }
    }

    /// T rows (tokens toks at positions pos() ..) through the
    /// model; the last row's logits into the engine's logits row 0.
    void run(const int32_t* toks, int T) {
        const auto& g = m->g;
        if (T > rows_now) {
            std::fprintf(stderr, "lamina: a prompt chunk of %d rows, its buffers %d\n", T, rows_now);
            throw Fatal{1};
        }
        const int N = (int) g.n_embd, hc = (int) g.hc;
        if (E.chunk_snaps.size() > snap_devs.size()) more_snap_devs();
        E.ensure(E.pos() + T);
        const int64_t p0 = E.pos();
        for (int t = 0; t < T; ++t) E.seq.push_back(toks[t]);
        const auto tt = Clock::now();
        const size_t n_mod = g.eng_layers.size(), eh = (size_t) g.eng_heads(), ed = (size_t) g.eng_dim;
        std::vector<uint32_t> r(n_mod * eh);
        for (int t = 0; t < T; ++t) {
            engram_ids(g, E.seq, p0 + t, r.data(), h_valid[t]);
            for (size_t e = 0; e < n_mod; ++e)
                std::memcpy(h_rows + (e * (size_t) T + (size_t) t) * eh, r.data() + e * eh, eh * 4);
            h_pos[t] = (int32_t) (p0 + t);
            h_tok[t] = toks[t];
        }
        const lamina::ngram::PleReader::Ticket ticket = E.reader.issue(h_rows, n_mod * (size_t) T * eh, h_raw);
        CK(cudaMemcpyAsync(tok_d, h_tok, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(pos_d, h_pos, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        // the dense steps' sub-chunks: as even as cap allows (one when the chunk fits)
        const int nsub = (T + cap - 1) / cap, ssz = (T + nsub - 1) / nsub;
        pass_p0 = p0;
        for (int r0 = 0; r0 < T; r0 += ssz) {   // the embeddings into the streams, a sub-chunk at a time
            const int ts = std::min(ssz, T - r0);
            K::iq_embed_rows(m->embd_type, m->embd, m->embd_row, g.n_vocab, tok_d + r0, ts, N, emb, st);
            if (E.img)   // a picture's cells: its rows, a copy a run
                for (int t = 0; t < ts;) {
                    const int32_t row = E.img->row(p0 + r0 + t);
                    if (row < 0) {
                        ++t;
                        continue;
                    }
                    int len = 1;
                    while (t + len < ts && E.img->row(p0 + r0 + t + len) == row + len) ++len;
                    CK(cudaMemcpyAsync(emb + (size_t) t * N, E.img->rows.data() + (size_t) row * N,
                                       (size_t) len * N * 4, cudaMemcpyHostToDevice, st));
                    t += len;
                }
            hy4k::k_streams_init<<<ts, hy4k::kThreads, 0, st>>>(emb, X + (size_t) r0 * hc * N, hc, N);
        }
        std::string err;
        if (!E.reader.collect(ticket, err)) {
            std::fprintf(stderr, "lamina: the Engram table: %s\n", err.c_str());
            throw Fatal{1};
        }
        E.table_ms += ms_since(tt);
        CK(cudaMemcpyAsync(eraw, h_raw, n_mod * (size_t) T * eh * ed, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(evalid, h_valid, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        std::fill(std::begin(ph), std::end(ph), 0.0);
        ph_t = Clock::now();
        const auto c0 = Clock::now();
        eng_T = T;
        const size_t ew = (size_t) g.eng_heads() * (size_t) g.eng_dim;
        auto view = [&](int r0, int ts) {   // a sub-chunk's views; the head-group tables for its rows (cap at most)
            Xc = X + (size_t) r0 * hc * N;
            postc = post + (size_t) r0 * hc;
            posc = pos_d + r0;
            erawc = eraw + (size_t) r0 * ew;
            evalidc = evalid + r0;
            if (ts != hids_T && ts <= cap) {
                hy4p::k_head_ids<<<(unsigned) ((kG * ts + 255) / 256), 256, 0, st>>>(hids, kG, ts);
                hy4p::k_head_bounds<<<1, 64, 0, st>>>(hb, kG, ts);
                hids_T = ts;
            }
        };
        if (halves_on && nsub >= 2 && devs.size() > 1 && !trace && tq_h2) {
            // two halves of sub-chunks: one half's dense steps on the main GPU while the tiers run the other's experts
            const int nh0 = (nsub + 1) / 2, T0 = std::min(T, nh0 * ssz);
            const int hr0[2] = {0, T0}, hT[2] = {T0, T - T0};
            auto dense_half = [&](Layer& l, int hf) {
                for (int r0 = hr0[hf]; r0 < hr0[hf] + hT[hf]; r0 += ssz) {
                    const int ts = std::min(ssz, hr0[hf] + hT[hf] - r0);
                    view(r0, ts);
                    front(l, ts, p0 + r0);
                    if (l.dense) dense_ffn(l, ts);
                    else moe_a(l, ts, r0, true);
                }
            };
            Layer* pend[2] = {nullptr, nullptr};   // the layer whose experts a half has started
            auto finish = [&](int hf) {
                if (pend[hf]) finish_moe(*pend[hf], hT[hf], hr0[hf], hf);
                pend[hf] = nullptr;
            };
            for (Layer& l : m->layers) {
                if (l.dense) {
                    finish(0);
                    finish(1);
                    dense_half(l, 0);
                    dense_half(l, 1);
                    continue;
                }
                finish(0);
                dense_half(l, 0);
                route_out(l, hT[0], hr0[0], 0);
                finish(1);   // queued before the host waits on half 0's routing: the GPU runs it while the host plans
                start_moe(l, hT[0], hr0[0], 0);
                pend[0] = &l;
                dense_half(l, 1);
                route_out(l, hT[1], hr0[1], 1);
                finish(0);
                start_moe(l, hT[1], hr0[1], 1);
                pend[1] = &l;
            }
            finish(0);
            finish(1);
        } else {
            for (Layer& l : m->layers) {
                for (int r0 = 0; r0 < T; r0 += ssz) {   // the dense steps a sub-chunk at a time
                    const int ts = std::min(ssz, T - r0);
                    view(r0, ts);
                    front(l, ts, p0 + r0);
                    mark(0);
                    if (l.dense) dense_ffn(l, ts);
                    else moe_a(l, ts, r0, nsub > 1);
                    mark(l.dense ? 5 : 1);
                }
                if (!l.dense) moe_b(l, T, nsub > 1);
                mark(5);
            }
        }
        view(0, T);   // the whole chunk's views (the MTP block's catch-up: a chunk of one sub-chunk, its tables made)
        // the head over the last row; with the MTP block kept up, every row's normed final hidden (y) for its catch-up;
        // with DFly's, the last row's in the ring (its picks kept above)
        if (E.drafts_on && m->has_mtp) {
            hy4k::k_hc_pre<<<T, 256, 0, st>>>(X, m->ohc_fn, m->ohc_scale, m->ohc_base, hc, N, hc, g.eps, g.hc_eps, 0.f,
                                              nullptr, xh, m->out_norm, y, g.eps);
            K::bf16_gemv_fp32_mmvf_multi(y + (size_t) (T - 1) * N, m->head, E.logits, N, g.n_vocab, 1, st);
            // the ring as of each snapshot inside the chunk (the block's first row after a restore there takes the
            // state of the row before): the ring before the chunk, the chunk's rows up to the snapshot; before the
            // catch-up, whose attention takes y
            for (size_t si = 0; si < E.chunk_snaps.size(); ++si) {
                const int64_t S = E.chunk_snaps[si];
                uint8_t* hr = snap_devs[si] + hring_off;
                CK(cudaMemcpyAsync(hr, E.hring, (size_t) Engine::kHRing * N * 4, cudaMemcpyDeviceToDevice, st));
                for (int64_t q = std::max<int64_t>(p0, S - Engine::kHRing); q < S; ++q)
                    CK(cudaMemcpyAsync(hr + (size_t) (q % Engine::kHRing) * N * 4, y + (size_t) (q - p0) * N,
                                       (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
            }
            mtp_catch_up(T, p0);
        } else {
            const float* Xl = X + (size_t) (T - 1) * hc * N;
            hy4k::k_hc_pre<<<1, 256, 0, st>>>(Xl, m->ohc_fn, m->ohc_scale, m->ohc_base, hc, N, hc, g.eps, g.hc_eps, 0.f,
                                              nullptr, xh, m->out_norm, xn, g.eps);
            K::bf16_gemv_fp32_mmvf_multi(xn, m->head, E.logits, N, g.n_vocab, 1, st);
            if (E.drafts_on) {
                hy4k::k_ring_store<<<1, 256, 0, st>>>(xn, E.hring, pos_d + (T - 1), Engine::kHRing, N);
                E.snap_base = p0 + T - 1;
                E.snap_rows = 1;
            }
        }
        for (size_t si = 0; si < E.chunk_snaps.size(); ++si) {   // the snapshots inside the chunk, in order
            const int64_t S = E.chunk_snaps[si];
            uint8_t* sd = snap_devs[si];
            for (const Layer& l : E.cl) {   // tails as they are (a snapshot at a multiple of the pool reads none)
                if (!l.has_idx) continue;
                const size_t li = l.mtp ? (size_t) g.n_layers : (size_t) l.il;
                CK(cudaMemcpyAsync(sd + tail_off[li], l.tail, (size_t) kTail * 2 * (size_t) g.idx_dim * 4,
                                   cudaMemcpyDeviceToDevice, st));
            }
            if (E.hring && !(E.drafts_on && m->has_mtp))   // DFly's: it re-anchors after a
                                                           // restore (the MTP block's above)
                CK(cudaMemcpyAsync(sd + hring_off, E.hring, (size_t) Engine::kHRing * N * 4, cudaMemcpyDeviceToDevice,
                                   st));
            std::vector<uint8_t> bytes(snap_size);
            CK(cudaMemcpyAsync(bytes.data(), sd, snap_size, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            E.add_snapshot(S, std::move(bytes), E.kept_at(S));
        }
        ++E.forwards;
        if (trace) {
            mark(5);
            std::fprintf(stderr,
                         "lamina: chunk of %d rows at %lld: %.0f ms; attention and indexer %.0f, routing %.0f, plan "
                         "%.0f, experts %.0f, the rest %.0f\n",
                         T, (long long) p0, ms_since(c0), ph[0], ph[1], ph[2], ph[3], ph[5]);
        }
    }

    /// The MTP block caught up over the chunk (its attention: K/V, pooled keys, tail), every row's state the trunk's
    /// normed final hidden of the row before (y; the ring's for the first), then the ring the chunk's last rows
    void mtp_catch_up(int T, int64_t p0) {
        const auto& g = m->g;
        const int N = (int) g.n_embd;
        Layer& l = m->mtp;
        // states: sy row 0 from the ring (zeros at position 0), rows 1 .. from y
        E.h_src[0] = Engine::ring_src(p0);
        CK(cudaMemcpyAsync(E.mtp_src, E.h_src, 4, cudaMemcpyHostToDevice, st));
        hy4k::k_gather_states<<<1, 256, 0, st>>>(E.hring, nullptr, E.mtp_src, sy, N);
        if (T > 1) CK(cudaMemcpyAsync(sy + N, y, (size_t) (T - 1) * N * 4, cudaMemcpyDeviceToDevice, st));
        const int keep = std::min(T, (int) Engine::kHRing);
        hy4k::k_ring_store<<<keep, 256, 0, st>>>(y + (size_t) (T - keep) * N, E.hring, pos_d + (T - keep),
                                                 Engine::kHRing, N);
        // [enorm(embedding) | hnorm(state)] in X (free now: 2N of its hc N a row), eh_proj into emb, its attention
        hy4k::k_rms<<<T, hy4k::kThreads, 0, st>>>(emb, N, m->mtp_enorm, X, 2 * N, N, g.eps);
        hy4k::k_rms<<<T, hy4k::kThreads, 0, st>>>(sy, N, m->mtp_hnorm, X + N, 2 * N, N, g.eps);
        q8(X, T, 2 * N, hdq);
        dense(m->eh_proj, hdq, T, emb);
        hy4k::k_rms<<<T, hy4k::kThreads, 0, st>>>(emb, N, l.attn_norm, xn, N, N, g.eps);
        attend(l, T, p0);
    }

    /// Engram, the attention, and the FFN's input (xn, its q8_1 rows in xq; iHC's post weights in `post`)
    void front(Layer& l, int T, int64_t p0) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, hc = (int) g.hc;
        if (l.eng >= 0) {
            const int ehn = (int) g.eng_heads(), edn = (int) g.eng_dim, w = ehn * edn;
            hy4k::k_engram_rows<<<dim3(ehn, T), 128, 0, st>>>(erawc + (size_t) l.eng * eng_T * w, l.e_lnorm, evalidc,
                                                              erows, ehn, edn, kEngramEps);
            for (int r0 = 0; r0 < T; r0 += kSub) {
                const int n = std::min(kSub, T - r0);
                q8(erows + (size_t) r0 * w, n, w, subq);
                dense(l.e_kv, subq, n, ekv);
                hy4k::k_engram_mix<<<n, 256, 0, st>>>(Xc + (size_t) r0 * hc * N, ekv, l.e_kv_b, l.e_gate, l.e_gamma,
                                                      l.e_hcs, evalidc + r0, hc, N, kEngramEps);
            }
        }
        hy4k::k_hc_pre<<<T, 256, 0, st>>>(Xc, l.hca_fn, l.hca_scale, l.hca_base, hc, N, 2 * hc, g.eps, g.hc_eps,
                                          g.hc_mag, postc, xh, l.attn_norm, xn, g.eps);
        attend(l, T, p0);
        hy4k::k_hc_post<<<T, hy4k::kThreads, 0, st>>>(Xc, y, postc, hc, N);
        hy4k::k_hc_pre<<<T, 256, 0, st>>>(Xc, l.hcf_fn, l.hcf_scale, l.hcf_base, hc, N, 2 * hc, g.eps, g.hc_eps,
                                          g.hc_mag, postc, xh, l.ffn_norm, xn, g.eps);
        q8(xn, T, N, xq);
    }

    /// A layer's attention over a chunk: its input normed in xn, its output (wo, the post-attention norm) in y
    void attend(Layer& l, int T, int64_t p0) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, H = l.H;
        q8(xn, T, N, xq);
        dense(l.q_a, xq, T, qa);
        hy4k::k_rms<<<T, hy4k::kThreads, 0, st>>>(qa, (int) g.q_lora, l.q_a_norm, qa, (int) g.q_lora, (int) g.q_lora,
                                                  g.eps);
        q8(qa, T, g.q_lora, qrq);
        dense(l.kv_a, xq, T, kva);
        const int64_t p_end = p0 + T;
        const int ring = (int) E.ring, n_swa = (int) g.n_swa;
        int lin0 = 0;
        if (l.swa) {
            // the window's cells in position order: the ring's before
            // the chunk, then the chunk's; the ring keeps the last
            lin0 = (int) std::max<int64_t>(0, p0 - n_swa + 1);
            __half* rh = reinterpret_cast<__half*>(l.kv);   // a ring's rows are halves
            if (p0 > lin0) hy4p::k_copy_cells<<<(unsigned) (p0 - lin0), 64, 0, st>>>(rh, ring, lin, 0, lin0, lin0);
            hy4k::k_kv_store<false>
                <<<T, hy4k::kThreads, 0, st>>>(kva, l.kv_a_norm, lin, posc, 0, lin0, l.theta, g.eps, E.mr);
            for (size_t si = 0; si < E.chunk_snaps.size() && !l.mtp; ++si) {   // the ring as of each snapshot's S
                const int64_t S = E.chunk_snaps[si];
                if (S <= pass_p0 || S < p0 || S >= p_end) continue;
                __half* dst = reinterpret_cast<__half*>(snap_devs[si] + ring_off[(size_t) l.il]);
                const int64_t a = std::max<int64_t>(0, S - ring);
                if (a < lin0)   // the oldest from the ring (still the chunk's predecessor's)
                    hy4p::k_copy_cells<<<(unsigned) (lin0 - a), 64, 0, st>>>(rh, ring, dst, ring, (int) a, 0);
                const int64_t b = std::max<int64_t>(a, lin0);
                hy4p::k_copy_cells<<<(unsigned) (S - b), 64, 0, st>>>(lin, 0, dst, ring, (int) b, lin0);
            }
            const int64_t pa = std::max<int64_t>(p0, p_end - ring);
            hy4p::k_copy_cells<<<(unsigned) (p_end - pa), 64, 0, st>>>(lin, 0, rh, ring, (int) pa, lin0);
        } else {
            (l.kv_q8 ? hy4k::k_kv_store<true> : hy4k::k_kv_store<false>) <<<T, hy4k::kThreads, 0, st>>>(
                kva, l.kv_a_norm, l.kv, posc, 0, 0, l.theta, g.eps, E.mr);
            indexer(l, T, p0);
            if (const int sl = m->dfly.on ? E.dfly_slot(l.il) : -1; sl >= 0) {   // DFly reads it: the last row's picks
                CK(cudaMemcpyAsync(E.snap_cells + (size_t) sl * kMaxT * g.idx_top_k,
                                   cells + (size_t) (T - 1) * g.idx_top_k, (size_t) g.idx_top_k * 4,
                                   cudaMemcpyDeviceToDevice, st));
                CK(cudaMemcpyAsync(E.snap_n + (size_t) sl * kMaxT, ncells + (T - 1), 4, cudaMemcpyDeviceToDevice, st));
            }
        }
        const float scale = 1.0f / sqrtf((float) g.k_head);
        const size_t kb_bytes = (size_t) hy4k::kLatent * ((size_t) g.nope() / 32 * 34);
        const size_t vb_bytes = (size_t) g.v_head * ((size_t) hy4k::kLatent / 32 * 34);
        for (int h0 = 0; h0 < H; h0 += kG) {
            dense(l.q_b, qrq, T, qg, kG * g.k_head, (size_t) h0 * g.k_head);
            hy4k::k_rope_norm<<<dim3(kG, T), 32, 0, st>>>(qg, kG, (int) g.k_head, (int) g.nope(), (int) g.n_rot, posc,
                                                          0, l.theta, E.mr);
            q8(qg, (int64_t) kG * T, g.nope(), hdq, 8, hids, g.k_head);
            MMQ::Product kb;
            kb.w = l.k_b.d + (size_t) h0 * kb_bytes;
            kb.type = l.k_b.type;
            kb.w_rows = hy4k::kLatent;
            kb.w_cols = g.nope();
            kb.expert_bytes = kb_bytes;
            kb.n = kG;
            kb.xq = hdq;
            kb.bounds = hb;
            kb.ids = hids;
            kb.ids_rows = hids_rows;
            kb.total_rows = (int64_t) kG * T;
            kb.max_rows = T;
            kb.dst = ql;
            kb.ld_dst = hy4k::kLatent;
            devs[0].mmq->run(kb, st);
            if (l.swa)
                hy4p::k_attn_prompt<false><<<T, 256, sizeof(hy4p::AttnSmem), st>>>(
                    ql, qg, lin, nullptr, nullptr, 0, posc, n_swa, lin0, l.sinks + h0, scale);
            else
                (l.kv_q8 ? hy4p::k_attn_prompt<true, true>
                         : hy4p::k_attn_prompt<true, false>) <<<T, 256, sizeof(hy4p::AttnSmem), st>>>(
                    ql, qg, l.kv, cells, ncells, (int) g.idx_top_k, posc, 0, 0, l.sinks + h0, scale);
            q8(ql, (int64_t) kG * T, hy4k::kLatent, hdq, 8, hids);
            MMQ::Product vb = kb;
            vb.w = l.v_b.d + (size_t) h0 * vb_bytes;
            vb.type = l.v_b.type;
            vb.w_rows = g.v_head;
            vb.w_cols = hy4k::kLatent;
            vb.expert_bytes = vb_bytes;
            vb.dst = qg;
            vb.ld_dst = g.v_head;
            devs[0].mmq->run(vb, st);
            dense(l.gate, xq, T, gt, kG * g.v_head, (size_t) h0 * g.v_head);
            const int64_t n_o = (int64_t) T * kG * g.v_head;
            hy4k::k_gate_mul<<<blocks(n_o), hy4k::kThreads, 0, st>>>(qg, gt, n_o);
            // into the wo input's column blocks of these heads
            q8(qg, T, (int64_t) kG * g.v_head, oq + (size_t) h0 * g.v_head / 128 * T * kQ8Block);
        }
        dense(l.wo, oq, T, y);
        hy4k::k_rms_inplace<<<T, hy4k::kThreads, 0, st>>>(y, l.post_attn, N, g.eps);
    }

    /// The sparse layer's indexer over a chunk: its rows' keys pooled into the groups they complete (the tail's cells
    /// before the chunk), the tail's last cells, each row's cells (every position while its visible groups are within
    /// the reach, else the top groups' by the tensor-core scores, R rows a pass).
    void indexer(Layer& l, int T, int64_t p0) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, ih = (int) g.idx_heads, idd = (int) g.idx_dim, pool_n = (int) g.idx_pool;
        dense(l.iq_b, qrq, T, iq);
        hy4k::k_rope_norm<<<dim3(ih, T), 32, 0, st>>>(iq, ih, idd, idd - (int) g.n_rot, (int) g.n_rot, posc, 0, l.theta,
                                                      E.mr);
        dense(l.ik, xq, T, ik);
        hy4k::k_rms<<<T, 128, 0, st>>>(ik, idd, l.ik_norm, ik, idd, idd, g.eps);
        q8(ik, T, idd, ikq);
        dense(l.icv, ikq, T, iv);
        dense(l.icg, ikq, T, ig);
        const int64_t p_end = p0 + T;
        const int64_t g_a = (p0 + pool_n) / pool_n - 1, g_b = p_end / pool_n;   // the groups the chunk completes
        if (g_b > g_a)
            hy4p::k_pool_chunk<<<(unsigned) (g_b - g_a), idd, 0, st>>>(iv, ig, l.tail, kTail, (int) p0, (int) g_a, idd,
                                                                       pool_n, l.ape, l.ipool_norm, g.eps, l.theta,
                                                                       (int) g.n_rot, l.pooled, E.mr);
        const int keep = std::min(T, kTail);
        hy4k::k_tail_store<<<keep, 128, 0, st>>>(iv + (size_t) (T - keep) * idd, ig + (size_t) (T - keep) * idd, l.tail,
                                                 posc + (T - keep), kTail, idd);
        hy4k::k_gemv_f32<<<dim3((unsigned) ((ih + 7) / 8), T), 256, 0, st>>>(l.iproj, xn, iw, N, ih,
                                                                             1.0f / sqrtf((float) (idd * ih)));
        const int reach = (int) (g.idx_top_k / pool_n - (g.idx_tail ? 1 : 0));
        const int ld = (int) (E.ctx / pool_n);
        for (int r0 = 0; r0 < T; r0 += R) {
            const int rr = std::min(R, T - r0);
            const int vis_max = (int) ((p0 + r0 + rr - 1) / pool_n);
            static const bool f32 = std::getenv("LAMINA_HY4_IDX_F32") != nullptr;   // the decode's FP32 scores (A/B)
            if (vis_max > reach && f32) {
                const size_t smem = ((size_t) ih * idd + ih) * sizeof(float);
                hy4k::k_index_scores_pos<<<dim3(kScoreBlocks, rr), 256, smem, st>>>(
                    iq + (size_t) r0 * ih * idd, iw + (size_t) r0 * ih, l.pooled, posc + r0, pool_n, reach, ih, idd,
                    scores, ld);
            } else if (vis_max > reach) {
                const int splits = std::max(1, std::min(64, (vis_max + 1023) / 1024));
                const int per = (vis_max + splits - 1) / splits / hy4p::kIdxTile * hy4p::kIdxTile + hy4p::kIdxTile;
                hy4p::k_index_scores_mma<<<dim3((unsigned) ((rr + 7) / 8), (unsigned) splits), 256, 0, st>>>(
                    iq, iw, l.pooled, posc, pool_n, reach, r0, rr, per, scores, ld);
            }
            hy4k::k_select_cells<<<rr, 1024, 0, st>>>(scores, ld, posc + r0, pool_n, reach,
                                                      cells + (size_t) r0 * g.idx_top_k, (int) g.idx_top_k,
                                                      ncells + r0);
        }
    }

    void dense_ffn(Layer& l, int T) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, hc = (int) g.hc, FF = (int) g.n_ff;
        for (int r0 = 0; r0 < T; r0 += kSub) {
            const int n = std::min(kSub, T - r0);
            q8(xn + (size_t) r0 * N, n, N, subq);
            dense(l.f_gate, subq, n, fg);
            dense(l.f_up, subq, n, fu);
            hy4k::k_swiglu<<<blocks((int64_t) n * FF), hy4k::kThreads, 0, st>>>(fg, fu, fh, (int64_t) n * FF);
            q8(fh, n, FF, subq);
            dense(l.f_down, subq, n, y + (size_t) r0 * N);
        }
        hy4k::k_rms_inplace<<<T, hy4k::kThreads, 0, st>>>(y, l.post_ffw, N, g.eps);
        hy4k::k_hc_post<<<T, hy4k::kThreads, 0, st>>>(Xc, y, postc, hc, N);
    }

    // ---- the routed experts.  The host takes each layer's routing, hands every entry to a GPU (its expert's slot,
    // else the streamer, which copies the expert from the arena) and plans each GPU's groups: experts of consecutive
    // blobs whose entries (up to kEmax) run as one MMQ product, their rows summed into the tokens' sums in pick order.
    struct Group {
        const uint8_t* w = nullptr;   // the first expert's blob on its GPU
        int n = 0, n_ent = 0, max_rows = 0, n_runs = 0;
        int o_bounds = 0, o_tok = 0, o_rtok = 0, o_rstart = 0, o_rent = 0, o_rw = 0;   // into the plan (int32)
    };
    struct Batch {   // the streamer's experts copied into one staging buffer, and its groups
        std::vector<int32_t> experts;
        int g0 = 0, g1 = 0;
    };
    struct DevPlan {
        std::vector<int32_t> P;
        std::vector<Group> groups;   // the batches' groups first, then the slots'
        std::vector<Batch> batches;
        int n_slot_groups = 0;
    };
    struct PlanSet {   // a half's plans: each GPU's groups, its entries by slot, the entries no GPU holds by expert
        std::vector<DevPlan> plans;
        std::vector<std::vector<std::vector<int32_t>>> slot_ents;   // [dev][slot]: entries
        std::vector<std::vector<int32_t>> free_ents;                // [expert]: entries no GPU holds
    };
    PlanSet ps[2];

    /// Groups over (blob index, entries) in ascending blob order from `base` (`stride` apart), split at kEmax entries
    void add_groups(DevPlan& dp, const uint8_t* blob0, size_t stride,
                    const std::vector<std::pair<int, const std::vector<int32_t>*>>& seq, const float* wts_h) {
        const int K = (int) m->g.top_k;
        std::vector<int32_t>& P = dp.P;
        size_t i = 0, off = 0;
        std::vector<int32_t> ents, cnt;
        std::vector<int> loc;
        while (i < seq.size()) {
            ents.clear();
            cnt.clear();
            const int ja = seq[i].first;
            while (i < seq.size() && (int) ents.size() < kEmax) {
                const int j = seq[i].first;
                const std::vector<int32_t>& v = *seq[i].second;
                const size_t take = std::min<size_t>((size_t) kEmax - ents.size(), v.size() - off);
                cnt.resize((size_t) (j - ja + 1), 0);
                cnt[(size_t) (j - ja)] += (int32_t) take;
                ents.insert(ents.end(), v.begin() + (ptrdiff_t) off, v.begin() + (ptrdiff_t) (off + take));
                off += take;
                if (off < v.size()) break;   // the rest of this expert's entries: the next group
                ++i;
                off = 0;
            }
            Group gr;
            gr.w = blob0 + (size_t) ja * stride;
            gr.n = (int) cnt.size();
            gr.n_ent = (int) ents.size();
            gr.max_rows = *std::max_element(cnt.begin(), cnt.end());
            gr.o_bounds = (int) P.size();
            int32_t acc = 0;
            P.push_back(0);
            for (const int32_t c : cnt) P.push_back(acc += c);
            gr.o_tok = (int) P.size();
            for (const int32_t e : ents) P.push_back(e / K);
            // the runs: the group's entries by token, in pick order
            loc.resize(ents.size());
            for (size_t k = 0; k < loc.size(); ++k) loc[k] = (int) k;
            std::sort(loc.begin(), loc.end(), [&](int a, int b) { return ents[(size_t) a] < ents[(size_t) b]; });
            std::vector<int32_t> rtok, rstart;
            for (size_t k = 0; k < loc.size(); ++k) {
                const int32_t tk = ents[(size_t) loc[k]] / K;
                if (rtok.empty() || rtok.back() != tk) {
                    rtok.push_back(tk);
                    rstart.push_back((int32_t) k);
                }
            }
            rstart.push_back((int32_t) loc.size());
            gr.n_runs = (int) rtok.size();
            gr.o_rtok = (int) P.size();
            P.insert(P.end(), rtok.begin(), rtok.end());
            gr.o_rstart = (int) P.size();
            P.insert(P.end(), rstart.begin(), rstart.end());
            gr.o_rent = (int) P.size();
            for (const int k : loc) P.push_back(k);
            gr.o_rw = (int) P.size();
            for (const int k : loc) {
                int32_t bits;
                std::memcpy(&bits, &wts_h[ents[(size_t) k]], 4);
                P.push_back(bits);
            }
            dp.groups.push_back(gr);
        }
    }

    /// The plan of T rows' routing (ids_h, wts_h: the host's copies from the rows' first) into half hf's PlanSet
    void plan_layer(Layer& l, int T, int hf, const int32_t* ids_h, const float* wts_h) {
        const auto& g = m->g;
        const int K = (int) g.top_k, nE = (int) g.n_expert;
        const size_t nd = devs.size();
        auto& plans = ps[hf].plans;
        auto& slot_ents = ps[hf].slot_ents;
        auto& free_ents = ps[hf].free_ents;
        plans.assign(nd, DevPlan{});
        slot_ents.resize(nd);
        for (size_t d = 0; d < nd; ++d) {
            slot_ents[d].resize((size_t) l.held[d].n);
            for (auto& v : slot_ents[d]) v.clear();
        }
        free_ents.resize((size_t) nE);
        for (auto& v : free_ents) v.clear();
        E.dump_route(1, l.il, T, ids_h);
        int64_t on_gpus = 0;
        for (int e = 0; e < T * K; ++e) {
            const int x = ids_h[e];
            l.use[(size_t) x] += 1.f;
            l.total[(size_t) x] += 1.f;
            int holder = -1;
            for (size_t d = 0; d < nd && holder < 0; ++d)
                if (l.held[d].n > 0 && l.held[d].slot_of[(size_t) x] >= 0) holder = (int) d;
            if (holder >= 0) {
                slot_ents[(size_t) holder][(size_t) l.held[(size_t) holder].slot_of[(size_t) x]].push_back(e);
                ++E.gpus[(size_t) holder].hits;
                ++on_gpus;
            } else {
                free_ents[(size_t) x].push_back(e);
            }
        }
        E.lookups += (int64_t) T * K;
        E.hits += on_gpus;
        std::vector<std::pair<int, const std::vector<int32_t>*>> seq;
        for (size_t d = 0; d < nd; ++d) {
            DevPlan& dp = plans[d];
            if ((int) d == streamer) {   // the experts no GPU holds, kStage a staging buffer
                std::vector<int32_t> used;
                for (int x = 0; x < nE; ++x)
                    if (!free_ents[(size_t) x].empty()) used.push_back(x);
                for (size_t b0 = 0; b0 < used.size(); b0 += kStage) {
                    Batch bt;
                    bt.experts.assign(used.begin() + (ptrdiff_t) b0,
                                      used.begin() + (ptrdiff_t) std::min(used.size(), b0 + kStage));
                    seq.clear();
                    for (size_t j = 0; j < bt.experts.size(); ++j)
                        seq.emplace_back((int) j, &free_ents[(size_t) bt.experts[j]]);
                    bt.g0 = (int) dp.groups.size();
                    add_groups(dp, devs[d].stage[dp.batches.size() % 2], l.stride, seq, wts_h);
                    bt.g1 = (int) dp.groups.size();
                    dp.batches.push_back(std::move(bt));
                }
            }
            seq.clear();
            for (int s = 0; s < l.held[d].n; ++s)
                if (!slot_ents[d][(size_t) s].empty()) seq.emplace_back(s, &slot_ents[d][(size_t) s]);
            const size_t g_before = dp.groups.size();
            add_groups(dp, l.held[d].slots, l.stride, seq, wts_h);
            dp.n_slot_groups = (int) (dp.groups.size() - g_before);
        }
    }

    void run_group(MoeDev& d, Layer& l, int T, const Group& gr, float* acc) {
        const auto& g = m->g;
        const int Ld = (int) g.n_latent, F = (int) g.n_ff_exp;
        hy4p::k_gather_q8<<<gr.n_ent, 128, 0, d.st>>>(reinterpret_cast<const uint4*>(d.tq),
                                                      reinterpret_cast<uint4*>(d.eq), d.plan + gr.o_tok, gr.n_ent, T,
                                                      Ld / 128);
        MMQ::Product gu;
        gu.w = gr.w;
        gu.type = l.fmt.gu_type;
        gu.w_rows = 2 * F;
        gu.w_cols = Ld;
        gu.expert_bytes = l.stride;
        gu.n = gr.n;
        gu.xq = d.eq;
        gu.bounds = d.plan + gr.o_bounds;
        gu.ids = d.ident;
        gu.ids_rows = d.ident_rows;
        gu.total_rows = gr.n_ent;
        gu.max_rows = gr.max_rows;
        gu.dst = d.gu;
        gu.ld_dst = 2 * F;
        d.mmq->run(gu, d.st);
        const int64_t nh = (int64_t) gr.n_ent * F;
        hy4p::k_swiglu_split<<<blocks(nh), hy4k::kThreads, 0, d.st>>>(d.gu, d.h, F, nh, l.fmt.swiglu_limit);
        MMQ::quantize(d.h, nullptr, d.hq, l.fmt.d_type, F, F, gr.n_ent, d.st);
        MMQ::Product dn = gu;
        dn.w = gr.w + l.fmt.down_off;
        dn.type = l.fmt.d_type;
        dn.w_rows = Ld;
        dn.w_cols = F;
        dn.xq = d.hq;
        dn.dst = d.rows;
        dn.ld_dst = Ld;
        d.mmq->run(dn, d.st);
        hy4p::k_accum_runs<<<gr.n_runs, 256, 0, d.st>>>(d.rows, d.plan + gr.o_rtok, d.plan + gr.o_rstart,
                                                        d.plan + gr.o_rent,
                                                        reinterpret_cast<const float*>(d.plan + gr.o_rw), acc, Ld);
    }

    /// GPU d's share of half hf's experts (T rows from row r0 of the chunk): its plan, its input (a tier's from the
    /// host), its groups (the streamer's batches copied from the arena between), a tier's sums out to the host
    void enqueue(MoeDev& d, Layer& l, int T, int hf, int r0) {
        const auto& g = m->g;
        const int Ld = (int) g.n_latent;
        DevPlan& dp = ps[hf].plans[(size_t) d.gi];
        lamina::core::DeviceScope ds(d.dev, main_dev, d.gi > 0);
        int32_t* plan_h = hf ? d.plan_h2 : d.plan_h;
        float* acc = d.gi > 0 ? d.acc : d.acc + (size_t) r0 * Ld;   // a tier's from its row 0 (a half at a time)
        std::memcpy(plan_h, dp.P.data(), dp.P.size() * 4);
        CK(cudaMemcpyAsync(d.plan, plan_h, dp.P.size() * 4, cudaMemcpyHostToDevice, d.st));
        CK(cudaMemsetAsync(acc, 0, (size_t) T * Ld * 4, d.st));
        if (d.gi > 0) CK(cudaMemcpyAsync(d.tq, hf ? tq_h2 : tq_h, MMQ::q8_bytes(T, Ld), cudaMemcpyHostToDevice, d.st));
        const size_t n_batch_groups = dp.groups.size() - (size_t) dp.n_slot_groups;
        for (size_t gi = n_batch_groups; gi < dp.groups.size(); ++gi) run_group(d, l, T, dp.groups[gi], acc);
        for (size_t b = 0; b < dp.batches.size(); ++b) {
            const Batch& bt = dp.batches[b];
            const int buf = (int) (b % 2);
            CK(cudaStreamWaitEvent(d.cst, d.used[buf], 0));
            for (size_t j = 0; j < bt.experts.size(); ++j)
                CK(cudaMemcpyAsync(d.stage[buf] + j * l.stride, l.blob[(size_t) bt.experts[j]], l.fmt.bytes,
                                   cudaMemcpyHostToDevice, d.cst));
            CK(cudaEventRecord(d.copied[buf], d.cst));
            CK(cudaStreamWaitEvent(d.st, d.copied[buf], 0));
            for (int gi = bt.g0; gi < bt.g1; ++gi) run_group(d, l, T, dp.groups[(size_t) gi], acc);
            CK(cudaEventRecord(d.used[buf], d.st));
        }
        if (d.gi > 0) {   // the sums out (as halves: sum_f16)
            if (sum_f16) {
                __half* a16 = d.acc16;
                hy4p::k_to_half<<<blocks((int64_t) T * Ld), hy4k::kThreads, 0, d.st>>>(acc, a16, (int64_t) T * Ld);
                CK(cudaMemcpyAsync(reinterpret_cast<uint8_t*>(d.acc_h) + (size_t) r0 * Ld * 2, a16, (size_t) T * Ld * 2,
                                   cudaMemcpyDeviceToHost, d.st));
            } else {
                CK(cudaMemcpyAsync(d.acc_h + (size_t) r0 * Ld, acc, (size_t) T * Ld * 4, cudaMemcpyDeviceToHost, d.st));
            }
            CK(cudaEventRecord(hf ? d.done2 : d.done, d.st));
        }
    }

    /// A sub-chunk's part of an MoE layer (its ts rows at r0 of the pass): the router's picks, routed_down's input, and
    /// with several sub-chunks the shared expert (its input the sub-chunk's xq)
    void moe_a(Layer& l, int ts, int r0, bool shared) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, nE = (int) g.n_expert, K = (int) g.top_k, Ld = (int) g.n_latent;
        lamina::kernels::Gemm32 rp;   // the router's logits [ts][nE], in floats
        rp.x = xn;
        rp.w = l.router;
        rp.y = logits_r;
        rp.T = ts;
        rp.N = nE;
        rp.K = N;
        lamina::kernels::gemm32(rp, {gsc, gsc_bytes}, st);
        hy4k::k_route<<<ts, std::max(32, ((nE + 31) / 32) * 32), 0, st>>>(logits_r, l.bias, nE, K, g.w_norm, g.w_scale,
                                                                          ids + (size_t) r0 * K, wts + (size_t) r0 * K);
        dense(l.r_down, xq, ts, rin + (size_t) r0 * Ld);
        if (shared) shared_expert(l, ts, sy + (size_t) r0 * N);
    }
    void shared_expert(Layer& l, int ts, float* out) {
        const int F = (int) m->g.n_ff_exp;
        dense(l.s_gate, xq, ts, sg);
        dense(l.s_up, xq, ts, su);
        hy4k::k_swiglu<<<blocks((int64_t) ts * F), hy4k::kThreads, 0, st>>>(sg, su, sh, (int64_t) ts * F);
        q8(sh, ts, F, shq);
        dense(l.s_down, shq, ts, out);
    }
    /// Half hf's routing out (T rows from row r0): routed_down's input as q8_1 rows (the main GPU's experts read them,
    /// the tiers a host copy), the picks and weights to the host; `routed` (routed2) marks them
    void route_out(Layer& l, int T, int r0, int hf) {
        const auto& g = m->g;
        const int K = (int) g.top_k, Ld = (int) g.n_latent;
        MoeDev& d0 = devs[0];
        q8(rin + (size_t) r0 * Ld, T, Ld, d0.tq, l.fmt.gu_type);
        CK(cudaMemcpyAsync(h_ids + (size_t) r0 * K, ids + (size_t) r0 * K, (size_t) T * K * 4, cudaMemcpyDeviceToHost,
                           st));
        CK(cudaMemcpyAsync(h_wts + (size_t) r0 * K, wts + (size_t) r0 * K, (size_t) T * K * 4, cudaMemcpyDeviceToHost,
                           st));
        if (devs.size() > 1)
            CK(cudaMemcpyAsync(hf ? tq_h2 : tq_h, d0.tq, MMQ::q8_bytes(T, Ld), cudaMemcpyDeviceToHost, st));
        CK(cudaEventRecord(hf ? routed2 : routed, st));
    }
    /// The host's wait for half hf's routing, its plan, every GPU's experts queued (the streamer's last)
    void start_moe(Layer& l, int T, int r0, int hf) {
        const int K = (int) m->g.top_k;
        CK(cudaEventSynchronize(hf ? routed2 : routed));
        mark(1);
        plan_layer(l, T, hf, h_ids + (size_t) r0 * K, h_wts + (size_t) r0 * K);
        mark(2);
        if (streamer != 0) enqueue(devs[0], l, T, hf, r0);
        for (size_t d = 1; d < devs.size(); ++d)
            if ((int) d != streamer) enqueue(devs[d], l, T, hf, r0);
        enqueue(devs[(size_t) streamer], l, T, hf, r0);
    }
    /// Half hf's end of an MoE layer: the tiers' sums added in their order, routed_up, the shared expert's rows, the
    /// norm, iHC's post (T rows from row r0)
    void finish_moe(Layer& l, int T, int r0, int hf) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, hc = (int) g.hc, Ld = (int) g.n_latent;
        MoeDev& d0 = devs[0];
        float* acc = d0.acc + (size_t) r0 * Ld;
        for (size_t d = 1; d < devs.size(); ++d) {
            CK(cudaStreamWaitEvent(st, hf ? devs[d].done2 : devs[d].done, 0));
            const size_t esz = sum_f16 ? 2 : 4;
            uint8_t* dst = reinterpret_cast<uint8_t*>(tmp) + (size_t) r0 * Ld * 4;
            CK(cudaMemcpyAsync(dst, reinterpret_cast<const uint8_t*>(devs[d].acc_h) + (size_t) r0 * Ld * esz,
                               (size_t) T * Ld * esz, cudaMemcpyHostToDevice, st));
            if (sum_f16)
                hy4p::k_add_half<<<blocks((int64_t) T * Ld), hy4k::kThreads, 0, st>>>(
                    acc, reinterpret_cast<const __half*>(dst), (int64_t) T * Ld);
            else
                hy4k::k_add<<<blocks((int64_t) T * Ld), hy4k::kThreads, 0, st>>>(
                    acc, reinterpret_cast<const float*>(dst), (int64_t) T * Ld);
        }
        mark(3);
        float* yr = y + (size_t) r0 * N;
        q8(acc, T, Ld, mq);
        dense(l.r_up, mq, T, yr);
        hy4k::k_add<<<blocks((int64_t) T * N), hy4k::kThreads, 0, st>>>(yr, sy + (size_t) r0 * N, (int64_t) T * N);
        hy4k::k_rms_inplace<<<T, hy4k::kThreads, 0, st>>>(yr, l.post_ffw, N, g.eps);
        hy4k::k_hc_post<<<T, hy4k::kThreads, 0, st>>>(X + (size_t) r0 * hc * N, yr, post + (size_t) r0 * hc, hc, N);
    }
    /// The chunk's MoE over its T rows in one piece: the plan, every GPU's experts (the ones no GPU holds streamed
    /// once), the sums, routed_up, the shared expert's rows added, the norm, iHC's post.  `shared_done`: the sub-chunks
    /// made the shared expert's rows; else (one sub-chunk) it runs here, beside the experts.
    void moe_b(Layer& l, int T, bool shared_done) {
        route_out(l, T, 0, 0);
        const auto t0 = Clock::now();
        start_moe(l, T, 0, 0);
        if (!shared_done) shared_expert(l, T, sy);
        finish_moe(l, T, 0, 0);
        ms_moe += ms_since(t0);
    }
};

// ------------------------------------------------------------------------------------------------ serve
/// lamina's engine protocol, one request at a time: GEN <max_new> [key=value ...] <ids>, or with --vision GENI
/// <max_new> [keys] <file> <ids> (the file lamina-vision's rows of the prompt's pictures) -> RESUME (the tokens the
/// state already held), PP lines while the prompt is read, a T line a token, DONE; STOP ends a request at its next
/// step, QUIT the loop.
struct Serve {
    Engine& E;
    int32_t eos = -1;
    lamina::program::ServeInput* in = nullptr;   // run()'s input lines, while it runs

    Serve(Engine& e, int32_t eos_id, int drafts_) : E(e), eos(eos_id), drafts(drafts_) {}
    int drafts = 0;                     // the drafts a window (E.drafts_on)
    std::vector<int32_t> prev_prompt;   // the request before's prompt: where the next one leaves it

    // The server's image encoder, which runs in this process, holds no VRAM of its own between pictures where the
    // server has it borrow (its "vision" section's "lend"): for a new picture it asks for memory of the main GPU,
    // mapped here from the chunks its last expert slots give up (and the free VRAM above the reserve), as the prompt
    // path's pass buffers are; given back, the slots grow again and the residents they displaced are queued back.
    std::unique_ptr<lamina::core::VmmRange> enc_vm;
    void enc_back() {
        if (!enc_vm) return;
        CK(cudaDeviceSynchronize());   // the encoder's last kernels
        for (int64_t c = 0; c < enc_vm->chunks(); ++c)
            if (enc_vm->mapped(c)) E.spare.push_back(enc_vm->unmap(c));
        enc_vm.reset();
        E.slots_back();
        const int q = E.queue_restore();
        std::fprintf(stderr,
                     "lamina: the image encoder's loan back to the main GPU's slots (%d residents queued back)\n", q);
    }
    void lend(const std::string& line, bool in_process) {
        long long dev = -1;
        unsigned long long bytes = 0;
        std::string why;
        const int s0 = E.gpus.empty() ? 0 : E.gpus[0].S;
        if (std::sscanf(line.c_str() + 5, "%lld %llu", &dev, &bytes) != 2 || bytes == 0) {
            why = "expected LEND <device> <bytes>";
        } else if (!in_process) {
            why = "a loan is memory of the engine's own process";
        } else if (enc_vm) {
            why = "a loan is out already";
        } else if (dev != E.main_dev) {
            why = "Hy4-flash's engine lends on its main GPU alone (GPU " + std::to_string(E.main_dev) + ")";
        } else if (!E.elastic) {
            why = "the main GPU's slots are not mapped in chunks (--no-kv-grow, or no VMM)";
        } else {
            auto vm = std::make_unique<lamina::core::VmmRange>();
            if (!vm->reserve(bytes)) {
                why = "no address range of " + std::to_string(bytes >> 20) + " MiB";
            } else if (!E.lend_chunks(vm->chunks())) {
                why = "the main GPU's slots cannot spare " + std::to_string(bytes >> 20) + " MiB";
            } else {
                auto take = [&]() -> lamina::core::VmmChunk {
                    if (E.spare.empty()) return 0;
                    const lamina::core::VmmChunk h = E.spare.back();
                    E.spare.pop_back();
                    return h;
                };
                if (vm->map_range(0, vm->chunks(), take)) {
                    enc_vm = std::move(vm);
                } else {
                    for (int64_t c = 0; c < vm->chunks(); ++c)
                        if (vm->mapped(c)) E.spare.push_back(vm->unmap(c));
                    E.slots_back();
                    E.queue_restore();
                    why = "the main GPU has no " + std::to_string(bytes >> 20) + " MiB to map";
                }
            }
        }
        if (why.empty()) {
            const long long out = (long long) (s0 - E.gpus[0].S) * (long long) E.lent_layers();
            serve_print("LENT %llu %llu %lld %lld\n", (unsigned long long) (uintptr_t) enc_vm->base(),
                        (unsigned long long) enc_vm->chunks() * lamina::core::vmm_granularity(), out, out);
        } else {
            serve_print("ERR no loan: %s\n", why.c_str());
        }
    }

    void request(const std::string& line) {
        const auto& g = E.m->g;
        if (enc_vm) {   // a request while a loan is out: the chunks come back first
            std::fprintf(stderr, "lamina: a request came while the image encoder's loan was out: taken back\n");
            enc_back();
        }
        E.warm_up();     // a request after a pause: the cards brought up beside its first work
        struct Ended {   // the request's end, for the next one's pause
            Engine& e;
            ~Ended() {
                e.last_end = Clock::now();
                e.served = true;
            }
        } ended{E};
        const bool geni = line.rfind("GENI ", 0) == 0;
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
            if (tk.empty() || eq == std::string::npos) {
                p = const_cast<char*>(st0);
                break;
            }
            if (tk.compare(0, eq, "img_keys") == 0) {
                for (const char* q = tk.c_str() + eq + 1; *q;) {
                    char* qe = nullptr;
                    const unsigned long long x = std::strtoull(q, &qe, 16);
                    if (qe == q || (*qe != ',' && *qe != '\0')) {
                        img_keys.clear();
                        break;
                    }
                    img_keys.push_back((uint64_t) x);
                    q = *qe == ',' ? qe + 1 : qe;
                }
            } else {
                smp.set(tk.substr(0, eq), tk.c_str() + eq + 1);   // keys of other paths (score) are left
            }
        }
        std::string emb;   // GENI: the request's lamina-vision file before the ids
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
            if (t < 0 || t >= g.n_vocab) {
                serve_print("ERR a token id is outside the vocabulary\n");
                return;
            }
        ImgReq ir;
        if (geni) {
            if (!E.vision) {
                serve_print("ERR this engine was started without --vision\n");
                return;
            }
            if (const std::string ve =
                    lamina::program::image_cells(emb, ids, {E.m->image_pad, E.m->video_pad}, g.n_embd, img_keys, ir);
                !ve.empty()) {
                serve_print("ERR %s\n", ve.c_str());
                return;
            }
        }
        E.mrope_set(geni ? &ir : nullptr);
        // (the room left, not the sum: that overflows for a huge max_new)
        if ((int64_t) ids.size() > E.ctx - kMaxT || max_new > E.ctx - kMaxT - (int64_t) ids.size()) {
            serve_print("ERR prompt (%zu tokens) + max_new (%lld) exceeds the context (%lld)\n", ids.size(), max_new,
                        (long long) (E.ctx - kMaxT));
            return;
        }
        in->stop.store(false);
        // what the state holds of this prompt: its common start, less one token when that is all of it (the last
        // prompt token's logits are the first pick)
        auto common_of = [&](const std::vector<int32_t>& s) {
            size_t c = 0;
            while (c < s.size() && c < ids.size() && s[c] == ids[c]) ++c;
            return c;
        };
        size_t common = common_of(E.seq);
        // a parked conversation the request continues from further on than from the live one: its own next request
        // (one that continues it whole, or superseded()) takes it back, any other borrows its start and leaves it
        // parked; the live one is parked in either case.  Else the live one is parked where the request leaves it (it
        // drops kParkMin or more of its cells and a kParkShare-th of it or more: a sibling's first message after one
        // long system prompt too) and is not its own next request.
        size_t best = 0;
        int64_t best_r = Engine::reusable(E.seq, Engine::snaps_at(E.snaps), ids);
        for (size_t i = 0; i < E.park.size(); ++i)
            if (const int64_t r = Engine::reusable(E.park[i].seq, Engine::snaps_at(E.park[i]), ids); r > best_r) {
                best = i + 1;
                best_r = r;
            }
        const int64_t drop = E.pos() - (int64_t) common;
        if (best > 0) {
            Engine::Parked& c = E.park[best - 1];
            if (best_r >= (int64_t) c.seq.size() || lamina::core::superseded(c.seq, ids, best_r)) {
                Engine::Parked back = std::move(c);
                E.park.erase(E.park.begin() + (ptrdiff_t) (best - 1));
                E.park_live(0);
                E.unpark(std::move(back));
            } else {
                c.pin = true;
                E.park_live(0);   // (the park may move: the lender found again)
                const auto lender =
                    std::find_if(E.park.begin(), E.park.end(), [](const Engine::Parked& x) { return x.pin; });
                E.borrow(*lender, best_r);
                lender->pin = false;
            }
            common = common_of(E.seq);
        } else if (drop >= Engine::kParkMin && drop * lamina::core::kParkShare >= E.pos() &&
                   !lamina::core::superseded(E.seq, ids, (int64_t) common)) {
            E.park_live((int64_t) common);
        }
        std::vector<float> lg;
        const Engine::Snap* whole = common == ids.size() ? E.snap_at((int64_t) common) : nullptr;
        if (whole != nullptr && whole->logits.empty()) whole = nullptr;
        if (whole) lg = whole->logits;   // the held prompt again: its snapshot has the first pick's logits
        else if (common == ids.size()) --common;
        const size_t reuse = (size_t) E.rewind((int64_t) common);
        if (reuse > 0) serve_print("RESUME %zu\n", reuse);
        const auto t0 = Clock::now();
        const int64_t fresh = (int64_t) (ids.size() - reuse);
        int64_t done = 0;
        auto last_pp = Clock::now();
        // the read's snapshots besides the multiples of kSnapEvery, each at the last multiple of the indexer's pool at
        // or before its place (and `past` cells or more past the start it resumes from): behind the prompt's last
        // picture or video; where it leaves the request before (its pictures compared: their cells' ids are their
        // keys') and a few cells before the end of a last message the request before held 256 tokens or more of, a
        // message that grows between requests; before the last message, its turn's first token (a client that sends
        // that message again after each new turn leaves the prompt there)
        {
            const int64_t n = (int64_t) ids.size();
            E.wanted.clear();
            auto want = [&](int64_t place, Engine::Kept why, int64_t past) {
                const int64_t at = place / g.idx_pool * g.idx_pool;
                if (at > (int64_t) reuse + past && at < n && at % Engine::kSnapEvery != 0 && E.snap_at(at) == nullptr &&
                    std::none_of(E.wanted.begin(), E.wanted.end(), [&](const auto& w) { return w.first == at; }))
                    E.wanted.emplace_back(at, why);
            };
            for (int64_t i = geni ? n : 0; i-- > 0;)
                if (ir.cell_row[(size_t) i] >= 0) {
                    want(i + 1, Engine::Kept::media, 0);
                    break;
                }
            int64_t d = 0;
            while (d < (int64_t) prev_prompt.size() && d < n - 1 && prev_prompt[(size_t) d] == ids[(size_t) d]) ++d;
            want(d, Engine::Kept::part, 256);
            const Model& md = *E.m;
            const bool tags = md.turn_start >= 0 && md.turn_middle >= 0 && md.turn_end >= 0;
            int64_t q = n;   // the last message's turn (the answer's: the generation prompt)
            for (int64_t i = n - 1; tags && i > (int64_t) reuse; --i)
                if (ids[(size_t) i] == md.turn_start) {
                    q = i;
                    break;
                }
            for (int64_t i = q - 1; tags && i > (int64_t) reuse; --i)
                if (ids[(size_t) i] == md.turn_end) {   // the message before it ends here
                    int64_t body = i;                   // its text's first cell
                    while (body > 0 && ids[(size_t) body - 1] != md.turn_middle) --body;
                    if (d - body >= 256) want(i - 4, Engine::Kept::part, 256);
                    int64_t open = body;   // the turn's first token, before its role
                    while (open > 0 && ids[(size_t) open] != md.turn_start) --open;
                    if (q < n) want(open, Engine::Kept::last, 256);
                    break;
                }
            std::sort(E.wanted.begin(), E.wanted.end());
            prev_prompt = ids;
        }
        E.img = geni ? &ir : nullptr;
        E.dump_route(2, 0, 0, nullptr);
        const int last = E.feed(ids.data() + reuse, fresh, [&](int64_t d) {
            done = d;
            if (done == fresh || ms_since(last_pp) > 250) {
                serve_print("PP %lld %lld %.0f %.1f\n", (long long) done, (long long) fresh, ms_since(t0),
                            (double) done / std::max(1e-9, ms_since(t0) / 1000.0));
                last_pp = Clock::now();
            }
            return !in->stop.load();
        });
        E.img = nullptr;
        E.wanted.clear();
        E.trace_print("the prompt");
        const bool cancelled = last < 0;
        const double prompt_ms = ms_since(t0);
        if (cancelled) {
            serve_print("DONE 0 %zu %.1f 0.0 cancel 0 0 %zu 0 0 0 0 0.0 %lld 0\n", ids.size(), prompt_ms, reuse,
                        (long long) done);
            return;
        }
        if (fresh > 0) {
            E.host_logits(last, lg);
            E.snapshot(lg);
        }
        E.settle(false);
        const int64_t landed0 = E.moves_landed;
        const int moved_at_start = E.move_at_start > 0 ? E.plan_moves(E.move_at_start, false) : 0;   // the slots toward
                                                                                                     // the prompt's
                                                                                                     // routing, beside
                                                                                                     // the answer
        std::mt19937_64 rng(lamina::program::draw_seed(smp.seed));
        std::vector<int32_t> hist;   // the penalties' history: the prompt's text (as llama.cpp's sampler, no picture)
        for (const int32_t t : ids)
            if (t >= 0) hist.push_back(t);
        const long long hits0 = E.hits, lookups0 = E.lookups;
        E.dump_route(3, 0, 0, nullptr);
        const auto d0 = Clock::now();
        long long produced = 0;
        const char* finish = "length";
        long long offered = 0, kept = 0;
        std::vector<int32_t> rows;   // drafted: the tokens the MTP block has not caught up with
        int32_t pre = -1;            // a target token sampled already (a window's first one that was no draft)
        // a token out: false when the answer ends there (finish set)
        auto emit = [&](int32_t t) {
            serve_print("T %d\n", t);
            ++produced;
            hist.push_back(t);
            if (t == eos) {
                finish = "stop";
                return false;
            }
            if (in->stop.load()) {
                finish = "cancel";
                return false;
            }
            if (produced >= max_new || E.pos() + kMaxT + 1 >= E.ctx - kMaxT) {
                finish = "length";
                return false;
            }
            return true;
        };
        while (true) {
            E.step_moves();
            const int nx = pre >= 0 ? pre : lamina::program::sample(lg, smp, hist, rng);
            pre = -1;
            if (nx < 0) {
                finish = "nonfinite";
                break;
            }
            if (!emit(nx)) break;
            const long long before = produced;
            if (E.drafts_on) {
                rows.push_back(nx);
                const int n = (int) std::min<long long>(drafts, max_new - produced);
                bool going = true;
                const Engine::Window w = E.drafted(
                    rows, n, lg, [&](std::vector<float>& l) { return lamina::program::sample(l, smp, hist, rng); },
                    [&](int32_t t) { return going = emit(t); });
                offered += (long long) w.drafts.size();
                kept += w.kept;
                if (!going) break;
                pre = w.next;
            } else {
                const int32_t one = nx;
                E.forward(&one, 1);
                E.host_logits(0, lg);
            }
            if (produced / 128 != before / 128) E.plan_moves(E.move_budget, true);
            E.maybe_lend(produced);
        }
        if (std::strcmp(finish, "nonfinite") == 0) {   // the state holds the non-finite rows: not continued
            E.forget();
            std::fprintf(stderr,
                         "lamina serve: non-finite logits: the request ends, and the next one reads its prompt from "
                         "the start\n");
        } else if (E.drafts_on && E.m->has_mtp && !rows.empty() && E.pos() >= (int64_t) rows.size()) {
            // the block caught up
            E.mtp_catch_up(rows.data(), (int64_t) rows.size(), E.pos() - (int64_t) rows.size());
        }
        const double decode_ms = ms_since(d0);
        E.trace_print("generation");
        const long long hits = E.hits - hits0, lookups = E.lookups - lookups0;
        serve_print("DONE %lld %zu %.1f %.1f %s %lld %lld %zu %lld %lld 0 0 0.0 %lld 0\n", produced, ids.size(),
                    prompt_ms, decode_ms, finish, kept, offered, reuse, hits, lookups, (long long) fresh);
        E.settle(false);
        const long long landed = (long long) (E.moves_landed - landed0);
        E.relax();
        const int queued = E.adapt ? E.plan_moves(512, true) : 0;
        E.save_profile(false);
        std::fprintf(stderr,
                     "lamina: expert copies: %d planned at the answer's start, %lld landed during it, %d planned "
                     "after it; MiB of VRAM free: %s\n",
                     moved_at_start, landed, queued, E.vram_free().c_str());
        std::fprintf(stderr,
                     "lamina: %zu prompt tokens (%zu reused) in %.1f ms, %lld generated in %.1f ms (%.2f tok/s), "
                     "%s%s\n",
                     ids.size(), reuse, prompt_ms, produced, decode_ms,
                     decode_ms > 0 ? 1000.0 * (double) produced / decode_ms : 0.0, finish,
                     offered > 0 ? (", drafts " + std::to_string(kept) + " of " + std::to_string(offered)).c_str()
                                 : "");
    }

    int run() {
        serve_print("READY %lld stop\n", (long long) (E.ctx - kMaxT));
        const lamina::program::ServeIo& io = lamina::program::serve_io();
        // Ctrl+C in the console ends the engine as QUIT does, before any request queued (the running one stops at its
        // next token), then the exit work runs (the learned profile).  In-process the host handles it.
        lamina::program::ServeInput input(true);
        in = &input;
        for (;;) {
            std::string line;
            if (!in->next(line)) return 0;
            if (line.empty()) continue;
            if (line.rfind("GEN ", 0) == 0 || line.rfind("GENI ", 0) == 0) {
                request(line);
            } else if (line.rfind("LEND ", 0) == 0) {
                lend(line, io.in_process);
            } else if (line == "UNLEND") {
                enc_back();
                serve_print("UNLENT\n");
            } else if (line == "QUIT") {
                in->quit();
            } else {
                serve_print("ERR expected: GEN <max_new> [key=value ...] <id,id,...> or GENI <max_new> [keys] <file> "
                            "<id,id,...>\n");
            }
        }
    }
};

int run_engine(int argc, char** argv, bool check_only) {
    std::string model, tokens, force, dump_full, emb_file;
    bool vision = false;
    int64_t max_ctx = 8192;
    int n_gen = 64, threads = 0;
    bool serve = false;
    double expert_vram = -1;
    std::string profile, profile_save;
    double save_every = 1.0;
    bool graphs = true;
    int prefill = 4096, prefill_min = 24, prefill_pass = 16384;
    double arena_pin = 72;
    double park_mib = 16384, reserve_mib = 6144;
    std::string spill_dir;    // --conversation-spill-dir: the park's disk tier (empty: none)
    int64_t spill_mib = -1;   // --conversation-spill-mib: its files' MiB at most (-1: its default; 0: none)
    bool kv_grow = true;
    bool file_cache = true;   // the model file read through the OS file cache (--no-file-cache: past it)
    bool arena_place = false;
    std::string kv = "fp16";
    bool use_mtp = true;
    std::string dfly_path, dfly_flag = "--drafter";   // (the name the line gave it: --drafter or an earlier one)
    int drafts = 0;
    float spec_min_p = 0.f;
    int lookup = 16;
    int warm_ms = 500;
    int prefill_lend = 0;
    bool adapt = true;
    bool tier_doorbell = false;
    std::vector<int> tier_devs;
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
        else if (a == "--max-context") max_ctx = std::atoll(next());
        else if (a == "--tokens") tokens = next();
        else if (a == "--tokens-file") {
            const char* f = next();
            FILE* fp = std::fopen(f, "rb");
            if (!fp) {
                std::fprintf(stderr, "lamina: cannot open %s\n", f);
                return 2;
            }
            char buf[1 << 16];
            size_t n;
            tokens.clear();   // (the last of --tokens and --tokens-file is the prompt)
            while ((n = std::fread(buf, 1, sizeof buf, fp)) > 0) tokens.append(buf, n);
            std::fclose(fp);
        } else if (a == "--n") n_gen = std::atoi(next());
        else if (a == "--force") force = next();
        else if (a == "--dump-full-logits") dump_full = next();
        else if (a == "--cpu-threads" || a == "--pool-workers") threads = std::atoi(next());
        else if (a == "--serve") serve = true;
        else if (a == "--expert-vram") expert_vram = std::atof(next());
        else if (a == "--expert-profile") profile = next();
        else if (a == "--expert-profile-save") profile_save = next();
        else if (a == "--expert-profile-save-every") save_every = std::atof(next());
        else if (a == "--no-graph") graphs = false;
        else if (a == "--prefill") prefill = std::atoi(next());   // rows a prompt chunk (0: the decode's forwards)
        else if (a == "--prefill-min")
            prefill_min = std::atoi(next());                                // a shorter prompt part:
                                                                            // the decode's forwards
        else if (a == "--prefill-pass") prefill_pass = std::atoi(next());   // rows of a long part's chunks (0: none)
        else if (a == "--arena-pin") arena_pin = std::atof(next());         // GiB of the arena in pinned memory at most
        else if (a == "--park-mib" || a == "--conversation-cache-mib") park_mib = std::atof(next());
        else if (a == "--ram-reserve-mib") reserve_mib = std::atof(next());
        else if (a == "--conversation-spill-dir") spill_dir = next();
        else if (a == "--conversation-spill-mib") spill_mib = std::max<int64_t>(0, std::atoll(next()));
        else if (a == "--no-kv-grow") kv_grow = false;
        else if (a == "--file-cache") file_cache = true;
        else if (a == "--arena-place")
            arena_place = true;   // the pinned places timed, the CPU's experts in the fast ones
        else if (a == "--no-arena-place") arena_place = false;
        else if (a == "--no-file-cache") file_cache = false;
        else if (a == "--kv") kv = next();           // the sparse layers' K / V: fp16 or q8_0
        else if (a == "--no-mtp") use_mtp = false;   // the MTP block not loaded (no drafts)
        else if (a == "--drafter" || a == "--dfly" || a == "--mtp" || a == "--dspark") {
            // DFly2's drafts (the checkpoint's last shard) in place of the MTP block; --mtp N, a number, is the earlier
            // releases' --drafts-per-step, taken with a warning so their configs still start
            const std::string v = next();
            if (!lamina::is_drafter_flag(a, v)) {
                drafts = std::max(0, std::min(kMaxT - 1, std::atoi(v.c_str())));
                std::fprintf(stderr,
                             "lamina: --mtp %s is --drafts-per-step %s on this path (--drafter names a drafter "
                             "file)\n",
                             v.c_str(), v.c_str());
            } else {
                dfly_flag = a;
                dfly_path = v;
            }
        } else if (a == "--drafts-per-step") drafts = std::max(0, std::min(kMaxT - 1, std::atoi(next())));
        else if (a == "--spec-min-p") spec_min_p = (float) std::atof(next());
        else if (a == "--spec-lookup") lookup = std::max(0, std::atoi(next()));   // prompt lookup's match (0: none)
        else if (a == "--warm-ms")
            warm_ms = std::max(0, std::atoi(next()));   // a request after a pause: the cards kept running
        else if (a == "--prefill-lend")
            prefill_lend = std::max(0, std::atoi(next()));   // the prompt buffers lent after N tokens
        else if (a == "--adapt-every")
            adapt = std::atoi(next()) != 0;                      // 0: the start's placement stays (Engine::adapt)
        else if (a == "--tier-doorbell") tier_doorbell = true;   // the tiers' shares queued behind doorbells
        else if (a == "--kv-grow") kv_grow = true;
        else if (a == "--vision") vision = true;    // pictures: GENI requests, --emb (the rows lamina-vision's)
        else if (a == "--emb") emb_file = next();   // generate mode: --tokens' pictures, a lamina-vision file
        else if (a == "--tier-gpus") {              // the expert tiers' GPUs (CUDA ordinals, the family switch's)
            for (const int32_t d : lamina::program::parse_ids(next())) tier_devs.push_back((int) d);
        } else {
            std::fprintf(stderr, "lamina: Hy4-flash's path does not take %s\n", a.c_str());
            return 2;
        }
    }
    if (model.empty()) {
        std::fprintf(stderr, "lamina: --native MODEL is required\n");
        return 2;
    }
    if (kv != "fp16" && kv != "q8_0") {
        std::fprintf(stderr, "lamina: --kv takes fp16 or q8_0 on this path\n");
        return 2;
    }
    std::string err;
    if (!dfly_path.empty()) {   // DFly's file read now: its config and the shard's header
        if (const std::string e = lamina::drafter_mismatch(
                dfly_flag, dfly_path, lamina::DrafterKind::DFly2, "Hy4-flash",
                "DFly2, the tencent/Hy4-flash checkpoint's last shard with its config.json beside it (the model file's "
                "MTP block without one)");
            !e.empty()) {
            std::fprintf(stderr, "lamina: %s\n", e.c_str());
            return 2;
        }
        lamina::hy4::DFlyConfig c;
        lamina::SafeTensors st;
        if (!lamina::hy4::read_dfly_config(std::filesystem::path(dfly_path).parent_path().string(), c, err) ||
            !st.open(dfly_path, err)) {
            std::fprintf(stderr, "lamina: %s\n", err.c_str());
            return 2;
        }
    }
    if (check_only) return 0;
    Model m;
    if (!load(model, m, 8, arena_pin, use_mtp && dfly_path.empty(), file_cache, {profile_save, profile}, serve, err) ||
        (!dfly_path.empty() && !load_dfly(dfly_path, m, err))) {
        std::fprintf(stderr, "lamina: %s\n", err.c_str());
        return 1;
    }
    Engine E;
    E.profile_path = profile;
    E.save_path = profile_save;
    E.save_every_min = save_every;
    E.use_graphs = graphs && E.trace < 3;
    E.elastic = kv_grow;
    E.kv_q8 = kv == "q8_0";
    E.drafts_on = (m.has_mtp || m.dfly.on) && drafts > 0;
    E.spec_min_p = spec_min_p;
    E.lookup_min = lookup;
    E.warm_ms = warm_ms;
    E.lend_after = prefill_lend;
    E.adapt = adapt;
    if (!adapt) E.move_at_start = E.move_budget = 0;
    E.tier_doorbell = tier_doorbell;
    if (lookup > 0)
        E.sfx = std::make_unique<lamina::spec::SuffixDrafter>(lookup, 32, (size_t) std::max<int64_t>(4096, max_ctx));
    E.vision = vision || !emb_file.empty();
    if (E.vision && m.image_pad < 0) {
        std::fprintf(stderr,
                     "lamina: the model's vocabulary has no image pad token, so --vision has no cells to fill\n");
        return 2;
    }
    E.init(m, max_ctx, threads);
    E.park_budget = (size_t) (std::max(0.0, park_mib) * (double) (1 << 20));
    E.ram_reserve = (size_t) (std::max(0.0, reserve_mib) * (double) (1 << 20));
    E.open_disk(spill_dir, spill_mib);   // the park's disk tier
    Prompt P(E);                         // its buffers before the slots take the GPUs' free memory
    // the MTP block kept up catches up over a chunk's rows at once: no passes past --prefill's
    const bool mtp_kept = E.drafts_on && m.has_mtp;
    if (prefill > 0 && P.init(prefill, tier_devs, mtp_kept ? 0 : prefill_pass)) {
        E.chunk_fn = [&P](const int32_t* t, int n) { P.run(t, n); };
        E.chunk_cap = P.cap;
        E.pass_cap = P.pass;
        E.base_fn = [&P](bool give) {
            if (give) P.base_give();
            else P.base_take();
        };
        E.pass_fn = [&P](bool on) {
            if (on) return P.pass_begin();
            P.pass_end();
            return false;
        };
        E.chunk_min = std::max(1, prefill_min);
    }
    E.place_slots(expert_vram, tier_devs);
    E.warm_streams();
    if (arena_place) {
        const char* dry = std::getenv("LAMINA_ARENA_PLACE_DRY");
        E.place_arena(dry != nullptr && std::atoi(dry) != 0);
    }
    if (serve) {
        Serve sv(E, m.eos, drafts);
        const int rc = sv.run();
        E.save_profile(true);
        return rc;
    }
    std::vector<int32_t> prompt = lamina::program::parse_ids(tokens.c_str());
    const std::vector<int32_t> forced = lamina::program::parse_ids(force.c_str());
    if (prompt.empty()) {
        std::fprintf(stderr, "lamina: --tokens is required\n");
        return 2;
    }
    if (!forced.empty()) n_gen = (int) forced.size() + 1;
    for (const std::vector<int32_t>* v : {(const std::vector<int32_t>*) &prompt, &forced})
        for (const int32_t t : *v)
            if (t < 0 || t >= m.g.n_vocab) {
                std::fprintf(stderr, "lamina: the token %d is outside the vocabulary (%lld)\n", (int) t,
                             (long long) m.g.n_vocab);
                return 2;
            }
    if ((int64_t) prompt.size() + n_gen + kMaxT > E.ctx) {
        std::fprintf(stderr, "lamina: the prompt (%zu tokens) and %d generated exceed the context (%lld)\n",
                     prompt.size(), n_gen, (long long) E.ctx);
        return 2;
    }
    ImgReq ir;
    if (!emb_file.empty()) {
        if (const std::string ve =
                lamina::program::image_cells(emb_file, prompt, {m.image_pad, m.video_pad}, m.g.n_embd, {}, ir);
            !ve.empty()) {
            std::fprintf(stderr, "lamina: --emb: %s\n", ve.c_str());
            return 2;
        }
        E.img = &ir;
        E.mrope_set(&ir);
    }
    FILE* ff = dump_full.empty() ? nullptr : std::fopen(dump_full.c_str(), "wb");
    const auto t0 = Clock::now();
    E.dump_route(2, 0, 0, nullptr);
    const int last = E.feed(prompt.data(), (int64_t) prompt.size(), nullptr);
    E.img = nullptr;
    std::vector<float> lg;
    E.host_logits(last, lg);
    const double pp_ms = ms_since(t0);
    E.trace_print("the prompt");
    const double cpu0 = E.cpu_ms, tab0 = E.table_ms;
    E.dump_route(3, 0, 0, nullptr);
    E.settle(false);
    if (E.move_at_start > 0) {
        const int q = E.plan_moves(E.move_at_start, false);
        std::fprintf(stderr, "lamina: %d expert copies queued at the answer's start\n", q);
    }
    const auto t1 = Clock::now();
    std::vector<int32_t> gen;
    int64_t offered = 0, kept = 0, windows = 0;
    if (E.drafts_on && forced.empty()) {   // drafted: windows of the token and the drafter's drafts, greedy
        std::vector<int32_t> rows;
        int32_t x = argmax(lg);
        while (true) {
            E.step_moves();
            gen.push_back(x);
            if (ff) std::fwrite(lg.data(), 4, lg.size(), ff);
            if ((int) gen.size() >= n_gen) break;
            rows.push_back(x);
            const int n = std::min(drafts, n_gen - (int) gen.size());
            const size_t before = gen.size();
            const Engine::Window w = E.drafted(
                rows, n, lg, [](const std::vector<float>& l) { return argmax(l); },
                [&](int32_t t) {
                    gen.push_back(t);
                    if (ff) std::fwrite(lg.data(), 4, lg.size(), ff);
                    return (int) gen.size() < n_gen;
                });
            ++windows;
            offered += (int64_t) w.drafts.size();
            kept += w.kept;
            if ((int) gen.size() >= n_gen) break;
            x = w.next >= 0 ? w.next : argmax(lg);
            if (gen.size() / 128 != before / 128) E.plan_moves(E.move_budget, true);
            E.maybe_lend((int64_t) gen.size());
        }
    } else {
        for (int i = 0; i < n_gen; ++i) {
            const int a = argmax(lg);
            gen.push_back(a);
            if (ff) std::fwrite(lg.data(), 4, lg.size(), ff);
            if (i + 1 == n_gen) break;
            const int32_t nx = i < (int) forced.size() ? forced[(size_t) i] : a;
            E.step_moves();
            E.forward(&nx, 1);
            E.host_logits(0, lg);
            if ((i + 1) % 128 == 0) E.plan_moves(E.move_budget, true);
        }
    }
    const double tg_ms = ms_since(t1);
    E.trace_print("generation");
    if (ff) std::fclose(ff);
    serve_print("gen:");
    for (int t : gen) serve_print(" %d", t);
    serve_print("\n");
    std::fprintf(stderr,
                 "lamina: prompt %zu tokens in %.1f ms (%.1f tok/s), %zu generated in %.1f ms (%.2f tok/s); "
                 "the CPU's experts %.1f ms, the table %.1f ms while generating; VRAM hits %.1f%%\n",
                 prompt.size(), pp_ms, prompt.size() / (pp_ms / 1000.0), gen.size(), tg_ms,
                 gen.size() / (tg_ms / 1000.0), E.cpu_ms - cpu0, E.table_ms - tab0,
                 E.lookups ? 100.0 * (double) E.hits / (double) E.lookups : 0.0);
    for (const Engine::Gpu& gp : E.gpus)
        std::fprintf(stderr, "lamina: GPU %d, %s: %.1f%% of the routed entries\n", gp.dev, gp.name.c_str(),
                     E.lookups ? 100.0 * (double) gp.hits / (double) E.lookups : 0.0);
    std::fprintf(stderr, "lamina: MiB of VRAM free: %s\n", E.vram_free().c_str());
    if (windows > 0)
        std::fprintf(stderr,
                     "lamina: %lld windows (%lld looked up), %lld drafts offered, %lld kept (%.1f%%), %.2f tokens "
                     "a window\n",
                     (long long) windows, (long long) E.looked_windows, (long long) offered, (long long) kept,
                     offered ? 100.0 * (double) kept / (double) offered : 0.0, (double) gen.size() / (double) windows);
    E.save_profile(true);
    return 0;
}

}   // namespace

int lamina::hy4::run(int argc, char** argv) {
    return lamina::core::run_owned([&] { return run_engine(argc, argv, false); });
}

int lamina::hy4::check_args(int argc, char** argv) {
    return lamina::core::check_owned([&] { return run_engine(argc, argv, true); });
}
