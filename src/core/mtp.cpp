// src/core/mtp.cpp - see include/lamina/core/mtp.hpp.
#include "lamina/core/mtp.hpp"

#include "lamina/artifact/draft_layer.hpp"
#include "lamina/artifact/gguf_reader.hpp"
#include "lamina/core/cuda_host.hpp"
#include "lamina/core/draft_pass.hpp"
#include "lamina/core/native_head.hpp"
#include "lamina/kernels/bf16_gemv.hpp"
#include "lamina/kernels/cpu/expert.hpp"
#include "lamina/kernels/draft_proxy.hpp"
#include "lamina/kernels/elementwise.hpp"
#include "lamina/kernels/fused_gr.hpp"
#include "lamina/kernels/gr.hpp"
#include "lamina/kernels/kv_q8.hpp"
#include "lamina/kernels/native_mmvq.hpp"
#include "lamina/kernels/native_moe.hpp"
#include "lamina/kernels/native_qsa.hpp"
#include "lamina/kernels/native_rope.hpp"
#include "lamina/kernels/native_router.hpp"
#include "lamina/kernels/qsa.hpp"
#include "lamina/kernels/qsa_decode_attn.hpp"
#include "lamina/kernels/quantize_act.hpp"
#include "lamina/kernels/rope.hpp"
#include "lamina/kernels/s2_expert_grouped.hpp"
#include "lamina/kernels/sampler.hpp"
#include "lamina/kernels/shared_expert.hpp"
#include "lamina/kernels/verify_kernels.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <vector>

namespace lamina::core {
namespace {

constexpr float EPS = 1e-6f;
constexpr int GGML_Q8_0 = 8;
using Clock = std::chrono::steady_clock;

bool read_file(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const std::streamsize n = f.tellg();
    f.seekg(0);
    out.resize((size_t) n);
    return (bool) f.read((char*) out.data(), n);
}

/// One Q2_0 expert's GGUF blocks -> its blob (cpu/expert.hpp; tools/mtp_rt.py's blob_of).  `gu`: 2*FF rows of H/64
/// blocks, the gate's rows then the up's; `dn`: H rows of FF/64 blocks; a block is its fp16 scale and 16 bytes of
/// codes.  The blob interleaves the gate/up rows (2r = gate r, 2r + 1 = up r) and keeps every row's codes in one plane,
/// its scales in another: [gate/up codes | down codes | gate/up scales | down scales].
void q2_blob(const uint8_t* gu, const uint8_t* dn, uint8_t* blob) {
    namespace X = lamina::kernels::cpu;
    uint8_t* codes = blob + X::O_GU_CODES;
    uint8_t* scales = blob + X::O_GU_SCALES;
    for (int r = 0; r < 2 * X::FF; ++r) {
        const uint8_t* row = gu + (size_t) ((r & 1) ? X::FF + r / 2 : r / 2) * X::SC_GU * X::BB;
        for (int b = 0; b < X::SC_GU; ++b, codes += X::BB - 2, scales += 2) {
            std::memcpy(codes, row + b * X::BB + 2, X::BB - 2);
            std::memcpy(scales, row + b * X::BB, 2);
        }
    }
    codes = blob + X::O_D_CODES;
    scales = blob + X::O_D_SCALES;
    for (int r = 0; r < X::H; ++r) {
        const uint8_t* row = dn + (size_t) r * X::SC_D * X::BB;
        for (int b = 0; b < X::SC_D; ++b, codes += X::BB - 2, scales += 2) {
            std::memcpy(codes, row + b * X::BB + 2, X::BB - 2);
            std::memcpy(scales, row + b * X::BB, 2);
        }
    }
}

}   // namespace

MtpDrafter::~MtpDrafter() {
    if (cs_) cudaStreamSynchronize(cs_);
    for (auto& e : round_exec_)
        if (e) cudaGraphExecDestroy(e);
    for (auto& e : step_exec_)
        if (e) cudaGraphExecDestroy(e);
    if (cs_) cudaStreamDestroy(cs_);
    if (side_) cudaStreamDestroy(side_);
    if (sh_fork_) cudaEventDestroy(sh_fork_);
    if (sh_join_) cudaEventDestroy(sh_join_);
    if (dense_) cudaFree(dense_);
    if (experts_) cudaFree(experts_);
    if (state_arena_) cudaFree(state_arena_);
    if (arena_) cudaFree(arena_);
    if (head_logits_) cudaFree(head_logits_);
    if (dhead_) cudaFree(dhead_);
    if (dvocab_) cudaFree(dvocab_);
    if (proxy_) cudaFree(proxy_);
    if (proxy_scratch_) cudaFree(proxy_scratch_);
    void* hosts[] = {h_tok_, h_step_, h_pos_, h_row_, h_out_, h_prob_};
    for (void* h : hosts)
        if (h) cudaFreeHost(h);
    qsa_state_free(st_);
}

const float* MtpDrafter::f32(const char* name) const {
    for (const auto& t : tensors_)
        if (t.name == name && t.kind == "f32") return (const float*) (dense_ + t.off);
    return nullptr;
}
const uint16_t* MtpDrafter::bf16(const char* name) const {
    for (const auto& t : tensors_)
        if (t.name == name && t.kind == "bf16") return (const uint16_t*) (dense_ + t.off);
    return nullptr;
}
const void* MtpDrafter::q8(const char* name) const {
    for (const auto& t : tensors_)
        if (t.name == name && t.kind == "q8_0") return dense_ + t.off;
    return nullptr;
}

bool MtpDrafter::load(const std::string& path, const ModelGeometry& g, SessionState& ss, int max_t, std::string& err,
                      int64_t window) {
    g_ = &g;
    ss_ = &ss;
    max_t_ = max_t;
    if (max_t < 1 || max_t > lamina::kernels::kVerifyMaxT) {
        err = "mtp: max_t out of range";
        return false;
    }
    const Clock::time_point t_load = Clock::now();
    uint64_t dense_bytes = 0;
    std::error_code ec;
    if (!(std::filesystem::is_directory(path, ec) ? load_folder(path, g, dense_bytes, err)
                                                  : load_file(path, g, dense_bytes, err)))
        return false;
    const char* required[] = {"fc_embedding.weight",
                              "fc_hidden.weight",
                              "self_attn.q_proj.weight",
                              "self_attn.k_proj.weight",
                              "self_attn.v_proj.weight",
                              "self_attn.o_proj.weight",
                              "mlp.shared_expert.gate_proj.weight",
                              "mlp.shared_expert.up_proj.weight",
                              "mlp.shared_expert.down_proj.weight"};
    for (const char* n : required)
        if (!q8(n)) {
            err = std::string("mtp: ") + n + " is missing (q8_0)";
            return false;
        }

    // ---- the layer's own K/V (dense attention: no indexer state is read), FP16 or INT8 (qsa_draft_kv_format)
    const lamina::kernels::QsaShapes s = lamina::kernels::qsa_shapes(g);
    const int64_t max_cells = ss.qsa_states[0].max_cells;
    const uint64_t sb = qsa_state_bytes(g, max_cells, false, qsa_draft_kv_format());
    if (cudaMalloc(&state_arena_, sb) != cudaSuccess) {
        err = "mtp: the K/V state does not fit";
        return false;
    }
    if (qsa_state_init(g, max_cells, state_arena_, st_, &ss.qsa_states[0], qsa_draft_kv_format()) == 0) {
        err = "mtp: state init failed";
        return false;
    }
    qsa_state_zero(st_, g, nullptr);
    cudaDeviceSynchronize();
    vram_ += sb;

    // ---- buffers
    window_ = (window > 0 && window < max_cells) ? window : 0;
    cap_ = (((window_ > 0 ? window_ : max_cells) + 63) / 64) * 64;
    attn_scratch_floats_ = (int64_t) lamina::kernels::qsa_decode_attn_scratch_floats(cap_, s);
    const uint64_t T = (uint64_t) max_t, N = (uint64_t) g.n_embd, HC = (uint64_t) g.hc, K = (uint64_t) ss.k;
    const uint64_t NH = (uint64_t) g.n_head, HD = (uint64_t) g.head_dim, NKV = (uint64_t) g.n_head_kv;
    const uint64_t R2 = 2 * T;   // step/pos rows: T catch-up rows + up to T-2 chain steps
    bool ok = mapped(T * 4 + 64, (void**) &h_tok_, (void**) &m_tok_) &&
              mapped(R2 * 4 * 4 + 64, (void**) &h_step_, (void**) &m_step_) &&
              mapped(R2 * NH * 4 + 64, (void**) &h_pos_, (void**) &m_pos_) &&
              mapped(64, (void**) &h_row_, (void**) &m_row_) &&
              mapped(T * 4 + 64, (void**) &h_out_, (void**) &m_out_) &&
              mapped(T * 4 + 64, (void**) &h_prob_, (void**) &m_prob_);
    if (!ok) {
        err = "mtp: mapped staging failed";
        return false;
    }
    auto carve = [&](Bump& b) {
        tok_ = b.take<int32_t>(T);
        step_ = b.take<int32_t>(R2 * 4);
        pos_ = b.take<int32_t>(R2 * NH);
        row_ = b.take<int32_t>(4);
        ident_ = b.take<int32_t>(T * (uint64_t) cap_);
        Rin_ = b.take<float>(T * HC * N);
        R_ = b.take<float>(T * HC * N);
        emb_ = b.take<float>(T * N);
        en_ = b.take<float>(T * N);
        e2_ = b.take<float>(T * N);
        hn_ = b.take<float>(T * HC * N);
        h2_ = b.take<float>(T * HC * N);
        mixed_ = b.take<float>(T * N);
        inj_ = b.take<float>(T * HC);
        inj2_ = b.take<float>(T * HC);
        lo_ = b.take<float>(T * (uint64_t) g.hc_lr);
        rs_ = b.take<float>(T * HC);
        bo_ = b.take<float>(T * N);
        grs_ = (float*) b.take<uint8_t>(lamina::kernels::fused_gr_scratch_bytes());
        xq_ = b.take<uint8_t>(lamina::kernels::native_q8_1_bytes((int) (NH * HD), 8));
        qfull_ = b.take<float>(T * NH * 2 * HD);
        qcur_ = b.take<float>(T * NH * HD);
        kcur_ = b.take<float>(T * NKV * HD);
        vcur_ = b.take<float>(T * NKV * HD);
        attn32_ = b.take<float>(T * NH * HD);
        attn_scratch_ = b.take<float>((uint64_t) attn_scratch_floats_);   // the full layer runs one row at a time
        logits_ = b.take<float>(T * (uint64_t) g.n_expert);
        w_ = b.take<float>(T * K);
        ids_ = b.take<int32_t>(T * K);
        shared_ = b.take<float>(T * N);
        parts_ = b.take<float>(T * K * N);
        y_ = b.take<float>(T * N);
        sample_ = b.take<float>(T * N);
        hit_slot_ = b.take<int32_t>(T * K);
        hit_dst_ = b.take<int32_t>(T * K);
        hit_count_ = b.take<int32_t>(4);
        grp_ptr_ = b.take<unsigned long long>(T * K);
        grp_start_ = b.take<int32_t>(T * K + 1);
        grp_counts_ = b.take<int32_t>(4);
        hit_xq_ = b.take<uint8_t>(T * (N / 32) * 34);
        hit_xs_ = b.take<float>(T * (N / 32));
        nat_xq_ = b.take<uint8_t>(T * (N / 32) * 36);
        rseq_ = b.take<uint32_t>(4);
        rcount_ = b.take<unsigned>(4);
        hit_scratch_ = b.take<uint8_t>(
            std::max<uint64_t>(lamina::kernels::moe_hit_grouped_scratch_bytes((int64_t) (T * K), g.n_embd, g.n_ff),
                               lamina::kernels::native_expert_scratch_bytes((int64_t) (T * K), g.n_ff)));
        sh_scratch_ = (float*) b.take<uint8_t>(lamina::kernels::shared_expert_scratch_bytes(g.n_ff));
        x_bf16_ = b.take<uint16_t>(N);
        out_ids_ = b.take<int32_t>(T + 4);
        probs_ = b.take<float>(T + 4);
        arg_scratch_ = b.take<uint8_t>(lamina::kernels::argmax_rows_scratch_bytes((int) T));
        top_scratch_ = b.take<uint8_t>(lamina::kernels::row_top_prob_scratch_bytes((int) T));
        dummy_inj_ = b.take<float>(HC);
    };
    Bump count;
    carve(count);
    if (cudaMalloc(&arena_, count.used) != cudaSuccess) {
        err = "mtp: buffers do not fit";
        return false;
    }
    cudaMemset(arena_, 0, count.used);
    Bump real;
    real.base = (uint8_t*) arena_;
    carve(real);
    vram_ += count.used;
    {
        std::vector<int32_t> id((size_t) (T * (uint64_t) cap_));
        for (uint64_t t = 0; t < T; ++t)
            for (int64_t i = 0; i < cap_; ++i) id[(size_t) (t * (uint64_t) cap_ + (uint64_t) i)] = (int32_t) i;
        cudaMemcpy(ident_, id.data(), id.size() * 4, cudaMemcpyHostToDevice);
    }
    if (cudaStreamCreateWithFlags(&cs_, cudaStreamNonBlocking) != cudaSuccess ||
        cudaStreamCreateWithFlags(&side_, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&sh_fork_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&sh_join_, cudaEventDisableTiming) != cudaSuccess) {
        err = "mtp: streams";
        return false;
    }
    char fmts[48] = "Q2_0 blobs";
    if (nat_.gu_type >= 0) std::snprintf(fmts, sizeof fmts, "ggml types %d/%d", nat_.gu_type, nat_.d_type);
    std::fprintf(stderr,
                 "lamina mtp: draft layer loaded, %.0f MiB of VRAM (experts %.0f as %s, dense %.0f), in %.2f s\n",
                 (double) vram_ / 1048576.0, (double) g.n_expert * blob_ / 1048576.0, fmts,
                 (double) dense_bytes / 1048576.0, std::chrono::duration<double>(Clock::now() - t_load).count());
    return true;
}

bool MtpDrafter::load_file(const std::string& path, const ModelGeometry& g, uint64_t& dense_bytes, std::string& err) {
    std::unique_ptr<GgufFile> f;   // mapped while its tensors go to VRAM
    try {
        f = std::make_unique<GgufFile>(path);
    } catch (const std::exception& e) {
        err = "mtp: " + path + ": " + e.what();
        return false;
    }
    const MetaValue* arch = f->get("general.architecture");
    const MetaValue* ver = f->get("lamina.mtp.version");
    if (arch == nullptr || arch->s != kDraftLayerArch || ver == nullptr || !ver->is_num() ||
        ver->u != kDraftLayerVersion) {
        err = "mtp: " + path + " is not a draft layer this engine reads (general.architecture " + kDraftLayerArch +
              ", lamina.mtp.version " + std::to_string(kDraftLayerVersion) + ": tools/mtp_rt.py writes one)";
        return false;
    }
    const uint64_t payload = f->file_size() - f->data_start();
    auto bytes_of = [&](const TensorInfo& t) -> uint64_t {   // 0: not a whole tensor of a known type inside the file
        const uint64_t b = tensor_payload_bytes(t);
        return b != 0 && t.offset <= payload && b <= payload - t.offset ? b : 0;
    };
    const TensorInfo* gu = f->find("mlp.experts.gate_up_proj");
    const TensorInfo* dn = f->find("mlp.experts.down_proj");
    const TensorInfo* dv = f->find("draft_vocab");
    // ---- the dense weights in one VRAM buffer, in the file's order, each at a 256-byte offset (zeros between)
    {
        std::vector<const TensorInfo*> dense;
        for (const TensorInfo& t : f->tensors())
            if (&t != gu && &t != dn && &t != dv) dense.push_back(&t);
        std::sort(dense.begin(), dense.end(),
                  [](const TensorInfo* a, const TensorInfo* b) { return a->offset < b->offset; });
        for (const TensorInfo* t : dense) {
            const char* kind = t->type == 0 ? "f32" : t->type == 30 ? "bf16" : t->type == GGML_Q8_0 ? "q8_0" : nullptr;
            const uint64_t b = bytes_of(*t);
            if (kind == nullptr || t->shape.size() > 2 || b == 0) {
                err = "mtp: " + path + ": " + t->name + " (" + t->type_name() +
                      ") is not a dense tensor the layer "
                      "takes (F32, BF16 or Q8_0, one or two dimensions, inside the file)";
                return false;
            }
            dense_bytes = (dense_bytes + 255) / 256 * 256;
            tensors_.push_back({t->name, kind, t->shape.size() > 1 ? (int64_t) t->shape[1] : 1, (int64_t) t->shape[0],
                                dense_bytes, b});
            dense_bytes += b;
        }
        if (dense.empty() || cudaMalloc((void**) &dense_, dense_bytes) != cudaSuccess) {
            err = "mtp: dense weights do not fit";
            return false;
        }
        cudaMemset(dense_, 0, dense_bytes);
        for (size_t i = 0; i < dense.size(); ++i)
            cudaMemcpy(dense_ + tensors_[i].off, f->tensor_data(*dense[i]), tensors_[i].bytes, cudaMemcpyHostToDevice);
        vram_ += dense_bytes;
    }
    // ---- the 512 routed experts, one blob each: Q2_0 in the blob layout of cpu/expert.hpp (q2_blob), other formats
    //      as their GGUF blocks [gate | up | down] for the native expert kernels
    {
        const uint64_t E = (uint64_t) g.n_expert, N = (uint64_t) g.n_embd, F = (uint64_t) g.n_ff;
        if (gu == nullptr || dn == nullptr || gu->shape != std::vector<uint64_t>{N, 2 * F, E} ||
            dn->shape != std::vector<uint64_t>{F, N, E} || bytes_of(*gu) == 0 || bytes_of(*dn) == 0) {
            err = "mtp: " + path + " has no mlp.experts.gate_up_proj [" + std::to_string(N) + ", " +
                  std::to_string(2 * F) + ", " + std::to_string(E) + "] and mlp.experts.down_proj [" +
                  std::to_string(F) + ", " + std::to_string(N) + ", " + std::to_string(E) + "] of a known format";
            return false;
        }
        const bool q2 = gu->type == 42 && dn->type == 42;
        if (q2) {
            blob_ = lamina::kernels::cpu::BLOB;
        } else {
            if (!lamina::kernels::native_expert_supported((int) gu->type, (int) dn->type, g.n_embd, g.n_ff)) {
                err = std::string("mtp: the experts' formats, ") + gu->type_name() + " / " + dn->type_name() +
                      ", have no GPU expert kernels";
                return false;
            }
            // the fast kernels under either --expert-math: drafts are verified, and both runs of a comparison
            // compute them here
            nat_ = lamina::kernels::native_expert_layout((int) gu->type, (int) dn->type, g.n_embd, g.n_ff);
            blob_ = nat_.bytes;
        }
        const uint64_t gu_e = bytes_of(*gu) / E, dn_e = bytes_of(*dn) / E;   // an expert's bytes of each
        if (blob_ != gu_e + dn_e) {
            err = "mtp: the experts' blocks do not make the expert layout's blobs";
            return false;
        }
        const uint64_t bytes = E * blob_;
        if (cudaMalloc((void**) &experts_, bytes) != cudaSuccess) {
            err = "mtp: the 512 experts do not fit in VRAM";
            return false;
        }
        const uint8_t* gsrc = f->tensor_data(*gu);
        const uint8_t* dsrc = f->tensor_data(*dn);
        constexpr uint64_t kPer = 32;   // experts a copy
        std::vector<uint8_t> chunk((size_t) (kPer * blob_));
        for (uint64_t e0 = 0; e0 < E; e0 += kPer) {
            const uint64_t n = std::min(kPer, E - e0);
            for (uint64_t e = 0; e < n; ++e) {
                uint8_t* blob = chunk.data() + e * blob_;
                const uint8_t *ge = gsrc + (e0 + e) * gu_e, *de = dsrc + (e0 + e) * dn_e;
                if (q2) {
                    q2_blob(ge, de, blob);
                } else {
                    std::memcpy(blob, ge, gu_e);
                    std::memcpy(blob + gu_e, de, dn_e);
                }
            }
            cudaMemcpy(experts_ + e0 * blob_, chunk.data(), n * blob_, cudaMemcpyHostToDevice);
        }
        vram_ += bytes;
    }
    // ---- the draft head's token ids (bind_head gathers their rows of the main head); without them the whole head
    if (dv != nullptr) {
        if (dv->type != 26 || dv->shape.size() != 1 || bytes_of(*dv) == 0) {
            err = "mtp: " + path + "'s draft_vocab is not a one-dimensional I32 tensor";
            return false;
        }
        dvocab_ids_.resize((size_t) dv->shape[0]);
        std::memcpy(dvocab_ids_.data(), f->tensor_data(*dv), dvocab_ids_.size() * sizeof(int32_t));
    }
    f.reset();
    return true;
}

bool MtpDrafter::load_folder(const std::string& dir, const ModelGeometry& g, uint64_t& dense_bytes, std::string& err) {
    // ---- the index and the dense weights, dense.bin as it is (its offsets are the VRAM buffer's)
    {
        std::ifstream idx(dir + "/dense.txt");
        if (!idx) {
            err = "mtp: " + dir + " is a folder without dense.txt: --drafter takes the draft layer's file, " +
                  kDraftLayerFile + " (" + kDraftLayerScript +
                  " downloads it), or a folder of its earlier layout (dense.txt, dense.bin, "
                  "experts.bin, draft_vocab.bin)";
            return false;
        }
        std::string line;
        while (std::getline(idx, line)) {
            if (line.empty()) continue;
            std::istringstream is(line);
            Tensor t;
            is >> t.name >> t.kind >> t.rows >> t.cols >> t.off >> t.bytes;
            if (!is) {
                err = "mtp: malformed dense.txt line: " + line;
                return false;
            }
            tensors_.push_back(t);
        }
        std::vector<uint8_t> blob;
        if (!read_file(dir + "/dense.bin", blob)) {
            err = "mtp: cannot read " + dir + "/dense.bin";
            return false;
        }
        // each entry of its kind's size, aligned, inside dense.bin: a broken folder is refused, not read past
        for (const Tensor& t : tensors_) {
            const uint64_t n = t.rows > 0 && t.cols > 0 ? (uint64_t) t.rows * (uint64_t) t.cols : 0;
            const uint64_t want = t.kind == "f32"                        ? n * 4
                                  : t.kind == "bf16"                     ? n * 2
                                  : t.kind == "q8_0" && t.cols % 32 == 0 ? n / 32 * 34
                                                                         : 0;
            if (want == 0 || t.bytes != want || t.off % 4 != 0 || t.off > blob.size() ||
                t.bytes > blob.size() - t.off) {
                err = "mtp: dense.txt's " + t.name + " (" + t.kind + " " + std::to_string(t.rows) + " x " +
                      std::to_string(t.cols) + ", " + std::to_string(t.bytes) + " bytes at " + std::to_string(t.off) +
                      ") is not a tensor of dense.bin's " + std::to_string(blob.size()) + " bytes";
                return false;
            }
        }
        if (cudaMalloc((void**) &dense_, blob.size()) != cudaSuccess) {
            err = "mtp: dense weights do not fit";
            return false;
        }
        cudaMemcpy(dense_, blob.data(), blob.size(), cudaMemcpyHostToDevice);
        dense_bytes = blob.size();
        vram_ += blob.size();
    }
    // ---- the 512 routed experts, one blob each: raw GGUF blocks [gate | up | down] in the formats experts.txt names
    //      (the ggml type ids of gate/up and down), or without it the Q2_0 blob layout of cpu/expert.hpp
    {
        blob_ = lamina::kernels::cpu::BLOB;
        if (std::ifstream fmt(dir + "/experts.txt"); fmt) {
            int gu = -1, dn = -1;
            fmt >> gu >> dn;
            if (!lamina::kernels::native_expert_supported(gu, dn, g.n_embd, g.n_ff)) {
                err = "mtp: experts.txt names formats the GPU expert kernels lack";
                return false;
            }
            nat_ = lamina::kernels::native_expert_layout(gu, dn, g.n_embd, g.n_ff);
            blob_ = nat_.bytes;
        }
        const uint64_t bytes = (uint64_t) g.n_expert * blob_;
        std::ifstream f(dir + "/experts.bin", std::ios::binary | std::ios::ate);
        if (!f) {
            err = "mtp: cannot open " + dir + "/experts.bin";
            return false;
        }
        if ((uint64_t) f.tellg() != bytes) {
            err = "mtp: experts.bin does not match its format";
            return false;
        }
        f.seekg(0);
        if (cudaMalloc((void**) &experts_, bytes) != cudaSuccess) {
            err = "mtp: the 512 experts do not fit in VRAM";
            return false;
        }
        std::vector<uint8_t> chunk(64u << 20);
        for (uint64_t off = 0; off < bytes;) {
            const uint64_t n = std::min<uint64_t>(chunk.size(), bytes - off);
            if (!f.read((char*) chunk.data(), (std::streamsize) n)) {
                err = "mtp: experts.bin is truncated";
                return false;
            }
            cudaMemcpy(experts_ + off, chunk.data(), n, cudaMemcpyHostToDevice);
            off += n;
        }
        vram_ += bytes;
    }
    // ---- the draft head's token ids, when tools/draft_vocab.py wrote them (none: the whole head)
    std::vector<uint8_t> raw;
    if (read_file(dir + "/draft_vocab.bin", raw) && raw.size() >= 4 && raw.size() % 4 == 0) {
        dvocab_ids_.resize(raw.size() / 4);
        std::memcpy(dvocab_ids_.data(), raw.data(), raw.size());
    }
    return true;
}

bool MtpDrafter::bind(const WeightTable& wt, const NativeHead* head, const float* window_R, std::string& err) {
    wt_ = &wt;
    head_ = head;
    window_R_ = window_R;
    const WeightRef* wo = wt.find("output.weight");
    if (!wo) {
        err = "mtp: output.weight is missing";
        return false;
    }
    n_vocab_ = wo->ne1;
    if (head == nullptr || !head->loaded()) {
        err = "mtp: the draft layer needs the native head (--native)";
        return false;
    }
    if (dhead_ == nullptr && !bind_head(head, err)) return false;
    if (dhead_ == nullptr && head->rows() != n_vocab_) {
        err = "mtp: the main head holds part of the vocabulary and the draft layer has no draft head of its own";
        return false;
    }
    if (head_logits_ == nullptr &&
        cudaMalloc((void**) &head_logits_, (size_t) max_t_ * (size_t) n_vocab_ * sizeof(float)) != cudaSuccess) {
        err = "mtp: the draft logits do not fit";
        return false;
    }
    return true;
}

bool MtpDrafter::bind_head(const NativeHead* head, std::string& err) {
    if (head == nullptr || !head->loaded()) {
        err = "mtp: the draft layer needs the native head (--native)";
        return false;
    }
    // the draft head's token subset, when the layer's file holds one (tools/draft_vocab.py, tools/mtp_rt.py)
    if (dhead_ == nullptr) {
        if (!dvocab_ids_.empty()) {
            n_dvocab_ = (int64_t) dvocab_ids_.size();
            for (const int32_t id : dvocab_ids_)   // the head's rows they are gathered from
                if (id < 0 || id >= head->rows()) {
                    err = "mtp: the draft vocabulary holds the token " + std::to_string(id) + ", outside the head's " +
                          std::to_string(head->rows()) + " rows";
                    return false;
                }
            const size_t ids_bytes = dvocab_ids_.size() * sizeof(int32_t);
            const int64_t row_bytes = (int64_t) head->row_bytes();   // a vocabulary row of the native head
            if (cudaMalloc((void**) &dvocab_, ids_bytes) != cudaSuccess ||
                cudaMalloc((void**) &dhead_, (size_t) (n_dvocab_ * row_bytes)) != cudaSuccess) {
                err = "mtp: the draft head does not fit";
                return false;
            }
            cudaMemcpy(dvocab_, dvocab_ids_.data(), ids_bytes, cudaMemcpyHostToDevice);
            lamina::kernels::gather_rows((const uint8_t*) head->weights(), row_bytes, dvocab_, n_dvocab_, dhead_,
                                         nullptr);
            cudaDeviceSynchronize();
            vram_ += (uint64_t) (n_dvocab_ * row_bytes) + ids_bytes;
            std::fprintf(stderr, "lamina mtp: draft head over %lld tokens (%.1f MiB)\n", (long long) n_dvocab_,
                         (double) (n_dvocab_ * row_bytes) / 1048576.0);
            // its 2-bit copy finds each draft's row (LAMINA_DRAFT_PROXY=0: the exact head over every row)
            const char* px = std::getenv("LAMINA_DRAFT_PROXY");
            const bool copy = px == nullptr || std::atoi(px) != 0;
            if (copy && lamina::kernels::draft_proxy_supported(head->type(), (int) g_->n_embd)) {
                const size_t pb = lamina::kernels::draft_proxy_bytes(n_dvocab_);
                const size_t sb = lamina::kernels::draft_proxy_scratch_bytes(n_dvocab_);
                if (cudaMalloc((void**) &proxy_, pb) != cudaSuccess || cudaMalloc(&proxy_scratch_, sb) != cudaSuccess) {
                    err = "mtp: the draft head's 2-bit copy does not fit";
                    return false;
                }
                cudaMemset(proxy_scratch_, 0, sb);
                lamina::kernels::draft_proxy_pack(head->type(), dhead_, n_dvocab_, proxy_, nullptr);
                cudaDeviceSynchronize();
                vram_ += pb + sb;
            } else if (copy) {   // a head type or width the copy does not take
                std::fprintf(stderr, "lamina mtp: draft head over %lld tokens: the exact head (no 2-bit copy for it)\n",
                             (long long) n_dvocab_);
            }
        }
    }
    return true;
}

// The layer for T rows.  full = false stops after the K/V append (the prompt only needs the cache).
bool MtpDrafter::record_forward(int T, int step_row0, cudaStream_t cs, std::string& err) {
    // step_row0 >= 0: the full layer on step rows [step_row0, +T); step_row0 < 0: K/V only on rows [-1 - step_row0, +T)
    const bool full = step_row0 >= 0;
    if (full && T != 1) {
        err = "mtp: the full layer runs one row at a time (its attention scratch is sized for one)";
        return false;
    }
    if (!record_front(T, full ? step_row0 : -1 - step_row0, cs, err)) return false;
    return !full || record_rest(step_row0, cs, err);
}

bool MtpDrafter::record_front(int T, int row0, cudaStream_t cs, std::string& err) {
    using namespace lamina::kernels;
    const ModelGeometry& g = *g_;
    const int64_t N = g.n_embd, HC = g.hc, NH = g.n_head, HD = g.head_dim, NKV = g.n_head_kv;
    const QsaShapes s = lamina::kernels::qsa_shapes(g);
    const int32_t* step = step_ + row0 * 4;
    const int32_t* pos = pos_ + row0 * NH;
    try {
        // ---- the two input branches
        if (!embed_rows(*wt_, g, tok_, T, emb_, cs, err)) return false;
        native_qsa_rms_norm_weighted(emb_, f32("pre_fc_norm_embedding.weight"), en_, (int) N, T, EPS, cs);
        native_quantize_q8_1(en_, xq_, (int) N, T, cs);
        native_mmvq(GGML_Q8_0, q8("fc_embedding.weight"), xq_, e2_, (int) N, (int) N, T, cs);
        native_qsa_rms_norm_weighted(Rin_, f32("pre_fc_norm_hidden.weight"), hn_, (int) (HC * N), T, EPS, cs);
        for (int c0 = 0; c0 < T * HC; c0 += 8) {
            const int nc = (int) std::min<int64_t>(8, T * HC - c0);
            native_quantize_q8_1(hn_ + (size_t) c0 * N, xq_, (int) N, nc, cs);
            native_mmvq(GGML_Q8_0, q8("fc_hidden.weight"), xq_, h2_ + (size_t) c0 * N, (int) N, (int) N, nc, cs);
        }
        add_streams_broadcast(h2_, e2_, R_, N, (int) HC, T, cs);
        // ---- the attention hyper-connection
        {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                fa[t].R = R_ + (size_t) t * HC * N;
                fa[t].R_out = R_ + (size_t) t * HC * N;
                fa[t].apply = false;
                fa[t].w_norm = f32("attn_hyper_connection.hc_norm.weight");
                fa[t].w_down = bf16("attn_hyper_connection.input_mix_weight_down.weight");
                fa[t].w_up = bf16("attn_hyper_connection.input_mix_weight_up.weight");
                fa[t].w_inject = bf16("attn_hyper_connection.block_inject_weight.weight");
                fa[t].eps = EPS;
                fa[t].lo = lo_ + t * g.hc_lr;
                fa[t].rs = rs_ + t * HC;
                fa[t].inject_out = inj_ + t * HC;
                fa[t].mixed = mixed_ + t * N;
            }
            fused_gr_read_multi(fa, T, grs_, cs);
        }
        // ---- attention: K/V into the layer's own cache
        native_quantize_q8_1(mixed_, xq_, (int) N, T, cs);
        native_mmvq(GGML_Q8_0, q8("self_attn.k_proj.weight"), xq_, kcur_, (int) N, (int) (NKV * HD), T, cs);
        native_mmvq(GGML_Q8_0, q8("self_attn.v_proj.weight"), xq_, vcur_, (int) N, (int) (NKV * HD), T, cs);
        for (int t = 0; t < T; ++t) {
            norm_rope(kcur_ + t * NKV * HD, f32("self_attn.k_norm.weight"), (int) NKV, (int) HD, pos + t * NH, cs);
            if (st_.kv_int8)
                kv_append_q8_step(st_.k_q, st_.v_q, st_.k_scale, st_.v_scale, st_.page_table, step + t * 4,
                                  kcur_ + t * NKV * HD, vcur_ + t * NKV * HD, s, cs);
            else
                kv_append_step(st_.k_pool, st_.v_pool, st_.page_table, step + t * 4, kcur_ + t * NKV * HD,
                               vcur_ + t * NKV * HD, s, cs);
        }
    } catch (const std::exception& e) {
        err = std::string("mtp: ") + e.what();
        return false;
    }
    return true;
}

void MtpDrafter::norm_rope(float* data, const float* gamma, int rows, int cols, const int32_t* p, cudaStream_t cs) {
    using namespace lamina::kernels;
    const QsaShapes s = lamina::kernels::qsa_shapes(*g_);
    native_qsa_rms_norm_weighted(data, gamma, data, cols, rows, EPS, cs);
    native_rope_apply(data, data, rows, cols, (int) s.n_rot, rope_scaling(), p, cs);
}

bool MtpDrafter::record_rest(int step_row, cudaStream_t cs, std::string& err) {
    using namespace lamina::kernels;
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    const int T = 1;
    const int64_t N = g.n_embd, HC = g.hc, K = ss.k, NH = g.n_head, HD = g.head_dim;
    const QsaShapes s = lamina::kernels::qsa_shapes(g);
    const GrShapes gs{g.n_embd, g.hc, g.hc_lr};
    const int32_t* step = step_ + step_row * 4;
    const int32_t* pos = pos_ + step_row * NH;
    try {
        // ---- dense attention over every cell
        native_mmvq(GGML_Q8_0, q8("self_attn.q_proj.weight"), xq_, qfull_, (int) N, (int) (NH * 2 * HD), T, cs);
        for (int t = 0; t < T; ++t) {
            float* qc = qcur_ + t * NH * HD;
            if (cudaMemcpy2DAsync(qc, (size_t) HD * 4, qfull_ + t * NH * 2 * HD, (size_t) HD * 2 * 4, (size_t) HD * 4,
                                  (size_t) NH, cudaMemcpyDeviceToDevice, cs) != cudaSuccess) {
                err = "mtp: q split failed";
                return false;
            }
            norm_rope(qc, f32("self_attn.q_norm.weight"), (int) NH, (int) HD, pos + t * NH, cs);
        }
        QsaAttnPools pools;
        pools.page_table = st_.page_table;
        if (st_.kv_int8) {
            pools.k_q = st_.k_q;
            pools.v_q = st_.v_q;
            pools.k_scale = st_.k_scale;
            pools.v_scale = st_.v_scale;
        } else {
            pools.k_pool = st_.k_pool;
            pools.v_pool = st_.v_pool;
        }
        if (window_ > 0) window_ids(const_cast<int32_t*>(step), T, (int) window_, ident_, cap_, cs);
        qsa_decode_attn_batch(qcur_, pools, ident_, step, cap_, s, attn_scratch_, attn32_, T, cs, qfull_);
        native_quantize_q8_1(attn32_, xq_, (int) (NH * HD), T, cs);
        native_mmvq(GGML_Q8_0, q8("self_attn.o_proj.weight"), xq_, bo_, (int) (NH * HD), (int) N, T, cs);
        // ---- the MLP hyper-connection (the attention write folded in)
        {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                fa[t].R = R_ + (size_t) t * HC * N;
                fa[t].R_out = R_ + (size_t) t * HC * N;
                fa[t].apply = true;
                fa[t].bo_prev = bo_ + t * N;
                fa[t].inj_prev = inj_ + t * HC;
                fa[t].w_norm = f32("mlp_hyper_connection.hc_norm.weight");
                fa[t].w_down = bf16("mlp_hyper_connection.input_mix_weight_down.weight");
                fa[t].w_up = bf16("mlp_hyper_connection.input_mix_weight_up.weight");
                fa[t].w_inject = bf16("mlp_hyper_connection.block_inject_weight.weight");
                fa[t].eps = EPS;
                fa[t].lo = lo_ + t * g.hc_lr;
                fa[t].rs = rs_ + t * HC;
                fa[t].inject_out = inj2_ + t * HC;
                fa[t].mixed = mixed_ + t * N;
            }
            fused_gr_read_multi(fa, T, grs_, cs);
        }
        // ---- MoE: the shared expert on a branch beside the router and the 512 resident experts, then the combine
        // and the write
        NativeSharedWeights nsw;
        nsw.gate_type = GGML_Q8_0;
        nsw.gate_data = q8("mlp.shared_expert.gate_proj.weight");
        nsw.up_type = GGML_Q8_0;
        nsw.up_data = q8("mlp.shared_expert.up_proj.weight");
        nsw.down_type = GGML_Q8_0;
        nsw.down_data = q8("mlp.shared_expert.down_proj.weight");
        nsw.q8_1 = xq_;   // the routed experts read nat_xq_ / hit_xq_
        const SForm none{};
        if (cudaEventRecord(sh_fork_, cs) != cudaSuccess || cudaStreamWaitEvent(side_, sh_fork_, 0) != cudaSuccess) {
            err = "mtp: the shared expert's branch";
            return false;
        }
        for (int t = 0; t < T; ++t) {
            f32_to_bf16_bulk(mixed_ + t * N, x_bf16_, N, side_);
            shared_expert(nullptr, nullptr, x_bf16_, none, nullptr, nullptr, nullptr, none, nullptr, nullptr, nullptr,
                          none, nullptr, nullptr, nullptr, bf16("mlp.shared_expert_gate.weight"), sh_scratch_,
                          shared_ + t * N, N, g.n_ff, 32, side_, mixed_ + t * N, &nsw);
        }
        if (cudaEventRecord(sh_join_, side_) != cudaSuccess) {
            err = "mtp: the shared expert's branch";
            return false;
        }
        VerifyRouterArgs ra;
        ra.x = mixed_;
        ra.w = bf16("mlp.gate.weight");
        ra.logits = logits_;
        ra.ids = ids_;
        ra.weights = w_;
        ra.xq1 = nat_.gu_type >= 0 ? nat_xq_ : nullptr;   // the native experts' q8_1 rows
        ra.seq = rseq_;
        ra.counter = rcount_;
        ra.n_tok = T;
        ra.n_embd = (int) N;
        ra.n_expert = (int) g.n_expert;
        verify_router(ra, cs);
        moe_group_resident(ids_, (int) (T * K), (int) K, experts_, (int64_t) blob_, grp_ptr_, grp_start_, grp_counts_,
                           hit_dst_, hit_slot_, cs);
        if (nat_.gu_type >= 0) {
            native_expert_grouped(nat_, grp_ptr_, grp_start_, grp_counts_, hit_dst_, hit_slot_, (int64_t) T * K,
                                  (int64_t) T * K, nat_xq_, (int64_t) (N / 32) * 36, hit_scratch_, parts_, cs);
        } else {
            quantize_q8_0_scaled(mixed_, hit_xq_, hit_xs_, (int64_t) T * N, cs);
            moe_grouped_s2(grp_ptr_, grp_start_, grp_counts_, hit_dst_, hit_slot_, (int64_t) T * K, (int64_t) T * K,
                           hit_xq_, hit_xs_, hit_scratch_, parts_, cs);
        }
        if (cudaStreamWaitEvent(cs, sh_join_, 0) != cudaSuccess) {
            err = "mtp: the shared expert's join";
            return false;
        }
        for (int t = 0; t < T; ++t) {
            native_moe_combine(parts_ + (size_t) t * K * N, w_ + t * K, shared_ + t * N, y_ + t * N, N, K, cs);
            gr_write(R_ + (size_t) t * HC * N, y_ + t * N, inj2_ + t * HC, gs, R_ + (size_t) t * HC * N, cs);
        }
        // ---- the final mixer and the main model's head
        {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                fa[t].R = R_ + (size_t) t * HC * N;
                fa[t].R_out = R_ + (size_t) t * HC * N;
                fa[t].apply = false;
                fa[t].w_norm = f32("hyper_connection_mixer.hc_norm.weight");
                fa[t].w_down = bf16("hyper_connection_mixer.input_mix_weight_down.weight");
                fa[t].w_up = bf16("hyper_connection_mixer.input_mix_weight_up.weight");
                fa[t].eps = EPS;
                fa[t].lo = lo_ + t * g.hc_lr;
                fa[t].rs = rs_ + t * HC;
                fa[t].mixed = sample_ + t * N;
            }
            fused_gr_read_multi(fa, T, grs_, cs);
        }
        native_quantize_q8_1(sample_, xq_, (int) N, T, cs);
        const bool sub = dhead_ != nullptr;
        const int64_t nv = sub ? n_dvocab_ : n_vocab_;
        if (proxy_ != nullptr) {   // the draft head's copy finds the row, the rows near it rescored exactly
            draft_proxy_top1(head_->type(), proxy_, dhead_, xq_, nv, dvocab_, out_ids_, probs_, proxy_scratch_, cs);
        } else {
            native_mmvq(head_->type(), sub ? dhead_ : head_->weights(), xq_, head_logits_, (int) N, (int) nv, T, cs);
            argmax_rows(head_logits_, T, (int) nv, arg_scratch_, out_ids_, cs);
            row_top_prob_split(head_logits_, T, (int) nv, out_ids_, probs_, top_scratch_, cs);
            if (sub) map_ids(out_ids_, dvocab_, T, cs);
        }
    } catch (const std::exception& e) {
        err = std::string("mtp: ") + e.what();
        return false;
    }
    return true;
}

namespace {
bool finish_capture(cudaStream_t cs, bool ok, cudaGraphExec_t& exec, const char* what, std::string& err) {
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs, &graph);
    if (!ok) {
        if (graph) cudaGraphDestroy(graph);
        return false;
    }
    if (ce != cudaSuccess || cudaGraphInstantiate(&exec, graph, 0) != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = std::string("mtp: ") + what + " capture: " + cudaGetErrorString(ce);
        return false;
    }
    cudaGraphDestroy(graph);
    // an explicit upload: the first launch's implicit one blocked behind a device-side spin (verify.cpp)
    cudaGraphUpload(exec, cs);
    cudaStreamSynchronize(cs);
    return true;
}
}   // namespace

bool MtpDrafter::capture_all(std::string& err) {
    for (int T = 1; T <= max_t_; ++T)
        if (!capture_round(T, err)) return false;
    for (int j = 1; j <= max_t_ - 2; ++j)
        if (!capture_step(j, err)) return false;
    return true;
}

bool MtpDrafter::capture_round(int T, std::string& err) {
    if (round_exec_[T]) return true;
    using namespace lamina::kernels;
    const int64_t HCN = g_->hc * g_->n_embd;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "mtp: begin capture";
        return false;
    }
    bool ok = true;
    // the catch-up: the layer's front for the window's T cells (their K/V), then its rest for row a only, on row
    // a's intermediates copied to row 0, at the cell the host staged in step row 2*max_t - 1; the draft chain is
    // one graph per step (`capture_step`) so the host can stop it when a draft is unlikely
    const int ra = 2 * max_t_ - 1;
    const int64_t N = g_->n_embd, NH = g_->n_head;
    const MappedCopy in[7] = {{tok_, m_tok_, T},
                              {step_, m_step_, (int64_t) 2 * T * 4},
                              {pos_, m_pos_, 2 * T * NH},
                              {row_, m_row_, 2},
                              {Rin_, window_R_, T * HCN},
                              {step_ + ra * 4, m_step_ + ra * 4, 4},
                              {pos_ + ra * NH, m_pos_ + ra * NH, NH}};
    copy_from_mapped_multi(in, 7, cs_);
    ok = record_front(T, 0, cs_, err);
    if (ok) {
        if (T > 1) {
            copy_row_to_first(row_, R_, HCN, inj_, g_->hc, mixed_, N, cs_);
            native_quantize_q8_1(mixed_, xq_, (int) N, 1, cs_);
        }
        ok = record_rest(ra, cs_, err);
    }
    if (ok) mtp_select(R_, HCN, out_ids_, row_ + 1, Rin_, tok_, m_out_, 0, cs_, probs_, m_prob_);
    return finish_capture(cs_, ok, round_exec_[T], "round", err);
}

// Chain step j (1..max_t-2): one row at the cell staged in step row `max_t + j - 1`, from the previous step's
// residual and token (left in Rin_[0] / tok_[0] by mtp_select); draft j and its probability to the mapped outputs.
bool MtpDrafter::capture_step(int j, std::string& err) {
    if (step_exec_[j]) return true;
    using namespace lamina::kernels;
    const int64_t HCN = g_->hc * g_->n_embd;
    const int row = max_t_ + j - 1;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "mtp: begin capture";
        return false;
    }
    const MappedCopy in[2] = {{step_ + row * 4, m_step_ + row * 4, 4},
                              {pos_ + row * g_->n_head, m_pos_ + row * g_->n_head, g_->n_head}};
    copy_from_mapped_multi(in, 2, cs_);
    bool ok = record_forward(1, row, cs_, err);
    if (ok) mtp_select(R_, HCN, out_ids_, row_ + 1, Rin_, tok_, m_out_, j, cs_, probs_, m_prob_);
    return finish_capture(cs_, ok, step_exec_[j], "step", err);
}

MtpPromptKv MtpDrafter::prompt_kv() {
    MtpPromptKv k;
    k.norm_emb = f32("pre_fc_norm_embedding.weight");
    k.norm_hidden = f32("pre_fc_norm_hidden.weight");
    k.fc_emb = q8("fc_embedding.weight");
    k.fc_hidden = q8("fc_hidden.weight");
    k.hc_norm = f32("attn_hyper_connection.hc_norm.weight");
    k.hc_down = bf16("attn_hyper_connection.input_mix_weight_down.weight");
    k.hc_up = bf16("attn_hyper_connection.input_mix_weight_up.weight");
    k.k_proj = q8("self_attn.k_proj.weight");
    k.v_proj = q8("self_attn.v_proj.weight");
    k.k_norm = f32("self_attn.k_norm.weight");
    k.kv = &st_;
    k.first_cell = (window_ > 0 && prompt_len_ > 0) ? prompt_len_ - window_ - 64 : 0;
    return k;
}

uint64_t MtpDrafter::tensor_bytes(const void* p) const {
    for (const auto& t : tensors_)
        if (dense_ + t.off == (const uint8_t*) p) return t.bytes;
    return 0;
}

void MtpDrafter::set_sampling(const lamina::kernels::SamplerParams& sp) {
    static const bool on = [] {
        const char* e = std::getenv("LAMINA_DRAFT_PASS");
        return e == nullptr || std::atoi(e) != 0;
    }();
    const bool penalties =
        sp.penalty_last_n > 0 && (sp.penalty_repeat != 1.0f || sp.penalty_freq != 0.0f || sp.penalty_present != 0.0f);
    // a temperature the sampler takes for greedy (sample_tokens) draws nothing; penalties move the verifier's
    // logits where the draft layer's stay
    pass_ = on && !sp.greedy && sp.temperature >= 1e-30f && !penalties && sp.logit_bias == nullptr;
    pass_seed_ = sp.seed;
    pass_temperature_ = sp.temperature;
    pass_min_p_ = -1.0f;   // the floors follow the temperature
}

bool MtpDrafter::draft(int T, const int32_t* tokens, int64_t p, int a, int32_t* drafts, std::string& err, float* probs,
                       float min_p, int* n_drafts, int* n_kept) {
    if (T < 1 || T > max_t_ || a < 0 || a >= T) {
        err = "mtp: draft arguments out of range";
        return false;
    }
    // the accepted rows' K/V and row a's layer: no read reaches a cell past the one being drafted, and the chain or the
    // next round writes the rejected rows' cells first
    T = a + 1;
    if (!capture_round(T, err)) return false;
    const Clock::time_point t0 = Clock::now();
    const int64_t NH = g_->n_head;
    auto put = [&](int row, int64_t cell) {
        h_step_[row * 4 + 0] = (int32_t) cell;
        h_step_[row * 4 + 1] = (int32_t) (cell + 1);
        h_step_[row * 4 + 2] = (int32_t) ((cell + 1) / 4);
        h_step_[row * 4 + 3] = (int32_t) (cell + 1);
        for (int64_t h = 0; h < NH; ++h) h_pos_[row * NH + h] = (int32_t) cell;
    };
    for (int t = 0; t < T; ++t) {
        h_tok_[t] = tokens[t];
        put(t, p + t);
    }
    put(2 * max_t_ - 1, p + a);
    h_row_[0] = a;
    h_row_[1] = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (cudaGraphLaunch(round_exec_[T], cs_) != cudaSuccess || cudaStreamSynchronize(cs_) != cudaSuccess) {
        err = std::string("mtp draft: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    // Whether draft j, of probability q under the layer, is kept.  A sampled request's draft is judged by its chance
    // to pass: chain step j drafts the token after cell p + a + j's, which its verify row draws with counter
    // p + a + j + 1 (row t of a window at pos0 draws Philox(seed, pos0 + t): Verifier::set_sampling).
    if (pass_ && pass_min_p_ != min_p) {
        for (int j = 0; j < 3; ++j) pass_floor_[j] = draft_pass_floor(j, min_p, pass_temperature_);
        pass_min_p_ = min_p;
    }
    const auto keeps = [&](int j, float q) {
        if (probs) probs[j] = q;
        if (!pass_) {
            chances_[j] = q;
            return q >= min_p;
        }
        const float u = lamina::kernels::philox_uniform(pass_seed_, (uint64_t) (p + a + j + 1));
        chances_[j] = draft_pass_chance(j, q, u, pass_temperature_);
        return chances_[j] >= pass_floor_[std::min(j, 2)];
    };
    drafts[0] = ((volatile int32_t*) h_out_)[0];
    bool kept = keeps(0, ((volatile float*) h_prob_)[0]);
    if (on_draft) on_draft(0, drafts[0], kept);
    int n = 1, nk = kept ? 1 : 0;
    // the chain goes on past a kept draft; draft j's K/V goes to cell p + a + j, so near the context's end it stops
    // at the last cell
    const int chain = (int) std::min<int64_t>(max_drafts_ > 0 ? std::min(max_drafts_, max_t_ - 1) : max_t_ - 1,
                                              st_.max_cells - (p + a));
    for (int j = 1; j < chain && kept; ++j) {
        if (!capture_step(j, err)) return false;
        put(max_t_ + j - 1, p + a + j);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (cudaGraphLaunch(step_exec_[j], cs_) != cudaSuccess || cudaStreamSynchronize(cs_) != cudaSuccess) {
            err = std::string("mtp draft step: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        drafts[j] = ((volatile int32_t*) h_out_)[j];
        kept = keeps(j, ((volatile float*) h_prob_)[j]);
        if (on_draft) on_draft(j, drafts[j], kept);
        ++n;
        nk += kept ? 1 : 0;
    }
    for (int j = n; j < max_t_ - 1; ++j) {
        drafts[j] = 0;
        chances_[j] = 0.0f;
        if (probs) probs[j] = 0.0f;
    }
    if (n_drafts) *n_drafts = n;
    if (n_kept) *n_kept = nk;
    ms_draft += ms_since(t0);
    ++rounds;
    return true;
}

bool MtpDrafter::draft_first(const float* R_row, int32_t token, int64_t cell, int32_t* drafts, std::string& err,
                             float* probs, float min_p, int* n_drafts, int* n_kept) {
    if (cudaMemcpy((void*) window_R_, R_row, (size_t) (g_->hc * g_->n_embd) * sizeof(float),
                   cudaMemcpyDeviceToDevice) != cudaSuccess) {
        err = "mtp: staging the first residual failed";
        return false;
    }
    return draft(1, &token, cell, 0, drafts, err, probs, min_p, n_drafts, n_kept);
}

}   // namespace lamina::core
