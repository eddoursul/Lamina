// src/core/model_check.cpp - the model files the engine takes, decided from their headers alone (model_check.hpp).
// Each rule is a load's own, in its words; the load's place is named beside it.
#include "lamina/core/model_check.hpp"

#include "lamina/artifact/drafter.hpp"
#include "lamina/artifact/gguf_reader.hpp"
#include "lamina/artifact/model_family.hpp"
#include "lamina/core/family_layout.hpp"
#include "lamina/core/geometry.hpp"
#include "lamina/core/native_dense.hpp"
#include "lamina/core/pack_image.hpp"
#include "lamina/hy4/geometry.hpp"
#include "lamina/inkling2/geometry.hpp"
#include "lamina/inkling2/kernel_limits.hpp"
#include "lamina/kernels/cpu/expert_layout.hpp"
#include "lamina/kernels/cpu/native_expert.hpp"
#include "lamina/kernels/iq_kernels.hpp"
#include "lamina/kernels/native_mmvq.hpp"
#include "lamina/kernels/ngram.hpp"

#include "ggml.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace lamina::core {
namespace {

namespace K = lamina::kernels;
namespace C = lamina::kernels::cpu;

std::string blk(int64_t l, const char* suffix) { return "blk." + std::to_string(l) + "." + suffix; }

bool is_float(uint32_t t) { return t == 0 || t == 1 || t == 30; }   // F32, F16, BF16

/// ggml has a dequantizer for the type (Inkling 2's floats and embedding: ggml's to_float)
bool ggml_reads(uint32_t t) {
    if (t == 0) return true;
    const ggml_type_traits* tr = t < GGML_TYPE_COUNT ? ggml_get_type_traits((ggml_type) t) : nullptr;
    return tr != nullptr && tr->to_float != nullptr;
}

/// NativeDense::load's directory checks of every shard (native_dense.hpp's ShardDirectory)
std::string dense_directory(const GgufModel& m) {
    ShardDirectory dir;
    for (size_t s = 0; s < m.size(); ++s)
        if (std::string e = dir.check(m.shard(s)); !e.empty()) return e;
    return {};
}

/// The dense matrices NativeDense serves, which the verify window must find served
bool served_name(const std::string& n) { return native_dense_name(n, true); }

}   // namespace

FileCheckOptions file_check_options(const std::vector<std::string>& args) {
    FileCheckOptions o;
    for (size_t i = 0; i < args.size(); ++i) {
        std::string a = args[i], v;
        const size_t eq = a.find('=');
        const bool joined = a.rfind("--", 0) == 0 && eq != std::string::npos;
        if (joined) {
            v = a.substr(eq + 1);
            a.resize(eq);
        }
        auto value = [&]() -> std::string {
            if (joined) return v;
            return i + 1 < args.size() ? args[++i] : std::string();
        };
        if (a == "--ple-gguf") o.ple_gguf = value();
        else if (a == "--hc-q8") o.hc_q8 = true;
        else if (a == "--no-hc-q8") o.hc_q8 = false;
        else if (a == "--expert-math") o.expert_exact = value() == "exact";
        else if (a == "--no-mtp") o.mtp = false;
        else if (a == "--vision") o.vision = true;
        else if (a == "--no-vision") o.vision = false;
        else if (a == "--emb") o.vision = !value().empty();
        else if (std::find(drafter_flags().begin(), drafter_flags().end(), a) != drafter_flags().end())
            o.mtp = value().empty() && o.mtp;
    }
    return o;
}

std::string flash_file_problem(const std::vector<std::string>& shards, const FileCheckOptions& o) {
    if (shards.empty()) return "--native MODEL.gguf is required";
    // the PLE table's file (generate.cpp, before the pack)
    if (o.ple_gguf.empty() && K::ple_table_shard(shards).empty())
        return "no shard of the --native model holds per_layer_token_embd; name the file with --ple-gguf";
    // the architecture and the geometry the kernels are built for (NativeDense / NativeHead: check_architecture)
    try {
        const GgufFile first(shards.front());
        if (std::string e = check_architecture(first); !e.empty()) return e;
    } catch (const std::exception& e) {
        return e.what();
    }
    // the pack: every tensor of a form or kernels, the routers and experts, the rotation (PackImage::build)
    PackImage pack;
    std::string err;
    if (!PackImage::build(shards.front(), pack, err)) return err;
    // the experts' layout as the CPU pool and the GPUs read it (expert_layout_load_text, then each layer's kernels)
    const ModelGeometry g0;
    C::ExpertLayout lay;
    if (!C::expert_layout_parse_text(pack.native_experts_text(), g0.n_layers, g0.n_expert, lay, err)) return err;
    const int q5k_mins = C::ggml_q5_K_mins_rounding();
    for (int64_t l = 0; l < lay.n_layers; ++l) {
        const C::NativeFmt& f = lay.fmt[(size_t) l];
        if (!K::native_expert_supported(f.gu_type, f.d_type, f.n_embd, f.n_ff))
            return "layer " + std::to_string(l) + "'s experts (GGML types " + std::to_string(f.gu_type) + " / " +
                   std::to_string(f.d_type) + ") have no GPU kernels";
        if (o.expert_exact && (!K::native_expert_exact_supported(f.gu_type, f.d_type, f.gu_act, f.d_act) ||
                               ((f.gu_type == 13 || f.d_type == 13) && q5k_mins < 0)))
            return "--expert-math exact has no bit-exact GPU kernels for layer " + std::to_string(l) +
                   "'s experts (GGML "
                   "types " +
                   std::to_string(f.gu_type) + " / " + std::to_string(f.d_type) + "); use --expert-math fast";
    }
    // the arena takes the compiled expert count (ArenaExpertSource::open)
    if (lay.n_expert != g0.n_expert)
        return "the model has " + std::to_string(lay.n_expert) + " experts a layer; this path runs " +
               std::to_string(g0.n_expert) +
               " (ArenaExpertSource: the expert layout was loaded for a different geometry)";
    try {
        const GgufModel m(shards);
        // the token embedding (NativeEmbed::load: its rows dequantized on the GPU)
        {
            const TensorInfo* t = m.find("token_embd.weight");
            if (!t || t->shape.size() != 2 || t->shape[0] != (uint64_t) g0.n_embd || t->shape[1] != 248320 ||
                !K::iq_supported((int) t->type) || g0.n_embd % 256)
                return "native embedding: token_embd.weight is absent, of another shape, or of a type without a GPU dequantizer";
        }
        // the dense matrices served in their GGUF form (NativeDense::load), every shard's directory checked
        if (std::string e = dense_directory(m); !e.empty()) return e;
        bool any = false;
        for (size_t s = 0; s < m.size(); ++s)
            for (const TensorInfo& t : m.shard(s).tensors()) {
                if (!served_name(t.name) || !K::native_mmvq_supported((int) t.type)) continue;
                if (t.shape.size() != 2 || t.shape[0] > INT_MAX || t.shape[1] > INT_MAX)
                    return "native dense: incompatible matrix " + t.name;
                any = true;
            }
        if (!any) return "native dense: no supported GDN/QSA matrices in supplied shards";
        // the hyper-connection matrices in 8 bits (HcQ8::load)
        if (o.hc_q8)
            for (size_t s = 0; s < m.size(); ++s)
                for (const TensorInfo& t : m.shard(s).tensors()) {
                    if (!hc_q8_name(t.name)) continue;
                    const std::string tn = t.type_name();
                    if (t.shape.size() != 2 || t.shape[0] % 32 != 0 ||
                        (tn != "Q8_0" && tn != "BF16" && tn != "F16" && tn != "F32"))
                        return "hc q8: " + t.name + " has another shape or type (" + tn + ")";
                }
        // the PLE table (PleTable::open: its shape, its row format, its span in the file) and the PLE's weights
        {
            if (std::string e = ple_table_problem(o.ple_gguf.empty() ? K::ple_table_shard(shards) : o.ple_gguf);
                !e.empty())
                return e;
            for (const char* n :
                 {"blk.1.ple_key.weight", "blk.1.ple_value.weight", "blk.1.ple_norm_key.weight",
                  "blk.1.ple_norm_query.weight", "blk.1.ple_norm_conv.weight", "blk.1.ple_conv1d.weight"})
                if (m.find(n) == nullptr)
                    return "the pack has no blk.1.ple_* tensors, so the PLE cannot be wired - and running without it is a "
                           "DIFFERENT MODEL";
        }
        // what the verify window and the prompt path read of each layer (Verifier::record_window, Prefill::run): every
        // tensor there, the projections served natively (an MMVQ type)
        for (int64_t l = 0; l < g0.n_layers; ++l) {
            std::vector<const char*> need = {
                "hc_attn_norm.weight",  "hc_attn_down.weight",       "hc_attn_up.weight",     "hc_attn_inject.weight",
                "hc_ffn_norm.weight",   "hc_ffn_down.weight",        "hc_ffn_up.weight",      "hc_ffn_inject.weight",
                "ffn_gate_inp.weight",  "ffn_gate_inp_shexp.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                "ffn_down_shexp.weight"};
            if (is_qsa_layer(g0, l))
                need.insert(need.end(), {"indexer.k_proj.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
                                         "attn_output.weight", "indexer.q_proj.weight", "attn_q_norm.weight",
                                         "attn_k_norm.weight", "indexer.q_norm.weight", "indexer.k_norm.weight"});
            else
                need.insert(need.end(),
                            {"attn_qkv.weight", "attn_gate.weight", "ssm_out.weight", "ssm_alpha.weight",
                             "ssm_beta.weight", "ssm_conv1d.weight", "ssm_norm.weight", "ssm_dt.bias", "ssm_a"});
            for (const char* n : need) {
                const std::string name = blk(l, n);
                const TensorInfo* t = m.find(name);
                if (t == nullptr) return name + " is missing";
                if (served_name(name) && (!K::native_mmvq_supported((int) t->type) || t->shape.size() != 2))
                    return "verify: " + name + " is not served natively (" + t->type_name() +
                           ", which the native MMVQ "
                           "does not take)";
            }
        }
        for (const char* n : {"output_hc_norm.weight", "output_hc_down.weight", "output_hc_up.weight"})
            if (m.find(n) == nullptr) return "verify: an output_hc_* weight is missing";
        // the head (NativeHead::load)
        {
            size_t at = 0;
            const TensorInfo* t = m.find("output.weight", &at);
            if (t == nullptr) return "output.weight is missing";
            if (!K::native_mmvq_supported((int) t->type) || t->shape.size() != 2 ||
                t->shape[0] != (uint64_t) g0.n_embd || t->shape[1] > INT_MAX)
                return "native head: expected a natively supported output.weight with the canonical head dimensions";
            const GgufFile& f = m.shard(at);
            const uint64_t bytes = K::native_mmvq_weight_bytes((int) t->type, (int) g0.n_embd, (int) t->shape[1]);
            const uint64_t payload = f.file_size() - f.data_start();
            if (t->offset > payload || bytes > payload - t->offset)
                return "native head: truncated output.weight payload";
        }
    } catch (const std::exception& e) {
        return e.what();
    }
    return {};
}

std::string qwen35_file_problem(const GgufModel& m) {
    ModelGeometry g;
    int64_t top_k = 0;
    std::string err;
    if (!geometry_from_gguf(m.meta(), g, top_k, err)) return err;
    if (g.family != ModelFamily::Qwen35Moe && g.family != ModelFamily::Qwen35)
        return "this path runs Qwen3.6-35B-A3B (qwen35moe) and Qwen3.8-27B (qwen35) files";
    const bool moe = g.family == ModelFamily::Qwen35Moe;
    if (!check_gguf_layout(m, g, err)) return err;
    if (g.head_dim != 256 || g.ssm_state_size != 128 || g.ssm_d_conv != 4 ||
        (moe && (g.n_expert > 512 || top_k > 15)) || g.n_head_kv < 1 || g.n_head % g.n_head_kv != 0 ||
        g.n_head / g.n_head_kv > 8)
        return "a geometry this driver's kernels do not take (head 256, state 128, conv 4, <= 512 experts, top-k <= 15, "
               "<= 8 query heads a KV head)";
    // each tensor's type as load() reads it (q36_engine.cu): f32 (upload_f32), mat (the native MMVQ), the experts
    std::string bad;
    auto f32 = [&](const std::string& name) {
        const TensorInfo* t = m.find(name);
        if (!bad.empty()) return;
        if (t == nullptr) bad = "missing " + name;
        else if (!is_float(t->type) && !(t->type == 8 && t->elements() % 32 == 0))
            bad = name + " is " + t->type_name() + ", expected F32, BF16, F16 or Q8_0";
    };
    auto mat = [&](const std::string& name) {
        const TensorInfo* t = m.find(name);
        if (!bad.empty()) return;
        if (t == nullptr) bad = "missing " + name;
        else if (!K::native_mmvq_supported((int) t->type))
            bad = name + " is " + t->type_name() + ", which the native MMVQ does not take";
    };
    const bool has_mtp = g.n_mtp_layers > 0;
    for (int64_t l = 0; l < g.n_layers + (has_mtp ? 1 : 0) && bad.empty(); ++l) {
        const bool attn = l == g.n_layers || is_qsa_layer(g, l);
        f32(blk(l, "attn_norm.weight"));
        f32(blk(l, "post_attention_norm.weight"));
        if (!attn) {
            mat(blk(l, "attn_qkv.weight"));
            mat(blk(l, "attn_gate.weight"));
            mat(blk(l, "ssm_out.weight"));
            for (const char* n : {"ssm_alpha.weight", "ssm_beta.weight", "ssm_dt.bias", "ssm_a", "ssm_conv1d.weight",
                                  "ssm_norm.weight"})
                f32(blk(l, n));
        } else {
            for (const char* n : {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight"})
                mat(blk(l, n));
            f32(blk(l, "attn_q_norm.weight"));
            f32(blk(l, "attn_k_norm.weight"));
        }
        if (!moe) {
            for (const char* n : {"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"}) mat(blk(l, n));
            continue;
        }
        f32(blk(l, "ffn_gate_inp.weight"));
        for (const char* n : {"ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"}) mat(blk(l, n));
        f32(blk(l, "ffn_gate_inp_shexp.weight"));
        if (!bad.empty()) break;
        const TensorInfo* tg = m.find(blk(l, "ffn_gate_exps.weight"));
        const TensorInfo* tu = m.find(blk(l, "ffn_up_exps.weight"));
        const TensorInfo* td = m.find(blk(l, "ffn_down_exps.weight"));
        if (!tg || !tu || !td)
            return "missing " + blk(l, !tg   ? "ffn_gate_exps.weight"
                                       : !tu ? "ffn_up_exps.weight"
                                             : "ffn_down_exps.weight");
        if (tg->type != tu->type) return blk(l, "ffn_gate_exps / ffn_up_exps differ in type");
        if (!K::native_expert_supported((int) tg->type, (int) td->type, g.n_embd, g.n_ff))
            return blk(l, "experts are ") + tg->type_name() + "/" + td->type_name() +
                   ", which the native expert kernels do not take";
    }
    if (has_mtp && bad.empty()) {
        mat(blk(g.n_layers, "nextn.eh_proj.weight"));
        for (const char* n : {"nextn.enorm.weight", "nextn.hnorm.weight", "nextn.shared_head_norm.weight"})
            f32(blk(g.n_layers, n));
    }
    f32("output_norm.weight");
    mat("output.weight");
    if (!bad.empty()) return bad;
    const TensorInfo* t = m.find("token_embd.weight");
    if (t == nullptr) return "missing token_embd.weight";
    if (!K::iq_supported((int) t->type) || t->shape[0] % 256 != 0)
        return std::string("token_embd.weight is ") + t->type_name() + ", which the GPU embedding does not take";
    return {};
}

std::string hy4_file_problem(const GgufModel& m, const FileCheckOptions& o) {
    hy4::Geometry g;
    if (std::string e = hy4::read_geometry(m.meta(), g); !e.empty()) return e;
    if (std::string e = hy4::check_layout(m, g); !e.empty()) return e;
    // each tensor's type as load() reads it (hy4_engine.cu): the Loader's f32 and mat, and the named ones
    std::string bad;
    auto find = [&](const std::string& name) {
        const TensorInfo* t = m.find(name);
        if (t == nullptr && bad.empty()) bad = "missing " + name;
        return bad.empty() ? t : nullptr;
    };
    auto f32 = [&](const std::string& name) {
        if (const TensorInfo* t = find(name); t && t->type != 0 && t->type != 30)
            bad = name + " is " + t->type_name() + ", expected F32 or BF16";
    };
    auto mat = [&](const std::string& name) {
        if (const TensorInfo* t = find(name);
            t && t->type != 8 && ((t->type != 30 && t->type != 0) || t->shape[0] % 32 != 0))
            bad = name + " is " + t->type_name() +
                  "; this path reads its dense matrices as Q8_0 (or BF16 / F32 made Q8_0)";
    };
    if (const TensorInfo* t = find("per_layer_token_embd.weight"); t && t->type != 43)
        return "per_layer_token_embd.weight is " + std::string(t->type_name()) + ", expected F8_E4M3";
    if (const TensorInfo* t = find("token_embd.weight"); t && !K::iq_supported((int) t->type))
        return std::string("token_embd.weight is ") + t->type_name() + ", which iq_embed_rows does not read";
    for (const char* n :
         {"output_norm.weight", "output_hc_fn.weight", "output_hc_base.weight", "output_hc_scale.weight"})
        f32(n);
    if (const TensorInfo* t = find("output.weight"); t && t->type != 30)
        return std::string("output.weight is ") + t->type_name() + ", expected BF16";
    if (!bad.empty()) return bad;
    const bool has_mtp = o.mtp && g.n_mtp > 0;
    for (int64_t l = 0; l < g.n_layers + (has_mtp ? 1 : 0) && bad.empty(); ++l) {
        const bool mtp = l >= g.n_layers;
        const std::string p = "blk." + std::to_string(l) + ".";
        const bool swa = g.swa[(size_t) l] != 0, has_idx = g.has_indexer(l), dense = !mtp && l < g.n_dense;
        const int eng = mtp ? -1 : g.engram_index(l);
        if (!swa && !has_idx) return p + ": a sparse layer without an indexer, which this path does not run";
        for (const char* n : {"attn_norm.weight", "attn_sinks.weight", "attn_q_a_norm.weight", "attn_kv_a_norm.weight",
                              "post_attention_norm.weight", "ffn_norm.weight", "post_ffw_norm.weight"})
            f32(p + n);
        for (const char* n : {"attn_q_a.weight", "attn_q_b.weight", "attn_kv_a_mqa.weight", "attn_k_b.weight",
                              "attn_v_b.weight", "attn_output.weight", "attn_gate.weight"})
            mat(p + n);
        if (mtp) {
            mat(p + "nextn.eh_proj.weight");
            for (const char* n : {"nextn.enorm.weight", "nextn.hnorm.weight", "nextn.shared_head_norm.weight"})
                f32(p + n);
        } else {
            for (const char* n : {"hc_attn_fn.weight", "hc_attn_base.weight", "hc_attn_scale.weight",
                                  "hc_ffn_fn.weight", "hc_ffn_base.weight", "hc_ffn_scale.weight"})
                f32(p + n);
        }
        if (has_idx) {
            for (const char* n : {"indexer.attn_q_b.weight", "indexer.attn_k.weight", "indexer_compressor_value.weight",
                                  "indexer_compressor_gate.weight"})
                mat(p + n);
            for (const char* n : {"indexer.k_norm.weight", "indexer_compressor_ape.weight",
                                  "indexer_compressor_norm.weight", "indexer.proj.weight"})
                f32(p + n);
        }
        if (eng >= 0) {
            mat(p + "engram_kv.weight");
            for (const char* n : {"engram_kv_b.weight", "engram_gate.weight", "engram_lookup_norm.weight",
                                  "engram_pre_gamma.weight", "engram_hc_scale.weight"})
                f32(p + n);
        }
        if (dense) {
            for (const char* n : {"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"}) mat(p + n);
            continue;
        }
        f32(p + "ffn_gate_inp.weight");
        f32(p + "exp_probs_b.bias");
        for (const char* n : {"ffn_routed_down.weight", "ffn_routed_up.weight", "ffn_gate_shexp.weight",
                              "ffn_up_shexp.weight", "ffn_down_shexp.weight"})
            mat(p + n);
        if (!bad.empty()) break;
        // the routed experts (load()'s blobs): gate and up of one type, the CPU and GPU kernels' format, their layout
        const TensorInfo *tg = find(p + "ffn_gate_exps.weight"), *tu = find(p + "ffn_up_exps.weight"),
                         *td = find(p + "ffn_down_exps.weight");
        if (!bad.empty()) break;
        if (tg->type != tu->type) return p + "ffn_gate_exps and ffn_up_exps differ in type";
        const bool bf16 = tg->type == 30 && td->type == 30;   // made Q8_0 as they are read
        const int gt = bf16 ? 8 : (int) tg->type, dt = bf16 ? 8 : (int) td->type;
        C::NativeFmt fmt;
        std::string err;
        if (!C::native_fmt(gt, dt, g.n_latent, g.n_ff_exp, fmt, err)) return p + "routed experts: " + err;
        const size_t gsz = (size_t) (tensor_payload_bytes(*tg) / (uint64_t) g.n_expert);
        const size_t dsz = (size_t) (tensor_payload_bytes(*td) / (uint64_t) g.n_expert);
        if (!bf16 && (gsz != fmt.up_off || 2 * gsz + dsz != fmt.bytes))
            return p + "routed experts: unexpected blob layout";
    }
    if (!bad.empty()) return bad;
    // pictures: the vocabulary's token for a picture's rows (run_engine)
    if (o.vision && m.meta().find_string("tokenizer.ggml.tokens", "<\xEF\xBD\x9Chy_image\xE2\x96\x81pad") < 0)
        return "the model's vocabulary has no image pad token, so --vision has no cells to fill";
    return {};
}

std::string inkling2_file_problem(const GgufModel& m, const FileCheckOptions& o) {
    inkling2::Geometry g;
    if (std::string e = inkling2::read_geometry(m.meta(), g); !e.empty()) return e;
    if (std::string e = inkling2::check_layout(m, g); !e.empty()) return e;
    if (std::string e = inkling2::kernel_limits(g); !e.empty()) return e;
    const int64_t N = g.n_embd, E = g.n_expert;
    // the experts (load(): the kernels' formats at this geometry, E blobs of the layout's rows)
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        const TensorInfo* tg = m.find(p + "ffn_gate_up_exps.weight");
        const TensorInfo* td = m.find(p + "ffn_down_exps.weight");
        if (!tg || !td) return "missing " + p + (tg ? "ffn_down_exps.weight" : "ffn_gate_up_exps.weight");
        if (!K::native_expert_supported((int) tg->type, (int) td->type, N, g.n_ff, true))
            return p + ": experts of " + tg->type_name() + " gate/up and " + td->type_name() +
                   " down rows, which the "
                   "expert kernels do not read at this geometry";
        const K::NativeExpertLayout gl = K::native_expert_layout((int) tg->type, (int) td->type, N, g.n_ff);
        if (tensor_payload_bytes(*tg) != (uint64_t) E * gl.down_off ||
            tensor_payload_bytes(*td) != (uint64_t) E * (gl.bytes - gl.down_off))
            return p + ": the experts' tensors are not " + std::to_string(E) + " blobs of the layout's rows";
    }
    // each dense tensor as load() reads it (inkling2_engine.cu): floats (F32, or a type ggml dequantizes), mat
    std::string bad;
    auto find = [&](const std::string& name) {
        const TensorInfo* t = m.find(name);
        if (t == nullptr && bad.empty()) bad = "missing " + name;
        return bad.empty() ? t : nullptr;
    };
    auto floats = [&](const std::string& name) {
        if (const TensorInfo* t = find(name); t && !ggml_reads(t->type))
            bad = name + " is " + t->type_name() + ", which has no dequantizer here";
    };
    auto mat = [&](const std::string& name) {
        const TensorInfo* t = find(name);
        if (t == nullptr) return;
        const int type = (int) t->type, n_in = (int) t->shape[0];
        uint64_t rows = 1;
        for (size_t i = 1; i < t->shape.size(); ++i) rows *= t->shape[i];
        const bool takes = type == 30 ? n_in % 2 == 0
                                      : (K::native_mmvq_supported(type) || K::iq_supported(type)) &&
                                            K::iq_row_bytes(type, n_in) * rows == tensor_payload_bytes(*t);
        if (!takes)
            bad = name + " is " + t->type_name() + " with rows of " + std::to_string(n_in) +
                  " values, which this path's "
                  "kernels do not read";
    };
    if (const TensorInfo* t = find("token_embd.weight"); t && !ggml_reads(t->type))
        return std::string("token_embd.weight is ") + t->type_name() + ", which has no dequantizer here";
    floats("token_embd_norm.weight");
    floats("output_norm.weight");
    floats("output_res_query.weight");
    mat("output.weight");
    for (int64_t l = 0; l < g.n_layers && bad.empty(); ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        for (const char* n : {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_rel.weight", "attn_gate.weight",
                              "attn_output.weight"})
            mat(p + n);
        for (const char* n :
             {"attn_sink_proj.weight", "attn_q_clip.weight", "attn_rel_clip.weight", "attn_rel_bias.weight",
              "attn_k_conv.weight", "attn_v_conv.weight", "attn_post_conv.weight", "ffn_post_conv.weight"})
            floats(p + n);
        if (g.full(l)) {
            for (const char* n : {"attn_q_inp_norm.weight", "attn_k_inp_norm.weight", "attn_v_inp_norm.weight",
                                  "attn_k_conv_proj.weight", "attn_v_conv_proj.weight"})
                floats(p + n);
        } else {
            floats(p + "attn_norm.weight");
        }
        if (l > 0) floats(p + "attn_res_query.weight");
        for (const char* n : {"ffn_res_query.weight", "ffn_norm.weight", "ffn_gate_inp.weight"}) floats(p + n);
        if (m.find(p + "exp_probs_b.bias") != nullptr) floats(p + "exp_probs_b.bias");
        if (g.looped(l))
            for (int c = 0; c < inkling2::kCtrlCount; ++c) {
                floats(inkling2::msg_query_at(g, l, c).first);
                for (const char* part : {"down", "gain", "forget"}) floats(inkling2::ctrl_name(g, l, c, part));
            }
    }
    if (!bad.empty()) return bad;
    // pictures: the token the chat template names for a picture's rows (run_engine)
    if (o.vision) {
        int64_t pad = -1;
        if (const MetaValue* v = m.meta().get("tokenizer.chat_template"); v != nullptr) {
            const std::string open = "<|content_image|><|";
            if (const size_t at = v->s.find(open); at != std::string::npos) {
                const size_t from = at + open.size() - 2, to = v->s.find("|>", from);
                if (to != std::string::npos && to - from < 64)
                    pad = m.meta().find_string("tokenizer.ggml.tokens", v->s.substr(from, to + 2 - from));
            }
        }
        if (pad < 0)
            return "the model's chat template names no token for a picture's rows, so --vision has no cells to fill";
    }
    return {};
}

std::string lora_file_problem(const GgufFile& f, const GgufModel* m) {
    auto str = [&](const char* k) -> std::string {
        const MetaValue* v = f.get(k);
        return v != nullptr && v->type == MetaType::STRING ? v->s : "";
    };
    // the adapter's own (q36_engine.cu's lora_read)
    if (str("general.type") != "adapter")
        return "general.type is '" + str("general.type") + "', not 'adapter' (a LoRA adapter's GGUF)";
    if (str("adapter.type") != "lora") return "adapter.type is '" + str("adapter.type") + "', not 'lora'";
    const std::string arch = str("general.architecture");
    ModelGeometry g;
    if (m != nullptr) {
        int64_t top_k = 0;
        std::string err;
        if (!geometry_from_gguf(m->meta(), g, top_k, err)) return "the model: " + err;
        if (arch != model_family_arch(g.family))
            return "an adapter of " + arch + ", the model is " + model_family_arch(g.family);
    } else if (const ModelFamily fam = model_family_from_arch(arch);
               fam != ModelFamily::Qwen35 && fam != ModelFamily::Qwen35Moe) {
        return "an adapter of " + (arch.empty() ? std::string("no architecture") : arch) +
               ": Lamina takes LoRA adapters on Qwen3.6-35B-A3B and Qwen3.8-27B (qwen35moe, qwen35)";
    }
    if (f.get("adapter.alora.invocation_tokens") != nullptr)
        return "an activated LoRA (aLoRA), which this path does not run";
    struct Pair {
        const TensorInfo *a = nullptr, *b = nullptr;
    };
    std::map<std::string, Pair> pairs;
    for (const TensorInfo& t : f.tensors()) {
        const std::string& n = t.name;
        auto ends = [&](const char* s) {
            const size_t k = std::char_traits<char>::length(s);
            return n.size() > k && n.compare(n.size() - k, k, s) == 0;
        };
        if (ends(".lora_a") || ends(".lora_b"))
            (ends(".lora_a") ? pairs[n.substr(0, n.size() - 7)].a : pairs[n.substr(0, n.size() - 7)].b) = &t;
        else if (!ends("_norm.weight")) return "the tensor " + n + " has an unexpected suffix";
    }
    for (const auto& [name, p] : pairs) {
        if (p.a == nullptr || p.b == nullptr)
            return "the pair of " + name + " lacks its " + (p.a ? "lora_b" : "lora_a");
        if (p.a->shape.size() != 2 || p.b->shape.size() != 2 || p.a->shape[1] != p.b->shape[0] || p.a->shape[1] == 0)
            return name + ": lora_a and lora_b do not pair (an adapter of llama.cpp's old finetune example?)";
        for (const TensorInfo* t : {p.a, p.b})
            if (t->type != 0 && t->type != 1 && t->type != 8 && t->type != 30)
                return t->name + " is " + t->type_name() + ": F32, F16, BF16 or Q8_0 pairs only";
    }
    if (pairs.empty()) return "no LoRA pairs";
    if (m == nullptr) return {};
    // each pair's product in the model (lora_target, load_loras): a matrix the path computes, of the pair's widths
    for (const auto& [name, p] : pairs) {
        int64_t n_in = 0, n_out = 0;
        std::string why;
        const int64_t l = name.rfind("blk.", 0) == 0 ? std::strtoll(name.c_str() + 4, nullptr, 10) : -1;
        const size_t dot = name.find('.', 4);
        const std::string s = dot == std::string::npos ? std::string() : name.substr(dot + 1);
        static const std::set<std::string> mats = {
            "attn_qkv.weight",       "attn_gate.weight",    "ssm_out.weight",       "attn_q.weight", "attn_k.weight",
            "attn_v.weight",         "attn_output.weight",  "ffn_gate.weight",      "ffn_up.weight", "ffn_down.weight",
            "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"};
        const TensorInfo* t = m->find(name);
        auto take = [&] {
            if (t == nullptr || t->shape.size() < 2) {
                why = "the model has no such tensor";
                return;
            }
            n_in = (int64_t) t->shape[0];
            n_out = (int64_t) t->shape[1];
        };
        if (name == "output.weight") take();
        else if (name == "token_embd.weight") why = "this path takes no pair for the embedding";
        else if (l < 0 || l > g.n_layers || (l == g.n_layers && g.n_mtp_layers == 0) || dot == std::string::npos)
            why = "the model has no such tensor";
        else if (mats.count(s) != 0 || (s == "nextn.eh_proj.weight" && l == g.n_layers)) take();
        else if ((s == "ssm_alpha.weight" || s == "ssm_beta.weight") && t != nullptr && l < g.n_layers &&
                 !is_qsa_layer(g, l)) {
            n_in = g.n_embd;
            n_out = g.ssm_v_heads;
        } else if (s.find("_exps.") != std::string::npos) why = "this path takes no pair for the routed experts";
        else if (s.rfind("ffn_gate_inp", 0) == 0)
            why = "this path takes no pair for the router or the shared expert's gate";
        else why = "the model has no such tensor";
        if (!why.empty()) return name + ": " + why;
        const int64_t rank = (int64_t) p.a->shape[1], padded = (rank + 7) / 8 * 8;
        if ((int64_t) p.a->shape[0] != n_in || (int64_t) p.b->shape[1] != n_out)
            return name + " is " + std::to_string(n_in) + " x " + std::to_string(n_out) + " in the model, its pair " +
                   std::to_string(p.a->shape[0]) + " x " + std::to_string(p.b->shape[1]) +
                   " (an adapter of another model?)";
        if (n_in % 8 != 0 || padded > 512)
            return name + ": rank " + std::to_string(rank) + " over " + std::to_string(n_in) +
                   " inputs (ranks up to 512, inputs a multiple of 8)";
    }
    return {};
}

std::string ple_table_problem(const std::string& path) {
    K::PleTable table;
    K::PleIoOptions io;
    io.mode = K::PleIo::Mmap;   // (the header's checks alone: nothing of the table is read)
    std::string err;
    if (!table.open(path, err, io)) return err;
    return {};
}

std::string model_file_problem(const std::string& path, const FileCheckOptions& o) {
    try {
        const std::vector<std::string> shards = gguf_split_paths(path);
        ModelFamily family = ModelFamily::Unknown;
        {
            const GgufFile first(shards.front());
            if (std::string e = check_architecture(first); !e.empty()) return e;
            family = model_family(first);
        }
        switch (family) {
            case ModelFamily::Qwen35Moe:
            case ModelFamily::Qwen35: return qwen35_file_problem(GgufModel(shards));
            case ModelFamily::HyV4Flash: return hy4_file_problem(GgufModel(shards), o);
            case ModelFamily::Inkling2: return inkling2_file_problem(GgufModel(shards), o);
            default: return flash_file_problem(shards, o);
        }
    } catch (const std::exception& e) {
        return e.what();
    }
}

}   // namespace lamina::core
