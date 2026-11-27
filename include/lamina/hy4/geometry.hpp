// include/lamina/hy4/geometry.hpp - Hy4-flash (GGUF architecture hy_v4_flash): the geometry read from a file's
// metadata, and its layout table (every tensor by name and GGUF shape) checked against the file's directory.
//
// The model as llama.cpp's PR #32517 computes it (src/models/hy-v4-flash.cpp; docs: done 2026-11-18 Hy4-flash support
// assessed): a trunk of `n_layers` layers, each with four hyper-connection streams (iHC) around gated MLA attention and
// an FFN, sandwich norms; the sliding-window layers attend their last `n_swa` cells, the sparse ones at most
// `idx_top_k` cells an indexer picks by groups of `idx_pool`; layer 0's FFN is dense, the others' latent experts;
// Engram rows from an n-gram hash join the streams before `eng_layers`; one MTP block after the trunk (a sparse layer
// without iHC).  Header-only and GPU-free: the engine path, the tools and the tests share it.
#pragma once

#include "lamina/artifact/gguf_reader.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

namespace lamina::hy4 {

struct Geometry {
    int64_t n_layers = 0;   ///< the trunk (block_count - nextn_predict_layers)
    int64_t n_mtp = 0;      ///< nextn_predict_layers: the MTP blocks after the trunk
    int64_t n_embd = 0;     ///< embedding_length (3,072)
    int64_t n_vocab = 0;    ///< the length of tokenizer.ggml.tokens
    float eps = 1e-5f;      ///< attention.layer_norm_rms_epsilon

    // gated MLA, one latent row a token (head_count_kv 1)
    int64_t q_lora = 0;            ///< attention.q_lora_rank (1,024)
    int64_t kv_lora = 0;           ///< attention.kv_lora_rank (512): the latent, also the attention's value width
    int64_t k_head = 0;            ///< attention.key_length_mla (256): a query head before absorption, nope + rope
    int64_t v_head = 0;            ///< attention.value_length_mla (256): a head's output after v_b
    int64_t n_rot = 0;             ///< rope.dimension_count (64), in consecutive pairs (ROPE_TYPE_NORM)
    std::vector<int64_t> n_head;   ///< attention.head_count, per layer of block_count (sparse 32, window 64)
    std::vector<uint8_t> swa;      ///< attention.sliding_window_pattern, per layer of block_count
    int64_t n_swa = 0;             ///< attention.sliding_window (512): a window layer's cells, its own included
    double rope_base = 0;          ///< rope.freq_base (1e7): the sparse layers'
    double rope_base_swa = 0;      ///< rope.freq_base_swa (1e4): the window layers'
    /// rope.dimension_sections: the M-RoPE sections of a picture's and a video frame's cells (t, h, w), the pairs
    /// interleaved; where the file names none (AngelSlim's), the checkpoint's mrope_section, as vLLM defaults it
    int64_t mrope[3] = {12, 10, 10};

    // the sparse layers' indexer
    int64_t idx_heads = 0;      ///< attention.indexer.head_count (16)
    int64_t idx_dim = 0;        ///< attention.indexer.key_length (128): rope on its last n_rot values
    int64_t idx_top_k = 0;      ///< attention.indexer.top_k (2,048): cells a query attends at most
    int64_t idx_pool = 0;       ///< attention.indexer.kpool (4): cells a pooled key covers
    bool idx_tail = false;      ///< attention.indexer.kpool_select_tail: the query's own group is always attended
    std::vector<uint8_t> idx;   ///< attention.indexer.types, per trunk layer (the MTP block always has one)

    // iHC
    int64_t hc = 0;     ///< hyper_connection.count (4 streams)
    float hc_eps = 0;   ///< hyper_connection.epsilon
    float hc_mag = 0;   ///< hyper_connection.magnitude (the post weights' scale)

    // FFN
    int64_t n_dense = 0;               ///< leading_dense_block_count (1)
    int64_t n_ff = 0;                  ///< feed_forward_length (13,824): the dense FFN's
    int64_t n_expert = 0;              ///< expert_count (512)
    int64_t top_k = 0;                 ///< expert_used_count (16)
    int64_t n_ff_exp = 0;              ///< expert_feed_forward_length (1,536): an
                                       ///< expert's hidden width, the shared expert's too
    int64_t n_latent = 0;              ///< expert_latent_length (1,536): the routed experts' input and output width
    int64_t n_shared = 0;              ///< expert_shared_count (1)
    float w_scale = 1;                 ///< expert_weights_scale (3.998)
    bool w_norm = false;               ///< expert_weights_norm
    std::vector<float> swiglu_limit;   ///< swiglu_clamp_exp per layer of block_count (10): the routed experts' only

    // Engram
    std::vector<int64_t> eng_layers;   ///< ple.layers (1, 11, 21, 31): the trunk layers its rows join, before attention
    int64_t ngram = 0;                 ///< ple.ngram_size (3): the token and two predecessors
    int64_t heads_per_ngram = 0;       ///< ple.heads_per_ngram (8)
    int64_t eng_dim = 0;               ///< embedding_length_per_layer_input (96): a row's values
    std::vector<uint64_t> eng_mult;    ///< ple.layer_multipliers [module][ngram]
    std::vector<uint64_t> eng_off;     ///< ple.head_offsets [module][head]: a head's first row in the table
    std::vector<uint64_t> eng_prime;   ///< ple.head_vocab_sizes [module][head]: its rows (the hash's modulus)
    int64_t eng_pad = 0;               ///< engram.pad_token_id (a missing predecessor)
    std::vector<int32_t> eng_map;      ///< engram.token_map [n_vocab]: the hash's compressed token ids
    uint64_t eng_rows = 0;             ///< the table's rows: the last head's end

    int64_t eng_heads() const { return (ngram - 1) * heads_per_ngram; }   ///< a module's rows a token (16)
    int64_t nope() const { return k_head - n_rot; }                       ///< 192
    int64_t kv_row() const { return kv_lora + n_rot; }                    ///< 576: a cached latent row
    bool sparse(int64_t l) const { return !swa[(size_t) l]; }
    bool has_indexer(int64_t l) const { return l >= n_layers || idx[(size_t) l]; }
    /// Layer l's Engram module, or -1
    int engram_index(int64_t l) const {
        for (size_t e = 0; e < eng_layers.size(); ++e)
            if (eng_layers[e] == l) return (int) e;
        return -1;
    }
    double base(int64_t l) const { return swa[(size_t) l] ? rope_base_swa : rope_base; }
};

/// Fills `g` from the metadata shard; "" or the first problem.
inline std::string read_geometry(const GgufFile& f, Geometry& g) {
    if (std::string e = check_architecture_hy4(f); !e.empty()) return e;
    g = Geometry{};
    const std::string p = "hy_v4_flash.";
    std::string err;
    auto num = [&](const char* k, bool need) -> const MetaValue* {
        const MetaValue* v = f.get(p + k);
        if (!v && need && err.empty()) err = "missing " + p + k;
        if (v && !v->is_num() && err.empty()) err = p + k + " is not a number";
        return v && v->is_num() ? v : nullptr;
    };
    auto i64 = [&](const char* k, int64_t& dst) {
        if (const MetaValue* v = num(k, true)) dst = (int64_t) v->u;
    };
    auto real = [&](const char* k, double& dst) {
        if (const MetaValue* v = num(k, true)) dst = v->num();
    };
    int64_t block_count = 0;
    i64("block_count", block_count);
    i64("nextn_predict_layers", g.n_mtp);
    i64("embedding_length", g.n_embd);
    double d = 0;
    real("attention.layer_norm_rms_epsilon", d);
    g.eps = (float) d;
    i64("attention.q_lora_rank", g.q_lora);
    i64("attention.kv_lora_rank", g.kv_lora);
    i64("attention.key_length_mla", g.k_head);
    i64("attention.value_length_mla", g.v_head);
    i64("rope.dimension_count", g.n_rot);
    i64("attention.sliding_window", g.n_swa);
    real("rope.freq_base", g.rope_base);
    real("rope.freq_base_swa", g.rope_base_swa);
    i64("attention.indexer.head_count", g.idx_heads);
    i64("attention.indexer.key_length", g.idx_dim);
    i64("attention.indexer.top_k", g.idx_top_k);
    i64("attention.indexer.kpool", g.idx_pool);
    if (const MetaValue* v = num("attention.indexer.kpool_select_tail", false)) g.idx_tail = v->u != 0;
    i64("hyper_connection.count", g.hc);
    real("hyper_connection.epsilon", d);
    g.hc_eps = (float) d;
    real("hyper_connection.magnitude", d);
    g.hc_mag = (float) d;
    i64("leading_dense_block_count", g.n_dense);
    i64("feed_forward_length", g.n_ff);
    i64("expert_count", g.n_expert);
    i64("expert_used_count", g.top_k);
    i64("expert_feed_forward_length", g.n_ff_exp);
    i64("expert_latent_length", g.n_latent);
    i64("expert_shared_count", g.n_shared);
    real("expert_weights_scale", d);
    g.w_scale = (float) d;
    if (const MetaValue* v = num("expert_weights_norm", true)) g.w_norm = v->u != 0;
    int64_t gating = 0;
    i64("expert_gating_func", gating);
    i64("ple.ngram_size", g.ngram);
    i64("ple.heads_per_ngram", g.heads_per_ngram);
    i64("embedding_length_per_layer_input", g.eng_dim);
    i64("engram.pad_token_id", g.eng_pad);
    if (!err.empty()) return err;

    auto ints = [&](const char* k, std::vector<uint64_t>& out) {
        if (!f.numbers(p + k, &out) && err.empty()) err = p + k + ": not an array of numbers";
    };
    std::vector<uint64_t> heads, swa, idx, layers, map;
    std::vector<double> limits;
    ints("attention.head_count", heads);
    ints("attention.sliding_window_pattern", swa);
    ints("attention.indexer.types", idx);
    ints("ple.layers", layers);
    ints("ple.layer_multipliers", g.eng_mult);
    ints("ple.head_offsets", g.eng_off);
    ints("ple.head_vocab_sizes", g.eng_prime);
    ints("engram.token_map", map);
    if (!f.numbers(p + "swiglu_clamp_exp", nullptr, &limits) && err.empty()) err = p + "swiglu_clamp_exp: not an array";
    if (std::vector<uint64_t> sec; f.get(p + "rope.dimension_sections") != nullptr) {
        if (!f.numbers(p + "rope.dimension_sections", &sec) || sec.size() < 3) {
            if (err.empty()) err = p + "rope.dimension_sections: not three numbers or more";
        } else {
            for (int k = 0; k < 3; ++k) g.mrope[k] = (int64_t) sec[(size_t) k];
        }
    }
    if (!err.empty()) return err;
    if (const MetaValue* v = f.get("tokenizer.ggml.tokens"); v && v->type == MetaType::ARRAY)
        g.n_vocab = (int64_t) v->count;

    auto bad = [&](const std::string& what) { return f.path() + ": " + what; };
    g.n_layers = block_count - g.n_mtp;
    const size_t nb = (size_t) block_count;
    if (g.n_layers <= 0 || g.n_mtp < 0 || g.n_mtp > 1) return bad("block_count / nextn_predict_layers out of range");
    if (heads.size() != nb || swa.size() != nb || limits.size() != nb)
        return bad("attention.head_count, sliding_window_pattern and swiglu_clamp_exp need block_count values");
    if (idx.size() != (size_t) g.n_layers) return bad("attention.indexer.types needs a value per trunk layer");
    for (size_t l = 0; l < nb; ++l) {
        g.n_head.push_back((int64_t) heads[l]);
        g.swa.push_back(swa[l] ? 1 : 0);
        g.swiglu_limit.push_back((float) limits[l]);
    }
    for (uint64_t v : idx) g.idx.push_back(v ? 1 : 0);
    if (g.n_mtp && g.swa.back()) return bad("the MTP block is a window layer");
    for (int64_t l = 0; l < g.n_layers; ++l)
        if (g.idx[(size_t) l] && g.swa[(size_t) l])
            return bad("layer " + std::to_string(l) + ": a window layer with an indexer");
    for (int64_t h : g.n_head)
        if (h <= 0) return bad("a layer of no heads");
    if (const MetaValue* kv = f.get(p + "attention.head_count_kv"); kv->u != 1)
        return bad("attention.head_count_kv is not 1 (MLA)");
    if (g.n_embd <= 0 || g.q_lora <= 0 || g.kv_lora <= 0 || g.n_rot <= 0 || g.k_head <= g.n_rot || g.v_head <= 0 ||
        g.n_swa <= 0 || g.idx_heads <= 0 || g.idx_dim <= g.n_rot || g.idx_pool < 2 || g.idx_top_k % g.idx_pool ||
        (g.idx_tail && g.idx_top_k / g.idx_pool < 2) || g.hc <= 0 || g.n_dense < 0 || g.n_ff <= 0 || g.n_expert <= 0 ||
        g.top_k <= 0 || g.top_k > g.n_expert || g.n_ff_exp <= 0 || g.n_latent <= 0 || g.n_shared != 1 || g.n_vocab <= 0)
        return bad("a geometry key is zero or out of range");
    if (gating != 2) return bad("expert_gating_func is not 2 (sigmoid)");
    // Engram: multipliers [module][ngram], offsets and primes [module][head]
    const size_t n_e = layers.size(), n_h = (size_t) g.eng_heads();
    if (n_e == 0 || g.ngram < 2 || g.heads_per_ngram <= 0 || g.eng_dim <= 0 ||
        g.eng_mult.size() != n_e * (size_t) g.ngram || g.eng_off.size() != n_e * n_h ||
        g.eng_prime.size() != n_e * n_h || map.size() != (size_t) g.n_vocab || g.eng_pad < 0 ||
        g.eng_pad >= (int64_t) map.size())
        return bad("the Engram metadata is inconsistent");
    for (size_t e = 0; e < n_e; ++e) {
        if ((int64_t) layers[e] >= g.n_layers || (e > 0 && layers[e] <= layers[e - 1]))
            return bad("an Engram layer out of range or out of order");
        g.eng_layers.push_back((int64_t) layers[e]);
    }
    for (size_t c = 0; c < g.eng_off.size(); ++c) {
        if (g.eng_prime[c] < 2 || g.eng_off[c] + g.eng_prime[c] > (uint64_t) INT32_MAX)
            return bad("an Engram head's rows out of range");
        g.eng_rows = std::max(g.eng_rows, g.eng_off[c] + g.eng_prime[c]);
    }
    for (uint64_t v : map) g.eng_map.push_back((int32_t) v);
    return {};
}

struct TensorWant {
    std::string name;
    std::vector<int64_t> shape;
};

/// Every tensor of block l (a trunk layer, or the MTP block at
/// l >= n_layers): GGUF names (blk.N.<suffix>), GGUF shapes.
inline std::vector<TensorWant> layer_tensors(const Geometry& g, int64_t l) {
    const int64_t N = g.n_embd, H = g.n_head[(size_t) l], hc = g.hc;
    const bool mtp = l >= g.n_layers;
    std::vector<TensorWant> t;
    auto add = [&](const char* s, std::vector<int64_t> shape) { t.push_back({s, std::move(shape)}); };
    add("attn_norm.weight", {N});
    add("attn_sinks.weight", {H});
    add("attn_q_a.weight", {N, g.q_lora});
    add("attn_q_a_norm.weight", {g.q_lora});
    add("attn_q_b.weight", {g.q_lora, H * g.k_head});
    add("attn_kv_a_mqa.weight", {N, g.kv_row()});
    add("attn_kv_a_norm.weight", {g.kv_lora});
    add("attn_k_b.weight", {g.nope(), g.kv_lora, H});
    add("attn_v_b.weight", {g.kv_lora, g.v_head, H});
    add("attn_output.weight", {H * g.v_head, N});
    add("attn_gate.weight", {N, H * g.v_head});
    add("post_attention_norm.weight", {N});
    add("ffn_norm.weight", {N});
    add("post_ffw_norm.weight", {N});
    if (!mtp) {
        add("hc_attn_fn.weight", {hc * N, 2 * hc});
        add("hc_attn_base.weight", {2 * hc});
        add("hc_attn_scale.weight", {2});
        add("hc_ffn_fn.weight", {hc * N, 2 * hc});
        add("hc_ffn_base.weight", {2 * hc});
        add("hc_ffn_scale.weight", {2});
    } else {
        add("nextn.enorm.weight", {N});
        add("nextn.hnorm.weight", {N});
        add("nextn.eh_proj.weight", {2 * N, N});
        add("nextn.shared_head_norm.weight", {N});
    }
    if (g.has_indexer(l)) {
        add("indexer.attn_k.weight", {N, g.idx_dim});
        add("indexer.k_norm.weight", {g.idx_dim});
        add("indexer_compressor_value.weight", {g.idx_dim, g.idx_dim});
        add("indexer_compressor_gate.weight", {g.idx_dim, g.idx_dim});
        add("indexer_compressor_ape.weight", {g.idx_dim, g.idx_pool});
        add("indexer_compressor_norm.weight", {g.idx_dim});
        add("indexer.attn_q_b.weight", {g.q_lora, g.idx_dim * g.idx_heads});
        add("indexer.proj.weight", {N, g.idx_heads});
    }
    if (!mtp && g.engram_index(l) >= 0) {
        add("engram_kv.weight", {g.eng_heads() * g.eng_dim, N * (hc + 1)});
        add("engram_kv_b.weight", {N, hc + 1});
        add("engram_gate.weight", {N, hc});
        add("engram_lookup_norm.weight", {g.eng_dim, g.eng_heads()});
        add("engram_pre_gamma.weight", {N});
        add("engram_hc_scale.weight", {N, hc});
    }
    if (!mtp && l < g.n_dense) {
        add("ffn_gate.weight", {N, g.n_ff});
        add("ffn_up.weight", {N, g.n_ff});
        add("ffn_down.weight", {g.n_ff, N});
    } else {
        const int64_t L = g.n_latent, F = g.n_ff_exp, E = g.n_expert;
        add("ffn_gate_inp.weight", {N, E});
        add("exp_probs_b.bias", {E});
        add("ffn_routed_down.weight", {N, L});
        add("ffn_routed_up.weight", {L, N});
        add("ffn_gate_exps.weight", {L, F, E});
        add("ffn_up_exps.weight", {L, F, E});
        add("ffn_down_exps.weight", {F, L, E});
        add("ffn_gate_shexp.weight", {N, F * g.n_shared});
        add("ffn_up_shexp.weight", {N, F * g.n_shared});
        add("ffn_down_shexp.weight", {F * g.n_shared, N});
    }
    return t;
}

inline std::vector<TensorWant> global_tensors(const Geometry& g) {
    return {{"token_embd.weight", {g.n_embd, g.n_vocab}},
            {"output_norm.weight", {g.n_embd}},
            {"output.weight", {g.n_embd, g.n_vocab}},
            {"output_hc_fn.weight", {g.hc * g.n_embd, g.hc}},
            {"output_hc_base.weight", {g.hc}},
            {"output_hc_scale.weight", {1}},
            {"per_layer_token_embd.weight", {g.eng_dim, (int64_t) g.eng_rows}}};
}

/// Every table entry present with its shape, inside its file, of a type with a known block size, and no tensor the
/// table does not name; `checked` counts them.
inline std::string check_layout(const GgufModel& m, const Geometry& g, size_t* checked = nullptr) {
    std::set<std::string> named;
    size_t n = 0;
    auto check = [&](const std::string& name, const std::vector<int64_t>& shape) -> std::string {
        named.insert(name);
        size_t s = 0;
        const TensorInfo* t = m.find(name, &s);
        if (!t) return "missing " + name;
        bool same = t->shape.size() == shape.size();
        for (size_t i = 0; same && i < shape.size(); ++i) same = t->shape[i] == (uint64_t) shape[i];
        if (!same) {
            std::string got, want;
            for (uint64_t x : t->shape) got += (got.empty() ? "" : ", ") + std::to_string(x);
            for (int64_t x : shape) want += (want.empty() ? "" : ", ") + std::to_string(x);
            return name + " is [" + got + "], Hy4-flash's layout wants [" + want + "]";
        }
        if (!m.in_bounds(*t, s))
            return name + " (" + t->type_name() + ") has no known size inside " + m.shard(s).path();
        ++n;
        return {};
    };
    for (const TensorWant& w : global_tensors(g))
        if (std::string e = check(w.name, w.shape); !e.empty()) return e;
    for (int64_t l = 0; l < g.n_layers + g.n_mtp; ++l)
        for (const TensorWant& w : layer_tensors(g, l))
            if (std::string e = check("blk." + std::to_string(l) + "." + w.name, w.shape); !e.empty()) return e;
    for (size_t s = 0; s < m.size(); ++s)
        for (const TensorInfo& t : m.shard(s).tensors())
            if (!named.count(t.name)) return t.name + " is not in Hy4-flash's layout";
    if (checked) *checked = n;
    return {};
}

}   // namespace lamina::hy4
