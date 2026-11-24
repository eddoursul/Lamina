// include/lamina/inkling2/geometry.hpp - Inkling 2 (GGUF architectures inkling2 and inkling2_loop; Thinking Machines
// Lab's Inkling 2 Mini): the geometry read from a file's metadata, and its layout table (every tensor by name and GGUF
// shape) checked against the file's directory.
//
// The model as llama.cpp computes it (src/models/inkling2.cpp; docs: done 2026-11-21 Inkling 2 Mini support assessed):
// `n_layers` layers in blocks of `res_block`, of which [loop_start, loop_end) run `n_pass` times a token, each pass a
// layer APPLICATION with K/V and convolution state of its own (a "visit": 48 for Inkling 2 Mini's 24 layers).  A window
// layer attends its last `n_swa` cells, a full one every cell; positions enter as a bias by distance (`n_rel` values a
// head against a table of `rel_extent` distances), no RoPE.  Every layer's FFN is `n_expert` experts of which `top_k`
// run, gate and up in one matrix.
//
// Two layouts of the same tensors exist.  unsloth's files (llama.cpp's PR #32695): architecture `inkling2`, block_count
// the VISITS with the per-layer arrays in visit order, `num_loops` / `loop.core_start` / `loop.core_end`, the loop's
// tensors `blk.N.loop_{attn,k,v,mlp}_{down,gain,forget}` and two tensors of message queries for all looped layers.
// Mainline llama.cpp's (PR #32694): architecture `inkling2_loop`, block_count the LAYERS, `loop.count` / `loop.start` /
// `loop.end`, `blk.N.loop_{q,k,v,ffn}_ctrl_{down,gain,forget}` and a layer's `loop_msg_query`.  Both are read here; the
// names differ only where `Geometry::mainline` says.  Header-only and GPU-free.
#pragma once

#include "lamina/artifact/gguf_reader.hpp"

#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace lamina::inkling2 {

/// The loop's four message controllers, in mainline's order (a layer's loop_msg_query columns)
enum Ctrl { kCtrlQ = 0, kCtrlK = 1, kCtrlV = 2, kCtrlFfn = 3, kCtrlCount = 4 };

struct Geometry {
    bool mainline = false;   ///< the file's layout: inkling2_loop (mainline llama.cpp's), else unsloth's inkling2
    std::string arch;        ///< general.architecture, the metadata keys' prefix
    int64_t n_layers = 0;    ///< layers with weights (24)
    int64_t n_embd = 0;      ///< embedding_length (2,304)
    int64_t n_vocab = 0;     ///< vocab_size_unpadded: the tokens (200,058)
    int64_t n_vocab_padded = 0;   ///< vocab_size: the embedding's and the head's rows (201,024)
    float eps = 1e-6f;       ///< attention.layer_norm_rms_epsilon

    // attention
    int64_t n_head = 0;      ///< attention.head_count (24): query heads, the same in every layer
    int64_t n_head_kv = 0;   ///< attention.head_count_kv (8): K/V heads
    int64_t head_dim = 0;    ///< attention.key_length (128)
    std::vector<uint8_t> swa;   ///< per layer: a window layer (attention.sliding_window_pattern)
    int64_t n_swa = 0;       ///< attention.sliding_window (1,024): a window layer's cells, its own included
    int64_t n_rel = 0;       ///< attention.relative_length (16): the distance bias's values a head
    int64_t rel_extent = 0;  ///< attention.relative_extent (1,024): distances with a bias (a full layer's table)
    float qk_clip = 0;       ///< attention.qk_clip (22.63): a head's q and k rows have at most this L2 norm
    float rel_clip = 0;      ///< attention.relative_clip (8)
    int64_t log_floor = 0;   ///< attention.log_scale_floor (8,192)
    float log_alpha = 0;     ///< attention.log_scale_alpha (0.1): q and r of a full layer times 1 + alpha ln(max(1, (pos + 1) / floor))
    int64_t n_conv = 0;      ///< shortconv.l_cache (4): the causal convolutions' taps
    int64_t res_block = 0;   ///< attn_res.block_size (6): layers a block

    // MoE
    int64_t n_expert = 0;    ///< expert_count (128)
    int64_t top_k = 0;       ///< expert_used_count (8)
    int64_t n_ff = 0;        ///< expert_feed_forward_length (1,152): an expert's hidden width
    float w_scale = 1;       ///< expert_weights_scale (8)
    float situ_beta = 0;     ///< activation.situ_beta (4): the gate's cap
    float situ_lbeta = 0;    ///< activation.situ_linear_beta (25): up's cap
    float softcap = 0;       ///< final_logit_softcapping (30), 0: none

    // the loop: layers [loop_start, loop_end) run n_pass times; n_pass 1: no loop
    int64_t loop_start = 0, loop_end = 0, n_pass = 1, gate_rank = 0;

    int64_t core() const { return n_pass > 1 ? loop_end - loop_start : 0; }
    /// layer applications a token (48)
    int64_t n_visits() const { return n_layers + (n_pass - 1) * core(); }
    /// visit v's layer, in the order they run: the layers before the loop, the looped ones pass by pass, the rest
    int64_t visit_layer(int64_t v) const {
        if (n_pass < 2 || v < loop_start) return v;
        if (v < loop_start + n_pass * core()) return loop_start + (v - loop_start) % core();
        return v - (n_pass - 1) * core();
    }
    /// visit v's pass (0 outside the loop)
    int64_t visit_pass(int64_t v) const {
        return n_pass > 1 && v >= loop_start && v < loop_start + n_pass * core() ? (v - loop_start) / core() : 0;
    }
    /// layer l's first visit
    int64_t first_visit(int64_t l) const { return n_pass > 1 && l >= loop_end ? l + (n_pass - 1) * core() : l; }
    bool looped(int64_t l) const { return n_pass > 1 && l >= loop_start && l < loop_end; }
    bool full(int64_t l) const { return !swa[(size_t) l]; }
    /// the visits that keep every cell (8)
    int64_t n_full_visits() const {
        int64_t n = 0;
        for (int64_t v = 0; v < n_visits(); ++v) n += full(visit_layer(v));
        return n;
    }
    int64_t q_width() const { return n_head * head_dim; }       ///< 3,072
    int64_t kv_width() const { return n_head_kv * head_dim; }   ///< 1,024: a cell's K, and its V
    /// a cell's K (or V) bytes: halves, or q8_0 blocks (34 bytes a 32 values)
    int64_t cell_bytes(bool q8) const { return q8 ? n_head_kv * (head_dim / 32) * 34 : kv_width() * 2; }
    int64_t rel_width() const { return n_head * n_rel; }        ///< 384
    /// a visit's convolution state: K, V, the attention's output and the MoE's, n_conv - 1 earlier rows each
    int64_t conv_channels() const { return 2 * kv_width() + 2 * n_embd; }
};

/// Fills `g` from the metadata shard; "" or the first problem.
inline std::string read_geometry(const GgufFile& f, Geometry& g) {
    if (std::string e = check_architecture_inkling2(f); !e.empty()) return e;
    g = Geometry{};
    g.arch = f.get("general.architecture")->s;
    g.mainline = g.arch == "inkling2_loop";
    const std::string p = g.arch + ".";
    std::string err;
    auto num = [&](const std::string& k, bool need) -> const MetaValue* {
        const MetaValue* v = f.get(p + k);
        if (!v && need && err.empty()) err = "missing " + p + k;
        if (v && !v->is_num() && err.empty()) err = p + k + " is not a number";
        return v && v->is_num() ? v : nullptr;
    };
    auto i64 = [&](const std::string& k, int64_t& dst, bool need = true) {
        if (const MetaValue* v = num(k, need)) dst = (int64_t) v->u;
    };
    auto real = [&](const std::string& k, float& dst, bool need = true) {
        if (const MetaValue* v = num(k, need)) dst = (float) v->num();
    };
    int64_t block_count = 0, key = 0, value = 0;
    i64("block_count", block_count);
    i64("embedding_length", g.n_embd);
    i64("vocab_size", g.n_vocab_padded);
    i64("vocab_size_unpadded", g.n_vocab);
    real("attention.layer_norm_rms_epsilon", g.eps);
    i64("attention.key_length", key);
    i64("attention.value_length", value);
    i64("attention.sliding_window", g.n_swa);
    i64("attention.relative_length", g.n_rel);
    i64("attention.relative_extent", g.rel_extent);
    real("attention.qk_clip", g.qk_clip);
    real("attention.relative_clip", g.rel_clip);
    i64("attention.log_scale_floor", g.log_floor);
    real("attention.log_scale_alpha", g.log_alpha);
    i64("shortconv.l_cache", g.n_conv);
    i64("attn_res.block_size", g.res_block);
    i64("expert_count", g.n_expert);
    i64("expert_used_count", g.top_k);
    i64("expert_feed_forward_length", g.n_ff);
    real("expert_weights_scale", g.w_scale, false);
    real("activation.situ_beta", g.situ_beta);
    real("activation.situ_linear_beta", g.situ_lbeta);
    real("final_logit_softcapping", g.softcap, false);
    // the loop, by the layout's keys: mainline's are required, an unsloth file without num_loops has none
    if (g.mainline) {
        i64("loop.count", g.n_pass);
        i64("loop.start", g.loop_start);
        i64("loop.end", g.loop_end);
        i64("loop.gate_rank", g.gate_rank);
        if (const MetaValue* v = num("loop.varying", false); v && v->u != 0 && err.empty())
            err = p + "loop.varying: a model of a varying number of passes, which this engine does not run";
    } else if (f.get(p + "num_loops")) {
        i64("num_loops", g.n_pass);
        if (g.n_pass > 1) {
            i64("loop.core_start", g.loop_start);
            i64("loop.core_end", g.loop_end);
            i64("loop.gate_rank", g.gate_rank);
        }
    }
    if (!err.empty()) return err;

    auto bad = [&](const std::string& what) { return f.path() + ": " + what; };
    if (g.n_pass < 1) return bad("the loop's pass count is under 1");
    if (g.n_pass == 1) g.loop_start = g.loop_end = g.gate_rank = 0;
    // unsloth's block_count is the visits, mainline's the layers
    g.n_layers = g.mainline ? block_count : block_count - (g.n_pass - 1) * (g.loop_end - g.loop_start);
    if (g.n_layers <= 0 || g.n_layers > 4096) return bad("block_count out of range");
    if (g.n_pass > 1 && (g.loop_start <= 0 || g.loop_start >= g.loop_end || g.loop_end > g.n_layers || g.gate_rank <= 0 ||
                         g.res_block <= 0 || g.loop_start % g.res_block || g.loop_end % g.res_block))
        return bad("the loop's bounds do not lie on block boundaries inside the layers");

    // the per-layer arrays: one value a layer (mainline) or a visit (unsloth), or a single number
    const size_t n_file = (size_t) (g.mainline ? g.n_layers : g.n_visits());
    auto per_layer = [&](const std::string& k, std::vector<uint64_t>& out) -> std::string {
        std::vector<uint64_t> raw;
        if (const MetaValue* v = f.get(p + k); v && v->is_num()) raw.assign(n_file, v->u);
        else if (!f.numbers(p + k, &raw)) return bad(p + k + " is not a number or an array of numbers");
        if (raw.size() != n_file) return bad(p + k + " has " + std::to_string(raw.size()) + " values, the layout " +
                                             std::to_string(n_file));
        out.resize((size_t) g.n_layers);
        for (int64_t l = 0; l < g.n_layers; ++l) out[(size_t) l] = raw[(size_t) (g.mainline ? l : g.first_visit(l))];
        // a looped layer is one layer: its visits say the same
        if (!g.mainline)
            for (int64_t v = 0; v < g.n_visits(); ++v)
                if (raw[(size_t) v] != out[(size_t) g.visit_layer(v)]) return bad(p + k + ": a layer's visits differ");
        return {};
    };
    std::vector<uint64_t> heads, heads_kv, swa;
    if (std::string e = per_layer("attention.head_count", heads); !e.empty()) return e;
    if (std::string e = per_layer("attention.head_count_kv", heads_kv); !e.empty()) return e;
    if (std::string e = per_layer("attention.sliding_window_pattern", swa); !e.empty()) return e;
    g.n_head = (int64_t) heads[0];
    g.n_head_kv = (int64_t) heads_kv[0];
    for (int64_t l = 0; l < g.n_layers; ++l) {
        if ((int64_t) heads[(size_t) l] != g.n_head || (int64_t) heads_kv[(size_t) l] != g.n_head_kv)
            return bad("the layers' head counts differ, which this engine does not run");
        g.swa.push_back(swa[(size_t) l] ? 1 : 0);
    }
    g.head_dim = key;
    if (key != value) return bad("attention.key_length and value_length differ");
    if (const MetaValue* v = f.get("tokenizer.ggml.tokens"); v && v->type == MetaType::ARRAY && (int64_t) v->count != g.n_vocab)
        return bad("the tokenizer's size is not vocab_size_unpadded");
    if (g.n_embd <= 0 || g.n_embd % 32 || g.n_head <= 0 || g.n_head_kv <= 0 || g.n_head % g.n_head_kv || g.head_dim <= 0 ||
        g.head_dim % 32 || g.n_swa <= 0 || g.n_rel <= 0 || g.rel_extent <= 0 || g.n_conv < 2 || g.n_conv > 8 ||
        g.res_block <= 0 || g.n_expert <= 0 || g.top_k <= 0 || g.top_k > g.n_expert || g.n_ff <= 0 || g.n_ff % 32 ||
        g.n_vocab <= 0 || g.n_vocab > g.n_vocab_padded || g.log_floor <= 0 || !(g.qk_clip > 0) || !(g.rel_clip > 0) ||
        !(g.situ_beta > 0) || !(g.situ_lbeta > 0) || !(g.eps > 0) || g.log_alpha < 0)
        return bad("a geometry key is zero or out of range");
    return {};
}

struct TensorWant {
    std::string name;             ///< GGUF name
    std::vector<int64_t> shape;   ///< GGUF order
    bool optional = false;        ///< a tensor some files leave out (an expert matrix's per-expert scale)
};

/// A looped layer's controller matrix for message `c`: "down" [2 n_embd, rank], "gain" and "forget" [rank, n_embd]
inline std::string ctrl_name(const Geometry& g, int64_t l, int c, const char* part) {
    static const char* const ours[] = {"attn", "k", "v", "mlp"};   // unsloth's
    static const char* const theirs[] = {"q", "k", "v", "ffn"};    // mainline's
    const std::string b = "blk." + std::to_string(l) + ".loop_";
    return g.mainline ? b + theirs[c] + "_ctrl_" + part + ".weight" : b + ours[c] + "_" + part + ".weight";
}

/// Where a looped layer's message query `c` lies: the tensor and the query's index in it (n_embd floats apart).
/// unsloth: `loop_res_query` holds a layer's (attn, mlp) pair and `loop_kv_res_query` its (k, v) pair, layer after
/// layer; mainline: the layer's own `loop_msg_query`, q, k, v, ffn.
inline std::pair<std::string, int64_t> msg_query_at(const Geometry& g, int64_t l, int c) {
    if (g.mainline) return {"blk." + std::to_string(l) + ".loop_msg_query.weight", c};
    const int64_t k = 2 * (l - g.loop_start);
    switch (c) {
    case kCtrlQ: return {"loop_res_query.weight", k};
    case kCtrlFfn: return {"loop_res_query.weight", k + 1};
    case kCtrlK: return {"loop_kv_res_query.weight", k};
    default: return {"loop_kv_res_query.weight", k + 1};
    }
}

/// Every tensor of layer l: GGUF names, GGUF shapes.
inline std::vector<TensorWant> layer_tensors(const Geometry& g, int64_t l) {
    const int64_t N = g.n_embd, Q = g.q_width(), KV = g.kv_width(), F = g.n_ff, E = g.n_expert, C = g.n_conv;
    const bool full = g.full(l);
    const std::string b = "blk." + std::to_string(l) + ".";
    std::vector<TensorWant> t;
    auto add = [&](const std::string& s, std::vector<int64_t> shape, bool opt = false) { t.push_back({s, std::move(shape), opt}); };
    add(b + "attn_q.weight", {N, Q});
    add(b + "attn_k.weight", {N, KV});
    add(b + "attn_v.weight", {N, KV});
    add(b + "attn_output.weight", {Q, N});
    add(b + "attn_gate.weight", {N, Q});
    add(b + "attn_q_clip.weight", {g.head_dim, g.n_head});
    add(b + "attn_sink_proj.weight", {N, g.n_head});
    add(b + "attn_rel.weight", {N, g.rel_width()});
    add(b + "attn_rel_bias.weight", {g.n_rel, full ? g.rel_extent : g.n_swa});
    add(b + "attn_rel_clip.weight", {g.n_rel, g.n_head});
    add(b + "attn_k_conv.weight", {C, KV});
    add(b + "attn_v_conv.weight", {C, KV});
    add(b + "attn_post_conv.weight", {C, N});
    add(b + "ffn_post_conv.weight", {C, N});
    if (full) {
        add(b + "attn_q_inp_norm.weight", {N});
        add(b + "attn_k_inp_norm.weight", {N});
        add(b + "attn_v_inp_norm.weight", {N});
        add(b + "attn_k_conv_proj.weight", {N, C * g.n_head_kv});
        add(b + "attn_v_conv_proj.weight", {N, C * g.n_head_kv});
    } else {
        add(b + "attn_norm.weight", {N});
    }
    if (l > 0) add(b + "attn_res_query.weight", full ? std::vector<int64_t>{N, 3} : std::vector<int64_t>{N});
    add(b + "ffn_res_query.weight", {N});
    add(b + "ffn_norm.weight", {N});
    add(b + "ffn_gate_inp.weight", {N, E});
    add(b + "exp_probs_b.bias", {E}, true);
    add(b + "ffn_gate_up_exps.weight", {N, 2 * F, E});
    add(b + "ffn_gate_up_exps.scale", {E}, true);
    add(b + "ffn_down_exps.weight", {F, N, E});
    add(b + "ffn_down_exps.scale", {E}, true);
    if (g.looped(l)) {
        for (int c = 0; c < kCtrlCount; ++c) {
            add(ctrl_name(g, l, c, "down"), {2 * N, g.gate_rank});
            add(ctrl_name(g, l, c, "gain"), {g.gate_rank, N});
            add(ctrl_name(g, l, c, "forget"), {g.gate_rank, N});
        }
        if (g.mainline) add(b + "loop_msg_query.weight", {N, kCtrlCount});
    }
    return t;
}

inline std::vector<TensorWant> global_tensors(const Geometry& g) {
    std::vector<TensorWant> t = {{"token_embd.weight", {g.n_embd, g.n_vocab_padded}},
                                 {"token_embd_norm.weight", {g.n_embd}},
                                 {"output.weight", {g.n_embd, g.n_vocab_padded}},
                                 {"output_norm.weight", {g.n_embd}},
                                 {"output_res_query.weight", {g.n_embd}}};
    if (g.n_pass > 1 && !g.mainline) {
        t.push_back({"loop_res_query.weight", {g.n_embd, 2 * g.core()}});
        t.push_back({"loop_kv_res_query.weight", {g.n_embd, 2 * g.core()}});
    }
    return t;
}

/// Every table entry present with its shape, inside its file, of a type with a known block size, and no tensor the
/// table does not name; `checked` counts them.
inline std::string check_layout(const GgufModel& m, const Geometry& g, size_t* checked = nullptr) {
    std::set<std::string> named;
    size_t n = 0;
    auto check = [&](const TensorWant& w) -> std::string {
        named.insert(w.name);
        size_t s = 0;
        const TensorInfo* t = m.find(w.name, &s);
        if (!t) return w.optional ? std::string() : "missing " + w.name;
        bool same = t->shape.size() == w.shape.size();
        for (size_t i = 0; same && i < w.shape.size(); ++i) same = t->shape[i] == (uint64_t) w.shape[i];
        if (!same) {
            std::string got, want;
            for (uint64_t x : t->shape) got += (got.empty() ? "" : ", ") + std::to_string(x);
            for (int64_t x : w.shape) want += (want.empty() ? "" : ", ") + std::to_string(x);
            return w.name + " is [" + got + "], Inkling 2's layout wants [" + want + "]";
        }
        if (!m.in_bounds(*t, s)) return w.name + " (" + t->type_name() + ") has no known size inside " + m.shard(s).path();
        ++n;
        return {};
    };
    for (const TensorWant& w : global_tensors(g))
        if (std::string e = check(w); !e.empty()) return e;
    for (int64_t l = 0; l < g.n_layers; ++l)
        for (const TensorWant& w : layer_tensors(g, l))
            if (std::string e = check(w); !e.empty()) return e;
    for (size_t s = 0; s < m.size(); ++s)
        for (const TensorInfo& t : m.shard(s).tensors())
            if (!named.count(t.name)) return t.name + " is not in Inkling 2's layout";
    if (checked) *checked = n;
    return {};
}

}  // namespace lamina::inkling2
