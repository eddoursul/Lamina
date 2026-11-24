// src/vision/ggml_encoder.cpp - the image encoder on the CPU, a ggml graph (include/lamina/vision/encoder.hpp).
//
// What llama.cpp's mtmd does for these two projector types at the pinned commit, kept operation for operation and in
// its order (tools/mtmd/clip.cpp, models/qwen3vl.cpp, models/hunyuanvl.cpp, MIT; Hy4's adapter and limits from pull
// request 32517's models/hy-v4-flash.cpp): the weights in their GGUF types in ggml's CPU backend, a graph built per
// picture for its size.  The graph's nodes are added in mtmd's order, so ggml's kernels see the same work and the rows
// come out the same.
#include "backends.hpp"

#include "lamina/vision/projector.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace lamina::vision {
namespace {

void quiet_log(ggml_log_level level, const char* text, void*) {
    if (level >= GGML_LOG_LEVEL_WARN) std::fputs(text, stderr);
}

// ggml's clock, its FP16 tables and its backends, once (what llama_backend_init does); its messages from warnings up
void backends_init() {
    static const bool once = [] {
        ggml_log_set(quiet_log, nullptr);
        ggml_time_init();
        ggml_init_params ip = {0, nullptr, false};
        ggml_free(ggml_init(ip));
        if (ggml_backend_reg_count() == 0) ggml_backend_load_all();
        return true;
    }();
    (void) once;
}

using Kind = ProjectorKind;

struct Layer {
    ggml_tensor* ln_1_w = nullptr;
    ggml_tensor* ln_1_b = nullptr;
    ggml_tensor* ln_2_w = nullptr;
    ggml_tensor* ln_2_b = nullptr;
    ggml_tensor* qkv_w = nullptr;   // the three projections as one matrix (Qwen's files) ...
    ggml_tensor* qkv_b = nullptr;
    ggml_tensor* q_w = nullptr;     // ... or each its own (Hy4's)
    ggml_tensor* q_b = nullptr;
    ggml_tensor* k_w = nullptr;
    ggml_tensor* k_b = nullptr;
    ggml_tensor* v_w = nullptr;
    ggml_tensor* v_b = nullptr;
    ggml_tensor* o_w = nullptr;
    ggml_tensor* o_b = nullptr;
    ggml_tensor* ff_up_w = nullptr;
    ggml_tensor* ff_up_b = nullptr;
    ggml_tensor* ff_down_w = nullptr;
    ggml_tensor* ff_down_b = nullptr;
    ggml_tensor* ls_1_w = nullptr;   // Inkling 2's: the block's attention and FFN outputs times a learned vector
    ggml_tensor* ls_2_w = nullptr;
};

struct Model : ProjectorParams {
    ggml_tensor* patch_embd_0 = nullptr;
    ggml_tensor* patch_embd_1 = nullptr;
    ggml_tensor* patch_bias = nullptr;
    ggml_tensor* position_embd = nullptr;
    ggml_tensor* pre_ln_w = nullptr;
    ggml_tensor* pre_ln_b = nullptr;
    ggml_tensor* post_ln_w = nullptr;
    ggml_tensor* post_ln_b = nullptr;
    std::vector<Layer> layers;
    ggml_tensor* mm_0_w = nullptr;
    ggml_tensor* mm_0_b = nullptr;
    ggml_tensor* mm_1_w = nullptr;
    ggml_tensor* mm_1_b = nullptr;
    ggml_tensor* mm_pre_norm_w = nullptr;   // Hy4's adapter
    ggml_tensor* image_newline = nullptr;
    ggml_tensor* class_embd = nullptr;      // Inkling 2's class token and its position
    ggml_tensor* class_pos_embd = nullptr;
};

constexpr int kMaxNodes = 8192;

// A picture's graph: the builders are mtmd's clip_graph's, by name.
struct Graph {
    const Model& m;
    ggml_context* ctx0;
    ggml_cgraph* gf;
    const int img_w, img_h;
    const int patch_size, n_patches_x, n_patches_y, n_patches;
    const int n_embd, n_head, n_head_kv, d_head, n_layer;
    const float eps, kq_scale;
    const bool flash;
    const int head_pad;      // the flash attention's heads widened by this many zeros (GgmlEncoder::open)
    const int n_batch = 1;   // a still picture
    // the inputs the encoder fills
    ggml_tensor* inp_raw = nullptr;
    ggml_tensor* positions = nullptr;   // Qwen's rotary positions
    ggml_tensor* pos_embd = nullptr;    // Hy4's resampled position table
    ggml_tensor* pos_w = nullptr;       // Inkling 2's rotary positions: a row's column and its row (the class token 0)
    ggml_tensor* pos_h = nullptr;
    // LAMINA_VISION_DUMP: tensors kept past the graph's run (each an output, so its buffer is not reused)
    bool dump = false;
    std::vector<std::pair<std::string, ggml_tensor*>> taps;
    void tap(ggml_tensor* t, const std::string& name) {
        if (!dump) return;
        ggml_set_output(t);
        taps.emplace_back(name, t);
    }
    bool tap_layer(int il) const { return il < 2 || il == n_layer - 1; }
    bool tap_attn = false;   // the next attention's output before its projection

    Graph(const Model& model, ggml_context* ctx, int w, int h, bool flash_attention, int pad)
        : m(model), ctx0(ctx), gf(ggml_new_graph_custom(ctx, kMaxNodes, false)), img_w(w), img_h(h),
          patch_size(model.patch_size), n_patches_x(w / model.patch_size), n_patches_y(h / model.patch_size),
          n_patches(n_patches_x * n_patches_y), n_embd(model.n_embd), n_head(model.n_head),
          n_head_kv(model.n_head_kv), d_head(model.d_head()),
          n_layer(model.n_layer), eps(model.eps), kq_scale(1.0f / sqrtf((float) d_head)), flash(flash_attention),
          head_pad(pad) {}

    ggml_tensor* build_inp_raw() {
        inp_raw = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, img_w, img_h, 3, n_batch);
        ggml_set_name(inp_raw, "inp_raw");
        ggml_set_input(inp_raw);
        return inp_raw;
    }

    ggml_tensor* build_norm(ggml_tensor* cur, ggml_tensor* mw, ggml_tensor* mb, bool rms, float norm_eps) const {
        cur = rms ? ggml_rms_norm(ctx0, cur, norm_eps) : ggml_norm(ctx0, cur, norm_eps);
        if (mw) cur = ggml_mul(ctx0, cur, mw);
        if (mb) cur = ggml_add(ctx0, cur, mb);
        return cur;
    }

    // up, GELU (the tanh form, ggml_gelu; `erf`: the exact one), down
    ggml_tensor* build_ffn(ggml_tensor* cur, ggml_tensor* up, ggml_tensor* up_b, ggml_tensor* down,
                           ggml_tensor* down_b, bool erf = false) const {
        ggml_tensor* tmp = ggml_mul_mat(ctx0, up, cur);
        if (up_b) tmp = ggml_add(ctx0, tmp, up_b);
        cur = erf ? ggml_gelu_erf(ctx0, tmp) : ggml_gelu(ctx0, tmp);
        cur = ggml_mul_mat(ctx0, down, cur);
        if (down_b) cur = ggml_add(ctx0, cur, down_b);
        return cur;
    }

    // q, k, v [d_head, n_head, n_pos, B] -> the block's attention output [n_embd, n_pos * B]
    ggml_tensor* build_attn(ggml_tensor* wo, ggml_tensor* wo_b, ggml_tensor* q_cur, ggml_tensor* k_cur,
                            ggml_tensor* v_cur) {
        // into the graph together, in this order
        ggml_build_forward_order(gf, q_cur);
        ggml_build_forward_order(gf, k_cur);
        ggml_build_forward_order(gf, v_cur);

        ggml_tensor* q = ggml_permute(ctx0, q_cur, 0, 2, 1, 3);
        ggml_tensor* k = ggml_permute(ctx0, k_cur, 0, 2, 1, 3);
        ggml_tensor* cur;
        if (flash) {
            // Padded heads: zeros after each head's values in q, k and v leave every score as it was and add zero
            // outputs, which are dropped again; the scale stays the true head size's.
            if (head_pad > 0) {
                q = ggml_permute(ctx0, ggml_pad(ctx0, q_cur, head_pad, 0, 0, 0), 0, 2, 1, 3);
                k = ggml_permute(ctx0, ggml_pad(ctx0, k_cur, head_pad, 0, 0, 0), 0, 2, 1, 3);
                v_cur = ggml_pad(ctx0, v_cur, head_pad, 0, 0, 0);
            }
            ggml_tensor* v = ggml_permute(ctx0, v_cur, 0, 2, 1, 3);
            k = ggml_cast(ctx0, k, GGML_TYPE_F16);
            v = ggml_cast(ctx0, v, GGML_TYPE_F16);
            cur = ggml_flash_attn_ext(ctx0, q, k, v, nullptr, kq_scale, 0.0f, 0.0f);
            ggml_prec_set_acc(cur, GGML_PREC_F32);
            if (head_pad > 0)
                cur = ggml_cont(ctx0, ggml_view_4d(ctx0, cur, d_head, cur->ne[1], cur->ne[2], cur->ne[3], cur->nb[1],
                                                   cur->nb[2], cur->nb[3], 0));
            cur = ggml_reshape_2d(ctx0, cur, cur->ne[0] * cur->ne[1], cur->ne[2] * cur->ne[3]);
        } else {
            ggml_tensor* v = ggml_permute(ctx0, v_cur, 1, 2, 0, 3);
            v = ggml_cont(ctx0, v);
            ggml_tensor* kq = ggml_mul_mat(ctx0, k, q);
            kq = ggml_soft_max_ext(ctx0, kq, nullptr, kq_scale, 0.0f);
            ggml_tensor* kqv = ggml_mul_mat(ctx0, v, kq);
            cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
            cur = ggml_cont_2d(ctx0, cur, cur->ne[0] * cur->ne[1], cur->ne[2] * cur->ne[3]);
        }
        if (tap_attn) tap(cur, "fa0");
        if (wo) cur = ggml_mul_mat(ctx0, wo, cur);
        if (wo_b) cur = ggml_add(ctx0, cur, wo_b);
        return cur;
    }

    // models/qwen3vl.cpp
    ggml_cgraph* build_qwen3vl() {
        const int batch_size = 1;
        const int n_pos = n_patches;
        const int num_position_ids = n_pos * 4;   // 4 values a position
        int mrope_sections[4] = {d_head / 4, d_head / 4, d_head / 4, d_head / 4};

        // a still picture: the two patch kernels over the same frame, summed
        build_inp_raw();
        ggml_tensor* inp =
            ggml_add(ctx0, ggml_conv_2d(ctx0, m.patch_embd_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1),
                     ggml_conv_2d(ctx0, m.patch_embd_1, inp_raw, patch_size, patch_size, 0, 0, 1, 1));

        // the patches in 2x2 groups from here on (the merger concatenates each group's four at the end)
        {
            inp = ggml_permute(ctx0, inp, 1, 2, 0, 3);   // [w, h, c, b] -> [c, w, h, b]
            inp = ggml_cont_4d(ctx0, inp, n_embd * 2, n_patches_x / 2, n_patches_y, batch_size);
            inp = ggml_reshape_4d(ctx0, inp, n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2));
            inp = ggml_permute(ctx0, inp, 0, 2, 1, 3);
            inp = ggml_cont_3d(ctx0, inp, n_embd, n_patches_x * n_patches_y, batch_size);
        }

        inp = ggml_add(ctx0, inp, m.patch_bias);

        // the learned position table at the picture's grid, in the same grouping
        ggml_tensor* learned_pos_embd = resize_position_embeddings((uint32_t) GGML_SCALE_MODE_BILINEAR |
                                                                   (uint32_t) GGML_SCALE_FLAG_ALIGN_CORNERS);
        learned_pos_embd = ggml_cont_4d(ctx0, learned_pos_embd, n_embd * 2, n_patches_x / 2, n_patches_y, batch_size);
        learned_pos_embd =
            ggml_reshape_4d(ctx0, learned_pos_embd, n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2));
        learned_pos_embd = ggml_permute(ctx0, learned_pos_embd, 0, 2, 1, 3);
        learned_pos_embd = ggml_cont_3d(ctx0, learned_pos_embd, n_embd, n_patches_x * n_patches_y, batch_size);
        inp = ggml_add(ctx0, inp, learned_pos_embd);

        ggml_tensor* inpL = inp;
        tap(inpL, "embed");

        positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_position_ids);
        ggml_set_name(positions, "positions");
        ggml_set_input(positions);

        if (m.pre_ln_w) inpL = build_norm(inpL, m.pre_ln_w, m.pre_ln_b, false, eps);

        for (int il = 0; il < n_layer; il++) {
            const Layer& layer = m.layers[(size_t) il];
            ggml_tensor* cur = inpL;

            cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, false, eps);
            if (il == 0) tap(cur, "ln0");
            tap_attn = il == 0;

            {
                cur = ggml_mul_mat(ctx0, layer.qkv_w, cur);
                cur = ggml_add(ctx0, cur, layer.qkv_b);

                ggml_tensor* Qcur = ggml_view_3d(ctx0, cur, d_head, n_head, n_pos, ggml_row_size(cur->type, d_head),
                                                 cur->nb[1], 0);
                ggml_tensor* Kcur = ggml_view_3d(ctx0, cur, d_head, n_head, n_pos, ggml_row_size(cur->type, d_head),
                                                 cur->nb[1], ggml_row_size(cur->type, n_embd));
                ggml_tensor* Vcur = ggml_view_3d(ctx0, cur, d_head, n_head, n_pos, ggml_row_size(cur->type, d_head),
                                                 cur->nb[1], ggml_row_size(cur->type, 2 * n_embd));

                Qcur = ggml_rope_multi(ctx0, Qcur, positions, nullptr, d_head / 2, mrope_sections,
                                       GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);
                Kcur = ggml_rope_multi(ctx0, Kcur, positions, nullptr, d_head / 2, mrope_sections,
                                       GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);

                cur = build_attn(layer.o_w, layer.o_b, Qcur, Kcur, Vcur);
            }
            if (il == 0) tap(cur, "attn0");

            cur = ggml_add(ctx0, cur, inpL);
            inpL = cur;

            cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, false, eps);
            cur = build_ffn(cur, layer.ff_up_w, layer.ff_up_b, layer.ff_down_w, layer.ff_down_b);

            cur = ggml_add(ctx0, inpL, cur);
            inpL = cur;
            if (tap_layer(il)) tap(inpL, "layer" + std::to_string(il));
        }

        if (m.post_ln_w) inpL = build_norm(inpL, m.post_ln_w, m.post_ln_b, false, eps);

        // the merger: a group's four patches as one row, two linear layers
        ggml_tensor* embeddings = inpL;
        embeddings = ggml_reshape_3d(ctx0, embeddings, n_embd * 4, n_pos / 4, batch_size);
        embeddings = build_ffn(embeddings, m.mm_0_w, m.mm_0_b, m.mm_1_w, m.mm_1_b);

        ggml_build_forward_expand(gf, embeddings);
        return gf;
    }

    // clip_graph::resize_position_embeddings
    ggml_tensor* resize_position_embeddings(uint32_t mode) const {
        ggml_tensor* pos = m.position_embd;
        const int height = img_h / patch_size;
        const int width = img_w / patch_size;
        const int n_per_side = (int) std::sqrt(pos->ne[1]);
        if (height == n_per_side && width == n_per_side) return pos;

        pos = ggml_reshape_3d(ctx0, pos, n_embd, n_per_side, n_per_side);   // -> (n_embd, n_per_side, n_per_side)
        pos = ggml_permute(ctx0, pos, 2, 0, 1, 3);                         // -> (n_per_side, n_per_side, n_embd)
        pos = ggml_interpolate(ctx0, pos, width, height, n_embd, 1, mode);  // -> (width, height, n_embd)
        pos = ggml_permute(ctx0, pos, 1, 2, 0, 3);                         // -> (n_embd, width, height)
        pos = ggml_cont_2d(ctx0, pos, n_embd, width * height);             // -> (n_embd, width * height)
        return pos;
    }

    // clip_graph::build_inp
    ggml_tensor* build_inp() {
        build_inp_raw();
        ggml_tensor* inp = ggml_conv_2d(ctx0, m.patch_embd_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
        inp = ggml_reshape_3d(ctx0, inp, n_patches, n_embd, n_batch);
        inp = ggml_cont(ctx0, ggml_transpose(ctx0, inp));
        if (m.patch_bias) inp = ggml_add(ctx0, inp, m.patch_bias);
        return inp;
    }

    // clip_graph::build_rope_2d without interleaved frequencies: a head's first half turned by pos_w, its second by
    // pos_h, each half's pairs over the base rope_theta
    ggml_tensor* build_rope_2d(ggml_tensor* cur) const {
        const int n_half = (int) (cur->ne[0] / 2);
        cur = ggml_rope_ext(ctx0, cur, pos_w, nullptr, n_half, 0, 0, m.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        cur = ggml_rope_ext(ctx0, cur, pos_h, nullptr, n_half, 0, 0, m.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        return ggml_rope_set_offset(cur, n_half);
    }

    // clip_graph::build_patch_merge_permute: [n_embd, patches] in rows of the grid -> a row a group of s x s patches,
    // the group's patches side by side in their own order, the groups in the grid's (the sides are whole groups)
    ggml_tensor* build_patch_merge_permute(ggml_tensor* cur, int s) const {
        const int width = n_patches_x, height = n_patches_y;
        cur = ggml_reshape_3d(ctx0, cur, n_embd, width, height);
        cur = ggml_reshape_3d(ctx0, cur, (int64_t) n_embd * s, width / s, height);
        cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
        cur = ggml_cont_3d(ctx0, cur, (int64_t) n_embd * s * s, height / s, width / s);
        cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
        return ggml_cont_2d(ctx0, cur, cur->ne[0], cur->ne[1] * cur->ne[2]);
    }

    // clip_graph::build_vit for a tower of separate q, k and v projections, with a learned position table; Inkling
    // 2's with its rotary positions on q and k, its two scales a block and the exact GELU
    ggml_tensor* build_vit(ggml_tensor* inp, int64_t n_pos, ggml_tensor* learned_pos_embd) {
        const int64_t B = inp->ne[2];
        const bool ink = m.kind == Kind::inkling2v;

        if (learned_pos_embd) inp = ggml_add(ctx0, inp, learned_pos_embd);

        inp = ggml_reshape_2d(ctx0, inp, n_embd, n_pos * B);
        ggml_tensor* inpL = inp;
        tap(inpL, "embed");

        if (m.pre_ln_w) inpL = build_norm(inpL, m.pre_ln_w, m.pre_ln_b, false, eps);

        for (int il = 0; il < n_layer; il++) {
            const Layer& layer = m.layers[(size_t) il];
            ggml_tensor* cur = inpL;

            cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, false, eps);
            if (il == 0) tap(cur, "ln0");
            tap_attn = il == 0;

            {
                ggml_tensor* Qcur = ggml_mul_mat(ctx0, layer.q_w, cur);
                if (layer.q_b) Qcur = ggml_add(ctx0, Qcur, layer.q_b);

                ggml_tensor* Kcur = ggml_mul_mat(ctx0, layer.k_w, cur);
                if (layer.k_b) Kcur = ggml_add(ctx0, Kcur, layer.k_b);

                ggml_tensor* Vcur = ggml_mul_mat(ctx0, layer.v_w, cur);
                if (layer.v_b) Vcur = ggml_add(ctx0, Vcur, layer.v_b);

                Qcur = ggml_reshape_4d(ctx0, Qcur, d_head, n_head, n_pos, B);
                Kcur = ggml_reshape_4d(ctx0, Kcur, d_head, n_head_kv, n_pos, B);
                Vcur = ggml_reshape_4d(ctx0, Vcur, d_head, n_head_kv, n_pos, B);
                if (ink) {
                    Qcur = build_rope_2d(Qcur);
                    Kcur = build_rope_2d(Kcur);
                }

                cur = build_attn(layer.o_w, layer.o_b, Qcur, Kcur, Vcur);
            }
            if (il == 0) tap(cur, "attn0");
            if (layer.ls_1_w) cur = ggml_mul(ctx0, cur, layer.ls_1_w);

            cur = ggml_add(ctx0, cur, inpL);
            inpL = cur;

            cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, false, eps);
            cur = build_ffn(cur, layer.ff_up_w, layer.ff_up_b, layer.ff_down_w, layer.ff_down_b, ink);
            if (layer.ls_2_w) cur = ggml_mul(ctx0, cur, layer.ls_2_w);

            cur = ggml_add(ctx0, inpL, cur);
            inpL = cur;
            if (tap_layer(il)) tap(inpL, "layer" + std::to_string(il));
        }

        if (m.post_ln_w) inpL = build_norm(inpL, m.post_ln_w, m.post_ln_b, false, eps);

        inpL = ggml_reshape_3d(ctx0, inpL, n_embd, inpL->ne[1] / B, B);
        return inpL;
    }

    // llama.cpp pull request 32517's models/hy-v4-flash.cpp
    ggml_cgraph* build_hy4() {
        const int merge = m.n_merge;
        const int pw = n_patches_x;
        const int ph = n_patches_y;

        // the position table at the picture's grid: an input, resampled on the host (hy4_position_table)
        pos_embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, ph * pw);
        ggml_set_name(pos_embd, "hunyuanvl_pos_embd");
        ggml_set_input(pos_embd);

        ggml_tensor* inp = build_inp();
        ggml_tensor* cur = build_vit(inp, n_patches, pos_embd);

        cur = build_norm(cur, m.mm_pre_norm_w, nullptr, true, eps);

        // [C, W*H] -> [W, H, C] for the convolutions
        cur = ggml_reshape_3d(ctx0, cur, n_embd, pw, ph);
        cur = ggml_permute(ctx0, cur, 2, 0, 1, 3);
        cur = ggml_cont(ctx0, cur);

        // a 2x2 convolution of stride 2, GELU (erf), a 1x1 convolution
        cur = ggml_conv_2d(ctx0, m.mm_0_w, cur, merge, merge, 0, 0, 1, 1);
        cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, m.mm_0_b, 1, 1, m.mm_0_b->ne[0]));
        cur = ggml_gelu_erf(ctx0, cur);
        cur = ggml_conv_2d(ctx0, m.mm_1_w, cur, 1, 1, 0, 0, 1, 1);
        cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, m.mm_1_b, 1, 1, m.mm_1_b->ne[0]));

        const int ow = pw / merge;
        const int oh = ph / merge;
        const int idim = (int) cur->ne[2];

        // the newline row after each row of cells
        ggml_tensor* nl = ggml_reshape_4d(ctx0, m.image_newline, 1, 1, idim, 1);
        nl = ggml_repeat_4d(ctx0, nl, 1, oh, idim, 1);
        cur = ggml_concat(ctx0, cur, nl, 0);

        // [OW+1, OH, OC] -> [OC, (OW+1)*OH]
        cur = ggml_permute(ctx0, cur, 1, 2, 0, 3);
        cur = ggml_cont_2d(ctx0, cur, idim, (int64_t) (ow + 1) * oh);

        ggml_build_forward_expand(gf, cur);
        return gf;
    }

    // unsloth's llama.cpp fork's models/inkling2v.cpp: a class token before the patches, the learned position table
    // resized bilinearly, rotary positions in two dimensions; the class token dropped, the patches merged 2 x 2, two
    // matrices with the exact GELU between them
    ggml_cgraph* build_inkling2v() {
        const int n_pos = n_patches + 1;
        pos_w = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
        ggml_set_name(pos_w, "pos_w");
        ggml_set_input(pos_w);
        pos_h = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
        ggml_set_name(pos_h, "pos_h");
        ggml_set_input(pos_h);

        ggml_tensor* cur = build_inp();
        cur = ggml_add(ctx0, cur, resize_position_embeddings((uint32_t) GGML_SCALE_MODE_BILINEAR));
        ggml_tensor* cls = ggml_add(ctx0, m.class_embd, m.class_pos_embd);
        cur = ggml_concat(ctx0, cls, cur, 1);

        cur = build_vit(cur, n_pos, nullptr);

        cur = ggml_view_2d(ctx0, cur, n_embd, n_patches, cur->nb[1], cur->nb[1]);   // (without the class token)
        cur = build_patch_merge_permute(cur, m.n_merge);
        cur = build_ffn(cur, m.mm_0_w, nullptr, m.mm_1_w, nullptr, true);

        ggml_build_forward_expand(gf, cur);
        return gf;
    }

    ggml_cgraph* build() {
        return m.kind == Kind::qwen3vl ? build_qwen3vl() : m.kind == Kind::hy4 ? build_hy4() : build_inkling2v();
    }
};

// Qwen's rotary positions (mtmd's encode): a patch's row and column, the patches in the 2x2 groups' order
std::vector<int32_t> qwen_positions(int pw, int ph, int merge) {
    const int num_patches = pw * ph;
    std::vector<int32_t> positions((size_t) num_patches * 4);
    int ptr = 0;
    for (int y = 0; y < ph; y += merge) {
        for (int x = 0; x < pw; x += merge) {
            for (int dy = 0; dy < 2; dy++) {
                for (int dx = 0; dx < 2; dx++) {
                    positions[(size_t) ptr] = y + dy;
                    positions[(size_t) num_patches + ptr] = x + dx;
                    positions[(size_t) 2 * num_patches + ptr] = y + dy;
                    positions[(size_t) 3 * num_patches + ptr] = x + dx;
                    ptr++;
                }
            }
        }
    }
    return positions;
}

class GgmlEncoder final : public Encoder {
public:
    GgmlEncoder() = default;
    ~GgmlEncoder() override {
        sched_.reset();
        buf_.reset();
        if (backend_ != nullptr) ggml_backend_free(backend_);
    }
    GgmlEncoder(const GgmlEncoder&) = delete;
    GgmlEncoder& operator=(const GgmlEncoder&) = delete;

    /// throws std::runtime_error
    void open(const EncoderParams& p) {
        flash_ = p.flash_attention != FlashAttention::off;
        // without a number of threads: one a core, at most 16 (a 256-thread server does not need 128 of them)
        threads_ = p.threads > 0 ? p.threads : (int) std::max(1u, std::min(16u, std::thread::hardware_concurrency() / 2));

        backend_ = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (backend_ == nullptr) throw std::runtime_error("ggml's CPU backend did not start");
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend_);
        sched_.reset(ggml_backend_sched_new(&backend_, &buft, 1, kMaxNodes, false, true));

        proj_.open(p);
        static_cast<ProjectorParams&>(m_) = proj_;
        load();
        // ggml's fast flash attention kernel takes heads that are whole vectors: 8 floats with AVX2, which this
        // tower's 72 are, 16 with AVX-512, where 72 go to a kernel several times slower.  A build with AVX-512 pads
        // them with zeros to 80 (a 975-token picture, Q8_0 projector, 16 threads: 7.0 s; unpadded 34.7 s, 20.5 s with
        // the attention as two products; an AVX2 build 9.05 s, its rows mtmd's bytes: mtmd pads none).
        // LAMINA_VISION_HEAD_PAD=N pads them to N values (0: none), for comparisons.
        {
            const int d_head = m_.d_head();
            int to = ggml_cpu_has_avx512() != 0 ? (d_head + 15) / 16 * 16 : d_head;
            if (const char* s = std::getenv("LAMINA_VISION_HEAD_PAD")) to = std::max(d_head, std::atoi(s));
            head_pad_ = to - d_head;
        }
        compute_meta_.resize((size_t) kMaxNodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(kMaxNodes, false));
    }

    int n_embd() const override { return m_.projection_dim; }

    bool encode(const Rgb& picture, Rows& out, std::string& err) override {
        try {
            encode_or_throw(picture, out);
            return true;
        } catch (const std::exception& e) {
            err = e.what();
            return false;
        }
    }

private:
    // ---- the projector file ----------------------------------------------------------------------------------------
    void load() {
        // the weights: each tensor's description copied into a context of its own, then allocated on the backend
        const int64_t n_tensors = gguf_get_n_tensors(proj_.gguf());
        ggml_init_params ip = {/*.mem_size =*/ (size_t) (n_tensors + 1) * ggml_tensor_overhead(), /*.mem_buffer =*/ nullptr,
                               /*.no_alloc =*/ true};
        data_.reset(ggml_init(ip));
        if (!data_) throw std::runtime_error("no memory for the projector's tensors");
        std::vector<ggml_tensor*> to_load;
        auto get = [&](const std::string& name, bool required = true) -> ggml_tensor* {
            ggml_tensor* cur = ggml_get_tensor(proj_.meta(), name.c_str());
            if (cur == nullptr) {
                if (required) throw std::runtime_error("the projector has no tensor " + name);
                return nullptr;
            }
            ggml_tensor* t = ggml_dup_tensor(data_.get(), cur);
            ggml_set_name(t, cur->name);
            to_load.push_back(t);
            return t;
        };
        auto blk = [](int il, const char* what) { return "v.blk." + std::to_string(il) + "." + what; };

        m_.pre_ln_w = get("v.pre_ln.weight", false);
        m_.pre_ln_b = get("v.pre_ln.bias", false);
        m_.post_ln_w = get("v.post_ln.weight", false);
        m_.post_ln_b = get("v.post_ln.bias", false);
        m_.patch_bias = get("v.patch_embd.bias", m_.kind == Kind::qwen3vl);
        m_.patch_embd_0 = get("v.patch_embd.weight");
        m_.patch_embd_1 = get("v.patch_embd.weight.1", m_.kind == Kind::qwen3vl);
        m_.position_embd = get("v.position_embd.weight");
        m_.layers.resize((size_t) m_.n_layer);
        for (int il = 0; il < m_.n_layer; ++il) {
            Layer& l = m_.layers[(size_t) il];
            if (m_.kind == Kind::qwen3vl) {
                l.qkv_w = get(blk(il, "attn_qkv.weight"));
                l.qkv_b = get(blk(il, "attn_qkv.bias"));
            } else {
                l.q_w = get(blk(il, "attn_q.weight"));
                l.q_b = get(blk(il, "attn_q.bias"), false);
                l.k_w = get(blk(il, "attn_k.weight"));
                l.k_b = get(blk(il, "attn_k.bias"), false);
                l.v_w = get(blk(il, "attn_v.weight"));
                l.v_b = get(blk(il, "attn_v.bias"), false);
            }
            l.o_w = get(blk(il, "attn_out.weight"));
            l.o_b = get(blk(il, "attn_out.bias"), false);
            l.ln_1_w = get(blk(il, "ln1.weight"), false);
            l.ln_1_b = get(blk(il, "ln1.bias"), false);
            l.ln_2_w = get(blk(il, "ln2.weight"), false);
            l.ln_2_b = get(blk(il, "ln2.bias"), false);
            l.ff_up_w = get(blk(il, "ffn_up.weight"));
            l.ff_up_b = get(blk(il, "ffn_up.bias"), false);
            l.ff_down_w = get(blk(il, "ffn_down.weight"));
            l.ff_down_b = get(blk(il, "ffn_down.bias"), false);
            l.ls_1_w = get(blk(il, "ls1.weight"), m_.kind == Kind::inkling2v);
            l.ls_2_w = get(blk(il, "ls2.weight"), m_.kind == Kind::inkling2v);
        }
        if (m_.kind == Kind::inkling2v) {   // its projector: two matrices, no biases
            m_.mm_0_w = get("mm.1.weight");
            m_.mm_1_w = get("mm.2.weight");
            m_.class_embd = get("v.class_embd");
            m_.class_pos_embd = get("v.class_position_embd.weight");
            if (m_.pre_ln_w == nullptr || m_.pre_ln_b == nullptr) throw std::runtime_error("its tower has no pre_ln");
        } else {
            m_.mm_0_w = get("mm.0.weight");
            m_.mm_0_b = get("mm.0.bias");
            m_.mm_1_w = get("mm.2.weight");
            m_.mm_1_b = get("mm.2.bias");
        }
        if (m_.kind == Kind::hy4) {
            m_.mm_pre_norm_w = get("mm.pre_norm.weight");
            m_.image_newline = get("v.image_newline");
            if (m_.position_embd->type != GGML_TYPE_F32)   // resampled on the host
                throw std::runtime_error("its position table is not F32");
            const int n_grid = (int) std::lround(std::sqrt((double) m_.position_embd->ne[1]));
            if ((int64_t) n_grid * n_grid != m_.position_embd->ne[1] || m_.position_embd->ne[0] != m_.n_embd)
                throw std::runtime_error("its position table is not a square grid of the tower's width");
        }
        // the rows' width is the projection's
        const int64_t width = m_.kind == Kind::qwen3vl ? m_.mm_1_b->ne[0]
                              : m_.kind == Kind::inkling2v ? m_.mm_1_w->ne[1] : m_.mm_1_w->ne[3];
        if (width != m_.projection_dim) throw std::runtime_error("its adapter's output is not its projection_dim wide");

        buf_.reset(ggml_backend_alloc_ctx_tensors_from_buft(data_.get(), ggml_backend_get_default_buffer_type(backend_)));
        if (!buf_) throw std::runtime_error("the projector's weights do not fit the RAM");
        ggml_backend_buffer_set_usage(buf_.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (ggml_tensor* t : to_load) {
            const TensorInfo* info = proj_.find(t->name);
            if (info == nullptr || info->bytes != ggml_nbytes(t))
                throw std::runtime_error(std::string("the projector has no tensor ") + t->name);
            proj_.read(t->name, *info, t->data);   // the CPU's buffer: read in place
        }
        if (m_.kind == Kind::hy4) {
            pos_host_.resize((size_t) ggml_nelements(m_.position_embd));
            ggml_backend_tensor_get(m_.position_embd, pos_host_.data(), 0, ggml_nbytes(m_.position_embd));
        }
    }

    // ---- a picture -------------------------------------------------------------------------------------------------
    ggml_cgraph* build(ggml_context_ptr& ctx0, std::unique_ptr<Graph>& g, int w, int h, bool flash) {
        ggml_init_params ip = {/*.mem_size =*/ compute_meta_.size(), /*.mem_buffer =*/ compute_meta_.data(),
                               /*.no_alloc =*/ true};
        ctx0.reset(ggml_init(ip));
        g = std::make_unique<Graph>(m_, ctx0.get(), w, h, flash, head_pad_);
        g->dump = std::getenv("LAMINA_VISION_DUMP") != nullptr;
        return g->build();
    }

    /// the buffers for a w x h picture's graph reserved
    void reserve(int w, int h) {
        ggml_context_ptr ctx0;
        std::unique_ptr<Graph> g;
        ggml_cgraph* gf = build(ctx0, g, w, h, flash_);
        if (!ggml_backend_sched_reserve(sched_.get(), gf))
            throw std::runtime_error("the encoder's buffers do not fit the RAM");
    }

    void encode_or_throw(const Rgb& picture, Rows& out) {
        if (picture.w <= 0 || picture.h <= 0 || picture.px.size() != (size_t) picture.w * (size_t) picture.h * 3)
            throw std::runtime_error("an empty picture");
        int tw = 0, th = 0;
        const std::vector<float> pixels = projector_planes(picture, m_, tw, th);

        // the first picture reserves the buffers, as mtmd's warm-up does
        if (!allocated_) {
            reserve(tw, th);
            allocated_ = true;
        }

        ggml_backend_sched_reset(sched_.get());
        ggml_context_ptr ctx0;
        std::unique_ptr<Graph> g;
        ggml_cgraph* gf = build(ctx0, g, tw, th, flash_);
        if (!ggml_backend_sched_alloc_graph(sched_.get(), gf))
            throw std::runtime_error("the picture's buffers do not fit the RAM");

        ggml_backend_tensor_set(g->inp_raw, pixels.data(), 0, ggml_nbytes(g->inp_raw));
        const int pw = tw / m_.patch_size, ph = th / m_.patch_size;
        if (m_.kind == Kind::qwen3vl) {
            const std::vector<int32_t> positions = qwen_positions(pw, ph, m_.n_merge);
            ggml_backend_tensor_set(g->positions, positions.data(), 0, ggml_nbytes(g->positions));
        } else if (m_.kind == Kind::inkling2v) {
            std::vector<int32_t> px{0}, py{0};   // the class token at 0, the patches from 1
            for (int y = 0; y < ph; ++y)
                for (int x = 0; x < pw; ++x) {
                    px.push_back(x + 1);
                    py.push_back(y + 1);
                }
            ggml_backend_tensor_set(g->pos_w, px.data(), 0, ggml_nbytes(g->pos_w));
            ggml_backend_tensor_set(g->pos_h, py.data(), 0, ggml_nbytes(g->pos_h));
        } else {
            const int n_grid = (int) std::lround(std::sqrt((double) m_.position_embd->ne[1]));
            const std::vector<float> table = hy4_position_table(pos_host_, m_.n_embd, n_grid, pw, ph);
            ggml_backend_tensor_set(g->pos_embd, table.data(), 0, ggml_nbytes(g->pos_embd));
        }

        ggml_backend_cpu_set_n_threads(backend_, threads_);
        const ggml_status status = ggml_backend_sched_graph_compute(sched_.get(), gf);
        if (status != GGML_STATUS_SUCCESS)
            throw std::runtime_error("the encoder's graph failed (ggml status " + std::to_string((int) status) + ")");

        for (const auto& [name, t] : g->taps) {
            std::vector<float> v((size_t) ggml_nelements(t));
            ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
            const std::string file = std::string(std::getenv("LAMINA_VISION_DUMP")) + "/" + name + ".f32";
            if (std::FILE* f = ggml_fopen(file.c_str(), "wb")) {
                std::fwrite(v.data(), 4, v.size(), f);
                std::fclose(f);
            }
        }
        ggml_tensor* rows = ggml_graph_node(gf, -1);
        const int n = (int) rows->ne[1];
        const int merged_x = pw / m_.n_merge, merged_y = ph / m_.n_merge;
        const int expected = m_.kind == Kind::hy4 ? (merged_x + 1) * merged_y : merged_x * merged_y;
        if (rows->ne[0] != m_.projection_dim || n != expected || ggml_nelements(rows) != (int64_t) n * m_.projection_dim)
            throw std::runtime_error("the encoder gave " + std::to_string(n) + " rows of " +
                                     std::to_string(rows->ne[0]) + " where " + std::to_string(expected) + " of " +
                                     std::to_string(m_.projection_dim) + " were due");
        out.n = n;
        out.n_embd = m_.projection_dim;
        if (m_.kind == Kind::qwen3vl) {
            out.nx = merged_x;
            out.ny = merged_y;
        } else {   // read at text positions: one row of n
            out.nx = n;
            out.ny = 1;
        }
        out.data.resize((size_t) n * (size_t) m_.projection_dim);
        ggml_backend_tensor_get(rows, out.data.data(), 0, ggml_nbytes(rows));
    }

    Projector proj_;
    Model m_;
    ggml_context_ptr data_;
    ggml_backend_t backend_ = nullptr;   // ggml's CPU backend
    ggml_backend_buffer_ptr buf_;
    ggml_backend_sched_ptr sched_;
    std::vector<uint8_t> compute_meta_;
    std::vector<float> pos_host_;   // Hy4's position table
    bool flash_ = true;             // flash attention (K and V in FP16); false: two products in FP32
    int threads_ = 4;
    int head_pad_ = 0;
    bool allocated_ = false;
};

}  // namespace

std::unique_ptr<Encoder> open_ggml_encoder(const EncoderParams& params, std::string& err) {
    backends_init();
    auto enc = std::make_unique<GgmlEncoder>();
    try {
        enc->open(params);
    } catch (const std::exception& e) {
        err = e.what();
        return nullptr;
    }
    return enc;
}

}  // namespace lamina::vision
