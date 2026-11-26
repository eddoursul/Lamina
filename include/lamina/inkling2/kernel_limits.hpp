// include/lamina/inkling2/kernel_limits.hpp - what Inkling 2's kernels take of a model's geometry (the sizes they are
// built for: kernel_sizes.hpp): the engine's load and lamina-server --check refuse a model past them alike.
#pragma once

#include "lamina/inkling2/geometry.hpp"
#include "lamina/inkling2/kernel_sizes.hpp"

#include <string>

namespace lamina::inkling2 {

/// "" or the first limit of the kernels that `g` passes
inline std::string kernel_limits(const Geometry& g) {
    namespace k = sizes;
    auto no = [](const std::string& what) { return "Inkling 2's path does not run " + what; };
    if (g.head_dim != 128) return no("a head of " + std::to_string(g.head_dim) + " values (its kernels take 128)");
    if (g.n_head / g.n_head_kv > k::kQH) return no("more than 4 query heads a K/V head");
    if (g.n_rel > k::kRelMax) return no("a distance bias of more than 32 values a head");
    if (g.n_head > k::kRingHeads || g.n_head * g.n_rel > k::kRingRs) return no("that many heads");
    if (g.n_expert > k::kExpertMax || g.top_k * k::kMaxT > k::kTB) return no("that many experts a layer or a token");
    if (g.n_pass > 1 && (g.gate_rank > k::kRankMax || g.gate_rank % 16 != 0)) return no("a loop gate of that rank");
    if ((g.n_layers + g.res_block - 1) / g.res_block > k::kBlk) return no("more than 4 blocks of layers");
    if (g.n_embd % 256 != 0 || g.q_width() % 256 != 0) return no("rows that are no multiple of 256 values");
    if (g.n_embd > 4 * k::kWide) return no("rows of more than 4,096 values");
    if (g.rel_extent % k::kTcCells != 0 || g.n_swa % k::kTcCells != 0)
        return no("a window or a table of distances that is no multiple of 32 cells");
    return {};
}

}  // namespace lamina::inkling2
