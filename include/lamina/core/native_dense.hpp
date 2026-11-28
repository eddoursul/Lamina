#pragma once

#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace lamina {
class GgufFile;
}

namespace lamina::core {
class WeightTable;

/// The dense matrices NativeDense serves in their GGUF form: a layer's projections and its shared expert
/// (blk.N.attn_qkv / attn_gate / ssm_out / attn_q / attn_k / attn_v / attn_output / ffn_{gate,up,down}_shexp), and
/// with `include_ple_key` blk.1.ple_key.weight
bool native_dense_name(const std::string& name, bool include_ple_key);
/// The hyper-connection matrices HcQ8 keeps in 8 bits: hc_attn_down / hc_attn_up / hc_ffn_down / hc_ffn_up of each
/// layer, output_hc_down and output_hc_up
bool hc_q8_name(const std::string& name);
/// NativeDense::load's checks of the shards' directories, a shard at a time in their order: the split metadata of a
/// later shard against the first's (its architecture checked by the caller), each tensor's block geometry, extent and
/// span inside the payload, no two spans overlapping.  "" or what fails ("native dense: ...").
struct ShardDirectory {
    std::string check(const lamina::GgufFile& gguf);

private:
    std::set<uint64_t> numbers_;
    uint64_t split_count_ = 0, split_tensors_ = 0;
    bool have_architecture_ = false;
    std::string first_arch_;
};

// Experimental GDN/QSA/shared-expert projection overrides. Upload unchanged native GGUF
// blocks once, then attach them to the matching canonical WeightRef. Unsupported
// types retain their canonical paths. Owns one Q8_1 scratch vector shared by all
// these projections, so use one ordered session stream and keep this object
// alive until all graphs that reference it have been destroyed and synchronized.
class NativeDense {
public:
    NativeDense() = default;
    ~NativeDense();
    NativeDense(const NativeDense&) = delete;
    NativeDense& operator=(const NativeDense&) = delete;
    bool load(const std::vector<std::string>& shards, WeightTable& table, std::string& err,
              bool include_ple_key = false);
    /// Plan v0.3 P1: the canonical tensor names `load` would serve natively from these shards (eligible name,
    /// supported type, 2-D), read from the GGUF headers only - so the canonical arena can skip them.
    static bool served_names(const std::vector<std::string>& shards, bool include_ple_key, std::set<std::string>& out,
                             std::string& err);
    uint64_t weight_bytes() const { return bytes_; }
    size_t tensor_count() const { return weights_.size(); }

private:
    std::vector<void*> weights_;
    void* scratch_ = nullptr;
    uint64_t bytes_ = 0;
};
/// --hc-q8: the hyper-connection projections' down and up matrices (hc_attn_*, hc_ffn_*, output_hc_*) only in 8 bits:
/// int8 codes and an f16 scale a 32 values, the GGUF's own Q8_0 values where it stores them so (UD-Q4_K_XL), else its
/// values rounded as ggml's quantize_row_q8_0_ref rounds them (IQ3_S's BF16).  Attached to the canonical WeightRef
/// (hc_codes, hc_scales), whose BF16 copy the arena skips (`names`): the verify window reads them in 8 bits (fused_gr's
/// q_down / q_up), the prompt path dequantizes them to BF16 once a layer (Q8_0 ones bitwise the pack's copy).
class HcQ8 {
public:
    HcQ8() = default;
    ~HcQ8();
    HcQ8(const HcQ8&) = delete;
    HcQ8& operator=(const HcQ8&) = delete;
    bool load(const std::vector<std::string>& shards, WeightTable& table, std::string& err);
    /// The canonical names of these matrices in the shards (from the headers only), for the arena to skip.
    static bool names(const std::vector<std::string>& shards, std::set<std::string>& out, std::string& err);
    uint64_t bytes() const { return bytes_; }
    size_t matrices() const { return allocs_.size() / 2; }
    size_t stored_q8_0() const { return q8_0_; }   ///< of them taken as the GGUF stores them
private:
    std::vector<void*> allocs_;
    uint64_t bytes_ = 0;
    size_t q8_0_ = 0;
};
}   // namespace lamina::core
