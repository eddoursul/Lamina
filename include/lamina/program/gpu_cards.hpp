// include/lamina/program/gpu_cards.hpp - --gpu's cards (a GPU and the settings it gives itself) and the units of the
// sizes and times the engine's options take.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace lamina::program {

/// A size with its unit, binary: 8G, 8GiB and 8GB are 8 GiB, as are 8g and 8gb; K, M, G and T, a fraction too (1.5G).
/// 0 needs none.  False (and `err` says why) for a number without a unit, a negative one or anything else.
bool parse_size(std::string_view text, uint64_t& bytes, std::string& err);
/// A time with its unit: 55us, 1.5ms, 1s; 0 needs none.  False (and `err` says why) otherwise.
bool parse_seconds(std::string_view text, double& seconds, std::string& err);
/// `bytes` in the largest unit that holds it whole (8G, 700M, 4096B), as parse_size reads it back.
std::string size_text(uint64_t bytes);
/// `seconds` as parse_seconds reads it back (55us, 1.5ms, 1s).
std::string seconds_text(double seconds);

/// One --gpu: a GPU the engine uses and the settings it gives itself; a setting it leaves out comes from the older
/// flags (--expert-cache, --second-gpu-gib, ...), else from the default.  The first --gpu runs the model, the others
/// are expert tiers, in their order.
struct GpuCard {
    std::string name;                      ///< as --main-gpu takes it, or "auto": every other GPU, as tiers
    bool cache_auto = false;               ///< cache=auto: all its free VRAM but the reserve
    std::optional<int64_t> cache_slots;    ///< cache=N: expert slots (the main GPU's, --expert-cache N)
    std::optional<uint64_t> cache_bytes;   ///< cache=SIZE
    std::optional<uint64_t> reserve;       ///< reserve=SIZE: the VRAM it leaves free
    bool vram_only_auto = false;           ///< vram-only=auto: what keeps the model's experts in the RAM it has
    std::optional<uint64_t> vram_only;     ///< vram-only=SIZE: the profile's experts it holds there only, not in RAM
    std::optional<uint64_t> prefetch;      ///< prefetch=SIZE (a tier): a layer's likeliest experts copied ahead
    std::optional<std::string> clocks;     ///< clocks=MIN-MAX/MEM in MHz (a tier): locked while generating; 0: none
    std::optional<std::pair<int, int>> layers;   ///< layers=A-B (a later card of Inkling 2's path): the layers it runs
    std::optional<double> sleep;           ///< sleep=TIME (a tier after the first): the CPU's time a layer below which
                                           ///< it sleeps
    bool has_cache() const { return cache_auto || cache_slots.has_value() || cache_bytes.has_value(); }
    bool has_vram_only() const { return vram_only_auto || vram_only.has_value(); }
};

/// "NAME[,key=value...]" as --gpu takes it (spaces around the items are dropped).  False (and `err` says why) for an
/// empty name, an unknown or repeated key, or a value of the wrong form.
bool parse_gpu_card(std::string_view text, GpuCard& card, std::string& err);
/// The card as --gpu takes it again: its name, then the settings it gives, in gpu_card_keys()'s order.
std::string gpu_card_text(const GpuCard& card);
/// The keys parse_gpu_card knows, in their order.
const std::vector<std::string_view>& gpu_card_keys();

}  // namespace lamina::program
