// src/hy4/hy4_dfly_file.cpp - DFly2's settings: the config.json beside its safetensors shard
#include "hy4_dfly_file.hpp"

#include <nlohmann/json.hpp>

#include <cstring>
#include <exception>
#include <filesystem>

namespace lamina::hy4 {

bool read_dfly_config(const std::string& dir, DFlyConfig& c, std::string& err) {
    const std::filesystem::path p = std::filesystem::path(dir) / "config.json";
    std::ifstream in(p);
    if (!in) {
        err = p.string() + " is missing: the checkpoint's config.json goes beside its DFly shard";
        return false;
    }
    const nlohmann::json j = nlohmann::json::parse(in, nullptr, false);
    if (j.is_discarded() || !j.is_object() || !j.contains("dfly_target_kv_layers")) {
        err = p.string() + ": no dfly_target_kv_layers (not Hy4-flash's config with DFly)";
        return false;
    }
    try {
        c.kv_layers = j.at("dfly_target_kv_layers").get<std::vector<int>>();
        std::vector<int> index = c.kv_layers;
        if (const auto& v = j.value("dfly_target_index_layer", nlohmann::json()); v.is_number())
            index.assign(c.kv_layers.size(), v.get<int>());
        else if (v.is_array()) index = v.get<std::vector<int>>();
        c.mask = j.at("dfly_mask_token_id").get<int32_t>();
        c.width = j.at("speculative_config").at("num_speculative_tokens").get<int>() + 1;
        c.top_k = j.at("dfly_target_index_topk").get<int>();
        const nlohmann::json window = j.value("dfly_context_local_window", nlohmann::json());
        if (index != c.kv_layers) {
            err = p.string() + ": DFly's index layers differ from its K/V layers, which this path does not run";
            return false;
        }
        if (!window.is_null() && window.get<int>() != 0) {
            err = p.string() + ": DFly's local window (dfly_context_local_window) is not run on this path";
            return false;
        }
        if (j.value("dfly_num_hidden_layers", 0) != (int) c.kv_layers.size()) {
            err = p.string() + ": dfly_num_hidden_layers is not one a K/V layer";
            return false;
        }
    } catch (const std::exception& e) {
        err = p.string() + ": " + e.what();
        return false;
    }
    return true;
}

}   // namespace lamina::hy4
