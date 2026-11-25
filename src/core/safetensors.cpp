// src/core/safetensors.cpp - a safetensors file's header and tensors
#include "lamina/core/safetensors.hpp"

#include <nlohmann/json.hpp>

#include <cstring>
#include <exception>
#include <filesystem>

namespace lamina {

bool SafeTensors::open(const std::string& path, std::string& err) {
    entries_.clear();
    in_.close();
    in_.clear();
    in_.open(std::filesystem::path(path), std::ios::binary);
    uint64_t n = 0;
    if (!in_ || !in_.read(reinterpret_cast<char*>(&n), 8) || n < 2 || n > (uint64_t{64} << 20)) {
        err = path + ": not a safetensors file";
        return false;
    }
    std::string text((size_t) n, '\0');
    if (!in_.read(text.data(), (std::streamsize) n)) {
        err = path + ": its header is cut short";
        return false;
    }
    const nlohmann::json head = nlohmann::json::parse(text, nullptr, false);
    if (head.is_discarded() || !head.is_object()) {
        err = path + ": its header is not a JSON object";
        return false;
    }
    try {
        for (const auto& [name, t] : head.items()) {
            if (name == "__metadata__") continue;
            Tensor e;
            e.dtype = t.at("dtype").get<std::string>();
            e.shape = t.at("shape").get<std::vector<int64_t>>();
            e.begin = t.at("data_offsets").at(0).get<uint64_t>();
            e.end = t.at("data_offsets").at(1).get<uint64_t>();
            if (e.end < e.begin) {
                err = path + ": " + name + "'s data ends before it begins";
                return false;
            }
            entries_.emplace(name, std::move(e));
        }
    } catch (const std::exception& e) {
        err = path + ": " + e.what();
        return false;
    }
    data_ = 8 + n;
    return true;
}

const SafeTensors::Tensor* SafeTensors::find(const std::string& name) const {
    const auto it = entries_.find(name);
    return it == entries_.end() ? nullptr : &it->second;
}

bool SafeTensors::bytes(const std::string& name, std::vector<uint8_t>& out) {
    const Tensor* e = find(name);
    if (e == nullptr) return false;
    out.resize((size_t) (e->end - e->begin));
    in_.clear();
    return (bool) in_.seekg((std::streamoff) (data_ + e->begin)) &&
           (bool) in_.read(reinterpret_cast<char*>(out.data()), (std::streamsize) out.size());
}

bool SafeTensors::read(const std::string& name, std::vector<float>& out, std::vector<int64_t>& shape) {
    const Tensor* e = find(name);
    if (e == nullptr) return false;
    const uint64_t width = e->dtype == "BF16" ? 2 : e->dtype == "F32" ? 4 : 0;
    uint64_t count = 1;
    for (const int64_t d : e->shape) count *= (uint64_t) d;
    std::vector<uint8_t> raw;
    if (width == 0 || e->end - e->begin != count * width || !bytes(name, raw)) return false;
    out.resize((size_t) count);
    if (width == 4) {
        std::memcpy(out.data(), raw.data(), raw.size());
    } else {
        for (size_t i = 0; i < out.size(); ++i) {
            const uint32_t u = (uint32_t) (raw[2 * i] | (raw[2 * i + 1] << 8)) << 16;
            std::memcpy(&out[i], &u, 4);
        }
    }
    shape = e->shape;
    return true;
}

}  // namespace lamina
