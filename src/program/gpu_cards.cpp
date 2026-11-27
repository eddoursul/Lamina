// src/program/gpu_cards.cpp - --gpu's cards and the units of sizes and times (gpu_cards.hpp).
#include "lamina/program/gpu_cards.hpp"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace lamina::program {
namespace {

std::string_view trim(std::string_view s) {
    while (!s.empty() && std::isspace((unsigned char) s.front())) s.remove_prefix(1);
    while (!s.empty() && std::isspace((unsigned char) s.back())) s.remove_suffix(1);
    return s;
}

std::string lower(std::string_view s) {
    std::string out(s);
    for (char& c : out) c = (char) std::tolower((unsigned char) c);
    return out;
}

/// The number at the start of `s` and the rest; false when it does not start with one (or is not finite).
bool number_then(std::string_view s, double& v, std::string_view& rest) {
    const std::string text(s);
    char* end = nullptr;
    v = std::strtod(text.c_str(), &end);
    if (end == text.c_str() || !std::isfinite(v)) return false;
    rest = trim(s.substr((size_t) (end - text.c_str())));
    return true;
}

bool all_digits(std::string_view s) {
    if (s.empty()) return false;
    for (const char c : s)
        if (!std::isdigit((unsigned char) c)) return false;
    return true;
}

}   // namespace

bool parse_size(std::string_view text, uint64_t& bytes, std::string& err) {
    const std::string_view s = trim(text);
    double v = 0.0;
    std::string_view unit;
    const std::string what = "'" + std::string(s) + "' is not a size: a number and its unit, as 8G or 700M";
    if (!number_then(s, v, unit) || v < 0.0) {
        err = what;
        return false;
    }
    if (unit.empty()) {
        if (v == 0.0) {
            bytes = 0;
            return true;
        }
        err = "'" + std::string(s) + "' needs its unit: 8G, 700M (binary: G is 1024 M)";
        return false;
    }
    const std::string u = lower(unit);
    struct Unit {
        const char* name;
        double mult;
    };
    static const Unit kUnits[] = {{"b", 1.0},
                                  {"k", 1024.0},
                                  {"kb", 1024.0},
                                  {"kib", 1024.0},
                                  {"m", 1048576.0},
                                  {"mb", 1048576.0},
                                  {"mib", 1048576.0},
                                  {"g", 1073741824.0},
                                  {"gb", 1073741824.0},
                                  {"gib", 1073741824.0},
                                  {"t", 1099511627776.0},
                                  {"tb", 1099511627776.0},
                                  {"tib", 1099511627776.0}};
    double mult = 0.0;
    for (const Unit& k : kUnits)
        if (u == k.name) mult = k.mult;
    if (mult == 0.0) {
        err = what;
        return false;
    }
    bytes = (uint64_t) std::llround(v * mult);
    return true;
}

bool parse_seconds(std::string_view text, double& seconds, std::string& err) {
    const std::string_view s = trim(text);
    double v = 0.0;
    std::string_view unit;
    if (!number_then(s, v, unit) || v < 0.0) {
        err = "'" + std::string(s) + "' is not a time: a number and its unit, as 55us, 1.5ms or 1s";
        return false;
    }
    const std::string u = lower(unit);
    if (u.empty() && v == 0.0) {
        seconds = 0.0;
        return true;
    }
    // divided, not multiplied by 1e-6: 55us is then exactly the double 55e-6
    const double per = u == "s" ? 1.0 : u == "ms" ? 1e3 : (u == "us" || u == "\xc2\xb5s") ? 1e6 : 0.0;
    if (per == 0.0) {
        err = u.empty() ? "'" + std::string(s) + "' needs its unit: 55us, 1.5ms, 1s"
                        : "'" + std::string(s) + "' is not a time: a number and its unit, as 55us, 1.5ms or 1s";
        return false;
    }
    seconds = v / per;
    return true;
}

std::string size_text(uint64_t bytes) {
    if (bytes == 0) return "0";
    static const std::pair<char, uint64_t> kUnits[] = {
        {'T', 1ull << 40}, {'G', 1ull << 30}, {'M', 1ull << 20}, {'K', 1ull << 10}};
    for (const auto& [letter, unit] : kUnits)
        if (bytes % unit == 0) return std::to_string(bytes / unit) + letter;
    return std::to_string(bytes) + "B";
}

std::string seconds_text(double seconds) {
    if (seconds == 0.0) return "0";
    char buf[48];
    if (seconds >= 1.0) std::snprintf(buf, sizeof buf, "%gs", seconds);
    else if (seconds >= 1e-3) std::snprintf(buf, sizeof buf, "%gms", seconds * 1e3);
    else std::snprintf(buf, sizeof buf, "%gus", seconds * 1e6);
    return buf;
}

const std::vector<std::string_view>& gpu_card_keys() {
    static const std::vector<std::string_view> keys = {"cache",  "reserve", "vram-only", "prefetch",
                                                       "clocks", "layers",  "sleep"};
    return keys;
}

bool parse_gpu_card(std::string_view text, GpuCard& card, std::string& err) {
    card = GpuCard{};
    std::vector<std::string_view> items;
    for (size_t at = 0;;) {
        const size_t comma = text.find(',', at);
        items.push_back(trim(text.substr(at, comma == std::string_view::npos ? std::string_view::npos : comma - at)));
        if (comma == std::string_view::npos) break;
        at = comma + 1;
    }
    card.name = std::string(items[0]);
    if (card.name.empty() || card.name.find('=') != std::string::npos) {
        err = "'" + std::string(text) +
              "': a card's name comes first (a number, a UUID's start, a PCI bus id or a "
              "name, or auto), then its settings as key=value";
        return false;
    }
    std::vector<std::string> seen;
    for (size_t i = 1; i < items.size(); ++i) {
        const std::string_view item = items[i];
        if (item.empty()) continue;
        const size_t eq = item.find('=');
        const std::string key = eq == std::string_view::npos ? lower(item) : lower(trim(item.substr(0, eq)));
        const std::string_view value = eq == std::string_view::npos ? std::string_view() : trim(item.substr(eq + 1));
        std::string known;
        for (const std::string_view k : gpu_card_keys()) known += (known.empty() ? "" : ", ") + std::string(k);
        bool is_key = false;
        for (const std::string_view k : gpu_card_keys()) is_key = is_key || key == k;
        if (!is_key) {
            err = "'" + std::string(item) + "': not a card's setting (" + known + ")";
            return false;
        }
        if (eq == std::string_view::npos || value.empty()) {
            err = "'" + std::string(item) + "': " + key + "=<value>";
            return false;
        }
        for (const std::string& s : seen)
            if (s == key) {
                err = "'" + std::string(text) + "': " + key + " twice";
                return false;
            }
        seen.push_back(key);
        std::string e;
        uint64_t bytes = 0;
        if (key == "cache") {
            if (lower(value) == "auto") {
                card.cache_auto = true;
            } else if (all_digits(value)) {
                card.cache_slots = std::atoll(std::string(value).c_str());
            } else if (parse_size(value, bytes, e)) {
                card.cache_bytes = bytes;
            } else {
                err = "cache=" + std::string(value) + ": auto, a size (16G) or a number of expert slots";
                return false;
            }
        } else if (key == "vram-only") {
            if (lower(value) == "auto") {
                card.vram_only_auto = true;
            } else if (parse_size(value, bytes, e)) {
                card.vram_only = bytes;
            } else {
                err = "vram-only=" + std::string(value) + ": auto or " + e;
                return false;
            }
        } else if (key == "reserve" || key == "prefetch") {
            if (!parse_size(value, bytes, e)) {
                err = key + "=" + std::string(value) + ": " + e;
                return false;
            }
            (key == "reserve" ? card.reserve : card.prefetch) = bytes;
        } else if (key == "clocks") {
            for (const char c : value)
                if (!std::isdigit((unsigned char) c) && c != '-' && c != '/') {
                    err = "clocks=" + std::string(value) +
                          ": <graphics min>-<graphics max>/<memory> in MHz "
                          "(1395-2100/9751), or 0";
                    return false;
                }
            card.clocks = std::string(value);
        } else if (key == "layers") {
            int a = -1, b = -1;
            char rest = 0;
            if (std::sscanf(std::string(value).c_str(), "%d-%d%c", &a, &b, &rest) != 2 || a < 0 || b < a) {
                err = "layers=" + std::string(value) + ": the first and the last layer the card runs (6-17)";
                return false;
            }
            card.layers = std::make_pair(a, b);
        } else {   // sleep
            double s = 0.0;
            if (!parse_seconds(value, s, e)) {
                err = "sleep=" + std::string(value) + ": " + e;
                return false;
            }
            card.sleep = s;
        }
    }
    return true;
}

std::string gpu_card_text(const GpuCard& c) {
    std::string out = c.name;
    if (c.cache_auto) out += ",cache=auto";
    else if (c.cache_slots) out += ",cache=" + std::to_string(*c.cache_slots);
    else if (c.cache_bytes) out += ",cache=" + size_text(*c.cache_bytes);
    if (c.reserve) out += ",reserve=" + size_text(*c.reserve);
    if (c.vram_only_auto) out += ",vram-only=auto";
    else if (c.vram_only) out += ",vram-only=" + size_text(*c.vram_only);
    if (c.prefetch) out += ",prefetch=" + size_text(*c.prefetch);
    if (c.clocks) out += ",clocks=" + *c.clocks;
    if (c.layers) out += ",layers=" + std::to_string(c.layers->first) + "-" + std::to_string(c.layers->second);
    if (c.sleep) out += ",sleep=" + seconds_text(*c.sleep);
    return out;
}

}   // namespace lamina::program
