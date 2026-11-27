// src/server/dialect.cpp - Hy4-flash's and Inkling 2's chat forms against Qwen's (dialect.hpp).
#include "dialect.hpp"

#include <algorithm>

namespace lamina {

std::optional<Hy4Tags> Hy4Tags::from_template(std::string_view src) {
    // {%- set HYTK = ':6124c78e' %} ... '<｜hy_start{}｜>'.format(HYTK)
    const size_t at = src.find("set HYTK = ");
    if (at == std::string_view::npos || src.find("hy_start") == std::string_view::npos) return std::nullopt;
    const size_t q = src.find_first_of("'\"", at);
    if (q == std::string_view::npos) return std::nullopt;
    const size_t e = src.find(src[q], q + 1);
    if (e == std::string_view::npos || e - q > 64) return std::nullopt;
    const std::string k(src.substr(q + 1, e - q - 1));
    Hy4Tags t;
    t.think_open = "<think" + k + ">";
    t.think_close = "</think" + k + ">";
    t.calls_open = "<tool_calls" + k + ">";
    t.calls_close = "</tool_calls" + k + ">";
    t.call_open = "<tool_call" + k + ">";
    t.call_close = "</tool_call" + k + ">";
    t.key_open = "<arg_key" + k + ">";
    t.key_close = "</arg_key" + k + ">";
    t.val_open = "<arg_value" + k + ">";
    t.val_close = "</arg_value" + k + ">";
    t.image_start = "<｜hy_image▁start" + k + "｜>";
    t.image_pad = "<｜hy_image▁pad" + k + "｜>";
    t.image_end = "<｜hy_image▁end" + k + "｜>";
    t.video_start = "<｜hy_video▁start" + k + "｜>";
    t.video_pad = "<｜hy_video▁pad" + k + "｜>";
    t.video_end = "<｜hy_video▁end" + k + "｜>";
    return t;
}

std::optional<std::string> TagTranslator::tag(std::string_view s) {
    const auto close_name = [&] {
        const bool was = name_;
        name_ = false;
        return std::string(was ? ">\n" : "");
    };
    if (s == t_.think_open) return std::string("<think>\n");
    if (s == t_.think_close) return std::string("</think>\n\n");
    if (s == t_.calls_open || s == t_.calls_close) return std::string();
    if (s == t_.call_open) {
        name_ = true;
        return std::string("<tool_call>\n<function=");
    }
    if (s == t_.key_open) return close_name() + "<parameter=";
    if (s == t_.key_close) return std::string(">\n");
    if (s == t_.val_open) return std::string();
    if (s == t_.val_close) return std::string("\n</parameter>\n");
    if (s == t_.call_close) return close_name() + "</function>\n</tool_call>";
    return std::nullopt;
}

std::string TagTranslator::piece(std::string_view s) {
    if (s.size() > 4 && s.front() == '<' && s.back() == '>')
        if (std::optional<std::string> r = tag(s)) return *r;
    return std::string(s);
}

std::string TagTranslator::text(std::string_view s) {
    std::string out;
    size_t i = 0;
    while (i < s.size()) {
        const size_t lt = s.find('<', i);
        if (lt == std::string_view::npos) break;
        const size_t gt = s.find('>', lt);
        if (gt == std::string_view::npos) break;
        if (std::optional<std::string> r = tag(s.substr(lt, gt + 1 - lt))) {
            out.append(s.substr(i, lt - i));
            out += *r;
            i = gt + 1;
        } else {
            out.append(s.substr(i, lt + 1 - i));
            i = lt + 1;
        }
    }
    out.append(s.substr(i));
    return out;
}

std::string qwen_to_hy4(std::string_view s, const Hy4Tags& t) {
    std::string out(s);
    auto replace = [&](std::string_view from, const std::string& to) {
        for (size_t at = 0; (at = out.find(from, at)) != std::string::npos; at += to.size())
            out.replace(at, from.size(), to);
    };
    // a named call's opening "<tool_call>\n<function=NAME>\n": Hy4's call opening and the name
    const std::string_view fs = "<tool_call>\n<function=";
    for (size_t at = 0; (at = out.find(fs, at)) != std::string::npos;) {
        size_t end = at + fs.size();
        const size_t gt = out.find(">\n", end);
        std::string name;
        if (gt != std::string::npos && out.find('<', end) > gt) {
            name = out.substr(end, gt - end);
            end = gt + 2;
        }
        const std::string to = t.calls_open + t.call_open + name;
        out.replace(at, end - at, to);
        at += to.size();
    }
    replace("</think>\n\n", t.think_close);
    replace("</think>", t.think_close);
    return out;
}

json hy4_kwargs(const json& kwargs) {
    json out = kwargs.is_object() ? kwargs : json::object();
    const json& et = jget(out, "enable_thinking");
    const bool off = et.is_boolean() && !et.get<bool>();
    if (out.is_object() && out.contains("enable_thinking")) out.erase("enable_thinking");
    if (off) {
        out["reasoning_effort"] = "no_think";
        return out;
    }
    const json& e = jget(out, "reasoning_effort");
    if (e.is_string()) {
        const std::string v = e.get<std::string>();
        if (v == "medium" || v == "xhigh") out["reasoning_effort"] = "high";   // Qwen's levels: Hy4 has low, high, max
    }
    return out;
}

std::optional<BlockTags> BlockTags::from_template(std::string_view src) {
    BlockTags t;
    for (const std::string* tag : {&t.model, &t.think, &t.text, &t.end})
        if (src.find(*tag) == std::string_view::npos) return std::nullopt;
    // a picture's marker: the tag the template writes right after <|content_image|>
    if (const size_t at = src.find(t.image + "<|"); at != std::string_view::npos) {
        const size_t from = at + t.image.size(), to = src.find("|>", from);
        if (to != std::string_view::npos && to - from < 64) t.image_pad = std::string(src.substr(from, to + 2 - from));
    }
    return t;
}

std::string block_piece(std::string_view s, const BlockTags& t) {
    if (s == t.model) return std::string(1, kBlkModel);
    if (s == t.think) return std::string(1, kBlkThink);
    if (s == t.text) return std::string(1, kBlkText);
    if (s == t.call) return std::string(1, kBlkCall);
    if (s == t.end) return std::string(1, kBlkEnd);
    // a block of another kind (a tag the template writes for no reply, <|content_invoke_tool_text|>): its text is
    // the reply's
    if (s.size() > 12 && s.compare(0, 10, "<|content_") == 0 && s.compare(s.size() - 2, 2, "|>") == 0)
        return std::string(1, kBlkText);
    std::string out(s);
    out.erase(std::remove_if(out.begin(), out.end(), is_block_control), out.end());
    return out;
}

std::string blocks_control(std::string_view s, const BlockTags& t) {
    std::string out;
    const std::pair<const std::string*, char> tags[] = {
        {&t.model, kBlkModel}, {&t.think, kBlkThink}, {&t.text, kBlkText}, {&t.call, kBlkCall}, {&t.end, kBlkEnd}};
    for (size_t i = 0; i < s.size();) {
        bool hit = false;
        for (const auto& [tag, c] : tags)
            if (s.compare(i, tag->size(), *tag) == 0) {
                out.push_back(c);
                i += tag->size();
                hit = true;
                break;
            }
        if (hit) continue;
        if (!is_block_control(s[i])) out.push_back(s[i]);
        ++i;
    }
    return out;
}

std::string qwen_to_blocks(std::string_view s, const BlockTags& t, bool in_block) {
    std::string out;
    const std::string_view think = "</think>", call = "<tool_call>\n<function=";
    auto drop_newlines = [&] {   // the newlines Qwen's form has around its tags are not this form's
        while (!out.empty() && out.back() == '\n') out.pop_back();
    };
    for (size_t i = 0; i < s.size();) {
        if (s.compare(i, think.size(), think) == 0) {
            drop_newlines();
            if (in_block) out += t.end;
            in_block = false;
            i += think.size();
            while (i < s.size() && s[i] == '\n') ++i;
            continue;
        }
        if (s.compare(i, call.size(), call) == 0) {
            const size_t name = i + call.size(), gt = s.find(">\n", name);
            if (gt != std::string_view::npos && s.find('<', name) > gt) {
                drop_newlines();
                if (in_block) out += t.end;
                out += t.model + std::string(s.substr(name, gt - name)) + t.call;
                in_block = true;
                i = gt + 2;
                continue;
            }
        }
        out.push_back(s[i++]);
    }
    if (!in_block) drop_newlines();   // text outside a block is no part of the form
    return out;
}

json block_kwargs(const json& kwargs) {
    json out = kwargs.is_object() ? kwargs : json::object();
    const json& et = jget(out, "enable_thinking");
    const bool off = et.is_boolean() && !et.get<bool>();
    if (out.contains("enable_thinking")) out.erase("enable_thinking");
    if (off) {
        out["reasoning_effort"] = "none";
        return out;
    }
    const json& e = jget(out, "reasoning_effort");
    if (e.is_string() && e.get_ref<const std::string&>() == "xhigh")
        out["reasoning_effort"] = "high";   // Qwen's top level
    return out;
}

}   // namespace lamina
