// src/server/dialect.hpp - the chat forms of a template that are not Qwen's: Hy4-flash's and Inkling 2's (below).
// Hy4-flash's tags carry the
// template's suffix (HYTK, ":6124c78e"): reasoning in <think:..>; calls as <tool_calls:..><tool_call:..>NAME
// <arg_key:..>K</arg_key:..><arg_value:..>V</arg_value:..></tool_call:..></tool_calls:..>; reasoning_effort no_think /
// low / high / max.  They are translated to Qwen's forms (<think>, <tool_call>\n<function=NAME>\n<parameter=K>\nV\n
// </parameter>\n</function>\n</tool_call>), which the output parser, the template's probes and the request flow read,
// and the server's own Qwen-form text (a forced call's opening, the close of thinking) back to Hy4's for a prompt.
#pragma once

#include "json.hpp"

#include <optional>
#include <string>
#include <string_view>

namespace lamina {

struct Hy4Tags {
    std::string think_open, think_close, calls_open, calls_close, call_open, call_close, key_open, key_close, val_open,
        val_close;
    std::string image_start, image_pad;   // a picture: <｜hy_image▁start｜><｜hy_image▁pad｜><｜hy_image▁end｜>
    /// a Hy4-flash template's tags (from its HYTK), or nullopt for another template
    static std::optional<Hy4Tags> from_template(std::string_view source);
};

/// Hy4's tags -> Qwen's.  `piece`: a token's text, translated when it is a whole tag (the tags are special tokens);
/// `text`: every tag in a rendered text.  A call's name is closed by the tag after it.
class TagTranslator {
public:
    explicit TagTranslator(Hy4Tags t) : t_(std::move(t)) {}
    std::string piece(std::string_view s);
    std::string text(std::string_view s);

private:
    std::optional<std::string> tag(std::string_view s);
    Hy4Tags t_;
    bool name_ = false;   // a call's name opened and not closed yet
};

/// the server's Qwen-form text for a prompt (the close of thinking, a forced call's opening) in Hy4's tags
std::string qwen_to_hy4(std::string_view s, const Hy4Tags& t);
/// a request's template kwargs (Qwen's: enable_thinking false, reasoning_effort low / medium / xhigh) as Hy4's
/// reasoning_effort (no_think / low / high; Hy4's own values pass)
json hy4_kwargs(const json& kwargs);

/// Inkling 2's chat form.  A reply is blocks between special tokens: <|message_model|>[NAME] then
/// <|content_thinking|>, <|content_text|> or <|content_invoke_tool_json|>, the block's text, <|end_message|>; a
/// call's text is the JSON {"name": NAME, "args": {...}}, its NAME in the header too.  The prompt has no generation
/// prompt: the reply opens its own first block.  The output parser reads such a reply in its block mode
/// (frontend.hpp), from the tags as control characters that no text of the model's holds (block_piece); the server's
/// own Qwen-form text is written back as tags for a prompt (qwen_to_blocks).
struct BlockTags {
    std::string model = "<|message_model|>", think = "<|content_thinking|>", text = "<|content_text|>",
                call = "<|content_invoke_tool_json|>", end = "<|end_message|>", image = "<|content_image|>";
    std::string image_pad;   ///< the marker the template writes after `image` for a picture's rows ("": none)
    /// a template of this form's tags, or nullopt for another template
    static std::optional<BlockTags> from_template(std::string_view source);
};
/// the tags as the output parser's block mode reads them
inline constexpr char kBlkModel = '\x1c', kBlkThink = '\x1d', kBlkText = '\x1e', kBlkCall = '\x1f', kBlkEnd = '\x1a';
inline bool is_block_control(char c) {
    return c == kBlkModel || c == kBlkThink || c == kBlkText || c == kBlkCall || c == kBlkEnd;
}
/// a token's text for the block mode: a whole tag (the tags are special tokens) as its control character, any other
/// text without those characters
std::string block_piece(std::string_view piece, const BlockTags& t);
/// every tag in `text` as its control character (the server's own text fed to the parser as the model's)
std::string blocks_control(std::string_view text, const BlockTags& t);
/// the server's Qwen-form text (the close of thinking, the wrap-up before it, a forced call's opening
/// "<tool_call>\n<function=NAME>\n") as this form's tags, for a prompt; `in_block`: the reply has a block open, which
/// a close of thinking or a call's opening ends first
std::string qwen_to_blocks(std::string_view s, const BlockTags& t, bool in_block);
/// a request's template kwargs (Qwen's: enable_thinking false, reasoning_effort low / medium / xhigh) as Inkling 2's
/// reasoning_effort (none / low / medium / high, the template's default)
json block_kwargs(const json& kwargs);

}   // namespace lamina
