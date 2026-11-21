// A reply in a block form (Inkling 2's: thinking, text and JSON call blocks between special tokens) through the output
// parser's block mode, a character at a time and whole: the events, a call's arguments streamed as written, a call that
// does not parse, a reply cut inside a call; and the dialect's translations (a token's piece, the server's Qwen-form
// text as tags, the template's effort). (CPU only.)
#include "dialect.hpp"
#include "frontend.hpp"

#include <cstdio>
#include <string>
#include <vector>

using lamina::Event;
using lamina::json;

static int fails = 0;
#define CHECK(c)                                                                  \
    do {                                                                          \
        if (!(c)) {                                                               \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);   \
            ++fails;                                                              \
        }                                                                         \
    } while (0)

static const json kTools = json::parse(R"([{"name": "write_file", "parameters": {"type": "object", "properties": {
    "path": {"type": "string"}, "content": {"type": "string"}}}}])");

struct Got {
    std::string reasoning, content, args, started;
    std::vector<std::pair<std::string, json>> calls;
    int arg_pieces = 0;
};

static void take(Got& g, const std::vector<Event>& evs) {
    for (const Event& e : evs) {
        if (e.kind == "reasoning") g.reasoning += e.text;
        else if (e.kind == "content") g.content += e.text;
        else if (e.kind == "tool_start") g.started = e.call->name;
        else if (e.kind == "tool_args") {
            g.args += e.text;
            ++g.arg_pieces;
        } else if (e.kind == "tool_call") g.calls.emplace_back(e.call->name, e.call->arguments);
    }
}

/// the reply's tokens through block_piece into a parser, `step` characters of plain text at a time
static Got run(const std::vector<std::string>& tokens, bool thinking, const json& tools, bool recover, size_t step,
               std::string* state = nullptr) {
    const lamina::BlockTags t;
    lamina::OutputParser p(thinking, tools, true, recover, true);
    Got g;
    for (const std::string& tok : tokens) {
        const std::string piece = lamina::block_piece(tok, t);
        for (size_t i = 0; i < piece.size(); i += step) take(g, p.feed(piece.substr(i, step)));
    }
    if (state) *state = p.state();
    take(g, p.finish("stop"));
    return g;
}

int main() {
    const lamina::BlockTags t;
    const std::string body = R"({"name":"write_file","args":{"content":"a \"q\" and a \\ and {braces}\n","path":"n.txt"}})";
    const std::vector<std::string> reply = {t.model, t.think, "Let me ", "write it.", t.end, t.model, t.text, "Here ",
                                            "it is.", t.end, t.model, "write_file", t.call, body, t.end};
    for (const size_t step : {(size_t) 1, (size_t) 7, (size_t) 4096}) {
        const Got g = run(reply, true, kTools, false, step);
        CHECK(g.reasoning == "Let me write it.");
        CHECK(g.content == "Here it is.");
        CHECK(g.started == "write_file");
        CHECK(g.calls.size() == 1 && g.calls[0].first == "write_file");
        CHECK(g.calls.size() == 1 && g.calls[0].second["path"] == "n.txt" &&
              g.calls[0].second["content"] == "a \"q\" and a \\ and {braces}\n");
        // the streamed pieces are the arguments' object as the model wrote it (the raw string outside CHECK: MSVC
        // does not stringize one)
        const std::string written = R"({"content":"a \"q\" and a \\ and {braces}\n","path":"n.txt"})";
        CHECK(g.args == written);
        CHECK(step != 1 || g.arg_pieces > 10);
    }
    {   // no thinking block: the text is the answer, with or without thinking expected; two text blocks are paragraphs
        for (const bool thinking : {true, false}) {
            const Got g = run({t.model, t.text, "One.", t.end, t.model, t.text, "Two.", t.end}, thinking, nullptr, false, 3);
            CHECK(g.reasoning.empty() && g.content == "One.\n\nTwo.");
        }
    }
    {   // the states the service's budgets and forced calls read
        std::string state;
        run({t.model, t.think, "thinking"}, true, kTools, false, 4, &state);
        CHECK(state == "reasoning");
        run({t.model, t.think, "thinking", t.end}, true, kTools, false, 4, &state);
        CHECK(state == "content");
        run({t.model, "write_file", t.call, "{\"na"}, true, kTools, false, 4, &state);
        CHECK(state == "call");
    }
    {   // arguments that are no object: read at the call's end, sent whole; a name in the JSON alone
        const Got g = run({t.model, "write_file", t.call, R"({"args":"{\"path\":\"x\"}","name":"write_file"})", t.end}, false,
                          kTools, false, 2);
        const std::string whole = R"({"path": "x"})";
        CHECK(g.calls.size() == 1 && g.calls[0].second == json({{"path", "x"}}) && g.args == whole);
        const Got h = run({t.model, t.call, R"({"name":"write_file","args":{}})", t.end}, false, kTools, false, 2);
        CHECK(h.started.empty() && h.calls.size() == 1 && h.calls[0].first == "write_file");
    }
    {   // a call that does not parse: the server's error, or with recovery the text it was
        bool threw = false;
        try {
            run({t.model, "write_file", t.call, "{\"name\": oops", t.end}, false, kTools, false, 5);
        } catch (const lamina::ToolCallError&) {
            threw = true;
        }
        CHECK(threw);
        const Got g = run({t.model, "write_file", t.call, "{\"name\": oops", t.end}, false, kTools, true, 5);
        CHECK(g.calls.empty() && g.content == "{\"name\": oops");
    }
    {   // a request without tools gets a call block's text; the tags' characters in the model's own text are dropped
        const Got g = run({t.model, "x", t.call, "{}", t.end, t.model, t.text, std::string("a") + lamina::kBlkCall + "b", t.end},
                          false, nullptr, false, 1);
        CHECK(g.calls.empty() && g.content == "{}ab");
    }
    // a tag the model spells in plain tokens is text: only a whole piece is a tag
    CHECK(lamina::block_piece("<|end_", t) == "<|end_" && lamina::block_piece(t.end, t) == std::string(1, lamina::kBlkEnd));
    CHECK(lamina::blocks_control(t.model + "f" + t.call, t) == std::string(1, lamina::kBlkModel) + "f" + lamina::kBlkCall);
    // the server's Qwen-form text as tags
    CHECK(lamina::qwen_to_blocks("<tool_call>\n<function=f>\n", t, false) == t.model + "f" + t.call);
    CHECK(lamina::qwen_to_blocks("\n\n<tool_call>\n<function=f>\n", t, true) == t.end + t.model + "f" + t.call);
    CHECK(lamina::qwen_to_blocks("\n</think>\n\n", t, true) == t.end);
    CHECK(lamina::qwen_to_blocks("\n</think>\n\n", t, false).empty());
    CHECK(lamina::qwen_to_blocks("\n\nEnough.\n</think>\n\n<tool_call>\n<function=f>\n", t, true) ==
          "\n\nEnough." + t.end + t.model + "f" + t.call);
    // the template's effort from Qwen's kwargs
    CHECK(lamina::block_kwargs({{"enable_thinking", false}}) == json({{"reasoning_effort", "none"}}));
    CHECK(lamina::block_kwargs({{"reasoning_effort", "xhigh"}}) == json({{"reasoning_effort", "high"}}));
    CHECK(lamina::block_kwargs({{"reasoning_effort", "low"}}) == json({{"reasoning_effort", "low"}}));
    CHECK(lamina::block_kwargs(json::object()) == json::object());
    // a template of the form, its picture marker; another template is none
    const auto tags = lamina::BlockTags::from_template(
        "<|message_model|><|content_thinking|><|content_text|><|end_message|>{{- rtok ~ \"<|content_image|><|unused_200054|>"
        "<|end_message|>\" -}}");
    CHECK(tags && tags->image_pad == "<|unused_200054|>");
    CHECK(!lamina::BlockTags::from_template("<|im_start|>assistant\n<think>"));
    if (fails) return 1;
    std::printf("block_form_test: ok\n");
    return 0;
}
