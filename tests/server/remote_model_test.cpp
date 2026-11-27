// lamina --check -m URL's pieces (src/server/remote_model.hpp), no model and no GPU: Hugging Face URLs as the
// server reads them, a GGUF header's length from its first bytes, a listing's models and the files read beside a model,
// the GGUF files that are no model by their headers (drafters, LoRA adapters, control vectors, Engram tables ...) and
// the checks a load makes of them, and against a Hub played here (cpp-httplib: its API's revision and paged tree,
// resolve's redirect to a download host that serves ranges, a gated repository, a server without ranges, a direct
// URL's missing shard) the listing and the headers' copies: sparse files of each file's size that the GGUF reader
// reads as it reads the file, the token sent to the Hub's host alone, one read for files of one id, a file's data not
// read; a split model's shard the listing holds under another name.  (CPU only.)
#include <httplib.h>

#include "json.hpp"
#include "remote_model.hpp"

#include "../core/gguf_fixture.hpp"
#include "lamina/artifact/drafter.hpp"
#include "lamina/artifact/gguf_reader.hpp"
#include "lamina/core/control_vector.hpp"
#include "lamina/core/geometry.hpp"
#include "lamina/core/model_check.hpp"
#include "model_config.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#endif

namespace fs = std::filesystem;

static int fails = 0;
#define CHECK(c)                                                                  \
    do {                                                                          \
        if (!(c)) {                                                               \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);   \
            ++fails;                                                              \
        }                                                                         \
    } while (0)

namespace {

std::string slurp(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

/// the bytes a file takes on its drive (a sparse file's: its written blocks alone)
uint64_t allocated(const fs::path& p) {
#ifdef _WIN32
    DWORD high = 0;
    const DWORD low = GetCompressedFileSizeW(p.c_str(), &high);
    return ((uint64_t) high << 32) | low;
#else
    struct stat st {};
    if (stat(p.c_str(), &st) != 0) return 0;
    return (uint64_t) st.st_blocks * 512;
#endif
}

bool parses(const std::string& url, lamina::HfTarget& t, const std::string& endpoint = "") {
    try {
        return lamina::parse_hf_url(url, t, endpoint);
    } catch (const lamina::RemoteError&) {
        return false;
    }
}
bool refused(const std::string& url) {
    lamina::HfTarget t;
    try {
        lamina::parse_hf_url(url, t);
    } catch (const lamina::RemoteError&) {
        return true;
    }
    return false;
}

void test_urls() {
    using lamina::is_url;
    CHECK(is_url("https://huggingface.co/a/b") && is_url("HTTP://x/y.gguf") && !is_url("D:/models/x.gguf") &&
          !is_url("hf.co/a/b") && !is_url("http:/x"));
    lamina::HfTarget t;
    CHECK(parses("https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF", t));
    CHECK(t.repo == "unsloth/Qwen3.8-Flash-Next-GGUF" && t.revision == "main" && t.path.empty() && !t.file && !t.dataset &&
          t.endpoint == "https://huggingface.co");
    CHECK(parses("https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/main/UD-Q4_K_XL", t));
    CHECK(t.path == "UD-Q4_K_XL" && !t.file && t.revision == "main");
    CHECK(parses("https://hf.co/o/r/blob/main/UD-Q4_K_XL/m-00001-of-00004.gguf?download=true#x", t));
    CHECK(t.repo == "o/r" && t.file && t.path == "UD-Q4_K_XL/m-00001-of-00004.gguf");
    CHECK(parses("https://huggingface.co/o/r/resolve/f5d175de/a%20b.gguf", t));
    CHECK(t.file && t.revision == "f5d175de" && t.path == "a b.gguf");
    CHECK(parses("https://www.huggingface.co/o/r/tree/refs%2Fpr%2F3/x", t));
    CHECK(t.revision == "refs/pr/3" && t.path == "x");
    CHECK(parses("https://huggingface.co/o/r/tree/refs/pr/3/x/y", t));
    CHECK(t.revision == "refs/pr/3" && t.path == "x/y");
    CHECK(parses("https://huggingface.co/datasets/o/r/tree/main/q", t));
    CHECK(t.dataset && t.repo == "o/r" && t.path == "q");
    CHECK(parses("https://huggingface.co/o/r/tree/dev", t));
    CHECK(t.revision == "dev" && t.path.empty() && !t.file);
    CHECK(parses("https://hf-mirror.com/o/r", t, "https://hf-mirror.com/"));
    CHECK(t.endpoint == "https://hf-mirror.com" && t.repo == "o/r");
    CHECK(parses("https://huggingface.co/o/r", t, "https://hf-mirror.com"));   // HF_ENDPOINT asked for any Hub URL
    CHECK(t.endpoint == "https://hf-mirror.com");
    CHECK(!parses("https://example.com/o/r/m.gguf", t));
    CHECK(!parses("https://hf-mirror.com/o/r", t));   // (no HF_ENDPOINT: another host)
    CHECK(refused("https://huggingface.co/spaces/o/r"));
    CHECK(refused("https://huggingface.co/unsloth"));
    CHECK(refused("https://huggingface.co/"));
    CHECK(refused("https://huggingface.co/o/r/discussions/1"));
    CHECK(refused("https://huggingface.co/o/r/blob/main"));
    CHECK(refused("https://huggingface.co/docs/hub"));
}

void test_header_length(const fs::path& dir) {
    using namespace fixture;
    const fs::path p = dir / "h.gguf";
    std::vector<std::string> vocab;
    for (int i = 0; i < 3000; ++i) vocab.push_back("token" + std::to_string(i));
    write(p, {str("general.architecture", "x"), strs("tokenizer.ggml.tokens", vocab), u32("x.block_count", 2)},
          {{"a.weight", {64, 4}, 8, 1}, {"b.weight", {32}, 0, 2}});
    const std::string b = slurp(p);
    lamina::GgufFile f(p.string());
    // the directory's end: the data's aligned start less the padding (the reader's own)
    uint64_t end = 0;
    for (uint64_t n = 1; n <= b.size(); ++n) {
        end = lamina::gguf_header_length((const uint8_t*) b.data(), n);
        if (end != 0) {
            CHECK(end == n);   // the first prefix that holds it is it
            break;
        }
    }
    CHECK(end > 0 && end <= f.data_start() && f.data_start() - end < 32);
    CHECK(lamina::gguf_header_length((const uint8_t*) b.data(), b.size()) == end);
    std::string bad = b;
    bad[0] = 'X';
    bool threw = false;
    try {
        lamina::gguf_header_length((const uint8_t*) bad.data(), bad.size());
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

/// gguf_drafter: a drafter told from a model by its header (or, for an architecture of no family, its name)
void test_drafters(const fs::path& dir) {
    using namespace fixture;
    auto file = [&](const std::string& name, std::vector<Kv> kv, std::vector<Tensor> ts) {
        const fs::path p = dir / name;
        write(p, kv, ts);
        return lamina::gguf_drafter(p.string());
    };
    const std::vector<Kv> q = {str("general.architecture", "qwen4exp"), u32("qwen4exp.block_count", 49),
                               u32("qwen4exp.nextn_predict_layers", 1)};
    // an MTP head (unsloth's mtp-*.gguf): layer 48 of 49 alone, with or without the embedding and the head
    lamina::GgufDrafter d = file("head.gguf", q, {{"blk.48.attn_q.weight", {32, 2}, 0, 1}, {"token_embd.weight", {32, 4}, 0, 2}});
    CHECK(d.what.find("MTP head") != std::string::npos && !d.read);
    // a model with its MTP block: its trunk too
    d = file("model.gguf", q, {{"blk.0.attn_q.weight", {32, 2}, 0, 1}, {"blk.48.attn_q.weight", {32, 2}, 0, 2}});
    CHECK(d.what.empty());
    // a split model's first shard (the metadata, no tensor) and a shard of the head's layer
    std::vector<Kv> s1 = q;
    for (const Kv& k : split_keys(0, 2, 1)) s1.push_back(k);
    CHECK(file("m-00001-of-00002.gguf", s1, {}).what.empty());
    std::vector<Kv> s2 = q;
    for (const Kv& k : split_keys(1, 2, 1)) s2.push_back(k);
    CHECK(file("m-00002-of-00002.gguf", s2, {{"blk.48.attn_q.weight", {32, 2}, 0, 1}}).what.empty());
    // another engine's drafters by their architecture
    d = file("hy4-flash_dfly_Q8_0.gguf", {str("general.architecture", "hy_v4_next_dfly")}, {{"blk.0.x.weight", {32}, 0, 1}});
    CHECK(d.what.find("DFly") != std::string::npos && !d.read);
    CHECK(file("e.gguf", {str("general.architecture", "eagle3")}, {{"x.weight", {32}, 0, 1}}).what.find("EAGLE-3") != std::string::npos);
    CHECK(file("f.gguf", {str("general.architecture", "dflash")}, {{"x.weight", {32}, 0, 1}}).what.find("DFlash") != std::string::npos);
    // an architecture of no family: by llama.cpp's side files' names alone
    CHECK(file("mtp-other.gguf", {str("general.architecture", "llama")}, {{"x.weight", {32}, 0, 1}}).what.find("named as llama.cpp names a drafter") != std::string::npos);
    CHECK(file("other.gguf", {str("general.architecture", "llama")}, {{"x.weight", {32}, 0, 1}}).what.empty());
    // a model of a family this engine runs is a model, whatever its name
    CHECK(file("mtp-model.gguf", q, {{"blk.0.attn_q.weight", {32, 2}, 0, 1}}).what.empty());
}

/// gguf_role: the GGUF files that are no model, by their headers, and whether Lamina reads them (the checks each load
/// makes: lora_file_problem, control_vector_problem, ple_table_problem)
void test_roles(const fs::path& dir) {
    using namespace fixture;
    auto role = [&](const std::string& name, std::vector<Kv> kv, std::vector<Tensor> ts) {
        const fs::path p = dir / name;
        write(p, kv, ts);
        return lamina::gguf_role(p.string());
    };
    const uint32_t F32 = 0, F16 = 1, Q4_0 = 2, IQ4_NL = 20, BF16 = 30;
    // a tiny Qwen3.8-27B (qwen35, a dense FFN), converted without its MTP block (no nextn_predict_layers)
    const std::vector<Kv> q35 = {str("general.architecture", "qwen35"), u32("qwen35.block_count", 4),
                                 u32("qwen35.embedding_length", 64), u32("qwen35.attention.head_count", 2),
                                 u32("qwen35.attention.head_count_kv", 1), u32("qwen35.attention.key_length", 16),
                                 u32("qwen35.attention.value_length", 16), u32("qwen35.ssm.conv_kernel", 4),
                                 u32("qwen35.ssm.state_size", 8), u32("qwen35.ssm.group_count", 2),
                                 u32("qwen35.ssm.time_step_rank", 4), u32("qwen35.ssm.inner_size", 32),
                                 u32("qwen35.full_attention_interval", 4), u32("qwen35.feed_forward_length", 32)};
    CHECK(role("model.gguf", q35, {{"blk.0.ffn_up.weight", {64, 32}, F32, 1}, {"blk.3.attn_q.weight", {64, 64}, F32, 2}})
              .kind.empty());

    // LoRA adapters (llama.cpp's convert_lora_to_gguf.py: lora_a [inputs, rank], lora_b [rank, outputs])
    const std::vector<Kv> lora = {str("general.architecture", "qwen35"), str("general.type", "adapter"),
                                  str("adapter.type", "lora"), f32("adapter.lora.alpha", 64)};
    const std::vector<Tensor> up = {{"blk.0.ffn_up.weight.lora_a", {64, 8}, F16, 1}, {"blk.0.ffn_up.weight.lora_b", {8, 32}, F16, 2}};
    lamina::GgufRole r = role("lora.gguf", lora, up);
    CHECK(r.kind == "LoRA" && r.problem.empty() && r.what == "a LoRA adapter of qwen35, alpha 64" &&
          r.use.find("--lora") != std::string::npos);
    r = role("lora-pair.gguf", lora, {{"blk.0.ffn_up.weight.lora_a", {64, 8}, F16, 1}, {"blk.0.ffn_up.weight.lora_b", {32, 32}, F16, 2}});
    CHECK(r.kind == "LoRA" && r.problem.find("do not pair") != std::string::npos);
    r = role("lora-q4.gguf", lora, {{"blk.0.ffn_up.weight.lora_a", {64, 32}, Q4_0, 1}, {"blk.0.ffn_up.weight.lora_b", {32, 32}, F16, 2}});
    CHECK(r.problem.find("F32, F16, BF16 or Q8_0 pairs only") != std::string::npos);
    r = role("lora-a.gguf", lora, {{"blk.0.ffn_up.weight.lora_a", {64, 8}, F16, 1}});
    CHECK(r.problem.find("lacks its lora_b") != std::string::npos);
    std::vector<Kv> flash = lora;
    flash[0] = str("general.architecture", "qwen4exp");
    r = role("lora-flash.gguf", flash, up);
    CHECK(r.kind == "LoRA" && r.problem.find("an adapter of qwen4exp") != std::string::npos && r.use.empty());
    std::vector<Kv> alora = lora;
    alora.push_back(u32s("adapter.alora.invocation_tokens", {1, 2}));
    CHECK(role("alora.gguf", alora, up).problem.find("aLoRA") != std::string::npos);
    std::vector<Kv> prompt = lora;
    prompt[2] = str("adapter.type", "prompt");
    r = role("adapter.gguf", prompt, {{"x.weight", {32}, F32, 1}});
    CHECK(r.kind == "adapter" && !r.problem.empty());
    // against the model: each pair's product and its widths (the qwen35 path's load)
    {
        const lamina::GgufModel m = lamina::GgufModel::open((dir / "model.gguf").string());
        CHECK(lamina::core::lora_file_problem(lamina::GgufFile((dir / "lora.gguf").string()), &m).empty());
        write(dir / "lora-wide.gguf", lora, {{"blk.0.ffn_up.weight.lora_a", {64, 8}, F16, 1}, {"blk.0.ffn_up.weight.lora_b", {8, 48}, F16, 2}});
        CHECK(lamina::core::lora_file_problem(lamina::GgufFile((dir / "lora-wide.gguf").string()), &m)
                  .find("is 64 x 32 in the model, its pair 64 x 48") != std::string::npos);
        write(dir / "lora-none.gguf", lora, {{"blk.9.ffn_up.weight.lora_a", {64, 8}, F16, 1}, {"blk.9.ffn_up.weight.lora_b", {8, 32}, F16, 2}});
        CHECK(lamina::core::lora_file_problem(lamina::GgufFile((dir / "lora-none.gguf").string()), &m) ==
              "blk.9.ffn_up.weight: the model has no such tensor");
        CHECK(lamina::core::lora_file_problem(lamina::GgufFile((dir / "lora-flash.gguf").string()), &m) ==
              "an adapter of qwen4exp, the model is qwen35");
    }

    // control vectors (llama.cpp's cvector-generator: direction.<layer> from layer 1, the model's width in F32)
    const uint64_t w = (uint64_t) lamina::core::ModelGeometry{}.n_embd;
    const std::vector<Kv> cv = {str("general.architecture", "controlvector"), str("controlvector.model_hint", "qwen4exp"),
                                u32("controlvector.layer_count", 2)};
    r = role("cvec.gguf", cv, {{"direction.1", {w}, F32, 1}, {"direction.2", {w}, F32, 2}});
    CHECK(r.kind == "control vector" && r.problem.empty() && r.what == "a control vector for qwen4exp, 2 layers" &&
          r.use.find("--control-vector") != std::string::npos);
    r = role("cvec-narrow.gguf", cv, {{"direction.1", {64}, F32, 1}});
    CHECK(r.kind == "control vector" && r.problem.find("must be " + std::to_string(w) + " F32 values") != std::string::npos);
    CHECK(role("cvec-zero.gguf", cv, {{"direction.0", {w}, F32, 1}}).problem.find("a layer from 1 on") != std::string::npos);
    CHECK(role("cvec-empty.gguf", cv, {}).problem.find("no direction") != std::string::npos);
    {
        const lamina::GgufFile f((dir / "cvec.gguf").string());
        CHECK(lamina::core::control_vector_problem(f, (int64_t) w).empty() && !lamina::core::control_vector_problem(f, 64).empty());
    }
    // Engram tables: Qwen3.8-Flash-Next's PLE (rows of 160 values), another model's (DeepSeek V4.1's, of 256)
    r = role("ple.gguf", {}, {{"per_layer_token_embd.weight", {160, 6}, BF16, 1}});
    CHECK(r.kind == "engram" && r.problem.empty() && r.what == "an Engram table of BF16 rows of 160 values" &&
          r.use.find("--ple-gguf") != std::string::npos);
    r = role("engram.gguf", split_keys(5, 6, 1), {{"per_layer_token_embd.weight", {256, 4}, IQ4_NL, 1}});
    CHECK(r.kind == "engram" && r.what == "an Engram table of IQ4_NL rows of 256 values, shard 6 of 6 of a split model" &&
          r.problem == "per_layer_token_embd.weight has rows of 256 values, and the PLE's are rows of 160 values");
    // the rest: an importance matrix, a later shard, a GGUF of no architecture
    CHECK(role("imatrix.gguf", {str("general.type", "imatrix")}, {{"x.in_sum2", {32}, F32, 1}}).kind == "imatrix");
    r = role("m-00002-of-00002.gguf", split_keys(1, 2, 1), {{"x.weight", {32}, F32, 1}});
    CHECK(r.kind == "shard" && r.what == "shard 2 of 2 of a split model");
    CHECK(role("bare.gguf", {}, {{"x.weight", {32}, F32, 1}}).kind == "other");
    // role_refusal: what a load of one as a model says
    CHECK(lamina::role_refusal(role("cvec.gguf", cv, {{"direction.1", {w}, F32, 1}})) ==
          "a control vector, not a model: a control vector for qwen4exp, 1 layers; Lamina reads it --control-vector, on "
          "Qwen3.8-Flash-Next");
}

lamina::RemoteListing listing_of(std::vector<std::string> paths, std::string focus = "") {
    lamina::RemoteListing l;
    std::sort(paths.begin(), paths.end());
    for (const std::string& p : paths) {
        lamina::RemoteFile f;
        f.path = p;
        f.url = "https://x/" + p;
        f.size = 10;
        f.id = f.url;
        l.files.push_back(f);
    }
    l.focus = std::move(focus);
    return l;
}

void test_models_and_beside() {
    const lamina::RemoteListing l = listing_of({"a/x-00001-of-00002.gguf", "a/x-00002-of-00002.gguf", "b/y-00001-of-00003.gguf",
                                                "b/y-00002-of-00003.gguf", "mmproj-F16.gguf", "README.md", "c/z.GGUF"});
    const std::vector<lamina::RemoteModel> ms = lamina::remote_models(l);
    CHECK(ms.size() == 4);
    if (ms.size() == 4) {
        CHECK(ms[0].path == "a/x-00001-of-00002.gguf" && ms[0].shards.size() == 2 && ms[0].missing.empty() && ms[0].bytes == 20);
        CHECK(ms[1].path == "b/y-00001-of-00003.gguf" && ms[1].shards.size() == 3 && ms[1].missing == "b/y-00003-of-00003.gguf");
        CHECK(ms[2].path == "c/z.GGUF" && ms[2].shards.size() == 1);
        CHECK(ms[3].path == "mmproj-F16.gguf");
    }
    lamina::RemoteListing f = l;
    f.focus = "a/x-00002-of-00002.gguf";
    const std::vector<lamina::RemoteModel> one = lamina::remote_models(f);
    CHECK(one.size() == 1 && one[0].path == "a/x-00001-of-00002.gguf");

    const lamina::RemoteListing b = listing_of({"m/model.gguf", "m/mmproj-F16.gguf", "m/MMPROJ-x.gguf", "m/dspark-a.gguf",
                                                "m/dspark/dspark-b.gguf", "m/mtp.safetensors", "m/config.json",
                                                "m/sub/config.json", "m/sub/model-00148-of-00148.safetensors",
                                                "m/.hidden/config.json", "m/other.gguf", "n/mmproj.gguf",
                                                "m/Lamina-Qwen3.8-Flash-Next-MTP.gguf", "m/dspark/other.gguf"});
    std::vector<std::string> got = lamina::files_beside(b, "m/model.gguf", {});
    std::vector<std::string> want = {"m/Lamina-Qwen3.8-Flash-Next-MTP.gguf", "m/MMPROJ-x.gguf", "m/config.json",
                                     "m/dspark-a.gguf", "m/dspark/dspark-b.gguf", "m/mmproj-F16.gguf",
                                     "m/mtp.safetensors", "m/sub/config.json"};
    CHECK(got == want);
    got = lamina::files_beside(b, "m/model.gguf", {{"m/sub/config.json", "{\"dfly_target_kv_layers\": [1, 2]}"}});
    want.push_back("m/sub/model-00148-of-00148.safetensors");
    std::sort(want.begin(), want.end());
    CHECK(got == want);
    // a model in the root: the folders inside it are the repository's
    const lamina::RemoteListing r = listing_of({"model.gguf", "mmproj.gguf", "Q4/m-00001-of-00002.gguf", "Q4/config.json"});
    CHECK((lamina::files_beside(r, "model.gguf", {}) == std::vector<std::string>{"Q4/config.json", "mmproj.gguf"}));
}

/// a Hub: its API (a revision, a tree listed two entries a page), resolve's redirect to a download host, that host's
/// ranges; a gated repository; a host without ranges
class Hub {
public:
    std::map<std::string, std::string> files;   // o/r's, by path
    std::map<std::string, std::string> sha;     // by path
    std::atomic<int> cdn_with_token{0}, hub_without_token{0};
    std::map<std::string, int> cdn_reads;       // by the sha
    std::mutex mu;
    std::string token = "hf_test";

    Hub() {
        srv_.Get(R"(/api/models/o/(r|gated)/revision/main)", [&](const httplib::Request& q, httplib::Response& a) {
            note(q);
            a.set_content("{\"sha\": \"c0ffee\", \"id\": \"o/r\"}", "application/json");
        });
        srv_.Get(R"(/api/models/o/private/revision/main)", [&](const httplib::Request&, httplib::Response& a) {
            a.status = 401;
            a.set_content("{\"error\": \"Repository not found\"}", "application/json");
        });
        srv_.Get(R"(/api/models/o/(r|gated)/tree/c0ffee(/.*)?)", [&](const httplib::Request& q, httplib::Response& a) {
            note(q);
            const std::string dir = q.matches[2].str().empty() ? "" : q.matches[2].str().substr(1) + "/";
            std::vector<std::string> paths;
            for (const auto& [p, b] : files)
                if (p.rfind(dir, 0) == 0) paths.push_back(p);
            if (paths.empty()) {
                a.status = 404;
                a.set_content("{\"error\": \"no such path\"}", "application/json");
                return;
            }
            const size_t from = q.has_param("cursor") ? std::stoul(q.get_param_value("cursor")) : 0;
            const size_t to = std::min(paths.size(), from + 2);
            std::string out = "[{\"type\": \"directory\", \"path\": \"" + dir + "zz\", \"oid\": \"d\", \"size\": 0}";
            for (size_t i = from; i < to; ++i) {
                const std::string& p = paths[i];
                out += ", {\"type\": \"file\", \"path\": \"" + p + "\", \"oid\": \"git" + std::to_string(i) +
                       "\", \"size\": " + std::to_string(files[p].size());
                if (sha.count(p)) out += ", \"lfs\": {\"oid\": \"" + sha[p] + "\", \"size\": " + std::to_string(files[p].size()) + "}";
                out += "}";
            }
            out += "]";
            if (to < paths.size())
                a.set_header("Link", "<" + q.path + "?recursive=true&cursor=" + std::to_string(to) + ">; rel=\"next\"");
            a.set_content(out, "application/json");
        });
        srv_.Get(R"(/o/(r|gated)/resolve/c0ffee/(.*))", [&](const httplib::Request& q, httplib::Response& a) {
            note(q);
            const std::string repo = q.matches[1].str(), p = q.matches[2].str();
            if (repo == "gated") {
                a.status = 403;
                a.set_content("{\"error\": \"Access to model o/gated is restricted\"}", "application/json");
                return;
            }
            if (!files.count(p)) {
                a.status = 404;
                return;
            }
            a.status = 302;
            a.set_content("Found. Redirecting", "text/plain");
            a.set_header("Location", "http://127.0.0.1:" + std::to_string(cdn_port_) + "/cdn/" + sha[p] + "?sig=1");
        });
        cdn_.Get(R"(/cdn/(\w+))", [&](const httplib::Request& q, httplib::Response& a) {   // the download host
            {
                std::lock_guard<std::mutex> lk(mu);
                ++cdn_reads[q.matches[1].str()];
            }
            if (q.has_header("Authorization")) ++cdn_with_token;
            for (const auto& [p, s] : sha)
                if (s == q.matches[1].str()) {
                    a.set_content(files[p], "application/octet-stream");   // (cpp-httplib answers a Range with 206)
                    return;
                }
            a.status = 404;
        });
        srv_.Get(R"(/plain/(.*))", [&](const httplib::Request& q, httplib::Response& a) {   // no ranges: 200, all of it
            const std::string p = q.matches[1].str();
            if (!plain_.count(p)) {
                a.status = 404;
                return;
            }
            const std::string body = plain_[p];
            a.status = 200;   // (a status set: cpp-httplib answers no range of it)
            a.set_content_provider(body.size(), "application/octet-stream",
                                   [body](size_t off, size_t len, httplib::DataSink& sink) {
                                       return sink.write(body.data() + off, std::min<size_t>(len, 65536));
                                   });
            a.set_header("Accept-Ranges", "none");
        });
        cdn_port_ = cdn_.bind_to_any_port("127.0.0.1");
        cdn_th_ = std::thread([this] { cdn_.listen_after_bind(); });
        cdn_.wait_until_ready();
        port_ = srv_.bind_to_any_port("127.0.0.1");
        th_ = std::thread([this] { srv_.listen_after_bind(); });
        srv_.wait_until_ready();
    }
    ~Hub() {
        srv_.stop();
        th_.join();
        cdn_.stop();
        cdn_th_.join();
    }
    std::string url() const { return "http://127.0.0.1:" + std::to_string(port_); }
    std::map<std::string, std::string> plain_;   // the host without ranges' files

private:
    void note(const httplib::Request& q) {
        if (q.get_header_value("Authorization") != "Bearer " + token) ++hub_without_token;
    }
    httplib::Server srv_, cdn_;   // the Hub, and its download host (another port: another origin)
    int port_ = 0, cdn_port_ = 0;
    std::thread th_, cdn_th_;
};

std::string model_bytes(const fs::path& dir, const std::string& name, std::vector<fixture::Kv> kv,
                        std::vector<fixture::Tensor> ts) {
    const fs::path p = dir / name;
    fixture::write(p, kv, ts);
    return slurp(p);
}

void test_hub(const fs::path& dir) {
    using namespace fixture;
    Hub hub;
    std::vector<std::string> vocab;
    for (int i = 0; i < 20000; ++i) vocab.push_back("t" + std::to_string(i));   // a header past the first 64 KiB read
    const std::string s1 = model_bytes(dir, "s1.gguf", [&] {
        std::vector<Kv> kv{str("general.architecture", "qwen4exp"), strs("tokenizer.ggml.tokens", vocab)};
        for (const Kv& k : split_keys(0, 2, 1)) kv.push_back(k);
        return kv;
    }(), {});
    const std::string s2 = model_bytes(dir, "s2.gguf", split_keys(1, 2, 1), {{"big.weight", {4096, 512}, 0, 3}});   // 8 MiB of data
    const std::string proj = model_bytes(dir, "p.gguf", {str("general.architecture", "clip")}, {{"p.weight", {32, 2}, 0, 4}});
    hub.files = {{"Q4/m-00001-of-00002.gguf", s1}, {"Q4/m-00002-of-00002.gguf", s2}, {"Q8/m-00001-of-00002.gguf", s1},
                 {"Q8/m-00002-of-00002.gguf", s2}, {"mmproj-F16.gguf", proj}, {"Q4/config.json", "{\"a\": 1}"},
                 {"README.md", "# r"}};
    hub.sha = {{"Q4/m-00001-of-00002.gguf", "aaa1"}, {"Q4/m-00002-of-00002.gguf", "bbb2"}, {"Q8/m-00001-of-00002.gguf", "aaa1"},
               {"Q8/m-00002-of-00002.gguf", "bbb2"}, {"mmproj-F16.gguf", "ccc3"}, {"Q4/config.json", "ddd4"}};

    // the listing: the commit, every page, sizes and ids
    lamina::RemoteListing l = lamina::list_remote(hub.url() + "/o/r", hub.token, hub.url());
    CHECK(l.hub && l.commit == "c0ffee" && l.repo == "o/r" && l.focus.empty() && l.files.size() == 7);
    CHECK(l.base == hub.url() + "/o/r/resolve/c0ffee/");
    const lamina::RemoteFile* f2 = l.find("Q4/m-00002-of-00002.gguf");
    CHECK(f2 != nullptr && f2->size == s2.size() && f2->sha256 == "bbb2" && f2->id == "bbb2");
    CHECK(l.find("README.md") != nullptr && l.find("README.md")->id.rfind("git:", 0) == 0);
    CHECK(hub.hub_without_token == 0);
    const std::vector<lamina::RemoteModel> ms = lamina::remote_models(l);
    CHECK(ms.size() == 3);

    {
        lamina::RemoteCopies copies(l, hub.token);
        copies.fetch({"Q4/m-00001-of-00002.gguf", "Q4/m-00002-of-00002.gguf", "Q8/m-00001-of-00002.gguf",
                      "Q8/m-00002-of-00002.gguf", "mmproj-F16.gguf", "Q4/config.json", "nothere.gguf"});
        CHECK(copies.error("nothere.gguf") == "not in the listing");
        for (const char* p : {"Q4/m-00001-of-00002.gguf", "Q4/m-00002-of-00002.gguf", "Q8/m-00002-of-00002.gguf", "mmproj-F16.gguf"}) {
            const std::string local = copies.local(p);
            CHECK(!local.empty() && copies.error(p).empty());
            if (local.empty()) continue;
            CHECK(fs::file_size(local) == hub.files[p].size());
            // the reader reads the copy as it reads the file
            const lamina::GgufFile a(local);
            const lamina::GgufFile b((dir / (std::string(p).find("00001") != std::string::npos ? "s1.gguf"
                                             : std::string(p).find("00002") != std::string::npos ? "s2.gguf" : "p.gguf")).string());
            CHECK(a.data_start() == b.data_start() && a.file_size() == b.file_size() && a.tensors().size() == b.tensors().size() &&
                  a.metadata().size() == b.metadata().size());
            for (size_t i = 0; i < a.tensors().size() && i < b.tensors().size(); ++i)
                CHECK(a.tensors()[i].name == b.tensors()[i].name && a.tensors()[i].offset == b.tensors()[i].offset &&
                      a.tensors()[i].type == b.tensors()[i].type && a.tensors()[i].shape == b.tensors()[i].shape);
        }
        const std::string big = copies.local("Q4/m-00002-of-00002.gguf");
        if (!big.empty()) CHECK(allocated(big) < 2 * 1048576);   // its 8 MiB of data: a hole
        CHECK(slurp(copies.local("Q4/config.json")) == "{\"a\": 1}");
        // one read an id; the 8 MiB shard's data not read
        CHECK(hub.cdn_reads["aaa1"] == 1 && hub.cdn_reads["bbb2"] == 1);
        CHECK(copies.bytes_read() < 2 * s1.size() + 2 * 1048576);
        CHECK(copies.files_read() == 4);
        CHECK(hub.cdn_with_token == 0 && hub.hub_without_token == 0);
        // the copies go with them
        const std::string root = copies.root();
        CHECK(fs::is_directory(root));
        lamina::RemoteListing& keep = l;
        (void) keep;
        static std::string gone;
        gone = root;
    }

    // a file's URL: its split model's shards; a folder's
    l = lamina::list_remote(hub.url() + "/o/r/blob/main/Q8/m-00002-of-00002.gguf", "", hub.url());
    CHECK(l.focus == "Q8/m-00002-of-00002.gguf" && l.scope == "Q8" && l.files.size() == 2);
    const std::vector<lamina::RemoteModel> one = lamina::remote_models(l);
    CHECK(one.size() == 1 && one[0].path == "Q8/m-00001-of-00002.gguf" && one[0].missing.empty());
    l = lamina::list_remote(hub.url() + "/o/r/tree/main/Q4", "", hub.url());
    CHECK(l.focus.empty() && l.scope == "Q4" && l.files.size() == 3);
    bool threw = false;
    try {
        lamina::list_remote(hub.url() + "/o/r/blob/main/Q4/none.gguf", "", hub.url());
    } catch (const lamina::RemoteError& e) {
        threw = std::string(e.what()).find("no file Q4/none.gguf") != std::string::npos;
    }
    CHECK(threw);
    threw = false;
    try {
        lamina::list_remote(hub.url() + "/o/private", "", hub.url());
    } catch (const lamina::RemoteError& e) {
        threw = std::string(e.what()).find("HF_TOKEN") != std::string::npos;
    }
    CHECK(threw);

    // a gated repository: listed, its files refused
    lamina::RemoteListing g = lamina::list_remote(hub.url() + "/o/gated", "", hub.url());
    {
        lamina::RemoteCopies copies(g, "");
        copies.fetch({"mmproj-F16.gguf"});
        CHECK(copies.local("mmproj-F16.gguf").empty() && copies.error("mmproj-F16.gguf").find("gated") != std::string::npos);
    }

    // a direct URL: a server without ranges, a split model's missing shard, a page that is no GGUF
    hub.plain_ = {{"d/m-00001-of-00002.gguf", s1}, {"d/x.gguf", s2}, {"d/page.gguf", "<html>no</html>"}};
    l = lamina::list_remote(hub.url() + "/plain/d/m-00001-of-00002.gguf", "");
    CHECK(!l.hub && l.files.size() == 2 && l.files[1].path == "m-00002-of-00002.gguf" &&
          l.files[1].url == hub.url() + "/plain/d/m-00002-of-00002.gguf");
    {
        lamina::RemoteCopies copies(l, "");
        copies.fetch({"m-00001-of-00002.gguf", "m-00002-of-00002.gguf"});
        CHECK(!copies.local("m-00001-of-00002.gguf").empty() && fs::file_size(copies.local("m-00001-of-00002.gguf")) == s1.size());
        CHECK(copies.error("m-00002-of-00002.gguf").find("404") != std::string::npos);
    }
    l = lamina::list_remote(hub.url() + "/plain/d/x.gguf", "");
    {
        lamina::RemoteCopies copies(l, "");
        copies.fetch({"x.gguf"});
        CHECK(!copies.local("x.gguf").empty() && fs::file_size(copies.local("x.gguf")) == s2.size());
        CHECK(copies.bytes_read() < 2 * 1048576);   // the reply stopped once the header was read
    }
    l = lamina::list_remote(hub.url() + "/plain/d/page.gguf", "");
    {
        lamina::RemoteCopies copies(l, "");
        copies.fetch({"page.gguf"});
        CHECK(copies.error("page.gguf").find("not a GGUF") != std::string::npos);
    }
}

void set_env(const char* k, const std::string& v) {
#ifdef _WIN32
    _putenv_s(k, v.c_str());
#else
    if (v.empty()) unsetenv(k);
    else setenv(k, v.c_str(), 1);
#endif
}

/// read_remote_models: what each GGUF of a repository is, and a split model's last shard the repository holds under
/// another name (one table shard for several quants, as DeepSeek V4.1's Engram): found by its split keys and read
/// under the shard's own name; a model whose shard is nowhere stays incomplete
void test_found(const fs::path& dir) {
    using namespace fixture;
    Hub hub;
    set_env("HF_ENDPOINT", hub.url());
    set_env("HF_TOKEN", hub.token);
    std::vector<Kv> k1 = {str("general.architecture", "qwen4exp")}, k3 = k1;
    for (const Kv& k : split_keys(0, 2, 2)) k1.push_back(k);
    for (const Kv& k : split_keys(0, 3, 3)) k3.push_back(k);
    const std::string s1 = model_bytes(dir, "f1.gguf", k1, {{"token_embd.weight", {32, 4}, 0, 1}});
    const std::string table = model_bytes(dir, "f2.gguf", split_keys(1, 2, 2), {{"per_layer_token_embd.weight", {256, 4}, 20, 2}});
    const std::string lonely = model_bytes(dir, "f3.gguf", k3, {{"token_embd.weight", {32, 4}, 0, 1}});
    const std::string cvec = model_bytes(dir, "f4.gguf", {str("general.architecture", "controlvector")},
                                         {{"direction.1", {(uint64_t) lamina::core::ModelGeometry{}.n_embd}, 0, 3}});
    hub.files = {{"A/m-00001-of-00002.gguf", s1}, {"B/m-00001-of-00002.gguf", s1}, {"engram.gguf", table},
                 {"C/n-00001-of-00003.gguf", lonely}, {"direction.gguf", cvec}};
    hub.sha = {{"A/m-00001-of-00002.gguf", "a1"}, {"B/m-00001-of-00002.gguf", "a1"}, {"engram.gguf", "e1"},
               {"C/n-00001-of-00003.gguf", "c1"}, {"direction.gguf", "d1"}};
    const lamina::RemoteModels r = lamina::read_remote_models(hub.url() + "/o/r");
    set_env("HF_ENDPOINT", "");
    set_env("HF_TOKEN", "");
    CHECK(r.found.size() == 5 && r.models().size() == 3);
    std::map<std::string, const lamina::RemoteFound*> by;
    for (const lamina::RemoteFound& f : r.found) by[f.model.path] = &f;
    for (const std::string m : {"A/m-00001-of-00002.gguf", "B/m-00001-of-00002.gguf"}) {
        const lamina::RemoteFound* f = by[m];
        CHECK(f != nullptr && f->kind == "model" && f->error.empty() && f->model.missing.empty() &&
              f->model.bytes == s1.size() + table.size());
        if (f == nullptr) continue;
        CHECK(f->notes.size() == 1 && f->notes[0].find("its shard 2 of 2 is engram.gguf") != std::string::npos);
        // the copy under the shard's own name: the reader reads the model's two shards from it
        const std::string second = m.substr(0, 2) + "m-00002-of-00002.gguf";
        CHECK(!r.copies->local(second).empty() && fs::file_size(r.copies->local(second)) == table.size());
        size_t tensors = 0;
        try {
            const lamina::GgufModel g = lamina::GgufModel::open(r.copies->local(m));
            tensors = g.find("per_layer_token_embd.weight") != nullptr && g.find("token_embd.weight") != nullptr ? 2 : 1;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "  %s: %s\n", m.c_str(), e.what());
        }
        CHECK(tensors == 2);
    }
    const lamina::RemoteFound* e = by["engram.gguf"];
    CHECK(e != nullptr && e->kind == "engram" && !e->read && e->notes.size() == 2 &&
          e->notes[0] == "shard 2 of 2 of A/m-00001-of-00002.gguf");
    const lamina::RemoteFound* c = by["C/n-00001-of-00003.gguf"];
    CHECK(c != nullptr && c->kind == "model" && c->error.find("missing model shard C/n-00002-of-00003.gguf") != std::string::npos);
    const lamina::RemoteFound* v = by["direction.gguf"];
    CHECK(v != nullptr && v->kind == "control vector" && v->read && v->what.find("a model it does not name") != std::string::npos);
    CHECK(hub.cdn_with_token == 0 && hub.hub_without_token == 0);
}

}  // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / ("remote_model_test_" + lamina::random_hex(8));   // (a run's own)
    fs::create_directories(dir);
    // (each part's name first: a part that ends the process is named by the last line)
    std::fprintf(stderr, "urls\n");
    test_urls();
    std::fprintf(stderr, "header length\n");
    test_header_length(dir);
    std::fprintf(stderr, "models and the files beside them\n");
    test_models_and_beside();
    std::fprintf(stderr, "drafters\n");
    test_drafters(dir);
    std::fprintf(stderr, "LoRA adapters, control vectors, Engram tables ...\n");
    test_roles(dir);
    std::fprintf(stderr, "a Hub\n");
    try {
        test_hub(dir);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: the Hub's part threw: %s\n", e.what());
        ++fails;
    }
    std::fprintf(stderr, "a repository's files, read\n");
    try {
        test_found(dir);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: the repository's part threw: %s\n", e.what());
        ++fails;
    }
    std::error_code ec;
    fs::remove_all(dir, ec);
    if (fails != 0) {
        std::fprintf(stderr, "%d failed\n", fails);
        return 1;
    }
    std::printf("remote_model_test: all passed\n");
    return 0;
}
