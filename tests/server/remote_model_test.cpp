// lamina-server --check -m URL's pieces (src/server/remote_model.hpp), no model and no GPU: Hugging Face URLs as the
// server reads them, a GGUF header's length from its first bytes, a listing's models and the files read beside a model,
// and against a Hub played here (cpp-httplib: its API's revision and paged tree, resolve's redirect to a download host
// that serves ranges, a gated repository, a server without ranges, a direct URL's missing shard) the listing and the
// headers' copies: sparse files of each file's size that the GGUF reader reads as it reads the file, the token sent to
// the Hub's host alone, one read for files of one id, a file's data not read.  (CPU only.)
#include <httplib.h>

#include "remote_model.hpp"

#include "../core/gguf_fixture.hpp"
#include "lamina/artifact/gguf_reader.hpp"

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

}  // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / ("remote_model_test_" + std::to_string(std::rand()));
    fs::create_directories(dir);
    // (each part's name first: a part that ends the process is named by the last line)
    std::fprintf(stderr, "urls\n");
    test_urls();
    std::fprintf(stderr, "header length\n");
    test_header_length(dir);
    std::fprintf(stderr, "models and the files beside them\n");
    test_models_and_beside();
    std::fprintf(stderr, "a Hub\n");
    try {
        test_hub(dir);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: the Hub's part threw: %s\n", e.what());
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
