// core/control_vector.hpp's host side: --control-vector*'s lists, and the tables built from GGUF control vectors as
// llama.cpp reads them (direction.<l> times each file's scale, summed per layer; the range; add and project; single:L)
// and the files it refuses.  CPU only: the files are written to a temporary folder.
#include "lamina/core/control_vector.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {
int fails = 0;
void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++fails;
    }
}

// A GGUF v3 file of F32 tensors with a string architecture and model hint, as llama-cvector-generator writes one.
struct Tensor {
    std::string name;
    std::vector<float> v;
    uint32_t type = 0;
};
void put_u32(std::string& b, uint32_t v) { b.append((const char*) &v, 4); }
void put_u64(std::string& b, uint64_t v) { b.append((const char*) &v, 8); }
void put_str(std::string& b, const std::string& s) {
    put_u64(b, s.size());
    b += s;
}
std::string write_gguf(const std::filesystem::path& dir, const char* file, const std::vector<Tensor>& ts,
                       const char* hint = "qwen4exp") {
    std::string b = "GGUF";
    put_u32(b, 3);
    put_u64(b, ts.size());
    put_u64(b, 2);
    put_str(b, "general.architecture");
    put_u32(b, 8);
    put_str(b, "controlvector");
    put_str(b, "controlvector.model_hint");
    put_u32(b, 8);
    put_str(b, hint);
    uint64_t off = 0;
    for (const Tensor& t : ts) {
        put_str(b, t.name);
        put_u32(b, 1);
        put_u64(b, t.type == 1 ? t.v.size() * 2 : t.v.size());   // F16: two values a float of `v`
        put_u32(b, t.type);
        put_u64(b, off);
        off += (t.v.size() * 4 + 31) / 32 * 32;
    }
    b.resize((b.size() + 31) / 32 * 32, '\0');
    for (const Tensor& t : ts) {
        const size_t at = b.size();
        b.append((const char*) t.v.data(), t.v.size() * 4);
        b.resize(at + (t.v.size() * 4 + 31) / 32 * 32, '\0');
    }
    const std::filesystem::path p = dir / file;
    std::FILE* f = std::fopen(p.string().c_str(), "wb");
    if (f != nullptr) {
        std::fwrite(b.data(), 1, b.size(), f);
        std::fclose(f);
    }
    return p.string();
}
std::vector<float> ramp(int n, float k) {
    std::vector<float> v((size_t) n);
    for (int j = 0; j < n; ++j) v[(size_t) j] = k * (float) (j + 1) * (j % 2 ? -1.0f : 1.0f);
    return v;
}
}  // namespace

int main() {
    using lamina::core::ControlVectorSpec;
    using lamina::core::ControlVectorTables;
    constexpr int E = 8, L = 6;   // n_embd, layers

    // ---- the lists
    {
        std::vector<ControlVectorSpec::File> f;
        std::string err;
        check(lamina::core::parse_control_vectors("D:/a/b.gguf:1.5,C:/x y.gguf:-1", true, f, err) && f.size() == 2 &&
                  f[0].path == "D:/a/b.gguf" && f[0].scale == 1.5f && f[1].path == "C:/x y.gguf" && f[1].scale == -1.0f,
              "FILE:SCALE list, drive letters kept");
        check(lamina::core::parse_control_vectors("a.gguf,b.gguf", false, f, err) && f.size() == 4 &&
                  f[2].path == "a.gguf" && f[2].scale == 1.0f && f[3].path == "b.gguf",
              "a plain list appends at scale 1");
        for (const char* bad : {"D:/x.gguf", "x.gguf:", "x.gguf:abc", "x.gguf:1e999", ":1", "a.gguf:1,", "x.gguf:1x"})
            check(!lamina::core::parse_control_vectors(bad, true, f, err), std::string("refused: ") + bad);
        check(!lamina::core::parse_control_vectors("a.gguf,,b.gguf", false, f, err), "refused: an empty name");
    }

    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "lamina_control_vector_test";
    std::filesystem::create_directories(dir);
    const std::vector<float> d1 = ramp(E, 0.5f), d2 = ramp(E, -0.25f);
    std::vector<Tensor> one;   // direction.1 .. direction.5, and one past the model's layers
    for (int l = 1; l < L; ++l) one.push_back({"direction." + std::to_string(l), ramp(E, (float) l)});
    one.push_back({"direction.9", ramp(E, 7.0f)});
    const std::string f1 = write_gguf(dir, "one.gguf", one);
    const std::string f2 = write_gguf(dir, "two.gguf", {{"direction.2", d1}, {"direction.4", d2}}, "llama");

    // ---- add: the scaled files summed per layer, the default range 1 .. L-1, u zero
    {
        ControlVectorSpec s;
        s.files = {{f1, 2.0f}, {f2, -1.0f}};
        ControlVectorTables t;
        std::string err;
        const bool ok = lamina::core::control_vector_tables(s, L, E, t, err);
        check(ok, "add: " + err);
        check(t.first == 1 && t.last == L - 1 && t.layers == std::vector<int64_t>({1, 2, 3, 4, 5}), "add: the range");
        check(t.warnings.size() == 1 && t.warnings[0].find("'llama'") != std::string::npos, "add: the hint warning");
        bool same = ok && t.rows.size() == (size_t) (5 * 2 * E);
        for (int i = 0; same && i < 5; ++i) {
            const int l = i + 1;
            const std::vector<float> base = ramp(E, (float) l);
            for (int j = 0; j < E; ++j) {
                double want = 2.0 * base[(size_t) j];
                if (l == 2) want -= d1[(size_t) j];
                if (l == 4) want -= d2[(size_t) j];
                same = same && t.rows[(size_t) (i * 2 * E + j)] == (float) want && t.rows[(size_t) (i * 2 * E + E + j)] == 0.0f;
            }
        }
        check(same, "add: a = the sum, u = 0");
    }
    // ---- project in a range: a zero, (h . u) u = |d| (h . v) v
    {
        ControlVectorSpec s;
        s.files = {{f1, 3.0f}};
        s.project = true;
        s.first = 2;
        s.last = 3;
        ControlVectorTables t;
        std::string err;
        const bool ok = lamina::core::control_vector_tables(s, L, E, t, err);
        check(ok && t.layers == std::vector<int64_t>({2, 3}), "project: layers 2..3 " + err);
        bool good = ok;
        for (int i = 0; good && i < 2; ++i) {
            const std::vector<float> base = ramp(E, (float) (i + 2));
            double nd = 0;
            for (float x : base) nd += 9.0 * x * x;
            nd = std::sqrt(nd);
            double hu = 0, hv = 0;
            for (int j = 0; j < E; ++j) {
                const double h = std::sin(1.0 + j), v = 3.0 * base[(size_t) j] / nd;
                hu += h * t.rows[(size_t) (i * 2 * E + E + j)];
                hv += h * v;
                good = good && t.rows[(size_t) (i * 2 * E + j)] == 0.0f;
            }
            for (int j = 0; j < E; ++j) {
                const double v = 3.0 * base[(size_t) j] / nd;
                const double got = hu * t.rows[(size_t) (i * 2 * E + E + j)], want = nd * hv * v;
                good = good && std::fabs(got - want) <= 1e-5 * (std::fabs(want) + 1e-3);
            }
        }
        check(good, "project: a = 0, (h . u) u = |d| (h . v) v");
    }
    // ---- single:L, and a range past the model clamped
    {
        ControlVectorSpec s;
        s.files = {{f1, 1.0f}};
        s.project = true;
        s.single = 4;
        s.first = 3;
        s.last = 40;
        ControlVectorTables t;
        std::string err;
        const bool ok = lamina::core::control_vector_tables(s, L, E, t, err);
        check(ok && t.last == L - 1 && t.layers == std::vector<int64_t>({3, 4, 5}), "single: layers 3..5 " + err);
        check(ok && std::memcmp(t.rows.data() + E, t.rows.data() + 5 * E, E * 4) == 0, "single: layer 4's u everywhere");
    }
    // ---- refused
    auto refused = [&](const ControlVectorSpec& s, const char* what) {
        ControlVectorTables t;
        std::string err;
        check(!lamina::core::control_vector_tables(s, L, E, t, err) && !err.empty(), std::string("refused: ") + what);
    };
    {
        ControlVectorSpec s;
        refused(s, "no file");
        s.files = {{(dir / "missing.gguf").string(), 1.0f}};
        refused(s, "a missing file");
        s.files = {{write_gguf(dir, "zero.gguf", {{"direction.0", d1}}), 1.0f}};
        refused(s, "direction.0");
        s.files = {{write_gguf(dir, "name.gguf", {{"token_embd.weight", d1}}), 1.0f}};
        refused(s, "another tensor");
        s.files = {{write_gguf(dir, "suffix.gguf", {{"direction.3x", d1}}), 1.0f}};
        refused(s, "direction.3x");
        s.files = {{write_gguf(dir, "width.gguf", {{"direction.1", ramp(E + 1, 1.0f)}}), 1.0f}};
        refused(s, "another n_embd");
        Tensor f16{"direction.1", ramp(E / 2, 1.0f), 1};
        s.files = {{write_gguf(dir, "type.gguf", {f16}), 1.0f}};
        refused(s, "an F16 tensor");
        s.files = {{f1, 1.0f}};
        s.first = 5;
        s.last = 4;
        refused(s, "an empty range");
        s.first = s.last = -1;
        s.single = 2;
        refused(s, "single:L in add mode");
        s.project = true;
        s.single = 9;
        refused(s, "single:L past the model");
        s.single = -1;
        s.files = {{f1, 0.0f}};
        refused(s, "scale 0: no direction left");
    }
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    if (fails == 0) std::puts("control_vector_test: OK");
    return fails == 0 ? 0 : 1;
}
