// tests/core/vision_records_test.cpp - the picture rows' reader (program/serve_io.hpp, read_vision_records): a list of
// the encoder's files reads as their records one after the other, the rows and pictures of one file of them all; a
// list in a list and a file the list names that is not there are errors. CPU only.
#include "lamina/program/serve_io.hpp"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using lamina::program::read_vision_records;
using lamina::program::VisionImage;

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++g_fail;
}

constexpr int32_t kEmbd = 4;

/// A record as the encoder writes it: 'SVE1', n, nx, ny, n_embd, then n x n_embd floats (row r's value i: base + r + i)
void record(std::FILE* f, int32_t nx, int32_t ny, float base) {
    const int32_t n = nx * ny, hdr[5] = {0x31455653, n, nx, ny, kEmbd};
    std::fwrite(hdr, sizeof hdr, 1, f);
    for (int32_t r = 0; r < n; ++r)
        for (int32_t i = 0; i < kEmbd; ++i) {
            const float v = base + (float) r + (float) i / 10.0f;
            std::fwrite(&v, sizeof v, 1, f);
        }
}

void text_file(const fs::path& p, const std::string& text) {
    std::FILE* f = std::fopen(p.string().c_str(), "wb");
    std::fwrite(text.data(), 1, text.size(), f);
    std::fclose(f);
}

bool same(const std::vector<VisionImage>& a, const std::vector<VisionImage>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i].n != b[i].n || a[i].nx != b[i].nx || a[i].ny != b[i].ny || a[i].off != b[i].off) return false;
    return true;
}

}   // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / "lamina-vision-records-test";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);
    const fs::path pic = dir / "pic.sve", video = dir / "video.sve", whole = dir / "whole.sve", list = dir / "req.svl";
    std::FILE* a = std::fopen(pic.string().c_str(), "wb");
    record(a, 3, 2, 100.0f);   // a picture of 6 rows
    std::fclose(a);
    std::FILE* b = std::fopen(video.string().c_str(), "wb");
    for (int g = 0; g < 3; ++g) record(b, 2, 2, 200.0f + 10.0f * (float) g);   // a video of three groups of 4
    std::fclose(b);
    std::FILE* w = std::fopen(whole.string().c_str(), "wb");   // the two one after the other, as one file
    record(w, 3, 2, 100.0f);
    for (int g = 0; g < 3; ++g) record(w, 2, 2, 200.0f + 10.0f * (float) g);
    std::fclose(w);
    text_file(list, "SVL1\n" + pic.string() + "\n" + video.string() + "\n");

    std::vector<float> rows_w, rows_l;
    std::vector<VisionImage> imgs_w, imgs_l;
    check(read_vision_records(whole.string(), kEmbd, rows_w, imgs_w).empty(), "one file of records reads");
    check(read_vision_records(list.string(), kEmbd, rows_l, imgs_l).empty(), "a list of the files reads");
    check(imgs_w.size() == 4 && rows_w.size() == (size_t) (6 + 12) * kEmbd, "the file: 4 records, 18 rows");
    check(rows_l == rows_w, "the list's rows are the file's, value for value");
    check(same(imgs_l, imgs_w), "the list's pictures are the file's: grids and offsets");

    std::vector<float> rows;
    std::vector<VisionImage> imgs;
    text_file(dir / "outer.svl", "SVL1\n" + list.string() + "\n");
    check(!read_vision_records((dir / "outer.svl").string(), kEmbd, rows, imgs).empty(), "a list in a list: an error");
    rows.clear();
    imgs.clear();
    text_file(dir / "gone.svl", "SVL1\n" + (dir / "gone.sve").string() + "\n");
    const std::string e = read_vision_records((dir / "gone.svl").string(), kEmbd, rows, imgs);
    check(e.rfind("cannot open", 0) == 0, "a file the list names that is not there: cannot open");
    rows.clear();
    imgs.clear();
    check(!read_vision_records(list.string(), kEmbd + 1, rows, imgs).empty(), "records of another width: an error");

    fs::remove_all(dir, ec);
    std::printf(g_fail == 0 ? "vision_records_test OK\n" : "vision_records_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
