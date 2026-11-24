// src/platform/spill_test.cpp - the prompt cache's disk tier, its platform part (CPU only, no GPU, no model):
//
//   1. a folder of this process's own under a root, named <pid>-<16 hex digits>, held while open and removed by
//      close();
//   2. a start removes the folders of that form whose lock nobody holds (an ended engine's), and keeps another
//      engine's live folder and anything else under the root;
//   3. unbuffered files: aligned writes and reads at offsets read back what was written, a file grows by a write
//      past its end, an unaligned request is refused, an existing file opens without being emptied.
#include "lamina/platform/spill.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

namespace fs = std::filesystem;
using lamina::platform::kSpillAlign;
using lamina::platform::SpillDir;
using lamina::platform::SpillFile;

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++g_fail;
}

std::string u8(const fs::path& p) {
    const std::u8string u = p.u8string();
    return std::string(u.begin(), u.end());
}

}  // namespace

int main() {
    std::printf("platform_spill_test\n");
    std::error_code ec;
    const fs::path root = fs::temp_directory_path(ec) / "lamina-spill-test";
    fs::remove_all(root, ec);
    fs::create_directories(root / "123-0123456789abcdef", ec);   // an ended engine's folder: no lock held
    std::ofstream(root / "123-0123456789abcdef" / "1.kv") << "old";
    fs::create_directories(root / "keep-me", ec);                  // not an engine's
    fs::create_directories(root / "77-0123456789ABCDEF", ec);      // not of that form (upper-case hex)
    {   // 1-2. the folders
        SpillDir a;
        std::string err;
        int cleaned = -1;
        check(a.open(u8(root), err, &cleaned), "a folder is taken under the root");
        const fs::path pa = fs::path(std::u8string(reinterpret_cast<const char8_t*>(a.path().data()), a.path().size()));
        check(fs::is_directory(pa) && pa.parent_path() == root, "inside the root");
        const std::string name = u8(pa.filename());
        check(name.size() > 17 && name[name.size() - 17] == '-', "named <pid>-<16 hex digits>");
        check(cleaned == 1 && !fs::exists(root / "123-0123456789abcdef"), "the ended engine's folder is removed");
        check(fs::exists(root / "keep-me") && fs::exists(root / "77-0123456789ABCDEF"),
              "folders of other names are kept");
        check(a.free_bytes() > 0, "the disk's free bytes are read");
        SpillDir b;
        check(b.open(u8(root), err, &cleaned) && cleaned == 0 && fs::is_directory(pa),
              "a second engine on the root keeps the first one's live folder");
        b.close();
        check(fs::is_directory(pa), "and closing it leaves that folder");

        // 3. the files
        const uint64_t n = 3 * kSpillAlign;
        auto* buf = (uint8_t*) lamina::platform::spill_alloc(n);
        auto* back = (uint8_t*) lamina::platform::spill_alloc(n);
        check(buf != nullptr && back != nullptr && (uintptr_t) buf % kSpillAlign == 0, "page-aligned buffers");
        for (uint64_t i = 0; i < n; ++i) buf[i] = (uint8_t) (i * 7 + 3);
        const std::string path = a.file("1.kv");
        {
            SpillFile f;
            check(f.open(path, true, err), "a file is made");
            check(f.write(0, buf, n, err), "an aligned write");
            check(f.write(5 * kSpillAlign, buf, kSpillAlign, err), "a write past its end");
            check(!f.write(100, buf, kSpillAlign, err), "an unaligned offset is refused");
            check(!f.write(0, buf + 1, kSpillAlign, err), "an unaligned buffer is refused");
            check(!f.write(0, buf, 100, err), "an unaligned length is refused");
        }
        {
            SpillFile f;
            check(!f.open(a.file("none.kv"), false, err), "a file that is not there does not open to read");
            check(f.open(path, true, err), "the file opens again");
            std::memset(back, 0, n);
            check(f.read(0, back, n, err) && std::memcmp(back, buf, n) == 0, "kept, and read back as written");
            std::memset(back, 0, n);
            check(f.read(5 * kSpillAlign, back, kSpillAlign, err) && std::memcmp(back, buf, kSpillAlign) == 0,
                  "the page past the old end too");
            check(!f.read(8 * kSpillAlign, back, kSpillAlign, err), "a read past its end fails");
        }
        check(fs::file_size(fs::path(std::u8string(reinterpret_cast<const char8_t*>(path.data()), path.size())), ec) ==
                  6 * kSpillAlign, "its size: whole pages");
        lamina::platform::spill_remove(path);
        check(!fs::exists(pa / "1.kv"), "spill_remove removes it");
        lamina::platform::spill_free(buf);
        lamina::platform::spill_free(back);
        a.close();
        check(!fs::exists(pa), "close() removes the folder");
    }
    fs::remove_all(root, ec);
    std::printf(g_fail == 0 ? "all passed\n" : "%d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
