// A picture's or video's file as a request names it (src/server/vision.hpp, media_file_path): paths and file:// URLs,
// network shares (read on Windows as \\host\share\x, refused elsewhere), Windows' device paths refused in every
// spelling. No file is opened.  (CPU only.)
#include "vision.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>

using lamina::device_path;
using lamina::media_file_path;
using lamina::share_path;

static int fails = 0;
#define CHECK(c)                                                                                                       \
    do {                                                                                                               \
        if (!(c)) {                                                                                                    \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);                                          \
            ++fails;                                                                                                   \
        }                                                                                                              \
    } while (0)

namespace {

/// the path media_file_path reads, or "refused: <its message>"
std::string path_of(const std::string& source) {
    try {
        return media_file_path(source, "a video");
    } catch (const std::invalid_argument& e) {
        return std::string("refused: ") + e.what();
    }
}

bool refused(const std::string& source, const char* words) {
    const std::string p = path_of(source);
    return p.rfind("refused: ", 0) == 0 && p.find(words) != std::string::npos;
}

}   // namespace

int main() {
    // paths as they are, file:// URLs as urllib's url2pathname reads them
    CHECK(path_of("clips/a b.mp4") == "clips/a b.mp4");
#ifdef _WIN32
    CHECK(path_of("E:/Films/x.mkv") == "E:/Films/x.mkv");
    CHECK(path_of("file:///E:/Short%20Films/Duet%20(2014)/x.mkv") == "E:/Short Films/Duet (2014)/x.mkv");
    CHECK(path_of("file://localhost/E:/x.mkv") == "E:/x.mkv");
    CHECK(path_of("file://E:/x.mkv") == "E:/x.mkv");   // a drive for a host: a client's spelling of file:///E:/
    CHECK(path_of("file:///E:/Egy%20ember,%20aki%20megeszi%20mag%C3%A1t/x.mkv") ==
          "E:/Egy ember, aki megeszi mag\xC3\xA1t/x.mkv");
#else
    CHECK(path_of("/home/u/x.mkv") == "/home/u/x.mkv");
    CHECK(path_of("file:///home/u/a%20b.mkv") == "/home/u/a b.mkv");
    CHECK(path_of("file://localhost/home/u/x.mkv") == "/home/u/x.mkv");
#endif

    // a network share's file, in each spelling
    for (const char* share : {"\\\\pc\\Films\\x.mkv", "//pc/Films/x.mkv", "file://pc/Films/x.mkv",
                              "file:////pc/Films/x.mkv", "file:///%2F%2Fpc/Films/x.mkv", "\\\\\\pc\\Films\\x.mkv"}) {
#ifdef _WIN32
        CHECK(path_of(share) == "\\\\pc\\Films\\x.mkv");
#else
        CHECK(refused(share, "network share"));
#endif
    }
#ifdef _WIN32
    CHECK(path_of("file://pc/Short%20Films/a%20b.mkv") == "\\\\pc\\Short Films\\a b.mkv");
    CHECK(path_of("\\\\192.168.1.20\\E$\\x.mkv") == "\\\\192.168.1.20\\E$\\x.mkv");
#endif

    // Windows' device and object namespaces name no file: refused everywhere, however spelled
    for (const char* device :
         {"\\\\.\\pipe\\x", "//./pipe/x", "\\\\?\\C:\\x.mkv", "\\\\?\\UNC\\pc\\Films\\x.mkv", "\\??\\C:\\x.mkv",
          "\\\\.", "file://./pipe/x", "file://?/C:/x.mkv", "file:///%2F%2F.%2Fpipe%2Fx", "\\\\\\.\\pipe\\x"})
        CHECK(refused(device, "names a device"));
    CHECK(!share_path("\\\\") && !share_path("C:\\x") && !share_path("/x") && share_path("//pc/Films"));
    CHECK(device_path("\\??\\UNC\\pc\\x") && !device_path("\\\\pc\\x") && !device_path("C:/x"));

    if (fails) std::fprintf(stderr, "%d failed\n", fails);
    else std::fprintf(stderr, "media_path_test: ok\n");
    return fails ? 1 : 0;
}
