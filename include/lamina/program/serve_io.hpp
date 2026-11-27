// include/lamina/program/serve_io.hpp - the --serve loop's line protocol, wherever it travels.
//
// The engine binary (`lamina-engine --serve`) reads requests on stdin and writes its replies on stdout. lamina
// runs the same loop on a thread of its own process, with in-process channels set here before lamina::program::run.
#pragma once

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace lamina::program {

struct ServeIo {
    /// the next line, without its line end; false once the input has ended
    std::function<bool(std::string&)> read_line;
    /// text of whole lines, written at once
    std::function<void(std::string_view)> write;
    /// in-process: the serve loop ends the input (end_input) and joins its reader thread before it returns (the
    /// reader's state is the loop's), and leaves Ctrl+C to the host process; stdin's reader ends with the process
    bool in_process = false;
    std::function<void()> end_input;
};

/// the channels the next run() uses (default: stdin and stdout)
void set_serve_io(ServeIo io);
const ServeIo& serve_io();
/// printf through serve_io().write
void serve_print(const char* fmt, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 1, 2)))
#endif
    ;
void serve_vprint(const char* fmt, va_list ap);
/// A warning for whoever runs the engine, one line: on stderr, and with --serve (`serving`) also a WARN line, which
/// lamina shows in its window
void serve_warn(bool serving, const std::string& text);

/// The serve loop's input reader at the loop's end: in-process the reader holds the loop's state, so its input ends
/// and it is joined before that state goes; stdin's reader ends with the process.
struct ReaderEnd {
    std::thread& t;
    const ServeIo& io;
    ~ReaderEnd() {
        if (!io.in_process) {
            t.detach();
            return;
        }
        if (io.end_input) io.end_input();
        t.join();
    }
};

/// A picture of a GENI request's embeddings file: its rows (n = nx x ny) and where they start in the rows read.
struct VisionImage {
    int64_t n, nx, ny;
    size_t off;
};
/// Reads the image encoder's file (one or more records: int32 'SVE1', n, nx, ny, n_embd, then n x n_embd floats, a
/// picture each, in prompt order), appending the rows to `rows` and the pictures to `imgs`: "" or the error.
std::string read_vision_records(const std::string& path, int64_t n_embd, std::vector<float>& rows,
                                std::vector<VisionImage>& imgs);
/// What the conversation cache compares: a picture is its grid and the server's key for it (`key`), or else its
/// embeddings (FNV-1a, with the seed and byte order of upstream's conversation cache).
uint64_t vision_image_key(const VisionImage& im, const uint64_t* key, const float* rows, int64_t n_embd);
/// A request's pictures over its cells (a model whose picture rows take one cell each): the image encoder's rows in
/// prompt order and each prompt cell's row (-1: text)
struct ImageCells {
    std::vector<float> rows;
    std::vector<int32_t> cell_row;
    int32_t row(int64_t p) const { return p >= 0 && p < (int64_t) cell_row.size() ? cell_row[(size_t) p] : -1; }
};
/// Reads the image encoder's file (read_vision_records) and lays the pictures over the prompt's runs of `pad`, a row a
/// cell; each such cell's id becomes negative, from the picture's key (`keys`: the server's, img_keys=, else the rows'
/// hash) and its row.  "" or the error.
std::string image_cells(const std::string& path, std::vector<int32_t>& ids, int32_t pad, int64_t n_embd,
                        const std::vector<uint64_t>& keys, ImageCells& r);

}   // namespace lamina::program
