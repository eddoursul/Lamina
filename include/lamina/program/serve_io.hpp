// include/lamina/program/serve_io.hpp - the --serve loop's line protocol, wherever it travels.
//
// The engine binary (`lamina-engine --serve`) reads requests on stdin and writes its replies on stdout. lamina
// runs the same loop on a thread of its own process, with in-process channels set here before lamina::program::run.
#pragma once

#include "lamina/platform/console_quit.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
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

/// The serve loop's input.  A thread of its own reads the lines (serve_io()'s, a line's CR dropped), so a STOP line
/// reaches a request that is still running: it sets `stop` at once (a request clears it as it starts and looks at it
/// between its steps), and every other line is queued for the loop.  Out of the process and with `console_quit` (a
/// loop with exit work), Ctrl+C in the console stands for a QUIT ahead of every line queued, `stop` set too; in-process
/// the host handles it.  At its end the input is ended and the reader joined in-process; stdin's reader ends with the
/// process.
class ServeInput {
    struct Queue {
        std::mutex mu;
        std::condition_variable cv;
        std::deque<std::string> lines;
        bool eof = false;
        std::atomic<bool> stop{false};
    };
    std::shared_ptr<Queue> q_;   // the reader's too: a reader left to the process's end reads no freed memory

public:
    explicit ServeInput(bool console_quit);
    ~ServeInput();
    ServeInput(const ServeInput&) = delete;
    ServeInput& operator=(const ServeInput&) = delete;

    std::atomic<bool>& stop;   ///< set by a STOP line (and a Ctrl+C)

    /// The next line, waited for; false once the input has ended with no line queued
    bool next(std::string& line);
    /// The next line where one is queued, else false at once
    bool poll(std::string& line);
    /// Whether a line is queued or the input has ended
    bool ready();
    /// Waits until ready() or `until`, and says which
    bool wait_until(std::chrono::steady_clock::time_point until);
    /// A QUIT line's work: the input ends there, the lines queued behind it dropped
    void quit();

    /// The queue under its lock (`eof`: the input has ended), for a loop that takes lines out of turn; `wait_for` waits
    /// there up to `d` for the reader's next line, the lock let go meanwhile
    struct Held {
        std::deque<std::string>& lines;
        const bool& eof;
        std::unique_lock<std::mutex>& lk;
        std::condition_variable& cv;
        void wait_for(std::chrono::milliseconds d) { cv.wait_for(lk, d); }
    };
    template <class F> decltype(auto) locked(F&& f) {
        std::unique_lock<std::mutex> lk(q_->mu);
        Held held{q_->lines, q_->eof, lk, q_->cv};
        return f(held);
    }

private:
    std::thread reader_;
    std::optional<platform::ConsoleQuit> console_quit_;
};

/// A picture of a GENI request's embeddings file: its rows (n = nx x ny) and where they start in the rows read.
struct VisionImage {
    int64_t n, nx, ny;
    size_t off;
};
/// Reads the image encoder's file (one or more records: int32 'SVE1', n, nx, ny, n_embd, then n x n_embd floats, a
/// picture each, in prompt order), or a list of such files ("SVL1", then a path a line, in prompt order: the server's
/// for a request, its encoder's files in place), appending the rows to `rows` and the pictures to `imgs`: "" or the
/// error.
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
    /// each prompt cell's M-RoPE position [cell][3] (t, h, w): text counts up by one; a record's cell (p, p + row, p +
    /// column) of its grid, the text after it from p + max(nx, ny); `next`: the first cell after the prompt's
    std::vector<int32_t> mrope;
    int64_t next = 0;
    int32_t row(int64_t p) const { return p >= 0 && p < (int64_t) cell_row.size() ? cell_row[(size_t) p] : -1; }
};
/// Reads the image encoder's file (read_vision_records) and lays the pictures over the prompt's runs of a pad (`pads`:
/// a picture's, a video group's), a record a run, a row a cell; each such cell's id becomes negative, from the
/// record's key (`keys`: the server's, img_keys=, else the rows' hash) and its row.  "" or the error.
std::string image_cells(const std::string& path, std::vector<int32_t>& ids, const std::vector<int32_t>& pads,
                        int64_t n_embd, const std::vector<uint64_t>& keys, ImageCells& r);

}   // namespace lamina::program
