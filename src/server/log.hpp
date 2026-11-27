// src/server/log.hpp - the server's console lines (the Python server's print(..., flush=True)).
#pragma once

#include <cstdio>
#include <functional>
#include <string_view>

namespace lamina {

/// one line (or several) on the console, stdout, written whole
void say(std::string_view text);
/// the same on stderr
void say_err(std::string_view text);
/// where say_err writes (the console's stderr, kept when the process's stderr goes to the engine's log)
void set_err_stream(std::FILE* f);

/// While one lives, say() and say_err() on its thread hand their text to `sink` in place of the console (`err`: say_err's)
/// - lamina --check on a model at a URL: its lines with the download's paths in place of the copies' it reads,
/// or a model's lines left out of a table.  They nest: the latest one takes the lines.
class SayRedirect {
public:
    explicit SayRedirect(std::function<void(std::string_view, bool)> sink);
    ~SayRedirect();
    SayRedirect(const SayRedirect&) = delete;
    SayRedirect& operator=(const SayRedirect&) = delete;
    const std::function<void(std::string_view, bool)>& sink() const { return sink_; }

private:
    std::function<void(std::string_view, bool)> sink_;
    SayRedirect* outer_;
};

}  // namespace lamina
