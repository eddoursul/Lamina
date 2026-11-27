#include "log.hpp"

#include <cstdio>
#include <mutex>

namespace lamina {
namespace {

std::mutex& mu() {
    static std::mutex m;
    return m;
}

std::FILE* g_err = nullptr;

void write(FILE* f, std::string_view text) {
    std::lock_guard<std::mutex> lk(mu());
    std::fwrite(text.data(), 1, text.size(), f);
    if (text.empty() || text.back() != '\n') std::fputc('\n', f);
    std::fflush(f);
}

thread_local SayRedirect* t_redirect = nullptr;

}   // namespace

SayRedirect::SayRedirect(std::function<void(std::string_view, bool)> sink)
    : sink_(std::move(sink)), outer_(t_redirect) {
    t_redirect = this;
}
SayRedirect::~SayRedirect() { t_redirect = outer_; }

void say(std::string_view text) {
    if (t_redirect != nullptr) {
        if (t_redirect->sink()) t_redirect->sink()(text, false);
        return;
    }
    write(stdout, text);
}
void say_err(std::string_view text) {
    if (t_redirect != nullptr) {
        if (t_redirect->sink()) t_redirect->sink()(text, true);
        return;
    }
    write(g_err != nullptr ? g_err : stderr, text);
}

void set_err_stream(std::FILE* f) {
    std::lock_guard<std::mutex> lk(mu());
    g_err = f;
}

}   // namespace lamina
