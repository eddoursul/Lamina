// program/gpu_cards.hpp: --gpu's cards (the name, then key=value settings) and the units of sizes and times, what each
// accepts and refuses, and the text a card reads back as.  CPU only.
#include "lamina/program/gpu_cards.hpp"

#include <cstdio>
#include <string>

namespace {
int fails = 0;
void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++fails;
    }
}

uint64_t size_of(const char* s) {
    uint64_t b = 0;
    std::string e;
    check(lamina::program::parse_size(s, b, e), std::string("size ") + s + ": " + e);
    return b;
}
bool size_refused(const char* s) {
    uint64_t b = 0;
    std::string e;
    return !lamina::program::parse_size(s, b, e) && !e.empty();
}
double seconds_of(const char* s) {
    double v = 0.0;
    std::string e;
    check(lamina::program::parse_seconds(s, v, e), std::string("time ") + s + ": " + e);
    return v;
}
bool time_refused(const char* s) {
    double v = 0.0;
    std::string e;
    return !lamina::program::parse_seconds(s, v, e) && !e.empty();
}
lamina::program::GpuCard card_of(const char* s) {
    lamina::program::GpuCard c;
    std::string e;
    check(lamina::program::parse_gpu_card(s, c, e), std::string("card ") + s + ": " + e);
    return c;
}
std::string card_refused(const char* s) {
    lamina::program::GpuCard c;
    std::string e;
    check(!lamina::program::parse_gpu_card(s, c, e), std::string("card ") + s + " accepted");
    return e;
}
}   // namespace

int main() {
    using lamina::program::gpu_card_text;
    // sizes: binary units, with or without i and B, any case, fractions; 0 alone
    check(size_of("8G") == 8ull << 30, "8G");
    check(size_of("8GiB") == 8ull << 30 && size_of("8gb") == 8ull << 30 && size_of(" 8 g ") == 8ull << 30,
          "8 GiB forms");
    check(size_of("700M") == 700ull << 20, "700M");
    check(size_of("1.5G") == 1536ull << 20, "1.5G");
    check(size_of("512K") == 512ull << 10 && size_of("2T") == 2ull << 40 && size_of("4096B") == 4096, "K, T, B");
    check(size_of("0") == 0, "0 alone");
    check(size_refused("8") && size_refused("700") && size_refused("-1G") && size_refused("8X") && size_refused("G") &&
              size_refused(""),
          "sizes without a unit, negative or malformed");
    check(lamina::program::size_text(8ull << 30) == "8G" && lamina::program::size_text(700ull << 20) == "700M" &&
              lamina::program::size_text(1536ull << 20) == "1536M" && lamina::program::size_text(4097) == "4097B" &&
              lamina::program::size_text(0) == "0",
          "size_text");
    // times
    check(seconds_of("55us") == 55e-6 && seconds_of("1.5ms") == 1.5e-3 && seconds_of("1s") == 1.0 &&
              seconds_of("0") == 0.0,
          "times");
    check(time_refused("1") && time_refused("55") && time_refused("1h") && time_refused("-1s"), "times refused");
    check(lamina::program::seconds_text(55e-6) == "55us" && lamina::program::seconds_text(1.0) == "1s" &&
              lamina::program::seconds_text(0.25) == "250ms",
          "seconds_text");
    // a card: the name alone, or with its settings
    {
        const auto c = card_of("RTX 5070 Ti");
        check(c.name == "RTX 5070 Ti" && !c.has_cache() && !c.reserve && !c.has_vram_only() && !c.clocks && !c.sleep,
              "a name alone");
    }
    {
        const auto c = card_of(" GPU-f26e9ba6 , clocks=1395-2100/9751, sleep=55us,vram-only=8G ,reserve=512M");
        check(c.name == "GPU-f26e9ba6" && c.clocks == "1395-2100/9751" && c.sleep && *c.sleep == 55e-6 &&
                  c.vram_only == 8ull << 30 && c.reserve == 512ull << 20,
              "a tier's settings");
        check(gpu_card_text(c) == "GPU-f26e9ba6,reserve=512M,vram-only=8G,clocks=1395-2100/9751,sleep=55us",
              "its text: " + gpu_card_text(c));
    }
    check(card_of("0,cache=auto").cache_auto && card_of("0,cache=9000").cache_slots == 9000 &&
              card_of("0,cache=16G").cache_bytes == 16ull << 30,
          "cache: auto, slots, a size");
    check(card_of("auto,vram-only=auto").vram_only_auto && card_of("10:00.0,prefetch=0").prefetch == 0ull,
          "auto names, vram-only=auto, prefetch=0, a bus id");
    check(card_of("RTX 3090,").name == "RTX 3090", "a trailing comma");
    // refused: no name, unknown, repeated or valueless keys, values without units
    check(!card_refused("").empty() && !card_refused(",cache=auto").empty() && !card_refused("cache=auto").empty(),
          "no name");
    check(card_refused("0,speed=fast").find("not a card's setting") != std::string::npos, "an unknown key");
    check(card_refused("0,reserve=1G,reserve=2G").find("twice") != std::string::npos, "a key twice");
    check(!card_refused("0,reserve").empty() && !card_refused("0,reserve=").empty(), "a key without its value");
    check(card_refused("0,reserve=700").find("unit") != std::string::npos, "a size without its unit");
    check(card_refused("0,sleep=55").find("unit") != std::string::npos, "a time without its unit");
    check(!card_refused("0,clocks=fast").empty() && !card_refused("0,vram-only=lots").empty(), "malformed values");
    if (fails == 0) std::printf("gpu_cards_test: all passed\n");
    return fails == 0 ? 0 : 1;
}
