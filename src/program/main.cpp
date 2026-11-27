// src/program/main.cpp - the `lamina-engine` binary: the engine on the command line, --serve on stdin and stdout.
#include "lamina/program/engine.hpp"

#include <cstdio>
#include <string_view>

int main(int argc, char** argv) {
    // UNBUFFERED, BECAUSE THE INTERESTING OUTPUT IS THE OUTPUT BEFORE A CRASH. `stdout` redirected to a pipe or a
    // file is block-buffered, so a program that dies loses every line it had already printed.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    // the program's, before either engine path's parser (no option of theirs takes "-v" for a value)
    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        if (a == "-v" || a == "--version") {
            std::fputs("lamina-engine " LAMINA_VERSION "\n", stdout);
            return 0;
        }
    }
    return lamina::program::run(argc, argv);
}
