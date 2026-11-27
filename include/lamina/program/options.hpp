// include/lamina/program/options.hpp - the engine's options as the help lists them: `lamina-engine --help` the
// common ones, `--help-all` every one, lamina's --help those its engine takes.  The paths' parsers read them.
#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace lamina::program {

/// Where an option is listed besides `lamina-engine --help-all`, which lists every one.
enum OptionList : unsigned {
    kListCommon = 1,   ///< `lamina-engine --help`: the options most runs set
    kListServed = 2,   ///< lamina's --help: an option for the engine a server runs
};

/// An option of the engine, a --gpu card's setting, or (an empty `flag`) the title of the section after it.
struct OptionHelp {
    const char* flag;    ///< "--max-context"; a card's setting "cache=auto|SIZE|N"
    const char* value;   ///< its value's form ("N", "fp16|q8_0"); "" for a switch or a card's setting
    unsigned lists;      ///< OptionList bits
    const char* help;    ///< the help (a section's title)
};

/// Every option the engine's paths take, by section.
const std::vector<OptionHelp>& engine_options();

/// A deprecated spelling the engine still takes: its value's form ("" for a switch) and what takes its place.
struct DeprecatedOption {
    const char* flag;
    const char* value;
    const char* instead;
};

/// The deprecated spellings, as --help-all lists them.
const std::vector<DeprecatedOption>& deprecated_options();

/// The deprecated spelling `flag` is; nullptr for any other option.  lamina names the ones a start uses.
const DeprecatedOption* deprecated_option(const char* flag);

/// The options listed in `lists` (0: every one), each section's title before its first one; then, with `deprecated`,
/// the deprecated spellings.  An option's help starts at column `col`, wrapped at `width`.
std::string options_help(unsigned lists, bool deprecated, size_t width, size_t col);

/// `text` wrapped at `width`: its first line from column `first_col`, the others after `indent` spaces.
std::string wrapped(std::string_view text, size_t indent, size_t width, size_t first_col);

/// An option's line: `left` after two spaces, its help from column `col` (on the next line where `left` reaches it).
std::string help_line(std::string_view left, std::string_view help, size_t width, size_t col);

}   // namespace lamina::program
