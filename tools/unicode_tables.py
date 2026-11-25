"""The code point tables of the server's text layer: src/server/unicode_tables.inc, from the Unicode Character Database.

Each code point's general category as one of eight classes (\\p{L} letter, \\p{M} mark, \\p{N} number, \\p{P}
punctuation, \\p{S} symbol, \\p{Z} separator, control for Cc, Cf, Co and Cs, and undefined for a code point the
database does not list), the White_Space property, and the simple lower and upper case mappings (UnicodeData.txt's
fields 13 and 12).  src/server/unicode.cpp reads them; the pre-tokenizer (src/server/pretokenize.cpp) splits a text
by these classes, and the Python-compatible string tests of src/server use them too.

    python tools/unicode_tables.py [--ucd DIR] [--out src/server/unicode_tables.inc]

DIR holds Unicode 15.1.0's UnicodeData.txt and PropList.txt; without it they are downloaded from unicode.org and
checked by their SHA-256.  15.1 is the version of the tables llama.cpp's tokenizer splits text with, so a pre-tokenizer
piece is the same in both (pretokenize_parity checks every code point against llama.cpp's table); the server's Python
string tests read Python 3.12's Unicode 15.0 from them by leaving out what 15.1 added (after_unicode_15).
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys
import urllib.request

VERSION = "15.1.0"
FILES = {   # name -> SHA-256 of Unicode 15.1.0's file
    "UnicodeData.txt": "2fc713e6a31a87c4850a37fe2caffa4218180fadb5de86b43a143ddb4581fb86",
    "PropList.txt": "05672956317b6296bc2ec3d6cef1f6452b57ff4f2efc6dc55b0a19277d5fcfd1",
}
MAX_CP = 0x110000

# the classes, as the bits of unicode.hpp's kCp* (one per code point)
UNDEFINED, NUMBER, LETTER, SEPARATOR, MARK, PUNCTUATION, SYMBOL, CONTROL = (1 << i for i in range(8))
CLASS_NAMES = {UNDEFINED: "Undefined", NUMBER: "Number", LETTER: "Letter", SEPARATOR: "Separator", MARK: "Mark",
               PUNCTUATION: "Punctuation", SYMBOL: "Symbol", CONTROL: "Control"}
CLASS_OF = {"C": CONTROL, "L": LETTER, "M": MARK, "N": NUMBER, "P": PUNCTUATION, "S": SYMBOL, "Z": SEPARATOR}


def read_files(ucd: str | None) -> dict[str, str]:
    out = {}
    for name, sha in FILES.items():
        if ucd:
            data = pathlib.Path(ucd, name).read_bytes()
        else:
            url = f"https://www.unicode.org/Public/{VERSION}/ucd/{name}"
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
        got = hashlib.sha256(data).hexdigest()
        if got != sha:
            sys.exit(f"{name}: SHA-256 {got}, not Unicode {VERSION}'s {sha}")
        out[name] = data.decode("utf-8")
    return out


def parse_unicode_data(text: str):
    """the class of every code point, and the (code point, mapping) pairs of the lower and upper case mappings"""
    classes = [UNDEFINED] * MAX_CP
    lower, upper = [], []
    first = None   # a range's <..., First> line, waiting for its <..., Last>
    for line in text.splitlines():
        f = line.split(";")
        cp, name, cat = int(f[0], 16), f[1], f[2]
        if cat[0] not in CLASS_OF or cat == "Cn":
            sys.exit(f"U+{cp:04X}: unexpected category {cat}")
        cls = CLASS_OF[cat[0]]
        if name.endswith(", First>"):
            first = (cp, cls)
            continue
        if name.endswith(", Last>"):
            if first is None or first[1] != cls:
                sys.exit(f"U+{cp:04X}: a range's last line without its first")
            for c in range(first[0], cp + 1):
                classes[c] = cls
            first = None
            continue
        classes[cp] = cls
        if f[13]:
            lower.append((cp, int(f[13], 16)))
        if f[12]:
            upper.append((cp, int(f[12], 16)))
    return classes, lower, upper


def parse_white_space(text: str) -> list[int]:
    out = []
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        cps, prop = (x.strip() for x in line.split(";"))
        if prop != "White_Space":
            continue
        a, _, b = cps.partition("..")
        out.extend(range(int(a, 16), int(b or a, 16) + 1))
    return sorted(out)


def rows(items: list[str], per_line: int) -> str:
    return "\n".join("    " + " ".join(items[i:i + per_line]) for i in range(0, len(items), per_line))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ucd", help="a folder with UnicodeData.txt and PropList.txt (default: download them)")
    ap.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parents[1] / "src/server/unicode_tables.inc"))
    a = ap.parse_args()
    files = read_files(a.ucd)
    classes, lower, upper = parse_unicode_data(files["UnicodeData.txt"])
    white = parse_white_space(files["PropList.txt"])

    runs = [(0, classes[0])]
    for cp in range(1, MAX_CP):
        if classes[cp] != runs[-1][1]:
            runs.append((cp, classes[cp]))
    text = f"""// src/server/unicode_tables.inc - Unicode {VERSION}'s code point tables (UnicodeData.txt, PropList.txt),
// written by tools/unicode_tables.py; src/server/unicode.cpp includes it.

// the general categories as runs of code points of one class: {{the run's first code point, its class}}, each to the
// next run's first (the last to 0x10FFFF)
constexpr CategoryRun kCategoryRuns[] = {{
{rows([f"{{0x{cp:06X}, kCp{CLASS_NAMES[c]}}}," for cp, c in runs], 4)}
}};

// White_Space
constexpr uint32_t kWhiteSpace[] = {{
{rows([f"0x{cp:04X}," for cp in white], 12)}
}};

// the simple case mappings, {{code point, its mapping}} by code point
constexpr CaseMapping kLowerCase[] = {{
{rows([f"{{0x{a:05X}, 0x{b:05X}}}," for a, b in lower], 5)}
}};
constexpr CaseMapping kUpperCase[] = {{
{rows([f"{{0x{a:05X}, 0x{b:05X}}}," for a, b in upper], 5)}
}};
"""
    pathlib.Path(a.out).write_text(text, encoding="utf-8", newline="\n")
    print(f"{a.out}: {len(runs)} category runs, {len(white)} White_Space, {len(lower)} lower and {len(upper)} upper "
          f"case mappings (Unicode {VERSION})")


if __name__ == "__main__":
    main()
