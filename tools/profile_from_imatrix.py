"""tools/profile_from_imatrix.py - an expert profile (`data/expert-profiles/<model>.bin`, make_profile.py's format) from a
llama.cpp importance matrix.

An importance matrix in GGUF form (llama-imatrix's output) holds, for each layer's expert matrices, a `.counts` tensor:
one number an expert, the calibration tokens routed to it.  A layer that runs several times a token (Inkling 2 Mini's
looped layers) holds every pass's routing in its one tensor, so the counts are the uses a token.  This tool ranks every
(layer, expert) pair by its count, the most routed first (equal counts by layer, then expert), and writes the profile
the engine fills its VRAM from (`--expert-profile`; the one in data/expert-profiles is built into the programs).

    python tools/profile_from_imatrix.py IMATRIX.gguf --out data/expert-profiles/inkling-2-mini.bin

Inkling 2 Mini's shipped profile is ggml-org/Inkling-2-Mini-GGUF's imatrix.gguf (302,080 calibration tokens).
"""
import argparse
import re
import struct
from pathlib import Path

MAGIC, VERSION = b"STRP", 1
SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
COUNTS = re.compile(r"blk\.(\d+)\.ffn_(gate_up|gate|up)_exps\.weight\.counts$")


def read_counts(path):
    """{layer: [count per expert]} from the matrix's expert `.counts` tensors (F32)."""
    with open(path, "rb") as f:
        def take(fmt):
            n = struct.calcsize(fmt)
            v = struct.unpack("<" + fmt, f.read(n))
            return v[0] if len(v) == 1 else v

        def skip(t):
            if t in SCALAR:
                f.seek(struct.calcsize(SCALAR[t]), 1)
            elif t == 8:
                f.seek(take("Q"), 1)
            elif t == 9:
                et, n = take("I"), take("Q")
                if et in SCALAR:
                    f.seek(n * struct.calcsize(SCALAR[et]), 1)
                else:
                    for _ in range(n):
                        skip(et)
            else:
                raise SystemExit(f"{path}: metadata value type {t}")

        if f.read(4) != b"GGUF":
            raise SystemExit(f"{path}: not a GGUF file")
        take("I")
        n_tensors, n_kv = take("Q"), take("Q")
        align = 32
        for _ in range(n_kv):
            key = f.read(take("Q")).decode("utf-8", "replace")
            t = take("I")
            if key == "general.alignment" and t == 4:
                align = take("I")
            else:
                skip(t)
        want = {}
        for _ in range(n_tensors):
            name = f.read(take("Q")).decode("utf-8", "replace")
            shape = [take("Q") for _ in range(take("I"))]
            ttype, off = take("I"), take("Q")
            m = COUNTS.match(name)
            if m and ttype == 0:
                n = 1
                for s in shape:
                    n *= s
                # a layer with separate gate and up matrices routes the same experts to both: one of them is enough
                want.setdefault(int(m.group(1)), (off, n))
        data = (f.tell() + align - 1) // align * align
        counts = {}
        for layer, (off, n) in want.items():
            f.seek(data + off)
            counts[layer] = list(struct.unpack(f"<{n}f", f.read(4 * n)))
    return counts


def write_profile(path, ranked, n_layers, n_expert):
    table = [[-1] * n_expert for _ in range(n_layers)]
    for slot, (layer, e) in enumerate(ranked):
        table[layer][e] = slot
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<5I", VERSION, n_layers, n_expert, len(ranked), len(ranked)))
        for layer, e in ranked:
            f.write(struct.pack("<HH", layer, e))
        for layer in range(n_layers):
            f.write(struct.pack("<%di" % n_expert, *table[layer]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("imatrix", help="a llama.cpp importance matrix (GGUF)")
    ap.add_argument("--out", required=True, help="the profile to write")
    a = ap.parse_args()

    counts = read_counts(a.imatrix)
    if not counts:
        raise SystemExit(f"{a.imatrix}: no expert .counts tensors (an importance matrix of a model without experts?)")
    n_layers = max(counts) + 1
    n_expert = len(next(iter(counts.values())))
    if sorted(counts) != list(range(n_layers)) or any(len(c) != n_expert for c in counts.values()):
        raise SystemExit(f"{a.imatrix}: the layers' expert counts do not form one {n_layers} x {n_expert} table")
    pairs = [(layer, e) for layer in range(n_layers) for e in range(n_expert)]
    ranked = sorted(pairs, key=lambda p: (-counts[p[0]][p[1]], p[0], p[1]))
    write_profile(a.out, ranked, n_layers, n_expert)
    total = sum(sum(c) for c in counts.values())
    half = sum(counts[layer][e] for layer, e in ranked[:len(ranked) // 2])
    print(f"wrote {a.out}: {n_layers} x {n_expert}, {len(ranked)} ranked pairs; the top half holds "
          f"{100.0 * half / total:.1f}% of the routing")


if __name__ == "__main__":
    main()
