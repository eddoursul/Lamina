"""tools/mtp_rt.py - the MTP draft layer as the one GGUF file the engine reads (`--mtp FILE`).

    python tools/mtp_rt.py --rt mtp/rt-distilled --out mtp/Lamina-Qwen3.8-Flash-Next-MTP.gguf
    python tools/mtp_rt.py --gguf mtp-q2_0.gguf --draft-vocab data/draft_vocab.bin --out layer.gguf
    python tools/mtp_rt.py --gguf layer.gguf --draft-vocab data/draft_vocab_fr.bin --out layer-fr.gguf

Sources:
  --rt DIR     a folder of the earlier runtime layout (dense.txt with dense.bin, experts.bin, draft_vocab.bin;
               experts.txt with experts in GGUF blocks), as the draft layer's training writes it: its bytes
               as they are, the Q2_0 experts' blobs put back into GGUF blocks; the file is checked against the folder
  --gguf FILE  tools/mtp_pack.py's file of the checkpoint's tensors (BF16), the dense ones put in the engine's forms
               below; or a file this tool wrote, copied (with --draft-vocab: another draft vocabulary)
  --draft-vocab FILE   the draft head's token ids (int32, tools/draft_vocab.py) in place of the source's

The file: general.architecture "qwen4exp-mtp", lamina.mtp.version 1 (the engine refuses any other), and
  the dense tensors by the names the engine looks up (self_attn.q_proj.weight, ...): the large projections Q8_0
              (ggml's reference rounding), the hyper-connection and router weights BF16, the RMSNorm weights F32 with
              the Gemma "+1" applied (vLLM's GemmaRMSNorm scales by 1 + w); the engine puts them in VRAM in this
              order, each at a 256-byte offset
  mlp.experts.gate_up_proj [512, 1280, 2560] (gate rows 0..639, up rows 640..1279) and mlp.experts.down_proj
              [512, 2560, 640], quantized along the input: Q2_0 the engine regroups at load into its blob layout
              (blob_of below; include/lamina/kernels/cpu/expert.hpp), other formats it reads as GGUF blocks with the
              native expert kernels
  draft_vocab I32, the token ids the draft head proposes from (none: the whole head)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import add_gguf_py  # noqa: E402
add_gguf_py()
import gguf  # noqa: E402  (the pinned llama.cpp gguf-py, which knows Q2_0 = type 42)

T = gguf.GGMLQuantizationType
H, FF = 2560, 640
BLOB = 3 * (H * FF * 18 // 64)   # a Q2_0 expert in the engine's blob layout
ARCH, VERSION = "qwen4exp-mtp", 1
GU, DN, VOCAB = "mlp.experts.gate_up_proj", "mlp.experts.down_proj", "draft_vocab"
KINDS = {"q8_0": T.Q8_0, "bf16": T.BF16, "f32": T.F32}
Q8 = {"fc_embedding.weight", "fc_hidden.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
      "self_attn.v_proj.weight", "self_attn.o_proj.weight", "self_attn.indexer.index_qk_proj.weight",
      "mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight", "mlp.shared_expert.down_proj.weight"}


def q8_0(x: np.ndarray) -> np.ndarray:
    """ggml quantize_row_q8_0_ref over the rows of a 2-D float32 array: 32-value blocks, fp16 d = amax / 127."""
    b = x.reshape(-1, 32).astype(np.float32)
    d = np.abs(b).max(axis=1) / 127.0
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0).astype(np.float32)
    v = b * inv[:, None]
    q = (np.sign(v) * np.floor(np.abs(v) + 0.5)).astype(np.int8)
    out = np.empty((b.shape[0], 34), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.reshape(x.shape[0], -1)


def blob_of(gu: np.ndarray, dn: np.ndarray) -> np.ndarray:
    """Q2_0 experts as GGUF blocks, gu (E, 1280, 720) and dn (E, 2560, 180), -> (E, BLOB) blobs: gate/up rows
    interleaved (2r = gate r, 2r + 1 = up r), then the down rows; each block's 16 code bytes in one plane, its fp16
    scale in another: [gate/up codes | down codes | gate/up scales | down scales], what the engine does at load."""
    e = gu.shape[0]
    gub = gu.reshape(e, 2 * FF, H // 64, 18)
    inter = np.empty_like(gub)
    inter[:, 0::2] = gub[:, :FF]
    inter[:, 1::2] = gub[:, FF:]
    dnb = dn.reshape(e, H, FF // 64, 18)
    parts = [inter[..., 2:], dnb[..., 2:], inter[..., :2], dnb[..., :2]]
    return np.concatenate([p.reshape(e, -1) for p in parts], axis=1)


def fused_of(blobs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """blob_of's inverse: (E, BLOB) -> gu (E, 1280, 720), dn (E, 2560, 180)."""
    e = blobs.shape[0]
    n_gc, n_dc, n_gs = 2 * FF * H // 4, H * FF // 4, 2 * FF * (H // 64) * 2
    gc = blobs[:, :n_gc].reshape(e, 2 * FF, H // 64, 16)
    dc = blobs[:, n_gc:n_gc + n_dc].reshape(e, H, FF // 64, 16)
    gs = blobs[:, n_gc + n_dc:n_gc + n_dc + n_gs].reshape(e, 2 * FF, H // 64, 2)
    ds = blobs[:, n_gc + n_dc + n_gs:].reshape(e, H, FF // 64, 2)
    inter = np.concatenate([gs, gc], axis=3)
    gu = np.empty_like(inter)
    gu[:, :FF] = inter[:, 0::2]
    gu[:, FF:] = inter[:, 1::2]
    return gu.reshape(e, 2 * FF, -1), np.concatenate([ds, dc], axis=3).reshape(e, H, -1)


def row_bytes(t: T, n: int) -> int:
    block, size = gguf.GGML_QUANT_SIZES[t]
    return n // block * size


class Layer:
    """The file's tensors as bytes: (name, GGUF type, uint8 array of the byte shape, outermost first)."""

    def __init__(self):
        self.dense: list[tuple[str, T, np.ndarray]] = []
        self.experts: list[tuple[str, T, np.ndarray]] = []
        self.vocab: np.ndarray | None = None
        self.meta: dict[str, str] = {}

    def add_dense(self, name: str, t: T, raw: np.ndarray, rows: int):
        self.dense.append((name, t, raw.reshape(-1) if rows == 1 else raw.reshape(rows, -1)))


def from_rt(rt: Path) -> Layer:
    lay = Layer()
    blob = np.fromfile(rt / "dense.bin", dtype=np.uint8)
    for line in (rt / "dense.txt").read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        name, kind, rows, cols, off, n = line.split()
        rows, cols, off, n = int(rows), int(cols), int(off), int(n)
        t = KINDS[kind]
        if n != rows * row_bytes(t, cols) or off + n > blob.size:
            sys.exit(f"{rt}/dense.txt: {line!r} is not a tensor of dense.bin")
        lay.add_dense(name, t, blob[off:off + n], rows)
    raw = np.fromfile(rt / "experts.bin", dtype=np.uint8)
    fmt = rt / "experts.txt"
    if fmt.exists():
        tg, td = (T(int(x)) for x in fmt.read_text(encoding="utf-8").split())
        gb, db = 2 * FF * row_bytes(tg, H), H * row_bytes(td, FF)
        raw = raw.reshape(-1, gb + db)
        gu, dn = raw[:, :gb].reshape(-1, 2 * FF, gb // (2 * FF)), raw[:, gb:].reshape(-1, H, db // H)
    else:
        tg = td = T.Q2_0
        gu, dn = fused_of(raw.reshape(-1, BLOB))
    lay.experts = [(GU, tg, gu), (DN, td, dn)]
    if (rt / "draft_vocab.bin").exists():
        lay.vocab = np.fromfile(rt / "draft_vocab.bin", dtype=np.int32)
    lay.meta["lamina.mtp.source"] = f"the runtime folder {rt.name}"
    return lay


def from_gguf(path: Path) -> Layer:
    """The source's tensors copied out of gguf-py's map of the file, which closes with the reader (Windows would keep
    the file open while a view of it lives)."""
    r = gguf.GGUFReader(path)
    lay = Layer()
    as_bytes = lambda d: d.view(np.uint8).reshape(*d.shape[:-1], -1)
    if r.get_field("lamina.mtp.version") is not None:   # a file of this tool's: its tensors as they are
        for t in r.tensors:
            data = np.array(t.data)
            if t.name == VOCAB:
                lay.vocab = data.astype(np.int32)
            elif t.name in (GU, DN):
                lay.experts.append((t.name, t.tensor_type, as_bytes(data)))
            else:
                lay.add_dense(t.name, t.tensor_type, as_bytes(data), int(t.shape[1]) if len(t.shape) > 1 else 1)
        for k in ("lamina.mtp.source",):
            if (f := r.get_field(k)) is not None:
                lay.meta[k] = f.contents()
        return lay
    for t in r.tensors:   # tools/mtp_pack.py's: mtp.layers.0.<name> or mtp.<name>
        short = t.name[len("mtp."):]
        short = short[len("layers.0."):] if short.startswith("layers.0.") else short
        data = np.array(t.data)
        if short in (GU, DN):
            lay.experts.append((short, t.tensor_type, data))
        elif t.tensor_type == T.F32:   # the raw GemmaRMSNorm w -> 1 + w
            lay.add_dense(short, T.F32, (data.astype(np.float32) + 1.0).reshape(-1).view(np.uint8), 1)
        else:   # BF16
            u16 = data.view(np.uint16).reshape(data.shape[0], -1)
            if short in Q8:
                lay.add_dense(short, T.Q8_0, q8_0((u16.astype(np.uint32) << 16).view(np.float32)), u16.shape[0])
            else:
                lay.add_dense(short, T.BF16, u16.view(np.uint8), u16.shape[0])
    lay.meta["lamina.mtp.source"] = f"tools/mtp_pack.py's {path.name}"
    return lay


def expert_format(lay: Layer) -> str:
    return "/".join(t.name for _, t, _ in lay.experts)


def write(lay: Layer, out: Path, name: str):
    lay.experts.sort(key=lambda x: x[0] != GU)
    if [x[0] for x in lay.experts] != [GU, DN]:
        sys.exit(f"the source has no {GU} and {DN}")
    w = gguf.GGUFWriter(str(out), ARCH)
    w.add_name(name)
    w.add_license("other")
    w.add_license_name("qwen-community-1.0")   # derived from Qwen3.8-Flash-Next's weights
    w.add_uint32("general.base_model.count", 1)
    w.add_string("general.base_model.0.name", "Qwen3.8 Flash Next")
    w.add_string("general.base_model.0.organization", "Qwen")
    w.add_string("general.base_model.0.repo_url", "https://huggingface.co/Qwen/Qwen3.8-Flash-Next")
    w.add_uint32("lamina.mtp.version", VERSION)
    w.add_string("lamina.mtp.expert_format", expert_format(lay))
    for k, v in lay.meta.items():
        w.add_string(k, v)
    for n, t, raw in lay.dense + lay.experts:
        w.add_tensor(n, np.ascontiguousarray(raw), raw_shape=raw.shape, raw_dtype=t)
    if lay.vocab is not None:
        w.add_tensor(VOCAB, lay.vocab.astype(np.int32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def check_rt(out: Path, rt: Path):
    """The file against the folder it came from: dense.bin's tensors, experts.bin by blob_of, draft_vocab.bin."""
    r = gguf.GGUFReader(out)
    tens = {t.name: t for t in r.tensors}
    blob = np.fromfile(rt / "dense.bin", dtype=np.uint8)
    for line in filter(str.strip, (rt / "dense.txt").read_text(encoding="utf-8").split("\n")):
        name, _, _, _, off, n = line.split()
        if not np.array_equal(np.asarray(tens[name].data).view(np.uint8).reshape(-1), blob[int(off):int(off) + int(n)]):
            sys.exit(f"{out}: {name} differs from {rt}/dense.bin")
    gu, dn = np.asarray(tens[GU].data), np.asarray(tens[DN].data)
    if (rt / "experts.txt").exists():
        got = np.concatenate([gu.reshape(gu.shape[0], -1), dn.reshape(dn.shape[0], -1)], axis=1)
    else:
        got = blob_of(gu, dn)
    if not np.array_equal(got.reshape(-1), np.fromfile(rt / "experts.bin", dtype=np.uint8)):
        sys.exit(f"{out}: the experts differ from {rt}/experts.bin")
    if (rt / "draft_vocab.bin").exists() and VOCAB in tens and not np.array_equal(
            np.asarray(tens[VOCAB].data), np.fromfile(rt / "draft_vocab.bin", dtype=np.int32)):
        sys.exit(f"{out}: draft_vocab differs from {rt}/draft_vocab.bin")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 epilog=__doc__.split("\n\n", 1)[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--rt", help="a runtime folder of the earlier layout")
    src.add_argument("--gguf", help="tools/mtp_pack.py's file, or a file this tool wrote")
    ap.add_argument("--draft-vocab", help="the draft head's token ids (int32) in place of the source's")
    ap.add_argument("--name", default="Lamina's draft layer for Qwen3.8-Flash-Next", help="general.name")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if a.gguf and out.resolve() == Path(a.gguf).resolve():
        sys.exit("--out is the source file: write another and replace it after")
    lay = from_rt(Path(a.rt)) if a.rt else from_gguf(Path(a.gguf))
    if a.draft_vocab:
        lay.vocab = np.fromfile(a.draft_vocab, dtype=np.int32)
    write(lay, out, a.name)
    if a.rt and not a.draft_vocab:
        check_rt(out, Path(a.rt))
        print(f"checked: {out.name} holds {a.rt}'s bytes")
    n_dense = sum(raw.nbytes for _, _, raw in lay.dense)
    print(f"{out}: {out.stat().st_size:,} bytes; {len(lay.dense)} dense tensors ({n_dense:,} bytes), experts "
          f"{expert_format(lay)}, draft vocabulary "
          + (f"{lay.vocab.size:,} tokens" if lay.vocab is not None else "none (the whole head)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
