# Setup

What a PC needs: [the README](../README.md#what-you-need). This page covers what setup does, the release binaries, the
config it writes, the GPUs, the engine options worth knowing, Qwen3.6-35B-A3B and Qwen3.8-27B, and Docker.

## Setup

From a clone of this repository:

```
setup.bat --model iq3_s          (Windows)
./setup.sh --model iq3_s         (Linux)
```

`setup.bat` and `setup.sh` make a Python environment, `.venv` (setup takes the standard library only; a build's
CMake and Ninja go into it), and run `setup.py` with the same arguments. The models: `iq3_s` (Qwen3.8-Flash-Next, ISTA-DASLab's GSQ-RCO IQ3_S),
`ud-q4_k_xl` (Qwen3.8-Flash-Next, unsloth's UD-Q4_K_XL), `qwen36-ud-q4_k_xl` (Qwen3.6-35B-A3B, unsloth's UD-Q4_K_XL)
and `qwen38-27b-ud-q6_k_m` (Qwen3.8-27B, unsloth's UD-Q6_K_M). The engine also runs Flash-Next's other GGUF files, most of unsloth's quants and
[Agention's Gyro-L](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF) among them: setup downloads none of those, so copy a
config, name the file after `--native` and give it a port of its own ([COMPARISON.md](COMPARISON.md#gyro-l-and-the-three-files-at-one-build)).
Of unsloth's Dynamic 3.1 quants it refuses UD-IQ1_S (a layer with its gate and up in two formats) and BF16, saying so
at load. A quant's n-gram table can be one of
unsloth's engram tables (BF16, Q8_0, Q5_1 or IQ4_NL): add `"--ple-gguf", "<the table's file>"` to the config's `args`,
or rename the file over the quant's table shard as unsloth's README says.
Without `--model`, setup builds the engine only.

| Option | |
| --- | --- |
| `--check` | say what this PC has (GPUs, RAM, disk, the tools) and what the model needs; change nothing |
| `--data-dir DIR` | where the model files go (default `../Lamina-data`, beside the clone) |
| `--vision on\|off\|gpu[:GPU]` | pictures (default on; the image encoder runs on the CPU). `gpu` builds it with CUDA (`build-vision-cuda`) and runs it on a GPU: the one named (name, UUID or nvidia-smi index), else the one with the least VRAM besides the main GPU, of those that drive no display if any; it takes ~1.3 GB there |
| `--main-gpu GPU` | the GPU the model runs on (default: the one with the most VRAM that drives no display) |
| `--context N` | the context in tokens (default: the example config's) |
| `--port N` | the server's port (default 8081 to 8084 by model) |
| `--update` | `git pull` this clone first |
| `--no-build` | use the programs as they are: a release's in `bin/` ([Release binaries](#release-binaries)), else those already built |
| `--cuda-archs A` | the CUDA architectures to build for, e.g. `86;120` (default: this PC's GPUs) |
| `--portable` | build the engine for other PCs (any AVX2 CPU; arm64: Armv8.2 with dotprod and fp16), as the releases and the Docker image are |

Setup asks nothing, installs no system software (it names what is missing) and moves or deletes no file. A step whose
result is already there is skipped, so setup runs again after an interruption or for another model:

1. For a build, CMake and Ninja into `.venv` and llama.cpp at the commit the engine is built against, into
   `third_party/llama.cpp`: ggml for the engine, mtmd for the image encoder. With `--no-build` nothing.
2. The server with the engine, `build/lamina-server` (`.exe`, with the CUDA runtime DLLs beside it; `build/lamina` is
   the engine alone), for this PC's GPUs, and the image encoder, `build-vision/bin/lamina-vision`: llama.cpp's mtmd
   library with the model's projector, run on the CPU beside the engine so the GPUs keep their VRAM for experts. The encoder is built for any AVX2 CPU: a native build on an AVX-512
   CPU loses flash attention and encodes about half as fast. With `--vision gpu` it is built with CUDA instead,
   `build-vision-cuda/bin/lamina-vision`, and runs on the GPU the option picks.
3. The model's files from Hugging Face at pinned revisions, into the data folder, resumed when interrupted and checked
   once by size and SHA-256 (`<file>.sha256` beside each records the check):
   [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
   [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF),
   [unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) and
   [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF). `HF_ENDPOINT` names a mirror.
4. Their preparation, for Flash-Next only: the MTP draft layer, `mtp/rt-distilled`, from
   [eddoursul/Lamina-Qwen3.8-Flash-Next-MTP](https://huggingface.co/eddoursul/Lamina-Qwen3.8-Flash-Next-MTP); with
   pictures, from the same repository the image projector in Q8_0 (ISTA-DASLab's BF16 one through `tools/mmproj_q8.py`),
   which the CPU encodes ~20% faster.
   Everything else comes from the GGUF files at start: the server reads the tokenizer and the chat template, the engine
   the experts, the projections and the float tensors (put in the forms it takes in memory, in milliseconds).
   Qwen3.6 and Qwen3.8-27B with pictures use their repositories' projectors (`mmproj-F16.gguf`) as they are.
5. The config, `lamina-<model>.json`, and its launcher, `run-<model>.bat` (`run-<model>.sh` on Linux), from
   `examples/<model>.json` with this PC's GPUs and data folder. A config written before keeps the keys the example
   lacks (an `api_key`, an `images` section).

On Windows on Arm the build takes LLVM's clang-cl for the C and C++ code (ggml's CPU code for Arm does not build with
MSVC; nvcc keeps MSVC's cl): Visual Studio's "C++ Clang Compiler for Windows" or LLVM's own install
(`LAMINA_CLANG_CL` names another).

## Release binaries

Each [release](https://github.com/eddoursul/Lamina/releases) has the programs built for Windows and Linux, on x64 and
arm64, in two downloads a platform:

- `lamina-<version>-bin-<os>-cuda-<x.y>-<cpu>`: the server with the engine (`lamina-server`), the engine alone
  (`lamina`) and the CPU image encoder (`lamina-vision`), with their licenses;
- `cudart-lamina-bin-<os>-cuda-<x.y>-<cpu>`: the CUDA runtime and cuBLAS they load, beside them (the same in every
  release of that CUDA version: one download serves the next releases).

Both are flat: extract them into one folder, `bin` in a clone of the release's tag (or of its source archive), and run
setup without the build:

```
git clone --branch <version> https://github.com/eddoursul/Lamina
cd Lamina
(extract both archives into bin\)
setup.bat --no-build --model iq3_s          (Windows)
./setup.sh --no-build --model iq3_s         (Linux)
```

Setup then needs neither the CUDA toolkit nor a compiler, nor llama.cpp or a Python package (the standard library
does), and the config and the launcher it writes run the programs in `bin/`. The programs are built for RTX 30, 40 and 50 GPUs (compute
capability 8.6, 8.9 and 12.x, the GB10's among them, with PTX for newer ones), any x64 CPU with AVX2 (the AVX-512
kernels where the CPU has them) or arm64 CPU with Armv8.2's dot product and FP16 instructions, and a driver for CUDA
13; the Linux ones for Ubuntu 22.04, Debian 12 or newer (glibc 2.34, GCC 12's libstdc++, OpenSSL 3), and without the
runtime archive they take the CUDA 13 libraries of the system (`LD_LIBRARY_PATH`). Other GPUs (an A100's 8.0, a GH200's 9.0), a
native build for the PC's CPU and `--vision gpu` (the encoder with CUDA) need setup's build. The arm64 programs are
built and their CPU tests run on each release, not yet run with a GPU.

The releases come from the GitHub workflow `.github/workflows/build.yml` at a `v*` tag; it also builds every push to
`main` and every pull request, and runs the tests that need no GPU. Its steps are `.github/ci.py`'s, which runs on a PC
the same way (`toolkit`, `build`, `test`, `package`).

## The config

`lamina-<model>.json` holds the engine's command line and the server's settings. Its keys from setup:

| Key | |
| --- | --- |
| `args` | the engine's options (below and `build/lamina --help`) |
| `cwd`, `lib_dirs` | the folder the engine runs in; on Linux, folders for the CUDA libraries (setup writes the toolkit's) |
| `model_name` | the name the API lists |
| `port`, `log` | the server's port; the engine's log file |
| `sampling` | the defaults for requests that send no sampling fields ([USING.md](USING.md#sampling)) |
| `vision` | the image encoder: its binary, the projector, the model, `threads` (the CPU's logical cores), or with `--vision gpu` `gpu` and `cuda_device` (the card it runs on) |
| `expert_profile_save`, `expert_profile_save_every` | what the run learned about the experts (below) |

The server's other keys (the network, API keys, idle unload, MCP servers, image generation):
[USING.md](USING.md#the-config). Relative paths are the clone's: the launchers start the server there.

**The learned expert profile.** The GPUs' expert caches start from a ranking of the experts by how often the model
routes to them (`data/expert-profile.bin`). With `expert_profile_save`, the engine also counts this PC's routing and
writes a profile (`<model>-learned.bin` in the data folder) at exit, on Ctrl+C and between requests at most once a
minute (`expert_profile_save_every`); the next start fills the caches from it. It ranks the experts by this PC's routing
and the profile it started from at equal weight. On the development PC, after a session of code and reasoning requests,
a round of fixed text took 9-14% less time on code, -3.8 to +0.9% on thinking and edits. Remove both keys to start from
the shipped profile every time.

## The GPUs

**`--main-gpu`** names the GPU the model runs on. Nothing else may use it: the engine synchronizes the GPU with its host
loop through spin-waits, and a display or another program on the same GPU makes the driver time-slice it, which slows
generation to a few tokens per second. Name it by its name or a part only its name has (`"RTX 3090"`), its UUID or the
start of it (`nvidia-smi -L` lists them), its PCI bus id, or its number as `nvidia-smi` numbers the GPUs (a number
moves when a card is added). The engine leaves `--vram-reserve-mib` (default 700) of it free and fills the rest with
experts (`--expert-cache auto`).

**`--second-gpu auto`** (the examples' setting) makes every other GPU a further expert tier, up to three, the fastest
link first: a short copy test at the start measures each GPU's rate from the host, and the log shows it. Each tier
holds the next experts of the profile and of the conversation and computes them beside the CPU; the first one also
computes the experts of a prompt that the main GPU lacks and part of the output head. A tier may drive the display: one
that does keeps 2048 MiB free, a headless one 512 (`--second-gpu-reserve-mib`). To fix the tiers and their order, name
them: `--second-gpu "RTX 5070 Ti,RTX 5060 Ti"`; `--second-gpu -1` uses none. Only the first tier takes part in
prompts. The others help generation, where an expert crosses the link once and stays, so a GPU on a slow link (an x1
or x4 slot) still pays as a later tier. Such a tier gets only short pieces of work, and a driver may lower its clocks
between them until its results come late: `--second-gpu-clocks` (below) locks them while the engine generates.

## Engine options

The example configs' options, and the ones that pay on some PCs. `build/lamina --help` lists every one.

| Option | |
| --- | --- |
| `--max-context N` | the context the K/V is sized for (the examples: 262,144, the trained context) |
| `--kv-grow` | the K/V takes VRAM only for the cells the requests reach; the expert cache holds the rest and gives it back as a conversation grows |
| `--kv fp16\|int8\|k8v4\|q4_0` | the K/V's format: 8-bit (`int8`), 8-bit keys and 4-bit values (`k8v4`), 4-bit both; smaller formats leave more VRAM to experts, at a cost in precision |
| `--spec T`, `--spec-min-p P`, `--spec-lookup N` | verify windows of up to T tokens (the last one and T-1 drafts), a draft kept while its probability under the draft layer stays above P (a sampled request's by the number its position is drawn with: [HOW_IT_WORKS.md](HOW_IT_WORKS.md#drafts)), and drafts from earlier text that repeats (16 tokens of it) |
| `--prefill N` | the prompt chunk (the examples: 16,384 tokens) |
| `--feed-max N` | prompt parts of up to N tokens go through verify windows instead of the prompt path |
| `--hc-q8` | the hyper-connection projections in 8 bits only: half their VRAM, read ~30% faster |
| `--prompt-shift F` | on a second GPU, a share of the experts the main GPU holds is computed beside it in long prompts |
| `--pcie-mode kernel --pcie-frac 0` | the experts no GPU holds are all computed by the CPU, none read by the main GPU over its link (the examples' setting) |
| `--prompt-split L` | off by default: a prompt of more than one chunk runs in two stages, the second GPU taking each chunk's layers L and up while the main GPU runs the next chunk |
| `--arena-place` | off by default: for RAM with slower regions (unpaired DIMMs), the experts the CPU reads are kept out of them |
| `--no-file-cache` | off by default: the experts are read past the OS file cache, which then holds no second copy of the model |
| `--second-gpu-clocks C[,..]` | off by default: per tier, its graphics and memory clocks locked through NVML while the engine generates, `<graphics min>-<graphics max>/<memory>` in MHz, the card's own (`0`: that tier's stay the driver's); needs administrator rights (root on Linux) |

The last four are measured on the development PC in [examples/](../examples/README.md); measure them on yours before
keeping them.

## Control vectors

A control vector in llama.cpp's format (a GGUF of `direction.<layer>` tensors, `general.architecture`
`controlvector`) steers Qwen3.8-Flash-Next at every token: after each layer of `--control-vector-layer-range A B`
(default 1-47, inclusive), on every hyper-connection stream h, as llama.cpp applies it with `build_cvec`:

| `--cvec-mode` | |
| --- | --- |
| `add` (the default, llama.cpp's) | h + S d |
| `project` (the [cvec-hooks](https://github.com/Cudila/llama.cpp/tree/cvec-hooks) fork's, which Strata applies) | h - \|S d\| (h . v) v, v = d / \|d\|: a unit direction at S 1 is removed |

`--control-vector-scaled FILE:S` takes a file at scale S (`--control-vector FILE`: 1); several files add up per layer,
and `--cvec-dir single:L` (project) applies layer L's direction in every layer of the range.

**The refusal direction of huihui-ai's abliterated releases**,
[Huihui-Qwen3.8-Flash-Next-refusal-direction-r-v2.gguf](https://huggingface.co/Pragmatism0220/Huihui-v2-Qwen3.8-Flash-Next-GSQ-RCO-Strata-abliterated-direction-GGUF),
takes out at runtime, from any quant of the base model, the direction their weights edit (their v1 removes it from what
each layer writes, v2 inverts it at 1.78 times that). In a config's `args`, as Strata's speed projection applied its
own vector:

```
"--control-vector-scaled", "<folder>/Huihui-Qwen3.8-Flash-Next-refusal-direction-r-v2.gguf:1.0",
"--control-vector-layer-range", "4", "44", "--cvec-mode", "project"
```

On the development PC, IQ3_S declined none of eight requests that it declined every one of without it, and began four
harmless answers word for word the same. That removes a safety behaviour: what the model writes with it is yours to
answer for. On ordinary text it moves the next-token distributions about as much as rounding does. Generation is
0.2-1% slower, a long prompt 0.4% (the prompt path applies the vector in the pass that writes the layer's output).
The vector holds for every request of the process: Strata's per-request switch is not there.

## Qwen3.6-35B-A3B and Qwen3.8-27B

The engine runs Qwen3.6-35B-A3B (GGUF architecture `qwen35moe`) and Qwen3.8-27B (`qwen35`, a dense model of the same
family) on a path of its own (`src/qwen35`): `lamina` switches to it from the model file's header, so their configs
name the file with `--native` like the others. It reads the GGUF as it is (no expert profile; the drafts come from the
file's own MTP block). K and V are fp16 (~22 KB a cell for Qwen3.6, ~68 KB for the 27B), or with `--kv q8_0` 8 bits
in llama.cpp's q8_0 rounding, about half: the 27B's full 262,144 cells then take ~9 GiB.

The drafts: `--mtp N` at most a step; `--spec-min-p P` ends a step's drafts before one whose joint probability with the
drafts before it, under the draft head, is below P; `--draft-vocab N` lets the draft head score only the first N token
ids (a draft outside them is a rejected draft, nothing else). The 27B's example takes 7, 0.3 and 65,536: up to 8
tokens a step on text that repeats its prompt, fewer where the drafts are unsure.

With `--second-gpu` (GPUs named as for `--main-gpu`, comma-separated, or `auto`: every other GPU in CUDA's order; the
examples have `auto`) the path splits the layers over the fewest of those GPUs, in order, that hold the whole model:
each takes the layers that fit beside their K/V and its buffers, the last the rest with the output head and the MTP
block, and each pass hands its activations from one GPU to the next. `--layer-split N,N,..` sets each GPU's layer
count. A prompt of several chunks (`--prefill`, the 27B's example 1,536 tokens) runs on the GPUs at once: the first
takes the next chunk while the second runs this one. Split, every Qwen3.6 expert is in VRAM and none takes RAM: on the
development PC, over the RTX 3090 and the RTX 5070 Ti (~4.7 GiB on the second), it generates 1.6-1.7x as fast and
reads prompts 1.9-3.3x as fast as on the 3090 alone ([COMPARISON.md](COMPARISON.md#qwen36-35b-a3b)). Where no split
holds every expert, or on one GPU, the main GPU keeps the experts that fit in its VRAM and the CPU computes the rest:
on a 24 GB GPU 212 of 256 a layer at the example's 65,536 cells, every one at 8,192 (`--context` sets another).
Qwen3.8-27B needs a split where one GPU does not hold it: its full 262,144 cells with K and V in q8_0 take ~36 GB of
VRAM (on the development PC the 5070 Ti first with 26 layers, then the 3090 with 38 and the head:
[COMPARISON.md](COMPARISON.md#qwen38-27b)).

Not on this path yet: the prompt cache's checkpoints (a follow-up turn whose history differs from what the engine holds
is read again from the start), `/v1/choose` and the GPU tiers. The config's `parallel` (batch slots) runs on one GPU.

## Docker

The `Dockerfile` builds the engine and the image encoder with setup for RTX 30, 40 and 50 GPUs (`CUDA_ARCHITECTURES`
narrows them). The container prepares a model in `/data` at its first start (later starts check the files) and serves
it on port 8080 with an API key:

```
docker build -t lamina .
docker run --rm --gpus all --ulimit memlock=-1 -p 127.0.0.1:8081:8080 -v lamina-data:/data \
           -e MODEL=iq3_s -e API_KEY=<a long random key> lamina
```

`VISION` (on, off), `CONTEXT` and `MAIN_GPU` pass the setup options of those names. The engine pins the experts in
RAM, hence `--ulimit memlock=-1`.

## Updating

`setup --update --model <model>` pulls this clone, rebuilds what changed and keeps the config's own keys. A model's
files stay where they are. With release binaries: check out the new release's tag, extract its archives into `bin/`
over the old ones and run setup with `--no-build`.
