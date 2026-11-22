# Setup

What a PC needs: [the README](../README.md#what-you-need). This page covers the build from the source, a model file
run with `lamina-server -m`, the release binaries, the config, the GPUs, the engine options worth knowing,
Qwen3.6-35B-A3B and Qwen3.8-27B, and Inkling 2 Mini.

## Building

From a clone of this repository, with the CUDA toolkit 13.0 or newer and, on Windows, Visual Studio 2022's C++ tools
(its free Build Tools with the workload "Desktop development with C++" do, or MSVC without Visual Studio, below), on
Linux GCC 11 or newer, CMake 3.22 or newer and Ninja (Ubuntu 22.04's own packages do):

```
build.bat                        (Windows)
./build.sh                       (Linux)
```

They build the preset `native` of `CMakePresets.json` into `build/`: `lamina-server` (the image encoder is a part of
it) and `lamina`, on Windows with the CUDA runtime's DLLs beside them (on Linux they take the toolkit's), for this PC's
CPU and its GPUs of the RTX 30 series and newer, as `nvidia-smi` names them at each build (after a card is added, build
again). The programs run from there: `build\lamina-server -m MODEL.gguf` ([below](#a-model-file)).
`build.bat portable` (`./build.sh portable`) builds them for other PCs into `build-portable/`, as the releases are: RTX
30, 40 and 50 GPUs (compute capability 8.6, 8.9 and 12.0, with PTX for newer ones), any x64 CPU with AVX2 or arm64 CPU
with Armv8.2's dot product and FP16 instructions, the C++ runtime linked in. CMake options after the preset's name go
to its configure (`build.bat native -DLAMINA_BUILD_TESTS=ON`, `--fresh` for a new start of the build folder). The
presets work with CMake alone too: `cmake --preset native`, `cmake --build --preset native` (on Windows in a Visual
Studio developer prompt).

- **llama.cpp**: the first configure downloads it at the commit the engine is built against, checks its SHA-256 and
  unpacks it into `third_party/llama.cpp`, without llama.cpp's web app (`tools/ui`, whose deep paths pass Windows'
  260-character limit). `-DLAMINA_GGML_DIR=<folder>` names another checkout.
- **The tools**: `build.bat` takes the MSVC the prompt has set up (a developer prompt's, or a portable one's: below),
  else finds Visual Studio 2022 (else 2019 or 2026) itself, and its CMake and Ninja unless the PATH has others; on
  Windows on Arm it compiles C and C++ with LLVM's clang-cl (Visual Studio's "C++ Clang Compiler for Windows" or
  LLVM's own: ggml's CPU code for Arm does not build with MSVC). nvcc is the one on the PATH, in `CUDA_PATH` or in
  `/usr/local/cuda`; `CUDACXX` names another, `CXX` the C++ compiler (on Linux nvcc's host compiler too:
  `CXX=g++-14` when the GCC is newer than the toolkit takes).
- **The checks**: a CUDA toolkit older than 13.0 stops the configure (engines built with CUDA 12.8 crashed in long
  prompts on RTX 50 GPUs), as do 13.2.0 and 13.2.1 for RTX 50 GPUs (their nvcc compiled some of the kernels wrong: the
  answers were garbage), and GPUs older than the RTX 30 series are left out.
- **Jobs**: half the logical cores compile and link at once, so that the PC stays usable, fewer when the free RAM holds
  fewer (1.5 GB a job: nvcc's largest files take up to 0.9 GB); `-DLAMINA_JOBS=N` sets another number. A build that
  stops is tried once more: CUDA's ptxas now and then fails to read a file it has just written.
- **What it builds**: the two programs. `-DLAMINA_BUILD_TESTS=ON` adds the tests (`ctest --test-dir build`), the
  parity checks and the tools, which otherwise build by name (`cmake --build build --target verify_parity`);
  `-DLAMINA_BUILD_VISION=OFF` leaves the image encoder out of the server, which then reads no pictures and does not
  start with a config that has a `vision` section.

**Without Visual Studio.** On Windows nvcc compiles with MSVC alone, and Visual Studio's installer is one way to get it:
[portable-msvc.py](https://gist.github.com/mmozeiko/7f3162ec2988e81e56d5c4e22cde9977) (Python 3.9 or newer) downloads
MSVC and the Windows SDK from Microsoft's servers into a folder, `msvc`, once Visual Studio's license is accepted (it
asks; ~0.5 GB downloaded, 1.2 GB unpacked). `--vs 2022` takes MSVC 14.44, which CUDA 13.0 and newer accept (the
script's default, Visual Studio 2026's compiler, needs CUDA 13.3 or newer). In a prompt with CMake and Ninja on the
PATH (cmake.org's and ninja-build.org's, or `pip install cmake ninja`):

```
python portable-msvc.py --vs 2022
msvc\setup_x64.bat
build.bat
```

The CUDA toolkit's Visual Studio integration is not needed.

`cmake --install build --prefix <folder>` puts the programs into a folder of their own, flat, as a release's two
downloads extract: the programs, `download-draft-layer.bat` (`.sh` on Linux) and the licenses (the component
`lamina`), and the CUDA runtime and cuBLAS the programs load, with NVIDIA's license (`cudart`; `--component` installs
one). Installed on Linux, the programs load the libraries in their own folder.

**The image encoder** is a part of `lamina-server` (half a megabyte of it): on the CPU a graph on ggml, on a GPU the
engine's own kernels, so a GPU it runs on is one the engine runs on (an RTX 30 or newer). A config's `vision` section
says which ([below](#a-model-file)); no program beside the server is needed for pictures.

The encoder reads the projectors of the engine's models: the type `qwen3vl_merger` (Qwen3.8-Flash-Next,
Qwen3.6-35B-A3B, Qwen3.8-27B) and Hy4-flash's `hy_v4_flash` (`mmproj-Hy4-flash-bf16.gguf`). The server refuses a
projector of another type by its name.

## A model file

`lamina-server -m MODEL.gguf` runs a model file as it lies, a split one by any of its shards:

```
lamina-server -m D:\Models\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
```

The server builds a config for the file and says in its window what it took:

- **The preset**, one of `examples/*.json`, built into the server and picked by the file's header: ISTA-DASLab's
  Qwen3.8-Flash-Next files take `iq3_s`, the family's other files (unsloth's quants, Agention's Gyro-L) `ud-q4_k_xl`,
  Qwen3.6-35B-A3B's `qwen36-ud-q4_k_xl`, Qwen3.8-27B's `qwen38-27b-ud-q6_k_m`, Hy4-flash's `hy4-flash-q4_k_m`,
  Inkling 2 Mini's `inkling2-mini-ud-iq4_xs`. Its engine options, sampling and port (8081 to 8087) are the config's; the
  model's name in the API is the file's, without the shard number.
- **The GPUs** ([below](#the-gpus)): the model on the card of the RTX 30 series or newer with the most VRAM that
  drives no display (the lower number on a tie), every other card an expert tier, as the preset's `"gpu"` (that card,
  then `auto`) has them; `--gpu NAME` names the main one, `--gpu 0,1` keeps it to those cards (the main one picked
  among them, the others its tiers).
- **Flash-Next's draft layer**, `Lamina-Qwen3.8-Flash-Next-MTP.gguf`, beside the model, else beside the server
  ([the README](../README.md#the-draft-layer)).
- **Pictures**: an image projector made for the model beside it (an `mmproj*.gguf` whose width is the model's; of
  several the smallest; Qwen3.6's and Qwen3.8-27B's repositories have theirs, `mmproj-F16.gguf`), read by the
  server's image encoder on the engine's main GPU, in VRAM it borrows from the engine for each new picture and gives
  back (the presets' `"gpu": true` and `"lend": true`). The encoder keeps no VRAM between pictures, so the GPUs hold
  what they hold without pictures, and a new picture of ~1,000 tokens takes about half a second (its weights read
  from the projector, the picture encoded: 0.3-0.6 s on an RTX 3090; Hy4-flash's pictures of up to ~4,100 rows 1-3 s)
  where the CPU takes 8-10 s. What the engine lends: Qwen3.8-Flash-Next the last slots of a card's expert cache, as a
  prompt's buffers take them (their experts are copied back after, ~0.1 s); Hy4-flash a part of its main GPU's expert
  slots; Qwen3.6-35B-A3B and Qwen3.8-27B the buffers a prompt's experts are streamed through where the experts do not
  all fit the GPUs, else VRAM they leave free (a layer split keeps the encoder's ~1 GB free on the main GPU); Inkling
  2 Mini VRAM its context leaves free for the loan from the start, or what its experts give up where they are in
  RAM. A picture
  whose loan finds no memory is encoded on the CPU, on its logical cores (the log says so once). With batch slots
  (`--parallel`) the engine lends beside the requests that are generating, and after a prompt one of them is still
  reading. The engine does not lend to a server with replicas: the CPU encodes there, as the window says,
  unless `--vision-cuda-device` names a card for the encoder to keep its VRAM on. `--no-vision-lend`: the encoder keeps its VRAM (the
  projector's size and 0.1-0.3 GB of buffers: 0.7 GB with Lamina's Q8_0 projector, 1.0 GB with an F16 one, 1.2 GB
  with Hy4-flash's) and encodes such a picture in 0.2 s, on a card besides the main one: the one with the least VRAM
  that drives no display, else any, else the main one; `--vision-cuda-device <UUID>` names another (`nvidia-smi -L`
  lists the UUIDs). On one RTX 3090 that kept VRAM costs Qwen3.6-35B-A3B 13 of the 212 experts a layer the card holds,
  2-4% of its generation speed and 17% of its prompts'. `--no-vision-gpu`: the encoder on the CPU alone. `--mmproj
  FILE` names a projector, `--no-mmproj` leaves pictures out; without a projector or the encoder the window says why
  there are none. For Qwen3.8-Flash-Next, download Lamina's Q8_0 projector into the model's folder:
  `mmproj-Qwen3.8-Flash-Next-Q8_0.gguf` (0.6 GB, ISTA-DASLab's BF16 one in Q8_0), in
  [eddoursul/Lamina-Qwen3.8-Flash-Next-MTP](https://huggingface.co/eddoursul/Lamina-Qwen3.8-Flash-Next-MTP) beside the
  draft layer. A loan for it is 0.7 GB where the BF16 one's is 1.0, and the CPU encodes a picture with it in ~20% less
  time than with the BF16 one and 280 MiB less RAM, with the same answers.
- **The expert profile**, Qwen3.8-Flash-Next's, built into the programs (`--expert-profile builtin`, the engine's
  default under `--serve`).
- **What a run writes**, beside the server: the engine's log (`<model>.log`) and what the run learned about the
  experts (`<model>-learned.bin`, [below](#the-config)), which the next start begins from.

The engine also runs Flash-Next's other GGUF files, most of unsloth's quants and
[Agention's Gyro-L](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF) among them. Of unsloth's Dynamic
3.1 quants it refuses UD-IQ1_S (a layer with its gate and up in two formats) and BF16, saying so at load. A quant's
n-gram table can be one of unsloth's engram tables (BF16, Q8_0, Q5_1 or IQ4_NL): `--ple-gguf <the table's file>`, on
the command line or in a config's `args`, or the file renamed over the quant's table shard as unsloth's README says.

**The draft layer.** `download-draft-layer.bat FOLDER` (`./download-draft-layer.sh FOLDER` on Linux; a release's archive
has its platform's script beside the programs) downloads `Lamina-Qwen3.8-Flash-Next-MTP.gguf` from
[eddoursul/Lamina-Qwen3.8-Flash-Next-MTP](https://huggingface.co/eddoursul/Lamina-Qwen3.8-Flash-Next-MTP) into a folder,
at a pinned revision, checked by its SHA-256 and a broken download resumed. A config whose `args` name no `mtp` takes
`Lamina-Qwen3.8-Flash-Next-MTP.gguf` from beside the model, else from beside the server, and the server's window says
which; elsewhere the `args` name it: `"mtp": "<folder>/Lamina-Qwen3.8-Flash-Next-MTP.gguf"`. A folder of the layer's
earlier layout (`dense.txt`, `dense.bin`, `experts.bin`, `draft_vocab.bin`), as earlier versions downloaded it to
`mtp/rt-distilled`, still works as `--mtp`. Flash-Next's server starts without the layer too, when none is found or the
named file or folder is not there, and warns in its window with the script's command. It then generates a token a verify
window, or prompt lookup's drafts (`--spec-lookup`). On the development PC, new text came at 44-61% of IQ3_S's speed
with the layer and 50-68% of UD-Q4_K_XL's, an edit that quotes its prompt at 92-96%. The layer's file holds its draft
vocabulary, the 40,525 tokens it proposes from; `tools/mtp_rt.py --gguf <the file> --draft-vocab <ids> --out <new file>`
writes one over another (`data/draft_vocab_fr.bin`, `draft_vocab_cyrillic.bin`, `tools/draft_vocab.py`).

Every key of a config is an option of the command line too ([USING.md](USING.md#the-config)), and a value given there
takes the place of the config file's, the file's that of the preset: `lamina-server -m MODEL.gguf --port 8090
--temperature 0.6`. A config file can be as short as `{"model": "D:/Models/x.gguf", "port": 8090}`: without `"args"`
the server builds the rest from the model's preset, as with `-m`, and the file's keys take the place of the preset's;
a config with `"args"` (an empty `{}` too) stays as it is, its `"model"` (or `-m`) naming the file. `"args"` holds the
engine's options that differ from its defaults, as an object of the options without their dashes (`"kv": "k8v4"`, a
switch `true` or `false`: `"kv-grow": false` is `--no-kv-grow`, a list for several values:
`"control-vector-layer-range": [4, 44]`) or as a list of words (`"--kv", "k8v4"`). What a run writes (the log, the
learned profile) goes beside that config file. An engine option on the command line, with the values after it, takes
the place of the preset's or the config's: `lamina-server -m MODEL.gguf --max-context 65536 --kv q4_0`. The engine's own
parser reads those options first, for the model's family, and the server stops on one it does not take ("unknown
argument: ...", a value out of range); `--native`, `--serve` and `--vision` are the server's to set (the model is
`"model"`, the pictures the `vision` section). `lamina-server --config OLD.json --write-config NEW.json` writes an
earlier config in this form: its GPUs as cards (the main GPU's and the per-tier flags as their settings, into `"gpu"`'s
own cards too), the other deprecated options renamed (a replica's own `args` too), its options at the engine's default
left out, `"args"` an object. Three keep their GPU flags, as the engine still reads them: a `"gpu"` of numbers (the
cards the engine sees), replicas (each one's `"gpu"` its group's numbers) and an `auto` card a flag would give a
setting.

`--write-config FILE` writes the config the command line runs to FILE (the preset's, the config file's and the command
line's settings in one) and stops; `lamina-server --config FILE` then runs it, and the web app's Model settings card
saves into it (a server started with `-m` alone has no file: the card's changes last until it stops). The log and the
learned profile then go beside FILE. An earlier FILE is kept as `FILE.bak`, and its keys outside the preset (an
`api_key`, an `images` section) carry over.

`--check` says what this PC has (the CPU, the RAM, the GPUs) and, with `-m` or `--config`, the model's file, the preset,
the GPUs it runs on, the RAM its experts take, the draft layer, the pictures, the port and the engine's command line,
then whether it would start; it loads nothing and exits with 1 when the model would not start (no usable GPU, too
little RAM, a file the engine does not read, a missing encoder or projector).

`--bench` measures the model the way it is used and writes a report to share: `lamina-server -m MODEL.gguf --bench`
(or `--config FILE --bench`) loads it on a port of its own, sends a fixed suite through the server's API, three runs of
each test, and stops: three short questions (decode speed, time to the first token), fresh prompts of ~4K and ~16K
tokens (prompt speed), and a coding agent's session of six turns with four tools (what the conversation cache
reused), 3-10 minutes after loading. Speeds are the engine's own timings, first-token times the client's clock;
thinking is off. The report goes into `bench-results/<date>-community-<GPU>-<model>/` beside the program:
`README.md`, `results.json` with every request, and the engine's log lines, with home folders, user and host names,
private addresses and API keys replaced. `--bench-clients N` adds N requests at once, `--bench-long` a ~32K prompt,
`--bench-compare OLD` puts an earlier report beside this one (`--bench-compare OLD NEW`: two reports, nothing run),
and `--bench-url http://127.0.0.1:8081` measures a server that runs instead. A Lamina server on the config's port
stops it before anything loads: the two would share the GPUs.

## Release binaries

Each [release](https://github.com/eddoursul/Lamina/releases) has the programs built for Windows and Linux, on x64 and
arm64, in two downloads a platform:

- `lamina-<version>-bin-<os>-cuda-<x.y>-<cpu>`: the server with the engine and the image encoder (`lamina-server`)
  and the engine alone (`lamina`), with their licenses, and the script that downloads Flash-Next's draft layer (`download-draft-layer.bat`, `.sh` on Linux);
- `cudart-lamina-bin-<os>-cuda-<x.y>-<cpu>`: the CUDA runtime and cuBLAS they load, beside them (the same in every
  release of that CUDA version: one download serves the next releases).

Both are flat: extract them into one folder, and `lamina-server -m` runs a model file from there
([above](#a-model-file)). The programs are built for RTX 30, 40 and 50 GPUs (compute capability 8.6, 8.9 and 12.x, the
GB10's among them, with PTX for newer ones), any x64 CPU with AVX2 (the AVX-512 kernels where the CPU has them) or arm64
CPU with Armv8.2's dot product and FP16 instructions, and a driver for CUDA 13; the Linux ones for glibc 2.34 and
OpenSSL 3, as Ubuntu 22.04, Debian 12, RHEL 9 (and Rocky Linux, AlmaLinux), Amazon Linux 2023, openSUSE Leap 15.6 and
newer systems have them (the C++ runtime is linked in), and without the runtime archive they take the CUDA 13 libraries
of the system (`LD_LIBRARY_PATH`). Other GPUs (an A100's 8.0, a GH200's 9.0) and a native build for the PC's CPU need a
build from the source ([above](#building)). The arm64 programs are built and their CPU tests run on each release, not
yet run with a GPU.

The releases come from the GitHub workflow `.github/workflows/build.yml` at a `v*` tag; it also builds every push to
`main` and every pull request, and runs the tests that need no GPU. Its steps are `.github/ci.py`'s, which runs on a PC
the same way (`toolkit`, `build`, `test`, `package`).

## The config

A config holds the model, its GPUs, the engine's options and the server's settings; `--write-config` writes one from a
preset ([above](#a-model-file)). Its keys:

| Key | |
| --- | --- |
| `model` | the model's GGUF (its first shard) |
| `gpu` | the GPUs as cards in their roles, each with its settings ([below](#the-gpus)) |
| `args` | the engine's options that differ from its defaults, an object (`"kv": "k8v4"`) or a list of words ([below](#engine-options) and `build/lamina --help`) |
| `cwd`, `lib_dirs` | the folder the engine runs in, where the config's relative paths start (the folder the server starts in when it has none); on Linux, folders put first on the library path |
| `model_name` | the name the API lists |
| `port`, `log` | the server's port; the engine's log file |
| `sampling` | the defaults for requests that send no sampling fields ([USING.md](USING.md#sampling)) |
| `vision` | the image encoder's files and settings: the projector (`mmproj`; `model` is taken and not read), `threads` (the CPU's logical cores), or `gpu` (`true`, or the card it runs on, named as in `"gpu"`) and `cuda_device` (that card by its UUID: the server does not start when that card is not there; without one the server picks a card as `--vision-gpu` does), `lend` (on a GPU: `true` has the encoder hold no VRAM between pictures and borrow the engine's on its card for each new one ([above](#a-model-file) says what each engine lends; without `cuda_device` that card is the engine's main one; the examples set `gpu` and `lend`), `q8` (a Q8_0 projector on a GPU: `mmq`, the default, multiplies its blocks as llama.cpp does, `fp16` its values as FP16, nearer the unquantized projector's embeddings; the answers measured the same). The encoder is a part of the server |
| `expert_profile_save`, `expert_profile_save_every` | what the run learned about the experts (below) |

The server's other keys (the network, API keys, idle unload, MCP servers, image generation):
[USING.md](USING.md#the-config).

**The learned expert profile.** The GPUs' expert caches start from a ranking of the experts by how often the model
routes to them (`data/expert-profiles`: one a model, built into the programs). With `expert_profile_save`, the engine
also counts this PC's routing and writes a profile (`<model>-learned.bin` beside the config file, beside the server
with `-m` alone) at exit, on Ctrl+C and between requests at most once a minute (`expert_profile_save_every`); the
next start fills the caches from it. It ranks the experts by this PC's routing and the profile it started from at
equal weight. On the
development PC, after a session of code and reasoning requests, a round of fixed text took 9-14% less time on code, -3.8
to +0.9% on thinking and edits. Remove both keys to start from the shipped profile every time.

## The GPUs

A config's `"gpu"` names the GPUs in their roles, each with its own settings: the first card runs the model, the others
hold experts as tiers, in their order (up to three). On the command line `--gpu`, again for each card, its settings
after its name:

```
"gpu": ["GPU-75af3083", "RTX 5070 Ti", {"id": "GPU-f26e9ba6", "clocks": "1395-2100/9751", "sleep": "55us"}]
--gpu GPU-75af3083 --gpu "RTX 5070 Ti" --gpu GPU-f26e9ba6,clocks=1395-2100/9751,sleep=55us
```

A card is named by its name or a part only its name has (`"RTX 3090"`), its UUID or the start of it (`nvidia-smi -L`
lists them), its PCI bus id, or its number as `nvidia-smi` numbers the GPUs (a number moves when a card is added).

**The first card runs the model.** Nothing else may use it: the engine synchronizes the GPU with its host loop through
spin-waits, and a display or another program on the same GPU makes the driver time-slice it, which slows generation to
a few tokens per second. It leaves 700 MiB of its VRAM free and fills the rest with experts.

**The cards after it are expert tiers**; `auto` after the first (the examples' setting) makes every other GPU one, the
fastest link first: a short copy test at the start measures each GPU's rate from the host, and the log shows it. Each
tier holds the next experts of the profile and of the conversation and computes them beside the CPU; the first one also
computes the experts of a prompt that the main GPU lacks and part of the output head. A tier may drive the display: one
that does keeps 2 GiB free, a headless one 512 MiB. Only the first tier takes part in prompts (and the second with
`--prompt-tiers 2`). The others help generation, where an expert crosses the link once and stays, so a GPU on a slow
link (an x1 or x4 slot) still pays as a later tier. Such a tier gets only short pieces of work, and a driver may lower
its clocks between them until its results come late: its `clocks` lock them while the engine generates.

A card's settings, each optional:

| Setting | |
| --- | --- |
| `cache=auto\|SIZE\|N` | its VRAM for experts: `auto` (the default) all but its reserve, or a size (`22G`); the first card's also N expert slots |
| `reserve=SIZE` | the VRAM it leaves free: the first card 700 MiB; a tier 2 GiB when the desktop uses it, else 512 MiB |
| `vram-only=SIZE\|auto` | the profile's experts it keeps only in its VRAM, out of the RAM: for a model whose experts outgrow the RAM; `auto` takes as many as keep them within it (the RAM less 4 GiB, and 6 GiB of it left available). The first card and the first two tiers (the second then computes its experts in prompts too); not with `sleep` |
| `prefetch=SIZE` | a tier: each layer's likeliest experts no GPU holds, copied to it ahead while the RAM is idle (12 MiB over a link of 20 GB/s or more, else none) |
| `clocks=MIN-MAX/MEM` | a tier: its graphics and memory clocks locked through NVML while the engine generates, in MHz, the card's own (`1395-2100/9751` for an RTX 3090; a memory clock of 0: the driver's); needs administrator rights (root on Linux). `--clocks-hold` and `--clocks-lead` time the locks around a request ([below](#engine-options)) |
| `sleep=TIME` | a tier after the first: it sleeps (its experts to the CPU, its lock released) while the CPU pool would need less than TIME a layer without it, averaged over a few seconds of generation; trades speed for the card's awake power, TIME by model file (`55us`) |

Sizes are binary (`8G`, `700M`), times have their unit (`55us`, `1.5ms`, `1s`). `"gpu"` as numbers alone (`[0, 1]`,
`--gpu 0,1`) keeps its earlier meaning: the cards the engine sees, the first its main one. An earlier config's
`--main-gpu`, `--second-gpu` and per-tier lists (`--second-gpu-clocks 0,1395-2100/9751`: a value per tier in their
order) still work, deprecated (`lamina --help-all` lists them), a card's own setting winning over them; of `--gpu`,
`--main-gpu` and `--second-gpu` the later takes the earlier's place (`--second-gpu -1` after a config's cards: no tiers).
lamina-server names the deprecated options a start uses in its window, each with what takes its place (an earlier
config's `--native` in its `"args"`, in place of `"model"`, too).

**Each GPU is checked** before the engine uses it: one older than the RTX 30 series (compute capability below 8.0), or
one the programs have no kernels for, stops the start as the main GPU and is left out as a tier or a layer split's GPU,
the log saying why. A build from the source is for this PC's GPUs: after a card is added, build again. The releases'
programs run on the RTX 30, 40 and 50 series.

## Engine options

The engine's defaults are the examples' values: a config's `args` holds what differs. The options the examples set, and
the ones that pay on some PCs; `build/lamina --help` lists the common ones, `--help-all` every one.

| Option | |
| --- | --- |
| `--max-context N` | the context the K/V is sized for (by default under `--serve` 262,144, the trained context) |
| `--kv-grow` | the default: the K/V takes VRAM only for the cells the requests reach; the expert cache holds the rest and gives it back as a conversation grows (`--no-kv-grow`: the whole context at start) |
| `--kv fp16\|int8\|k8v4\|q4_0` | the K/V's format: 8-bit (`int8`), 8-bit keys and 4-bit values (`k8v4`), 4-bit both; smaller formats leave more VRAM to experts, at a cost in precision |
| `--spec T`, `--spec-min-p P`, `--spec-lookup N` | verify windows of up to T tokens (the last one and T-1 drafts), a draft kept while its probability under the draft layer stays above P (a sampled request's by the number its position is drawn with: [HOW_IT_WORKS.md](HOW_IT_WORKS.md#drafts)), and drafts from earlier text that repeats (16 tokens of it); 4, 0.5 and 16 by default |
| `--prefill N` | the prompt chunk (by default 16,384 tokens, halved until its buffers fit the caches' lendable slots) |
| `--feed-max N` | prompt parts of up to N tokens go through verify windows instead of the prompt path (46 by default) |
| `--hc-q8` | the default: the hyper-connection projections in 8 bits only, half their VRAM, read ~30% faster (`--no-hc-q8`: BF16) |
| `--prompt-shift F` | on a second GPU, a share of the experts the main GPU holds is computed beside it in long prompts |
| `--pcie-frac F`, `--pcie-mode M` | 0 by default: the experts no GPU holds are all computed by the CPU, none read by the main GPU over its link (over an x4 link the CPU is sooner); a share F of them read over it otherwise (`kernel`, the default mode: a copy kernel in the window's graph) |
| `--prompt-split L[,M]` | off by default: a prompt of more than one chunk runs in two stages, the second GPU taking each chunk's layers L and up while the main GPU runs the next chunk; `L,M` with two or more tiers: three stages, the tier after the first taking the layers L to M-1 (with one tier: two stages, split halfway between L and M) |
| `--prompt-split-from N` | off by default: with `--prompt-split`, a prompt part of N tokens or more that fits one chunk is read as two chunks of half its size through the stages (a shorter one runs on the main GPU's stage alone) |
| `--prompt-tiers 2` | off by default: with two or more tiers, the tier after the first computes the experts it holds in prompts too, where they otherwise stream to the first tier |
| `--arena-place` | off by default: for RAM with slower regions (unpaired DIMMs), the experts the CPU reads are kept out of them |
| `--no-file-cache` | off by default: the experts are read past the OS file cache, which then holds no second copy of the model |
| `--clocks-hold TIME` | with a card's `clocks` under `--serve`: the locks stay this long after a request (`5s` by default); a locked card draws its awake power until then |
| `--clocks-lead TIME` | off by default (the locks are asked for from a batched prompt part's last chunk on): they are asked for this long before the part ends, by its progress; a lock is at speed ~0.6 s after it is asked for |

The options off by default, and the cards' `clocks` and `sleep`, are measured on the development PC in
[examples/](../examples/README.md); measure them on yours before keeping them.

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
"control-vector-scaled": "<folder>/Huihui-Qwen3.8-Flash-Next-refusal-direction-r-v2.gguf:1.0",
"control-vector-layer-range": [4, 44], "cvec-mode": "project"
```

On the development PC, IQ3_S declined none of eight requests that it declined every one of without it, and began four
harmless answers word for word the same. That removes a safety behaviour: what the model writes with it is yours to
answer for. On ordinary text it moves the next-token distributions about as much as rounding does. Generation is
0.2-1% slower, a long prompt 0.4% (the prompt path applies the vector in the pass that writes the layer's output).
The vector holds for every request of the process: Strata's per-request switch is not there.

## Qwen3.6-35B-A3B and Qwen3.8-27B

The engine runs Qwen3.6-35B-A3B (GGUF architecture `qwen35moe`) and Qwen3.8-27B (`qwen35`, a dense model of the same
family) on a path of its own (`src/qwen35`): `lamina` switches to it from the model file's header, so their configs
name the file with `"model"` like the others. It reads the GGUF as it is (no expert profile; the drafts come from the
file's own MTP block). K and V are fp16 (~22 KB a cell for Qwen3.6, ~68 KB for the 27B), or with `--kv q8_0` 8 bits
in llama.cpp's q8_0 rounding, about half: the 27B's full 262,144 cells then take ~9 GiB.

The drafts: `--drafts-per-step N` at most a step (an earlier config's `--mtp N` is taken as it, with a warning);
`--spec-min-p P` ends a step's drafts before one whose joint probability with the drafts before it, under the draft
head, is below P; `--draft-vocab N` lets the draft head score only the first N token ids (a draft outside them is a
rejected draft, nothing else). The 27B's example takes 7, 0.3 and 65,536: up to 8 tokens a step on text that repeats
its prompt, fewer where the drafts are unsure.

With more cards than one in `"gpu"` (named as above, or `auto` after the first: every other GPU in CUDA's order; the
examples have the main GPU, then `auto`) the path splits the layers over the fewest of those GPUs, in order, that hold
the whole model (their settings are the Flash-Next path's: this one sizes its own):
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

The prompt cache is Flash-Next's, with its options ([USING.md](USING.md#behaviour)): checkpoints of the recurrent
state (64 MiB each for Qwen3.6, 151 MiB for the 27B) and conversations parked in RAM with their K and V; the same
request again reads nothing. Not on this
path yet: `/v1/choose` and the GPU tiers. The config's `parallel` (batch slots) runs on one GPU: each of its sessions
holds a conversation with checkpoints of its own, a request runs in the session that holds its conversation, and
the conversations the sessions do not hold are parked in RAM (`--prompt-cache 0`: each slot keeps its conversation
and continues it only when the request extends it).

## Inkling 2 Mini

The engine runs Thinking Machines Lab's Inkling 2 Mini (GGUF architectures `inkling2`, unsloth's files, and
`inkling2_loop`, mainline llama.cpp's layout of the same tensors: unsloth's `Mainline_Support/*.gguf_file`) on a
path of its own (`src/inkling2`): `lamina` switches to it from the model file's header. Every quant of unsloth's
repository runs, from UD-IQ1_S (7.2 GiB) to UD-Q8_K_XL (21.3 GiB); `lamina-server -m FILE --check` says whether a
card holds one and how much context is left beside it. The model is 24 layers of which 12 run three times a token, 128 experts
a layer with 8 used; five layers of six attend their last 1,024 tokens and the sixth every token, so K and V are kept
whole for 8 of the 48 layer applications: 32.0 KiB a token in FP16, 17.0 with `--kv q8_0`.

It runs on one GPU, which holds the whole file (the experts in the file's own formats: NVFP4 in most layers of
unsloth's quants) beside the K and V; the embedding table stays in RAM (~0.5 GB). A config's later `"gpu"` cards
hold the K and V the first has no VRAM for (`--cell-gpus`): its oldest cells move to them as a conversation grows,
each card scores its cells for every token, and the context is what the cards hold together, up to the model's
1,048,576 tokens. A card's `clocks=` locks its clocks while it holds cells and a request runs (a card that idles
between its scans answers late; the lock needs administrator rights), and its `reserve=SIZE` is the VRAM it leaves
free (700 MiB, or 1,536 where the desktop draws on it). `lamina-server --check` says what each card holds: the
model and K and V of how many tokens, in FP16 and in q8_0. Where no card holds the model, one later card can run a
run of its layers instead - `layers=A-B` (`{"id": "RTX 3060", "layers": "6-17"}`): the layers' weights and their
part of the context are on it, and the answers are one card's to the bit. The run holds the layers that repeat
(Inkling 2 Mini: 6-17, half of the weights) all or not at all, and leaves the first and the last layer on the
first card. Those layers are three quarters of a token's work: the faster card runs them best (an RTX 3090 with
them on an RTX 5070 Ti generates 15-25% faster than the 3090 alone), and a prompt is read on the two cards at
once (30-50% faster there; two cards on slow links gain less). The two hold the cells of as many tokens as
the fuller one has room for beside its layers, the other holds older cells in what it has free, and cards after
the two hold more, as without a split. Both cards take `clocks=` here (each works a part of a token and clocks down in the
other's), the later one `reserve=`. Without `layers=` the engine does this by itself with the card after the first
where the first would keep under 80% of the experts (two 12 GB cards: all the experts on cards instead of a third
of them in RAM); `--layer-split 0` keeps the experts in RAM. Its options: `--kv fp16 | q8_0`;
`--max-context N` (not set: what the free VRAM holds K and V for, at most the model's 1,048,576 tokens and not under
32,768; set: a sequence ends where the VRAM does); `--vram-reserve-mib N` (the VRAM left free: 700, or 1,536 on a
card the desktop draws on); `--no-kv-grow` allocates the context's K and V at the start, where they are otherwise
mapped as a conversation grows; `--no-graph`, `--no-file-cache`, `--load-threads N`.

A card smaller than the file, or than the file and the least context beside it (32,768 tokens; a set
`--max-context` where no other card holds K and V), runs it with the experts in RAM (UD-IQ4_XS: 13.1 GiB, pinned;
what the RAM has no room for above `--ram-reserve-mib` stays on the card alone, and a start that finds room in
neither is refused with the numbers). The card has the experts used most, in the VRAM its K and V leave: filled at
the start by a ranking (the one built into the programs; `--expert-profile FILE` another), they give way as a
conversation grows and come back after it. The CPU computes the experts a token routes that the card lacks
(`--cpu-threads N`: its pool's threads, a worker a core by default), so the speed follows their share: 2-5% of them
on a 16 GB card with UD-IQ4_XS (an RTX 5070 Ti: 135-319 tok/s through the API, llama.cpp 21), 11-21% with 6 GiB of
experts on the card (106 tok/s in an agent's session where the whole model on an RTX 3090 gives 171). A prompt's
chunk sends the RAM experts it routes to the card - the next layer application's while the card computes the
current one's, as many as the card's link carries meanwhile: a prompt is read at the card's own speed on a fast
link (PCIe 4.0 or 5.0 x16), at a third of it on four lanes with 6 GiB of experts on the card. Which experts
are on the card follows the conversation, and `--expert-profile-save FILE` (the config's `expert_profile_save`,
which `-m` sets; `--expert-profile-save-every MIN`, 1) keeps the ranking a run's routing makes for the next start.
`--expert-cache SIZE` (the first card's `cache=SIZE`: `6G`) caps the experts' VRAM. With experts in RAM a request
sent again may write another text: which experts the CPU computes follows the routing counted so far.

Drafts: a generated token is a window of rows through the model, the last token and up to 7 drafts after it; a
draft stays while it is the token the request's sampler picks (the sampler runs on the GPU, a row's draw by its
token's position), and a window's rows leave what a row at a time leaves, to the bit, so a greedy text is the same
with drafts and without. The drafts are the tokens that followed the text's last 16 tokens or more where they
occurred before (`--spec-lookup N`, prompt lookup; 0: none), else those of the model's block drafter, DSpark:
`lamina-server` takes the smallest `dspark-*.gguf` beside the model, else in its `dspark/` folder (unsloth's
`dspark-Inkling-2-Mini-Q8_0.gguf`), `--dspark FILE` names one, `--drafts-per-step 0` runs without (N: its drafts a
window at most, 7). Its drafts go on while their joint probability under its head stays at `--spec-min-p` (0.7).
It takes 0.40 GiB of VRAM for its weights and ~0.35 GiB for its context cells (made from the model's own states of
every position; `--dspark-reach N`, 32,768: those its layer without a window keeps) and buffers, and is left out
where the card alone holds the context and would hold under 32,768 tokens beside it. UD-IQ4_XS on an RTX 3090: 108
tok/s without drafts, 120 after a short prompt, 140 thinking, 248 where the answer quotes its prompt.

A prompt is read in chunks of `--prefill N` rows (1,024; their buffers take ~0.8 GiB of VRAM; 0: 8 rows at a time
through the generation's kernels), which grow to `--prefill-max N` rows (4,096: 2.3 GiB more, 14% faster at 21K
tokens) for a long prompt while the card's free VRAM holds the larger buffers beside the context's cells, and are
back after the request. With experts in RAM the larger buffers take the VRAM of the experts used least for a part
of two such chunks or more, and those experts are back on the card before the first token after it (an RTX 5070 Ti
alone: 11% faster at 21K tokens, 23% at 86K). Where the card alone holds the context and would hold under 32,768 tokens (or
under `--max-context`) beside those buffers, the rows are halved until it does, down to 128 (a line says so: the
21.3 GiB UD-Q8_K_XL on a 24 GB card). On an RTX 50 a chunk's expert products run on the card's FP4 tensor cores, as
llama.cpp's do. The prompt cache keeps conversations as on the other paths (`--prompt-cache`, `--prompt-cache-every`,
`--conversation-cache-mib`, `--conversation-cache-slots`, `--conversation-cache-min-tokens`, `--ram-reserve-mib`):
a checkpoint holds the window layers' last cells and the convolutions' state (169 MiB in FP16, 92 in q8_0) and
leaves through an image of that size in VRAM; a conversation another request replaces is parked with its K and V -
on the card while the VRAM free above the reserve holds the copy (a switch back is then a copy on the device: two
conversations of 82K and 8K tokens in turns take 0.1 s a switch where the way to RAM and back takes 0.5), in RAM
once the card needs the room for cells, a long prompt's larger chunk or the image encoder (`--park-vram 0`: in RAM
at once).

Its chat form is its own: a reply is blocks of thinking, text and tool calls between special tokens, a call's
arguments JSON. The server reads them as such (a call's arguments are streamed as the model writes them) and writes
the server's own forms for it (a forced call's opening, the close of thinking at a budget). `reasoning_effort`
none, low, medium and high are the template's levels 0, 0.2, 0.7 and 0.9; at none the model still writes a short
thinking block, which comes as reasoning. The preset samples at temperature 1.0 over the 64 likeliest tokens (the
model card: the plain distribution). Pictures: the repository's `mmproj-F32.gguf` beside the model (`-m` finds it;
a config's `vision` section names it). Its processor does not resize: a picture is a token a patch of 40 x 40 px, its
sides padded to whole patches, up to 4,096 tokens (the section's `max_tokens`; one past the limit is resized to fit):
a 2,560 x 1,440 screenshot is 2,304 tokens, encoded in 0.5-0.6 s on an RTX 3090 (0.9 GB of VRAM while it works: the
projector's F32 matrices are FP16 on the card).

## Updating

`git pull` and the build again update a clone's programs; the configs and the model files stay where they are. With
release binaries, extract the new release's archives over the old ones.

**A clone an earlier version's setup made.** `setup.bat` and `setup.sh` are gone. Their `--update` pulls, then stops
at the `setup.py` it runs after the pull ("can't open file '...setup.py'", on Windows also "The batch file cannot be
found"): the pull went through, and nothing is built. `build.bat` (`./build.sh`) builds the programs into the same
`build/`, where the `run-<model>.bat` (`.sh`) and `lamina-<model>.json` setup wrote run them as before. The configs
need no change: their `vision` section's `exe`, which named setup's encoder in `build-vision/bin` or
`build-vision-cuda/bin`, is not read (the encoder is a part of `lamina-server`; a build with `-DLAMINA_BUILD_VISION=OFF` leaves such a
config unable to start), and the draft
layer's folder `mtp/rt-distilled` still loads.

- **CMake and Ninja**: `build.bat` takes Visual Studio's ("C++ CMake tools for Windows", part of the workload "Desktop
  development with C++"), `build.sh` the PATH's (`sudo apt install cmake ninja-build`). `build/` keeps the Ninja it was
  configured with: when that one is gone, its configure stops ("Running '...ninja' '--version' failed") until
  `build.bat --fresh` (`./build.sh --fresh`).
- **The CUDA toolkit**: `build/` keeps the nvcc setup chose; `CUDACXX=<path to nvcc>` with `--fresh` takes another
  ([above](#building)).
- **What setup did otherwise**: `setup --check` is `lamina-server --config lamina-<model>.json --check`; `BENCH.bat`
  and `bench.sh` are `lamina-server --config lamina-<model>.json --bench`, and `--bench-compare <an earlier report's
  folder>` reads the reports they wrote; a model setup did not download runs with `lamina-server -m`
  ([above](#a-model-file)), which downloads nothing. The image generator `tools/image/install.py` built
  (`build-image-cuda`) and the config's `images` section keep working; it is built again by hand
  ([USING.md](USING.md#image-generation)).
- **What nothing uses now**: `build-vision/` and `build-vision-cuda/` (setup's encoders). `third_party/llama.cpp` is
  the build's.
