# Troubleshooting

The engine's log is the config's `log` (`<model>.log` beside the config file, beside the server with `-m` alone); the
server writes to its window. `lamina-server -m MODEL.gguf --check` (or `--config FILE --check`) says what the PC has
and what a model needs.

## Building

**`setup --update` ends with "can't open file '...setup.py'"** (on Windows also "The batch file cannot be found").
Setup is gone; an earlier version's `--update` pulls, then runs the pulled `setup.py`. The pull went through: build
with `build.bat` (`./build.sh`) ([SETUP.md](SETUP.md#updating), A clone an earlier version's setup made).

**"Visual Studio's C++ tools are needed".** `build.bat` found no Visual Studio 2022 (nor 2019 or 2026) with the C++
tools: install Visual Studio 2022 or its free Build Tools with the workload "Desktop development with C++", which has
CMake and Ninja too, or run `build.bat` from a developer prompt of the one to use, or after a portable MSVC's
`setup_x64.bat` ([SETUP.md](SETUP.md#building), Without Visual Studio).

**"CMake 3.22 or newer is needed", "Ninja is needed", or CMake's "CMake 3.22 or higher is required".** Install them:
on Windows Visual Studio's "C++ CMake tools for Windows" (part of that workload) or cmake.org's; on Linux the
distribution's `cmake` and `ninja-build`.

**"No CMAKE_CUDA_COMPILER could be found".** The CUDA toolkit is not installed, or its nvcc is neither on the PATH nor
in `CUDA_PATH` or `/usr/local/cuda`: install the toolkit 13.0 or newer, or name its nvcc, `CUDACXX=<path to nvcc>`,
and build with `--fresh` (`build.bat --fresh`), as the build folder keeps the first compiler it found.

**"Lamina builds with the CUDA toolkit 13.0 or newer, and ... is 12.x".** Name a newer toolkit's nvcc with `CUDACXX`
and build with `--fresh`.

**"CUDA 13.2.0 or 13.2.1 compiles some of the engine's kernels wrong for RTX 50 GPUs".** The answers would be
garbage: use CUDA 13.0, 13.1 or 13.2.2 and newer (`CUDACXX`, `--fresh`), or build for older GPUs only
(`-DCMAKE_CUDA_ARCHITECTURES=86`).

**"CMAKE_CUDA_ARCHITECTURES is native, and nvidia-smi names no NVIDIA GPU ...".** The NVIDIA driver is missing, or
`nvidia-smi` is not on the PATH, or the GPUs are older than the RTX 30 series. For another PC's GPUs: `build.bat
portable`, or name them (`-DCMAKE_CUDA_ARCHITECTURES=86;120`, quoted for `build.bat`).

**Linux: the build stops at `unsupported GNU version`.** The GCC is newer than the CUDA toolkit accepts. Install an
older one beside it (g++-14) and name it: `CXX=g++-14 ./build.sh --fresh` (nvcc then takes it too; `CUDAHOSTCXX`
names nvcc's alone).

**"llama.cpp ...: the download failed".** The build fetches llama.cpp from GitHub at its first configure: check the
network (an HTTPS proxy is taken from `https_proxy`) and build again. **"... holds a llama.cpp this build did not
download"**: `third_party/llama.cpp` is a copy of another origin; move it away, or name it with
`-DLAMINA_GGML_DIR=<folder>`.

## Starting

**The PC is slow or stops responding for a minute or two while the model loads.** The engine reads the experts into
RAM and locks them there for the GPUs (the server's window says when). Wait; the next starts are faster while the
model file stays in the OS file cache.

**"not enough RAM: the model's experts take ...".** The experts stay locked in RAM while the model runs (the README's
table), and this PC's RAM cannot hold them beside the ~4 GiB the system needs. Use a smaller quant: IQ3_S takes ~28 GB
less than UD-Q4_K_XL. With GPUs of ample VRAM, a card's `vram-only=auto` in `"gpu"` keeps as many experts there
instead of in RAM as the RAM needs ([SETUP.md](SETUP.md#the-gpus)). Qwen3.6's experts take RAM only when they do not
all fit in VRAM: more GPUs (cards in `"gpu"`) or a shorter context avoid it.

**"RAM is tight: ...", or the PC pages heavily.** The model fits, but less than 6 GiB of RAM would stay available
beside it: other programs hold the rest, or the model leaves little. Windows pages them out to disk while the model
runs, and Linux may end the engine. Close other programs, or use a smaller quant.

**"...: not a model Lamina runs: its architecture is ...", or "...: a drafter, not a model: ..." (a LoRA adapter, a
control vector, an Engram table ...).** `-m` names a GGUF that is no model of these families: an image projector
(`mmproj-*.gguf`), a drafter (the draft layer, DSpark, another engine's MTP head or drafter), a file a model is run
with (a LoRA adapter: `--lora`; a control vector: `--control-vector`; Qwen3.8-Flash-Next's PLE table: `--ple-gguf`),
another model. Name the model's file, a split one's first shard or any other; the message says whether Lamina reads
the file beside a model, and how.

**"--mmproj ...: the projector of a model N wide, and this one is M", "... not an image projector", or "its type is
...".** The file `--mmproj` names belongs to another model, is no projector, or is a projector of a type or a tower the
image encoder does not read: name the one of the model's repository, or leave `--mmproj` out and the server takes the
one beside the model. `--check` says why a projector beside the model is not taken (`pictures: off (...)`), and
`--check -m URL` which of a repository's projectors the encoder reads.

**"unknown argument: ..." and "the engine of this model does not take an option of the command line (above)".** An
option on `lamina-server`'s command line that is not the server's goes to the engine, whose parser refused it: a typo,
a value it does not take, or an option of Qwen3.8-Flash-Next's engine given for Qwen3.6-35B-A3B or Qwen3.8-27B (or the
other way). `lamina-server --help` lists every option with the models that take it,
[PARAMETERS.md](PARAMETERS.md) with their defaults.

**"the config's "..." is no key lamina-server reads (lamina-server --help lists them): ignored".** A key of the config
file that no option names: a typo, or a key of another program or an earlier version. The server starts without it;
`lamina-server --help` lists every key beside its option.

**"the vision section (--vision-*) names no image projector".** Pictures were asked for (a `--vision-...` option, or a
`vision` section in the config) and no projector names the model's: give one with `--mmproj FILE`, or turn pictures
off with `--no-mmproj`.

**"port N is already in use - is Lamina (or another server) already running?".** Another server listens on the
config's port: close it, or start this one with `--port`.

**"config "vision": this lamina-server was built without the image encoder (LAMINA_BUILD_VISION=OFF)".** The
image encoder is a part of `lamina-server`, and this one was built without it. Build it with the encoder (the
default: `build.bat`, `./build.sh`) or take a release's, or take the `vision` section and `"--vision"` out of the
config to run without pictures. The `exe` a config of an earlier version's setup holds is not read
([SETUP.md](SETUP.md#updating)).

**"the vision encoder did not start: its GPU ... is not there (...)".** The config's `vision` section asks for a GPU
(`"gpu": true`) and its `cuda_device` names a card that is not in this PC (moved, replaced, or a config from another
PC), or the driver gives no GPU. `nvidia-smi -L` lists the cards with their UUIDs: name one there, or set `"gpu"` to
`false`.

**"the vision encoder did not start on GPU N, ...: its compute capability is ..., and Lamina needs 8.0 or newer".**
On a GPU the encoder runs on the engine's kernels, which need an RTX 30 series card or newer (up to 0.3.2 it
ran on older cards too). Name such a card in the `vision` section's `cuda_device`, or set `"gpu"` to `false`: the
encoder then runs on the CPU.

**"WARNING: no draft layer (...)".** Qwen3.8-Flash-Next's draft layer, `Lamina-Qwen3.8-Flash-Next-MTP.gguf`, is neither
beside the model nor beside the server, or the config's `args` name a file that is not there (`"--drafter", "<file>"`,
or `"--mtp"`, its earlier name), or a folder without the files of the layer's earlier layout (`dense.txt`, `dense.bin`,
`experts.bin`). The server runs, and the model writes new text at roughly half to two thirds of its speed. Run the
command the warning gives (`download-draft-layer.bat FOLDER`, `./download-draft-layer.sh FOLDER` on Linux: 0.8 GB), do
what it adds about the config, and start the server again ([SETUP.md](SETUP.md#a-model-file), The draft layer).

**"mtp: ... is not a draft layer this engine reads".** The file `--drafter` names is another GGUF (a model, a
projector) or the draft layer of another format: download the layer again with the script.

**"--drafter FILE is ... another model's: ...'s drafter is ...".** The file is a drafter, but another model's (Inkling
2's DSpark, Hy4-flash's DFly2, Qwen3.8-Flash-Next's draft layer): the line names the model's own. Every model takes its
drafter as `--drafter`, a config's `"drafter"`; `--mtp`, `--dspark` and `--dfly` are its earlier names.

**"this build has no kernels for its compute capability, ...".** The engine was built for other GPUs: a card added after
the build, or a release's programs on a card other than the RTX 30, 40 and 50 series. Build again from the source
(`build.bat`, `./build.sh`): the build is for this PC's GPUs. As the main GPU, such a card stops the start; as
a tier it is left out.

**"its compute capability is ..., and Lamina needs 8.0 or newer".** The card is older than the RTX 30 series: name
another first in `"gpu"`.

**"no CUDA device is visible (CUDA driver version is insufficient for CUDA runtime version)".** The NVIDIA driver is
older than the CUDA runtime linked into the programs: install a driver for CUDA 13 or newer.

**Linux: "version `GLIBC_2.34' not found", or "libssl.so.3: cannot open shared object file".** A release's programs
need glibc 2.34 and OpenSSL 3, as Ubuntu 22.04, Debian 12, RHEL 9 and newer systems have them (OpenSSL 3 is the
`libssl3` or `openssl-libs` package); an older system cannot run them.

**Linux: "mlock failed (raise ulimit -l)".** The engine locks the experts it keeps in RAM without pinning them, so
that the system does not page them out, and the locked-memory limit (`ulimit -l`) is below what they take: they stay
pageable. Raise it for the server: `memlock` in `/etc/security/limits.conf`, `LimitMEMLOCK=infinity` in a systemd
unit, `--ulimit memlock=-1` for a container.

**"Windows blocked ... Smart App Control".** Smart App Control on Windows 11 refuses programs that are not signed, and
the programs are built on this PC. Turn it off (Windows Security > App & browser control > Smart App Control
settings).

**"ExpertCache: cudaMalloc(...) failed: out of memory" although the GPU has VRAM free.** On Windows, every allocation
on a GPU also counts against Windows' commit limit (RAM and the page file), and the locked experts already take much of
it. Give the page file a fixed size, the same initial and maximum size (e.g. 65,536 MB: System > About > Advanced system
settings > Performance > Advanced > Virtual memory), and restart Windows: a page file that grows on demand may not grow
in time.

**"MiB of VRAM free with everything loaded - LOW: requests may stall".** The expert cache and the K/V left the GPU
almost no VRAM; Windows then pages the GPU's memory, and a request can stall. Raise the main GPU's `reserve` in the
config's `"gpu"` (its first card: `{"id": "...", "reserve": "1G"}`; default 700 MiB), or lower `--max-context`, and close
what else uses that GPU.

## While it answers

**Generation is far slower than expected, a few tokens a second.** Something else uses the main GPU: a display
plugged into it, a browser or a game drawing on it, another CUDA program. The engine's GPU waits on the host by
spinning, and a GPU shared in time slices stalls it. Plug the displays into another GPU (`-m` picks a GPU that drives
no display as the main one) and check `nvidia-smi` for other processes on it.

**It is slower than [COMPARISON.md](COMPARISON.md).** Those are the development PC's numbers: three GPUs holding most
experts, a 3090's VRAM bandwidth, 8 cores and DDR5-5600 for the rest. Fewer or smaller GPUs leave more experts to the
CPU, and the RAM's bandwidth sets its speed: run the RAM at its rated speed (EXPO or XMP), in both channels.
`lamina-server -m MODEL.gguf --bench` measures an install the way it is used, through the API, and writes a report
without personal details, to compare after a change or to attach to an issue
([SETUP.md](SETUP.md#a-model-file)).

**Answers turn into nonsense, or the PC crashes under load.** Check the RAM first: an overclock that passes everyday
use can flip bits when the CPU and the GPUs' copies read it at once, which on the development PC (DDR5-6000 XMP with
four DIMMs) gave crashes and changing answers. Test at the RAM's stock speed. A display driver reset followed by a
blue screen in `nvlddmkm` (0x116) points at the NVIDIA driver: try another version.

**"the engine stopped unexpectedly (exit code ...)".** The engine ended mid-request; the next request starts it again.
The end of its log says why. On Linux the usual cause is the RAM running out: `sudo dmesg | grep -i -E 'killed
process|out of memory'`.

**"request exceeds the context window".** The conversation and the answer's room are longer than the context. Start a
new conversation, or start the server with a larger `--max-context` (with `--kv-grow`, the configs' setting, a larger
context takes VRAM only as a conversation reaches it).

**"the engine said nothing for ... s during the request".** The engine and the server lost step, or the engine froze
(a driver stall): the server ends the engine and the next request starts it again. A prompt that takes longer than
that on a slow PC: raise `engine_silence_s` in the config (default 300).

**"this server was started without the vision encoder".** The server started without pictures: no image projector
beside the model (`--mmproj FILE` names one), `--no-mmproj`, or a config without a `vision` section. The server's
window said why at the start.

**Pictures take seconds.** A new picture is encoded once, in about half a second on the engine's GPU, which lends
the encoder the VRAM for it. Seconds a picture (about 9 for ~1,000 image tokens on an 8-core CPU) mean the CPU
encodes it: a config whose `vision` section has `"gpu": false` or no `gpu` (`--vision-gpu --vision-lend` on the
command line, or `"gpu": true, "lend": true` there), `--no-vision-gpu`, or a loan that found no memory, which the
engine's log says once ("the image encoder ... found no VRAM for a picture (...): the CPU encodes", with what the
engine answered; `--no-vision-lend` puts the encoder on a GPU with VRAM of its own). With batch slots (`--parallel`) a picture that arrives while another
request's long prompt is being read is encoded when that prompt is read: the engine's log then says how long the
encoder waited ("... waiting for the engine to lend"). The config's `vision.max_tokens` caps a picture's tokens
(1,024 in the configs).

**Two runs of one greedy request differ.** Where an expert is computed changes its rounding a little, and the GPUs'
expert caches change as their copies land, so runs can part at a near-tie
([HOW_IT_WORKS.md](HOW_IT_WORKS.md#exactness)).

## Reporting a problem

Open an [issue](https://github.com/eddoursul/Lamina/issues) with the engine's log and the server's output, the output
of `--check` (with the model's `-m` or `--config`), the commit (`git rev-parse --short HEAD`) and what the request was.
