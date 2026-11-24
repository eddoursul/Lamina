# Lamina

Lamina is an inference engine, forked from [Strata](https://github.com/Niko1221/Strata) and meticulously optimized to run on several consumer cards, 5070 Ti + two 3090s in my case. Single card setups benefit from some optimizations as well.

Supported models:
- [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (unsloth UD-IQ1_M through Q8_0, ISTA-DASLab's GSQ-RCO IQ3_S, Gyro-L)
- [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (unsloth)
- [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (unsloth)
- [Hy4-flash](https://huggingface.co/AngelSlim/Hy4-flash-GGUF) (AngelSlim)
- [Inkling 2 Mini](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) (unsloth's quants, both layouts of the files; on one GPU, a smaller one than the file with the experts in RAM and those used most on the card; the context also over other GPUs; its drafter; pictures)

Speeds against llama.cpp and Strata on one PC are in [docs/COMPARISON.md](docs/COMPARISON.md).

Support for other quants may be added upon request.

## Requirements

- **An NVIDIA GPU**, RTX 30, 40 or 50 series, or the workstation cards of those generations, with a driver for CUDA 13. The model should run on one GPU that nothing else uses: no display, no other program. Up to three more GPUs become further expert tiers, these may drive the display.
- **RAM** for the experts:

  | Model file | RAM taken |
  | --- | --- |
  | Qwen3.8-Flash-Next, [ISTA-DASLab's GSQ-RCO IQ3_S](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | ~53 GB |
  | Qwen3.8-Flash-Next, [unsloth's UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | ~81 GB |
  | Qwen3.6-35B-A3B, [unsloth's UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) | ~22 GB on one GPU, ~2 GB split over GPUs that hold it |
  | Qwen3.8-27B, [unsloth's UD-Q6_K_M](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | ~2 GB |
  | Inkling 2 Mini, [unsloth's UD-IQ4_XS](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) | ~1 GB (the model in ~16 GB of VRAM); ~14 GB on a smaller card |

- **An SSD**.
- **A 64-bit x86 CPU with AVX2**. 64-bit Arm support is built but not tested with a GPU. The CPU computes the experts no GPU holds, so its cores and the RAM's bandwidth set much of the speed.
- **Windows 10 / 11 or Linux** - no Apple hardware to test on, sorry.
- **To build:** [docs/SETUP.md](docs/SETUP.md#building).

## Install

Extract both archives of a [release](https://github.com/eddoursul/Lamina/releases) for your platform into one folder.

### MTP

When using Qwen 3.8 Flash Next (any supported quant), download Lamina's optimized draft layer [Lamina-Qwen3.8-Flash-Next-MTP.gguf](https://huggingface.co/eddoursul/Lamina-Qwen3.8-Flash-Next-MTP/blob/main/Lamina-Qwen3.8-Flash-Next-MTP.gguf) (0.8 GB):

```
download-draft-layer.bat MODEL_FOLDER      (Windows)
./download-draft-layer.sh MODEL_FOLDER     (Linux)
```

Load with `--mtp <file>` or the configuration line:
`"--mtp", "<folder>/Lamina-Qwen3.8-Flash-Next-MTP.gguf"`

`--mtp` does not accept other MTP formats.

## Run

```
lamina-server --help
lamina-server -m Qwen3.8-27B-UD-Q6_K_XL.gguf
lamina-server -m Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --mtp Lamina-Qwen3.8-Flash-Next-MTP.gguf
```

Generate a configuration file:
```
lamina-server -m Qwen3.8-27B-UD-Q6_K_XL.gguf <other parameters> --write-config qwen27b.json
; Run with the created config file
lamina-server --config qwen27b.json
```

How to connect clients, images, image generation, MCP tools and serving on a network:
[docs/USING.md](docs/USING.md); every option: [docs/PARAMETERS.md](docs/PARAMETERS.md).

## A fixed chat template

The server renders prompts with the chat template in the model's file. Lamina also includes
[froggeric's fixed template for Qwen 3.5, 3.6 and 3.8](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)
(v22.5) in `data/chat-template-froggeric-v22.5.jinja`. Qwen3.8-Flash-Next was observed to hallucinate less on short prompts with this template enabled.

To enable, add this key to the model's config or use `--chat-template-file` on the command line:

```json
"chat_template_file": "data/chat-template-froggeric-v22.5.jinja"
```

## Documentation

- [docs/SETUP.md](docs/SETUP.md): the build, a model file, the release binaries, the configs, choosing the GPUs,
  Qwen3.6 and Qwen3.8-27B.
- [docs/USING.md](docs/USING.md): the server, its API and web app, clients, pictures, image generation, MCP tools.
- [docs/PARAMETERS.md](docs/PARAMETERS.md): every option of the server and the engine, its default and what it
  trades.
- [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md): the engine.
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md): when something goes wrong.
- [docs/COMPARISON.md](docs/COMPARISON.md): speeds, memory and agreement with llama.cpp, against llama.cpp and
  Strata.
- [examples/](examples/README.md): the configs of the PC Lamina is developed on.
- [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md): reporting problems, pull requests, what the
  server exposes.

*The docs are not pruned from AI fluff yet.*

## License and credits

Lamina is under the MIT license ([LICENSE](LICENSE)), as Strata is. Some parts carry their own: `third_party/ggml`
(MIT, llama.cpp's), the web app's font, [Outfit](https://github.com/Outfitio/Outfit-Fonts) (SIL Open Font License 1.1,
`src/server/web/fonts/OFL.txt`), froggeric's chat template (Apache-2.0, unmodified,
`data/chat-template-froggeric-v22.5.LICENSE`), and what lamina-server takes from llama.cpp's checkout (its Jinja engine,
cpp-httplib, nlohmann/json, stb_image: MIT). It builds on:

- [Strata](https://github.com/Niko1221/Strata) by Niko1221 and its contributors: the engine Lamina grew from, with ideas
  Strata took from [Splash](https://github.com/incoai/splash), [ninfer](https://github.com/Neroued/ninfer) and
  [HyperQwen](https://github.com/syv-ai/HyperQwen).
- [llama.cpp and ggml](https://github.com/ggml-org/llama.cpp) (MIT): the CPU and CUDA kernels it shares, the prompt
  path's matrix products, the image encoder's tensor library and the arithmetic it follows (llama.cpp's multimodal
  library), the GGUF tools. The build downloads it at a pinned commit.
- The models: [Qwen](https://huggingface.co/Qwen)'s, under their licenses, in the quantizations of
  [ISTA-DASLab](https://huggingface.co/ISTA-DASLab) and [unsloth](https://huggingface.co/unsloth).
- The draft layer, [eddoursul/Lamina-Qwen3.8-Flash-Next-MTP](https://huggingface.co/eddoursul/Lamina-Qwen3.8-Flash-Next-MTP):
  Qwen's MTP layer, its dense part trained on this engine's output, under the Qwen license.
