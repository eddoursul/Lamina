# Parameters

Every option of `lamina-server`: the server's own, then the engine's it passes on. `lamina-server --help` lists the
same in short.

## How options are given

- **The server's options** are keys of the config file too: `--port 8090` is `"port": 8090`, and a section's key has
  the section's name before its own (`vision.max_tokens` is `--vision-max-tokens`). The tables name each key.
- **The engine's options** are the config's `"args"`, without their dashes: `--max-context 65536` is
  `"args": {"max-context": 65536}`, a switch `true` or `false` (`"kv-grow": false` is `--no-kv-grow`), a list for
  several values (`"control-vector-layer-range": [4, 44]`).
- **Precedence**: the command line over the config file over the preset `-m` builds for the model file (one of
  [examples/](../examples/README.md)). `--check` shows the engine's command line a start would run, `--write-config`
  writes the whole config.
- **Values**: a switch takes `--no-<name>` for off. A list takes commas (`--aliases a,b`) or JSON. Sizes are binary
  (`8G`, `700M`), times have their unit (`55us`, `1.5ms`, `1s`). A path on the command line starts from the folder the
  server was started in, one in a config from its `cwd`.
- **Models**: Flash-Next is Qwen3.8-Flash-Next, Qwen3.6 is Qwen3.6-35B-A3B, Inkling 2 is Inkling 2 Mini. Each model
  family runs on an engine path of its own, which takes its own options: an option its path does not take stops the
  start, the server saying which.
- **Defaults** are the engine's own; [the presets](#the-presets-engine-options) change some.
- **Trade-offs**: the figures were measured on the development PC (an RTX 3090 running the model, an RTX 5070 Ti and a
  second RTX 3090 as tiers, an 8-core Ryzen 7 9700X, DDR5-5600) and move with the GPUs, their links and the RAM.
  Measure an option on your PC before keeping it: `lamina-server -m MODEL.gguf --bench` before and after.

## Commands

| Command | What it does |
| --- | --- |
| `--config FILE` | Start from this config file. |
| `--write-config FILE` | Write the config the command line runs (the preset's, the file's and the command line's settings in one) to FILE and stop. An earlier FILE is kept as `FILE.bak`, and its keys outside the preset stay. |
| `--check` | Say what the PC has and, with `-m` or `--config`, what the model needs, where it would run and whether it starts (exit code 1: it would not). Nothing is loaded. |
| `--bench` | Measure the model through the server's own API on a port of its own: short questions, prompts of ~4K and ~16K tokens and a coding agent's session, three runs each (3-10 minutes); write a report into `bench-results/` beside the program, with personal details removed, and stop. |
| `--bench-url URL` | The same against a Lamina server that runs; nothing is loaded. |
| `--bench-runs N` | Runs per test (default 3). |
| `--bench-clients N` | Also N requests at once (batch slots run them together). |
| `--bench-long` | Also a prompt of ~32K tokens. |
| `--bench-think` | Thinking on (the suite turns it off, so that the answers stay short and comparable). |
| `--bench-out DIR` | Where the report's folder goes. |
| `--bench-compare OLD [NEW]` | An earlier report (its folder or `results.json`) beside this run's; with NEW and no `--bench`, two reports side by side. |
| `-h`, `--help` | The options in short. |
| `-v`, `--version` | The program's version. |

## The server's options

### The model

| Option | Key | Default | What it does |
| --- | --- | --- | --- |
| `-m`, `--model GGUF` | `model` | | The model's file, a split one by any shard. Without the config's `"args"`, the file's preset supplies the engine's options, and the server picks the GPUs and finds the files beside the model ([SETUP.md](SETUP.md#a-model-file)). |
| `--mmproj FILE` | `vision.mmproj` | with `-m`: the `mmproj*.gguf` made for the model beside it | The image projector: pictures on ([USING.md](USING.md#pictures)). |
| `--no-mmproj` | `vision` (`false`) | | No pictures. |
| `--vision-gpu` | `vision.gpu` | on with `-m` | The image encoder on a GPU; `--no-vision-gpu`: on the CPU, 8-10 s for a picture of ~1,000 tokens instead of 0.3-0.6 s. |
| `--vision-lend` | `vision.lend` | on with `-m` | The encoder holds no VRAM between pictures: for each new one it borrows VRAM the engine lends on its card ([SETUP.md](SETUP.md#a-model-file) says what each model lends). `--no-vision-lend`: it keeps 0.7-1.2 GB of VRAM and encodes a picture in ~0.2 s. |
| `--vision-cuda-device UUID` | `vision.cuda_device` | with `lend`, the engine's main GPU; else the card with the least VRAM that drives no display | The encoder's GPU (`nvidia-smi -L` lists the UUIDs). |
| `--vision-q8 mmq\|fp16` | `vision.q8` | `mmq` | A Q8_0 projector's products on a GPU: as llama.cpp computes them (the inputs rounded to 8 bits), or in FP16, nearer the unquantized projector. The answers measured the same. |
| `--vision-threads N` | `vision.threads` | with `-m`: the logical cores | The encoder's threads on the CPU. |
| `--vision-max-tokens N` | `vision.max_tokens` | the projector's own, 4,096; Flash-Next's, Qwen3.6's and Qwen3.8-27B's presets 1,024 | A picture's tokens at most: a larger one is scaled down. More tokens show more detail and take longer to read. |
| `--vision-min-tokens N` | `vision.min_tokens` | the projector's own | A picture's tokens at least. |
| `--vision-model GGUF` | `vision.model` | | Taken and not read (earlier versions read the encoder's vocabulary from it). |
| `--model-name NAME` | `model_name` | with `-m`: the file's name | The model's name in `/v1/models`. Any name is served. |
| `--aliases A,B` (`--alias`) | `aliases` | | More names the model is listed under. |
| `--lora-models JSON` | `lora_models` | | Model names that run with the engine's LoRA adapters at given scales: `{"NAME": [{"id": 0, "scale": 1}]}` ([USING.md](USING.md#lora-adapters)). |
| `--arch ARCH` | `arch` | the file's | The architecture whose model card gives the sampling a request leaves out. |
| `--chat-template-file FILE` | `chat_template_file` | the model file's | A chat template in its place ([the README](../README.md#a-fixed-chat-template)). |

### Running the engine

| Option | Key | Default | What it does |
| --- | --- | --- | --- |
| `--gpu CARD[,key=value...]` | `gpu` | with `-m`: the RTX 30 or newer with the most VRAM that drives no display, then `auto` | The GPUs in their roles: [GPUs](#gpus). |
| `--parallel N` | `parallel` | 1 | Qwen3.6 and Qwen3.8-27B: up to N requests (8 at most) decoding together in batch slots, on one GPU. Each slot holds a session's K/V. |
| `--replicas N\|JSON` | `replicas` | | One engine on each of N groups of the `gpu` list, the requests spread over them; `[{"gpus": [0, 1]}, {"gpus": [2, 3]}]` names the groups. Each engine pins its own experts in RAM. |
| `--replica-cpus` | `replica_cpus` | on | Linux: each replica on CPU cores of its own. |
| `--replica-load-tokens N` | `replica_load_tokens` | 3000 | A request in flight counts as N prompt tokens when a replica is picked. |
| `--expert-profile-save FILE` | `expert_profile_save` | with `-m`: `<model>-learned.bin` beside the program | Flash-Next, Hy4-flash and Inkling 2 (with experts in RAM): the experts' use the run counted, blended with the ranking it started from, written at exit and between requests; the next start fills the GPUs from it. On the development PC, after a session of code and reasoning requests, fixed text took 9-14% less time on code (-4 to +1% on thinking and edits). |
| `--expert-profile-save-every MIN` | `expert_profile_save_every` | 1 with `-m`; else Flash-Next 10, the others 1 | Minutes between those saves (0: at exit only). |
| `--engine-silence-s S` | `engine_silence_s` | 300 | A request ends when the engine says nothing for S seconds, and the engine starts again (0: no limit). |
| `--effort-position start\|end` | `effort_position` | `start` | `end`: a request's non-default reasoning effort goes into a short turn before the answer instead of the top of the prompt, so that a request that only changes the effort reuses the cached conversation. |
| `--log FILE` (`--log-file`) | `log` | `<model>.log` beside the config file, or beside the program with `-m` alone | The engine's log. |
| `--cwd DIR` | `cwd` | the folder the server starts in | The engine's folder, where the config's relative paths start. |
| `--env NAME=VALUE` | `env` | | A variable of the engine's environment (again for more). |
| `--lib-dirs D1,D2` | `lib_dirs` | | Folders searched first for the CUDA libraries. |

### The network and access

| Option | Key | Default | What it does |
| --- | --- | --- | --- |
| `--host HOST` | `host` | `127.0.0.1`: this PC only | `0.0.0.0` opens the server to the network: set an API key ([USING.md](USING.md#from-other-devices)). |
| `--port N` | `port` | 8095; with `-m` the preset's, 8081-8087 | The port (0: one the system picks). |
| `--api-key KEY` | `api_key` | `$LAMINA_API_KEY` | The key the API asks for (`Authorization: Bearer KEY` or `x-api-key`); several comma-separated. |
| `--cors-origins O1,O2` | `cors_origins` | none | Web pages' origins allowed to call `/v1/*` (`*`: any). |
| `--trusted-origins O1,O2` | `trusted_origins` | | Origins trusted as the server's own pages: the web app's settings and MCP tools through a proxy or a tunnel. |
| `--allowed-hosts H1,H2` | `allowed_hosts` | `$LAMINA_ALLOWED_HOSTS` | More `Host` names answered without an API key (`*`: any). |
| `--offline` | `offline` | off | No image URLs fetched, no MCP servers started, no `before_load` command run. |
| `--api-monitor` | `api_monitor` | off | The request monitor at `/api-monitor`: the last 100 prompts and answers, in memory. |
| `--mcp-servers JSON` | `mcp_servers` | | MCP servers for the web app's chat, in Claude Desktop's form (`{"name": {"command": ...}}`). |
| `--mcp-config FILE` | `mcp_config` | | MCP servers from a file in that form (`{"mcpServers": {...}}`). |
| `--mcp-timeout-s S` | `mcp.timeout_s` | 60 | A tool call's time limit. |
| `--mcp-start-timeout-s S` | `mcp.start_timeout_s` | 120 | An MCP server's time to start. |
| `--mcp-max-result-chars N` | `mcp.max_result_chars` | 20000 | A tool result's characters at most. |
| `--mcp-max-rounds N` | `mcp.max_rounds` | 8 | Tool-call rounds of one chat reply at most. |

### Loading and stopping

| Option | Key | Default | What it does |
| --- | --- | --- | --- |
| `--lazy-load` (`--lazy`) | `lazy_load` | off | Start without the model: the first request loads it. |
| `--idle-unload-s S` (`--idle-unload`) | `idle_unload_s` | 0: never | Unload the model after S seconds without requests; the next request loads it again, quickly while the model file stays in the OS file cache. |
| `--min-free-vram-mib N` | `min_free_vram_mib` | 0 | Load an unloaded model only when N MiB of VRAM are free (waiting up to 15 s), else answer 503. |
| `--before-load CMD` | `before_load` | | A command run before the model loads again, such as one that unloads another server's model (a JSON list: the program and its arguments). |
| `--before-load-timeout-s S` | `before_load_timeout_s` | 120 | That command's time limit. |
| `--before-load-ignore-failure` | `before_load_ignore_failure` | off | Load the model even when that command fails. |
| `--before-load-strict` | `before_load_strict` | off | Answer 503 when that command fails. |
| `--max-queue N` | `max_queue` | no limit | Inference calls that may wait beyond the running ones, 0-4096 (0: a busy engine refuses the next). |
| `--queue-timeout S` | `queue_timeout` | | The longest wait for the engine, 0.1-600 s (with `--max-queue`). |
| `--shutdown-timeout S` | `shutdown_timeout` | | At a stop, running requests may finish for up to S seconds (0-300). |
| `--ready-file FILE` | `ready_file` | | A JSON record of this instance once it serves, for a supervisor (an absolute path, a new file). |
| `--open-browser` (`--open`) | `open_browser` | off | Open the web app in the browser once the model is ready. |
| `--prevent-sleep` | `prevent_sleep` | off | Windows: the PC stays awake while a request runs or waits. |

### Request defaults

The settings of a request that leaves them out ([USING.md](USING.md#sampling)). Without them, sampling follows the
model card's settings for the request's thinking state (Flash-Next thinking: temperature 1.0, top_p 0.95, top_k 20;
not thinking: 0.7, 0.8, 20, presence_penalty 1.5). The presets sample at temperature 0.7, top_p 0.95, top_k 20, min_p
0, presence_penalty 0; Hy4-flash's at 0.9, 1.0, 64 and Inkling 2's at 1.0, 1.0, 64.

| Option | Key | Default | What it does |
| --- | --- | --- | --- |
| `--temperature T` (`--temp`) | `sampling.temperature` | | 0: greedy. Sampled, fewer drafts pass than greedy: Flash-Next thinks 9-10% slower and writes code 5-9% slower. |
| `--top-p P` | `sampling.top_p` | | |
| `--top-k K` | `sampling.top_k` | | 1-64. |
| `--min-p P` | `sampling.min_p` | | |
| `--presence-penalty X` | `sampling.presence_penalty` | | |
| `--frequency-penalty X` | `sampling.frequency_penalty` | | |
| `--repetition-penalty X` (`--repeat-penalty`) | `sampling.repetition_penalty` | 1: off | |
| `--penalty-last-n N` (`--repeat-last-n`) | `sampling.penalty_last_n` | | The tokens the repetition penalty looks back over. |
| `--seed N` | `sampling.seed` | | |
| `--reasoning-effort LEVEL` | `reasoning_effort` | the model's own: `high` | `none`, `low`, `medium` or `high`. |
| `--reasoning-budget-tokens N` (`--reasoning-budget`) | `reasoning_budget_tokens` | 0: no cap | The thinking at most: the server then ends it with a short wrap-up, and the model answers from there. |
| `--answer-room` | `answer_room` | on | With thinking on, no budget and a `max_tokens` of 1,024 or more, the thinking is wrapped up when a quarter of `max_tokens` (at least 512 tokens) is left, so that the answer has room; without `max_tokens`, at three quarters of the context. |
| `--fit-max-tokens` | `fit_max_tokens` | on | A `max_tokens` past the context is shortened to the room left; off: a 400, as llama.cpp answers. |
| `--anthropic-thinking model\|on_request` | `anthropic_thinking` | `model` | An Anthropic request that asks for no thinking: `model` thinks as the model's template does, `on_request` gets none (Claude Code's short helper calls want it). |

### Replies

| Option | Key | Default | What it does |
| --- | --- | --- | --- |
| `--tool-call-recovery` | `tool_call_recovery` | off | Also read tool calls the model writes in a few other forms (`<parameter=NAME>`, JSON inside `<tool_call>`, a bare `<function=NAME>` block). |
| `--repeat-stop-tokens N` | `repeat_stop_tokens` | 256 | A reply that repeats one token N times ends as `"length"` (0: off). |
| `--reasoning-close-retry` | `reasoning_close_retry` | off | A thinking that ended with no answer is closed and the reply continues once. |
| `--literal-markers` | `literal_markers` | on | An end-of-turn token drawn where the text quotes it (a code span or fence, after an opening quote, in a tool call's open parameter, mid-thinking) becomes the marker's text and the reply goes on, up to 64 times a reply; off: the turn ends there. |
| `--short-opening-tokens N` | `short_opening_tokens` | 0: off | A conversation's first message of up to N tokens is read at `medium` effort: a bare greeting at the top effort can make the model invent instructions. The next turn's prompt then starts differently and is read again. |
| `--reasoning-loop-recovery stop\|recover\|false` | `reasoning_loop_recovery` | `false` | A reply whose reasoning repeats whole passages ends (`stop`) or restarts (`recover`). |
| `--codex-compaction-cache` | `codex_compaction_cache` | off | Codex's compaction requests rendered with the conversation's tools, so that they reuse the cached conversation. |
| `--codex-thread-titles` | `codex_thread_titles` | off | Codex's thread-title requests answered without the engine. |
| `--env-to-user` | `env_to_user` | off | opencode's `<env>` block moved to the first user message, so that the system turn, and the cached prompt with it, stays the same between sessions. |

### Image generation

The `images` section ([USING.md](USING.md#image-generation)).

| Option | Key | Default | What it does |
| --- | --- | --- | --- |
| `--images-exe FILE` | `images.exe` | | stable-diffusion.cpp's `sd-cli`. |
| `--images-diffusion-model FILE` | `images.diffusion_model` | | The denoiser. |
| `--images-vae FILE` | `images.vae` | | The VAE. |
| `--images-llm FILE` | `images.llm` | | The text encoder. |
| `--images-model-name NAME` | `images.model_name` | `qwen-image-2.1-turbo` | The image model's name in the API. |
| `--images-backend NAME` | `images.backend` | | `sd-cli`'s backend, `cuda` or `vulkan`: the variable `--images-gpu` sets for it. |
| `--images-quant Q` | `images.quant` | | The denoiser's quant, reported with each job's figures. |
| `--images-args A,B` | `images.args` | | `sd-cli`'s own options (a JSON list for ones with commas). |
| `--images-native` | `images.native` | off | Lamina's runner of the denoiser inside `sd-cli` (a CUDA build with its patch). |
| `--images-gpu N` | `images.gpu` | | The GPU `sd-cli` runs on (its `nvidia-smi` number or UUID). |
| `--images-gpu-policy auto\|keep\|unload` | `images.gpu_policy` | `auto` | The text model during a job: kept loaded, stopped, or by the free VRAM. |
| `--images-reload lazy\|eager` | `images.reload` | `lazy` | A model stopped for a job loads again with the next request, or at the job's end. |
| `--images-need-mib N` | `images.need_mib` | by the job's size | The VRAM a job takes. |
| `--images-out-dir DIR` | `images.out_dir` | a temporary folder | Where the images go. |
| `--images-keep-files N` | `images.keep_files` | 64 | The newest images kept there. |
| `--images-steps N` | `images.steps` | 8 | The denoising steps of a request that names none. |
| `--images-sigmas S` | `images.sigmas` | Turbo's schedule's | The schedule's sigmas. |
| `--images-max-n N` | `images.max_n` | 4 | Images a request at most. |
| `--images-max-side N` | `images.max_side` | 2048 | An image's side at most. |
| `--images-timeout-s S` | `images.timeout_s` | 3600 | A job's time limit. |
| `--images-lib-dirs D1,D2` | `images.lib_dirs` | | Folders searched first for `sd-cli`'s libraries. |
| `--images-env NAME=VALUE` | `images.env` | | A variable of `sd-cli`'s environment (again for more). |

### Tests

| Option | Key | What it does |
| --- | --- | --- |
| `--engine lamina\|mock` | `engine` | `mock`: a scripted engine in place of the model, for the server's tests. |
| `--tokenizer GGUF` | `tokenizer` | The mock engine's tokenizer and chat template: a model's file. |
| `--script TEXT` | `scripts` | The mock engine's answer; again for more, the requests get them in turn. |
| `--script-cycle` | `script_cycle` | The scripts again from the first, not the last repeated. |
| `--script-delay S` | `script_delay` | Seconds between the mock engine's tokens. |

## The engine's options

### The presets' engine options

What `-m` sets for each model file beside the defaults below (the examples' configs):

| Preset | `"args"` |
| --- | --- |
| Flash-Next, ISTA-DASLab's IQ3_S (`iq3_s`) | `--mtp` (the draft layer beside the model), `--kv k8v4`, `--prompt-shift 0.2` |
| Flash-Next, the others (`ud-q4_k_xl`) | `--mtp`, `--kv int8`, `--head-split 0` |
| Qwen3.6 (`qwen36-ud-q4_k_xl`) | `--max-context 65536`, `--drafts-per-step 2` |
| Qwen3.8-27B (`qwen38-27b-ud-q6_k_m`) | `--max-context 262144`, `--prefill 1536`, `--kv q8_0`, `--drafts-per-step 7`, `--spec-min-p 0.3`, `--draft-vocab 65536` |
| Hy4-flash (`hy4-flash-q4_k_m`) | `--max-context 524288`, `--drafts-per-step 4`, `--spec-min-p 0.3` |
| Inkling 2 (`inkling2-mini-ud-iq4_xs`) | none; the server finds its drafter (`--dspark`) |

### GPUs

`--gpu CARD[,key=value...]`, again for each GPU, names the GPUs in their roles; a config's `"gpu"` holds the same
cards (`["RTX 3090", {"id": "GPU-f26e9ba6", "clocks": "1395-2100/9751"}]`). CARD is a name only one card has
(`"RTX 5070 Ti"`), a UUID or its start (`GPU-f26e9ba6`, `nvidia-smi -L` lists them), a PCI bus id (`10:00.0`), or a
number as `nvidia-smi` numbers the GPUs (it moves when a card is added); `auto` after the first card is every other
GPU, the fastest link first. Numbers alone (`--gpu 0,1`) are the cards the engine sees, the first its main one.

- **The first card runs the model.** Nothing else may use it: a display or another program on it makes the driver
  share it in time slices, which slows generation to a few tokens a second.
- **Flash-Next**: the cards after it are expert tiers, in their order (up to three), each holding the next experts in
  line and computing them beside the CPU; the first tier also computes the prompts' experts the main GPU lacks and
  part of the output head. A tier may drive the display. A tier on a slow link (x1, x4) still pays: an expert crosses
  it once and stays.
- **Hy4-flash**: the cards after it are expert tiers too, sized by their free VRAM; their settings are not used.
- **Qwen3.6 and Qwen3.8-27B**: the cards are the layer split's GPUs in their order (`--layer-split`); their settings
  are not used.
- **Inkling 2**: a later card holds the context's cells the first has no room for; with `layers=` it runs a run of
  the model's layers, with `cache=` a copy of the experts in RAM.

A card's settings, each optional:

| Setting | Models | Default | What it does, and its trade-off |
| --- | --- | --- | --- |
| `cache=auto\|SIZE\|N` | Flash-Next, Inkling 2 | `auto`: all but its reserve | Its VRAM for experts (`22G`); the first card's also N expert slots. Inkling 2: the first card's experts at most (`6G`; below their size the rest stay in RAM); a later card's `cache=` holds a copy of every expert in RAM and computes a token's entries of them in place of the CPU, all or none: with drafts 28-42% faster on an RTX 3090 that holds 6 of the 13 GiB of experts. |
| `reserve=SIZE` | Flash-Next, Inkling 2 | the first card 700M (Inkling 2: 1.5G where the desktop draws on it); a tier 2G where the desktop draws on it, else 512M | The VRAM it leaves free. Less gives experts more room and the desktop or other programs less; under ~128 MiB free (the log warns) Windows pages the GPU's memory and a request can stall. |
| `vram-only=SIZE\|auto` | Flash-Next | none | The profile's experts the card keeps only in its VRAM, out of the RAM: for a model whose experts outgrow the RAM. `auto`: as many as keep them within the RAM less 4 GiB, with 6 GiB of it left available. The first card and the first two tiers, not with `sleep`. Those experts never move, so the card's adaptive cache has fewer slots to follow the conversation with. |
| `prefetch=SIZE` | Flash-Next, a tier | 12M over a link of 20 GB/s or more, else none | Each layer's likeliest experts no GPU holds, copied to the tier ahead while the RAM is idle. The copies take RAM bandwidth the CPU needs: 16M was up to 3.6% slower than 12M, and 4M over an x4 link made code 6% slower. |
| `clocks=MIN-MAX/MEM` | Flash-Next, a tier; Inkling 2, any card | none: the driver's | Its graphics and memory clocks in MHz, locked through NVML while the engine works (`1395-2100/9751` for an RTX 3090; a memory clock of 0 stays the driver's; `nvidia-smi --query-gpu=name,clocks.max.graphics,clocks.max.memory --format=csv` reads a card's highest). A tier with little work, which the driver clocks down between its shares, then answers in time: the development PC's third GPU without it made rounds 4-22% longer. A locked card draws its awake power (~110 W against ~12 W for an RTX 3090) until `--clocks-hold` after a request. Needs administrator rights (root on Linux); an engine ended by force leaves the lock on (`nvidia-smi -rgc -rmc` ends it). |
| `sleep=TIME` | Flash-Next, a tier after the first | none | The tier sleeps (its experts go to the tiers before it and the CPU, its lock is released) while the CPU would need less than TIME a layer without it, averaged over a few seconds, and wakes when the text turns heavy. Saves the card's awake power for some speed: on the development PC IQ3_S at `55us` slept through 59-63% of an agent's session for 10.5% fewer joules a token within ~1% of the time; code and edits kept the card awake. TIME is a model file's and a PC's: start from what `lamina --stats` shows on thinking text, and compare whole sessions. |
| `layers=A-B` | Inkling 2, a later card | none (below) | The card runs those layers (`6-17`, the looped ones: all or none) with their weights and their part of the context; the answers are one card's to the bit. Without it, the engine does this by itself where the first card would keep under 80% of the experts. An RTX 3090 with the layers on an RTX 5070 Ti generated 15-25% faster than the 3090 alone and read prompts 30-50% faster. |

### The tiers and the CPU

| Option | Models | Default | What it does, and its trade-off |
| --- | --- | --- | --- |
| `--clocks-hold TIME` | Flash-Next, Inkling 2 | `5s`; Inkling 2 `1s` | How long the cards' clock locks stay after a request. A shorter hold saves the locked power in pauses; a request after a longer pause starts on a card still at its idle clocks (a lock is at speed ~0.6 s after it is asked for). With `--clocks-lead 1s` at `1s` an agent's session took 3.2-3.5% less energy, no measurable time. |
| `--clocks-lead TIME` | Flash-Next | off: from a batched prompt part's last chunk on | The locks are asked for this long before a batched prompt part ends, by its progress. Too short a lead starts generation on a card at its idle clocks. |
| `--tier-min SIZE` | Flash-Next | `4M` | A layer's missed experts from which the tiers take their shares: below it the CPU is quicker than the tiers' ~100 us round trip. |
| `--no-tier-adapt` | Flash-Next | off | The tiers keep the experts the start placed, while the main GPU's go on adapting. |
| `--head-split F` | Flash-Next | 0.45 | With a tier: the first tier's share of the output head's rows, read in parallel with the main GPU's. IQ3_S at 0.45: code 5.8% faster than with the whole head on the main GPU; a tier busy at a window's end loses instead (UD-Q4_K_XL edits 6.5% faster at 0, which its preset sets). |
| `--pcie-frac F` | Flash-Next | 0 | The share of a layer's experts no GPU holds that the main GPU reads from RAM over its link instead of the CPU computing them. Over an x4 link the CPU is sooner. |
| `--pcie-mode kernel\|dma\|direct` | Flash-Next | `kernel` | How that share reaches the main GPU: a copy kernel in the window's graph, the copy engine, or the expert kernels reading the mapped RAM. |
| `--pool-workers N` | every model (`--cpu-threads` too, but on Flash-Next) | 0: a worker a physical core (Flash-Next: but the one that drives the GPUs) | The threads that compute the experts no GPU holds. They are bound by the RAM's bandwidth: SMT threads did not help. |

### The context

| Option | Models | Default | What it does, and its trade-off |
| --- | --- | --- | --- |
| `--max-context N` | every model (`--ctx` too on Qwen3.6 and Qwen3.8-27B) | Flash-Next: 262,144, the trained context; Qwen3.6, Qwen3.8-27B: 32,768; Hy4-flash: 8,192; Inkling 2: what the free VRAM holds K and V for, up to 1,048,576 | The context in tokens. A request longer than it is refused. With `--kv-grow` a larger context takes VRAM only as conversations reach it; Qwen3.6 and Qwen3.8-27B allocate the whole K/V at the start (Qwen3.6 ~22 KB a token in fp16, Qwen3.8-27B ~68 KB, ~36 KB in q8_0). Inkling 2 not set: what the free VRAM holds, at least 32,768 tokens; set: a sequence ends where the VRAM does, and with `--no-kv-grow` a start past it is refused. Flash-Next at most 1,466,368 on a GPU with 99 KiB of shared memory a block (630,784 on 48 KiB). |
| `--kv FORMAT` | every model | `fp16` | The K/V's format. Flash-Next: `fp16`, `int8` (8 bits), `k8v4` (8-bit keys, 4-bit values) or `q4_0` (4 bits): at 262,144 tokens 6.2, 3.1, 2.4 and 1.7 GiB, the VRAM saved going to experts, the speed within 2%. Against llama.cpp `int8` is at the rounding's floor, `k8v4` 2-3 times it, `q4_0` further; `fp16` gained nothing measurable even after 128K tokens. The other models: `fp16` or `q8_0` (about half): the Qwen3.8-27B's full context fits two GPUs only in `q8_0`, Hy4-flash's 524,288 tokens take 4.4 GiB in it instead of 7.5 (more experts on the main GPU), Inkling 2's 17.0 KiB a token instead of 32.0. |
| `--kv-grow` | Flash-Next, Hy4-flash, Inkling 2 | on | The K/V takes VRAM only for the cells the requests reach: the experts hold the rest and give it back as a conversation grows. `--no-kv-grow`: the whole context's at the start, at the cost of experts on the GPU (on Flash-Next up to 2-6% longer rounds). |
| `--rope-scaling none\|linear\|yarn` | Flash-Next | `none` | Rotary positions for a context past the trained 262,144 tokens, as llama.cpp computes them. Fixed for the run, so it changes the output at every length: at 524,288 tokens with `yarn` and `--rope-scale 2` both models found passphrases placed at 65K, 283K and 515K tokens of one prompt. |
| `--rope-scale F` | Flash-Next | | The context's extension factor (2: 524,288 tokens). |
| `--rope-freq-base N` | Flash-Next | 0: the model's, 1e7 | The rotation base. |
| `--rope-freq-scale F` | Flash-Next | 0: 1 / `--rope-scale` | The frequency scale. |
| `--yarn-orig-ctx N` | Flash-Next | 0: 262,144 | YaRN's trained context. |
| `--yarn-ext-factor F` | Flash-Next | below 0: 1 with `yarn` | YaRN's extrapolation mix. |
| `--yarn-attn-factor F` | Flash-Next | 1 | YaRN's attention factor. |
| `--yarn-beta-fast F`, `--yarn-beta-slow F` | Flash-Next | 32, 1 | YaRN's correction bounds. |

### Prompts

| Option | Models | Default | What it does, and its trade-off |
| --- | --- | --- | --- |
| `--prefill N\|auto` | every model (`--chunk` too on Qwen3.6 and Qwen3.8-27B) | Flash-Next: `auto`, 16,384, halved until its buffers fit the expert caches' lendable slots; Qwen3.6, Qwen3.8-27B: `auto`, 2,048 with a layer split or experts in RAM, else the largest of 4,096, 2,048 and 1,024 that fits beside the experts; Hy4-flash: 4,096; Inkling 2: 1,024 | A prompt's chunk in tokens. Every chunk reads each expert its tokens use once, so larger chunks read long prompts faster and take more VRAM (Flash-Next's borrow expert slots while a prompt runs). On Flash-Next 32K chunks read 32K-131K prompts 1-3.5% faster; 8,192 read IQ3_S's 12K-32K prompts 5-18% sooner but UD-Q4_K_XL's 16-18% later. Hy4-flash and Inkling 2 at 0: every prompt through the generation's kernels (slow). |
| `--feed-max N` | Flash-Next | 46 | A new prompt part of up to N tokens, such as the next turn's message, runs through verify windows instead of the batched prompt path, whose call has a fixed cost (~0.2 s): the two break even at ~45 tokens. |
| `--prompt-shift F` | Flash-Next | 0 | With a tier, in chunks of 8K tokens or more: the first tier also computes the likeliest share F of each layer's experts the main GPU holds. IQ3_S's prompts 1-2% faster at 0.2 (its preset), UD-Q4_K_XL's none. |
| `--prompt-split L[,M]` | Flash-Next | 0: off | With a tier: a prompt of more than one chunk runs in two stages, the first tier running each chunk's layers L and up while the main GPU runs the next chunk. It pays where the tier reads a chunk about as fast as both GPUs together, as the development PC's RTX 5070 Ti beside its RTX 3090: at 26, prompts of 32K-128K tokens 1.24-1.55x as fast with IQ3_S, 1.13-1.43x with UD-Q4_K_XL, for 0.7 GiB more pinned RAM, ~0.3 GiB of the tier and 6-8 GiB of its cache lent while such a prompt runs. `L,M` with two or more tiers: three stages, the tier after the first running the layers L to M-1 (with one tier, two stages split halfway between): at 17,35 IQ3_S's 64K / 128K prompts another 14% / 21% faster, UD-Q4_K_XL's no faster, for ~0.9 GiB more pinned RAM and that tier at work through a prompt (170-200 W). The best layer differs between generate mode and the server: measure it through the server. |
| `--prompt-split-from N` | Flash-Next | 0: off | With `--prompt-split`: a prompt part of N tokens or more that fits one chunk is read as two halves through the stages. The stages' fixed cost sets N: ~9K for IQ3_S (a 16K part 17-19% sooner, an 8K one 3% later), no size for UD-Q4_K_XL. |
| `--prompt-tiers N` | Flash-Next | 1 | With two or more tiers: the first N compute their experts in prompts, where the later tiers' experts otherwise stream to the first. At 2, UD-Q4_K_XL's parts of 2-4K tokens read 15-25% sooner and 64K-128K prompts 6%, IQ3_S's 4-10% later (its small experts stream sooner than rows cross two x4 links); ~0.65 GiB of that card's VRAM, which then works through a prompt (~140 W). |
| `--no-prefill-borrow` | Flash-Next | off | The prompt path's buffers keep VRAM of their own instead of borrowing the expert caches' last slots while a prompt runs: fewer experts on the GPU, nothing to copy back after a prompt. |
| `--no-warmup` | Flash-Next, Qwen3.6, Qwen3.8-27B | off | No warm-up before the server is ready: the start is 1.2-1.4 s shorter, and the first request loads the prompt path's kernels (IQ3_S's first 2K prompt 1.1-1.4 s instead of 0.76). |

### Drafts

Each step of generation verifies the last token and drafts after it in one pass; every token, drafted or not, is the
one the model itself picks.

| Option | Models | Default | What it does, and its trade-off |
| --- | --- | --- | --- |
| `--mtp FILE` | Flash-Next | the server: `Lamina-Qwen3.8-Flash-Next-MTP.gguf` beside the model, else beside the program | The draft layer ([the README](../README.md#mtp)), or a folder of its earlier layout. It takes 1.1 GiB of VRAM; without it a step yields one token, or prompt lookup's drafts: new text at 44-61% of IQ3_S's speed with it, 50-68% of UD-Q4_K_XL's, an edit that quotes its prompt at 92-96%. |
| `--spec T` | Flash-Next | 4 | Steps of up to T tokens, the last one and T-1 drafts (2-8). Every further token routes its own experts: 5 and 6 were slower than 4. |
| `--spec-min-p P` | every model | 0.5; Qwen3.6, Qwen3.8-27B, Hy4-flash: 0 | Drafts while their joint probability under the drafter stays at P or more (Flash-Next and Inkling 2 weigh a sampled request's drafts by their chance to pass at their positions' draws). Lower lets more drafts in, which pay where they pass and cost where they do not: Flash-Next IQ3_XXS thinking at 0 / 0.3 / 0.5 / 0.7: 60.5 / 68.0 / 73.5 / 71.8 tokens/s; the Qwen3.8-27B's and Hy4-flash's presets 0.3. |
| `--spec-lookup N` | Flash-Next, Hy4-flash, Inkling 2 | 16 | Prompt lookup: where the text's last N tokens or more occurred before, up to 7 of the tokens that followed there are the step's drafts (0: off). An answer that quotes its prompt, such as a file in an edit, passes ~7.5 tokens a step (Flash-Next: edits 23-26% faster); matches of 8 tokens drafted wrongly on thinking text, of 16 never. 22-28 MiB of VRAM. |
| `--mtp-window N` | Flash-Next | 32768 | The context cells the draft layer attends to (0: every cell). Fewer pass fewer drafts after long prompts: 2,048 / 8,192 cells passed 2.59 / 2.64 tokens a step after a 16K prompt instead of 2.82. |
| `--drafts-per-step N` | Qwen3.6, Qwen3.8-27B, Hy4-flash, Inkling 2 | 0: none; Inkling 2: 7 | At most N drafts a step (0-7; Hy4-flash with `--dfly` 0-5) of the model file's own MTP block, Inkling 2's of its drafter (0: none, no drafter loaded). More pay where they pass: the Qwen3.8-27B at 7 passes 7.4 tokens a step on an edit. An earlier config's `--mtp N` is taken as it on Qwen3.6 and Qwen3.8-27B. |
| `--draft-vocab N` | Qwen3.6, Qwen3.8-27B | the whole vocabulary | The draft head scores the first N token ids only: generation 4-8% faster at 65,536, but drafts miss the tokens past them (code: 63% of the drafts accepted against 70%). |
| `--dspark FILE` | Inkling 2 | the server: the smallest `dspark-*.gguf` beside the model, else in its `dspark/` folder | The model's block drafter (unsloth's `dspark-Inkling-2-Mini-Q8_0.gguf`): 0.4 GiB of weights and ~0.8 GiB of context cells and buffers, left out where the first card alone holds the context and would hold under 32,768 tokens beside it. UD-IQ4_XS on an RTX 3090: 108 -> 120 tokens/s after a short prompt, 107 -> 140 thinking, 105 -> 248 on an edit (with prompt lookup); its context on a 24 GB card 184K tokens instead of 217K. |
| `--dspark-reach N` | Inkling 2 | 131,072; 32,768 with experts in RAM; no more than a set `--max-context` | The context cells the drafter's layer without a window keeps, 4.5 KiB each. After an 86K-token prompt 3.54 tokens a step against 3.20 at 32,768. |
| `--dfly FILE` | Hy4-flash | none | Tencent's DFly2 drafter (the tencent/Hy4-flash checkpoint's last shard, `model-00148-of-00148.safetensors`, its `config.json` beside it) in place of the file's MTP block: ~0.8 GiB of the main GPU, the block's 3.6 GiB of RAM saved; an agent's session 2.2% less time a token and 5.2% less energy than with the MTP block. |
| `--no-mtp` | Hy4-flash | off | The file's MTP block left out: no drafts without `--dfly`, 3.6 GiB of RAM and ~5 s of the start saved. |

### The prompt cache

A request continues from the longest start of its prompt the engine holds; a conversation another request replaces
is parked with its K/V and its checkpoints ([USING.md](USING.md#the-prompt-cache)).

| Option | Models | Default | What it does, and its trade-off |
| --- | --- | --- | --- |
| `--prompt-cache N` | Flash-Next, Qwen3.6, Qwen3.8-27B, Inkling 2 | 16 | The checkpoints of the sequence's fixed-size state each conversation keeps in RAM, at each turn and along long prompts; past N they thin out back from its end. Each takes RAM (Flash-Next 113 MiB, Qwen3.6 64 MiB, Qwen3.8-27B 151 MiB, Inkling 2 169 MiB in FP16 and 92 in q8_0); a follow-up turn after 16-18K tokens starts in 0.2 s instead of 6.9 (Flash-Next). 0: none (Flash-Next: every request reads its whole prompt; the others continue the conversation they hold where a request extends it). |
| `--prompt-cache-every N` | Flash-Next, Qwen3.6, Qwen3.8-27B, Inkling 2 | 16384 | The checkpoints' interval inside a long prompt, in whole chunks. |
| `--conversation-cache-mib MIB` | every model (`--park-mib` too on Hy4-flash) | 16384 | RAM for the parked conversations, so that switching back to one continues where it was (a chat app's title request, a subagent); the least recently used leaves first, whole. Each holds its K/V (Flash-Next in `k8v4`: 12.5 KB a token) and its checkpoints. 0: none in RAM. |
| `--conversation-cache-slots N` | Flash-Next, Qwen3.6, Qwen3.8-27B, Inkling 2 | 0: as many as fit | Parked conversations at most. |
| `--conversation-cache-min-tokens N` | Flash-Next, Qwen3.6, Qwen3.8-27B, Inkling 2 | 1024 (Hy4-flash: 1,024 always) | A shorter conversation is not parked. |
| `--ram-reserve-mib MIB` | every model | 6144 | The parked conversations take no RAM below MIB available to the system, and leave between requests when it runs below: the experts are pinned, so a park beside a large arena would swap the rest of the system out. 0: only their budget limits them. |
| `--conversation-spill-dir DIR` | every model | none | The conversations the RAM does not hold, and those it evicts, go to files in a folder of the engine's own under DIR, written from the GPUs past the file cache, and are read back when a request continues them; one taken back keeps its file, so that its next park writes only what is new. With `--conversation-cache-mib 0` every parked conversation goes there. Where the RAM cannot hold an agent's long conversations, a switch reads seconds of disk instead of minutes of prompt: a 17K-token conversation parked in ~0.2 s and back in ~0.1 s on an NVMe drive, two of 480K tokens in turns 1.9-2.1 s a switch (2.8-3.1 s through RAM). |
| `--conversation-spill-mib MIB` | every model | 32 GiB or a quarter of the disk's free space, the less | Those files' size at most; never the disk's last 4 GiB. 0: none. |
| `--park-vram 0\|1` | Inkling 2 | 1 | A parked conversation's K and V stay on the cards where they are (up to three conversations beside the live one) until the VRAM is needed: two conversations in turns switch in tens of ms instead of ~0.5 s through RAM. Not with experts in RAM. 0: to RAM at once. |

### The experts

| Option | Models | Default | What it does, and its trade-off |
| --- | --- | --- | --- |
| `--expert-profile P` | Flash-Next, Hy4-flash, Inkling 2 | Flash-Next, Inkling 2: `builtin`, the model's ranking in the programs; Hy4-flash: none | The ranking of the experts by use that the GPUs are filled from at the start, or a file `expert_profile_save` wrote (the server takes that one once it exists). Inkling 2 reads one only with experts in RAM. |
| `--adapt-every N` | Flash-Next (0: Hy4-flash too) | 1 | Every N steps the GPUs' caches swap in the conversation's most routed experts they lack. Less often follows the conversation less (every 4 steps: code 10-13% slower). 0: the start's placement stays, and runs of a request repeat bitwise. |
| `--adapt-swaps N` | Flash-Next | 96 | An update's moves at most (384 measured the same). |
| `--adapt-decay F` | Flash-Next | 0.7 | The share of the routing counts kept after an update (between 0 and 1). |
| `--deterministic-adapt` | Flash-Next | off | Each update settled before the next, so that runs repeat bitwise while the caches adapt: a little slower (UD-Q4_K_XL thinking 96.8 tokens/s against ~105). |
| `--arena-place` | Flash-Next, Hy4-flash | off (`--no-arena-place`) | At the start, and between requests, the RAM is timed and the experts the CPU reads are kept out of its slow regions, as unpaired DIMMs make (the development PC's 2 x 32 + 2 x 48 GB): there UD-Q4_K_XL wrote code 5.4% and edits 6.7% faster, IQ3_S 1.4-6.5%. Elsewhere it finds nothing to move and costs ~2-5 s at the start. |
| `--arena-pin G` | Hy4-flash | 72 | GiB of the experts' RAM pinned: a GPU streams a prompt's experts from pinned RAM at ~55 GB/s against ~24 from pageable (an edit's 2,254-token prompt 580 against 163 tokens/s); pinning takes ~14 s more of the start, and the driver pins ~76 GiB of a process at most. |
| `--no-file-cache` | Flash-Next, Hy4-flash, Inkling 2 | off (`--file-cache`) | The experts (Hy4-flash and Inkling 2: the model file) are read past the OS file cache. Where the RAM holds the experts and the cached file together, the cached file settles on pages the next starts' experts would have had: on the development PC (160 GB) UD-Q4_K_XL's generation fell 7-8% over a session's starts, and stays at a fresh boot's speed with the option. Each start then reads from the drive (~5 s more there). With less RAM it costs only the start time. |
| `--hc-q8` | Flash-Next | on | The hyper-connection projections in 8 bits only: 644 MiB of VRAM, read ~30% faster. `--no-hc-q8`: BF16 copies, 1.27 GiB, a step 2-4% longer. |
| `--expert-math fast\|exact` | Flash-Next | `fast` | `exact` (K-quant files): the GPUs compute their experts with the CPU's arithmetic, so that the output does not depend on where an expert is computed; 22-28% slower generation. For tests. |
| `--expert-vram G` | Qwen3.6, Hy4-flash, Inkling 2 | Qwen3.6: what the card has free; Hy4-flash: the main GPU's free VRAM less 700 MiB (1.5 GiB where the desktop draws on it) | GiB of VRAM for the routed experts. Qwen3.6 on one GPU: below their size the CPU computes the rest (212 of 256 a layer fit a 24 GB GPU at 65,536 tokens of context). Hy4-flash: the main GPU's expert slots. Inkling 2: the first card's `cache=`. |
| `--hybrid-rows N` | Qwen3.6 | 128 | With experts in RAM: a prompt chunk of more rows streams its experts to the GPU, a smaller batch runs them from the VRAM slots and the CPU (at least 8). |
| `--ple-gguf PATH` | Flash-Next | the model's shard that holds it | The per-layer embedding table from another GGUF, such as one of unsloth's engram tables (BF16, Q8_0, Q5_1 or IQ4_NL). |
| `--ple-io direct\|ram\|mmap` | Flash-Next | `direct` | The table's reads: from the drive past the OS file cache, the table never in RAM; `ram`: read into RAM after the start (28.8 GB in IQ4_NL), a 16K prompt of new text 0.5 s sooner but generation slower (code 3-14%), the CPU reading its experts beside it; `mmap`: through a mapping, for comparisons. |
| `--load-threads N` | Inkling 2 | 8 | The threads that read the model file. |

### Control vectors

Flash-Next only ([SETUP.md](SETUP.md#control-vectors)). Generation is 0.2-1% slower with a vector, a long prompt
0.4%.

| Option | Default | What it does |
| --- | --- | --- |
| `--control-vector F[,F]` | | A control vector in llama.cpp's GGUF format (`direction.<layer>` tensors), applied to every hyper-connection stream after each layer, at scale 1; several files add up. |
| `--control-vector-scaled F:S[,F:S]` | | The same at scale S. |
| `--control-vector-layer-range A B` | 1 to the last | The layers it applies after, inclusive. |
| `--cvec-mode add\|project` | `add` | `add`: h + S d, as llama.cpp applies it; `project`: h - \|S d\| (h . v) v with v = d / \|d\|, a unit direction at S 1 removed. |
| `--cvec-dir per-layer\|single:L` | `per-layer` | With `project`: each layer's own direction, or layer L's in every layer. |

### LoRA adapters

Qwen3.6 and Qwen3.8-27B only ([USING.md](USING.md#lora-adapters)). An adapter takes its pairs' VRAM (304 MiB for
harness-hub's pi adapter); pi on the Qwen3.8-27B made prompts 6% and generation 5% slower (llama.cpp 17% and 12%).

| Option | What it does |
| --- | --- |
| `--lora F[,F]` | Adapters in llama.cpp's GGUF format, each at scale 1: each product with a pair adds s B (A x), s the scale times the adapter's alpha over its rank. |
| `--lora-scaled F:S[,F:S]` | The same at scale S (the scale after the last colon, so that a drive's colon stays the path's). |
| `--lora-init-without-apply` | The adapters loaded at scale 0: a request's `"lora"` or `POST /lora-adapters` sets their scales. |

### Layer splits and batch slots

| Option | Models | Default | What it does, and its trade-off |
| --- | --- | --- | --- |
| `--layer-split LIST` | Qwen3.6, Qwen3.8-27B, Inkling 2 | Qwen3.6, Qwen3.8-27B: the fewest cards that hold the model, filled in order; Inkling 2: by itself where the first card lacks room | Qwen3.6 and Qwen3.8-27B: each card's layer count in their order (`26,38`); a prompt's chunks run on the cards at once. Split, Qwen3.6 keeps every expert in VRAM: over an RTX 3090 and an RTX 5070 Ti it generated 1.6-1.7x and read prompts 1.9-3.3x as fast as on the 3090 alone. Inkling 2: `DEV:A-B`, the layers A to B on the card of CUDA number DEV (a card's `layers=` gives it), or `0`: no split by itself, the experts the first card lacks room for in RAM. |
| `--batch-mtp` | Qwen3.6, Qwen3.8-27B | off | With batch slots (`--parallel`) and `--drafts-per-step`: each session's step verifies its drafts too. |

### Hy4-flash

| Option | Default | What it does, and its trade-off |
| --- | --- | --- |
| `--prefill-min N` | 24 | A prompt part shorter than N tokens goes through the generation's kernels, where the prompt path's fixed cost (~0.3 s) would be the larger. |
| `--prefill-pass N` | 16384 | A long prompt part's chunks of up to N rows, each in two halves: the main GPU runs one half's dense steps while the tiers run the other half's experts. Its buffers (1.56 GiB at 16,384) are lent by the main GPU's expert slots for the part: the 16K prompt in 9.5-9.7 s instead of 11.5, 64K in 35.4 s instead of 43.7. 0: none. |
| `--prefill-lend N` | 0: off | An answer that has run N tokens lends the prompt path's own buffers on the main GPU (~1.5 GiB) to its expert slots; the next prompt part takes them back (~65 ms). For long answers after the last long prompt. |
| `--warm-ms N` | 500 | A request after a pause of a second or more runs a low-priority kernel on each GPU for N ms, so that their clocks are up while its first forwards run: short follow-ups after a pause 1.16 s instead of 1.38, for ~18 J a warm-up. 0: off. |
| `--tier-doorbell` | off | Each tier's shares of a forward queued at its start, behind doorbells the host rings instead of launches: no faster (the CPU sets the pace), and the waiting tiers draw 60-70 W more. |

### Inkling 2 Mini

| Option | Default | What it does, and its trade-off |
| --- | --- | --- |
| `--layer-split-back N` | 0 | With `layers=` on a later card: the loop's last N of its 36 applications run on the first card, with their layers' weights a second time there. Where the later card is the slower one, prompts and generation both gain; where the first card is (an RTX 3090 before an RTX 5070 Ti), prompts of 21K-86K tokens gained 20-28% at 8 and an agent's session lost 11%. |
| `--cell-spread N` | 0: from when the first card is full | With cells on later cards: the cards share the cells evenly from N tokens on, each scoring its share at once. At 49,152 on three cards a 171K-token prompt read at 2,297 tokens/s instead of 2,070 and generation after it at 126 instead of 110, for the other cards' power while they score (~127 and ~71 W against 10 and 20 idle). |
| `--prefill-max N` | 4096 | The rows a long prompt's chunk grows to while the free VRAM holds the larger buffers (2.3 GiB more at 4,096), back to `--prefill`'s after the request: a 21K-token prompt 14-17% sooner. With experts in RAM the larger buffers take the VRAM of the experts used least, which are back before the first token. At most `--prefill`: fixed chunks. |

### Set from the cards

The engine's own spellings of what the `--gpu` cards give, as CUDA numbers. lamina-server sets them from the cards;
they also take the place of the cards' values when given.

| Option | Models | What it is |
| --- | --- | --- |
| `--stage-gpus LIST` | Qwen3.6, Qwen3.8-27B | The layer split's GPUs, the cards in their order. |
| `--tier-gpus LIST` | Hy4-flash | The expert tiers' GPUs, the cards after the first (on Flash-Next a deprecated spelling, below). |
| `--cell-gpus LIST` | Inkling 2 | The cards that hold the context's cells. |
| `--cell-gpu-clocks SPEC[,SPEC]`, `--cell-gpu-reserve-mib R[,R]` | Inkling 2 | Their `clocks=` (0: none) and `reserve=` in MiB (`-`: by the card's use). |
| `--layer-split-clocks SPEC`, `--layer-split-reserve-mib N` | Inkling 2 | The `clocks=` and `reserve=` of the card with `layers=`. |
| `--main-gpu-clocks SPEC` | Inkling 2 | The first card's `clocks=`. |
| `--expert-gpu D` | Inkling 2 | The later card with `cache=`: the copy of the experts in RAM. |
| `--expert-cache auto\|SIZE`, `--vram-reserve-mib N` | Inkling 2 | The first card's `cache=` and `reserve=` (on Flash-Next deprecated spellings, below). |

### Deprecated spellings

Still taken, each named in the server's window at a start with what takes its place; `--write-config` writes a config
without them. A card's own setting wins over them, a per-tier list's values go to the tiers in order (its last to
those past it), and of `--gpu`, `--main-gpu` and `--second-gpu` the later takes the earlier's place (`--second-gpu -1`
after the cards: no tiers).

| Option | In its place |
| --- | --- |
| `--main-gpu GPU` | the first `--gpu` |
| `--second-gpu G[,G,G]`, `--tier-gpus G[,G,G]` | the later `--gpu` cards, or `auto` (-1: none) |
| `--expert-cache N\|auto` | the first card's `cache=` (N slots; 0: the profile's count) |
| `--vram-reserve-mib M` | the first card's `reserve=` in MiB |
| `--vram-pin-gib G` | the first card's `vram-only=` in GiB |
| `--second-gpu-pin-gib G` | the first tier's `vram-only=` in GiB |
| `--second-gpu-gib G[,..]` | each tier's `cache=` in GiB (0: auto) |
| `--second-gpu-reserve-mib R[,..]` | each tier's `reserve=` in MiB |
| `--second-gpu-prefetch-mb M[,..]` | each tier's `prefetch=` in MiB |
| `--second-gpu-clocks C[,..]` | each tier's `clocks=` (0: the driver's) |
| `--second-gpu-sleep-us T[,..]` | each tier's `sleep=` in microseconds (0: none) |
| `--second-gpu-clocks-hold S`, `--second-gpu-clocks-lead S` | `--clocks-hold`, `--clocks-lead` in seconds |
| `--second-gpu-min-mb M` | `--tier-min` in MiB |
| `--second-gpu-adapt`, `--no-second-gpu-adapt` | `--tier-adapt`, `--no-tier-adapt` |

### The engine alone

`lamina --native MODEL.gguf` runs the engine without the server: `--serve` and `--vision` are the line protocol and
the pictures lamina-server speaks; with `--tokens` it generates from token ids, for tests and measurements.
`lamina --help` lists the common options, `lamina --help-all` every one.

| Option | Models | What it does |
| --- | --- | --- |
| `-h`, `--help`, `--help-all` | every model | The common options; every one, then the deprecated spellings. |
| `--native MODEL.gguf` | every model (`--model` too, but on Flash-Next) | The model's GGUF, its first shard (lamina-server: `-m`). Its architecture picks the engine's path. |
| `--serve` | every model | Stay resident and take requests on stdin, as lamina-server sends them. |
| `--vision` | every model | `--serve` takes pictures too: the image encoder's rows beside a request's tokens. |
| `--batch N` (`--slots`) | Qwen3.6, Qwen3.8-27B | N sessions decoding together on one GPU (lamina-server: `--parallel`). |
| `--tail-role-token ID` | Flash-Next, Qwen3.6, Qwen3.8-27B | A turn of this role right before the last one stays out of its checkpoint (lamina-server sets it with `--effort-position end`; -1: off). |
| `--expert-profile-save P`, `--expert-profile-save-every MIN` | Flash-Next, Hy4-flash, Inkling 2 | lamina-server's `expert_profile_save` and `expert_profile_save_every`. |
| `--tokens LIST`, `--tokens-file PATH` | every model | Generate mode's prompt as token ids, comma-separated or from a file. |
| `--max-new N` (`--n`) | every model (`--n` on Hy4-flash and Inkling 2) | Tokens to generate (Flash-Next 16, the others 64). |
| `--stop-eos`, `--eos-ids LIST` | Flash-Next | Stop at an end-of-turn token (`--serve` always does), or at these. |
| `--temperature F`, `--top-p F`, `--top-k N`, `--min-p F`, `--seed N` | Flash-Next, Inkling 2 | Generate mode's sampling (greedy by default). |
| `--spec-follow PATH` | Flash-Next | Emit this continuation (token ids) and keep the drafts that match it, so that settings and builds compare on one text. |
| `--force IDS`, `--dump-full-logits F`, `--emb FILE` | Qwen3.6, Qwen3.8-27B, Hy4-flash, Inkling 2 | Run along these tokens, write every row's logits, take a picture's rows (parity runs against llama.cpp). |
| `--bench-pp N`, `--bench-tg N` | Qwen3.6, Qwen3.8-27B | A prompt of N tokens read and N tokens generated, timed. |
| `--no-graph` (`--no-token-graph`) | Qwen3.6, Qwen3.8-27B, Hy4-flash, Inkling 2 | Every kernel launched each step instead of the step's captured graph (slower). |
| `--expert-slots N` | Qwen3.6 | With experts in RAM: N experts a layer in VRAM. |
| `--main-cells N` | Inkling 2 | The first card keeps N cells at most, so that cells reach the other cards at a short context. |
| `--spec-split` | Flash-Next | A step's tokens in two groups, the GPU's work on one beside the CPU's on the other: the same output, ~7% slower. |
| `--disable-hit` | Flash-Next | Every routed expert computed by the CPU (`--expert-math exact`'s reference). |
| `--no-prompt-offload` | Flash-Next | With a tier: the prompt path streams the experts the main GPU lacks to it instead of computing them on the first tier (slower). |
| `--dump-routing PATH` | Flash-Next | The routed experts' ids and weights per layer and position. |
| `--window-hashes PATH` | Flash-Next | Per step a hash of its final rows and its picks: builds that do the same arithmetic write the same file (a fixed text through `--spec-follow`, `--adapt-every 0`). |
| `--window-logits PATH` | Flash-Next | Per emitted token the 64 likeliest with their log-probabilities, for comparisons with llama.cpp. |
| `--dump-draft-data DIR` | Flash-Next | With `--serve`: each request's positions for the draft layer's training. |
| `--stats`, `--window-profile`, `--prefill-profile` | Flash-Next | The time by stage at the end; with `--stats` the GPU time of each stage of a step (~5% slower); the GPU time of each section of the prompt path. |
