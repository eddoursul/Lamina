# Using Lamina

The server, `lamina`, runs the engine in its own process and serves the OpenAI, Anthropic and Responses APIs and
a web app. It starts with a model file, `lamina serve -m MODEL.gguf`, or a config, `lamina serve --config FILE`
([SETUP.md](SETUP.md#a-model-file)), and listens on this PC only, on the config's port (the presets': 8081 for
ISTA-DASLab's Flash-Next files, 8082 for the family's others, 8083 for Qwen3.6-35B-A3B, 8084 for Qwen3.8-27B, 8086 for
Hy4-flash, 8087 for Inkling 2 Mini). Ctrl+C in its window stops it; the engine saves what it learned about the experts
first. Every option, the server's and the engine's: [PARAMETERS.md](PARAMETERS.md).

## The web app

`http://127.0.0.1:8081/`: **Chat** (with pictures, the thinking level, MCP tools, chats kept in the browser),
**Monitor** (what the engine is doing, the last requests, the GPUs, CPU and RAM) and **About** (the addresses to give
clients, the settings). Its **Model settings** card changes a few settings of the running server and saves them into
its config file ([below](#the-config)).

## The API

| | Endpoint |
| --- | --- |
| OpenAI Chat Completions (stream and non-stream, tools) | `POST /v1/chat/completions` |
| Anthropic Messages (stream and non-stream, tools) | `POST /v1/messages`, `POST /v1/messages/count_tokens` |
| OpenAI Responses (stream and non-stream, tools; stateless, [below](#codex-cli-and-the-responses-api)) | `POST /v1/responses` |
| A choice among single-token options in one pass ([below](#one-pass-choices)) | `POST /v1/choose` |
| The model list, health | `GET /v1/models`, `GET /models`, `GET /health` |
| The model's properties (llama-server's fields) | `GET /props` |
| What the model is doing now | `GET /status`, `GET /slots` |
| Everything the Monitor shows; Prometheus' format with vLLM's names on `Accept: text/plain` | `GET /metrics` |
| Request totals in Prometheus' format under llama-server's names | `GET /metrics/prometheus` |
| The MCP servers, their state and tools | `GET /mcp` |
| The prompt a chat request is read as; a text's token ids and back (llama-server's fields) | `POST /apply-template`, `POST /tokenize`, `POST /detokenize` |
| Image generation, when set up ([below](#image-generation)) | `POST /v1/images/generations` |
| Unload the model now, or load it ahead of a request | `POST /unload`, `POST /load` |
| The LoRA adapters and their scales (llama-server's; [below](#lora-adapters)) | `GET /lora-adapters`, `POST /lora-adapters` |

```bash
curl http://127.0.0.1:8081/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "Write a haiku about GPUs."}], "max_tokens": 512}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8081/v1", api_key="none")
r = client.chat.completions.create(model="local", messages=[{"role": "user", "content": "Hello!"}])
print(r.choices[0].message.content)
```

Any model name is served; `/v1/models` lists the config's `model_name` and its `aliases`. An answer's `timings` use
llama.cpp's names, with the drafts offered and accepted, and with GPU tiers the tiers' work in `expert_tiers` (the
first tier's fields `second_*`, the next ones' `third_*` and `fourth_*`); a tier with `sleep=` adds the rounds it slept
through (`*_rounds`, `*_asleep_rounds`, `*_sleeps`) and the measure its threshold is held against, the CPU's
microseconds a layer without the tier: `*_awake_us` predicted over the answer's rounds awake, `*_asleep_us` measured
over its rounds asleep ([PARAMETERS.md](PARAMETERS.md#gpus)). `/metrics` in Prometheus' format carries vLLM's metric names,
so dashboards written for vLLM read this server, and the Monitor's other facts as `lamina:<key>`;
[monitoring/](monitoring/) has a Prometheus scrape config, a Prometheus Operator `ServiceMonitor` and a Grafana
dashboard.

## Clients

- **Apps with an OpenAI-compatible provider**: base URL `http://127.0.0.1:8081/v1`, any API key (or the config's
  `api_key`). Zed's llama.cpp provider reads `/props` and offers tools from its `chat_template_caps`.
- **Claude Code**: `ANTHROPIC_BASE_URL=http://127.0.0.1:8081`, `ANTHROPIC_MODEL` set to a Claude model name it knows
  (the server ignores the name), and any `ANTHROPIC_AUTH_TOKEN` (or the `api_key`).
- **OpenCode**: a provider in `opencode.jsonc`, with the server's context (its `--max-context`); OpenCode compacts the
  conversation before `limit.context`, so keep `limit.output` well under it:

  ```jsonc
  {
    "$schema": "https://opencode.ai/config.json",
    "provider": {
      "lamina": {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Lamina",
        "options": { "baseURL": "http://127.0.0.1:8081/v1", "apiKey": "none" },
        "models": {
          "flash-next": {
            "name": "Qwen3.8-Flash-Next",
            "limit": { "context": 262144, "output": 32768 },
            "options": { "reasoningEffort": "high" },
            "variants": { "low": { "reasoningEffort": "low" }, "none": { "reasoningEffort": "none" } }
          }
        }
      }
    },
    "model": "lamina/flash-next"
  }
  ```

- **Codex CLI**: [below](#codex-cli-and-the-responses-api).
- **A terminal chat**: `python chat.py --port 8081` in a clone (Python 3.10 or newer, no packages; `/image <path>`
  attaches a picture, `/think` sets the thinking level, `/reset` starts over).

## Thinking

The model thinks before it answers (streamed as `reasoning_content`; Anthropic: `thinking` blocks). How much, per
request:

| API | |
| --- | --- |
| OpenAI | `"reasoning_effort": "none" \| "low" \| "medium" \| "high"` (also `"reasoning": {"effort": ...}`, or `"chat_template_kwargs": {"enable_thinking": false}`) |
| Anthropic | `"output_config": {"effort": "low" \| "medium" \| "high"}`, `"thinking": {"type": "disabled"}`, or `"thinking": {"type": "enabled", "budget_tokens": N}` (under 2K = low, under 8K = medium, more = high) |

Without a setting the model uses its own default, high. The levels are instructions the model was trained with, not
a token limit; `"reasoning_budget_tokens": N` in a request caps the thinking, the server then ending it with a short
wrap-up. By default the thinking also leaves the answer room within `max_tokens`. The server-wide defaults and switches
(`--reasoning-effort`, `--reasoning-budget-tokens`, `--answer-room`, `--effort-position`, `--anthropic-thinking`):
[PARAMETERS.md](PARAMETERS.md#request-defaults).

## Sampling

A request's `temperature`, `top_p`, `top_k`, `min_p`, `seed`, stop strings (`stop`; Anthropic `stop_sequences`) and
penalties (`presence_penalty`, `frequency_penalty`, `repetition_penalty` with `penalty_last_n`; llama.cpp's
`repeat_penalty` and `repeat_last_n` too) are honored. What a request leaves out comes from the config's `sampling`
block, else from the model card's settings for the request's thinking state (both:
[PARAMETERS.md](PARAMETERS.md#request-defaults)). `"temperature": 0` decodes greedy; `top_k` keeps at most 64
candidates.

Every token, drafted or not, is the model's own pick: with sampling, a draft is kept only when it equals the token the
model sampled for its position, so the drafts change the speed, never the distribution. The expert caches adapt to the
conversation in the background, so a sampled answer repeats run to run only with `--adapt-every 0`.

## The prompt cache

A request continues from the longest start of its prompt the engine holds, so the next turn of a chat or an agent reads
only what is new. Each conversation keeps checkpoints of its state in RAM (at each turn and along long prompts); a
conversation another request replaces (a chat app's title request, a subagent) is parked in RAM with its K/V and its
checkpoints, so switching back continues where it was, and with `--conversation-spill-dir` on disk where the RAM
cannot hold it. A request that shares only a start with another conversation (subagents, sessions with one system
prompt) borrows that start. With batch slots (`parallel`) each session holds a conversation, and requests run in the
session that holds theirs. The engine keeps the model's own token ids of each reply for the next turn. Responses report
the reused tokens (`usage.prompt_tokens_details.cached_tokens`; Anthropic: `cache_read_input_tokens`). The cache's
sizes and limits: [PARAMETERS.md](PARAMETERS.md#the-prompt-cache).

## Behaviour

- **Streaming**: with `"stream": true` the thinking, the answer and tool calls (name first, then the arguments piece by
  piece) arrive as they are made; a long prompt sends keep-alives so that agents do not time out, and
  `"return_progress": true` sends llama.cpp's `prompt_progress` (total, cache, processed, time_ms) per prompt chunk
  instead. Closing the connection stops the model, so the next request starts at once.
- **Tool calls** are taken out of the text when the request declares tools; without them the answer is the model's
  text as written (`--tool-call-recovery` reads a few other forms too).
- **Context**: a request longer than the context is refused (400), never cut. One whose `max_tokens` would run past the
  context is answered with `max_tokens` shortened to the room left (agents ask for 32,000 on every request);
  `--no-fit-max-tokens` gives llama.cpp's 400 instead.
- **Guards**: a reply that repeats one token 256 times ends as `"length"`; a verify window whose logits are not finite
  emits nothing, the engine drops its cached state and the server retries the request once if nothing was sent yet; a
  request the engine stays silent on for 300 s ends. The model writes `<|im_end|>` and `<|endoftext|>` as the one token
  that ends a turn, also when it means to quote them: drawn where the text quotes it, the token becomes the marker's
  text and the reply goes on. Each has its switch, as have the optional recoveries of a thinking with no answer and of
  a reasoning loop: [PARAMETERS.md](PARAMETERS.md#replies).
- **A greeting**: a tool-using agent's prompt and an opening message with no task ("hello") can make the model invent
  instructions in its thinking, and act on them, at the top reasoning effort (on llama.cpp too); `--short-opening-tokens
  4` reads such a message at `medium`.
- **JSON output**: `response_format` with `json_object` or `json_schema` (an object at the schema's root) asks the model
  for JSON and validates the answer: one generation, no hidden retry, a 502 (`structured_output_failed`) when it does
  not hold. The schema is checked with the `jsonschema` package when it is installed; without it, only that the answer
  is one JSON object. Prompting and validation, not constrained decoding.

## LoRA adapters

Qwen3.6-35B-A3B and Qwen3.8-27B take LoRA adapters in llama.cpp's GGUF format (`convert_lora_to_gguf.py`'s) with
llama-server's options (`--lora`, `--lora-scaled`, `--lora-init-without-apply`:
[PARAMETERS.md](PARAMETERS.md#lora-adapters)) and API:

```json
"args": {"lora": "D:\\adapters\\Qwen3.8-27B-pi-lora-F16.gguf,D:\\adapters\\Qwen3.8-27B-opencode-lora-F16.gguf"}
```

The adapters are numbered in their order, and several add up:

- `GET /lora-adapters`: `[{"id": 0, "path": ..., "scale": 1.0, "task_name": ..., "prompt_prefix": ...}, ...]`, the
  scales a request that names none takes.
- `POST /lora-adapters` with `[{"id": 0, "scale": 0.5}]` sets those scales, every adapter left out at 0.
- A request's `"lora": [{"id": 1, "scale": 1.0}]` (Chat Completions, Messages, Responses) runs it with those
  scales, every adapter left out at 0; an id that is not an adapter's, or a scale that is not a number, is a 400.
- The config's `lora_models` names sets of scales: `"lora_models": {"qwen3.8-27b-pi": [{"id": 0, "scale": 1}]}`. A
  request naming one as its model (and no `"lora"` of its own) runs with them, so a client that only picks a model
  name picks an adapter; `/v1/models` lists them beside the model.

A request with other scales starts a conversation of its own (the prompt cache keeps both), and with batch slots
requests with different adapters decode together. The adapters' pairs may be F16, BF16, F32 or Q8_0, at any projection
but the embedding, the router and the routed experts; activated LoRA (aLoRA) adapters are refused.

## Pictures

With an image projector beside the model (or a config's `vision` section), OpenAI `image_url` parts and Anthropic
`image` blocks (base64 data, http(s) URLs or file paths) go through the server's image encoder and reach the model as
up to 1,024 tokens each in the presets (a 640x480 photo is ~300; Inkling 2 Mini's are not resized: a token a 40 x 40
px patch, up to 4,096). JPEG, PNG, BMP and GIF are read as they are, WebP, TIFF, AVIF and others converted to PNG
first. A picture is encoded once, also when a chat app sends it again every turn. On Windows a file path can be a
network share's (`\\host\share\x.png` or `file://host/share/x.png`), read with the account the server runs under;
elsewhere, mount the share and name the file by its path there.

By default the encoder runs on the engine's main GPU in VRAM it borrows from the engine for each new picture, so the
GPUs keep their VRAM for the model: 0.3-0.6 s a picture of ~1,000 tokens, where the CPU takes 8-10 s. On the development
PC the first question about such a picture takes 1.3 s with Qwen3.8-Flash-Next's IQ3_S. An encoder that keeps its own
VRAM (`--no-vision-lend`, 0.7-1.2 GB) encodes such a picture in 0.2 s: for a card with VRAM to spare. The options:
[PARAMETERS.md](PARAMETERS.md#the-model); what each model lends: [SETUP.md](SETUP.md#a-model-file). `--offline`
refuses URLs.

## Videos

Qwen3.8-Flash-Next, Qwen3.6-35B-A3B, Qwen3.8-27B and Hy4-flash read videos through the same projector and encoder
(Inkling 2 Mini's takes none): an OpenAI-style `video_url` part (vLLM's form), llama.cpp's `input_video` part
(`{"data": ...}` or `{"url": ...}`), a Responses API `input_video` part (its `video_url`) or a `video` block shaped as
an Anthropic `image` block, each a data: URL, an http(s) URL (up to 1 GiB) or a file path (a network share's too, as
a picture's). ffmpeg and ffprobe, 5.1 or newer, read the file (any format they read): on the PATH, or
`--vision-ffmpeg` names ffmpeg (ffprobe beside it).

A video is sampled and sized as its model's own processor does it: 2 frames a second, each group of frames after its
time in the prompt (`<12.5 seconds>`), so the model can tell when something happens.

- **Qwen:** up to 768 frames, a pair of frames a group, a frame at most 768 tokens. The whole video is held to about
  12,300 tokens, plus ~10 a group for its time and tags.
- **Hy4-flash:** up to 2,048 frames, one a group, held to about 60,000 tokens.

A part of a video: `"video_start"` and `"video_end"` beside the part's `"type"`, in seconds (Qwen's own
`qwen-vl-utils` keys; either one alone runs from the start or to the end). The part is planned as a whole video of its
frames is, so a short part of a long video gets the budget to itself (each frame larger), and its times stay the
video's (`<3005.0 seconds>`). ffmpeg seeks to it rather than decoding from the start.

A 3:42 film at 720p comes to ~13,300 prompt tokens with the Qwen models and ~57,000 with Hy4-flash. Its frames are read in
1.6 s and encoded in 2-8 s on the development PC, and the prompt is read at the model's usual rate. A video is encoded
once and kept (up to 32 videos or 4 GiB of encoded rows), also when a chat app sends it again every turn, and the next
turn of the conversation reads only what follows it (with the Qwen models another question about it in a new
conversation too). The web app attaches videos of up to 150 MB.

## Image generation

Optional: [Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) in
[unsloth's GGUF quants](https://huggingface.co/unsloth/Qwen-Image-2.1-Turbo-GGUF), run by
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)'s `sd-cli`. The model is under the Qwen
Research License, not Apache 2.0: read [its license](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/main/LICENSE)
before using the images commercially. It takes three things, then a start of the server with them:

- **`sd-cli`**, built from stable-diffusion.cpp at the commit Lamina was tested with, `2988060a`, with CUDA:

  ```
  git clone --recursive https://github.com/leejet/stable-diffusion.cpp
  cd stable-diffusion.cpp
  git checkout 2988060a4338a5dcd94721c964a8e87003c1ac8b
  git submodule update
  cmake -B build -DCMAKE_BUILD_TYPE=Release -DSD_CUDA=ON -DSD_BUILD_EXAMPLES=ON -DSD_WEBP=OFF -DSD_WEBM=OFF -DSD_BUILD_SHARED_LIBS=OFF
  cmake --build build --config Release --target sd-cli
  ```

  `python tools/image/sdcpp_native/apply.py <the checkout>` before the build adds Lamina's own runner of the model's
  denoiser, which the section's `"native": true` then takes.
- **The model's three files**: a denoiser from [unsloth/Qwen-Image-2.1-Turbo-GGUF](https://huggingface.co/unsloth/Qwen-Image-2.1-Turbo-GGUF)
  by the GPU's VRAM (`qwen-image-2.1-turbo-Q4_K_M.gguf` up to 8 GB, `-Q6_K` up to 12 GB, `-Q8_0` above: 4.2, 6.3 and
  7.6 GB), the VAE `vae/qwen_image_2.1_vae_bf16.safetensors` (0.7 GB) of
  [unsloth/Qwen-Image-2.1-FP8](https://huggingface.co/unsloth/Qwen-Image-2.1-FP8), and the text encoder
  `Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf` (5.1 GB) of
  [unsloth/Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen3-VL-8B-Instruct-GGUF).
- **An `images` section** in the config that names them (its other keys:
  [PARAMETERS.md](PARAMETERS.md#image-generation); the server's window says at the start which file is missing):

  ```json
  "images": {"exe": "D:/sd/build/bin/Release/sd-cli.exe", "backend": "cuda",
             "diffusion_model": "D:/Models/qwen-image/qwen-image-2.1-turbo-Q8_0.gguf",
             "vae": "D:/Models/qwen-image/qwen_image_2.1_vae_bf16.safetensors",
             "llm": "D:/Models/qwen-image/Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf",
             "args": ["--diffusion-fa"], "gpu_policy": "auto"}
  ```

`POST /v1/images/generations` takes OpenAI's request: `prompt`, `size` (`WIDTHxHEIGHT`, each side a multiple of 32 from
256 to 2048; default 1024x1024), `n` (1-4), `response_format` (`b64_json`, or `url`: the server keeps the newest 64),
plus `seed` (the answer says which it used) and `steps` (8, Turbo's schedule). A job takes the server's turn on the GPU:
a chat request waits for the image and the other way round. The section's `gpu_policy` keeps the engine loaded beside
the job (`keep`), stops it for the job (`unload`: the next request starts it again and reads its conversation again),
or picks by the free VRAM (`auto`, the default).

## MCP tools

MCP servers named in the config's `mcp_servers` (Claude Desktop's format; or a file with `--mcp-config`) give the web
app's chat their tools: the server starts them (or connects to them over Streamable HTTP), lists them at `GET /mcp`, and
runs the calls the model makes. Only requests that ask for them get them (`"lamina_mcp": true`, which the web app
sends); other clients see the plain API. `--offline` starts none.

## Sharing the GPU

On a PC that also games or runs another model server, `--idle-unload-s` unloads the model after a time without
requests, `--lazy-load` starts without it, `--min-free-vram-mib` loads it only when the VRAM is free, and
`--before-load` runs a command first, such as one that unloads another server's model
([PARAMETERS.md](PARAMETERS.md#loading-and-stopping)). `POST /unload` unloads it now (`?wait=1` waits for a running
request first) and `POST /load` loads it, both with `Content-Type: application/json`. Unloading ends the engine and the
image encoder, so their VRAM and RAM go back (the server keeps each GPU's CUDA context: about 200 MiB on the main GPU,
less on the others); the model files stay in the OS file cache, so loading again is quick while that RAM is not needed
elsewhere.

**Behind [llama-swap](https://github.com/mostlygeek/llama-swap)**, which starts the server a request names and stops
the others: run the server itself with llama-swap's port, so that stopping it stops the engine too
(`cmd: sh -c 'cd /path/to/Lamina && exec build/lamina serve --config iq3_s.json --port ${PORT}'`),
with `checkEndpoint: /health` (the server listens once the model has loaded) and a `healthCheckTimeout` of a few
minutes. llama-swap passes the client's `Host` through: without an API key, add the name clients reach it by to
`allowed_hosts` ([below](#from-other-devices)), or those requests get 403.

## From other devices

The server listens on this PC only (`127.0.0.1`). `"host": "0.0.0.0"` in the config (or `--host`) opens it to the
network, with an API key: `"api_key": "a-long-secret"` (or the `LAMINA_API_KEY` variable; several keys
comma-separated or as a list), which clients then send as their API key. The server window prints this PC's addresses;
on Windows the firewall blocks them until lamina is allowed on private networks. Through a tunnel or a proxy, set
the key first, and add the address the page is reached at to `"trusted_origins"` for the web app's settings and MCP
tools. A request without the right key is refused before its body is read. The server has no deadline for a request's
headers, though: a client that sends them a byte at a time holds one of its threads, so on a network you do not trust,
put a reverse proxy in front of it.

- **Browsers (CORS)**: off by default; `"cors_origins": ["https://chat.example.com"]` lets those pages call `/v1/*`.
- **Host names**: without an API key the server answers only `Host` names it knows (localhost, IP addresses, this PC's
  name when it listens beyond it), against DNS rebinding; `"allowed_hosts"` (or `LAMINA_ALLOWED_HOSTS`) adds others.
- **Web pages without a key**: a browser's `POST` to `/v1/*` from a page other than the server's own, localhost, an
  allowed host or a trusted origin gets 403; clients that send no `Origin` (curl, the SDKs) are not affected.

## The config

The server reads a config file at its start (`--config FILE`), or with `-m` builds one from the model file
([SETUP.md](SETUP.md#a-model-file); `lamina write-config` writes it; its keys: [SETUP.md](SETUP.md#the-config)). Every
key is an option of the command line too and the other way round, the engine's options in the config's `"args"`, and the
command line's values take the place of the file's, the file's those of the preset: `lamina serve --config my.json
--port 8090 --temperature 0.6` ([PARAMETERS.md](PARAMETERS.md#how-options-are-given)). A key the server does not read (a
typo) is named in its window and ignored. The command line's alone are the commands (`lamina help` lists them),
`--config` and the options of `check` and `bench`.

The web app's Model settings card shows the values the server runs with (each set on the command line marked so) of
`sampling` (`temperature`, `top_p`, `top_k`, `min_p`), `reasoning_budget_tokens`, `fit_max_tokens`,
`anthropic_thinking`, `effort_position`, `aliases`, `idle_unload_s`, `lazy_load`, `engine_silence_s`, `api_monitor`,
`open_browser` and the main GPU's VRAM reserve (the first card's `reserve` in `"gpu"`). **Apply** changes them for the
running server at once; `effort_position`, `lazy_load`, `open_browser` and the reserve wait for the next start (marked
*at start-up*). **Save configuration**, with `--config`, writes into that file the values that differ from what it
gives (the earlier file kept as `<config>.bak`, its other keys as they were); in a config that names a model and has
no `"args"`, a value equal to the preset's takes the key out of the file. Without a config file (`-m` alone) the
card's changes last until the server stops.

## One-pass choices

`POST /v1/choose` answers a multiple-choice question without generating: the chat is rendered as for
`/v1/chat/completions` with thinking off (so it shares a conversation's cached prompt), `answer_prefix` (optional)
follows the assistant header, and each option's probability is read from the head's logits there. Options must be
single tokens (letters are; after a prefix ending in a word, `" A"` is one token and `"A"` another):

```bash
curl http://127.0.0.1:8081/v1/choose -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "Is Paris the capital of France? A) yes B) no. Answer with the letter."}],
  "options": ["A", "B"]}'
# {"probs": {"A": 0.99988, "B": 0.00012}, "logprobs": {"A": -0.0009, "B": -9.005}, "mass": 0.9992, "choice": "A",
#  "confidence": 0.99988, "prompt_tokens": 31, "reused": 0, "seconds": 0.71}
```

`probs` are renormalized over the options; `mass` is what they got of everything the model could have said there.
Not on the path of Qwen3.6-35B-A3B and Qwen3.8-27B yet.

## Codex CLI and the Responses API

`POST /v1/responses` takes OpenAI's Responses requests, on the same path as chat completions: `input` as text or items
(messages, reasoning, function and custom tool calls and their outputs), `instructions`, `tools` (function, namespace
and custom tools; hosted tools such as `web_search` are left out), `reasoning.effort`, `max_output_tokens`,
`text.format` (the JSON formats above), and streaming with Responses' events. It is stateless: nothing is stored, so
the client sends the whole conversation every time (Codex does), and `previous_response_id`, `conversation`,
`background` and the retrieve, delete and cancel endpoints are refused. The thinking comes back as a `reasoning` item;
sending it back with the conversation keeps the prompt cache.

**Codex CLI**, in `~/.codex/config.toml` (Windows: `%USERPROFILE%\.codex\config.toml`):

```toml
model = "qwen3.8-flash-next"
model_provider = "lamina"

[model_providers.lamina]
name = "Lamina"
base_url = "http://127.0.0.1:8081/v1"
wire_api = "responses"
# env_key = "LAMINA_API_KEY"    # only with an api_key: the variable holding it
```

Codex warns that it lacks the model's metadata; that is only the warning. `--codex-compaction-cache` renders Codex's
compaction requests with the conversation's tools, so they reuse the cached conversation; `--codex-thread-titles`
answers Codex's thread-title requests without the engine.
