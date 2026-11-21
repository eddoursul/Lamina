# Using Lamina

The server, `lamina-server`, runs the engine in its own process and serves the OpenAI, Anthropic and Responses APIs and
a web app. It starts with a model file, `lamina-server -m MODEL.gguf`, or a config, `lamina-server --config FILE`
([SETUP.md](SETUP.md#a-model-file)). It listens on the config's `port` (the presets': 8081 for ISTA-DASLab's Flash-Next
files, 8082 for the family's others, 8083 for Qwen3.6-35B-A3B, 8084 for Qwen3.8-27B, 8086 for Hy4-flash, 8087 for
Inkling 2 Mini; `--port` overrides it), on this PC
only unless told otherwise ([below](#from-other-devices)). Ctrl+C in its window stops the engine, which saves its
learned expert profile, and the server.

## The web app

`http://127.0.0.1:8081/`: **Chat** (with pictures, the thinking level, MCP tools, chats kept in the browser),
**Monitor** (what the engine is doing, the last requests, the GPUs, CPU and RAM) and **About** (the addresses to give
clients, the settings). Its **Model settings** card changes a few settings for the running server and writes them into
its config file ([below](#the-config)); only the page itself can change them.

## The API

| | Endpoint |
| --- | --- |
| OpenAI Chat Completions (stream and non-stream, tools) | `POST /v1/chat/completions` |
| Anthropic Messages (stream and non-stream, tools) | `POST /v1/messages`, `POST /v1/messages/count_tokens` |
| OpenAI Responses (stream and non-stream, tools; stateless, [below](#codex-cli-and-the-responses-api)) | `POST /v1/responses` |
| A choice among single-token options in one pass, nothing generated ([below](#one-pass-choices)) | `POST /v1/choose` |
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
llama.cpp's names, with the drafts offered and accepted. Prometheus' scrape of `/metrics` carries vLLM's metric names
(`vllm:num_requests_running`, the token and request counters, the draft counters, time to first token and between
tokens), so dashboards written for vLLM read this server; the Monitor's other facts come under `lamina:` and their
JSON key (`lamina:live_tok_s`, `lamina:gpu_util`, ...). Ready to use in [monitoring/](monitoring/): a Prometheus scrape
config (`prometheus.yml`, the API key sent as a bearer token), a Prometheus Operator `ServiceMonitor` and a Grafana
dashboard of both name sets (`grafana-lamina.json`).

## Clients

- **Apps with an OpenAI-compatible provider**: base URL `http://127.0.0.1:8081/v1`, any API key (or the config's
  `api_key`). Zed's llama.cpp provider reads `/props` and offers tools from its `chat_template_caps`.
- **Claude Code**: `ANTHROPIC_BASE_URL=http://127.0.0.1:8081`, `ANTHROPIC_MODEL` set to a Claude model name it knows
  (the server ignores the name), and any `ANTHROPIC_AUTH_TOKEN` (or the `api_key`).
- **OpenCode**: a provider in `opencode.jsonc`, with the server's context (its `--max-context`):

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

  OpenCode compacts the conversation before `limit.context`; keep `limit.output` well under it.
- **Codex CLI**: [below](#codex-cli-and-the-responses-api).
- **A terminal chat**: `python chat.py --port 8081` in a clone (Python 3.10 or newer, no packages; `/image <path>`
  attaches a picture, `/think` sets the thinking level, `/reset` starts over).

## Thinking

The model thinks before it answers (streamed as `reasoning_content`; Anthropic: `thinking` blocks). Choose how much
per request:

| API | |
| --- | --- |
| OpenAI | `"reasoning_effort": "none" \| "low" \| "medium" \| "high"` (also `"reasoning": {"effort": ...}`, or `"chat_template_kwargs": {"enable_thinking": false}`) |
| Anthropic | `"output_config": {"effort": "low" \| "medium" \| "high"}`, `"thinking": {"type": "disabled"}`, or `"thinking": {"type": "enabled", "budget_tokens": N}` (under 2K = low, under 8K = medium, more = high) |

Without a setting the model uses its own default, high; `"reasoning_effort"` in the config sets another for requests
that name none. The levels are instructions the model was trained with, not a token limit:

- **A thinking budget**: `"reasoning_budget_tokens": N` in a request or the config caps the thinking at N tokens; the
  server then ends it with a short wrap-up and `</think>`, and the model answers from there (nothing is read again).
- **The answer keeps its room**: with thinking on, no budget and a `max_tokens` of 1,024 or more, the thinking is
  wrapped up when a quarter of the cap (at least 512 tokens) is left; without `max_tokens`, at three quarters of the
  context. `"answer_room": false` in the config turns it off.
- **`"effort_position": "end"`** in the config puts a non-default effort in a short system turn before the answer
  instead of the top of the prompt, so a request that only changes the effort reuses the cached conversation.
- **`"anthropic_thinking": "on_request"`**: an Anthropic request that asks for no thinking gets none (Anthropic's rule;
  Claude Code's short helper calls want it). The default thinks as the model's template does.

## Sampling

A request's `temperature`, `top_p`, `top_k`, `min_p`, `seed`, stop strings (`stop`; Anthropic `stop_sequences`) and
penalties (`presence_penalty`, `frequency_penalty`, `repetition_penalty` with `penalty_last_n`; llama.cpp's
`repeat_penalty` and `repeat_last_n` too) are honored. The config's `sampling` block sets the defaults for requests that
leave them out; what it leaves out comes from the model card's settings for the request's thinking state (thinking:
temperature 1.0, top_p 0.95, top_k 20; not thinking: temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5).
The example configs sample at temperature 0.7, top_p 0.95, top_k 20, min_p 0, presence_penalty 0. `"temperature": 0`
decodes greedy. `top_k` keeps at most 64 candidates.

Every token, drafted or not, is the model's own pick: with sampling, a draft is kept only when it equals the token the
model sampled for its position, so the drafts change the speed, never the distribution. The expert tiers adapt to the
conversation in the background, so a sampled answer is reproducible run to run only with `--adapt-every 0` in the
engine's arguments.

## LoRA adapters

Qwen3.6-35B-A3B and Qwen3.8-27B take LoRA adapters in llama.cpp's GGUF format (`convert_lora_to_gguf.py`'s), with
llama-server's options and API. In the config's `args` (or on the command line):

| Option | |
| --- | --- |
| `--lora F[,F]` | the adapters, each at scale 1 |
| `--lora-scaled F:S[,F:S]` | the same at scale S (the scale after the last `:`, so `D:\adapters\pi.gguf:0.5` reads) |
| `--lora-init-without-apply` | the adapters loaded at scale 0: requests or `POST /lora-adapters` set them |

```json
"args": {"lora": "D:\\adapters\\Qwen3.8-27B-pi-lora-F16.gguf,D:\\adapters\\Qwen3.8-27B-opencode-lora-F16.gguf"}
```

Each product with a pair adds `s B (A x)`, `s` the scale times the adapter's alpha over its rank (llama.cpp's). Several
adapters add up. The adapters are numbered in their order:

- `GET /lora-adapters`: `[{"id": 0, "path": ..., "scale": 1.0, "task_name": ..., "prompt_prefix": ...}, ...]`, the
  scales a request that names none takes.
- `POST /lora-adapters` with `[{"id": 0, "scale": 0.5}]` sets those scales, every adapter left out at 0.
- A request's `"lora": [{"id": 1, "scale": 1.0}]` (Chat Completions, Messages, Responses) runs it with those
  scales, every adapter left out at 0; an id that is not an adapter's, or a scale that is not a number, is a 400.
- The config's `lora_models` gives sets of scales names: `"lora_models": {"qwen3.8-27b-pi": [{"id": 0, "scale": 1}],
  "qwen3.8-27b-opencode": [{"id": 1, "scale": 1}]}`. A request naming one as its model (and no `"lora"` of its own)
  runs with them, so a client that only picks a model name picks an adapter; `/v1/models` lists them beside the
  model.

A conversation's state belongs to the scales it was read with: a request with other scales starts a conversation of
its own and reads its prompt (llama.cpp's cache does the same), the prompt cache keeps both, and with batch slots
(`parallel`) each runs in a session of its own, so requests with different adapters decode together. The adapters' pairs may be F16, BF16, F32 or Q8_0, at any of the model's
projections but the embedding, the router and the routed experts; activated LoRA (aLoRA) adapters are refused.
Flash-Next and Hy4-flash take no adapters.

## Behaviour

- **Streaming**: with `"stream": true` the thinking, the answer and tool calls (name first, then the arguments piece by
  piece) arrive as they are made; a long prompt sends keep-alives so agents do not time out, and `"return_progress":
  true` sends llama.cpp's `prompt_progress` (total, cache, processed, time_ms) per prompt chunk instead. Closing the
  connection stops the model, so the next request starts at once.
- **Tool calls** are taken out of the text when the request declares tools; without them the answer is the model's
  text as written. `"tool_call_recovery": true` in the config also reads calls the model writes in a few other forms
  (`<parameter=NAME>`, JSON inside `<tool_call>`, a bare `<function=NAME>` block).
- **Context**: a request longer than the context is refused (400), never cut. One whose `max_tokens` would run past the
  context is answered with `max_tokens` shortened to the room left (agents ask for 32,000 on every request);
  `"fit_max_tokens": false` in the config (or `--no-fit-max-tokens`) gives llama.cpp's 400 instead.
- **The prompt cache**: a request continues from the longest start of its prompt the engine holds, so the next turn of a
  chat or an agent reads only what is new. Each conversation keeps its own checkpoints of the sequence state in RAM
  (at each turn and in long prompts; `--prompt-cache N`, default 16 per conversation), and a conversation another
  request replaces (a chat app's title request, a subagent) is parked in RAM with its K/V and its checkpoints
  (`--conversation-cache-mib MIB`, default 16384; `--conversation-cache-slots N` caps their number, by default only
  the memory does) so switching back continues where it was. A parked conversation leaves only whole, the least
  recently used first; `--conversation-cache-min-tokens N` (default 1024) keeps shorter ones out. They never take
  the RAM below `--ram-reserve-mib MIB` (default 6144) available to the system, and leave between requests when it
  runs below it: the expert arena is pinned, so without it a park beside a large arena swaps the rest of the system
  instead (0: only the budget limits them). The Monitor's
  Conversation cache card shows them. Qwen3.6-35B-A3B and Qwen3.8-27B have the same cache. With batch slots
  (`parallel`) each of the engine's sessions holds a conversation with its checkpoints: a request runs in the
  session that holds its conversation, a subagent starts from what it shares with another one, also one that is
  still answering, and the park takes the conversations the sessions do not hold. The engine keeps the model's own
  token ids of each reply for the next turn.
  Responses report the reused tokens
  (`usage.prompt_tokens_details.cached_tokens`; Anthropic: `cache_read_input_tokens`).
- **Guards**: a reply that repeats one token 256 times ends as `"length"` (`"repeat_stop_tokens"` sets the run, 0 turns
  it off); a verify window whose logits are not finite emits nothing, the engine drops its cached state and the server
  retries the request once if nothing was sent yet. `"reasoning_close_retry": true` closes a thinking that ended with
  no answer and continues once; `"reasoning_loop_recovery": "stop"` or `"recover"` ends or restarts a reply whose
  reasoning repeats whole passages. A request the engine stays silent on for `"engine_silence_s"` (default 300) ends.
  The model writes `<|im_end|>` and `<|endoftext|>` as the one token that ends a turn, also when it means to quote them:
  drawn inside a code span or fence, after an opening quote, in a tool call's open parameter or in the middle of its
  thinking, the token becomes the marker's text and the reply goes on, up to 64 times a reply (`"literal_markers": false`
  ends the turn there).
- **A greeting**: a tool-using agent's prompt and an opening message with no task ("hello") can make the model invent
  instructions in its thinking, and act on them, at the top reasoning effort (on llama.cpp too).
  `"short_opening_tokens": 4` reads a conversation's first message of up to that many tokens as at `medium`, which has no
  effort sentence; the next turn's prompt starts differently, so it is read again.
- **JSON output**: `response_format` with `json_object` or `json_schema` (an object at the schema's root) asks the model
  for JSON and validates the answer: one generation, no hidden retry, a 502 (`structured_output_failed`) when it does
  not hold. The schema is checked with the `jsonschema` package when it is installed; without it, only that the answer
  is one JSON object. Prompting and validation, not constrained decoding.
- **Model aliases**: `"aliases": ["qwen", "local-model"]` lists the model under those names too.
- **`"prevent_sleep": true`** keeps Windows from sleeping while a request runs or waits.

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

Codex warns that it lacks the model's metadata; that is only the warning. `"codex_compaction_cache": true` in the
config renders Codex's compaction requests with the conversation's tools, so they reuse the cached conversation;
`"codex_thread_titles": true` answers Codex's thread-title requests without the engine.

## Pictures

With pictures on (an image projector beside the model, or a config's `vision` section), OpenAI `image_url` parts and
Anthropic `image` blocks (base64 data, http(s) URLs or local file paths) go through the server's image encoder and
reach the model as up to 1,024 tokens each
(the `vision` section's `max_tokens`; a 640x480 photo is ~300). JPEG, PNG, BMP and GIF are read as they are; WebP, TIFF,
AVIF and others are converted to PNG first. A picture is encoded once (one sent again, as chat apps do every turn, is
not encoded again), by default on the engine's main GPU in VRAM the encoder borrows from the engine for it (the
`vision` section's `gpu` and `lend`, which `-m`'s presets set: [SETUP.md](SETUP.md#a-model-file) says what each engine
lends), so the GPUs keep their VRAM for the model: 0.3-0.6 s a picture of ~1,000 tokens. On the development PC the
first question about such a picture takes 1.3 s with Qwen3.8-Flash-Next's IQ3_S (1.6 s with UD-Q4_K_XL) where the
CPU's encoder (`--no-vision-gpu`; the section's `threads`) takes 8.3-8.5 s on its 8-core Ryzen 7 9700X; with the engine and the
encoder on one RTX 3090, 4.4-4.5 s against 11.7-12.2 s (Qwen3.6-35B-A3B there: 2.2-2.3 s against 9.6-9.9 s, with the
same experts in VRAM and the same text speed as without pictures). An encoder that keeps its weights in VRAM
(`--no-vision-lend`; 0.7-1.2 GB of the card `cuda_device` names) encodes such a picture in 0.2 s: for a card with
VRAM to spare, as the development PC's last expert tier, where the first question takes 0.9-1.1 s. `--offline`
refuses URLs.
Qwen3.6-35B-A3B and Qwen3.8-27B read pictures with their repositories' own projectors.

## Image generation

Optional: [Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) in
[unsloth's GGUF quants](https://huggingface.co/unsloth/Qwen-Image-2.1-Turbo-GGUF), run by
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)'s `sd-cli`. The model is under the Qwen
Research License, not Apache 2.0: read [its license](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/main/LICENSE)
before using the images commercially.

It takes three things, then a start of the server with them:

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
- **An `images` section** in the config that names them:

  ```json
  "images": {"exe": "D:/sd/build/bin/Release/sd-cli.exe", "backend": "cuda",
             "diffusion_model": "D:/Models/qwen-image/qwen-image-2.1-turbo-Q8_0.gguf",
             "vae": "D:/Models/qwen-image/qwen_image_2.1_vae_bf16.safetensors",
             "llm": "D:/Models/qwen-image/Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf",
             "args": ["--diffusion-fa"], "gpu_policy": "auto"}
  ```

  `gpu` names the card `sd-cli` runs on (its `nvidia-smi` number or UUID), and the server's window says at the start
  which of the files is missing.

`POST /v1/images/generations` takes OpenAI's request: `prompt`, `size` (`WIDTHxHEIGHT`, each side a multiple of 32 from
256 to 2048; default 1024x1024), `n` (1-4), `response_format` (`b64_json`, or `url`: the server keeps the newest 64),
plus `seed` (the answer says which it used) and `steps` (8, Turbo's schedule). A job takes the server's turn on the GPU:
a chat request waits for the image and the other way round. The section's `gpu_policy` keeps the engine loaded beside
the job (`keep`), stops it for the job (`unload`: the next request starts it again, or the job's end with
`"reload": "eager"`, and reads its conversation again), or picks by the free VRAM (`auto`, the default).

## MCP tools

MCP servers named in the config's `mcp_servers` (Claude Desktop's format; or a file with `--mcp-config`) give the web
app's chat their tools: the server starts them (or connects to them over Streamable HTTP), lists them at `GET /mcp`, and
runs the calls the model makes. Only requests that ask for them get them (`"lamina_mcp": true`, which the web app
sends); other clients see the plain API. `--offline` starts none.

## Sharing the GPU

On a PC that also games or runs another model server, these keys (also server options) give the VRAM back:

| Config key | Option | |
| --- | --- | --- |
| `"idle_unload_s": 600` | `--idle-unload 600` | unload the model after 600 s without requests; the next request loads it again |
| `"min_free_vram_mib": 11000` | `--min-free-vram-mib 11000` | load an unloaded model only when that much VRAM is free (waiting up to 15 s), else answer 503 |
| `"before_load": "cmd"` | `--before-load "cmd"` | a command run before the model loads again (one that unloads another server's model); `before_load_timeout_s`, `before_load_strict`, `before_load_ignore_failure` say what a failure does |
| `"lazy_load": true` | `--lazy` | start the API unloaded; the first request loads the model |

`POST /unload` unloads it now (`?wait=1` waits for a running request first) and `POST /load` loads it, both with
`Content-Type: application/json`. Unloading ends the engine and the image encoder, so their VRAM and RAM go back (the
server keeps each GPU's CUDA context: about 200 MiB on the main GPU, less on the others); the model files stay in the
OS file cache, so loading again is quick while that RAM is not needed elsewhere.

**Behind [llama-swap](https://github.com/mostlygeek/llama-swap)**, which starts the server a request names and stops
the others: run the server itself with llama-swap's port, so that stopping it stops the engine too
(`cmd: sh -c 'cd /path/to/Lamina && exec build/lamina-server --config iq3_s.json --port ${PORT}'`),
with `checkEndpoint: /health` (the server listens once the model has loaded) and a `healthCheckTimeout` of a few
minutes. llama-swap passes the client's `Host` through: without an API key, add the name clients reach it by to
`allowed_hosts` ([below](#from-other-devices)), or those requests get 403.

## Replicas

`"replicas": N` with a `"gpu"` list of numbers cuts the GPUs into N groups and runs one engine on each, the requests spread over
them (or `"replicas": [{"gpus": [0, 1]}, {"gpus": [2, 3]}]`). Each engine pins its own experts in RAM.

## From other devices

The server listens on this PC only (`127.0.0.1`). `"host": "0.0.0.0"` in the config (or `--host`) opens it to the
network, with an API key: `"api_key": "a-long-secret"` (or the `LAMINA_API_KEY` variable; several keys
comma-separated or as a list), which clients then send as their API key. The server window prints this PC's addresses;
on Windows the firewall blocks them until lamina-server is allowed on private networks. Through a tunnel or a proxy, set the
key first, and add the address the page is reached at to `"trusted_origins"` for the web app's settings and MCP tools.
A request without the right key is refused before its body is read. The server has no deadline for a request's
headers, though: a client that sends them a byte at a time holds one of its threads, so on a network you do not trust,
put a reverse proxy in front of it.

- **Browsers (CORS)**: off by default; `"cors_origins": ["https://chat.example.com"]` lets those pages call `/v1/*`.
- **Host names**: without an API key the server answers only `Host` names it knows (localhost, IP addresses, this PC's
  name when it listens beyond it), against DNS rebinding; `"allowed_hosts"` (or `LAMINA_ALLOWED_HOSTS`) adds others.
- **Web pages without a key**: a browser's `POST` to `/v1/*` from a page other than the server's own, localhost, an
  allowed host or a trusted origin gets 403; clients that send no `Origin` (curl, the SDKs) are not affected.

## The config

The server reads a config file at its start (`--config FILE`), or with `-m` builds one from the model file
([SETUP.md](SETUP.md#a-model-file); `--write-config` writes it). The preset's keys: [SETUP.md](SETUP.md#the-config).

**Every key is an option of the command line too, and every option a key** (`lamina-server --help` lists them all,
each with its key). A value given on the command line takes the place of the config file's, and the file's takes the
place of the preset's that `-m` (or the file's `"model"`) builds from: `lamina-server --config my.json --port 8090
--temperature 0.6`. An option is its key with `-` for `_`, and a section's key comes after the section's name
(`vision.max_tokens` is `--vision-max-tokens`, `images.quant` `--images-quant`, `mcp.timeout_s` `--mcp-timeout-s`). Set
apart: the `sampling` block's keys go by llama.cpp's names (`--temperature` or `--temp`, `--top-p`, `--top-k`,
`--min-p`, `--presence-penalty`, `--frequency-penalty`, `--repetition-penalty`, `--penalty-last-n`, `--seed`),
`vision.mmproj` is `--mmproj` and `"vision": false` `--no-mmproj`, `model` also `-m`, `scripts` `--script`, and the
earlier spellings stay: `--lazy`, `--idle-unload`, `--open`, `--alias`, `--log-file`, `--reasoning-budget`. A switch
takes `--no-<name>` for off (`--no-answer-room`). A list takes commas (`--aliases a,b`) or JSON, `--env` and
`--images-env` take `NAME=VALUE` (again for more), and `--replicas`, `--mcp-servers` and `--before-load` JSON where
they need it. Any other option is the engine's: the config's `args`, which the command line's take the place of. A
path given on the command line starts from the folder the server was started in, one in a config from its `cwd`. A
key the server does not read (a typo) is named in its window and ignored. Only the commands are the command line's
alone: `--config`, `--write-config`, `--check`, `--help` and `--version` (`-v`).

| Key | |
| --- | --- |
| `port`, `host`, `api_key` | where it listens and the key it asks for |
| `cors_origins`, `trusted_origins`, `allowed_hosts` | browsers and host names ([above](#from-other-devices)) |
| `sampling`, `reasoning_effort`, `reasoning_budget_tokens`, `answer_room` | the defaults for requests ([above](#thinking)) |
| `fit_max_tokens`, `anthropic_thinking`, `effort_position`, `aliases` | [above](#behaviour) |
| `lora_models` | model names that run with the engine's LoRA adapters ([above](#lora-adapters)) |
| `tool_call_recovery`, `repeat_stop_tokens`, `reasoning_close_retry`, `literal_markers`, `short_opening_tokens`, `reasoning_loop_recovery`, `engine_silence_s` | [above](#behaviour) |
| `chat_template_file` | a chat template file in place of the model file's ([the README](../README.md#a-fixed-chat-template)) |
| `codex_compaction_cache`, `codex_thread_titles`, `env_to_user` | Codex's compaction and titles; opencode's `<env>` block moved to the first user message, so the system turn stays the same between sessions |
| `idle_unload_s`, `min_free_vram_mib`, `before_load`, `lazy_load` | [sharing the GPU](#sharing-the-gpu) |
| `mcp_servers`, `images`, `replicas`, `parallel` | MCP tools, image generation, replicas; `parallel`: the batch slots of Qwen3.6 and Qwen3.8-27B (`--batch`, one GPU) |
| `api_monitor`, `open_browser`, `prevent_sleep` | the request monitor at `/api-monitor` (the last 100 prompts and answers in memory); a browser at the start; Windows awake while busy |
| `max_queue`, `queue_timeout`, `shutdown_timeout`, `offline`, `ready_file`, `mcp_config` | the calls that may wait and for how long; the time running requests get at a stop; no network sources; an instance record for a supervisor; MCP servers from a file |
| `model`, `vision`, `model_name`, `arch`, `log`, `cwd`, `env`, `lib_dirs`, `replica_cpus`, `replica_load_tokens` | the model and its pictures, its name in the API, the engine's log, folder and environment, the replicas' CPUs and routing ([SETUP.md](SETUP.md#the-config)) |
| `gpu`, `args` | the GPUs as cards in their roles, each with its settings ([SETUP.md](SETUP.md#the-gpus)); the engine's options that differ from its defaults, an object or a list ([SETUP.md](SETUP.md#a-model-file)) |

The web app's Model settings card shows the values the server runs with (the command line's over the config file's
over the preset's, each set on the command line marked so) of `sampling` (`temperature`, `top_p`, `top_k`, `min_p`),
`reasoning_budget_tokens`, `fit_max_tokens`, `anthropic_thinking`, `effort_position`, `aliases`, `idle_unload_s`,
`lazy_load`, `engine_silence_s`, `api_monitor`, `open_browser` and `vram_reserve_mib`, the main GPU's reserve: the
first card's `reserve` in `"gpu"` (a change goes there and takes an earlier `--vram-reserve-mib` out of `"args"`), in
a config whose `"gpu"` is numbers the engine's `--vram-reserve-mib`.

- **Apply** changes them for the running server, at once. `effort_position`, `lazy_load`, `open_browser` and
  `--vram-reserve-mib` are read when the server starts (marked *at start-up*): a change to them waits for the next
  start, once saved.
- **Save configuration**, there when the server was started with `--config`, writes into that file the values that
  are not what it gives (the earlier file kept as `<config>.bak`) and leaves its other keys as they are. In a config
  that names a model and has no `"args"`, a value equal to the preset's takes the key out of the file, an empty one
  over a preset's value is written as `null`, and the engine's option is not written: such a config holds none
  (`--write-config` writes one with `"args"`).
- Without a config file (`-m` alone) the card's changes last until the server stops, and the start-up settings come
  from the command line.
