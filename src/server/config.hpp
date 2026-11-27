// src/server/config.hpp - the run config as the server reads it: the engine's arguments and what
// the config's other keys add to them, the GPUs it names, the environment the engine runs in, the offline policy.
#pragma once

#include "json.hpp"
#include "process.hpp"
#include "tokenizer.hpp"

#include <optional>
#include <string>
#include <vector>

namespace lamina {

/// a config that cannot be used: the server does not start (main prints "[lamina] config <what>")
struct ConfigError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/// the file's JSON (a BOM allowed: Notepad writes one); a mistake is said with its place in the file
json read_config(const std::string& path);
/// #2267: an option written as one element of "args" ("--ple-io mmap", "--ple-io=mmap") is split into two (said once)
void split_joined_args(json& cfg);
/// #1641: "--reasoning-effort X" in the args is the config's reasoning_effort, not an engine option (said once)
void lift_reasoning_effort_arg(json& cfg);
/// Qwen3.8-Flash-Next's draft layer (`arch` "qwen4exp"), which no model file holds: args that name none (--drafter FILE,
/// or an earlier name of it: artifact/drafter.hpp; of several the last, the others taken out) take kDraftLayerFile
/// (artifact/draft_layer.hpp) beside the model, else beside the program; a named file that is not there, or a folder
/// without the earlier layout's files, is taken out of them: the engine starts and generates without drafts.  The line for the server's window: the file found, or the warning with the download's command; empty when
/// the args name a file or a folder of the earlier layout that is there, or the model is another (Qwen3.6-35B-A3B and
/// Qwen3.8-27B draft with their files' own MTP block)
std::string resolve_draft_layer(json& cfg, const std::string& arch);
/// The drafter file beside the model of Inkling 2 (`arch` "inkling2" or "inkling2_loop") and of Hy4-flash
/// ("hy_v4_flash"), for args that name none (--drafter FILE, or an earlier name of it, as above) and ask for drafts
/// (--drafts-per-step: Inkling 2's 7 by default, Hy4-flash's none). Inkling 2: the smallest dspark-*.gguf in the
/// model's folder, else in its dspark/ folder (unsloth's layout), else the checkpoint's mtp.safetensors in the model's
/// folder (Lucebox's layout). Hy4-flash: DFly2, a safetensors shard of model.dfly_layers.* with the checkpoint's
/// config.json beside it (dfly_target_kv_layers), in the model's folder, else in a folder inside it (the first by
/// name; hidden ones not looked into), in place of the model file's MTP block. A named file that is not there is
/// taken out of them. The line for the server's window: the file found, or the warning; empty where none is found or
/// one is named.
std::string resolve_drafter(json& cfg, const std::string& arch);
/// the config's "args" as the engine's words: a list as it is, an object's flags in its order (true a switch, false
/// its --no- form, null left out, a list the flag's values; a value that is no text as JSON writes it)
json args_as_list(const json& args);
/// "args" as an object of flags when it reads back as the same words (each word after a flag its value, no flag twice,
/// no value before the first flag; a value becomes a number when it writes back the same); nullopt otherwise
std::optional<json> args_as_object(const json& args);
/// "args" given as an object into its list: the server reads lists (read_config, runconfig::load and the presets do it)
void args_to_list(json& cfg);
/// the config's "gpu" in the earlier form: nvidia-smi's numbers alone (0, [0, 2], "0,2"), the cards the engine sees
/// (CUDA_VISIBLE_DEVICES); empty for none, or for cards (gpu_cards)
std::vector<int> gpu_list(const json& cfg);
/// the config's "gpu" as --gpu's cards, in their order (program/gpu_cards.hpp): the first runs the model, the others
/// are its tiers; each a number, a name, a UUID's start, a PCI bus id or auto, or an object of "id" and the card's
/// settings ({"id": "GPU-f26e9ba6", "clocks": "1395-2100/9751"}).  nullopt for none, or numbers alone (gpu_list).
/// ConfigError for a card the engine would refuse.
std::optional<std::vector<std::string>> gpu_cards(const json& cfg);
/// gpu_cards' cards as objects: "id" and each setting by its key in lower case, all text
std::optional<std::vector<json>> card_objects(const json& cfg);
/// card objects as "gpu" is written: a card with its "id" alone as that id, else its object; every card an object
/// when each would be a number alone (numbers alone are gpu_list's, the cards the engine sees)
json cards_json(const std::vector<json>& cards);
/// a "gpu" value as one list: its numbers ("0,2" as 0 and 2) or its cards, each an item
json gpu_canonical(const json& gpu);
/// the cards the engine works on, as --gpu takes them: "gpu"'s, else the args' --gpu, else their --main-gpu and
/// --second-gpu (names alone); empty when none names one
std::vector<std::string> engine_cards(const json& cfg);
/// a config in the form --write-config writes: the older GPU flags as cards of "gpu" (--main-gpu and --second-gpu as
/// their names, the main GPU's flags and the per-tier lists as their settings, a card's own setting winning), but
/// beside a "gpu" of numbers and where a setting would go to an auto card; the other older names renamed
/// (--clocks-hold, --tier-min ...); with `flash` (a Qwen3.8-Flash-Next config) the options at the engine's default
/// left out; --native as "model", --vision as its section, "lib_dirs" left out, and "args" as an object (an empty
/// one too: a config without "args" is the preset's)
json modern_config(json cfg, bool flash);
struct GpuInfo {
    int index;
    std::string name, bus_id, uuid;
    int64_t memory_mib = 0;   ///< its VRAM
    int cc = 0;               ///< compute capability x 10 (86: 8.6), 0 when nvidia-smi does not say
    bool display = false;     ///< a display is active on it
    std::string driver{};     ///< the NVIDIA driver's version
};
/// the NVIDIA cards as nvidia-smi lists them (empty without nvidia-smi)
std::vector<GpuInfo> nvidia_gpus();
/// the card `spec` names, read as the engine reads --main-gpu / --second-gpu; -1 when none or several match, with
/// the reason in `why` when given
int find_gpu(const std::string& spec, const std::vector<GpuInfo>& gpus, std::string* why = nullptr);
/// the cards the engine works on (the config's "gpu", else --main-gpu and --second-gpu), the first its main one, among
/// `gpus` (nvidia-smi's list; a number is taken as given when it is empty); empty, with the reason in `why` when
/// given, when --main-gpu names no card or several, where the engine stops
std::vector<int> engine_gpus(const json& cfg, const std::vector<GpuInfo>& gpus, std::string* why = nullptr);
/// the same over nvidia-smi's list, read when the args name a GPU
std::vector<int> engine_gpus(const json& cfg);
/// #458: --tail-role-token for "effort_position": "end" (nullopt: the effort stays at the top); ConfigError for a
/// value other than "start" / "end"
std::optional<std::vector<std::string>> effort_end_args(const json& cfg, const Tokenizer& tok);
/// #481: seconds the engine may print nothing during a request (0: no limit); ConfigError for a bad value
double engine_silence_s(const json& cfg);
/// the engine's arguments: the config's, with what "vision", "parallel" and "expert_profile_save" ask for
std::vector<std::string> engine_args(const json& cfg);
/// the deprecated options a start uses, "FLAG VALUE: what takes its place" each, once: the engine's deprecated
/// spellings (program::deprecated_option) in its command line `line` (engine_args), and --native in place of "model".  `file`: the config as its file has it (the server writes its "model" into the merged
/// config's args as --native)
std::vector<std::string> deprecated_options(const json& file, const std::vector<std::string>& line);
/// An engine option on lamina's command line: the flag ("--max-context") and the values that follow it.
using EngineOption = std::vector<std::string>;
/// `opts` into the config's args, each in place of the args' own instances of its flag (with their values: the words
/// after a flag that do not start with "--") and of its other form ("--kv-grow" for "--no-kv-grow" and back; a drafter
/// file's --drafter for its earlier names, --mtp, --dspark and --dfly, and back: Hy4-flash's --no-mtp is no form of
/// them), at the end in their order.  The engine takes an option's last value, so the command line's wins either way; the args stay
/// without the old ones for a config that is written.
void apply_engine_options(json& cfg, const std::vector<EngineOption>& opts);
/// the model file of the engine's arguments (--native), relative ones the config's cwd's
std::string model_file(const json& cfg);
/// the engine's environment, set in this process before the engine starts: "gpu" (CUDA_DEVICE_ORDER and
/// CUDA_VISIBLE_DEVICES), the config's "env", the offline policy
void apply_engine_env(const json& cfg, bool offline);
/// After apply_engine_env, for the image encoder, which runs in this process too: where "gpu" is numbers (the process
/// narrowed to them) and the vision section's "cuda_device" is none of them, that card is visible after them and the
/// engine's devices are CUDA's first ones, as many as the numbers (core::engine_gpu_count).  `gpus`: nvidia-smi's
/// list, read when empty.
void show_vision_gpu(const json& cfg, std::vector<GpuInfo>& gpus);
/// the variables every child gets under --offline
const std::vector<std::pair<std::string, std::string>>& offline_env();
/// why this config cannot be strictly offline (empty: it can)
std::vector<std::string> offline_blockers(const json& cfg);

}  // namespace lamina
