// src/program/options.cpp - the engine's options as the help lists them (options.hpp).
#include "lamina/program/options.hpp"

#include <cstring>

namespace lamina::program {

namespace {

constexpr unsigned C = kListCommon, S = kListServed;

}   // namespace

const std::vector<OptionHelp>& engine_options() {
    static const std::vector<OptionHelp> table = {
        {"", "", 0, "The model's files"},
        {"--native", "MODEL.gguf", C,
         "the model's GGUF, its first shard (the others are found by name; --model on the other paths too). Its "
         "architecture picks the engine's path: Qwen3.8-Flash-Next, Qwen3.6-35B-A3B and Qwen3.8-27B, Hy4-flash or "
         "Inkling 2 Mini (lamina: -m)"},
        {"--ple-gguf", "PATH", S,
         "Flash-Next: the per-layer embedding table from another GGUF, such as one of unsloth's engram tables "
         "(default: the model's shard that holds it)"},
        {"--ple-io", "direct|ram|mmap", S,
         "Flash-Next: the table's reads. direct (the default): from the drive past the OS file cache, the table never "
         "in RAM; ram: read into RAM after the start (28.8 GB in IQ4_NL), a new prompt a little sooner and generation "
         "slower; mmap: through a mapping, for comparisons"},
        {"--no-file-cache", "", S,
         "the experts (Hy4-flash and Inkling 2: the model file) read past the OS file cache, which then keeps no "
         "second copy of them in RAM; each start reads them from the drive. --file-cache: through it, the default"},
        {"--load-threads", "N", S, "Inkling 2: the threads that read the model file (default 8)"},

        {"", "", 0, "GPUs"},
        {"--gpu", "CARD[,key=value...]", C,
         "a GPU the engine uses, the option again for each: the first runs the model, the others hold experts as "
         "tiers, in their order (up to three). CARD: a name only one card has (\"RTX 5070 Ti\"), a UUID or its start "
         "(GPU-f26e9ba6), a PCI bus id (10:00.0) or a number as nvidia-smi numbers the GPUs; auto after the first: "
         "every other GPU, the fastest link first. Its settings below. Qwen3.6 and Qwen3.8-27B: the cards are the "
         "layer split's GPUs in order, their settings unused; Inkling 2: a later card holds context cells, runs "
         "layers (layers=) or holds the RAM experts' copy (cache=). Example: --gpu GPU-75af3083 --gpu \"RTX 5070 Ti\" "
         "--gpu GPU-f26e9ba6,clocks=1395-2100/9751"},
        {"", "", 0,
         "A --gpu card's settings: --gpu CARD,key=value,..., a config's card {\"id\": CARD, \"key\": \"value\"}; sizes "
         "binary (8G, 700M), times with their unit (55us, 1.5ms, 1s)"},
        {"cache=auto|SIZE|N", "", C | S,
         "its VRAM for experts: auto (the default) all but its reserve, or a size (22G); the first card's also N "
         "expert slots. Inkling 2: the first card's experts at most; a later card's, a copy of the experts in RAM"},
        {"reserve=SIZE", "", C | S,
         "the VRAM it leaves free: the first card 700M (Inkling 2: 1.5G where the desktop draws on it); a tier 2G "
         "where the desktop draws on it, else 512M"},
        {"vram-only=SIZE|auto", "", C | S,
         "Flash-Next: the profile's experts it keeps only in its VRAM, out of the RAM, for experts that outgrow the "
         "RAM (auto: as many as keep them within it); the first card and the first two tiers, not with sleep"},
        {"prefetch=SIZE", "", C | S,
         "Flash-Next, a tier: each layer's likeliest experts no GPU holds, copied to it ahead while the RAM is idle "
         "(default 12M over a link of 20 GB/s or more, else none)"},
        {"clocks=MIN-MAX/MEM", "", C | S,
         "Flash-Next, a tier; Inkling 2, any card: its graphics and memory clocks in MHz, locked through NVML while "
         "the engine works (1395-2100/9751 for an RTX 3090; a memory clock of 0: the driver's); needs administrator "
         "rights (root on Linux)"},
        {"sleep=TIME", "", C | S,
         "Flash-Next, a tier after the first: it sleeps (its experts to the CPU, its lock released) while the CPU "
         "pool would need less than TIME a layer without it: the card's awake power saved for some speed"},
        {"layers=A-B", "", C | S,
         "Inkling 2, a later card: it runs those layers (6-17) with their weights and their part of the context"},

        {"", "", 0, "The tiers and the CPU"},
        {"--clocks-hold", "TIME", S,
         "Flash-Next and Inkling 2, --serve: the cards' clock locks (clocks=) stay this long after a request "
         "(default 5s, Inkling 2 1s); a locked card draws its awake power meanwhile"},
        {"--clocks-lead", "TIME", S,
         "Flash-Next, --serve: the locks are asked for this long before a batched prompt part ends, by its progress "
         "(default off: from its last chunk on); a lock is at speed ~0.6 s after it is asked for"},
        {"--tier-min", "SIZE", S,
         "Flash-Next: a layer's missed experts from which the tiers take their shares (default 4M; below it the CPU "
         "is quicker than their ~100 us round trip)"},
        {"--no-tier-adapt", "", S,
         "Flash-Next: the tiers keep the experts the start placed, while the main GPU's go on adapting "
         "(--tier-adapt: every card adapts, the default)"},
        {"--head-split", "F", S,
         "Flash-Next with a tier: the first tier's share of the output head's rows (default 0.45; 0: the whole head "
         "on the main GPU)"},
        {"--pcie-frac", "F", S,
         "Flash-Next: the share of a layer's experts no GPU holds that the main GPU reads from RAM over its link "
         "instead of the CPU computing them (default 0: over an x4 link the CPU is sooner)"},
        {"--pcie-mode", "kernel|dma|direct", S,
         "Flash-Next: how that share reaches the main GPU: a copy kernel in the window's graph (the default), the "
         "copy engine, or the expert kernels reading the RAM"},
        {"--pool-workers", "N", S,
         "the CPU expert pool's workers (default 0: a worker a physical core, on Flash-Next but the one that drives "
         "the GPUs); --cpu-threads on the other paths too"},

        {"", "", 0, "The context"},
        {"--max-context", "N", C | S,
         "the context in tokens. Flash-Next: under --serve 262,144, the trained context, else 4096; Qwen3.6 and "
         "Qwen3.8-27B: 32,768 (--ctx); Hy4-flash: 8,192; Inkling 2: what the free VRAM holds, up to 1,048,576 (set, "
         "a sequence ends where the VRAM does)"},
        {"--kv", "FORMAT", C | S,
         "the K/V's format. Flash-Next: fp16 (the default), int8 (half fp16's VRAM), k8v4 (8-bit keys, 4-bit values: "
         "23% less than int8) or q4_0 (45% less); the others: fp16 (the default) or q8_0 (about half). A smaller "
         "format leaves VRAM to experts and context, at a cost in precision"},
        {"--kv-grow", "", C | S,
         "Flash-Next, Hy4-flash and Inkling 2, --serve: the K/V takes VRAM only for the cells the requests reach, "
         "the experts holding the rest until a conversation needs it (the default); --no-kv-grow: the whole "
         "context's at the start"},
        {"--rope-scaling", "none|linear|yarn", S,
         "Flash-Next: rotary positions for a context past the trained 262,144 tokens, as llama.cpp computes them "
         "(default none); fixed for the run, so the output changes at every length"},
        {"--rope-scale", "F", S, "Flash-Next: the context's extension factor (2: 524,288 tokens)"},
        {"--rope-freq-base", "N", S, "Flash-Next: the rotation base (0, the default: the model's 1e7)"},
        {"--rope-freq-scale", "F", S, "Flash-Next: the frequency scale (0, the default: 1 / --rope-scale)"},
        {"--yarn-orig-ctx", "N", S, "Flash-Next: YaRN's trained context (0, the default: 262,144)"},
        {"--yarn-ext-factor", "F", S, "Flash-Next: YaRN's extrapolation mix (below 0, the default: 1 for yarn)"},
        {"--yarn-attn-factor", "F", S, "Flash-Next: YaRN's attention factor (default 1)"},
        {"--yarn-beta-fast", "F", S, "Flash-Next: YaRN's fast correction bound (default 32)"},
        {"--yarn-beta-slow", "F", S, "Flash-Next: YaRN's slow correction bound (default 1)"},

        {"", "", 0, "Prompts"},
        {"--prefill", "N|auto", C | S,
         "a prompt's chunk in tokens: larger chunks read long prompts faster and take more VRAM. Flash-Next: auto "
         "(the default) 16,384, halved until its buffers fit the caches' lendable slots; Qwen3.6 and Qwen3.8-27B: "
         "auto, 2,048 with a layer split or experts in RAM, else the largest of 4,096 to 1,024 that fits beside the "
         "experts (--chunk); "
         "Hy4-flash: 4,096; Inkling 2: 1,024 (both 0: every prompt through the decode's forwards)"},
        {"--feed-max", "N", S,
         "Flash-Next, --serve: a new prompt part of up to N tokens runs through verify windows instead of the batched "
         "prompt path, whose call has a fixed cost (default 46)"},
        {"--prompt-shift", "F", S,
         "Flash-Next with a tier: in prompt chunks of 8K tokens or more the first tier also computes the likeliest "
         "share F (0-1) of each layer's experts the main GPU holds (default 0)"},
        {"--prompt-split", "L[,M]", S,
         "Flash-Next with a tier: a prompt of more than one chunk runs in two stages, the first tier running each "
         "chunk's layers L and up while the main GPU runs the next chunk (default 0: off). L,M with two or more "
         "tiers: three stages, the tier after the first running the layers L to M-1. It pays where the tier reads a "
         "chunk about as fast as both GPUs together; it takes pinned RAM and lends the tier's cache while it runs"},
        {"--prompt-split-from", "N", S,
         "Flash-Next with --prompt-split: a prompt part of N tokens or more that fits one chunk is read as two halves "
         "through the stages (default 0: in one chunk)"},
        {"--prompt-tiers", "N", S,
         "Flash-Next with two or more tiers: the first N tiers compute their experts in prompts (default 1: the later "
         "tiers' experts stream to the first; 2: the second computes those it holds, at its card's VRAM and power)"},
        {"--no-prefill-borrow", "", S,
         "Flash-Next: the prompt path's buffers keep VRAM of their own instead of borrowing the expert caches' last "
         "slots while a prompt runs: fewer experts in VRAM, nothing to refill after a prompt"},
        {"--no-warmup", "", S,
         "Flash-Next, Qwen3.6 and Qwen3.8-27B, --serve: no warm-up before the engine is ready: a quicker start, and "
         "the first request loads the prompt path's kernels"},

        {"", "", 0, "Drafts"},
        {"--drafter", "FILE", C | S,
         "the model's drafter, a file of its own (--mtp, --dspark and --dfly, its earlier names, the same option on "
         "every path; another model's drafter is refused, named). Flash-Next: its draft layer, "
         "Lamina-Qwen3.8-Flash-Next-MTP.gguf (the download-draft-layer script fetches it; lamina finds it "
         "beside the model or the program), or a folder of its earlier layout; without it, a token a verify window, "
         "or prompt lookup's drafts. Inkling 2: its block drafter, a dspark-*.gguf or the checkpoint's "
         "mtp.safetensors (lamina takes the smallest dspark-*.gguf beside the model, else in its dspark/ "
         "folder, else mtp.safetensors beside it): ~1.2 GiB of VRAM with its cells and buffers. Hy4-flash: Tencent's "
         "DFly2 drafter (the tencent/Hy4-flash checkpoint's last shard, its config.json beside it; lamina takes "
         "one beside the model, else in a folder inside the model's, where --drafts-per-step asks for drafts) in place "
         "of the file's MTP block: ~0.8 GiB of the main GPU, the block's 3.6 GiB of RAM saved. Qwen3.6 and Qwen3.8-27B: "
         "none (they draft with the file's own MTP block; a file named is ignored)"},
        {"--spec", "T", C | S,
         "Flash-Next: verify windows of up to T tokens, the last one and T-1 drafts (2-8, default 4): every further "
         "token routes its own experts"},
        {"--spec-min-p", "P", S,
         "drafts while their joint probability under the drafter stays at P or more (default 0.5; Qwen3.6, "
         "Qwen3.8-27B and Hy4-flash: 0, every draft); Flash-Next and Inkling 2 weigh a sampled request's drafts by "
         "their chance to pass at their positions' draws"},
        {"--spec-lookup", "N", S,
         "Flash-Next, Hy4-flash and Inkling 2: where the text's last N tokens or more occurred before, up to 7 of the "
         "tokens that followed there are the window's drafts (prompt lookup; default 16; 0: off)"},
        {"--mtp-window", "N", S,
         "Flash-Next: the context cells the draft layer attends to (default 32768; 0: every cell)"},
        {"--drafts-per-step", "N", S,
         "Qwen3.6, Qwen3.8-27B and Hy4-flash: at most N drafts (0-7) of the model file's own MTP block a step "
         "(default 0: none; Hy4-flash with --drafter: 0-5); Inkling 2: its drafter's drafts a window at most (default "
         "7; 0: no drafter). An earlier config's --mtp N (a number) is taken as it"},
        {"--draft-vocab", "N", S,
         "Qwen3.6 and Qwen3.8-27B: the draft head scores the first N token ids only (default: all): quicker drafts "
         "that miss the other tokens"},
        {"--dspark-reach", "N", S,
         "Inkling 2: the context cells the drafter's layer without a window keeps, 4.5 KiB each (default 131,072; "
         "32,768 with experts in RAM; no more than a set --max-context)"},
        {"--no-mtp", "", S,
         "Hy4-flash: the file's MTP block left out (no drafts without --drafter; 3.6 GiB of RAM saved)"},

        {"", "", 0, "The prompt cache (--serve)"},
        {"--prompt-cache", "N", S,
         "Flash-Next, Qwen3.6, Qwen3.8-27B and Inkling 2: the checkpoints of the sequence state each conversation "
         "keeps in RAM, so that a request continues from the longest start of its prompt the engine holds (default "
         "16; 0: off)"},
        {"--prompt-cache-every", "N", S,
         "Flash-Next, Qwen3.6, Qwen3.8-27B and Inkling 2: the checkpoints' interval inside a long prompt, in whole "
         "chunks (default 16384)"},
        {"--conversation-cache-mib", "MIB", S,
         "RAM for the conversations another request replaced, their K/V and checkpoints, so that switching back "
         "continues where it was; the least recently used leaves first (default 16384; Hy4-flash: --park-mib too; 0: "
         "none in RAM)"},
        {"--conversation-cache-slots", "N", S,
         "Flash-Next, Qwen3.6, Qwen3.8-27B and Inkling 2: such conversations at most (default 0: as many as fit)"},
        {"--conversation-cache-min-tokens", "N", S,
         "Flash-Next, Qwen3.6, Qwen3.8-27B and Inkling 2: a shorter conversation is not kept (default 1024)"},
        {"--ram-reserve-mib", "MIB", S,
         "the conversations take no RAM below MIB available to the system, and leave between requests when it runs "
         "below (default 6144; 0: only their budget limits them)"},
        {"--conversation-spill-dir", "DIR", S,
         "the conversations the RAM does not hold, and those it evicts, go to files in a folder of the engine's own "
         "under DIR, read back when a request continues them (default: none)"},
        {"--conversation-spill-mib", "MIB", S,
         "those files' size at most (default: 32 GiB or a quarter of the disk's free space, the less; never its last "
         "4 GiB free; 0: none)"},
        {"--park-vram", "0|1", S,
         "Inkling 2: a parked conversation's K and V stay on the cards until the VRAM is needed (1, the default; not "
         "with experts in RAM); 0: to RAM at once"},

        {"", "", 0, "The experts"},
        {"--expert-profile", "P", C | S,
         "Flash-Next, Hy4-flash and Inkling 2: the ranking of the experts the GPUs are filled from: builtin, "
         "Flash-Next's and Inkling 2's in the programs (their default under --serve), or a file "
         "--expert-profile-save wrote"},
        {"--adapt-every", "N", S,
         "Flash-Next: every N verify rounds the GPUs swap in the conversation's most routed experts they lack "
         "(default 1); 0 (Hy4-flash too): the start's placement stays, and every run writes the same output"},
        {"--adapt-swaps", "N", S, "Flash-Next: an update's moves at most (default 96)"},
        {"--adapt-decay", "F", S,
         "Flash-Next: the share of the routing counts kept after an update, between 0 and 1 (default 0.7)"},
        {"--deterministic-adapt", "", S,
         "Flash-Next: each update settled before the next, so that runs repeat bitwise while the GPUs adapt "
         "(a little slower)"},
        {"--arena-place", "", S,
         "Flash-Next and Hy4-flash: at the start, the RAM is timed and the experts the CPU reads are kept out of its "
         "slow regions (as unpaired DIMMs make; ~2-5 s more at the start, nothing gained on RAM without them; "
         "--no-arena-place: off, the default)"},
        {"--arena-pin", "G", S,
         "Hy4-flash: GiB of the experts' RAM pinned (default 72): a pinned expert streams to a GPU over twice as fast, "
         "and the start takes longer"},
        {"--hc-q8", "", S,
         "Flash-Next: the hyper-connection projections in 8 bits only (the default: 644 MiB of VRAM, read ~30% "
         "faster); --no-hc-q8: BF16 copies, 1.27 GiB"},
        {"--expert-math", "fast|exact", S,
         "Flash-Next: the GPUs' expert arithmetic: fast (the default, llama.cpp's CUDA kernels') or exact (K-quant "
         "files: the CPU's, so the output does not depend on where an expert is computed; slower, for tests)"},
        {"--expert-vram", "G", S,
         "GiB of VRAM for the experts. Qwen3.6 on one GPU (default: what the card has free; less than they take: "
         "the CPU computes the rest); Hy4-flash: the main GPU's expert slots (default: its free VRAM less 700 MiB, "
         "1.5 GiB where the desktop draws on it); Inkling 2: the first card's cache="},
        {"--hybrid-rows", "N", S,
         "Qwen3.6 with experts in RAM: a prompt chunk of more rows streams its experts to the GPU, a smaller batch "
         "runs them from the VRAM slots and the CPU (default 128, at least 8)"},

        {"", "", 0, "Control vectors (Qwen3.8-Flash-Next)"},
        {"--control-vector", "F[,F]", S,
         "a control vector in llama.cpp's GGUF format (direction.<layer> tensors), applied to every hyper-connection "
         "stream after each layer at scale 1 (several files add up)"},
        {"--control-vector-scaled", "F:S[,F:S]", S, "the same at scale S"},
        {"--control-vector-layer-range", "A B", S, "the layers it applies after, inclusive (default 1 to the last)"},
        {"--cvec-mode", "add|project", S,
         "add: h + S d (llama.cpp's, the default); project: h - |S d| (h . v) v with v = d / |d|, a unit direction "
         "at S 1 removed"},
        {"--cvec-dir", "per-layer|single:L", S,
         "project: each layer's own direction (the default), or layer L's in every layer"},

        {"", "", 0, "LoRA adapters (Qwen3.6-35B-A3B and Qwen3.8-27B)"},
        {"--lora", "F[,F]", S,
         "adapters in llama.cpp's GGUF format, each at scale 1 (each product with a pair adds s B A x, s the scale "
         "times alpha over the rank)"},
        {"--lora-scaled", "F:S[,F:S]", S, "the same at scale S (the scale after the last colon)"},
        {"--lora-init-without-apply", "", S,
         "the adapters loaded at scale 0: a request's \"lora\" or POST /lora-adapters sets their scales"},

        {"", "", 0, "Layer splits and batch slots"},
        {"--layer-split", "LIST", S,
         "Qwen3.6 and Qwen3.8-27B: each GPU's layer count, N,N,.. in the cards' order (default: the fewest GPUs that "
         "hold the model, filled in order); Inkling 2: DEV:A-B, the layers A to B on the card of CUDA number DEV (a "
         "card's layers= gives it), or 0: no split of its own (where the first card lacks room, experts stay in RAM)"},
        {"--batch-mtp", "", S,
         "Qwen3.6 and Qwen3.8-27B with batch slots (lamina's --parallel) and --drafts-per-step: each "
         "session's step verifies its drafts too"},

        {"", "", 0, "Hy4-flash"},
        {"--prefill-min", "N", S,
         "a prompt part shorter than N tokens goes through the decode's forwards (default 24)"},
        {"--prefill-pass", "N", S,
         "a long prompt part's chunks of up to N rows, each in two halves, the main GPU's dense steps "
         "beside the tiers' experts, its buffers lent by the main GPU's expert slots (default 16384; 0: none)"},
        {"--prefill-lend", "N", S,
         "an answer that has run N tokens lends the prompt path's buffers on the main GPU to its expert "
         "slots until the next prompt part (default 0: off)"},
        {"--warm-ms", "N", S,
         "a request after a pause of a second or more runs a low-priority kernel on each GPU for N ms, so "
         "that their clocks are up (default 500; 0: off)"},
        {"--tier-doorbell", "", S,
         "each tier's shares of a forward queued at its start, behind doorbells the host rings (off by "
         "default: no faster, and the waiting tiers draw more power)"},

        {"", "", 0, "Inkling 2 Mini"},
        {"--layer-split-back", "N", S,
         "with layers= on a later card: the loop's last N of its 36 applications run on the first card, "
         "those layers' weights a second time there (default 0): a share of the work moved to the first card"},
        {"--cell-spread", "N", S,
         "with cells on later cards: the cards share the cells evenly from N tokens on, each scoring its "
         "share at once, for their power (default 0: from when the first card is full)"},
        {"--prefill-max", "N", S,
         "the rows a long prompt's chunk grows to while the free VRAM holds its larger buffers (default "
         "4096: 2.3 GiB more; at most --prefill: fixed chunks)"},

        {"", "", 0, "Set from the --gpu cards (CUDA numbers)"},
        {"--stage-gpus", "LIST", 0, "Qwen3.6 and Qwen3.8-27B: the layer split's GPUs, the cards in their order"},
        {"--tier-gpus", "LIST", 0, "Hy4-flash: the expert tiers' GPUs, the cards after the first"},
        {"--cell-gpus", "LIST", 0, "Inkling 2: the cards that hold the context's cells the first has no room for"},
        {"--cell-gpu-clocks", "SPEC[,SPEC]", 0, "Inkling 2: their clocks= (0: none)"},
        {"--cell-gpu-reserve-mib", "R[,R]", 0, "Inkling 2: their reserve= in MiB (-: by the card's use)"},
        {"--layer-split-clocks", "SPEC", 0, "Inkling 2: the clocks= of the card with layers="},
        {"--layer-split-reserve-mib", "N", 0, "Inkling 2: its reserve= in MiB"},
        {"--main-gpu-clocks", "SPEC", 0, "Inkling 2: the first card's clocks="},
        {"--expert-gpu", "D", 0, "Inkling 2: the later card with cache=, the RAM experts' copy"},
        {"--expert-cache", "auto|SIZE", 0, "Inkling 2: the first card's cache="},
        {"--vram-reserve-mib", "N", 0, "Inkling 2: the first card's reserve= in MiB"},

        {"", "", 0, "Serve mode (lamina sets these)"},
        {"--serve", "", C, "stay resident and take requests on stdin, as lamina sends them"},
        {"--vision", "", C, "--serve takes pictures too, the image encoder's rows beside a request's tokens"},
        {"--batch", "N", 0,
         "Qwen3.6 and Qwen3.8-27B: N sessions decoding together on one GPU (--slots; lamina's parallel)"},
        {"--tail-role-token", "ID", 0,
         "Flash-Next, Qwen3.6 and Qwen3.8-27B: a turn of this role right before the last one stays out of its "
         "checkpoint (lamina's effort_position end; default -1: off)"},
        {"--expert-profile-save", "P", 0,
         "Flash-Next, Hy4-flash and Inkling 2: what the run's routing learned, blended with the profile it started "
         "from, written to P at the end and between requests (lamina's expert_profile_save)"},
        {"--expert-profile-save-every", "MIN", 0,
         "minutes between those saves (Flash-Next 10, the others 1; 0: at the end only)"},

        {"", "", 0, "Generate mode, tests and measurements"},
        {"--tokens", "LIST", 0, "the prompt as comma-separated token ids"},
        {"--tokens-file", "PATH", 0, "the same from a file, commas or whitespace"},
        {"--max-new", "N", 0,
         "tokens to generate (Flash-Next: default 16; Qwen3.6 and Qwen3.8-27B: 64, --n too; Hy4-flash and Inkling 2: "
         "--n, 64)"},
        {"--stop-eos", "", 0, "Flash-Next: stop at an end-of-turn token (--serve always does)"},
        {"--eos-ids", "LIST", 0, "Flash-Next: the end-of-turn tokens instead of 248044,248046 (with --stop-eos)"},
        {"--temperature", "F", 0, "Flash-Next and Inkling 2: sample at this temperature (default 0: greedy)"},
        {"--top-p", "F", 0, "Flash-Next and Inkling 2: top_p (default 1)"},
        {"--top-k", "N", 0, "Flash-Next and Inkling 2: top_k (default 20)"},
        {"--min-p", "F", 0, "Flash-Next and Inkling 2: min_p (default 0)"},
        {"--seed", "N", 0, "Flash-Next and Inkling 2: the sampler's seed (default 0; Flash-Next: from the clock)"},
        {"--spec-follow", "PATH", 0,
         "Flash-Next: emit this continuation (token ids) and keep the drafts that match it, so that settings and "
         "builds compare on one text"},
        {"--force", "IDS", 0,
         "Qwen3.6, Qwen3.8-27B, Hy4-flash and Inkling 2: run along these tokens instead of the model's picks"},
        {"--emb", "FILE", 0,
         "Qwen3.6, Qwen3.8-27B, Hy4-flash and Inkling 2: the prompt's pictures, a file of the image encoder's rows"},
        {"--dump-full-logits", "F", 0, "Qwen3.6, Qwen3.8-27B, Hy4-flash and Inkling 2: every row's logits to F"},
        {"--bench-pp", "N", 0, "Qwen3.6 and Qwen3.8-27B: a prompt of N tokens read and timed"},
        {"--bench-tg", "N", 0, "Qwen3.6 and Qwen3.8-27B: N tokens generated and timed"},
        {"--no-graph", "", 0,
         "Qwen3.6, Qwen3.8-27B (--no-token-graph too), Hy4-flash and Inkling 2: every kernel launched each step "
         "instead of its captured graph (slower)"},
        {"--expert-slots", "N", 0, "Qwen3.6 with experts in RAM: N experts a layer in VRAM"},
        {"--main-cells", "N", 0,
         "Inkling 2: the first card keeps N cells at most, so that cells reach the others early"},
        {"--spec-split", "", 0,
         "Flash-Next: a window's tokens in two groups, the GPU's work on one beside the CPU's on the other (the same "
         "output, ~7% slower)"},
        {"--disable-hit", "", 0,
         "Flash-Next: every routed expert computed by the CPU, the cache allocated (exact's reference)"},
        {"--no-prompt-offload", "", 0,
         "Flash-Next with a tier: the prompt path streams the experts the main GPU lacks to it instead of computing "
         "them on the first tier"},
        {"--dump-routing", "PATH", 0, "Flash-Next: the routed experts' ids and weights per layer and position"},
        {"--window-hashes", "PATH", 0,
         "Flash-Next: per verify window a hash of its final rows and its argmaxes: builds that do the same arithmetic "
         "write the same file (a fixed text through --spec-follow, --adapt-every 0; not --serve)"},
        {"--window-logits", "PATH", 0,
         "Flash-Next: per emitted token the 64 likeliest with their log-probabilities, for comparisons with "
         "llama.cpp (not --serve)"},
        {"--dump-draft-data", "DIR", 0,
         "Flash-Next, --serve: each request's positions for the draft layer's training (the final rows, the windows' "
         "64 likeliest tokens and drafts); prompts then read in batches whatever --feed-max"},
        {"--stats", "", 0, "Flash-Next: the time by stage at the end"},
        {"--window-profile", "", 0,
         "Flash-Next with --stats: the GPU time of each stage of the verify window (timestamps in the graph, ~2 us "
         "each)"},
        {"--prefill-profile", "", 0,
         "Flash-Next: the GPU time of each section of the batched prompt path (--serve: each batched part's)"},

        {"", "", 0, "Help"},
        {"-h, --help", "", C, "the options most runs set"},
        {"--help-all", "", C, "every option, then the deprecated spellings"},
        {"-v, --version", "", C, "the program's version"},
    };
    return table;
}

const std::vector<DeprecatedOption>& deprecated_options() {
    static const std::vector<DeprecatedOption> table = {
        {"--main-gpu", "GPU", "the first --gpu's card"},
        {"--second-gpu", "G[,G,G]", "the later --gpu's cards, or auto (-1: none)"},
        {"--tier-gpus", "G[,G,G]", "the later --gpu's cards, as --second-gpu"},
        {"--expert-cache", "N|auto", "the main GPU's cache= (N slots; 0: the profile's count)"},
        {"--vram-reserve-mib", "M", "the main GPU's reserve= in MiB"},
        {"--vram-pin-gib", "G", "the main GPU's vram-only= in GiB"},
        {"--second-gpu-pin-gib", "G", "the first tier's vram-only= in GiB"},
        {"--second-gpu-gib", "G[,..]", "each tier's cache= in GiB (0: auto)"},
        {"--second-gpu-reserve-mib", "R[,..]", "each tier's reserve= in MiB"},
        {"--second-gpu-prefetch-mb", "M[,..]", "each tier's prefetch= in MiB"},
        {"--second-gpu-clocks", "C[,..]", "each tier's clocks= (0: the driver's)"},
        {"--second-gpu-sleep-us", "T[,..]", "each tier's sleep= in microseconds (0: none)"},
        {"--second-gpu-clocks-hold", "S", "--clocks-hold in seconds"},
        {"--second-gpu-clocks-lead", "S", "--clocks-lead in seconds"},
        {"--second-gpu-min-mb", "M", "--tier-min in MiB"},
        {"--second-gpu-adapt", "", "--tier-adapt"},
        {"--no-second-gpu-adapt", "", "--no-tier-adapt"},
    };
    return table;
}

const DeprecatedOption* deprecated_option(const char* flag) {
    for (const DeprecatedOption& d : deprecated_options())
        if (std::strcmp(d.flag, flag) == 0) return &d;
    return nullptr;
}

std::string wrapped(std::string_view text, size_t indent, size_t width, size_t first_col) {
    std::string out, line;
    size_t col = first_col;
    for (size_t at = 0; at < text.size();) {
        const size_t sp = std::min(text.find(' ', at), text.size());
        const std::string_view w = text.substr(at, sp - at);
        at = sp + 1;
        if (w.empty()) continue;
        if (!line.empty() && col + 1 + w.size() > width) {
            out += line + "\n" + std::string(indent, ' ');
            line.clear();
            col = indent;
        }
        if (!line.empty()) {
            line += ' ';
            ++col;
        }
        line += w;
        col += w.size();
    }
    return out + line + "\n";
}

std::string help_line(std::string_view left, std::string_view help, size_t width, size_t col) {
    std::string l = "  " + std::string(left);
    if (l.size() + 1 > col) return l + "\n" + std::string(col, ' ') + wrapped(help, col, width, col);
    l += std::string(col - l.size(), ' ');
    return l + wrapped(help, col, width, l.size());
}

std::string options_help(unsigned lists, bool deprecated, size_t width, size_t col) {
    std::string out = wrapped("An option that names models is theirs alone: Flash-Next (Qwen3.8-Flash-Next), Qwen3.6 "
                              "(Qwen3.6-35B-A3B), Qwen3.8-27B, Hy4-flash, Inkling 2 (Inkling 2 Mini).",
                              0, width, 0);
    const char* title = nullptr;
    for (const OptionHelp& o : engine_options()) {
        if (*o.flag == '\0') {
            title = o.help;
            continue;
        }
        if (lists != 0 && (o.lists & lists) == 0) continue;
        if (title != nullptr) {
            out += "\n" + wrapped(title, 0, width, 0);
            title = nullptr;
        }
        out += help_line(std::string(o.flag) + (*o.value ? " " : "") + o.value, o.help, width, col);
    }
    if (!deprecated) return out;
    out += "\n" + wrapped("Deprecated spellings, still taken: a card's own setting wins over them, a per-tier list's "
                          "values go to the tiers in order (its last to those past it), and of --gpu, --main-gpu and "
                          "--second-gpu the later takes the earlier's place (--second-gpu -1 after the cards: no "
                          "tiers); lamina names the ones a start uses",
                          0, width, 0);
    for (const DeprecatedOption& d : deprecated_options())
        out += help_line(std::string(d.flag) + (*d.value ? " " : "") + d.value, d.instead, width, col);
    return out;
}

}   // namespace lamina::program
