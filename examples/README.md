# Example configs

The configs Lamina is developed with, on a PC with two RTX 3090s, an RTX 5070 Ti and 160 GB of RAM
(their speeds: [COMPARISON.md](../docs/COMPARISON.md)). They are built into `lamina-server` as the presets of `-m`,
which takes the model file, the GPU, the draft layer, the projector and the files a run writes from the PC instead
([SETUP.md](../docs/SETUP.md#a-model-file)); their paths are placeholders:

- `iq3_s.json`: Qwen3.8-Flash-Next, ISTA-DASLab's GSQ-RCO IQ3_S (~53 GB of RAM taken).
- `ud-q4_k_xl.json`: Qwen3.8-Flash-Next, unsloth's UD-Q4_K_XL (~81 GB of RAM taken).
- `qwen36-ud-q4_k_xl.json`: Qwen3.6-35B-A3B, unsloth's UD-Q4_K_XL, on the engine's qwen35 path: its layers split over
  the fewest GPUs that hold every expert, else on the main GPU with the CPU computing the experts it lacks.
- `qwen38-27b-ud-q6_k_m.json`: Qwen3.8-27B, unsloth's UD-Q6_K_M, on the same path, its layers split over the fewest
  GPUs that hold it at the full 262,144 cells, K and V in q8_0 (~36 GB of VRAM: a 24 GB and a 16 GB GPU).
- `hy4-flash-q4_k_m.json`: Hy4-flash, AngelSlim's Q4_K_M (both shards), on the engine's hy4 path: its dense weights
  and 524,288 cells of K and V on the main GPU (~16 GB of VRAM; `"kv": "q8_0"` in `"args"` takes them in 4.4 GiB
  instead of 7.5, which leaves more of the routed experts to the GPU at long contexts), the routed experts in RAM for
  the CPU (~90 GB), the Engram table (59 GiB) read from the drive.

- `inkling2-mini-ud-iq4_xs.json`: Inkling 2 Mini, unsloth's UD-IQ4_XS (the preset of the model's other quants too), on the engine's
  inkling2 path: the whole model on the main GPU (UD-IQ4_XS: ~16 GiB of VRAM at a short context, ~1 GB of RAM), the context what the VRAM left holds
  K and V for (32.0 KiB a token in FP16, 17.0 with `"kv": "q8_0"` in `"args"`; ~213,000 tokens on a 24 GB card in
  FP16, ~414,000 in q8_0), `"max-context"` in `"args"` to set one. No pictures and no drafts yet.

`lamina-server -m MODEL.gguf --write-config FILE` writes one with the PC's own files and GPUs. `run-<model>.bat.example`
is a launcher for each: copied to `run-<model>.bat`, it starts the example from the clone's folder, the model's files
where its paths say (`../Lamina-data`). Each names its model (`"model"`), its GPUs (`"gpu"`: the main one, then
`auto`) and in `"args"` the engine's options that differ from its defaults. Their `"vision"` section has the image
encoder on the engine's main GPU, in VRAM it borrows from the engine for each new picture (`"gpu": true, "lend":
true`, no card named: [SETUP.md](../docs/SETUP.md#a-model-file)); the development PC's configs keep it on a card with
VRAM to spare instead. Their engine options:
[SETUP.md](../docs/SETUP.md#engine-options); the GPUs' cards: [SETUP.md](../docs/SETUP.md#the-gpus); the server's keys:
[USING.md](../docs/USING.md#the-config).

## The development PC's own options

The development PC's configs name their GPUs as cards (`"gpu": ["GPU-75af3083", "RTX 5070 Ti", {"id": "GPU-f26e9ba6",
"clocks": "1395-2100/9751"}]`, its two RTX 3090s by their UUIDs' starts, where the examples take every other GPU with
`auto`), and its Flash-Next configs add options measured for its hardware, which the examples leave out. Each pays only
where a PC has the same trait: measure it through the server before keeping it. Its Qwen3.8-27B config puts the RTX
5070 Ti first (the first card, which `-m` sets to a PC's own choice): that card reads
prompts ~1.44x as fast as the RTX 3090 and takes the first 26 layers, the 3090 the other 38 with the head, and the
prompt chunks run on both at once, so the prompts read ~11% faster than in the other order.

- **`--arena-place`** (both): for RAM whose regions read at different rates, as unpaired DIMMs make (here two 32 GB and
  two 48 GB modules leave 32 GB on one channel), the engine times its RAM arena at the start and keeps the experts the
  CPU reads out of the slow region, again between requests. Elsewhere it finds nothing to move and costs ~2 s at the
  start.
- **`--no-file-cache`** (UD-Q4_K_XL): the engine reads the experts from the GGUF past the OS file cache. Where the RAM
  holds the 72 GiB arena and the cached 72 GB file together (this PC's 160 GB), the file settles in the cache on pages
  the later starts' arenas would have had: here generation fell 7-8% over a session's starts, and with the option it
  stays at a fresh boot's speed. Each start then reads the experts from the drive (here 4.3-4.4 GiB/s against 6 from
  the cache, ~5 s more). With less RAM the cache cannot keep the file beside the arena, and the option costs only the
  start time.
- **`--prompt-split 26`** (both): a prompt of more than one chunk runs in two stages, the first tier taking each
  chunk's layers 26-47 whole while the main GPU runs the next chunk. It pays where the tier alone reads a chunk about as
  fast as both GPUs together: this PC's 5070 Ti runs the dense steps 1.44x as fast as its 3090, and prompts of
  32K-128K tokens read 1.13-1.55x as fast. It takes 0.7 GiB more pinned RAM and lends 6-8 GiB of the tier's expert
  cache while such a prompt runs. Its layer needs measuring too, through the server: here generate mode put IQ3_S's
  best layer at 23 (6-11% faster than 26), while through the API 26 was 2-4% faster than 23.
  - Three more switches of the prompt path were measured here with the second RTX 3090 as the tier after the first;
    this PC's configs do not set them.
  - `--prompt-split L,M` runs three stages, that tier's GPU the layers L to M-1 between the main GPU's and the first
    tier's: at 17,35 IQ3_S's prompts of 64K / 128K tokens read 14% / 21% faster than in two stages through the
    server (10.4 / 18.9 s against 11.9 / 22.8), UD-Q4_K_XL's no faster (the first tier then streams its larger
    experts for three stages at once). It takes ~0.9 GiB more pinned RAM and ~0.25 GiB more of the first tier's VRAM,
    lends that tier's cache as well, and keeps its GPU at work from a prompt's first chunk (170-200 W through a
    prompt, where it otherwise idles). Without that tier the same value runs two stages at the layer halfway between.
  - `--prompt-split-from N` reads a prompt part of N tokens or more that fits one chunk as two halves through the
    stages. The stages' fixed cost sets N: here ~9K tokens for IQ3_S (a 12K part 10-12% sooner, 16K 17-19%, 8K 3%
    later; the server's 16K prompt in 4.0 s against 5.1 with three stages), and no size for UD-Q4_K_XL.
  - `--prompt-tiers 2` has the tier after the first compute the experts it holds in prompts, its input and sums over
    its own link: UD-Q4_K_XL's parts of 2-4K tokens read 15-25% sooner and its prompts of 64K-128K 6% (the first
    tier streams fewer of its large experts), IQ3_S's 4-10% later (its small experts stream sooner than rows cross
    two x4 links). It takes ~0.65 GiB of that card's VRAM and keeps the card at work through a prompt (~140 W).
- **The third GPU's `clocks=1395-2100/9751`** (all three, on its card): its graphics clock locked at 1395-2100 MHz
  and its memory clock at 9,751 while the engine generates (NVML's locks, the ones `nvidia-smi -lgc` and `-lmc` set). The engine sets them from a request's start, or from a long
  prompt's last chunk, until 5 s after the request (its own timing: the two switches below shorten it), so the card
  idles at ~12 W instead of the ~110 W it draws locked.
  It pays for a tier whose work is sparse: this PC's second RTX 3090, as the third GPU, gets only the verify windows'
  short shares, the Windows driver drops it to its idle clocks between them, its rows come late and its moves copy at
  a quarter of the link's rate, and generation takes 4-22% longer (with an RTX 5060 Ti in that place: 0-7%). A tier
  with more to do keeps its clocks: the same RTX 3090 as the only tier gains ~1% from the lock (1-3% on thinking
  text), and the first of two tiers, the display's card, nothing. No Linux driver was measured.
  - The clocks are a card's own: `nvidia-smi --query-gpu=name,clocks.max.graphics,clocks.max.memory --format=csv`
    reads each card's highest. Here the memory clock is locked at its highest and the graphics range starts at two
    thirds of its highest: under these shares the driver holds the clock at the range's floor, and a higher floor
    draws more. NVML takes values outside a card's range without a word: a faster card's (`2700-3090/14001`) held
    this RTX 3090 at 1,965 MHz and 149 W idle against 114 W, and values below its range lock nothing.
  - The engine's process needs administrator rights (root on Linux). Without them, or on a card NVML has no lock
    for (it documents the graphics clock's from the RTX 20 series on and the memory clock's from the RTX 30 series
    on; a memory clock of `0`, as in `1395-2100/0`, leaves that one to the driver), the engine says so at the first
    request and runs at the driver's clocks.
  - Each card has its own: with `auto` tiers every one takes the `auto` card's settings, so name the tiers to give
    each card its own values.
  - An engine ended by force leaves its lock: the next engine's first release ends it, or `nvidia-smi -rgc -rmc`.
  - `--clocks-hold 1s --clocks-lead 1s` (this PC's configs set them with the lock): the lock
    stays 1 s after a request instead of 5, and is asked for 1 s before a batched prompt part ends, by the part's
    progress, instead of from its last chunk on. Over an agent's session replayed here (60 requests, a third of its
    time in pauses of up to 12 s) that took 3.2-3.5% of the energy the GPUs and the CPU drew: the card's locked time
    in the pauses and through one-chunk prompts. No time that showed: the generation +0.2% over four rounds, where
    two runs of one arm differ by up to 2.2%. A lock is at speed ~0.6 s after it is asked for, so a shorter lead
    starts generation on a card still at its idle clocks; a short hold pays where the next request rarely comes
    within it (here 72% of an agent's come within a second).
  - A card's `sleep=TIME` (this PC's IQ3_S config gives the third GPU `sleep=55us`), on a tier after the first: it
    sleeps (its experts go to the tiers before it and the CPU, its lock is released) while the CPU pool would
    need less than T microseconds a layer without it, averaged over a few seconds of generation, and wakes when the
    text turns heavy. It gives up speed for the card's awake power, by an amount that is the model file's and the
    PC's: here IQ3_S at 55 slept through 59-63% of a session's rounds for 10.5% fewer joules a token, its time a
    token within ~1% of the config's (code and edits kept the card, worth 11-13% of their time), Gyro-L at 100
    through 51% for 6.7% fewer joules and 2% more time, and UD-Q4_K_XL's load never falls that low. `--stats` prints
    the measure at the end of a `lamina` run, predicted while the tier works and measured while it sleeps: start
    from what a thinking text shows, and compare whole sessions with and without the switch, several each (single
    runs differ by 2%).
- **`dfly`** (Hy4-flash): Tencent's DFly2 drafter, a download of its own (the tencent/Hy4-flash checkpoint's last
  shard, `model-00148-of-00148.safetensors`, 1.7 GB, with its `config.json` beside it), drafts in place of the file's
  MTP block, which the example drafts with: `"dfly": "<the shard>"` in the args. Here, over an agent's session
  replayed, 2.2% less time a token and 5.2% less energy, 3.6 GiB less RAM (no MTP block), 0.8 GiB more of the main GPU.
