// include/lamina/core/gpu_select.hpp - GPUs named in the options, whether this build's kernels run on one, a short
// probe that orders and sizes the GPU tiers, and a tier's clocks locked while the engine generates.
//
// A device ordinal moves when a card is added: here an RTX 5060 Ti on an M.2 adapter took bus 02 and made the RTX
// 3090 device 2 instead of 1, and its slot's bus 0F became 10, so a PCI bus id moves too.  A UUID stays with its
// card, and so does a name while no other card has it: --main-gpu and --second-gpu take any of the four.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace lamina::core {

/// One visible CUDA device, as the options can name it (no context is made to list it).
struct GpuInfo {
    int ordinal = -1;
    std::string name;     ///< cudaDeviceProp::name, "NVIDIA GeForce RTX 3090"
    std::string bus_id;   ///< "0000:10:00.0"
    std::string uuid;     ///< "GPU-f26e9ba6-34e6-e46a-a700-b8349f7e5456", as nvidia-smi prints it
    int sms = 0;
    int cc = 0;   ///< compute capability x 10 (86: 8.6)
};
/// The engine's devices are CUDA's first `n`: lamina narrows its process to the engine's cards
/// (CUDA_VISIBLE_DEVICES) and adds after them a card its image encoder runs on that is none of theirs.  n < 0 (as at
/// the start): every visible device.
void engine_gpu_count(int n);
/// The engine's devices, each one's ordinal its index: every visible one unless engine_gpu_count says fewer; `all`:
/// every visible one regardless.
std::vector<GpuInfo> list_gpus(bool all = false);

/// "" when the engine can run on `device`, else why not, to follow the GPU's name in a message: a compute capability
/// below 8.0, no kernels for it in this build (CMAKE_CUDA_ARCHITECTURES), or a driver that cannot load them.  The
/// last two are asked of the driver, as a launch would load a kernel (CUB's probe), before anything is made on
/// `device`: it is current after it, with its context, which goes again when the kernels do not run there.
std::string gpu_unusable(int device);

/// Whether the GPU at PCI bus id `bus_id` (nvidia-smi's "00000000:02:00.0") is in the desktop's use, as NVML sees it: a
/// display active there, or a process's graphics context on it (dwm.exe and the desktop's apps on Windows: the display
/// flag alone reads off while the monitor sleeps).  False where NVML cannot say (`known` false then).
bool gpu_in_desktop_use(const std::string& bus_id);
bool gpu_in_desktop_use(const std::string& bus_id, bool& known);
/// The VRAM a card leaves free where nothing sets it, in bytes: 700 MiB, or 1.5 GiB on one the desktop or another
/// graphics program draws on (gpu_in_desktop_use; also where NVML cannot say)
uint64_t gpu_default_reserve(int device);

/// The ordinal `spec` names: an ordinal ("2"), a UUID or a prefix of it that only one card has ("GPU-f26e9ba6"), a PCI
/// bus id ("10:00.0", "0000:10:00.0", nvidia-smi's "00000000:10:00.0"), or a name or a part of one that only one card
/// has ("RTX 3090", case-insensitive; a whole name wins over a part of another; a number no ordinal is, as a word of
/// one: "3090", not "7").  -1
/// with `why` when none or several match (then each one's ordinal, name and UUID).  lamina's find_gpu
/// (src/server/config.cpp) reads names the same way.
int find_gpu(const std::vector<GpuInfo>& gpus, const std::string& spec, std::string& why);

/// A GPU another part of this process keeps memory on (lamina's image encoder runs in the server's process,
/// on a card of its own choice): counted, a keep for each one that holds it.  The engine's own use of a card needs
/// none.
void gpu_keep(int device, bool keep);
/// The context a query made on `device`, a GPU the engine then leaves out, goes with all it holds
/// (cudaDeviceReset), unless the card is kept (gpu_keep): then nothing is done.  `device` is current after it.
void gpu_reset_unused(int device);

/// What another part of this process will borrow of `device` between requests (lamina's image encoder, where
/// its section says "lend": the serve protocol's LEND), in bytes; 0 takes the word back.  Said before the engine
/// loads: an engine that sizes itself by the card's free VRAM and has no cache to lend from (the qwen35 path) then
/// has that much to lend - its prompt path's staging buffers made that large, or the VRAM left free past its reserve.
void gpu_loan_ask(int device, uint64_t bytes);
/// The bytes asked of `device` (0: none).
uint64_t gpu_loan_asked(int device);

/// A tier's GPU as a short probe saw it: the pinned host-to-device rate of 64 MiB copies (GB/s, the best of five
/// after the link is woken, and of five more for as long as those were faster), and whether the desktop or another
/// graphics program uses it (NVML: a display active or a process's graphics context; unknown when NVML is not there:
/// true, the larger reserve).
struct GpuProbe {
    double h2d_gbps = 0;
    bool display = true;
    bool display_known = false;
};
/// The devices' probes: their links woken together (copies until each link is at its full PCIe generation, 0.4-0.5 s
/// here), then each one's copies timed alone; `main_device` current after.
std::vector<GpuProbe> probe_gpus(const std::vector<int>& devices, int main_device);

/// GPUs' clocks locked while the engine generates (--second-gpu-clocks): under a tier's short shares the driver lowers
/// that card's clocks between them, and its rows come late; locked all the time it draws its awake power while the
/// engine idles or reads a prompt.  The locks are NVML's (nvidia-smi's -lgc and -lmc: they need administrator rights
/// and stay until released or a reboot), set and released on a thread of this object's: a lock is at speed about
/// half a second after it is asked for.  Each GPU has its own: one whose tier sleeps (sleep) is left out.
class GpuClocks {
public:
    GpuClocks() = default;
    ~GpuClocks();   ///< releases the locks it holds
    GpuClocks(const GpuClocks&) = delete;
    GpuClocks& operator=(const GpuClocks&) = delete;

    /// `spec` as "<graphics min>-<graphics max>/<memory>" MHz ("1395-2100/9751"; a memory clock of 0 leaves that
    /// one to the driver): that lock for `device` (a CUDA ordinal).  False with `why` for another form; values
    /// outside the card's range are NVML's to judge, and it takes them without a word (above the range: the card's
    /// top clocks; below it: no lock).  Before the first hold().
    bool add(int device, const std::string& spec, std::string& why);
    /// Generation nears or runs: the locks are set unless they are, and a release that was coming is off.  Returns at
    /// once.
    void hold();
    /// Generation is over: the locks are released `seconds` from now unless hold() comes first.
    void release_after(double seconds);
    /// `device`'s tier sleeps or wakes (--second-gpu-sleep-us): asleep, its lock is released at once and hold()
    /// leaves it out.  Returns at once; a device without a lock of its own: nothing.
    void sleep(int device, bool asleep);
    /// Whether `device` runs at its locked clocks: its lock was set kAtSpeedSeconds ago or longer.  True as well for
    /// a device without a lock of its own and when the locks were refused (the driver's clocks, as without them).
    bool at_speed(int device) const;
    /// A lock is at speed 0.55 s after NVML returns (an RTX 3090, 100 ms samples).
    static constexpr double kAtSpeedSeconds = 0.6;
    /// The locks of the object that held some, released on the calling thread: for an exit that runs no destructor
    /// (std::exit runs it by itself; std::_Exit does not).
    static void release_now();

private:
    struct Gpu {
        int device = -1;
        void* nvml = nullptr;
        unsigned gr_min = 0, gr_max = 0, mem = 0;
    };
    struct State;
    void run();
    void start();
    bool set(bool lock);
    bool set_one(const Gpu& g, bool lock);
    std::vector<Gpu> gpus_;
    State* st_ = nullptr;
};

}   // namespace lamina::core
