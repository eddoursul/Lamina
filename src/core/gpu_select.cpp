// src/core/gpu_select.cpp - see include/lamina/core/gpu_select.hpp.
#include "lamina/core/gpu_select.hpp"

#include "lamina/core/cuda_fault.hpp"

#include <cuda_runtime.h>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <regex>
#include <thread>

namespace lamina::core {
namespace {

std::string lower(std::string s) {
    for (char& c : s) c = (char) std::tolower((unsigned char) c);
    return s;
}

std::string trimmed(const std::string& s) {
    const size_t a = s.find_first_not_of(" \t\"'"), b = s.find_last_not_of(" \t\"'");
    return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
}

GpuInfo info_of(int ordinal) {
    GpuInfo g;
    g.ordinal = ordinal;
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, ordinal) != cudaSuccess) {
        (void) cudaGetLastError();
        return g;
    }
    g.name = p.name;
    g.sms = p.multiProcessorCount;
    g.cc = p.major * 10 + p.minor;
    char buf[64];
    std::snprintf(buf, sizeof buf, "%04x:%02x:%02x.0", (unsigned) p.pciDomainID, (unsigned) p.pciBusID,
                  (unsigned) p.pciDeviceID);
    g.bus_id = buf;
    const auto* u = (const unsigned char*) p.uuid.bytes;
    std::snprintf(buf, sizeof buf, "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", u[0],
                  u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
    g.uuid = buf;
    return g;
}

// NVML from the driver (nvml.dll, libnvidia-ml.so.1), loaded at the first use: whether a display is active on a GPU and
// how many processes hold graphics contexts there, its PCIe link's generation now and at most (the lower of the
// card's and its root port's), and its clock locks (null where the library has none).  Without NVML every handle is
// null.
struct Nvml {
    using Init = int (*)();
    using ByBus = int (*)(const char*, void**);
    using Get = int (*)(void*, int*);
    using GetU = int (*)(void*, unsigned*);
    using Procs = int (*)(void*, unsigned*, void*);
    using Lock = int (*)(void*, unsigned, unsigned);
    using Reset = int (*)(void*);
    using ErrorString = const char* (*)(int);
    ByBus by_bus = nullptr;
    Get display = nullptr;
    GetU gen = nullptr, max_gen = nullptr;
    Procs graphics = nullptr;
    Lock lock_graphics = nullptr, lock_memory = nullptr;
    Reset reset_graphics = nullptr, reset_memory = nullptr;
    ErrorString error = nullptr;
    bool ok = false;
    Nvml() {
#if defined(_WIN32)
        HMODULE h = LoadLibraryA("nvml.dll");
        auto sym = [&](const char* n) { return h ? (void*) GetProcAddress(h, n) : nullptr; };
#else
        void* h = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
        auto sym = [&](const char* n) { return h ? dlsym(h, n) : nullptr; };
#endif
        const auto init = (Init) sym("nvmlInit_v2");
        by_bus = (ByBus) sym("nvmlDeviceGetHandleByPciBusId_v2");
        display = (Get) sym("nvmlDeviceGetDisplayActive");
        gen = (GetU) sym("nvmlDeviceGetCurrPcieLinkGeneration");
        max_gen = (GetU) sym("nvmlDeviceGetMaxPcieLinkGeneration");
        graphics = (Procs) sym("nvmlDeviceGetGraphicsRunningProcesses_v3");
        lock_graphics = (Lock) sym("nvmlDeviceSetGpuLockedClocks");
        lock_memory = (Lock) sym("nvmlDeviceSetMemoryLockedClocks");
        reset_graphics = (Reset) sym("nvmlDeviceResetGpuLockedClocks");
        reset_memory = (Reset) sym("nvmlDeviceResetMemoryLockedClocks");
        error = (ErrorString) sym("nvmlErrorString");
        ok = init != nullptr && by_bus != nullptr && display != nullptr && gen != nullptr && max_gen != nullptr &&
             graphics != nullptr && init() == 0;
    }
    static Nvml& get() {
        static Nvml n;
        return n;
    }
    void* handle(const std::string& bus_id) const {
        void* d = nullptr;
        return ok && !bus_id.empty() && by_bus(bus_id.c_str(), &d) == 0 ? d : nullptr;
    }
};

// a link at its full generation: one the copies woke (idle, PCIe 1.0 or 2.0)
bool awake(void* h) {
    unsigned now = 0, most = 0;
    return h != nullptr && Nvml::get().gen(h, &now) == 0 && Nvml::get().max_gen(h, &most) == 0 && now >= most;
}

// The desktop's GPU, or another's in graphics use: a display active there or a process's graphics context (dwm.exe and
// the desktop's apps on Windows).  The display flag alone was off for a few minutes while the PC sat unattended.  False
// with `known` false when NVML cannot say.
bool desktop(void* h, bool& known) {
    int active = 0;
    unsigned procs = 0;
    const int r = h != nullptr ? Nvml::get().graphics(h, &procs, nullptr) : -1;
    // 7: NVML_ERROR_INSUFFICIENT_SIZE, processes to list
    known = h != nullptr && Nvml::get().display(h, &active) == 0 && (r == 0 || r == 7);
    return known && (active != 0 || r == 7 || procs > 0);
}

}   // namespace

bool gpu_in_desktop_use(const std::string& bus_id) {
    bool known = false;
    return desktop(Nvml::get().handle(bus_id), known);
}

bool gpu_in_desktop_use(const std::string& bus_id, bool& known) {
    known = false;
    return desktop(Nvml::get().handle(bus_id), known);
}

uint64_t gpu_default_reserve(int device) {
    bool known = false;
    const bool in_use = gpu_in_desktop_use(info_of(device).bus_id, known);
    return (uint64_t) (known && !in_use ? 700 : 1536) << 20;
}

namespace {
std::atomic<int> g_engine_gpus{-1};
}   // namespace

void engine_gpu_count(int n) { g_engine_gpus = n; }

std::vector<GpuInfo> list_gpus(bool all) {
    std::vector<GpuInfo> out;
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) {
        (void) cudaGetLastError();
        return out;
    }
    if (const int engine = g_engine_gpus.load(); !all && engine >= 0) n = std::min(n, engine);
    for (int i = 0; i < n; ++i) out.push_back(info_of(i));
    return out;
}

int find_gpu(const std::vector<GpuInfo>& gpus, const std::string& spec_in, std::string& why) {
    const std::string spec = trimmed(spec_in), ls = lower(spec);
    if (spec.empty()) {
        why = "no GPU named";
        return -1;
    }
    if (gpus.empty()) {   // CUDA's reason: no driver, one too old for this build's CUDA runtime, none visible
        int n = 0;
        const cudaError_t e = cudaGetDeviceCount(&n);
        (void) cudaGetLastError();
        why = "no CUDA device is visible";
        if (e != cudaSuccess) why += std::string(" (") + cudaGetErrorString(e) + ")";
        if (e == cudaErrorInsufficientDriver)
            why += ": install an NVIDIA driver for CUDA " + std::to_string(CUDART_VERSION / 1000) + " or newer";
        return -1;
    }
    std::vector<const GpuInfo*> hits;
    const bool number =
        std::all_of(spec.begin(), spec.end(), [](char c) { return std::isdigit((unsigned char) c) != 0; });
    if (number) {   // an ordinal, else a part of a name ("3090")
        const int k = std::atoi(spec.c_str());
        for (const GpuInfo& g : gpus)
            if (g.ordinal == k) return k;
    }
    static const std::regex bus(R"(^(?:([0-9a-fA-F]+):)?([0-9a-fA-F]+):([0-9a-fA-F]+)(?:\.[0-9a-fA-F]+)?$)");
    std::smatch m;
    if (ls.rfind("gpu-", 0) == 0) {   // a UUID, or its start
        for (const GpuInfo& g : gpus)
            if (!g.uuid.empty() && lower(g.uuid).rfind(ls, 0) == 0) hits.push_back(&g);
    } else if (std::regex_match(spec, m, bus)) {   // [domain:]bus:device[.function], hexadecimal
        const unsigned long dom = m[1].matched ? std::strtoul(m[1].str().c_str(), nullptr, 16) : 0;
        const unsigned long b = std::strtoul(m[2].str().c_str(), nullptr, 16);
        const unsigned long d = std::strtoul(m[3].str().c_str(), nullptr, 16);
        for (const GpuInfo& g : gpus) {
            unsigned gd = 0, gb = 0, gv = 0;
            if (std::sscanf(g.bus_id.c_str(), "%x:%x:%x", &gd, &gb, &gv) == 3 && gd == dom && gb == b && gv == d)
                hits.push_back(&g);
        }
    } else {   // a name: a whole one first, then a part of one (a number only as a word of it: "3090", not "7")
        auto part = [&](const std::string& name) {
            for (size_t at = name.find(ls); at != std::string::npos; at = name.find(ls, at + 1)) {
                const size_t end = at + ls.size();
                if (!number || ((at == 0 || !std::isalnum((unsigned char) name[at - 1])) &&
                                (end == name.size() || !std::isalnum((unsigned char) name[end]))))
                    return true;
            }
            return false;
        };
        for (const GpuInfo& g : gpus)
            if (lower(g.name) == ls) hits.push_back(&g);
        if (hits.empty())
            for (const GpuInfo& g : gpus)
                if (part(lower(g.name))) hits.push_back(&g);
    }
    if (hits.size() == 1) return hits[0]->ordinal;
    if (hits.empty() && number) {
        why = "no visible CUDA device " + spec + " (" + std::to_string(gpus.size()) + " visible) or GPU named so";
    } else if (hits.empty()) {
        why = "no visible GPU is " + spec;
    } else {
        why = "several GPUs match " + spec + " (name one by its UUID):";
        for (const GpuInfo* g : hits) why += " " + std::to_string(g->ordinal) + " " + g->name + " " + g->uuid + ";";
        why.pop_back();
    }
    return -1;
}

namespace {
std::mutex g_keep_mu;
std::vector<int> g_keep;   // a device's keeps, by its ordinal
}   // namespace

void gpu_keep(int device, bool keep) {
    if (device < 0) return;
    const std::lock_guard<std::mutex> lk(g_keep_mu);
    if (g_keep.size() <= (size_t) device) g_keep.resize((size_t) device + 1, 0);
    g_keep[(size_t) device] = std::max(0, g_keep[(size_t) device] + (keep ? 1 : -1));
}

void gpu_reset_unused(int device) {
    (void) cudaSetDevice(device);
    {
        const std::lock_guard<std::mutex> lk(g_keep_mu);
        if ((size_t) device < g_keep.size() && g_keep[(size_t) device] > 0) return;
    }
    (void) cudaDeviceReset();
    (void) cudaGetLastError();
}

namespace {
std::vector<uint64_t> g_loan;   // a device's asked loan, by its ordinal (g_keep_mu)
}   // namespace

void gpu_loan_ask(int device, uint64_t bytes) {
    if (device < 0) return;
    const std::lock_guard<std::mutex> lk(g_keep_mu);
    if (g_loan.size() <= (size_t) device) g_loan.resize((size_t) device + 1, 0);
    g_loan[(size_t) device] = bytes;
}

uint64_t gpu_loan_asked(int device) {
    const std::lock_guard<std::mutex> lk(g_keep_mu);
    return device >= 0 && (size_t) device < g_loan.size() ? g_loan[(size_t) device] : 0;
}

std::vector<GpuProbe> probe_gpus(const std::vector<int>& devices, int main_device) {
    struct Dev {
        void* buf = nullptr;
        cudaStream_t s = nullptr;
        cudaEvent_t a = nullptr, b = nullptr;   // timing
        cudaEvent_t done[2] = {};               // the copies in flight, in turn
        long queued = 0;
        void* nvml = nullptr;
        bool ok = false, up = false;
    };
    const size_t n = devices.size();
    constexpr size_t kBytes = 64ull << 20;
    std::vector<GpuProbe> out(n);
    std::vector<Dev> d(n);
    const Nvml& nvml = Nvml::get();
    void* host = nullptr;
    const bool pinned = cudaHostAlloc(&host, kBytes, cudaHostAllocPortable) == cudaSuccess;
    bool all_known = true;   // every GPU's P-state readable
    for (size_t i = 0; i < n; ++i) {
        Dev& x = d[i];
        x.nvml = nvml.handle(info_of(devices[i]).bus_id);
        out[i].display = desktop(x.nvml, out[i].display_known) || !out[i].display_known;
        x.ok = pinned && cudaSetDevice(devices[i]) == cudaSuccess && cudaMalloc(&x.buf, kBytes) == cudaSuccess &&
               cudaStreamCreateWithFlags(&x.s, cudaStreamNonBlocking) == cudaSuccess &&
               cudaEventCreate(&x.a) == cudaSuccess && cudaEventCreate(&x.b) == cudaSuccess &&
               cudaEventCreateWithFlags(&x.done[0], cudaEventDisableTiming) == cudaSuccess &&
               cudaEventCreateWithFlags(&x.done[1], cudaEventDisableTiming) == cudaSuccess;
        all_known = all_known && (!x.ok || x.nvml != nullptr);
    }
    // The links woken, all at once: an idle one runs at PCIe 1.0 or 2.0 rates until 0.2-0.5 s of copies raise its
    // P-state (P8, or a display's P5), and its link with it.  Two copies in flight on each, a new one as soon as one is
    // done: with the link idle between pairs of copies, for no more than the 2 ms between the polls, the 5070 Ti stayed
    // in P5.  Until each link is at its full generation, at most 1 s; without NVML 0.6 s.  (Waiting for a P-state of P2
    // or faster ran into the 1 s limit on cold starts, the 5070 Ti's link at PCIe 5.0 by then.)
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        bool up = true;
        for (size_t i = 0; i < n; ++i) {
            Dev& x = d[i];
            if (!x.ok) continue;
            cudaSetDevice(devices[i]);
            while (x.queued < 2 || cudaEventQuery(x.done[x.queued % 2]) == cudaSuccess) {
                cudaMemcpyAsync(x.buf, host, kBytes, cudaMemcpyHostToDevice, x.s);
                cudaEventRecord(x.done[x.queued % 2], x.s);
                ++x.queued;
            }
            x.up = x.up || awake(x.nvml);
            up = up && x.up;
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if ((all_known && up) || ms >= (all_known ? 1000.0 : 600.0)) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    for (Dev& x : d)
        if (x.ok) cudaStreamSynchronize(x.s);
    // then each link alone: the best of five copies, and of five more while those were still faster.  A link can
    // report its full generation before it copies at it: about one start in fifty the 5070 Ti's first five read 7.3
    // GB/s here (54 in the others), which left its prefetch off and, with "auto", would put it behind a 3090.
    for (size_t i = 0; i < n; ++i) {
        Dev& x = d[i];
        for (int fives = 0; x.ok && fives < 40; ++fives) {
            double best = 0;
            for (int r = 0; r < 5; ++r) {
                float ms = 0;
                cudaSetDevice(devices[i]);
                cudaEventRecord(x.a, x.s);
                cudaMemcpyAsync(x.buf, host, kBytes, cudaMemcpyHostToDevice, x.s);
                cudaEventRecord(x.b, x.s);
                if (cudaEventSynchronize(x.b) == cudaSuccess && cudaEventElapsedTime(&ms, x.a, x.b) == cudaSuccess &&
                    ms > 0)
                    best = std::max(best, (double) kBytes / (double) ms / 1e6);
            }
            const bool faster = best > 1.05 * out[i].h2d_gbps;
            out[i].h2d_gbps = std::max(out[i].h2d_gbps, best);
            if (!faster) break;
        }
        cudaSetDevice(devices[i]);
        for (cudaEvent_t e : {x.a, x.b, x.done[0], x.done[1]})
            if (e) cudaEventDestroy(e);
        if (x.s) cudaStreamDestroy(x.s);
        if (x.buf) cudaFree(x.buf);
    }
    if (pinned) cudaFreeHost(host);
    (void) cudaGetLastError();
    cudaSetDevice(main_device);
    return out;
}

struct GpuClocks::State {
    std::mutex mu;
    std::condition_variable cv;
    std::thread thread;
    std::atomic<bool> want{false}, off{false};   // generation nears or runs; a lock was refused
    bool quit = false;
    uint64_t wishes = 0;                           // counts the calls that change what the locks should be
    std::chrono::steady_clock::time_point due{};   // a release's time
    // per GPU of gpus_: its tier sleeps; its lock is set, and since when
    std::vector<char> asleep, locked;
    std::vector<std::chrono::steady_clock::time_point> since;
};

namespace {
std::atomic<GpuClocks*> g_clocks{nullptr};   // the one that has held locks (an engine at a time), for release_now
}

void GpuClocks::release_now() {
    GpuClocks* c = g_clocks.load();
    if (c != nullptr) c->set(false);
}

GpuClocks::~GpuClocks() {
    if (st_ == nullptr) return;
    {
        GpuClocks* self = this;
        g_clocks.compare_exchange_strong(self, nullptr);
    }
    {
        const std::lock_guard<std::mutex> lk(st_->mu);
        st_->quit = true;
    }
    st_->cv.notify_all();
    st_->thread.join();
    delete st_;
}

bool GpuClocks::add(int device, const std::string& spec, std::string& why) {
    Gpu g;
    g.device = device;
    char rest = 0;
    if (std::sscanf(spec.c_str(), "%u-%u/%u%c", &g.gr_min, &g.gr_max, &g.mem, &rest) != 3 || g.gr_min == 0 ||
        g.gr_max < g.gr_min) {
        why = "a GPU's clocks are <graphics min>-<graphics max>/<memory> in MHz (1395-2100/9751), not " + spec;
        return false;
    }
    g.nvml = Nvml::get().handle(info_of(device).bus_id);
    gpus_.push_back(g);
    return true;
}

// One GPU's locks set or released; false when refused (said once: `off`), and then nothing of it stays locked.
bool GpuClocks::set_one(const Gpu& g, bool lock) {
    const Nvml& n = Nvml::get();
    int r = -1;
    const char* clock = "graphics";   // the one a refusal is about
    const bool nvml = g.nvml != nullptr && n.lock_graphics != nullptr && n.lock_memory != nullptr &&
                      n.reset_graphics != nullptr && n.reset_memory != nullptr;
    if (nvml) {
        if (lock) {
            r = n.lock_graphics(g.nvml, g.gr_min, g.gr_max);
            if (r == 0 && g.mem > 0) {
                clock = "memory";
                r = n.lock_memory(g.nvml, g.mem, g.mem);
                if (r != 0) n.reset_graphics(g.nvml);
            }
        } else {
            r = n.reset_graphics(g.nvml);
            const int rm = g.mem > 0 ? n.reset_memory(g.nvml) : 0;
            if (r == 0) r = rm;
        }
    }
    if (r == 0) return true;
    // 4: NVML_ERROR_NO_PERMISSION.  NVML documents the graphics clock's lock from Volta on (the RTX 20 series) and the
    // memory clock's from Ampere on (the RTX 30 series), "Not Supported" for a card without one.
    if (lock && !st_->off.exchange(true))
        std::fprintf(stderr, "lamina: GPU %d's %s clock could not be locked (%s%s): it runs at the driver's clocks\n",
                     g.device, clock,
                     g.nvml == nullptr || r < 0 ? "no NVML"
                     : n.error != nullptr       ? n.error(r)
                                                : "NVML refused",
                     r == 4 ? ": the lock needs administrator rights, root on Linux" : "");
    return false;
}

// Every GPU's locks set or released, whatever its tier does; false when one was refused.
bool GpuClocks::set(bool lock) {
    bool all = true;
    for (const Gpu& g : gpus_) all = set_one(g, lock) && all;
    return all;
}

void GpuClocks::run() {
    using clock = std::chrono::steady_clock;
    State& s = *st_;
    std::unique_lock<std::mutex> lk(s.mu);
    const size_t n = gpus_.size();
    while (!s.quit && !s.off.load()) {
        // the locks as they should stand: each GPU's whose tier is awake, while generation nears or runs and until
        // a release that is still to come; one call to the driver, then the wishes are read again
        const bool held = s.want.load() || clock::now() < s.due;
        size_t todo = n;
        for (size_t i = 0; i < n && todo == n; ++i)
            if ((held && !s.asleep[i]) != (s.locked[i] != 0)) todo = i;
        if (todo < n) {
            const bool lock = !s.locked[todo];
            lk.unlock();
            const bool ok = set_one(gpus_[todo], lock);
            lk.lock();
            if (lock && !ok) break;   // refused (`off`): none stays locked alone
            s.locked[todo] = lock ? 1 : 0;
            s.since[todo] = clock::now();
            continue;
        }
        const uint64_t seen = s.wishes;
        const auto changed = [&] { return s.quit || s.wishes != seen; };
        bool any = false;
        for (const char c : s.locked) any = any || c != 0;
        if (!s.want.load() && any) s.cv.wait_until(lk, s.due, changed);   // a release is to come
        else s.cv.wait(lk, changed);
    }
    for (size_t i = 0; i < n; ++i) {
        if (!s.locked[i]) continue;
        lk.unlock();
        set_one(gpus_[i], false);
        lk.lock();
        s.locked[i] = 0;
    }
}

void GpuClocks::start() {
    if (st_ != nullptr) return;
    st_ = new State;
    st_->asleep.assign(gpus_.size(), 0);
    st_->locked.assign(gpus_.size(), 0);
    st_->since.assign(gpus_.size(), std::chrono::steady_clock::time_point{});
    st_->thread = std::thread([this] { run(); });
    g_clocks.store(this);
    set_cuda_fault_release(&GpuClocks::release_now);   // a CUDA fault releases them before it is handed on
    static const int at_exit = std::atexit([] { GpuClocks::release_now(); });   // std::exit from a failed launch
    (void) at_exit;
}

void GpuClocks::hold() {
    if (gpus_.empty()) return;
    start();
    if (st_->want.load(std::memory_order_relaxed) || st_->off.load(std::memory_order_relaxed)) return;
    {
        const std::lock_guard<std::mutex> lk(st_->mu);
        st_->want.store(true);
        ++st_->wishes;
    }
    st_->cv.notify_one();
}

void GpuClocks::release_after(double seconds) {
    if (st_ == nullptr || !st_->want.load(std::memory_order_relaxed)) return;
    {
        const std::lock_guard<std::mutex> lk(st_->mu);
        st_->due = std::chrono::steady_clock::now() + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                                          std::chrono::duration<double>(seconds));
        st_->want.store(false);
        ++st_->wishes;
    }
    st_->cv.notify_one();
}

void GpuClocks::sleep(int device, bool asleep) {
    for (size_t i = 0; i < gpus_.size(); ++i) {
        if (gpus_[i].device != device) continue;
        start();
        {
            const std::lock_guard<std::mutex> lk(st_->mu);
            if ((st_->asleep[i] != 0) == asleep) return;
            st_->asleep[i] = asleep ? 1 : 0;
            ++st_->wishes;
        }
        st_->cv.notify_one();
        return;
    }
}

bool GpuClocks::at_speed(int device) const {
    for (size_t i = 0; i < gpus_.size(); ++i) {
        if (gpus_[i].device != device) continue;
        if (st_ == nullptr) return false;   // its lock was never asked for
        if (st_->off.load(std::memory_order_relaxed)) return true;
        const std::lock_guard<std::mutex> lk(st_->mu);
        return st_->locked[i] != 0 &&
               std::chrono::duration<double>(std::chrono::steady_clock::now() - st_->since[i]).count() >=
                   kAtSpeedSeconds;
    }
    return true;
}

}   // namespace lamina::core
