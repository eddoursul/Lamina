#include "lamina/core/cuda_fault.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include <atomic>
#include <mutex>

namespace lamina::core {
namespace {

std::mutex g_mu;
std::atomic<bool> g_faulted{false};
std::string g_what;   // (under g_mu, set before g_faulted)
std::atomic<void (*)()> g_release{nullptr};
std::atomic<void (*)(const std::string&)> g_hook{nullptr};

/// cuDevicePrimaryCtxGetState through the runtime's entry points (no driver library linked), or null
decltype(&cuDevicePrimaryCtxGetState) primary_state() {
    static const auto fn = []() -> decltype(&cuDevicePrimaryCtxGetState) {
        cudaDriverEntryPointQueryResult q{};
        void* p = nullptr;
#if CUDART_VERSION >= 12050
        const cudaError_t e =
            cudaGetDriverEntryPointByVersion("cuDevicePrimaryCtxGetState", &p, 12000, cudaEnableDefault, &q);
#else   // the versioned query arrived in CUDA 12.5
        const cudaError_t e = cudaGetDriverEntryPoint("cuDevicePrimaryCtxGetState", &p, cudaEnableDefault, &q);
#endif
        if (e != cudaSuccess || q != cudaDriverEntryPointSuccess || p == nullptr) {
            (void) cudaGetLastError();
            return nullptr;
        }
        return (decltype(&cuDevicePrimaryCtxGetState)) p;
    }();
    return fn;
}

}   // namespace

std::string cuda_fault_probe() {
    int cur = -1, n = 0;
    if (cudaGetDevice(&cur) != cudaSuccess) cur = -1;
    if (cudaGetDeviceCount(&n) != cudaSuccess) n = 0;
    const auto state = primary_state();
    std::string found;
    for (int d = 0; d < n && found.empty(); ++d) {
        if (state != nullptr) {   // a GPU the process has no context on is not given one now
            unsigned flags = 0;
            int active = 0;
            if (state((CUdevice) d, &flags, &active) != CUDA_SUCCESS || !active) continue;
        } else if (d != cur) {
            continue;
        }
        cudaError_t e = cudaSetDevice(d);
        size_t free_b = 0, total_b = 0;
        if (e == cudaSuccess) e = cudaMemGetInfo(&free_b, &total_b);
        if (e != cudaSuccess && e != cudaErrorMemoryAllocation)
            found = "GPU " + std::to_string(d) + ": " + cudaGetErrorString(e);
    }
    if (cur >= 0) (void) cudaSetDevice(cur);
    return found;
}

void set_cuda_fault_release(void (*release)()) { g_release.store(release); }
void set_cuda_fault_hook(void (*hook)(const std::string& what)) { g_hook.store(hook); }

bool report_cuda_fault(const std::string& what) {
    {
        const std::lock_guard<std::mutex> lk(g_mu);
        if (g_faulted.load()) return false;
        g_what = what;
        g_faulted.store(true);
    }
    if (void (*release)() = g_release.load()) release();
    if (void (*hook)(const std::string&) = g_hook.load()) hook(what);
    return true;
}

bool cuda_faulted() { return g_faulted.load(); }

std::string cuda_fault_what() {
    const std::lock_guard<std::mutex> lk(g_mu);
    return g_what;
}

int failed_call_code() {
    if (cuda_faulted()) return kCudaFault;
    const std::string w = cuda_fault_probe();
    if (w.empty()) return 1;
    report_cuda_fault(w);
    return kCudaFault;
}

}   // namespace lamina::core
