// include/lamina/core/cuda_host.hpp - host-side helpers the engine's parts share: a GPU current for one scope, a CUDA
// call that may fail, mapped host memory, a bump allocator over one buffer, milliseconds on the steady clock.
#pragma once

#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace lamina::core {

/// `dev` current for one scope (when `set`: a tier's GPU, a remote runner's), `main_dev` again afterwards
struct DeviceScope {
    int main;
    bool set;
    DeviceScope(int dev, int main_dev, bool set_dev = true) : main(main_dev), set(set_dev) {
        if (set) cudaSetDevice(dev);
    }
    ~DeviceScope() {
        if (set) cudaSetDevice(main);
    }
    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;
};

/// `dev` current for one scope (-1: the current one stays) and the one current before it again afterwards; with `on`
/// false nothing at all (`fallback`: the one restored where the current one cannot be read)
struct OnDevice {
    int prev = -1;
    explicit OnDevice(int dev, bool on = true, int fallback = 0) {
        if (!on) return;
        prev = fallback;
        int cur = 0;
        if (cudaGetDevice(&cur) == cudaSuccess) prev = cur;
        if (dev >= 0 && dev != prev) cudaSetDevice(dev);
    }
    ~OnDevice() {
        if (prev >= 0) cudaSetDevice(prev);
    }
    OnDevice(const OnDevice&) = delete;
    OnDevice& operator=(const OnDevice&) = delete;
};

/// A CUDA call that may fail without ending the run (a checkpoint's copy): false then, the error cleared
inline bool soft(cudaError_t e) {
    if (e == cudaSuccess) return true;
    (void) cudaGetLastError();
    return false;
}

/// zeroed host memory the GPU maps: the host pointer in *h, the device's in *d
inline bool mapped(size_t bytes, void** h, void** d, unsigned flags = cudaHostAllocMapped) {
    if (cudaHostAlloc(h, bytes, flags) != cudaSuccess) return false;
    std::memset(*h, 0, bytes);
    return cudaHostGetDevicePointer(d, *h, 0) == cudaSuccess;
}

/// 256-byte-aligned pieces carved off one buffer; without one (base null) it only adds up their size
struct Bump {
    uint8_t* base = nullptr;
    uint64_t used = 0;
    template <typename T> T* take(uint64_t n) {
        T* p = base ? (T*) (base + used) : nullptr;
        used += (n * sizeof(T) + 255) & ~255ull;
        return p;
    }
};

inline double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

}   // namespace lamina::core
