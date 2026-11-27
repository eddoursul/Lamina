// src/core/gpu_usable.cu - see include/lamina/core/gpu_select.hpp: whether this build's kernels run on a GPU.
#include "lamina/core/gpu_select.hpp"

#include <cuda_runtime.h>

#include <string>

#ifndef LAMINA_CUDA_ARCHS   // CMAKE_CUDA_ARCHITECTURES, comma-separated (CMakeLists.txt)
#define LAMINA_CUDA_ARCHS "?"
#endif

namespace lamina::core {
namespace {

// Built for the same architectures as every kernel of the engine: whether its image loads on a GPU is whether theirs
// do, as for CUB's EmptyKernel.
__global__ void usable_probe() {}

}   // namespace

std::string gpu_unusable(int device) {
    cudaDeviceProp p{};
    if (const cudaError_t e = cudaGetDeviceProperties(&p, device); e != cudaSuccess) {
        (void) cudaGetLastError();
        return std::string("its properties cannot be read (") + cudaGetErrorString(e) + ")";
    }
    const std::string cc = std::to_string(p.major) + "." + std::to_string(p.minor);
    if (p.major < 8)   // the kernels' tf32 mma and bf16 math (CMakeLists.txt refuses older architectures)
        return "its compute capability is " + cc + ", and Lamina needs 8.0 or newer (RTX 30, 40 or 50 series)";
    cudaFuncAttributes a{};
    cudaError_t e = cudaSetDevice(device);
    if (e == cudaSuccess) {
        e = cudaFuncGetAttributes(&a, usable_probe);
        if (e != cudaSuccess) gpu_reset_unused(device);   // the context made for the probe goes: nothing runs there
    }
    (void) cudaGetLastError();
    if (e == cudaSuccess) return {};
#ifdef _WIN32
#define LAMINA_BUILD_SCRIPT "build.bat"
#else
#define LAMINA_BUILD_SCRIPT "./build.sh"
#endif
    if (e == cudaErrorNoKernelImageForDevice || e == cudaErrorInvalidDeviceFunction)
        return "this build has no kernels for its compute capability, " + cc +
               " (built for " LAMINA_CUDA_ARCHS "): build Lamina for it (" LAMINA_BUILD_SCRIPT
               " in a clone builds for this PC's GPUs; "
               "CMAKE_CUDA_ARCHITECTURES with " +
               std::to_string(p.major * 10 + p.minor) + ")";
    if (e == cudaErrorUnsupportedPtxVersion || e == cudaErrorInsufficientDriver)
        return std::string("its driver cannot load this build's kernels (") + cudaGetErrorString(e) +
               "): update the NVIDIA driver";
    return std::string("it cannot run this build's kernels (") + cudaGetErrorString(e) + ")";
}

}   // namespace lamina::core
