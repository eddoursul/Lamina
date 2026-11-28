// include/lamina/core/cuda_fault.hpp - a CUDA fault (an illegal address, a launch failure: a GPU's context is lost for
// the whole process, and every later call on it fails) wherever a model path sees it: the clock locks released at
// once, then the fault handed on - lamina's server gives the engine up (no new engine in its process), a standalone
// engine ends.  An out-of-memory is no fault: the context goes on.
#pragma once

#include <string>

namespace lamina::core {

/// A run's exit code when a CUDA fault ended it
constexpr int kCudaFault = 3;

/// The first GPU whose context holds a fault, of those the process has made a context on: "GPU n: <error>"; empty
/// when none does.  Each is asked without waiting for its queued work (cudaMemGetInfo); the current device stays.
std::string cuda_fault_probe();

/// What releases the clock locks (GpuClocks registers it as it takes its first lock)
void set_cuda_fault_release(void (*release)());
/// What hands a fault on once the locks are released (the serve loop's FAULT line, a standalone run's end)
void set_cuda_fault_hook(void (*hook)(const std::string& what));

/// A fault seen: the clock locks released, then the hook; once per process (true for the call that reported it)
bool report_cuda_fault(const std::string& what);
/// Whether a fault was reported in this process, and what it was
bool cuda_faulted();
std::string cuda_fault_what();

/// The code a run ends with after a failed CUDA call: kCudaFault where a GPU's context holds a fault (reported), else 1
int failed_call_code();

}   // namespace lamina::core
