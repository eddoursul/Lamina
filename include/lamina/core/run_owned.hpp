// include/lamina/core/run_owned.hpp - a model path's run (the qwen35, Hy4-flash and Inkling 2 paths): its end on an
// error (Fatal, LAMINA_CK), and what it allocates freed when it ends, on an error too (Owned), with the helpers that
// record each allocation.  The engine may run in the server's process (lamina), which an exit would end, and the
// server starts it again there after an unload: a run gives back all it took.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <vector>

/// A CUDA call that must succeed: a failure prints the call, its error and its place, and ends the run (Fatal{1})
#define LAMINA_CK(x)                                                                                                   \
    do {                                                                                                               \
        cudaError_t e_ = (x);                                                                                          \
        if (e_ != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "lamina: %s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__);   \
            throw ::lamina::core::Fatal{1};                                                                            \
        }                                                                                                              \
    } while (0)

namespace lamina::core {

/// The run's end on an error (LAMINA_CK's, a missing tensor's ...): its message is printed, run_owned returns `code`
struct Fatal {
    int code;
};

/// What a run allocates, freed when it ends: device, pinned and heap memory, streams, events and graphs, each recorded
/// by the helpers below.  One run at a time per process.
struct Owned {
    std::mutex mu;
    std::vector<void*> dev, pinned, heap;
    std::vector<cudaStream_t> streams;
    std::vector<cudaEvent_t> events;
    std::vector<cudaGraphExec_t> graphs;
    Owned() = default;
    Owned(const Owned&) = delete;
    Owned& operator=(const Owned&) = delete;
    ~Owned() {
        cudaDeviceSynchronize();
        for (cudaGraphExec_t g : graphs) cudaGraphExecDestroy(g);
        for (cudaEvent_t e : events) cudaEventDestroy(e);
        for (cudaStream_t st : streams) cudaStreamDestroy(st);
        for (void* d : dev) cudaFree(d);
        for (void* h : pinned) cudaFreeHost(h);
        for (void* h : heap) std::free(h);
        (void) cudaGetLastError();
    }
};
/// The running run's lists (null outside a run: the helpers record nothing, and what they made is the caller's to free)
inline Owned* g_owned = nullptr;

/// `v` into the running run's `list`
template <class V, class T> void own(std::vector<V> Owned::* list, T v) {
    if (g_owned == nullptr) return;
    std::lock_guard<std::mutex> lk(g_owned->mu);
    (g_owned->*list).push_back((V) v);
}
/// `v` out of it again (the caller frees it now)
template <class V, class T> void disown(std::vector<V> Owned::* list, T v) {
    if (g_owned == nullptr) return;
    std::lock_guard<std::mutex> lk(g_owned->mu);
    std::vector<V>& l = g_owned->*list;
    l.erase(std::remove(l.begin(), l.end(), (V) v), l.end());
}

// What may be refused, the refusal the caller's to handle (its error then still set): `bytes` of device memory, and of
// pinned host memory with cudaHostAlloc's `flags`, each in *p.
template <class T> cudaError_t dmalloc(T** p, size_t bytes) {
    const cudaError_t e = cudaMalloc((void**) p, bytes);
    if (e == cudaSuccess) own(&Owned::dev, (void*) *p);
    return e;
}
template <class T> cudaError_t hmalloc(T** p, size_t bytes, unsigned flags) {
    const cudaError_t e = cudaHostAlloc((void**) p, bytes, flags);
    if (e == cudaSuccess) own(&Owned::pinned, (void*) *p);
    return e;
}
/// Device memory a helper allocated, freed now
inline cudaError_t dfree(void* p) {
    disown(&Owned::dev, p);
    return cudaFree(p);
}
/// Pinned host memory a helper allocated, freed now
inline cudaError_t hfree(void* p) {
    disown(&Owned::pinned, p);
    return cudaFreeHost(p);
}

// What the run cannot go on without (LAMINA_CK ends it where it is refused): `n` T's, 256 bytes at least, of device
// memory, and of pinned host memory every GPU may copy from (portable); zeroed where the name says so (pages given back
// and taken again are not).
template <class T> T* dalloc(size_t n) {
    void* p = nullptr;
    LAMINA_CK(cudaMalloc(&p, std::max<size_t>(n * sizeof(T), 256)));
    own(&Owned::dev, p);
    return (T*) p;
}
template <class T> T* dzero(size_t n) {
    T* p = dalloc<T>(n);
    LAMINA_CK(cudaMemset(p, 0, std::max<size_t>(n * sizeof(T), 256)));
    return p;
}
template <class T> T* halloc(size_t n) {
    void* p = nullptr;
    LAMINA_CK(cudaHostAlloc(&p, std::max<size_t>(n * sizeof(T), 256), cudaHostAllocPortable));
    own(&Owned::pinned, p);
    return (T*) p;
}
template <class T> T* hzero(size_t n) {
    T* p = halloc<T>(n);
    std::memset((void*) p, 0, std::max<size_t>(n * sizeof(T), 256));
    return p;
}
/// Zeroed pinned host memory the GPUs read through the mapping: the host's pointer, the current GPU's in *dev
template <class T> T* hmapped(size_t n, T** dev) {
    void* p = nullptr;
    LAMINA_CK(cudaHostAlloc(&p, std::max<size_t>(n * sizeof(T), 256), cudaHostAllocMapped | cudaHostAllocPortable));
    own(&Owned::pinned, p);
    std::memset(p, 0, std::max<size_t>(n * sizeof(T), 256));
    LAMINA_CK(cudaHostGetDevicePointer((void**) dev, p, 0));
    return (T*) p;
}
/// A stream of the current GPU's (at `priority`, cudaStreamCreateWithPriority's)
inline cudaStream_t stream_new(unsigned flags = cudaStreamNonBlocking) {
    cudaStream_t s = nullptr;
    LAMINA_CK(cudaStreamCreateWithFlags(&s, flags));
    own(&Owned::streams, s);
    return s;
}
inline cudaStream_t stream_new(unsigned flags, int priority) {
    cudaStream_t s = nullptr;
    LAMINA_CK(cudaStreamCreateWithPriority(&s, flags, priority));
    own(&Owned::streams, s);
    return s;
}
/// An event of the current GPU's
inline cudaEvent_t event_new(unsigned flags = cudaEventDisableTiming) {
    cudaEvent_t e = nullptr;
    LAMINA_CK(cudaEventCreateWithFlags(&e, flags));
    own(&Owned::events, e);
    return e;
}
/// `graph` instantiated (the graph itself stays the caller's)
inline cudaGraphExec_t graph_new(cudaGraph_t graph) {
    cudaGraphExec_t x = nullptr;
    LAMINA_CK(cudaGraphInstantiate(&x, graph, 0));
    own(&Owned::graphs, x);
    return x;
}
/// The work captured on stream `s` since its capture began, as a graph
inline cudaGraphExec_t graph_end(cudaStream_t s) {
    cudaGraph_t graph = nullptr;
    LAMINA_CK(cudaStreamEndCapture(s, &graph));
    const cudaGraphExec_t x = graph_new(graph);
    LAMINA_CK(cudaGraphDestroy(graph));
    return x;
}
/// A graph the pointers it was captured on no longer hold for, freed now (or once a launch of it is through)
inline void graph_drop(cudaGraphExec_t g) {
    if (g == nullptr) return;
    disown(&Owned::graphs, g);
    (void) cudaGraphExecDestroy(g);
}
/// `n` bytes of the heap (null where refused)
inline void* heap_new(size_t n) {
    void* h = std::malloc(n);
    if (h != nullptr) own(&Owned::heap, h);
    return h;
}

/// A path's run (`f`: the engine's run, an int its exit code) under an Owned: what it allocates goes when it ends,
/// on an error too (Fatal's code; another exception's message printed, 1)
template <class F> int run_owned(F&& f) {
    Owned owned;
    g_owned = &owned;
    struct Unset {
        ~Unset() { g_owned = nullptr; }
    } unset;
    try {
        return f();
    } catch (const Fatal& end) {
        return end.code;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "lamina: %s\n", e.what());
        return 1;
    }
}
/// A path's command line checked (`f`: the engine's run in its check mode, which allocates nothing): Fatal's code
template <class F> int check_owned(F&& f) {
    try {
        return f();
    } catch (const Fatal& end) {
        return end.code;
    }
}

}   // namespace lamina::core
