// src/core/gguf_loader.cpp - see include/lamina/core/gguf_loader.hpp.
#include "lamina/core/gguf_loader.hpp"

#include "lamina/artifact/dequant.hpp"
#include "lamina/core/run_owned.hpp"
#include "lamina/kernels/bf16_bits.hpp"
#include "lamina/kernels/f16_bits.hpp"
#include "lamina/platform/slice_read.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <new>
#include <thread>
#include <utility>

namespace lamina::core {

bool tensor_floats(const lamina::TensorInfo& t, const uint8_t* src, std::vector<float>& h) {
    h.resize((size_t) t.elements());
    if (t.type == 0) std::memcpy(h.data(), src, h.size() * 4);
    else if (t.type == 30)
        for (size_t i = 0; i < h.size(); ++i)
            h[i] = lamina::kernels::f32_from_bf16((uint16_t) (src[2 * i] | (src[2 * i + 1] << 8)));
    else if (t.type == 1)   // F16
        for (size_t i = 0; i < h.size(); ++i)
            h[i] = lamina::kernels::f32_from_f16((uint16_t) (src[2 * i] | (src[2 * i + 1] << 8)));
    else if (t.type == 8 && h.size() % 32 == 0)   // Q8_0: blocks of an fp16 scale and 32 int8
        for (size_t b = 0; b < h.size() / 32; ++b) lamina::dequantize_q8_0(src + b * 34, h.data() + b * 32);
    else return false;
    return true;
}

const float* floats_on_device(const std::vector<float>& h) {
    float* d = dalloc<float>(h.size());
    LAMINA_CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    return d;
}

// (upstream PR #2673's form: one synchronous cudaMemcpy a matrix took the mapping's faults one at a time, UD-Q4_K_XL's
// 2,975 MiB of dense matrices in ~1.5-2 s with the file cached)
bool upload_parallel(const std::vector<Upload>& jobs, std::string& err) {
    constexpr uint64_t PIECE = 8ull << 20;
    std::vector<std::pair<size_t, uint64_t>> pieces;   // (job, offset)
    for (size_t j = 0; j < jobs.size(); ++j)
        for (uint64_t o = 0; o < jobs[j].bytes; o += PIECE) pieces.emplace_back(j, o);
    if (pieces.empty()) return true;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        err = "native dense upload: no device";
        return false;
    }
    std::atomic<size_t> next{0};
    std::atomic<bool> bad{false};
    std::atomic<int> first_err{(int) cudaSuccess};   // a worker's error (the runtime's last error is per thread)
    auto fail = [&](cudaError_t e) {
        int none = (int) cudaSuccess;
        first_err.compare_exchange_strong(none, (int) e);
        bad = true;
    };
    auto worker = [&] {
        cudaStream_t s = nullptr;
        uint8_t* stage = nullptr;
        cudaEvent_t done[2] = {nullptr, nullptr};
        bool live[2] = {false, false};
        cudaError_t e = cudaSetDevice(device);
        if (e == cudaSuccess) e = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        if (e == cudaSuccess) e = cudaHostAlloc((void**) &stage, 2 * PIECE, cudaHostAllocDefault);
        if (e == cudaSuccess) e = cudaEventCreateWithFlags(&done[0], cudaEventDisableTiming);
        if (e == cudaSuccess) e = cudaEventCreateWithFlags(&done[1], cudaEventDisableTiming);
        if (e != cudaSuccess) fail(e);
        for (int h = 0; !bad; h ^= 1) {
            const size_t i = next.fetch_add(1);
            if (i >= pieces.size()) break;
            const auto [j, o] = pieces[i];
            const uint64_t n = std::min(PIECE, jobs[j].bytes - o);
            if (live[h] && (e = cudaEventSynchronize(done[h])) != cudaSuccess) {
                fail(e);
                break;
            }
            std::memcpy(stage + h * PIECE, jobs[j].src + o, (size_t) n);
            e = cudaMemcpyAsync((uint8_t*) jobs[j].dev + o, stage + h * PIECE, (size_t) n, cudaMemcpyHostToDevice, s);
            if (e == cudaSuccess) e = cudaEventRecord(done[h], s);
            if (e != cudaSuccess) {
                fail(e);
                break;
            }
            live[h] = true;
        }
        if (s != nullptr && (e = cudaStreamSynchronize(s)) != cudaSuccess) fail(e);
        for (cudaEvent_t ev : done)
            if (ev) cudaEventDestroy(ev);
        if (stage) cudaFreeHost(stage);
        if (s) cudaStreamDestroy(s);
    };
    const size_t threads = std::min<size_t>(8, pieces.size());
    std::vector<std::thread> pool;
    for (size_t k = 1; k < threads; ++k) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
    // the tails' memsets, on the legacy stream beside the workers' non-blocking ones, landed too
    if (const cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess) fail(e);
    if (bad) {
        err = std::string("native dense upload: ") + cudaGetErrorString((cudaError_t) first_err.load());
        return false;
    }
    return true;
}

bool GgufLoader::read_ahead(const std::function<bool(const std::string&)>& want, int threads, bool uncached,
                            std::string& e) {
    std::vector<lamina::platform::SliceRun> runs;
    std::vector<std::pair<const lamina::TensorInfo*, uint64_t>> at;
    uint64_t total = 0;
    for (size_t s = 0; s < gm.size(); ++s)
        for (const lamina::TensorInfo& t : gm.shard(s).tensors()) {
            const uint64_t n = lamina::tensor_payload_bytes(t);
            if (n == 0 || !gm.in_bounds(t, s) || !want(t.name)) continue;
            runs.push_back({gm.shard(s).path(), gm.shard(s).data_start() + t.offset, n, total, n, 1, 0});
            at.emplace_back(&t, total);
            total += (n + 63) / 64 * 64;
        }
    pre_buf_.reset(new (std::nothrow) uint8_t[std::max<uint64_t>(total, 64)]);
    if (!pre_buf_) return true;   // the mapping's, then
    uint8_t* b = pre_buf_.get();
    if (!lamina::platform::read_slices(runs, threads, uncached, (uint64_t) 32 << 20, [b](int) { return b; }, e))
        return false;
    for (const auto& [t, o] : at) pre_[t] = b + o;
    return true;
}

void GgufLoader::drop_ahead() {
    pre_.clear();
    pre_buf_.reset();
}

const lamina::TensorInfo* GgufLoader::find(const std::string& name, size_t* s) const {
    const lamina::TensorInfo* t = gm.find(name, s);
    if (!t) {
        std::fprintf(stderr, "lamina: missing %s\n", name.c_str());
        throw Fatal{1};
    }
    return t;
}

const uint8_t* GgufLoader::host(const std::string& name, const lamina::TensorInfo** ti) const {
    size_t s = 0;
    const lamina::TensorInfo* t = find(name, &s);
    if (ti) *ti = t;
    if (const auto it = pre_.find(t); it != pre_.end()) return it->second;
    return gm.shard(s).tensor_data(*t);
}

uint8_t* GgufLoader::device_bytes(const std::string& name, const lamina::TensorInfo** ti) const {
    const lamina::TensorInfo* t = nullptr;
    const uint8_t* src = host(name, &t);
    if (ti) *ti = t;
    const uint64_t n = lamina::tensor_payload_bytes(*t);
    uint8_t* d = dalloc<uint8_t>(n + kWeightPad);
    LAMINA_CK(cudaMemcpy(d, src, n, cudaMemcpyHostToDevice));
    LAMINA_CK(cudaMemset(d + n, 0, kWeightPad));
    return d;
}

uint8_t* GgufLoader::device_room(const std::string& name, std::vector<Upload>& jobs,
                                 const lamina::TensorInfo** ti) const {
    const lamina::TensorInfo* t = nullptr;
    const uint8_t* src = host(name, &t);
    if (ti) *ti = t;
    const uint64_t n = lamina::tensor_payload_bytes(*t);
    uint8_t* d = dalloc<uint8_t>(n + kWeightPad);
    LAMINA_CK(cudaMemset(d + n, 0, kWeightPad));   // (the tail apart from the copy: upload_parallel waits for it)
    jobs.push_back({d, src, n});
    return d;
}

std::vector<float> GgufLoader::floats(const std::string& name) const {
    const lamina::TensorInfo* t = nullptr;
    const uint8_t* src = host(name, &t);
    std::vector<float> h;
    if (!tensor_floats(*t, src, h)) {
        std::fprintf(stderr, "lamina: %s is %s, expected F32, BF16, F16 or Q8_0\n", name.c_str(), t->type_name());
        throw Fatal{1};
    }
    return h;
}

}   // namespace lamina::core
