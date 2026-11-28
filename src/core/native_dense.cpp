#include "lamina/core/native_dense.hpp"

#include "lamina/artifact/gguf_reader.hpp"
#include "lamina/core/gguf_loader.hpp"
#include "lamina/core/weights.hpp"
#include "lamina/kernels/bf16_bits.hpp"
#include "lamina/kernels/cpu/q8_0_rows.hpp"
#include "lamina/kernels/f16_bits.hpp"
#include "lamina/kernels/native_mmvq.hpp"
#include "lamina/platform/cpu_arch.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <thread>

namespace lamina::core {
bool native_dense_name(const std::string& name, bool include_ple_key) {
    if (name.rfind("blk.", 0) != 0) return false;
    if (include_ple_key && name == "blk.1.ple_key.weight") return true;
    static const char* suffixes[] = {".attn_qkv.weight",      ".attn_gate.weight",      ".ssm_out.weight",
                                     ".attn_q.weight",        ".attn_k.weight",         ".attn_v.weight",
                                     ".attn_output.weight",   ".ffn_gate_shexp.weight", ".ffn_up_shexp.weight",
                                     ".ffn_down_shexp.weight"};
    for (const char* suffix : suffixes)
        if (name.ends_with(suffix)) return true;
    return false;
}

bool hc_q8_name(const std::string& name) {
    if (name.rfind("blk.", 0) != 0) return name == "output_hc_down.weight" || name == "output_hc_up.weight";
    return name.ends_with(".hc_attn_down.weight") || name.ends_with(".hc_attn_up.weight") ||
           name.ends_with(".hc_ffn_down.weight") || name.ends_with(".hc_ffn_up.weight");
}

std::string ShardDirectory::check(const lamina::GgufFile& gguf) {
    const auto* count = gguf.get("split.count");
    const auto* number = gguf.get("split.no");
    const auto* tensors = gguf.get("split.tensors.count");
    const auto* architecture = gguf.get("general.architecture");
    if (number && !numbers_.insert(number->u).second) return "native dense: duplicate split shard number";
    // A re-split GGUF (Huihui's Q8_0) repeats general.architecture on every shard but keeps the architecture's numeric
    // keys (block_count ...) on the first one only: the full guard is the first shard's (or a single file's); a later
    // shard that names an architecture must name the same one
    if (architecture && (!number || number->u == 0)) {
        have_architecture_ = true;
        first_arch_ = architecture->s;
        if (count && number && tensors && number->u == 0 && count->u > 1) {
            split_count_ = count->u;
            split_tensors_ = tensors->u;
        }
    } else if (!have_architecture_ || !split_count_ || !count || !number || !tensors || count->u != split_count_ ||
               number->u == 0 || number->u >= split_count_ || tensors->u != split_tensors_ ||
               (architecture && architecture->s != first_arch_)) {
        return "native dense: additional shard must match the architecture-validated first shard's split metadata";
    }
    std::vector<uint64_t> offsets;
    for (const auto& tensor : gguf.tensors()) offsets.push_back(tensor.offset);
    std::sort(offsets.begin(), offsets.end());
    if (std::adjacent_find(offsets.begin(), offsets.end()) != offsets.end())
        return "native dense: tensor payload offsets overlap";
    // every directory span, the tensors not uploaded too: an ignored tensor must not overlap the matrix after it
    const uint64_t payload = gguf.file_size() - gguf.data_start();
    for (const auto& tensor : gguf.tensors()) {
        int block_elements = 0, block_bytes = 0;
        uint64_t elements = 1;
        if (tensor.shape.empty() || !lamina::block_geometry(tensor.type, block_elements, block_bytes) ||
            tensor.shape[0] % (uint64_t) block_elements != 0)
            return "native dense: invalid block geometry " + tensor.name;
        for (uint64_t dimension : tensor.shape) {
            if (!dimension || elements > (std::numeric_limits<uint64_t>::max)() / dimension)
                return "native dense: invalid tensor extent " + tensor.name;
            elements *= dimension;
        }
        const uint64_t blocks = elements / (uint64_t) block_elements;
        if (blocks > (std::numeric_limits<uint64_t>::max)() / (uint64_t) block_bytes)
            return "native dense: tensor byte count overflow " + tensor.name;
        const uint64_t bytes = blocks * (uint64_t) block_bytes;
        if (tensor.offset > payload || bytes > payload - tensor.offset)
            return "native dense: truncated payload " + tensor.name;
        const auto next = std::upper_bound(offsets.begin(), offsets.end(), tensor.offset);
        if (next != offsets.end() && bytes > *next - tensor.offset)
            return "native dense: overlapping payload " + tensor.name;
    }
    return {};
}

namespace {
struct DeviceFree {
    void operator()(void* p) const {
        if (p) cudaFree(p);
    }
};
using DevicePtr = std::unique_ptr<void, DeviceFree>;
struct Pending {
    WeightRef* ref;
    int type;
    uint64_t bytes;
    DevicePtr data;
};
}   // namespace

bool NativeDense::served_names(const std::vector<std::string>& shards, bool include_ple_key, std::set<std::string>& out,
                               std::string& err) {
    try {
        for (const auto& path : shards) {
            lamina::GgufFile gguf(path);
            for (const auto& tensor : gguf.tensors())
                if (native_dense_name(tensor.name, include_ple_key) &&
                    lamina::kernels::native_mmvq_supported(tensor.type) && tensor.shape.size() == 2)
                    out.insert(tensor.name);
        }
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}

NativeDense::~NativeDense() {
    if (scratch_) cudaFree(scratch_);
    for (void* p : weights_) cudaFree(p);
}

bool NativeDense::load(const std::vector<std::string>& shards, WeightTable& table, std::string& err,
                       bool include_ple_key) {
    if (scratch_ || !weights_.empty()) {
        err = "native dense: already loaded";
        return false;
    }
    if (shards.empty()) {
        err = "native dense: at least one GGUF shard is required";
        return false;
    }
    try {
        std::vector<Pending> pending;
        std::set<std::string> seen;
        int max_in = 0;
        uint64_t total = 0;
        ShardDirectory dir;
        for (const auto& path : shards) {
            lamina::GgufFile gguf(path);
            const auto* number = gguf.get("split.no");
            if (gguf.get("general.architecture") && (!number || number->u == 0)) {   // (the first shard's guard)
                err = lamina::check_architecture(gguf);
                if (!err.empty()) return false;
            }
            err = dir.check(gguf);
            if (!err.empty()) return false;
            std::vector<Upload> jobs;
            for (const auto& tensor : gguf.tensors()) {
                if (!native_dense_name(tensor.name, include_ple_key)) continue;
                if (!seen.insert(tensor.name).second) {
                    err = "native dense: duplicate tensor " + tensor.name;
                    return false;
                }
                auto found = table.table_.find(tensor.name);
                if (found == table.table_.end()) {
                    err = "native dense: tensor absent from canonical table: " + tensor.name;
                    return false;
                }
                auto& ref = found->second;
                if (ref.native_data) {
                    err = "native dense: override already attached";
                    return false;
                }
                if (!lamina::kernels::native_mmvq_supported(tensor.type)) continue;
                if (!ref.quantized() || tensor.shape.size() != 2 || ref.ne0 <= 0 || ref.ne0 > INT_MAX || ref.ne1 <= 0 ||
                    ref.ne1 > INT_MAX || tensor.shape[0] != (uint64_t) ref.ne0 ||
                    tensor.shape[1] != (uint64_t) ref.ne1) {
                    err = "native dense: incompatible matrix " + tensor.name;
                    return false;
                }
                const auto bytes = lamina::kernels::native_mmvq_weight_bytes(tensor.type, (int) ref.ne0, (int) ref.ne1);
                // a zeroed tail: the prompt path's MMQ reads past the last row of a matrix whose rows are not whole
                // 256-value steps (the shared expert's down projection, 640), and NaN times a zero input is NaN
                constexpr uint64_t kTail = 4096;
                void* allocation = nullptr;
                auto status = cudaMalloc(&allocation, bytes + kTail);
                DevicePtr data(allocation);
                if (status == cudaSuccess) status = cudaMemset((uint8_t*) data.get() + bytes, 0, kTail);
                if (status != cudaSuccess) {
                    err = "native dense upload " + tensor.name + ": " + cudaGetErrorString(status);
                    return false;
                }
                jobs.push_back(Upload{data.get(), (const uint8_t*) gguf.tensor_data(tensor), bytes});
                max_in = std::max(max_in, (int) ref.ne0);
                total += bytes;
                pending.push_back(Pending{&ref, (int) tensor.type, bytes, std::move(data)});
            }
            if (!upload_parallel(jobs, err)) return false;
        }
        if (pending.empty()) {
            err = "native dense: no supported GDN/QSA matrices in supplied shards";
            return false;
        }
        void* allocation = nullptr;
        const auto status = cudaMalloc(&allocation, lamina::kernels::native_q8_1_bytes(max_in));
        DevicePtr scratch(allocation);
        if (status != cudaSuccess) {
            err = std::string("native dense scratch: ") + cudaGetErrorString(status);
            return false;
        }
        // All checks and allocations finish before publishing any reference.
        weights_.reserve(pending.size());
        for (auto& item : pending) {
            item.ref->native_data = item.data.get();
            item.ref->native_type = item.type;
            item.ref->native_q8_1 = scratch.get();
            weights_.push_back(item.data.release());
        }
        scratch_ = scratch.release();
        bytes_ = total;
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}

bool HcQ8::names(const std::vector<std::string>& shards, std::set<std::string>& out, std::string& err) {
    try {
        for (const auto& path : shards) {
            lamina::GgufFile gguf(path);
            for (const auto& t : gguf.tensors())
                if (hc_q8_name(t.name)) out.insert(t.name);
        }
        return true;
    } catch (const std::exception& error) {
        err = std::string("hc q8: ") + error.what();
        return false;
    }
}

HcQ8::~HcQ8() {
    for (void* p : allocs_) cudaFree(p);
}

bool HcQ8::load(const std::vector<std::string>& shards, WeightTable& table, std::string& err) {
    try {
        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess) {
            err = "hc q8: no device";
            return false;
        }
        struct Job {
            const lamina::TensorInfo* t;
            WeightRef* ref;
            bool q8_0;   // the GGUF's own Q8_0 blocks, split into codes and scales
            int kind;    // else the rows' type: 0 F32, 1 F16, 2 BF16
            void* dq = nullptr;
            void* ds = nullptr;
        };
        for (const auto& path : shards) {
            lamina::GgufFile gguf(path);
            std::vector<Job> jobs;
            for (const auto& t : gguf.tensors()) {
                if (!hc_q8_name(t.name)) continue;
                auto found = table.table_.find(t.name);
                if (found == table.table_.end()) {
                    err = "hc q8: tensor absent from the canonical table: " + t.name;
                    return false;
                }
                WeightRef& ref = found->second;
                const std::string tn = t.type_name();
                if (t.shape.size() != 2 || t.shape[0] != (uint64_t) ref.ne0 || t.shape[1] != (uint64_t) ref.ne1 ||
                    ref.ne0 % 32 != 0 || (tn != "Q8_0" && tn != "BF16" && tn != "F16" && tn != "F32")) {
                    err = "hc q8: " + t.name + " has another shape or type (" + tn + ")";
                    return false;
                }
                Job j{&t, &ref, tn == "Q8_0", tn == "F32" ? 0 : tn == "F16" ? 1 : 2};
                const size_t n = (size_t) (ref.ne0 * ref.ne1);
                if (cudaMalloc(&j.dq, n) != cudaSuccess || cudaMalloc(&j.ds, n / 32 * 2) != cudaSuccess) {
                    if (j.dq) cudaFree(j.dq);
                    err = "hc q8: no VRAM for " + t.name;
                    return false;
                }
                allocs_.push_back(j.dq);
                allocs_.push_back(j.ds);
                jobs.push_back(j);
            }
            // a matrix at a time to each of up to 8 threads (one thread rounded IQ3_S's 194 BF16 matrices in 2 s)
            std::atomic<size_t> next{0};
            std::atomic<bool> bad{false};
            std::mutex err_mu;
            auto worker = [&] {
                std::vector<int8_t> q;
                std::vector<uint16_t> s;
                std::vector<float> row;
                if (cudaSetDevice(device) != cudaSuccess) bad = true;
                for (size_t k; !bad && (k = next++) < jobs.size();) {
                    const Job& j = jobs[k];
                    const int64_t cols = j.ref->ne0, rows = j.ref->ne1;
                    q.resize((size_t) (rows * cols));
                    s.resize((size_t) (rows * cols / 32));
                    row.resize((size_t) cols);
                    const uint8_t* src = gguf.tensor_data(*j.t);
                    for (int64_t r = 0; r < rows; ++r) {
                        int8_t* qr = q.data() + r * cols;
                        uint16_t* sr = s.data() + r * (cols / 32);
                        if (j.q8_0) {   // as stored: {f16 d, 32 codes} a block
                            const uint8_t* bl = src + (size_t) (r * (cols / 32)) * 34;
                            for (int64_t b = 0; b < cols / 32; ++b) {
                                std::memcpy(sr + b, bl + b * 34, 2);
                                std::memcpy(qr + b * 32, bl + b * 34 + 2, 32);
                            }
                            continue;
                        }
                        for (int64_t c = 0; c < cols; ++c) {
                            const size_t i = (size_t) (r * cols + c);
                            if (j.kind == 0) {
                                std::memcpy(&row[(size_t) c], src + i * 4, 4);
                            } else {
                                uint16_t h;
                                std::memcpy(&h, src + i * 2, 2);
                                if (j.kind == 1) {
                                    row[(size_t) c] = lamina::kernels::f32_from_f16(h);
                                } else {
                                    row[(size_t) c] = lamina::kernels::f32_from_bf16(h);
                                }
                            }
                        }
                        lamina::kernels::cpu::quantize_q8_0_row_split(row.data(), cols, qr, sr);
                    }
                    if (cudaMemcpy(j.dq, q.data(), q.size(), cudaMemcpyHostToDevice) != cudaSuccess ||
                        cudaMemcpy(j.ds, s.data(), s.size() * 2, cudaMemcpyHostToDevice) != cudaSuccess) {
                        std::lock_guard<std::mutex> lk(err_mu);
                        if (!bad) err = "hc q8: upload of " + j.t->name + ": " + cudaGetErrorString(cudaGetLastError());
                        bad = true;
                    }
                }
            };
            const size_t threads = std::min<size_t>(8, jobs.size());
            std::vector<std::thread> pool;
            for (size_t k = 1; k < threads; ++k) pool.emplace_back(worker);
            if (threads > 0) worker();
            for (auto& th : pool) th.join();
            if (bad) {
                if (err.empty()) err = "hc q8: no device on a loader thread";
                return false;
            }
            for (const Job& j : jobs) {
                bytes_ += (uint64_t) (j.ref->ne0 * j.ref->ne1) / 32 * 34;   // the codes and an f16 scale a 32
                if (j.q8_0) ++q8_0_;
                j.ref->hc_codes = (const int8_t*) j.dq;
                j.ref->hc_scales = (const uint16_t*) j.ds;
            }
        }
        return true;
    } catch (const std::exception& error) {
        err = std::string("hc q8: ") + error.what();
        return false;
    }
}
}   // namespace lamina::core
