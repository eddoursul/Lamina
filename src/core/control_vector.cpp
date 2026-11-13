// src/core/control_vector.cpp - see include/lamina/core/control_vector.hpp.
#include "lamina/core/control_vector.hpp"

#include "lamina/artifact/gguf_reader.hpp"
#include "lamina/kernels/cvec.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>

namespace lamina::core {

bool parse_control_vectors(const std::string& list, bool scaled, std::vector<ControlVectorSpec::File>& out,
                           std::string& err) {
    for (size_t at = 0;;) {
        const size_t comma = std::min(list.find(',', at), list.size());
        const std::string item = list.substr(at, comma - at);
        ControlVectorSpec::File f;
        f.path = item;
        if (scaled) {
            const size_t colon = item.rfind(':');
            const std::string num = colon == std::string::npos ? std::string() : item.substr(colon + 1);
            char* end = nullptr;
            errno = 0;
            const double s = num.empty() ? 0.0 : std::strtod(num.c_str(), &end);
            if (colon == std::string::npos || colon == 0 || num.empty() || *end != '\0' || errno != 0 ||
                !std::isfinite(s)) {
                err = "--control-vector-scaled: expected FILE:SCALE, got '" + item + "'";
                return false;
            }
            f.path = item.substr(0, colon);
            f.scale = (float) s;
        }
        if (f.path.empty()) {
            err = scaled ? "--control-vector-scaled: an empty file name" : "--control-vector: an empty file name";
            return false;
        }
        out.push_back(f);
        if (comma == list.size()) return true;
        at = comma + 1;
    }
}

bool control_vector_tables(const ControlVectorSpec& spec, int64_t n_layers, int64_t n_embd, ControlVectorTables& out,
                           std::string& err) {
    out = ControlVectorTables{};
    if (spec.files.empty()) {
        err = "control vector: no file";
        return false;
    }
    // every file's directions times its scale, summed per layer (llama.cpp's common_control_vector_load)
    std::vector<double> sum((size_t) (n_layers * n_embd), 0.0);
    std::vector<bool> have((size_t) n_layers, false);
    for (const ControlVectorSpec::File& f : spec.files) {
        try {
            const lamina::GgufFile gguf(f.path);
            if (const lamina::MetaValue* hint = gguf.get("controlvector.model_hint"))
                if (hint->type == lamina::MetaType::STRING && hint->s != "qwen4exp")
                    out.warnings.push_back(f.path + " was made for '" + hint->s + "', not qwen4exp");
            int found = 0;
            for (const lamina::TensorInfo& t : gguf.tensors()) {
                long l = -1;
                char* end = nullptr;
                if (t.name.rfind("direction.", 0) == 0 && t.name[10] >= '0' && t.name[10] <= '9')
                    l = std::strtol(t.name.c_str() + 10, &end, 10);
                if (l < 1 || end == nullptr || *end != '\0') {
                    err = f.path + ": " + t.name + " is not a direction.<layer> tensor of a layer from 1 on";
                    return false;
                }
                if (t.type != 0 || t.shape.size() != 1 || t.shape[0] != (uint64_t) n_embd) {
                    err = f.path + ": " + t.name + " must be " + std::to_string((long long) n_embd) +
                          " F32 values (the model's n_embd)";
                    return false;
                }
                const uint64_t bytes = (uint64_t) n_embd * 4;
                if (t.offset > gguf.file_size() || gguf.data_start() + t.offset + bytes > gguf.file_size()) {
                    err = f.path + ": " + t.name + " lies past the end of the file";
                    return false;
                }
                ++found;
                if (l >= n_layers) continue;   // past the model's layers: llama.cpp ignores it too
                std::vector<float> v((size_t) n_embd);
                std::memcpy(v.data(), gguf.tensor_data(t), bytes);
                for (int64_t j = 0; j < n_embd; ++j) {
                    if (!std::isfinite(v[(size_t) j])) {
                        err = f.path + ": " + t.name + " holds a value that is not finite";
                        return false;
                    }
                    sum[(size_t) (l * n_embd + j)] += (double) f.scale * v[(size_t) j];
                }
                have[(size_t) l] = true;
            }
            if (found == 0) {
                err = f.path + ": no direction.<layer> tensors (not a control vector)";
                return false;
            }
        } catch (const std::exception& e) {
            err = f.path + ": " + e.what();
            return false;
        }
    }
    out.first = spec.first <= 0 ? 1 : spec.first;
    out.last = spec.last <= 0 || spec.last >= n_layers ? n_layers - 1 : spec.last;
    if (spec.single >= 0 && !spec.project) {
        err = "--cvec-dir single:L applies to --cvec-mode project";
        return false;
    }
    if (spec.single >= 0 && (spec.single >= n_layers || !have[(size_t) spec.single])) {
        err = "--cvec-dir single:" + std::to_string(spec.single) + ": the files have no direction for that layer";
        return false;
    }
    for (int64_t l = out.first; l <= out.last; ++l) {
        const int64_t src = spec.single >= 0 ? spec.single : l;
        if (!have[(size_t) src]) continue;
        const double* d = sum.data() + (size_t) (src * n_embd);
        double nrm = 0.0;
        for (int64_t j = 0; j < n_embd; ++j) nrm += d[j] * d[j];
        nrm = std::sqrt(nrm);
        if (nrm == 0.0) continue;
        const size_t at = out.rows.size();
        out.rows.resize(at + (size_t) (2 * n_embd), 0.0f);
        float* a = out.rows.data() + at;
        float* u = a + n_embd;
        const double k = 1.0 / std::sqrt(nrm);   // u = sqrt(s) v, s = |d|, v = d / |d|
        for (int64_t j = 0; j < n_embd; ++j) {
            if (spec.project) u[j] = (float) (d[j] * k);
            else a[j] = (float) d[j];
        }
        out.layers.push_back(l);
    }
    if (out.layers.empty()) {
        err = "control vector: no direction in layers " + std::to_string((long long) out.first) + ".." +
              std::to_string((long long) out.last);
        return false;
    }
    return true;
}

ControlVector::~ControlVector() {
    if (dev_ != nullptr) cudaFree(dev_);
}

bool ControlVector::load(const ControlVectorSpec& spec, const ModelGeometry& g, WeightTable& table, std::string& err) {
    if (!lamina::kernels::cvec_supported(g.n_embd)) {
        err = "control vector: no kernel for n_embd " + std::to_string((long long) g.n_embd);
        return false;
    }
    if (!control_vector_tables(spec, g.n_layers, g.n_embd, t_, err)) return false;
    const size_t bytes = t_.rows.size() * sizeof(float);
    if (cudaMalloc(&dev_, bytes) != cudaSuccess ||
        cudaMemcpy(dev_, t_.rows.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        err = "control vector: cannot place its " + std::to_string(bytes) + " bytes on the GPU";
        return false;
    }
    for (size_t i = 0; i < t_.layers.size(); ++i) {
        WeightRef r;
        r.data = (const float*) dev_ + i * (size_t) (2 * g.n_embd);
        r.bytes = (uint64_t) (2 * g.n_embd) * sizeof(float);
        r.ne0 = g.n_embd;
        r.ne1 = 2;
        r.elements = 2 * g.n_embd;
        r.kind = WeightKind::F32;
        table.put("blk." + std::to_string((long long) t_.layers[i]) + ".cvec", r);
    }
    return true;
}

}  // namespace lamina::core
