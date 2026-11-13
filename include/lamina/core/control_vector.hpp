// include/lamina/core/control_vector.hpp - control vectors (--control-vector, llama.cpp's GGUF format) as a layer
// weight: `blk.<l>.cvec`, the vectors [a, u] (2 x n_embd f32) that the verify windows and the prompt path apply after
// layer l's FFN write (kernels/cvec.hpp).  A prompt split's second stage copies it with the layer's other weights.
//
// The files are read as llama.cpp's common_control_vector_load reads them: every tensor `direction.<l>` (l >= 1, F32,
// one dimension of n_embd values) times its file's scale, summed per layer over the files; layer 0 has none, and a
// layer past the model's is ignored.  The layers steered are those of --control-vector-layer-range (inclusive; by
// default 1 .. the last) whose vector d_l is not zero:
//
//   add (llama.cpp's):  h <- h + d_l                                     a = d_l, u = 0
//   project (--cvec-mode project, the cvec-hooks fork's):
//                       h <- h - |d_l| (h . v) v,  v = d_l / |d_l|        a = 0, u = d_l / sqrt(|d_l|)
//
// on each hyper-connection stream h.  --cvec-dir single:L (project): layer L's d for every layer of the range.  So a
// unit direction at scale 1 removes its component from every stream; at 1.78 it inverts 78% of it.
#pragma once

#include "lamina/core/geometry.hpp"
#include "lamina/core/weights.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace lamina::core {

struct ControlVectorSpec {
    struct File {
        std::string path;
        float scale = 1.0f;
    };
    std::vector<File> files;
    int first = -1, last = -1;   ///< --control-vector-layer-range; <= 0: the defaults, 1 and the last layer
    bool project = false;        ///< --cvec-mode project
    int single = -1;             ///< --cvec-dir single:L, -1: per-layer
};

/// --control-vector FILE[,FILE...] (scale 1) or, `scaled`, --control-vector-scaled FILE:SCALE[,...] (the scale after
/// the last colon, so a drive letter stays in the path), appended to `out`.  False with `err` on a malformed item.
bool parse_control_vectors(const std::string& list, bool scaled, std::vector<ControlVectorSpec::File>& out,
                           std::string& err);

/// The steered layers' vectors, built on the host.
struct ControlVectorTables {
    std::vector<int64_t> layers;         ///< ascending
    std::vector<float> rows;             ///< per steered layer [a, u], n_embd floats each
    int64_t first = 0, last = 0;         ///< the range as resolved
    std::vector<std::string> warnings;   ///< a file made for another model (controlvector.model_hint)
};
bool control_vector_tables(const ControlVectorSpec& spec, int64_t n_layers, int64_t n_embd, ControlVectorTables& out,
                           std::string& err);

/// The tables on the current device, put into `table` as `blk.<l>.cvec`.
class ControlVector {
public:
    ControlVector() = default;
    ~ControlVector();
    ControlVector(const ControlVector&) = delete;
    ControlVector& operator=(const ControlVector&) = delete;
    bool load(const ControlVectorSpec& spec, const ModelGeometry& g, WeightTable& table, std::string& err);
    const ControlVectorTables& tables() const { return t_; }

private:
    void* dev_ = nullptr;
    ControlVectorTables t_;
};

}  // namespace lamina::core
