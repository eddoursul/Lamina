// include/lamina/kernels/cpu/miss_jobs.hpp - the CPU pool's jobs for the routed entries no GPU holds (the qwen35 path's
// hybrid mode, Hy4-flash's RAM experts): in entry order, one job per expert and a new one past MAXT tokens (the rows a
// job holds pick the kernel: the outputs depend on that order), each token's activation quantized once, at its first
// such entry.  Its arrays are kept between calls (no map a layer).
#pragma once

#include "lamina/kernels/cpu/native_expert.hpp"
#include "lamina/kernels/cpu/pool.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace lamina::kernels::cpu {

class MissJobs {
public:
    std::vector<ExpertJobMulti> jobs;   ///< the call's jobs, for the pool

    /// A call's start: `T` tokens whose rows are at `h` (`ld` floats apart), their activations quantized into `acts`
    /// (kNativeActBytes each) in `fmt`; experts 0 .. n_expert - 1
    void begin(int T, int n_expert, const NativeFmt& fmt, const float* h, size_t ld, uint8_t* acts) {
        jobs.clear();
        open_.assign((size_t) n_expert, -1);
        quantized_.assign((size_t) T, 0);
        fmt_ = &fmt;
        h_ = h;
        ld_ = ld;
        acts_ = acts;
    }
    /// An entry no GPU holds: token t's row through the expert `ex` (its blob), its output row at `out`
    void add(int t, int ex, const uint8_t* blob, float* out) {
        uint8_t* const act = acts_ + (size_t) t * kNativeActBytes;
        if (!quantized_[(size_t) t]) {
            native_quant_act(*fmt_, h_ + (size_t) t * ld_, act);
            quantized_[(size_t) t] = 1;
        }
        int& o = open_[(size_t) ex];
        if (o < 0 || jobs[(size_t) o].nt == MAXT) {
            ExpertJobMulti j;
            j.blob = blob;
            o = (int) jobs.size();
            jobs.push_back(j);
        }
        ExpertJobMulti& j = jobs[(size_t) o];
        j.nact[j.nt] = act;
        j.out[j.nt] = out;
        ++j.nt;
    }

private:
    std::vector<int> open_;   // each expert's job with room, -1
    std::vector<char> quantized_;
    const NativeFmt* fmt_ = nullptr;
    const float* h_ = nullptr;
    size_t ld_ = 0;
    uint8_t* acts_ = nullptr;
};

}   // namespace lamina::kernels::cpu
