// include/lamina/kernels/k_scales.hpp - the K-quants' (Q4_K, Q5_K) 6-bit scale and min of sub-block j from a
// super-block's 12 scale bytes, as ggml's get_scale_min_k4: host and device, any integer type for the two.
#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define LAMINA_KS_HD __host__ __device__ __forceinline__
#else
#define LAMINA_KS_HD inline
#endif

namespace lamina::kernels {

template <class T> LAMINA_KS_HD void scale_min_k4(int j, const uint8_t* q, T& d, T& m) {
    if (j < 4) {
        d = (T) (q[j] & 63);
        m = (T) (q[j + 4] & 63);
    } else {
        d = (T) ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        m = (T) ((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

}   // namespace lamina::kernels
