// include/lamina/kernels/warp.cuh - the kernels' warp-level helpers: sums and maxima over a warp's 32 lanes (the XOR
// butterfly: every lane gets the result, the additions always in one order), a block of 128 threads' sum, and the
// ldmatrix, cp.async and mma instructions of the tensor-core kernels.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace lamina::kernels {

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ int warp_sum(int v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

/// The largest of a warp's (value, index) pairs, the lower index of equals: every lane's (the XOR butterfly)
__device__ __forceinline__ void warp_argmax(float& v, int& i) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, o);
        const int oi = __shfl_xor_sync(0xffffffffu, i, o);
        if (ov > v || (ov == v && oi < i)) {
            v = ov;
            i = oi;
        }
    }
}

/// The sum over a block of 128 threads: `part` 4 floats of shared memory, read after one barrier (a caller that writes
/// them again first waits for every thread's read).
__device__ __forceinline__ float block128_sum(float v, float* part) {
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = v;
    __syncthreads();
    return part[0] + part[1] + part[2] + part[3];
}

__device__ __forceinline__ void ldsm4(unsigned (&r)[4], const void* s) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((unsigned) __cvta_generic_to_shared(s)));
}
__device__ __forceinline__ void ldsm4t(unsigned (&r)[4], const void* s) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((unsigned) __cvta_generic_to_shared(s)));
}
/// 16 bytes from global memory to shared memory, in the background (cp.async.cg)
__device__ __forceinline__ void cp_async16(void* s, const void* g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned) __cvta_generic_to_shared(s)), "l"(g));
}
/// 16 bytes, `bytes` of them from g and zeros after (cp.async.cg with a source size: a tile's missing rows as zeros)
__device__ __forceinline__ void cp_async16_fill(void* s, const void* g, unsigned bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"((unsigned) __cvta_generic_to_shared(s)),
                 "l"(g), "r"(bytes));
}
/// 8 bytes (cp.async.ca: a source that is no multiple of 16 bytes into its buffer)
__device__ __forceinline__ void cp_async8(void* s, const void* g) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"((unsigned) __cvta_generic_to_shared(s)), "l"(g));
}
/// 4 bytes (cp.async.ca)
__device__ __forceinline__ void cp_async4(void* s, const void* g) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"((unsigned) __cvta_generic_to_shared(s)), "l"(g));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
/// waits until at most N committed groups are still in flight
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

/// d += a b on the tensor cores: a 16 x 16 tile of halves by a 16 x 8 one, FP32 sums (mma.m16n8k16, the operands as
/// ldmatrix leaves them)
__device__ __forceinline__ void mma16816(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
/// The same with BF16 operands (mma.m16n8k16 .bf16)
__device__ __forceinline__ void mma16816_bf16(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
/// d += a b on the int8 tensor cores: a 16 x 32 tile of signed bytes by a 32 x 8 one, int32 sums (mma.m16n8k32 .s8)
__device__ __forceinline__ void mma16832_s8(int (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
/// two floats as the halves of a word, rounded to nearest (an mma operand's)
__device__ __forceinline__ unsigned h2(float a, float b) {
    const __half2 v = __floats2half2_rn(a, b);
    return *reinterpret_cast<const unsigned*>(&v);
}

}   // namespace lamina::kernels
