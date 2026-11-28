// include/lamina/kernels/expert_groups.cuh - the routed entries (T tokens x K picks, entry e = t * K + k) grouped per
// expert on the device, for the qwen35 path's and Hy4-flash's grouped expert kernels: one group per expert used, in
// ascending id, its blob's address, its entries in entry order (ent_dst the entry, ent_tok its token).  Three forms
// write the same arrays: the serial one (one thread's walk, any count of entries), the rank one (n_ent <= 64) and the
// scan one (n_ent <= 128, n_expert <= 512).  HY: only the entries whose expert holds a VRAM slot (slot_of[x] >= 0),
// the group's blob at blobs + slot * blob; else every entry, the blob at blobs + x * blob.  With them, the two copies
// the paths share: the CPU pool's rows into the entries' rows, and picture rows into a chunk's embeddings.  Integer
// work and copies alone: each file's copy of them (an unnamed namespace) is the same in either compile mode.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace lamina::kernels {
namespace {

template <bool HY> __device__ __forceinline__ bool group_held(const int32_t* slot_of, int x) {
    return !HY || slot_of[x] >= 0;
}
template <bool HY>
__device__ __forceinline__ unsigned long long group_blob(const uint8_t* blobs, unsigned long long blob,
                                                         const int32_t* slot_of, int x) {
    return (unsigned long long) (blobs + (size_t) (HY ? slot_of[x] : x) * blob);
}

/// The serial form: one block; its first thread walks the experts (CAP at most: the counts in shared memory) and the
/// entries
template <int CAP, bool HY>
__global__ void group_serial(const int32_t* ids, int n_ent, int K_, int n_expert, const int32_t* slot_of,
                             const uint8_t* blobs, unsigned long long blob, unsigned long long* grp_ptr,
                             int32_t* grp_start, int32_t* n_groups, int32_t* ent_dst, int32_t* ent_tok) {
    __shared__ int cnt[CAP], off[CAP];
    for (int x = threadIdx.x; x < n_expert; x += blockDim.x) cnt[x] = 0;
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x)
        if (group_held<HY>(slot_of, ids[e])) atomicAdd(&cnt[ids[e]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int g = 0, pos = 0;
        for (int x = 0; x < n_expert; ++x) {
            off[x] = pos;
            if (cnt[x] == 0) continue;
            grp_ptr[g] = group_blob<HY>(blobs, blob, slot_of, x);
            grp_start[g] = pos;
            pos += cnt[x];
            ++g;
        }
        grp_start[g] = pos;
        *n_groups = g;
        for (int e = 0; e < n_ent; ++e) {
            if (!group_held<HY>(slot_of, ids[e])) continue;
            const int p = off[ids[e]]++;
            ent_dst[p] = e;
            ent_tok[p] = e / K_;
        }
    }
}

/// The rank form, n_ent <= 64: one block of 64 threads ranks each entry by (expert id, entry) in parallel
template <bool HY>
__global__ void group_rank(const int32_t* ids, int n_ent, int K_, const int32_t* slot_of, const uint8_t* blobs,
                           unsigned long long blob, unsigned long long* grp_ptr, int32_t* grp_start, int32_t* n_groups,
                           int32_t* ent_dst, int32_t* ent_tok) {
    __shared__ int sid[64];
    const int e = threadIdx.x;
    const int id = e < n_ent ? ids[e] : 0;
    const bool valid = e < n_ent && group_held<HY>(slot_of, id);
    int rank = 0;
    if (valid) {
        for (int e2 = 0; e2 < n_ent; ++e2) {
            const int id2 = ids[e2];
            if (!group_held<HY>(slot_of, id2)) continue;
            if (id2 < id || (id2 == id && e2 < e)) ++rank;
        }
        ent_dst[rank] = e;
        ent_tok[rank] = e / K_;
        sid[rank] = id;
    }
    __syncthreads();
    int nv = 0;
    for (int e2 = 0; e2 < n_ent; ++e2) nv += group_held<HY>(slot_of, ids[e2]) ? 1 : 0;
    if (valid && (rank == 0 || sid[rank - 1] != sid[rank])) {
        int g = 0;
        for (int q = 1; q < rank; ++q) g += sid[q - 1] != sid[q] ? 1 : 0;
        g += rank > 0 ? 1 : 0;
        grp_ptr[g] = group_blob<HY>(blobs, blob, slot_of, id);
        grp_start[g] = rank;
    }
    if (e == 0) {
        int G = nv > 0 ? 1 : 0;
        for (int q = 1; q < nv; ++q) G += sid[q - 1] != sid[q] ? 1 : 0;
        grp_start[G] = nv;
        *n_groups = G;
    }
}

/// The scan form, n_ent <= 128 and n_expert <= 512: one block of 512 threads - the experts' counts, two scans over them
/// (an expert's group and its first place), each entry's place by the entries of its expert before it
template <bool HY>
__global__ void __launch_bounds__(512) group_scan(const int32_t* ids, int n_ent, int K_, int n_expert,
                                                  const int32_t* slot_of, const uint8_t* blobs, unsigned long long blob,
                                                  unsigned long long* grp_ptr, int32_t* grp_start, int32_t* n_groups,
                                                  int32_t* ent_dst, int32_t* ent_tok) {
    __shared__ int cnt[512];
    __shared__ int wsum[2][16];
    __shared__ int e_ids[128];
    const int x = threadIdx.x, lane = x & 31, warp = x >> 5;
    cnt[x] = 0;
    if (x < n_ent) e_ids[x] = ids[x];
    __syncthreads();
    if (x < n_ent && group_held<HY>(slot_of, e_ids[x])) atomicAdd(&cnt[e_ids[x]], 1);
    __syncthreads();
    const int c = x < n_expert ? cnt[x] : 0;
    // inclusive warp scans of (c > 0) and c, then the warps' totals
    int g = c > 0 ? 1 : 0, o = c;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const int g2 = __shfl_up_sync(0xffffffffu, g, d), o2 = __shfl_up_sync(0xffffffffu, o, d);
        if (lane >= d) {
            g += g2;
            o += o2;
        }
    }
    if (lane == 31) {
        wsum[0][warp] = g;
        wsum[1][warp] = o;
    }
    __syncthreads();
    int gb = 0, ob = 0;
    for (int i = 0; i < warp; ++i) {
        gb += wsum[0][i];
        ob += wsum[1][i];
    }
    const int grp = gb + g - (c > 0 ? 1 : 0), off = ob + o - c;   // exclusive
    if (c > 0) {
        grp_ptr[grp] = group_blob<HY>(blobs, blob, slot_of, x);
        grp_start[grp] = off;
    }
    if (x == 511) {
        int gt = 0, ot = 0;
        for (int i = 0; i < 16; ++i) {
            gt += wsum[0][i];
            ot += wsum[1][i];
        }
        grp_start[gt] = ot;
        *n_groups = gt;
    }
    __syncthreads();
    cnt[x] = off;   // each expert's first place
    __syncthreads();
    if (x < n_ent) {
        const int ex = e_ids[x];
        if (group_held<HY>(slot_of, ex)) {
            int rank = 0;
            for (int e = 0; e < x; ++e) rank += e_ids[e] == ex ? 1 : 0;
            const int p = cnt[ex] + rank;
            ent_dst[p] = x;
            ent_tok[p] = x / K_;
        }
    }
}

/// The CPU pool's rows into the entries' rows: parts[e] = cpu[e] where miss[e].  A block per entry.
__global__ void merge_cpu_rows(float* parts, const float* cpu, const int32_t* miss, int n) {
    const int e = blockIdx.x;
    if (!miss[e]) return;
    for (int i = threadIdx.x; i < n; i += blockDim.x) parts[(size_t) e * n + i] = cpu[(size_t) e * n + i];
}

/// Picture rows over a chunk's embeddings: x[r] = img[map[r]] for the rows with map[r] >= 0.  A block per row.
__global__ void put_rows(float* x, const float* img, const int32_t* map, int n) {
    const int r = blockIdx.x;
    const int m = map[r];
    if (m < 0) return;
    for (int i = threadIdx.x; i < n; i += blockDim.x) x[(size_t) r * n + i] = img[(size_t) m * n + i];
}

}   // namespace
}   // namespace lamina::kernels
