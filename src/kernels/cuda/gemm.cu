// src/kernels/cuda/gemm.cu - see include/lamina/kernels/gemm.hpp.
//
// The kernels are gemm_tc.cuh's templates; this file holds their instances and the choice among them.  A product's
// time is its blocks' work at the instance's rate, a launch's and a block's fixed parts, and, with K split, the
// second launch that adds the partials up: plan() prices every instance at a few split counts by that model, its
// constants and the instances' rates fitted per card generation to measured sweeps of the paths' shapes, and takes
// the cheapest.  What decides most: an SM has four tensor units, a warp is bound to one, and a unit takes one warp's
// product at a time, so a card is full with as many blocks as it has SMs and a block's warps on every unit; a
// product of few rows has few tiles, and K is split to make them enough.  A product of a few outputs a row has a
// kernel of its own (a block a row), priced beside the tiles.
#include "lamina/kernels/gemm.hpp"

#include "gemm_tc.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace lamina::kernels {
namespace {

using namespace gemmk;

[[noreturn]] void die(const char* what, cudaError_t e) {
    std::fprintf(stderr, "gemm: %s: %s\n", what, cudaGetErrorString(e));
    std::exit(1);
}
void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) die(what, e);
}

constexpr int kMaxDevices = 16;

// ---- the card: its SM count and the model's constants of its generation
struct Model {
    int gen;           // which of an instance's rates
    double r_tc;       // products of 16 x 8 x 16 an SM a microsecond, its tensor units busy
    double r_f32;      // float multiply-adds an SM a microsecond, in thousands
    double floor;      // a launch (us)
    double ob;         // a block's start and stores (us): the tensor-core kernel's
    double ob_f32;     // the float kernel's
    double red;        // the reduce's bytes a us
    double dram;       // the card's memory, bytes a us
    double rows;       // the row kernel: bytes a us through the cache (a row reads W whole)
    double rows_min;   // its least time (us)
    double l2;         // the L2 cache (bytes)
};
// RTX 3090 (sm_86) and RTX 5070 Ti (sm_120); a card of another generation takes the nearer one's
constexpr Model kAmpere{0, 211.0, 89.0, 7.2, 2.9, 2.7, 1.05e6, 0.65e6, 2.5e6, 13.0, 6e6};
constexpr Model kBlackwell{1, 357.0, 155.0, 6.1, 1.45, 0.6, 4.5e6, 0.71e6, 3.0e6, 8.0, 48e6};
struct Card {
    int sms = 0;
    const Model* m = nullptr;
};
const Card& card() {
    static Card cards[kMaxDevices];
    static std::atomic<bool> have[kMaxDevices];
    int dev = 0;
    if (const cudaError_t e = cudaGetDevice(&dev); e != cudaSuccess) die("cudaGetDevice", e);
    if (dev < 0 || dev >= kMaxDevices) {
        std::fprintf(stderr, "gemm: device %d (at most %d)\n", dev, kMaxDevices);
        std::exit(1);
    }
    if (!have[dev].load(std::memory_order_acquire)) {
        int sms = 0, major = 0;
        if (const cudaError_t e = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev); e != cudaSuccess)
            die("the card's SM count", e);
        if (const cudaError_t e = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
            e != cudaSuccess)
            die("the card's architecture", e);
        cards[dev].sms = std::max(1, sms);   // (two threads may write the same values)
        cards[dev].m = major >= 9 ? &kBlackwell : &kAmpere;
        have[dev].store(true, std::memory_order_release);
    }
    return cards[dev];
}
int device() {
    int dev = 0;
    cudaGetDevice(&dev);
    return dev;
}

// A kernel's dynamic shared memory above the default limit is set once a device.
struct SmemOnce {
    std::atomic<uint32_t> done{0};
    template <typename F> void operator()(F kernel, int bytes) {
        const uint32_t bit = 1u << device();
        if (done.load(std::memory_order_relaxed) & bit) return;
        if (const cudaError_t e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
            e != cudaSuccess)
            die("a kernel's shared memory", e);
        done.fetch_or(bit, std::memory_order_relaxed);
    }
};

// ---- an instance as the model prices it
struct Form {
    int bm, bn;           // rows of X and of W a block
    int warps, smem;      // a block's warps and shared memory
    int bk, es;           // a slab of K; bytes a value
    double work;          // a slab's: products of 16 x 8 x 16, or float multiply-adds in thousands
    double rate, share;   // of the card's rate with an SM's units busy; of that with four warps an SM
};

/// An instance's time (us) at a split count by the model: its blocks in rounds over the SMs - as many at once as an
/// SM's shared memory holds, then the rest - each round the blocks' work at the instance's rate and their starts and
/// stores; a split's second launch; never under what the operands and the sums take through the card's memory.
double price(const Card& c, const Form& f, int z, int64_t T, int64_t N, int64_t K) {
    const Model& m = *c.m;
    const bool f32 = f.es == 4;
    const int64_t cap = std::max(1, std::min(4, 99000 / f.smem));
    const int64_t blocks = ((T + f.bm - 1) / f.bm) * ((N + f.bn - 1) / f.bn) * z;
    const int64_t slabs = ((K + z - 1) / z + f.bk - 1) / f.bk;
    const double work = f.work * (double) slabs, r = (f32 ? m.r_f32 : m.r_tc) * f.rate;
    const int64_t per_sm = (blocks + c.sms - 1) / c.sms, full = per_sm / cap, rest = per_sm % cap;
    const auto with = [&](int64_t warps) { return warps >= 8 ? 1.0 : f.share * std::min(1.0, (double) warps / 4.0); };
    double t = (double) (full * cap) * work / (r * with(cap * f.warps));
    if (rest > 0) t += (double) rest * work / (r * with(rest * f.warps));
    t += (double) (full + (rest > 0 ? 1 : 0)) * (f32 ? m.ob_f32 : m.ob) + m.floor;
    if (z > 1) t += m.floor + (double) (z + 1) * (double) T * (double) N * 4 / m.red;
    return std::max(t, m.floor + ((double) (T + N) * (double) K * f.es + (double) T * (double) N * 4) / m.dram);
}

/// The row kernel's time (us): a block a row, each reading X's row and W whole.
double price_rows(const Card& c, int64_t T, int64_t N, int64_t K) {
    const Model& m = *c.m;
    const double x = (double) T * (double) K * 2;
    return std::max(m.rows_min, m.floor + std::max(x * (double) (N + 1) / m.rows, x / m.dram));
}

// ---- the tensor-core instances
struct TcCfg {
    int MT, NT, WM, WN, BK, ST;
    double rate[2], share[2];                            // [Model::gen]
    void (*run[2])(const TcArgs&, dim3, cudaStream_t);   // [bf16]
    constexpr int bm() const { return 16 * MT * WM; }
    constexpr int bn() const { return 16 * NT * WN; }
    Form form(int gen) const {
        return {bm(),
                bn(),
                WM * WN,
                ST * (bm() + bn()) * BK * 2,
                BK,
                2,
                (double) (bm() / 16) * (bn() / 8) * (BK / 16),
                rate[gen],
                share[gen]};
    }
    void name(char* out, size_t n) const { std::snprintf(out, n, "%d.%d.%d.%d.%d.%d", MT, NT, WM, WN, BK, ST); }
};
template <bool BF, int MT, int NT, int WM, int WN, int BK, int ST, bool PL>
void run_tc(const TcArgs& a, dim3 grid, cudaStream_t st) {
    static SmemOnce once;
    constexpr int smem = tc_smem_bytes<MT, NT, WM, WN, BK, ST>();
    once(tc_kernel<BF, MT, NT, WM, WN, BK, ST, PL>, smem);
    tc_kernel<BF, MT, NT, WM, WN, BK, ST, PL><<<grid, 32 * WM * WN, smem, st>>>(a);
}
#define LAMINA_TC(MT, NT, WM, WN, BK, ST, R86, E86, R120, E120)                                                        \
    TcCfg {                                                                                                            \
        MT, NT, WM, WN, BK, ST, {R86, R120}, {E86, E120}, {                                                            \
            &run_tc<false, MT, NT, WM, WN, BK, ST, false>, &run_tc<true, MT, NT, WM, WN, BK, ST, false>                \
        }                                                                                                              \
    }
constexpr TcCfg kTc[] = {
#include "gemm_tc_cfgs.inc"
};
// rows that are not 16-byte aligned: plain loads, one tile
constexpr TcCfg kTcPlain{
    2,          2,          2,
    2,          32,         3,
    {0.0, 0.0}, {0.0, 0.0}, {&run_tc<false, 2, 2, 2, 2, 32, 3, true>, &run_tc<true, 2, 2, 2, 2, 32, 3, true>}};

// ---- the FP32 instances: GM x GN threads of TM x TN sums
struct SgCfg {
    int GM, GN, TM, TN, ST;
    double rate[2], share[2];   // [Model::gen]
    void (*run)(const SgArgs&, dim3, cudaStream_t);
    constexpr int bm() const { return GM * TM; }
    constexpr int bn() const { return GN * TN; }
    Form form(int gen) const {
        return {bm(),
                bn(),
                GM * GN / 32,
                ST * (bm() + bn()) * kSgK * 4,
                kSgK,
                4,
                (double) bm() * bn() * kSgK / 1000.0,
                rate[gen],
                share[gen]};
    }
    void name(char* out, size_t n) const { std::snprintf(out, n, "s%d.%d.%d.%d.%d", GM, GN, TM, TN, ST); }
};
template <int GM, int GN, int TM, int TN, int ST, bool PL> void run_sg(const SgArgs& a, dim3 grid, cudaStream_t st) {
    static SmemOnce once;
    constexpr int smem = sg_smem_bytes<GM, GN, TM, TN, ST>();
    once(sg_kernel<GM, GN, TM, TN, ST, PL>, smem);
    sg_kernel<GM, GN, TM, TN, ST, PL><<<grid, GM * GN, smem, st>>>(a);
}
#define LAMINA_SG(GM, GN, TM, TN, ST, R86, E86, R120, E120)                                                            \
    SgCfg { GM, GN, TM, TN, ST, {R86, R120}, {E86, E120}, &run_sg<GM, GN, TM, TN, ST, false> }
constexpr SgCfg kSg[] = {
#include "gemm_sg_cfgs.inc"
};
constexpr SgCfg kSgPlain{16, 16, 4, 4, 3, {0.0, 0.0}, {0.0, 0.0}, &run_sg<16, 16, 4, 4, 3, true>};

struct Plan {
    int cfg = -1;   // the instance's index (-1: the kernel outside the list - the row kernel)
    int splits = 1;
    bool t_inner = false;
};

/// The cheapest of a list's instances at a few split counts, or the kernel outside the list where its time `other`
/// (us) is less.  A thread keeps its last plans: a chunk repeats its shapes layer after layer.
template <typename Cfg, size_t n>
Plan plan(const Card& c, const Cfg (&list)[n], int64_t T, int64_t N, int64_t K, size_t scratch_bytes, double other) {
    struct Slot {
        int64_t T, N, K;
        size_t scratch;
        int dev;
        Plan p;
    };
    constexpr int kSlots = 32;
    thread_local Slot cache[kSlots];
    thread_local int filled = 0, next = 0;
    const int dev = device();
    for (int i = 0; i < filled; ++i)
        if (cache[i].T == T && cache[i].N == N && cache[i].K == K && cache[i].scratch == scratch_bytes &&
            cache[i].dev == dev)
            return cache[i].p;
    Plan best;
    double best_t = other;
    int es = 2;
    const size_t part = (size_t) ((T * N + 3) / 4 * 4) * 4;   // a split's partials
    const int z_room = part > 0 ? (int) std::min<size_t>(scratch_bytes / part, 4096) : 1;
    for (size_t i = 0; i < n; ++i) {
        const Form f = list[i].form(c.m->gen);
        es = f.es;
        const int64_t tiles = ((T + f.bm - 1) / f.bm) * ((N + f.bn - 1) / f.bn);
        const int z_max = (int) std::max<int64_t>(1, std::min<int64_t>((K + f.bk - 1) / f.bk, z_room));
        int last = 0;
        for (const int per : {0, 1, 2, 3, 4, 6, 8}) {   // blocks an SM (0: K whole)
            int z = per == 0 ? 1 : (int) std::min<int64_t>(z_max, ((int64_t) per * c.sms + tiles - 1) / tiles);
            if (z > 1) {   // as the launch rounds it: a split's K in whole slabs
                const int64_t kz = ((K + z - 1) / z + f.bk - 1) / f.bk * f.bk;
                z = (int) ((K + kz - 1) / kz);
            }
            if (z == last) continue;
            last = z;
            const double t = price(c, f, z, T, N, K);
            if (t < best_t) {
                best_t = t;
                best.cfg = (int) i;
                best.splits = z;
            }
        }
    }
    // X's tiles the inner index where W is past the cache and X the smaller: W then comes from the card's memory once
    best.t_inner = (double) N * (double) K * es > c.m->l2 && T < N;
    cache[next] = {T, N, K, scratch_bytes, dev, best};
    next = (next + 1) % kSlots;
    filled = std::min(filled + 1, kSlots);
    return best;
}

bool aligned16(const void* p) { return ((uintptr_t) p & 15) == 0; }
size_t part_bytes(int64_t T, int64_t N) { return (size_t) ((T * N + 3) / 4 * 4) * 4; }

/// A launch's split count as it runs, K's parts in whole slabs: `kz` a part's length.
int whole_slabs(int64_t K, int splits, int slab, int64_t& kz) {
    kz = K;
    if (splits <= 1) return 1;
    kz = ((K + splits - 1) / splits + slab - 1) / slab * slab;
    return (int) ((K + kz - 1) / kz);
}

void launch_tc(const TcCfg& f, int splits, bool t_inner, const Gemm16& p, int64_t ldx, int64_t ldy,
               const GemmScratch& scratch, cudaStream_t st) {
    const int BM = f.bm(), BN = f.bn();
    const int nT = (int) ((p.T + BM - 1) / BM), nN = (int) ((p.N + BN - 1) / BN);
    TcArgs a{};
    a.x = p.x;
    a.w = p.w;
    a.ldx = ldx;
    a.T = (int) p.T;
    a.N = (int) p.N;
    a.K = (int) p.K;
    a.nI = t_inner ? nT : nN;
    a.nO = t_inner ? nN : nT;
    a.ci = 0;
    int64_t kz = 0;
    splits = whole_slabs(p.K, splits, f.BK, kz);
    a.kz = (int) kz;
    const int esz = p.y_half ? 2 : 4;
    // a store of two values: Y's pairs lie at multiples of their size
    const bool pair = ((uintptr_t) p.y % (2 * esz)) == 0 && ldy % 2 == 0;
    if (splits > 1) {
        const long long zs = (p.T * p.N + 3) / 4 * 4;
        a.y = scratch.p;
        a.ldy = p.N;
        a.zstride = zs;
        a.alpha = 1.0f;
        a.beta = 0.0f;
        a.flags = (p.N % 2 == 0 ? kYPair : 0) | (t_inner ? kTInner : 0);
        f.run[p.bf16](a, dim3((unsigned) (nT * nN), 1, (unsigned) splits), st);
        const long long n4 = (p.T * p.N + 3) / 4;
        const unsigned blocks = (unsigned) ((n4 + 255) / 256);
        if (p.y_half)
            tc_reduce_kernel<true><<<blocks, 256, 0, st>>>((const float*) scratch.p, p.y, ldy, zs, (int) p.T, (int) p.N,
                                                           splits, p.alpha, p.beta);
        else
            tc_reduce_kernel<false><<<blocks, 256, 0, st>>>((const float*) scratch.p, p.y, ldy, zs, (int) p.T,
                                                            (int) p.N, splits, p.alpha, p.beta);
    } else {
        a.y = p.y;
        a.ldy = ldy;
        a.zstride = 0;
        a.alpha = p.alpha;
        a.beta = p.beta;
        a.flags = (p.y_half ? kYHalf : 0) | (pair ? kYPair : 0) | (t_inner ? kTInner : 0);
        f.run[p.bf16](a, dim3((unsigned) (nT * nN), 1, 1), st);
    }
}

void launch_sg(const SgCfg& f, int splits, bool t_inner, const Gemm32& p, int64_t ldx, int64_t ldy,
               const GemmScratch& scratch, cudaStream_t st) {
    const int BM = f.bm(), BN = f.bn();
    const int nT = (int) ((p.T + BM - 1) / BM), nN = (int) ((p.N + BN - 1) / BN);
    SgArgs a{};
    a.x = p.x;
    a.w = p.w;
    a.ldx = ldx;
    a.T = (int) p.T;
    a.N = (int) p.N;
    a.K = (int) p.K;
    a.nI = t_inner ? nT : nN;
    a.nO = t_inner ? nN : nT;
    a.ci = 0;
    int64_t kz = 0;
    splits = whole_slabs(p.K, splits, kSgK, kz);
    a.kz = (int) kz;
    if (splits > 1) {
        const long long zs = (p.T * p.N + 3) / 4 * 4;
        a.y = (float*) scratch.p;
        a.ldy = p.N;
        a.zstride = zs;
        a.alpha = 1.0f;
        a.beta = 0.0f;
        // a store of four values: Y's fours lie at multiples of 16 bytes
        a.flags = (t_inner ? kTInner : 0) | (aligned16(scratch.p) && p.N % 4 == 0 ? kYQuad : 0);
        f.run(a, dim3((unsigned) (nT * nN), 1, (unsigned) splits), st);
        const long long n4 = (p.T * p.N + 3) / 4;
        tc_reduce_kernel<false><<<(unsigned) ((n4 + 255) / 256), 256, 0, st>>>(
            (const float*) scratch.p, p.y, ldy, zs, (int) p.T, (int) p.N, splits, p.alpha, p.beta);
    } else {
        a.y = p.y;
        a.ldy = ldy;
        a.zstride = 0;
        a.alpha = p.alpha;
        a.beta = p.beta;
        a.flags = (t_inner ? kTInner : 0) | (aligned16(p.y) && ldy % 4 == 0 ? kYQuad : 0);
        f.run(a, dim3((unsigned) (nT * nN), 1, 1), st);
    }
}

/// Rows that take plain loads: 64 x 64 tiles, K split for a block an SM where the scratch has the room.
int plain_splits(const Card& c, int64_t T, int64_t N, int64_t K, const GemmScratch& scratch) {
    const size_t part = part_bytes(T, N);
    const int64_t tiles = ((T + 63) / 64) * ((N + 63) / 64);
    int z = (int) std::min<int64_t>((c.sms + tiles - 1) / tiles, (K + 31) / 32);
    z = (int) std::min<size_t>((size_t) z, scratch.p != nullptr && part > 0 ? scratch.bytes / part : 1);
    return std::max(1, z);
}

[[noreturn]] void bad_form(const char* what, int form, int splits, int64_t T, int64_t N, int64_t K) {
    std::fprintf(stderr, "%s: instance %d, %d splits of %lld x %lld x %lld\n", what, form, splits, (long long) T,
                 (long long) N, (long long) K);
    std::exit(1);
}

}   // namespace

size_t gemm_scratch_bytes(int64_t T, int64_t N) { return std::min<size_t>(32 * part_bytes(T, N), (size_t) 32 << 20); }

void gemm16(const Gemm16& p, const GemmScratch& scratch, void* stream) {
    if (p.T <= 0 || p.N <= 0) return;
    if (p.K <= 0 || p.T > INT32_MAX || p.N > INT32_MAX || p.K > INT32_MAX) {
        std::fprintf(stderr, "gemm16: %lld x %lld x %lld\n", (long long) p.T, (long long) p.N, (long long) p.K);
        std::exit(1);
    }
    const cudaStream_t st = (cudaStream_t) stream;
    const int64_t ldx = p.ldx > 0 ? p.ldx : p.K, ldy = p.ldy > 0 ? p.ldy : p.N;
    const Card& c = card();
    if (!(aligned16(p.x) && aligned16(p.w) && ldx % 8 == 0 && p.K % 8 == 0)) {
        launch_tc(kTcPlain, plain_splits(c, p.T, p.N, p.K, scratch), false, p, ldx, ldy, scratch, st);
        check("gemm16");
        return;
    }
    // a few outputs a row: the row kernel where the model has it cheaper than the tiles
    const Plan pl = plan(c, kTc, p.T, p.N, p.K, scratch.p != nullptr ? scratch.bytes : 0,
                         p.N <= kRdN ? price_rows(c, p.T, p.N, p.K) : 1e300);
    if (pl.cfg < 0) {
        RdArgs a{};
        a.x = p.x;
        a.w = p.w;
        a.y = p.y;
        a.ldx = ldx;
        a.ldy = ldy;
        a.T = (int) p.T;
        a.N = (int) p.N;
        a.K = (int) p.K;
        a.wpr = 8;
        a.alpha = p.alpha;
        a.beta = p.beta;
        a.flags = p.y_half ? kYHalf : 0;
        if (p.bf16) rd_kernel<true><<<(unsigned) p.T, 256, 0, st>>>(a);
        else rd_kernel<false><<<(unsigned) p.T, 256, 0, st>>>(a);
    } else {
        launch_tc(kTc[pl.cfg], pl.splits, pl.t_inner, p, ldx, ldy, scratch, st);
    }
    check("gemm16");
}

int gemm16_plan(int64_t T, int64_t N, int64_t K, size_t scratch_bytes, char* name, size_t name_size) {
    const Card& c = card();
    const Plan pl = plan(c, kTc, T, N, K, scratch_bytes, N <= kRdN ? price_rows(c, T, N, K) : 1e300);
    if (pl.cfg < 0) std::snprintf(name, name_size, "rows");
    else kTc[pl.cfg].name(name, name_size);
    return pl.splits;
}

int gemm16_forms(int form, char* name, size_t name_size) {
    constexpr int n = (int) (sizeof kTc / sizeof kTc[0]);
    if (form >= 0 && form < n && name != nullptr) kTc[form].name(name, name_size);
    return n;
}

void gemm16_form(int form, int splits, bool x_inner, const Gemm16& p, const GemmScratch& scratch, void* stream) {
    if (p.T <= 0 || p.N <= 0) return;
    const int64_t ldx = p.ldx > 0 ? p.ldx : p.K, ldy = p.ldy > 0 ? p.ldy : p.N;
    if (form < 0 || form >= gemm16_forms(-1, nullptr, 0) || p.K <= 0 || !aligned16(p.x) || !aligned16(p.w) ||
        ldx % 8 != 0 || p.K % 8 != 0 || splits < 1 ||
        (splits > 1 && (scratch.p == nullptr || scratch.bytes / part_bytes(p.T, p.N) < (size_t) splits)))
        bad_form("gemm16_form", form, splits, p.T, p.N, p.K);
    launch_tc(kTc[form], splits, x_inner, p, ldx, ldy, scratch, (cudaStream_t) stream);
    check("gemm16_form");
}

void gemm16_grouped(const uint16_t* const* x, const uint16_t* const* w, float* const* y, const int* rows, int n,
                    int64_t N, int64_t K, bool bf16, void* stream) {
    if (n <= 0 || N <= 0) return;
    if (K <= 0 || K % 8 != 0 || N > INT32_MAX || K > INT32_MAX) {
        std::fprintf(stderr, "gemm16_grouped: %lld x %lld (K a multiple of 8)\n", (long long) N, (long long) K);
        std::exit(1);
    }
    // 64 x 128 tiles, eight warps: an expert's rows are few, and a batch's tiles fill the card without splitting K
    constexpr int MT = 2, NT = 2, WM = 2, WN = 4, BK = 32, ST = 3, BM = 16 * MT * WM, BN = 16 * NT * WN;
    constexpr int smem = tc_smem_bytes<MT, NT, WM, WN, BK, ST>();
    static SmemOnce once[2];
    const cudaStream_t st = (cudaStream_t) stream;
    const int nN = (int) ((N + BN - 1) / BN);
    for (int i0 = 0; i0 < n; i0 += kTcGroupMax) {
        TcGroupArgs a{};
        a.x = x + i0;
        a.w = w + i0;
        a.y = y + i0;
        a.n = std::min(kTcGroupMax, n - i0);
        a.N = (int) N;
        a.K = (int) K;
        int blocks = 0;
        for (int i = 0; i < a.n; ++i) {
            a.first[i] = blocks;
            a.rows[i] = rows[i0 + i];
            blocks += (std::max(rows[i0 + i], 0) + BM - 1) / BM * nN;
        }
        a.first[a.n] = blocks;
        if (blocks == 0) continue;
        if (bf16) {
            once[1](tc_group_kernel<true, MT, NT, WM, WN, BK, ST>, smem);
            tc_group_kernel<true, MT, NT, WM, WN, BK, ST><<<(unsigned) blocks, 32 * WM * WN, smem, st>>>(a);
        } else {
            once[0](tc_group_kernel<false, MT, NT, WM, WN, BK, ST>, smem);
            tc_group_kernel<false, MT, NT, WM, WN, BK, ST><<<(unsigned) blocks, 32 * WM * WN, smem, st>>>(a);
        }
    }
    check("gemm16_grouped");
}

void gemm32(const Gemm32& p, const GemmScratch& scratch, void* stream) {
    if (p.T <= 0 || p.N <= 0) return;
    if (p.K <= 0 || p.T > INT32_MAX || p.N > INT32_MAX || p.K > INT32_MAX) {
        std::fprintf(stderr, "gemm32: %lld x %lld x %lld\n", (long long) p.T, (long long) p.N, (long long) p.K);
        std::exit(1);
    }
    const cudaStream_t st = (cudaStream_t) stream;
    const int64_t ldx = p.ldx > 0 ? p.ldx : p.K, ldy = p.ldy > 0 ? p.ldy : p.N;
    const Card& c = card();
    if (!(aligned16(p.x) && aligned16(p.w) && ldx % 4 == 0 && p.K % 4 == 0)) {
        launch_sg(kSgPlain, plain_splits(c, p.T, p.N, p.K, scratch), false, p, ldx, ldy, scratch, st);
    } else {
        const Plan pl = plan(c, kSg, p.T, p.N, p.K, scratch.p != nullptr ? scratch.bytes : 0, 1e300);
        launch_sg(kSg[pl.cfg], pl.splits, pl.t_inner, p, ldx, ldy, scratch, st);
    }
    check("gemm32");
}

int gemm32_plan(int64_t T, int64_t N, int64_t K, size_t scratch_bytes, char* name, size_t name_size) {
    const Plan pl = plan(card(), kSg, T, N, K, scratch_bytes, 1e300);
    kSg[pl.cfg].name(name, name_size);
    return pl.splits;
}

int gemm32_forms(int form, char* name, size_t name_size) {
    constexpr int n = (int) (sizeof kSg / sizeof kSg[0]);
    if (form >= 0 && form < n && name != nullptr) kSg[form].name(name, name_size);
    return n;
}

void gemm32_form(int form, int splits, bool x_inner, const Gemm32& p, const GemmScratch& scratch, void* stream) {
    if (p.T <= 0 || p.N <= 0) return;
    const int64_t ldx = p.ldx > 0 ? p.ldx : p.K, ldy = p.ldy > 0 ? p.ldy : p.N;
    if (form < 0 || form >= gemm32_forms(-1, nullptr, 0) || p.K <= 0 || !aligned16(p.x) || !aligned16(p.w) ||
        ldx % 4 != 0 || p.K % 4 != 0 || splits < 1 ||
        (splits > 1 && (scratch.p == nullptr || scratch.bytes / part_bytes(p.T, p.N) < (size_t) splits)))
        bad_form("gemm32_form", form, splits, p.T, p.N, p.K);
    launch_sg(kSg[form], splits, x_inner, p, ldx, ldy, scratch, (cudaStream_t) stream);
    check("gemm32_form");
}

}   // namespace lamina::kernels
