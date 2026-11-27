// include/lamina/inkling2/kernel_sizes.hpp - the sizes Inkling 2's kernels are built for (src/inkling2/
// inkling2_kernels.cuh takes its constants from here), host code's too: what a model's geometry may be is checked
// against them where no CUDA header can be read (kernel_limits.hpp).
#pragma once

namespace lamina::inkling2::sizes {

inline constexpr int kMaxT = 8;          // rows a forward takes at most (native_mmvq's columns)
inline constexpr int kBlk = 4;           // block inputs a mix reads at most (the embedding and three blocks' outputs)
inline constexpr int kTB = 256;          // threads of the row kernels
inline constexpr int kWide = 1024;       // of those whose time is their threads' loop
                                         // over a row: the mix, the norm, the bias
inline constexpr int kRankMax = 64;      // the loop gate's rank at most
inline constexpr int kRelMax = 32;       // the distance bias's values a head at most
inline constexpr int kRingRs = 1024;     // a row's r values, every head's
inline constexpr int kRingHeads = 64;    // heads at most
inline constexpr int kQH = 4;            // query heads a K/V head at most
inline constexpr int kExpertMax = 256;   // experts a layer at most
inline constexpr int kTcCells = 32;      // cells a tile

}   // namespace lamina::inkling2::sizes
