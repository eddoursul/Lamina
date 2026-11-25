// gemm.cu's tensor-core instances: LAMINA_TC(MT, NT, WM, WN, BK, ST, r86, e86, r120, e120) - a warp's 16 MT x 16 NT
// sums, WM x WN warps a block, K in slabs of BK through ST buffers (a block: 16 MT WM rows of X against 16 NT WN rows
// of W).  r: its products' rate as a share of what the card's tensor units issue; e: of that with four warps an SM
// (a block of four alone on its SM: each tensor unit has one warp, and idles while it copies and loads).  Fitted to
// sweeps of the prompt paths' shapes on an RTX 3090 (86) and an RTX 5070 Ti (120).  Every one keeps its sums and
// operands in registers (no stack frame).
LAMINA_TC(4, 4, 2, 2, 32, 3, 0.83, 1.00, 0.94, 1.00),   // 128 x 128, four warps of 64 x 64: two blocks an SM
LAMINA_TC(2, 4, 4, 2, 64, 3, 0.88, 1.00, 0.92, 1.00),   // 128 x 128, eight warps
LAMINA_TC(2, 4, 4, 1, 32, 3, 0.82, 0.99, 0.95, 1.00),   // 128 x 64, four warps
LAMINA_TC(2, 2, 4, 2, 32, 3, 0.83, 1.00, 0.88, 1.00),   // 128 x 64, eight warps
LAMINA_TC(2, 2, 2, 4, 32, 3, 0.83, 1.00, 0.85, 1.00),   // 64 x 128
LAMINA_TC(2, 2, 2, 4, 64, 3, 0.80, 1.00, 0.91, 1.00),
LAMINA_TC(2, 2, 2, 2, 64, 3, 0.78, 1.00, 0.96, 1.00),   // 64 x 64, four warps
LAMINA_TC(2, 2, 2, 2, 64, 4, 0.84, 0.87, 0.95, 0.99),
LAMINA_TC(1, 2, 4, 2, 32, 3, 0.77, 1.00, 0.85, 1.00),   // 64 x 64, eight warps
LAMINA_TC(2, 1, 2, 4, 32, 3, 0.80, 1.00, 0.84, 1.00),
LAMINA_TC(2, 5, 4, 1, 32, 3, 0.92, 0.98, 0.89, 1.00),   // 128 x 80, x 160: a low rank of 320, and wide tiles
LAMINA_TC(2, 5, 4, 2, 32, 3, 0.90, 1.00, 0.85, 1.00),
LAMINA_TC(2, 3, 4, 1, 32, 3, 0.84, 0.96, 0.89, 0.96),   // 128 x 48, x 16: narrow outputs
LAMINA_TC(2, 1, 4, 1, 32, 3, 0.58, 0.55, 0.64, 0.61),   // (alone on an SM: as its blocks of a whole K run)
LAMINA_TC(2, 2, 1, 4, 32, 3, 0.72, 0.96, 0.80, 0.75),   // 32 x 128, 32 x 64, 16 x 256: few rows
LAMINA_TC(2, 2, 1, 2, 64, 3, 0.76, 0.87, 0.82, 0.99),
LAMINA_TC(1, 2, 1, 8, 32, 3, 0.51, 1.00, 0.52, 1.00),
