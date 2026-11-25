// gemm.cu's FP32 instances: LAMINA_SG(GM, GN, TM, TN, ST, r86, e86, r120, e120) - GM x GN threads of TM x TN sums (a
// block: GM TM rows of X against GN TN rows of W), K in slabs of 32 through ST buffers.  r: its rate as a share of the
// card's; e: of that with four warps an SM.  Fitted to sweeps of the paths' FP32 shapes on an RTX 3090 (86) and an
// RTX 5070 Ti (120).
LAMINA_SG(16, 16, 8, 8, 2, 1.14, 1.00, 1.14, 1.00),   // 128 x 128
LAMINA_SG(16, 16, 8, 4, 2, 1.02, 1.00, 1.22, 1.00),   // 128 x 64
LAMINA_SG(16, 16, 4, 8, 2, 1.11, 1.00, 1.10, 1.00),   // 64 x 128
LAMINA_SG(16, 16, 4, 4, 2, 1.04, 1.00, 1.06, 1.00),   // 64 x 64
LAMINA_SG(8, 16, 4, 4, 3, 0.94, 0.77, 0.88, 0.68),    // 32 x 64, four warps: few rows
LAMINA_SG(16, 16, 2, 2, 3, 0.64, 1.00, 0.70, 1.00),   // 32 x 32
