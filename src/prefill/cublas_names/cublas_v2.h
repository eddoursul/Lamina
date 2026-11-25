// cublas_v2.h - the names of cuBLAS that llama.cpp's ggml-cuda headers mention (common.cuh: a context's handle, made
// on demand, and an error's text), for a CUDA toolkit installed without cuBLAS.  The engine calls none of them and
// links no cuBLAS: CMakeLists.txt puts this folder on the MMQ library's include path only where the toolkit has no
// cublas_v2.h of its own.  Declarations alone, as far as those headers go at the pinned llama.cpp commit.
#pragma once

#include <cuda_runtime.h>

#include <stddef.h>

#define CUBLAS_VER_MAJOR 13
#define CUBLAS_VER_MINOR 0

struct cublasContext;
typedef struct cublasContext* cublasHandle_t;

typedef enum {
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_STATUS_NOT_INITIALIZED = 1,
    CUBLAS_STATUS_ALLOC_FAILED = 3,
    CUBLAS_STATUS_INVALID_VALUE = 7,
    CUBLAS_STATUS_ARCH_MISMATCH = 8,
    CUBLAS_STATUS_MAPPING_ERROR = 11,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
    CUBLAS_STATUS_INTERNAL_ERROR = 14,
    CUBLAS_STATUS_NOT_SUPPORTED = 15,
    CUBLAS_STATUS_LICENSE_ERROR = 16
} cublasStatus_t;

typedef enum {
    CUBLAS_DEFAULT_MATH = 0,
    CUBLAS_TENSOR_OP_MATH = 1,
    CUBLAS_PEDANTIC_MATH = 2,
    CUBLAS_TF32_TENSOR_OP_MATH = 3
} cublasMath_t;

#ifdef __cplusplus
extern "C" {
#endif
cublasStatus_t cublasCreate(cublasHandle_t* handle);
cublasStatus_t cublasDestroy(cublasHandle_t handle);
cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode);
cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t stream);
cublasStatus_t cublasSetWorkspace(cublasHandle_t handle, void* workspace, size_t bytes);
const char* cublasGetStatusString(cublasStatus_t status);
#ifdef __cplusplus
}
#endif
