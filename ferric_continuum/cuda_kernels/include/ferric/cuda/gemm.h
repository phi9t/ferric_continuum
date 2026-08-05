#ifndef FERRIC_CUDA_GEMM_H_
#define FERRIC_CUDA_GEMM_H_

// Single-precision dense matrix multiply: C = A * B.
//
// Row-major FP32 layout, matching tnsr's `raw_linear_forward`:
//   A is (m x k), B is (k x n), C is (m x n), all row-major and contiguous.
// Pointers are HOST pointers; the implementation copies to the device, launches
// the kernel, and copies the result back. This convenience convention keeps
// lesson and FFI packaging simple; device-pointer / stream variants may be
// added later without changing this entry point.

#include "ferric/cuda/status.h"

#ifdef __cplusplus
extern "C" {
#endif

// Naive one-thread-per-output-element GEMM.
//
// Returns FERRIC_CUDA_ERR_INVALID_ARG if any dimension is negative or, when a
// dimension is positive, the corresponding pointer is null. Zero-sized problems
// (any of m, n, k == 0) succeed as a no-op. Returns FERRIC_CUDA_ERR_DEVICE if a
// CUDA runtime call fails.
FerricCudaStatus ferric_cuda_gemm_f32(int m, int n, int k, const float* a_host,
                                      const float* b_host, float* c_host);

// Tiled shared-memory GEMM with identical numerics and contract to the naive
// version. Provided so lessons and benchmarks can contrast the two.
FerricCudaStatus ferric_cuda_gemm_tiled_f32(int m, int n, int k,
                                            const float* a_host,
                                            const float* b_host, float* c_host);

#ifdef __cplusplus
}
#endif

#endif  // FERRIC_CUDA_GEMM_H_
