#ifndef FERRIC_CONTINUUM_CUDA_KERNELS_COMMON_CUDA_CHECK_HH_
#define FERRIC_CONTINUUM_CUDA_KERNELS_COMMON_CUDA_CHECK_HH_

// Error-handling helpers for CUDA runtime calls inside kernel implementations.
//
// `FERRIC_CUDA_CHECK(call)` evaluates a CUDA runtime expression and, on failure,
// bails out of the enclosing function by `return`ing FERRIC_CUDA_ERR_DEVICE.
// It is meant for use inside functions whose return type is FerricCudaStatus
// (the kernel C ABI). The failing call, file, and line are reported on stderr so
// device failures are never silent.

#include <cuda_runtime.h>

#include <cstdio>

#include "ferric/cuda/status.h"

// Return FERRIC_CUDA_ERR_DEVICE from the current function if `call` fails.
#define FERRIC_CUDA_CHECK(call)                                              \
  do {                                                                       \
    const cudaError_t ferric_cuda_check_status_ = (call);                    \
    if (ferric_cuda_check_status_ != cudaSuccess) {                          \
      std::fprintf(stderr, "FERRIC_CUDA_CHECK(%s) failed at %s:%d: %s\n",    \
                   #call, __FILE__, __LINE__,                                \
                   cudaGetErrorString(ferric_cuda_check_status_));           \
      return FERRIC_CUDA_ERR_DEVICE;                                         \
    }                                                                        \
  } while (0)

// After a kernel launch, check both the launch error and the synchronous
// completion. Use this instead of a bare FERRIC_CUDA_CHECK so that both the
// dispatch-time and execution-time errors are surfaced.
#define FERRIC_CUDA_CHECK_KERNEL()                                           \
  do {                                                                       \
    FERRIC_CUDA_CHECK(cudaGetLastError());                                   \
    FERRIC_CUDA_CHECK(cudaDeviceSynchronize());                             \
  } while (0)

#endif  // FERRIC_CONTINUUM_CUDA_KERNELS_COMMON_CUDA_CHECK_HH_
