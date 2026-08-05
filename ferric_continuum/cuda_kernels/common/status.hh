#ifndef FERRIC_CONTINUUM_CUDA_KERNELS_COMMON_STATUS_HH_
#define FERRIC_CONTINUUM_CUDA_KERNELS_COMMON_STATUS_HH_

// Shared status enum for the Ferric CUDA kernel C ABI.
//
// The public kernel headers under `include/ferric/cuda/` are pure C so they can
// be consumed from C++ demos and from Rust FFI alike. This header re-exports the
// same enum for internal C++ use and provides a human-readable name helper.

#include "ferric/cuda/status.h"

namespace ferric_continuum::cuda_kernels {

// Returns a short, stable string for a status code (never nullptr).
inline const char* StatusName(FerricCudaStatus status) {
  switch (status) {
    case FERRIC_CUDA_OK:
      return "FERRIC_CUDA_OK";
    case FERRIC_CUDA_ERR_INVALID_ARG:
      return "FERRIC_CUDA_ERR_INVALID_ARG";
    case FERRIC_CUDA_ERR_DEVICE:
      return "FERRIC_CUDA_ERR_DEVICE";
  }
  return "FERRIC_CUDA_ERR_UNKNOWN";
}

}  // namespace ferric_continuum::cuda_kernels

#endif  // FERRIC_CONTINUUM_CUDA_KERNELS_COMMON_STATUS_HH_
