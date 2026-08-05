#ifndef FERRIC_CUDA_STATUS_H_
#define FERRIC_CUDA_STATUS_H_

// Status codes returned by every Ferric CUDA kernel entry point.
//
// This header is pure C so it can be included from C++ (demos, gtests, the
// kernel .cu files) and mirrored by Rust FFI without an ABI mismatch.

#ifdef __cplusplus
extern "C" {
#endif

typedef enum FerricCudaStatus {
  // The call succeeded and outputs are valid.
  FERRIC_CUDA_OK = 0,
  // A shape or pointer argument failed validation; no device work was launched.
  FERRIC_CUDA_ERR_INVALID_ARG = 1,
  // A CUDA runtime call failed (allocation, copy, launch, or sync). Outputs are
  // undefined.
  FERRIC_CUDA_ERR_DEVICE = 2,
} FerricCudaStatus;

#ifdef __cplusplus
}
#endif

#endif  // FERRIC_CUDA_STATUS_H_
