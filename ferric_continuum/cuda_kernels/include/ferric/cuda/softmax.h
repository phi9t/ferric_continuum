#ifndef FERRIC_CUDA_SOFTMAX_H_
#define FERRIC_CUDA_SOFTMAX_H_

// Numerically-stable row softmax over the last dimension.
//
// Input `x_host` and output `out_host` are row-major FP32 host pointers of shape
// (rows x cols): softmax is computed independently for each of the `rows` rows,
// over its `cols` entries, using the max-subtraction trick. This matches tnsr's
// `raw_softmax` (softmax over the last dim). In-place is allowed
// (out_host == x_host).

#include "ferric/cuda/status.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns FERRIC_CUDA_ERR_INVALID_ARG if rows/cols are negative, or if a
// positive-sized problem is given null pointers. rows == 0 or cols == 0 is a
// no-op success. Returns FERRIC_CUDA_ERR_DEVICE on CUDA runtime failure.
FerricCudaStatus ferric_cuda_softmax_f32(int rows, int cols,
                                         const float* x_host, float* out_host);

#ifdef __cplusplus
}
#endif

#endif  // FERRIC_CUDA_SOFTMAX_H_
