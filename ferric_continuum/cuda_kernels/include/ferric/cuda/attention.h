#ifndef FERRIC_CUDA_ATTENTION_H_
#define FERRIC_CUDA_ATTENTION_H_

// Single-head scaled dot-product attention: out = softmax(Q Kᵀ / sqrt(d)) V.
//
// Row-major FP32 host pointers:
//   q_host, k_host, v_host, out_host are each (seq x dim).
// The scores matrix S = Q Kᵀ is (seq x seq), scaled by 1/sqrt(dim), row-softmaxed
// to P, then out = P V is (seq x dim). This mirrors tnsr's toy attention path
// (single head, no mask). It is a teaching/reference primitive, not a fused
// flash-attention kernel.

#include "ferric/cuda/status.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns FERRIC_CUDA_ERR_INVALID_ARG on negative dims or null pointers for a
// positive-sized problem. seq == 0 or dim == 0 is a no-op success. Returns
// FERRIC_CUDA_ERR_DEVICE on CUDA runtime failure.
FerricCudaStatus ferric_cuda_attention_f32(int seq, int dim,
                                           const float* q_host,
                                           const float* k_host,
                                           const float* v_host,
                                           float* out_host);

#ifdef __cplusplus
}
#endif

#endif  // FERRIC_CUDA_ATTENTION_H_
