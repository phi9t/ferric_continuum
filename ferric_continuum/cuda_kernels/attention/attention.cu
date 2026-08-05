// Single-head scaled dot-product attention behind ferric_cuda_attention_f32.
//
//   S = Q Kᵀ / sqrt(dim)   (seq x seq)
//   P = row_softmax(S)      (seq x seq)
//   out = P V               (seq x dim)
//
// One thread block handles one query row i: it computes that row of scores into
// shared memory, does a stable softmax over it, then accumulates the weighted
// sum of V rows. dim is assumed modest (teaching primitive), so scores live in
// shared memory sized to seq floats.

#include "ferric/cuda/attention.h"

#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstddef>

#include "ferric_continuum/cuda_kernels/common/cuda_check.hh"
#include "ferric_continuum/cuda_kernels/common/device_buffer.hh"

namespace ferric_continuum::cuda_kernels {
namespace {

__global__ void AttentionRowKernel(int seq, int dim, float scale,
                                    const float* q, const float* k,
                                    const float* v, float* out) {
  // scores[0..seq) holds this query row's scaled logits, then softmax weights.
  extern __shared__ float scores[];
  const int i = blockIdx.x;  // query index
  const float* q_row = q + static_cast<std::size_t>(i) * dim;

  // Each thread computes scores for a strided set of key indices j.
  for (int j = threadIdx.x; j < seq; j += blockDim.x) {
    const float* k_row = k + static_cast<std::size_t>(j) * dim;
    float dot = 0.0f;
    for (int d = 0; d < dim; ++d) {
      dot += q_row[d] * k_row[d];
    }
    scores[j] = dot * scale;
  }
  __syncthreads();

  // Stable softmax over scores[0..seq) performed by thread 0. seq is small for
  // this teaching primitive, so a single-thread pass keeps the code clear and
  // avoids a second reduction buffer.
  if (threadIdx.x == 0) {
    float max_v = -FLT_MAX;
    for (int j = 0; j < seq; ++j) {
      max_v = fmaxf(max_v, scores[j]);
    }
    float sum = 0.0f;
    for (int j = 0; j < seq; ++j) {
      const float e = expf(scores[j] - max_v);
      scores[j] = e;
      sum += e;
    }
    const float inv = 1.0f / sum;
    for (int j = 0; j < seq; ++j) {
      scores[j] *= inv;
    }
  }
  __syncthreads();

  // out[i, d] = sum_j P[i, j] * V[j, d]; threads split the dim dimension.
  float* out_row = out + static_cast<std::size_t>(i) * dim;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int j = 0; j < seq; ++j) {
      acc += scores[j] * v[static_cast<std::size_t>(j) * dim + d];
    }
    out_row[d] = acc;
  }
}

}  // namespace
}  // namespace ferric_continuum::cuda_kernels

extern "C" FerricCudaStatus ferric_cuda_attention_f32(int seq, int dim,
                                                      const float* q_host,
                                                      const float* k_host,
                                                      const float* v_host,
                                                      float* out_host) {
  using ferric_continuum::cuda_kernels::AttentionRowKernel;
  using ferric_continuum::cuda_kernels::DeviceBuffer;

  if (seq < 0 || dim < 0) {
    return FERRIC_CUDA_ERR_INVALID_ARG;
  }
  if (seq == 0 || dim == 0) {
    return FERRIC_CUDA_OK;  // no-op
  }
  if (q_host == nullptr || k_host == nullptr || v_host == nullptr ||
      out_host == nullptr) {
    return FERRIC_CUDA_ERR_INVALID_ARG;
  }

  const std::size_t qkv_elems = static_cast<std::size_t>(seq) * dim;
  DeviceBuffer<float> d_q(qkv_elems);
  DeviceBuffer<float> d_k(qkv_elems);
  DeviceBuffer<float> d_v(qkv_elems);
  DeviceBuffer<float> d_out(qkv_elems);
  if (!d_q.ok() || !d_k.ok() || !d_v.ok() || !d_out.ok()) {
    return FERRIC_CUDA_ERR_DEVICE;
  }

  FERRIC_CUDA_CHECK(
      cudaMemcpy(d_q.data(), q_host, d_q.bytes(), cudaMemcpyHostToDevice));
  FERRIC_CUDA_CHECK(
      cudaMemcpy(d_k.data(), k_host, d_k.bytes(), cudaMemcpyHostToDevice));
  FERRIC_CUDA_CHECK(
      cudaMemcpy(d_v.data(), v_host, d_v.bytes(), cudaMemcpyHostToDevice));

  const float scale = 1.0f / std::sqrt(static_cast<float>(dim));
  const int threads = seq < 256 ? (seq < dim ? dim : seq) : 256;
  const int capped_threads = threads < 256 ? threads : 256;
  const std::size_t shared_bytes = static_cast<std::size_t>(seq) * sizeof(float);
  AttentionRowKernel<<<seq, capped_threads, shared_bytes>>>(
      seq, dim, scale, d_q.data(), d_k.data(), d_v.data(), d_out.data());
  FERRIC_CUDA_CHECK_KERNEL();

  FERRIC_CUDA_CHECK(cudaMemcpy(out_host, d_out.data(), d_out.bytes(),
                               cudaMemcpyDeviceToHost));
  return FERRIC_CUDA_OK;
}
