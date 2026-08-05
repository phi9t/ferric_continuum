// Numerically-stable row softmax behind the ferric_cuda_softmax_f32 C ABI.
//
// One thread block processes one row of the (rows x cols) matrix. Threads
// cooperatively find the row max, then the sum of exp(x - max), via shared-memory
// reductions, and finally write the normalized probabilities.

#include "ferric/cuda/softmax.h"

#include <cuda_runtime.h>

#include <cfloat>
#include <cstddef>

#include "ferric_continuum/cuda_kernels/common/cuda_check.hh"
#include "ferric_continuum/cuda_kernels/common/device_buffer.hh"

namespace ferric_continuum::cuda_kernels {
namespace {

constexpr int kThreads = 256;

// Block-wide reduction using op over `shared`; result ends up in shared[0].
template <typename Op>
__device__ void BlockReduce(float* shared, Op op) {
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] =
          op(shared[threadIdx.x], shared[threadIdx.x + stride]);
    }
    __syncthreads();
  }
}

__global__ void SoftmaxRowKernel(int cols, const float* x, float* out) {
  extern __shared__ float scratch[];
  const int row = blockIdx.x;
  const float* row_in = x + static_cast<std::size_t>(row) * cols;
  float* row_out = out + static_cast<std::size_t>(row) * cols;

  // Row max.
  float local_max = -FLT_MAX;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    local_max = fmaxf(local_max, row_in[i]);
  }
  scratch[threadIdx.x] = local_max;
  __syncthreads();
  BlockReduce(scratch, [] __device__(float a, float b) { return fmaxf(a, b); });
  const float row_max = scratch[0];
  __syncthreads();

  // Row sum of exp(x - max).
  float local_sum = 0.0f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    local_sum += expf(row_in[i] - row_max);
  }
  scratch[threadIdx.x] = local_sum;
  __syncthreads();
  BlockReduce(scratch, [] __device__(float a, float b) { return a + b; });
  const float row_sum = scratch[0];
  const float inv_sum = 1.0f / row_sum;

  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    row_out[i] = expf(row_in[i] - row_max) * inv_sum;
  }
}

}  // namespace
}  // namespace ferric_continuum::cuda_kernels

extern "C" FerricCudaStatus ferric_cuda_softmax_f32(int rows, int cols,
                                                    const float* x_host,
                                                    float* out_host) {
  using ferric_continuum::cuda_kernels::DeviceBuffer;
  using ferric_continuum::cuda_kernels::kThreads;
  using ferric_continuum::cuda_kernels::SoftmaxRowKernel;

  if (rows < 0 || cols < 0) {
    return FERRIC_CUDA_ERR_INVALID_ARG;
  }
  if (rows == 0 || cols == 0) {
    return FERRIC_CUDA_OK;  // no-op
  }
  if (x_host == nullptr || out_host == nullptr) {
    return FERRIC_CUDA_ERR_INVALID_ARG;
  }

  const std::size_t elems = static_cast<std::size_t>(rows) * cols;
  DeviceBuffer<float> d_x(elems);
  DeviceBuffer<float> d_out(elems);
  if (!d_x.ok() || !d_out.ok()) {
    return FERRIC_CUDA_ERR_DEVICE;
  }

  FERRIC_CUDA_CHECK(
      cudaMemcpy(d_x.data(), x_host, d_x.bytes(), cudaMemcpyHostToDevice));

  // Launch a power-of-two thread count so the shared-memory reduction halves
  // cleanly. Threads beyond `cols` simply contribute the reduction identity
  // (-FLT_MAX for max, 0 for sum) because the strided loops skip them.
  int threads = 1;
  while (threads < cols && threads < kThreads) {
    threads <<= 1;
  }
  const std::size_t shared_bytes = threads * sizeof(float);
  SoftmaxRowKernel<<<rows, threads, shared_bytes>>>(cols, d_x.data(),
                                                    d_out.data());
  FERRIC_CUDA_CHECK_KERNEL();

  FERRIC_CUDA_CHECK(cudaMemcpy(out_host, d_out.data(), d_out.bytes(),
                               cudaMemcpyDeviceToHost));
  return FERRIC_CUDA_OK;
}
