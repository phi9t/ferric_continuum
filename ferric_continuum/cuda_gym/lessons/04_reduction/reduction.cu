// Lesson 04 — Reduction.
//
// Summing an array is the archetypal parallel reduction. Each block loads a
// slice into shared memory and reduces it with a logarithmic tree
// (halving stride each step, __syncthreads between steps). Thread 0 of each
// block then atomicAdds its partial sum into a single global accumulator,
// combining the blocks without a second launch.

#include "ferric_continuum/cuda_gym/lessons/04_reduction/reduction.hh"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace ferric_continuum::cuda_gym::reduction {
namespace {

constexpr unsigned int kThreadsPerBlock = 256;

void ThrowOnCudaError(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(status));
  }
}

__global__ void SumKernel(std::size_t n, const float* in, float* out) {
  __shared__ float scratch[kThreadsPerBlock];
  const std::size_t gid =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  scratch[threadIdx.x] = (gid < n) ? in[gid] : 0.0f;
  __syncthreads();

  // Tree reduction within the block (blockDim.x is a power of two).
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicAdd(out, scratch[0]);
  }
}

}  // namespace

float Sum(const std::vector<float>& host_in) {
  const std::size_t n = host_in.size();
  if (n == 0) {
    return 0.0f;
  }

  const std::size_t bytes = n * sizeof(float);
  float* d_in = nullptr;
  float* d_out = nullptr;
  ThrowOnCudaError(cudaMalloc(&d_in, bytes), "cudaMalloc(d_in)");
  cudaError_t malloc_out = cudaMalloc(&d_out, sizeof(float));
  if (malloc_out != cudaSuccess) {
    cudaFree(d_in);
    ThrowOnCudaError(malloc_out, "cudaMalloc(d_out)");
  }

  ThrowOnCudaError(cudaMemcpy(d_in, host_in.data(), bytes,
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(H2D)");
  ThrowOnCudaError(cudaMemset(d_out, 0, sizeof(float)), "cudaMemset(d_out)");

  const unsigned int blocks =
      static_cast<unsigned int>((n + kThreadsPerBlock - 1) / kThreadsPerBlock);
  SumKernel<<<blocks, kThreadsPerBlock>>>(n, d_in, d_out);

  cudaError_t launch = cudaGetLastError();
  if (launch != cudaSuccess) {
    cudaFree(d_in);
    cudaFree(d_out);
    ThrowOnCudaError(launch, "SumKernel launch");
  }
  ThrowOnCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  float result = 0.0f;
  ThrowOnCudaError(
      cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost),
      "cudaMemcpy(D2H)");

  cudaFree(d_in);
  cudaFree(d_out);
  return result;
}

}  // namespace ferric_continuum::cuda_gym::reduction
