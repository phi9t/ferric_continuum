// SAXPY (single-precision a*x + y) CUDA kernel and host launcher.
//
// This is a minimal, self-contained example that exercises the rules_cuda
// toolchain wired up in MODULE.bazel / .bazelrc. Build it with:
//
//     bazel build --config=cuda //ferric_continuum/cuda:saxpy
//
// It intentionally keeps error handling explicit rather than relying on a
// macro so the control flow stays readable.

#include "ferric_continuum/cuda/saxpy.hh"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace ferric_continuum::cuda {
namespace {

void ThrowOnCudaError(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(status));
  }
}

__global__ void SaxpyKernel(std::size_t n, float a, const float* x, float* y) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a * x[i] + y[i];
  }
}

}  // namespace

void Saxpy(float a, const std::vector<float>& x, std::vector<float>& y) {
  if (x.size() != y.size()) {
    throw std::invalid_argument("Saxpy: x and y must have the same length");
  }
  const std::size_t n = x.size();
  if (n == 0) {
    return;
  }

  const std::size_t bytes = n * sizeof(float);
  float* d_x = nullptr;
  float* d_y = nullptr;

  ThrowOnCudaError(cudaMalloc(&d_x, bytes), "cudaMalloc(d_x)");
  cudaError_t malloc_y = cudaMalloc(&d_y, bytes);
  if (malloc_y != cudaSuccess) {
    cudaFree(d_x);
    ThrowOnCudaError(malloc_y, "cudaMalloc(d_y)");
  }

  ThrowOnCudaError(cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy(H2D x)");
  ThrowOnCudaError(cudaMemcpy(d_y, y.data(), bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy(H2D y)");

  constexpr unsigned int kThreadsPerBlock = 256;
  const unsigned int blocks =
      static_cast<unsigned int>((n + kThreadsPerBlock - 1) / kThreadsPerBlock);
  SaxpyKernel<<<blocks, kThreadsPerBlock>>>(n, a, d_x, d_y);

  ThrowOnCudaError(cudaGetLastError(), "SaxpyKernel launch");
  ThrowOnCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  ThrowOnCudaError(cudaMemcpy(y.data(), d_y, bytes, cudaMemcpyDeviceToHost),
                   "cudaMemcpy(D2H y)");

  cudaFree(d_x);
  cudaFree(d_y);
}

}  // namespace ferric_continuum::cuda
