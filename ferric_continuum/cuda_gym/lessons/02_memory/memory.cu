// Lesson 02 — Memory.
//
// The mechanics every kernel relies on: allocate device memory, copy host→device
// (H2D), do work (or not), copy device→host (D2H), and free. A plain round-trip
// is bit-exact; scaling on the device makes the traversal observable.

#include "ferric_continuum/cuda_gym/lessons/02_memory/memory.hh"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace ferric_continuum::cuda_gym::memory {
namespace {

void ThrowOnCudaError(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(status));
  }
}

__global__ void ScaleKernel(std::size_t n, float factor, float* data) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    data[i] *= factor;
  }
}

std::vector<float> RoundTripImpl(const std::vector<float>& host_in,
                                 bool scale, float factor) {
  std::vector<float> host_out(host_in.size());
  const std::size_t n = host_in.size();
  if (n == 0) {
    return host_out;
  }

  const std::size_t bytes = n * sizeof(float);
  float* d_data = nullptr;
  ThrowOnCudaError(cudaMalloc(&d_data, bytes), "cudaMalloc");

  ThrowOnCudaError(
      cudaMemcpy(d_data, host_in.data(), bytes, cudaMemcpyHostToDevice),
      "cudaMemcpy(H2D)");

  if (scale) {
    constexpr unsigned int kThreadsPerBlock = 256;
    const unsigned int blocks = static_cast<unsigned int>(
        (n + kThreadsPerBlock - 1) / kThreadsPerBlock);
    ScaleKernel<<<blocks, kThreadsPerBlock>>>(n, factor, d_data);
    cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) {
      cudaFree(d_data);
      ThrowOnCudaError(launch, "ScaleKernel launch");
    }
    cudaError_t sync = cudaDeviceSynchronize();
    if (sync != cudaSuccess) {
      cudaFree(d_data);
      ThrowOnCudaError(sync, "cudaDeviceSynchronize");
    }
  }

  ThrowOnCudaError(
      cudaMemcpy(host_out.data(), d_data, bytes, cudaMemcpyDeviceToHost),
      "cudaMemcpy(D2H)");
  cudaFree(d_data);
  return host_out;
}

}  // namespace

std::vector<float> RoundTrip(const std::vector<float>& host_in) {
  return RoundTripImpl(host_in, /*scale=*/false, 1.0f);
}

std::vector<float> RoundTripScaled(const std::vector<float>& host_in,
                                   float factor) {
  return RoundTripImpl(host_in, /*scale=*/true, factor);
}

}  // namespace ferric_continuum::cuda_gym::memory
