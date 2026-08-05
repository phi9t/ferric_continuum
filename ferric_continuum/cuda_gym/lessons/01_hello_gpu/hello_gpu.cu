// Lesson 01 — Hello GPU.
//
// The gym's entry point: query the device you are running on, then perform the
// canonical first kernel launch (SAXPY, y = a*x + y). Concepts: device
// properties, grid/block launch geometry, host<->device copies, and explicit
// error checking after a launch.

#include "ferric_continuum/cuda_gym/lessons/01_hello_gpu/hello_gpu.hh"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace ferric_continuum::cuda_gym::hello_gpu {
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

DeviceInfo QueryDevice() {
  int device_count = 0;
  ThrowOnCudaError(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
  if (device_count == 0) {
    throw std::runtime_error("QueryDevice: no CUDA device available");
  }

  int device_id = 0;
  ThrowOnCudaError(cudaGetDevice(&device_id), "cudaGetDevice");

  cudaDeviceProp props{};
  ThrowOnCudaError(cudaGetDeviceProperties(&props, device_id),
                   "cudaGetDeviceProperties");

  DeviceInfo info;
  info.device_id = device_id;
  info.name = props.name;
  info.compute_major = props.major;
  info.compute_minor = props.minor;
  info.multiprocessor_count = props.multiProcessorCount;
  info.total_global_mem_bytes = props.totalGlobalMem;
  return info;
}

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

}  // namespace ferric_continuum::cuda_gym::hello_gpu
