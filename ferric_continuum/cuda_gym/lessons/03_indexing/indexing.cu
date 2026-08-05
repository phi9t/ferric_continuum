// Lesson 03 — Indexing.
//
// How threads map to data. In 1D, the global index is
// blockIdx.x*blockDim.x + threadIdx.x; a bounds guard handles the ragged last
// block. In 2D, threadIdx/blockIdx have .x and .y — by convention x → column,
// y → row for a row-major matrix.

#include "ferric_continuum/cuda_gym/lessons/03_indexing/indexing.hh"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace ferric_continuum::cuda_gym::indexing {
namespace {

void ThrowOnCudaError(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(status));
  }
}

__global__ void IotaKernel(std::size_t n, float* out) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {  // guard: last block may overrun n
    out[i] = static_cast<float>(i);
  }
}

__global__ void RowMajorIndicesKernel(int rows, int cols, float* out) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;  // x → column
  const int row = blockIdx.y * blockDim.y + threadIdx.y;  // y → row
  if (row < rows && col < cols) {
    out[row * cols + col] = static_cast<float>(row * cols + col);
  }
}

float* MallocOrThrow(std::size_t bytes) {
  float* p = nullptr;
  ThrowOnCudaError(cudaMalloc(&p, bytes), "cudaMalloc");
  return p;
}

}  // namespace

std::vector<float> Iota(std::size_t n) {
  std::vector<float> out(n);
  if (n == 0) {
    return out;
  }
  const std::size_t bytes = n * sizeof(float);
  float* d_out = MallocOrThrow(bytes);

  constexpr unsigned int kThreadsPerBlock = 256;
  const unsigned int blocks =
      static_cast<unsigned int>((n + kThreadsPerBlock - 1) / kThreadsPerBlock);
  IotaKernel<<<blocks, kThreadsPerBlock>>>(n, d_out);

  cudaError_t launch = cudaGetLastError();
  if (launch != cudaSuccess) {
    cudaFree(d_out);
    ThrowOnCudaError(launch, "IotaKernel launch");
  }
  ThrowOnCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  ThrowOnCudaError(
      cudaMemcpy(out.data(), d_out, bytes, cudaMemcpyDeviceToHost),
      "cudaMemcpy(D2H)");
  cudaFree(d_out);
  return out;
}

std::vector<float> RowMajorIndices(int rows, int cols) {
  if (rows < 0 || cols < 0) {
    throw std::invalid_argument("RowMajorIndices: negative dimension");
  }
  std::vector<float> out(static_cast<std::size_t>(rows) * cols);
  if (rows == 0 || cols == 0) {
    return out;
  }
  const std::size_t bytes = out.size() * sizeof(float);
  float* d_out = MallocOrThrow(bytes);

  const dim3 block(16, 16);
  const dim3 grid((cols + block.x - 1) / block.x,
                  (rows + block.y - 1) / block.y);
  RowMajorIndicesKernel<<<grid, block>>>(rows, cols, d_out);

  cudaError_t launch = cudaGetLastError();
  if (launch != cudaSuccess) {
    cudaFree(d_out);
    ThrowOnCudaError(launch, "RowMajorIndicesKernel launch");
  }
  ThrowOnCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  ThrowOnCudaError(
      cudaMemcpy(out.data(), d_out, bytes, cudaMemcpyDeviceToHost),
      "cudaMemcpy(D2H)");
  cudaFree(d_out);
  return out;
}

}  // namespace ferric_continuum::cuda_gym::indexing
