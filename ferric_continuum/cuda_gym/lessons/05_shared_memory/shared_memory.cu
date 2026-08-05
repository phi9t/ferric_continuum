// Lesson 05 — Shared memory.
//
// Shared memory is fast, block-local scratch. Here y = M x: one block per row,
// with the block cooperatively staging tiles of the shared vector x into shared
// memory. Every thread in the block reuses each loaded tile, cutting global
// loads of x from (threads * cols) down to (cols). Each thread accumulates a
// strided partial dot product, then a tree reduction combines them into y[row].

#include "ferric_continuum/cuda_gym/lessons/05_shared_memory/shared_memory.hh"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace ferric_continuum::cuda_gym::shared_memory {
namespace {

constexpr int kThreads = 128;
constexpr int kTile = kThreads;  // stage x one block-width tile at a time

void ThrowOnCudaError(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(status));
  }
}

__global__ void MatVecKernel(int rows, int cols, const float* m, const float* x,
                             float* y) {
  __shared__ float x_tile[kTile];
  __shared__ float partial[kThreads];
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const float* m_row = m + static_cast<std::size_t>(row) * cols;

  float acc = 0.0f;
  for (int base = 0; base < cols; base += kTile) {
    // Cooperatively load a tile of x into shared memory.
    const int idx = base + threadIdx.x;
    if (threadIdx.x < kTile && idx < cols) {
      x_tile[threadIdx.x] = x[idx];
    }
    __syncthreads();

    const int tile_len = min(kTile, cols - base);
    for (int j = threadIdx.x; j < tile_len; j += blockDim.x) {
      acc += m_row[base + j] * x_tile[j];
    }
    __syncthreads();
  }

  // Reduce the per-thread partials to y[row].
  partial[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    y[row] = partial[0];
  }
}

}  // namespace

std::vector<float> MatVec(int rows, int cols, const std::vector<float>& matrix,
                          const std::vector<float>& x) {
  if (rows < 0 || cols < 0) {
    throw std::invalid_argument("MatVec: negative dimension");
  }
  if (matrix.size() != static_cast<std::size_t>(rows) * cols ||
      x.size() != static_cast<std::size_t>(cols)) {
    throw std::invalid_argument("MatVec: size mismatch");
  }
  std::vector<float> y(static_cast<std::size_t>(rows), 0.0f);
  if (rows == 0 || cols == 0) {
    return y;
  }

  const std::size_t m_bytes = matrix.size() * sizeof(float);
  const std::size_t x_bytes = x.size() * sizeof(float);
  const std::size_t y_bytes = y.size() * sizeof(float);
  float* d_m = nullptr;
  float* d_x = nullptr;
  float* d_y = nullptr;
  ThrowOnCudaError(cudaMalloc(&d_m, m_bytes), "cudaMalloc(d_m)");
  if (cudaMalloc(&d_x, x_bytes) != cudaSuccess) {
    cudaFree(d_m);
    throw std::runtime_error("cudaMalloc(d_x) failed");
  }
  if (cudaMalloc(&d_y, y_bytes) != cudaSuccess) {
    cudaFree(d_m);
    cudaFree(d_x);
    throw std::runtime_error("cudaMalloc(d_y) failed");
  }

  ThrowOnCudaError(
      cudaMemcpy(d_m, matrix.data(), m_bytes, cudaMemcpyHostToDevice),
      "cudaMemcpy(H2D m)");
  ThrowOnCudaError(cudaMemcpy(d_x, x.data(), x_bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy(H2D x)");

  MatVecKernel<<<rows, kThreads>>>(rows, cols, d_m, d_x, d_y);
  cudaError_t launch = cudaGetLastError();
  if (launch != cudaSuccess) {
    cudaFree(d_m);
    cudaFree(d_x);
    cudaFree(d_y);
    ThrowOnCudaError(launch, "MatVecKernel launch");
  }
  ThrowOnCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  ThrowOnCudaError(cudaMemcpy(y.data(), d_y, y_bytes, cudaMemcpyDeviceToHost),
                   "cudaMemcpy(D2H y)");
  cudaFree(d_m);
  cudaFree(d_x);
  cudaFree(d_y);
  return y;
}

}  // namespace ferric_continuum::cuda_gym::shared_memory
