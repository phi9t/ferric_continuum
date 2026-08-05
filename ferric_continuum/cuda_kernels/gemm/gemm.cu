// FP32 GEMM kernels (naive + tiled) behind the ferric_cuda_gemm_f32 C ABI.
//
// C = A * B, row-major, A:(m x k) B:(k x n) C:(m x n). The host wrappers own
// device memory via DeviceBuffer and report failures through FerricCudaStatus.

#include "ferric/cuda/gemm.h"

#include <cuda_runtime.h>

#include <cstddef>

#include "ferric_continuum/cuda_kernels/common/cuda_check.hh"
#include "ferric_continuum/cuda_kernels/common/device_buffer.hh"

namespace ferric_continuum::cuda_kernels {
namespace {

constexpr int kTile = 16;

// One thread per output element C[row, col].
__global__ void GemmNaiveKernel(int m, int n, int k, const float* a,
                                 const float* b, float* c) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) {
    return;
  }
  float acc = 0.0f;
  for (int i = 0; i < k; ++i) {
    acc += a[row * k + i] * b[i * n + col];
  }
  c[row * n + col] = acc;
}

// Tiled GEMM: each block cooperatively loads kTile x kTile tiles of A and B into
// shared memory and accumulates the partial products.
__global__ void GemmTiledKernel(int m, int n, int k, const float* a,
                                 const float* b, float* c) {
  __shared__ float a_tile[kTile][kTile];
  __shared__ float b_tile[kTile][kTile];

  const int row = blockIdx.y * kTile + threadIdx.y;
  const int col = blockIdx.x * kTile + threadIdx.x;

  float acc = 0.0f;
  const int num_tiles = (k + kTile - 1) / kTile;
  for (int t = 0; t < num_tiles; ++t) {
    const int a_col = t * kTile + threadIdx.x;
    const int b_row = t * kTile + threadIdx.y;

    a_tile[threadIdx.y][threadIdx.x] =
        (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
    b_tile[threadIdx.y][threadIdx.x] =
        (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;
    __syncthreads();

    for (int i = 0; i < kTile; ++i) {
      acc += a_tile[threadIdx.y][i] * b_tile[i][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < m && col < n) {
    c[row * n + col] = acc;
  }
}

// Validates args and returns true iff the problem is a no-op (any dim == 0).
// Sets *status on invalid args.
bool ValidateGemm(int m, int n, int k, const float* a, const float* b,
                  const float* c, FerricCudaStatus* status) {
  if (m < 0 || n < 0 || k < 0) {
    *status = FERRIC_CUDA_ERR_INVALID_ARG;
    return false;
  }
  if (m == 0 || n == 0 || k == 0) {
    *status = FERRIC_CUDA_OK;
    return true;  // no-op
  }
  if (a == nullptr || b == nullptr || c == nullptr) {
    *status = FERRIC_CUDA_ERR_INVALID_ARG;
    return false;
  }
  *status = FERRIC_CUDA_OK;
  return false;
}

FerricCudaStatus RunGemm(int m, int n, int k, const float* a_host,
                         const float* b_host, float* c_host, bool tiled) {
  FerricCudaStatus status = FERRIC_CUDA_OK;
  const bool noop = ValidateGemm(m, n, k, a_host, b_host, c_host, &status);
  if (status != FERRIC_CUDA_OK || noop) {
    return status;
  }

  const std::size_t a_elems = static_cast<std::size_t>(m) * k;
  const std::size_t b_elems = static_cast<std::size_t>(k) * n;
  const std::size_t c_elems = static_cast<std::size_t>(m) * n;

  DeviceBuffer<float> d_a(a_elems);
  DeviceBuffer<float> d_b(b_elems);
  DeviceBuffer<float> d_c(c_elems);
  if (!d_a.ok() || !d_b.ok() || !d_c.ok()) {
    return FERRIC_CUDA_ERR_DEVICE;
  }

  FERRIC_CUDA_CHECK(cudaMemcpy(d_a.data(), a_host, d_a.bytes(),
                               cudaMemcpyHostToDevice));
  FERRIC_CUDA_CHECK(cudaMemcpy(d_b.data(), b_host, d_b.bytes(),
                               cudaMemcpyHostToDevice));

  const dim3 block(kTile, kTile);
  const dim3 grid((n + kTile - 1) / kTile, (m + kTile - 1) / kTile);
  if (tiled) {
    GemmTiledKernel<<<grid, block>>>(m, n, k, d_a.data(), d_b.data(),
                                     d_c.data());
  } else {
    GemmNaiveKernel<<<grid, block>>>(m, n, k, d_a.data(), d_b.data(),
                                     d_c.data());
  }
  FERRIC_CUDA_CHECK_KERNEL();

  FERRIC_CUDA_CHECK(cudaMemcpy(c_host, d_c.data(), d_c.bytes(),
                               cudaMemcpyDeviceToHost));
  return FERRIC_CUDA_OK;
}

}  // namespace
}  // namespace ferric_continuum::cuda_kernels

extern "C" FerricCudaStatus ferric_cuda_gemm_f32(int m, int n, int k,
                                                 const float* a_host,
                                                 const float* b_host,
                                                 float* c_host) {
  return ferric_continuum::cuda_kernels::RunGemm(m, n, k, a_host, b_host, c_host,
                                                 /*tiled=*/false);
}

extern "C" FerricCudaStatus ferric_cuda_gemm_tiled_f32(int m, int n, int k,
                                                       const float* a_host,
                                                       const float* b_host,
                                                       float* c_host) {
  return ferric_continuum::cuda_kernels::RunGemm(m, n, k, a_host, b_host, c_host,
                                                 /*tiled=*/true);
}
