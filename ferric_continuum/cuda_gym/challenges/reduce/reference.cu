// Reference: multi-block tree reduction into a host-accumulated partials sum.

#include "ferric_continuum/cuda_gym/challenges/harness/challenge_io.hh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <vector>

namespace {

using ferric_continuum::cuda_gym::challenges::CaseSpec;
using ferric_continuum::cuda_gym::challenges::CheckCuda;
using ferric_continuum::cuda_gym::challenges::CheckCudaKernel;
using ferric_continuum::cuda_gym::challenges::PrintResult;
using ferric_continuum::cuda_gym::challenges::RandomInput;
using ferric_continuum::cuda_gym::challenges::TimeMs;

constexpr int kThreads = 256;

__global__ void ReduceBlockKernel(const float* in, float* partials, int n) {
  __shared__ float smem[kThreads];
  const int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0f;
  while (i < n) {
    sum += in[i];
    i += blockDim.x * gridDim.x;
  }
  smem[tid] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] += smem[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    partials[blockIdx.x] = smem[0];
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s '<json>'\n", argv[0]);
    return 2;
  }
  try {
    const CaseSpec spec(argv[1]);
    const int n = spec.GetInt("n");
    const unsigned seed = static_cast<unsigned>(spec.GetInt("seed"));
    if (n < 0) {
      PrintResult(1, {}, {}, 0.0);
      return 0;
    }

    const auto a = RandomInput(static_cast<std::size_t>(n), seed, 1);
    std::vector<float> out(1, 0.0f);

    float* d_a = nullptr;
    float* d_partials = nullptr;
    const int blocks =
        n > 0 ? std::min(1024, (n + kThreads - 1) / kThreads) : 0;
    std::vector<float> partials(static_cast<std::size_t>(blocks), 0.0f);

    if (n > 0) {
      CheckCuda(cudaMalloc(&d_a, a.size() * sizeof(float)), "cudaMalloc(a)");
      CheckCuda(cudaMalloc(&d_partials, blocks * sizeof(float)),
                "cudaMalloc(partials)");
      CheckCuda(cudaMemcpy(d_a, a.data(), a.size() * sizeof(float),
                           cudaMemcpyHostToDevice),
                "H2D(a)");
    }

    const double ms = TimeMs([&] {
      if (n == 0) {
        out[0] = 0.0f;
        return;
      }
      ReduceBlockKernel<<<blocks, kThreads>>>(d_a, d_partials, n);
      CheckCudaKernel("ReduceBlockKernel");
      CheckCuda(cudaMemcpy(partials.data(), d_partials, blocks * sizeof(float),
                           cudaMemcpyDeviceToHost),
                "D2H(partials)");
      float total = 0.0f;
      for (float p : partials) {
        total += p;
      }
      out[0] = total;
    });

    if (d_a) CheckCuda(cudaFree(d_a), "cudaFree(a)");
    if (d_partials) CheckCuda(cudaFree(d_partials), "cudaFree(partials)");

    PrintResult(0, {1}, out, ms);
    return 0;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "reduce reference failed: %s\n", ex.what());
    PrintResult(2, {}, {}, 0.0);
    return 0;
  }
}
