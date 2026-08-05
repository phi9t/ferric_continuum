// Reference solution: element-wise vector add.

#include "ferric_continuum/cuda_gym/challenges/harness/challenge_io.hh"

#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

namespace {

using ferric_continuum::cuda_gym::challenges::CaseSpec;
using ferric_continuum::cuda_gym::challenges::CheckCuda;
using ferric_continuum::cuda_gym::challenges::CheckCudaKernel;
using ferric_continuum::cuda_gym::challenges::PrintResult;
using ferric_continuum::cuda_gym::challenges::RandomInput;
using ferric_continuum::cuda_gym::challenges::TimeMs;

__global__ void VectorAddKernel(const float* a, const float* b, float* out,
                                int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] + b[i];
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
    const auto b = RandomInput(static_cast<std::size_t>(n), seed, 2);
    std::vector<float> out(static_cast<std::size_t>(n));

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_out = nullptr;
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    if (n > 0) {
      CheckCuda(cudaMalloc(&d_a, bytes), "cudaMalloc(a)");
      CheckCuda(cudaMalloc(&d_b, bytes), "cudaMalloc(b)");
      CheckCuda(cudaMalloc(&d_out, bytes), "cudaMalloc(out)");
      CheckCuda(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice),
                "H2D(a)");
      CheckCuda(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice),
                "H2D(b)");
    }

    const double ms = TimeMs([&] {
      if (n > 0) {
        constexpr int kThreads = 256;
        const int blocks = (n + kThreads - 1) / kThreads;
        VectorAddKernel<<<blocks, kThreads>>>(d_a, d_b, d_out, n);
        CheckCudaKernel("VectorAddKernel");
        CheckCuda(cudaMemcpy(out.data(), d_out, bytes, cudaMemcpyDeviceToHost),
                  "D2H(out)");
      }
    });

    if (d_a) CheckCuda(cudaFree(d_a), "cudaFree(a)");
    if (d_b) CheckCuda(cudaFree(d_b), "cudaFree(b)");
    if (d_out) CheckCuda(cudaFree(d_out), "cudaFree(out)");

    PrintResult(0, {n}, out, ms);
    return 0;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "vector_add reference failed: %s\n", ex.what());
    PrintResult(2, {}, {}, 0.0);
    return 0;
  }
}
