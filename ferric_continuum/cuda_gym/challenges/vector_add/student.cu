// STUDENT STUB — fill in the kernel + launch to pass :grade.
// See description.md for the I/O contract.

#include "ferric_continuum/cuda_gym/challenges/harness/challenge_io.hh"

#include <cuda_runtime.h>

#include <cstdio>
#include <string>
#include <vector>

namespace {

using ferric_continuum::cuda_gym::challenges::CaseSpec;
using ferric_continuum::cuda_gym::challenges::PrintResult;
using ferric_continuum::cuda_gym::challenges::RandomInput;
using ferric_continuum::cuda_gym::challenges::TimeMs;

// TODO: write a __global__ kernel that computes out[i] = a[i] + b[i].

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s '<json>'\n", argv[0]);
    return 2;
  }
  const CaseSpec spec(argv[1]);
  const int n = spec.GetInt("n");
  const unsigned seed = static_cast<unsigned>(spec.GetInt("seed"));

  if (n < 0) {
    PrintResult(1, {}, {}, 0.0);
    return 0;
  }

  const auto a = RandomInput(static_cast<std::size_t>(n), seed, 1);
  const auto b = RandomInput(static_cast<std::size_t>(n), seed, 2);
  std::vector<float> out(static_cast<std::size_t>(n), 0.0f);

  // TODO: cudaMalloc, H2D, launch, D2H. Leave zeros until then so grade fails.
  (void)a;
  (void)b;

  PrintResult(0, {n}, out, 0.0);
  return 0;
}
