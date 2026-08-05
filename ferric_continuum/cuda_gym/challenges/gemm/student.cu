// STUDENT STUB — implement C = A * B (row-major FP32).

#include "ferric_continuum/cuda_gym/challenges/harness/challenge_io.hh"

#include <cstdio>
#include <vector>

namespace {

using ferric_continuum::cuda_gym::challenges::CaseSpec;
using ferric_continuum::cuda_gym::challenges::PrintResult;
using ferric_continuum::cuda_gym::challenges::RandomInput;

// TODO: GEMM kernel.

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s '<json>'\n", argv[0]);
    return 2;
  }
  const CaseSpec spec(argv[1]);
  const int m = spec.GetInt("m");
  const int n = spec.GetInt("n");
  const int k = spec.GetInt("k");
  const unsigned seed = static_cast<unsigned>(spec.GetInt("seed"));
  if (m < 0 || n < 0 || k < 0) {
    PrintResult(1, {}, {}, 0.0);
    return 0;
  }
  (void)RandomInput(static_cast<std::size_t>(m) * k, seed, 1);
  (void)RandomInput(static_cast<std::size_t>(k) * n, seed, 2);
  std::vector<float> c(static_cast<std::size_t>(m) * n, 0.0f);
  PrintResult(0, {m, n}, c, 0.0);
  return 0;
}
