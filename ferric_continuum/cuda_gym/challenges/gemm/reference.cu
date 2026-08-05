// Reference GEMM: delegates to shared cuda_kernels C ABI.

#include "ferric/cuda/gemm.h"
#include "ferric_continuum/cuda_gym/challenges/harness/challenge_io.hh"

#include <cstdio>
#include <vector>

namespace {

using ferric_continuum::cuda_gym::challenges::CaseSpec;
using ferric_continuum::cuda_gym::challenges::PrintResult;
using ferric_continuum::cuda_gym::challenges::RandomInput;
using ferric_continuum::cuda_gym::challenges::TimeMs;

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

  const auto a =
      RandomInput(static_cast<std::size_t>(m) * static_cast<std::size_t>(k),
                  seed, 1);
  const auto b =
      RandomInput(static_cast<std::size_t>(k) * static_cast<std::size_t>(n),
                  seed, 2);
  std::vector<float> c(static_cast<std::size_t>(m) * static_cast<std::size_t>(n),
                       0.0f);

  int status = FERRIC_CUDA_OK;
  const double ms = TimeMs([&] {
    status = ferric_cuda_gemm_tiled_f32(m, n, k, a.data(), b.data(), c.data());
  });

  PrintResult(status, {m, n}, c, ms);
  return 0;
}
