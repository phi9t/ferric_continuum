// STUDENT STUB — implement a reduction that sums n floats into out[0].

#include "ferric_continuum/cuda_gym/challenges/harness/challenge_io.hh"

#include <cstdio>
#include <vector>

namespace {

using ferric_continuum::cuda_gym::challenges::CaseSpec;
using ferric_continuum::cuda_gym::challenges::PrintResult;
using ferric_continuum::cuda_gym::challenges::RandomInput;

// TODO: write a reduction kernel.

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
  (void)RandomInput(static_cast<std::size_t>(n), seed, 1);
  // Wrong on purpose until filled in (except empty sum = 0).
  std::vector<float> out(1, n == 0 ? 0.0f : 1.0f);
  PrintResult(0, {1}, out, 0.0);
  return 0;
}
