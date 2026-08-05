// STUDENT STUB — implement row-wise stable softmax.

#include "ferric_continuum/cuda_gym/challenges/harness/challenge_io.hh"

#include <cstdio>
#include <vector>

namespace {

using ferric_continuum::cuda_gym::challenges::CaseSpec;
using ferric_continuum::cuda_gym::challenges::PrintResult;
using ferric_continuum::cuda_gym::challenges::RandomInput;

// TODO: softmax kernel.

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s '<json>'\n", argv[0]);
    return 2;
  }
  const CaseSpec spec(argv[1]);
  const int rows = spec.GetInt("rows");
  const int cols = spec.GetInt("cols");
  const unsigned seed = static_cast<unsigned>(spec.GetInt("seed"));
  if (rows < 0 || cols < 0) {
    PrintResult(1, {}, {}, 0.0);
    return 0;
  }
  (void)RandomInput(static_cast<std::size_t>(rows) * cols, seed, 1);
  std::vector<float> out(static_cast<std::size_t>(rows) * cols, 0.0f);
  PrintResult(0, {rows, cols}, out, 0.0);
  return 0;
}
