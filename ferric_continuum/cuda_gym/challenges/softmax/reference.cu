// Reference softmax: shared cuda_kernels C ABI.

#include "ferric/cuda/softmax.h"
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
  const int rows = spec.GetInt("rows");
  const int cols = spec.GetInt("cols");
  const unsigned seed = static_cast<unsigned>(spec.GetInt("seed"));
  if (rows < 0 || cols < 0) {
    PrintResult(1, {}, {}, 0.0);
    return 0;
  }

  const auto x =
      RandomInput(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols),
                  seed, 1);
  std::vector<float> out(x.size(), 0.0f);

  int status = FERRIC_CUDA_OK;
  const double ms = TimeMs([&] {
    status = ferric_cuda_softmax_f32(rows, cols, x.data(), out.data());
  });

  PrintResult(status, {rows, cols}, out, ms);
  return 0;
}
