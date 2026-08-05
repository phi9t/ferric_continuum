// Lesson 05 demo: y = M x with shared-memory tiling of x.

#include <vector>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"

#include "ferric_continuum/cuda_gym/lessons/05_shared_memory/shared_memory.hh"

namespace sm = ferric_continuum::cuda_gym::shared_memory;

int main() {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  LOG(INFO) << "=== Lesson 05: Shared Memory ===";

  // 3x4 matrix, x = [1,1,1,1] -> y = row sums.
  const int rows = 3, cols = 4;
  const std::vector<float> m = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const std::vector<float> x = {1, 1, 1, 1};
  const std::vector<float> y = sm::MatVec(rows, cols, m, x);

  for (int r = 0; r < rows; ++r) {
    LOG(INFO) << absl::StrCat("y[", r, "] = ", y[r]);
  }
  return 0;
}
