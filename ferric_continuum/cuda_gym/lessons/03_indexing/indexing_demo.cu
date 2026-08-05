// Lesson 03 demo: 1D iota and a 2D row-major index fill.

#include <vector>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"

#include "ferric_continuum/cuda_gym/lessons/03_indexing/indexing.hh"

namespace idx = ferric_continuum::cuda_gym::indexing;

int main() {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  LOG(INFO) << "=== Lesson 03: Indexing ===";

  // 1D: size deliberately not a multiple of the block size (256).
  const std::vector<float> flat = idx::Iota(300);
  LOG(INFO) << absl::StrCat("Iota(300): first=", flat.front(),
                            " last=", flat.back());

  // 2D: 3x4 row-major grid.
  const int rows = 3, cols = 4;
  const std::vector<float> mat = idx::RowMajorIndices(rows, cols);
  for (int r = 0; r < rows; ++r) {
    std::string line;
    for (int c = 0; c < cols; ++c) {
      absl::StrAppend(&line, mat[r * cols + c], " ");
    }
    LOG(INFO) << absl::StrCat("row ", r, ": ", line);
  }

  return 0;
}
