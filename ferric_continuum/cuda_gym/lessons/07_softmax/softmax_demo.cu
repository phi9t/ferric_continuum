// Lesson 07 demo: row softmax over the shared cuda_kernels C ABI.

#include <vector>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"

#include "ferric/cuda/softmax.h"

int main() {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  LOG(INFO) << "=== Lesson 07: Softmax (cuda_kernels) ===";

  const int rows = 2, cols = 4;
  const std::vector<float> x = {1, 2, 3, 4, 4, 3, 2, 1};
  std::vector<float> out(x.size(), 0.0F);

  const FerricCudaStatus st =
      ferric_cuda_softmax_f32(rows, cols, x.data(), out.data());
  LOG(INFO) << absl::StrCat("status=", static_cast<int>(st));

  for (int r = 0; r < rows; ++r) {
    float sum = 0.0F;
    std::string line;
    for (int c = 0; c < cols; ++c) {
      const float p = out[r * cols + c];
      sum += p;
      absl::StrAppend(&line, p, " ");
    }
    LOG(INFO) << absl::StrCat("row ", r, ": ", line, " (sum=", sum, ")");
  }
  return 0;
}
