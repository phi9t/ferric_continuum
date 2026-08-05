// Lesson 04 demo: sum a large array on the GPU and compare to the host sum.

#include <numeric>
#include <vector>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"

#include "ferric_continuum/cuda_gym/lessons/04_reduction/reduction.hh"

namespace red = ferric_continuum::cuda_gym::reduction;

int main() {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  LOG(INFO) << "=== Lesson 04: Reduction ===";

  const std::size_t n = 1 << 20;  // ~1M elements
  std::vector<float> data(n, 1.0F);
  const float gpu_sum = red::Sum(data);
  const double host_sum = std::accumulate(data.begin(), data.end(), 0.0);

  LOG(INFO) << absl::StrCat("n=", n, " gpu_sum=", gpu_sum,
                            " host_sum=", host_sum);
  return 0;
}
