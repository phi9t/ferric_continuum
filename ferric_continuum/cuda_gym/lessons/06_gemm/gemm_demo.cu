// Lesson 06 demo: GEMM over the shared cuda_kernels C ABI, timed.

#include <chrono>
#include <random>
#include <vector>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"

#include "ferric/cuda/gemm.h"

int main() {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  LOG(INFO) << "=== Lesson 06: GEMM (cuda_kernels) ===";

  const int m = 256, n = 256, k = 256;
  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  std::vector<float> a(static_cast<std::size_t>(m) * k);
  std::vector<float> b(static_cast<std::size_t>(k) * n);
  std::vector<float> c(static_cast<std::size_t>(m) * n, 0.0F);
  for (auto& v : a) v = dist(rng);
  for (auto& v : b) v = dist(rng);

  auto time_call = [&](auto fn, const char* label) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    const FerricCudaStatus st = fn(m, n, k, a.data(), b.data(), c.data());
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    LOG(INFO) << absl::StrCat(label, ": status=", static_cast<int>(st),
                              " C[0]=", c[0], " wall_ms=", ms);
  };

  time_call(ferric_cuda_gemm_f32, "naive");
  time_call(ferric_cuda_gemm_tiled_f32, "tiled");
  return 0;
}
