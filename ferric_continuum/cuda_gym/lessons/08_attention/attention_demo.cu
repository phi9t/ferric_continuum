// Lesson 08 demo: single-head attention over the shared cuda_kernels C ABI.

#include <random>
#include <vector>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"

#include "ferric/cuda/attention.h"

int main() {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  LOG(INFO) << "=== Lesson 08: Attention (cuda_kernels) ===";

  const int seq = 4, dim = 8;
  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  std::vector<float> q(static_cast<std::size_t>(seq) * dim);
  std::vector<float> k(q.size());
  std::vector<float> v(q.size());
  std::vector<float> out(q.size(), 0.0F);
  for (auto& x : q) x = dist(rng);
  for (auto& x : k) x = dist(rng);
  for (auto& x : v) x = dist(rng);

  const FerricCudaStatus st = ferric_cuda_attention_f32(
      seq, dim, q.data(), k.data(), v.data(), out.data());
  LOG(INFO) << absl::StrCat("status=", static_cast<int>(st), " seq=", seq,
                            " dim=", dim);

  for (int i = 0; i < seq; ++i) {
    LOG(INFO) << absl::StrCat("out[", i, ",0]=", out[i * dim + 0],
                              " out[", i, ",1]=", out[i * dim + 1]);
  }
  return 0;
}
