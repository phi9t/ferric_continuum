// Lesson 08 test: single-head attention over the shared kernel matches a CPU
// reference on a tiny QKV.

#include <algorithm>
#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "ferric/cuda/attention.h"

namespace {

std::vector<float> CpuAttention(int seq, int dim, const std::vector<float>& q,
                                const std::vector<float>& k,
                                const std::vector<float>& v) {
  const float scale = 1.0F / std::sqrt(static_cast<float>(dim));
  std::vector<float> out(static_cast<std::size_t>(seq) * dim, 0.0F);
  std::vector<float> scores(seq);
  for (int i = 0; i < seq; ++i) {
    float max_v = -1e30F;
    for (int j = 0; j < seq; ++j) {
      float dot = 0.0F;
      for (int d = 0; d < dim; ++d) dot += q[i * dim + d] * k[j * dim + d];
      scores[j] = dot * scale;
      max_v = std::max(max_v, scores[j]);
    }
    float sum = 0.0F;
    for (int j = 0; j < seq; ++j) {
      scores[j] = std::exp(scores[j] - max_v);
      sum += scores[j];
    }
    for (int d = 0; d < dim; ++d) {
      float acc = 0.0F;
      for (int j = 0; j < seq; ++j) acc += (scores[j] / sum) * v[j * dim + d];
      out[i * dim + d] = acc;
    }
  }
  return out;
}

TEST(AttentionLessonTest, MatchesCpuTiny) {
  const int seq = 3, dim = 4;
  const std::vector<float> q = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
  const std::vector<float> k = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
  const std::vector<float> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<float> out(q.size(), 0.0F);

  ASSERT_EQ(ferric_cuda_attention_f32(seq, dim, q.data(), k.data(), v.data(),
                                      out.data()),
            FERRIC_CUDA_OK);
  const std::vector<float> want = CpuAttention(seq, dim, q, k, v);
  ASSERT_EQ(out.size(), want.size());
  for (std::size_t i = 0; i < want.size(); ++i) {
    EXPECT_NEAR(out[i], want[i], 1e-5F + 1e-4F * std::fabs(want[i]));
  }
}

}  // namespace
