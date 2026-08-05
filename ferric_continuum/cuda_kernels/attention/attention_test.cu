// Golden test for ferric_cuda_attention_f32 vs a CPU reference:
//   out = softmax(Q Kᵀ / sqrt(dim)) V, single head. Requires a GPU at runtime.

#include "ferric/cuda/attention.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "gtest/gtest.h"

namespace {

std::vector<float> CpuAttention(int seq, int dim, const std::vector<float>& q,
                                const std::vector<float>& k,
                                const std::vector<float>& v) {
  const float scale = 1.0f / std::sqrt(static_cast<float>(dim));
  std::vector<float> out(static_cast<std::size_t>(seq) * dim, 0.0f);
  std::vector<float> scores(seq);
  for (int i = 0; i < seq; ++i) {
    // scores[j] = scale * <q_i, k_j>
    float max_v = -std::numeric_limits<float>::infinity();
    for (int j = 0; j < seq; ++j) {
      float dot = 0.0f;
      for (int d = 0; d < dim; ++d) {
        dot += q[i * dim + d] * k[j * dim + d];
      }
      scores[j] = dot * scale;
      max_v = std::max(max_v, scores[j]);
    }
    float sum = 0.0f;
    for (int j = 0; j < seq; ++j) {
      scores[j] = std::exp(scores[j] - max_v);
      sum += scores[j];
    }
    for (int j = 0; j < seq; ++j) {
      scores[j] /= sum;
    }
    for (int d = 0; d < dim; ++d) {
      float acc = 0.0f;
      for (int j = 0; j < seq; ++j) {
        acc += scores[j] * v[j * dim + d];
      }
      out[i * dim + d] = acc;
    }
  }
  return out;
}

std::vector<float> RandomMatrix(int rows, int cols, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> m(static_cast<std::size_t>(rows) * cols);
  for (auto& val : m) {
    val = dist(rng);
  }
  return m;
}

void ExpectClose(const std::vector<float>& got, const std::vector<float>& want) {
  ASSERT_EQ(got.size(), want.size());
  for (std::size_t i = 0; i < want.size(); ++i) {
    const float tol = 1e-5f + 1e-4f * std::fabs(want[i]);
    EXPECT_NEAR(got[i], want[i], tol) << "at index " << i;
  }
}

TEST(AttentionTest, MatchesCpu) {
  const int seq = 20, dim = 24;
  const auto q = RandomMatrix(seq, dim, 11);
  const auto k = RandomMatrix(seq, dim, 12);
  const auto v = RandomMatrix(seq, dim, 13);
  std::vector<float> out(static_cast<std::size_t>(seq) * dim, 0.0f);

  ASSERT_EQ(ferric_cuda_attention_f32(seq, dim, q.data(), k.data(), v.data(),
                                      out.data()),
            FERRIC_CUDA_OK);
  ExpectClose(out, CpuAttention(seq, dim, q, k, v));
}

TEST(AttentionTest, RejectsNullForNonEmpty) {
  EXPECT_EQ(ferric_cuda_attention_f32(2, 2, nullptr, nullptr, nullptr, nullptr),
            FERRIC_CUDA_ERR_INVALID_ARG);
}

TEST(AttentionTest, ZeroSizeIsNoop) {
  EXPECT_EQ(ferric_cuda_attention_f32(0, 4, nullptr, nullptr, nullptr, nullptr),
            FERRIC_CUDA_OK);
}

}  // namespace
