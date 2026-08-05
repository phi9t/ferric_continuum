// Golden test for ferric_cuda_softmax_f32 vs a CPU reference (stable row
// softmax over the last dim). Requires a GPU at runtime.

#include "ferric/cuda/softmax.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "gtest/gtest.h"

namespace {

std::vector<float> CpuSoftmax(int rows, int cols,
                              const std::vector<float>& x) {
  std::vector<float> out(x.size());
  for (int r = 0; r < rows; ++r) {
    const float* row = x.data() + static_cast<std::size_t>(r) * cols;
    float max_v = -std::numeric_limits<float>::infinity();
    for (int c = 0; c < cols; ++c) {
      max_v = std::max(max_v, row[c]);
    }
    float sum = 0.0f;
    for (int c = 0; c < cols; ++c) {
      sum += std::exp(row[c] - max_v);
    }
    for (int c = 0; c < cols; ++c) {
      out[static_cast<std::size_t>(r) * cols + c] =
          std::exp(row[c] - max_v) / sum;
    }
  }
  return out;
}

std::vector<float> RandomMatrix(int rows, int cols, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
  std::vector<float> m(static_cast<std::size_t>(rows) * cols);
  for (auto& v : m) {
    v = dist(rng);
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

TEST(SoftmaxTest, MatchesCpu) {
  const int rows = 23, cols = 130;  // cols not a power of two
  const auto x = RandomMatrix(rows, cols, 7);
  std::vector<float> out(x.size(), 0.0f);

  ASSERT_EQ(ferric_cuda_softmax_f32(rows, cols, x.data(), out.data()),
            FERRIC_CUDA_OK);
  ExpectClose(out, CpuSoftmax(rows, cols, x));
}

TEST(SoftmaxTest, RowsSumToOne) {
  const int rows = 8, cols = 512;
  const auto x = RandomMatrix(rows, cols, 8);
  std::vector<float> out(x.size(), 0.0f);

  ASSERT_EQ(ferric_cuda_softmax_f32(rows, cols, x.data(), out.data()),
            FERRIC_CUDA_OK);
  for (int r = 0; r < rows; ++r) {
    float sum = 0.0f;
    for (int c = 0; c < cols; ++c) {
      sum += out[static_cast<std::size_t>(r) * cols + c];
    }
    EXPECT_NEAR(sum, 1.0f, 1e-4f);
  }
}

TEST(SoftmaxTest, RejectsNullForNonEmpty) {
  EXPECT_EQ(ferric_cuda_softmax_f32(2, 2, nullptr, nullptr),
            FERRIC_CUDA_ERR_INVALID_ARG);
}

TEST(SoftmaxTest, ZeroSizeIsNoop) {
  EXPECT_EQ(ferric_cuda_softmax_f32(0, 4, nullptr, nullptr), FERRIC_CUDA_OK);
}

}  // namespace
