#include "ferric_continuum/cuda_gym/lessons/05_shared_memory/shared_memory.hh"

#include <cmath>
#include <random>
#include <vector>

#include "gtest/gtest.h"

namespace ferric_continuum::cuda_gym::shared_memory {
namespace {

std::vector<float> CpuMatVec(int rows, int cols, const std::vector<float>& m,
                             const std::vector<float>& x) {
  std::vector<float> y(rows, 0.0f);
  for (int r = 0; r < rows; ++r) {
    float acc = 0.0f;
    for (int c = 0; c < cols; ++c) {
      acc += m[r * cols + c] * x[c];
    }
    y[r] = acc;
  }
  return y;
}

TEST(SharedMemoryTest, RowSums) {
  const int rows = 3, cols = 4;
  const std::vector<float> m = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const std::vector<float> x = {1, 1, 1, 1};
  const std::vector<float> y = MatVec(rows, cols, m, x);
  const std::vector<float> expected = {10, 26, 42};
  ASSERT_EQ(y.size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(y[i], expected[i]);
  }
}

TEST(SharedMemoryTest, MatchesCpuWideCols) {
  // cols larger than the tile width forces multiple tile iterations.
  const int rows = 17, cols = 300;
  std::mt19937 rng(99);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> m(static_cast<std::size_t>(rows) * cols);
  std::vector<float> x(cols);
  for (auto& v : m) v = dist(rng);
  for (auto& v : x) v = dist(rng);

  const std::vector<float> got = MatVec(rows, cols, m, x);
  const std::vector<float> want = CpuMatVec(rows, cols, m, x);
  ASSERT_EQ(got.size(), want.size());
  for (std::size_t i = 0; i < want.size(); ++i) {
    EXPECT_NEAR(got[i], want[i], 1e-5f + 1e-4f * std::fabs(want[i]));
  }
}

TEST(SharedMemoryTest, RejectsSizeMismatch) {
  EXPECT_THROW(MatVec(2, 3, {1, 2, 3}, {1, 1, 1}), std::invalid_argument);
}

}  // namespace
}  // namespace ferric_continuum::cuda_gym::shared_memory
