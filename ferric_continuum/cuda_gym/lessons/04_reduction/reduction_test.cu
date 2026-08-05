#include "ferric_continuum/cuda_gym/lessons/04_reduction/reduction.hh"

#include <cmath>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"

namespace ferric_continuum::cuda_gym::reduction {
namespace {

TEST(ReductionTest, SumOfOnes) {
  const std::size_t n = 100000;
  const std::vector<float> data(n, 1.0F);
  EXPECT_FLOAT_EQ(Sum(data), static_cast<float>(n));
}

TEST(ReductionTest, SumMatchesHost) {
  const std::size_t n = 1 << 16;
  std::vector<float> data(n);
  for (std::size_t i = 0; i < n; ++i) {
    data[i] = static_cast<float>((i % 7) - 3) * 0.5F;
  }
  const double host = std::accumulate(data.begin(), data.end(), 0.0);
  // fp32 reduction of 64K terms: allow a small relative tolerance.
  EXPECT_NEAR(Sum(data), static_cast<float>(host),
              1e-5F + 1e-4F * std::fabs(static_cast<float>(host)));
}

TEST(ReductionTest, EmptyIsZero) {
  EXPECT_FLOAT_EQ(Sum({}), 0.0F);
}

TEST(ReductionTest, SingleElement) {
  EXPECT_FLOAT_EQ(Sum({42.0F}), 42.0F);
}

}  // namespace
}  // namespace ferric_continuum::cuda_gym::reduction
