#include "ferric_continuum/cuda_gym/lessons/03_indexing/indexing.hh"

#include <vector>

#include "gtest/gtest.h"

namespace ferric_continuum::cuda_gym::indexing {
namespace {

TEST(IndexingTest, IotaMapsFlatIndex) {
  // 300 is not a multiple of the 256-thread block, exercising the bounds guard.
  const std::vector<float> out = Iota(300);
  ASSERT_EQ(out.size(), 300u);
  for (std::size_t i = 0; i < out.size(); ++i) {
    EXPECT_FLOAT_EQ(out[i], static_cast<float>(i));
  }
}

TEST(IndexingTest, RowMajorIndices2D) {
  const int rows = 5, cols = 7;
  const std::vector<float> out = RowMajorIndices(rows, cols);
  ASSERT_EQ(out.size(), static_cast<std::size_t>(rows * cols));
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      EXPECT_FLOAT_EQ(out[r * cols + c], static_cast<float>(r * cols + c));
    }
  }
}

TEST(IndexingTest, EmptyIota) {
  EXPECT_TRUE(Iota(0).empty());
}

}  // namespace
}  // namespace ferric_continuum::cuda_gym::indexing
