#include "ferric_continuum/cuda_gym/lessons/01_hello_gpu/hello_gpu.hh"

#include <vector>

#include "gtest/gtest.h"

namespace ferric_continuum::cuda_gym::hello_gpu {
namespace {

TEST(HelloGpuTest, DeviceIsQueryable) {
  const DeviceInfo info = QueryDevice();
  EXPECT_FALSE(info.name.empty());
  EXPECT_GT(info.compute_major, 0);
  EXPECT_GT(info.multiprocessor_count, 0);
  EXPECT_GT(info.total_global_mem_bytes, 0u);
}

TEST(HelloGpuTest, SaxpyComputesAxPlusY) {
  const float a = 2.0F;
  const std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F};
  std::vector<float> y = {10.0F, 20.0F, 30.0F, 40.0F};

  Saxpy(a, x, y);

  const std::vector<float> expected = {12.0F, 24.0F, 36.0F, 48.0F};
  ASSERT_EQ(y.size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(y[i], expected[i]);
  }
}

TEST(HelloGpuTest, SaxpyEmptyIsNoop) {
  std::vector<float> y;
  Saxpy(3.0F, {}, y);
  EXPECT_TRUE(y.empty());
}

TEST(HelloGpuTest, SaxpyMismatchedLengthsThrow) {
  const std::vector<float> x = {1.0F, 2.0F};
  std::vector<float> y = {1.0F};
  EXPECT_THROW(Saxpy(1.0F, x, y), std::invalid_argument);
}

}  // namespace
}  // namespace ferric_continuum::cuda_gym::hello_gpu
