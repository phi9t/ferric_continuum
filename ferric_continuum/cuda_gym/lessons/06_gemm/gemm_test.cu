// Lesson 06 test: the shared GEMM kernel matches a CPU reference. The kernel is
// covered thoroughly in cuda_kernels; this asserts the lesson wiring works.

#include <vector>

#include "gtest/gtest.h"

#include "ferric/cuda/gemm.h"

namespace {

TEST(GemmLessonTest, MatchesCpuSmall) {
  // A(2x3) * B(3x2) = C(2x2).
  const int m = 2, n = 2, k = 3;
  const std::vector<float> a = {1, 2, 3, 4, 5, 6};
  const std::vector<float> b = {7, 8, 9, 10, 11, 12};
  std::vector<float> c(4, 0.0F);

  ASSERT_EQ(ferric_cuda_gemm_tiled_f32(m, n, k, a.data(), b.data(), c.data()),
            FERRIC_CUDA_OK);

  // Row 0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
  // Row 1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
  EXPECT_FLOAT_EQ(c[0], 58.0F);
  EXPECT_FLOAT_EQ(c[1], 64.0F);
  EXPECT_FLOAT_EQ(c[2], 139.0F);
  EXPECT_FLOAT_EQ(c[3], 154.0F);
}

}  // namespace
