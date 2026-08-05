// Lesson 07 test: shared softmax kernel produces a valid probability row.

#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "ferric/cuda/softmax.h"

namespace {

TEST(SoftmaxLessonTest, RowSumsToOneAndOrdered) {
  const int rows = 1, cols = 4;
  const std::vector<float> x = {1, 2, 3, 4};
  std::vector<float> out(x.size(), 0.0F);

  ASSERT_EQ(ferric_cuda_softmax_f32(rows, cols, x.data(), out.data()),
            FERRIC_CUDA_OK);

  float sum = 0.0F;
  for (float p : out) {
    EXPECT_GE(p, 0.0F);
    sum += p;
  }
  EXPECT_NEAR(sum, 1.0F, 1e-5F);
  // Monotonic input -> monotonic probabilities.
  EXPECT_LT(out[0], out[1]);
  EXPECT_LT(out[1], out[2]);
  EXPECT_LT(out[2], out[3]);
}

}  // namespace
