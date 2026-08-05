#include "ferric_continuum/cuda_gym/lessons/02_memory/memory.hh"

#include <vector>

#include "gtest/gtest.h"

namespace ferric_continuum::cuda_gym::memory {
namespace {

TEST(MemoryTest, RoundTripIsBitExact) {
  const std::vector<float> in = {1.5F, -2.25F, 3.125F, 0.0F, 42.0F};
  const std::vector<float> out = RoundTrip(in);
  ASSERT_EQ(out.size(), in.size());
  for (std::size_t i = 0; i < in.size(); ++i) {
    // Plain copy: expect bit-exact equality.
    EXPECT_EQ(out[i], in[i]);
  }
}

TEST(MemoryTest, RoundTripScaled) {
  const std::vector<float> in = {1.0F, 2.0F, 3.0F};
  const std::vector<float> out = RoundTripScaled(in, 2.0F);
  const std::vector<float> expected = {2.0F, 4.0F, 6.0F};
  ASSERT_EQ(out.size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(out[i], expected[i]);
  }
}

TEST(MemoryTest, EmptyRoundTrip) {
  EXPECT_TRUE(RoundTrip({}).empty());
}

}  // namespace
}  // namespace ferric_continuum::cuda_gym::memory
