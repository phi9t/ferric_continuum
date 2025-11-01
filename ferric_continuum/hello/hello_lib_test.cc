#include "hello_lib.hh"

#include <gtest/gtest.h>

namespace ferric {
namespace {

TEST(HelloLibTest, GenerateGreeting) {
  EXPECT_EQ(generate_greeting("World"),
            "Hello, World! Welcome to Ferric Continuum (C++ Edition with Abseil)");
  EXPECT_EQ(generate_greeting("Ferric"),
            "Hello, Ferric! Welcome to Ferric Continuum (C++ Edition with Abseil)");
}

TEST(HelloLibTest, Fibonacci) {
  EXPECT_EQ(fibonacci(0), 0);
  EXPECT_EQ(fibonacci(1), 1);
  EXPECT_EQ(fibonacci(2), 1);
  EXPECT_EQ(fibonacci(3), 2);
  EXPECT_EQ(fibonacci(4), 3);
  EXPECT_EQ(fibonacci(5), 5);
  EXPECT_EQ(fibonacci(10), 55);
  EXPECT_EQ(fibonacci(20), 6765);
}

TEST(HelloLibTest, IsPrime) {
  EXPECT_FALSE(is_prime(0));
  EXPECT_FALSE(is_prime(1));
  EXPECT_TRUE(is_prime(2));
  EXPECT_TRUE(is_prime(3));
  EXPECT_FALSE(is_prime(4));
  EXPECT_TRUE(is_prime(5));
  EXPECT_FALSE(is_prime(9));
  EXPECT_TRUE(is_prime(97));
  EXPECT_FALSE(is_prime(100));
}

TEST(HelloLibTest, PrimesUpTo) {
  auto primes = primes_up_to(20);
  std::vector<uint64_t> expected = {2, 3, 5, 7, 11, 13, 17, 19};
  EXPECT_EQ(primes, expected);
}

TEST(HelloLibTest, FormatNumberList) {
  std::vector<uint64_t> numbers = {2, 3, 5, 7, 11};
  EXPECT_EQ(format_number_list(numbers), "2, 3, 5, 7, 11");

  std::vector<uint64_t> empty;
  EXPECT_EQ(format_number_list(empty), "");

  std::vector<uint64_t> single = {42};
  EXPECT_EQ(format_number_list(single), "42");
}

}  // namespace
}  // namespace ferric
