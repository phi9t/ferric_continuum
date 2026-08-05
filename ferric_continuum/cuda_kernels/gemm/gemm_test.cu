// Golden test for ferric_cuda_gemm_f32 / ferric_cuda_gemm_tiled_f32 vs a CPU
// reference. Requires a GPU at runtime (tagged requires-gpu).

#include "ferric/cuda/gemm.h"

#include <cmath>
#include <random>
#include <vector>

#include "gtest/gtest.h"

namespace {

// Row-major CPU reference: C = A(m x k) * B(k x n).
std::vector<float> CpuGemm(int m, int n, int k, const std::vector<float>& a,
                           const std::vector<float>& b) {
  std::vector<float> c(static_cast<std::size_t>(m) * n, 0.0f);
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float acc = 0.0f;
      for (int i = 0; i < k; ++i) {
        acc += a[row * k + i] * b[i * n + col];
      }
      c[row * n + col] = acc;
    }
  }
  return c;
}

std::vector<float> RandomMatrix(int rows, int cols, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> m(static_cast<std::size_t>(rows) * cols);
  for (auto& v : m) {
    v = dist(rng);
  }
  return m;
}

// fp32 policy: rtol 1e-4, atol 1e-5.
void ExpectClose(const std::vector<float>& got, const std::vector<float>& want) {
  ASSERT_EQ(got.size(), want.size());
  for (std::size_t i = 0; i < want.size(); ++i) {
    const float tol = 1e-5f + 1e-4f * std::fabs(want[i]);
    EXPECT_NEAR(got[i], want[i], tol) << "at index " << i;
  }
}

TEST(GemmTest, NaiveMatchesCpu) {
  const int m = 37, n = 41, k = 29;
  const auto a = RandomMatrix(m, k, 1);
  const auto b = RandomMatrix(k, n, 2);
  std::vector<float> c(static_cast<std::size_t>(m) * n, 0.0f);

  ASSERT_EQ(ferric_cuda_gemm_f32(m, n, k, a.data(), b.data(), c.data()),
            FERRIC_CUDA_OK);
  ExpectClose(c, CpuGemm(m, n, k, a, b));
}

TEST(GemmTest, TiledMatchesCpu) {
  const int m = 64, n = 48, k = 80;
  const auto a = RandomMatrix(m, k, 3);
  const auto b = RandomMatrix(k, n, 4);
  std::vector<float> c(static_cast<std::size_t>(m) * n, 0.0f);

  ASSERT_EQ(ferric_cuda_gemm_tiled_f32(m, n, k, a.data(), b.data(), c.data()),
            FERRIC_CUDA_OK);
  ExpectClose(c, CpuGemm(m, n, k, a, b));
}

TEST(GemmTest, TiledAndNaiveAgree) {
  const int m = 33, n = 17, k = 65;  // deliberately non-multiples of the tile
  const auto a = RandomMatrix(m, k, 5);
  const auto b = RandomMatrix(k, n, 6);
  std::vector<float> c_naive(static_cast<std::size_t>(m) * n, 0.0f);
  std::vector<float> c_tiled(static_cast<std::size_t>(m) * n, 0.0f);

  ASSERT_EQ(ferric_cuda_gemm_f32(m, n, k, a.data(), b.data(), c_naive.data()),
            FERRIC_CUDA_OK);
  ASSERT_EQ(
      ferric_cuda_gemm_tiled_f32(m, n, k, a.data(), b.data(), c_tiled.data()),
      FERRIC_CUDA_OK);
  ExpectClose(c_tiled, c_naive);
}

TEST(GemmTest, RejectsNegativeDims) {
  float x = 0.0f;
  EXPECT_EQ(ferric_cuda_gemm_f32(-1, 1, 1, &x, &x, &x),
            FERRIC_CUDA_ERR_INVALID_ARG);
}

TEST(GemmTest, RejectsNullForNonEmpty) {
  EXPECT_EQ(ferric_cuda_gemm_f32(2, 2, 2, nullptr, nullptr, nullptr),
            FERRIC_CUDA_ERR_INVALID_ARG);
}

TEST(GemmTest, ZeroSizeIsNoop) {
  EXPECT_EQ(ferric_cuda_gemm_f32(0, 4, 4, nullptr, nullptr, nullptr),
            FERRIC_CUDA_OK);
}

}  // namespace
