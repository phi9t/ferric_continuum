#include "hello_lib.hh"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

#include <cmath>

namespace ferric {

std::string generate_greeting(absl::string_view name) {
  // Using Abseil's StrCat for efficient string concatenation
  return absl::StrCat("Hello, ", name, "! Welcome to Ferric Continuum (C++ Edition with Abseil)");
}

uint64_t fibonacci(int n) {
  if (n <= 1)
    return n;

  uint64_t a = 0, b = 1;
  for (int i = 2; i <= n; ++i) {
    uint64_t temp = a + b;
    a = b;
    b = temp;
  }
  return b;
}

bool is_prime(uint64_t n) {
  if (n < 2)
    return false;
  if (n == 2)
    return true;
  if (n % 2 == 0)
    return false;

  uint64_t limit = static_cast<uint64_t>(std::sqrt(n));
  for (uint64_t i = 3; i <= limit; i += 2) {
    if (n % i == 0)
      return false;
  }
  return true;
}

std::vector<uint64_t> primes_up_to(uint64_t n) {
  std::vector<uint64_t> primes;
  for (uint64_t i = 2; i <= n; ++i) {
    if (is_prime(i)) {
      primes.push_back(i);
    }
  }
  return primes;
}

std::string format_number_list(const std::vector<uint64_t>& numbers) {
  // Using Abseil's StrJoin for elegant list formatting
  return absl::StrJoin(numbers, ", ");
}

}  // namespace ferric
