#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"

#include "hello_lib.hh"

int main() {
  // Initialize Abseil logging
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  LOG(INFO) << ferric::generate_greeting("World");

  // Demonstrate fibonacci
  LOG(INFO) << "Fibonacci sequence (first 10 numbers):";
  for (int i = 0; i < 10; ++i) {
    LOG(INFO) << "fib(" << i << ") = " << ferric::fibonacci(i);
  }

  // Demonstrate prime checking with Abseil string formatting
  LOG(INFO) << "Prime numbers up to 50:";
  auto primes = ferric::primes_up_to(50);
  LOG(INFO) << ferric::format_number_list(primes);

  return 0;
}
