#pragma once

#include "absl/strings/string_view.h"

#include <string>
#include <vector>

namespace ferric {

// Generate a greeting message using Abseil string utilities
std::string generate_greeting(absl::string_view name);

// Calculate fibonacci number (demonstrating simple algorithm)
uint64_t fibonacci(int n);

// Check if a number is prime
bool is_prime(uint64_t n);

// Get all prime numbers up to n
std::vector<uint64_t> primes_up_to(uint64_t n);

// Format a list of numbers as a comma-separated string using Abseil
std::string format_number_list(const std::vector<uint64_t>& numbers);

}  // namespace ferric
