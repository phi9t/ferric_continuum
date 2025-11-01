#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"

#include "value_semantics.hh"

namespace ff = ferric::foundation;

int main() {
  // Initialize Abseil logging
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  LOG(INFO) << "=== Value Semantics Demo ===";

  // Create original point
  ff::Point p1(3.0, 4.0);
  LOG(INFO) << "p1: " << p1.to_string();
  LOG(INFO) << "Distance from origin: " << p1.distance_from_origin();

  // Copy creates independent value
  ff::Point p2 = p1;
  LOG(INFO) << "p2 (copied from p1): " << p2.to_string();

  // Translate p2 - p1 remains unchanged
  p2 = p2.translate(2.0, 1.0);
  LOG(INFO) << "After translating p2:";
  LOG(INFO) << "  p1: " << p1.to_string() << " (unchanged!)";
  LOG(INFO) << "  p2: " << p2.to_string() << " (translated)";

  // Assignment also creates independent copy
  ff::Point p3(0.0, 0.0);
  p3 = p1;
  LOG(INFO) << "p3 assigned from p1: " << p3.to_string();

  LOG(INFO) << "Key Point: Each object is an independent value.";
  LOG(INFO) << "Modifying a copy doesn't affect the original.";

  return 0;
}
