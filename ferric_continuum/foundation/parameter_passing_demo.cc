#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"

#include <utility>

#include "parameter_passing.hh"

namespace ff = ferric::foundation;

int main() {
  // Initialize Abseil logging
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  LOG(INFO) << "=== Parameter Passing Demo ===";

  ff::Rectangle rect{10.0, 5.0};
  LOG(INFO) << "Original: " << rect.to_string();

  // 1. Pass by const reference - read-only, no copy
  LOG(INFO) << "1. Pass by const reference:";
  double area = ff::compute_area_by_const_ref(rect);
  LOG(INFO) << "   Area: " << area;
  LOG(INFO) << "   Original unchanged: " << rect.to_string();
  LOG(INFO) << "   Use case: Efficient read-only access";

  // 2. Pass by value - creates copy
  LOG(INFO) << "2. Pass by value (creates copy):";
  area = ff::compute_area_by_value(rect);
  LOG(INFO) << "   Computed area (width doubled internally): " << area;
  LOG(INFO) << "   Original unchanged: " << rect.to_string();
  LOG(INFO) << "   Use case: Need local modifications without affecting caller";

  // 3. Pass by reference - modifies original
  LOG(INFO) << "3. Pass by non-const reference:";
  LOG(INFO) << "   Before: " << rect.to_string();
  ff::scale_by_reference(rect, 2.0);
  LOG(INFO) << "   After:  " << rect.to_string() << " (modified!)";
  LOG(INFO) << "   Use case: Need to modify the original object";

  // Reset
  rect = {10.0, 5.0};

  // 4. Pass by pointer - nullable, can modify
  LOG(INFO) << "4. Pass by pointer:";
  LOG(INFO) << "   Before: " << rect.to_string();
  ff::scale_by_pointer(&rect, 2.0);
  LOG(INFO) << "   After:  " << rect.to_string() << " (modified!)";
  ff::scale_by_pointer(nullptr, 2.0);  // Safe - nullptr check inside
  LOG(INFO) << "   nullptr is safe (checked inside function)";
  LOG(INFO) << "   Use case: Optional parameters or C-style APIs";

  // 5. Pass by rvalue reference - consumes temporary
  LOG(INFO) << "5. Pass by rvalue reference:";
  ff::Rectangle temp{5.0, 3.0};
  LOG(INFO) << "   Temp before: " << temp.to_string();
  ff::Rectangle result = ff::transform_by_rvalue(std::move(temp), 3.0);
  LOG(INFO) << "   Result: " << result.to_string();
  LOG(INFO) << "   Use case: Efficiently consuming temporary objects";

  LOG(INFO) << "Guidelines:";
  LOG(INFO) << "- const& : Default choice for read-only access";
  LOG(INFO) << "- value  : Small objects or when you need a local copy";
  LOG(INFO) << "- &      : When you need to modify the original";
  LOG(INFO) << "- *      : For optional parameters or C APIs";
  LOG(INFO) << "- &&     : For move semantics and consuming temporaries";

  return 0;
}
