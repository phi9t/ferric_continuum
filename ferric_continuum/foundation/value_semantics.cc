#include "value_semantics.hh"

#include "absl/strings/str_format.h"

#include <cmath>

namespace ferric::foundation {

Point::Point(double x, double y) : x_(x), y_(y) {}

Point Point::translate(double dx, double dy) const {
  // Returns a new Point, doesn't modify this one
  return Point(x_ + dx, y_ + dy);
}

double Point::distance_from_origin() const {
  return std::sqrt(x_ * x_ + y_ * y_);
}

std::string Point::to_string() const {
  return absl::StrFormat("Point(%.2f, %.2f)", x_, y_);
}

void demonstrate_independent_copies() {
  // This function exists just to show the concept
  // See the demo and test files for actual usage
}

}  // namespace ferric::foundation
