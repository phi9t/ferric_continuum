#pragma once

#include <string>

namespace ferric::foundation {

/// Simple 2D Point class demonstrating value semantics
/// When you copy a Point, you get an independent copy
class Point {
 public:
  Point(double x, double y);

  // Compiler-generated copy operations work correctly (value semantics)
  Point(const Point&) = default;
  Point& operator=(const Point&) = default;

  // Accessors
  double x() const { return x_; }
  double y() const { return y_; }

  // Operations create new values
  Point translate(double dx, double dy) const;
  double distance_from_origin() const;

  std::string to_string() const;

 private:
  double x_;
  double y_;
};

/// Demonstrates that modifying a copy doesn't affect the original
void demonstrate_independent_copies();

}  // namespace ferric::foundation
