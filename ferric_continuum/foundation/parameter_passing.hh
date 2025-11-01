#pragma once

#include <string>

namespace ferric::foundation {

/// Simple struct to demonstrate parameter passing
struct Rectangle {
  double width;
  double height;

  double area() const { return width * height; }
  std::string to_string() const;
};

// =============================================================================
// Different Parameter Passing Strategies
// =============================================================================

/// 1. Pass by const reference - efficient for read-only access
/// Best for: Large objects you need to read but not modify
double compute_area_by_const_ref(const Rectangle& rect);

/// 2. Pass by value - creates a copy
/// Best for: Small objects, or when you need to modify a local copy
double compute_area_by_value(Rectangle rect);

/// 3. Pass by non-const reference - allows modification
/// Best for: When you need to modify the original object
void scale_by_reference(Rectangle& rect, double factor);

/// 4. Pass by pointer - nullable, can modify
/// Best for: Optional parameters or C-style APIs
void scale_by_pointer(Rectangle* rect, double factor);

/// 5. Pass by rvalue reference - takes ownership
/// Best for: Consuming temporary objects efficiently
Rectangle transform_by_rvalue(Rectangle&& rect, double scale);

}  // namespace ferric::foundation
