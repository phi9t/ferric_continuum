#include "parameter_passing.hh"

#include "absl/strings/str_format.h"

namespace ferric::foundation {

std::string Rectangle::to_string() const {
  return absl::StrFormat("Rectangle(%.2f x %.2f)", width, height);
}

double compute_area_by_const_ref(const Rectangle& rect) {
  // Can't modify rect - compiler enforces this
  return rect.area();
}

double compute_area_by_value(Rectangle rect) {
  // rect is a copy - modifying it won't affect caller's object
  rect.width *= 2;  // This change is local only
  return rect.area();
}

void scale_by_reference(Rectangle& rect, double factor) {
  // Modifies the original object
  rect.width *= factor;
  rect.height *= factor;
}

void scale_by_pointer(Rectangle* rect, double factor) {
  if (rect) {  // nullptr check needed!
    rect->width *= factor;
    rect->height *= factor;
  }
}

Rectangle transform_by_rvalue(Rectangle&& rect, double scale) {
  // rect is about to be destroyed, we can safely modify it
  rect.width *= scale;
  rect.height *= scale;
  return rect;  // Move return
}

}  // namespace ferric::foundation
