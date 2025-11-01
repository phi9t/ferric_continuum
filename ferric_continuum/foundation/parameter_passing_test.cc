#include "parameter_passing.hh"

#include <gtest/gtest.h>
#include <utility>

namespace ferric::foundation {
namespace {

TEST(ParameterPassingTest, RectangleConstruction) {
  Rectangle rect{10.0, 5.0};
  EXPECT_DOUBLE_EQ(rect.width, 10.0);
  EXPECT_DOUBLE_EQ(rect.height, 5.0);
}

TEST(ParameterPassingTest, RectangleArea) {
  Rectangle rect{10.0, 5.0};
  EXPECT_DOUBLE_EQ(rect.area(), 50.0);

  Rectangle rect2{7.5, 4.0};
  EXPECT_DOUBLE_EQ(rect2.area(), 30.0);

  Rectangle rect3{0.0, 10.0};
  EXPECT_DOUBLE_EQ(rect3.area(), 0.0);
}

TEST(ParameterPassingTest, RectangleToString) {
  Rectangle rect{10.5, 5.5};
  std::string str = rect.to_string();
  EXPECT_FALSE(str.empty());
  // Should contain dimensions
  EXPECT_NE(str.find("10.5"), std::string::npos);
  EXPECT_NE(str.find("5.5"), std::string::npos);
}

TEST(ParameterPassingTest, ConstReference) {
  Rectangle rect{10.0, 5.0};
  double area = compute_area_by_const_ref(rect);

  EXPECT_DOUBLE_EQ(area, 50.0);
  EXPECT_DOUBLE_EQ(rect.width, 10.0);  // Unchanged
  EXPECT_DOUBLE_EQ(rect.height, 5.0);  // Unchanged
}

TEST(ParameterPassingTest, ByValue) {
  Rectangle rect{10.0, 5.0};
  double area = compute_area_by_value(rect);

  EXPECT_DOUBLE_EQ(area, 100.0);       // Width doubled inside
  EXPECT_DOUBLE_EQ(rect.width, 10.0);  // Original unchanged
  EXPECT_DOUBLE_EQ(rect.height, 5.0);  // Original unchanged
}

TEST(ParameterPassingTest, ByValueMultipleCalls) {
  Rectangle rect{5.0, 4.0};

  double area1 = compute_area_by_value(rect);
  double area2 = compute_area_by_value(rect);

  // Both calls should produce same result (rect unchanged)
  EXPECT_DOUBLE_EQ(area1, area2);
  EXPECT_DOUBLE_EQ(rect.width, 5.0);
}

TEST(ParameterPassingTest, ByReference) {
  Rectangle rect{10.0, 5.0};
  scale_by_reference(rect, 2.0);

  EXPECT_DOUBLE_EQ(rect.width, 20.0);
  EXPECT_DOUBLE_EQ(rect.height, 10.0);
}

TEST(ParameterPassingTest, ByReferenceMultipleTimes) {
  Rectangle rect{10.0, 5.0};
  scale_by_reference(rect, 2.0);
  scale_by_reference(rect, 0.5);

  EXPECT_DOUBLE_EQ(rect.width, 10.0);
  EXPECT_DOUBLE_EQ(rect.height, 5.0);
}

TEST(ParameterPassingTest, ByPointer) {
  Rectangle rect{10.0, 5.0};
  scale_by_pointer(&rect, 2.0);

  EXPECT_DOUBLE_EQ(rect.width, 20.0);
  EXPECT_DOUBLE_EQ(rect.height, 10.0);

  // nullptr is safe
  scale_by_pointer(nullptr, 2.0);  // Should not crash
}

TEST(ParameterPassingTest, ByPointerNullptrSafety) {
  // Multiple nullptr calls should be safe
  scale_by_pointer(nullptr, 1.0);
  scale_by_pointer(nullptr, 2.0);
  scale_by_pointer(nullptr, 0.5);
  // If we get here, test passes
}

TEST(ParameterPassingTest, RvalueReference) {
  Rectangle rect{5.0, 3.0};
  Rectangle result = transform_by_rvalue(std::move(rect), 3.0);

  EXPECT_DOUBLE_EQ(result.width, 15.0);
  EXPECT_DOUBLE_EQ(result.height, 9.0);
}

TEST(ParameterPassingTest, RvalueReferenceWithTemporary) {
  Rectangle result = transform_by_rvalue(Rectangle{4.0, 2.0}, 2.5);

  EXPECT_DOUBLE_EQ(result.width, 10.0);
  EXPECT_DOUBLE_EQ(result.height, 5.0);
}

TEST(ParameterPassingTest, RvalueReferenceChaining) {
  Rectangle result = transform_by_rvalue(transform_by_rvalue(Rectangle{2.0, 3.0}, 2.0), 3.0);

  EXPECT_DOUBLE_EQ(result.width, 12.0);
  EXPECT_DOUBLE_EQ(result.height, 18.0);
}

}  // namespace
}  // namespace ferric::foundation
