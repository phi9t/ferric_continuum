#include "value_semantics.hh"

#include <gtest/gtest.h>

namespace ferric::foundation {
namespace {

TEST(ValueSemanticsTest, Constructor) {
  Point p(3.0, 4.0);
  EXPECT_DOUBLE_EQ(p.x(), 3.0);
  EXPECT_DOUBLE_EQ(p.y(), 4.0);
}

TEST(ValueSemanticsTest, Accessors) {
  Point p(5.5, 10.2);
  EXPECT_DOUBLE_EQ(p.x(), 5.5);
  EXPECT_DOUBLE_EQ(p.y(), 10.2);
}

TEST(ValueSemanticsTest, Translate) {
  Point p(3.0, 4.0);
  Point p2 = p.translate(2.0, 1.0);

  // Original unchanged
  EXPECT_DOUBLE_EQ(p.x(), 3.0);
  EXPECT_DOUBLE_EQ(p.y(), 4.0);

  // New point with translation
  EXPECT_DOUBLE_EQ(p2.x(), 5.0);
  EXPECT_DOUBLE_EQ(p2.y(), 5.0);
}

TEST(ValueSemanticsTest, TranslateNegative) {
  Point p(10.0, 10.0);
  Point p2 = p.translate(-3.0, -5.0);

  EXPECT_DOUBLE_EQ(p2.x(), 7.0);
  EXPECT_DOUBLE_EQ(p2.y(), 5.0);
}

TEST(ValueSemanticsTest, DistanceFromOrigin) {
  Point p1(3.0, 4.0);
  EXPECT_DOUBLE_EQ(p1.distance_from_origin(), 5.0);

  Point p2(0.0, 0.0);
  EXPECT_DOUBLE_EQ(p2.distance_from_origin(), 0.0);

  Point p3(5.0, 0.0);
  EXPECT_DOUBLE_EQ(p3.distance_from_origin(), 5.0);

  Point p4(0.0, 12.0);
  EXPECT_DOUBLE_EQ(p4.distance_from_origin(), 12.0);
}

TEST(ValueSemanticsTest, ToString) {
  Point p(3.5, 4.2);
  std::string str = p.to_string();
  EXPECT_FALSE(str.empty());
  // Should contain the coordinates
  EXPECT_NE(str.find("3.5"), std::string::npos);
  EXPECT_NE(str.find("4.2"), std::string::npos);
}

TEST(ValueSemanticsTest, IndependentCopies) {
  Point p1(3.0, 4.0);
  Point p2 = p1;  // Copy

  // Verify copy was made
  EXPECT_EQ(p1.x(), p2.x());
  EXPECT_EQ(p1.y(), p2.y());

  // Translate p2
  p2 = p2.translate(1.0, 1.0);

  // p1 should be unchanged
  EXPECT_EQ(p1.x(), 3.0);
  EXPECT_EQ(p1.y(), 4.0);

  // p2 should be changed
  EXPECT_EQ(p2.x(), 4.0);
  EXPECT_EQ(p2.y(), 5.0);
}

TEST(ValueSemanticsTest, CopyConstructor) {
  Point p1(10.0, 20.0);
  Point p2(p1);

  EXPECT_DOUBLE_EQ(p2.x(), 10.0);
  EXPECT_DOUBLE_EQ(p2.y(), 20.0);

  // Modify copy doesn't affect original
  p2 = p2.translate(1.0, 1.0);
  EXPECT_DOUBLE_EQ(p1.x(), 10.0);
  EXPECT_DOUBLE_EQ(p1.y(), 20.0);
}

TEST(ValueSemanticsTest, AssignmentOperator) {
  Point p1(10.0, 20.0);
  Point p2(0.0, 0.0);

  p2 = p1;

  EXPECT_DOUBLE_EQ(p2.x(), 10.0);
  EXPECT_DOUBLE_EQ(p2.y(), 20.0);

  // Modify assigned doesn't affect original
  p2 = p2.translate(5.0, 5.0);
  EXPECT_DOUBLE_EQ(p1.x(), 10.0);
  EXPECT_DOUBLE_EQ(p1.y(), 20.0);
}

}  // namespace
}  // namespace ferric::foundation
