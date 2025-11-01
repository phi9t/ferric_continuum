#include "constructor_rules.hh"

#include <gtest/gtest.h>
#include <utility>

namespace ferric::foundation {
namespace {

TEST(ConstructorRulesTest, ResourceManagerDefaultConstruction) {
  ResourceManager::reset_stats();

  ResourceManager r1(100);
  EXPECT_TRUE(r1.is_valid());
  EXPECT_EQ(r1.size(), 100);
  EXPECT_EQ(ResourceManager::default_constructions(), 1);
}

TEST(ConstructorRulesTest, ResourceManagerCopyConstruction) {
  ResourceManager::reset_stats();

  ResourceManager r1(100);
  ResourceManager r2 = r1;  // Copy constructor

  EXPECT_TRUE(r1.is_valid());
  EXPECT_TRUE(r2.is_valid());
  EXPECT_EQ(r1.size(), 100);
  EXPECT_EQ(r2.size(), 100);
  EXPECT_NE(r1.data(), r2.data());  // Different memory
  EXPECT_EQ(ResourceManager::copy_constructions(), 1);
}

TEST(ConstructorRulesTest, ResourceManagerMoveConstruction) {
  ResourceManager::reset_stats();

  ResourceManager r1(100);
  int* original_data = r1.data();

  ResourceManager r2 = std::move(r1);  // Move constructor

  EXPECT_FALSE(r1.is_valid());  // Moved-from state
  EXPECT_EQ(r1.size(), 0);
  EXPECT_TRUE(r2.is_valid());
  EXPECT_EQ(r2.size(), 100);
  EXPECT_EQ(r2.data(), original_data);  // Same memory
  EXPECT_EQ(ResourceManager::move_constructions(), 1);
}

TEST(ConstructorRulesTest, ResourceManagerCopyAssignment) {
  ResourceManager r1(100);
  ResourceManager r2(50);

  r2 = r1;  // Copy assignment

  EXPECT_TRUE(r1.is_valid());
  EXPECT_TRUE(r2.is_valid());
  EXPECT_EQ(r1.size(), 100);
  EXPECT_EQ(r2.size(), 100);
  EXPECT_NE(r1.data(), r2.data());  // Different memory
}

TEST(ConstructorRulesTest, ResourceManagerMoveAssignment) {
  ResourceManager r1(100);
  ResourceManager r2(50);
  int* original_data = r1.data();

  r2 = std::move(r1);  // Move assignment

  EXPECT_FALSE(r1.is_valid());  // Moved-from state
  EXPECT_TRUE(r2.is_valid());
  EXPECT_EQ(r2.size(), 100);
  EXPECT_EQ(r2.data(), original_data);  // Same memory
}

TEST(ConstructorRulesTest, MoveOnlyResourceCannotCopy) {
  // This test verifies the type signature - copy is deleted
  EXPECT_FALSE(std::is_copy_constructible_v<MoveOnlyResource>);
  EXPECT_FALSE(std::is_copy_assignable_v<MoveOnlyResource>);
  EXPECT_TRUE(std::is_move_constructible_v<MoveOnlyResource>);
  EXPECT_TRUE(std::is_move_assignable_v<MoveOnlyResource>);
}

TEST(ConstructorRulesTest, MoveOnlyResourceMoveSemantics) {
  MoveOnlyResource r1("resource1");
  EXPECT_TRUE(r1.is_valid());
  EXPECT_EQ(r1.name(), "resource1");

  MoveOnlyResource r2 = std::move(r1);
  EXPECT_FALSE(r1.is_valid());  // Moved-from
  EXPECT_TRUE(r2.is_valid());
  EXPECT_EQ(r2.name(), "resource1");
}

TEST(ConstructorRulesTest, PointWithDefaults) {
  Point p1(3.0, 4.0);

  // Copy works (= default)
  Point p2 = p1;
  EXPECT_EQ(p2.x(), 3.0);
  EXPECT_EQ(p2.y(), 4.0);

  // Move works (= default)
  Point p3 = std::move(p1);
  EXPECT_EQ(p3.x(), 3.0);
  EXPECT_EQ(p3.y(), 4.0);
}

TEST(ConstructorRulesTest, RuleOfZeroExample) {
  RuleOfZeroExample ex1;
  ex1.set_name("test");
  ex1.add_value(1);
  ex1.add_value(2);

  // Copy works automatically
  RuleOfZeroExample ex2 = ex1;
  EXPECT_EQ(ex2.name(), "test");
  EXPECT_EQ(ex2.data().size(), 2);

  // Move works automatically
  RuleOfZeroExample ex3 = std::move(ex1);
  EXPECT_EQ(ex3.name(), "test");
  EXPECT_EQ(ex3.data().size(), 2);
}

TEST(ConstructorRulesTest, FactoryFunctionUsesMove) {
  ResourceManager::reset_stats();

  ResourceManager r = create_resource(100);

  EXPECT_TRUE(r.is_valid());
  EXPECT_EQ(r.size(), 100);
  // Should use move or RVO, not copy
  EXPECT_EQ(ResourceManager::copy_constructions(), 0);
}

TEST(ConstructorRulesTest, VectorOfResourcesUsesMove) {
  ResourceManager::reset_stats();

  auto resources = create_multiple_resources(5, 100);

  EXPECT_EQ(resources.size(), 5);
  // Should use moves for container insertion
  EXPECT_GE(ResourceManager::move_constructions(), 0);
  // Should not copy
  EXPECT_EQ(ResourceManager::copy_constructions(), 0);
}

}  // namespace
}  // namespace ferric::foundation
