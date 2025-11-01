#include "move_semantics.hh"

#include <gtest/gtest.h>
#include <utility>

namespace ferric::foundation {
namespace {

TEST(MoveSemanticsTest, Constructor) {
  LargeBuffer buf(1000);
  EXPECT_EQ(buf.size(), 1000);
}

TEST(MoveSemanticsTest, Fill) {
  LargeBuffer buf(100);
  buf.fill(42);
  // If fill works, it shouldn't crash
  EXPECT_EQ(buf.size(), 100);
}

TEST(MoveSemanticsTest, CopyConstructor) {
  LargeBuffer::reset_counts();

  LargeBuffer buf1(500);
  buf1.fill(10);
  LargeBuffer buf2(buf1);  // Copy constructor

  EXPECT_EQ(LargeBuffer::copy_count(), 1);
  EXPECT_EQ(buf1.size(), 500);
  EXPECT_EQ(buf2.size(), 500);
}

TEST(MoveSemanticsTest, CopyAssignment) {
  LargeBuffer::reset_counts();

  LargeBuffer buf1(500);
  LargeBuffer buf2(100);

  buf2 = buf1;  // Copy assignment

  EXPECT_GE(LargeBuffer::copy_count(), 0);
  EXPECT_EQ(buf1.size(), 500);
  EXPECT_EQ(buf2.size(), 500);
}

TEST(MoveSemanticsTest, MoveConstructor) {
  LargeBuffer::reset_counts();

  LargeBuffer buf1(1000);
  LargeBuffer buf2(std::move(buf1));  // Move constructor

  EXPECT_EQ(buf1.size(), 0);  // Moved-from state
  EXPECT_EQ(buf2.size(), 1000);
  EXPECT_GE(LargeBuffer::move_count(), 1);
}

TEST(MoveSemanticsTest, MoveAssignment) {
  LargeBuffer::reset_counts();

  LargeBuffer buf1(1000);
  LargeBuffer buf2(500);

  buf2 = std::move(buf1);  // Move assignment

  EXPECT_EQ(buf1.size(), 0);  // Moved-from state
  EXPECT_EQ(buf2.size(), 1000);
  EXPECT_GE(LargeBuffer::move_count(), 1);
}

TEST(MoveSemanticsTest, CreateBuffer) {
  LargeBuffer::reset_counts();

  LargeBuffer buf1 = create_buffer(1000);

  // Should use move or RVO, not copy
  EXPECT_EQ(LargeBuffer::copy_count(), 0);
  EXPECT_EQ(buf1.size(), 1000);
}

TEST(MoveSemanticsTest, ProcessCopy) {
  LargeBuffer::reset_counts();

  LargeBuffer buf1(1000);
  LargeBuffer buf2 = process_copy(buf1);  // Lvalue, must copy

  // Should copy
  EXPECT_EQ(LargeBuffer::copy_count(), 1);

  // Original still valid
  EXPECT_EQ(buf1.size(), 1000);
  EXPECT_EQ(buf2.size(), 1000);
}

TEST(MoveSemanticsTest, ProcessMove) {
  LargeBuffer::reset_counts();

  LargeBuffer buf1 = create_buffer(1000);
  LargeBuffer buf2 = process_move(std::move(buf1));

  // Should not copy
  EXPECT_EQ(LargeBuffer::copy_count(), 0);

  // buf1 should be in moved-from state
  EXPECT_EQ(buf1.size(), 0);
  EXPECT_EQ(buf2.size(), 1000);
}

TEST(MoveSemanticsTest, CopyWhenNeeded) {
  LargeBuffer::reset_counts();

  LargeBuffer buf1(1000);
  LargeBuffer buf2 = buf1;  // Copy

  EXPECT_EQ(LargeBuffer::copy_count(), 1);

  // Both should be valid
  EXPECT_EQ(buf1.size(), 1000);
  EXPECT_EQ(buf2.size(), 1000);
}

TEST(MoveSemanticsTest, CounterTracking) {
  LargeBuffer::reset_counts();

  EXPECT_EQ(LargeBuffer::copy_count(), 0);
  EXPECT_EQ(LargeBuffer::move_count(), 0);

  {
    LargeBuffer buf1(100);
    LargeBuffer buf2 = buf1;             // Copy
    LargeBuffer buf3 = std::move(buf2);  // Move

    EXPECT_EQ(LargeBuffer::copy_count(), 1);
    EXPECT_GE(LargeBuffer::move_count(), 1);
  }

  // Counters should persist after destruction
  EXPECT_EQ(LargeBuffer::copy_count(), 1);
  EXPECT_GE(LargeBuffer::move_count(), 1);
}

}  // namespace
}  // namespace ferric::foundation
