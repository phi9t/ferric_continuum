#pragma once

#include <cstddef>
#include <string>

namespace ferric::foundation {

/// A class managing a large resource to demonstrate move semantics
/// Copying is expensive, but moving is cheap
class LargeBuffer {
 public:
  explicit LargeBuffer(size_t size);

  // Copy operations are expensive (deep copy)
  LargeBuffer(const LargeBuffer& other);
  LargeBuffer& operator=(const LargeBuffer& other);

  // Move operations are cheap (transfer ownership)
  LargeBuffer(LargeBuffer&& other) noexcept;
  LargeBuffer& operator=(LargeBuffer&& other) noexcept;

  ~LargeBuffer();

  size_t size() const { return size_; }
  void fill(int value);

  // Counters to track operations
  static size_t copy_count() { return copy_count_; }
  static size_t move_count() { return move_count_; }
  static void reset_counts();

 private:
  int* data_;
  size_t size_;

  static size_t copy_count_;
  static size_t move_count_;
};

/// Function returning by value - uses move optimization
LargeBuffer create_buffer(size_t size);

/// Function taking by value - copies if passed lvalue, moves if passed rvalue
LargeBuffer process_copy(LargeBuffer buf);

/// Function taking rvalue reference - always moves
LargeBuffer process_move(LargeBuffer&& buf);

}  // namespace ferric::foundation
