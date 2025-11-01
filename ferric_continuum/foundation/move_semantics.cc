#include "move_semantics.hh"

#include <utility>

namespace ferric::foundation {

size_t LargeBuffer::copy_count_ = 0;
size_t LargeBuffer::move_count_ = 0;

LargeBuffer::LargeBuffer(size_t size) : data_(new int[size]), size_(size) {
  for (size_t i = 0; i < size_; ++i) {
    data_[i] = 0;
  }
}

// Copy constructor - expensive operation
LargeBuffer::LargeBuffer(const LargeBuffer& other)
    : data_(new int[other.size_]), size_(other.size_) {
  for (size_t i = 0; i < size_; ++i) {
    data_[i] = other.data_[i];
  }
  ++copy_count_;
}

LargeBuffer& LargeBuffer::operator=(const LargeBuffer& other) {
  if (this != &other) {
    delete[] data_;
    size_ = other.size_;
    data_ = new int[size_];
    for (size_t i = 0; i < size_; ++i) {
      data_[i] = other.data_[i];
    }
    ++copy_count_;
  }
  return *this;
}

// Move constructor - cheap operation (just pointer swap)
LargeBuffer::LargeBuffer(LargeBuffer&& other) noexcept : data_(other.data_), size_(other.size_) {
  other.data_ = nullptr;
  other.size_ = 0;
  ++move_count_;
}

LargeBuffer& LargeBuffer::operator=(LargeBuffer&& other) noexcept {
  if (this != &other) {
    delete[] data_;
    data_ = other.data_;
    size_ = other.size_;
    other.data_ = nullptr;
    other.size_ = 0;
    ++move_count_;
  }
  return *this;
}

LargeBuffer::~LargeBuffer() {
  delete[] data_;
}

void LargeBuffer::fill(int value) {
  for (size_t i = 0; i < size_; ++i) {
    data_[i] = value;
  }
}

void LargeBuffer::reset_counts() {
  copy_count_ = 0;
  move_count_ = 0;
}

LargeBuffer create_buffer(size_t size) {
  LargeBuffer buf(size);
  buf.fill(42);
  return buf;  // Return value optimization or move
}

LargeBuffer process_copy(LargeBuffer buf) {
  buf.fill(100);
  return buf;  // Move on return
}

LargeBuffer process_move(LargeBuffer&& buf) {
  buf.fill(200);
  return std::move(buf);  // Explicit move
}

}  // namespace ferric::foundation
