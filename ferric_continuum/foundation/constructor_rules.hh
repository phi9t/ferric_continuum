#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace ferric::foundation {

// Example 1: Full Rule of Five implementation
// When managing resources, define all special member functions
class ResourceManager {
 public:
  // 1. Default constructor
  ResourceManager();
  explicit ResourceManager(size_t size);

  // 2. Destructor
  ~ResourceManager();

  // 3. Copy constructor
  ResourceManager(const ResourceManager& other);

  // 4. Copy assignment operator
  ResourceManager& operator=(const ResourceManager& other);

  // 5. Move constructor
  ResourceManager(ResourceManager&& other) noexcept;

  // 6. Move assignment operator
  ResourceManager& operator=(ResourceManager&& other) noexcept;

  // Accessors
  size_t size() const { return size_; }
  int* data() const { return data_; }
  bool is_valid() const { return data_ != nullptr; }

  // Statistics
  static int default_constructions() { return default_constructions_; }
  static int copy_constructions() { return copy_constructions_; }
  static int move_constructions() { return move_constructions_; }
  static int destructions() { return destructions_; }
  static void reset_stats();

 private:
  int* data_;
  size_t size_;

  static inline int default_constructions_ = 0;
  static inline int copy_constructions_ = 0;
  static inline int move_constructions_ = 0;
  static inline int destructions_ = 0;
};

// Example 2: Move-only type
// Useful for RAII types that shouldn't be copied (file handles, unique resources)
class MoveOnlyResource {
 public:
  explicit MoveOnlyResource(const std::string& name);
  ~MoveOnlyResource();

  // Delete copy operations
  MoveOnlyResource(const MoveOnlyResource&) = delete;
  MoveOnlyResource& operator=(const MoveOnlyResource&) = delete;

  // Allow move operations
  MoveOnlyResource(MoveOnlyResource&& other) noexcept;
  MoveOnlyResource& operator=(MoveOnlyResource&& other) noexcept;

  const std::string& name() const { return name_; }
  bool is_valid() const { return valid_; }

 private:
  std::string name_;
  bool valid_;
};

// Example 3: Copyable type with defaults
// When compiler-generated versions are correct, use = default
class Point {
 public:
  Point(double x, double y) : x_(x), y_(y) {}

  // Compiler-generated versions are perfect for this simple type
  Point(const Point&) = default;
  Point& operator=(const Point&) = default;
  Point(Point&&) = default;
  Point& operator=(Point&&) = default;
  ~Point() = default;

  double x() const { return x_; }
  double y() const { return y_; }

 private:
  double x_, y_;
};

// Example 4: Rule of Zero (BEST PRACTICE)
// Use RAII types (smart pointers, containers) - no manual resource management
class RuleOfZeroExample {
 public:
  RuleOfZeroExample() = default;

  // No need to define any special member functions!
  // std::vector and std::string handle everything automatically

  void add_value(int value) { data_.push_back(value); }
  void set_name(const std::string& name) { name_ = name; }

  const std::vector<int>& data() const { return data_; }
  const std::string& name() const { return name_; }

 private:
  std::vector<int> data_;
  std::string name_;
};

// Factory functions to demonstrate usage
ResourceManager create_resource(size_t size);
MoveOnlyResource create_unique_resource(const std::string& name);
std::vector<ResourceManager> create_multiple_resources(size_t count, size_t size);

}  // namespace ferric::foundation
