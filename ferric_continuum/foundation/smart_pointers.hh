#pragma once

#include <memory>
#include <vector>

namespace ferric::foundation {

/// Simple node class for demonstrating smart pointers
class Node {
 public:
  explicit Node(int value);

  int value() const { return value_; }
  Node* next() const { return next_.get(); }

  // Add a node to the end of the chain (unique ownership)
  void append(std::unique_ptr<Node> node);

 private:
  int value_;
  std::unique_ptr<Node> next_;  // Exclusive ownership
};

// =============================================================================
// unique_ptr - Exclusive Ownership
// =============================================================================

/// Creates a linked list with unique_ptr
/// Demonstrates: RAII, automatic cleanup, no manual delete needed
std::unique_ptr<Node> create_list(const std::vector<int>& values);

/// Count nodes in list (raw pointer for observation only)
size_t count_nodes(const Node* head);

// =============================================================================
// shared_ptr - Shared Ownership
// =============================================================================

class Resource {
 public:
  explicit Resource(int id);
  ~Resource();

  int id() const { return id_; }

  // Track number of Resource instances alive
  static int instance_count() { return instance_count_; }

 private:
  int id_;
  static int instance_count_;
};

/// Create a shared resource
std::shared_ptr<Resource> create_shared_resource(int id);

/// Share ownership - multiple pointers to same resource
std::vector<std::shared_ptr<Resource>> share_resource(std::shared_ptr<Resource> resource,
                                                      int copies);

// =============================================================================
// RAII Pattern
// =============================================================================

/// RAII wrapper simulating file handle
/// Resource acquired in constructor, released in destructor
class FileGuard {
 public:
  explicit FileGuard(const std::string& filename);
  ~FileGuard();

  // Non-copyable (file handles shouldn't be copied)
  FileGuard(const FileGuard&) = delete;
  FileGuard& operator=(const FileGuard&) = delete;

  // Movable (transfer ownership)
  FileGuard(FileGuard&& other) noexcept;
  FileGuard& operator=(FileGuard&& other) noexcept;

  bool is_open() const { return is_open_; }
  const std::string& filename() const { return filename_; }

 private:
  void close();

  std::string filename_;
  bool is_open_;
};

}  // namespace ferric::foundation
