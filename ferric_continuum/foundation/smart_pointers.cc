#include "smart_pointers.hh"

#include <utility>

namespace ferric::foundation {

// =============================================================================
// Node Implementation
// =============================================================================

Node::Node(int value) : value_(value), next_(nullptr) {}

void Node::append(std::unique_ptr<Node> node) {
  if (next_) {
    next_->append(std::move(node));
  } else {
    next_ = std::move(node);
  }
}

std::unique_ptr<Node> create_list(const std::vector<int>& values) {
  if (values.empty()) {
    return nullptr;
  }

  auto head = std::make_unique<Node>(values[0]);

  for (size_t i = 1; i < values.size(); ++i) {
    head->append(std::make_unique<Node>(values[i]));
  }

  return head;  // Ownership transferred to caller
}

size_t count_nodes(const Node* head) {
  size_t count = 0;
  const Node* current = head;

  while (current) {
    ++count;
    current = current->next();
  }

  return count;
}

// =============================================================================
// Resource Implementation (shared_ptr)
// =============================================================================

int Resource::instance_count_ = 0;

Resource::Resource(int id) : id_(id) {
  ++instance_count_;
}

Resource::~Resource() {
  --instance_count_;
}

std::shared_ptr<Resource> create_shared_resource(int id) {
  return std::make_shared<Resource>(id);
}

std::vector<std::shared_ptr<Resource>> share_resource(std::shared_ptr<Resource> resource,
                                                      int copies) {
  std::vector<std::shared_ptr<Resource>> result;

  for (int i = 0; i < copies; ++i) {
    result.push_back(resource);  // Share ownership
  }

  return result;
}

// =============================================================================
// FileGuard Implementation (RAII)
// =============================================================================

FileGuard::FileGuard(const std::string& filename) : filename_(filename), is_open_(true) {
  // In real code: file_ = fopen(filename.c_str(), "r");
}

FileGuard::~FileGuard() {
  close();
}

FileGuard::FileGuard(FileGuard&& other) noexcept
    : filename_(std::move(other.filename_)), is_open_(other.is_open_) {
  other.is_open_ = false;
}

FileGuard& FileGuard::operator=(FileGuard&& other) noexcept {
  if (this != &other) {
    close();
    filename_ = std::move(other.filename_);
    is_open_ = other.is_open_;
    other.is_open_ = false;
  }
  return *this;
}

void FileGuard::close() {
  if (is_open_) {
    // In real code: fclose(file_);
    is_open_ = false;
  }
}

}  // namespace ferric::foundation
