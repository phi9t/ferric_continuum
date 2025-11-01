#include "constructor_rules.hh"

#include <algorithm>

namespace ferric::foundation {

// ===== ResourceManager Implementation =====

ResourceManager::ResourceManager() : data_(nullptr), size_(0) {
  ++default_constructions_;
}

ResourceManager::ResourceManager(size_t size)
    : data_(size > 0 ? new int[size] : nullptr), size_(size) {
  if (data_) {
    std::fill(data_, data_ + size_, 0);
  }
  ++default_constructions_;
}

ResourceManager::~ResourceManager() {
  delete[] data_;
  ++destructions_;
}

ResourceManager::ResourceManager(const ResourceManager& other)
    : data_(other.size_ > 0 ? new int[other.size_] : nullptr), size_(other.size_) {
  if (data_ && other.data_) {
    std::copy(other.data_, other.data_ + size_, data_);
  }
  ++copy_constructions_;
}

ResourceManager& ResourceManager::operator=(const ResourceManager& other) {
  if (this != &other) {
    // Clean up existing resources
    delete[] data_;

    // Allocate and copy new resources
    size_ = other.size_;
    data_ = size_ > 0 ? new int[size_] : nullptr;
    if (data_ && other.data_) {
      std::copy(other.data_, other.data_ + size_, data_);
    }
  }
  return *this;
}

ResourceManager::ResourceManager(ResourceManager&& other) noexcept
    : data_(other.data_), size_(other.size_) {
  other.data_ = nullptr;
  other.size_ = 0;
  ++move_constructions_;
}

ResourceManager& ResourceManager::operator=(ResourceManager&& other) noexcept {
  if (this != &other) {
    // Clean up existing resources
    delete[] data_;

    // Transfer ownership
    data_ = other.data_;
    size_ = other.size_;

    // Leave other in valid state
    other.data_ = nullptr;
    other.size_ = 0;
  }
  return *this;
}

void ResourceManager::reset_stats() {
  default_constructions_ = 0;
  copy_constructions_ = 0;
  move_constructions_ = 0;
  destructions_ = 0;
}

// ===== MoveOnlyResource Implementation =====

MoveOnlyResource::MoveOnlyResource(const std::string& name) : name_(name), valid_(true) {}

MoveOnlyResource::~MoveOnlyResource() {
  valid_ = false;
}

MoveOnlyResource::MoveOnlyResource(MoveOnlyResource&& other) noexcept
    : name_(std::move(other.name_)), valid_(other.valid_) {
  other.valid_ = false;
}

MoveOnlyResource& MoveOnlyResource::operator=(MoveOnlyResource&& other) noexcept {
  if (this != &other) {
    name_ = std::move(other.name_);
    valid_ = other.valid_;
    other.valid_ = false;
  }
  return *this;
}

// ===== Factory Functions =====

ResourceManager create_resource(size_t size) {
  return ResourceManager(size);  // Move or RVO
}

MoveOnlyResource create_unique_resource(const std::string& name) {
  return MoveOnlyResource(name);  // Move or RVO
}

std::vector<ResourceManager> create_multiple_resources(size_t count, size_t size) {
  std::vector<ResourceManager> resources;
  resources.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    resources.push_back(ResourceManager(size));
  }
  return resources;  // Move
}

}  // namespace ferric::foundation
