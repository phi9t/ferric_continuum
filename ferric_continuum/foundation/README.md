# Foundation: Core C++ and Rust Concepts

This directory contains foundational examples demonstrating core C++ and Rust concepts side-by-side. Each concept includes library implementations, demo programs, and comprehensive tests for both languages.

## Overview

The foundation examples illustrate key differences and similarities between C++ and Rust in:
- Memory management and ownership
- Value vs. reference semantics
- Resource management (RAII)
- Parameter passing conventions
- Move semantics and performance optimization

## Table of Contents

1. [Value Semantics](#1-value-semantics)
2. [Move Semantics](#2-move-semantics)
3. [Parameter Passing](#3-parameter-passing)
4. [Smart Pointers & RAII](#4-smart-pointers--raii)
5. [Constructor Rules (Rule of Five)](#5-constructor-rules)

---

## 1. Value Semantics

**Concept**: Each object is an independent value. Copying creates a new, independent object.

### C++ Value Semantics

Value semantics is the default in C++. When you copy an object, you get an independent duplicate.

**Example:**

```cpp
namespace ferric::foundation {

class Point {
 public:
  Point(double x, double y) : x_(x), y_(y) {}
  
  // Copy operations create independent values
  Point(const Point& other) = default;
  Point& operator=(const Point& other) = default;
  
  // Operations return new values
  Point translate(double dx, double dy) const {
    return Point(x_ + dx, y_ + dy);
  }
  
  double distance_from_origin() const {
    return std::sqrt(x_ * x_ + y_ * y_);
  }
  
 private:
  double x_, y_;
};

}  // namespace ferric::foundation
```

**Usage:**

```cpp
ferric::foundation::Point p1(3.0, 4.0);
ferric::foundation::Point p2 = p1;  // Independent copy

p2 = p2.translate(2.0, 1.0);  // p1 remains unchanged
// p1 is still (3.0, 4.0)
// p2 is now (5.0, 5.0)
```

### Rust Value Semantics

In Rust, types with `Copy` trait have value semantics. Only simple types can be `Copy`.

**Example:**

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    x: f64,
    y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }
    
    pub fn translate(&self, dx: f64, dy: f64) -> Self {
        Point {
            x: self.x + dx,
            y: self.y + dy,
        }
    }
    
    pub fn distance_from_origin(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
}
```

**Usage:**

```rust
let p1 = Point::new(3.0, 4.0);
let p2 = p1;  // Copy happens automatically (Copy trait)

let p2 = p2.translate(2.0, 1.0);  // p1 remains unchanged
// p1 is still (3.0, 4.0)
// p2 is now (5.0, 5.0)
```

### Key Differences

| Aspect | C++ | Rust |
|--------|-----|------|
| Default behavior | Copy by default | Move by default |
| Value semantics | Automatic for all types | Only for types with `Copy` trait |
| Opt-in | Move requires `std::move()` | Copy requires `.clone()` |
| Safety | Can lead to expensive copies | Compiler enforces explicit copying |

**Build and run:**
```bash
bazel run //ferric_continuum/foundation:value_semantics_demo_cc
bazel run //ferric_continuum/foundation:value_semantics_demo_rs
```

---

## 2. Move Semantics

**Concept**: Transfer ownership of resources without copying. Enables efficient resource management and zero-cost abstractions.

### C++ Move Semantics

Move semantics allow transferring resources from one object to another without copying.

**Example:**

```cpp
namespace ferric::foundation {

class LargeBuffer {
 public:
  explicit LargeBuffer(size_t size) 
      : data_(new int[size]), size_(size) {
    ++alloc_count_;
  }
  
  // Copy constructor - expensive
  LargeBuffer(const LargeBuffer& other) 
      : data_(new int[other.size_]), size_(other.size_) {
    std::copy(other.data_, other.data_ + size_, data_);
    ++copy_count_;
  }
  
  // Move constructor - efficient
  LargeBuffer(LargeBuffer&& other) noexcept 
      : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
    ++move_count_;
  }
  
  // Move assignment operator
  LargeBuffer& operator=(LargeBuffer&& other) noexcept {
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
  
  ~LargeBuffer() {
    delete[] data_;
    ++dealloc_count_;
  }
  
 private:
  int* data_;
  size_t size_;
  
  static inline int alloc_count_ = 0;
  static inline int copy_count_ = 0;
  static inline int move_count_ = 0;
  static inline int dealloc_count_ = 0;
};

// Factory function - return by value uses move
LargeBuffer create_buffer(size_t size) {
  return LargeBuffer(size);  // Move or RVO
}

}  // namespace ferric::foundation
```

**Usage:**

```cpp
namespace ff = ferric::foundation;

// Efficient: move or RVO (Return Value Optimization)
ff::LargeBuffer buf1 = ff::create_buffer(10000);

// Expensive: copy (buf1 is lvalue)
ff::LargeBuffer buf2 = buf1;

// Efficient: explicit move
ff::LargeBuffer buf3 = std::move(buf1);  // buf1 is now in "moved-from" state
```

### Rust Move Semantics

In Rust, moves are the **default** behavior for types without the `Copy` trait.

**Example:**

```rust
pub struct LargeBuffer {
    data: Vec<i32>,
}

impl LargeBuffer {
    pub fn new(size: usize) -> Self {
        LargeBuffer {
            data: vec![0; size],
        }
    }
    
    pub fn size(&self) -> usize {
        self.data.len()
    }
}

// Factory function - ownership transferred automatically
pub fn create_buffer(size: usize) -> LargeBuffer {
    LargeBuffer::new(size)
}
```

**Usage:**

```rust
// Efficient: ownership transferred (no copy)
let buf1 = create_buffer(10000);

// Move: buf1 is moved into buf2 (buf1 can't be used anymore)
let buf2 = buf1;  // buf1 is MOVED, not copied

// Explicit copy if needed
let buf3 = buf2.clone();  // buf2 remains valid
```

### Key Differences

| Aspect | C++ | Rust |
|--------|-----|------|
| Default | Copy by default | Move by default |
| Explicit move | `std::move()` required | Automatic for non-Copy types |
| Explicit copy | Copy constructor called | `.clone()` required |
| Safety | Can use moved-from objects (unsafe) | Compiler prevents use after move |
| Performance | Opt-in optimization | Zero-cost by default |

**Build and run:**
```bash
bazel run //ferric_continuum/foundation:move_semantics_demo_cc
bazel run //ferric_continuum/foundation:move_semantics_demo_rs
```

---

## 3. Parameter Passing

**Concept**: How to efficiently and safely pass arguments to functions. Different conventions have different performance and semantic implications.

### C++ Parameter Passing Conventions

C++ offers multiple ways to pass parameters, each with different semantics:

```cpp
namespace ferric::foundation {

struct Rectangle {
  double width;
  double height;
};

// 1. Pass by const reference - read-only, no copy (MOST COMMON)
double compute_area_by_const_ref(const Rectangle& rect) {
  return rect.width * rect.height;
}

// 2. Pass by value - creates copy
double compute_area_by_value(Rectangle rect) {
  rect.width *= 2;  // Local modification
  return rect.width * rect.height;
}

// 3. Pass by non-const reference - modify original
void scale_by_reference(Rectangle& rect, double factor) {
  rect.width *= factor;
  rect.height *= factor;
}

// 4. Pass by pointer - nullable, C-style
void scale_by_pointer(Rectangle* rect, double factor) {
  if (rect) {  // Must check for nullptr
    rect->width *= factor;
    rect->height *= factor;
  }
}

// 5. Pass by rvalue reference - consume temporaries
Rectangle transform_by_rvalue(Rectangle&& rect, double factor) {
  rect.width *= factor;
  rect.height *= factor;
  return rect;  // Move
}

}  // namespace ferric::foundation
```

**Usage:**

```cpp
namespace ff = ferric::foundation;

ff::Rectangle rect{10.0, 5.0};

// Read-only access
double area = ff::compute_area_by_const_ref(rect);  // rect unchanged

// Modify in place
ff::scale_by_reference(rect, 2.0);  // rect is now 20.0 x 10.0

// Nullable pointer
ff::scale_by_pointer(&rect, 0.5);  // rect is now 10.0 x 5.0
ff::scale_by_pointer(nullptr, 2.0);  // Safe - checks for null

// Consume temporary
ff::Rectangle result = ff::transform_by_rvalue(std::move(rect), 3.0);
```

### Rust Parameter Passing Conventions

Rust has three main parameter passing conventions:

```rust
pub struct Rectangle {
    pub width: f64,
    pub height: f64,
}

// 1. Pass by immutable reference - read-only (MOST COMMON)
pub fn compute_area_by_ref(rect: &Rectangle) -> f64 {
    rect.width * rect.height
}

// 2. Pass by value - take ownership or copy (for Copy types)
pub fn compute_area_by_value(mut rect: Rectangle) -> f64 {
    rect.width *= 2.0;  // Local modification
    rect.width * rect.height
}

// 3. Pass by mutable reference - modify original
pub fn scale_by_mut_ref(rect: &mut Rectangle, factor: f64) {
    rect.width *= factor;
    rect.height *= factor;
}

// 4. Taking ownership - consume the value
pub fn transform_by_value(mut rect: Rectangle, factor: f64) -> Rectangle {
    rect.width *= factor;
    rect.height *= factor;
    rect
}
```

**Usage:**

```rust
let mut rect = Rectangle { width: 10.0, height: 5.0 };

// Read-only access (borrow)
let area = compute_area_by_ref(&rect);  // rect still usable

// Modify in place (mutable borrow)
scale_by_mut_ref(&mut rect, 2.0);  // rect is now 20.0 x 10.0

// Take ownership (move)
let result = transform_by_value(rect, 3.0);  // rect can't be used anymore
```

### Key Differences

| C++ Convention | Rust Equivalent | When to Use |
|----------------|-----------------|-------------|
| `const T&` | `&T` | Default for read-only access |
| `T` (small types) | `T` (Copy types) | Small, trivially copyable types |
| `T&` | `&mut T` | Need to modify original |
| `T*` | `Option<&T>` | Optional parameters (use references instead) |
| `T&&` | `T` | Consuming temporaries (automatic in Rust) |

**Guidelines:**

**C++:**
- Default: `const T&` for read-only
- Use `T` for small types (primitives, small PODs)
- Use `T&` when modifying
- Avoid `T*` for parameters (use references)
- Use `T&&` for move semantics

**Rust:**
- Default: `&T` for read-only (most common)
- Use `&mut T` when modifying
- Use `T` to take ownership (move for non-Copy, copy for Copy)
- No null pointers - use `Option<&T>` if needed

**Build and run:**
```bash
bazel run //ferric_continuum/foundation:parameter_passing_demo_cc
bazel run //ferric_continuum/foundation:parameter_passing_demo_rs
```

---

## 4. Smart Pointers & RAII

**Concept**: Automatic resource management. Resources (memory, files, locks) are tied to object lifetime and automatically cleaned up.

### C++ Smart Pointers

C++ provides smart pointers for automatic memory management:

```cpp
namespace ferric::foundation {

// Manual memory management with raw pointers (DON'T DO THIS)
struct Node {
  int value;
  Node* next;  // Raw pointer - manual cleanup needed
};

// RAII with unique_ptr - exclusive ownership
std::unique_ptr<Node> create_list(const std::vector<int>& values) {
  std::unique_ptr<Node> head = nullptr;
  for (auto it = values.rbegin(); it != values.rend(); ++it) {
    auto node = std::make_unique<Node>();
    node->value = *it;
    node->next = head.release();  // Transfer ownership
    head = std::move(node);
  }
  return head;  // Move ownership to caller
}

// RAII with shared_ptr - shared ownership
class Resource {
 public:
  explicit Resource(int id) : id_(id) { ++count_; }
  ~Resource() { --count_; }
  
  int id() const { return id_; }
  static int instance_count() { return count_; }
  
 private:
  int id_;
  static inline int count_ = 0;
};

std::shared_ptr<Resource> create_shared_resource(int id) {
  return std::make_shared<Resource>(id);
}

// Share ownership with multiple references
std::vector<std::shared_ptr<Resource>> share_resource(
    std::shared_ptr<Resource> res, int count) {
  std::vector<std::shared_ptr<Resource>> copies;
  for (int i = 0; i < count; ++i) {
    copies.push_back(res);  // Increment reference count
  }
  return copies;
}

// RAII for file handles
class FileGuard {
 public:
  explicit FileGuard(const std::string& filename) 
      : filename_(filename), open_(true) {
    // In real code: open file
  }
  
  ~FileGuard() {
    if (open_) {
      // Automatic cleanup
    }
  }
  
  // Move-only
  FileGuard(FileGuard&& other) noexcept 
      : filename_(std::move(other.filename_)), open_(other.open_) {
    other.open_ = false;
  }
  
  FileGuard(const FileGuard&) = delete;
  FileGuard& operator=(const FileGuard&) = delete;
  
  bool is_open() const { return open_; }
  const std::string& filename() const { return filename_; }
  
 private:
  std::string filename_;
  bool open_;
};

}  // namespace ferric::foundation
```

**Usage:**

```cpp
namespace ff = ferric::foundation;

// unique_ptr - exclusive ownership
{
  auto list = ff::create_list({1, 2, 3, 4, 5});
  auto list2 = std::move(list);  // Transfer ownership
  // list is now nullptr
  // list2 owns the data
}  // Automatic cleanup

// shared_ptr - shared ownership
{
  auto resource = ff::create_shared_resource(42);
  LOG(INFO) << "Use count: " << resource.use_count();  // 1
  
  {
    auto shared = ff::share_resource(resource, 3);
    LOG(INFO) << "Use count: " << resource.use_count();  // 4
  }  // shared vector destroyed
  
  LOG(INFO) << "Use count: " << resource.use_count();  // 1
}  // resource destroyed when last shared_ptr goes away

// RAII for resources
{
  ff::FileGuard file("data.txt");
  // Use file...
}  // File automatically closed in destructor
```

### Rust Smart Pointers

Rust has similar concepts with compile-time ownership checking:

```rust
use std::rc::Rc;
use std::cell::RefCell;

// Box - exclusive ownership (like unique_ptr)
pub struct Node {
    value: i32,
    next: Option<Box<Node>>,
}

pub fn create_list(values: &[i32]) -> Option<Box<Node>> {
    let mut head: Option<Box<Node>> = None;
    for &value in values.iter().rev() {
        let node = Box::new(Node {
            value,
            next: head,
        });
        head = Some(node);
    }
    head
}

// Rc - shared ownership (like shared_ptr, single-threaded)
pub struct Resource {
    id: i32,
}

impl Resource {
    pub fn new(id: i32) -> Rc<Self> {
        Rc::new(Resource { id })
    }
    
    pub fn id(&self) -> i32 {
        self.id
    }
}

pub fn share_resource(res: Rc<Resource>, count: usize) -> Vec<Rc<Resource>> {
    let mut copies = Vec::new();
    for _ in 0..count {
        copies.push(Rc::clone(&res));  // Increment reference count
    }
    copies
}

// RAII with Drop trait
pub struct FileGuard {
    filename: String,
    open: bool,
}

impl FileGuard {
    pub fn new(filename: &str) -> Self {
        FileGuard {
            filename: filename.to_string(),
            open: true,
        }
    }
    
    pub fn is_open(&self) -> bool {
        self.open
    }
    
    pub fn filename(&self) -> &str {
        &self.filename
    }
}

impl Drop for FileGuard {
    fn drop(&mut self) {
        if self.open {
            // Automatic cleanup
        }
    }
}

// Interior mutability with RefCell
pub struct Counter(Rc<RefCell<i32>>);

impl Counter {
    pub fn new() -> Self {
        Counter(Rc::new(RefCell::new(0)))
    }
    
    pub fn increment(&self) {
        *self.0.borrow_mut() += 1;
    }
    
    pub fn get(&self) -> i32 {
        *self.0.borrow()
    }
}
```

**Usage:**

```rust
// Box - exclusive ownership
{
    let list = create_list(&[1, 2, 3, 4, 5]);
    let list2 = list;  // Move ownership
    // list can't be used anymore
}  // Automatic cleanup

// Rc - shared ownership
{
    let resource = Resource::new(42);
    info!(count = Rc::strong_count(&resource), "Use count");  // 1
    
    {
        let shared = share_resource(Rc::clone(&resource), 3);
        info!(count = Rc::strong_count(&resource), "Use count");  // 4
    }  // shared vector dropped
    
    info!(count = Rc::strong_count(&resource), "Use count");  // 1
}  // resource dropped when last Rc goes away

// RAII with Drop
{
    let file = FileGuard::new("data.txt");
    // Use file...
}  // File automatically closed in Drop::drop
```

### Key Differences

| C++ | Rust | Notes |
|-----|------|-------|
| `std::unique_ptr<T>` | `Box<T>` | Exclusive ownership |
| `std::shared_ptr<T>` | `Rc<T>` | Shared ownership (single-threaded) |
| `std::shared_ptr<T>` | `Arc<T>` | Shared ownership (thread-safe) |
| Destructor | `Drop` trait | Automatic cleanup |
| Raw pointers | Unsafe pointers | Must use `unsafe` block |
| Manual memory management | Compile-time ownership | Rust prevents memory errors at compile time |

**Build and run:**
```bash
bazel run //ferric_continuum/foundation:smart_pointers_demo_cc
bazel run //ferric_continuum/foundation:smart_pointers_demo_rs
```

---

## 5. Constructor Rules (Rule of Five)

**Concept**: Understanding C++ special member functions and their role in resource management and move semantics.

C++ provides five special member functions that control object lifecycle and copying/moving behavior:

### The Special Member Functions

```cpp
class ResourceManager {
 public:
  // 1. Default constructor
  ResourceManager() 
      : data_(nullptr), size_(0) {
  }
  
  // 2. Destructor
  ~ResourceManager() {
    delete[] data_;
  }
  
  // 3. Copy constructor
  ResourceManager(const ResourceManager& other) 
      : data_(new int[other.size_]), size_(other.size_) {
    std::copy(other.data_, other.data_ + size_, data_);
  }
  
  // 4. Copy assignment operator
  ResourceManager& operator=(const ResourceManager& other) {
    if (this != &other) {
      // Clean up existing resources
      delete[] data_;
      
      // Allocate and copy new resources
      data_ = new int[other.size_];
      size_ = other.size_;
      std::copy(other.data_, other.data_ + size_, data_);
    }
    return *this;
  }
  
  // 5. Move constructor
  ResourceManager(ResourceManager&& other) noexcept 
      : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }
  
  // 6. Move assignment operator
  ResourceManager& operator=(ResourceManager&& other) noexcept {
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
  
 private:
  int* data_;
  size_t size_;
};
```

### Rule of Five

**If you define or delete any of the following, you should explicitly define or delete all of them:**

1. **Destructor**: Cleans up resources when an object is destroyed
2. **Copy constructor**: Creates a new object as a copy of an existing one
3. **Copy assignment operator**: Assigns values from one object to another
4. **Move constructor**: Transfers ownership from one object to another
5. **Move assignment operator**: Assigns ownership from one object to another

### When to Define Explicitly

- **Define destructor** when managing dynamic memory or other resources (files, locks, network connections)
- **Define copy operations** when deep copying is necessary (owning raw pointers, file handles)
- **Define move operations** to enable efficient resource transfer and avoid copies
- **Use `= default`** when the compiler-generated version is correct
- **Use `= delete`** to prevent copying/moving (make type move-only or non-copyable)

### Examples

**Move-only type:**
```cpp
class MoveOnly {
 public:
  MoveOnly() = default;
  ~MoveOnly() = default;
  
  // Delete copy operations
  MoveOnly(const MoveOnly&) = delete;
  MoveOnly& operator=(const MoveOnly&) = delete;
  
  // Allow move operations
  MoveOnly(MoveOnly&&) = default;
  MoveOnly& operator=(MoveOnly&&) = default;
};
```

**Copyable type with automatic defaults:**
```cpp
class Point {
 public:
  Point(double x, double y) : x_(x), y_(y) {}
  
  // Compiler-generated versions are perfect
  Point(const Point&) = default;
  Point& operator=(const Point&) = default;
  Point(Point&&) = default;
  Point& operator=(Point&&) = default;
  ~Point() = default;
  
 private:
  double x_, y_;
};
```

### Rule of Zero

**Better approach:** Don't manage resources directly. Use RAII types (smart pointers, containers).

```cpp
class BetterResourceManager {
 public:
  BetterResourceManager() = default;
  
  // No need to define special member functions!
  // std::vector handles everything
  
 private:
  std::vector<int> data_;  // RAII container
};
```

### Rust Equivalent

Rust doesn't have this complexity because:
- Ownership is enforced at compile time
- Move is the default
- `Clone` trait for explicit copying
- `Drop` trait for cleanup (automatically implemented for most types)

```rust
// Move-only by default
struct MoveOnly {
    data: Vec<i32>,
}

// Add Clone for copyability
#[derive(Clone)]
struct Copyable {
    data: Vec<i32>,
}

// Custom cleanup with Drop
impl Drop for MoveOnly {
    fn drop(&mut self) {
        // Cleanup code
    }
}
```

---

## Building and Testing

### Build all foundation libraries:
```bash
bazel build //ferric_continuum/foundation:all
```

### Run all demos:
```bash
# C++ demos
bazel run //ferric_continuum/foundation:value_semantics_demo_cc
bazel run //ferric_continuum/foundation:move_semantics_demo_cc
bazel run //ferric_continuum/foundation:parameter_passing_demo_cc
bazel run //ferric_continuum/foundation:smart_pointers_demo_cc
bazel run //ferric_continuum/foundation:constructor_rules_demo_cc

# Rust demos
bazel run //ferric_continuum/foundation:value_semantics_demo_rs
bazel run //ferric_continuum/foundation:move_semantics_demo_rs
bazel run //ferric_continuum/foundation:parameter_passing_demo_rs
bazel run //ferric_continuum/foundation:smart_pointers_demo_rs
```

### Run all tests:
```bash
bazel test //ferric_continuum/foundation:all
```

---

## Key Takeaways

### C++
- **Value semantics by default**: Copying is implicit, moving requires `std::move()`
- **Rule of Five**: Manage special member functions carefully when handling resources
- **Smart pointers**: Use `std::unique_ptr` and `std::shared_ptr` for automatic memory management
- **RAII**: Tie resource lifetime to object lifetime
- **Parameter passing**: Default to `const T&` for read-only, `T&` for modify, `T&&` for move

### Rust
- **Move semantics by default**: Ownership transfer is automatic for non-Copy types
- **Borrow checker**: Compile-time verification of memory safety
- **Explicit copying**: Use `.clone()` when you need a copy
- **Drop trait**: Automatic cleanup, usually no manual implementation needed
- **Parameter passing**: Default to `&T` for read-only, `&mut T` for modify, `T` for ownership transfer

### Comparing C++ and Rust

| Feature | C++ Approach | Rust Approach |
|---------|--------------|---------------|
| Default | Copy | Move |
| Memory safety | Runtime (with care) | Compile-time (guaranteed) |
| Resource management | RAII (manual) | RAII (automatic via Drop) |
| Ownership | Convention-based | Enforced by compiler |
| Performance | Zero-cost with care | Zero-cost by design |
| Learning curve | Steep (many rules) | Steep (different model) |

Both languages enable high-performance, safe code. C++ offers flexibility and control with manual management, while Rust enforces safety at compile time with a steeper initial learning curve but fewer runtime surprises.

---

## Further Reading

- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)
- [The Rust Book - Ownership](https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html)
- [Smart Pointers in C++](https://en.cppreference.com/book/intro/smart_pointers)
- [Rust Smart Pointers](https://doc.rust-lang.org/book/ch15-00-smart-pointers.html)
