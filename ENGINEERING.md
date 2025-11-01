# ENGINEERING.md

Engineering standards and best practices for Ferric Continuum development.

---

## Table of Contents

1. [Philosophy](#philosophy)
2. [Google C++ Style Guide](#google-c-style-guide)
3. [Project-Specific Standards](#project-specific-standards)
   - [Namespace Usage](#namespace-usage-c)
   - [Logging Standards](#logging-standards)
   - [Testing and Benchmarking Standards](#testing-and-benchmarking-standards)
4. [C++ Core Guidelines](#c-core-guidelines)
5. [Modern C++ Checklist](#modern-c-checklist)
6. [Rust Best Practices](#rust-best-practices)
7. [Tooling and Automation](#tooling-and-automation)
8. [Recommendations for This Project](#recommendations-for-this-project)

---

## Philosophy

Ferric Continuum prioritizes **correctness, performance, and maintainability** in that order. Code should be:

- **Self-documenting** - express intent directly through types and names
- **Safe by default** - prevent errors at compile time when possible
- **Observable** - instrumented for profiling and debugging
- **Reproducible** - hermetic builds with deterministic outputs

---

## Google C++ Style Guide

The [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) provides battle-tested practices for large-scale codebases. Key principles:

### Code Organization

**Namespaces**
- Always use namespaces to avoid global scope pollution
- Never use `using namespace` directives (especially not `using namespace std;`)
- Prefer fully qualified names or targeted `using` declarations

**Header Files**
- Every `.h` file must be self-contained and compile independently
- Use include guards: `_<PROJECT>___<PATH>___<FILE>__H_`
- Apply "Include What You Use" (IWYU) - only include headers for symbols you directly reference
- Avoid forward declarations when possible; they hide dependencies

```cpp
// Good: self-contained header
#ifndef _FERRIC_CONTINUUM___CPP___MATRIX_H_
#define _FERRIC_CONTINUUM___CPP___MATRIX_H_

#include <memory>
#include <vector>

namespace ferric {

class Matrix {
public:
  explicit Matrix(size_t rows, size_t cols);
  // ...
private:
  std::vector<double> data_;
};

}  // namespace ferric

#endif  // _FERRIC_CONTINUUM___CPP___MATRIX_H_
```

### Memory & Ownership

**Smart Pointers**
- Prefer `std::unique_ptr` for exclusive ownership
- Use `std::shared_ptr` only with strong justification (shared ownership is complex)
- Never use raw `new`/`delete` in application code
- Avoid `std::auto_ptr` (deprecated)

```cpp
// Good: clear ownership with unique_ptr
std::unique_ptr<Matrix> CreateMatrix(size_t rows, size_t cols) {
  return std::make_unique<Matrix>(rows, cols);
}

// Bad: raw pointer with unclear ownership
Matrix* CreateMatrix(size_t rows, size_t cols) {
  return new Matrix(rows, cols);  // Who deletes this?
}
```

**Google-Specific Constraints**
- **No exceptions** - Use error codes, `std::optional`, or `absl::StatusOr`
- **No RTTI** (`dynamic_cast`, `typeid`) except in test code
- Avoid static storage objects with non-trivial destructors

### Function & Variable Design

**Optimize for Readers**
- Code is read far more than it's written - prioritize clarity
- Keep functions small (~40 lines is a reasonable threshold)
- Use descriptive names that convey purpose

**Variable Scope**
- Declare variables in the narrowest scope possible
- Initialize at declaration rather than assign later
- Use `const` liberally to document immutability

```cpp
// Good: narrow scope, immediate initialization
void ProcessData(const std::vector<int>& data) {
  for (const auto& value : data) {
    const int processed = Transform(value);
    Store(processed);
  }
}

// Bad: wide scope, delayed initialization
void ProcessData(const std::vector<int>& data) {
  int processed;
  for (const auto& value : data) {
    processed = Transform(value);
    Store(processed);
  }
}
```

**Return Values Over Output Parameters**
- Prefer returning values (by value or move) over output parameters
- Modern C++ move semantics make this efficient

```cpp
// Good: return by value
std::vector<double> ComputeResults(const Matrix& input) {
  std::vector<double> results;
  // ... compute
  return results;  // Move semantics apply
}

// Bad: output parameter
void ComputeResults(const Matrix& input, std::vector<double>* output) {
  output->clear();
  // ... compute
}
```

### Class Design

**Encapsulation**
- Make data members `private` unless they're constants
- Use getters/setters judiciously - don't expose implementation details

**Constructors**
- Use `explicit` for single-argument constructors to prevent implicit conversions
- Explicitly define or delete copy/move constructors and assignment operators

```cpp
class Matrix {
public:
  explicit Matrix(size_t size);  // explicit prevents Matrix m = 5;

  // Explicitly declare copy/move semantics
  Matrix(const Matrix&) = delete;
  Matrix& operator=(const Matrix&) = delete;
  Matrix(Matrix&&) = default;
  Matrix& operator=(Matrix&&) = default;

private:
  std::vector<double> data_;
};
```

**Inheritance**
- Prefer composition over inheritance
- Public inheritance only (no protected/private inheritance)
- Mark overrides with `override` or `final`, not `virtual`

```cpp
class Base {
public:
  virtual void Process() = 0;
  virtual ~Base() = default;
};

class Derived : public Base {
public:
  void Process() override;  // Good: explicit override
  // void Process();        // Bad: implicit override
};
```

### Modern C++ Features (C++20 Target)

**Type Deduction**
- Use `auto` for complex types and iterator loops
- Avoid `auto` when the type isn't obvious from context

**Const Correctness**
- Use `const` for all immutable references and pointers
- Mark member functions `const` when they don't modify state

**Compile-Time Evaluation**
- Use `constexpr` liberally for values and functions that can be computed at compile time

```cpp
constexpr double kPi = 3.14159265358979323846;

constexpr int Square(int x) {
  return x * x;
}

std::array<int, Square(10)> buffer;  // Computed at compile time
```

**Casts**
- Prefer C++-style casts (`static_cast`, `const_cast`, `reinterpret_cast`)
- Never use C-style casts
- For arithmetic conversions, prefer brace initialization: `int64_t{value}`

---

## Project-Specific Standards

These standards are specific to Ferric Continuum and complement the general C++ and Rust guidelines.

### Namespace Usage (C++)

**Never use `using namespace` declarations.** Instead, use namespace-qualified names or create namespace aliases.

```cpp
// ❌ BAD - Pollutes global namespace
using namespace std;
using namespace ferric::foundation;

void process() {
  cout << "Hello" << endl;
}

// ✅ GOOD - Explicit qualification
void process() {
  std::cout << "Hello" << std::endl;
  ferric::foundation::Point p(1.0, 2.0);
}

// ✅ GOOD - Namespace alias for shorter syntax
namespace ff = ferric::foundation;

void process() {
  ff::Point p(1.0, 2.0);
  ff::Vector v = p.normalize();
}
```

**Benefits:**
- Explicit namespace usage prevents naming conflicts
- Makes code dependencies clear and auditable
- Follows C++ Core Guidelines and Google C++ Style Guide
- Namespace aliases provide brevity without sacrificing clarity
- Easier to track symbol origins when debugging

**Exceptions:**
- `using` declarations for specific symbols in implementation files (`.cpp`) are acceptable:
  ```cpp
  // Acceptable in .cpp files only
  using std::vector;
  using std::unique_ptr;
  ```
- Never use `using namespace` in header files

### Logging Standards

**Use structured logging instead of raw print statements** for all code beyond trivial examples.

#### C++ Logging

Use **Abseil logging** (`absl/log/log.h`) for all C++ code, including demos and examples.

**Log Levels:**
- `LOG(INFO)` - General information
- `LOG(WARNING)` - Warning conditions
- `LOG(ERROR)` - Error conditions
- `LOG(FATAL)` - Fatal errors (terminates program)

**Initialization** (required in `main()`):
```cpp
#include "absl/log/log.h"
#include "absl/log/initialize.h"
#include "absl/log/globals.h"

int main() {
  // Initialize Abseil logging
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  
  LOG(INFO) << "Program started";
  // ... rest of program ...
  return 0;
}
```

**BUILD.bazel dependencies:**
```python
cc_binary(
    name = "my_program",
    srcs = ["my_program.cc"],
    deps = [
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:initialize",
        "@abseil-cpp//absl/log:globals",
        # ... other deps
    ],
)
```

**Usage examples:**
```cpp
// ❌ BAD - Raw output to stdout/stderr
std::cout << "Processing " << count << " items" << std::endl;
std::cerr << "Error: Buffer full" << std::endl;

// ✅ GOOD - Structured logging
LOG(INFO) << "Processing " << count << " items";
LOG(WARNING) << "Buffer nearly full: " << usage << "% capacity";
LOG(ERROR) << "Failed to allocate memory: " << size << " bytes";

// With context
LOG(INFO) << "Matrix multiplication: " << rows << "x" << cols
          << " in " << duration_ms << "ms";
```

#### Rust Logging

Use the **tracing** crate for all Rust code, including demos and examples.

**Macros:** `trace!`, `debug!`, `info!`, `warn!`, `error!`

**Initialization** (required in `main()`):
```rust
use tracing::{info, Level};
use tracing_subscriber;

fn main() {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();
    
    info!("Program started");
    // ... rest of program ...
}
```

**BUILD.bazel dependencies:**
```python
rust_binary(
    name = "my_program",
    srcs = ["my_program.rs"],
    edition = "2021",
    deps = [
        "@crates//:tracing",
        "@crates//:tracing-subscriber",
        # ... other deps
    ],
)
```

**Usage examples:**
```rust
use tracing::{info, warn, error, debug};

// ❌ BAD - Raw output
println!("Processing {} items", count);
eprintln!("Error: Buffer full");

// ✅ GOOD - Structured logging
info!(count, "Processing items");
warn!(usage = %usage, "Buffer nearly full");
error!(size, "Failed to allocate memory");

// With structured fields
info!(
    rows = matrix.rows,
    cols = matrix.cols,
    duration_ms = elapsed.as_millis(),
    "Matrix multiplication complete"
);

// Spans for tracing execution
use tracing::instrument;

#[instrument]
async fn process_data(data: &Data) -> Result<Output> {
    debug!("Starting data processing");
    // Function execution is automatically traced
    // ...
}
```

#### Benefits of Structured Logging

- **Configurable**: Output levels and destinations can be changed at runtime
- **Structured data**: Machine-parseable for log aggregation and analysis
- **Performance**: Can be compiled out or filtered with minimal overhead
- **Integration**: Works with observability tools (Prometheus, Grafana, ELK stack)
- **Context**: Automatically captures file, line, and module information
- **Distributed tracing**: Spans enable tracking across async operations and microservices

#### When Raw Output Is Acceptable

- **Build scripts**: Simple build/generation scripts that aren't part of the runtime system
- **Quick debugging**: Temporary debugging during development (remove before commit)
- **Interactive tools**: CLI tools where user-facing output is the primary purpose (though consider `clap` with logging)

**All demo programs and examples use structured logging** to demonstrate best practices. This includes files in `ferric_continuum/*/` directories with `*_demo.{cc,rs}` suffixes. Production code must always use the appropriate logging framework.

### Testing and Benchmarking Standards

#### Unit Testing

**C++ Testing (Google Test)**

```cpp
#include "gtest/gtest.h"
#include "my_module.h"

TEST(MyModuleTest, BasicFunctionality) {
  MyClass obj;
  EXPECT_EQ(obj.compute(5), 25);
  EXPECT_TRUE(obj.is_valid());
}

TEST(MyModuleTest, EdgeCases) {
  MyClass obj;
  EXPECT_THROW(obj.compute(-1), std::invalid_argument);
}
```

**Rust Testing (Built-in)**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_functionality() {
        let obj = MyStruct::new();
        assert_eq!(obj.compute(5), 25);
        assert!(obj.is_valid());
    }

    #[test]
    #[should_panic(expected = "invalid argument")]
    fn test_edge_cases() {
        let obj = MyStruct::new();
        obj.compute(-1);
    }
}
```

#### Benchmarking

**C++ Benchmarks (Google Benchmark)**

```cpp
#include "benchmark/benchmark.h"

static void BM_MatrixMultiply(benchmark::State& state) {
  Matrix a(state.range(0), state.range(0));
  Matrix b(state.range(0), state.range(0));

  for (auto _ : state) {
    Matrix c = a * b;
    benchmark::DoNotOptimize(c);
  }

  state.SetComplexityN(state.range(0));
}

BENCHMARK(BM_MatrixMultiply)
    ->RangeMultiplier(2)
    ->Range(8, 512)
    ->Complexity();
```

**Rust Benchmarks (Criterion)**

```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn matrix_multiply_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("MatrixMultiply");

    for size in [8, 16, 32, 64, 128, 256, 512] {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |b, &size| {
                let a = Matrix::new(size, size);
                let b = Matrix::new(size, size);
                b.iter(|| &a * &b);
            }
        );
    }

    group.finish();
}

criterion_group!(benches, matrix_multiply_benchmark);
criterion_main!(benches);
```

#### Performance Profiling

**Tools by Platform:**

- **C++**: Google Benchmark, perf, valgrind, gprof
- **Rust**: Criterion.rs, cargo flamegraph, perf
- **Cross-platform**: Intel VTune, AMD μProf

**Running Profilers:**

```bash
# C++ with perf
bazel build --config=opt //cpp:matrix_mul_benchmark
perf record -g bazel-bin/cpp/matrix_mul_benchmark
perf report

# Rust with flamegraph
cargo install flamegraph
cargo flamegraph --bench my_benchmark

# Valgrind for memory profiling
valgrind --tool=massif bazel-bin/cpp/my_binary
ms_print massif.out.*
```

**Best Practices:**
- Always profile in release/optimized builds
- Run benchmarks multiple times to account for variance
- Track performance over time with baseline comparisons
- Document performance characteristics in code comments
- Commit benchmark results to track regressions (via PerfSmith agent)

---

## C++ Core Guidelines

The [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines) (by Bjarne Stroustrup and Herb Sutter) represent the ISO C++ community's best practices.

### Fundamental Philosophy

**P.1: Express ideas directly in code**
- Use language features that match your intent
- Don't encode meaning in comments that could be types

**P.4: Ideally, a program should be statically type safe**
- Catch errors at compile time
- Use strong types instead of primitives

**P.5: Prefer compile-time checking to run-time checking**
- Static verification is faster and safer

**P.8: Don't leak any resources**
- Use RAII consistently
- Pair acquisition with initialization, cleanup with destruction

### Resource Management (RAII)

**Make ownership explicit**
- Never transfer ownership by raw pointer or reference (I.11)
- Use smart pointers to document ownership

```cpp
// Good: explicit ownership
std::unique_ptr<Resource> resource = AcquireResource();

// Bad: ambiguous ownership
Resource* resource = AcquireResource();  // Who owns this?
```

**RAII Pattern**
- Acquire resources in constructors
- Release resources in destructors
- Compiler guarantees cleanup

```cpp
class FileHandle {
public:
  explicit FileHandle(const std::string& path)
    : fd_(open(path.c_str(), O_RDONLY)) {
    if (fd_ < 0) throw std::runtime_error("Failed to open file");
  }

  ~FileHandle() {
    if (fd_ >= 0) close(fd_);
  }

  // Delete copy, allow move
  FileHandle(const FileHandle&) = delete;
  FileHandle& operator=(const FileHandle&) = delete;
  FileHandle(FileHandle&& other) noexcept : fd_(other.fd_) {
    other.fd_ = -1;
  }

  int get() const { return fd_; }

private:
  int fd_;
};
```

### Interface Design

**I.4: Make interfaces precisely and strongly typed**
- Use specific types instead of generic `int` or `void*`
- Let the type system enforce constraints

```cpp
// Good: strong types
struct RowIndex { size_t value; };
struct ColIndex { size_t value; };

double GetElement(const Matrix& m, RowIndex row, ColIndex col);

// Bad: ambiguous primitives
double GetElement(const Matrix& m, size_t row, size_t col);
// Easy to swap row and col by mistake
```

**I.5: State preconditions**
**I.7: State postconditions**
- Use `Expects()` and `Ensures()` for contracts (C++20 contracts coming in future standard)

```cpp
int Divide(int a, int b) {
  Expects(b != 0);  // Precondition
  int result = a / b;
  Ensures(result * b == a);  // Postcondition
  return result;
}
```

**I.23: Keep the number of function arguments low**
- Too many parameters suggest missing abstractions
- Consider creating a parameter object

```cpp
// Bad: too many parameters
void CreateWindow(int x, int y, int width, int height,
                  bool resizable, bool visible, const std::string& title);

// Good: parameter object
struct WindowConfig {
  Point position;
  Size dimensions;
  bool resizable = true;
  bool visible = true;
  std::string title;
};

void CreateWindow(const WindowConfig& config);
```

### Error Handling

**I.10: Use exceptions to signal a failure to perform a required task**
- Errors should not be ignorable
- Return values can be ignored; exceptions cannot

**Note:** This conflicts with Google's no-exceptions policy. For Ferric Continuum, we recommend exceptions for new code unless interfacing with Google libraries.

**P.7: Catch run-time errors early**
- Validate at the point of use
- Fail fast rather than propagating invalid state

### Concurrency

**CP.1: Assume that your code will run as part of a multi-threaded program**
- Design for thread safety from the start

**Data races are bugs**
- Use proper synchronization (mutexes, atomics)
- Remember: "You cannot have a race condition on immutable data"
- Prefer immutable data structures when possible

```cpp
// Good: immutable shared state
class Config {
public:
  Config(std::string name, int value)
    : name_(std::move(name)), value_(value) {}

  const std::string& name() const { return name_; }
  int value() const { return value_; }

private:
  const std::string name_;
  const int value_;
};

// Thread-safe: read-only access from multiple threads
std::shared_ptr<const Config> shared_config;
```

---

## Modern C++ Checklist

Use this checklist for code reviews and development:

### Memory & Ownership
- [ ] Use `std::unique_ptr` for exclusive ownership
- [ ] Use `std::shared_ptr` only when shared ownership is truly needed
- [ ] No raw `new`/`delete` in application code
- [ ] All resources managed via RAII (files, sockets, mutexes, etc.)
- [ ] Ownership is clear from types and signatures

### Type Safety
- [ ] Use strong types instead of primitive types for domain concepts
- [ ] Mark single-argument constructors `explicit`
- [ ] Use `const` for all immutable references and member functions
- [ ] Prefer value semantics and move operations
- [ ] Use `enum class` instead of plain `enum`

### Function Design
- [ ] Functions are small and focused (< 40-50 lines)
- [ ] Return values preferred over output parameters
- [ ] Parameters are `const&` for input, value for small types
- [ ] Preconditions and postconditions are documented/enforced

### Class Design
- [ ] Data members are `private` (or `protected` for inheritance)
- [ ] Copy/move constructors explicitly defined or deleted
- [ ] Virtual destructors for polymorphic base classes
- [ ] `override` or `final` used for virtual method overrides

### Modern C++ Features
- [ ] Use `auto` for complex types and iterators
- [ ] Use `constexpr` for compile-time constants and functions
- [ ] Use range-based `for` loops when iterating
- [ ] Use C++-style casts (`static_cast`, etc.)
- [ ] Use `nullptr` instead of `NULL` or `0`

### Code Organization
- [ ] Headers are self-contained and compile independently
- [ ] Include guards or `#pragma once` used
- [ ] "Include What You Use" (IWYU) followed
- [ ] Code is in namespaces (not global scope)
- [ ] No `using namespace` in headers

### Performance
- [ ] Move semantics used for expensive-to-copy types
- [ ] Reserve capacity for vectors when size is known
- [ ] Avoid unnecessary copies (pass by `const&` or value)
- [ ] Profile before optimizing - don't assume

### Concurrency (for multi-threaded code)
- [ ] Shared mutable state is properly synchronized
- [ ] Prefer immutable data for sharing between threads
- [ ] Avoid data races (use thread sanitizer to verify)
- [ ] Use RAII for lock management (`std::lock_guard`, `std::unique_lock`)

---

## Rust Best Practices

For Rust components in Ferric Continuum, follow these principles:

### Ownership & Borrowing
- Let the borrow checker guide your design
- Prefer borrowing (`&T`, `&mut T`) over ownership (`T`)
- Use `Rc`/`Arc` only when shared ownership is truly needed
- Avoid `RefCell`/`Mutex` unless interior mutability is required

### Error Handling
- Use `Result<T, E>` for recoverable errors
- Use `Option<T>` for optional values
- Propagate errors with `?` operator
- Reserve `panic!` for unrecoverable errors and bugs

```rust
// Good: composable error handling
fn read_config(path: &Path) -> Result<Config, io::Error> {
    let contents = fs::read_to_string(path)?;
    let config = parse_config(&contents)?;
    Ok(config)
}
```

### Type Safety
- Use newtype pattern for domain concepts
- Leverage `enum` for state machines and variants
- Make invalid states unrepresentable

```rust
// Good: invalid states impossible
struct UserId(u64);
struct UserName(String);

struct User {
    id: UserId,
    name: UserName,
}
```

### Performance
- Avoid allocations in hot paths
- Use iterators and `collect()` instead of manual loops
- Profile with `cargo bench` and `criterion`
- Use `#[inline]` judiciously for small, hot functions

### Async Code (Tokio)
- Prefer async/await over manual futures
- Avoid blocking operations in async contexts
- Use `tokio::spawn` for concurrent tasks
- Be careful with `Arc<Mutex<T>>` - consider message passing

---

## Tooling and Automation

Automated tooling ensures code quality, consistency, and reduces manual effort. Ferric Continuum uses industry-standard tools for formatting and linting.

### C++ Tools

#### clang-format

**Purpose**: Automatic code formatting for C++ files.

**Configuration**: `.clang-format` in repository root (based on Google style).

**Usage**:
```bash
# Format a single file
clang-format -i path/to/file.cpp

# Format all C++ files in project
find . -name "*.cpp" -o -name "*.h" -o -name "*.cc" -o -name "*.hpp" | xargs clang-format -i

# Check formatting without modifying (CI mode)
clang-format --dry-run --Werror path/to/file.cpp

# Use the provided script
./scripts/format.sh          # Format all files
./scripts/format.sh --check  # Check only (CI)
```

**Key Benefits**:
- Eliminates formatting debates
- Ensures consistency across the codebase
- Automatic fixes for most style issues

#### clangd

**Purpose**: C++ Language Server Protocol (LSP) implementation for IDE/editor integration.

**Features**:
- Code completion
- Go-to-definition
- Find references
- Real-time diagnostics
- Inline documentation

**Configuration**: `.clangd` in repository root.

**Setup**:
```bash
# Generate compile_commands.json for clangd
bazel run @hedron_compile_commands//:refresh_all

# Most editors auto-detect clangd; verify it's in your PATH
which clangd
```

**Editor Support**:
- **VS Code / Cursor / Trae**: Install "clangd" extension from marketplace
- **Emacs / Doom Emacs**: Use lsp-mode with clangd (see Editor Configuration section below)

#### clang-tidy

**Purpose**: C++ static analyzer and linter with auto-fix capabilities.

**Configuration**: `.clang-tidy` in repository root.

**Usage**:
```bash
# Run clang-tidy on a single file
clang-tidy path/to/file.cpp -- -I./include

# Run with auto-fix
clang-tidy --fix path/to/file.cpp -- -I./include

# Run on all files with compile_commands.json
clang-tidy -p . path/to/file.cpp

# Use the provided script
./scripts/lint.sh           # Lint and report issues
./scripts/lint.sh --fix     # Lint and auto-fix
```

**Enabled Checks** (see `.clang-tidy` for full list):
- `modernize-*` - Suggests modern C++ idioms (auto, nullptr, range-for)
- `readability-*` - Improves code readability
- `performance-*` - Identifies performance issues
- `bugprone-*` - Catches common bugs
- `cppcoreguidelines-*` - Enforces C++ Core Guidelines

**Auto-fixable Issues**:
- Use `nullptr` instead of `NULL`
- Use `auto` for obvious types
- Use range-based for loops
- Add `override` keywords
- Remove unnecessary copies

### Rust Tools

#### rustfmt

**Purpose**: Official Rust code formatter.

**Configuration**: `rustfmt.toml` in repository root.

**Usage**:
```bash
# Format all Rust code in project
cargo fmt

# Check formatting without modifying
cargo fmt -- --check

# Format specific files
rustfmt src/main.rs
```

**Integration**:
- Runs automatically in most Rust-aware editors
- Required in CI pipeline

#### clippy

**Purpose**: Official Rust linter with hundreds of lint rules and auto-fix support.

**Configuration**: `.clippy.toml` and `Cargo.toml` lint settings.

**Usage**:
```bash
# Run clippy on all targets
cargo clippy --all-targets --all-features

# Run with auto-fix
cargo clippy --fix --all-targets --all-features

# Deny warnings (CI mode)
cargo clippy -- -D warnings

# Explain a specific lint
cargo clippy -- -W clippy::lint-name --explain
```

**Key Lint Categories**:
- Correctness (deny by default) - Probable bugs
- Suspicious - Code that's likely wrong
- Style - Code style recommendations
- Complexity - Overly complex code patterns
- Performance - Performance issues
- Pedantic - Extra nitpicky lints (opt-in)

**Common Auto-fixable Issues**:
- Unnecessary clones
- Redundant closures
- Verbose iterator patterns
- Inefficient string operations

### Integrated Automation Scripts

#### scripts/format.sh

Formats all C++ and Rust code in the repository.

```bash
# Format all code
./scripts/format.sh

# Check formatting only (exits with error if formatting needed)
./scripts/format.sh --check

# Format specific language
./scripts/format.sh --cpp-only
./scripts/format.sh --rust-only
```

#### scripts/lint.sh

Runs all linters with optional auto-fix.

```bash
# Run all linters (report mode)
./scripts/lint.sh

# Run with auto-fix
./scripts/lint.sh --fix

# Run specific linter
./scripts/lint.sh --cpp-only
./scripts/lint.sh --rust-only
```

### Bazel Integration

The `.bazelrc` file includes:

```bash
# Build with all warnings
build --copt=-Wall --copt=-Wextra

# Run clang-tidy on C++ builds
build --aspects=@bazel_clang_tidy//clang_tidy:clang_tidy.bzl%clang_tidy_aspect
build --output_groups=report

# Format check target
bazel run //:format_check  # Check formatting
bazel run //:format        # Apply formatting
```

### Pre-commit Integration (Optional)

For automatic formatting/linting before commits:

```bash
# Install pre-commit hook
./scripts/install-hooks.sh

# Manually run pre-commit checks
./scripts/pre-commit.sh
```

**Hook Actions**:
1. Run clang-format on staged C++ files
2. Run rustfmt on staged Rust files
3. Run clang-tidy on modified C++ files
4. Run clippy on Rust workspace
5. Fail commit if issues found (or auto-fix if configured)

### CI/CD Integration

All formatting and linting tools run in continuous integration:

```yaml
# CI Pipeline Steps
- name: Check Formatting
  run: ./scripts/format.sh --check

- name: Run Linters
  run: ./scripts/lint.sh

- name: Run Clippy
  run: cargo clippy --all-targets -- -D warnings
```

### Tool Installation

**C++ Tools**:
```bash
# Ubuntu/Debian
sudo apt install clang-format clang-tidy clangd

# macOS
brew install clang-format llvm

# From source or LLVM releases
# https://releases.llvm.org/
```

**Rust Tools**:
```bash
# Install rustfmt and clippy (included with rustup)
rustup component add rustfmt clippy

# Update to latest versions
rustup update
```

### Best Practices

1. **Format before committing**: Run `./scripts/format.sh` before every commit
2. **Fix lint issues**: Address linter warnings; don't suppress unless justified
3. **Use auto-fix judiciously**: Review auto-fixes before committing
4. **Configure your editor**: Enable format-on-save for both C++ and Rust
5. **Keep tools updated**: Update clang-tools and Rust components regularly
6. **Document suppressions**: If suppressing a lint, add a comment explaining why

### Quick Reference: Daily Workflow

```bash
# 1. Before starting work - ensure tools are available
which clang-format clang-tidy clangd
cargo fmt --version
cargo clippy --version

# 2. During development - format frequently
./scripts/format.sh                    # Format all code
./scripts/format.sh --cpp-only         # C++ only
./scripts/format.sh --rust-only        # Rust only

# 3. Before committing - check and fix issues
./scripts/format.sh --check            # Verify formatting
./scripts/lint.sh                      # Check for linting issues
./scripts/lint.sh --fix                # Auto-fix what's possible

# 4. Review changes
git diff                               # Review all changes including auto-fixes

# 5. Bazel integration (optional)
bazel build --config=clippy //...      # Rust linting
bazel build --config=rustfmt //...     # Rust formatting check
bazel build --config=ci //...          # Full CI checks locally
```

### CI/CD Pipeline Integration

```yaml
# Example CI pipeline configuration
name: Code Quality

on: [push, pull_request]

jobs:
  format-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install C++ tools
        run: sudo apt install clang-format clang-tidy

      - name: Install Rust tools
        run: rustup component add rustfmt clippy

      - name: Check formatting
        run: ./scripts/format.sh --check

      - name: Run linters
        run: ./scripts/lint.sh

      - name: Build with CI config
        run: bazel build --config=ci //...
```

### Common Auto-Fix Examples

#### C++ Auto-Fixes (clang-tidy)

```cpp
// Before: Using NULL instead of nullptr
void* ptr = NULL;

// After: Modern C++ (auto-fixed by clang-tidy)
void* ptr = nullptr;
```

```cpp
// Before: Missing override keyword
class Base {
  virtual void foo() {}
};

class Derived : public Base {
  void foo() {}  // Warning: missing override
};

// After: Auto-fixed
class Derived : public Base {
  void foo() override {}
};
```

```cpp
// Before: Unnecessary copy
for (std::string item : vec) {  // Copies each string
  process(item);
}

// After: Auto-fixed to use const reference
for (const std::string& item : vec) {
  process(item);
}
```

```cpp
// Before: Old-style loop
for (size_t i = 0; i < vec.size(); ++i) {
  process(vec[i]);
}

// After: Modernized (auto-fixed)
for (auto& item : vec) {
  process(item);
}
```

#### Rust Auto-Fixes (clippy)

```rust
// Before: Unnecessary clone
let s = String::from("hello");
let t = s.clone();
println!("{}", s);

// After: Auto-fixed - remove unnecessary clone
let s = String::from("hello");
let t = &s;
println!("{}", s);
```

```rust
// Before: Inefficient string operation
let s = format!("Hello");

// After: Auto-fixed
let s = "Hello".to_string();
```

```rust
// Before: Redundant closure
let doubled: Vec<i32> = vec.iter().map(|x| x * 2).collect();

// After: Auto-fixed (when appropriate)
let doubled: Vec<i32> = vec.iter().map(|x| x * 2).collect();
// Note: clippy suggests improvements based on context
```

### Troubleshooting

**Issue**: `clang-format: command not found`
```bash
# Ubuntu/Debian
sudo apt install clang-format

# macOS
brew install clang-format

# Verify installation
clang-format --version
```

**Issue**: `clang-tidy` requires `compile_commands.json`
```bash
# Generate compilation database with Bazel
bazel run @hedron_compile_commands//:refresh_all

# Or use bear for non-Bazel projects
bear -- make
```

**Issue**: Rust tools not found
```bash
# Install rustup if needed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Add components
rustup component add rustfmt clippy

# Update to latest
rustup update
```

**Issue**: Format/lint scripts fail on CI
```bash
# Ensure scripts are executable
chmod +x scripts/format.sh scripts/lint.sh

# Test locally first
./scripts/format.sh --check
./scripts/lint.sh
```

**Issue**: Too many false positives from clang-tidy
```bash
# Disable specific checks in .clang-tidy
# Or suppress inline with NOLINT comment
void legacy_function() {  // NOLINT(modernize-use-trailing-return-type)
  // Justification: Legacy API compatibility
}
```

### Performance Tips

**For large codebases:**
```bash
# Run formatters in parallel on specific directories
find cpp/ -name "*.cpp" | xargs -P 4 clang-format -i

# Lint only changed files
git diff --name-only --diff-filter=ACM | grep -E '\.(cpp|cc|h|hpp)$' | xargs clang-tidy --fix

# Cache clang-tidy results
export CLANG_TIDY_CACHE_DIR=~/.cache/clang-tidy
```

**For faster feedback during development:**
```bash
# Format only staged files
git diff --cached --name-only --diff-filter=ACM | grep -E '\.(cpp|cc|h|hpp)$' | xargs clang-format -i

# Lint only current directory
./scripts/lint.sh --cpp-only $(pwd)
```

### Editor Configuration Examples

#### VS Code / Cursor / Trae

Create or update `.vscode/settings.json` in your project root:

```json
{
  "editor.formatOnSave": true,
  "editor.defaultFormatter": null,

  // C++ Configuration
  "C_Cpp.codeAnalysis.clangTidy.enabled": true,
  "C_Cpp.codeAnalysis.clangTidy.path": "/usr/bin/clang-tidy",
  "C_Cpp.default.configurationProvider": "llvm-vs-code-extensions.vscode-clangd",

  // clangd LSP
  "clangd.path": "/usr/bin/clangd",
  "clangd.arguments": [
    "--background-index",
    "--clang-tidy",
    "--completion-style=detailed",
    "--header-insertion=iwyu",
    "--pch-storage=memory"
  ],

  // Rust Configuration
  "[rust]": {
    "editor.defaultFormatter": "rust-lang.rust-analyzer",
    "editor.formatOnSave": true
  },
  "rust-analyzer.check.command": "clippy",
  "rust-analyzer.checkOnSave": true,
  "rust-analyzer.rustfmt.extraArgs": [
    "+nightly"
  ],

  // File associations
  "files.associations": {
    "*.bazel": "starlark",
    "BUILD": "starlark",
    "WORKSPACE": "starlark"
  }
}
```

**Required Extensions:**
- **C++**: [clangd](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd)
- **Rust**: [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer)
- **Bazel**: [Bazel](https://marketplace.visualstudio.com/items?itemName=BazelBuild.vscode-bazel)

**Install Extensions:**
```bash
code --install-extension llvm-vs-code-extensions.vscode-clangd
code --install-extension rust-lang.rust-analyzer
code --install-extension BazelBuild.vscode-bazel
```

#### Emacs (Doom Emacs)

Add to your Doom Emacs configuration (`~/.doom.d/config.el`):

```elisp
;; C++ Configuration with clangd and clang-format
(after! lsp-mode
  (setq lsp-clients-clangd-args
        '("--background-index"
          "--clang-tidy"
          "--completion-style=detailed"
          "--header-insertion=iwyu"
          "--pch-storage=memory"))

  ;; Enable clangd for C/C++
  (setq lsp-clients-clangd-executable "/usr/bin/clangd"))

;; Auto-format on save for C/C++
(add-hook 'c-mode-hook
          (lambda ()
            (add-hook 'before-save-hook #'clang-format-buffer nil 'local)))

(add-hook 'c++-mode-hook
          (lambda ()
            (add-hook 'before-save-hook #'clang-format-buffer nil 'local)))

;; Rust Configuration with rust-analyzer
(after! rustic
  (setq rustic-format-on-save t)
  (setq rustic-lsp-server 'rust-analyzer)
  (setq rustic-analyzer-command '("rust-analyzer"))

  ;; Use clippy for checks
  (setq rustic-flycheck-clippy-params "--all-targets --all-features"))

;; Bazel support
(use-package! bazel
  :mode (("\\.bazel\\'" . bazel-mode)
         ("BUILD\\'" . bazel-mode)
         ("WORKSPACE\\'" . bazel-mode)))
```

**Required Doom modules** (add to `~/.doom.d/init.el`):

```elisp
(doom! :lang
       (cc +lsp)           ; C/C++ with LSP
       (rust +lsp)         ; Rust with LSP

       :tools
       lsp                 ; Language Server Protocol
       format              ; Auto-formatting

       :checkers
       syntax)             ; Syntax checking
```

**Install required packages:**
```bash
# Ensure clang-format package is available
doom sync
doom install

# For clang-format buffer command
M-x package-install RET clang-format RET
```

**Alternative: using format-all (simpler)**

```elisp
;; In config.el
(use-package! format-all
  :commands format-all-mode
  :hook (prog-mode . format-all-mode)
  :config
  (setq-default format-all-formatters
                '(("C" (clang-format))
                  ("C++" (clang-format))
                  ("Rust" (rustfmt)))))
```

---

## Recommendations for This Project

Given Ferric Continuum's goals (performance, HPC, distributed computing), we recommend:

### C++ Components
1. **Follow C++ Core Guidelines as primary reference**
   - Allow exceptions (unlike Google style) for error handling
   - Use RTTI sparingly, but don't ban it entirely
   - Target C++20, prepare for C++23 features

2. **Google Style Guide for code organization**
   - Namespace conventions
   - Header file practices
   - Naming conventions

3. **Performance-critical code**
   - Profile before optimizing
   - Use compiler-specific attributes when beneficial
   - Document performance assumptions and constraints

### Rust Components
1. **Standard Rust idioms**
   - Follow Clippy recommendations
   - Use `rustfmt` for formatting
   - Run `cargo clippy` in CI

2. **Async runtime (Tokio)**
   - Use for I/O-bound operations (network, file I/O)
   - Avoid for CPU-bound work (use thread pools instead)

3. **FFI with C++**
   - Use `cxx` crate for safe C++/Rust interop
   - Minimize boundary crossings
   - Document ownership transfer clearly

### Testing & Quality
- Unit tests for all components
- Integration tests for agent interactions
- Benchmark suite for performance tracking
- Use sanitizers (ASan, TSan, UBSan) in CI
- Document performance characteristics

### Documentation
- Document design decisions in code comments
- Use Doxygen for C++ API docs
- Use rustdoc for Rust API docs
- Maintain architectural decision records (ADRs)

---

## References

- [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)
- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- [Effective Modern C++](https://www.oreilly.com/library/view/effective-modern-c/9781491908419/) by Scott Meyers
- [The Rust Programming Language](https://doc.rust-lang.org/book/)
