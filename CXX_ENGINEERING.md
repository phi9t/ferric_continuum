# CXX_ENGINEERING.md

C++ engineering fundamentals for Ferric Continuum. This is the C++-only guide; see `ENGINEERING.md` for cross-language standards, tooling, and Rust guidance.

---

## Core Principles

- **Correctness first**: prefer clarity over cleverness.
- **Explicit ownership**: types should make ownership obvious.
- **RAII everywhere**: tie resources to object lifetime.
- **Small, focused functions**: favor readability and testability.
- **Observable behavior**: use structured logging, not raw stdout.

---

## Code Organization

### Namespaces

- Never use `using namespace` (especially in headers).
- Prefer explicit qualification or local `using` declarations in `.cc` files.

```cpp
// Good: explicit qualification
std::string name = ferric::foundation::MakeName();

// Acceptable in .cc files only
using std::string;
```

### Headers

- Headers use the `.hh` extension.
- Use `#pragma once`.
- Keep headers self-contained and include what you use.

```cpp
#pragma once

#include <string>

namespace ferric::foundation {

class Widget {
 public:
  explicit Widget(std::string name);
  const std::string& name() const;

 private:
  std::string name_;
};

}  // namespace ferric::foundation
```

---

## Ownership and Memory

- Prefer `std::unique_ptr` for exclusive ownership.
- Use `std::shared_ptr` only when shared ownership is required.
- Avoid raw `new`/`delete` in application code.
- Use `const` for immutable data and references.

```cpp
std::unique_ptr<Widget> CreateWidget() {
  return std::make_unique<Widget>();
}
```

---

## Function and Variable Design

- Keep functions small (roughly 40 lines or less).
- Prefer returning values over output parameters.
- Narrow variable scope and initialize at declaration.

```cpp
std::vector<double> ComputeResults(const Matrix& input) {
  std::vector<double> results;
  // ... compute
  return results;  // Move semantics apply
}
```

---

## Class Design

- Use `explicit` for single-argument constructors.
- Define or delete copy/move operations deliberately.
- Use `override` for virtual overrides.

```cpp
class Buffer {
 public:
  explicit Buffer(size_t size);

  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  Buffer(Buffer&&) = default;
  Buffer& operator=(Buffer&&) = default;
};
```

---

## Modern C++ Features

- Use `auto` for complex types and iterators when the type is obvious.
- Use `constexpr` for compile-time constants and functions.
- Prefer C++-style casts (`static_cast`, `const_cast`, `reinterpret_cast`).
- Use `nullptr` instead of `NULL` or `0`.

```cpp
constexpr int Square(int x) {
  return x * x;
}

std::array<int, Square(10)> buffer;
```

---

## Error Handling

- Prefer `absl::Status` / `absl::StatusOr` for recoverable errors.
- Use `CHECK` / `LOG(FATAL)` for invariants that must not fail.
- Avoid exceptions unless required by an external API.

---

## Concurrency

- Assume code may run in a multi-threaded context.
- Avoid shared mutable state; prefer immutable data or explicit synchronization.
- Use RAII for lock management (`std::lock_guard`, `std::unique_lock`).

---

## Logging

Use Abseil logging for all non-trivial C++ code.

```cpp
#include "absl/log/log.h"
#include "absl/log/initialize.h"
#include "absl/log/globals.h"

int main() {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  LOG(INFO) << "Program started";
}
```

---

## Testing

- Unit tests use GoogleTest.
- Keep tests small and deterministic.
- Prefer value-based assertions over behavioral coupling.

```cpp
#include "gtest/gtest.h"
#include "my_module.h"

TEST(MyModuleTest, BasicFunctionality) {
  MyClass obj;
  EXPECT_EQ(obj.compute(5), 25);
  EXPECT_TRUE(obj.is_valid());
}
```

---

## Benchmarking

Use Google Benchmark for performance measurements.

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

---

## Modern C++ Checklist (Quick)

### Memory and Ownership
- [ ] Use `std::unique_ptr` for exclusive ownership
- [ ] Use `std::shared_ptr` only when shared ownership is needed
- [ ] No raw `new`/`delete` in application code
- [ ] All resources managed via RAII
- [ ] Ownership is clear from types and signatures

### Type Safety
- [ ] Use strong types instead of primitive types for domain concepts
- [ ] Mark single-argument constructors `explicit`
- [ ] Use `const` for immutable references and member functions
- [ ] Prefer value semantics and move operations
- [ ] Use `enum class` instead of plain `enum`

### Function Design
- [ ] Functions are small and focused
- [ ] Return values preferred over output parameters
- [ ] Parameters are `const&` for input, value for small types
- [ ] Preconditions and postconditions are documented or enforced

### Class Design
- [ ] Data members are `private` (or `protected` for inheritance)
- [ ] Copy/move constructors explicitly defined or deleted
- [ ] Virtual destructors for polymorphic base classes
- [ ] `override` used for virtual method overrides

### Code Organization
- [ ] Headers are self-contained and use `#pragma once`
- [ ] Include what you use
- [ ] Code lives in namespaces (not global scope)
- [ ] No `using namespace` in headers

### Performance
- [ ] Move semantics used for expensive-to-copy types
- [ ] Reserve capacity for vectors when size is known
- [ ] Avoid unnecessary copies (pass by `const&` or value)
- [ ] Profile before optimizing

### Concurrency
- [ ] Shared mutable state is synchronized
- [ ] Prefer immutable data for sharing between threads
- [ ] Avoid data races (use thread sanitizer)
- [ ] RAII for locks (`std::lock_guard`, `std::unique_lock`)

---

## References

- `ENGINEERING.md` for tooling and cross-language standards.
- Google C++ Style Guide: https://google.github.io/styleguide/cppguide.html
- C++ Core Guidelines: https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines
