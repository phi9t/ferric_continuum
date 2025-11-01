# Ferric Continuum - Quick Start Guide

## What We've Built

A working example demonstrating C++ and Rust collocated in the same directory, built with Bazel.

## Key Technologies

### Build System
- **Bazel 8.4.2** with Bzlmod (modern module system)
- Hermetic, reproducible builds
- Support for C++20, Rust 2021, Python 3.12

### C++ Stack
- **C++20** standard
- **Abseil** (Google's C++ library) for utilities
- **GoogleTest** for unit testing
- **LLVM 18.1.8** toolchain
- Header files use **`.hh`** extension
- Include guards use **`#pragma once`**

### Rust Stack
- **Rust 1.81.0** (edition 2021)
- Tokio, Tonic for async/distributed
- Criterion for benchmarking
- Built-in testing with `#[cfg(test)]`

### Python Stack
- **Python 3.12** (default)
- Spack for HPC library management
- FastAPI, Pandas, Polars

## Build and Run

### Build Everything
```bash
bazel build //...
```

### Build C++ Hello World
```bash
bazel build //ferric_continuum/hello:hello_cc
```

### Build Rust Hello World
```bash
bazel build //ferric_continuum/hello:hello_rs
```

### Run C++ Program
```bash
bazel run //ferric_continuum/hello:hello_cc
```

**Output:**
```
Hello, World! Welcome to Ferric Continuum (C++ Edition with Abseil)

Fibonacci sequence (first 10 numbers):
fib(0) = 0
fib(1) = 1
...
fib(9) = 34

Prime numbers up to 50:
2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47
```

### Run Rust Program
```bash
bazel run //ferric_continuum/hello:hello_rs
```

**Output:**
```
Hello, World! Welcome to Ferric Continuum (Rust Edition)

Fibonacci sequence (first 10 numbers):
fib(0) = 0
fib(1) = 1
...
fib(9) = 34

Prime numbers up to 50:
2 3 5 7 11 13 17 19 23 29 31 37 41 43 47
```

## Testing

### Run All Tests
```bash
bazel test //ferric_continuum/hello/...
```

### Run C++ Tests Only
```bash
bazel test //ferric_continuum/hello:hello_lib_cc_test
```

### Run Rust Tests Only
```bash
bazel test //ferric_continuum/hello:hello_lib_rs_test
```

**Test Results:**
```
✓ C++ Tests: 5/5 passed (Fibonacci, IsPrime, PrimesUpTo, GenerateGreeting, FormatNumberList)
✓ Rust Tests: 4/4 passed (fibonacci, is_prime, primes_up_to, generate_greeting)
```

## Key Features Demonstrated

### 1. Collocated Multi-Language Development
- C++ and Rust in the same directory
- Shared `BUILD.bazel` managing both languages
- Side-by-side comparison of implementations

### 2. Modern C++ with Abseil
```cpp
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

std::string generate_greeting(absl::string_view name) {
  return absl::StrCat("Hello, ", name, "! Welcome to Ferric Continuum");
}

std::string format_number_list(const std::vector<uint64_t>& numbers) {
  return absl::StrJoin(numbers, ", ");
}
```

### 3. Idiomatic Rust
```rust
pub fn generate_greeting(name: &str) -> String {
    format!("Hello, {}! Welcome to Ferric Continuum", name)
}

pub fn primes_up_to(n: u64) -> Vec<u64> {
    (2..=n).filter(|&i| is_prime(i)).collect()
}
```

### 4. Comprehensive Testing
- **C++**: GoogleTest framework with separate test files
- **Rust**: Built-in testing with `#[test]` attributes

### 5. Code Organization Standards
- **C++ headers**: `.hh` extension with `#pragma once`
- **C++ source**: `.cc` extension
- **Rust**: Standard `.rs` extension
- **Namespace**: All C++ code in `namespace ferric`

## Build Configuration Highlights

### `.bazelrc` Features
```bash
# C++ standard
build --cxxopt=-std=c++20

# Rust edition
build --@rules_rust//:rust_edition=2021

# Python version
build --@rules_python//python/config_settings:python_version=3.12

# Optimization profiles
build:opt --compilation_mode=opt --copt=-O3
build:debug --compilation_mode=dbg --copt=-g

# Sanitizers
build:asan --copt=-fsanitize=address --linkopt=-fsanitize=address
build:tsan --copt=-fsanitize=thread --linkopt=-fsanitize=thread

# Rust tooling
build:clippy --aspects=@rules_rust//rust:defs.bzl%rust_clippy_aspect
build:rustfmt --aspects=@rules_rust//rust:defs.bzl%rustfmt_aspect
```

## Troubleshooting

### Clean Build
```bash
bazel clean
```

### Rebuild Everything
```bash
bazel clean && bazel build //...
```

### View Dependency Graph
```bash
bazel query --output=graph //ferric_continuum/hello/... > graph.dot
```

### Check for Linting Issues
```bash
# Rust
bazel build --config=clippy //ferric_continuum/hello/...

# C++ (requires clang-tidy setup)
bazel build --config=asan //ferric_continuum/hello/...
```

## Performance Tips

- Use `--config=opt` for optimized builds
- Use `--config=fastbuild` for faster iteration
- Enable remote caching for larger projects
- Adjust `--jobs=N` for parallel builds

## Documentation

- **Architecture**: See `AGENTS.md`
- **Full Setup**: See `README.md`
- **Engineering Guide**: See `ENGINEERING.md`
- **Module Details**: See `ferric_continuum/hello/README.md`

---
