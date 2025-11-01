# Ferric Continuum

*Forging performance through parallelism and precision — in C++ and Rust.*

A high-performance computing system combining C++ and Rust, featuring distributed workloads, benchmarking, and autonomous agent-based orchestration.

---

## What We've Built

A working HPC system demonstrating:
- **C++ and Rust collocated** in the same directories, organized by functionality
- **Foundation libraries** illustrating core systems programming concepts (value semantics, move semantics, smart pointers, RAII, constructor rules)
- **Hermetic builds** with Bazel for reproducible, isolated compilation
- **Comprehensive testing** with 74 unit tests (100% passing)
- **Modern tooling** with structured logging, formatting, and linting

---

## Key Technologies

### Build System
- **Bazel 8.4.2** with Bzlmod (modern module system)
- Hermetic, reproducible builds
- Support for C++20, Rust 2021, Python 3.12

### C++ Stack
- **C++20** standard
- **Abseil** (Google's C++ library) for utilities and structured logging
- **GoogleTest** for unit testing
- **LLVM 18.1.8** toolchain
- Header files use **`.hh`** extension
- Include guards use **`#pragma once`**
- **Never use `using namespace`** declarations

### Rust Stack
- **Rust 1.81.0** (edition 2021)
- **Tracing** crate for structured logging
- Tokio, Tonic for async/distributed computing
- Criterion for benchmarking
- Built-in testing with `#[cfg(test)]`

### Python Stack
- **Python 3.12** (default)
- Spack for HPC library management
- FastAPI, Pandas, Polars

---

## Prerequisites

- **Bazel 8.4.2+** - Hermetic build system ([install](https://bazel.build/install))
- **Python 3.12+** - For Python-based agents
- **C++ compiler** - Clang 16+ or GCC 13+ (C++20 support required)
- **Rust toolchain** - Managed automatically by Bazel
- **Spack** (optional) - HPC library management (CUDA, MPI, BLAS)

### Installation

```bash
# Check Bazel version
bazel --version  # Should be 8.4.2 or newer

# Install Bazelisk (recommended for version management)
# macOS
brew install bazelisk

# Linux
npm install -g @bazel/bazelisk
```

---

## Quick Start

### Build Everything

```bash
# Build all targets
bazel build //...

# Build with optimizations
bazel build --config=opt //...

# Build in debug mode
bazel build --config=debug //...
```

### Run Tests

```bash
# Run all tests
bazel test //...

# Run with verbose output
bazel test --test_output=all //...

# Run with coverage
bazel coverage //...
```

**Test Results:**
- ✅ **74 unit tests** (100% passing)
- ✅ C++ tests using GoogleTest
- ✅ Rust tests using built-in test framework

---

## Examples

### Hello World (C++)

```bash
# Build and run C++ hello world
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

### Hello World (Rust)

```bash
# Build and run Rust hello world
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

### Foundation Libraries

Explore foundational C++ and Rust concepts:

```bash
# Value semantics (copy vs move)
bazel run //ferric_continuum/foundation:value_semantics_demo_cc
bazel run //ferric_continuum/foundation:value_semantics_demo_rs

# Move semantics
bazel run //ferric_continuum/foundation:move_semantics_demo_cc
bazel run //ferric_continuum/foundation:move_semantics_demo_rs

# Parameter passing
bazel run //ferric_continuum/foundation:parameter_passing_demo_cc
bazel run //ferric_continuum/foundation:parameter_passing_demo_rs

# Smart pointers & RAII
bazel run //ferric_continuum/foundation:smart_pointers_demo_cc
bazel run //ferric_continuum/foundation:smart_pointers_demo_rs

# Constructor rules (Rule of Five)
bazel run //ferric_continuum/foundation:constructor_rules_demo_cc
```

See `ferric_continuum/foundation/README.md` for detailed documentation.

---

## Development Workflow

### Building C++ Components

```bash
# Build a specific C++ target
bazel build //ferric_continuum/hello:hello_cc

# Run C++ tests
bazel test //ferric_continuum/hello:hello_lib_cc_test

# Build with address sanitizer
bazel build --config=asan //ferric_continuum/hello:hello_cc

# Build with thread sanitizer
bazel build --config=tsan //ferric_continuum/hello:hello_cc
```

### Building Rust Components

```bash
# Build a specific Rust target
bazel build //ferric_continuum/hello:hello_rs

# Run Rust tests
bazel test //ferric_continuum/hello:hello_lib_rs_test

# Run Clippy (Rust linter)
bazel build --config=clippy //ferric_continuum/...

# Run Rustfmt
bazel build --config=rustfmt //ferric_continuum/...
```

### Code Quality

```bash
# Format all C++ code
find ferric_continuum -name "*.cc" -o -name "*.hh" | xargs clang-format -i

# Format all Rust code
find ferric_continuum -name "*.rs" | xargs rustfmt --edition 2021

# Or use the convenience scripts
./scripts/format.sh
./scripts/lint.sh
```

---

## Key Features Demonstrated

### 1. Collocated Multi-Language Development

C++ and Rust code organized by functionality, not by language:

```
ferric_continuum/foundation/
├── BUILD.bazel              # Builds both C++ and Rust
├── value_semantics.hh       # C++ header
├── value_semantics.cc       # C++ implementation
├── value_semantics_demo.cc  # C++ demo
├── value_semantics_test.cc  # C++ tests
├── value_semantics.rs       # Rust implementation
├── value_semantics_demo.rs  # Rust demo
└── README.md                # Comparative documentation
```

### 2. Modern C++ with Abseil

```cpp
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/log/log.h"

std::string generate_greeting(absl::string_view name) {
  return absl::StrCat("Hello, ", name, "! Welcome to Ferric Continuum");
}

std::string format_number_list(const std::vector<uint64_t>& numbers) {
  return absl::StrJoin(numbers, ", ");
}

int main() {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  
  LOG(INFO) << "Program started";
  // ... program logic ...
  return 0;
}
```

### 3. Idiomatic Rust with Tracing

```rust
use tracing::{info, Level};
use tracing_subscriber;

pub fn generate_greeting(name: &str) -> String {
    format!("Hello, {}! Welcome to Ferric Continuum", name)
}

pub fn primes_up_to(n: u64) -> Vec<u64> {
    (2..=n).filter(|&i| is_prime(i)).collect()
}

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();
    
    info!("Program started");
    // ... program logic ...
}
```

### 4. Comprehensive Testing

**C++ with GoogleTest:**
```cpp
#include "gtest/gtest.h"
#include "value_semantics.hh"

TEST(ValueSemanticsTest, CopyConstructor) {
  ferric::foundation::Buffer original(10);
  ferric::foundation::Buffer copy(original);
  EXPECT_EQ(copy.size(), 10);
}
```

**Rust with built-in testing:**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fibonacci() {
        assert_eq!(fibonacci(0), 0);
        assert_eq!(fibonacci(1), 1);
        assert_eq!(fibonacci(10), 55);
    }
}
```

### 5. Coding Standards Enforcement

- **C++ headers**: `.hh` extension with `#pragma once`
- **C++ source**: `.cc` extension
- **Namespace**: All C++ code in `namespace ferric`
- **No `using namespace`** declarations in C++
- **Structured logging**: Abseil (C++) and tracing (Rust)
- **Formatting**: clang-format (C++) and rustfmt (Rust)
- **Linting**: clang-tidy (C++) and clippy (Rust)

---

## Running Agents

```bash
# BuildMind - Build orchestration
bazel run -- //agents/build_mind --targets=//ferric_continuum/matrix:matrix_mul_cc

# PerfSmith - Benchmarking
bazel run -- //agents/perf_smith --compare=baseline.json

# AsyncHermes - Distributed workload management
bazel run -- //agents/async_hermes --nodes=4 --role=trainer

# ContinuumSupervisor - Meta-controller
bazel run -- //agents/continuum_supervisor --summary
```

---

## Integration with Spack

Spack provides HPC libraries (CUDA, BLAS, LAPACK, MPI) for high-performance computing:

```bash
# Create and activate Spack environment
spack env create ferric_continuum
spack env activate ferric_continuum

# Install HPC libraries
spack add openblas
spack add lapack
spack add openmpi

# Find installed packages
spack find

# SpackSentinel monitors and syncs these with Bazel
bazel run -- //agents:spack_sentinel --sync
```

---

## Configuration

### Bazel Configuration (.bazelrc)

The `.bazelrc` file provides build configurations:

**Optimization profiles:**
- `--config=opt` - Optimized builds (-O3)
- `--config=fastbuild` - Fast builds for iteration
- `--config=debug` - Debug builds with symbols

**Sanitizers:**
- `--config=asan` - Address sanitizer
- `--config=tsan` - Thread sanitizer

**Rust tooling:**
- `--config=clippy` - Rust linting
- `--config=rustfmt` - Rust formatting

**Example configuration:**
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
```

### User-Specific Configuration

Create `.bazelrc.user` for personal overrides (ignored by git):

```bash
# Example .bazelrc.user
build --jobs=16
build --remote_cache=grpc://localhost:9092
```

### Update Python Dependencies

```bash
# Update requirements_lock.txt from requirements.txt
bazel run //:requirements.update
```

---

## Troubleshooting

### Clean Build

```bash
# Clean build artifacts
bazel clean

# Deep clean (removes all cached data)
bazel clean --expunge
```

### Rebuild Everything

```bash
bazel clean && bazel build //...
```

### View Dependency Graph

```bash
bazel query --output=graph //ferric_continuum/hello/... > graph.dot
```

### Check for Issues

```bash
# Run linters
./scripts/lint.sh

# Format code
./scripts/format.sh

# Run tests with verbose output
bazel test --test_output=all //...
```

---

## Performance Tips

- Use `--config=opt` for optimized builds (-O3)
- Use `--config=fastbuild` for faster iteration during development
- Enable remote caching for larger projects
- Adjust `--jobs=N` for parallel builds (default: CPU cores)
- Use `bazel build --config=asan` to catch memory errors early

---

## Documentation

- **[AGENTS.md](AGENTS.md)** - Agent architecture and 10-week curriculum
- **[ENGINEERING.md](ENGINEERING.md)** - Comprehensive coding standards and tooling
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to contribute
- **[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)** - Community guidelines
- **[SECURITY.md](SECURITY.md)** - Security policy
- **[CHANGELOG.md](CHANGELOG.md)** - Version history
- **[ferric_continuum/foundation/README.md](ferric_continuum/foundation/README.md)** - Foundation libraries documentation

---

## Development Principles

- **Hermetic builds** - All builds are reproducible and isolated
- **Collocated code** - C++ and Rust organized by functionality, not language
- **Comprehensive testing** - Every function has unit tests
- **Structured logging** - Use Abseil (C++) and tracing (Rust), never raw stdout
- **Code quality** - Automated formatting and linting enforced
- **Documentation** - Every public API must be documented

See [ENGINEERING.md](ENGINEERING.md) for complete coding standards.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Ferric Continuum** — *Forging performance through parallelism and precision.*
