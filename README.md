# Ferric Continuum

[![CI](https://github.com/USERNAME/ferric_continuum/workflows/CI/badge.svg)](https://github.com/USERNAME/ferric_continuum/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Bazel](https://img.shields.io/badge/Build-Bazel_8.4.2-43A047.svg)](https://bazel.build/)
[![C++](https://img.shields.io/badge/C++-17/20-00599C.svg?logo=c%2B%2B)](https://isocpp.org/)
[![Rust](https://img.shields.io/badge/Rust-1.81-CE422B.svg?logo=rust)](https://www.rust-lang.org/)
[![Code Style](https://img.shields.io/badge/code_style-Google-4285F4.svg)](https://google.github.io/styleguide/cppguide.html)

**Tagline:** *Forging performance through parallelism and precision — in C++ and Rust.*

A high-performance computing system that combines C++ and Rust for distributed workloads, benchmarking, and autonomous agent-based orchestration.

> 🚀 **Featured:** Comprehensive foundation libraries comparing C++ and Rust approaches to systems programming
> 
> 📚 **Learning Resource:** 10-week curriculum teaching HPC concepts in both languages
> 
> ⚙️ **Production Ready:** 74 unit tests, 100% passing, comprehensive documentation

---

## Quick Links

- [📖 Documentation](#documentation)
- [🚀 Quick Start](#quick-start)
- [🏗️ Architecture](#agent-architecture)
- [🤝 Contributing](CONTRIBUTING.md)
- [📜 License](LICENSE)
- [📋 Code of Conduct](CODE_OF_CONDUCT.md)

---

## Overview

Ferric Continuum is an HPC system featuring autonomous agents that manage:

- **Building and testing** C++ and Rust components via Bazel
- **Dependency management** and toolchains via Spack
- **Distributed workloads** and performance profiling
- **Benchmarking** and continuous experimentation

Each agent operates as a self-contained subsystem, capable of acting independently or collaborating with others—inspired by the specialized Minds of the Culture universe.

---

## Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes
- **[AGENTS.md](AGENTS.md)** - Agent architecture and 10-week curriculum
- **[ENGINEERING.md](ENGINEERING.md)** - Coding standards and best practices
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to contribute
- **[TEST_COVERAGE_SUMMARY.md](TEST_COVERAGE_SUMMARY.md)** - Comprehensive test coverage report

---

## Features

### 🎯 Foundation Libraries
Comprehensive C++ and Rust implementations demonstrating:
- **Value Semantics** - Copy vs. move semantics
- **Move Semantics** - Efficient resource transfer
- **Parameter Passing** - Best practices for function parameters
- **Smart Pointers & RAII** - Automatic resource management
- **Constructor Rules** - Rule of Five and Rule of Zero

### ✅ Production Quality
- **74 test cases** with 100% pass rate
- **Structured logging** (Abseil for C++, tracing for Rust)
- **Formatted code** (clang-format, rustfmt)
- **Comprehensive documentation** for every API

### 🔧 Modern Build System
- **Bazel 8.4.2** for hermetic, reproducible builds
- **Bzlmod** for dependency management
- **Spack integration** for HPC libraries
- **Multi-language support** (C++, Rust, Python)

---

## Project Structure

```
ferric_continuum/
├── MODULE.bazel              # Bazel module dependencies (Bzlmod)
├── BUILD.bazel               # Root build configuration
├── .bazelrc                  # Bazel build flags and options
├── .bazelversion             # Pinned Bazel version
├── requirements.txt          # Python dependencies
├── requirements_lock.txt     # Locked Python dependencies
├── AGENTS.md                 # Agent architecture and curriculum
├── agents/                   # Agent implementations
│   ├── build_mind/           # Build orchestration (Python)
│   ├── perf_smith/           # Benchmarking and profiling (Rust)
│   ├── async_hermes/         # Distributed task management (Rust)
│   ├── spack_sentinel/      # Toolchain monitoring (Python)
│   └── continuum_supervisor/ # Meta-controller (Python)
├── ferric_continuum/         # Core libraries (C++ and Rust collocated)
│   ├── hello/               # Hello World examples (hello.cc, hello.rs)
│   ├── matrix/              # Matrix operations (matrix_mul.cc, matrix_mul.rs)
│   ├── parallel/            # Parallel algorithms (both languages)
│   ├── distributed/         # Distributed computing primitives
│   └── common/              # Shared utilities and data structures
└── examples/                 # Example projects and tutorials
    ├── week1_hello/         # Week 1: Hello World in both languages
    ├── week2_build/         # Week 2: Multi-language builds
    └── ...                  # Additional curriculum examples
```

---

## Design Philosophy: Collocated C++ and Rust

Ferric Continuum organizes code **by functionality rather than by language**. C++ and Rust implementations live side-by-side in the same directories, enabling:

- **Direct comparison** of language approaches to the same problem
- **Shared interfaces** and interoperability patterns
- **Unified documentation** for multi-language features
- **Natural learning progression** comparing both implementations

Example structure:
```
ferric_continuum/matrix/
├── BUILD.bazel           # Defines both C++ and Rust targets
├── matrix_mul.cc         # C++ implementation
├── matrix_mul.hh         # C++ header
├── matrix_mul_test.cc    # C++ tests
├── matrix_mul.rs         # Rust implementation
├── matrix_mul_test.rs    # Rust tests
└── README.md             # Comparative docs for both
```

This approach reflects real-world polyglot systems where multiple languages collaborate within the same codebase.

---

## Prerequisites

- **Bazel 8.4.2+** - Install from [bazel.build](https://bazel.build/install)
- **Python 3.12+** - For agent development
- **Rust 1.81.0+** - Managed by Bazel (automatic)
- **C++ compiler** - Clang 18+ or GCC 13+ (managed by Bazel)
- **Spack** (optional) - For HPC library management

---

## Quick Start

### 1. Verify Bazel Installation

```bash
# Check Bazel version
bazel --version  # Should be 8.4.2 or newer

# If not installed, use Bazelisk (recommended):
# macOS
brew install bazelisk

# Linux
npm install -g @bazel/bazelisk
```

### 2. Build All Targets

```bash
# Build everything
bazel build //...

# Build with optimizations
bazel build --config=opt //...

# Build in debug mode
bazel build --config=debug //...
```

### 3. Run Tests

```bash
# Run all tests
bazel test //...

# Run with verbose output
bazel test --test_output=all //...

# Run with coverage
bazel coverage //...
```

### 4. Update Python Dependencies

```bash
# Update requirements_lock.txt from requirements.txt
bazel run //:requirements.update
```

---

## Development Workflow

### Building C++ Components

```bash
# Build a specific C++ target
bazel build //ferric_continuum/matrix:matrix_mul_cc

# Run C++ tests
bazel test //ferric_continuum/matrix:matrix_mul_cc_test

# Build with address sanitizer
bazel build --config=asan //ferric_continuum/matrix:matrix_mul_cc
```

### Building Rust Components

```bash
# Build a specific Rust target
bazel build //ferric_continuum/parallel:parallel_sort_rs

# Run Rust tests
bazel test //ferric_continuum/parallel:parallel_sort_rs_test

# Run Clippy (Rust linter)
bazel build --config=clippy //ferric_continuum/...

# Run Rustfmt
bazel build --config=rustfmt //ferric_continuum/...
```

### Running Agents

```bash
# BuildMind - Build orchestration
bazel run -- //agents/build_mind --targets=//ferric_continuum/matrix:matrix_mul_cc //ferric_continuum/parallel:parallel_sort_rs

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

# Find what packages are installed
spack find

# Install HPC libraries
spack add openblas
spack add lapack
spack add openmpi

# SpackSentinel monitors and syncs these with Bazel
bazel run -- //agents:spack_sentinel --sync
```

---

## Configuration

### Bazel Configuration (.bazelrc)

The `.bazelrc` file provides build configurations:

- `--config=opt` - Optimized builds (-O3)
- `--config=debug` - Debug builds with symbols
- `--config=asan` - Address sanitizer
- `--config=tsan` - Thread sanitizer
- `--config=clippy` - Rust linting
- `--config=rustfmt` - Rust formatting

### User-Specific Configuration

Create `.bazelrc.user` for personal overrides (ignored by git):

```bash
# Example .bazelrc.user
build --jobs=16
build --remote_cache=grpc://localhost:9092
```

---

## Benchmarking

### C++ Benchmarks (Google Benchmark)

```bash
# Run C++ benchmarks
bazel run //ferric_continuum/matrix:matrix_mul_cc_benchmark

# Generate benchmark report
bazel run //ferric_continuum/matrix:matrix_mul_cc_benchmark -- --benchmark_format=json > results.json
```

### Rust Benchmarks (Criterion)

```bash
# Run Rust benchmarks
bazel run -- //ferric_continuum/parallel:parallel_sort_rs_benchmark

# Criterion automatically generates HTML reports
open bazel-bin/ferric_continuum/parallel/parallel_sort_rs_benchmark/criterion/report/index.html
```

---

## Observability

All agents expose metrics in Prometheus format and emit structured logs:

```bash
# Start observability dashboard
bazel run -- //observability:dashboard

# View metrics at http://localhost:9090
# View Grafana at http://localhost:3000
```

---

## 10-Week Learning Curriculum

See [AGENTS.md](AGENTS.md) for the complete 10-week curriculum covering:

- Bazel and Spack setup
- C++ and Rust interoperability
- Parallel algorithms and concurrency
- Benchmarking and profiling
- Distributed systems with gRPC
- CI/CD and observability
- Meta-controller development

---

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on how to get started.

- 📝 **[Code of Conduct](CODE_OF_CONDUCT.md)** - Community guidelines
- 🐛 **[Report a Bug](https://github.com/USERNAME/ferric_continuum/issues/new?template=bug_report.md)** - File a bug report
- 💡 **[Request a Feature](https://github.com/USERNAME/ferric_continuum/issues/new?template=feature_request.md)** - Suggest an enhancement
- 🔒 **[Security Policy](SECURITY.md)** - Report security vulnerabilities

This project follows hermetic, reproducible build practices:

1. All builds must pass: `bazel build //...`
2. All tests must pass: `bazel test //...`
3. C++ code follows Google C++ Style Guide
4. Rust code must pass `clippy` and `rustfmt`
5. Python code must pass `mypy`, `black`, and `ruff`

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

Inspired by the autonomous Minds of Iain M. Banks' Culture universe, where specialized intelligences collaborate to solve complex problems.

- Built with [Bazel](https://bazel.build/), [Abseil](https://abseil.io/), and [Tokio](https://tokio.rs/)
- Following [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) and [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)

*"The Culture is a society formed from about eight or ten different humanoid races, living in artificial habitats and worldships scattered throughout the galaxy."*

---

## Contact

- **Issues**: [GitHub Issues](https://github.com/USERNAME/ferric_continuum/issues)
- **Discussions**: [GitHub Discussions](https://github.com/USERNAME/ferric_continuum/discussions)

---

<div align="center">

**Ferric Continuum** — *Forging performance through parallelism and precision.*

Made with ❤️ in C++ and Rust

[![Stars](https://img.shields.io/github/stars/USERNAME/ferric_continuum?style=social)](https://github.com/USERNAME/ferric_continuum/stargazers)
[![Forks](https://img.shields.io/github/forks/USERNAME/ferric_continuum?style=social)](https://github.com/USERNAME/ferric_continuum/network/members)

</div>

