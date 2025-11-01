# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Ferric Continuum** is a high-performance computing system combining C++ and Rust for parallel workloads. The architecture is based on autonomous "agent" modules inspired by the Culture series, each responsible for specific aspects of the build, test, and deployment pipeline.

## Build System

This project uses **Bazel** as the primary build system, with **Spack** for managing external dependencies and toolchains.

### Common Commands

```bash
# Build all targets
bazel build //...

# Build specific C++ target
bazel build //cpp:matrix_mul

# Build specific Rust target
bazel build //rust:parallel_sort

# Run tests
bazel test //...

# Run specific test
bazel test //cpp:matrix_mul_test

# Clean build artifacts
bazel clean
```

### Spack Environment Management

```bash
# Activate the Ferric Continuum environment
spack env activate ferric-env

# Install core dependencies (CUDA, BLAS, LAPACK, MPI)
spack install cuda openblas lapack

# View installed packages
spack find
```

## Architecture

The system is organized around five key agents:

### 1. **BuildMind** (Python)
- Configures and compiles C++/Rust targets using Bazel
- Interfaces with Spack-provisioned compilers and libraries
- Entry point: `//agents:buildmind`

### 2. **PerfSmith** (Rust)
- Runs benchmarks and gathers profiling data
- Tracks performance regressions
- Uses Google Benchmark (C++) and Criterion.rs (Rust)
- Entry point: `//agents:perfsmith`

### 3. **AsyncHermes** (Rust)
- Manages distributed task orchestration
- Handles message passing between nodes
- Built on Tokio and gRPC/Tonic
- Entry point: `//agents:asynchromes`

### 4. **SpackSentinel** (Python)
- Monitors Spack environments
- Synchronizes toolchain packages for Bazel builds
- Manages CUDA, BLAS, LAPACK, MPI dependencies

### 5. **ContinuumSupervisor** (Python)
- Meta-controller coordinating all agents
- Orchestrates experiments and generates reports
- Entry point: `//agents:continuum_supervisor`

## Development Principles

- **Hermetic builds**: All builds must be reproducible and isolated
- **Observability**: Each agent exposes metrics, logs, and telemetry
- **API-first**: Every agent provides both CLI and API interfaces
- **Containerized**: Agents run in Bazel containers or Kubernetes pods
- **Async communication**: Agents communicate through async channels, local sockets, or the Bazel workspace graph

## Coding Standards

All C++ and Rust code must follow the engineering standards documented in [ENGINEERING.md](ENGINEERING.md). Key highlights:

- **C++**: Follow C++ Core Guidelines with Google Style Guide conventions for code organization
- **Rust**: Follow standard Rust idioms with Clippy recommendations
- **Memory Safety**: Use RAII, smart pointers, and ownership semantics
- **Type Safety**: Prefer strong types over primitives
- **Testing**: Unit tests required for all components, benchmarks for performance-critical code

Refer to ENGINEERING.md for detailed best practices, checklists, and examples.

## Workflow Integration

The typical development workflow follows this pipeline:

1. **SpackSentinel** provisions toolchains and libraries
2. **BuildMind** compiles and links artifacts
3. **PerfSmith** profiles and benchmarks them
4. **AsyncHermes** deploys and manages distributed workloads
5. **ContinuumSupervisor** oversees the pipeline and aggregates metrics

## Code Organization

The repository is structured by agent and language:

- `/agents/` - Agent implementations (buildmind, perfsmith, asynchromes, spacksentinel, continuum_supervisor)
- `/cpp/` - C++ performance-critical components (e.g., matrix operations)
- `/rust/` - Rust parallel algorithms and async runtime components

## Testing and Benchmarking

```bash
# Run all tests
bazel test //...

# Run specific test
bazel test //cpp:matrix_mul_test

# Run performance suite with baseline comparison
bazel run //agents:perfsmith -- --compare=baseline.json

# Run specific benchmarks
bazel run //cpp:matrix_mul_benchmark
bazel run //rust:parallel_sort_benchmark

# Deploy distributed workload across 4 nodes
bazel run //agents:asynchromes -- --nodes=4 --role=trainer
```

**Testing Frameworks:**
- **C++**: Google Test and Google Benchmark
- **Rust**: Built-in testing and Criterion.rs

**Performance Profiling Tools:**
- C++: perf, valgrind, gprof, Intel VTune
- Rust: cargo flamegraph, perf
- PerfSmith agent aggregates and tracks results across runs

See [ENGINEERING.md § Testing and Benchmarking Standards](ENGINEERING.md#testing-and-benchmarking-standards) for detailed examples and best practices.

## Code Quality Tools

Quick reference for formatting and linting. See [ENGINEERING.md § Tooling and Automation](ENGINEERING.md#tooling-and-automation) for comprehensive documentation.

### Formatting

```bash
# Format all code (C++ and Rust)
./scripts/format.sh

# Check formatting (CI mode)
./scripts/format.sh --check
```

### Linting

```bash
# Run all linters
./scripts/lint.sh

# Run with auto-fix
./scripts/lint.sh --fix
```

### Bazel Integration

```bash
# Run Clippy on Rust code
bazel build --config=clippy //...

# Run Rust formatting check
bazel build --config=rustfmt //...

# CI mode (warnings as errors)
bazel build --config=ci //...
```

### Tool Installation

```bash
# C++ Tools (Ubuntu/Debian)
sudo apt install clang-format clang-tidy clangd

# C++ Tools (macOS)
brew install clang-format llvm

# Rust Tools (included with rustup)
rustup component add rustfmt clippy
```

**Editor Configuration:** See [ENGINEERING.md § Editor Configuration](ENGINEERING.md#editor-configuration-examples) for VS Code, Cursor, Trae, and Emacs setup.

## Toolchain Requirements

- Bazel (build system)
- Spack (package manager for HPC dependencies)
- C++17 or later compiler
- Rust stable toolchain
- CUDA (for GPU workloads)
- MPI (for distributed computing)
- OpenBLAS/LAPACK (for linear algebra)
