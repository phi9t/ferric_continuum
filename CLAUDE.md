# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Project Overview

Ferric Continuum is a Bazel-based, multi-language playground focused on side-by-side C++/Rust examples and a Python/C++ optimizer prototype. The agent architecture is **planned** but not yet implemented.

## Build System

Use **Bazel** for all builds and tests.

### Common Commands

```bash
# Build everything
bazel build //...

# Run all tests
bazel test //...
```

### Targeted Runs

```bash
# Hello world examples
bazel run //ferric_continuum/hello:hello_cc
bazel run //ferric_continuum/hello:hello_rs

# Foundation demos
bazel run //ferric_continuum/foundation:value_semantics_demo_cc
bazel run //ferric_continuum/foundation:value_semantics_demo_rs

# Muon optimizer demo
bazel run //ferric_continuum/optimizers/muon:muon_demo

# CUDA gym (opt-in; needs toolkit + GPU)
bazel test --config=cuda //ferric_continuum/cuda_kernels/...
bazel run  --config=cuda //ferric_continuum/cuda_gym/lessons/01_hello_gpu:hello_gpu_demo
```

### Targeted Tests

```bash
# Hello tests
bazel test //ferric_continuum/hello:hello_lib_cc_test
bazel test //ferric_continuum/hello:hello_lib_rs_test

# Foundation tests
bazel test //ferric_continuum/foundation:all

# Muon Python test
bazel test //ferric_continuum/optimizers/muon:muon_py_test

# CUDA (GPU-tagged; skip on CPU CI)
bazel test --config=cuda //ferric_continuum/cuda_gym/lessons/...
bazel test --config=cuda //ferric_continuum/tnsr:cuda_forward_tests
```

## Repository Layout

- `ferric_continuum/hello` - C++/Rust hello world example
- `ferric_continuum/foundation` - Core concepts with demos and tests
- `ferric_continuum/optimizers/muon` - Muon optimizer (pybind11 + numpy)

## Python Dependencies

Python dependencies are managed via `requirements.txt` and `requirements_lock.txt`. Muon tests rely on `numpy` and `pytest` via Bazel's pip integration.

## Coding Standards

Follow `ENGINEERING.md` for C++ and Rust best practices, logging, and testing conventions.

## Agent Architecture (Planned)

The agent system is documented in `AGENTS.md`, but no `/agents` directory or agent binaries exist in this repo yet.
