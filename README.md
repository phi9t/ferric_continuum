# Ferric Continuum

*Forging performance through parallelism and precision — in C++ and Rust.*

Ferric Continuum is a multi-language systems playground built with Bazel. It focuses on side-by-side C++/Rust examples, clear teaching artifacts, and a Python/C++ optimizer prototype.

The repo name nods to a line in *Use of Weapons* about Minds blurring the boundary between tactics and strategy:

> “The Minds did not assume such distinctions; to them, there was no cut-off between the two. Tactics cohered into strategy, strategy disintegrated into tactics, in the sliding scale of their dialectical moral algebra.”

In that spirit, this codebase blends C++, Rust, and Python so tensor work can flow across the stack without hard cutoffs between systems-level kernels, safe concurrency, and high-level orchestration.

---

## What Lives Here

- **Collocated C++ and Rust examples** in `ferric_continuum/hello` and `ferric_continuum/foundation`.
- **Foundation modules** covering value semantics, move semantics, parameter passing, smart pointers/RAII, and constructor rules.
- **Muon optimizer prototype** in `ferric_continuum/optimizers/muon` using a C++ backend exposed to Python via pybind11.
- **CUDA example** in `ferric_continuum/cuda` (opt-in SAXPY kernel via `rules_cuda`).
- **Bazel-first workflows** for builds, tests, and demos.

---

## Quick Start

```bash
# Build everything
bazel build //...

# Run all tests
bazel test //...
```

---

## Examples

### Hello World

```bash
bazel run //ferric_continuum/hello:hello_cc
bazel run //ferric_continuum/hello:hello_rs
```

### Foundation Demos

```bash
bazel run //ferric_continuum/foundation:value_semantics_demo_cc
bazel run //ferric_continuum/foundation:value_semantics_demo_rs

bazel run //ferric_continuum/foundation:move_semantics_demo_cc
bazel run //ferric_continuum/foundation:move_semantics_demo_rs

bazel run //ferric_continuum/foundation:parameter_passing_demo_cc
bazel run //ferric_continuum/foundation:parameter_passing_demo_rs

bazel run //ferric_continuum/foundation:smart_pointers_demo_cc
bazel run //ferric_continuum/foundation:smart_pointers_demo_rs

bazel run //ferric_continuum/foundation:constructor_rules_demo_cc
```

### Muon Optimizer (Python + C++)

```bash
# Run the demo
bazel run //ferric_continuum/optimizers/muon:muon_demo

# Run the Python test
bazel test //ferric_continuum/optimizers/muon:muon_py_test
```

### CUDA (opt-in, GPU)

CUDA is disabled by default so CPU-only builds and CI stay hermetic. Enable it
with `--config=cuda` on a machine with a CUDA toolkit:

```bash
# Build the SAXPY kernel (no GPU needed to compile).
bazel build --config=cuda //ferric_continuum/cuda:saxpy

# Run the test (requires a GPU).
bazel test --config=cuda //ferric_continuum/cuda:saxpy_test
```

See `ferric_continuum/cuda/README.md` for architecture selection and compiler
options.

---

## Repository Layout

```
ferric_continuum/
├── hello/                 # C++/Rust hello world example
├── foundation/            # Core C++/Rust concepts with demos and tests
├── cuda/                  # Opt-in CUDA example (SAXPY via rules_cuda)
└── optimizers/muon/        # Muon optimizer (pybind11 + numpy)
```

---

## Documentation

- `AGENTS.md` - Agent roadmap and design notes (planned system)
- `CXX_ENGINEERING.md` - C++ engineering fundamentals (short guide)
- `ENGINEERING.md` - Coding standards and tooling guidance
- `ferric_continuum/hello/README.md` - Hello world walkthrough
- `ferric_continuum/foundation/README.md` - Foundation module deep dive

---

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

---

**Ferric Continuum** — *Forging performance through parallelism and precision.*
