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
- **CUDA gym** in `ferric_continuum/cuda_gym` (lessons + challenges) and shared kernels in `ferric_continuum/cuda_kernels`, with an opt-in GPU forward path for `tnsr`.
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

### CUDA Gym (opt-in, GPU)

CUDA is disabled by default so CPU-only builds and CI stay hermetic. Enable it
with `--config=cuda` on a machine with a CUDA toolkit and GPU.

**Prerequisites:** a locally-installed CUDA toolkit (auto-detected via
`CUDA_PATH` or `/usr/local/cuda`) and a CUDA-capable GPU to *run* tests/demos.
The default arch list covers Ampere→Blackwell including **sm_100 (B200)**; note
CUDA 13.x dropped Volta (`compute_70`). Trim to your GPU for faster builds, e.g.
`--config=cuda --cuda_archs=compute_100:sm_100`.

```bash
# Shared production kernels / lessons (wildcards work under --config=cuda)
bazel test --config=cuda //ferric_continuum/cuda_kernels/...
bazel test --config=cuda //ferric_continuum/cuda_gym/lessons/...

# Lesson 01 demo
bazel run  --config=cuda //ferric_continuum/cuda_gym/lessons/01_hello_gpu:hello_gpu_demo

# Challenge self-check (green). Student :grade fails until stubs are filled.
bazel test --config=cuda //ferric_continuum/cuda_gym/challenges/vector_add:grade_reference

# tnsr GPU forward (matmul + softmax)
bazel test --config=cuda //ferric_continuum/tnsr:cuda_forward_tests
```

See `ferric_continuum/cuda_gym/README.md` for the full lesson order, challenge
workflow, and architecture flags (`--cuda_archs=...`).

---

## Repository Layout

```
ferric_continuum/
├── hello/                 # C++/Rust hello world example
├── foundation/            # Core C++/Rust concepts with demos and tests
├── cuda_gym/              # CUDA lessons + graded challenges
├── cuda_kernels/          # Shared production GEMM / softmax / attention kernels
├── tnsr/                  # Transformer autograd library (optional CUDA fwd)
└── optimizers/muon/       # Muon optimizer (pybind11 + numpy)
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
