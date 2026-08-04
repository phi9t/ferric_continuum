# CUDA Example

A minimal SAXPY (`y = a*x + y`) kernel that exercises the
[`rules_cuda`](https://github.com/bazel-contrib/rules_cuda) toolchain wired into
this repo. It doubles as a template for adding your own GPU kernels.

## Why it's opt-in

CUDA is **disabled by default** so that CPU-only development and CI stay
hermetic and require no toolkit. The default build sets:

```
build --@rules_cuda//cuda:enable=False
```

and the targets here are tagged `manual`, so `bazel build //...` and
`bazel test //...` skip them. Enable CUDA with `--config=cuda`.

## Prerequisites

- A CUDA toolkit installed locally (auto-detected via `CUDA_PATH` or
  `/usr/local/cuda`), or a hermetic redist configured in `MODULE.bazel` (see the
  commented `cuda.redist_json` block there).
- A CUDA-capable GPU to *run* the test (build does not need one).

## Build & run

```bash
# Compile the kernel library (no GPU required to build).
bazel build --config=cuda //ferric_continuum/cuda:saxpy

# Run the test (requires a GPU at runtime).
bazel test --config=cuda //ferric_continuum/cuda:saxpy_test

# Target a specific architecture to speed up compilation, e.g. Ampere:
bazel build --config=cuda --cuda_archs=compute_80:compute_80,sm_80 \
  //ferric_continuum/cuda:saxpy

# Use clang instead of nvcc as the CUDA compiler:
bazel build --config=cuda --cuda_compiler=clang //ferric_continuum/cuda:saxpy
```

## Files

| File | Purpose |
|------|---------|
| `saxpy.hh` | Host-side declaration of `Saxpy`. |
| `saxpy.cu` | CUDA kernel + host launcher with explicit error handling. |
| `saxpy_test.cu` | GoogleTest coverage (correctness, empty input, length mismatch). |
| `BUILD.bazel` | `cuda_library` / `cuda_test` targets. |

## Configuration knobs

Defined in `.bazelrc` (flag aliases) and `MODULE.bazel` (toolchain):

| Flag / alias | Meaning |
|--------------|---------|
| `--config=cuda` | Enable CUDA and set default archs/compiler. |
| `--cuda_archs=...` | Override the target GPU architectures. |
| `--cuda_compiler=nvcc\|clang` | Select the device compiler. |
| `--cuda_enable=True\|False` | Toggle rules_cuda directly. |
