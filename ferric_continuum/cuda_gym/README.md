# CUDA Gym

Progressive CUDA track for Ferric Continuum: lessons, graded challenges, and
shared kernels (`//ferric_continuum/cuda_kernels`) that also feed the optional
`tnsr` GPU forward path.

## Prerequisites

- NVIDIA GPU + installed CUDA toolkit (`nvcc` on `PATH` or `CUDA_PATH` set)
- Enable CUDA for Bazel: **`--config=cuda`**

CUDA targets are tagged `cuda` / `requires-gpu`. Default `.bazelrc` filters
(`-cuda`) keep them out of CPU `//...`; `--config=cuda` clears those filters
so package wildcards work. Student `:grade` also carries Bazel's `manual` tag
(wildcards skip it; an explicit label still runs — and fails until solved).

```bash
# All production kernel tests
bazel test --config=cuda //ferric_continuum/cuda_kernels/...

# All lessons
bazel test --config=cuda //ferric_continuum/cuda_gym/lessons/...

# A lesson demo
bazel run  --config=cuda //ferric_continuum/cuda_gym/lessons/01_hello_gpu:hello_gpu_demo

# Challenge self-check (green). Student :grade is opt-in / expected-fail.
bazel test --config=cuda //ferric_continuum/cuda_gym/challenges/vector_add:grade_reference
bazel test --config=cuda //ferric_continuum/cuda_gym/challenges/vector_add:grade

# tnsr GPU forward (matmul + softmax)
bazel test --config=cuda //ferric_continuum/tnsr:cuda_forward_tests
```

Override architecture for faster local builds, e.g. B200 / sm_100:

```bash
bazel test --config=cuda --cuda_archs=compute_100:sm_100 //ferric_continuum/cuda_kernels:gemm_test
```

## Lessons

| # | Package | Concept |
|---|---------|---------|
| 01 | `lessons/01_hello_gpu` | Device query + first kernel (SAXPY) |
| 02 | `lessons/02_memory` | H2D / D2H round-trip |
| 03 | `lessons/03_indexing` | 1D / 2D thread mapping |
| 04 | `lessons/04_reduction` | Parallel reduction |
| 05 | `lessons/05_shared_memory` | Shared-memory tiling |
| 06 | `lessons/06_gemm` | GEMM (via `cuda_kernels`) |
| 07 | `lessons/07_softmax` | Stable row softmax |
| 08 | `lessons/08_attention` | Single-head attention primitive |

Each lesson: library (or kernel wrapper) + demo binary + gtest. See per-lesson `README.md`.

## Challenges

Fill-in track under `challenges/`. Each has `student.cu` (stubs), `reference.cu`,
`cases.json`, and:

- `:grade_reference` — reference vs itself (CI / wildcard green path)
- `:grade` — student vs reference (fails until you implement `student.cu`;
  tagged `manual` so wildcards skip it; run the label explicitly to grade)

| Challenge | Maps to lessons |
|-----------|-----------------|
| `vector_add` | 01–03 |
| `reduce` | 04–05 |
| `gemm` | 06 |
| `softmax` | 07 |

## Shared kernels + tnsr

Production FP32 host-pointer APIs live in `//ferric_continuum/cuda_kernels`
(`ferric_cuda_gemm_f32`, `ferric_cuda_softmax_f32`, attention). With
`--config=cuda`, `tnsr` enables crate feature `cuda` and routes matmul / softmax
**forward** through those kernels; backward stays on the existing CPU path.

Force CPU at runtime without rebuilding: `FERRIC_TNSR_DEVICE=cpu`.
