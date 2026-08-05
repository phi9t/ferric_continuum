# Lesson 06 — GEMM

**Concept:** dense matrix multiply `C = A·B`, and the payoff of tiling.

This lesson is a **thin demo/test over `cuda_kernels`** — the naive and tiled
GEMM kernels live in `//ferric_continuum/cuda_kernels:gemm` and are shared with
`tnsr`'s GPU path. The demo runs both variants on a 256³ problem and prints wall
time so you can compare naive vs shared-memory tiled.

## Files

| File | Purpose |
|------|---------|
| `gemm_demo.cu` | Runs naive + tiled GEMM, prints result and timing. |
| `gemm_test.cu` | gtest: a small hand-checked product via the shared kernel. |

## Run

```bash
bazel run  --config=cuda //ferric_continuum/cuda_gym/lessons/06_gemm:gemm_demo
bazel test --config=cuda //ferric_continuum/cuda_gym/lessons/06_gemm:gemm_test
```

See `//ferric_continuum/cuda_kernels` for the kernel implementations and their
golden tests.
