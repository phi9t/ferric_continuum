# Lesson 03 — Indexing

**Concept:** mapping threads to data in 1D and 2D.

- **1D:** the global index is `blockIdx.x * blockDim.x + threadIdx.x`. A bounds
  guard (`if (i < n)`) handles the ragged final block when `n` is not a multiple
  of the block size — `Iota(300)` with 256-thread blocks exercises exactly this.
- **2D:** `threadIdx`/`blockIdx` have `.x` and `.y`. By convention x → column,
  y → row for a row-major matrix, giving `out[row*cols + col]`.

## Files

| File | Purpose |
|------|---------|
| `indexing.hh` / `indexing.cu` | `Iota` (1D) and `RowMajorIndices` (2D). |
| `indexing_demo.cu` | Prints a 1D iota and a small 2D index grid. |
| `indexing_test.cu` | gtest: flat map correctness, 2D map, ragged last block. |

## Run

```bash
bazel run  --config=cuda //ferric_continuum/cuda_gym/lessons/03_indexing:indexing_demo
bazel test --config=cuda //ferric_continuum/cuda_gym/lessons/03_indexing:indexing_test
```
