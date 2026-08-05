# Lesson 04 — Reduction

**Concept:** a two-level parallel sum.

1. **Tree reduction in shared memory:** each block loads its slice into shared
   memory and reduces it by halving the stride each step, with a
   `__syncthreads()` between steps (`blockDim.x` is a power of two).
2. **Cross-block combine with atomics:** thread 0 of each block `atomicAdd`s its
   partial sum into a single global accumulator, avoiding a second kernel launch.

## Files

| File | Purpose |
|------|---------|
| `reduction.hh` / `reduction.cu` | `Sum` over a device array. |
| `reduction_demo.cu` | Sums ~1M elements and compares to the host sum. |
| `reduction_test.cu` | gtest: sum of ones, host match, empty, single element. |

## Run

```bash
bazel run  --config=cuda //ferric_continuum/cuda_gym/lessons/04_reduction:reduction_demo
bazel test --config=cuda //ferric_continuum/cuda_gym/lessons/04_reduction:reduction_test
```
