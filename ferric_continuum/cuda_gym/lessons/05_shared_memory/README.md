# Lesson 05 — Shared Memory

**Concept:** staging reused data in fast, block-local shared memory.

`y = M x` with one block per row: the block cooperatively loads tiles of the
shared vector `x` into shared memory, so every thread reuses each loaded chunk
instead of re-reading `x` from global memory. Global loads of `x` drop from
`threads × cols` to `cols`. Each thread accumulates a strided partial dot
product; a tree reduction then combines them into `y[row]`.

## Files

| File | Purpose |
|------|---------|
| `shared_memory.hh` / `shared_memory.cu` | `MatVec`. |
| `shared_memory_demo.cu` | `y = M·[1,1,1,1]` (row sums). |
| `shared_memory_test.cu` | gtest: row sums, host match with wide cols, mismatch. |

## Run

```bash
bazel run  --config=cuda //ferric_continuum/cuda_gym/lessons/05_shared_memory:shared_memory_demo
bazel test --config=cuda //ferric_continuum/cuda_gym/lessons/05_shared_memory:shared_memory_test
```
