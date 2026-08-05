# Lesson 02 — Memory

**Concept:** the device memory lifecycle — `cudaMalloc`, `cudaMemcpy` (H2D and
D2H), and `cudaFree`.

- `RoundTrip` copies a host buffer to the device and straight back: a plain
  round-trip is **bit-exact**.
- `RoundTripScaled` scales each element on the device so the traversal is
  observable, showing the H2D → compute → D2H pattern every later kernel uses.

## Files

| File | Purpose |
|------|---------|
| `memory.hh` / `memory.cu` | `RoundTrip` and `RoundTripScaled`. |
| `memory_demo.cu` | Round-trips and scales a small buffer. |
| `memory_test.cu` | gtest: bit-exact round-trip, scaling, empty input. |

## Run

```bash
bazel run  --config=cuda //ferric_continuum/cuda_gym/lessons/02_memory:memory_demo
bazel test --config=cuda //ferric_continuum/cuda_gym/lessons/02_memory:memory_test
```
