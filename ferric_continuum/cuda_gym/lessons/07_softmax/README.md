# Lesson 07 — Softmax

**Concept:** the numerically-stable row softmax used throughout attention.

A **thin demo/test over `cuda_kernels`**: the stable row-softmax kernel lives in
`//ferric_continuum/cuda_kernels:softmax` (max-subtraction, per-row block
reduction) and is shared with `tnsr`. The demo softmaxes two rows and confirms
each sums to ≈ 1.

## Files

| File | Purpose |
|------|---------|
| `softmax_demo.cu` | Softmaxes two rows, prints probabilities and row sums. |
| `softmax_test.cu` | gtest: row sums to 1, monotonic input → monotonic output. |

## Run

```bash
bazel run  --config=cuda //ferric_continuum/cuda_gym/lessons/07_softmax:softmax_demo
bazel test --config=cuda //ferric_continuum/cuda_gym/lessons/07_softmax:softmax_test
```
