# Lesson 08 — Attention

**Concept:** single-head scaled dot-product attention,
`out = softmax(Q Kᵀ / √d) V`.

The capstone of the teaching track and a **thin demo/test over `cuda_kernels`**:
the kernel lives in `//ferric_continuum/cuda_kernels:attention` (one block per
query row: scaled scores → stable softmax → weighted sum of V) and mirrors
`tnsr`'s toy attention. The demo runs a tiny QKV; the test compares against a CPU
reference.

## Files

| File | Purpose |
|------|---------|
| `attention_demo.cu` | Runs attention on a small random QKV, prints outputs. |
| `attention_test.cu` | gtest: matches a CPU reference on a tiny QKV. |

## Run

```bash
bazel run  --config=cuda //ferric_continuum/cuda_gym/lessons/08_attention:attention_demo
bazel test --config=cuda //ferric_continuum/cuda_gym/lessons/08_attention:attention_test
```
