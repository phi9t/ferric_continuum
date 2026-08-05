# Ferric CUDA Kernels

Reference-quality FP32 CUDA kernels with a small **C ABI** surface. They serve
two consumers:

- the advanced gym lessons (`cuda_gym/lessons/06_gemm`, `07_softmax`,
  `08_attention`), which are thin demos/tests over these kernels; and
- `tnsr`'s opt-in GPU forward path (feature `cuda`), which links the C ABI via
  Rust FFI.

## Layout

| Path | Contents |
|------|----------|
| `include/ferric/cuda/*.h` | Pure-C public headers (`status.h`, `gemm.h`, `softmax.h`, `attention.h`). |
| `common/` | `cuda_check.hh` (`FERRIC_CUDA_CHECK`), `status.hh`, RAII `DeviceBuffer`. |
| `gemm/` | Naive + tiled shared-memory GEMM. |
| `softmax/` | Stable row softmax over the last dim. |
| `attention/` | Single-head scaled QKᵀ·softmax·V. |

## C ABI

All entry points are `extern "C"`, take **row-major FP32 host pointers**, and
return a `FerricCudaStatus`:

```c
FerricCudaStatus ferric_cuda_gemm_f32(int m, int n, int k,
                                      const float* a, const float* b, float* c);
FerricCudaStatus ferric_cuda_softmax_f32(int rows, int cols,
                                         const float* x, float* out);
FerricCudaStatus ferric_cuda_attention_f32(int seq, int dim, const float* q,
                                           const float* k, const float* v,
                                           float* out);
```

Contract:

- Shape/pointer validation at the boundary → `FERRIC_CUDA_ERR_INVALID_ARG`
  (negative dims, or null pointers for a positive-sized problem).
- Any zero dimension is a no-op success.
- CUDA runtime failures → `FERRIC_CUDA_ERR_DEVICE` (never silently ignored;
  launches are followed by `cudaGetLastError` + `cudaDeviceSynchronize`).

GEMM matches `tnsr`'s `raw_linear_forward` (A:`m×k`, B:`k×n`, C:`m×n`); softmax
matches `raw_softmax` (stable, over the last dim).

## Build & test

CUDA is opt-in. Targets are tagged `cuda`; default tag filters exclude them
from CPU `//...`. On a CUDA machine with a GPU:

```bash
bazel build --config=cuda //ferric_continuum/cuda_kernels/...
bazel test  --config=cuda //ferric_continuum/cuda_kernels/...
```

Each test compares the kernel against a CPU reference at the fp32 tolerance
`rtol 1e-4 / atol 1e-5`.
