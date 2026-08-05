# Lesson 01 — Hello GPU

**Concept:** device properties + the first kernel launch.

You learn how to:

- query the CUDA device you are running on (`cudaGetDeviceProperties`), and
- launch your first kernel — SAXPY, `y = a*x + y` — with an explicit
  grid/block geometry, host↔device copies, and post-launch error checking.

SAXPY is the canonical "hello world" of GPU compute and the seed the rest of the
gym grows from. (This lesson absorbs the old `ferric_continuum/cuda/saxpy`
example.)

## Files

| File | Purpose |
|------|---------|
| `hello_gpu.hh` | `QueryDevice()` + `Saxpy()` declarations. |
| `hello_gpu.cu` | Device query and the SAXPY kernel/launcher. |
| `hello_gpu_demo.cu` | Prints device info and runs SAXPY (Abseil logging). |
| `hello_gpu_test.cu` | gtest: device is queryable, SAXPY correctness, edge cases. |

## Run

```bash
bazel run  --config=cuda //ferric_continuum/cuda_gym/lessons/01_hello_gpu:hello_gpu_demo
bazel test --config=cuda //ferric_continuum/cuda_gym/lessons/01_hello_gpu:hello_gpu_test
```
