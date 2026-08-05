# CUDA Gym Design

**Date:** 2026-08-04  
**Status:** Approved for implementation planning  
**Repo:** Ferric Continuum  

## Summary

Add a **CUDA gym** to Ferric Continuum: progressive teaching modules (like `foundation`), a graded challenge track with a thin Python harness, and real shared CUDA kernels that power advanced lessons **and** a first GPU path for `tnsr`.

**Target stack:** CUDA C++ kernels; Python for challenge grading; gtest for lesson/kernel correctness; GPU and NVCC required for CUDA targets (no CPU GPU simulator).

## Goals

1. **Teach (Phase A):** Eight ML-oriented CUDA lessons with lib + demo + gtest + concept notes.
2. **Train (Phase B):** Fill-in challenges with reference solutions, case matrices, and a Python grader (correctness always; timing reported; soft performance budgets optional when easy).
3. **Ship (Phase C):** Production kernels in `cuda_kernels/` used by lessons 06–08 and by `tnsr` for CUDA matmul and softmax (attention kernel when ready).

## Non-goals (v1)

- Multi-GPU / NCCL
- Mixed precision (FP16/BF16/INT8) as primary path (FP32 only)
- Hermetic CUDA Docker or CPU fallback simulators
- Nsight automation or Criterion/Google Benchmark as first-class targets
- Rust-authored kernels as the primary implementation (Rust hosts tnsr; kernels are C++/CUDA)
- Changing CPU-only default of `bazel test //...` for non-GPU CI

## Constraints and decisions

| Decision | Choice |
|----------|--------|
| Curriculum depth | ML track: core five + GEMM + softmax + attention primitives |
| Build without CUDA | CUDA targets/config only; otherwise load docs and inert package stubs; no failures for default builds |
| GPU | Required to *run* GPU tests/demos |
| Languages | CUDA C++ kernels; Python harness for challenges |
| Architecture | Foundation-style lessons + separate challenges + shared `cuda_kernels` |
| v1 C | Real kernels for `tnsr`, not just a doc seam |

## Architecture

```
ferric_continuum/
├── cuda_kernels/                 # Shared production kernels (C ABI surface)
│   ├── common/                   # cuda_check, error types, device buffers
│   ├── gemm/
│   ├── softmax/
│   ├── attention/                # single-head toy attention primitive
│   └── include/ferric/cuda/      # public headers
├── cuda_gym/
│   ├── README.md                 # how to enable --config=cuda, lesson map
│   ├── lessons/
│   │   ├── 01_hello_gpu/
│   │   ├── 02_memory/
│   │   ├── 03_indexing/
│   │   ├── 04_reduction/
│   │   ├── 05_shared_memory/
│   │   ├── 06_gemm/              # wraps / demos cuda_kernels gemm
│   │   ├── 07_softmax/
│   │   └── 08_attention/
│   └── challenges/
│       ├── harness/              # Python grader + optional pybind
│       ├── vector_add/
│       ├── reduce/
│       ├── gemm/
│       └── softmax/
└── tnsr/
    └── … CUDA feature/dispatch → FFI into cuda_kernels
```

### Responsibility split

| Unit | Responsibility | Depends on |
|------|----------------|------------|
| `cuda_kernels` | Reference-quality kernels + C ABI + gtest goldens | CUDA toolkit, Abseil optional |
| `cuda_gym/lessons/*` | One concept each: teaching code, demo, gtest, short README | common helpers; 06–08 also `cuda_kernels` |
| `cuda_gym/challenges/*` | Student stubs, cases, description; reference not the default student target | same kernel problem contracts |
| `challenges/harness` | Load cases, run student vs reference, report JSON | pybind11 extension + numpy/pytest (subprocess JSON only if pybind blocked) |
| `tnsr` (feature `cuda`) | Device path for matmul + softmax (attention optional) | C ABI of `cuda_kernels` via FFI |

### Data / call flow

**Lesson demo:** host allocates → H2D → kernel launch → D2H → Abseil log / print of results and simple timing.

**Challenge grade:** harness loads `cases.json` → builds/runs student binary or pybind module → numerical compare to reference → exit non-zero on failure; print student ms and reference ms.

**tnsr CUDA op:** Rust `TensorValue` host buffer → `cuda_kernels` API (copy + launch + copy back, or reuse device buffer if later API grows) → fill output tensor. CPU path remains default when feature off.

### C ABI for shared kernels

Public headers under `cuda_kernels/include/ferric/cuda/` expose a C-compatible surface so both C++ demos and Rust FFI can link cleanly. Example shape (illustrative names):

```c
// ferric/cuda/gemm.h
#ifdef __cplusplus
extern "C" {
#endif

typedef enum FerricCudaStatus {
  FERRIC_CUDA_OK = 0,
  FERRIC_CUDA_ERR_INVALID_ARG = 1,
  FERRIC_CUDA_ERR_DEVICE = 2,
} FerricCudaStatus;

// v1 layout: row-major FP32. Device-pointer APIs may land later without changing this name's host convention.
FerricCudaStatus ferric_cuda_gemm_f32(
    int m, int n, int k,
    const float* a_host,
    const float* b_host,
    float* c_host);

#ifdef __cplusplus
}
#endif
```

v1 may start with host-pointer convenience APIs for simplicity, then add device-pointer / stream overloads without breaking lesson packaging.

### tnsr integration detail

Today `tnsr` is CPU-only pure Rust. v1 CUDA support:

1. Bazel (or Cargo feature `cuda`) builds a small `tnsr_cuda` static library / `cc_library` wrapping `cuda_kernels`.
2. Rust links via `extern "C"` blocks and optional `build.rs` / Bazel `rustc` link opts when CUDA config is on.
3. Ops: `raw_linear_forward` (and softmax) check a runtime or compile-time device flag and call FFI instead of CPU loops for eligible shapes.
4. Autograd policy for v1: **GPU forward only** for matmul and softmax; after the op returns host data, **backward stays on the existing CPU path**. Full GPU backward is explicitly out of v1.

## Curriculum (Phase A)

| # | Lesson | Core idea | Demo | Test |
|---|--------|-----------|------|------|
| 01 | hello_gpu | device props, first launch | device name, grid/block | kernel side-effect or result flag |
| 02 | memory | malloc, H2D, D2H | round-trip buffer | bit-exact |
| 03 | indexing | thread/block maps | vector + small 2D | correct maps, no OOB |
| 04 | reduction | tree / atomics | large array sum | match host |
| 05 | shared_memory | tiled load | tiled row sum / matvec | match host |
| 06 | gemm | naive + tiled | C=AB vs host time | fp32 atol/rtol |
| 07 | softmax | stable row softmax | rows sum ≈ 1 | max abs err |
| 08 | attention | scaled QK + softmax + V | tiny QKV | match host reference |

Lessons follow Ferric patterns: library + `*_demo` binary + gtest; demos use Abseil logging where practical.

## Challenges (Phase B)

v1 set: `vector_add`, `reduce`, `gemm`, `softmax`.

Per challenge:

```
challenges/<name>/
  description.md
  student.cu          # TODO stubs (default student entry)
  reference.cu        # complete implementation (separate Bazel target)
  cases.json          # shapes, seeds, rtol, atol, optional time_budget_ms
```

### Python harness

- Package: `//ferric_continuum/cuda_gym/challenges/harness`
- Flow: load cases → call student and reference pybind modules (or JSON binaries) → `numpy.allclose` → report wall time
- Entry: `bazel test //ferric_continuum/cuda_gym/challenges/...` with GPU tags; optional grade CLI
- Default student Bazel targets link `student.cu` only; CI also builds/tests reference targets separately
- Prefer pybind11 (same pattern as Muon) for in-process grading

## Build system

1. Add CUDA rules (prefer `rules_cuda` / local CUDA toolkit detection compatible with Bazel 7+ / existing MODULE.bazel).
2. Introduce user-facing flag: `bazel build --config=cuda //ferric_continuum/cuda_gym/...`
3. `.bazelrc` entries:
   - `build:cuda --//...` enabling CUDA toolchains / defines
4. All GPU tests: `tags = ["cuda", "requires-gpu", "exclusive"]` (exclusive optional if device contention).
5. Default CI remains CPU-only; optional separate workflow (or documented manual job) for GPU runners — **not required on day one**, but tags ready.

If `rules_cuda` integration is blocked, fall back to `nvcc` `genrule` / custom rule minimal surface documented in MODULE/BUILD — still gated by `--config=cuda`.

## Error handling

- Macro/helper `FERRIC_CUDA_CHECK(call)` maps `cudaError_t` to status or fatal Abseil log in demos.
- Kernels do not silently ignore launch failures; demos call `cudaDeviceSynchronize` after launch and check errors.
- Python harness: device error → failed test with stderr captured, no hang (timeout on test rules).
- Shape / arg validation at C ABI boundary: return `FERRIC_CUDA_ERR_INVALID_ARG`.

## Testing strategy

| Layer | Mechanism | When |
|-------|-----------|------|
| Lessons | gtest | CUDA config + GPU |
| cuda_kernels | gtest goldens (shared shapes with lesson 06–08) | same |
| Challenges | pytest + harness | same |
| tnsr CUDA | Rust tests, GPU-tagged, small shapes | same |
| Numerical policy | fp32: rtol 1e-4 / atol 1e-5 (tunable per case) | documented in cases |

## Documentation updates

- Root `README.md`: CUDA gym section + `--config=cuda` prerequisites
- `ferric_continuum/cuda_gym/README.md`: lesson order, challenge workflow, GPU requirements
- `CLAUDE.md` / `AGENTS.md` optional one-liner for agents
- `tnsr/SCALING_BOOK_MAP.md`: mark GPU chapter mapping as partially covered when kernels land

## Implementation phasing (for planning)

1. **Bootstrap:** CUDA Bazel config, `cuda_kernels/common`, hello_gpu lesson end-to-end.
2. **Core lessons 02–05:** memory through shared memory.
3. **Production kernels:** GEMM + softmax (+ attention kernel).
4. **Lessons 06–08:** thin demos/tests over kernels.
5. **Challenge harness + four challenges.**
6. **tnsr FFI + CUDA matmul/softmax forward path + tags.**
7. **Docs polish.**

## Success criteria

- With `--config=cuda` on a CUDA machine: all lesson demos run; all tagged CUDA tests pass.
- Challenge student stubs **fail** until filled; reference targets **pass**.
- `tnsr` with CUDA feature: matmul and softmax forward match CPU within tolerances on tested shapes.
- Without CUDA config: repository builds/tests remain green on existing CPU CI.

## Open points deferred to implementation plan

- Exact `rules_cuda` version and MODULE.bazel snippets
- Whether challenge link uses pybind11 vs CUDA binary JSON protocol (prefer pybind if toolchain fits existing muoon style; else subprocess JSON is fine)
- Device pointer residency API for longer-term zero-copy autograd
- Optional soft perf budgets in cases.json once timing variance is measured
