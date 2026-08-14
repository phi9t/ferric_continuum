# `tnsr` repository map

A compact guide to where model semantics, execution, CUDA kernels, and parallelism research live—and where they do not.

## The three-layer map

| Layer | Start here | What it does | Reality level |
|---|---|---|---|
| Tensor engine | `tnsr/src/tensor.rs`, `autograd.rs`, `ops/` | Stores f32 host tensors, records operations, executes forward/backward math | Real execution |
| Model | `tnsr/src/qwen3.rs`, `qwen3_load.rs`, `bin/qwen3_infer.rs` | Builds and loads Qwen3, runs prefill and greedy generation | Real execution |
| CUDA bridge | `tnsr/src/cuda_ffi.rs`, `cuda_kernels/` | Runs selected forward operations through a host-buffer C ABI | Real, but not GPU-resident |
| Scaling lab | `tnsr/src/scaling/` | Computes FLOPs/bytes and simulates shards or collectives in one process | Analytical/simulated |
| CUDA curriculum | `cuda_gym/lessons/`, `cuda_gym/challenges/` | Teaches and grades standalone kernel concepts | Real CUDA exercises |

## Qwen3 inference trace

```text
src/bin/qwen3_infer.rs
  → qwen3_load::load_qwen3
  → Qwen3Model::forward
    → embedding
    → Qwen3Block::forward × L
      → RMSNorm
      → Qwen3Attention::forward
        → linear(Q, K, V) → optional CUDA GEMM
        → reshape → Q/K norm → RoPE
        → gqa_attention → CPU fused scores/mask/softmax/mix
        → linear(O) → optional CUDA GEMM
      → residual add → RMSNorm
      → linear(gate, up) → optional CUDA GEMM
      → SiLU × multiply
      → linear(down) → optional CUDA GEMM
    → final RMSNorm
    → linear(lm_head) → optional CUDA GEMM
  → host-side argmax → append token → repeat full forward
```

## CUDA boundary

| Operation | Generic block | Qwen3 path | Backward |
|---|---|---|---|
| Dense linear | CUDA-eligible | CUDA-eligible | CPU |
| Standalone softmax | CUDA-eligible | Not used by fused GQA | CPU |
| Attention scores/mix | CPU | CPU fused GQA | CPU |
| Norm, RoPE, activation, residual | CPU | CPU | CPU |

> **Performance warning:** Every CUDA bridge call allocates device buffers, copies host inputs to the device, launches a kernel, and copies results back. This is a correctness bridge, not yet an efficient GPU tensor runtime.

## Build switch

```bash
# CPU default
bazel test //ferric_continuum/tnsr:tnsr_tests

# Link CUDA feature and kernels; requires toolkit + GPU to run
bazel test --config=cuda //ferric_continuum/tnsr:cuda_forward_tests

# Debug a CUDA build through the CPU path without rebuilding
FERRIC_TNSR_DEVICE=cpu bazel run --config=cuda //ferric_continuum/tnsr:tnsr_demo
```

## Research compass

| Question | Inspect first | Evidence to demand |
|---|---|---|
| Is this mathematically correct? | `ops/`, gradient and agreement tests | Golden values, gradient checks, CPU/GPU error bounds |
| Does this run on GPU? | Call site → `cuda_ffi.rs` → C header → `.cu` | Build feature, kernel launch, profiler trace |
| Does this scale across devices? | `scaling/distributed/` | Separate formula/simulation claims from real transport |
| Will parallelism help? | `op_cost.rs`, `roofline.rs`, collective costs | Compute/communication overlap and topology-aware measurements |
| Is inference efficient? | `qwen3_infer.rs`, `scaling/inference.rs` | Prefill/decode split, KV-cache behavior, bytes per generated token |

[Return to Lesson 0001](../lessons/0001-three-layers-of-tnsr.md) · [Continue to context-parallel Qwen3](../lessons/0002-context-parallel-qwen3.md)
