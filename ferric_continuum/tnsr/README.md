# tnsr

`ferric_continuum/tnsr` is the monorepo's Bazel-first Rust tensor and
transformer teaching module. It focuses on readable reverse-mode autograd,
activation checkpointing, transformer block mechanics, debug traces, symbolic
scaling reports, and small Qwen3 experiments.

The default build is CPU-only and f32; selected forward kernels can use the
monorepo CUDA path through `//ferric_continuum/cuda_kernels` when built with
Bazel `--config=cuda`.

## Build and Test

Start with Bazel. The pinned binary avoids host `bazel` version drift:

```sh
/data02/home/philip.yang/.local/bin/bazel-9.2.0 build //ferric_continuum/tnsr:tnsr
/data02/home/philip.yang/.local/bin/bazel-9.2.0 test //ferric_continuum/tnsr:lib_tests
/data02/home/philip.yang/.local/bin/bazel-9.2.0 build //ferric_continuum/tnsr:tnsr_doc
```

Run the current module behavior suite with:

```sh
/data02/home/philip.yang/.local/bin/bazel-9.2.0 test \
  //ferric_continuum/tnsr:tnsr_tests \
  //ferric_continuum/tnsr:debug_trace_tests \
  //ferric_continuum/tnsr:inference_tests \
  //ferric_continuum/tnsr:mesh_sim_tests \
  //ferric_continuum/tnsr:scaling_tests \
  //ferric_continuum/tnsr:scaling_memory_tests \
  //ferric_continuum/tnsr:distributed_tests \
  //ferric_continuum/tnsr:qwen3_tests \
  //ferric_continuum/tnsr:context_parallel_gqa_tests \
  //ferric_continuum/tnsr:playground_cli_tests \
  //ferric_continuum/tnsr:lib_tests
```

CUDA forward agreement is separate, opt-in, and requires a CUDA GPU:

```sh
/data02/home/philip.yang/.local/bin/bazel-9.2.0 test --config=cuda //ferric_continuum/tnsr:cuda_forward_tests
```

Cargo metadata exists for Rust editor tooling and local supplementary checks,
but Bazel is the supported build and verification path.

## Playground

Use `tnsr_demo` when you want a single maintained example that exercises the
current APIs:

```sh
/data02/home/philip.yang/.local/bin/bazel-9.2.0 run //ferric_continuum/tnsr:tnsr_demo -- --help
/data02/home/philip.yang/.local/bin/bazel-9.2.0 run //ferric_continuum/tnsr:tnsr_demo -- \
  --dot /tmp/tnsr/block.dot \
  --trace-json /tmp/tnsr/trace.json
```

The demo runs:

- uncheckpointed tiny transformer forward/backward
- whole-block checkpointing
- selective transformer checkpointing
- DOT graph output
- stable `tnsr.debug_trace` JSON output
- scaling, memory, and distributed parallelism reports

Generated artifacts belong in caller-selected temporary paths such as `/tmp`.
Do not check generated DOT or JSON output into the repo.

## Training Path

Forward ops live under `tnsr::ops`. Each op computes a `TensorValue` first. If
gradient mode is enabled and any input requires gradients, the op creates an
`OpCall` for the output tensor.

`Engine::new()` creates the debug recorder used by the backward pass and by
trace exports. Call it before building the graph you want to inspect:

```rust
use tnsr::autograd::Engine;
use tnsr::ops::{basic, linear};
use tnsr::tensor::{Shape, Tensor, TensorValue};

let mut engine = Engine::new();
let x = Tensor::from_value(
    TensorValue::from_vec(Shape(vec![1, 2]), vec![1.0, 2.0]),
    true,
);
let w = Tensor::from_value(
    TensorValue::from_vec(Shape(vec![2, 1]), vec![0.5, -1.5]),
    true,
);
let y = linear::linear(&x, &w, "proj");
let loss = basic::sum(&y, "loss");
engine.backward(&loss);
println!("{}", engine.debug.trace_json_pretty());
```

Backward numerical values are saved through `SavedTensor`:

- `Materialized` stores activation values for backward.
- `Borrowed` keeps parameter references without copying values.
- `Recompute` records checkpoint handles and replays the forward closure on
  first unpack.

## Debug Trace JSON

`DebugRecorder::trace_json()` exports a stable Rust-owned schema:

- `schema`: currently `tnsr.debug_trace`
- `schema_version`: currently `1`
- `events`: replayable ordered event stream
- `forward_ops`: normalized forward op records with shapes and edges
- `saved_sites`: save/unpack records
- `checkpoints`: checkpoint enter/exit/recompute records
- `backward_ops`: reverse-mode operation applications
- `grad_accumulations`: gradient accumulation and leaf-write events

The schema is tested by `//ferric_continuum/tnsr:debug_trace_tests`.

## Inference Primitives

Model-agnostic decode helpers live in `tnsr::inference`:

- `RopeConfig` and `apply_rope` rotate raw `[B,T,D]` query/key values by
  absolute token position.
- `KvCache` stores per-sequence key/value prefixes as `[T,D]` buffers.
- `DecodeRequest`, `AttentionStep`, `BatchStats`, and `ContinuousBatcher`
  provide deterministic FIFO batching for prefill/decode work.

These helpers are not a serving stack. Tokenization, sampling policy, network
serving, and optimized multi-head kernels remain outside this module surface.
Qwen3's differentiable path continues to use `ops::rope` and `ops::gqa`.

## Scaling and Distributed Reports

`tnsr::scaling` turns the reference block into executable estimates:

- `model_stats`: parameter counts for the current `TransformerConfig`
- `op_cost`: FLOPs and activation-byte formulas by op family
- `report`: aggregate transformer scaling report and formatter
- `memory`: training memory, activation checkpointing, KV cache, and ZeRO stage
  estimates
- `roofline`: compute-vs-memory bounds for a named hardware profile
- `sharding`: local algebra for sharded matmul cases
- `distributed`: symbolic DDP/FSDP/TP/PP reports plus single-process
  collective simulations

Distributed reports are estimates and simulations, not runtime distributed
execution. They are verified through `//ferric_continuum/tnsr:distributed_tests`.

`SCALING_BOOK_MAP.md` has the longer formula map.

## Qwen3

Qwen3 support is a monorepo extension beyond the small teaching block:

- `//ferric_continuum/tnsr:qwen3_demo`
- `//ferric_continuum/tnsr:qwen3_generate`
- `//ferric_continuum/tnsr:qwen3_op_trace`
- `//ferric_continuum/tnsr:qwen3_infer`
- `ferric_continuum/tnsr/tools/verify_hf_compat.sh`

Real-weight compatibility checks require a local HuggingFace model directory
and are only required when Qwen3 math, loader, tokenizer, RoPE/GQA behavior, or
logits output changes.
